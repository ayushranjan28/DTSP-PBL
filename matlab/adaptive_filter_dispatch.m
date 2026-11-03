function [filtered, pred_class, strategy_name, reason] = adaptive_filter_dispatch(audio, fs)
% ADAPTIVE_FILTER_DISPATCH Classifies interference and dispatches the optimal DSP filter.
%
% Inputs:
%   audio         - 1D audio sample vector
%   fs            - Sampling frequency in Hz (default: 16000)
%
% Outputs:
%   filtered      - Cleaned audio vector
%   pred_class    - Predicted noise category string
%   strategy_name - Selected DSP filter name
%   reason        - Technical explanation for the chosen filter topology

    if nargin < 2, fs = 16000; end

    % Extract feature fingerprint
    feats = extract_audio_features(audio, fs);

    % Decision Logic mapping
    if feats.hum_50hz_ratio > 0.05 || (abs(feats.dominant_peak_freq - 50) < 6 && feats.energy_ratio_rumble > 0.15)
        pred_class = 'electrical_hum_50hz';
        strategy_name = 'Harmonic Notch Filter (50 Hz)';
        reason = 'High narrowband energy detected at 50 Hz mains fundamental. Harmonic notch cuts hum with zero damage to speech formants.';
        filtered = apply_notch_filter(audio, 50.0, 35.0, true, 4, fs, true);

    elseif feats.hum_60hz_ratio > 0.05 || (abs(feats.dominant_peak_freq - 60) < 6 && feats.energy_ratio_rumble > 0.15)
        pred_class = 'electrical_hum_60hz';
        strategy_name = 'Harmonic Notch Filter (60 Hz)';
        reason = 'High narrowband energy detected at 60 Hz mains fundamental. Harmonic notch deployed with Q=35.';
        filtered = apply_notch_filter(audio, 60.0, 35.0, true, 4, fs, true);

    elseif feats.energy_ratio_rumble > 0.35 && feats.spectral_centroid < 800
        pred_class = 'low_frequency_rumble';
        strategy_name = 'Butterworth Highpass SOS (150 Hz)';
        reason = 'Acoustic rumble concentrated below 150 Hz (HVAC/handling). 4th-order high-pass Butterworth filter deployed.';
        filtered = apply_butterworth_filter(audio, 150.0, 4, 'high', fs, true);

    elseif feats.energy_ratio_high > 0.30 && feats.spectral_centroid > 2800
        pred_class = 'high_frequency_hiss';
        strategy_name = 'Linear-Phase Low-Pass FIR (3800 Hz)';
        reason = 'High-frequency thermal/tape hiss detected above 3.5 kHz. Linear-phase FIR lowpass filter rolls off hiss.';
        filtered = apply_fir_filter(audio, 3800.0, 101, 'low', 'hamming', fs, true);

    elseif feats.rms < 0.02
        pred_class = 'clean_speech';
        strategy_name = 'Subtle Denoiser / Pass-Through';
        reason = 'Audio classified as clean speech with nominal background noise. Processing bypassed to avoid speech coloration.';
        filtered = audio;

    else
        pred_class = 'broadband_noise';
        strategy_name = 'Audio Wiener Filter (Decision-Directed)';
        reason = 'Broadband distributed noise detected across spectrum. Decision-Directed Wiener filter tracks speech non-stationarity.';
        filtered = wiener_filter_audio(audio, fs, 0.95, 1.25, -18.0);
    end
end
