function feats = extract_audio_features(audio, fs)
% EXTRACT_AUDIO_FEATURES Computes 10 physical and spectral features for noise classification.
% Works in base MATLAB without requiring Signal Processing Toolbox.
%
% Inputs:
%   audio - 1D audio sample vector
%   fs    - Sampling rate in Hz (default: 16000)
%
% Outputs:
%   feats - Struct containing computed features:
%           .rms, .zcr, .spectral_centroid, .energy_ratio_rumble,
%           .energy_ratio_high, .dominant_peak_freq, .hum_50hz_ratio,
%           .hum_60hz_ratio

    if nargin < 2, fs = 16000; end
    audio = audio(:);
    n = length(audio);

    % 1. RMS Energy
    feats.rms = sqrt(mean(audio .^ 2));

    % 2. Zero-Crossing Rate
    feats.zcr = sum(abs(diff(audio > 0))) / (n - 1);

    % 3. FFT Magnitude Spectrum with Native Hamming Window
    n_fft = 2048;
    win = 0.54 - 0.46 * cos(2 * pi * (0:n-1)' / max(1, n - 1));
    freqs = linspace(0, fs/2, n_fft/2 + 1)';
    
    pad_len = max(n, n_fft);
    padded_x = zeros(pad_len, 1);
    padded_x(1:n) = audio .* win;
    
    X = abs(fft(padded_x, n_fft));
    X = X(1:n_fft/2 + 1);
    P = X .^ 2;
    total_power = sum(P) + 1e-12;

    % 4. Spectral Centroid
    feats.spectral_centroid = sum(freqs .* X) / (sum(X) + 1e-12);

    % 5. Sub-band Energies
    mask_rumble = (freqs >= 0) & (freqs < 150);
    feats.energy_ratio_rumble = sum(P(mask_rumble)) / total_power;

    mask_high = (freqs >= 3000) & (freqs <= fs/2);
    feats.energy_ratio_high = sum(P(mask_high)) / total_power;

    % 6. Dominant Peak Frequency
    [~, max_idx] = max(P);
    feats.dominant_peak_freq = freqs(max_idx);

    % 7. Mains Hum Ratios (50 Hz and 60 Hz)
    mask_50 = (freqs >= 44) & (freqs <= 56);
    feats.hum_50hz_ratio = sum(P(mask_50)) / total_power;

    mask_60 = (freqs >= 54) & (freqs <= 66);
    feats.hum_60hz_ratio = sum(P(mask_60)) / total_power;
end
