%% Adaptive ML-Assisted Audio Noise Reduction Using FIR/IIR Filters
% Master Academic MATLAB Demonstration Script
% 
% Demonstrates classical DSP (FIR, Butterworth SOS, Notch biquads),
% STFT Spectral Gating, Decision-Directed Audio Wiener filtering, and
% content-adaptive rule-based filter dispatching.
%
% Runs in any MATLAB environment (base MATLAB R2020a-R2025b or GNU Octave).

clear; close all; clc;
fprintf('=================================================================\n');
fprintf('  SonicPure DSP: Adaptive Audio Noise Reduction Demonstration   \n');
fprintf('=================================================================\n\n');

%% 1. Load or Synthesize Clean Speech Audio
fs = 16000; % 16 kHz standard sample rate
sample_wav_path = fullfile('..', 'data', 'samples', 'clean_sample.wav');

clean_loaded = false;
if exist(sample_wav_path, 'file')
    try
        fprintf('[1/5] Loading clean speech from %s...\n', sample_wav_path);
        [clean_speech, in_fs] = audioread(sample_wav_path);
        if in_fs ~= fs
            % Native resample if resample function missing
            if exist('resample', 'file')
                clean_speech = resample(clean_speech, fs, in_fs);
            end
        end
        if size(clean_speech, 2) > 1
            clean_speech = mean(clean_speech, 2);
        end
        clean_loaded = true;
    catch
        clean_loaded = false;
    end
end

if ~clean_loaded
    fprintf('[1/5] Synthesizing academic speech formants (vocal tract model)...\n');
    duration = 3.5;
    t = (0:1/fs:duration-1/fs)';
    f0 = 135.0 - 15.0 * (t / duration) + 3.0 * sin(2*pi*5.0*t);
    phase = 2 * pi * cumsum(f0) / fs;
    
    glottal = zeros(size(t));
    for h = 1:20
        glottal = glottal + (1 / h^0.8) * sin(h * phase);
    end
    
    formant1 = sin(2*pi*650*t) .* exp(-mod(t*2.0, 0.3));
    formant2 = 0.5 * sin(2*pi*1550*t) .* exp(-mod(t*2.5, 0.3));
    envelope = max(0.05, sin(2*pi*3.5*t).^2);
    clean_speech = (0.6 * glottal + 0.3 * formant1 + 0.2 * formant2) .* envelope;
    clean_speech = 0.8 * clean_speech / max(abs(clean_speech));
end

clean_speech = clean_speech(:);
n_samples = length(clean_speech);

%% 2. Synthesize Calibrated Noise & Mix at Target SNR (5 dB)
target_snr_db = 5.0;
fprintf('[2/5] Synthesizing 50 Hz powerline hum and mixing at %0.1f dB SNR...\n', target_snr_db);

t = (0:n_samples-1)' / fs;
% 50 Hz harmonic hum
hum_noise = sin(2*pi*50*t) + 0.5*sin(2*pi*100*t) + 0.33*sin(2*pi*150*t) + 0.25*sin(2*pi*200*t);
hum_noise = hum_noise / sqrt(mean(hum_noise.^2));

% SNR Power scaling: SNR = 10 * log10( P_signal / P_noise )
p_signal = mean(clean_speech .^ 2);
p_noise_target = p_signal / (10 ^ (target_snr_db / 10));
scaled_noise = hum_noise * sqrt(p_noise_target);
noisy_audio = clean_speech + scaled_noise;

measured_snr_in = 10 * log10(p_signal / mean((noisy_audio - clean_speech).^2));
fprintf('      Measured Input SNR = %0.2f dB\n\n', measured_snr_in);

%% 3. Execute All 6 Denoising Strategies & Benchmark
fprintf('[3/5] Processing audio through all 6 filtering methods...\n');
methods = {
    '1. Baseline (Noisy Input)',
    '2. FIR Bandpass (300-3400 Hz)',
    '3. Butterworth Highpass (150 Hz)',
    '4. Harmonic Notch (50 Hz)',
    '5. STFT Spectral Gating',
    '6. Audio Wiener Filter (DD)',
    '7. ML-Assisted Adaptive Filtering'
};

outputs = cell(length(methods), 1);
runtimes = zeros(length(methods), 1);

% 1. Baseline
outputs{1} = noisy_audio;
runtimes(1) = 0.0001;

% 2. FIR Filter
tic;
outputs{2} = apply_fir_filter(noisy_audio, [300, 3400], 101, 'bandpass', 'hamming', fs, true);
runtimes(2) = toc;

% 3. Butterworth Highpass
tic;
outputs{3} = apply_butterworth_filter(noisy_audio, 150.0, 4, 'high', fs, true);
runtimes(3) = toc;

% 4. Harmonic Notch
tic;
outputs{4} = apply_notch_filter(noisy_audio, 50.0, 35.0, true, 4, fs, true);
runtimes(4) = toc;

% 5. Spectral Gating
tic;
outputs{5} = spectral_gating(noisy_audio, fs, -18.0, 1.5, 512, 128);
runtimes(5) = toc;

% 6. Wiener Filter
tic;
outputs{6} = wiener_filter_audio(noisy_audio, fs, 0.95, 1.25, -18.0, 512, 128);
runtimes(6) = toc;

% 7. ML-Assisted Adaptive Filtering
tic;
[outputs{7}, pred_class, strat_name, strat_reason] = adaptive_filter_dispatch(noisy_audio, fs);
runtimes(7) = toc;

%% 4. Calculate Objective Performance Metrics
fprintf('[4/5] Calculating objective DSP metrics...\n\n');
out_snr = zeros(length(methods), 1);
delta_snr = zeros(length(methods), 1);
rmse_vals = zeros(length(methods), 1);
pearson_r = zeros(length(methods), 1);

for i = 1:length(methods)
    sig = outputs{i};
    err = sig - clean_speech;
    p_err = mean(err .^ 2);
    out_snr(i) = 10 * log10(p_signal / max(p_err, 1e-12));
    delta_snr(i) = out_snr(i) - measured_snr_in;
    rmse_vals(i) = sqrt(p_err);
    r_mat = corrcoef(clean_speech, sig);
    pearson_r(i) = r_mat(1, 2);
end

% Print ASCII Table
fprintf('----------------------------------------------------------------------------------------\n');
fprintf('  %-34s | %-10s | %-10s | %-8s | %-9s \n', 'Method', 'Output SNR', 'Delta SNR', 'RMSE', 'Time (ms)');
fprintf('----------------------------------------------------------------------------------------\n');
for i = 1:length(methods)
    fprintf('  %-34s | %6.2f dB | %+6.2f dB | %0.4f | %6.1f ms \n', ...
        methods{i}, out_snr(i), delta_snr(i), rmse_vals(i), runtimes(i)*1000);
end
fprintf('----------------------------------------------------------------------------------------\n\n');

fprintf('ML Adaptive Diagnosis: %s\n', pred_class);
fprintf('Selected Filter:       %s\n', strat_name);
fprintf('Rationale:             %s\n\n', strat_reason);

%% 5. Generate Publication-Quality Visualizations
fprintf('[5/5] Generating publication-quality figures...\n');

% Figure 1: Waveforms
try
    figure('Name', 'SonicPure DSP Waveforms', 'Color', 'w', 'Position', [100, 100, 900, 600]);
    subplot(3, 1, 1);
    plot(t, noisy_audio, 'Color', [0.95, 0.25, 0.35], 'LineWidth', 1.0);
    grid on; title('Noisy Input Signal (Speech + 50 Hz Hum @ 5 dB SNR)');
    ylabel('Amplitude'); ylim([-1.1, 1.1]);

    subplot(3, 1, 2);
    plot(t, outputs{7}, 'Color', [0.06, 0.72, 0.50], 'LineWidth', 1.0);
    grid on; title(sprintf('Cleaned Signal via Adaptive Filtering (%s)', strat_name));
    ylabel('Amplitude'); ylim([-1.1, 1.1]);

    subplot(3, 1, 3);
    plot(t, clean_speech, 'Color', [0.05, 0.65, 0.95], 'LineWidth', 1.0);
    grid on; title('Clean Ground-Truth Speech Reference');
    xlabel('Time (seconds)'); ylabel('Amplitude'); ylim([-1.1, 1.1]);
catch
end

% Figure 2: Spectrograms (Native STFT fallback if spectrogram toolbox missing)
try
    figure('Name', 'SonicPure DSP Spectrograms', 'Color', 'w', 'Position', [150, 150, 950, 420]);
    subplot(1, 2, 1);
    plot_stft_image(noisy_audio, fs, 'Noisy Input Spectrogram');

    subplot(1, 2, 2);
    plot_stft_image(outputs{7}, fs, sprintf('Denoised Spectrogram (%s)', strat_name));
catch
end

% Save output audio
out_dir = fullfile('outputs');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
try
    audiowrite(fullfile(out_dir, 'matlab_noisy_input.wav'), noisy_audio, fs);
    audiowrite(fullfile(out_dir, 'matlab_denoised_adaptive.wav'), outputs{7}, fs);
    fprintf('Saved audio outputs to %s/\n', out_dir);
catch
end

fprintf('Demo complete!\n');

function plot_stft_image(x, fs, title_str)
    n_fft = 512;
    hop = 128;
    win = 0.5 * (1 - cos(2*pi*(0:n_fft-1)'/n_fft));
    n_frames = floor((length(x) - n_fft) / hop) + 1;
    S = zeros(n_fft/2 + 1, n_frames);
    for k = 1:n_frames
        idx = (k-1)*hop + (1:n_fft);
        X = fft(x(idx) .* win);
        S(:, k) = abs(X(1:n_fft/2 + 1));
    end
    t_axis = (0:n_frames-1) * hop / fs;
    f_axis = linspace(0, fs/2, n_fft/2 + 1);
    imagesc(t_axis, f_axis, 20*log10(max(S, 1e-4)));
    axis xy; colormap('hot'); colorbar;
    xlabel('Time (s)'); ylabel('Frequency (Hz)');
    title(title_str);
end
