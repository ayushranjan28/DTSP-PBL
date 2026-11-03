function [denoised, gain_matrix] = wiener_filter_audio(audio, fs, alpha_smooth, overestimation_beta, gain_floor_db, n_fft, hop_length)
% WIENER_FILTER_AUDIO Ephraim-Malah Decision-Directed audio Wiener filter.
% Pure MATLAB implementation requiring no external toolboxes.
%
% Inputs:
%   audio               - 1D vector of audio samples
%   fs                  - Sampling frequency in Hz (default: 16000)
%   alpha_smooth        - Decision-directed smoothing factor (default: 0.95)
%   overestimation_beta - Noise overestimation factor (default: 1.25)
%   gain_floor_db       - Minimum allowable filter gain in dB (default: -18.0)
%   n_fft               - FFT window size (default: 512)
%   hop_length          - Hop size (default: 128)

    if nargin < 2, fs = 16000; end
    if nargin < 3, alpha_smooth = 0.95; end
    if nargin < 4, overestimation_beta = 1.25; end
    if nargin < 5, gain_floor_db = -18.0; end
    if nargin < 6, n_fft = 512; end
    if nargin < 7, hop_length = 128; end

    audio = audio(:);
    n_samples = length(audio);

    % Native Periodic Hann Window
    n_idx = (0:n_fft-1)';
    win = 0.5 * (1 - cos(2 * pi * n_idx / n_fft));

    n_frames = floor((n_samples - n_fft) / hop_length) + 1;
    stft_mat = zeros(n_fft, n_frames);

    for i = 1:n_frames
        idx = (i-1)*hop_length + (1:n_fft);
        stft_mat(:, i) = fft(audio(idx) .* win);
    end

    n_bins = n_fft / 2 + 1;
    pos_stft = stft_mat(1:n_bins, :);
    mag = abs(pos_stft);
    phase = angle(pos_stft);
    power = mag .^ 2;

    % Noise PSD estimation from quietest 20% frames
    frame_energy = sum(power, 1);
    energy_thresh = native_prctile(frame_energy, 20);
    quiet_idx = find(frame_energy <= energy_thresh);
    if length(quiet_idx) < 3
        [~, sorted_idx] = sort(frame_energy);
        quiet_idx = sorted_idx(1:max(3, round(n_frames * 0.1)));
    end

    noise_psd = mean(power(:, quiet_idx), 2);
    noise_psd = max(noise_psd, 1e-12);
    scaled_noise_psd = overestimation_beta * noise_psd;

    gain_min = 10 ^ (gain_floor_db / 20.0);
    gain_matrix = zeros(n_bins, n_frames);
    s_hat_mag_prev = zeros(n_bins, 1);

    % Decision-Directed recursive loop across frames
    for i = 1:n_frames
        p_frame = power(:, i);

        % A posteriori SNR
        gamma = max(p_frame ./ scaled_noise_psd, 1e-4);

        if i == 1
            xi = max(gamma - 1.0, 0.0);
        else
            prev_snr = (s_hat_mag_prev .^ 2) ./ scaled_noise_psd;
            xi = alpha_smooth * prev_snr + (1.0 - alpha_smooth) * max(gamma - 1.0, 0.0);
        end

        % Wiener gain transfer function H = xi / (xi + 1)
        h = xi ./ (xi + 1.0);
        h = max(gain_min, min(1.0, h));

        gain_matrix(:, i) = h;
        s_hat_mag_prev = h .* mag(:, i);
    end

    % Reconstruct modified complex STFT
    clean_pos = (gain_matrix .* mag) .* exp(1j * phase);
    clean_full = zeros(n_fft, n_frames);
    clean_full(1:n_bins, :) = clean_pos;
    clean_full(n_bins+1:end, :) = conj(clean_pos(end-1:-1:2, :));

    % Overlap-Add
    denoised = zeros(n_samples + n_fft, 1);
    win_sum = zeros(n_samples + n_fft, 1);

    for i = 1:n_frames
        idx = (i-1)*hop_length + (1:n_fft);
        time_frame = real(ifft(clean_full(:, i)));
        denoised(idx) = denoised(idx) + time_frame .* win;
        win_sum(idx) = win_sum(idx) + win .^ 2;
    end

    nonzero = win_sum > 1e-4;
    denoised(nonzero) = denoised(nonzero) ./ win_sum(nonzero);
    denoised = single(denoised(1:n_samples));
end

function val = native_prctile(x, p)
    sx = sort(x);
    k = max(1, min(length(sx), round((p / 100) * (length(sx) - 1) + 1)));
    val = sx(k);
end
