function [denoised, gain_mask] = spectral_gating(audio, fs, suppression_db, over_subtraction, n_fft, hop_length)
% SPECTRAL_GATING STFT Spectral noise subtraction with over-subtraction and 2D smoothing.
% Pure MATLAB implementation requiring no external toolboxes.
%
% Inputs:
%   audio            - 1D vector of audio samples
%   fs               - Sampling frequency in Hz (default: 16000)
%   suppression_db   - Minimum gain floor in dB (default: -18.0)
%   over_subtraction - Multiplier for estimated noise profile (default: 1.5)
%   n_fft            - FFT frame size (default: 512)
%   hop_length       - Hop size between successive frames (default: 128)

    if nargin < 2, fs = 16000; end
    if nargin < 3, suppression_db = -18.0; end
    if nargin < 4, over_subtraction = 1.5; end
    if nargin < 5, n_fft = 512; end
    if nargin < 6, hop_length = 128; end

    audio = audio(:);
    n_samples = length(audio);

    % Native Periodic Hann Window
    n_idx = (0:n_fft-1)';
    win = 0.5 * (1 - cos(2 * pi * n_idx / n_fft));

    % Compute STFT via standard framing
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

    % Estimate noise profile from the quietest 20% frames
    frame_energy = sum(power, 1);
    energy_thresh = native_prctile(frame_energy, 20);
    quiet_idx = find(frame_energy <= energy_thresh);
    if length(quiet_idx) < 3
        [~, sorted_idx] = sort(frame_energy);
        quiet_idx = sorted_idx(1:max(3, round(n_frames * 0.1)));
    end

    noise_mag = mean(mag(:, quiet_idx), 2);
    noise_threshold = over_subtraction * noise_mag;

    % Linear gain floor
    gain_floor = 10 ^ (suppression_db / 20.0);

    % Spectral Subtraction Mask
    sub = max(0, (mag - repmat(noise_threshold, 1, n_frames)) ./ (mag + 1e-9));
    gain_mask = max(gain_floor, sub);

    % 2D Smoothing (Box blur) to suppress musical noise
    h_smooth = ones(3, 3) / 9;
    gain_mask = conv2(gain_mask, h_smooth, 'same');
    gain_mask = max(gain_floor, min(1.0, gain_mask));

    % Reconstruct modified complex STFT
    clean_pos = (gain_mask .* mag) .* exp(1j * phase);

    clean_full = zeros(n_fft, n_frames);
    clean_full(1:n_bins, :) = clean_pos;
    clean_full(n_bins+1:end, :) = conj(clean_pos(end-1:-1:2, :));

    % Overlap-Add Inverse STFT
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
