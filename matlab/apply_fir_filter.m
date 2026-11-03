function [filtered, b, delay_ms] = apply_fir_filter(audio, cutoff_hz, num_taps, btype, window_name, fs, zero_phase)
% APPLY_FIR_FILTER Designs and executes a linear-phase FIR filter.
% Runs in any MATLAB environment (uses pure native math if Signal Processing Toolbox is absent).
%
% Inputs:
%   audio       - 1D vector of audio samples
%   cutoff_hz   - Cutoff frequency in Hz (scalar for low/high, [low high] for bandpass)
%   num_taps    - Number of filter taps / order+1
%   btype       - 'low', 'high', or 'bandpass'
%   window_name - 'hamming', 'hann', 'blackman', or 'rect' (default: 'hamming')
%   fs          - Sampling frequency in Hz (default: 16000)
%   zero_phase  - Boolean: true for zero-phase (filtfilt), false for causal
%
% Outputs:
%   filtered    - 1D filtered audio vector
%   b           - Filter impulse response coefficients
%   delay_ms    - Group delay in milliseconds

    if nargin < 4, btype = 'low'; end
    if nargin < 5, window_name = 'hamming'; end
    if nargin < 6, fs = 16000; end
    if nargin < 7, zero_phase = true; end

    nyquist = fs / 2;
    order = num_taps - 1;

    % Ensure order is even for highpass/bandpass (Type I linear phase)
    if (strcmpi(btype, 'high') || strcmpi(btype, 'bandpass')) && mod(order, 2) ~= 0
        order = order + 1;
    end
    M = order;
    N = M + 1;

    % Pure MATLAB window implementation (no toolbox required)
    n_idx = (0:M)';
    switch lower(window_name)
        case 'hann'
            win = 0.5 * (1 - cos(2 * pi * n_idx / M));
        case 'blackman'
            win = 0.42 - 0.5 * cos(2 * pi * n_idx / M) + 0.08 * cos(4 * pi * n_idx / M);
        case 'rect'
            win = ones(N, 1);
        otherwise % hamming
            win = 0.54 - 0.46 * cos(2 * pi * n_idx / M);
    end

    % Check if fir1 from Signal Processing Toolbox is available
    use_fir1 = false;
    if exist('fir1', 'file')
        try
            Wn = cutoff_hz / nyquist;
            if strcmpi(btype, 'bandpass')
                b = fir1(order, Wn, 'bandpass', win);
            elseif strcmpi(btype, 'high')
                b = fir1(order, Wn, 'high', win);
            else
                b = fir1(order, Wn, 'low', win);
            end
            use_fir1 = true;
        catch
            use_fir1 = false;
        end
    end

    if ~use_fir1
        % Analytical Windowed-Sinc FIR Design (Pure Base MATLAB)
        b = design_sinc_fir(cutoff_hz, M, btype, win, fs);
    end

    b = b(:)'; % row vector

    % Filtering with causal or zero-phase forward-backward
    audio = audio(:);
    if zero_phase
        if exist('filtfilt', 'file')
            try
                filtered = filtfilt(b, 1, audio);
            catch
                filtered = bidir_filter(b, [1], audio);
            end
        else
            filtered = bidir_filter(b, [1], audio);
        end
        delay_ms = 0;
    else
        filtered = filter(b, 1, audio);
        delay_ms = (order / 2 / fs) * 1000;
    end

    filtered = single(filtered);
end

function b = design_sinc_fir(cutoff_hz, M, btype, win, fs)
    n = (0:M)' - M / 2;
    zero_idx = (n == 0);
    hd = zeros(size(n));

    if strcmpi(btype, 'low')
        wc = 2 * pi * cutoff_hz(1) / fs;
        hd(zero_idx) = wc / pi;
        hd(~zero_idx) = sin(wc * n(~zero_idx)) ./ (pi * n(~zero_idx));
        b = hd .* win;
        b = b / sum(b); % Unity gain at DC

    elseif strcmpi(btype, 'high')
        wc = 2 * pi * cutoff_hz(1) / fs;
        hd(zero_idx) = 1 - wc / pi;
        hd(~zero_idx) = -sin(wc * n(~zero_idx)) ./ (pi * n(~zero_idx));
        b = hd .* win;
        % Unity gain at Nyquist
        gain_nyq = sum(b .* ((-1).^(0:M)'));
        if abs(gain_nyq) > 1e-4, b = b / gain_nyq; end

    else % bandpass
        wc1 = 2 * pi * cutoff_hz(1) / fs;
        wc2 = 2 * pi * cutoff_hz(2) / fs;
        hd(zero_idx) = (wc2 - wc1) / pi;
        hd(~zero_idx) = (sin(wc2 * n(~zero_idx)) - sin(wc1 * n(~zero_idx))) ./ (pi * n(~zero_idx));
        b = hd .* win;
        % Normalize at center frequency
        w_mid = (wc1 + wc2) / 2;
        center_gain = abs(sum(b .* exp(-1j * w_mid * (0:M)')));
        if center_gain > 1e-4, b = b / center_gain; end
    end
end

function y = bidir_filter(b, a, x)
    % Native forward-backward bidirectional filter fallback for zero-phase
    y_fwd = filter(b, a, x);
    y_rev = flipud(y_fwd);
    y_back = filter(b, a, y_rev);
    y = flipud(y_back);
end
