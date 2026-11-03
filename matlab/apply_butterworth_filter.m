function [filtered, b_out, a_out] = apply_butterworth_filter(audio, cutoff_hz, order, btype, fs, zero_phase)
% APPLY_BUTTERWORTH_FILTER Butterworth IIR filter with native base MATLAB fallback.
%
% Inputs:
%   audio       - 1D vector of audio samples
%   cutoff_hz   - Cutoff frequency in Hz
%   order       - Filter order (default: 4)
%   btype       - 'low', 'high', or 'bandpass'
%   fs          - Sampling frequency in Hz (default: 16000)
%   zero_phase  - Boolean: true for zero-phase forward-backward filtering
%
% Outputs:
%   filtered    - 1D filtered audio vector
%   b_out, a_out- Numerator and denominator polynomial coefficients

    if nargin < 3, order = 4; end
    if nargin < 4, btype = 'high'; end
    if nargin < 5, fs = 16000; end
    if nargin < 6, zero_phase = true; end

    audio = audio(:);
    nyquist = fs / 2;

    use_butter = false;
    if exist('butter', 'file')
        try
            Wn = cutoff_hz / nyquist;
            if strcmpi(btype, 'bandpass')
                [b, a] = butter(order, Wn, 'bandpass');
            elseif strcmpi(btype, 'high')
                [b, a] = butter(order, Wn, 'high');
            else
                [b, a] = butter(order, Wn, 'low');
            end
            use_butter = true;
        catch
            use_butter = false;
        end
    end

    if ~use_butter
        % Analytical Bilinear-Transformed Butterworth Biquad Cascade
        [b, a] = design_butterworth_biquad_cascade(cutoff_hz, order, btype, fs);
    end

    b_out = b;
    a_out = a;

    if zero_phase
        if exist('filtfilt', 'file')
            try
                filtered = filtfilt(b, a, audio);
            catch
                filtered = bidir_iir(b, a, audio);
            end
        else
            filtered = bidir_iir(b, a, audio);
        end
    else
        filtered = filter(b, a, audio);
    end

    filtered = single(filtered);
end

function [b, a] = design_butterworth_biquad_cascade(cutoff_hz, order, btype, fs)
    % Designs cascaded 2nd-order Butterworth biquads via bilinear transform
    fc = cutoff_hz(1);
    omega = tan(pi * fc / fs);
    n_biquads = ceil(order / 2);
    
    b = 1;
    a = 1;
    
    for k = 1:n_biquads
        theta = (2*k - 1) * pi / (2 * (n_biquads * 2));
        q = 1 / (2 * sin(theta));
        
        if strcmpi(btype, 'high')
            D = 1 + (omega / q) + omega^2;
            b_sec = [1, -2, 1] / D;
            a_sec = [1, 2*(omega^2 - 1)/D, (1 - (omega/q) + omega^2)/D];
        else % lowpass
            D = 1 + (omega / q) + omega^2;
            b_sec = (omega^2 * [1, 2, 1]) / D;
            a_sec = [1, 2*(omega^2 - 1)/D, (1 - (omega/q) + omega^2)/D];
        end
        b = conv(b, b_sec);
        a = conv(a, a_sec);
    end
end

function y = bidir_iir(b, a, x)
    y_fwd = filter(b, a, x);
    y_rev = flipud(y_fwd);
    y_back = filter(b, a, y_rev);
    y = flipud(y_back);
end
