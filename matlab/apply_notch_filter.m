function [filtered, notch_freqs] = apply_notch_filter(audio, notch_freq_hz, q_factor, include_harmonics, max_harmonics, fs, zero_phase)
% APPLY_NOTCH_FILTER Attenuates narrowband periodic hum (50/60 Hz and harmonics).
% Works in base MATLAB without requiring Signal Processing Toolbox.
%
% Inputs:
%   audio             - 1D vector of audio samples
%   notch_freq_hz     - Fundamental interference frequency (default: 50)
%   q_factor          - Quality factor Q = f0 / delta_f (default: 30)
%   include_harmonics - Boolean: filter integer multiples (default: true)
%   max_harmonics     - Number of harmonics to filter up to Nyquist (default: 4)
%   fs                - Sampling frequency in Hz (default: 16000)
%   zero_phase        - Boolean: true for zero-phase (default: true)
%
% Outputs:
%   filtered          - 1D filtered audio vector
%   notch_freqs       - List of filtered notch frequencies

    if nargin < 2, notch_freq_hz = 50.0; end
    if nargin < 3, q_factor = 30.0; end
    if nargin < 4, include_harmonics = true; end
    if nargin < 5, max_harmonics = 4; end
    if nargin < 6, fs = 16000; end
    if nargin < 7, zero_phase = true; end

    nyquist = fs / 2;
    filtered = audio(:);
    notch_freqs = [];

    num_h = 1;
    if include_harmonics
        num_h = max_harmonics;
    end

    for h = 1:num_h
        f0 = notch_freq_hz * h;
        if f0 >= (nyquist - 20)
            break;
        end

        use_iirnotch = false;
        if exist('iirnotch', 'file')
            try
                wo = f0 / nyquist;
                bw = wo / q_factor;
                [b, a] = iirnotch(wo, bw);
                use_iirnotch = true;
            catch
                use_iirnotch = false;
            end
        end

        if ~use_iirnotch
            % Analytical 2nd-order Notch Biquad (Pure Math)
            w0 = 2 * pi * f0 / fs;
            bw_rad = 2 * pi * (f0 / q_factor) / fs;
            beta = tan(bw_rad / 2);
            r = (1 - beta) / (1 + beta);
            
            % Normalized notch biquad coefficients
            b0 = (1 + r) / 2;
            b = [b0, -2 * b0 * cos(w0), b0];
            a = [1, -(1 + r) * cos(w0), r];
        end

        if zero_phase
            if exist('filtfilt', 'file')
                try
                    filtered = filtfilt(b, a, filtered);
                catch
                    filtered = bidir_notch(b, a, filtered);
                end
            else
                filtered = bidir_notch(b, a, filtered);
            end
        else
            filtered = filter(b, a, filtered);
        end

        notch_freqs = [notch_freqs, f0];
    end

    filtered = single(filtered);
end

function y = bidir_notch(b, a, x)
    y_fwd = filter(b, a, x);
    y_rev = flipud(y_fwd);
    y_back = filter(b, a, y_rev);
    y = flipud(y_back);
end
