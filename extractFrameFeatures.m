function F = extractFrameFeatures(x, fs, overlapSec, withMfcc, withProsody)
%EXTRACTFRAMEFEATURES  Frame-level features with standard Audio Toolbox
%functions only (replaces the missing audioAdvFeatureExtractor).
%
%   F = extractFrameFeatures(x, fs, overlapSec, withMfcc, withProsody)
%   returns a [numFeatures x numFrames] single matrix for one speech segment.
%
%   Window : 25 ms Hamming; hop = 25 ms - overlapSec
%   MFCC   : 13 coefficients + delta + delta-delta (39 values), log energy ignored
%   Prosody: pitch (F0), dF0, ddF0, short-time energy, dE, ddE (6 values)
%            deltas are first differences, x(i) - x(i-1)
%
%   Returns [] if the segment is shorter than one window.

    winLen = round(0.025*fs);
    ovl    = round(overlapSec*fs);
    hop    = winLen - ovl;
    x = x(:);
    if numel(x) < winLen + hop
        F = []; return
    end
    win = hamming(winLen, "periodic");
    nFrames = floor((numel(x) - winLen)/hop) + 1;

    blocks = {};
    if withMfcc
        [c, d, dd] = mfcc(x, fs, "Window", win, "OverlapLength", ovl, ...
                          "NumCoeffs", 13, "LogEnergy", "Ignore");
        blocks{end+1} = [c d dd];
    end
    if withProsody
        f0 = pitch(x, fs, "WindowLength", winLen, "OverlapLength", ovl, "Range", [50 400]);
        idx = (0:nFrames-1)'*hop + (1:winLen);           % frame sample indices
        frames = x(idx) .* win';
        E = sum(frames.^2, 2);
        n = min(numel(f0), numel(E));
        f0 = f0(1:n); E = E(1:n);
        blocks{end+1} = [f0, delta1(f0), delta1(delta1(f0)), E, delta1(E), delta1(delta1(E))];
    end
    n = min(cellfun(@(b) size(b,1), blocks));
    blocks = cellfun(@(b) b(1:n,:), blocks, 'UniformOutput', false);
    F = single([blocks{:}]');
end

function d = delta1(v)
    d = [0; diff(v)];
end
