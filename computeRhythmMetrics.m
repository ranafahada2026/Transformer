function [m, info] = computeRhythmMetrics(phnFile, corpus, sampleRate)
%COMPUTERHYTHMMETRICS  Utterance-level rhythm metrics from ONE phonetic
%segmentation file (Ramus, Nespor & Mehler, 1999).
%
%   m = computeRhythmMetrics(phnFile, corpus, sampleRate) returns a 1x4
%   vector [%V %C dV dC] computed only from the segmentation of this
%   utterance. No class label is used anywhere in this function.
%
%   phnFile    : path to a .PHN/.phn file with lines "start end label"
%   corpus     : "TIMIT" or "SAAVB" (selects the phone-class mapping)
%   sampleRate : rate (Hz) of the integer sample indices in the file
%                (TIMIT: 16000; SAAVB: the WAV rate, 8000). Files whose
%                times contain a decimal point are read as seconds.
%
%   Procedure
%     1. Every phone is mapped to V (vocalic), C (consonantal) or P (pause).
%     2. Pauses and unlabeled segments are excluded and break intervals.
%     3. Adjacent phones of the same class are merged into one interval.
%     4. %V = 100 * total vocalic duration / (vocalic + consonantal duration)
%        %C = 100 - %V   (so %V + %C = 100 by construction)
%        dV, dC = standard deviation of vocalic / consonantal interval
%        durations, in milliseconds.
%
%   info returns the interval counts and the per-phone classes so the
%   mapping can be checked.

    [t0, t1, lab, isSeconds] = readPhn(phnFile);
    if isSeconds
        scale = 1000;                 % s -> ms
    else
        scale = 1000 / sampleRate;    % samples -> ms
    end
    dur = (t1 - t0) * scale;

    cls = repmat('P', numel(lab), 1);
    for k = 1:numel(lab)
        cls(k) = phoneClass(lab{k}, corpus);
    end

    % Merge adjacent phones of the same class; pauses break intervals.
    ivClass = '';  ivDur = [];  prev = 'P';
    for k = 1:numel(lab)
        c = cls(k);
        if c == 'P'
            prev = 'P';
            continue
        end
        if c == prev
            ivDur(end) = ivDur(end) + dur(k);
        else
            ivClass(end+1) = c;         %#ok<AGROW>
            ivDur(end+1)   = dur(k);    %#ok<AGROW>
        end
        prev = c;
    end

    V = ivDur(ivClass == 'V');
    C = ivDur(ivClass == 'C');
    total = sum(V) + sum(C);
    Vper = 100 * sum(V) / total;
    Cper = 100 * sum(C) / total;
    dV = NaN; dC = NaN;
    if numel(V) > 1, dV = std(V); end
    if numel(C) > 1, dC = std(C); end
    m = [Vper, Cper, dV, dC];

    info.nVocalicIntervals     = numel(V);
    info.nConsonantalIntervals = numel(C);
    info.nUnlabeledSegments    = sum(cellfun(@isempty, lab));
    info.labels  = lab;
    info.classes = cls;
end

% -------------------------------------------------------------------------
function [t0, t1, lab, isSeconds] = readPhn(phnFile)
    txt   = fileread(phnFile);
    lines = regexp(txt, '\r?\n', 'split');
    t0 = []; t1 = []; lab = {}; isSeconds = false;
    for k = 1:numel(lines)
        p = strsplit(strtrim(lines{k}));
        if numel(p) < 2 || isempty(p{1}), continue, end
        a = str2double(p{1});  b = str2double(p{2});
        if isnan(a) || isnan(b), continue, end
        if contains(p{1}, '.') || contains(p{2}, '.'), isSeconds = true; end
        t0(end+1,1) = a;  t1(end+1,1) = b;          %#ok<AGROW>
        if numel(p) >= 3, lab{end+1,1} = p{3};      %#ok<AGROW>
        else,             lab{end+1,1} = '';        %#ok<AGROW>
        end
    end
    if isempty(t0)
        error('computeRhythmMetrics:noTimes', ...
            'No time-aligned segments found in %s', phnFile);
    end
end

% -------------------------------------------------------------------------
function c = phoneClass(label, corpus)
    l = lower(strtrim(label));
    if isempty(l), c = 'P'; return, end
    switch upper(string(corpus))
        case "TIMIT"
            % TIMIT phone set (Garofolo et al., 1993).
            pauses = {'h#','pau','epi'};
            vowels = {'iy','ih','eh','ey','ae','aa','aw','ay','ah','ao', ...
                      'oy','ow','uh','uw','ux','er','ax','ix','axr','ax-h'};
            % Syllabic consonants (el, em, en, eng) are treated as
            % consonantal here; move them to 'vowels' if you prefer.
            if any(strcmp(l, pauses)),     c = 'P';
            elseif any(strcmp(l, vowels)), c = 'V';
            else,                          c = 'C';   % incl. closures
            end
        case "SAAVB"
            % SAAVB labels: phone symbol + 2-3 code characters, e.g.
            % as10 (short /a/), as21 (long /a/), is10, us21, ls10, hz11.
            % Vowel labels start with a, i or u.
            if any(strcmp(l, {'sil','sil#','sill','sp'})) || ...
                    startsWith(l, 'sil') || startsWith(l, 'nsp')
                c = 'P';
            elseif any(l(1) == 'aiu')
                c = 'V';
            else
                c = 'C';
            end
        otherwise
            error('Unknown corpus "%s"', corpus);
    end
end
