%% DialectClassification_TRF_SAAVB3.m
% Improved version of DialectClassification_TRF_SAAVB2.m (early fusion kept:
% the 4 rhythm metrics are appended to every frame before the encoder).
%
% Changes relative to SAAVB2:
%  1. Standard pre-norm residuals:  h = x + Attn(LN(x)),  y = h + FFN(LN(h))
%  2. Positional embedding of size d_model (128) instead of 1
%  3. Statistics pooling (mean + std over time) instead of the last frame
%  4. Attention dropout, FFN dropout, L2 weight decay and SpecAugment
%  5. Network with the best validation loss is kept (early stopping)
%  6. Several seeds: mean +/- SD, and an ensemble of their posteriors
%  7. Short files are repeated to one full sequence, so no file is lost
%  8. Rhythm metrics are normalized per file (each file has weight 1)
%
% Choose settings on the VALIDATION accuracy; report the TEST accuracy once.

clearvars; clc; close all;

%% ===================== SETTINGS =====================
corpus      = "SAAVB";
dataFolder  = fullfile('.', 'SAAVB_sp');
numClasses  = 9;

overlapSec  = 0.015;   % 25 ms window, 10 ms hop (as in SAAVB2)
seqLen      = 50;      % frames per sequence (as in SAAVB2)
seqOverlap  = 40;      % overlapping frames between sequences (as in SAAVB2)

P.dModel      = 128;   % model width (numKeyChannels)
P.numHeads    = 4;
P.ffnUnits    = 128;   % FFN hidden units (numhiddenUnits)
P.headUnits   = 128;   % hidden FC in the classification head
P.attnDropout = 0.1;
P.ffnDropout  = 0.1;
P.headDropout = 0.5;

l2          = 1e-3;    % L2 weight decay
maxEpochs   = 50;
learnRate   = 1e-4;
miniBatch   = 64;
patience    = 8;       % validation checks without improvement before stopping
useSpecAug  = true;    % one masked copy of every training sequence
seeds       = 1:5;     % one training run per seed

%% ===================== DATA =====================
parts = ["train","validation","test"];
D = struct();
for p = parts
    D.(p) = loadSplit(dataFolder, p);
    fprintf('%-10s %4d files\n', p, numel(D.(p).files));
end
classes = categories(D.train.labels);
for p = parts, D.(p).labels = setcats(D.(p).labels, classes); end
countcats(D.train.labels)'

%% ===================== FEATURES (cached) =====================
Fs = audioinfo(D.train.files(1)).SampleRate;
useAdv = ~isempty(which('audioAdvFeatureExtractor'));
extractor = [];
if useAdv
    extractor = audioAdvFeatureExtractor( ...
        "SampleRate",Fs, ...
        "Window",hamming(round(0.025*Fs),"periodic"), ...
        "OverlapLength",round(overlapSec*Fs), ...
        "pitch",true, "pitchDelta",true, "pitchDeltaDelta",true, ...
        "energy",true, "energyDelta",true, "energyDeltaDelta",true, ...
        "mfcc",true);
else
    fprintf('audioAdvFeatureExtractor not found: using extractFrameFeatures.m\n');
end

cacheFile = sprintf('features_%s_ovl%dms_adv%d.mat', corpus, round(overlapSec*1000), useAdv);
if isfile(cacheFile)
    load(cacheFile, 'A');
else
    A = struct();
    for p = parts
        fprintf('Extracting features: %s ...\n', p);
        A.(p) = extractSet(D.(p).files, extractor, useAdv, overlapSec);
    end
    save(cacheFile, 'A', '-v7.3');
end

% Normalization statistics from the TRAINING set only:
% acoustic features over all training frames, rhythm over training FILES.
allTr = cat(2, A.train{:});
allTr(~isfinite(allTr)) = nan;
muA = mean(allTr, 2, 'omitnan');  sdA = std(allTr, 0, 2, 'omitnan');
sdA(sdA == 0 | isnan(sdA)) = 1;
clear allTr
muR = mean(D.train.rhythm, 1, 'omitnan');  sdR = std(D.train.rhythm, 0, 1, 'omitnan');
sdR(sdR == 0 | isnan(sdR)) = 1;
nAc = numel(muA);                          % acoustic channels (rhythm adds 4)

seqStep = seqLen - seqOverlap;
S = struct();
for p = parts
    [S.(p).X, S.(p).perFile] = makeSequences(A.(p), D.(p).rhythm, muA, sdA, muR, sdR, seqLen, seqStep);
    S.(p).Y = repelem(D.(p).labels, S.(p).perFile);
    fprintf('%-10s %6d sequences of %d frames (%d channels)\n', p, numel(S.(p).X), seqLen, nAc + 4);
end

%% ===================== TRAINING RUNS =====================
valAcc = zeros(numel(seeds),1); testAcc = valAcc; secs = valAcc;
postSum = 0;
nets = cell(numel(seeds),1);
for s = 1:numel(seeds)
    rng(seeds(s));
    Xtr = S.train.X;  Ytr = S.train.Y;
    if useSpecAug
        Xaug = cellfun(@(x) specAugment(x, nAc), Xtr, 'UniformOutput', false);
        Xtr = [Xtr; Xaug];  Ytr = [Ytr; Ytr];
    end

    lgraph = buildNet(nAc + 4, numClasses, seqLen, P);
    args = {"MaxEpochs",maxEpochs, "MiniBatchSize",miniBatch, ...
        "InitialLearnRate",learnRate, "LearnRateSchedule","piecewise", ...
        "LearnRateDropFactor",0.5, "LearnRateDropPeriod",10, ...
        "L2Regularization",l2, "GradientThreshold",1, ...
        "Shuffle","every-epoch", ...
        "ValidationData",{S.validation.X, S.validation.Y}, ...
        "ValidationFrequency",max(1, floor(numel(Ytr)/miniBatch)), ...
        "ValidationPatience",patience, ...
        "Plots","training-progress", "Verbose",false, ...
        "ExecutionEnvironment","auto"};
    try
        options = trainingOptions("adam", args{:}, "OutputNetwork","best-validation");
    catch
        options = trainingOptions("adam", args{:}, "OutputNetwork","best-validation-loss");
    end

    tic;
    nets{s} = trainNetwork(Xtr, Ytr, lgraph, options);
    secs(s) = toc;

    valAcc(s)        = fileAccuracy(nets{s}, S.validation, D.validation.labels);
    [testAcc(s), pt] = fileAccuracy(nets{s}, S.test, D.test.labels);
    postSum = postSum + pt;
    fprintf('Seed %d: validation = %.2f%%   test = %.2f%%   time = %s\n', seeds(s), ...
        valAcc(s), testAcc(s), char(duration(0,0,secs(s),'Format','hh:mm:ss')));
end

%% ===================== RESULTS (per file) =====================
fprintf('\n%s | %d seeds\n', corpus, numel(seeds));
fprintf('Validation: %.2f%% +/- %.2f%%\n', mean(valAcc), std(valAcc));
fprintf('Test:       %.2f%% +/- %.2f%%\n', mean(testAcc), std(testAcc));

[~, k] = max(postSum, [], 2);
predEns = categorical(classes(k), classes);
ensAcc = 100 * mean(predEns == D.test.labels);
fprintf('Ensemble of %d runs (mean posterior): test = %.2f%%\n', numel(seeds), ensAcc);

figure
cm = confusionchart(D.test.labels, predEns, 'Title', 'Test accuracy per file (ensemble)');
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

R = table(seeds(:), valAcc, testAcc, secs, 'VariableNames', {'Seed','ValAcc','TestAcc','Seconds'});
writetable(R, sprintf('results_%s.csv', corpus));
save(sprintf('trainedNets_%s.mat', corpus), 'nets', 'classes', 'muA', 'sdA', 'muR', 'sdR', 'P', 'R', 'ensAcc');
disp('Done.')

%% ===================== LOCAL FUNCTIONS =====================
function D = loadSplit(loc, part)
% Files of one split with their labels and their own rhythm metrics (TSV).
    ads = audioDatastore(fullfile(loc, part), 'IncludeSubfolders', true);
    tsv = fullfile(loc, part, part + "_audio_info.tsv");
    opts = detectImportOptions(tsv, 'FileType','text', 'PreserveVariableNames',true, ...
        'NumHeaderLines',0, 'Delimiter','\t');
    M = readtable(tsv, opts);
    [~, nm, ex] = fileparts(ads.Files);
    [found, row] = ismember(string(strcat(nm, ex)), string(M.Audio_File));
    assert(all(found), 'Some %s files are missing from the TSV file', part);
    M = M(row, :);
    D.files  = string(ads.Files);
    D.labels = categorical(string(M.dialect));
    D.rhythm = [M.Vper, M.Cper, M.dV, M.dC];
end

function A = extractSet(files, extractor, useAdv, overlapSec)
% Frame features of the speech parts of every file (silence removed,
% speech segments joined), one [channels x frames] matrix per file.
    A = cell(numel(files), 1);
    parfor i = 1:numel(files)
        [x, fs] = audioread(files(i));  x = x(:,1);
        idx = detectSpeech(x, fs);
        if isempty(idx), idx = [1 numel(x)]; end
        F = [];
        for k = 1:size(idx,1)
            F = [F, frameFeatures(x(idx(k,1):idx(k,2)), fs, extractor, useAdv, overlapSec)]; %#ok<AGROW>
        end
        if isempty(F)                                   % very short file: use it whole
            F = frameFeatures(x, fs, extractor, useAdv, overlapSec);
        end
        A{i} = single(F);
    end
end

function f = frameFeatures(seg, fs, extractor, useAdv, overlapSec)
% MFCC (39) + pitch/energy with deltas (6) for one speech segment.
    if useAdv
        if numel(seg) < round(0.025*fs), f = []; return, end
        f = extract(extractor, seg)';
    else
        f = extractFrameFeatures(seg, fs, overlapSec, true, true);
    end
end

function [X, perFile] = makeSequences(A, rhythm, muA, sdA, muR, sdR, seqLen, seqStep)
% Normalize, append the file's rhythm vector as 4 constant channels (early
% fusion) and cut into fixed-length sequences. Files shorter than one
% sequence are repeated, so every file gives at least one sequence.
    X = {};  perFile = zeros(numel(A), 1);
    nAc = numel(muA);
    for i = 1:numel(A)
        F = A{i};
        if isempty(F), F = repmat(muA, 1, 1); end
        Z = (F - muA) ./ sdA;  Z(~isfinite(Z)) = 0;
        r = (rhythm(i,:) - muR) ./ sdR;  r(~isfinite(r)) = 0;
        if size(Z,2) < seqLen
            Z = repmat(Z, 1, ceil(seqLen/size(Z,2)));
        end
        Z = [Z(1:nAc,:); repmat(single(r'), 1, size(Z,2))];
        starts = 1:seqStep:(size(Z,2) - seqLen + 1);
        for st = starts
            X{end+1,1} = Z(:, st:st+seqLen-1); %#ok<AGROW>
        end
        perFile(i) = numel(starts);
    end
end

function X = specAugment(X, nAc)
% Two time masks (up to 10% of the frames) and one frequency mask (up to
% 15% of the acoustic channels); the 4 rhythm channels are left untouched.
    T = size(X,2);
    for m = 1:2
        w = randi([0, max(1, round(0.10*T))]);  t0 = randi([1, max(1, T-w+1)]);
        X(1:nAc, t0:min(T, t0+w-1)) = 0;
    end
    w = randi([0, max(1, round(0.15*nAc))]);  f0 = randi([1, max(1, nAc-w+1)]);
    X(f0:min(nAc, f0+w-1), :) = 0;
end

function lgraph = buildNet(inputSize, numClasses, seqLen, P)
% Projection + positional embedding, one pre-norm transformer block,
% statistics pooling, classification head.
    layers = [
        sequenceInputLayer(inputSize, Name="input")
        fullyConnectedLayer(P.dModel, Name="input_proj")
        positionEmbeddingLayer(P.dModel, seqLen, Name="pos_emb")
        additionLayer(2, Name="embeddings_add")

        layerNormalizationLayer(Name="b1_norm1")
        selfAttentionLayer(P.numHeads, P.dModel, DropoutProbability=P.attnDropout, Name="b1_attn")
        additionLayer(2, Name="b1_add1")
        layerNormalizationLayer(Name="b1_norm2")
        fullyConnectedLayer(P.ffnUnits, Name="b1_ff1")
        reluLayer(Name="b1_relu")
        dropoutLayer(P.ffnDropout, Name="b1_drop")
        fullyConnectedLayer(P.dModel, Name="b1_ff2")
        additionLayer(2, Name="b1_add2")

        layerNormalizationLayer(Name="final_norm")
        functionLayer(@statsPool, Formattable=true, Name="stats_pool")
        dropoutLayer(P.headDropout, Name="head_drop")
        fullyConnectedLayer(P.headUnits, Name="head_fc")
        reluLayer(Name="head_relu")
        fullyConnectedLayer(numClasses, Name="out")
        softmaxLayer(Name="softmax")
        classificationLayer(Name="class")];
    lgraph = layerGraph(layers);
    lgraph = connectLayers(lgraph, "input_proj", "embeddings_add/in2");   % x = proj + pos
    lgraph = connectLayers(lgraph, "embeddings_add", "b1_add1/in2");      % h = x + Attn(LN(x))
    lgraph = connectLayers(lgraph, "b1_add1", "b1_add2/in2");             % y = h + FFN(LN(h))
end

function Y = statsPool(X)
% Mean and standard deviation over time -> [2C x B] ("CB")
    t = finddim(X, "T");
    mu = mean(X, t);
    sd = sqrt(mean((X - mu).^2, t) + 1e-5);
    Y = stripdims(cat(finddim(X, "C"), mu, sd));
    Y = dlarray(reshape(Y, size(Y,1), size(Y,2)), "CB");
end

function [acc, post] = fileAccuracy(net, S, labels)
% File decision = class with the highest mean posterior over its sequences.
% Posterior columns are returned in the order of categories(labels).
    Pr = predict(net, S.X, "MiniBatchSize", 64);
    post = zeros(numel(labels), size(Pr,2));
    c = 0;
    for f = 1:numel(labels)
        post(f,:) = mean(Pr(c + (1:S.perFile(f)), :), 1);
        c = c + S.perFile(f);
    end
    [~, ord] = ismember(categories(labels), string(net.Layers(end).Classes));
    post = post(:, ord);
    [~, k] = max(post, [], 2);
    cats = categories(labels);
    acc = 100 * mean(string(cats(k)) == string(labels(:)));
end
