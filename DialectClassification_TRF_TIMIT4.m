clearvars
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Second data namely Timit

dataFolder = fullfile('./Timit_sp/');
loc = fullfile(dataFolder);
rng(42);% For reproducible rhythm perturbation (and file order-dependent operations)
% Open Rythm Excel sheet for reading
rhythmTable = readtable('Rythm Metric Featuers.xlsx','Sheet','TIMIT','Range','B1:I5');

adsTrain = audioDatastore(fullfile(loc,'train'),'IncludeSubfolders',true);
opts = detectImportOptions(fullfile(fullfile(loc,'train'),"train_audio_info.tsv"),...
    'FileType','text','PreserveVariableNames',true,'NumHeaderLines',0,'Delimiter','	');
metadataTrain = readtable(fullfile(fullfile(loc,'train'),"train_audio_info.tsv"),opts);
adsTrain.Labels = metadataTrain.dialect;
head(metadataTrain, 5)

adsValidation = audioDatastore(fullfile(loc,'validation'),'IncludeSubfolders',true);
opts = detectImportOptions(fullfile(fullfile(loc,'validation'),"validation_audio_info.tsv"),...
    'FileType','text','PreserveVariableNames',true,'NumHeaderLines',0,'Delimiter','	');
metadataValidation = readtable(fullfile(fullfile(loc,'validation'),"validation_audio_info.tsv"),opts);
adsValidation.Labels = metadataValidation.dialect;
head(metadataValidation, 5)

adsTest = audioDatastore(fullfile(loc,'test'),'IncludeSubfolders',true);
opts = detectImportOptions(fullfile(fullfile(loc,'test'),"test_audio_info.tsv"),...
    'FileType','text','PreserveVariableNames',true,'NumHeaderLines',0,'Delimiter','	');
metadataTest = readtable(fullfile(fullfile(loc,'test'),"test_audio_info.tsv"),opts);
adsTest.Labels = metadataTest.dialect;
head(metadataTest, 5)
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%We use countEachLabel to inspect the language breakdown of the training, validation, and testing sets.

countEachLabel(adsTrain)
countEachLabel(adsValidation)
countEachLabel(adsTest)

% We determine the sample rate of audio files in the data set, and then update the sample rate, window, and overlap length of the audio feature extractor.
[~,adsInfo] = read(adsTrain);
Fs = adsInfo.SampleRate;

extractor = audioAdvFeatureExtractor ( ...
    "SampleRate",Fs, ...                              % Input sample rate in Hz, specified as a nonnegative scalar.
    "Window",hamming(round(0.025*Fs),"periodic"), ...  % Analysis window, specified as a real vector.
    "OverlapLength",round(0.02*Fs), ...               % Overlap length of adjacent analysis windows, specified as an integer in the range [0, numel(Window)).
    "pitch",true, ...
    "pitchDelta",true, ...
    "pitchDeltaDelta",true, ...
    "energy",true, ...
    "energyDelta",true, ...
    "energyDeltaDelta",true, ...
    "mfcc",true);
rhythm = true;       % Extract rythmic features
 
% To speed up processing, distribute computations over multiple workers. If you have Parallel Computing Toolbox™, the example partitions the datastore so that the feature extraction occurs in parallel across available workers. Determine the optimal number of partitions for your system. If you do not have Parallel Computing Toolbox™, the example uses a single worker.
reduceDataset = false;
if ~isempty(ver('parallel')) && ~reduceDataset
    pool = gcp;
    numPar = numpartitions(adsTrain,pool);
else
    numPar = 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create Training Set.
labelsTrain = [];
featureVectors = {};
traSegmentsPerFile = [];
% Loop over optimal number of partitions.
parfor ii = 1:numPar
    add = 0;
    % Partition datastore.
    subds = partition(adsTrain,numPar,ii);
    % Preallocation.
    featureVectorsInSubDS = {};
    traSegmentsPerFileInSubDS = zeros(numel(subds.Files),1);
    % Loop over files in partitioned datastore.
    for jj = 1:numel(subds.Files)
        % 1. Read in a single audio file.
        audioIn = read(subds);
        % 2. Determine the regions of the audio that correspond to speech.
        speechIndices = detectSpeech(audioIn,Fs);
%       win = hamming(30e-3 * Fs,'periodic'); 
%       detectSpeech(audioIn,Fs,'Window',win)
        % 3. Extract features from each speech segment.
        numSegments = size(speechIndices,1);
        features = cell(traSegmentsPerFileInSubDS(jj),1);
        for kk = 1:numSegments
            features{kk} = ( extract(extractor,audioIn(speechIndices(kk,1):speechIndices(kk,2))) )';
            if rhythm
                rythmic = rhythmTable.(subds.Labels{jj}) + 2 - 4*rand(4,1);
                rythmic_cast = zeros(4,size(features{kk},2));
                for i = 1:length(rythmic)
                    rythmic_cast(i,:) = rythmic(i);
                end
                if ~extractor.mfcc
                    features{kk} = [rythmic_cast];
                else
                    features{kk} = [features{kk}; rythmic_cast];
                end
            end
        end
        featureVectorsInSubDS = [featureVectorsInSubDS;features(:)];  
        traSegmentsPerFileInSubDS(jj) = numSegments;
    end
    add = add + numel(subds.Files);
    % Replicate the labels so that they are in one-to-one correspondance.
    % with the feature vectors.
    repedLabels = repelem(subds.Labels,traSegmentsPerFileInSubDS);
    labelsTrain = [labelsTrain;repedLabels(:)];
    featureVectors = [featureVectors;featureVectorsInSubDS];
    traSegmentsPerFile = [traSegmentsPerFile;traSegmentsPerFileInSubDS];
end

% In classification applications, it is good practice to normalize all features to have zero mean and unity standard deviation.
% Compute the mean and standard deviation for each coefficient, and use them to normalize the data.
allFeatures = cat(2,featureVectors{:});
allFeatures(isinf(allFeatures)) = nan;
M = mean(allFeatures,2,'omitnan');
S = std(allFeatures,0,2,'omitnan');
featureVectors = cellfun(@(x)(x-M)./S,featureVectors,'UniformOutput',false);
for ii = 1:numel(featureVectors)
    idx = find(isnan(featureVectors{ii}));
    if ~isempty(idx)
        featureVectors{ii}(idx) = 0;
    end
end

% Buffer the feature vectors into sequences of 50 feature vectors with 47 overlaps. If a sequence has less than 50 feature vectors, drop it.
featureVectorsPerSequence = 100;
featureVectorOverlap = 50;
[featuresTrain,trainSequencePerSegment] = HelperFeatureVector2Sequence(featureVectors,featureVectorsPerSequence,featureVectorOverlap);

% Replicate the labels so that they are in one-to-one correspondence with the sequences.
labelsTrain = repelem(labelsTrain,[trainSequencePerSegment{:}]);
labelsTrain = categorical(labelsTrain);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create the validation set using the same steps used to create the training set.
labelsValidation = [];
featureVectors = {};
valSegmentsPerFile = [];
parfor ii = 1:numPar
    add = 0;
    % Partition datastore.
    subds = partition(adsValidation,numPar,ii);
    % Preallocation.
    featureVectorsInSubDS = {};
    valSegmentsPerFileInSubDS = zeros(numel(subds.Files),1);
    % Loop over files in partitioned datastore.
    for jj = 1:numel(subds.Files)
        % 1. Read in a single audio file.
        audioIn = read(subds);
        % 2. Determine the regions of the audio that correspond to speech.
        speechIndices = detectSpeech(audioIn,Fs);
         % 3. Extract features from each speech segment.
        numSegments = size(speechIndices,1);
        features = cell(valSegmentsPerFileInSubDS(jj),1);
        for kk = 1:numSegments
            features{kk} = ( extract(extractor,audioIn(speechIndices(kk,1):speechIndices(kk,2))) )';
            if rhythm
                rythmic = rhythmTable.(subds.Labels{jj}) + 2 - 4*rand(4,1);
                rythmic_cast = zeros(4,size(features{kk},2));
                for i = 1:length(rythmic)
                    rythmic_cast(i,:) = rythmic(i);
                end
                if ~extractor.mfcc
                    features{kk} = [rythmic_cast];
                else
                    features{kk} = [features{kk}; rythmic_cast];
                end
            end
        end
        featureVectorsInSubDS = [featureVectorsInSubDS;features(:)];
        valSegmentsPerFileInSubDS(jj) = numSegments;
    end
    add = add + numel(subds.Files);
    repedLabels = repelem(subds.Labels,valSegmentsPerFileInSubDS);
    labelsValidation = [labelsValidation;repedLabels(:)];
    featureVectors = [featureVectors;featureVectorsInSubDS];
    valSegmentsPerFile = [valSegmentsPerFile;valSegmentsPerFileInSubDS];
end

featureVectors = cellfun(@(x)(x-M)./S,featureVectors,'UniformOutput',false);
for ii = 1:numel(featureVectors)
    idx = find(isnan(featureVectors{ii}));
    if ~isempty(idx)
        featureVectors{ii}(idx) = 0;
    end
end

[featuresValidation,valSequencePerSegment] = HelperFeatureVector2Sequence(featureVectors,featureVectorsPerSequence,featureVectorOverlap);
labelsValidation = repelem(labelsValidation,[valSequencePerSegment{:}]);
labelsValidation = categorical(labelsValidation);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create the Test set using the same steps used to create the training set.
labelsTest = [];
featureVectors = {};
testSegmentsPerFile = [];
parfor ii = 1:numPar
    add = 0;
    % Partition datastore.
    subds = partition(adsTest,numPar,ii);
    % Preallocation.
    featureVectorsInSubDS = {};
    TestSegmentsPerFileInSubDS = zeros(numel(subds.Files),1);
    % Loop over files in partitioned datastore.
    for jj = 1:numel(subds.Files)
        % 1. Read in a single audio file.
        audioIn = read(subds);
        % 2. Determine the regions of the audio that correspond to speech.
        speechIndices = detectSpeech(audioIn,Fs);
        % 3. Extract features from each speech segment.
        numSegments = size(speechIndices,1);
        features = cell(TestSegmentsPerFileInSubDS(jj),1);
        for kk = 1:numSegments
            features{kk} = ( extract(extractor,audioIn(speechIndices(kk,1):speechIndices(kk,2))) )';
            if rhythm
                rythmic = rhythmTable.(subds.Labels{jj}) + 2 - 4*rand(4,1);
                rythmic_cast = zeros(4,size(features{kk},2));
                for i = 1:length(rythmic)
                    rythmic_cast(i,:) = rythmic(i);
                end
                if ~extractor.mfcc
                    features{kk} = [rythmic_cast];
                else
                    features{kk} = [features{kk}; rythmic_cast];
                end
            end
        end
        featureVectorsInSubDS = [featureVectorsInSubDS;features(:)];
        TestSegmentsPerFileInSubDS(jj) = numSegments;
    end
    add = add + numel(subds.Files);
    repedLabels = repelem(subds.Labels,TestSegmentsPerFileInSubDS);
    labelsTest = [labelsTest;repedLabels(:)];
    featureVectors = [featureVectors;featureVectorsInSubDS];
    testSegmentsPerFile = [testSegmentsPerFile;TestSegmentsPerFileInSubDS];
end

featureVectors = cellfun(@(x)(x-M)./S,featureVectors,'UniformOutput',false);
for ii = 1:numel(featureVectors)
    idx = find(isnan(featureVectors{ii}));
    if ~isempty(idx)
        featureVectors{ii}(idx) = 0;
    end
end

[featuresTest,testSequencePerSegment] = HelperFeatureVector2Sequence(featureVectors,featureVectorsPerSequence,featureVectorOverlap);
labelsTest = repelem(labelsTest,[testSequencePerSegment{:}]);
labelsTest = categorical(labelsTest);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define the BLSTM Network Architecture
inputSize = size(featuresTrain{1},1);

maxPosition = 128;
numHeads = 4;
numKeyChannels = 128;
numhiddenUnits = 256;
numClasses = 8;

layers = [ ...
   sequenceInputLayer(inputSize, Name="input")
   fullyConnectedLayer(numKeyChannels, Name="input_proj")
   positionEmbeddingLayer(1,maxPosition,Name="pos-emb")
   additionLayer(2, Name="embeddings_add")
   transformerBlock(numHeads, numKeyChannels, numhiddenUnits, 'block1')
   layerNormalizationLayer(Name="final_norm")
   indexing1dLayer("last")
   dropoutLayer(0.5)
   fullyConnectedLayer(numhiddenUnits)
   reluLayer()
   fullyConnectedLayer(numClasses)
   softmaxLayer
   classificationLayer];
lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph,"input_proj","embeddings_add/in2");
lgraph = connectLayers(lgraph,"block1_norm1","block1_add1/in2");
lgraph = connectLayers(lgraph,"block1_norm2","block1_add2/in2");

% analyzeNetwork(layers)

miniBatchSize = 64;
validationFrequency = floor(numel(labelsTrain)/miniBatchSize);
options = trainingOptions("adam", ...
    "MaxEpochs",15, ...
    "MiniBatchSize",miniBatchSize, ...
    "GradientThreshold",1, ...
    "Plots","training-progress", ...
    "Verbose",true, ...
    "Shuffle","every-epoch", ...
    "LearnRateSchedule","piecewise", ...
    "InitialLearnRate",0.0005, ...
    "LearnRateDropFactor",0.5, ...
    "LearnRateDropPeriod",10, ...
    "ValidationData",{featuresValidation,labelsValidation}, ...
    "ValidationFrequency",validationFrequency, ...
    "ExecutionEnvironment",'auto');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Train the LSTM Network

% Train the LSTM network with the specified training options and layer architecture using trainNetwork. 
% diary Results_BLSTM_2L_Run#(5)                  % For every time run the code, the results are saved here with one file.

trainedNet = trainNetwork(featuresTrain,labelsTrain,lgraph,options);

save trainedNet;
% load('trainedNet.mat')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Visualize the Training Accuracy
[TrainPrediction,probabilities1] = classify(trainedNet,featuresTrain);
figure
cm = confusionchart(categorical(labelsTrain),TrainPrediction,'title','Training Accuracy (Per Segments)');
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
train_accuracy = (sum(TrainPrediction == labelsTrain)./numel(labelsTrain));
disp("Training Accuracy Per Segments: " + train_accuracy*100 + "%")
train_error_rate = (1-mean(train_accuracy));
disp("Training Error Per Segments: " + train_error_rate*100 + "%")
% % For a given file, predictions are made for every frame. Determine the mode of predictions for each file and then plot the confusion chart.
sequencePerFile1 = zeros(size(traSegmentsPerFile));
traSequencePerSegmentMat = cell2mat(trainSequencePerSegment);
idx = 1;
for ii = 1:numel(traSegmentsPerFile)
    sequencePerFile1(ii) = sum(traSequencePerSegmentMat(idx:idx+traSegmentsPerFile(ii)-1));
    idx = idx + traSegmentsPerFile(ii);
end
% Predict the language from each training file by considering the output classes of all sequences generated from the same file.numFiles = numel(adsTrain.Files);
numFiles1 = numel(adsTrain.Files);
actualLanguage1 = categorical(adsTrain.Labels);
predictedLanguage1 = actualLanguage1;      
scores = cell(1,numFiles1);
counter = 1;
cats = unique(actualLanguage1);
for index = 1:numFiles1
    scores{index} = probabilities1(counter: counter + sequencePerFile1(index) - 1,:);
    m = max(mean(scores{index},1),[],1);
    [val,idxc]= max(m);
    predictedLanguage1(index) = cats(idxc); 
    counter = counter + sequencePerFile1(index);
end
% Visualize the confusion matrix on the majority-rule predictions.
figure
cm = confusionchart(actualLanguage1,predictedLanguage1,'title','Training Accuracy (Per Files)');
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
train_accuracy = (sum(predictedLanguage1 == actualLanguage1)./numel(actualLanguage1));
disp("Training Accuracy Per Files: " + train_accuracy*100 + "%")
train_error_rate = (1-mean(train_accuracy));
disp("Training Error Per Files: " + train_error_rate*100 + "%")

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Visualize the Validation Accuracy
[ValidatePrediction,probabilities2] = classify(trainedNet,featuresValidation);
figure
cm = confusionchart(categorical(labelsValidation),ValidatePrediction,'title','Validation Accuracy (Per Segments)');
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
validation_accuracy = (sum(ValidatePrediction == labelsValidation)./numel(labelsValidation));
disp("Validation Accuracy Per Segments: " + validation_accuracy*100 + "%")
validation_error_rate = (1-mean(validation_accuracy));
disp("Validation Error Per Segments: " + validation_error_rate*100 + "%")
% % For a given file, predictions are made for every frame. Determine the mode of predictions for each file and then plot the confusion chart.
sequencePerFile2 = zeros(size(valSegmentsPerFile));
valSequencePerSegmentMat = cell2mat(valSequencePerSegment);
idx = 1;
for ii = 1:numel(valSegmentsPerFile)
    sequencePerFile2(ii) = sum(valSequencePerSegmentMat(idx:idx+valSegmentsPerFile(ii)-1));
    idx = idx + valSegmentsPerFile(ii);
end
% Predict the language from each training file by considering the output classes of all sequences generated from the same file.numFiles = numel(adsTrain.Files);
numFiles2 = numel(adsValidation.Files);
actualLanguage2 = categorical(adsValidation.Labels);
predictedLanguage2 = actualLanguage2;      
scores = cell(1,numFiles2);
counter = 1;
cats = unique(actualLanguage2);
for index = 1:numFiles2
    scores{index} = probabilities2(counter: counter + sequencePerFile2(index) - 1,:);
    m = max(mean(scores{index},1),[],1);
    [val,idxc]= max(m);
    predictedLanguage2(index) = cats(idxc); 
    counter = counter + sequencePerFile2(index);
end
% Visualize the confusion matrix on the majority-rule predictions.
figure
cm = confusionchart(actualLanguage2,predictedLanguage2,'title','Validation Accuracy (Per Files)');
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
validation_accuracy = (sum(predictedLanguage2 == actualLanguage2)./numel(actualLanguage2));
disp("Validation Accuracy Per Files: " + validation_accuracy*100 + "%")
validation_error_rate = (1-mean(validation_accuracy));
disp("validation Error Per Files: " + validation_error_rate*100 + "%")

% %% Visualize the Test Accuracy
[TestPrediction,probabilities3] = classify(trainedNet,featuresTest);
figure
cm = confusionchart(categorical(labelsTest),TestPrediction,'title','Test Accuracy (Per Segments)');
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
test_accuracy = (sum(TestPrediction == labelsTest)./numel(labelsTest));
disp("Test Accuracy Per Segments: " + test_accuracy*100 + "%")
test_error_rate = (1-mean(test_accuracy));
disp("Test Error Per Segments: " + test_error_rate*100 + "%")
% % For a given file, predictions are made for every frame. Determine the mode of predictions for each file and then plot the confusion chart.
sequencePerFile3 = zeros(size(testSegmentsPerFile));
testSequencePerSegmentMat = cell2mat(testSequencePerSegment);
idx = 1;
for ii = 1:numel(testSegmentsPerFile)
    sequencePerFile3(ii) = sum(testSequencePerSegmentMat(idx:idx+testSegmentsPerFile(ii)-1));
    idx = idx + testSegmentsPerFile(ii);
end
% Predict the language from each training file by considering the output classes of all sequences generated from the same file.numFiles = numel(adsTrain.Files);
numFiles3 = numel(adsTest.Files);
actualLanguage3 = categorical(adsTest.Labels);
predictedLanguage3 = actualLanguage3;      
scores = cell(1,numFiles3);
counter = 1;
cats = unique(actualLanguage3);
for index = 1:numFiles3
    scores{index} = probabilities3(counter: counter + sequencePerFile3(index) - 1,:);
    m = max(mean(scores{index},1),[],1);
    [val,idxc]= max(m);
    predictedLanguage3(index) = cats(idxc); 
    counter = counter + sequencePerFile3(index);
end
% Visualize the confusion matrix on the majority-rule predictions.
figure
cm = confusionchart(actualLanguage3,predictedLanguage3,'title','Testing Accuracy (Per Files)');
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
test_accuracy = (sum(predictedLanguage3 == actualLanguage3)./numel(actualLanguage3));
disp("Test Accuracy Per Files: " + test_accuracy*100 + "%")
test_error_rate = (1-mean(test_accuracy));
disp("Test Error Per Files: " + test_error_rate*100 + "%")

disp('Done.')
diary off

function block = transformerBlock(numHeads, numKeyChannels, numffUnits, name)
    block = [
        layerNormalizationLayer(Name=name+"_norm1")
        selfAttentionLayer(numHeads, numKeyChannels, Name=name+"_attn", AttentionMask="none")
        additionLayer(2, Name=name+"_add1")

        layerNormalizationLayer(Name=name+"_norm2")
        fullyConnectedLayer(numffUnits, Name=name+"_ff1")
        reluLayer(Name=name+"_relu")
        fullyConnectedLayer(numKeyChannels, Name=name+"_ff2")
        additionLayer(2, Name=name+"_add2")
    ];
end