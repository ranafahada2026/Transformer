clearvars
clc
close all
% Specify the path for the Timit folder
baseFolder = '.\Timit';
% CHANGED: find the corpus folder automatically if it is not at this path
if ~isfolder(fullfile(baseFolder, 'DR1'))
    d = dir(fullfile('.', '**'));
    d = d([d.isdir] & strcmpi({d.name}, 'DR1') & ~contains({d.folder}, '_sp'));
    if isempty(d)
        error('Folder "DR1" not found under %s. Put the corpus folder next to this script.', pwd);
    end
    baseFolder = d(1).folder;
end
fprintf('Corpus folder: %s\n', baseFolder);
TimitFolder = '.\Timit_sp';
if ~isfolder(TimitFolder), mkdir(TimitFolder); end

% Split ratios for training, testing, and validation
trainRatio = 0.7;
valRatio = 0.1;
testRatio = 0.2;

% Types of folders required
folderTypes = {};

% Ensure the main storage folders are created
trainFolder = fullfile(TimitFolder, 'train');
testFolder = fullfile(TimitFolder, 'test');
valFolder = fullfile(TimitFolder, 'validation');

if exist(trainFolder, 'dir')
    rmdir(trainFolder, 's');
end
if exist(testFolder, 'dir')
    rmdir(testFolder, 's');
end
if exist(valFolder, 'dir')
    rmdir(valFolder, 's');
end

if ~isfolder(trainFolder), mkdir(trainFolder); end
if ~isfolder(testFolder), mkdir(testFolder); end
if ~isfolder(valFolder), mkdir(valFolder); end

% Open TSV files for writing
trainTsvFile = fullfile(trainFolder, 'train_audio_info.tsv');
testTsvFile = fullfile(testFolder, 'test_audio_info.tsv');
valTsvFile = fullfile(valFolder, 'validation_audio_info.tsv');

trainFileID = fopen(trainTsvFile, 'w');
testFileID = fopen(testTsvFile, 'w');
valFileID = fopen(valTsvFile, 'w');

% CHANGED: rhythm metrics are no longer read from 'Rythm Metric Featuers.xlsx'
% (per-dialect averages looked up with the label). They are computed below
% for EACH file from its own .PHN segmentation by computeRhythmMetrics.m.

% Write headers to TSV files
fprintf(trainFileID, 'Audio_File\tdialect\tVper\tCper\tdV\tdC\n');
fprintf(testFileID, 'Audio_File\tdialect\tVper\tCper\tdV\tdC\n');
fprintf(valFileID, 'Audio_File\tdialect\tVper\tCper\tdV\tdC\n');

% Iterate through all the main folders
MainFolders = dir(baseFolder); % List all directories in baseFolder
for i = 1:length(MainFolders)
    if MainFolders(i).isdir && ~startsWith(MainFolders(i).name, '.')
        MainPath = fullfile(baseFolder, MainFolders(i).name);

        regionFolders = dir(MainPath);
        for j = 1:length(regionFolders)
            if regionFolders(j).isdir && ~startsWith(regionFolders(j).name, '.')
                % Region Folder
                regionPath = fullfile(MainPath, regionFolders(j).name);
                
                % Iterate through the types of folders: audio and text
                for k = 1
                    folderPath = fullfile(regionPath);
                    
                    if isfolder(folderPath)
                        % Get lists of .wav files
                        wavFiles = dir(fullfile(folderPath, '*.wav'));
                        fileNames = {wavFiles.name};
                        
                        % Split the files
                        [trainFiles, testFiles, valFiles] = splitFiles(fileNames, trainRatio, testRatio);
                        
                        % Storage paths for the split files by type
                        trainFolderType = fullfile(trainFolder, 'audio_text'); 
                        testFolderType = fullfile(testFolder, 'audio_text');
                        valFolderType = fullfile(valFolder, 'audio_text');
                        
                        % Ensure the storage folders are created by type
                        if ~isfolder(trainFolderType), mkdir(trainFolderType); end
                        if ~isfolder(testFolderType), mkdir(testFolderType); end
                        if ~isfolder(valFolderType), mkdir(valFolderType); end
                        
                        % Copy files and write to TSV files
                        for fileIdx = 1:numel(trainFiles)
                            srcPath = fullfile(folderPath, trainFiles{fileIdx});
                            destPath = fullfile(trainFolderType, [num2str(i) '_' num2str(j) '_' num2str(fileIdx) '_' MainFolders(i).name '_' regionFolders(j).name '_' trainFiles{fileIdx}]);
                            copyfile(srcPath, destPath);
                            rythmic = fileRhythmTIMIT(srcPath);   % CHANGED: from this file's own .PHN
                            fprintf(trainFileID, '%s\t%s\t%.6f\t%.6f\t%.6f\t%.6f\n', [num2str(i) '_' num2str(j) '_' num2str(fileIdx) '_' MainFolders(i).name '_' regionFolders(j).name '_' trainFiles{fileIdx}],...
                                MainFolders(i).name, rythmic(1), rythmic(2), rythmic(3), rythmic(4));
                        end
                        for fileIdx = 1:numel(testFiles)
                            srcPath = fullfile(folderPath, testFiles{fileIdx});
                            destPath = fullfile(testFolderType, [num2str(i) '_' num2str(j) '_' num2str(fileIdx) '_' MainFolders(i).name '_' regionFolders(j).name '_' testFiles{fileIdx}]);
                            copyfile(srcPath, destPath);
                            rythmic = fileRhythmTIMIT(srcPath);   % CHANGED: from this file's own .PHN
                            fprintf(testFileID, '%s\t%s\t%.6f\t%.6f\t%.6f\t%.6f\n', [num2str(i) '_' num2str(j) '_' num2str(fileIdx) '_' MainFolders(i).name '_' regionFolders(j).name '_' testFiles{fileIdx}],...
                                MainFolders(i).name, rythmic(1), rythmic(2), rythmic(3), rythmic(4));
                        end
                        for fileIdx = 1:numel(valFiles)
                            srcPath = fullfile(folderPath, valFiles{fileIdx});
                            destPath = fullfile(valFolderType, [num2str(i) '_' num2str(j) '_' num2str(fileIdx) '_' MainFolders(i).name '_' regionFolders(j).name '_' valFiles{fileIdx}]);
                            copyfile(srcPath, destPath);
                            rythmic = fileRhythmTIMIT(srcPath);   % CHANGED: from this file's own .PHN
                            fprintf(valFileID, '%s\t%s\t%.6f\t%.6f\t%.6f\t%.6f\n', [num2str(i) '_' num2str(j) '_' num2str(fileIdx) '_' MainFolders(i).name '_' regionFolders(j).name '_' valFiles{fileIdx}],...
                                MainFolders(i).name, rythmic(1), rythmic(2), rythmic(3), rythmic(4));
                        end
                    end
                end
            end
        end
    end
end

% Close the TSV files
fclose(trainFileID);
fclose(testFileID);
fclose(valFileID);

disp('Files have been organized into train, test, and validation folders, and audio file information has been recorded in separate TSV files.');

% CHANGED: rhythm metrics of one TIMIT WAV file from the .PHN file with the
% same name in the same folder (TIMIT .PHN times are samples at 16 kHz).
function r = fileRhythmTIMIT(wavPath)
    [folder, base] = fileparts(wavPath);
    cand = dir(fullfile(folder, [base '.*']));
    cand = cand(endsWith(lower({cand.name}), '.phn'));
    if isempty(cand), error('No .PHN file found for %s', wavPath); end
    r = computeRhythmMetrics(fullfile(cand(1).folder, cand(1).name), "TIMIT", 16000);
end

% Function to split files
function [trainFiles, testFiles, valFiles] = splitFiles(files, trainRatio, testRatio)
    numFiles = numel(files);
    trainCount = round(trainRatio * numFiles);
    testCount = round(testRatio * numFiles);
    
    % Split the files
    trainFiles = files(1:trainCount);
    testFiles = files(trainCount+1:trainCount+testCount);
    valFiles = files(trainCount+testCount+1:end);
end