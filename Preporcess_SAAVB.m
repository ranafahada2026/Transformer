clearvars
clc
close all
% Specify the path for the SAAVB folder
baseFolder = '.\SAAVB';
% CHANGED: find the corpus folder automatically if it is not at this path
if ~isfolder(fullfile(baseFolder, '41Female'))
    d = dir(fullfile('.', '**'));
    d = d([d.isdir] & strcmpi({d.name}, '41Female') & ~contains({d.folder}, '_sp'));
    if isempty(d)
        error('Folder "41Female" not found under %s. Put the corpus folder next to this script.', pwd);
    end
    baseFolder = d(1).folder;
end
fprintf('Corpus folder: %s\n', baseFolder);
SAAVBFolder = '.\SAAVB_sp';
if ~isfolder(SAAVBFolder), mkdir(SAAVBFolder); end

% Split ratios for training, testing, and validation
trainRatio = 0.7;
valRatio = 0.1;
testRatio = 0.2;

% Types of folders required
folderTypes = {'audio', 'txt'};

% Ensure the main storage folders are created
trainFolder = fullfile(SAAVBFolder, 'train');
testFolder = fullfile(SAAVBFolder, 'test');
valFolder = fullfile(SAAVBFolder, 'validation');

% Delete the folders if they exist, then recreate them
if exist(trainFolder, 'dir'), rmdir(trainFolder, 's'); end
if exist(testFolder, 'dir'), rmdir(testFolder, 's'); end
if exist(valFolder, 'dir'), rmdir(valFolder, 's'); end

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
rng(42);   % reproducible split

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
                switch regionFolders(j).name
                    case 'مكة'
                        region_name = 'makkah';
                    case 'المدينة'
                        region_name = 'madinah';
                    case 'الرياض'
                        region_name = 'riyadh';
                    case 'الشرقية'
                        region_name = 'shargyah';
                    case 'القصيم'
                        region_name = 'qassiem';
                    case 'تبوك'
                        region_name = 'tbook';
                    case 'حائل'
                        region_name = 'hail';
                    case 'عرعر'
                        region_name = 'ararr';
                    case 'عسير'
                        region_name = 'aseir';
                    otherwise
                        region_name = regionFolders(j).name;
                end
                
                % Region Folder
                regionPath = fullfile(MainPath, regionFolders(j).name);
                
                % Iterate through the types of folders: audio and text
                for k = 1:numel(folderTypes)
                    folderPath = fullfile(regionPath, folderTypes{k});
                    
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
                            destPath = fullfile(trainFolderType, [num2str(i) '_' num2str(j) '_' num2str(fileIdx) '_' region_name '_' trainFiles{fileIdx}]);
                            copyfile(srcPath, destPath);
                            rythmic = fileRhythm(regionPath, srcPath);   % CHANGED: from this file's own .PHN
                            fprintf(trainFileID, '%s\t%s\t%.6f\t%.6f\t%.6f\t%.6f\n', [num2str(i) '_' num2str(j) '_' num2str(fileIdx) '_' region_name '_' trainFiles{fileIdx}],...
                                region_name, rythmic(1), rythmic(2), rythmic(3), rythmic(4));
                        end
                        for fileIdx = 1:numel(testFiles)
                            srcPath = fullfile(folderPath, testFiles{fileIdx});
                            destPath = fullfile(testFolderType, [num2str(i) '_' num2str(j) '_' num2str(fileIdx) '_' region_name '_' testFiles{fileIdx}]);
                            copyfile(srcPath, destPath);
                            rythmic = fileRhythm(regionPath, srcPath);   % CHANGED: from this file's own .PHN
                            fprintf(testFileID, '%s\t%s\t%.6f\t%.6f\t%.6f\t%.6f\n', [num2str(i) '_' num2str(j) '_' num2str(fileIdx) '_' region_name '_' testFiles{fileIdx}],...
                                region_name, rythmic(1), rythmic(2), rythmic(3), rythmic(4));
                        end
                        for fileIdx = 1:numel(valFiles)
                            srcPath = fullfile(folderPath, valFiles{fileIdx});
                            destPath = fullfile(valFolderType, [num2str(i) '_' num2str(j) '_' num2str(fileIdx) '_' region_name '_' valFiles{fileIdx}]);
                            copyfile(srcPath, destPath);
                            rythmic = fileRhythm(regionPath, srcPath);   % CHANGED: from this file's own .PHN
                            fprintf(valFileID, '%s\t%s\t%.6f\t%.6f\t%.6f\t%.6f\n', [num2str(i) '_' num2str(j) '_' num2str(fileIdx) '_' region_name '_' valFiles{fileIdx}],...
                                region_name, rythmic(1), rythmic(2), rythmic(3), rythmic(4));
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

% CHANGED: rhythm metrics of one WAV file from its time-aligned .PHN file
% (N<name>.PHN or <name>.PHN in the region's txt/Text folder; the
% SAAVB*_<name>.phn files contain labels only and are skipped).
function r = fileRhythm(regionPath, wavPath)
    [~, base] = fileparts(wavPath);
    cand = dir(fullfile(regionPath, '**', '*.*'));
    cand = cand(~[cand.isdir] & endsWith(lower({cand.name}), '.phn'));
    for c = 1:numel(cand)
        [~, cb] = fileparts(cand(c).name);
        if strcmp(cb, base) || strcmp(cb, ['N' base])
            ai = audioinfo(wavPath);
            r = computeRhythmMetrics(fullfile(cand(c).folder, cand(c).name), "SAAVB", ai.SampleRate);
            return
        end
    end
    error('No time-aligned .PHN file found for %s', wavPath);
end

% Function to split files
function [trainFiles, testFiles, valFiles] = splitFiles(files, trainRatio, testRatio)
    numFiles = numel(files);
    trainCount = round(trainRatio * numFiles);
    testCount = round(testRatio * numFiles);
    % Randomly shuffle the files
    ids = randperm(numFiles);
    
    % Split the files
    trainFiles = files(ids(1:trainCount));
    testFiles = files(ids(trainCount+1:trainCount+testCount));
    valFiles = files(ids(trainCount+testCount+1:end));
end