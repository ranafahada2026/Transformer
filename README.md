Rhythm-aware Transformer for Spoken Dialect Classification (SAAVB & TIMIT)

Description

This repository contains MATLAB code for a rhythm-aware transformer-based dialect classification system. The pipeline includes preprocessing, feature extraction, sequence construction, and classification for both Arabic (SAAVB) and English (TIMIT) speech datasets.

Dataset Information
SAAVB (Arabic)
Dataset: Saudi Accented Arabic Voice Bank
DOI/URL: uploaded.
TIMIT (English)
Dataset: TIMIT Acoustic-Phonetic Continuous Speech Corpus
URL: https://catalog.ldc.upenn.edu/LDC93S1
Code Information
Preporcess_SAAVB.m → Prepares SAAVB dataset
Preprocess_TIMIT.m → Prepares TIMIT dataset
DialectClassification_TRF_SAAVB2.m → Transformer model for SAAVB
DialectClassification_TRF_TIMIT4.m → Transformer model for TIMIT
Usage Instructions
Step 1: Prepare datasets
Place SAAVB in: ./SAAVB/
Place TIMIT in: ./Timit/
Ensure file: Rythm Metric Featuers.xlsx is in working directory
Step 2: Preprocess data

Run:
Preporcess_SAAVB
Preprocess_TIMIT
Step 3: Train models
DialectClassification_TRF_SAAVB2
DialectClassification_TRF_TIMIT4
Requirements
MATLAB
Audio Toolbox
Deep Learning Toolbox
Parallel Computing Toolbox (optional)
Methodology
Data preprocessing and splitting
Speech segmentation using detectSpeech
Feature extraction (pitch, energy, MFCCs)
Rhythm feature integration
Feature normalization
Sequence construction
Transformer-based classification
Evaluation (segment-level and file-level)
Reproducibility


