# Rhythm-aware Transformer for Spoken Dialect Classification (SAAVB & TIMIT)

## Description

This repository contains MATLAB code for a rhythm-aware, transformer-based spoken dialect classifier. It runs on two corpora: Arabic (SAAVB) and English (TIMIT). The pipeline has five stages:

1. Data splitting.
2. Rhythm metrics for each utterance, computed from its own phonetic segmentation.
3. Frame-level acoustic feature extraction.
4. Sequence construction with early fusion of the rhythm metrics.
5. Transformer classification, evaluated per file over several random seeds.

## Dataset Information

### SAAVB (Arabic)
- **Dataset:** Saudi Accented Arabic Voice Bank
- **Developer:** Computer and Electronics Research Institute, King Abdulaziz City for Science and Technology (KACST), Riyadh, Saudi Arabia
- **Content:** 1,033 Saudi speakers from all regions of Saudi Arabia, reading Modern Standard Arabic with a Saudi accent and giving spontaneous answers. The speech is telephone speech.
- **Reference paper DOI:** https://doi.org/10.1016/S1319-1578(08)80004-3
- **Availability:** the corpus is not openly downloadable. It is distributed by KACST.
- **Classes (9 regions):** makkah, madinah, riyadh, shargyah, qassiem, tbook, hail, ararr, aseir
- **Required structure:** `./SAAVB/<speaker group, e.g. 41Female>/<region folder in Arabic>/audio/*.wav` and `.../txt/*.wav`. Each WAV file needs a time-aligned phone file, `<name>.PHN` or `N<name>.PHN`, somewhere under its region folder. Files named `SAAVB*_<name>.phn` contain labels only and are ignored.

### TIMIT (English)
- **Dataset:** TIMIT Acoustic-Phonetic Continuous Speech Corpus
- **URL:** https://catalog.ldc.upenn.edu/LDC93S1
- **Classes (8 dialect regions):** DR1–DR8
- **Required structure:** `./Timit/DR1 … DR8/<speaker>/*.wav`, with the `.PHN` file of the same name in the same folder. TIMIT sample indices are at 16 kHz.

## Code Information

| File | Purpose |
|---|---|
| `Preporcess_SAAVB.m` | Splits SAAVB into train/validation/test (70/10/20, random, `rng(42)`), copies the WAV files and writes the TSV files with the rhythm metrics of each file |
| `Preprocess_TIMIT.m` | The same for TIMIT (70/10/20 per speaker folder, in file order) |
| `computeRhythmMetrics.m` | Computes `%V`, `%C`, `ΔV` and `ΔC` (Ramus, Nespor & Mehler, 1999) for **one** utterance from its `.PHN` segmentation. It does not use the class label |
| `extractFrameFeatures.m` | Frame features using only standard Audio Toolbox functions. It is used when `audioAdvFeatureExtractor` is not on the path |
| `DialectClassification_TRF_SAAVB3.m` | Transformer training and evaluation for SAAVB |
| `DialectClassification_TRF_TIMIT5.m` | Transformer training and evaluation for TIMIT |

Both corpus folders are found automatically if they are not at `./SAAVB` or `./Timit`. The scripts search the working directory for `41Female` or `DR1`.

## Usage Instructions

**Step 1 – Place the datasets**
- SAAVB in `./SAAVB/`
- TIMIT in `./Timit/`
- Put all `.m` files in the working directory or on the MATLAB path.

**Step 2 – Preprocess the data**
```matlab
Preporcess_SAAVB
Preprocess_TIMIT
```
This step creates `./SAAVB_sp/` and `./Timit_sp/`. Each has the subfolders `train/`, `validation/` and `test/`. Each subfolder holds:
- `audio_text/*.wav`
- `<split>_audio_info.tsv`, with the columns `Audio_File  dialect  Vper  Cper  dV  dC`

> **Note:** Each preprocessing run deletes and rebuilds the `_sp` folders. Afterwards, delete the old feature cache (`features_<corpus>_ovl*ms_adv*.mat`). The training script reloads the cache without checking whether the split has changed.

**Step 3 – Train and evaluate**
```matlab
DialectClassification_TRF_SAAVB3
DialectClassification_TRF_TIMIT5
```
Outputs for each corpus:
- `results_<corpus>.csv`: validation and test accuracy and training time for each seed
- `trainedNets_<corpus>.mat`: the trained networks, normalization statistics, hyperparameters and ensemble accuracy
- A confusion chart of the ensemble on the test set

## Requirements

- MATLAB **R2023a or later**. The scripts need `selfAttentionLayer` and `positionEmbeddingLayer`.
- Audio Toolbox (`detectSpeech`, `mfcc`, `pitch`, `audioDatastore`)
- Signal Processing Toolbox (`hamming`)
- Deep Learning Toolbox
- Parallel Computing Toolbox (optional). Without it, `parfor` runs serially.
- `audioAdvFeatureExtractor` (optional). If it is missing, `extractFrameFeatures.m` is used. The cache file name records which extractor was used (`adv0` or `adv1`).

## Methodology

1. **Data splitting:** 70% train, 10% validation and 20% test. Labels come from the folder structure: the dialect region for TIMIT and the city/region for SAAVB.

2. **Rhythm metrics, one set per utterance** (`computeRhythmMetrics.m`):
   - Each phone is mapped to vocalic (V), consonantal (C) or pause (P).
     - TIMIT: vowels are `iy ih eh ey ae aa aw ay ah ao oy ow uh uw ux er ax ix axr ax-h`. Pauses are `h#`, `pau` and `epi`. Closures and syllabic consonants count as consonantal.
     - SAAVB: labels that start with `a`, `i` or `u` are vowels. `sil*`, `sp` and `nsp*` are pauses.
   - Pauses and unlabeled segments are dropped and break intervals. Adjacent phones of the same class are merged into one interval.
   - `%V = 100·ΣV / (ΣV + ΣC)` and `%C = 100 − %V`. `ΔV` and `ΔC` are the standard deviations of the vocalic and consonantal interval durations, in ms.

3. **Speech segmentation:** `detectSpeech` removes silence and the speech segments are joined. A file with no detected speech is used whole.

4. **Acoustic features (45 per frame):** 25 ms Hamming window, plus:
   - 13 MFCCs with Δ and ΔΔ (39 values)
   - Pitch, F0 in the 50–400 Hz range, with Δ and ΔΔ
   - Short-time energy with Δ and ΔΔ

   The hop is 5 ms for TIMIT and 10 ms for SAAVB.

5. **Normalization:** all statistics come from the **training set only**. Acoustic features are z-scored using the mean and SD over all training frames. Rhythm metrics are z-scored using the mean and SD over training files, so each file has weight 1.

6. **Sequence construction (early fusion):** the 4 normalized rhythm metrics of a file are appended to every one of its frames as constant channels (45 + 4 = 49 inputs). Each file is then cut into overlapping sequences. A file shorter than one sequence is repeated, so no file is lost.

   | | TIMIT | SAAVB |
   |---|---|---|
   | Frames per sequence | 100 (0.5 s) | 50 (0.5 s) |
   | Overlap between sequences | 50 frames | 40 frames |

7. **Transformer classifier:**
   - Linear projection to d_model = 128, plus a learned positional embedding
   - One pre-norm transformer block: `h = x + Attn(LN(x))` with 4 heads and attention dropout 0.1, then `y = h + FFN(LN(h))` with ReLU and dropout 0.1
   - Final LayerNorm, then statistics pooling (mean and SD over time)
   - Dropout 0.5, then a hidden FC layer with ReLU, then softmax

   The FFN and head hidden layers have 256 units for TIMIT and 128 for SAAVB.

8. **Training:**
   - Adam optimizer, mini-batch of 64, gradient clipping at 1, L2 weight decay of 1e-3
   - Piecewise learning rate, halved every 10 epochs. TIMIT starts at 5e-4 for 15 epochs; SAAVB starts at 1e-4 for 50 epochs.
   - **SpecAugment:** each training sequence gets one masked copy, with 2 time masks of up to 10% of the frames and 1 frequency mask of up to 15% of the acoustic channels. The rhythm channels are not masked.
   - **Early stopping:** validation once per epoch, patience of 8, and the network with the best validation loss is kept.

9. **Evaluation (file level):** each file is assigned the class with the highest mean posterior over its sequences. The scripts report:
   - Validation and test accuracy as mean ± SD over 5 seeds (`seeds = 1:5`)
   - Test accuracy of the **ensemble**, which averages the posteriors of the 5 networks
   - A confusion chart with row- and column-normalized summaries

## Reproducibility

- **Fixed randomness:** the SAAVB split uses `rng(42)` and the TIMIT split is deterministic, by file order. Each training run uses `rng(seed)` with `seeds = 1:5`. GPU training can still vary slightly between runs.
- **Model selection:** choose all settings on **validation** accuracy. Report test accuracy once, as mean ± SD over seeds, and give the ensemble result separately.
- **Feature cache:** cached features are stored in `features_<corpus>_ovl<ms>ms_adv<0|1>.mat`. Delete this file after re-running preprocessing.
- **Speaker overlap:** the split is made inside each speaker or region folder, so the same speaker can appear in more than one split. For a speaker-independent evaluation, split by speaker instead.
- **Redundant metric:** `%C = 100 − %V`, so `Cper` adds no new information.
## References

- Alghamdi, M., Alhargan, F., Alkanhal, M., Alkhairy, A., Eldesouki, M., & Alenazi, A. (2008). Saudi Accented Arabic Voice Bank. *Journal of King Saud University – Computer and Information Sciences*, 20, 45–64. https://doi.org/10.1016/S1319-1578(08)80004-3
- Ramus, F., Nespor, M., & Mehler, J. (1999). Correlates of linguistic rhythm in the speech signal. *Cognition*, 73(3), 265–292.
- Garofolo, J. S., et al. (1993). *TIMIT Acoustic-Phonetic Continuous Speech Corpus* (LDC93S1). Linguistic Data Consortium.
