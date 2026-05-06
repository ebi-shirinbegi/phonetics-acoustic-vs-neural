# Acoustic and Neural Representations in a Phonetically Aligned Speech Corpus

A research project for **Advanced Statistics**, M1 Computational Linguistics,
Université Paris Cité — Academic year 2025–2026.

Author: Mohammad Ebrahim **SHARIFI**.
Instructor: Prof. Guillaume **Wisniewski**.

## What the project does

I compare two families of speech representations on the
[Russian–French Interference Corpus](https://www.ortolang.fr/market/corpora/ru-fr_interference)
(19 speakers, 12 French target words, 6 repetitions each, ~7 300 phoneme
tokens):

- **Acoustic** features extracted with Praat (`parselmouth`):
  F1, F2, F3 at midpoint, plus 25 % / 75 % trajectory points, mean f0,
  duration, spectral centre of gravity.
- **Neural** representations from two pre-trained encoders:
  Whisper-medium (layers 6 and 20) and XLS-R-large (layers 3, 12 and 20).

Across nine sections I run descriptive statistics, hypothesis tests
(t / Mann-Whitney / permutation, Mantel), nearest-centroid classification,
linear mixed-effects models, ROPE analysis, and hierarchical clustering.

## Repository layout
src/                     # All analysis scripts, one per pipeline stage
data/                    # DVC-tracked outputs (csv, npz)
figures/                 # DVC-tracked plots (png)
corpus_raw/              # Original corpus (not tracked)
dvc.yaml                 # Pipeline definition
pixi.toml                # Reproducible environment

## Reproducing the project

I use [pixi](https://pixi.sh) for the environment and
[DVC](https://dvc.org) for the pipeline.

```bash
# 1. Set up the environment
pixi install

# 2. Reproduce the entire pipeline from raw audio to figures
pixi run dvc repro
```

The full pipeline takes about 35 minutes on an Apple M4 (most of it is
neural feature extraction on MPS).

## Pipeline stages

1. `parse`     — read TextGrid files, build phoneme manifest
2. `clean`     — drop annotation artefacts, fold diacritics
3. `acoustic`  — Praat features per phoneme token
4. `neural`    — Whisper + XLS-R embeddings (5 layers)
5. `normalise` — Lobanov, PCA, UMAP
6. `analyse_5_1`–`analyse_9` — eight analysis stages, one per section

## Outputs

All numerical results live as `.csv` in `data/section*` and all figures as
`.png` in `figures/`. They feed directly into the report.