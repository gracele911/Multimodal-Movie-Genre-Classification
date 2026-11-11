# Multimodal Movie Genre Classification

## Overview

This individual project investigates multi-label movie
genre classification using plot summaries and poster
images from the Simplified MM-IMDb dataset.

The objective is to determine whether combining text
and visual information improves genre classification
compared with unimodal baselines.

Four approaches are implemented:

1. Text-only: DistilBERT + MLP
2. Image-only: CLIP Vision + MLP
3. Early Fusion: concatenated embeddings + joint MLP
4. Late Fusion: weighted probability averaging

## Dataset

The Simplified MM-IMDb dataset contains:

- 25,959 movie records
- Plot summaries
- Poster images
- 23 binary genre labels
- Multiple genres per movie

The dataset files are:

- `data.npy`
- `images.npz`

Raw data is not included in this repository.

Obtain the original dataset from the course-provided
source. Verify the data licence before redistribution.

## Methodology

### Text Feature Extraction

- Encoder: `distilbert/distilbert-base-uncased`
- Encoder frozen during feature extraction
- Maximum sequence length: 256 tokens
- Representation: first-token hidden state
- Embedding dimensions: 768

### Image Feature Extraction

- Encoder: `openai/clip-vit-base-patch32`
- Frozen CLIP Vision encoder
- CLIP image preprocessing
- Representation: `CLIPVisionModel.pooler_output`
- Embedding dimensions: 768

The vision pooler output is used rather than
the projected 512-dimensional CLIP embedding.

### Classification Models

Each MLP uses:

- Hidden layer: 256 neurons
- ReLU activation
- Dropout: 0.3
- Output layer: 23 logits
- Loss: BCEWithLogitsLoss
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 256
- Maximum epochs: 20
- Early stopping patience: 4

Early Fusion concatenates text and image embeddings
into a 1,536-dimensional feature vector.

Late Fusion combines independently predicted
probabilities using validation-selected weights.

## Data Splitting

A duplicate-aware, group-based splitting strategy
was used to reduce leakage between partitions.

- Development set: 18,176 movies
- Independent holdout test: 7,783 movies

Five group-aware cross-validation folds were used
on the development set.

Each outer training partition was further split
into model-fitting and inner-validation subsets.

Feature scalers were fitted only on model-fitting data.

Decision thresholds and late-fusion weights were
selected using inner-validation data.

The independent test set was not used for tuning.

The exact saved split files are included in `splits/`.

## Results

### Five-Fold Cross-Validation

| Model | Macro F1 (Mean ± SD) | Micro F1 |
|---|---:|---:|
| Text-only | 0.5749 ± 0.0053 | 0.6430 |
| Image-only | 0.2896 ± 0.0079 | 0.4523 |
| Early Fusion | 0.5825 ± 0.0077 | 0.6587 |
| Late Fusion | 0.5774 ± 0.0069 | 0.6474 |

### Independent Holdout Test

| Model | Macro F1 | Micro F1 |
|---|---:|---:|
| Text-only | 0.5701 | 0.6240 |
| Image-only | 0.2858 | 0.4688 |
| Early Fusion | 0.5788 | 0.6473 |
| Late Fusion | 0.5814 | 0.6376 |

Early Fusion was selected as the primary model
based on development cross-validation Macro F1.

Late Fusion achieved the highest observed test
Macro F1, while Early Fusion achieved the highest
test Micro F1.

The improvements over Text-only were modest.

## Final Model Calibration

A separate group-aware calibration split was
created within the development dataset.

- Final training: 16,359 movies
- Calibration: 1,817 movies

Final classification thresholds:

| Model | Threshold |
|---|---:|
| Text-only | 0.175 |
| Image-only | 0.175 |
| Early Fusion | 0.225 |
| Late Fusion | 0.200 |

Late Fusion weights:

- Text: 0.80
- Image: 0.20

These settings were selected before test evaluation.

## Per-Genre Results

Early Fusion improved F1 for 16 of the 23 genres
compared with Text-only.

Late Fusion improved F1 for 18 of the 23 genres.

Animation showed the largest improvement under
Early Fusion.

Rare genres, particularly Film-Noir, remained
difficult to classify.

Detailed genre-level results are available in:

`results/metrics/per_genre_f1_comparison.csv`

## Error Analysis

Early Fusion reduced total genre-label errors
compared with Text-only.

| Model | Total Label Errors | Exact Match Accuracy |
|---|---:|---:|
| Text-only | 18,498 | 0.0794 |
| Image-only | 27,319 | 0.0245 |
| Early Fusion | 15,771 | 0.1190 |
| Late Fusion | 17,188 | 0.0914 |

Early Fusion mainly reduced false positives,
while also introducing some additional false negatives.

Six qualitative error cases were selected to
illustrate improvements and degradations.

## Repository Structure

```text
multimodal-movie-genre-classification/
├── notebooks/
│   ├── movie_preprocessing.ipynb
│   ├── movie_feature_extraction.ipynb
│   └── movie_fusion_evaluation.ipynb
├── splits/
│   ├── group_ids.npy
│   ├── split_indices.npz
│   └── cv_fold_indices.npz
├── results/
│   ├── metrics/
│   └── figures/
├── README.md
├── requirements.txt
└── .gitignore
```

## How to Reproduce

### 1. Environment

The experiments were implemented in Python,
using Google Colab and Google Drive.

Install the dependencies:

```bash
pip install -r requirements.txt
```

GPU acceleration is recommended for encoder
feature extraction.

### 2. Prepare Data

Obtain the original dataset files:

- `data.npy`
- `images.npz`

Place them in a Google Drive project directory.

The original implementation uses:

```text
/content/drive/MyDrive/Multi Modal - Project 2.2
```

Update the notebook paths if using a different
directory or a local Jupyter environment.

Copy the contents of `splits/` into the project
directory to preserve the original data partitions.

### 3. Execute Notebooks

Recommended order:

1. `movie_preprocessing.ipynb`
2. `movie_feature_extraction.ipynb`
3. `movie_fusion_evaluation.ipynb`

The notebooks retain research-stage experiments and are written for Google Colab. Update `PROJECT_DIR` when using another Drive directory. The preprocessing notebook also supports the `MOVIE_PROJECT_DIR` environment variable.

When reproducing the reported experiments, retain the supplied split files rather than generating new random partitions. The preprocessing notebook contains exploratory split-generation cells; if those cells are rerun, use the saved files in `splits/` for the exact reported partitions.

The feature extraction notebook produces:

- `text_embeddings.npy`
- `image_embeddings.npy`

The evaluation notebook generates trained
classifiers, calibration predictions, performance
tables and figures.

### 4. Expected Outputs

The `results/` directory contains:

- Cross-validation summaries
- Independent test metrics
- Per-genre performance
- Classification error analysis
- Comparison figures

Large datasets, cached embeddings and checkpoints
are intentionally excluded from GitHub.

## Limitations

- Strong class imbalance across 23 genres
- Weak standalone performance of poster features
- Frozen pretrained feature encoders
- Possible information loss from center cropping
- Variable poster quality
- Modest improvements from multimodal fusion
- Higher classifier capacity in Early Fusion

These are potential contributing factors.
No ablation experiment established a single cause.

## Reproducibility Notes

- Main experimental seed: 42
- Final training seed: 142
- Group-aware splitting
- Training-only feature standardisation
- Inner-validation threshold optimisation
- Independent holdout test evaluation

The repository preserves the reported experimental
configuration. Minor numerical differences may arise
from library versions, hardware and execution order.

## References

- DistilBERT: https://arxiv.org/abs/1910.01108
- CLIP: https://arxiv.org/abs/2103.00020
