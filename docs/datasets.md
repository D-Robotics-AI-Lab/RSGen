# Datasets Preparation

## 1. Data Processing
Our data processing pipeline follows the approach used in **[CC-Diff](https://github.com/AZZMM/CC-Diff/blob/main/data_tools/data_process.md)**. For more details, please refer to the original work.


---

## 2. DIOR-RSVG Dataset
**Directory Structure:**
```text
DIOR-RSVG/
├── Annotations/  # Contains XML annotation files
├── train/        # Training set images
├── val/          # Validation set images
└── test/         # Test set images
└── rsvg_emb.pt     # Pre-computed embeddings 
```

## 3. DOTA Dataset
**Directory Structure:**
```text
DOTA/
├── filter_train/
│   ├── images/     # Training images
│   └── labelTxt/   # Training labels (TXT files)
├── filter_val/
│   ├── images/     # Validation images
│   └── labelTxt/   # Validation labels (TXT files)
└── dota_emb.pt     # Pre-computed embeddings
```
