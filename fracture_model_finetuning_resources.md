# Fracture Detection Model — Fine-Tuning Resources

## Key Papers
- **Deep Learning in Fracture Detection: A Narrative Review** — https://pmc.ncbi.nlm.nih.gov/articles/PMC7144272/
- **A Deep Learning Framework for Automated Fracture Detection and Localization** (YOLOv8 + calibration) — https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2026.1806133/full
- **Cross-Center Validation of DL Model for MSK Fracture Detection** (multi-site YOLO) — https://www.medrxiv.org/content/10.1101/2024.01.17.24301244.full.pdf
- **Scaphoid Fracture Classification Fine-Tuning** (EfficientNetB1, hyperparameter detail) — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12826633/
- **Ensemble DL Model for Fracture Detection** (fine-tuned ResNet50 + ensemble) — https://www.clinicalradiologyonline.net/article/S0009-9260(24)00419-7/fulltext

## ⭐ RibFrac (Chest/Rib Fracture — CT) — Key Dataset for Rib Fractures
- **Size:** 7,473 annotated traumatic rib fractures from 900 patients (660 CT scans in the public challenge subset: 420 train / 60 tuning / 120 test)
- **Format:** 3D CT volumes (NIfTI) with voxel-level instance segmentation masks; 4 clinical categories (buckle, nondisplaced, displaced, segmental)
- **Tasks:** Detection/instance segmentation track (FROC metric) + classification/grading track (F1 metric)
- **Access:** Free download via the official challenge site — https://ribfrac.grand-challenge.org/ (Grand Challenge account required, no cost)
- **Reference paper (FracNet):** https://www.sciencedirect.com/science/article/pii/S2352396420304825
- **Note:** CT-based, not plain chest X-ray. Public large-scale X-ray-only rib fracture datasets are much rarer.

## Public Datasets & Access (Other Body Regions)

### FracAtlas
- **Size:** 4,083 X-ray images (717 fractured), COCO / VGG / YOLO / Pascal VOC annotation formats
- **Access:** Free, no registration required.
  - Original release (Figshare): https://doi.org/10.6084/m9.figshare.22363012 (~323 MB, direct download)
  - Hugging Face loader (easiest for Python/PyTorch): https://huggingface.co/datasets/yh0701/FracAtlas_dataset
    ```python
    from datasets import load_dataset
    ds = load_dataset("yh0701/FracAtlas_dataset")
    ```
  - Roboflow (pre-converted to YOLOv5/v8/v9/v11/v12, COCO): https://universe.roboflow.com/fracatlas/fracatlas-rtiu9
  - Kaggle mirror: https://www.kaggle.com/datasets/mahmudulhasantasin/fracatlas-original-dataset
- **License:** CC-BY 4.0 — must cite Abedeen et al., *Scientific Data* (2023)

### Bone Fracture Multi-Region X-ray Dataset (Kaggle)
- **Size:** 10,580 images, multi-region classification
- **Access:** Kaggle account required (free); download via web UI or Kaggle API:
  ```bash
  kaggle datasets download -d <dataset-slug>
  ```
  (search "bone fracture multi-region" on Kaggle to get the exact slug)

### Bone Break Classification Dataset (Kaggle)
- **Access:** Same as above — Kaggle account + `kaggle datasets download`
  - https://www.kaggle.com/datasets/pkdarabi/bone-break-classification-image-dataset

### MURA (Stanford)
- **Size:** ~40,561 images / 14,863 studies, upper extremity, normal/abnormal labels
- **Access:** Requires registration — request via Stanford AIMI: https://aimi.stanford.edu/datasets/mura-msk-xrays
  - You fill a short form (research use); a download link is emailed. No cost, no data use agreement signature typically required (varies by dataset version).
  - Note: official **test set** is not public; only train/validation.

### Mendeley Fracture Dataset
- **Size:** 2,511 images (1,211 simple, 1,173 comminuted, 127 normal)
- **Access:** Free direct download from Mendeley Data (search "bone fracture dataset Mendeley"), no registration needed.

