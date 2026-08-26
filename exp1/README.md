# Experiment 1: Structured Disease Classification

Experiment 1 evaluates how well frozen foundation-model image representations support supervised prediction of the 14 structured MIMIC-CXR disease labels. It is the primary linear-probing and classifier-head experiment described in the project proposal and report. The foundation models are not fine-tuned; only downstream classifier heads are trained.

The experiment compares six individual representations and an Early Fusion representation:

```text
MedSigLIP, BioViL-T, EVA-X, CheXFound, CheXagent, CXR_Foundation
Early_Fusion = concatenation of the six model representations
```

## Evaluation design

### Cohort

The scripts read the consolidated LanceDB table:

```text
../embeddings/MIMIC-CXR-JPG/complete_embeddings_MIMIC-CXR-JPG
```

Rows are restricted to images not marked `ignore == 1`, postero-anterior images, and the official MIMIC-CXR-JPG patient-level train, validation, or test split. Most scripts accept `val`, `valid`, or `validate`; the independent MLP implementation currently expects exactly `validate`. Confirm split values in the table before a complete run.

The 14 categories are:

```text
Atelectasis, Cardiomegaly, Consolidation, Edema,
Enlarged_Cardiomediastinum, Fracture, Lung_Lesion, Lung_Opacity,
Pleural_Effusion, Pneumonia, Pneumothorax, Pleural_Other,
Support_Devices, No_Finding
```

### Labels

The active configurations in the GPU logistic-regression, XGBoost, and MLP scripts use `NegBio_labels`; commented alternatives use `CheXpert_labels`. In the active binary processing functions, only value `1` is positive and every other value, including uncertainty and unmentioned values, becomes `0`.

This is an analysis convention, not a claim that uncertain labels are truly negative. Record the label source and recoding policy with every result. Older CheXpert-oriented artifacts remain in the repository, so inspect the output directory and summary CSV before comparing runs.

### Representation variants

Individual models are evaluated with:

- `raw`: the stored model representation.
- `l2`: the stored L2-normalised representation.
- `pca_95`: PCA fitted on the training split of the raw representation, retaining 95% of variance, then applied to validation and test data.

Early Fusion concatenates the corresponding six model vectors. PCA and standardisation objects are saved with the trained artifacts and must be reused for later inference.

## Classifier heads

### PyTorch logistic regression

`gpu_fit_models.py` trains one multilabel linear head per model and variant. Inputs are standardised with `StandardScaler`; outputs are 14 logits trained with `BCEWithLogitsLoss`. A square-root positive-class weight reduces the effect of class imbalance. Training uses AdamW, validation macro AUROC for early stopping, and optional PyTorch `DataParallel`.

### XGBoost

`xgboost_train.py` trains one binary XGBoost classifier per disease, model, and representation variant. The 14 disease jobs are distributed across four GPUs using Joblib. For `pca_95`, the script standardises first and then applies PCA. This order is part of the saved inference contract.

### Shared multilayer perceptron

`mlp_train.py` performs a fixed grid search over:

```text
alpha:   0.5, 0.67, 1.0, 1.5, 2.0
dropout: 0.2, 0.5
depth:   1 or 2 hidden layers
```

The first hidden width is `int(alpha * input_dim)`. A second hidden layer, when used, has width `max(int(k1 / 2), 14)`. Each hidden block uses Linear, LayerNorm, GELU, and Dropout. The best configuration is selected by validation macro AUROC. MLP runs use raw embeddings and include Early Fusion.

### Independent disease-wise MLP

`independent_mlp.py` runs the same 20-configuration grid separately for each of the 14 diseases. Each disease receives a scalar binary output and its own positive-class weight. The best configuration is selected independently by validation AUROC, producing per-disease weights and reconstructed 14-column probability matrices.

## Running the experiment

Python 3.12 or newer is specified in `pyproject.toml`. Install the environment:

```bash
cd exp1
uv sync
```

The scripts are intended for a machine with enough RAM to materialise the consolidated table and preferably four NVIDIA A6000-class GPUs. Run one head family at a time:

```bash
uv run gpu_fit_models.py
uv run xgboost_train.py
uv run mlp_train.py
uv run independent_mlp.py
```

`main.py` is only a placeholder and does not orchestrate these jobs. Before a full run, confirm the table, QC column, PA metadata, embedding columns, label source, and output directory. The training scripts write predictable filenames and may overwrite prior artifacts.

## Artifact directories

Typical outputs are organised by label source and classifier family:

```text
CheXpert_labels/
├── torch_lr_artifacts/
├── xgboost_evaluation_artifacts/
├── mlp_grid_artifacts/
└── independent_mlp_artifacts/

NegBio_labels/
├── negbio_torch_lr_artifacts/
├── negbio_xgboost_evaluation_artifacts/
├── negbio_mlp_grid_artifacts/
└── negbio_independent_mlp_artifacts/
```

Depending on the run, active scripts may write directly into a label-specific directory such as `negbio_torch_lr_artifacts`. Do not infer the label source only from a filename; inspect the output directory and saved `y_*_true.npy` arrays.

Common outputs include:

| Output | Purpose |
| --- | --- |
| `*_val_probs.npy` | Validation probabilities for the 14 diseases |
| `*_test_probs.npy` | Test probabilities for the 14 diseases |
| `y_val_true.npy`, `y_test_true.npy` | Ground-truth matrices |
| `*_weights.pt` or `*_best_weights.pt` | PyTorch classifier weights |
| `*.json` | Per-disease XGBoost model |
| `*_scaler.joblib` | Standardisation transform |
| `*_pca_object.joblib` | PCA transform |
| `*_summary.csv` | AUROC and configuration summary |

## Metrics and interpretation

The training scripts report macro AUROC across diseases. The visualisation workflow also computes disease-level AUROC and AUPRC from saved test probabilities. AUPRC is particularly informative for rare findings.

This is a representation probe, not an end-to-end fine-tuning benchmark. Differences can reflect embedding quality, dimensionality, preprocessing, label noise, class imbalance, or classifier capacity. The outputs support the report's conclusion that CheXagent is often marginally strongest in linear probing, while nonlinear heads do not consistently improve on simpler linear probes. Interpret that conclusion together with the exact label policy, cohort filters, and artifact version.

## Related files

- `evaluate_all_heads.py` contains an older or alternate all-head evaluation path.
- `feature_explorer.py` explores embedding features.
- `generate_plots.py` creates plots from selected evaluation artifacts.
- `streamlit_app.py` is an experiment-specific interactive application.
- `../vis` consolidates predictions into figures and metrics tables.
