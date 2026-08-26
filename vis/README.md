# Visualisation and Results Consolidation

This folder consolidates prediction arrays from Experiments 1 and 2, computes disease-level test metrics, and generates figures for comparing foundation models, classifier heads, label sources, and training-data percentages.

It is a post-processing folder. It does not train foundation models or classifier heads, and it does not extract image embeddings. Its main inputs are `.npy` prediction arrays and the ground-truth arrays produced by `exp1` and `exp2`.

## Scope

The visualisations cover:

- Experiment 1: full-data structured disease classification.
- Experiment 2: nested 1%, 5%, and 10% training-data conditions.
- Six foundation models plus Early Fusion.
- Logistic regression, XGBoost, shared MLP, and independent MLP heads.
- Binary and four-class output conventions.
- CheXpert and NegBio label sources.
- Raw, L2-normalised, and PCA-95 embedding variants where available.

The 14 diseases are:

```text
Atelectasis, Cardiomegaly, Consolidation, Edema,
Enlarged_Cardiomediastinum, Fracture, Lung_Lesion, Lung_Opacity,
Pleural_Effusion, Pneumonia, Pneumothorax, Pleural_Other,
Support_Devices, No_Finding
```

## Data flow

```text
exp1 classifier artifacts   exp2 classifier artifacts
             |                         |
             +------ prediction arrays (.npy)
                                |
                                v
                     vis/test_probs/
                                |
                                v
                       compute_metrics.py
                                |
                                v
                       master_metrics.csv
                                |
                                v
                       generate_figures.py
                                |
                                v
                     figs/*.png and CSV outputs
```

The folder also contains `interactive_consolidator.py`, which interactively maps source
prediction files into the standard `test_probs` naming scheme, and `roll_call.py`, which
inventories the staged files.

## Directory contents

| File or directory | Role |
| --- | --- |
| `compute_metrics.py` | Loads staged test predictions and calculates AUROC/AUPRC |
| `generate_figures.py` | Generates the four summary figures |
| `interactive_consolidator.py` | Interactively copies and renames prediction arrays |
| `roll_call.py` | Checks staged-file completeness and writes an inventory |
| `master_metrics.csv` | Disease-level consolidated metrics |
| `inventory_roll_call.csv` | Parsed inventory of staged prediction arrays |
| `figs/` | Generated figures and visual assets |
| `test_probs/` | Staged test probability arrays |
| `compute_metrics.py` | Uses LanceDB ground truth and staged predictions |

`main.py` is an auxiliary placeholder unless it is updated for a particular run.

## Environment

Python 3.12 or newer is specified in `pyproject.toml`. The declared dependencies include
LanceDB, Pandas, Plotly, scikit-learn, Seaborn, Streamlit, and Tabulate.

Install from this directory:

```bash
cd vis
uv sync
```

The metric and figure scripts use Matplotlib/Seaborn-style plotting and scikit-learn
metrics. A GPU is not required for this folder, although loading large prediction arrays
can require substantial RAM.

## Prediction-file convention

Files in `test_probs/` must use this structure:

```text
<Model>_<Head>_<Label>_<Variant>.npy
```

Examples:

```text
MedSigLIP_LR_CheXpert_raw.npy
BioViL-T_XGB_10pct_pca95.npy
Early_Fusion_s2_NegBio_raw.npy
```

The recognised values are:

```text
Models:   MedSigLIP, BioViL-T, EVA-X, CheXFound, CheXagent,
          CXR_Foundation, Early_Fusion
Heads:    LR, XGB, s2, s4, i2, i4
Labels:   CheXpert, NegBio, 1pct, 5pct, 10pct
Variants: raw, l2, pca95
```

The parser handles model names containing underscores by matching the known model prefix
first. Do not add extra underscore-separated fields to filenames without updating
`compute_metrics.py`.

## Staging predictions

`interactive_consolidator.py` searches both Experiment 1 and Experiment 2 for NumPy
arrays, excludes validation and ground-truth arrays, checks that the first dimension is
996 test samples, and prompts for the model, head, label condition, and variant. It then
copies the selected array into `test_probs/` with a standard filename.

Run it from this directory:

```bash
cd vis
uv run interactive_consolidator.py
```

The utility is interactive. It scans:

```text
../exp1/**/*.npy
../exp2/artifacts/**/*.npy
```

It ignores files with `y_` or `val` in the filename and skips hidden virtual-environment
paths. Review each proposed mapping carefully, especially for files from old CheXpert or
NegBio runs.

After staging files, inspect the inventory:

```bash
uv run roll_call.py
```

This writes `inventory_roll_call.csv` and prints completeness tables by model/head and
label/variant. It is a naming and coverage check, not a numerical validation.

## Computing metrics

`compute_metrics.py` loads test ground truths from:

```text
../embeddings/MIMIC-CXR-JPG/complete_embeddings_MIMIC-CXR-JPG
```

It filters to non-ignored PA images in the official test split. It constructs both binary
and four-class targets from the nested `CheXpert_labels` and `NegBio_labels` structs:

- `1` becomes positive.
- `0` becomes negative.
- `-1` becomes the uncertain class for four-class evaluation.
- `-2` or missing values become the unmentioned class for four-class evaluation.
- Binary evaluation maps every value other than `1` to `0`.

Run:

```bash
uv run compute_metrics.py
```

For each staged prediction array, the script calculates per-disease AUROC and AUPRC and
writes:

```text
master_metrics.csv
```

The output columns are:

```text
Model, Head, Label, Var, Disease, AUC, AUPRC
```

Binary heads (`LR`, `XGB`, `s2`, `i2`) use the disease probability column directly.
Four-class heads (`s4`, `i4`) use one-vs-rest macro averages across the four output
classes. The code accepts either `(samples, 4, diseases)` or `(samples, diseases, 4)`
probability layouts.

The prediction arrays and the ground-truth rows must have identical ordering. A matching
shape alone does not prove matching row order, so keep the same filtered test cohort and
ordering in every training script.

## Generating figures

`generate_figures.py` reads `master_metrics.csv` and writes four PNG figures in the
current working directory:

```bash
uv run generate_figures.py
```

The figures are:

1. `fig1_label_efficiency.png`: mean AUROC versus 1%, 5%, 10%, and 100% labelled data.
   The `CheXpert` label condition is treated as the 100% baseline by the current script.
2. `fig2_labeler_sensitivity.png`: full-data raw-embedding comparison of CheXpert and
   NegBio labels.
3. `fig3_head_comparison.png`: classifier-head comparison on raw full-data CheXpert
   results.
4. `fig4_disease_heatmap.png`: maximum per-disease AUROC across heads for each model on
   full-data CheXpert raw embeddings.

The current figure script saves to the working directory rather than explicitly to
`figs/`. Run it from `vis/figs` if you want output there, or move/copy generated figures
into `figs/` after review.

## Reading the figures correctly

### Label efficiency

The scarcity plot compares the same held-out test set across training-data percentages.
It is intended to show how quickly each representation reaches full-data performance.
The 100% point is a label-source baseline, so it must use the same labeller and task
convention as the scarcity runs for a strictly controlled comparison.

### Labeler sensitivity

The CheXpert-versus-NegBio figure measures sensitivity to the NLP-derived ground truth,
not model robustness to an independent clinical reference standard. Differences can
reflect report parsing and label-generation behavior.

### Head comparison

The head figure compares mean disease-level AUROC across models. It should not be read as
an isolated test of architectural complexity because each head family may have different
regularisation, scaling, optimization, and hyperparameter-search behavior.

### Disease heatmap

The heatmap takes the maximum AUROC over available heads for each model and disease. It
is therefore a best-head summary, not the performance of one common classifier setting.
Use `master_metrics.csv` for the underlying head-specific values.

## Reproducibility checklist

1. Confirm that every staged array has 996 rows and corresponds to the same filtered test
   cohort.
2. Run `roll_call.py` and inspect `inventory_roll_call.csv` for duplicate or missing
   combinations.
3. Confirm whether each filename's label field means a labeller (`CheXpert`, `NegBio`)
   or a training-data percentage (`1pct`, `5pct`, `10pct`).
4. Run `compute_metrics.py` before generating figures.
5. Preserve `master_metrics.csv` alongside the staged arrays used to produce it.
6. Record the git state, label policy, QC exclusion policy, view filter, and split filter.
7. Treat figures as derived outputs; regenerate them when the metric CSV changes.

## Relationship to the experiments

`../exp1` supplies full-data classifier results for structured disease classification.
`../exp2` supplies the 1%, 5%, and 10% label-efficiency results. This folder does not
retrain either experiment; it aligns their saved test probabilities into a common schema
and makes the comparisons visible.

The results are research analyses of frozen representations and NLP-derived labels. The
probabilities and figures are not clinical diagnoses or a validated clinical decision
system.
