# Experiment 2: Label-Efficiency Analysis

Experiment 2 tests how foundation-model representations perform when only a small fraction of the available labelled training data is used. It implements the second experiment in the project proposal: compare model performance at 1%, 5%, and 10% of the training data, while evaluating on the same held-out validation and test cohorts.

## Sampling design

`random_sample.py` reads:

```text
../embeddings/MIMIC-CXR-JPG/complete_embeddings_MIMIC-CXR-JPG
```

and writes an updated table:

```text
../embeddings/MIMIC-CXR-JPG/sampled_embeddings_MIMIC-CXR-JPG
```

The output retains complete embedding records and adds three integer indicator columns:

```text
sample_1_percent
sample_5_percent
sample_10_percent
```

An indicator value of `1` means that the row belongs to that training subset. The subsets are nested: every 1% row is in the 5% subset, and every 5% row is in the 10% subset. Sampling is performed independently within each `(ViewCodeSequence_CodeMeaning, split)` stratum, with NumPy seed `42` and stochastic rounding for small expected counts.

The script writes realised and expected counts to `stratified_sampling_log.csv`. It creates the named sampled table with `mode="overwrite"`; protect an existing sampled table before regenerating it.

## What is held constant

The sampled training flags are intended to change only the amount of labelled training data. The model representations, validation/test rows, and core task remain the same as Experiment 1. Downstream training should continue to exclude QC-excluded rows where `ignore == 1`, non-PA images when reproducing the report's PA-only analysis, and rows outside the official train/validation/test partition.

The 1%, 5%, and 10% subsets are training-data conditions, not new test sets. Do not resample validation or test images into these subsets.

## Classifier jobs

The `scripts/` directory contains one set of training scripts for each sampling level:

```text
scripts/1p/
scripts/5p/
scripts/10p/
```

Each level contains variants of the Experiment 1 classifier families:

- `gpu_fit_models.py`: multilabel PyTorch logistic regression.
- `xgboost_train.py`: independent disease-wise XGBoost classifiers.
- `mlp_train.py`: shared multilabel MLP grid search.
- `independent_mlp.py`: independent disease-wise MLP grid search.
- `shared_4class.py`: shared four-class label treatment.
- `independent_4class.py`: independent four-class label treatment.

Outputs are organised as:

```text
artifacts/<level>/<classifier_family>/
```

where `<level>` is `1p`, `5p`, or `10p`. The artifact tree contains per-model and per-disease weights, probabilities, PCA/scaler objects, and summary files. The scripts under each level are separate copies with level-specific paths or settings; inspect their active constants before assuming they are interchangeable.

## Running the sampling step

Experiment 2's top-level environment is Python 3.12 or newer, but its `pyproject.toml` does not declare dependencies. Use an environment containing at least LanceDB, Pandas, and NumPy, plus the dependencies required by the classifier scripts.

```bash
cd exp2
uv sync
uv run random_sample.py
```

The top-level `main.py` executes, in order, the six scripts in `scripts/1p/`, the six scripts in `scripts/5p/`, and the six scripts in `scripts/10p/`:

```bash
uv run main.py
```

Because this launches a long sequence of GPU training jobs and stops on the first failure, run level-specific scripts directly when debugging or when only one data condition is needed.

## Analysis conditions

The proposal frames the comparison around six individual models and Early Fusion:

```text
MedSigLIP, BioViL-T, EVA-X, CheXFound, CheXagent, CXR_Foundation, Early_Fusion
```

The scarcity curves compare macro AUROC at:

```text
1%, 5%, 10%, and 100% labelled training data
```

The `100%` point is the full-label baseline from Experiment 1. In `vis`, it is represented by the `CheXpert` or `NegBio` label condition, depending on which prediction files were consolidated. Keep the labeller consistent when comparing the baseline to scarcity conditions.

## Artifact interpretation

Prediction filenames generally encode model, head, label/data condition, and embedding variant. Examples from the visualisation staging convention are:

```text
MedSigLIP_LR_1pct_raw.npy
Early_Fusion_XGB_10pct_pca95.npy
BioViL-T_s2_NegBio_raw.npy
```

The visualisation parser recognises:

```text
Models:   MedSigLIP, BioViL-T, EVA-X, CheXFound, CheXagent,
          CXR_Foundation, Early_Fusion
Heads:    LR, XGB, s2, s4, i2, i4
Labels:   CheXpert, NegBio, 1pct, 5pct, 10pct
Variants: raw, l2, pca95
```

Binary heads (`LR`, `XGB`, `s2`, `i2`) are evaluated against binary targets. Four-class heads (`s4`, `i4`) are evaluated one-vs-rest across the four output states. The exact mapping is implemented in `../vis/compute_metrics.py`.

## Reproducibility checks

Before interpreting a scarcity result, verify:

1. `stratified_sampling_log.csv` records realised counts for every view/split stratum.
2. The 1%, 5%, and 10% flags are nested.
3. Validation and test arrays have matching row order across classifier families.
4. The same QC exclusion, PA-view restriction, label source, and split definitions are used at every data percentage.
5. PCA and scalers are fitted on the relevant training subset only.
6. No test probabilities from a different label policy have been copied into the same comparison directory.

Small strata can produce realised counts that differ from nominal percentages due to stochastic rounding. The sampling log, rather than the requested percentage alone, is the authoritative cohort record.

## Research interpretation

This experiment addresses label efficiency, not pre-training efficiency or compute efficiency. A model that performs well at 1% may provide a more useful representation when labelled data are scarce, but the comparison remains sensitive to label noise, class prevalence, classifier regularisation, and the nested sampling procedure.

Use saved per-disease metrics to determine whether a macro trend is driven by common labels or by improvements on rare diseases.

## Related files

- `random_sample.py` creates nested sample flags and the sampling log.
- `main.py` runs all three level-specific script groups sequentially.
- `artifacts/` stores outputs from the 1%, 5%, and 10% jobs.
- `../exp1` provides the full-data classifier baselines.
- `../vis` consolidates predictions from both experiments.
