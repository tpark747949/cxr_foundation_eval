<p align="center"><img src="docs/carleton.png" alt="Carleton College" height="86">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="docs/hms.png" alt="Harvard Medical School" height="86">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="docs/mgh.png" alt="Massachusetts General Hospital" height="86"></p>






<h1 align="center">Evaluating Chest X-ray Foundation Models</h1>

<p align="center">
  A MIMIC-CXR evaluation of general biomedical and chest X-ray-specific foundation-model representations
</p>

<p align="center">
  <strong>Taejoon Park</strong><br>
  21 August 2026
</p>

## Project overview

This repository evaluates whether chest X-ray-specific foundation models outperform a general biomedical vision-language model on downstream thoracic radiology tasks. The study uses MIMIC-CXR-JPG v2.1.0 images, metadata, structured labels, and associated MIMIC-CXR reports.

The central question is:

> Do CXR-specific foundation models outperform general medical vision-language foundation models on MIMIC-CXR, and under what conditions?

The project compares frozen representations rather than retraining the foundation models. Image and text embeddings are extracted in model-specific environments, consolidated into LanceDB tables, and evaluated through supervised classification, label-efficiency, zero-shot classification, and image-report retrieval.

## Project materials

These documents provide the project direction, presentation summary, and full written report:

| Material | Description |
| --- | --- |
| [Research plan](docs/research%20plan.md) | Original study motivation, aims, candidate models, dataset, and proposed experimental design |
| [Projection presentation](docs/08.21_final.pdf) | Slide deck covering the question, methods, experiments, and primary outcomes |
| [Project report](docs/Project%20report.pdf) | Written report describing the completed evaluation, results, discussion, and future directions |

The research plan defines four aims: full-data disease classification, label-efficiency analysis, zero-shot classification, and image-report alignment. The codebase implements those aims across `exp1` through `exp4`, with derived metrics and figures in `vis`.

## Study design at a glance

| Experiment | Question | Main models | Main outputs |
| --- | --- | --- | --- |
| [Experiment 1](exp1/README.md) | How well do frozen image embeddings support structured disease classification? | Six foundation models plus Early Fusion | AUROC/AUPRC, classifier heads, per-disease predictions |
| [Experiment 2](exp2/README.md) | How much performance is retained with 1%, 5%, or 10% of labelled training data? | Six foundation models plus Early Fusion | Label-efficiency curves and scarcity-condition artifacts |
| [Experiment 3](exp3/README.md) | Can disease status be predicted without a fitted classifier using image-text alignment? | MedSigLIP, CXR Foundation, BioViL-T | Zero-shot continuous scores, AUROC/AUPRC, ROC/PR dashboard |
| [Experiment 4](exp4/README.md) | Can images retrieve their reports and reports retrieve their images? | MedSigLIP, CXR Foundation, BioViL-T | Recall@1/5/10, MRR, retrieval dashboard |
| [Visualisations](vis/README.md) | How are results consolidated and compared across models and heads? | Results from Experiments 1 and 2 | `master_metrics.csv` and derived figures |

## Foundation models

The repository evaluates six model families overall:

- **MedSigLIP**: general biomedical image-text pre-training across chest X-rays and other biomedical modalities.
- **CheXagent**: chest X-ray vision-language model based on a SigLIP-style architecture.
- **CXR Foundation**: CXR-specific ELIXR-based image-text representation.
- **BioViL-T**: chest X-ray temporal vision-language representation with projected image and text spaces.
- **CheXFound**: vision-centric ViT-L chest X-ray foundation model.
- **EVA-X**: self-supervised chest X-ray ViT.

The model set depends on the task. Experiments 1 and 2 use all six models and Early Fusion. Experiments 3 and 4 use only the three models with the required image-text spaces: MedSigLIP, CXR Foundation, and BioViL-T.

## Dataset and cohort

The primary data sources are:

- **MIMIC-CXR-JPG v2.1.0**: 377,110 de-identified chest radiographs with metadata and CheXpert-style structured labels.
- **MIMIC-CXR v2.1.0**: associated de-identified radiology reports, with multiple images potentially belonging to one clinical study.

The study uses the official patient-level train, validation, and test partitions. Supervised classification is generally restricted to postero-anterior images and excludes records marked by the quality-control process. The zero-shot and retrieval scripts have their own filters, documented in their respective READMEs, so cohort definitions should be checked before comparing experiments.

The 14 structured categories are:

```text
Atelectasis, Cardiomegaly, Consolidation, Edema,
Enlarged_Cardiomediastinum, Fracture, Lung_Lesion, Lung_Opacity,
Pleural_Effusion, Pneumonia, Pneumothorax, Pleural_Other,
Support_Devices, No_Finding
```

### Quality control

The data pipeline identified 85 severely corrupted, mostly black images, likely caused by scanning or collimation problems. These images can still produce apparently valid model embeddings. They are marked with `ignore = 1` in the consolidated image data and should be excluded from headline metrics.

MIMIC-CXR and MIMIC-CXR-JPG are credentialed datasets. Access, storage, sharing, and processing must follow the PhysioNet data-use agreement. Do not commit the source data or upload images and reports to unauthorised third-party services.

## Repository map

```text
cxr_foundation_eval/
├── data/          Dataset preparation, quality control, and cohort analysis
├── embeddings/    LanceDB image, phrase, and report embedding stores
├── eval/          Isolated model environments and embedding extraction scripts
├── exp1/          Full-data structured disease classification
├── exp2/          Label-efficiency sampling and training
├── exp3/          Zero-shot disease classification
├── exp4/          Cross-modal image-report retrieval
├── vis/           Metrics consolidation and result visualisation
└── docs/          Research plan, presentation, report, and affiliation logos
```

## Directory guides

Each major directory has its own README with implementation details, commands, and known caveats:

- [Data processing and QC](data/README.md)
- [Embedding and multimodal feature stores](embeddings/README.md)
- [Foundation-model evaluation environments](eval/README.md)
- [Experiment 1: structured disease classification](exp1/README.md)
- [Experiment 2: label-efficiency analysis](exp2/README.md)
- [Experiment 3: zero-shot disease classification](exp3/README.md)
- [Experiment 4: cross-modal retrieval](exp4/README.md)
- [Visualisation and result consolidation](vis/README.md)

These guides are intentionally more specific than this overview. Start with the relevant directory README before running a model extraction or training pipeline.

## End-to-end workflow

The intended dependency flow is:

```text
PhysioNet MIMIC-CXR-JPG + MIMIC-CXR
                  |
                  v
        data/ preparation and QC
                  |
                  v
        eval/ model-specific inference
                  |
                  v
        embeddings/ LanceDB artifacts
             /          \
            v            v
        exp1/exp2     exp3/exp4
            |
            v
          vis/
```

A typical full-data workflow is:

1. Obtain and arrange the credentialed MIMIC-CXR-JPG and MIMIC-CXR data.
2. Run quality profiling and manually review suspicious images in `data/`.
3. Mark exclusions and propagate the `ignore` field to the embedding records.
4. Extract per-model image, report, and phrase embeddings in `eval/`.
5. Merge compatible model tables in `embeddings/`.
6. Run Experiment 1 for frozen-embedding disease classification.
7. Run Experiment 2 to create nested 1%, 5%, and 10% training cohorts and train the same families of heads.
8. Run Experiment 3 for positive-negative prompt scoring.
9. Run Experiment 4 for image-report and report-image retrieval.
10. Consolidate predictions and figures with `vis/`.

Most pipelines use `uv` and maintain a separate environment per model or experiment. The repository also contains GPU-specific code and large binary LanceDB artifacts; start with a small smoke test before launching a complete MIMIC pass.

## Representation and evaluation conventions

### Image embeddings

The consolidated image table is stored under:

```text
embeddings/MIMIC-CXR-JPG/complete_embeddings_MIMIC-CXR-JPG.lance
```

Model-prefixed columns include raw and L2-normalised vectors. Expected dimensions are:

| Model | Expected image vector size |
| --- | ---: |
| MedSigLIP | 1,152 |
| CXR Foundation | 4,096 flattened values, interpreted as 32 x 128 in text alignment tasks |
| BioViL-T | 128 |
| EVA-X | 768 |
| CheXagent | 1,024 |
| CheXFound | 1,024 |

### Labels

The image table contains both `CheXpert_labels` and `NegBio_labels`. Depending on the run, binary processing maps only `1` to positive and maps `0`, `-1`, and `-2` to negative. Experiment 3 exposes more explicit policies for uncertain and unmentioned labels. Always record the label source, recoding policy, view filter, split filter, and QC exclusion policy.

### Metrics

- **AUROC**: threshold-independent ranking performance for binary disease prediction.
- **AUPRC**: precision-recall performance, especially informative for rare diseases.
- **Recall@k**: whether a correct report or image appears in the first `k` retrieved candidates.
- **MRR**: average reciprocal rank of the first correct retrieval.

## Environment and storage notes

There is no single root environment for the entire repository. Each major computational directory has its own `pyproject.toml` and often its own lockfile. The model environments under `eval/` are intentionally isolated because their framework and version requirements differ.

Common requirements include:

- Python versions ranging from 3.9.16 for CheXFound to 3.12 for most analysis projects.
- CUDA-capable PyTorch and/or TensorFlow for embedding extraction and GPU classifier training.
- Sufficient local storage and memory for MIMIC-CXR-JPG, LanceDB tables, and materialised feature matrices.
- Hugging Face or upstream model checkpoint access for models that download weights at runtime.

The full data and embedding stores are large. Avoid loading all vectors into pandas when Arrow or LanceDB queries are sufficient, and do not edit `.lance` internal transaction or version files manually.

## Main findings represented in the report

The completed report presents these broad observations:

- MedSigLIP was not significantly divergent from CXR-specific models in supervised linear probing and zero-shot classification.
- CheXagent was marginally strongest in the reported linear-probing comparison.
- Nonlinear MLP and XGBoost heads did not consistently improve over simpler linear probing.
- MedSigLIP performed strongly in cross-modal image-report retrieval.
- Several labels, particularly atelectasis, remained difficult across model and head choices.
- Broader biomedical pre-training may provide transferable visual and semantic primitives that remain competitive with domain-specific pre-training.

These are conclusions of the documented experiments and protocols, not universal claims about every checkpoint, institution, preprocessing pipeline, or clinical use case.

## Reproducibility and responsible use

Before interpreting a result, preserve:

- The exact source table and artifact directory.
- The model checkpoint and extraction environment.
- The train/validation/test filters and patient-level split.
- The CheXpert or NegBio label source and uncertainty policy.
- The `ignore` quality-control exclusion policy.
- The embedding variant (`raw`, `l2`, or PCA-reduced).
- The classifier head, hyperparameters, and early-stopping behavior.
- The script version used to produce derived CSV, JSON, Parquet, or PNG outputs.

This repository is a research evaluation of pretrained representations and NLP-derived labels. Its outputs are not clinical diagnoses, medical advice, or a validated clinical decision-support system.

## Acknowledgements

The project direction and guidance were provided by Professor Bongjin Lee, with support and opportunity from Professor Jonghye Woo. The study uses publicly documented foundation models and the credentialed MIMIC-CXR and MIMIC-CXR-JPG resources.
