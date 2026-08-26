# Embeddings and Multimodal Features

This folder contains the LanceDB tables used by the chest radiology foundation-model
evaluation. The artifacts were produced from MIMIC-CXR-JPG v2.1.0 and MIMIC-CXR
reports, and are consumed by the experiments in `exp1` through `exp4` and by the
visualisation scripts in `vis`.

The central research question is whether broad biomedical pre-training transfers as
well as chest-X-ray-specific pre-training. The stored features support three kinds of
evaluation described in the project report:

- Frozen image embeddings for structured 14-label disease classification and linear,
  MLP, or XGBoost probing.
- Image-text joint embeddings for zero-shot disease classification using positive and
  negative prompts.
- Image and report embeddings for cross-modal retrieval between studies and reports.

## Important quality-control note

The MIMIC-CXR collection contains 85 severely corrupted images, likely because of scanning or collimation problems. Several models extracted
embeddings for these images without raising an error, so a successfully computed vector
does not imply that the input image was valid. These records are marked `ignore = 1` in
the consolidated image table and should be excluded from evaluation unless an analysis
explicitly studies failure cases.

Two examples are:

```text
0539ee33-9d402e49-a9cc6d36-7aabc539-3d80a62b...
MIMIC-CXR-JPG/2.1.0/files/p10/p10291098/s57194260/0539ee33-9d80a62b....jpg

14a5423b-9989fc33-123ce6f1-4cc7ca9a-9a3d2179...
MIMIC-CXR-JPG/2.1.0/files/p13/p13579794/s51003958/14a5423b-9989fc33-123ce6f1-4cc7ca9a-9a3d2179.jpg
```

The examples above are abbreviated identifiers. The authoritative paths and IDs are
stored in the LanceDB tables and in the source dataset.

MIMIC-CXR-JPG/2.1.0/files/p10/p10291098/s57194260/0539ee33-9d402e49-a9cc6d36-7aabc539-3d80a62b...jpg


```text
embeddings/
├── MIMIC-CXR-JPG/                 Image and image-model LanceDB tables
│   ├── <model>_embeddings_*.lance  Per-model source tables
│   ├── CheXagent_MIMIC_Part_*.lance Sharded CheXagent source tables
│   ├── CheXfound_MIMIC.lance       CheXFound latecomer table
│   ├── complete_embeddings_*.lance Consolidated table before/after append
│   ├── fixed_embeddings_*.lance    Consolidated table from merge_mimic.py
│   └── sampled_embeddings_*.lance  Sampled image-feature table
├── phrases/                         Prompt-pair LanceDB tables
├── reports/                         Image-report text embedding tables
├── merge_mimic.py                   Joins image-model tables
├── append_column.py                 Appends CheXFound columns
├── merge_phrases.py                 Joins phrase embeddings
├── main.py                          Placeholder entry point
└── pyproject.toml                   Python dependencies and project metadata
```

The `.lance` directories are LanceDB tables, not ordinary collections of individual
embedding files. Avoid editing their internal `_transactions`, `_versions`, or `data`
directories manually.

## Installation

Python 3.12 or newer is required. From this directory, install the declared dependencies
with the project tool of your choice. For example, with `uv`:

```bash
cd embeddings
uv sync
```

The declared runtime dependencies are:

- `lancedb` for opening and writing LanceDB tables.
- `duckdb` for joining Arrow tables using SQL.
- `pylance` for Lance-related Python support.

## LanceDB artifact inventory

### Image embeddings

The `MIMIC-CXR-JPG` database contains source tables for the following models:

| Model | Source table(s) | Stored vector size |
| --- | --- | ---: |
| MedSigLIP | `MedSigLIP_embeddings_MIMIC-CXR-JPG` | 1,152 |
| CXR Foundation | `CXR_Foundation_embeddings_MIMIC-CXR-JPG` | 4,096 |
| BioViL-T | `BioViL-T_embeddings_MIMIC-CXR-JPG` | 128 |
| EVA-X | `EVA-X_embeddings_MIMIC-CXR-JPG` | 768 |
| CheXagent | `CheXagent_MIMIC_Part_0` through `Part_3` | 1,024 |
| CheXFound | `CheXfound_MIMIC` | 1,024 |

Each source model table generally exposes `embedding_raw` and `embedding_l2`. The
consolidated tables rename these columns to include the model name, for example
`MedSigLIP_raw` and `MedSigLIP_l2`.

### Reports and phrases

`reports/` contains report-level LanceDB tables for `BioViL-T`, `CheXagent`,
`CXR_Foundation`, and `MedSigLIP`. These support image-report retrieval. Reports are
represented using the `Impression` and `Findings` sections; when a study has multiple
images, downstream retrieval code can aggregate image vectors at study level.

`phrases/` contains model-specific prompt embedding tables for disease phrases. The
available tables are `MedSigLIP`, `CXR_Foundation`, `BioViL-T`, `CheXagent`, and
`complete_phrases`. The consolidated phrase schema contains one row per disease and
positive and negative vectors for the supported models.

## Consolidated image schema

The complete image table is keyed by `dicom_id`. Its metadata columns are:

| Column | Meaning |
| --- | --- |
| `path` | Relative path to the MIMIC-CXR-JPG image |
| `dicom_id` | Unique image identifier and join key |
| `study_id` | Clinical study identifier |
| `subject_id` | De-identified patient identifier |
| `split` | Dataset split, such as `train`, `validate`, or `test` |
| `PerformedProcedureStepDescription` | Procedure description |
| `ViewPosition` | Radiographic view position |
| `image_size` | Struct containing `Rows` and `Columns` |
| `StudyDate`, `StudyTime` | Study timing metadata |
| `ProcedureCodeSequence_CodeMeaning` | Procedure code description |
| `ViewCodeSequence_CodeMeaning` | View code description |
| `PatientOrientationCodeSequence_CodeMeaning` | Patient orientation metadata |
| `CheXpert_labels` | Nested 14-label CheXpert struct |
| `NegBio_labels` | Nested 14-label NegBio struct |
| `cxr_similarity`, `cxr_similarity1` | Image quality/similarity metrics used by QC analyses |
| `ignore` | QC exclusion flag; `1` means exclude the image |

The 14 disease fields, in both label structs, are:

```text
Atelectasis, Cardiomegaly, Consolidation, Edema,
Enlarged_Cardiomediastinum, Fracture, Lung_Lesion, Lung_Opacity,
Pleural_Effusion, Pneumonia, Pneumothorax, Pleural_Other,
Support_Devices, No_Finding
```

Label values follow the project convention: positive `1`, negative `0`, and uncertain
or unmentioned `-1`. Check the consuming experiment before treating `-1` as a negative
label; many multilabel evaluations mask or filter these values.

The consolidated image vectors currently documented in this workspace are:

```text
MedSigLIP_raw / MedSigLIP_l2       1152 floats
CXR_Foundation_raw / _l2          4096 floats
BioViL-T_raw / BioViL-T_l2          128 floats
EVA-X_raw / EVA-X_l2                768 floats
CheXagent_raw / CheXagent_l2       1024 floats
CheXFound_raw / CheXFound_l2       1024 floats
```

`raw` vectors preserve the extracted representation. `l2` vectors are L2-normalised
representations and are suitable for cosine-similarity workflows. Confirm the exact
schema with LanceDB before combining artifacts generated at different points in the
pipeline.

## Phrase schema

Each phrase table has one `disease` row and paired vectors:

```text
disease: string
<Model>_positive_embedding: fixed-size float vector
<Model>_negative_embedding: fixed-size float vector
```

The prompt pairs are used by zero-shot classification: image-to-positive and
image-to-negative similarities are compared for each disease. The report used paired
prompts with similar vocabulary and length to reduce prompt-format effects.

## Merge workflows

### Build the fixed image table

`merge_mimic.py` opens the MedSigLIP, CXR Foundation, BioViL-T, EVA-X, and CheXagent
tables from `MIMIC-CXR-JPG`, converts them to Arrow, and joins them on `dicom_id`.
It writes the result to `fixed_embeddings_MIMIC-CXR-JPG`.

```bash
cd embeddings
python merge_mimic.py
```

This is an **inner join**. Any image absent from one source table is removed from the
result. The script currently does not include CheXFound in that join, despite the
repository also containing CheXFound artifacts.

### Append CheXFound to the complete table

`append_column.py` joins `complete_embeddings_MIMIC-CXR-JPG` with `CheXfound_MIMIC` on
`dicom_id`, adds `CheXFound_raw` and `CheXFound_l2`, and overwrites the complete table.

```bash
cd embeddings
python append_column.py
```

The script uses an absolute path rooted at `~/cxr_foundation_eval/embeddings`, so change
`uri` if the repository is elsewhere. It is also an inner join and uses
`mode="overwrite"`; make a backup or copy the LanceDB directory before running it.

### Merge phrase tables

`merge_phrases.py` joins the `MedSigLIP` and `CXR_Foundation` phrase tables on `disease`
and overwrites `phrases/complete_phrases`.

```bash
cd embeddings
python merge_phrases.py
```

The current script merges only those two phrase tables. The other model-specific phrase
tables remain available as separate artifacts.

## Reading a table

LanceDB tables can be inspected from Python without loading every vector into a pandas
DataFrame:

```python
import lancedb

db = lancedb.connect("embeddings/MIMIC-CXR-JPG")
table = db.open_table("complete_embeddings_MIMIC-CXR-JPG")

print(table.schema)
sample = table.search().limit(5).to_arrow()
print(sample.column_names)
```

For a model-specific table, replace the table name and select its `embedding_raw`,
`embedding_l2`, and `dicom_id` columns. Filter `ignore == 0` and use the official
patient-level split before fitting or evaluating models.

## Reproducibility and interpretation

- Join image features on `dicom_id`; join studies and reports using the study/report
  identifiers supplied by the corresponding tables.
- Preserve the official patient-level train, validation, and test split. Do not create
  a random image split that can place studies from the same patient in multiple sets.
- Exclude `ignore == 1` images from headline metrics and report the exclusion policy.
- Treat CheXpert and NegBio as separate label sources; they are not interchangeable
  ground truth annotations.
- Record whether `raw` or `l2` vectors were used, because normalisation changes both
  linear probing and cosine-similarity behavior.
- The artifacts are large and binary. Keep them outside ordinary source diffs and avoid
  regenerating tables in place unless the inputs and schema are recorded.

## Related project material

- `../exp1` evaluates frozen image embeddings with classifier heads.
- `../exp3` evaluates zero-shot disease classification.
- `../exp4` evaluates cross-modal retrieval.
- `../vis` generates cohort and evaluation visualisations.
- `../data` contains dataset processing, QC, and analysis utilities.