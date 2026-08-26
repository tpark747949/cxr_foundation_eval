# Foundation Model Evaluation Environments

This directory contains the model-specific inference environments used in the chest
radiology foundation-model evaluation. There are seven subfolders:

- Six folders contain one foundation model, its checkpoint or checkpoint-loading code,
  single-image extraction, and bulk MIMIC-CXR-JPG extraction utilities.
- `scripts` contains the shared interactive evaluator, artifact collection utility, and
  inference diagnostics. It does not contain a seventh foundation model.

The project compares general biomedical pre-training with chest-X-ray-specific
pre-training on MIMIC-CXR-JPG v2.1.0. The generated image features are stored in
`../embeddings/MIMIC-CXR-JPG`; report and prompt features are stored under
`../embeddings/reports` and `../embeddings/phrases`.

## Evaluation architecture

The repository separates model inference from downstream evaluation:

```text
MIMIC-CXR-JPG images and metadata
              |
              v
  model-specific embedding_extract_mimic.py
              |
              v
  LanceDB tables in ../embeddings/MIMIC-CXR-JPG
              |
              +--> exp1: frozen-embedding classification
              +--> exp2: sampling and cohort analysis
              +--> exp3: zero-shot classification
              +--> exp4: image-report retrieval
              +--> vis: visualisation and quality analysis

single uploaded image
              |
              v
  scripts/app.py -> <model>/extract.py -> classifier artifacts
```

The bulk extractors preserve image metadata, patient/study identifiers, official split
information, CheXpert and NegBio labels, and both raw and L2-normalised vectors. The
single-image extractors write a temporary `.npy` file for the caller and are not intended
to populate the LanceDB datasets directly.

## Common prerequisites

### Data

The extraction scripts expect the following layout relative to each model directory:

```text
../../data/MIMIC-CXR-JPG/2.1.0/
├── IMAGE_FILENAMES
├── mimic-cxr-2.0.0-metadata.csv.gz
├── mimic-cxr-2.0.0-split.csv.gz
├── mimic-cxr-2.0.0-chexpert.csv.gz
├── mimic-cxr-2.0.0-negbio.csv.gz
└── files/                         MIMIC-CXR-JPG image hierarchy
```

The scripts also expect to write to `../../embeddings/MIMIC-CXR-JPG`. Obtain MIMIC-CXR
and MIMIC-CXR-JPG through PhysioNet and follow their credentialing, data-use, and
handling requirements. Do not commit the data or generated LanceDB tables.

### Python environments

Most folders use `uv` and include a `.python-version`, `pyproject.toml`, and `uv.lock`.
Set up a model environment from that model's directory:

```bash
cd eval/MedSigLIP
uv sync
```

Repeat the command in the folder for the model being run. The environments are
intentionally separate because the model implementations have incompatible dependency
requirements. The declared Python constraints are:

| Folder | Python constraint | Main framework or dependency |
| --- | --- | --- |
| `BioViL-T` | `>=3.12` | PyTorch, torchvision, Hugging Face Hub, PIL |
| `CheXagent` | `>=3.12` | Transformers, PyTorch, Hugging Face Hub |
| `CheXFound` | `==3.9.16` | PyTorch, NumPy `<2`, CheXFound source tree |
| `CXR_Foundation` | `>=3.11` | TensorFlow, TensorFlow Text, PyArrow, Hugging Face Hub |
| `EVA-X` | `>=3.10` | PyTorch, torchvision, local EVA-X source |
| `MedSigLIP` | `>=3.12` | Transformers, TensorFlow, PyTorch, Hugging Face Hub |
| `scripts` | `>=3.12` | Runtime dependencies are imported by the scripts |

The model extractors require a CUDA-capable PyTorch or TensorFlow installation where
applicable. GPU IDs, batch sizes, worker counts, and paths are hard-coded in several
bulk scripts; review and adjust them for the target machine before launching a full
run.

## Standard command patterns

### Single-image extraction

Every model folder has an `extract.py` that accepts two positional arguments:

```bash
cd eval/<model>
uv run extract.py /path/to/image.jpg /path/to/output.npy
```

The output is a NumPy array containing the model's image representation. The shared
Streamlit application invokes this same interface through `uv run`, with the working
directory set to the selected model folder.

The single-image interface is useful for smoke tests and interactive inference. It is
not equivalent to the bulk pipeline in every detail: preprocessing, device selection,
checkpoint loading, and whether the output is a projected or backbone representation
are model-specific and described below.

### Bulk MIMIC extraction

The usual bulk command is:

```bash
cd eval/<model>
uv run embedding_extract_mimic.py
```

These scripts read the MIMIC metadata files, iterate over `IMAGE_FILENAMES`, run model
inference, and create or overwrite a LanceDB table. They use `mode="overwrite"`, so
protect an existing table before rerunning a pipeline. Most use inner metadata merges
and retain failed image records with null embeddings or log failures, depending on the
model implementation.

Run bulk jobs on the intended compute node. A complete MIMIC pass is large and can
consume substantial GPU time, memory, and local storage.

## Model folders

### `BioViL-T`

BioViL-T is a chest-X-ray vision-language model. The local image wrapper reconstructs a
ResNet-50 backbone and a learned projection to a 128-dimensional global image vector.
The bulk extractor uses the projected global representation, resizes images to 512,
crops the center to 480 x 480, applies ImageNet normalisation, and stores 128-dimensional
`embedding_raw` and `embedding_l2` vectors.

Key files:

- `extract.py`: downloads `biovil_t_image_model_proj_size_128.pt` from
  `microsoft/BiomedVLP-BioViL-T` and extracts one normalised image vector.
- `embedding_extract_mimic.py`: bulk image extraction and LanceDB table creation.
- `report_recall.py`: creates BioViL-T joint image and report embeddings for the test
  split; report text is divided into Findings and Impression sections.
- `zero_shot.py`, `autozeroshot.py`: zero-shot prompt and classification workflows.
- `phrase_grounding.ipynb`: exploratory phrase-grounding work.
- `sample_extract.py`: small extraction or sampling utility.

The bulk image table is named `BioViL-T_embeddings_MIMIC-CXR-JPG`. BioViL-T report
features are written to the `BioViL-T` table under `../embeddings/reports`.

### `CheXagent`

CheXagent uses the Stanford AIMI XraySigLIP checkpoint:
`StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli`. The single-image extractor uses
Hugging Face `AutoProcessor` and `SiglipVisionModel`, returning a normalised projected
pooler vector. The bulk extractor distributes the MIMIC rows over GPU workers and writes
one LanceDB partition per worker.

Key files:

- `extract.py`: single-image XraySigLIP extraction.
- `embedding_extract_mimic.py`: multiprocessing bulk extraction using `AVAILABLE_GPUS`
  and tables named `CheXagent_MIMIC_Part_<gpu_id>`.
- `embedding_extract_mimic copy.py`: alternate or copied extraction implementation;
  treat it as historical unless its differences are specifically required.
- `merge_tables.py`: merges the GPU partitions.
- `report_recall.py`: joint image/report retrieval features.
- `zero_shot.py`, `autozeroshot.py`: zero-shot workflows.
- `recall_fix.py`: retrieval-related correction utility.
- `main.py`, `sample_extract.py`: auxiliary entry points.

The model is large and the bulk script assumes the image set is readable without the
per-item recovery path used by some other extractors. Check GPU allocation and available
VRAM before running it.

### `CheXFound`

CheXFound is the vision-centric ViT-L foundation model included in the repository as a
source tree under `chexfound/`. It uses a 512 x 512 evaluation transform and extracts the
class-token representation from the final intermediate block. The supplied checkpoint
layout is:

```text
checkpoints/config.yaml
checkpoints/teacher_checkpoint.pth
checkpoints/example/
```

The environment is intentionally different from the other folders: `pyproject.toml`
requires Python 3.9.16 and NumPy `<2`, while `conda-extras.yaml` describes the upstream
conda environment. Prefer the upstream environment instructions in the model README
when the `uv` environment cannot import the bundled CheXFound package.

Key files:

- `extract.py`: single-image class-token extraction using the bundled checkpoint.
- `embedding_extract_mimic.py`: multiprocessing bulk extraction with four configured
  GPU workers and tables named `CheXfound_MIMIC_Part_<gpu_id>`.
- `merge_tables.py`: merges CheXFound worker partitions.
- `batch1_recovery.py`, `batch2_recovery.py`: recovery utilities for interrupted or
  incomplete extraction batches.
- `main.py`, `sample_extract.py`: auxiliary entry points.
- `chexfound/`: bundled training, evaluation, model, data, and utility code.

CheXFound's bulk tables use a model-specific capitalization (`CheXfound_MIMIC...`) that
differs from the folder name (`CheXFound`) and from the consolidated embedding column
name (`CheXFound_raw` or `CheXFound_l2`). Preserve those names when joining artifacts.
The bulk script uses overwrite mode and does not provide the same per-image validity
fallback as the safer dataset wrappers, so validate paths before a large run.

### `CXR_Foundation`

CXR Foundation uses Google's ELIXR-C pooled image model followed by the PAX ELIXR-B
text/Q-Former component. The pipeline converts each image to a greyscale serialized
TensorFlow PNG example, obtains image features from the ELIXR-C SavedModel, then obtains
joint image embeddings from the Q-Former. The flattened stored representation is
32 x 128 = 4,096 floats.

Key files:

- `extract.py`: downloads the required SavedModels from `google/cxr-foundation` into
  `checkpoints/hf` and extracts one image vector.
- `embedding_extract_mimic.py`: TensorFlow bulk extraction, stored as
  `CXR_Foundation_embeddings_MIMIC-CXR-JPG`.
- `embedding_extract_mimic_gpu.py`: alternate multiprocessing/GPU bulk implementation.
- `merge_tables.py`: table assembly utility.
- `report_recall.py`: report retrieval workflow.
- `zero_shot.py`, `sample_extract.py`: auxiliary workflows.

The bulk script uses TensorFlow inference with a static batch size of one for the Q-Former
input even though images are loaded through a PyTorch DataLoader. It limits TensorFlow
GPU memory growth and writes records in chunks. Confirm that both SavedModel directories
exist under `checkpoints/hf` before starting a run.

### `EVA-X`

EVA-X is a self-supervised chest-X-ray ViT. The folder contains the local model source in
`EVA-X/` and the base checkpoint:

```text
checkpoints/eva_x_base_patch16_merged520k_mim.pt
```

The extractor resizes inputs to 224 x 224, converts grayscale images to three channels,
normalises with ImageNet statistics, and uses the CLS token from `forward_features`. The
bulk implementation probes the output dimension dynamically; the expected base-model
representation is 768 floats.

Key files:

- `extract.py`: single-image extraction using the local `eva_x_base_patch16` model.
- `embedding_extract_mimic.py`: bulk extraction and creation of
  `EVA-X_embeddings_MIMIC-CXR-JPG`.
- `EVA-X/`: bundled upstream EVA-X implementation and task code.
- `sample_extract.py`: small extraction utility.

The scripts select `cuda` for inference, and the bulk script defaults to `cuda:1` when
CUDA is available. Change that device selection if GPU 1 is unavailable. The checkpoint
path is relative to the model folder.

### `MedSigLIP`

MedSigLIP is the general biomedical comparison model. The extractor loads
`google/medsiglip-448` through Hugging Face Transformers, resizes inputs to 448 x 448,
and uses the vision model's 1,152-dimensional `pooler_output`. Both raw and L2-normalised
vectors are stored by the bulk pipeline.

Key files:

- `extract.py`: single-image vision embedding extraction using the TensorFlow resize
  operation followed by the Transformers processor.
- `embedding_extract_mimic.py`: bulk extraction and creation of
  `MedSigLIP_embeddings_MIMIC-CXR-JPG`.
- `report_recall.py`: image/report joint embedding and retrieval preparation.
- `zero_shot.py`: positive-negative prompt zero-shot classification.
- `cosine_dissimilar.py`: cosine similarity analysis utility.
- `text_embedding.pt`: a `[1, 1152]` tensor for the phrase `a photo of a chest x-ray`.
- `sample_extract.py`: small extraction utility.

MedSigLIP's text embedding artifact is a convenience feature for experiments and should
not be confused with the disease-specific prompt vectors in `../embeddings/phrases`.
The bulk script records missing or unreadable images with null vectors and logs errors to
`mimic_processing_errors.log`.

## Shared `scripts` folder

The `scripts` folder is the common orchestration layer. It has its own Python metadata,
but `pyproject.toml` intentionally declares no dependencies even though the scripts
import packages such as Streamlit, PyTorch, XGBoost, Joblib, LanceDB, and NumPy. Install
those packages in the environment used to run the shared tools, or use an environment
that already contains the project evaluation stack.

### `app.py`: interactive classifier

Start the Streamlit evaluator from this directory:

```bash
cd eval/scripts
streamlit run app.py
```

The app:

1. Accepts a JPG, PNG, or JPEG upload.
2. Calls the selected model's `extract.py` through that model's isolated `uv` environment.
3. Loads a classifier head and optional preprocessing artifacts from `scripts/artifacts`.
4. Displays probabilities for the 14 CheXpert categories.

Supported model selections are `MedSigLIP`, `BioViL-T`, `EVA-X`, `CheXFound`, `CheXagent`,
`CXR_Foundation`, and `Early_Fusion`. `Early_Fusion` concatenates the single-image
outputs from the six model folders. Supported heads are `LR`, `XGB`, `s2` (shared MLP),
and `i2` (independent MLP). PCA/scaler processing is available for the linear and XGB
paths; the MLP paths use raw vectors.

The app assumes the repository is located at `~/cxr_foundation_eval`, because
`PROJECT_ROOT` is constructed from that path. Change it before running elsewhere. It
also assumes standardized artifact filenames in `scripts/artifacts` and that the input
vector dimensions agree with the saved classifier weights.

### `collect_artifacts.py`: standardize trained heads

Run from `eval/scripts`:

```bash
python collect_artifacts.py
```

The utility scans `../../exp1/CheXpert_labels` for `.pt`, `.joblib`, `.json`, and `.pkl`
files, skips `3class` and `4class` outputs, infers model/head/variant/target names, and
copies recognized files into `scripts/artifacts`. It does not train models. Existing
files with the same destination names may be overwritten by the copy operation, so
inspect the artifact directory before rerunning it.

### `debug_inference.py`: artifact sweep

Run from `eval/scripts`:

```bash
python debug_inference.py
```

This diagnostic opens `../embeddings/MIMIC-CXR-JPG/complete_embeddings_MIMIC-CXR-JPG`,
checks model/head/variant combinations, applies PCA and scalers where required, loads
classifier artifacts, and prints a viability report. It uses Atelectasis for a small
prediction smoke test. It is an artifact compatibility check, not a benchmark and not a
replacement for the official evaluation scripts.

## Output conventions

### Image tables

Bulk image extractors generally create one LanceDB table per model under
`../embeddings/MIMIC-CXR-JPG`. Their common columns are:

```text
path, dicom_id, image_size, study_id, subject_id, split,
CheXpert_labels, NegBio_labels, embedding_raw, embedding_l2
```

The consolidated table renames vectors with model prefixes, for example:

```text
MedSigLIP_raw, MedSigLIP_l2
CXR_Foundation_raw, CXR_Foundation_l2
BioViL-T_raw, BioViL-T_l2
EVA-X_raw, EVA-X_l2
CheXagent_raw, CheXagent_l2
CheXFound_raw, CheXFound_l2
```

Expected dimensions are 1,152, 4,096, 128, 768, 1,024, and 1,024 respectively. Verify
the actual LanceDB schema when working with regenerated tables.

### Labels and splits

The 14 structured categories are:

```text
Atelectasis, Cardiomegaly, Consolidation, Edema,
Enlarged_Cardiomediastinum, Fracture, Lung_Lesion, Lung_Opacity,
Pleural_Effusion, Pneumonia, Pneumothorax, Pleural_Other,
Support_Devices, No_Finding
```

The source MIMIC labels use positive `1`, negative `0`, uncertain `-1`, and unmentioned
`-2` in these extraction scripts. This differs from some downstream summaries that
collapse uncertain and unmentioned values together. Check the consuming experiment
before recoding labels.

Use the official patient-level `split` values for train, validation, and test. Do not
make an image-level random split, because multiple images and studies can belong to the
same patient.

### Quality control

The upstream project identified 85 severely corrupted, mostly black images. A valid
embedding can still be produced for such an image. The consolidated image artifacts
therefore use the `ignore` flag when available; exclude `ignore == 1` from headline
metrics and retain the policy in experiment logs. Some model-specific bulk tables were
created before that flag was appended, so confirm the table schema and join against the
consolidated QC metadata when necessary.

## Recommended workflow

1. Confirm PhysioNet access and the MIMIC-CXR-JPG directory layout.
2. Create only the model environment needed for the next extraction.
3. Run `sample_extract.py` or `extract.py` on one image and inspect the vector shape.
4. Run the corresponding bulk extractor on a small, reversible test subset.
5. Verify the LanceDB table schema, row count, null-vector count, and split coverage.
6. Back up existing tables before any command that uses overwrite mode.
7. Merge model tables only after checking that `dicom_id` coverage is compatible.
8. Run the downstream experiment with official patient splits and the QC exclusion policy.
9. Use `scripts/collect_artifacts.py` before interactive classifier inference.
10. Run `scripts/debug_inference.py` and then `streamlit run app.py` for a qualitative
    single-image smoke test.

## Relationship to the project report

The evaluation implemented by these folders supports the report's three principal
experiments:

- Structured disease classification uses frozen image vectors from all six models and
  evaluates linear probing, XGBoost, and MLP heads with AUROC and AUPRC.
- Zero-shot classification uses image-text models and positive-negative disease prompts,
  evaluated with AUROC and AUPRC.
- Cross-modal retrieval matches image/study representations to Findings and Impression
  report representations using Recall@k and mean reciprocal rank.

The model outputs are representations for research evaluation, not clinical diagnoses.
They must not be used as a standalone clinical decision system.
