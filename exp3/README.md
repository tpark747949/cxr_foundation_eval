# Experiment 3: Zero-Shot Disease Classification

Experiment 3 evaluates zero-shot chest X-ray disease classification using image-text representations. Unlike Experiments 1 and 2, no supervised disease classifier is trained here. Each test image is scored against positive and negative text prompts for each disease, and the score difference is used as the model output.

Only three foundation models are evaluated in this experiment:

- `MedSigLIP`
- `CXR_Foundation`
- `BioViL-T`

CheXagent and the other models in the wider project are not evaluated by this experiment, even though related artifacts may exist elsewhere in the repository.

## Objective

The experiment tests whether the image-text joint latent space already separates disease-positive from disease-negative examples without fitting a task-specific classification head. The main metrics are AUROC and AUPRC, computed from continuous image-text scores.

The evaluated diseases are:

```text
Atelectasis, Cardiomegaly, Consolidation, Edema,
Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity,
Pleural Effusion, Pleural Other, Pneumonia, Pneumothorax,
Support Devices, No Finding
```

## Inputs

The batch evaluation script reads image embeddings and prompt embeddings from LanceDB:

```text
Image database:  ../embeddings/MIMIC-CXR-JPG
Image table:     fixed_embeddings_MIMIC-CXR-JPG
Phrase database: ../embeddings/phrases
Phrase table:    complete_phrases
```

It also opens precomputed score tables for BioViL-T and CheXagent:

```text
../embeddings/phrases/BioViL-T
../embeddings/phrases/CheXagent
```

Only the BioViL-T table is used among those two precomputed score sources because CheXagent is not included in `MODELS`. The extra CheXagent lookup is retained in the script but does not make CheXagent part of the evaluated model set.

The image query selects:

```text
split = 'test' AND ignore = 0
```

Unlike the PA-restricted supervised experiments, `zero_shot_classification.py` does not add a postero-anterior view filter. If PA-only results are required, add or apply that filter consistently before comparing with Experiments 1 and 2.

## Prompt embeddings

`complete_phrases` contains one row per disease and positive/negative prompt embeddings for the supported models. Conceptually, each row contains:

```text
disease
<Model>_positive_embedding
<Model>_negative_embedding
```

The prompt pairs are designed to express the presence and absence of the same disease. For example:

```text
Positive: "radiographic findings consistent with atelectasis"
Negative: "no evidence of atelectasis, lungs are fully inflated"
```

The prompt embeddings must belong to the same model-specific joint space as the corresponding image embedding. Do not compare an image vector from one model against another model's text vectors.

## Classification methods

This experiment contains different scoring methods because the three models do not expose identical embedding structures or stored score artifacts.

### 1. Standard differential cosine score

For a global image embedding, the script computes cosine similarity to the positive and negative prompt vectors and subtracts the negative similarity from the positive similarity:

$$
S(d, x) = \cos(x, t_d^+) - \cos(x, t_d^-)
$$

where $x$ is the image embedding and $t_d^+$ and $t_d^-$ are the positive and negative prompt embeddings for disease $d$.

This path is used for MedSigLIP:

```text
score = cosine(image, positive_prompt) - cosine(image, negative_prompt)
```

A larger score indicates stronger relative alignment with the positive prompt.

### 2. CXR Foundation patch-wise max score

CXR Foundation image vectors are stored as a flattened 4,096-element representation that the script reshapes to `(32, 128)`. It computes positive and negative cosine similarities independently for each of the 32 components, then subtracts the maximum negative similarity from the maximum positive similarity:

$$
S(d, x) = \max_i \cos(x_i, t_d^+) - \max_i \cos(x_i, t_d^-)
$$

This is a different method from the global score. It is intended to preserve the strongest patch/token-level alignment rather than average all 32 components into one vector. It is used for CXR Foundation.

The reshape requires exactly 4,096 values and 128-dimensional prompt vectors. A shape mismatch indicates incompatible artifacts rather than a classification failure.

### 3. Precomputed continuous scores

For BioViL-T, the script reads continuous disease scores from the precomputed `phrases/BioViL-T` LanceDB table using `dicom_id` as the lookup key. It does not recompute BioViL-T scores from the `complete_phrases` prompt table in the current implementation.

These values are expected to contain one continuous score per disease in the table's `prediction` struct. Missing image IDs or missing disease values become `NaN` and are ignored by downstream metric calculations where appropriate.

This means BioViL-T's current path is operationally different from MedSigLIP and CXR Foundation. When reproducing or extending the experiment, record whether scores were recomputed from prompts or loaded from the precomputed table.

## Output

Run the batch evaluation from this directory:

```bash
cd exp3
uv sync
uv run zero_shot_classification.py
```

The script writes:

```text
zeroshot_evaluation_results.parquet
```

The Parquet file contains identifiers and labels, followed by one continuous score column per model and disease. Important column patterns are:

```text
dicom_id, study_id, subject_id, view_position
label_chexpert_<disease>
label_negbio_<disease>
score_MedSigLIP_<disease>
score_CXR_Foundation_<disease>
score_BioViL-T_<disease>
```

The output is the input to the Streamlit application. It is not a table of hard class predictions; AUROC and AUPRC should be computed from the continuous score columns.

## Interactive application

`app.py` provides an interactive ROC and precision-recall viewer for the saved Parquet output.

Start it with:

```bash
cd exp3
uv run streamlit run app.py
```

The application supports:

- Selecting `CheXpert` or `NegBio` as the ground-truth source.
- Viewing macro-average curves or one disease at a time.
- Comparing any subset of the three evaluated models.
- Mapping uncertain labels (`-1`) to positive, negative, or ignored.
- Mapping unmentioned/missing labels (`-2` or `NaN`) to negative or dropping them.
- Displaying random baselines and approximate 95% null regions for ROC and PR curves.

The default model selection contains exactly `MedSigLIP`, `CXR_Foundation`, and `BioViL-T`.

### Label policies in the app

For each disease, the app first removes rows with missing model scores. It then applies the selected label policies:

- `U-Ones`: map `-1` to `1`.
- `U-Zeros`: map `-1` to `0`.
- `U-Ignore`: remove `-1` rows.
- `Map -2 / NaN to 0`: treat unmentioned or missing labels as absent.
- `Drop -2 / NaN`: remove unmentioned or missing labels.

Only final labels in `{0, 1}` are passed to binary ROC and PR metric functions. Different policy choices can change the evaluated cohort and the resulting metrics, especially for rare diseases.

## Environment

Python 3.12 or newer is specified in `pyproject.toml`. Install the declared dependencies with `uv`:

```bash
cd exp3
uv sync
```

The environment includes LanceDB, Pandas, Plotly, scikit-learn, Seaborn, Streamlit, and supporting typing packages. Batch evaluation requires access to the embedding and phrase LanceDB directories. The interactive app requires the generated Parquet file in the current directory.

## Reproducibility checklist

1. Confirm that `fixed_embeddings_MIMIC-CXR-JPG` exists and contains the three required image columns.
2. Confirm that `complete_phrases` contains the required prompt columns for MedSigLIP and CXR Foundation.
3. Confirm that the BioViL-T precomputed score table uses the same `dicom_id` identifiers as the selected test images.
4. Record the exact test filter, including whether the PA-view restriction was applied.
5. Record the ground-truth source and uncertainty/unmentioned-label policy.
6. Keep model-specific image and text vectors in matching latent spaces.
7. Check score columns for missing values before computing metrics.
8. Preserve the generated Parquet file with the prompt-table and image-table versions used to create it.

## Interpretation

This experiment measures zero-shot image-text alignment, not the quality of a supervised classifier head. A high score means that the image is relatively closer to a positive prompt than to its paired negative prompt under the selected scoring method. Results can be affected by prompt wording, embedding normalization, patch aggregation, precomputed-score provenance, label policy, and disease prevalence.

The report compares this experiment with the supervised linear-probing results. The two experiments answer different questions: supervised probing asks how easily a frozen representation can be organized by fitted labels, while zero-shot classification asks whether the pretrained joint space itself provides clinically useful disease ordering.

These outputs are research measurements and are not clinical diagnoses or a validated clinical decision system.

## Related files

- `zero_shot_classification.py`: batch score generation.
- `app.py`: interactive ROC/PR analysis.
- `../embeddings/README.md`: embedding and phrase table schemas.
- `../eval/README.md`: model-specific extraction environments.
- `../exp1/README.md`: supervised structured-label classification.
- `../vis/README.md`: consolidated visualisation and metric workflows.
