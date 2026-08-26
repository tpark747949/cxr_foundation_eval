# Experiment 4: Cross-Modal Image-Report Retrieval

Experiment 4 evaluates whether foundation-model image and text representations can retrieve the matching clinical report for a chest X-ray, and retrieve the matching image for a report. It is the cross-modal retrieval experiment described in the project report.

The experiment compares three image-text foundation models:

- `MedSigLIP`
- `CXR_Foundation`
- `BioViL-T`

The retrieval implementation also contains a lookup path for precomputed CheXagent report scores in related artifacts, but `CheXagent` is not included in the active `MODELS` list in `retrieval.py`. Therefore, the current Experiment 4 run evaluates only the three models above.

## Research question

The task measures cross-modal alignment rather than supervised disease classification:

- Given an image, can the system rank its corresponding report highly?
- Given a report, can the system rank its corresponding image highly?
- Does aggregating Findings, Impression, or study-level representations change retrieval quality?

The project uses Recall@1, Recall@5, Recall@10, and mean reciprocal rank (MRR).

## Data sources

The retrieval script reads image embeddings from:

```text
../embeddings/MIMIC-CXR-JPG/complete_embeddings_MIMIC-CXR-JPG
```

and report embeddings from:

```text
../embeddings/reports/<model>
```

The three expected report tables are:

```text
../embeddings/reports/MedSigLIP
../embeddings/reports/CXR_Foundation
../embeddings/reports/BioViL-T
```

The image table is initially filtered to `split = 'test'`. For each model, the script then restricts images to studies present in that model's report table. This produces a model-specific aligned candidate pool.

The intended identifiers are:

- `dicom_id`: image identifier.
- `study_id`: image/report matching key.
- `subject_id`: patient identifier used for provenance and cohort analysis.

A clinical study may contain multiple images, while the report table contains report-level Findings and Impression representations. The retrieval code accounts for that many-images-to-one-study structure through both direct study matching and centroid evaluation.

## Retrieval directions

### Image-to-report (I2R)

Each image is a query. Candidate reports are ranked by descending image-report similarity. A result is correct when the retrieved report belongs to the same `study_id` as the image query.

### Report-to-image (R2I)

Each report is a query. Candidate images are ranked by descending report-image similarity. A result is correct when the retrieved image belongs to the same `study_id` as the report query.

The two directions have different query counts because multiple images can correspond to a single report and because missing Findings or Impression embeddings can remove candidates from one direction.

## Report representation strategies

The script evaluates four strategies, stored in the `Section` column.

### `findings`

Ranks images against the report's Findings embedding. Reports without a valid Findings vector are excluded from the relevant query/candidate calculation.

### `impression`

Ranks images against the report's Impression embedding. Reports without a valid Impression vector are excluded from the relevant calculation.

### `softmax`

Combines the Findings and Impression similarities using a log-add-exp operation:

$$
S_{soft}(x, r) = \log\left(\exp(S_f(x,r)) + \exp(S_i(x,r))\right)
$$

This uses whichever report sections are valid and combines their evidence in similarity space. It is named `softmax` in the output even though the implementation uses `torch.logaddexp`, not a probability softmax.

### `centroid_1to1`

Creates one image centroid and one report centroid per study:

- The image centroid is the mean of all image vectors belonging to the study, followed by L2 normalisation.
- The report centroid is the mean of the valid Findings and Impression vectors, followed by L2 normalisation.
- Each study becomes one image candidate and one report candidate.

This is the cleanest one-to-one study-level comparison because it removes the advantage or ambiguity introduced by studies with multiple views.

## Similarity computation

For MedSigLIP and BioViL-T, the script computes image-report similarity using matrix multiplication between image vectors and transposed report vectors. The embeddings are expected to be L2-normalised, so the dot product approximates cosine similarity.

CXR Foundation has a flattened 4,096-value image representation. The script reshapes each vector to `(32, 128)` and compares each of the 32 components with the report vector using `torch.einsum`. It then aggregates component similarities with `torch.logsumexp`:

```text
CXR Foundation image shape: (32, 128)
CXR Foundation report shape: (128,)
Aggregated score: logsumexp over the 32 component similarities
```

This is a model-specific aggregation and is not directly identical to the global dot-product calculation used for the other two models. Keep this distinction in mind when interpreting cross-model differences.

## Retrieval metrics

For each query, the code finds the rank of the first correct candidate. If a query has multiple correct candidates, the best-ranked matching candidate determines its rank.

For ranks $r_1, r_2, \ldots, r_n$:

$$
R@k = \frac{1}{n}\sum_{j=1}^{n} \mathbf{1}[r_j \leq k]
$$

and:

$$
MRR = \frac{1}{n}\sum_{j=1}^{n} \frac{1}{r_j}
$$

The output metrics are:

```text
I2R_R@1, I2R_R@5, I2R_R@10, I2R_MRR
R2I_R@1, R2I_R@5, R2I_R@10, R2I_MRR
```

`Candidate_Pool_Size` records the number of candidates ranked for the image-to-report calculation. `Query_Count_I2R` records the number of I2R queries with at least one valid matching candidate. The JSON artifact also includes `Query_Count_R2I`; the current CSV writer does not include that field.

## Running the evaluation

Python 3.12 or newer is specified in `pyproject.toml`. Install the environment:

```bash
cd exp4
uv sync
```

The retrieval script expects the repository to be located at `~/cxr_foundation_eval`, because its embedding paths are expanded from that home-directory location. Update `URI_IMAGES` and `URI_REPORTS` if the repository is elsewhere.

Run the batch evaluation with:

```bash
uv run retrieval.py
```

The script writes:

```text
evaluation_results_stratified.csv
```

The name is historical; the current script evaluates test-set retrieval strategies rather than performing the stratified label-efficiency sampling used in Experiment 2.

## Output schema

The current CSV contains one overall row for each model and report strategy:

```text
Model
Section
Disease
Label
I2R_R@1
I2R_R@5
I2R_R@10
I2R_MRR
R2I_R@1
R2I_R@5
R2I_R@10
R2I_MRR
Query_Count_I2R
Candidate_Pool_Size
```

For the current implementation, `Disease` is `Overall` and `Label` is `All`. Disease labels are not used to stratify the retrieval metrics in `retrieval.py`; they are retained as fixed output fields for compatibility with the wider evaluation tooling.

An accompanying `evaluation_results_stratified.json` is present in the repository. It contains a richer, older or separately generated result set with per-disease rows, CheXagent entries, and `Query_Count_R2I`. Do not combine it with the current CSV without confirming that both artifacts were generated from the same code, table versions, filters, and model set.

## Interactive dashboard

`app.py` provides a Streamlit dashboard for comparing retrieval metrics with random-ranking baselines.

Start it from this directory:

```bash
cd exp4
uv run streamlit run app.py
```

The dashboard supports:

- Image-to-report and report-to-image views.
- Findings, Impression, softmax, and one-to-one centroid strategies.
- Bar charts for R@1, R@5, R@10, and MRR.
- Approximate random-chance expectations.
- 95% null cutoffs calculated from a binomial model for Recall@k.
- An MRR baseline based on the expected reciprocal rank of a random ordering.

The dashboard loads `evaluation_results_stratified.csv` from the current working directory. Start it from `exp4` unless the path handling is changed.

## Null baselines

For a candidate pool of size $M$, the expected random-ranking recall is approximated by:

$$
E[R@k] = \min\left(\frac{k}{M}, 1\right)
$$

The dashboard treats the number of queries as $N$ and uses a binomial 95th-percentile cutoff to show how far an observed Recall@k must exceed random chance before it appears unusual under that null model.

The MRR baseline is calculated from the harmonic mean of reciprocal ranks under a uniformly random ordering. These are statistical reference points, not confidence intervals for model performance and not evidence of clinical utility.

## Reproducibility checklist

1. Confirm the image and report tables use matching model names and embedding spaces.
2. Confirm report `study_id` values match the image table's `study_id` values.
3. Verify whether the image table contains L2-normalised model vectors.
4. Record the test split filter and whether ignored QC images have already been removed.
5. Record the number of images and reports retained for each model after alignment.
6. Keep the Findings, Impression, softmax, and centroid strategy definitions unchanged when comparing models.
7. Treat CXR Foundation's `(32, 128)` aggregation separately from global dot-product similarity.
8. Preserve the exact CSV/JSON generation script because the checked-in artifacts have different schemas.
9. Compare both I2R and R2I; they answer different retrieval questions.

## Interpretation

Retrieval performance depends on candidate-pool size, study multiplicity, missing section embeddings, vector normalisation, and the choice of report aggregation. A higher Recall@k means the correct study appears within the first $k$ ranked candidates more often; it does not mean the report is clinically complete or that the model can generate a report.

The project report describes stronger retrieval performance for MedSigLIP than for the CXR-specific comparison models. This result should be interpreted as evidence about the evaluated image-report representation and ranking protocol, not as a universal claim about all checkpoints or all clinical institutions.

These outputs are research measurements and are not clinical diagnoses or a validated clinical decision system.

## Related material

- `../embeddings/README.md`: image and report embedding tables.
- `../exp3/README.md`: zero-shot image-text classification.
- `../vis/README.md`: broader result consolidation and visualisation conventions.
