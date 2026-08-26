# MIMIC-CXR Data Folder

This folder contains scripts, utilities, and configuration for processing, analyzing, and quality-checking the **MIMIC-CXR-JPG** dataset (v2.1.0), a large-scale collection of chest X-ray images from the Medical Information Mart for Intensive Care (MIMIC) database. The folder integrates with the embeddings pipeline to compute and analyze foundation model representations of CXR images.

## Overview

The MIMIC-CXR dataset contains:
- **~377,000 CXR images** from **~65,000 patients**
- **CheXpert-style labels** for 14 pathologies (positive, uncertain, negative, no-mention)
- **Structured metadata** including patient demographics, study information, and acquisition parameters
- **Quality control assessments** for image integrity and anomalies

This folder provides a complete pipeline for:
1. **Data profiling** - Statistical analysis of image properties and label distributions
2. **Quality control** - Detection of corrupted, collimated, or anomalous images
3. **Embedding computation** - Integration with foundation model embeddings from `../embeddings/`
4. **Consensus metrics** - Cross-model abnormality scoring using Z-scores
5. **Dataset analysis** - Comprehensive statistics on distribution, pathology prevalence, and patient representation

## Folder Structure

```
data/
├── MIMIC-CXR-JPG/              # Symlink/reference to the actual MIMIC-CXR-JPG dataset (2.1.0)
│   └── 2.1.0/                  # Official MIMIC-CXR-JPG release
│       ├── files/              # JPEG images organized by subject_id/study_id
│       ├── mimic-cxr-*.csv.gz  # Metadata splits (train/validate/test)
│       └── mimic-cxr-chexpert-labels.csv.gz  # CheXpert-style pathology labels
├── sample_images/              # Subset of sample images for quick testing
│
├── Python Scripts (Utilities & Analysis)
│   ├── main.py                 # Entry point placeholder
│   ├── app.py                  # Streamlit dashboard for QC exploration
│   ├── profile_mimic.py        # Compute QC metrics (image statistics, corruption detection)
│   ├── extract_suspicious.py   # Extract outlier images for manual review
│   ├── compute_zscores.py      # Calculate Z-scores from foundation model embeddings
│   ├── compute_consensus.py    # Combine Z-scores across models for consensus abnormality
│   ├── cosine_similarity.py    # Compute CXR-text similarity using MedSigLIP embeddings
│   ├── cosine_similarity_histogram.py  # Visualize similarity distribution
│   ├── analyses.py             # Pathology distribution and label analysis
│   ├── split_analysis.py       # Analyze train/validate/test split characteristics
│   ├── study_counts.py         # Dataset size statistics (patients, studies, images)
│   ├── matrix_analyses.py      # Cross-pathology and view-code analysis
│   └── soft_delete.py          # Flag excluded/suspicious images in embeddings
│
├── Data Files (Generated Outputs)
│   ├── qc_metrics.csv          # Image-level QC metrics (see QC Metrics section)
│   ├── embedding_zscores.csv   # Foundation model abnormality Z-scores per image
│   ├── suspicious_images.txt   # Paths to images flagged as outliers
│   └── mimic_exclude_list.txt  # DICOM IDs to exclude from analysis (soft-delete)
│
├── Configuration
│   ├── pyproject.toml          # Python dependencies and project metadata
│   ├── .python-version         # Python version specification
│   └── README.md               # This file
```

## Key Dependencies

From `pyproject.toml`:
- **lancedb** - Vector database for efficient embedding storage and retrieval
- **pandas** - Data manipulation and CSV processing
- **numpy** - Numerical computing
- **matplotlib / seaborn** - Visualization
- **Pillow** - Image loading and analysis
- **streamlit** - Interactive dashboard
- **tqdm** - Progress bars
- **polars** - Fast data processing (alternative to pandas)

## Script Descriptions

### Data Profiling & Quality Control

#### `profile_mimic.py`
Analyzes all MIMIC-CXR images to compute quality metrics. Runs in parallel across 16 workers.

**Metrics Computed:**
- `overall_mean` - Mean pixel intensity (detects underexposure/overexposure)
- `overall_std` - Standard deviation of pixel intensity (detects blank/dead-pixel scans)
- `min_half_mean` - Minimum mean intensity of top/bottom/left/right halves (detects collimation errors)
- `corner_mean` - Mean intensity of four 10×10 corner blocks (detects photometric inversion)
- `corrupted` - Boolean flag for unreadable/physically damaged images

**Output:** `qc_metrics.csv`

**Usage:**
```bash
python profile_mimic.py [--limit SAMPLE_LIMIT] [--workers NUM_WORKERS]
```

#### `extract_suspicious.py`
Identifies and exports outlier images for manual review based on QC metrics.

**Suspicion Categories:**
- **Collimation errors** - Lowest `min_half_mean` (half-black images)
- **Blank/dead pixel scans** - Lowest `overall_std` (no contrast)
- **Underexposed** - Lowest `overall_mean` (too dark)
- **Overexposed** - Highest `overall_mean` (too bright)
- **Photometric inversion** - Highest `corner_mean` (inverted pixel values)

**Output:** `suspicious_images.txt` (one path per line)

**Usage:**
```bash
python extract_suspicious.py --input qc_metrics.csv --output suspicious_images.txt --n 100
```

### Embedding Integration

#### `compute_zscores.py`
Converts raw embedding magnitudes into Z-scores for anomaly detection across multiple foundation models.

**Models Processed:**
- MedSigLIP
- CXR_Foundation
- BioViL-T
- EVA-X
- CheXagent
- CheXFound

**Computation:**
1. Loads embedding vectors from LanceDB
2. Computes L2 norm (magnitude) for each embedding
3. Calculates Z-scores: `(magnitude - mean) / std_dev`
4. Outputs per-model Z-scores and raw magnitudes

**Output:** `embedding_zscores.csv` with columns:
- `dicom_id`
- `{MODEL}_raw_mag` - Raw embedding magnitude
- `{MODEL}_mag_zscore` - Standardized abnormality score

**Usage:**
```bash
python compute_zscores.py
```

#### `compute_consensus.py`
Combines Z-scores across all models to create unified abnormality metrics.

**Consensus Methods:**
- `consensus_mean_zscore` - Average abnormality across all models
- `consensus_min_zscore` - Minimum abnormality (images anomalous in ALL models)

**Usage:**
```bash
python compute_consensus.py
```

#### `cosine_similarity.py`
Computes image-text similarity between CXR visual embeddings and a reference text embedding using MedSigLIP.

**Process:**
1. Loads MedSigLIP text embedding (pre-computed reference)
2. Retrieves all CXR visual embeddings from LanceDB
3. Computes cosine similarity: `sim = V · T / (||V|| × ||T||)`
4. Appends `cxr_similarity1` column to embeddings table

**Output:** Updated LanceDB table with `cxr_similarity1` column

**Usage:**
```bash
python cosine_similarity.py
```

#### `cosine_similarity_histogram.py`
Visualizes the distribution of CXR-text similarities.

**Output:** `cosine_similarity.png` histogram

**Usage:**
```bash
python cosine_similarity_histogram.py
```

#### `soft_delete.py`
Flags suspicious/excluded images in the LanceDB embeddings table without physically deleting them.

**Process:**
1. Reads DICOM IDs from `mimic_exclude_list.txt`
2. Creates binary `ignore` column in LanceDB table
3. Allows downstream analyses to filter these rows

**Usage:**
```bash
python soft_delete.py
```

### Dataset Analysis

#### `study_counts.py`
Computes basic dataset statistics and visualizations.

**Statistics Generated:**
- Total patients, studies, and images
- Distribution of images per study (histogram)
- Distribution of studies per patient (log scale)
- Distribution of images per patient (log scale)

**Data Source:** `MIMIC-CXR-JPG/2.1.0/mimic-cxr-2.0.0-split.csv.gz`

**Usage:**
```bash
python study_counts.py
```

#### `analyses.py`
Comprehensive pathology label analysis with multiple visualizations.

**Analyses:**
1. **Pathology Distribution** - Count of positive, uncertain, and negative labels per pathology
2. **Patient Representation Inequality** - Lorenz curves showing label distribution fairness across train/validate/test
3. **Label Mention Heatmap** - Cross-tabulation of pathology mentions by dataset split

**Pathologies Included:**
- Atelectasis, Cardiomegaly, Consolidation, Edema
- Enlarged_Cardiomediastinum, Fracture, Lung_Lesion
- Lung_Opacity, Pleural_Effusion, Pneumonia
- Pneumotharax, Pleural_Other, Support_Devices, No_Finding

**Usage:**
```bash
python analyses.py
```

#### `split_analysis.py`
Deep analysis of dataset splits (train/validate/test) with focus on pathology distribution and representation.

**Outputs:**
1. Positive pathology mention heatmap by view code and split
2. Patient representation inequality curves per pathology
3. Comorbidity burden analysis (concurrent pathologies per image)

**Usage:**
```bash
python split_analysis.py
```

#### `matrix_analyses.py`
Advanced matrix and heatmap analyses for pathology co-occurrence and view-specific patterns.

**Focus Areas:**
- Cross-pathology relationships
- View code (AP/PA/Lateral) specific patterns
- Label density and label correlation structures

**Usage:**
```bash
python matrix_analyses.py
```

### Interactive Dashboard

#### `app.py`
Streamlit-based interactive QC dashboard for exploring dataset quality and metrics.

**Features:**
- Load and filter QC metrics from `qc_metrics.csv`
- Display image previews from MIMIC-CXR-JPG
- Visualize Z-score distributions
- Search and sort by various QC criteria
- Integration with LanceDB for embedding-based retrieval

**Configuration:**
```python
DATA_DIR = "./MIMIC-CXR-JPG/2.1.0"
QC_METRICS_CSV = "./qc_metrics.csv"
ZSCORE_CSV = "./embedding_zscores.csv"
LANCEDB_URI = "../embeddings/MIMIC-CXR-JPG"
```

**Usage:**
```bash
streamlit run app.py
```

## Data Files

### `qc_metrics.csv`
Generated by `profile_mimic.py`. Contains image-level quality metrics.

**Columns:**
| Column | Type | Description |
|--------|------|-------------|
| `dicom_id` | str | Unique DICOM identifier |
| `path` | str | Relative path to JPEG file |
| `corrupted` | bool | True if image couldn't be loaded |
| `overall_mean` | float | Mean pixel intensity (0-255) |
| `overall_std` | float | Standard deviation of pixel intensity |
| `min_half_mean` | float | Minimum mean of four image halves |
| `corner_mean` | float | Mean intensity of corner blocks |

### `embedding_zscores.csv`
Generated by `compute_zscores.py`. Contains foundation model abnormality metrics.

**Columns:**
| Column | Type | Description |
|--------|------|-------------|
| `dicom_id` | str | Unique DICOM identifier |
| `{MODEL}_raw_mag` | float | L2 norm of embedding vector |
| `{MODEL}_mag_zscore` | float | Z-score of embedding magnitude |
| `consensus_mean_zscore` | float | Average Z-score across models |
| `consensus_min_zscore` | float | Minimum Z-score across models |

### `suspicious_images.txt`
Generated by `extract_suspicious.py`. Contains file paths of flagged outliers.

**Format:** One path per line, relative to MIMIC-CXR-JPG root

### `mimic_exclude_list.txt`
Manually curated list of DICOM IDs to exclude from analysis.

**Format:** One DICOM ID per line

**Used by:** `soft_delete.py` to flag images in embeddings

## Workflow: From Raw Data to Analyzed Embeddings

### 1. Initial Setup
```bash
# Ensure MIMIC-CXR-JPG dataset is available (symlinked or copied)
# and embeddings are pre-computed in ../embeddings/MIMIC-CXR-JPG
```

### 2. Quality Profiling
```bash
python profile_mimic.py
# → Generates qc_metrics.csv
```

### 3. Identify Outliers
```bash
python extract_suspicious.py
# → Generates suspicious_images.txt
# → Review manually and update mimic_exclude_list.txt
```

### 4. Compute Abnormality Scores
```bash
python compute_zscores.py
# → Generates embedding_zscores.csv with model Z-scores
```

### 5. Create Consensus Metrics
```bash
python compute_consensus.py
# → Appends consensus_*_zscore columns to embedding_zscores.csv
```

### 6. Compute Image-Text Similarity
```bash
python cosine_similarity.py
# → Computes similarity scores with text embeddings
python cosine_similarity_histogram.py
# → Visualizes distribution
```

### 7. Flag Excluded Images
```bash
python soft_delete.py
# → Adds 'ignore' column to LanceDB embeddings table
```

### 8. Dataset Analysis
```bash
python study_counts.py      # Basic statistics
python analyses.py          # Pathology distribution
python split_analysis.py    # Train/validate/test characteristics
python matrix_analyses.py   # Advanced cross-tabulations
```

### 9. Interactive Exploration
```bash
streamlit run app.py
# → Launch dashboard at http://localhost:8501
```

## LanceDB Integration

All scripts that interact with embeddings use **LanceDB**, a vector database optimized for AI workloads.

**Database Location:** `../embeddings/MIMIC-CXR-JPG`

**Main Table:** `complete_embeddings_MIMIC-CXR-JPG` (or `fixed_embeddings_MIMIC-CXR-JPG`)

**Columns Include:**
- `dicom_id` - Unique identifier
- `path` - File path
- `{MODEL}_raw` - Raw embedding vector (list of floats)
- `{MODEL}_l2` - L2-normalized embedding
- `CheXpert_labels` - Struct of pathology labels (1/-1/0/-2)
- Additional metadata (view code, demographics, etc.)

**Connection Pattern:**
```python
import lancedb
db = lancedb.connect("../embeddings/MIMIC-CXR-JPG")
table = db.open_table("complete_embeddings_MIMIC-CXR-JPG")
df = table.to_pandas()  # or table.to_arrow() for PyArrow
```

## Key Metrics & Visualizations

### From Attached Images

1. **cxr_similarity1 Histogram** - Distribution of image-text similarity scores (bimodal, centered near 0)

2. **Distribution Plots**
   - Images per study: Most studies have 1-2 images (right-skewed)
   - Studies per patient: Most patients have 1-20 studies (log-scale heavy tail)
   - Images per patient: Similar pattern to studies (median ~5 images)

3. **Pathology Label Distribution** (Bar Chart)
   - "No Finding" is most common (~141k mentions)
   - Support Devices and Pleural Effusion also frequent
   - Fracture and Pleural Other are rare

4. **View Code × Pathology Heatmap**
   - Antero-Posterior (AP) views dominate dataset
   - Different pathologies detected at different rates by view
   - Pneumonia and Edema have strong AP bias

5. **Image Dimension Density** (Scatter)
   - Most images are ~2500×3000 pixels
   - Two clusters visible (different equipment/protocols)
   - Outliers at extreme dimensions

6. **Patient Representation (Lorenz Curves)**
   - Slight inequality in label distribution
   - Train/validate/test splits are relatively balanced
   - All curves similarly bowed

7. **Comorbidity Burden** (Box Plot)
   - Average ~2 pathologies per image
   - Ranges from 0-8+ concurrent findings
   - Similar distributions across splits

## Running Analysis in Batch Mode

All scripts support command-line invocation:

```bash
# Full pipeline execution
python profile_mimic.py && \
python extract_suspicious.py && \
python compute_zscores.py && \
python compute_consensus.py && \
python cosine_similarity.py && \
python soft_delete.py

# Analysis visualizations
python study_counts.py
python analyses.py
python split_analysis.py
python matrix_analyses.py
```

## Notes & Considerations

- **Data Privacy:** MIMIC-CXR requires credentialed access; ensure compliance with data use agreements
- **Disk Space:** Full dataset ~150 GB; consider using `SAMPLE_LIMIT` in `profile_mimic.py` for testing
- **GPU Acceleration:** `cosine_similarity.py` uses GPU if available (CUDA); falls back to CPU
- **Performance:** LanceDB query scans are optimized for large-scale analysis but memory usage scales with table size
- **Metadata Caching:** Streamlit app uses `@st.cache_resource` and `@st.cache_data` for performance; clear cache if data changes

## Future Enhancements

- [ ] Automated data validation pipeline
- [ ] Integration with model-agnostic interpretability tools
- [ ] Multi-modal contrastive analysis (image-report-label triples)
- [ ] Federated learning pipeline for decentralized model training
- [ ] Real-time streaming QC for production deployments
