import os
import io
import png
import logging
import pandas as pd
import numpy as np
import pyarrow as pa
import lancedb
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Suppress overly verbose TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import tensorflow_text

# ==========================================
# 0. LOGGING & TENSORFLOW MEMORY SETUP
# ==========================================
logging.basicConfig(
    filename="mimic_processing_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.error("Testing connection.")

# Prevent TensorFlow from instantly hogging all 48GB of A6000 VRAM
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"TF Memory Growth Error: {e}")

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
DATA_DIR = "../../data/MIMIC-CXR-JPG/2.1.0"  # Change to your symlink target
METADATA_DIR = "../../data/MIMIC-CXR-JPG/2.1.0"
URI = "../../embeddings/MIMIC-CXR-JPG"
BATCH_SIZE = 128                                     # Adjust based on memory consumption
NUM_WORKERS = 8                                      # Multi-process CPU data loading
WRITE_BUFFER_SIZE = 2500                             # LanceDB chunking

# IMPORTANT: Set this to the flattened size of the CXR-Foundation output.
# If output is [batch, 32, 128], flattened dim is 32 * 128 = 4096.
EMBEDDING_DIM = 4096

# ==========================================
# 1. LOAD AND MERGE METADATA (CHEXPERT + NEGBIO)
# ==========================================
print("Loading and indexing metadata files...")

with open(os.path.join(METADATA_DIR, "IMAGE_FILENAMES"), "r") as f:
    paths = [line.strip() for line in f if line.strip()]
df_paths = pd.DataFrame({"path": paths})
df_paths["dicom_id"] = df_paths["path"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

df_meta = pd.read_csv(os.path.join(METADATA_DIR, "mimic-cxr-2.0.0-metadata.csv.gz"))
df_split = pd.read_csv(os.path.join(METADATA_DIR, "mimic-cxr-2.0.0-split.csv.gz"))
df_chexpert = pd.read_csv(os.path.join(METADATA_DIR, "mimic-cxr-2.0.0-chexpert.csv.gz"))
df_negbio = pd.read_csv(os.path.join(METADATA_DIR, "mimic-cxr-2.0.0-negbio.csv.gz"))

label_cols = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
    "Fracture", "Lung Lesion", "Lung Opacity", "Pleural Effusion", "Pneumonia",
    "Pneumothorax", "Pleural Other", "Support Devices", "No Finding"
]

# Rename columns to prevent collision
chex_rename = {col: f"{col.replace(' ', '_')}_chex" for col in label_cols}
neg_rename = {col: f"{col.replace(' ', '_')}_neg" for col in label_cols}
df_chexpert = df_chexpert.rename(columns=chex_rename)
df_negbio = df_negbio.rename(columns=neg_rename)

# Merges
df_master = df_paths.merge(df_meta, on="dicom_id", how="left")
df_master = df_master.merge(df_split[["dicom_id", "split"]], on="dicom_id", how="left")
df_master = df_master.merge(df_chexpert, on=["subject_id", "study_id"], how="left")
df_master = df_master.merge(df_negbio, on=["subject_id", "study_id"], how="left")

# Hardened Numeric Coercion
df_master["Rows"] = pd.to_numeric(df_master["Rows"], errors="coerce").fillna(0).astype(np.uint16)
df_master["Columns"] = pd.to_numeric(df_master["Columns"], errors="coerce").fillna(0).astype(np.uint16)
df_master["study_id"] = pd.to_numeric(df_master["study_id"], errors="coerce").fillna(0).astype(np.uint32)
df_master["subject_id"] = pd.to_numeric(df_master["subject_id"], errors="coerce").fillna(0).astype(np.uint32)

clean_label_cols = [col.replace(' ', '_') for col in label_cols]
for col in clean_label_cols:
    df_master[f"{col}_chex"] = pd.to_numeric(df_master[f"{col}_chex"], errors="coerce").fillna(-2).astype(np.int8)
    df_master[f"{col}_neg"] = pd.to_numeric(df_master[f"{col}_neg"], errors="coerce").fillna(-2).astype(np.int8)

print(f"Total records to process: {len(df_master)}")

# ==========================================
# 2. PYTORCH DATASET DESIGN (CPU MULTIPROCESSING)
# ==========================================
def png_to_tfexample(image_array: np.ndarray) -> bytes:
    """Creates a serialized tf.train.Example from a NumPy array."""
    image = image_array.astype(np.float32)
    image -= image.min()

    if image_array.dtype == np.uint8:
        pixel_array = image.astype(np.uint8)
        bitdepth = 8
    else:
        max_val = image.max()
        if max_val > 0:
            image *= 65535 / max_val
        pixel_array = image.astype(np.uint16)
        bitdepth = 16

    output = io.BytesIO()
    png.Writer(
        width=pixel_array.shape[1],
        height=pixel_array.shape[0],
        greyscale=True,
        bitdepth=bitdepth
    ).write(output, pixel_array.tolist())
    
    example = tf.train.Example()
    features = example.features.feature
    features['image/encoded'].bytes_list.value.append(output.getvalue())
    features['image/format'].bytes_list.value.append(b'png')

    return example.SerializeToString()

class MimicCxrTfDataset(Dataset):
    def __init__(self, dataframe, base_dir):
        self.df = dataframe
        self.base_dir = base_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.base_dir, row["path"])
        
        try:
            with Image.open(img_path) as img:
                img_array = np.array(img.convert("L"))
                
            serialized_str = png_to_tfexample(img_array)
            return serialized_str, idx, True
            
        except Exception as e:
            logging.error(f"Failed at {idx} | Path: {img_path} | Error: {str(e)}")
            return b"", idx, False

dataloader = DataLoader(
    MimicCxrTfDataset(df_master, DATA_DIR), 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS,
    pin_memory=False 
)

# ==========================================
# 3. LANCEDB SETTING
# ==========================================
db = lancedb.connect(URI)

schema = pa.schema([
    pa.field("path", pa.string()),
    pa.field("dicom_id", pa.string()),
    pa.field("PerformedProcedureStepDescription", pa.string()),
    pa.field("ViewPosition", pa.string()),
    pa.field("image_size", pa.struct([pa.field("Rows", pa.uint16()), pa.field("Columns", pa.uint16())])),
    pa.field("StudyDate", pa.string()),
    pa.field("StudyTime", pa.string()),
    pa.field("ProcedureCodeSequence_CodeMeaning", pa.string()),
    pa.field("ViewCodeSequence_CodeMeaning", pa.string()),
    pa.field("PatientOrientationCodeSequence_CodeMeaning", pa.string()),
    pa.field("study_id", pa.uint32()),   
    pa.field("subject_id", pa.uint32()), 
    pa.field("split", pa.string()),
    pa.field("CheXpert_labels", pa.struct([pa.field(col, pa.int8()) for col in clean_label_cols])),
    pa.field("NegBio_labels", pa.struct([pa.field(col, pa.int8()) for col in clean_label_cols])),
    pa.field("embedding_raw", pa.list_(pa.float32(), EMBEDDING_DIM)),
    pa.field("embedding_l2", pa.list_(pa.float32(), EMBEDDING_DIM)),
])

table = db.create_table("CXR_Foundation_embeddings_MIMIC-CXR-JPG", schema=schema, mode="overwrite")

# ==========================================
# 4. INITIALIZE TF MODELS
# ==========================================
print("Loading CXR-Foundation models...")
elixrc_model = tf.saved_model.load('./checkpoints/hf/elixr-c-v2-pooled')
elixrc_infer = elixrc_model.signatures['serving_default']

qformer_model = tf.saved_model.load("./checkpoints/hf/pax-elixr-b-text")
qformer_infer = qformer_model.signatures['serving_default']

# ==========================================
# 5. INFERENCE & DATABASE POPULATION (BATCH=1 FOR TF)
# ==========================================
print("Starting inference pipeline...")
buffer_records = []

# Pre-allocate the static zero tensors required by the Q-Former (Batch Size MUST be 1)
static_ids = tf.zeros((1, 1, 128), dtype=tf.int32)
static_paddings = tf.zeros((1, 1, 128), dtype=tf.float32)

for batch_strings, indices, success_flags in dataloader:
    indices = indices.numpy()
    success_flags = success_flags.numpy()
    
    raw_embeddings_batch = [None] * len(indices)
    l2_embeddings_batch = [None] * len(indices)
    
    valid_indices = [i for i, success in enumerate(success_flags) if success]
    
    if valid_indices:
        valid_strings = [batch_strings[i] for i in valid_indices]
        
        # Iterate through the pre-processed batch one by one for TensorFlow
        for exact_gpu_idx, original_batch_idx in enumerate(valid_indices):
            single_string = valid_strings[exact_gpu_idx]
            
            # 1. ELIXR-C Pass (Requires a 1D tensor of shape (1,))
            tf_string = tf.constant([single_string])
            elixrc_output = elixrc_infer(input_example=tf_string)
            elixrc_embedding = elixrc_output['feature_maps_0']
            
            # 2. QFormer Pass (Requires image_feature shape (1, 8, 8, 1376))
            qformer_input = {
                'image_feature': elixrc_embedding,
                'ids': static_ids,
                'paddings': static_paddings,
            }
            qformer_output = qformer_infer(**qformer_input)
            
            # 3. Flatten & Normalize
            raw_out = qformer_output['all_contrastive_img_emb']  # Shape is (1, 32, 128)
            
            # Flatten to a 1D tensor of shape (4096,)
            raw_out_flat = tf.reshape(raw_out, (-1,))
            l2_out_flat = tf.math.l2_normalize(raw_out_flat, axis=-1)
            
            # Map back to positions
            raw_embeddings_batch[original_batch_idx] = raw_out_flat.numpy().tolist()
            l2_embeddings_batch[original_batch_idx] = l2_out_flat.numpy().tolist()

    # Build rows
    for idx_in_batch, global_idx in enumerate(indices):
        row = df_master.iloc[global_idx]
        
        record = {
            "path": str(row["path"]),
            "dicom_id": str(row["dicom_id"]),
            "PerformedProcedureStepDescription": str(row["PerformedProcedureStepDescription"]) if pd.notna(row["PerformedProcedureStepDescription"]) else None,
            "ViewPosition": str(row["ViewPosition"]) if pd.notna(row["ViewPosition"]) else None,
            "image_size": {"Rows": int(row["Rows"]), "Columns": int(row["Columns"])},
            "StudyDate": str(row["StudyDate"]) if pd.notna(row["StudyDate"]) else None,
            "StudyTime": str(row["StudyTime"]) if pd.notna(row["StudyTime"]) else None,
            "ProcedureCodeSequence_CodeMeaning": str(row["ProcedureCodeSequence_CodeMeaning"]) if pd.notna(row["ProcedureCodeSequence_CodeMeaning"]) else None,
            "ViewCodeSequence_CodeMeaning": str(row["ViewCodeSequence_CodeMeaning"]) if pd.notna(row["ViewCodeSequence_CodeMeaning"]) else None,
            "PatientOrientationCodeSequence_CodeMeaning": str(row["PatientOrientationCodeSequence_CodeMeaning"]) if pd.notna(row["PatientOrientationCodeSequence_CodeMeaning"]) else None,
            "study_id": int(row["study_id"]),
            "subject_id": int(row["subject_id"]),
            "split": str(row["split"]) if pd.notna(row["split"]) else None,
            "CheXpert_labels": {col: int(row[f"{col}_chex"]) for col in clean_label_cols},
            "NegBio_labels": {col: int(row[f"{col}_neg"]) for col in clean_label_cols},
            "embedding_raw": raw_embeddings_batch[idx_in_batch],
            "embedding_l2": l2_embeddings_batch[idx_in_batch],
        }
        buffer_records.append(record)
        
    # Write to LanceDB in chunks safely
    if len(buffer_records) >= WRITE_BUFFER_SIZE:
        try:
            table.add(buffer_records)
            print(f"Indexed up to row {global_idx + 1}/{len(df_master)}")
        except Exception as e:
            logging.error(f"CRITICAL: Failed to insert batch ending at row {global_idx}. Error: {str(e)}")
        finally:
            buffer_records = []

# Final flush
if buffer_records:
    try:
        table.add(buffer_records)
    except Exception as e:
        logging.error(f"CRITICAL: Failed to insert final batch. Error: {str(e)}")

print(f"Finished pipeline! Table total rows: {len(table)}")