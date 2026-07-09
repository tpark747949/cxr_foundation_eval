import os
import gzip
import pandas as pd
import numpy as np
import pyarrow as pa
import lancedb
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoProcessor, SiglipVisionModel

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
DATA_DIR = "../../data/MIMIC-CXR-JPG/2.1.0"  # Change to your symlink target
METADATA_DIR = "../../data/MIMIC-CXR-JPG/2.1.0"                 # Directory containing your csv.gz files
URI = "../../embeddings/MIMIC-CXR-JPG"
BATCH_SIZE = 256                                     # Optimized for A6000 48GB VRAM
NUM_WORKERS = 8                                      # Multi-process data loading
INPUT_DIMENSION = (448, 448)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# =========================================
# Set up logger
# =========================================
import logging

# Configure logging to write to a file
logging.basicConfig(
    filename="mimic_processing_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.error("Testing connection.")

# ==========================================
# 1. LOAD AND MERGE METADATA
# ==========================================
print("Loading and indexing metadata files...")

# Read IMAGE_FILENAMES.txt
with open(os.path.join(METADATA_DIR, "IMAGE_FILENAMES"), "r") as f:
    paths = [line.strip() for line in f if line.strip()]
df_paths = pd.DataFrame({"path": paths})
# Extract dicom_id from the filename stem
df_paths["dicom_id"] = df_paths["path"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

# Read metadata CSVs
# Read metadata CSVs
df_meta = pd.read_csv(os.path.join(METADATA_DIR, "mimic-cxr-2.0.0-metadata.csv.gz"))
df_split = pd.read_csv(os.path.join(METADATA_DIR, "mimic-cxr-2.0.0-split.csv.gz"))
df_chexpert = pd.read_csv(os.path.join(METADATA_DIR, "mimic-cxr-2.0.0-chexpert.csv.gz"))
df_negbio = pd.read_csv(os.path.join(METADATA_DIR, "mimic-cxr-2.0.0-negbio.csv.gz"))

label_cols = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
    "Fracture", "Lung Lesion", "Lung Opacity", "Pleural Effusion", "Pneumonia",
    "Pneumothorax", "Pleural Other", "Support Devices", "No Finding"
]

# Create renamed column mapping to avoid collisions (and replace spaces with underscores)
chex_rename = {col: f"{col.replace(' ', '_')}_chex" for col in label_cols}
neg_rename = {col: f"{col.replace(' ', '_')}_neg" for col in label_cols}

df_chexpert = df_chexpert.rename(columns=chex_rename)
df_negbio = df_negbio.rename(columns=neg_rename)

# Stepwise merge
df_master = df_paths.merge(df_meta, on="dicom_id", how="left")
df_master = df_master.merge(df_split[["dicom_id", "split"]], on="dicom_id", how="left")
df_master = df_master.merge(df_chexpert, on=["subject_id", "study_id"], how="left")
df_master = df_master.merge(df_negbio, on=["subject_id", "study_id"], how="left")

# Hardened coercion (Handling missing data)
df_master["Rows"] = pd.to_numeric(df_master["Rows"], errors="coerce").fillna(0).astype(np.uint16)
df_master["Columns"] = pd.to_numeric(df_master["Columns"], errors="coerce").fillna(0).astype(np.uint16)
df_master["study_id"] = pd.to_numeric(df_master["study_id"], errors="coerce").fillna(0).astype(np.uint32)
df_master["subject_id"] = pd.to_numeric(df_master["subject_id"], errors="coerce").fillna(0).astype(np.uint32)

# Apply our -2 sentinel value to both sets of labels
clean_label_cols = [col.replace(' ', '_') for col in label_cols]
for col in clean_label_cols:
    df_master[f"{col}_chex"] = pd.to_numeric(df_master[f"{col}_chex"], errors="coerce").fillna(-2).astype(np.int8)
    df_master[f"{col}_neg"] = pd.to_numeric(df_master[f"{col}_neg"], errors="coerce").fillna(-2).astype(np.int8)

print(f"Total records to process: {len(df_master)}")

# ==========================================
# 2. PYTORCH DATASET DESIGN
# ==========================================
class MimicCxrDataset(Dataset):
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
                img = img.convert("RGB").resize(INPUT_DIMENSION, Image.Resampling.BILINEAR)
                img_array = np.array(img)
            return img_array, idx, True  # True = Success
            
        except Exception as e:
            # Log the exact file and the error reason
            logging.error(f"Failed to load image at index {idx} | Path: {img_path} | Error: {str(e)}")
            
            # Return a dummy array and False so the main loop knows to handle it
            dummy_array = np.zeros((INPUT_DIMENSION[0], INPUT_DIMENSION[1], 3), dtype=np.uint8)
            return dummy_array, idx, False  # False = Corrupted/Missing

dataset = MimicCxrDataset(df_master, DATA_DIR)
dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS, 
    pin_memory=True
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
    pa.field("study_id", pa.uint32()),   # Upgraded to uint32 to handle 8-digit IDs
    pa.field("subject_id", pa.uint32()), # Upgraded to uint32 to handle 8-digit IDs
    pa.field("split", pa.string()),
    pa.field("CheXpert_labels", pa.struct([pa.field(col, pa.int8()) for col in clean_label_cols])),
    pa.field("NegBio_labels", pa.struct([pa.field(col, pa.int8()) for col in clean_label_cols])),
    pa.field("embedding_raw", pa.list_(pa.float32(), 1152)),
    pa.field("embedding_l2", pa.list_(pa.float32(), 1152)),
])

table = db.create_table("MedSigLIP_embeddings_MIMIC-CXR-JPG", schema=schema, mode="overwrite")

# ==========================================
# 4. INITIALIZE MODEL
# ==========================================
print("Loading MedSigLIP model...")
model = SiglipVisionModel.from_pretrained("google/medsiglip-448").to(DEVICE)
processor = AutoProcessor.from_pretrained("google/medsiglip-448")
model.eval()

# ==========================================
# 5. BATCH INFERENCE & DATABASE POPULATION
# ==========================================
print("Starting inference pipeline...")

# We accumulate records in chunks before writing to LanceDB for maximum I/O performance
buffer_records = []
WRITE_BUFFER_SIZE = 5000 

with torch.no_grad():
    for imgs_batch, indices, success_flags in dataloader:
        
        # Convert tensors to numpy/lists for easier handling
        indices = indices.numpy()
        success_flags = success_flags.numpy()
        
        # Track embeddings for this entire batch (including potential Nones)
        raw_embeddings_batch = [None] * len(indices)
        l2_embeddings_batch = [None] * len(indices)
        
        # Filter out only the valid images to send to the A6000
        valid_indices_in_batch = [i for i, success in enumerate(success_flags) if success]
        
        if valid_indices_in_batch:
            # Extract only valid images for the GPU
            valid_imgs = [Image.fromarray(imgs_batch[i].numpy()) for i in valid_indices_in_batch]
            
            inputs = processor(images=valid_imgs, padding="max_length", return_tensors="pt").to(DEVICE)
            outputs = model(**inputs)
            
            raw_outs = outputs["pooler_output"]
            l2_outs = F.normalize(raw_outs, p=2, dim=-1)
            
            # Move back to CPU and map them back to their correct batch positions
            raw_outs = raw_outs.cpu().numpy()
            l2_outs = l2_outs.cpu().numpy()
            
            for exact_gpu_idx, original_batch_idx in enumerate(valid_indices_in_batch):
                raw_embeddings_batch[original_batch_idx] = raw_outs[exact_gpu_idx].tolist()
                l2_embeddings_batch[original_batch_idx] = l2_outs[exact_gpu_idx].tolist()

        # Build records for LanceDB
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
                # Will be a list of floats if successful, or None if the file was corrupt
                "embedding_raw": raw_embeddings_batch[idx_in_batch],
                "embedding_l2": l2_embeddings_batch[idx_in_batch],
            }
            buffer_records.append(record)
            
        # Write to LanceDB in chunks
        if len(buffer_records) >= WRITE_BUFFER_SIZE:
            try:
                table.add(buffer_records)
                print(f"Indexed up to row {global_idx + 1}/{len(df_master)}")
            except Exception as e:
                # If PyArrow rejects the batch due to a schema error, log it and keep going!
                logging.error(f"CRITICAL: Failed to insert batch ending at row {global_idx}. Error: {str(e)}")
            finally:
                # Clear the buffer regardless of success or failure to avoid infinite loops
                buffer_records = []
# Append any remaining records
if buffer_records:
    table.add(buffer_records)

print(f"Finished pipeline! Table total rows: {len(table)}")