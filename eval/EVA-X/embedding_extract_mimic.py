import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import lancedb
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ==========================================
# 0. DIRECTORY & PATH MANAGEMENT
# ==========================================
# Point Python to your cloned EVA-X subfolder
EVA_X_DIR = Path(__file__).parent / "EVA-X" if "__file__" in locals() else Path("./EVA-X")
sys.path.append(str(EVA_X_DIR))

from eva_x import eva_x_base_patch16

# ==========================================
# 1. CONFIGURATION & LOGGING
# ==========================================
logging.basicConfig(
    filename="evax_processing_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

DATA_DIR = "../../data/MIMIC-CXR-JPG/2.1.0"  # Change to your symlink target
METADATA_DIR = "../../data/MIMIC-CXR-JPG/2.1.0"                 # Directory containing your csv.gz files
URI = "../../embeddings/MIMIC-CXR-JPG"
CHECKPOINT_PATH = "checkpoints/eva_x_base_patch16_merged520k_mim.pt"

BATCH_SIZE = 1280             # A6000 (48GB) can easily swallow a batch size of 256 for a Base ViT
NUM_WORKERS = 8              
WRITE_BUFFER_SIZE = 5000     

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==========================================
# 2. LOAD AND MERGE METADATA
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

df_chexpert = df_chexpert.rename(columns={col: f"{col.replace(' ', '_')}_chex" for col in label_cols})
df_negbio = df_negbio.rename(columns={col: f"{col.replace(' ', '_')}_neg" for col in label_cols})

df_master = df_paths.merge(df_meta, on="dicom_id", how="left")
df_master = df_master.merge(df_split[["dicom_id", "split"]], on="dicom_id", how="left")
df_master = df_master.merge(df_chexpert, on=["subject_id", "study_id"], how="left")
df_master = df_master.merge(df_negbio, on=["subject_id", "study_id"], how="left")

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
# 3. PYTORCH DATASET DESIGN (WITH CHANNELS MATCHING)
# ==========================================
class MimicCxrEvaXDataset(Dataset):
    def __init__(self, dataframe, base_dir):
        self.df = dataframe
        self.base_dir = base_dir
        
        # Matches your precise input transformation requirements
        self.transform = T.Compose([
            T.Resize((224, 224)),                   
            T.ToTensor(),                           
            T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x[:3, :, :]), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.base_dir, row["path"])
        
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(img)
            return img_tensor, idx, True
            
        except Exception as e:
            logging.error(f"Failed at {idx} | Path: {img_path} | Error: {str(e)}")
            return torch.zeros((3, 224, 224), dtype=torch.float32), idx, False

dataloader = DataLoader(
    MimicCxrEvaXDataset(df_master, DATA_DIR), 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# ==========================================
# 4. INITIALIZE EVA-X & DYNAMIC SHAPE CHECK
# ==========================================
print("Loading EVA-X model and checkpoint...")
image_model = eva_x_base_patch16(pretrained=CHECKPOINT_PATH)
image_model = image_model.to(DEVICE)
image_model.eval()

print("Performing dynamic dimension probe...")
with torch.no_grad():
    dummy_input = torch.zeros((1, 3, 224, 224), device=DEVICE)
    
    # Use forward_features instead of a standard forward pass
    dummy_features = image_model.forward_features(dummy_input)
    
    # A ViT returns [Batch, Num_Tokens, Hidden_Dim]. E.g., [1, 197, 768]
    # We slice [:, 0, :] to isolate the CLS token for the global representation
    if len(dummy_features.shape) == 3:
        dummy_features = dummy_features[:, 0, :]
        
    EMBEDDING_DIM = dummy_features.shape[-1]
    
print(f"Dynamic Check Successful! Model feature dimensionality: {EMBEDDING_DIM}")

# ==========================================
# 5. LANCEDB SCHEMATICS
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

table = db.create_table("EVA-X_embeddings_MIMIC-CXR-JPG", schema=schema, mode="overwrite")

# ==========================================
# 6. PIPELINED INFERENCE LOOP
# ==========================================
print("Starting EVA-X inference pipeline...")
buffer_records = []

with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
    for imgs_batch, indices, success_flags in dataloader:
        
        success_mask = success_flags.bool()
        raw_embeddings_batch = [None] * len(indices)
        l2_embeddings_batch = [None] * len(indices)
        
        if success_mask.any():
            valid_imgs = imgs_batch[success_mask].to(DEVICE, non_blocking=True)
            
            # --- NEW LOGIC: Extract raw features ---
            raw_outs = image_model.forward_features(valid_imgs)
            
            # Isolate the CLS token (Index 0 of the token sequence)
            if len(raw_outs.shape) == 3:
                raw_outs = raw_outs[:, 0, :]
            
            # L2 Normalize the 768-dimensional feature vector
            l2_outs = raw_outs / raw_outs.norm(p=2, dim=-1, keepdim=True)
            
            # Cast down safely to single precision floats for standard array serialization
            raw_outs_np = raw_outs.cpu().to(torch.float32).numpy()
            l2_outs_np = l2_outs.cpu().to(torch.float32).numpy()
            
            valid_indices = success_mask.nonzero(as_tuple=True)[0].tolist()
            for exact_gpu_idx, original_batch_idx in enumerate(valid_indices):
                raw_embeddings_batch[original_batch_idx] = raw_outs_np[exact_gpu_idx].tolist()
                l2_embeddings_batch[original_batch_idx] = l2_outs_np[exact_gpu_idx].tolist()

        # Compile database writes
        indices_np = indices.numpy()
        for idx_in_batch, global_idx in enumerate(indices_np):
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
            
        # Commit to LanceDB disk store
        if len(buffer_records) >= WRITE_BUFFER_SIZE:
            try:
                table.add(buffer_records)
                print(f"Indexed up to row {global_idx + 1}/{len(df_master)}")
            except Exception as e:
                logging.error(f"CRITICAL: Failed to insert batch ending at row {global_idx}. Error: {str(e)}")
            finally:
                buffer_records = []

# Final structural clean-up flush
if buffer_records:
    try:
        table.add(buffer_records)
    except Exception as e:
        logging.error(f"CRITICAL: Failed to insert final batch. Error: {str(e)}")

print(f"Finished pipeline! Table total rows: {len(table)}")