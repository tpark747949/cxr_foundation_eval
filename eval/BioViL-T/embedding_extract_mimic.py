import os
import logging
import pandas as pd
import numpy as np
import pyarrow as pa
import lancedb
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download
from PIL import Image

# ==========================================
# 0. CONFIGURATION & LOGGING
# ==========================================
logging.basicConfig(
    filename="biovilt_processing_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

DATA_DIR = "../../data/MIMIC-CXR-JPG/2.1.0"  # Change to your symlink target
METADATA_DIR = "../../data/MIMIC-CXR-JPG/2.1.0"                 # Directory containing your csv.gz files
URI = "../../embeddings/MIMIC-CXR-JPG"

BATCH_SIZE = 1024             # BioViL-T (ResNet50) is very lightweight, 256 or 512 will easily fit on an A6000
NUM_WORKERS = 8              
WRITE_BUFFER_SIZE = 5000     
EMBEDDING_DIM = 128          # BioViL-T projected dimension

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

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

chex_rename = {col: f"{col.replace(' ', '_')}_chex" for col in label_cols}
neg_rename = {col: f"{col.replace(' ', '_')}_neg" for col in label_cols}
df_chexpert = df_chexpert.rename(columns=chex_rename)
df_negbio = df_negbio.rename(columns=neg_rename)

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
# 2. PYTORCH DATASET DESIGN
# ==========================================
class MimicCxrBioVilDataset(Dataset):
    def __init__(self, dataframe, base_dir):
        self.df = dataframe
        self.base_dir = base_dir
        # Torchvision transforms execute efficiently in C++
        self.transform = T.Compose([
            T.Resize(512),
            T.CenterCrop(480),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.base_dir, row["path"])
        
        try:
            # Must be RGB for ResNet50
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(img)
            return img_tensor, idx, True
            
        except Exception as e:
            logging.error(f"Failed at {idx} | Path: {img_path} | Error: {str(e)}")
            # Return a dummy tensor of the exact correct shape to satisfy the collate function
            return torch.zeros((3, 480, 480), dtype=torch.float32), idx, False

dataloader = DataLoader(
    MimicCxrBioVilDataset(df_master, DATA_DIR), 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS,
    pin_memory=True  # Back to True for PyTorch Tensors!
)

# ==========================================
# 3. INITIALIZE BIOVIL-T MODEL
# ==========================================
class BioViLTVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50()
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() 
        self.projector = nn.Linear(num_features, EMBEDDING_DIM)

    def forward(self, x):
        features = self.backbone(x)
        return self.projector(features) # Removed the OutputContainer class

print("Downloading/Loading BioViL-T weights...")
checkpoint_path = hf_hub_download(
    repo_id="microsoft/BiomedVLP-BioViL-T", 
    filename="biovil_t_image_model_proj_size_128.pt"
)

image_model = BioViLTVisionModel()
state_dict = torch.load(checkpoint_path, map_location="cpu")
clean_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
image_model.load_state_dict(clean_state_dict, strict=False)

image_model = image_model.to(DEVICE)
image_model.eval()

# ==========================================
# 4. LANCEDB SETTING
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

table = db.create_table("BioViL-T_embeddings_MIMIC-CXR-JPG", schema=schema, mode="overwrite")

# ==========================================
# 5. INFERENCE LOOP
# ==========================================
print("Starting inference pipeline...")
buffer_records = []

with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
    for imgs_batch, indices, success_flags in dataloader:
        
        # Convert flags to a boolean mask
        success_mask = success_flags.bool()
        
        # Track embeddings for this batch (initializing with Nones for corrupted files)
        raw_embeddings_batch = [None] * len(indices)
        l2_embeddings_batch = [None] * len(indices)
        
        # If there are any valid images in this batch, process them
        if success_mask.any():
            # Elegant PyTorch filtering: Only send valid images to the GPU
            valid_imgs = imgs_batch[success_mask].to(DEVICE, non_blocking=True)
            
            raw_outs = image_model(valid_imgs)
            l2_outs = raw_outs / raw_outs.norm(p=2, dim=-1, keepdim=True)
            
            # Move back to CPU as standard floats
            raw_outs_np = raw_outs.cpu().to(torch.float32).numpy()
            l2_outs_np = l2_outs.cpu().to(torch.float32).numpy()
            
            # Map valid embeddings back to their original positions in the batch
            valid_indices = success_mask.nonzero(as_tuple=True)[0].tolist()
            for exact_gpu_idx, original_batch_idx in enumerate(valid_indices):
                raw_embeddings_batch[original_batch_idx] = raw_outs_np[exact_gpu_idx].tolist()
                l2_embeddings_batch[original_batch_idx] = l2_outs_np[exact_gpu_idx].tolist()

        # Build LanceDB Records
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
            
        # Flush to LanceDB
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