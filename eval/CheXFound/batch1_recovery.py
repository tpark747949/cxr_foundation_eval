import os
import pandas as pd
import numpy as np
import pyarrow as pa
import lancedb
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from argparse import Namespace

# Import your custom modules
from chexfound.eval.setup import setup_and_build_model
from chexfound.data.transforms import make_classification_eval_transform

# ==========================================
# 0. GLOBAL CONFIGURATION
# ==========================================
DATA_DIR = "../../data/MIMIC-CXR-JPG/2.1.0"
METADATA_DIR = "../../data/MIMIC-CXR-JPG/2.1.0"
URI = "../../embeddings/MIMIC-CXR-JPG"
BASE_DIR = './checkpoints/'

TARGET_GPU = 0
BATCH_SIZE = 128               
NUM_WORKERS = 8        
WRITE_BUFFER_SIZE = 2500     

CHEXFOUND_ARGS = Namespace(
    config_file=os.path.join(BASE_DIR, 'config.yaml'),
    pretrained_weights=os.path.join(BASE_DIR, 'teacher_checkpoint.pth'),
    output_dir=os.path.join(BASE_DIR, 'example'),
    opts=[],
    image_size=512,
    patch_size=16,
    n_register_tokens=4,
    n_last_blocks=4,
    return_class_token=True,
    num_classes=40,
    num_heads=8,
)

# ==========================================
# 1. DATASET DEFINITION
# ==========================================
class MimicCxrChexFoundDataset(Dataset):
    def __init__(self, dataframe, base_dir, transform):
        self.df = dataframe
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.base_dir, row["path"])
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(img)
        except Exception as e:
            # Catch completely corrupted image files
            print(f"\n[WARNING] Failed to load image {img_path}: {e}")
            img_tensor = torch.zeros((3, CHEXFOUND_ARGS.image_size, CHEXFOUND_ARGS.image_size))
            
        return img_tensor, idx

# ==========================================
# 2. MAIN PIPELINE
# ==========================================
if __name__ == '__main__':
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

    # Setup Device
    torch.cuda.set_device(TARGET_GPU)
    device = torch.device(f"cuda:{TARGET_GPU}")
    print(f"PyTorch initialized on {device}. Using bfloat16 to prevent overflows.")

    # Model Setup
    eval_transform = make_classification_eval_transform(
        resize_size=CHEXFOUND_ARGS.image_size, crop_size=CHEXFOUND_ARGS.image_size
    )

    model, _ = setup_and_build_model(CHEXFOUND_ARGS)

    state_dict = torch.load(CHEXFOUND_ARGS.pretrained_weights, map_location="cpu")['teacher']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('backbone'):
            ls = k.split('.')
            if 'blocks' in k:
                new_k = '.'.join([ls[1], *ls[3:]])
            else:
                new_k = '.'.join(ls[1:])
        else:
            new_k = k
        new_state_dict.update({new_k: v})

    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # Dimension Probe
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        dummy_input = torch.zeros((1, 3, CHEXFOUND_ARGS.image_size, CHEXFOUND_ARGS.image_size), device=device)
        dummy_features = model.get_intermediate_layers(
            dummy_input, n=CHEXFOUND_ARGS.n_last_blocks, return_class_token=CHEXFOUND_ARGS.return_class_token
        )
        dummy_output = dummy_features[-1][1]
        EMBEDDING_DIM = dummy_output.shape[-1]

    # LanceDB Schema Setup
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

    db = lancedb.connect(URI)
    table_name = "CheXfound_MIMIC"
    table = db.create_table(table_name, schema=schema, mode="overwrite")

    # Dataloader
    dataset = MimicCxrChexFoundDataset(df_master, DATA_DIR, eval_transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True 
    )

    # Inference Loop
    buffer_records = []
    
    # EXPLICITLY FORCING BFLOAT16 HERE
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for imgs_batch, indices in dataloader:
            valid_imgs = imgs_batch.to(device, non_blocking=True)
            
            features = model.get_intermediate_layers(
                valid_imgs, n=CHEXFOUND_ARGS.n_last_blocks, return_class_token=CHEXFOUND_ARGS.return_class_token
            )
            
            raw_outs = features[-1][1]
            
            # --- SAFETY NET ---
            if torch.isnan(raw_outs).any() or torch.isinf(raw_outs).any():
                raw_outs = torch.nan_to_num(raw_outs, nan=0.0, posinf=0.0, neginf=0.0)
                bad_indices = indices[torch.isnan(features[-1][1]).any(dim=-1).cpu()]
                for bad_idx in bad_indices:
                    bad_dicom = df_master.iloc[bad_idx.item()]["dicom_id"]
                    print(f"\n[WARNING] Math overflow detected on DICOM ID: {bad_dicom}. Saved as zero-vector.")
            # ------------------

            l2_outs = raw_outs / (raw_outs.norm(p=2, dim=-1, keepdim=True) + 1e-8) 
            
            raw_outs_np = raw_outs.cpu().to(torch.float32).numpy()
            l2_outs_np = l2_outs.cpu().to(torch.float32).numpy()
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
                    "embedding_raw": raw_outs_np[idx_in_batch].tolist(),
                    "embedding_l2": l2_outs_np[idx_in_batch].tolist(),
                }
                buffer_records.append(record)
                
            if len(buffer_records) >= WRITE_BUFFER_SIZE:
                table.add(buffer_records)
                buffer_records = []
                print(f"Indexed up to row {global_idx + 1}/{len(df_master)}")

    if buffer_records:
        table.add(buffer_records)
        
    print(f"Finished! Total rows in {table_name}: {len(table)}")