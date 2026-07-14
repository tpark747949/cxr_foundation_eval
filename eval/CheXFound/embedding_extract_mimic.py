import os
import pandas as pd
import numpy as np
import pyarrow as pa
import lancedb
import multiprocessing as mp
from PIL import Image
from argparse import Namespace

# ==========================================
# 0. GLOBAL CONFIGURATION & PICKLABLE DATASET
# ==========================================
# CRITICAL: Do NOT import torch, torchvision, or chexfound here!

DATA_DIR = "../../data/MIMIC-CXR-JPG/2.1.0"  # Change to your symlink target
METADATA_DIR = "../../data/MIMIC-CXR-JPG/2.1.0"
URI = "../../embeddings/MIMIC-CXR-JPG"
BASE_DIR = './checkpoints/'

AVAILABLE_GPUS = [0, 1, 2, 3]  
BATCH_SIZE = 256       
NUM_WORKERS_PER_GPU = 8        
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

class MimicCxrChexFoundDataset(object):
    """
    Defined at the module level so it is 100% picklable by background DataLoader workers.
    Inherits from object to avoid importing torch in the global scope.
    """
    def __init__(self, dataframe, base_dir, transform):
        self.df = dataframe
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.base_dir, row["path"])
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor, idx

# ==========================================
# 1. THE ISOLATED GPU WORKER
# ==========================================
def run_gpu_worker(gpu_id, df_chunk, clean_label_cols, args, data_dir, uri):
    """
    This worker runs in total isolation. By setting the environment variable 
    BEFORE importing PyTorch, we guarantee zero CUDA context collisions.
    """
    
    # 1. BLIND THE PROCESS BEFORE PYTORCH WAKES UP
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 2. LATE IMPORTS - PyTorch initializes here, seeing only 1 GPU.
    import torch
    from torch.utils.data import DataLoader
    from chexfound.eval.setup import setup_and_build_model
    from chexfound.data.transforms import make_classification_eval_transform

    # Because of CUDA_VISIBLE_DEVICES, "cuda:0" safely maps to the assigned GPU.
    device = torch.device("cuda:0")
    print(f"[GPU {gpu_id}] Initializing. PyTorch sees {torch.cuda.device_count()} GPU(s). Processing {len(df_chunk)} images...")

    # 3. INITIALIZE MODEL & TRANSFORMS
    eval_transform = make_classification_eval_transform(
        resize_size=args.image_size, crop_size=args.image_size
    )

    model, autocast_dtype = setup_and_build_model(args)

    state_dict = torch.load(args.pretrained_weights, map_location="cpu")['teacher']
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

    # 4. DYNAMIC DIMENSION PROBE
    with torch.no_grad():
        dummy_input = torch.zeros((1, 3, args.image_size, args.image_size), device=device)
        dummy_features = model.get_intermediate_layers(
            dummy_input, n=args.n_last_blocks, return_class_token=args.return_class_token
        )
        dummy_output = dummy_features[-1][1]
        EMBEDDING_DIM = dummy_output.shape[-1]
    
    # 5. LANCEDB SCHEMA
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

    db = lancedb.connect(uri)
    table_name = f"CheXfound_MIMIC_Part_{gpu_id}"
    table = db.create_table(table_name, schema=schema, mode="overwrite")

    # 6. DATALOADER
    dataloader = DataLoader(
        MimicCxrChexFoundDataset(df_chunk, data_dir, eval_transform), 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS_PER_GPU,
        pin_memory=True 
    )

    # 7. INFERENCE LOOP
    buffer_records = []
    
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=autocast_dtype):
        for imgs_batch, indices in dataloader:
            valid_imgs = imgs_batch.to(device, non_blocking=True)
            
            features = model.get_intermediate_layers(
                valid_imgs, n=args.n_last_blocks, return_class_token=args.return_class_token
            )
            
            raw_outs = features[-1][1]
            l2_outs = raw_outs / raw_outs.norm(p=2, dim=-1, keepdim=True)
            
            raw_outs_np = raw_outs.cpu().to(torch.float32).numpy()
            l2_outs_np = l2_outs.cpu().to(torch.float32).numpy()
            indices_np = indices.numpy()

            for idx_in_batch, global_idx in enumerate(indices_np):
                row = df_chunk.iloc[global_idx]
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
                print(f"[GPU {gpu_id}] Indexed up to row {global_idx + 1}/{len(df_chunk)}")

    if buffer_records:
        table.add(buffer_records)
        
    print(f"[GPU {gpu_id}] Finished! Total rows in {table_name}: {len(table)}")

# ==========================================
# 2. MAIN PROCESS COORDINATOR
# ==========================================
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
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

    chunk_indices = np.array_split(np.arange(len(df_master)), len(AVAILABLE_GPUS))
    chunks = [df_master.iloc[indices].reset_index(drop=True) for indices in chunk_indices]

    print(f"Launching {len(AVAILABLE_GPUS)} GPU workers...")
    
    processes = []
    for i, gpu_id in enumerate(AVAILABLE_GPUS):
        p = mp.Process(target=run_gpu_worker, args=(gpu_id, chunks[i], clean_label_cols, CHEXFOUND_ARGS, DATA_DIR, URI))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All GPUs have finished processing!")