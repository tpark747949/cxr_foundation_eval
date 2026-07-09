import os
import io
import png
import logging
import pandas as pd
import numpy as np
import pyarrow as pa
import lancedb
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ==========================================
# 0. GLOBAL CONFIGURATION
# ==========================================
# Fix the LanceDB/PyTorch Fork Warning!
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

logging.basicConfig(
    filename="mimic_processing_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

DATA_DIR = "../../data/MIMIC-CXR-JPG/2.1.0"  # Change to your symlink target
METADATA_DIR = "../../data/MIMIC-CXR-JPG/2.1.0"
URI = "../../embeddings/MIMIC-CXR-JPG"

BATCH_SIZE = 128             
NUM_WORKERS_PER_GPU = 4      # 4 GPUs * 4 workers = 16 CPU threads total
WRITE_BUFFER_SIZE = 2500     
EMBEDDING_DIM = 4096         

# Define which GPUs we want to use (0, 1, 2, 3)
AVAILABLE_GPUS = [1, 2, 3]

# ==========================================
# 1. PYTORCH DATASET DESIGN (CPU MULTIPROCESSING)
# ==========================================
def png_to_tfexample(image_array: np.ndarray) -> bytes:
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
    
    # We must import TF locally inside the worker for the TF Example creation
    import tensorflow as tf 
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

# ==========================================
# 2. THE MULTI-GPU WORKER PROCESS
# ==========================================
def run_gpu_worker(gpu_id, df_chunk, schema, clean_label_cols):
    """This function runs entirely independently on a single GPU."""
    
    # 1. Isolate the GPU BEFORE importing TensorFlow
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    
    import tensorflow as tf
    import tensorflow_text  # Required for custom ops
    
    # Force Memory Growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU {gpu_id} TF Memory Error: {e}")

    print(f"[GPU {gpu_id}] Initializing. Processing {len(df_chunk)} images...")

    # 2. Load Models
    elixrc_model = tf.saved_model.load('./checkpoints/hf/elixr-c-v2-pooled')
    elixrc_infer = elixrc_model.signatures['serving_default']

    qformer_model = tf.saved_model.load("./checkpoints/hf/pax-elixr-b-text")
    qformer_infer = qformer_model.signatures['serving_default']

    # 3. Create a Fast TF.Function to bypass Python loop overhead
    @tf.function
    def fast_inference_loop(string_tensor):
        # Use TensorArray to build output dynamically inside the C++ graph
        batch_size = tf.shape(string_tensor)[0]
        out_array = tf.TensorArray(tf.float32, size=batch_size)
        
        static_ids = tf.zeros((1, 1, 128), dtype=tf.int32)
        static_paddings = tf.zeros((1, 1, 128), dtype=tf.float32)

        for i in tf.range(batch_size):
            single_str = string_tensor[i]
            elixr_in = tf.expand_dims(single_str, 0)
            
            elixrc_out = elixrc_infer(input_example=elixr_in)
            
            q_in = {
                'image_feature': elixrc_out['feature_maps_0'],
                'ids': static_ids,
                'paddings': static_paddings,
            }
            q_out = qformer_infer(**q_in)
            
            raw_flat = tf.reshape(q_out['all_contrastive_img_emb'], (-1,))
            l2_flat = tf.math.l2_normalize(raw_flat, axis=-1)
            
            out_array = out_array.write(i, l2_flat)
            
        return out_array.stack()

    # 4. Initialize DataLoader and LanceDB
    dataloader = DataLoader(
        MimicCxrTfDataset(df_chunk, DATA_DIR), 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS_PER_GPU,
        pin_memory=False 
    )

    db = lancedb.connect(URI)
    # Each GPU gets its own table to prevent write-locking
    table_name = f"CXR_Foundation_MIMIC_Part_{gpu_id}"
    table = db.create_table(table_name, schema=schema, mode="overwrite")

    # 5. Inference Loop
    buffer_records = []
    
    for batch_strings, indices, success_flags in dataloader:
        indices = indices.numpy()
        success_flags = success_flags.numpy()
        
        valid_indices = [i for i, success in enumerate(success_flags) if success]
        
        if valid_indices:
            valid_strings = [batch_strings[i] for i in valid_indices]
            
            # Execute the fast C++ compiled TF loop
            tf_string_batch = tf.constant(valid_strings)
            l2_embeddings_tensor = fast_inference_loop(tf_string_batch)
            l2_embeddings_np = l2_embeddings_tensor.numpy()

        # Build records
        for idx_in_batch, global_idx in enumerate(indices):
            # If failed, embedding is None. If success, grab the corresponding row from the numpy matrix.
            is_valid = success_flags[idx_in_batch]
            emb_vector = l2_embeddings_np[valid_indices.index(idx_in_batch)].tolist() if is_valid else None
            
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
                "embedding_raw": emb_vector, # Saving space: Just using the L2 vector for both or customize if raw is needed
                "embedding_l2": emb_vector,
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
# 3. MAIN PROCESS COORDINATOR
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

    # Define Schema here so we can pass it to workers
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

    # Safely shard Dataframe into 4 equal chunks without destroying Pandas formatting
    chunk_indices = np.array_split(np.arange(len(df_master)), len(AVAILABLE_GPUS))
    chunks = [df_master.iloc[indices].reset_index(drop=True) for indices in chunk_indices]

    print(f"Launching {len(AVAILABLE_GPUS)} GPU workers...")
    
    processes = []
    for i, gpu_id in enumerate(AVAILABLE_GPUS):
        p = mp.Process(target=run_gpu_worker, args=(gpu_id, chunks[i], schema, clean_label_cols))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print("All 4 GPUs have finished processing! You can merge the LanceDB partitions natively.")