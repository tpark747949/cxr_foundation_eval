import lancedb
import numpy as np
import pandas as pd
from tqdm import tqdm

LANCEDB_URI = "../embeddings/MIMIC-CXR-JPG"
TABLE_NAME = "complete_embeddings_MIMIC-CXR-JPG" # Update if your table name differs
OUTPUT_CSV = "./embedding_zscores.csv"

MODELS = ["MedSigLIP", "CXR_Foundation", "BioViL-T", "EVA-X", "CheXagent", "CheXFound"]

print("Connecting to LanceDB...")
db = lancedb.connect(LANCEDB_URI)
table = db.open_table(TABLE_NAME)

results = {"dicom_id": []}
for m in MODELS:
    results[f"{m}_mag_zscore"] = []
    results[f"{m}_raw_mag"] = [] # Keeping raw magnitude just in case

print("Fetching data from LanceDB in chunks...")
# We use LanceDB's scanner to stream batches to Pandas
batches = table.search().select(["dicom_id"] + [f"{m}_raw" for m in MODELS]).to_batches()

# 1. First pass: compute all magnitudes
df_list = []
for batch in tqdm(batches, desc="Computing L2 Norms"):
    df_chunk = batch.to_pandas()
    
    chunk_data = {"dicom_id": df_chunk["dicom_id"]}
    
    for m in MODELS:
        col_name = f"{m}_raw"
        if col_name in df_chunk.columns:
            # Convert list of floats to 2D numpy array, then compute norm along axis 1
            arr = np.vstack(df_chunk[col_name].values)
            magnitudes = np.linalg.norm(arr, axis=1)
            chunk_data[f"{m}_raw_mag"] = magnitudes
            
    df_list.append(pd.DataFrame(chunk_data))

df_all = pd.concat(df_list, ignore_index=True)

# 2. Second pass: compute absolute Z-scores
print("Computing absolute Z-scores...")
for m in MODELS:
    mag_col = f"{m}_raw_mag"
    if mag_col in df_all.columns:
        mean_val = df_all[mag_col].mean()
        std_val = df_all[mag_col].std()
        
        # Absolute Z-Score calculation
        df_all[f"{m}_mag_zscore"] = np.abs((df_all[mag_col] - mean_val) / std_val)

# Drop the raw magnitudes to keep the CSV tiny, keep only dicom_id and z-scores
cols_to_keep = ["dicom_id"] + [f"{m}_mag_zscore" for m in MODELS if f"{m}_raw_mag" in df_all.columns]
df_final = df_all[cols_to_keep]

df_final.to_csv(OUTPUT_CSV, index=False)
print(f"Done! Z-scores saved to {OUTPUT_CSV}")