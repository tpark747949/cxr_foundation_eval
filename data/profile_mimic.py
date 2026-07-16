import os
import pandas as pd
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

DATA_DIR = "MIMIC-CXR-JPG/2.1.0"
METADATA_DIR = "MIMIC-CXR-JPG/2.1.0"
OUTPUT_CSV = "./mimic_qc.csv"
NUM_WORKERS = 16  # Adjust based on your CPU cores
SAMPLE_LIMIT = None  # Set to None to run on the entire 377k dataset

def analyze_image(row):
    path, dicom_id = row["path"], row["dicom_id"]
    img_path = os.path.join(DATA_DIR, path)
    
    try:
        # Load as grayscale and resize to 128x128 to make analysis lightning fast
        with Image.open(img_path) as img:
            img_gray = img.convert("L").resize((128, 128))
            arr = np.array(img_gray, dtype=np.float32)
    except Exception as e:
        return {"dicom_id": dicom_id, "corrupted": True}

    h, w = arr.shape
    
    # 1. Calculate Halves
    top_half = arr[:h//2, :]
    bottom_half = arr[h//2:, :]
    left_half = arr[:, :w//2]
    right_half = arr[:, w//2:]
    
    # 2. Calculate Corners (10x10 pixel blocks in corners)
    c_size = 10
    corners = [
        arr[:c_size, :c_size],       # Top-Left
        arr[:c_size, -c_size:],      # Top-Right
        arr[-c_size:, :c_size],      # Bottom-Left
        arr[-c_size:, -c_size:]      # Bottom-Right
    ]
    corner_mean = np.mean([np.mean(c) for c in corners])

    return {
        "dicom_id": dicom_id,
        "path": path,
        "corrupted": False,
        "overall_mean": float(np.mean(arr)),
        "overall_std": float(np.std(arr)),
        "mean_top": float(np.mean(top_half)),
        "mean_bottom": float(np.mean(bottom_half)),
        "mean_left": float(np.mean(left_half)),
        "mean_right": float(np.mean(right_half)),
        "min_half_mean": float(min(np.mean(top_half), np.mean(bottom_half), np.mean(left_half), np.mean(right_half))),
        "corner_mean": float(corner_mean)
    }

if __name__ == "__main__":
    print("Loading image paths...")
    with open(os.path.join(METADATA_DIR, "IMAGE_FILENAMES"), "r") as f:
        paths = [line.strip() for line in f if line.strip()]
    
    df = pd.DataFrame({"path": paths})
    df["dicom_id"] = df["path"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    
    if SAMPLE_LIMIT:
        print(f"Limiting profiling to a sample of {SAMPLE_LIMIT} images...")
        df = df.sample(n=SAMPLE_LIMIT, random_state=42).reset_index(drop=True)

    records = df.to_dict(orient="records")
    
    print(f"Profiling images using {NUM_WORKERS} threads...")
    results = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for res in tqdm(executor.map(analyze_image, records), total=len(records)):
            results.append(res)
            
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"Metrics saved to {OUTPUT_CSV}!")