import os
import shutil
from pathlib import Path

# --- Schema Definitions ---
MODELS = ["MedSigLIP", "BioViL-T", "EVA-X", "CheXFound", "CheXagent", "CXR_Foundation", "Early_Fusion"]
CHEXPERT_DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged_Cardiomediastinum", "Fracture", "Lung_Lesion",
    "Lung_Opacity", "Pleural_Effusion", "Pneumonia",
    "Pneumothorax", "Pleural_Other", "Support_Devices", "No_Finding"
]

# Mapping directory names to our standardized Head names
HEAD_MAP = {
    "independent": "i2",
    "shared_mlp": "s2",
    "torch_lr": "LR",
    "xgboost": "XGB"
}

DEST_DIR = Path("artifacts")

def parse_artifact_name(path, filename):
    path_str = str(path).lower()
    
    # 1. Guess Model
    model = next((m for m in MODELS if m.lower() in filename.lower()), None)
    if "early_fusion" in filename.lower(): 
        model = "Early_Fusion"
        
    if not model:
        return None
        
    # 2. Guess Head (based on parent directories)
    head = "unknown"
    for dir_key, head_code in HEAD_MAP.items():
        if dir_key in path_str:
            head = head_code
            break

    if head == "unknown":
        print("Warning: Could not determine head for file:", path)
            
    # 3. Guess Variance (Preprocessing)
    var = "raw"
    if "l2" in filename.lower(): 
        var = "l2"
    elif "pca_95" in filename.lower() or "pca95" in filename.lower(): 
        var = "pca95"
        
    # 4. Guess Target (Disease name, or scaler/weights type)
    target = "unknown"
    disease = next((d for d in CHEXPERT_DISEASES if d.lower() in filename.lower()), None)

    
    if disease:
        target = disease
    elif "scaler" in filename.lower():
        target = "scaler"
    elif "pca" in filename.lower() and path.suffix in [".joblib", ".pkl"]:
        target = "pca_transformer"
    elif "weights" in filename.lower() or path.suffix == ".pt":
        target = "weights" # Shared heads usually output a single weights file for all diseases

    
    if target == "unknown":
        print("Warning: Could not determine target for file:", path)
        
    return f"{model}_{head}_{var}_{target}{path.suffix}"

def main():
    DEST_DIR.mkdir(exist_ok=True)
    
    # We only want the 100% data models trained on CheXpert for clinical inference
    source_dir = Path("../../exp1/CheXpert_labels")
    if not source_dir.exists():
        print(f"Error: Could not find {source_dir.resolve()}")
        return

    print(f"Scanning {source_dir} for model artifacts...\n")
    
    all_files = []
    for ext in ["*.pt", "*.joblib", "*.json", "*.pkl"]:
        all_files.extend(list(source_dir.rglob(ext)))
        
    processed = 0
    
    for path in all_files:
        # Skip hidden files/directories
        if "4class" in str(path).lower() or "3class" in str(path).lower():
            continue
            
        new_name = parse_artifact_name(path, path.name)
        if not new_name:
            continue
            
        dest_path = DEST_DIR / new_name
        shutil.copy2(path, dest_path)
        processed += 1
        
    print(f"✅ Successfully collected and standardized {processed} artifacts into {DEST_DIR.resolve()}")

if __name__ == "__main__":
    main()