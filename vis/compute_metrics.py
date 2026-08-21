import lancedb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

DB_URI = "../embeddings/MIMIC-CXR-JPG"
TABLE_NAME = "complete_embeddings_MIMIC-CXR-JPG"

CHEXPERT_DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged_Cardiomediastinum", "Fracture", "Lung_Lesion",
    "Lung_Opacity", "Pleural_Effusion", "Pneumonia",
    "Pneumothorax", "Pleural_Other", "Support_Devices", "No_Finding"
]

MODELS = ["MedSigLIP", "BioViL-T", "EVA-X", "CheXFound", "CheXagent", "CXR_Foundation", "Early_Fusion"]
HEADS = ["LR", "XGB", "s2", "s4", "i2", "i4"]
LABELS = ["CheXpert", "NegBio", "1pct", "5pct", "10pct"]
VARS = ["raw", "l2", "pca95"]

def safe_binary_auc(y_t, y_p):
    return roc_auc_score(y_t, y_p) if len(np.unique(y_t)) > 1 else np.nan

def safe_binary_auprc(y_t, y_p):
    return average_precision_score(y_t, y_p) if len(np.unique(y_t)) > 1 else np.nan

def safe_ovr_auc(y_t, y_p):
    aucs = [roc_auc_score((y_t == c).astype(int), y_p[:, c]) for c in range(4) if len(np.unique((y_t == c).astype(int))) == 2]
    return np.mean(aucs) if aucs else np.nan

def safe_ovr_auprc(y_t, y_p):
    auprcs = [average_precision_score((y_t == c).astype(int), y_p[:, c]) for c in range(4) if len(np.unique((y_t == c).astype(int))) == 2]
    return np.mean(auprcs) if auprcs else np.nan

def load_ground_truths():
    df = lancedb.connect(DB_URI).open_table(TABLE_NAME).to_pandas()
    test_df = df[(df["ignore"] != 1) & (df["ViewCodeSequence_CodeMeaning"] == "postero-anterior") & (df["split"] == "test")]
    
    def to_binary(col):
        return np.array([[1.0 if r[d] == 1.0 else 0.0 for d in CHEXPERT_DISEASES] for r in col])
        
    def to_4class(col):
        matrix = np.zeros((len(col), len(CHEXPERT_DISEASES)), dtype=np.int64)
        for i, r in enumerate(col):
            for j, d in enumerate(CHEXPERT_DISEASES):
                v = r[d]
                matrix[i, j] = 1 if v == 1.0 else (2 if v == -1.0 else (3 if v == -2.0 or np.isnan(v) else 0))
        return matrix

    return {
        "CheXpert_binary": to_binary(test_df["CheXpert_labels"]), 
        "CheXpert_4class": to_4class(test_df["CheXpert_labels"]),
        "NegBio_binary": to_binary(test_df["NegBio_labels"]), 
        "NegBio_4class": to_4class(test_df["NegBio_labels"])
    }

def main():
    gt = load_ground_truths()
    records = []
    
    for f in Path("test_probs").glob("*.npy"):
        stem = f.stem
        
        # 1. Safely extract the exact model name without using split()
        model = next((m for m in MODELS if stem.startswith(m)), None)
        if not model:
            print(f"Skipping {stem} - Unrecognized model prefix.")
            continue
            
        # 2. Strip the model name and the connecting underscore to get the rest
        rest = stem[len(model)+1:]
        parts = rest.split('_')
        
        if len(parts) != 3:
            print(f"Skipping {stem} - Could not parse remaining parts: {parts}")
            continue
            
        head, label, var = parts
        
        # Ensure parsed elements exist in your provided lists
        if head not in HEADS or label not in LABELS or var not in VARS:
            print(f"Skipping {stem} - Invalid parameter match.")
            continue
            
        # 3. Determine ground truth mapping
        is_4class = head in ["s4", "i4"]
        gt_suffix = "4class" if is_4class else "binary"
        gt_prefix = "NegBio" if label == "NegBio" else "CheXpert"
        
        y_true = gt[f"{gt_prefix}_{gt_suffix}"]
        probs = np.load(f)
        
        # 4. Compute AUC and AUPRC for each disease
        for d_idx, disease in enumerate(CHEXPERT_DISEASES):
            y_t = y_true[:, d_idx]
            
            if is_4class:
                # Handle varying 3D shapes: (996, 4, 14) vs (996, 14, 4)
                y_p = probs[:, :, d_idx] if probs.shape[1] == 4 else probs[:, d_idx, :]
                auc = safe_ovr_auc(y_t, y_p)
                auprc = safe_ovr_auprc(y_t, y_p)
            else:
                auc = safe_binary_auc(y_t, probs[:, d_idx])
                auprc = safe_binary_auprc(y_t, probs[:, d_idx])
                
            records.append({
                "Model": model, 
                "Head": head, 
                "Label": label, 
                "Var": var, 
                "Disease": disease, 
                "AUC": auc,
                "AUPRC": auprc
            })
            
    pd.DataFrame(records).to_csv("master_metrics.csv", index=False)
    print(f"Saved master_metrics.csv successfully! Computed {len(records)} rows.")

if __name__ == "__main__":
    main()