import os
import joblib
import lancedb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

# --- Configuration & Constants ---
DB_URI = "../embeddings/MIMIC-CXR-JPG"
TABLE_NAME = "complete_embeddings_MIMIC-CXR-JPG"
# OUTPUT_DIR = "xgboost_evaluation_artifacts"
OUTPUT_DIR = "negbio_xgboost_evaluation_artifacts"
NUM_GPUS = 4  # Matches your 4x A6000 setup

# Early Fusion is excluded per your experimental design
MODELS = ["MedSigLIP", "BioViL-T", "EVA-X", "CheXFound", "CheXagent", "CXR_Foundation"]
VARIANTS = ["raw", "l2", "pca_95"]

CHEXPERT_DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged_Cardiomediastinum", "Fracture", "Lung_Lesion",
    "Lung_Opacity", "Pleural_Effusion", "Pneumonia",
    "Pneumothorax", "Pleural_Other", "Support_Devices", "No_Finding"
]

def train_single_disease(disease_idx, model_name, variant, X_tr, y_tr_1d, X_v, y_v_1d, X_te, output_dir):
    """
    Trains a single XGBoost binary classifier on a strictly isolated GPU process.
    """
    # 1. Sandbox the worker to a single GPU BEFORE touching CUDA
    gpu_id = disease_idx % NUM_GPUS
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 2. Import GPU libraries only AFTER the environment is sandboxed
    import torch
    import xgboost as xgb
    
    disease_name = CHEXPERT_DISEASES[disease_idx]
    # The device is always 'cuda:0' because the worker only sees its assigned GPU
    device = "cuda:0" 
    
    # 3. Move data to GPU explicitly to avoid the Device Mismatch warning
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr_1d, dtype=torch.float32, device=device)
    X_v_t = torch.tensor(X_v, dtype=torch.float32, device=device)
    y_v_t = torch.tensor(y_v_1d, dtype=torch.float32, device=device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)
    
    pos_count = np.sum(y_tr_1d)
    neg_count = len(y_tr_1d) - pos_count
    spw = min(neg_count / max(pos_count, 1), 50.0) 

    clf = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.05,
        colsample_bytree=0.2,
        subsample=0.8,
        scale_pos_weight=spw,
        tree_method='hist',
        device=device,
        random_state=42,
        eval_metric='auc',
        early_stopping_rounds=20
    )

    clf.fit(X_tr_t, y_tr_t, eval_set=[(X_v_t, y_v_t)], verbose=False)
    
    # 4. Save the model directly from the worker (prevents IPC pickling crashes)
    model_path = os.path.join(output_dir, f"{model_name}_{variant}_{disease_name}.json")
    clf.save_model(model_path)
    
    # 5. Extract probabilities and safely cast back to CPU numpy arrays
    val_probs = clf.predict_proba(X_v_t)[:, 1]
    test_probs = clf.predict_proba(X_te_t)[:, 1]
    
    if hasattr(val_probs, 'cpu'): val_probs = val_probs.cpu().numpy()
    if hasattr(test_probs, 'cpu'): test_probs = test_probs.cpu().numpy()
    
    return disease_idx, val_probs, test_probs


def calculate_auc(y_true, y_prob):
    aucs = [
        roc_auc_score(y_true[:, i], y_prob[:, i]) 
        if len(np.unique(y_true[:, i])) > 1 else np.nan 
        for i in range(len(CHEXPERT_DISEASES))
    ]
    return np.nanmean(aucs)


def process_labels(df_labels, strategy="zeros"):
    labels_matrix = np.zeros((len(df_labels), len(CHEXPERT_DISEASES)))
    for idx, row in enumerate(df_labels):
        for d_idx, disease in enumerate(CHEXPERT_DISEASES):
            val = row[disease]
            if val == 1.0:
                labels_matrix[idx, d_idx] = 1.0
            elif val == -1.0:
                labels_matrix[idx, d_idx] = 1.0 if strategy == "ones" else 0.0
            else:
                labels_matrix[idx, d_idx] = 0.0
    return labels_matrix


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    db = lancedb.connect(DB_URI)
    tbl = db.open_table(TABLE_NAME)
    df = tbl.to_pandas()

    # Masks
    valid_data_mask = (df["ignore"] != 1)
    view_data_mask = (df["ViewCodeSequence_CodeMeaning"] == "postero-anterior")
    base_mask = valid_data_mask & view_data_mask

    train_mask = base_mask & (df["split"] == "train")
    val_mask = base_mask & (df["split"].isin(["val", "valid", "validate"]))
    test_mask = base_mask & (df["split"] == "test")

    # labels
    # y_train = process_labels(df.loc[train_mask, "CheXpert_labels"].values)
    # y_val = process_labels(df.loc[val_mask, "CheXpert_labels"].values)
    # y_test = process_labels(df.loc[test_mask, "CheXpert_labels"].values)

    y_train = process_labels(df.loc[train_mask, "NegBio_labels"].values)
    y_val = process_labels(df.loc[val_mask, "NegBio_labels"].values)
    y_test = process_labels(df.loc[test_mask, "NegBio_labels"].values)

    np.save(os.path.join(OUTPUT_DIR, "y_val_true.npy"), y_val)
    np.save(os.path.join(OUTPUT_DIR, "y_test_true.npy"), y_test)

    results = []

    for model_name in MODELS:
        for variant in VARIANTS:
            col_key = f"{model_name}_raw" if variant == "pca_95" else f"{model_name}_{variant}"
            
            X_tr = np.vstack(df.loc[train_mask, col_key].values)
            X_v = np.vstack(df.loc[val_mask, col_key].values)
            X_te = np.vstack(df.loc[test_mask, col_key].values)

            print(f"\n--- Training XGBoost: {model_name} | Variant: {variant} ---")

            if variant == "pca_95":
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_v = scaler.transform(X_v)
                X_te = scaler.transform(X_te)

                pca = PCA(n_components=0.95, svd_solver="auto")
                # pca = PCA(n_components=0.95, svd_solver="randomized", random_state=42)
                X_tr = pca.fit_transform(X_tr)
                X_v = pca.transform(X_v)
                X_te = pca.transform(X_te)
                
                # Save artifacts for Streamlit inference
                joblib.dump(scaler, os.path.join(OUTPUT_DIR, f"{model_name}_pca_95_scaler.joblib"))
                joblib.dump(pca, os.path.join(OUTPUT_DIR, f"{model_name}_pca_95_object.joblib"))

            # Parallelize the 14 diseases across the 4 GPUs
            # loky automatically memmaps the massive numpy arrays, saving System RAM
            parallel_results = Parallel(n_jobs=NUM_GPUS, backend="loky")(
                delayed(train_single_disease)(
                    d_idx, model_name, variant, X_tr, y_train[:, d_idx], 
                    X_v, y_val[:, d_idx], X_te, OUTPUT_DIR
                ) for d_idx in range(len(CHEXPERT_DISEASES))
            )
            
            # Reconstruct the multi-label probability matrices from the parallel workers
            val_probs = np.zeros((len(X_v), len(CHEXPERT_DISEASES)))
            test_probs = np.zeros((len(X_te), len(CHEXPERT_DISEASES)))
            
            # We no longer receive 'clf' back to avoid the CUDA crash
            for d_idx, v_prob, t_prob in parallel_results:
                val_probs[:, d_idx] = v_prob
                test_probs[:, d_idx] = t_prob
                # Save individual disease booster
                # clf.save_model(os.path.join(OUTPUT_DIR, f"{model_name}_{variant}_{CHEXPERT_DISEASES[d_idx]}.json"))

            # Save the final matrix for evaluation
            np.save(os.path.join(OUTPUT_DIR, f"{model_name}_{variant}_val_probs.npy"), val_probs)
            np.save(os.path.join(OUTPUT_DIR, f"{model_name}_{variant}_test_probs.npy"), test_probs)

            v_auc = calculate_auc(y_val, val_probs)
            t_auc = calculate_auc(y_test, test_probs)
            print(f"--> Result [{model_name} - {variant}] Val AUC: {v_auc:.4f} | Test AUC: {t_auc:.4f}")
            
            results.append({
                "Model": model_name, 
                "Variant": variant, 
                "Val_AUC": v_auc, 
                "Test_AUC": t_auc
            })

    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, "xgboost_roc_auc_summary.csv"), index=False)
    print("\nXGBoost Training completed! All models and predictions saved.")

if __name__ == "__main__":
    main()