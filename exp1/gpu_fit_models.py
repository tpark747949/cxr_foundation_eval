import os
import copy
import joblib
import lancedb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- Configuration & Constants ---
DB_URI = "../embeddings/MIMIC-CXR-JPG"
TABLE_NAME = "complete_embeddings_MIMIC-CXR-JPG"
#OUTPUT_DIR = "torch_lr_artifacts"
OUTPUT_DIR = "negbio_torch_lr_artifacts"


MODELS = ["MedSigLIP", "BioViL-T", "EVA-X", "CheXFound", "CheXagent", "CXR_Foundation"]
# Variants: Raw, L2-normalized, and PCA-reduced (fitted on Raw)
VARIANTS = ["raw", "l2", "pca_95"]

CHEXPERT_DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged_Cardiomediastinum", "Fracture", "Lung_Lesion",
    "Lung_Opacity", "Pleural_Effusion", "Pneumonia",
    "Pneumothorax", "Pleural_Other", "Support_Devices", "No_Finding"
]

# --- PyTorch Linear Model (No Dropout) ---
class MultiLabelLogReg(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0002):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = -np.inf
        self.best_weights = None

    def __call__(self, current_score, model):
        if current_score >= self.best_score + self.min_delta:
            self.best_score = current_score
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def calculate_auc(y_true, y_prob):
    aucs = [
        roc_auc_score(y_true[:, i], y_prob[:, i]) 
        if len(np.unique(y_true[:, i])) > 1 else np.nan 
        for i in range(len(CHEXPERT_DISEASES))
    ]
    return np.nanmean(aucs)

# --- Core GPU Trainer ---
def train_model(
    model_id, X_train, y_train, X_val, y_val, X_test, y_test, 
    batch_size=4096, epochs=100, lr=1e-3, weight_decay=1e-3
):
    print(f"\n--- Training: {model_id} | Input Dim: {X_train.shape[1]} ---")
    
    # Scale Data
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Convert to Tensors
    X_tr_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32)
    X_v_t = torch.tensor(X_val_s, dtype=torch.float32)
    X_te_t = torch.tensor(X_test_s, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True, num_workers=4)

    # Setup PyTorch Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiLabelLogReg(X_train.shape[1], y_train.shape[1])
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # Mild square-root class weighting to avoid extreme penalties
    pos_counts = y_train.sum(axis=0)
    neg_counts = len(y_train) - pos_counts
    pos_weight = np.sqrt(np.where(pos_counts > 0, neg_counts / pos_counts, 1.0))
    pos_weight_t = torch.tensor(pos_weight, dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=5, min_delta=0.0005)

    X_v_t = X_v_t.to(device)

    # Fast Training Loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(X_v_t)).cpu().numpy()
            
        current_val_auc = calculate_auc(y_val, val_probs)
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f} | Val AUC: {current_val_auc:.4f}")

        if early_stopping(current_val_auc, model):
            print(f"Early stopping at epoch {epoch+1}. Best Val AUC: {early_stopping.best_score:.4f}")
            break
            
    # Load Best Model
    model.load_state_dict(early_stopping.best_weights)
    
    # Final Inference
    model.eval()
    with torch.no_grad():
        X_te_t = X_te_t.to(device)
        best_val_probs = torch.sigmoid(model(X_v_t)).cpu().numpy()
        test_probs = torch.sigmoid(model(X_te_t)).cpu().numpy()
        
    np.save(os.path.join(OUTPUT_DIR, f"{model_id}_val_probs.npy"), best_val_probs)
    np.save(os.path.join(OUTPUT_DIR, f"{model_id}_test_probs.npy"), test_probs)

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"{model_id}_weights.pt"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, f"{model_id}_scaler.joblib"))

    best_val_auc = calculate_auc(y_val, best_val_probs)
    test_auc = calculate_auc(y_test, test_probs)
    
    print(f"--> Result [{model_id}] Val AUC: {best_val_auc:.4f} | Test AUC: {test_auc:.4f}")
    return best_val_auc, test_auc


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

    torch.multiprocessing.set_start_method('spawn', force=True)

    # Masks
    valid_data_mask = (df["ignore"] != 1)
    view_data_mask = (df["ViewCodeSequence_CodeMeaning"] == "postero-anterior")
    base_mask = valid_data_mask & view_data_mask

    train_mask = base_mask & (df["split"] == "train")
    val_mask = base_mask & (df["split"].isin(["val", "valid", "validate"]))
    test_mask = base_mask & (df["split"] == "test")

    # Labels
    #y_train = process_labels(df.loc[train_mask, "CheXpert_labels"])
    #y_val = process_labels(df.loc[val_mask, "CheXpert_labels"])
    #y_test = process_labels(df.loc[test_mask, "CheXpert_labels"])

    y_train = process_labels(df.loc[train_mask, "NegBio_labels"])
    y_val = process_labels(df.loc[val_mask, "NegBio_labels"])
    y_test = process_labels(df.loc[test_mask, "NegBio_labels"])

    np.save(os.path.join(OUTPUT_DIR, "y_val_true.npy"), y_val)
    np.save(os.path.join(OUTPUT_DIR, "y_test_true.npy"), y_test)

    results = []

    # 1. Individual Foundation Models
    for model_name in MODELS:
        for variant in ["raw", "l2"]:
            col_key = f"{model_name}_{variant}"
            X_tr = np.vstack(df.loc[train_mask, col_key].values)
            X_v = np.vstack(df.loc[val_mask, col_key].values)
            X_te = np.vstack(df.loc[test_mask, col_key].values)

            # Fit Raw / L2
            v_auc, t_auc = train_model(col_key, X_tr, y_train, X_v, y_val, X_te, y_test)
            results.append({"Model": model_name, "Variant": variant, "Val_AUC": v_auc, "Test_AUC": t_auc})

        # Generate PCA_95 variant from RAW embeddings
        raw_key = f"{model_name}_raw"
        X_tr_raw = np.vstack(df.loc[train_mask, raw_key].values)
        X_v_raw = np.vstack(df.loc[val_mask, raw_key].values)
        X_te_raw = np.vstack(df.loc[test_mask, raw_key].values)

        pca = PCA(n_components=0.95, svd_solver="auto")
        X_tr_pca = pca.fit_transform(X_tr_raw)
        X_v_pca = pca.transform(X_v_raw)
        X_te_pca = pca.transform(X_te_raw)

        joblib.dump(pca, os.path.join(OUTPUT_DIR, f"{model_name}_pca_object.joblib"))

        v_auc, t_auc = train_model(f"{model_name}_pca_95", X_tr_pca, y_train, X_v_pca, y_val, X_te_pca, y_test)
        results.append({"Model": model_name, "Variant": "pca_95", "Val_AUC": v_auc, "Test_AUC": t_auc})


    # 2. Early Fusion Ensemble
    for variant in ["raw", "l2"]:
        X_tr_f = np.hstack([np.vstack(df.loc[train_mask, f"{m}_{variant}"].values) for m in MODELS])
        X_v_f = np.hstack([np.vstack(df.loc[val_mask, f"{m}_{variant}"].values) for m in MODELS])
        X_te_f = np.hstack([np.vstack(df.loc[test_mask, f"{m}_{variant}"].values) for m in MODELS])

        v_auc, t_auc = train_model(f"Early_Fusion_{variant}", X_tr_f, y_train, X_v_f, y_val, X_te_f, y_test)
        results.append({"Model": "Early_Fusion", "Variant": variant, "Val_AUC": v_auc, "Test_AUC": t_auc})

    # Early Fusion PCA_95
    X_tr_f_raw = np.hstack([np.vstack(df.loc[train_mask, f"{m}_raw"].values) for m in MODELS])
    X_v_f_raw = np.hstack([np.vstack(df.loc[val_mask, f"{m}_raw"].values) for m in MODELS])
    X_te_f_raw = np.hstack([np.vstack(df.loc[test_mask, f"{m}_raw"].values) for m in MODELS])

    pca_f = PCA(n_components=0.95, svd_solver="auto")
    X_tr_f_pca = pca_f.fit_transform(X_tr_f_raw)
    X_v_f_pca = pca_f.transform(X_v_f_raw)
    X_te_f_pca = pca_f.transform(X_te_f_raw)

    joblib.dump(pca_f, os.path.join(OUTPUT_DIR, "Early_Fusion_pca_object.joblib"))

    v_auc, t_auc = train_model("Early_Fusion_pca_95", X_tr_f_pca, y_train, X_v_f_pca, y_val, X_te_f_pca, y_test)
    results.append({"Model": "Early_Fusion", "Variant": "pca_95", "Val_AUC": v_auc, "Test_AUC": t_auc})

    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, "roc_auc_summary.csv"), index=False)
    print("\nTraining completed! All evaluation artifacts generated.")

if __name__ == "__main__":
    main()