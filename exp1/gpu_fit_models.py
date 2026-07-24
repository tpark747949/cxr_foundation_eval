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

# --- Configuration & Constants ---
DB_URI = "../embeddings/MIMIC-CXR-JPG"
TABLE_NAME = "complete_embeddings_MIMIC-CXR-JPG"
OUTPUT_DIR = "gpu_evaluation_artifacts"

MODELS = ["MedSigLIP", "BioViL-T", "EVA-X", "CheXFound", "CheXagent", "CXR_Foundation"]
VARIANTS = ["raw", "l2"]

CHEXPERT_DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged_Cardiomediastinum", "Fracture", "Lung_Lesion",
    "Lung_Opacity", "Pleural_Effusion", "Pneumonia",
    "Pneumothorax", "Pleural_Other", "Support_Devices", "No_Finding"
]

# --- PyTorch Model & Early Stopping ---
class MultiLabelLogReg(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # A single linear layer without activation is mathematically equivalent to LR logits
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)

class EarlyStopping:
    """Stops training if validation AUC doesn't improve after a given patience."""
    def __init__(self, patience=5, min_delta=0.001):
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

# --- Core Training Function ---
def train_pytorch_model(
    model_id, X_train, y_train, X_val, y_val, X_test, y_test, 
    batch_size=2048, epochs=100, lr=1e-3, weight_decay=1e-2
):
    print(f"\n--- Training: {model_id} (Features: {X_train.shape[1]}) ---")
    
    # Scale Data (CPU side)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(OUTPUT_DIR, f"scaler_{model_id}.joblib"))

    # Convert to Tensors
    X_tr_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32)
    X_v_t = torch.tensor(X_val_s, dtype=torch.float32)
    X_te_t = torch.tensor(X_test_s, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize Model & Multi-GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiLabelLogReg(X_train.shape[1], y_train.shape[1])
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # Handle Class Imbalance dynamically
    pos_counts = y_train.sum(axis=0)
    neg_counts = len(y_train) - pos_counts
    # Avoid division by zero if a class has 0 positives
    pos_weight = np.where(pos_counts > 0, neg_counts / pos_counts, 1.0)
    pos_weight_t = torch.tensor(pos_weight, dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    # Move validation sets to GPU for fast evaluation
    X_v_t = X_v_t.to(device)
    
    def calculate_auc(y_true, y_prob):
        aucs = [
            roc_auc_score(y_true[:, i], y_prob[:, i]) 
            if len(np.unique(y_true[:, i])) > 1 else np.nan 
            for i in range(len(CHEXPERT_DISEASES))
        ]
        return np.nanmean(aucs)

    # Training Loop
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

        # Validation Step
        model.eval()
        with torch.no_grad():
            val_logits = model(X_v_t)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            
        current_val_auc = calculate_auc(y_val, val_probs)
        
        print(f"Epoch {epoch+1:03d} | Loss: {epoch_loss/len(train_loader):.4f} | Val AUC: {current_val_auc:.4f}")

        # Check Early Stopping
        if early_stopping(current_val_auc, model):
            print(f"Early stopping triggered at epoch {epoch+1}. Restoring best weights (AUC: {early_stopping.best_score:.4f}).")
            break
            
    # Load best weights
    model.load_state_dict(early_stopping.best_weights)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"model_{model_id}.pt"))

    # Final Evaluation (Best Model)
    model.eval()
    with torch.no_grad():
        X_te_t = X_te_t.to(device)
        best_val_probs = torch.sigmoid(model(X_v_t)).cpu().numpy()
        test_probs = torch.sigmoid(model(X_te_t)).cpu().numpy()
        
    np.save(os.path.join(OUTPUT_DIR, f"{model_id}_val_probs.npy"), best_val_probs)
    np.save(os.path.join(OUTPUT_DIR, f"{model_id}_test_probs.npy"), test_probs)

    best_val_auc = calculate_auc(y_val, best_val_probs)
    test_auc = calculate_auc(y_test, test_probs)
    
    print(f"--- FINAL {model_id} | Val AUC: {best_val_auc:.4f} | Test AUC: {test_auc:.4f} ---")
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

    # Filters
    valid_data_mask = (df["ignore"] != 1)
    view_data_mask = (df["ViewCodeSequence_CodeMeaning"] == "postero-anterior")
    base_mask = valid_data_mask & view_data_mask

    train_mask = base_mask & (df["split"] == "train")
    val_mask = base_mask & (df["split"].isin(["val", "valid", "validate"]))
    test_mask = base_mask & (df["split"] == "test")

    # Labels
    y_train = process_labels(df.loc[train_mask, "CheXpert_labels"])
    y_val = process_labels(df.loc[val_mask, "CheXpert_labels"])
    y_test = process_labels(df.loc[test_mask, "CheXpert_labels"])

    np.save(os.path.join(OUTPUT_DIR, "y_val_true.npy"), y_val)
    np.save(os.path.join(OUTPUT_DIR, "y_test_true.npy"), y_test)

    results = []

    # 1. Foundation Models
    for model_name in MODELS:
        for variant in VARIANTS:
            col_key = f"{model_name}_{variant}"
            X_train = np.vstack(df.loc[train_mask, col_key].values)
            X_val = np.vstack(df.loc[val_mask, col_key].values)
            X_test = np.vstack(df.loc[test_mask, col_key].values)
            
            # Weight decay maps conceptually to the inverse of C in sklearn
            # Smaller embeddings need less regularization, large ones (CXR_Found) need more
            wd = 1e-1 if model_name == "CXR_Foundation" else 1e-2
            
            v_auc, t_auc = train_pytorch_model(
                col_key, X_train, y_train, X_val, y_val, X_test, y_test, 
                weight_decay=wd
            )
            results.append({"Model": model_name, "Variant": variant, "Val_AUC": v_auc, "Test_AUC": t_auc})

    # 2. Early Fusion
    for variant in VARIANTS:
        X_train_fusion = np.hstack([np.vstack(df.loc[train_mask, f"{m}_{variant}"].values) for m in MODELS])
        X_val_fusion = np.hstack([np.vstack(df.loc[val_mask, f"{m}_{variant}"].values) for m in MODELS])
        X_test_fusion = np.hstack([np.vstack(df.loc[test_mask, f"{m}_{variant}"].values) for m in MODELS])
        
        # Very high weight decay for the ~8k dimensional fusion space
        v_auc, t_auc = train_pytorch_model(
            f"Early_Fusion_{variant}", X_train_fusion, y_train, X_val_fusion, y_val, X_test_fusion, y_test, 
            weight_decay=0.5
        )
        results.append({"Model": "Early_Fusion", "Variant": variant, "Val_AUC": v_auc, "Test_AUC": t_auc})

    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, "roc_auc_summary.csv"), index=False)
    print("All models trained and artifacts saved.")

if __name__ == "__main__":
    main()