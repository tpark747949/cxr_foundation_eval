import os
import lancedb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
DB_URI = "../embeddings/MIMIC-CXR-JPG"
TABLE_NAME = "sampled_embeddings_MIMIC-CXR-JPG"
# OUTPUT_DIR = "shared_4class_artifacts"
OUTPUT_DIR = "../../artifacts/1p/shared_4class_mlp"

MODELS = ["MedSigLIP", "BioViL-T", "EVA-X", "CheXFound", "CheXagent", "CXR_Foundation", "Early_Fusion"]
CHEXPERT_DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged_Cardiomediastinum", "Fracture", "Lung_Lesion",
    "Lung_Opacity", "Pleural_Effusion", "Pneumonia",
    "Pneumothorax", "Pleural_Other", "Support_Devices", "No_Finding"
]

def process_4class_labels(df_labels):
    labels_matrix = np.zeros((len(df_labels), len(CHEXPERT_DISEASES)), dtype=np.int64)
    for idx, row in enumerate(df_labels):
        for d_idx, disease in enumerate(CHEXPERT_DISEASES):
            val = row[disease]
            if val == 1.0:
                labels_matrix[idx, d_idx] = 1 
            elif val == -1.0:
                labels_matrix[idx, d_idx] = 2 
            elif val == -2.0 or np.isnan(val):
                labels_matrix[idx, d_idx] = 3 
            else:
                labels_matrix[idx, d_idx] = 0 
    return labels_matrix

def safe_macro_auc(y_true_1d, y_prob_2d, num_classes=4):
    """Safely computes One-vs-Rest Macro AUC, completely ignoring missing classes."""
    aucs = []
    for c in range(num_classes):
        y_binary = (y_true_1d == c).astype(int)
        # Only calculate AUC if the split contains both positive and negative examples for this class
        if len(np.unique(y_binary)) == 2:
            aucs.append(roc_auc_score(y_binary, y_prob_2d[:, c]))
    return np.mean(aucs) if len(aucs) > 0 else 0.5

def compute_4class_metrics(y_true, y_prob):
    probs_permuted = np.transpose(y_prob, (0, 2, 1)) # -> (N, 14, 4)
    all_aucs = []
    
    for d_idx in range(14):
        y_t = y_true[:, d_idx]
        y_p = probs_permuted[:, d_idx, :] 
        all_aucs.append(safe_macro_auc(y_t, y_p, 4))
                
    return np.mean(all_aucs)

class Shared4ClassMLP(nn.Module):
    def __init__(self, input_dim, num_diseases=14, num_classes=4, alpha=1.0, dropout_p=0.5):
        super().__init__()
        self.num_diseases = num_diseases
        self.num_classes = num_classes
        k1 = int(input_dim * alpha)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, k1),
            nn.LayerNorm(k1),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(k1, num_diseases * num_classes)
        )
            
    def forward(self, x):
        out = self.net(x)
        return out.view(-1, self.num_diseases, self.num_classes).permute(0, 2, 1)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    db = lancedb.connect(DB_URI)
    df = db.open_table(TABLE_NAME).to_pandas()

    # Masks
    valid_data_mask = (df["ignore"] != 1)
    view_data_mask = (df["ViewCodeSequence_CodeMeaning"] == "postero-anterior")
    sample_mask = (df["sample_1_percent"] == 1)
    base_mask = valid_data_mask & view_data_mask & sample_mask

    train_mask = base_mask & (df["split"] == "train")
    val_mask = base_mask & (df["split"].isin(["val", "valid", "validate"]))
    test_mask = base_mask & (df["split"] == "test")

    # Labels
    y_train = process_4class_labels(df.loc[train_mask, "CheXpert_labels"])
    y_val = process_4class_labels(df.loc[val_mask, "CheXpert_labels"])
    y_test = process_4class_labels(df.loc[test_mask, "CheXpert_labels"])

    # y_train = process_labels(df.loc[train_mask, "NegBio_labels"])
    # y_val = process_labels(df.loc[val_mask, "NegBio_labels"])
    # y_test = process_labels(df.loc[test_mask, "NegBio_labels"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for model_name in MODELS:
        print(f"\n=== Training Shared 4-Class MLP for {model_name} ===")
        
        if model_name == "Early_Fusion":
            X_tr_raw = np.hstack([np.vstack(df.loc[train_mask, f"{m}_raw"].values) for m in MODELS[:-1]])
            X_v_raw = np.hstack([np.vstack(df.loc[val_mask, f"{m}_raw"].values) for m in MODELS[:-1]])
            X_te_raw = np.hstack([np.vstack(df.loc[test_mask, f"{m}_raw"].values) for m in MODELS[:-1]])
        else:
            X_tr_raw = np.vstack(df.loc[train_mask, f"{model_name}_raw"].values)
            X_v_raw = np.vstack(df.loc[val_mask, f"{model_name}_raw"].values)
            X_te_raw = np.vstack(df.loc[test_mask, f"{model_name}_raw"].values)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_v = scaler.transform(X_v_raw)
        X_te = scaler.transform(X_te_raw)

        X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
        y_tr_t = torch.tensor(y_train, dtype=torch.long, device=device)
        X_v_t = torch.tensor(X_v, dtype=torch.float32, device=device)
        X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)
        
        train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=2048, shuffle=True)
        
        model = Shared4ClassMLP(input_dim=X_tr.shape[1]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        best_val_auc = -np.inf
        best_weights = None
        patience = 0
        
        for epoch in range(100):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
                
            model.eval()
            with torch.no_grad():
                val_probs = torch.softmax(model(X_v_t), dim=1).cpu().numpy()
            
            mean_auc = compute_4class_metrics(y_val, val_probs)
            if mean_auc > best_val_auc + 0.0001:
                best_val_auc = mean_auc
                best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 5:
                    break
                    
        # Failsafe in case it immediately broke on Epoch 0 with no improvement 
        if best_weights is None:
             best_weights = {k: v.cpu() for k, v in model.state_dict().items()}

        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
        model.eval()
        with torch.no_grad():
            test_probs = torch.softmax(model(X_te_t), dim=1).cpu().numpy()
            
        final_test_auc = compute_4class_metrics(y_test, test_probs)
        print(f"[{model_name}] Final Test Macro AUC across 4 classes: {final_test_auc:.4f}")
        np.save(os.path.join(OUTPUT_DIR, f"{model_name}_shared_4class_probs.npy"), test_probs)

if __name__ == "__main__":
    main()