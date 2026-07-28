import os
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
from joblib import Parallel, delayed

# --- Configuration & Constants ---
DB_URI = "../embeddings/MIMIC-CXR-JPG"
TABLE_NAME = "complete_embeddings_MIMIC-CXR-JPG"
OUTPUT_DIR = "mlp_grid_artifacts"
NUM_GPUS = 4  # Matches your 4x A6000 setup

MODELS = ["MedSigLIP", "BioViL-T", "EVA-X", "CheXFound", "CheXagent", "CXR_Foundation"]
CHEXPERT_DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged_Cardiomediastinum", "Fracture", "Lung_Lesion",
    "Lung_Opacity", "Pleural_Effusion", "Pneumonia",
    "Pneumothorax", "Pleural_Other", "Support_Devices", "No_Finding"
]

# --- Harmonic Grid Definition ---
ALPHAS = [0.5, 0.67, 1.0, 1.5, 2.0]
DROPOUTS = [0.2, 0.5]
DEPTHS = [1, 2]


# --- MLP Architecture ---
class HarmonicMLP(nn.Module):
    def __init__(self, input_dim, alpha, dropout_p, depth, num_classes=14):
        super().__init__()
        k1 = int(input_dim * alpha)
        
        layers = [
            nn.Linear(input_dim, k1),
            nn.LayerNorm(k1),
            nn.GELU(),
            nn.Dropout(dropout_p)
        ]
        
        if depth == 2:
            k2 = max(int(k1 / 2), num_classes) # Ensure it doesn't compress below output size
            layers.extend([
                nn.Linear(k1, k2),
                nn.LayerNorm(k2),
                nn.GELU(),
                nn.Dropout(dropout_p)
            ])
            self.net = nn.Sequential(*layers, nn.Linear(k2, num_classes))
        else:
            self.net = nn.Sequential(*layers, nn.Linear(k1, num_classes))
            
    def forward(self, x):
        return self.net(x)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = -np.inf

    def __call__(self, current_score):
        if current_score >= self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0
            return False, True # Stop: False, Is_Best: True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True, False
        return False, False


def calculate_auc(y_true, y_prob):
    aucs = [
        roc_auc_score(y_true[:, i], y_prob[:, i]) 
        if len(np.unique(y_true[:, i])) > 1 else np.nan 
        for i in range(len(CHEXPERT_DISEASES))
    ]
    return np.nanmean(aucs)


# --- GPU Worker Function ---
def train_mlp_trial(trial_idx, config, model_name, X_tr, y_tr, X_v, y_v, X_te, y_te):
    """
    Sandboxed worker to run a single hyperparameter configuration on a dedicated GPU.
    """
    gpu_id = trial_idx % NUM_GPUS
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    
    device = torch.device("cuda:0")
    
    # Standardize data
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_v_s = scaler.transform(X_v)
    X_te_s = scaler.transform(X_te)
    
    # To Tensors
    X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    X_v_t = torch.tensor(X_v_s, dtype=torch.float32, device=device)
    X_te_t = torch.tensor(X_te_s, dtype=torch.float32, device=device)
    
    # DataLoader
    dataset = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(dataset, batch_size=2048, shuffle=True)
    
    # Model Setup
    model = HarmonicMLP(
        input_dim=X_tr.shape[1], 
        alpha=config['alpha'], 
        dropout_p=config['dropout'], 
        depth=config['depth']
    ).to(device)
    
    # Loss & Optimizer
    pos_counts = y_tr.sum(axis=0)
    neg_counts = len(y_tr) - pos_counts
    pos_weight = np.sqrt(np.where(pos_counts > 0, neg_counts / pos_counts, 1.0))
    pos_weight_t = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    early_stopping = EarlyStopping(patience=6, min_delta=0.001)
    
    best_weights = None
    
    # Training Loop
    for epoch in range(200):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(X_v_t)).cpu().numpy()
            
        current_auc = calculate_auc(y_v, val_probs)
        stop, is_best = early_stopping(current_auc)
        
        if is_best:
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
            
        if stop:
            break
            
    # Load best weights for final test evaluation
    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    model.eval()
    with torch.no_grad():
        best_val_probs = torch.sigmoid(model(X_v_t)).cpu().numpy()
        best_test_probs = torch.sigmoid(model(X_te_t)).cpu().numpy()
        
    # 2. Fix the calculation inside the function
    final_val_auc = early_stopping.best_score
    final_test_auc = calculate_auc(y_te, best_test_probs)  # ✅ Fixed to use y_te
    
    return config, final_val_auc, final_test_auc, best_val_probs, best_test_probs, best_weights, scaler


def process_labels(df_labels):
    labels_matrix = np.zeros((len(df_labels), len(CHEXPERT_DISEASES)))
    for idx, row in enumerate(df_labels):
        for d_idx, disease in enumerate(CHEXPERT_DISEASES):
            val = row[disease]
            labels_matrix[idx, d_idx] = 1.0 if val == 1.0 else 0.0
    return labels_matrix


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    db = lancedb.connect(DB_URI)
    tbl = db.open_table(TABLE_NAME)
    df = tbl.to_pandas()

    valid_mask = (df["ignore"] != 1) & (df["ViewCodeSequence_CodeMeaning"] == "postero-anterior")
    train_mask = valid_mask & (df["split"] == "train")
    val_mask = valid_mask & (df["split"].isin(["val", "valid", "validate"]))
    test_mask = valid_mask & (df["split"] == "test")

    y_train = process_labels(df.loc[train_mask, "CheXpert_labels"])
    y_val = process_labels(df.loc[val_mask, "CheXpert_labels"])
    y_test = process_labels(df.loc[test_mask, "CheXpert_labels"])

    np.save(os.path.join(OUTPUT_DIR, "y_val_true.npy"), y_val)
    np.save(os.path.join(OUTPUT_DIR, "y_test_true.npy"), y_test)

    # Build the Grid
    grid = [{"alpha": a, "dropout": d, "depth": dp} for a in ALPHAS for d in DROPOUTS for dp in DEPTHS]
    
    # We will test the 6 models + Early Fusion (Raw only, no L2/PCA needed for MLPs)
    evaluation_queue = MODELS + ["Early_Fusion"]
    
    final_summary = []

    for model_name in evaluation_queue:
        print(f"\n=== Starting Grid Search for {model_name} ===")
        
        # Load Data
        if model_name == "Early_Fusion":
            X_tr = np.hstack([np.vstack(df.loc[train_mask, f"{m}_raw"].values) for m in MODELS])
            X_v = np.hstack([np.vstack(df.loc[val_mask, f"{m}_raw"].values) for m in MODELS])
            X_te = np.hstack([np.vstack(df.loc[test_mask, f"{m}_raw"].values) for m in MODELS])
        else:
            X_tr = np.vstack(df.loc[train_mask, f"{model_name}_raw"].values)
            X_v = np.vstack(df.loc[val_mask, f"{model_name}_raw"].values)
            X_te = np.vstack(df.loc[test_mask, f"{model_name}_raw"].values)

        # Run Grid in Parallel across 4 GPUs
        results = Parallel(n_jobs=NUM_GPUS, backend="loky")(
            delayed(train_mlp_trial)(
                idx, config, model_name, X_tr, y_train, X_v, y_val, X_te, y_test  # ✅ Added y_test
            ) for idx, config in enumerate(grid)
        )
        
        # Find Best Config based on Validation AUC
        best_result = max(results, key=lambda x: x[1])
        best_config, best_v_auc, _, best_v_probs, best_t_probs, best_weights, best_scaler = best_result
        
        # Calculate actual Test AUC
        best_t_auc = calculate_auc(y_test, best_t_probs)
        
        print(f"--> BEST [{model_name}]: {best_config} | Val AUC: {best_v_auc:.4f} | Test AUC: {best_t_auc:.4f}")
        
        final_summary.append({
            "Model": model_name,
            "Best_Alpha": best_config['alpha'],
            "Best_Dropout": best_config['dropout'],
            "Best_Depth": best_config['depth'],
            "Val_AUC": best_v_auc,
            "Test_AUC": best_t_auc
        })
        
        # Save Artifacts for the best configuration
        np.save(os.path.join(OUTPUT_DIR, f"{model_name}_val_probs.npy"), best_v_probs)
        np.save(os.path.join(OUTPUT_DIR, f"{model_name}_test_probs.npy"), best_t_probs)
        torch.save(best_weights, os.path.join(OUTPUT_DIR, f"{model_name}_best_weights.pt"))
        joblib.dump(best_scaler, os.path.join(OUTPUT_DIR, f"{model_name}_scaler.joblib"))

    # Save final report
    pd.DataFrame(final_summary).to_csv(os.path.join(OUTPUT_DIR, "mlp_grid_summary.csv"), index=False)
    print("\nMLP Grid Search Completed. Results saved.")

if __name__ == "__main__":
    main()