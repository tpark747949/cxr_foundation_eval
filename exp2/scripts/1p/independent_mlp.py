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

# --- Patched Configuration ---
DB_URI = "../embeddings/MIMIC-CXR-JPG"
TABLE_NAME = "sampled_embeddings_MIMIC-CXR-JPG"
# OUTPUT_DIR = "independent_mlp_artifacts"
OUTPUT_DIR = "../../artifacts/1p/independent_binary_mlp"
NUM_GPUS = 4

MODELS = ["MedSigLIP", "BioViL-T", "EVA-X", "CheXFound", "CheXagent", "CXR_Foundation", "Early_Fusion"]
CHEXPERT_DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged_Cardiomediastinum", "Fracture", "Lung_Lesion",
    "Lung_Opacity", "Pleural_Effusion", "Pneumonia",
    "Pneumothorax", "Pleural_Other", "Support_Devices", "No_Finding"
]

# Grid Definition
ALPHAS = [0.5, 0.67, 1.0, 1.5, 2.0]
DROPOUTS = [0.2, 0.5]
DEPTHS = [1, 2]

class SingleDiseaseMLP(nn.Module):
    def __init__(self, input_dim, alpha, dropout_p, depth):
        super().__init__()
        k1 = int(input_dim * alpha)
        
        layers = [
            nn.Linear(input_dim, k1),
            nn.LayerNorm(k1),
            nn.GELU(),
            nn.Dropout(dropout_p)
        ]
        
        if depth == 2:
            k2 = max(int(k1 / 2), 2)
            layers.extend([
                nn.Linear(k1, k2),
                nn.LayerNorm(k2),
                nn.GELU(),
                nn.Dropout(dropout_p)
            ])
            self.net = nn.Sequential(*layers, nn.Linear(k2, 1)) # Output is 1 (Binary)
        else:
            self.net = nn.Sequential(*layers, nn.Linear(k1, 1))
            
    def forward(self, x):
        return self.net(x).squeeze(-1) # Output shape (Batch,)

class EarlyStopping:
    def __init__(self, patience=6, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = -np.inf

    def __call__(self, current_score):
        if current_score >= self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0
            return False, True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True, False
        return False, False

def train_single_trial(task_idx, disease_idx, config, X_tr, y_tr_1d, X_v, y_v_1d, X_te, y_te_1d):
    """
    Trains a single architecture configuration for a SINGLE disease on a dedicated GPU.
    """
    gpu_id = task_idx % NUM_GPUS
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    
    device = torch.device("cuda:0")
    
    # Send pre-scaled data to GPU
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr_1d, dtype=torch.float32, device=device)
    X_v_t = torch.tensor(X_v, dtype=torch.float32, device=device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)
    
    dataset = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(dataset, batch_size=2048, shuffle=True)
    
    model = SingleDiseaseMLP(
        input_dim=X_tr.shape[1], 
        alpha=config['alpha'], 
        dropout_p=config['dropout'], 
        depth=config['depth']
    ).to(device)
    
    # Dynamic Pos Weight for this specific disease
    pos_count = y_tr_1d.sum()
    neg_count = len(y_tr_1d) - pos_count
    # Square root damping to prevent extreme loss spikes on rare diseases
    pos_weight_val = np.sqrt(neg_count / max(pos_count, 1))
    pos_weight_t = torch.tensor(pos_weight_val, dtype=torch.float32, device=device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    early_stopping = EarlyStopping(patience=6, min_delta=0.001)
    
    best_weights = None
    
    for epoch in range(50):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            
        # Validation on single disease
        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(X_v_t)).cpu().numpy()
            
        try:
            current_auc = roc_auc_score(y_v_1d, val_probs)
        except ValueError: # Failsafe if validation set missing positive cases
            current_auc = 0.5 

        stop, is_best = early_stopping(current_auc)
        
        if is_best:
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
            
        if stop:
            break
            
    # Load best and do final inference
    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    model.eval()
    with torch.no_grad():
        best_v_probs = torch.sigmoid(model(X_v_t)).cpu().numpy()
        test_probs = torch.sigmoid(model(X_te_t)).cpu().numpy()
        
    try:
        final_test_auc = roc_auc_score(y_te_1d, test_probs)
    except ValueError:
        final_test_auc = np.nan
        
    return disease_idx, config, early_stopping.best_score, final_test_auc, best_v_probs, test_probs, best_weights

def process_labels(df_labels):
    labels_matrix = np.zeros((len(df_labels), len(CHEXPERT_DISEASES)))
    for idx, row in enumerate(df_labels):
        for d_idx, disease in enumerate(CHEXPERT_DISEASES):
            labels_matrix[idx, d_idx] = 1.0 if row[disease] == 1.0 else 0.0
    return labels_matrix

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    db = lancedb.connect(DB_URI)
    tbl = db.open_table(TABLE_NAME)
    df = tbl.to_pandas()

    # Masks
    valid_data_mask = (df["ignore"] != 1)
    view_data_mask = (df["ViewCodeSequence_CodeMeaning"] == "postero-anterior")
    sample_mask = (df["sample_1_percent"] == 1)
    base_mask = valid_data_mask & view_data_mask & sample_mask

    train_mask = base_mask & (df["split"] == "train")
    val_mask = base_mask & (df["split"].isin(["val", "valid", "validate"]))
    test_mask = base_mask & (df["split"] == "test")

    # Labels
    y_train = process_labels(df.loc[train_mask, "CheXpert_labels"])
    y_val = process_labels(df.loc[val_mask, "CheXpert_labels"])
    y_test = process_labels(df.loc[test_mask, "CheXpert_labels"])

    # y_train = process_labels(df.loc[train_mask, "NegBio_labels"])
    # y_val = process_labels(df.loc[val_mask, "NegBio_labels"])
    # y_test = process_labels(df.loc[test_mask, "NegBio_labels"])

    np.save(os.path.join(OUTPUT_DIR, "y_val_true.npy"), y_val)
    np.save(os.path.join(OUTPUT_DIR, "y_test_true.npy"), y_test)

    grid = [{"alpha": a, "dropout": d, "depth": dp} for a in ALPHAS for d in DROPOUTS for dp in DEPTHS]
    final_summary = []

    for model_name in MODELS:
        print(f"\n=== Training 14 Independent MLPs for {model_name} ===")
        
        if model_name == "Early_Fusion":
            X_tr_raw = np.hstack([np.vstack(df.loc[train_mask, f"{m}_raw"].values) for m in MODELS[:-1]])
            X_v_raw = np.hstack([np.vstack(df.loc[val_mask, f"{m}_raw"].values) for m in MODELS[:-1]])
            X_te_raw = np.hstack([np.vstack(df.loc[test_mask, f"{m}_raw"].values) for m in MODELS[:-1]])
        else:
            X_tr_raw = np.vstack(df.loc[train_mask, f"{model_name}_raw"].values)
            X_v_raw = np.vstack(df.loc[val_mask, f"{model_name}_raw"].values)
            X_te_raw = np.vstack(df.loc[test_mask, f"{model_name}_raw"].values)

        # Scale ONCE per model in the main thread
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_v = scaler.transform(X_v_raw)
        X_te = scaler.transform(X_te_raw)
        joblib.dump(scaler, os.path.join(OUTPUT_DIR, f"{model_name}_scaler.joblib"))

        # Build task list: 14 diseases * 20 configs = 280 tasks
        tasks = []
        for d_idx in range(len(CHEXPERT_DISEASES)):
            for config in grid:
                tasks.append((d_idx, config))

        results = Parallel(n_jobs=NUM_GPUS, backend="loky")(
            delayed(train_single_trial)(
                task_idx, d_idx, config, 
                X_tr, y_train[:, d_idx], 
                X_v, y_val[:, d_idx], 
                X_te, y_test[:, d_idx]
            ) for task_idx, (d_idx, config) in enumerate(tasks)
        )
        
        # Aggregate the best results for each disease
        best_val_probs_matrix = np.zeros((len(X_v), len(CHEXPERT_DISEASES)))
        best_test_probs_matrix = np.zeros((len(X_te), len(CHEXPERT_DISEASES)))
        
        for d_idx, disease in enumerate(CHEXPERT_DISEASES):
            # Filter results for this specific disease
            disease_results = [r for r in results if r[0] == d_idx]
            
            # Find the best config based on Validation AUC
            best_run = max(disease_results, key=lambda x: x[2])
            _, best_config, best_v_auc, best_t_auc, v_probs, t_probs, best_weights = best_run
            
            # Reconstruct matrices
            best_val_probs_matrix[:, d_idx] = v_probs
            best_test_probs_matrix[:, d_idx] = t_probs
            
            print(f"[{disease}] Best Config: {best_config} | Test AUC: {best_t_auc:.4f}")
            
            # Save independent weights
            torch.save(best_weights, os.path.join(OUTPUT_DIR, f"{model_name}_{disease}_best_weights.pt"))
            
            final_summary.append({
                "Model": model_name,
                "Disease": disease,
                "Best_Alpha": best_config['alpha'],
                "Best_Dropout": best_config['dropout'],
                "Best_Depth": best_config['depth'],
                "Val_AUC": best_v_auc,
                "Test_AUC": best_t_auc
            })
            
        # Save reconstructed full matrices for the evaluation suite
        np.save(os.path.join(OUTPUT_DIR, f"{model_name}_val_probs.npy"), best_val_probs_matrix)
        np.save(os.path.join(OUTPUT_DIR, f"{model_name}_test_probs.npy"), best_test_probs_matrix)

    pd.DataFrame(final_summary).to_csv(os.path.join(OUTPUT_DIR, "independent_mlp_grid_summary.csv"), index=False)
    print("\n14-Independent-MLP Grid Search Completed. Results saved.")

if __name__ == "__main__":
    main()