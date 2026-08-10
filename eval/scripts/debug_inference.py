import os
import torch
import torch.nn as nn
import lancedb
import joblib
import numpy as np
import xgboost as xgb
from pathlib import Path

LANCE_DB_PATH = os.path.expanduser("~/cxr_foundation_eval/embeddings/MIMIC-CXR-JPG/")
ARTIFACTS_DIR = Path("artifacts")

MODELS = ["MedSigLIP", "BioViL-T", "EVA-X", "CheXFound", "CheXagent", "CXR_Foundation", "Early_Fusion"]
HEADS = ["LR", "XGB", "s2", "i2"]
VARS = ["raw", "l2", "pca95"]
TEST_DISEASE = "Atelectasis"

# --- PyTorch Class Definitions ---

class MultiLabelLogReg(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)

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
            k2 = max(int(k1 / 2), num_classes)
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

class SingleDiseaseMLP(nn.Module):
    # Added num_classes to dynamically handle the checkpoint's 4-output structure
    def __init__(self, input_dim, alpha, dropout_p, depth, num_classes=1):
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
            self.net = nn.Sequential(*layers, nn.Linear(k2, num_classes)) 
        else:
            self.net = nn.Sequential(*layers, nn.Linear(k1, num_classes))
            
    def forward(self, x):
        out = self.net(x)
        # Squeeze only if it's strictly binary, otherwise return raw logits
        if out.shape[-1] == 1:
            return out.squeeze(-1)
        return out

# --- Helper Functions ---

def clean_state_dict(state_dict):
    """Removes 'module.' prefixes if the model was saved via DataParallel."""
    return {k.replace('module.', ''): v for k, v in state_dict.items()}

def load_mlp_from_state_dict(state_dict, model_type="s2"):
    """Dynamically reconstructs the MLP based on weight shapes."""
    w0_shape = state_dict['net.0.weight'].shape
    k1, input_dim = w0_shape[0], w0_shape[1]
    
    alpha = k1 / input_dim
    depth = 2 if 'net.8.weight' in state_dict else 1
    
    # Infer exact number of output classes from the final layer
    last_layer_key = 'net.8.weight' if depth == 2 else 'net.4.weight'
    num_classes = state_dict[last_layer_key].shape[0]
    
    dropout_p = 0.0 # Disabled for inference
    
    if model_type == "s2":
        model = HarmonicMLP(input_dim, alpha, dropout_p, depth, num_classes)
    elif model_type == "i2":
        model = SingleDiseaseMLP(input_dim, alpha, dropout_p, depth, num_classes)
        
    model.load_state_dict(state_dict)
    model.eval()
    return model

def get_embeddings_for_model(db, model_name, var, limit=3):
    db_var = "raw" if var == "pca95" else var
    col_key = f"{model_name}_{db_var}"
    
    table = db.open_table("complete_embeddings_MIMIC-CXR-JPG")
    df = table.search().limit(limit).to_pandas()
    
    if col_key not in df.columns:
        return None
    return np.vstack(df[col_key].values)

def apply_preprocessing(embeddings, model, head, var):
    processed = embeddings.copy()
    if var == "pca95":
        # Check both naming conventions for the PCA artifact
        scaler_path = ARTIFACTS_DIR / f"{model}_{head}_{var}_scaler.joblib"
        pca_path = ARTIFACTS_DIR / f"{model}_{head}_{var}_pca_transformer.joblib"
        alt_pca_path = ARTIFACTS_DIR / f"{model}_{head}_raw_pca_transformer.joblib" # The fallback
        
        if not scaler_path.exists():
            return None
            
        if pca_path.exists():
            pca_file_to_load = pca_path
        elif alt_pca_path.exists():
            pca_file_to_load = alt_pca_path
        else:
            return None
            
        scaler = joblib.load(scaler_path)
        pca = joblib.load(pca_file_to_load)
        
        # CORRECT ORDER for LR: PCA first (1152 -> 119), then Scaler normalizes the 119
        # CORRECT ORDER for XGB: Scaler first (1152 -> 1152), then PCA reduces to 119
        if head == "lr" or head == "LR":
            processed = pca.transform(processed)
            processed = scaler.transform(processed)
        elif head == "xgb" or head == "XGB":
            processed = scaler.transform(processed)
            processed = pca.transform(processed)

        
    return processed

# --- Main Sweep ---

def main():
    print(f"Connecting to LanceDB at {LANCE_DB_PATH}...\n")
    db = lancedb.connect(LANCE_DB_PATH)
    
    results = []
    
    for model in MODELS:
        for head in HEADS:
            for var in VARS:
                combo_name = f"{model} | {head} | {var}"
                
                # Filter out unnecessary combos early
                if model == "Early_Fusion":
                    continue # Skipping Early Fusion for now until we build the concat logic
                if head in ["s2", "i2"] and var in ["l2", "pca95"]:
                    results.append((combo_name, "⏭️ Skipped (MLPs only use raw embeddings)"))
                    continue
                
                # 1. Check Data
                try:
                    embeddings = get_embeddings_for_model(db, model, var)
                    if embeddings is None:
                        results.append((combo_name, "❌ Missing DB Column"))
                        continue
                except Exception as e:
                    results.append((combo_name, f"❌ DB Error: {e}"))
                    continue
                    
                # 2. Check Preprocessing Artifacts
                try:
                    processed_embs = apply_preprocessing(embeddings, model, head, var)
                    if processed_embs is None:
                        results.append((combo_name, "❌ Missing PCA/Scaler Artifacts"))
                        continue
                except Exception as e:
                    results.append((combo_name, f"❌ Preprocessing Error: {e}"))
                    continue

                # 3. Check Inference Artifacts & Run
                try:
                    tensor_embs = torch.tensor(processed_embs, dtype=torch.float32)
                    
                    if head == "XGB":
                        model_path = ARTIFACTS_DIR / f"{model}_{head}_{var}_{TEST_DISEASE}.json"
                        if not model_path.exists():
                            results.append((combo_name, "❌ Missing XGB Model"))
                            continue
                            
                        bst = xgb.Booster()
                        bst.load_model(model_path)
                        preds = bst.predict(xgb.DMatrix(processed_embs))
                        results.append((combo_name, f"✅ OK (XGB Preds: {preds[0]:.3f}, {preds[1]:.3f}, {preds[2]:.3f})"))
                        
                    elif head in ["LR", "i2", "s2"]:
                        model_path = ARTIFACTS_DIR / f"{model}_{head}_{var}_{TEST_DISEASE}.pt"
                        if not model_path.exists():
                            model_path = ARTIFACTS_DIR / f"{model}_{head}_{var}_weights.pt"
                        
                        if not model_path.exists():
                            results.append((combo_name, "❌ Missing PyTorch Weights"))
                            continue
                            
                        state_dict = clean_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
                        
                        if head == "LR":
                            if "linear.weight" in state_dict:
                                input_dim = processed_embs.shape[1]
                                num_classes = state_dict["linear.weight"].shape[0]
                                
                                model_lr = MultiLabelLogReg(input_dim, num_classes)
                                model_lr.load_state_dict(state_dict)
                                model_lr.eval()
                                
                                with torch.no_grad():
                                    logits = model_lr(tensor_embs)
                                    preds = torch.sigmoid(logits).numpy()
                                    # Handle varying output dimensions for formatting
                                    pred_str = ", ".join([f"{p:.3f}" for p in preds[0][:3]])
                                    results.append((combo_name, f"✅ OK (LR Preds: {pred_str})"))
                            else:
                                results.append((combo_name, f"❌ LR key mismatch. Found keys: {list(state_dict.keys())[:3]}..."))
                                
                        elif head in ["s2", "i2"]:
                            model_mlp = load_mlp_from_state_dict(state_dict, model_type=head)
                            
                            with torch.no_grad():
                                logits = model_mlp(tensor_embs)
                                preds = torch.sigmoid(logits).numpy()
                                
                                # Formatting handles multi-class outputs safely
                                if preds.ndim > 1:
                                    pred_str = ", ".join([f"{p:.3f}" for p in preds[0][:3]])
                                else:
                                    pred_str = ", ".join([f"{p:.3f}" for p in preds[:3]])
                                    
                                results.append((combo_name, f"✅ OK ({head} Preds: {pred_str})"))
                
                except Exception as e:
                    results.append((combo_name, f"❌ Inference Error: {e}"))

    # Print Summary Report
    print("=== SWEEP REPORT ===")
    success_count = sum(1 for r in results if "✅" in r[1])
    skipped_count = sum(1 for r in results if "⏭️" in r[1])
    
    for combo, status in results:
        print(f"{combo.ljust(35)} -> {status}")
        
    print(f"\nTotal Viable Combinations: {success_count} / {len(results) - skipped_count}")

if __name__ == "__main__":
    main()