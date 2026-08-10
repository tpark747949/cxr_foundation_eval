import streamlit as st
import subprocess
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
import joblib
import tempfile
import os
from pathlib import Path
from PIL import Image

# --- Configuration ---
PROJECT_ROOT = Path("~/cxr_foundation_eval").expanduser()
ARTIFACTS_DIR = PROJECT_ROOT / "eval" / "scripts" / "artifacts"

MODELS = ["MedSigLIP", "BioViL-T", "EVA-X", "CheXFound", "CheXagent", "CXR_Foundation", "Early_Fusion"]
HEADS = ["LR", "XGB", "s2", "i2"]
VARS = ["raw", "l2", "pca95"]
CLASSES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
    "Fracture", "Lung Lesion", "Lung Opacity", "Pleural Effusion", "Pneumonia",
    "Pneumothorax", "Pleural Other", "Support Devices", "No Finding"
]

# --- PyTorch Classes ---
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
        layers = [nn.Linear(input_dim, k1), nn.LayerNorm(k1), nn.GELU(), nn.Dropout(dropout_p)]
        if depth == 2:
            k2 = max(int(k1 / 2), num_classes)
            layers.extend([nn.Linear(k1, k2), nn.LayerNorm(k2), nn.GELU(), nn.Dropout(dropout_p)])
            self.net = nn.Sequential(*layers, nn.Linear(k2, num_classes))
        else:
            self.net = nn.Sequential(*layers, nn.Linear(k1, num_classes))
    def forward(self, x):
        return self.net(x)

class SingleDiseaseMLP(nn.Module):
    def __init__(self, input_dim, alpha, dropout_p, depth, num_classes=1):
        super().__init__()
        k1 = int(input_dim * alpha)
        layers = [nn.Linear(input_dim, k1), nn.LayerNorm(k1), nn.GELU(), nn.Dropout(dropout_p)]
        if depth == 2:
            k2 = max(int(k1 / 2), 2)
            layers.extend([nn.Linear(k1, k2), nn.LayerNorm(k2), nn.GELU(), nn.Dropout(dropout_p)])
            self.net = nn.Sequential(*layers, nn.Linear(k2, num_classes)) 
        else:
            self.net = nn.Sequential(*layers, nn.Linear(k1, num_classes))
    def forward(self, x):
        out = self.net(x)
        if out.shape[-1] == 1: return out.squeeze(-1)
        return out

def load_mlp_from_state_dict(state_dict, model_type="s2"):
    w0_shape = state_dict['net.0.weight'].shape
    k1, input_dim = w0_shape[0], w0_shape[1]
    alpha = k1 / input_dim
    depth = 2 if 'net.8.weight' in state_dict else 1
    last_layer_key = 'net.8.weight' if depth == 2 else 'net.4.weight'
    num_classes = state_dict[last_layer_key].shape[0]
    
    if model_type == "s2": model = HarmonicMLP(input_dim, alpha, 0.0, depth, num_classes)
    elif model_type == "i2": model = SingleDiseaseMLP(input_dim, alpha, 0.0, depth, num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# --- Orchestration Functions ---
def get_embedding_via_subprocess(foundation_model, image_path):
    """Bridges Streamlit to the isolated uv environments."""
    model_dir = PROJECT_ROOT / "eval" / foundation_model
    script_path = model_dir / "extract.py"
    output_npy = model_dir / "temp_embedding.npy"
    
    cmd = ["uv", "run", str(script_path), str(image_path), str(output_npy)]
    
    try:
        subprocess.run(cmd, cwd=model_dir, check=True, capture_output=True, text=True)
        embedding = np.load(output_npy)
        output_npy.unlink() # Clean up
        return embedding
    except subprocess.CalledProcessError as e:
        st.error(f"Error in {foundation_model} subprocess:\n{e.stderr}")
        return None

def apply_preprocessing(embeddings, model, head, var):
    """Applies PCA/Scaler dynamically based on our tested logic."""
    processed = embeddings.copy()
    if var == "pca95":
        scaler_path = ARTIFACTS_DIR / f"{model}_{head}_{var}_scaler.joblib"
        pca_path = ARTIFACTS_DIR / f"{model}_{head}_{var}_pca_transformer.joblib"
        alt_pca_path = ARTIFACTS_DIR / f"{model}_{head}_raw_pca_transformer.joblib"
        
        pca_file_to_load = pca_path if pca_path.exists() else alt_pca_path
        scaler, pca = joblib.load(scaler_path), joblib.load(pca_file_to_load)
        
        # Applying the preprocessing order fix
        if head.upper() == "LR":
            processed = pca.transform(processed)
            processed = scaler.transform(processed)
        elif head.upper() == "XGB":
            processed = scaler.transform(processed)
            processed = pca.transform(processed)
    return processed

# --- UI Layout ---
st.set_page_config(page_title="CXR Foundation Evaluator", layout="wide")
st.title("CXR Foundation Model Evaluator")
st.markdown("Upload a postero-anterior chest X-ray to generate embeddings and run multi-label disease classification.")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Configuration")
    selected_model = st.selectbox("Foundation Model", MODELS)
    selected_head = st.selectbox("Classifier Head", HEADS)
    
    # MLPs only use raw
    available_vars = ["raw"] if selected_head in ["s2", "i2"] else VARS
    selected_var = st.selectbox("Embedding Variant", available_vars)
    
    uploaded_file = st.file_uploader("Upload X-Ray Image", type=["jpg", "png", "jpeg"])

with col2:
    st.header("2. Inference Results")
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-Ray", width=300)
        
        if st.button("Run Inference", type="primary"):
            with st.spinner(f"Extracting embeddings using {selected_model}..."):
                # Save uploaded file to temp path
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                # 1. Extract Embedding
                if selected_model == "Early_Fusion":
                    st.info("Extracting from all models for Early Fusion...")
                    embs = []
                    for fm in [m for m in MODELS if m != "Early_Fusion"]:
                        emb = get_embedding_via_subprocess(fm, tmp_path)
                        if emb is not None: embs.append(emb)
                    raw_embedding = np.concatenate(embs, axis=1) if embs else None
                else:
                    raw_embedding = get_embedding_via_subprocess(selected_model, tmp_path)
                
                os.remove(tmp_path) # Clean up image

            if raw_embedding is not None:
                with st.spinner("Running classifier..."):
                    try:
                        # 2. Preprocess
                        processed_emb = apply_preprocessing(raw_embedding, selected_model, selected_head, selected_var)
                        tensor_emb = torch.tensor(processed_emb, dtype=torch.float32)
                        
# 3. Predict
                        preds = []
                        
                        # --- INDEPENDENT MODELS (One per disease) ---
                        if selected_head in ["XGB", "i2"]:
                            for disease in CLASSES:
                                disease_safe = disease.replace(" ", "_")
                                
                                if selected_head == "XGB":
                                    model_path = ARTIFACTS_DIR / f"{selected_model}_{selected_head}_{selected_var}_{disease_safe}.json"
                                    bst = xgb.Booster()
                                    bst.load_model(model_path)
                                    # .item() safely extracts the scalar from arrays like [0.85] or [[0.85]]
                                    p = bst.predict(xgb.DMatrix(processed_emb)).item()
                                    preds.append(p)
                                    
                                elif selected_head == "i2":
                                    model_path = ARTIFACTS_DIR / f"{selected_model}_{selected_head}_{selected_var}_{disease_safe}.pt"
                                    state_dict = {k.replace('module.', ''): v for k, v in torch.load(model_path, map_location="cpu", weights_only=True).items()}
                                    
                                    model_mlp = load_mlp_from_state_dict(state_dict, model_type="i2")
                                    with torch.no_grad():
                                        out = torch.sigmoid(model_mlp(tensor_emb)).numpy()
                                        preds.append(out.item())

                        # --- MULTI-LABEL MODELS (One model predicts all 14) ---
                        else: 
                            # Fallback logic for how multi-label weights might be named
                            model_path = ARTIFACTS_DIR / f"{selected_model}_{selected_head}_{selected_var}_weights.pt"
                            if not model_path.exists():
                                model_path = ARTIFACTS_DIR / f"{selected_model}_{selected_head}_{selected_var}.pt"
                                
                            state_dict = {k.replace('module.', ''): v for k, v in torch.load(model_path, map_location="cpu", weights_only=True).items()}
                            
                            if selected_head == "LR":
                                model_lr = MultiLabelLogReg(processed_emb.shape[1], state_dict["linear.weight"].shape[0])
                                model_lr.load_state_dict(state_dict)
                                model_lr.eval()
                                with torch.no_grad(): 
                                    # .flatten() guarantees a 1D array before converting to a Python list
                                    preds = torch.sigmoid(model_lr(tensor_emb)).numpy().flatten().tolist()
                                    
                            elif selected_head == "s2":
                                model_mlp = load_mlp_from_state_dict(state_dict, model_type="s2")
                                with torch.no_grad():
                                    out = torch.sigmoid(model_mlp(tensor_emb)).numpy()
                                    preds = out.flatten().tolist()
                        
                        # 4. Display Results
                        st.success("Inference Complete!")
                        
                        # Create a clean results table
                        results_dict = {disease: float(prob) for disease, prob in zip(CLASSES, preds)}
                        sorted_results = sorted(results_dict.items(), key=lambda item: item[1], reverse=True)
                        
                        # Use Streamlit metrics for the top 3 findings
                        st.subheader("Top Findings")
                        c1, c2, c3 = st.columns(3)
                        c1.metric(sorted_results[0][0], f"{sorted_results[0][1]*100:.1f}%")
                        c2.metric(sorted_results[1][0], f"{sorted_results[1][1]*100:.1f}%")
                        c3.metric(sorted_results[2][0], f"{sorted_results[2][1]*100:.1f}%")
                        
                        with st.expander("View All 14 Categories"):
                            st.table({"Disease": [x[0] for x in sorted_results], "Probability": [f"{x[1]*100:.2f}%" for x in sorted_results]})

                    except Exception as e:
                        st.error(f"Inference pipeline failed: {str(e)}")