import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, confusion_matrix

OUTPUT_DIR = "evaluation_artifacts"
MODELS = ["MedSigLIP", "BioViL-T", "EVA-X", "CheXFound", "CheXagent", "CXR_Foundation", "Early_Fusion"]
VARIANTS = ["raw", "l2"]
CHEXPERT_DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged_Cardiomediastinum", "Fracture", "Lung_Lesion",
    "Lung_Opacity", "Pleural_Effusion", "Pneumonia",
    "Pneumothorax", "Pleural_Other", "Support_Devices", "No_Finding"
]

st.set_page_config(page_title="VLM CheXpert Evaluation", layout="wide")
st.title("VLM Embedding Decision Boundary Explorer")

# Sidebar Selections
st.sidebar.header("Configuration")
model_sel = st.sidebar.selectbox("Select Model", MODELS)
variant_sel = st.sidebar.selectbox("Select Variant", VARIANTS)
split_sel = st.sidebar.radio("Select Split", ["val", "test"])
disease_sel = st.sidebar.selectbox("Select Disease", CHEXPERT_DISEASES)
disease_idx = CHEXPERT_DISEASES.index(disease_sel)

# Load Data
@st.cache_data
def load_data(model, variant, split):
    true_path = os.path.join(OUTPUT_DIR, f"y_{split}_true.npy")
    prob_path = os.path.join(OUTPUT_DIR, f"{model}_{variant}_{split}_probs.npy")
    if not os.path.exists(true_path) or not os.path.exists(prob_path):
        return None, None
    return np.load(true_path), np.load(prob_path)

y_true_all, y_prob_all = load_data(model_sel, variant_sel, split_sel)

if y_true_all is None:
    st.error(f"Data not found for {model_sel}_{variant_sel} on {split_sel} split. Did you run the training script?")
else:
    y_true = y_true_all[:, disease_idx]
    y_prob = y_prob_all[:, disease_idx]

    # Metrics
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Probability Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(y_prob[y_true == 0], bins=50, alpha=0.5, label='Negative', color='blue', density=True)
        ax.hist(y_prob[y_true == 1], bins=50, alpha=0.5, label='Positive', color='red', density=True)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Density")
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("ROC Curve")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.legend(loc="lower right")
        st.pyplot(fig2)

    # Interactive Threshold Tuning
    st.markdown("---")
    st.subheader("Fine-Grained Threshold Tuning")
    threshold = st.slider("Decision Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    met_col1, met_col2, met_col3, met_col4 = st.columns(4)
    met_col1.metric("Precision", f"{precision:.3f}")
    met_col2.metric("Recall (Sensitivity)", f"{recall:.3f}")
    met_col3.metric("Specificity", f"{specificity:.3f}")
    met_col4.metric("F1 Score", f"{f1:.3f}")