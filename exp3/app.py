import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="VLM Zero-Shot Evaluator", layout="wide")
st.title("Zero-Shot Contrastive VLM Evaluation")

DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", 
    "Lung Opacity", "Pleural Effusion", "Pleural Other", 
    "Pneumonia", "Pneumothorax", "Support Devices", "No Finding"
]

MODELS = ["MedSigLIP", "CXR_Foundation", "BioViL-T", "CheXagent"]
COLOR_MAP = {"MedSigLIP": "#1f77b4", "CXR_Foundation": "#ff7f0e", "BioViL-T": "#2ca02c", "CheXagent": "#d62728"}

# ==========================================
# DATA LOADING
# ==========================================
@st.cache_data
def load_data():
    return pd.read_parquet("zeroshot_evaluation_results.parquet")

try:
    df = load_data()
except FileNotFoundError:
    st.error("Could not find 'zeroshot_evaluation_results.parquet'. Please run the evaluation script first.")
    st.stop()

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("1. Data & Labels")
label_type = st.sidebar.radio("Ground Truth Source:", ["chexpert", "negbio"])

st.sidebar.header("2. Label Policy Options")
u_policy = st.sidebar.selectbox(
    "Uncertainty Policy (-1):",
    ["U-Ones (Map -1 to 1)", "U-Zeros (Map -1 to 0)", "U-Ignore (Drop -1)"]
)

unmentioned_policy = st.sidebar.selectbox(
    "Unmentioned Policy (-2 / NaN):",
    ["Map -2 / NaN to 0 (Presume Absent)", "Drop -2 / NaN"]
)

st.sidebar.header("3. Display Options")
view_mode = st.sidebar.selectbox("Display Mode:", ["Macro Average Overview"] + DISEASES)
selected_models = st.sidebar.multiselect("Models to Compare:", MODELS, default=MODELS)

# ==========================================
# LABEL PROCESSING FUNCTION
# ==========================================
def get_clean_data(df, disease, model, label_type, u_policy, unmentioned_policy):
    struct_key = disease.replace(" ", "_")
    lbl_col = f"label_{label_type}_{struct_key}"
    score_col = f"score_{model}_{struct_key}"
    
    subset = df[[lbl_col, score_col]].copy()
    
    # Drop rows where prediction score is missing
    subset = subset.dropna(subset=[score_col])
    
    # 1. Handle Unmentioned (-2.0 or NaN)
    if unmentioned_policy == "Map -2 / NaN to 0 (Presume Absent)":
        subset[lbl_col] = subset[lbl_col].fillna(0.0)
        subset[lbl_col] = subset[lbl_col].replace(-2.0, 0.0)
    elif unmentioned_policy == "Drop -2 / NaN":
        subset = subset.dropna(subset=[lbl_col])
        subset = subset[subset[lbl_col] != -2.0]
        
    # 2. Handle Uncertainty (-1.0)
    if u_policy == "U-Ones (Map -1 to 1)":
        subset[lbl_col] = subset[lbl_col].replace(-1.0, 1.0)
    elif u_policy == "U-Zeros (Map -1 to 0)":
        subset[lbl_col] = subset[lbl_col].replace(-1.0, 0.0)
    elif u_policy == "U-Ignore (Drop -1)":
        subset = subset[subset[lbl_col] != -1.0]
        
    # Strictly isolate binary targets (0.0 or 1.0)
    subset = subset[subset[lbl_col].isin([0.0, 1.0])]
    
    return subset[lbl_col].values, subset[score_col].values

# Helper function to plot statistical null region
def plot_null_region(ax, n_pos, n_neg):
    fpr_space = np.linspace(0, 1, 200)
    
    # Variance under random guessing
    variance_null = fpr_space * (1 - fpr_space) * ((1 / n_pos) + (1 / n_neg))
    upper_bound = fpr_space + 1.96 * np.sqrt(variance_null)
    upper_bound = np.clip(upper_bound, 0, 1)
    
    ax.plot([0, 1], [0, 1], linestyle='--', color='darkgray', zorder=1)
    ax.fill_between(
        fpr_space, 0, upper_bound, 
        color='lightcoral', alpha=0.15, zorder=0,
        label=f'Null Region (p > 0.05)'
    )

# ==========================================
# MAIN PLOTTING LOGIC
# ==========================================
fig, ax = plt.subplots(figsize=(10, 8))
common_fpr = np.linspace(0, 1, 200)

if view_mode == "Macro Average Overview":
    st.subheader(f"Macro Average ROC across all 14 Diseases ({label_type.capitalize()} Labels)")
    st.caption(f"Settings: **Uncertainty (-1)** = {u_policy} | **Unmentioned (-2/NaN)** = {unmentioned_policy}")
    
    total_pos = 0
    total_neg = 0
    
    for model in selected_models:
        tprs = []
        aucs = []
        
        for disease in DISEASES:
            y_true, y_scores = get_clean_data(df, disease, model, label_type, u_policy, unmentioned_policy)
            
            if len(np.unique(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                roc_auc = auc(fpr, tpr)
                
                interp_tpr = np.interp(common_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(roc_auc)
                
                # Count aggregate sample sizes across dataset (using first model iteration)
                if model == selected_models[0]:
                    total_pos += np.sum(y_true == 1)
                    total_neg += np.sum(y_true == 0)
        
        if tprs:
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(common_fpr, mean_tpr)
            
            ax.plot(common_fpr, mean_tpr, label=f'{model} (Macro AUC = {mean_auc:.3f})', 
                    color=COLOR_MAP[model], linewidth=2.5)

    if total_pos > 0 and total_neg > 0:
        plot_null_region(ax, total_pos, total_neg)
        ax.set_title(f"Macro Average ROC (Aggregate N+ = {total_pos:,}, N- = {total_neg:,})", fontsize=14)

else:
    # Individual Disease View
    disease = view_mode
    st.subheader(f"ROC Curve for {disease} ({label_type.capitalize()} Labels)")
    st.caption(f"Settings: **Uncertainty (-1)** = {u_policy} | **Unmentioned (-2/NaN)** = {unmentioned_policy}")
    
    n_pos, n_neg = 0, 0
    
    for model in selected_models:
        y_true, y_scores = get_clean_data(df, disease, model, label_type, u_policy, unmentioned_policy)
        
        if len(np.unique(y_true)) < 2:
            st.warning(f"Not enough class diversity for {model} on {disease} with selected label settings. Skipping.")
            continue
            
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, label=f'{model} (AUC = {roc_auc:.3f})', 
                color=COLOR_MAP[model], linewidth=2)

    if n_pos > 0 and n_neg > 0:
        plot_null_region(ax, n_pos, n_neg)
        ax.set_title(f"{disease} (N+ = {n_pos:,}, N- = {n_neg:,})", fontsize=14)

# Plot Formatting
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
ax.grid(True, linestyle='--', alpha=0.5)

st.pyplot(fig)

if st.checkbox("Show Raw DataFrame"):
    st.dataframe(df)