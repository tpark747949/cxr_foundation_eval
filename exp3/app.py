import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

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

MODELS = ["MedSigLIP", "CXR_Foundation", "BioViL-T"]
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
    subset = subset.dropna(subset=[score_col])
    
    if unmentioned_policy == "Map -2 / NaN to 0 (Presume Absent)":
        subset[lbl_col] = subset[lbl_col].fillna(0.0)
        subset[lbl_col] = subset[lbl_col].replace(-2.0, 0.0)
    elif unmentioned_policy == "Drop -2 / NaN":
        subset = subset.dropna(subset=[lbl_col])
        subset = subset[subset[lbl_col] != -2.0]
        
    if u_policy == "U-Ones (Map -1 to 1)":
        subset[lbl_col] = subset[lbl_col].replace(-1.0, 1.0)
    elif u_policy == "U-Zeros (Map -1 to 0)":
        subset[lbl_col] = subset[lbl_col].replace(-1.0, 0.0)
    elif u_policy == "U-Ignore (Drop -1)":
        subset = subset[subset[lbl_col] != -1.0]
        
    subset = subset[subset[lbl_col].isin([0.0, 1.0])]
    return subset[lbl_col].values, subset[score_col].values


# ==========================================
# NULL REGION HELPERS
# ==========================================
def add_roc_null_region(fig, n_pos, n_neg):
    total = n_pos + n_neg
    if total == 0: return
    
    fpr_space = np.linspace(0, 1, 200)
    
    # Variance of random guessing
    se = np.sqrt(fpr_space * (1 - fpr_space) * ((1 / n_pos) + (1 / n_neg)))
    upper = np.clip(fpr_space + 1.96 * se, 0, 1)
    lower = np.clip(fpr_space - 1.96 * se, 0, 1)
    
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray', width=1.5), name='Random Baseline', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=fpr_space, y=lower, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=fpr_space, y=upper, mode='lines', fill='tonexty', fillcolor='rgba(255, 99, 71, 0.15)', line=dict(width=0), name='Null Region (95% CI)', hoverinfo='skip'))

def add_pr_null_region(fig, n_pos, n_neg):
    total = n_pos + n_neg
    if total == 0: return
    p_base = n_pos / total
    
    recall_space = np.linspace(0.01, 1.0, 200)
    se = np.sqrt(p_base * (1 - p_base) / (recall_space * total))
    upper = np.clip(p_base + 1.96 * se, 0, 1)
    lower = np.clip(p_base - 1.96 * se, 0, 1)
    
    recall_space = np.insert(recall_space, 0, 0)
    upper = np.insert(upper, 0, 1)
    lower = np.insert(lower, 0, 0)
    
    fig.add_trace(go.Scatter(x=[0, 1], y=[p_base, p_base], mode='lines', line=dict(dash='dash', color='gray', width=1.5), name=f'Random Baseline ({p_base:.3f})', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=recall_space, y=lower, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=recall_space, y=upper, mode='lines', fill='tonexty', fillcolor='rgba(255, 99, 71, 0.15)', line=dict(width=0), name='Null Region (95% CI)', hoverinfo='skip'))


# ==========================================
# MAIN PLOTTING LOGIC
# ==========================================
fig_roc = go.Figure()
fig_pr = go.Figure()

if view_mode == "Macro Average Overview":
    st.subheader(f"Macro Average Curves ({label_type.capitalize()} Labels)")
    st.caption(f"Settings: **Uncertainty (-1)** = {u_policy} | **Unmentioned (-2/NaN)** = {unmentioned_policy}")
    
    common_x = np.linspace(0, 1, 200)
    total_pos, total_neg = 0, 0
    
    for model in selected_models:
        tprs, aucs = [], []
        prs, aps = [], []
        
        for disease in DISEASES:
            y_true, y_scores = get_clean_data(df, disease, model, label_type, u_policy, unmentioned_policy)
            
            if len(np.unique(y_true)) > 1:
                # ROC Calculations
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                roc_auc = auc(fpr, tpr)
                interp_tpr = np.interp(common_x, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(roc_auc)
                
                # PR Calculations
                p, r, _ = precision_recall_curve(y_true, y_scores)
                ap = average_precision_score(y_true, y_scores)
                sort_idx = np.argsort(r)
                interp_p = np.interp(common_x, r[sort_idx], p[sort_idx])
                prs.append(interp_p)
                aps.append(ap)
                
                if model == selected_models[0]:
                    total_pos += np.sum(y_true == 1)
                    total_neg += np.sum(y_true == 0)
        
        if tprs and prs:
            # Aggregate ROC
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = np.mean(aucs)
            fig_roc.add_trace(go.Scatter(x=common_x, y=mean_tpr, mode='lines', name=f'{model} (AUC = {mean_auc:.3f})', line=dict(color=COLOR_MAP.get(model, 'black'), width=3), hovertemplate="FPR: %{x:.2f}<br>TPR: %{y:.2f}"))
            
            # Aggregate PR
            mean_pr = np.mean(prs, axis=0)
            mean_ap = np.mean(aps)
            fig_pr.add_trace(go.Scatter(x=common_x, y=mean_pr, mode='lines', name=f'{model} (AUPRC = {mean_ap:.3f})', line=dict(color=COLOR_MAP.get(model, 'black'), width=3), hovertemplate="Recall: %{x:.2f}<br>Precision: %{y:.2f}"))

    if total_pos > 0:
        add_roc_null_region(fig_roc, total_pos, total_neg)
        add_pr_null_region(fig_pr, total_pos, total_neg)
        title_suffix = f" (Aggregate N+ = {total_pos:,}, N- = {total_neg:,})"

else:
    disease = view_mode
    st.subheader(f"{disease} Curves ({label_type.capitalize()} Labels)")
    st.caption(f"Settings: **Uncertainty (-1)** = {u_policy} | **Unmentioned (-2/NaN)** = {unmentioned_policy}")
    
    n_pos, n_neg = 0, 0
    
    for model in selected_models:
        y_true, y_scores = get_clean_data(df, disease, model, label_type, u_policy, unmentioned_policy)
        
        if len(np.unique(y_true)) < 2:
            st.warning(f"Not enough class diversity for {model}. Skipping.")
            continue
            
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{model} (AUC = {roc_auc:.3f})', line=dict(color=COLOR_MAP.get(model, 'black'), width=3), hovertemplate="FPR: %{x:.2f}<br>TPR: %{y:.2f}"))
        
        # PR
        p, r, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        fig_pr.add_trace(go.Scatter(x=r, y=p, mode='lines', name=f'{model} (AUPRC = {ap:.3f})', line=dict(color=COLOR_MAP.get(model, 'black'), width=3), hovertemplate="Recall: %{x:.2f}<br>Precision: %{y:.2f}"))

    if n_pos > 0:
        add_roc_null_region(fig_roc, n_pos, n_neg)
        add_pr_null_region(fig_pr, n_pos, n_neg)
        title_suffix = f" (N+ = {n_pos:,}, N- = {n_neg:,})"

# ==========================================
# RENDER LAYOUT
# ==========================================
layout_kwargs = dict(
    height=600, hovermode="x unified", template="plotly_white",
    legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98, bgcolor="rgba(255,255,255,0.8)", bordercolor="black", borderwidth=1)
)

fig_roc.update_layout(title=f"Receiver Operating Characteristic{title_suffix}", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", xaxis=dict(range=[0, 1.01]), yaxis=dict(range=[0, 1.05]), **layout_kwargs)

# Move PR legend to top right to avoid covering the typical curve shape
pr_layout_kwargs = layout_kwargs.copy()
pr_layout_kwargs["legend"] = dict(yanchor="top", y=0.98, xanchor="right", x=0.98, bgcolor="rgba(255,255,255,0.8)", bordercolor="black", borderwidth=1)
fig_pr.update_layout(title=f"Precision-Recall Curve{title_suffix}", xaxis_title="Recall", yaxis_title="Precision", xaxis=dict(range=[0, 1.01]), yaxis=dict(range=[0, 1.05]), **pr_layout_kwargs)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_roc, use_container_width=True)
with col2:
    st.plotly_chart(fig_pr, use_container_width=True)

if st.checkbox("Show Raw DataFrame"):
    st.dataframe(df)