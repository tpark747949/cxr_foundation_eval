import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="CXR Retrieval Evaluation", layout="wide")
st.title("CXR Image-Report Retrieval Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv("evaluation_results_stratified.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error("Could not find evaluation_results_stratified.csv.")
    st.stop()

# --- Statistical Null Calculator ---
def compute_null_baseline(N_queries, M_candidates, p_val=0.05):
    """Calculates random guessing baseline and 95% null confidence threshold."""
    if N_queries <= 0 or M_candidates <= 0:
        return {}
    
    r1_exp = min(1.0 / M_candidates, 1.0)
    r5_exp = min(5.0 / M_candidates, 1.0)
    r10_exp = min(10.0 / M_candidates, 1.0)

    # 95th percentile null cutoff via Binomial distribution
    r1_95 = stats.binom.ppf(1 - p_val, N_queries, r1_exp) / N_queries
    r5_95 = stats.binom.ppf(1 - p_val, N_queries, r5_exp) / N_queries
    r10_95 = stats.binom.ppf(1 - p_val, N_queries, r10_exp) / N_queries

    # Harmonic mean approximation for MRR
    harmonic_M = np.sum(1.0 / np.arange(1, M_candidates + 1))
    mrr_exp = harmonic_M / M_candidates
    harmonic2_M = np.sum(1.0 / (np.arange(1, M_candidates + 1) ** 2))
    var_rr = (harmonic2_M / M_candidates) - (mrr_exp ** 2)
    std_mrr = np.sqrt(max(var_rr, 1e-9) / N_queries)
    mrr_95 = mrr_exp + stats.norm.ppf(1 - p_val) * std_mrr

    return {
        "R@1": (r1_exp, r1_95), "R@5": (r5_exp, r5_95),
        "R@10": (r10_exp, r10_95), "MRR": (mrr_exp, mrr_95)
    }

# --- Sidebar Controls ---
st.sidebar.header("Global Filters")
selected_task = st.sidebar.radio("Retrieval Task", ["Image-to-Report (I2R)", "Report-to-Image (R2I)"])
task_prefix = "I2R" if selected_task == "Image-to-Report (I2R)" else "R2I"

selected_section = st.sidebar.selectbox(
    "Report Strategy", 
    options=["centroid_1to1", "softmax", "findings", "impression"],
    format_func=lambda x: "1:1 Geometric Centroid" if x == "centroid_1to1" else x.capitalize()
)

st.header(f"Performance ({selected_section}) vs. Null Baseline")

df_filtered = df[(df["Disease"] == "Overall") & (df["Section"] == selected_section)]

if not df_filtered.empty:
    N_q = int(df_filtered["Query_Count_I2R"].iloc[0])
    M_pool = int(df_filtered["Candidate_Pool_Size"].iloc[0]) if "Candidate_Pool_Size" in df_filtered.columns else 3269
    
    null_bounds = compute_null_baseline(N_q, M_pool)
    
    # Statistical Callout Banner
    st.info(
        f"**Null Baseline (Random Chance) for N={N_q} queries, M={M_pool} pool:**\n"
        f"- **R@1:** Expected = `{null_bounds['R@1'][0]:.4f}` | 95% Cutoff (p=0.05) = `{null_bounds['R@1'][1]:.4f}`\n"
        f"- **R@5:** Expected = `{null_bounds['R@5'][0]:.4f}` | 95% Cutoff (p=0.05) = `{null_bounds['R@5'][1]:.4f}`\n"
        f"- **MRR:** Expected = `{null_bounds['MRR'][0]:.4f}` | 95% Cutoff (p=0.05) = `{null_bounds['MRR'][1]:.4f}`"
    )

    metrics = ["R@1", "R@5", "R@10", "MRR"]
    plot_data = []
    for metric in metrics:
        for _, row in df_filtered.iterrows():
            plot_data.append({
                "Model": row["Model"],
                "Metric": metric,
                "Score": row[f"{task_prefix}_{metric}"],
                "Null_95": null_bounds[metric][1]
            })

    df_plot = pd.DataFrame(plot_data)

    fig = px.bar(
        df_plot, x="Metric", y="Score", color="Model", barmode="group",
        text_auto='.3f', title="Model Metrics vs. Statistical Null Range (Red Lines)",
        color_discrete_sequence=px.colors.qualitative.Prism
    )

    # Add red line indicators for the p=0.05 null threshold
    for m in metrics:
        fig.add_shape(
            type="line", x0=m, x1=m, y0=0, y1=null_bounds[m][1],
            line=dict(color="Red", width=3, dash="dash")
        )

    st.plotly_chart(fig, use_container_width=True)