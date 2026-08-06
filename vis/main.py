# dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="CXR Foundation Evaluation Dashboard", layout="wide")

st.title("CXR Foundation Models: Evaluation & Sensitivity Dashboard")

@st.cache_data
def load_data():
    df1 = pd.read_csv("exp1_master_results.csv")
    df2 = pd.read_csv("exp2_master_results.csv")
    return df1, df2

try:
    df_exp1, df_exp2 = load_data()
except Exception as e:
    st.error("Please run `python generate_master_csvs.py` first to create the summary CSV files.")
    st.stop()

tab1, tab2, tab3 = st.tabs([
    "Exp 1: Sensitivity & Labeler Comparison", 
    "Exp 2: Label Efficiency Curves", 
    "Disease Granularity Matrix"
])

# --- TAB 1: EXP 1 ---
with tab1:
    st.header("Experiment 1: CheXpert vs NegBio Sensitivity")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_artifact = st.selectbox(
            "Select Architecture Group", 
            options=df_exp1["Artifact_Group"].unique()
        )
    with col2:
        agg_metric = st.selectbox("Aggregation Level", ["Macro Mean", "Per-Disease"])

    filtered_exp1 = df_exp1[df_exp1["Artifact_Group"] == selected_artifact]
    
    if agg_metric == "Macro Mean":
        macro_df = filtered_exp1.groupby(["Model", "Label_Source"])["AUC"].mean().reset_index()
        fig = px.bar(
            macro_df, 
            x="Model", 
            y="AUC", 
            color="Label_Source", 
            barmode="group",
            text_auto=".3f",
            title=f"Macro AUROC: CheXpert vs NegBio ({selected_artifact})"
        )
        fig.update_layout(yaxis_range=[0.5, 1.0])
        st.plotly_chart(fig, use_container_width=True)
    else:
        disease_df = filtered_exp1.groupby(["Model", "Label_Source", "Disease"])["AUC"].mean().reset_index()
        selected_disease = st.selectbox("Select Pathology", options=df_exp1["Disease"].unique())
        sub_d = disease_df[disease_df["Disease"] == selected_disease]
        fig = px.bar(
            sub_d, 
            x="Model", 
            y="AUC", 
            color="Label_Source", 
            barmode="group",
            text_auto=".3f",
            title=f"{selected_disease} AUROC: CheXpert vs NegBio"
        )
        fig.update_layout(yaxis_range=[0.5, 1.0])
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: EXP 2 ---
with tab2:
    st.header("Experiment 2: Label Efficiency (1% vs 5% vs 10%)")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_classifier = st.selectbox("Select Classifier", options=df_exp2["Classifier"].unique())
    
    filtered_exp2 = df_exp2[df_exp2["Classifier"] == selected_classifier]
    
    # Map percentage strings to numeric values for clean plotting
    pct_map = {"1p": 1, "5p": 5, "10p": 10}
    filtered_exp2["Pct_Num"] = filtered_exp2["Sample_Pct"].map(pct_map)
    
    curve_df = filtered_exp2.groupby(["Model", "Pct_Num"])["AUC"].mean().reset_index()
    
    fig = px.line(
        curve_df, 
        x="Pct_Num", 
        y="AUC", 
        color="Model", 
        markers=True,
        labels={"Pct_Num": "Training Data Percentage (%)", "AUC": "Macro Mean AUROC"},
        title=f"Label Efficiency Scaling Curves ({selected_classifier})"
    )
    fig.update_layout(xaxis=dict(tickvals=[1, 5, 10]), yaxis_range=[0.5, 1.0])
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: HEATMAP ---
with tab3:
    st.header("Disease Granularity Heatmap")
    
    exp_choice = st.radio("Choose Experiment Source", ["Exp 1", "Exp 2"])
    
    if exp_choice == "Exp 1":
        source_df = df_exp1
        sub_filter = st.selectbox("Artifact Group", options=source_df["Artifact_Group"].unique())
        matrix_df = source_df[source_df["Artifact_Group"] == sub_filter]
        pivot_df = matrix_df.pivot_table(index="Disease", columns="Model", values="AUC", aggfunc="mean")
    else:
        source_df = df_exp2
        sub_pct = st.selectbox("Sample Percentage", options=source_df["Sample_Pct"].unique())
        sub_cls = st.selectbox("Classifier Architecture", options=source_df["Classifier"].unique())
        matrix_df = source_df[(source_df["Sample_Pct"] == sub_pct) & (source_df["Classifier"] == sub_cls)]
        pivot_df = matrix_df.pivot_table(index="Disease", columns="Model", values="AUC", aggfunc="mean")

    fig = px.imshow(
        pivot_df, 
        text_auto=".2f", 
        color_continuous_scale="Viridis",
        aspect="auto",
        title="Disease-Level Performance Matrix"
    )
    st.plotly_chart(fig, use_container_width=True)