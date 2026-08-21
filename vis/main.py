import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide", page_title="CXR Foundation Eval")

@st.cache_data
def load_data():
    return pd.read_csv("master_metrics.csv")

df = load_data()

st.title("CXR Foundation Models: Deep Exploration")

# --- Global Sidebar Filters ---
st.sidebar.header("Metric Selection")
selected_metric = st.sidebar.radio("Evaluate Performance Using:", ["AUC", "AUPRC"])

st.sidebar.header("Global Filters")
selected_models = st.sidebar.multiselect("Models", df["Model"].unique(), default=df["Model"].unique())
selected_heads = st.sidebar.multiselect("Classifier Heads", df["Head"].unique(), default=["LR", "XGB"])
selected_vars = st.sidebar.multiselect("Embeddings (Var)", df["Var"].unique(), default=["raw"])
selected_diseases = st.sidebar.multiselect("Diseases", df["Disease"].unique(), default=df["Disease"].unique())

filtered_df = df[
    (df["Model"].isin(selected_models)) & 
    (df["Head"].isin(selected_heads)) & 
    (df["Var"].isin(selected_vars)) & 
    (df["Disease"].isin(selected_diseases))
]

# --- Tabbed Interface ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Data Scarcity (Exp 2)", 
    "Head Architecture (Exp 1)", 
    "Per-Disease Heatmap", 
    "Raw Data Explorer"
])

with tab1:
    st.subheader(f"Label Efficiency: How much data do these models really need? ({selected_metric})")
    st.markdown("Watch how complex heads (MLPs) collapse at 1% data, while simple heads (LR) survive.")
    
    eff_df = filtered_df[filtered_df["Label"].isin(["1pct", "5pct", "10pct", "CheXpert"])].copy()
    pct_map = {"1pct": 1, "5pct": 5, "10pct": 10, "CheXpert": 100}
    eff_df["Data_Pct"] = eff_df["Label"].map(pct_map)
    
    # Toggle to separate or average heads
    separate_heads = st.checkbox("Plot separate lines for each Classifier Head", value=True)
    
    if separate_heads:
        # Create a combined column for the legend
        eff_df["Model + Head"] = eff_df["Model"] + " (" + eff_df["Head"] + ")"
        mean_eff = eff_df.groupby(["Model + Head", "Model", "Head", "Data_Pct"])[selected_metric].mean().reset_index()
        fig1 = px.line(mean_eff, x="Data_Pct", y=selected_metric, color="Model", line_dash="Head", markers=True, log_x=True,
                       hover_name="Model + Head", labels={"Data_Pct": "Training Data Percentage", selected_metric: f"Macro Mean {selected_metric}"})
    else:
        mean_eff = eff_df.groupby(["Model", "Data_Pct"])[selected_metric].mean().reset_index()
        fig1 = px.line(mean_eff, x="Data_Pct", y=selected_metric, color="Model", markers=True, log_x=True,
                       labels={"Data_Pct": "Training Data Percentage", selected_metric: f"Macro Mean {selected_metric} (Averaged across selected heads)"})
        
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.subheader(f"Linear Separability ({selected_metric}): The Delta Between Simple and Complex Heads")
    st.markdown("A good foundation model should achieve high performance with just Logistic Regression.")
    
    # Isolate 100% data
    head_df = filtered_df[filtered_df["Label"] == "CheXpert"]
    
    fig2 = px.box(head_df, x="Model", y=selected_metric, color="Head", points="all",
                  category_orders={"Head": ["LR", "XGB", "s2", "i2", "s4", "i4"]},
                  labels={selected_metric: f"Distribution of {selected_metric}"})
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader(f"Disease Fingerprints ({selected_metric}): Who is good at what?")
    
    heat_df = filtered_df[filtered_df["Label"] == "CheXpert"]
    # Group by Model and Disease, averaging whatever heads/vars are left in the global filter
    heat_pivot = heat_df.groupby(["Model", "Disease"])[selected_metric].mean().unstack()
    
    fig3 = px.imshow(heat_pivot, text_auto=".3f", aspect="auto", color_continuous_scale="Viridis",
                     labels={"color": f"Mean {selected_metric}"})
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.subheader("Raw Data View")
    st.dataframe(filtered_df.sort_values(by=selected_metric, ascending=False), use_container_width=True)