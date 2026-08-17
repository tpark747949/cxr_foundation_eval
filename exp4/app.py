import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(page_title="CXR Retrieval Evaluation", layout="wide")
st.title("CXR Image-Report Retrieval Dashboard")

# --- Data Loading ---
@st.cache_data
def load_data():
    df = pd.read_csv("evaluation_results_stratified.csv")
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Could not find `evaluation_results_stratified.csv`. Make sure you've run the evaluation script.")
    st.stop()

# --- Sidebar Controls ---
st.sidebar.header("Navigation")
view_mode = st.sidebar.radio("Select View", ["Overall Performance", "Disease Explorer"])

st.sidebar.header("Global Filters")
selected_task = st.sidebar.radio("Retrieval Task", ["Image-to-Report (I2R)", "Report-to-Image (R2I)"])
task_prefix = "I2R" if selected_task == "Image-to-Report (I2R)" else "R2I"

selected_section = st.sidebar.selectbox(
    "Report Section", 
    options=["softmax", "findings", "impression"],
    format_func=lambda x: x.capitalize()
)

# --- Overall Performance View ---
if view_mode == "Overall Performance":
    st.header(f"Overall {selected_task} Performance ({selected_section.capitalize()})")
    
    # Filter for Overall metrics
    df_overall = df[(df["Disease"] == "Overall") & (df["Section"] == selected_section)]
    
    if df_overall.empty:
        st.warning("No overall data found for these filters.")
    else:
        # Prepare data for plotting
        metrics = ["R@1", "R@5", "R@10", "MRR"]
        plot_data = []
        for metric in metrics:
            col_name = f"{task_prefix}_{metric}"
            for _, row in df_overall.iterrows():
                plot_data.append({
                    "Model": row["Model"],
                    "Metric": metric,
                    "Score": row[col_name]
                })
        
        df_plot = pd.DataFrame(plot_data)
        
        # Plot Grouped Bar Chart
        fig = px.bar(
            df_plot, 
            x="Metric", 
            y="Score", 
            color="Model", 
            barmode="group",
            text_auto='.3f',
            title=f"Overall Metrics Comparison",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig.update_layout(yaxis_title="Score", yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        # Show raw data
        st.subheader("Raw Results")
        st.dataframe(df_overall[["Model", f"{task_prefix}_R@1", f"{task_prefix}_R@5", f"{task_prefix}_R@10", f"{task_prefix}_MRR", f"Query_Count_{task_prefix}"]], use_container_width=True)

# --- Disease Explorer View ---
elif view_mode == "Disease Explorer":
    st.header("Disease Stratification Explorer")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_metric = st.selectbox("Select Metric to Visualize", ["R@1", "R@5", "R@10", "MRR"], index=2)
    with col2:
        label_mapping = {
            1: "1 (Positive)", 
            0: "0 (Negative)", 
            -1: "-1 (Uncertain)", 
            -2: "-2 (No Mention)"
        }
        selected_label_val = st.selectbox("CheXpert Label", options=[1, 0, -1, -2], format_func=lambda x: label_mapping[x])
    
    # Filter for specific disease labels (exclude "Overall")
    df_disease = df[
        (df["Disease"] != "Overall") & 
        (df["Section"] == selected_section) & 
        (df["Label"] == str(selected_label_val))  # Ensure string matching if CSV saved as string
    ]
    
    # Sometimes Pandas reads it as int, sometimes string, let's cast to be safe
    df["Label"] = df["Label"].astype(str)
    df_disease = df[
        (df["Disease"] != "Overall") & 
        (df["Section"] == selected_section) & 
        (df["Label"] == str(selected_label_val))
    ]
    
    metric_col = f"{task_prefix}_{selected_metric}"
    
    if df_disease.empty:
        st.warning("No data found for this combination.")
    else:
        # Create a pivot table for the Heatmap: Diseases (Rows) x Models (Columns)
        pivot_df = df_disease.pivot(index="Disease", columns="Model", values=metric_col)
        
        # Plot Heatmap
        fig = px.imshow(
            pivot_df, 
            text_auto=".3f", 
            aspect="auto",
            color_continuous_scale="Viridis",
            title=f"{selected_metric} for {task_prefix} across Diseases (Label: {label_mapping[selected_label_val]})"
        )
        fig.update_layout(xaxis_title="Model", yaxis_title="Disease")
        st.plotly_chart(fig, use_container_width=True)
        
        # Support/Query Count check (to see if a high score is just due to N=1)
        st.subheader("Query Counts (N)")
        st.caption("Always check query counts. High retrieval scores on very low sample sizes can be misleading.")
        count_col = f"Query_Count_{task_prefix}"
        count_pivot = df_disease.pivot(index="Disease", columns="Model", values=count_col)
        st.dataframe(count_pivot, use_container_width=True)