import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Ensure page width allows for large charts
st.set_page_config(page_title="Embedding EDA", layout="wide")
st.title("Feature Distribution Explorer")

@st.cache_data
def generate_mock_data():
    # In reality, load your LanceDB embeddings here. 
    # Using mock data for demonstration purposes.
    raw_data = np.random.lognormal(mean=0.0, sigma=1.0, size=(10000, 50))
    scaler = StandardScaler()
    std_data = scaler.fit_transform(raw_data)
    return raw_data, std_data

raw_data, std_data = generate_mock_data()
num_features = raw_data.shape[1]

st.sidebar.header("Controls")
# Allow user to pick up to 10 features to compare at once
selected_features = st.sidebar.multiselect(
    "Select Features to Overlay (Max 10 for performance):",
    options=[f"Feature_{i}" for i in range(num_features)],
    default=["Feature_0", "Feature_1", "Feature_2"],
    max_selections=10
)

view_mode = st.sidebar.radio("Data Transformation", ["Raw", "Standardized", "Both"])

st.markdown("""
**Pro-tip for interactions:**
* **Hover** over the chart for density values.
* **Single-click** a feature in the legend to toggle its visibility.
* **Double-click** a feature in the legend to isolate it (hides all others). Double-click again to reset.
""")

if not selected_features:
    st.warning("Please select at least one feature from the sidebar.")
else:
    fig = go.Figure()

    for feat_name in selected_features:
        idx = int(feat_name.split("_")[1])
        
        if view_mode in ["Raw", "Both"]:
            fig.add_trace(go.Histogram(
                x=raw_data[:, idx],
                name=f"{feat_name} (Raw)",
                opacity=0.6 if view_mode == "Both" else 0.75,
                histnorm='probability density'
            ))
            
        if view_mode in ["Standardized", "Both"]:
            fig.add_trace(go.Histogram(
                x=std_data[:, idx],
                name=f"{feat_name} (Std)",
                opacity=0.6 if view_mode == "Both" else 0.75,
                histnorm='probability density'
            ))

    # barmode='overlay' allows the distributions to sit transparently on top of each other
    fig.update_layout(
        barmode='overlay',
        title="Probability Density of Selected Features",
        xaxis_title="Feature Value",
        yaxis_title="Density",
        legend=dict(title="Click to toggle, Double-click to isolate")
    )
    
    st.plotly_chart(fig, use_container_width=True)