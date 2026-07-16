import streamlit as st
import pandas as pd
import os
from PIL import Image

# Config
DATA_DIR = "MIMIC-CXR-JPG/2.1.0"
METRIC_CSV = "./mimic_qc.csv"

st.set_page_config(layout="wide", page_title="MIMIC-CXR QC Dashboard")
st.title("🔍 MIMIC-CXR 'Dumb' Tool Quality Control Panel")

@st.cache_data
def load_data():
    if not os.path.exists(METRIC_CSV):
        st.error(f"Could not find {METRIC_CSV}. Please run profile_dataset.py first!")
        st.stop()
    df = pd.read_csv(METRIC_CSV)
    # Exclude failed loads
    df = df[df["corrupted"] == False].reset_index(drop=True)
    return df

df = load_data()

st.sidebar.metric("Total Profiled Images", len(df))
num_display = st.sidebar.slider("Number of outliers to load per category", 5, 50, 10)

# Define Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔲 Half-Black / Collimation Errors", 
    "🌫️ Low Variance / Blank Scans", 
    "☀️ Exposure / Pure Black & White", 
    "🔄 Photometric Inversions (White Corners)"
])

# Helper function to render a scrollable row of images
def render_image_row(dataframe, metric_name, title):
    st.subheader(title)
    cols = st.columns(num_display)
    for idx, (_, row) in enumerate(dataframe.iterrows()):
        col = cols[idx % num_display]
        img_path = os.path.join(DATA_DIR, row["path"])
        
        try:
            img = Image.open(img_path)
            col.image(img, use_container_width=True)
            col.caption(f"**DICOM ID:**\n`{row['dicom_id'][:8]}...`")
            col.caption(f"**{metric_name}:** `{row[metric_name]:.2f}`")
        except Exception as e:
            col.error(f"Error loading image")

# --- TAB 1: Half-Black / Collimation ---
with tab1:
    st.write("Images where at least one half (Top, Bottom, Left, or Right) is completely dark.")
    # Sort ascending by min_half_mean (lowest mean of any half)
    half_black_outliers = df.sort_values(by="min_half_mean", ascending=True).head(num_display)
    render_image_row(half_black_outliers, "min_half_mean", "Top Half-Black Outliers (Lowest Half-Mean)")

# --- TAB 2: Low Variance / Blank Scans ---
with tab2:
    st.write("Images with incredibly low contrast or gray solid blocks (lowest standard deviation).")
    low_var_outliers = df.sort_values(by="overall_std", ascending=True).head(num_display)
    render_image_row(low_var_outliers, "overall_std", "Top Low Variance Outliers (Flat Images)")

# --- TAB 3: Extreme Exposure ---
with tab3:
    col_left, col_right = st.columns(2)
    
    # Mostly Black Scans
    dark_outliers = df.sort_values(by="overall_mean", ascending=True).head(num_display)
    with col_left:
        render_image_row(dark_outliers, "overall_mean", "Underexposed Outliers (Lowest Mean Intensity)")
        
    # Mostly White Scans
    bright_outliers = df.sort_values(by="overall_mean", ascending=False).head(num_display)
    with col_right:
        render_image_row(bright_outliers, "overall_mean", "Overexposed Outliers (Highest Mean Intensity)")

# --- TAB 4: Photometric Inversion ---
with tab4:
    st.write("Normal chest X-rays have black background corners. If corners are bright white, the image is likely inverted.")
    inverted_outliers = df.sort_values(by="corner_mean", ascending=False).head(num_display)
    render_image_row(inverted_outliers, "corner_mean", "Inversion Outliers (Highest Corner Brightness)")