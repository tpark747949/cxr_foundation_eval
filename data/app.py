import streamlit as st
import pandas as pd
import lancedb
import os
from PIL import Image

# ==========================================
# 0. CONFIGURATION
# ==========================================
DATA_DIR = "./MIMIC-CXR-JPG/2.1.0"
QC_METRICS_CSV = "./qc_metrics.csv"
ZSCORE_CSV = "./embedding_zscores.csv"
LANCEDB_URI = "../embeddings/MIMIC-CXR-JPG"
TABLE_NAME = "complete_embeddings_MIMIC-CXR-JPG" 

st.set_page_config(layout="wide", page_title="MIMIC-CXR QC Dashboard")
st.title("MIMIC-CXR Quality Control Panel")

# ==========================================
# 1. DATA LOADING & CONNECTIONS
# ==========================================
@st.cache_resource
def get_lancedb_table():
    db = lancedb.connect(LANCEDB_URI)
    return db.open_table(TABLE_NAME)

tbl = get_lancedb_table()

@st.cache_data
def load_local_metrics():
    if not os.path.exists(QC_METRICS_CSV):
        st.error(f"Could not find {QC_METRICS_CSV}. Please run profile_dataset.py first!")
        st.stop()
        
    df_metrics = pd.read_csv(QC_METRICS_CSV)
    df_metrics = df_metrics[df_metrics["corrupted"] == False].reset_index(drop=True)
    
    if os.path.exists(ZSCORE_CSV):
        df_zscores = pd.read_csv(ZSCORE_CSV)
        df_master = df_metrics.merge(df_zscores, on="dicom_id", how="left")
    else:
        st.warning(f"Could not find {ZSCORE_CSV}. Magnitude metrics won't be available.")
        df_master = df_metrics
        
    return df_master

df_master = load_local_metrics()

# Identify available model z-scores dynamically
zscore_cols = [c for c in df_master.columns if c.endswith("_mag_zscore") and "consensus" not in c]
available_models = [c.replace("_mag_zscore", "") for c in zscore_cols]

# ==========================================
# 2. SIDEBAR: METADATA FILTERS (LANCEDB)
# ==========================================
st.sidebar.header("LanceDB Metadata Filters")

@st.cache_data
def get_unique_lancedb_values(column_name):
    df = tbl.search().select([column_name]).to_pandas()
    return df[column_name].dropna().unique().tolist()

view_positions = st.sidebar.multiselect("View Position", options=get_unique_lancedb_values("ViewCodeSequence_CodeMeaning"))
procedures = st.sidebar.multiselect("Procedure Description", options=get_unique_lancedb_values("PerformedProcedureStepDescription"))
orientations = st.sidebar.multiselect("Patient Orientation", options=get_unique_lancedb_values("PatientOrientationCodeSequence_CodeMeaning"))

# Build LanceDB SQL string
sql_filters = []
if view_positions:
    sql_filters.append(f"ViewCodeSequence_CodeMeaning IN ({', '.join([f'{chr(39)}{v}{chr(39)}' for v in view_positions])})")
if procedures:
    sql_filters.append(f"PerformedProcedureStepDescription IN ({', '.join([f'{chr(39)}{p}{chr(39)}' for p in procedures])})")
if orientations:
    sql_filters.append(f"PatientOrientationCodeSequence_CodeMeaning IN ({', '.join([f'{chr(39)}{o}{chr(39)}' for o in orientations])})")

final_filter_string = " AND ".join(sql_filters)

# Query LanceDB just for the valid dicom_ids based on metadata filters
if final_filter_string:
    with st.spinner("Applying metadata filters in LanceDB..."):
        # A search without a vector just acts as a fast SQL scan
        valid_records = tbl.search().where(final_filter_string).select(["dicom_id"]).to_pandas()
        valid_dicoms = valid_records["dicom_id"].tolist()
        
    df_filtered = df_master[df_master["dicom_id"].isin(valid_dicoms)]
else:
    df_filtered = df_master

st.sidebar.metric("Images Matching Filters", len(df_filtered))

# ==========================================
# 3. SIDEBAR: DISPLAY SETTINGS
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("Display Settings")
images_per_page = st.sidebar.slider("Images per page", 4, 100, 20, step=4)
cols_per_row = st.sidebar.slider("Columns per row", 2, 8, 4)
page_num = st.sidebar.number_input("Page Number", min_value=1, value=1)

# ==========================================
# 4. GRID RENDERER
# ==========================================
def render_image_grid(dataframe, sort_col, ascending, title):
    st.subheader(title)
    
    sorted_df = dataframe.sort_values(by=sort_col, ascending=ascending).reset_index(drop=True)
    start_idx = (page_num - 1) * images_per_page
    end_idx = start_idx + images_per_page
    window_df = sorted_df.iloc[start_idx:end_idx]
    
    if window_df.empty:
        st.info("No more images to display on this page.")
        return

    for i in range(0, len(window_df), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(window_df):
                row = window_df.iloc[i + j]
                img_path = os.path.join(DATA_DIR, row["path"])
                
                with col:
                    try:
                        img = Image.open(img_path)
                        # Updated to Streamlit's new width parameter specification
                        st.image(img, width="stretch")
                        st.caption(f"**ID:** `{row['dicom_id'][:8]}...`")
                        st.caption(f"**{sort_col}:** `{row[sort_col]:.4f}`")
                    except Exception:
                        st.error("Error loading image")

# ==========================================
# 5. UI TABS
# ==========================================
tabs = st.tabs([
    "Model Consensus Outliers",
    "Collimation Errors", 
    "Blank Scans", 
    "Over/Under Exposure", 
    "Inversions"
])

# --- TAB 1: Embedding Z-Scores ---
with tabs[0]:
    st.write("Images with the most mathematically abnormal embedding magnitudes ($|Z|$-score).")
    
    score_choices = ["Consensus: Broke All Models (Minimum |Z|)", "Consensus: Average Abnormality (Mean |Z|)"] + available_models
    selected_score = st.selectbox("Select Outlier Scoring Metric", score_choices)
    
    if selected_score == "Consensus: Broke All Models (Minimum |Z|)":
        target_col = "consensus_min_zscore"
        title_text = "Universal Outliers (Broke every model simultaneously)"
    elif selected_score == "Consensus: Average Abnormality (Mean |Z|)":
        target_col = "consensus_mean_zscore"
        title_text = "Highest Average Outlier Score across all architectures"
    else:
        target_col = f"{selected_score}_mag_zscore"
        title_text = f"Highest $|Z|$-Scores for {selected_score}"
        
    if target_col in df_filtered.columns:
        render_image_grid(df_filtered, target_col, ascending=False, title=title_text)

# --- TAB 2: Half-Black / Collimation ---
with tabs[1]:
    render_image_grid(df_filtered, "min_half_mean", ascending=True, title="Lowest Half-Mean")

# --- TAB 3: Low Variance / Blank Scans ---
with tabs[2]:
    render_image_grid(df_filtered, "overall_std", ascending=True, title="Lowest Variance")

# --- TAB 4: Extreme Exposure ---
with tabs[3]:
    exposure_type = st.radio("Select Exposure Type", ["Underexposed (Too Dark)", "Overexposed (Too Bright)"], horizontal=True)
    if "Under" in exposure_type:
        render_image_grid(df_filtered, "overall_mean", ascending=True, title="Underexposed Outliers")
    else:
        render_image_grid(df_filtered, "overall_mean", ascending=False, title="Overexposed Outliers")

# --- TAB 5: Photometric Inversion ---
with tabs[4]:
    render_image_grid(df_filtered, "corner_mean", ascending=False, title="Brightest Corners")