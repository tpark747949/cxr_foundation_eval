import matplotlib
# Force headless backend before importing pyplot
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import pandas as pd
import numpy as np
import lancedb

# 1. Connect to LanceDB
print("Connecting to LanceDB...")
db = lancedb.connect("../embeddings/MIMIC-CXR-JPG")
table = db.open_table("complete_embeddings_MIMIC-CXR-JPG")
df = table.to_pandas()
print(f"Loaded {len(df)} records.")

# 2. Define the exact order as stored in your PyArrow struct
pathologies = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
    "Fracture", "Lung Lesion", "Lung Opacity", "Pleural Effusion", "Pneumonia",
    "Pneumothorax", "Pleural Other", "Support Devices", "No Finding"
]

# 3. Unpack the PyArrow Struct
# Pandas converts PyArrow structs into a Series of dictionaries.
# Wrapping .tolist() in a DataFrame instantly expands dict keys into columns.
print("Unpacking label structs...")
labels_df = pd.DataFrame(df['CheXpert_labels'].tolist())

# Force column names to match our list exactly, in case your `clean_label_cols` 
# swapped spaces for underscores during the initial database ingestion.
labels_df.columns = pathologies

# Merge back into the main dataframe
df = pd.concat([df.drop(columns=['CheXpert_labels', 'NegBio_labels'], errors='ignore'), labels_df], axis=1)


# ==========================================
# MATRIX PANEL 1: Counts by ViewCodeSequence
# ==========================================
print("Generating ViewCodeSequence Heatmap...")
plt.figure(figsize=(12, 8))
view_col = 'ViewCodeSequence_CodeMeaning'

# Defensively clean DICOM text and filter out NaNs
df[view_col] = df[view_col].astype(str).str.lower().str.strip()
clean_views = df[~df[view_col].isin(['nan', 'none', 'null', ''])]

# Get top 3 legitimate views
top_views = clean_views[view_col].value_counts().nlargest(3).index.tolist()
view_df = df[df[view_col].isin(top_views)]

matrix_data = {}
for view in top_views:
    sub_df = view_df[view_df[view_col] == view]
    matrix_data[view.title()] = [int((sub_df[p] == 1).sum()) for p in pathologies]

view_matrix = pd.DataFrame(matrix_data, index=pathologies)

sns.heatmap(view_matrix, annot=True, fmt="d", cmap="Purples", norm=LogNorm(), cbar_kws={'label': 'Log10 Count'})
plt.title('Positive Pathology Counts by View Code Sequence', fontsize=14, pad=15)
plt.xlabel('View Code Meaning')
plt.ylabel('Pathology')
plt.tight_layout()

heatmap_filename = 'pathology_by_view_matrix.png'
plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved -> {heatmap_filename}")


# ==========================================
# MATRIX PANEL 2: Lorenz Curves
# ==========================================
print("Generating Lorenz Curves Panel...")
fig, axes = plt.subplots(4, 4, figsize=(20, 18), sharex=True, sharey=True)
axes = axes.flatten()
colors = {'train': '#1f77b4', 'validate': '#ff7f0e', 'test': '#2ca02c'}

for i, p in enumerate(pathologies):
    ax = axes[i]
    # Filter for positive mentions
    path_df = df[df[p] == 1]
    
    for split_name in ['train', 'validate', 'test']:
        split_sub = path_df[path_df['split'] == split_name]
        if split_sub.empty:
            continue
            
        patient_counts = split_sub['subject_id'].value_counts().sort_values(ascending=False).values
        cum_cases = np.cumsum(patient_counts) / patient_counts.sum() * 100
        cum_pts = np.arange(1, len(patient_counts) + 1) / len(patient_counts) * 100
        
        cum_pts = np.insert(cum_pts, 0, 0)
        cum_cases = np.insert(cum_cases, 0, 0)
        
        ax.plot(cum_pts, cum_cases, label=split_name, color=colors[split_name], lw=2)
    
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.5, lw=1)
    ax.set_title(f"{p}", fontsize=12, weight='bold')
    
    if i >= 10: ax.set_xlabel('% of Patients')
    if i % 4 == 0: ax.set_ylabel('Cum. % of Positive Imgs')
    if i == 0: ax.legend(loc='upper left')

# Delete unused axes in the 4x4 grid (since we have 14 pathologies)
for j in range(len(pathologies), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle('Patient Representation Inequality (Lorenz Curves) Per Pathology', fontsize=18, weight='bold', y=0.95)
plt.tight_layout(rect=[0, 0, 1, 0.93])

lorenz_filename = 'pathology_lorenz_curves.png'
plt.savefig(lorenz_filename, dpi=300)
plt.close()
print(f"Saved -> {lorenz_filename}")