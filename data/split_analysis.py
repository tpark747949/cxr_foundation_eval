import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import lancedb

# 1. Connect to LanceDB and load data
db = lancedb.connect("../embeddings/MIMIC-CXR-JPG")
table = db.open_table("complete_embeddings_MIMIC-CXR-JPG")
df = table.to_pandas()
print(df.columns.tolist())

pathologies = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", 
    "Lung Opacity", "Pleural Effusion", "Pneumonia", 
    "Pneumothorax", "Pleural Other", "Support Devices", "No Finding"
]

# Expand the CheXpert_labels list column into 14 distinct columns
# Note: Ensure your LanceDB schema returns lists; if it returns strings, you may need ast.literal_eval
labels_df = pd.DataFrame(df['CheXpert_labels'].to_list(), columns=pathologies)
df = pd.concat([df, labels_df], axis=1)

# 3. Set up the visualization grid
sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 2)

# Assuming `df` is already loaded and pathologies are unpacked as 1s, 0s, -1s, -2s.


# ==========================================
# 1. THE LEAKAGE TEST
# ==========================================
print("--- Data Leakage Audit ---")
# Count how many unique splits each patient and study appears in
patient_splits = df.groupby('subject_id')['split'].nunique()
study_splits = df.groupby('study_id')['split'].nunique()

leaked_patients = patient_splits[patient_splits > 1]
leaked_studies = study_splits[study_splits > 1]

print(f"Patients spanning multiple splits: {len(leaked_patients)}")
print(f"Studies spanning multiple splits:  {len(leaked_studies)}")

if len(leaked_patients) == 0:
    print("Result: CLEAN. All images for a given patient remain in their designated split.")
else:
    print("Result: WARNING! Data leakage detected.")

# ==========================================
# VISUALIZATION SETUP
# ==========================================
sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(2, 2)

# --- Plot A: Pathology Distribution Across Splits (Logarithmic Heatmap) ---
ax1 = fig.add_subplot(gs[0, 0])

# Calculate frequency of POSITIVE (1) labels per split
split_path_counts = df[df['split'].isin(['train', 'validate', 'test'])].groupby('split')[pathologies].apply(
    lambda x: (x == 1).sum()
).T

# Order columns logically
if 'validate' in split_path_counts.columns:
    split_path_counts = split_path_counts[['train', 'validate', 'test']]

# Use a logarithmic color scale because "Pleural Effusion" dwarfs "Fracture"
sns.heatmap(split_path_counts, cmap="YlOrRd", norm=LogNorm(), annot=True, fmt='d', ax=ax1, cbar_kws={'label': 'Log Count'})
ax1.set_title('Positive Pathology Mentions by Split (Log Scale)', fontsize=14)
ax1.set_ylabel('Pathology')
ax1.set_xlabel('Dataset Split')

# --- Plot B: The "Over-weight" Patient Curve (Lorenz Curve) ---
ax2 = fig.add_subplot(gs[0, 1])

# Do a few "frequent flyer" patients dominate the test/val sets?
colors = {'train': 'blue', 'validate': 'orange', 'test': 'green'}
for split_name in ['train', 'validate', 'test']:
    split_df = df[df['split'] == split_name]
    if split_df.empty: continue
        
    # Get image counts per patient, sorted descending
    patient_counts = split_df['subject_id'].value_counts().sort_values(ascending=False).values
    
    # Calculate cumulative percentages
    cum_images = np.cumsum(patient_counts) / patient_counts.sum() * 100
    cum_patients = np.arange(1, len(patient_counts) + 1) / len(patient_counts) * 100
    
    ax2.plot(cum_patients, cum_images, label=split_name, color=colors[split_name], lw=2)

# Plot the line of perfect equality (every patient has the same number of images)
ax2.plot([0, 100], [0, 100], 'k--', label='Perfect Equality (1 pt = 1 img)')
ax2.set_title('Patient Representation Inequality (Lorenz Curve)', fontsize=14)
ax2.set_xlabel('% of Patients (ranked by image volume)')
ax2.set_ylabel('Cumulative % of Images in Split')
ax2.legend()

# --- Plot C: Disease Burden / Comorbidity Count ---
ax3 = fig.add_subplot(gs[1, :])

# Calculate how many simultaneous positive labels each image has
df['disease_burden'] = (df[pathologies] == 1).sum(axis=1)

# Boxenplot (enhanced boxplot for large datasets) showing the distribution of comorbidities
sns.boxenplot(data=df, x='split', y='disease_burden', order=['train', 'validate', 'test'], palette='Set2', ax=ax3)
ax3.set_title('Comorbidity Burden: Number of Concurrent Pathologies per Image', fontsize=14)
ax3.set_ylabel('Count of Positive Labels (0-14)')
ax3.set_xlabel('Split')

plt.tight_layout()
output_path = "split_analysis.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Histogram successfully saved to {output_path}")