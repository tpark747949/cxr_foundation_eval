import lancedb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm


# 1. Connect to LanceDB and load data
db = lancedb.connect("../embeddings/MIMIC-CXR-JPG")
table = db.open_table("complete_embeddings_MIMIC-CXR-JPG")
df = table.to_pandas()
print(df.columns.tolist())

# 2. Unpack the pathology labels
# The list provided is mostly alphabetical, with No Finding at the end (standard CheXpert)
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

# --- Plot A: Pathology Distribution & Uncertainty ---
ax1 = fig.add_subplot(gs[0, :])

# Count frequencies of 1 (Positive), 0 (Negative), and -1 (Uncertain). Ignoring -2 (No Mention).
label_counts = []
for p in pathologies:
    counts = df[p].value_counts()
    label_counts.append({
        'Pathology': p,
        'Positive (1)': counts.get(1, 0),
        'Uncertain (-1)': counts.get(-1, 0),
        'Negative (0)': counts.get(0, 0)
    })

counts_df = pd.DataFrame(label_counts).set_index('Pathology')
counts_df.sort_values('Positive (1)', ascending=False).plot(
    kind='bar', stacked=True, ax=ax1, 
    color=['#d62728', '#ff7f0e', '#1f77b4'], alpha=0.8
)
ax1.set_title('Pathology Label Mentions (Positive, Uncertain, Negative)', fontsize=14)
ax1.set_ylabel('Number of Reports')
ax1.set_xlabel('')
ax1.tick_params(axis='x', rotation=45)

# --- Plot B: Clinical Acuity (View vs Orientation) ---
ax2 = fig.add_subplot(gs[1, 0])

# Filter out nulls and rare views for a clean heatmap
view_orientation = df.dropna(subset=['ViewPosition', 'PatientOrientationCodeSequence_CodeMeaning'])
view_counts = pd.crosstab(
    view_orientation['ViewPosition'], 
    view_orientation['PatientOrientationCodeSequence_CodeMeaning']
)

sns.heatmap(view_counts, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False)
ax2.set_title('Image Acquisition: View vs. Orientation', fontsize=14)
ax2.set_ylabel('View Position (AP / PA / LATERAL)')
ax2.set_xlabel('Patient Orientation')

# --- Plot C: Hardware & Image Cropping (Dimensions) ---
ax3 = fig.add_subplot(gs[1, 1])

# Hexbin plot of Rows vs Columns to show density of image dimensions
# High density areas indicate standard detector sizes; spread indicates manual cropping
# Split the string on 'x' and assign to new columns
df['Columns'] = df['image_size'].str.get('Columns')
df['Rows'] = df['image_size'].str.get('Rows')
print(df[['image_size', 'Columns', 'Rows']].head())
hb = ax3.hexbin(df['Columns'], df['Rows'], gridsize=40, cmap='inferno', bins='log')
# hb = ax3.hexbin(df['Columns'], df['Rows'], gridsize=40, norm=LogNorm(vmin=1), cmap='inferno')
ax3.set_title('Image Dimension Density (Log Scale)', fontsize=14)
ax3.set_xlabel('Width (Columns)')
ax3.set_ylabel('Height (Rows)')
fig.colorbar(hb, ax=ax3, label='log10(count)')

plt.tight_layout()
output_path = "analyses.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Histogram successfully saved to {output_path}")


# 4. Summary Statistics Printout
print("--- Acuity Check ---")
ap_count = df[df['ViewPosition'] == 'AP'].shape[0]
pa_count = df[df['ViewPosition'] == 'PA'].shape[0]
print(f"AP (Often Portable/ICU) Views: {ap_count}")
print(f"PA (Often Ambulatory) Views:   {pa_count}")
if pa_count > 0:
    print(f"AP to PA Ratio:                {ap_count/pa_count:.2f}")