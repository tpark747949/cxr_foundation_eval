import os
import lancedb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import umap  # pip install umap-learn

# 1. Connect to LanceDB and load the data
db_path = os.path.expanduser("~/cxr_foundation_eval/embeddings/MIMIC-CXR-JPG")
print(f"Connecting to LanceDB at: {db_path}")

db = lancedb.connect(db_path)
table = db.open_table("fixed_embeddings_MIMIC-CXR-JPG")

print("Loading data into memory...")
df = table.search().to_pandas()

# OPTIMIZATION: UMAP on ~377k 1024D vectors is extremely heavy. 
# We take a random sample of 50,000 to make this run in a reasonable time.
# Change this number based on your RAM and CPU capabilities.
sample_size = min(50000, len(df))
print(f"Downsampling to {sample_size} random scans for UMAP...")
df = df.sample(n=sample_size, random_state=42).copy()

# 2. Extract embeddings and perform UMAP
print("Running UMAP to reduce 1024D -> 2D (This may take a few minutes)...")
embeddings = np.stack(df["CheXagent_raw"].values)

# n_neighbors and min_dist are UMAP hyperparameters you can tweak to change cluster tightness
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
umap_result = reducer.fit_transform(embeddings)

df["umap_x"] = umap_result[:, 0]
df["umap_y"] = umap_result[:, 1]

# 3. Define the labels of interest
labels_list = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged_Cardiomediastinum", "Fracture", "Lung_Lesion",
    "Lung_Opacity", "Pleural_Effusion", "Pneumonia", "Pneumothorax",
    "Pleural_Other", "Support_Devices", "No_Finding"
]

# 4. Set up the plot
plt.figure(figsize=(16, 12))
colors = sns.color_palette("husl", len(labels_list))
sns.set_style("whitegrid")

# --- NEW: Plot the neutral background density ---
print("Plotting neutral background points...")
# We can downsample the background points even further if 50k is still too visually dense
bg_sample = df.sample(n=min(20000, len(df)), random_state=42)
plt.scatter(
    bg_sample["umap_x"], 
    bg_sample["umap_y"], 
    color="gray", 
    alpha=0.15,  # Highly transparent 
    s=3,         # Small point size
    zorder=1     # Ensure points stay under the contours
)

# We will store custom legend handles here
legend_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
           markersize=8, label=f"Background Scans (n={len(bg_sample)})")
]

print("Plotting contours...")
for idx, label in enumerate(labels_list):
    # Extract structural label
    df[label] = df["CheXpert_labels"].apply(lambda x: x.get(label) if isinstance(x, dict) else None)
    
    # Filter for positive cases
    positive_subset = df[df[label] == 1]
    
    if len(positive_subset) < 10:
        print(f"  Skipping {label} - not enough positive samples ({len(positive_subset)}).")
        continue

    print(f"  Drawing {label} (n={len(positive_subset)})...")
    
    # Plot KDE contour
    sns.kdeplot(
        x=positive_subset["umap_x"],
        y=positive_subset["umap_y"],
        thresh=0.3,  # Top 70% density
        levels=2,
        color=colors[idx],
        linewidths=2.5,
        alpha=0.9,
        zorder=2     # Draw on top of scatter points
    )
    
    # --- NEW: Create a custom line object for the legend ---
    custom_line = Line2D([0], [0], color=colors[idx], lw=2.5, label=f"{label} (n={len(positive_subset)})")
    legend_handles.append(custom_line)

# 5. Finalize aesthetics and save
plt.title("CheXagent Latent Space: UMAP with CheXpert Label Density Contours", fontsize=16)
plt.xlabel("UMAP Component 1", fontsize=12)
plt.ylabel("UMAP Component 2", fontsize=12)

# Pass the custom handles to the legend
plt.legend(handles=legend_handles, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., fontsize=11)
plt.tight_layout()

output_path = "chexagent_umap_contours.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Done! Plot saved to {os.path.abspath(output_path)}")