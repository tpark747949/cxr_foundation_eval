import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("master_metrics.csv")
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# --- Figure 1: Label Efficiency (Exp 2) ---
plt.figure(figsize=(10, 6))
# Filter to raw embeddings, LR/XGB heads, and efficiency labels + CheXpert baseline
eff_df = df[(df["Var"] == "raw") & (df["Label"].isin(["1pct", "5pct", "10pct", "CheXpert"]))].copy()
# Map strings to numeric for the X-axis (CheXpert is our 100% baseline)
pct_map = {"1pct": 1, "5pct": 5, "10pct": 10, "CheXpert": 100}
eff_df["Data_Pct"] = eff_df["Label"].map(pct_map)
mean_eff = eff_df.groupby(["Model", "Data_Pct"])["AUC"].mean().reset_index()

sns.lineplot(data=mean_eff, x="Data_Pct", y="AUC", hue="Model", marker="o", linewidth=2)
plt.xscale("log") # Log scale shows 1, 5, 10, 100 clearly
plt.xticks([1, 5, 10, 100], ["1%", "5%", "10%", "100%"])
plt.title("Label Efficiency: Foundation Models under Data Scarcity", weight="bold")
plt.ylabel("Macro Mean AUROC")
plt.xlabel("Training Data Percentage (Log Scale)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("fig1_label_efficiency.png", dpi=300)
print("Saved fig1_label_efficiency.png")

# --- Figure 2: CheXpert vs NegBio Sensitivity (Exp 1) ---
plt.figure(figsize=(12, 6))
# Compare 100% training on CheXpert vs NegBio
gt_df = df[(df["Var"] == "raw") & (df["Label"].isin(["CheXpert", "NegBio"]))]
mean_gt = gt_df.groupby(["Model", "Label"])["AUC"].mean().reset_index()

sns.barplot(data=mean_gt, x="Model", y="AUC", hue="Label", palette=["#2ca02c", "#d62728"])
plt.title("Ground Truth Sensitivity: CheXpert vs NegBio Labellers", weight="bold")
plt.ylabel("Macro Mean AUROC")
plt.xlabel("Foundation Model")
# Set y-limit intelligently based on data
y_min = max(0.5, mean_gt["AUC"].min() - 0.05)
y_max = mean_gt["AUC"].max() + 0.05
plt.ylim(y_min, y_max)
plt.legend(title="NLP Labeller")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("fig2_labeler_sensitivity.png", dpi=300)
print("Saved fig2_labeler_sensitivity.png")

# --- Figure 3: Classifier Head Architecture Comparison ---
plt.figure(figsize=(10, 6))
# Compare heads on the baseline 100% data
head_df = df[(df["Var"] == "raw") & (df["Label"] == "CheXpert")]
mean_head = head_df.groupby(["Model", "Head"])["AUC"].mean().reset_index()

sns.boxplot(data=mean_head, x="Head", y="AUC", color="lightblue", order=["LR", "XGB", "s2", "i2", "s4", "i4"])
sns.stripplot(data=mean_head, x="Head", y="AUC", color="black", alpha=0.6, order=["LR", "XGB", "s2", "i2", "s4", "i4"])
plt.title("Classifier Head Complexity vs Performance across all Models", weight="bold")
plt.ylabel("Macro Mean AUROC")
plt.xlabel("Classifier Head Architecture")
plt.tight_layout()
plt.savefig("fig3_head_comparison.png", dpi=300)
print("Saved fig3_head_comparison.png")

# --- Figure 4: Per-Disease Performance Heatmap ---
plt.figure(figsize=(14, 8))
# Pivot the data to get Models on the Y axis and Diseases on the X axis
# Averaging across heads for the 100% CheXpert raw embeddings
heat_df = head_df.groupby(["Model", "Disease"])["AUC"].max().reset_index()
heat_pivot = heat_df.pivot(index="Model", columns="Disease", values="AUC")

sns.heatmap(heat_pivot, annot=True, cmap="YlGnBu", fmt=".3f", cbar_kws={'label': 'Mean AUROC'})
plt.title("Per-Disease Performance (100% CheXpert labels, max performance across heads)", weight="bold")
plt.ylabel("Foundation Model")
plt.xlabel("Pathology")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("fig4_disease_heatmap.png", dpi=300)
print("Saved fig4_disease_heatmap.png")