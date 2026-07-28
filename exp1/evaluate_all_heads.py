import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.utils import resample

# --- Configuration ---
ARTIFACT_DIRS = {
    "Logistic Regression": "torch_lr_artifacts",
    "XGBoost": "xgboost_evaluation_artifacts",
    "MLP": "mlp_grid_artifacts"
}

CHEXPERT_DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged_Cardiomediastinum", "Fracture", "Lung_Lesion",
    "Lung_Opacity", "Pleural_Effusion", "Pneumonia",
    "Pneumothorax", "Pleural_Other", "Support_Devices", "No_Finding"
]

MODELS = ["MedSigLIP", "BioViL-T", "EVA-X", "CheXFound", "CheXagent", "CXR_Foundation", "Early_Fusion"]

# Set aesthetic styling
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'


def compute_auprc(y_true, y_prob):
    """Calculates Area Under Precision-Recall Curve (AUPRC)."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return auc(recall, precision)


def bootstrap_ci(y_true, y_prob, metric_fn, n_bootstraps=500, ci=95, seed=42):
    """Computes bootstrapped 95% Confidence Intervals for a metric."""
    np.random.seed(seed)
    scores = []
    n_samples = len(y_true)
    
    for _ in range(n_bootstraps):
        indices = np.random.randint(0, n_samples, n_samples)
        if len(np.unique(y_true[indices])) < 2:
            continue # Skip samples without both classes
        score = metric_fn(y_true[indices], y_prob[indices])
        scores.append(score)
        
    lower = np.percentile(scores, (100 - ci) / 2)
    upper = np.percentile(scores, 100 - (100 - ci) / 2)
    return lower, upper


def load_predictions_and_evaluate():
    """Loads true labels and predicted probabilities across all heads and models."""
    records = []
    
    # Load True Test Labels (from any artifact directory)
    y_true_path = os.path.join(list(ARTIFACT_DIRS.values())[0], "y_test_true.npy")
    if not os.path.exists(y_true_path):
        raise FileNotFoundError(f"True labels file not found at {y_true_path}")
    y_test_true = np.load(y_true_path)

    for head_name, artifact_dir in ARTIFACT_DIRS.items():
        if not os.path.exists(artifact_dir):
            continue
            
        for model in MODELS:
            # Construct file pattern for test probabilities
            prob_file = os.path.join(artifact_dir, f"{model}_raw_test_probs.npy") 
            if not os.path.exists(prob_file):
                prob_file = os.path.join(artifact_dir, f"{model}_test_probs.npy")
            if not os.path.exists(prob_file):
                continue
                
            probs = np.load(prob_file)
            
            for d_idx, disease in enumerate(CHEXPERT_DISEASES):
                y_t = y_test_true[:, d_idx]
                y_p = probs[:, d_idx]
                
                if len(np.unique(y_t)) < 2:
                    continue
                    
                roc_val = roc_auc_score(y_t, y_p)
                pr_val = compute_auprc(y_t, y_p)
                
                # Compute CIs
                roc_low, roc_high = bootstrap_ci(y_t, y_p, roc_auc_score)
                
                records.append({
                    "Head": head_name,
                    "Foundation_Model": model,
                    "Disease": disease,
                    "AUROC": roc_val,
                    "AUROC_Low": roc_low,
                    "AUROC_High": roc_high,
                    "AUPRC": pr_val,
                    "Positives": int(np.sum(y_t))
                })
                
    return pd.DataFrame(records)


# --- Plotting Functions ---

def plot_per_pathology_heatmap(df):
    """Figure 1: Heatmap of AUROC across all Pathologies vs Classifier Heads."""
    pivot = df.groupby(["Head", "Disease"])["AUROC"].mean().unstack(level=0)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Mean AUROC'})
    plt.title("Pathology-Level AUROC Across Classifier Heads", fontsize=14, fontweight='bold')
    plt.xlabel("Classifier Head", fontsize=12)
    plt.ylabel("Pathology", fontsize=12)
    plt.tight_layout()
    plt.savefig("fig1_pathology_head_heatmap.png", dpi=300)
    plt.close()


def plot_head_comparison_with_ci(df, target_model="Early_Fusion"):
    """Figure 2: Head-to-Head Comparison with 95% Confidence Intervals for a given Model."""
    sub_df = df[df["Foundation_Model"] == target_model].copy()
    if sub_df.empty:
        sub_df = df.copy() # Fallback to average across models
        
    plt.figure(figsize=(16, 7))
    
    # Order diseases by prevalence (rarest last)
    disease_order = sub_df.groupby("Disease")["Positives"].mean().sort_values(ascending=False).index
    
    ax = sns.barplot(
        data=sub_df, 
        x="Disease", 
        y="AUROC", 
        hue="Head", 
        order=disease_order,
        palette="muted"
    )
    
    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.ylim(0.5, 1.0)
    plt.title(f"Per-Disease Head Comparison with Bootstrap CIs ({target_model})", fontsize=14, fontweight='bold')
    plt.ylabel("Test AUROC", fontsize=12)
    plt.axhline(0.5, linestyle="--", color="gray", alpha=0.7)
    plt.legend(title="Classifier Head", loc="lower right")
    plt.tight_layout()
    plt.savefig("fig2_per_disease_head_comparison.png", dpi=300)
    plt.close()


def plot_auroc_vs_auprc_scatter(df):
    """Figure 3: Scatter plot showing AUROC vs AUPRC to highlight rare disease performance drops."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, 
        x="AUROC", 
        y="AUPRC", 
        hue="Disease", 
        style="Head", 
        s=100, 
        alpha=0.8
    )
    plt.title("AUROC vs. AUPRC Across Diseases and Classifier Heads", fontsize=14, fontweight='bold')
    plt.xlabel("AUROC (Sensitivity vs. Specificity)", fontsize=12)
    plt.ylabel("AUPRC (Precision vs. Recall)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("fig3_auroc_vs_auprc_scatter.png", dpi=300)
    plt.close()


def main():
    print("Extracting predictions and computing statistics...")
    df_metrics = load_predictions_and_evaluate()
    
    if df_metrics.empty:
        print("No evaluation artifacts found. Ensure outputs exist in specified directories.")
        return
        
    df_metrics.to_csv("comprehensive_disease_evaluation.csv", index=False)
    print("Metrics CSV exported to 'comprehensive_disease_evaluation.csv'.")
    
    print("Generating visualizations...")
    plot_per_pathology_heatmap(df_metrics)
    plot_head_comparison_with_ci(df_metrics)
    plot_auroc_vs_auprc_scatter(df_metrics)
    print("Visualizations successfully saved as PNGs!")

if __name__ == "__main__":
    main()