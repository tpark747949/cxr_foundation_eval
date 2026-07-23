import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "evaluation_artifacts"

def main():
    csv_path = os.path.join(OUTPUT_DIR, "roc_auc_summary.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Run train_models.py first. Could not find {csv_path}")

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Model', y='Test_AUC', hue='Variant', palette='viridis')
    plt.title('Mean Test ROC-AUC across Foundation Models (PA View Only)')
    plt.ylabel('Mean ROC-AUC')
    plt.xlabel('Model')
    plt.ylim(0.5, 1.0)
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random Chance')
    plt.xticks(rotation=45)
    plt.legend(title='Embedding Type')
    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUT_DIR, 'roc_auc_results.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Saved visualization to {plot_path}")

if __name__ == "__main__":
    main()