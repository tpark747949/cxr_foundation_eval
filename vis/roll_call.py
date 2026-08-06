import pandas as pd
from pathlib import Path

DEST_DIR = Path("test_probs")

def roll_call():
    print("Taking roll call of test_probs/...\n")
    files = list(DEST_DIR.glob("*.npy"))
    
    records = []
    for f in files:
        # Expected format: Model_Head_Label_Var.npy
        # e.g., MedSigLIP_LR_1pct_pca95.npy OR Early_Fusion_LR_1pct_pca95.npy
        stem = f.stem
        parts = stem.split('_')
        
        if len(parts) >= 4:
            var = parts[-1]
            label = parts[-2]
            head = parts[-3]
            # Join everything else back together for models with underscores
            model = "_".join(parts[:-3])
            
            records.append({
                "Model": model,
                "Head": head,
                "Label": label,
                "Var": var,
                "Filename": f.name
            })
        else:
            print(f"⚠️ Warning: Couldn't parse structure of {f.name}")
            
    df = pd.DataFrame(records)
    
    if df.empty:
        print("No valid files found in test_probs/")
        return
        
    print(f"Total valid arrays collected: {len(df)}\n")
    
    # 1. Model vs Head completeness
    print("=== Model vs Classifier Head (Count) ===")
    pivot_head = pd.crosstab(df['Model'], df['Head'], margins=True)
    print(pivot_head.to_markdown())
    print("\n")
    
    # 2. Label vs Variation completeness
    print("=== Label Target vs Variance (Count) ===")
    pivot_var = pd.crosstab(df['Label'], df['Var'], margins=True)
    print(pivot_var.to_markdown())
    print("\n")
    
    # Save a master list you can inspect
    out_csv = "inventory_roll_call.csv"
    df.sort_values(['Model', 'Head', 'Label', 'Var']).to_csv(out_csv, index=False)
    print(f"✅ Full detailed inventory saved to {out_csv}")

if __name__ == "__main__":
    roll_call()