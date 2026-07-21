import pandas as pd
import numpy as np

ZSCORE_CSV = "./embedding_zscores.csv"

print("Loading embedding Z-scores...")
df = pd.read_csv(ZSCORE_CSV)

# Dynamically find all the model Z-score columns
zscore_cols = [c for c in df.columns if c.endswith("_mag_zscore")]

print(f"Found {len(zscore_cols)} model vectors to combine.")

# 1. Consensus Mean: Average abnormality across all models
df["consensus_mean_zscore"] = df[zscore_cols].mean(axis=1)

# 2. Consensus Minimum: The lowest abnormality score among all models 
# (High value means it broke EVERY single model)
df["consensus_min_zscore"] = df[zscore_cols].min(axis=1)

# Save the updated dataframe
df.to_csv(ZSCORE_CSV, index=False)
print(f"Consensus metrics successfully appended to {ZSCORE_CSV}!")