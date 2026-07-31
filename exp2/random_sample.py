import os
import lancedb
import pandas as pd
import numpy as np

# 1. Configuration
LANCEDB_URI = os.path.expanduser("~/cxr_foundation_eval/embeddings/MIMIC-CXR-JPG")
TABLE_NAME = "complete_embeddings_MIMIC-CXR-JPG"
NEW_TABLE_NAME = "sampled_embeddings_MIMIC-CXR-JPG"
LOG_FILE = "stratified_sampling_log.csv"
SEED = 42

def get_stochastic_size(n_total, fraction):
    """
    Returns an integer sample size using stochastic rounding.
    e.g., n=40, fraction=0.01 -> expected=0.4 -> 40% chance of 1, 60% chance of 0.
    """
    expected = n_total * fraction
    base = int(expected)
    prob = expected - base
    return base + (1 if np.random.rand() < prob else 0)

def main():
    print("Connecting to LanceDB and loading data...")
    db = lancedb.connect(LANCEDB_URI)
    table = db.open_table(TABLE_NAME)
    df = table.to_pandas()
    
    # 2. Prepare the Stratification Key
    # We group by ViewCode and Split. The independent disease marginals 
    # will naturally balance due to the stochastic random sampling.
    print("Preparing stratification keys...")
    df['stratify_key'] = df.apply(
        lambda row: (row['ViewCodeSequence_CodeMeaning'], row['split']), 
        axis=1
    )

    df['sample_1_percent'] = 0
    df['sample_5_percent'] = 0
    df['sample_10_percent'] = 0

    # 3. Perform Nested Stratified Sampling with Stochastic Rounding
    print("Performing stochastic stratified sampling...")
    np.random.seed(SEED)
    log_records = []

    for name, group in df.groupby('stratify_key'):
        n_total = len(group)
        
        # Calculate sample sizes probabilistically
        n_1 = get_stochastic_size(n_total, 0.01)
        
        # To maintain strict nesting (1% is a subset of 5%), we calculate the 
        # *additional* samples needed to reach 5%, rather than calculating 5% of total.
        n_5_target = get_stochastic_size(n_total, 0.05)
        n_5_additional = max(0, n_5_target - n_1)
        n_5 = n_1 + n_5_additional
        
        n_10_target = get_stochastic_size(n_total, 0.10)
        n_10_additional = max(0, n_10_target - n_5)
        n_10 = n_5 + n_10_additional

        # Shuffle indices randomly
        shuffled_idx = np.random.permutation(group.index)

        # Slice the shuffled indices (Nested: 1% is inside 5% is inside 10%)
        idx_1 = shuffled_idx[:n_1]
        idx_5 = shuffled_idx[:n_5]
        idx_10 = shuffled_idx[:n_10]

        # Assign binary flags
        df.loc[idx_1, 'sample_1_percent'] = 1
        df.loc[idx_5, 'sample_5_percent'] = 1
        df.loc[idx_10, 'sample_10_percent'] = 1

        # Extract values for logging
        view_code, split_val = name
        log_records.append({
            'ViewCode': view_code,
            'Split': split_val,
            'Total_Population': n_total,
            'Expected_1_pct': n_total * 0.01,
            'Actual_1_pct': n_1,
            'Expected_5_pct': n_total * 0.05,
            'Actual_5_pct': n_5,
            'Expected_10_pct': n_total * 0.10,
            'Actual_10_pct': n_10
        })

    # 4. Generate the Manuscript Log
    print(f"Saving sampling log to {LOG_FILE}...")
    log_df = pd.DataFrame(log_records)
    log_df = log_df.sort_values(by='Total_Population', ascending=False)
    log_df.to_csv(LOG_FILE, index=False)

    # 5. Clean up temporary columns and save back to LanceDB
    print(f"Saving updated data to new LanceDB table: {NEW_TABLE_NAME}...")
    df = df.drop(columns=['stratify_key'])
    
    db.create_table(NEW_TABLE_NAME, data=df, mode="overwrite")
    print("Done!")

if __name__ == "__main__":
    main()