import lancedb
import pandas as pd
import numpy as np
import torch
import json
import os
from pathlib import Path

# --- Configuration ---
MODELS = ["MedSigLIP", "CXR_Foundation", "BioViL-T", "CheXagent"]
DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
    "Enlarged_Cardiomediastinum", "Fracture", "Lung_Lesion", 
    "Lung_Opacity", "Pleural_Effusion", "Pleural_Other", 
    "Pneumonia", "Pneumothorax", "Support_Devices", "No_Finding"
]
LABEL_VALS = [1, 0, -1, -2]

URI_IMAGES = os.path.expanduser("~/cxr_foundation_eval/embeddings/MIMIC-CXR-JPG")
URI_REPORTS = os.path.expanduser("~/cxr_foundation_eval/embeddings/reports")

# --- Helper Functions ---
def parse_chexpert_labels(raw_label):
    """Safely parses CheXpert labels whether stored as a dict/struct or list/array."""
    if isinstance(raw_label, dict):
        return [raw_label.get(d, -2) for d in DISEASES]
    elif isinstance(raw_label, (list, np.ndarray, tuple)):
        return list(raw_label)
    return [-2] * len(DISEASES)

def get_retrieval_metrics(S, query_indices, target_mask, is_r2i=False):
    """
    S: Similarity matrix. For I2R, shape is (N_I, N_R). For R2I, queries/targets are flipped.
    query_indices: list/tensor of valid query indices to evaluate.
    target_mask: Boolean tensor same shape as S. True where target is a match.
    is_r2i: If True, S is (N_R, N_I) and target_mask is (N_R, N_I)
    """
    if len(query_indices) == 0:
        return {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0, "MRR": 0.0, "count": 0}

    S_queries = S[query_indices]
    mask_queries = target_mask[query_indices]
    
    # Sort scores descending
    sorted_indices = torch.argsort(S_queries, dim=1, descending=True)
    
    ranks = []
    for i in range(len(query_indices)):
        sorted_matches = mask_queries[i, sorted_indices[i]]
        true_positions = torch.where(sorted_matches)[0]
        if len(true_positions) > 0:
            ranks.append(true_positions[0].item() + 1)
            
    ranks = np.array(ranks)
    count = len(ranks)
    
    if count == 0:
        return {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0, "MRR": 0.0, "count": 0}

    return {
        "R@1": float(np.mean(ranks <= 1)),
        "R@5": float(np.mean(ranks <= 5)),
        "R@10": float(np.mean(ranks <= 10)),
        "MRR": float(np.mean(1.0 / ranks)),
        "count": count
    }


def main():
    print("Connecting to Image LanceDB...")
    db_images = lancedb.connect(URI_IMAGES)
    table_images = db_images.open_table("complete_embeddings_MIMIC-CXR-JPG")
    
    # Fetch test set images
    df_images = table_images.search().where("split = 'test'").to_pandas()
    print(f"Loaded {len(df_images)} images from the test set.")
    
    db_reports = lancedb.connect(URI_REPORTS)
    
    results = []

    for model_name in MODELS:
        print(f"\n--- Processing Model: {model_name} ---")
        
        try:
            table_reports = db_reports.open_table(model_name)
            df_reports = table_reports.to_pandas()
        except Exception as e:
            print(f"Could not load report table for {model_name}. Skipping. ({e})")
            continue
            
        # Align image and report studies
        report_study_ids = df_reports['study_id'].unique()
        df_img_filtered = df_images[df_images['study_id'].isin(report_study_ids)].reset_index(drop=True)
        test_study_ids = df_img_filtered['study_id'].unique()
        df_rpt_filtered = df_reports[df_reports['study_id'].isin(test_study_ids)].reset_index(drop=True)
        
        N_I = len(df_img_filtered)
        N_R = len(df_rpt_filtered)
        print(f"Aligned: {N_I} Images mapped to {N_R} Reports (Studies).")
        
        # Build alignment mappings
        img_studies = torch.tensor(df_img_filtered['study_id'].values)
        rpt_studies = torch.tensor(df_rpt_filtered['study_id'].values)
        
        target_mask_i2r = (img_studies.unsqueeze(1) == rpt_studies.unsqueeze(0))
        target_mask_r2i = target_mask_i2r.T
        
        # Cast to int before argmax to avoid boolean argmax RuntimeError
        img_to_rpt_idx = target_mask_i2r.int().argmax(dim=1)
        
        # --- Tensor Extraction ---
        img_col = f"{model_name}_l2"
        I_tensor = torch.from_numpy(np.array(df_img_filtered[img_col].tolist(), dtype=np.float32))
        
        dim_t = len([x for x in df_rpt_filtered['findings_embedding'].values if x is not None][0])
        
        f_list, valid_f = [], []
        for x in df_rpt_filtered['findings_embedding'].values:
            if x is not None:
                f_list.append(x)
                valid_f.append(True)
            else:
                f_list.append([0.0]*dim_t)
                valid_f.append(False)
                
        i_list, valid_i = [], []
        for x in df_rpt_filtered['impression_embedding'].values:
            if x is not None:
                i_list.append(x)
                valid_i.append(True)
            else:
                i_list.append([0.0]*dim_t)
                valid_i.append(False)

        T_f = torch.from_numpy(np.array(f_list, dtype=np.float32))
        T_i = torch.from_numpy(np.array(i_list, dtype=np.float32))
        valid_f = torch.tensor(valid_f, dtype=torch.bool)
        valid_i = torch.tensor(valid_i, dtype=torch.bool)
        
        # --- Similarity Matrix Computation ---
        if model_name == "CXR_Foundation":
            I_tensor = I_tensor.view(N_I, 32, 128)
            S_f_raw = torch.einsum('ipd, rd -> irp', I_tensor, T_f)
            S_i_raw = torch.einsum('ipd, rd -> irp', I_tensor, T_i)
            
            S_f = torch.logsumexp(S_f_raw, dim=2)
            S_i = torch.logsumexp(S_i_raw, dim=2)
        else:
            S_f = torch.matmul(I_tensor, T_f.T)
            S_i = torch.matmul(I_tensor, T_i.T)

        S_f[:, ~valid_f] = float('-inf')
        S_i[:, ~valid_i] = float('-inf')

        S_soft = torch.logaddexp(S_f, S_i)

        S_f_r2i = S_f.T
        S_i_r2i = S_i.T
        S_soft_r2i = S_soft.T

        sections = {
            "findings": (S_f, S_f_r2i, valid_f),
            "impression": (S_i, S_i_r2i, valid_i),
            "softmax": (S_soft, S_soft_r2i, valid_f | valid_i)
        }

        # Disease / Label mappings
        labels_list = [parse_chexpert_labels(x) for x in df_img_filtered['CheXpert_labels'].values]
        study_to_labels = dict(zip(df_img_filtered['study_id'].values, labels_list))
        rpt_labels_list = [study_to_labels[s.item()] for s in rpt_studies]
        rpt_labels_tensor = torch.from_numpy(np.array(rpt_labels_list, dtype=np.int64))

        # --- Evaluate ---
        for sec_name, (S_i2r, S_r2i, valid_r_mask) in sections.items():
            
            valid_img_mask = valid_r_mask[img_to_rpt_idx]
            valid_image_indices = torch.where(valid_img_mask)[0]
            valid_report_indices = torch.where(valid_r_mask)[0]

            # 1. Overall Metrics
            met_i2r = get_retrieval_metrics(S_i2r, valid_image_indices, target_mask_i2r, is_r2i=False)
            met_r2i = get_retrieval_metrics(S_r2i, valid_report_indices, target_mask_r2i, is_r2i=True)

            results.append({
                "Model": model_name,
                "Section": sec_name,
                "Disease": "Overall",
                "Label": "All",
                "I2R_R@1": met_i2r["R@1"], "I2R_R@5": met_i2r["R@5"], "I2R_R@10": met_i2r["R@10"], "I2R_MRR": met_i2r["MRR"],
                "R2I_R@1": met_r2i["R@1"], "R2I_R@5": met_r2i["R@5"], "R2I_R@10": met_r2i["R@10"], "R2I_MRR": met_r2i["MRR"],
                "Query_Count_I2R": met_i2r["count"],
                "Query_Count_R2I": met_r2i["count"]
            })

            # 2. Stratified Metrics
            for d_idx, disease in enumerate(DISEASES):
                for label in LABEL_VALS:
                    condition_rpt_mask = (rpt_labels_tensor[:, d_idx] == label)
                    subset_rpt_indices = torch.where(condition_rpt_mask & valid_r_mask)[0]
                    
                    subset_img_mask = condition_rpt_mask[img_to_rpt_idx]
                    subset_img_indices = torch.where(subset_img_mask & valid_img_mask)[0]

                    s_met_i2r = get_retrieval_metrics(S_i2r, subset_img_indices, target_mask_i2r, is_r2i=False)
                    s_met_r2i = get_retrieval_metrics(S_r2i, subset_rpt_indices, target_mask_r2i, is_r2i=True)

                    results.append({
                        "Model": model_name,
                        "Section": sec_name,
                        "Disease": disease,
                        "Label": label,
                        "I2R_R@1": s_met_i2r["R@1"], "I2R_R@5": s_met_i2r["R@5"], "I2R_R@10": s_met_i2r["R@10"], "I2R_MRR": s_met_i2r["MRR"],
                        "R2I_R@1": s_met_r2i["R@1"], "R2I_R@5": s_met_r2i["R@5"], "R2I_R@10": s_met_r2i["R@10"], "R2I_MRR": s_met_r2i["MRR"],
                        "Query_Count_I2R": s_met_i2r["count"],
                        "Query_Count_R2I": s_met_r2i["count"]
                    })

    # Export
    out_file = "evaluation_results_stratified.json"
    print(f"\nEvaluation Complete. Saving to {out_file}")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    df_out = pd.DataFrame(results)
    df_out.to_csv("evaluation_results_stratified.csv", index=False)
    print("Also saved as CSV: evaluation_results_stratified.csv")

if __name__ == "__main__":
    main()