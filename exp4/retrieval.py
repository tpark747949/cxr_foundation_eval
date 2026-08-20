import lancedb
import pandas as pd
import numpy as np
import torch
import os

MODELS = ["MedSigLIP", "CXR_Foundation", "BioViL-T", "CheXagent"]
DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", 
    "Lung Opacity", "Pleural Effusion", "Pleural Other", 
    "Pneumonia", "Pneumothorax", "Support Devices", "No Finding"
]

URI_IMAGES = os.path.expanduser("~/cxr_foundation_eval/embeddings/MIMIC-CXR-JPG")
URI_REPORTS = os.path.expanduser("~/cxr_foundation_eval/embeddings/reports")

def parse_chexpert_labels(raw_label):
    if isinstance(raw_label, dict):
        return [raw_label.get(d, -2) for d in DISEASES]
    elif isinstance(raw_label, (list, np.ndarray, tuple)):
        return list(raw_label)
    return [-2] * len(DISEASES)

def get_retrieval_metrics(S, query_indices, target_mask):
    if len(query_indices) == 0:
        return {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0, "MRR": 0.0, "count": 0}

    S_queries = S[query_indices]
    mask_queries = target_mask[query_indices]
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
    db_images = lancedb.connect(URI_IMAGES)
    table_images = db_images.open_table("complete_embeddings_MIMIC-CXR-JPG")
    df_images_base = table_images.search().where("split = 'test'").to_pandas()
    db_reports = lancedb.connect(URI_REPORTS)
    
    results = []

    for model_name in MODELS:
        print(f"\n--- Processing Model: {model_name} ---")
        try:
            table_reports = db_reports.open_table(model_name)
            df_reports = table_reports.to_pandas()
        except Exception as e:
            print(f"Could not load report table for {model_name}: {e}")
            continue

        report_study_ids = df_reports['study_id'].unique()
        df_img_filtered = df_images_base[df_images_base['study_id'].isin(report_study_ids)].reset_index(drop=True)
        
        # Load image embeddings directly from the main table
        col_name = f"{model_name}_l2" if f"{model_name}_l2" in df_img_filtered.columns else f"{model_name}"
        I_tensor = torch.from_numpy(np.array(df_img_filtered[col_name].tolist(), dtype=np.float32))

        test_study_ids = df_img_filtered['study_id'].unique()
        df_rpt_filtered = df_reports[df_reports['study_id'].isin(test_study_ids)].reset_index(drop=True)

        N_I, N_R = len(df_img_filtered), len(df_rpt_filtered)
        
        # Explicit type conversion to avoid UInt32 vs Long conflicts
        img_studies = torch.tensor(df_img_filtered['study_id'].values.astype(np.int64), dtype=torch.int64)
        rpt_studies = torch.tensor(df_rpt_filtered['study_id'].values.astype(np.int64), dtype=torch.int64)
        
        target_mask_i2r = (img_studies.unsqueeze(1) == rpt_studies.unsqueeze(0))
        img_to_rpt_idx = target_mask_i2r.int().argmax(dim=1)

        # Text features
        dim_t = len([x for x in df_rpt_filtered['findings_embedding'].values if x is not None][0])
        f_list = [x if x is not None else [0.0]*dim_t for x in df_rpt_filtered['findings_embedding'].values]
        i_list = [x if x is not None else [0.0]*dim_t for x in df_rpt_filtered['impression_embedding'].values]

        T_f = torch.from_numpy(np.array(f_list, dtype=np.float32))
        T_i = torch.from_numpy(np.array(i_list, dtype=np.float32))

        print(f"Are all {model_name} text embeddings identical? ", torch.allclose(T_f[0], T_f[10]))
        
        valid_f = torch.tensor([x is not None for x in df_rpt_filtered['findings_embedding'].values])
        valid_i = torch.tensor([x is not None for x in df_rpt_filtered['impression_embedding'].values])

        # --- Similarity Computation ---
        if model_name == "CXR_Foundation":
            I_tensor_reshaped = I_tensor.view(N_I, 32, 128)
            S_f_raw = torch.einsum('ipd, rd -> irp', I_tensor_reshaped, T_f)
            S_i_raw = torch.einsum('ipd, rd -> irp', I_tensor_reshaped, T_i)
            
            S_f = torch.logsumexp(S_f_raw, dim=2)
            S_i = torch.logsumexp(S_i_raw, dim=2)
        else:
            S_f = torch.matmul(I_tensor, T_f.T)
            S_i = torch.matmul(I_tensor, T_i.T)

        S_f[:, ~valid_f] = float('-inf')
        S_i[:, ~valid_i] = float('-inf')
        S_soft = torch.logaddexp(S_f, S_i)

        # --- FEATURE: Geometric Centroid 1:1 Matching ---
        unique_studies = torch.unique(rpt_studies)
        study_img_centroids = []
        study_rpt_centroids = []

        for s_id in unique_studies:
            img_mask = (img_studies == s_id)
            img_centroid = I_tensor[img_mask].mean(dim=0)
            norm = img_centroid.norm()
            if norm > 0:
                img_centroid = img_centroid / norm
            study_img_centroids.append(img_centroid)

            rpt_idx = (rpt_studies == s_id).nonzero(as_tuple=True)[0][0]
            rpt_vecs = []
            if valid_f[rpt_idx]: rpt_vecs.append(T_f[rpt_idx])
            if valid_i[rpt_idx]: rpt_vecs.append(T_i[rpt_idx])
            
            if len(rpt_vecs) > 0:
                rpt_centroid = torch.stack(rpt_vecs).mean(dim=0)
                r_norm = rpt_centroid.norm()
                if r_norm > 0:
                    rpt_centroid = rpt_centroid / r_norm
            else:
                rpt_centroid = torch.zeros(dim_t)
            study_rpt_centroids.append(rpt_centroid)

        S_img_centroid = torch.stack(study_img_centroids)
        S_rpt_centroid = torch.stack(study_rpt_centroids)

        if model_name == "CXR_Foundation":
            S_img_c_reshaped = S_img_centroid.view(-1, 32, 128)
            S_1to1_raw = torch.einsum('ipd, rd -> irp', S_img_c_reshaped, S_rpt_centroid)
            S_1to1 = torch.logsumexp(S_1to1_raw, dim=2)
        else:
            S_1to1 = torch.matmul(S_img_centroid, S_rpt_centroid.T)

        target_mask_1to1 = torch.eye(len(unique_studies), dtype=torch.bool)

        sections = {
            "findings": (S_f, S_f.T, valid_f, target_mask_i2r, target_mask_i2r.T),
            "impression": (S_i, S_i.T, valid_i, target_mask_i2r, target_mask_i2r.T),
            "softmax": (S_soft, S_soft.T, valid_f | valid_i, target_mask_i2r, target_mask_i2r.T),
            "centroid_1to1": (S_1to1, S_1to1.T, torch.ones(len(unique_studies), dtype=torch.bool), target_mask_1to1, target_mask_1to1)
        }

        for sec_name, (S_i2r, S_r2i, valid_r_mask, mask_i2r, mask_r2i) in sections.items():
            if sec_name == "centroid_1to1":
                v_img_idx = torch.arange(len(unique_studies))
                v_rpt_idx = torch.arange(len(unique_studies))
            else:
                v_img_idx = torch.where(valid_r_mask[img_to_rpt_idx])[0]
                v_rpt_idx = torch.where(valid_r_mask)[0]

            met_i2r = get_retrieval_metrics(S_i2r, v_img_idx, mask_i2r)
            met_r2i = get_retrieval_metrics(S_r2i, v_rpt_idx, mask_r2i)

            results.append({
                "Model": model_name, "Section": sec_name, "Disease": "Overall", "Label": "All",
                "I2R_R@1": met_i2r["R@1"], "I2R_R@5": met_i2r["R@5"], "I2R_R@10": met_i2r["R@10"], "I2R_MRR": met_i2r["MRR"],
                "R2I_R@1": met_r2i["R@1"], "R2I_R@5": met_r2i["R@5"], "R2I_R@10": met_r2i["R@10"], "R2I_MRR": met_r2i["MRR"],
                "Query_Count_I2R": met_i2r["count"], "Candidate_Pool_Size": S_i2r.shape[1]
            })

    pd.DataFrame(results).to_csv("evaluation_results_stratified.csv", index=False)
    print("\nSaved updated results to evaluation_results_stratified.csv")

if __name__ == "__main__":
    main()