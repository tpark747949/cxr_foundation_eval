import lancedb
import numpy as np
import pandas as pd
from typing import Dict, List

# Configuration
URI_IMG = "../embeddings/MIMIC-CXR-JPG"
TABLE_IMG = "fixed_embeddings_MIMIC-CXR-JPG"

URI_TXT = "../embeddings/phrases"
TABLE_TXT = "complete_phrases"

DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", 
    "Lung Opacity", "Pleural Effusion", "Pleural Other", 
    "Pneumonia", "Pneumothorax", "Support Devices", "No Finding"
]

MODELS = ["MedSigLIP", "CXR_Foundation", "BioViL-T", "CheXagent"]

def compute_similarity(img_emb: np.ndarray, txt_emb: np.ndarray) -> float:
    """Computes cosine similarity."""
    norm_img = np.linalg.norm(img_emb)
    norm_txt = np.linalg.norm(txt_emb)
    if norm_img == 0 or norm_txt == 0:
        return 0.0
    return float(np.dot(img_emb, txt_emb) / (norm_img * norm_txt))

def compute_cxr_foundation_score(img_emb: np.ndarray, pos_txt: np.ndarray, neg_txt: np.ndarray) -> float:
    """Handles the 32x128 patch-pooling quirk for CXR_Foundation."""
    img_emb = np.reshape(img_emb, (32, 128))
    pos_sims, neg_sims = [], []
    
    for i in range(32):
        pos_sims.append(compute_similarity(img_emb[i], pos_txt))
        neg_sims.append(compute_similarity(img_emb[i], neg_txt))
        
    return float(np.max(pos_sims) - np.max(neg_sims))

def compute_standard_score(img_emb: np.ndarray, pos_txt: np.ndarray, neg_txt: np.ndarray) -> float:
    """Standard differential zero-shot score for global embeddings."""
    pos_sim = compute_similarity(img_emb, pos_txt)
    neg_sim = compute_similarity(img_emb, neg_txt)
    return float(pos_sim - neg_sim)

def main():
    print("Connecting to LanceDB...")
    db_img = lancedb.connect(URI_IMG)
    db_txt = lancedb.connect(URI_TXT)
    
    tbl_img = db_img.open_table(TABLE_IMG)
    tbl_txt = db_txt.open_table(TABLE_TXT)
    
    # Open pre-computed continuous prediction tables
    tbl_chexagent = db_txt.open_table("CheXagent")
    tbl_biovil = db_txt.open_table("BioViL-T")
    
    print("Fetching text embeddings and pre-computed tables...")
    df_txt = tbl_txt.search().to_pandas()
    txt_lookup = {row["disease"]: row for _, row in df_txt.iterrows()}
    
    print("Building lookup maps for pre-computed continuous scores...")
    df_chexagent = tbl_chexagent.search().to_pandas()
    chexagent_lookup = {row["dicom_id"]: row["prediction"] for _, row in df_chexagent.iterrows()}
    
    df_biovil = tbl_biovil.search().to_pandas()
    biovil_lookup = {row["dicom_id"]: row["prediction"] for _, row in df_biovil.iterrows()}
    
    print("Fetching test set image embeddings (split = 'test' AND ignore = 0)...")
    df_img = tbl_img.search().where("split = 'test' AND ignore = 0").to_pandas()
    
    results = {
        "dicom_id": df_img["dicom_id"].tolist(),
        "study_id": df_img["study_id"].tolist(),
        "subject_id": df_img["subject_id"].tolist(),
        "view_position": df_img["ViewCodeSequence_CodeMeaning"].tolist()
    }
    
    print("Parsing NLP Labels (CheXpert & NegBio)...")
    for disease in DISEASES:
        struct_key = disease.replace(" ", "_")
        
        results[f"label_chexpert_{struct_key}"] = [
            float(row.get(struct_key, np.nan)) if row is not None else np.nan 
            for row in df_img["CheXpert_labels"]
        ]
        results[f"label_negbio_{struct_key}"] = [
            float(row.get(struct_key, np.nan)) if row is not None else np.nan 
            for row in df_img["NegBio_labels"]
        ]
    
    print("Processing scores for all models...")
    for model in MODELS:
        print(f"  -> Extracting/Computing scores for {model}...")
        
        # --- PATH 1: Pre-computed Continuous Scores (CheXagent & BioViL-T) ---
        if model in ["CheXagent", "BioViL-T"]:
            lookup = chexagent_lookup if model == "CheXagent" else biovil_lookup
            
            for disease in DISEASES:
                struct_key = disease.replace(" ", "_")
                scores = []
                
                for d_id in results["dicom_id"]:
                    if d_id in lookup and lookup[d_id] is not None:
                        val = lookup[d_id].get(struct_key, np.nan)
                        scores.append(float(val) if val is not None else np.nan)
                    else:
                        scores.append(np.nan)
                        
                results[f"score_{model}_{struct_key}"] = scores
            continue

        # --- PATH 2: On-the-fly Continuous Score Computation (MedSigLIP & CXR_Foundation) ---
        img_col = f"{model}_l2" 
        txt_pos_col, txt_neg_col = f"{model}_positive_embedding", f"{model}_negative_embedding"
            
        for disease in DISEASES:
            struct_key = disease.replace(" ", "_")
            scores = []
            
            pos_txt = np.array(txt_lookup[disease][txt_pos_col])
            neg_txt = np.array(txt_lookup[disease][txt_neg_col])
            
            for img_emb in df_img[img_col]:
                img_emb_np = np.array(img_emb)
                
                if model == "CXR_Foundation":
                    score = compute_cxr_foundation_score(img_emb_np, pos_txt, neg_txt)
                else:
                    score = compute_standard_score(img_emb_np, pos_txt, neg_txt)
                    
                scores.append(score)
                
            results[f"score_{model}_{struct_key}"] = scores

    print("Compiling and saving evaluation dataset...")
    df_results = pd.DataFrame(results)
    
    output_path = "zeroshot_evaluation_results.parquet"
    df_results.to_parquet(output_path, index=False)
    print(f"Successfully saved continuous scores to {output_path}")

if __name__ == "__main__":
    main()