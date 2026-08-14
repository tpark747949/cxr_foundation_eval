import lancedb
import numpy as np
import pandas as pd
from typing import Dict, List

# Configuration
URI_IMG = "../embeddings/MIMIC-CXR-JPG"
TABLE_IMG = "complete_embeddings_MIMIC-CXR-JPG"

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
    return np.dot(img_emb, txt_emb) / (norm_img * norm_txt)

def compute_cxr_foundation_score(img_emb: np.ndarray, pos_txt: np.ndarray, neg_txt: np.ndarray) -> float:
    """Handles the 32x128 patch-pooling quirk for CXR_Foundation."""
    img_emb = np.reshape(img_emb, (32, 128))
    pos_sims, neg_sims = [], []
    
    for i in range(32):
        pos_sims.append(compute_similarity(img_emb[i], pos_txt))
        neg_sims.append(compute_similarity(img_emb[i], neg_txt))
        
    # Multi-Instance Learning: Max pool the similarities, then take difference
    return np.max(pos_sims) - np.max(neg_sims)

def compute_standard_score(img_emb: np.ndarray, pos_txt: np.ndarray, neg_txt: np.ndarray) -> float:
    """Standard differential zero-shot score for global embeddings."""
    pos_sim = compute_similarity(img_emb, pos_txt)
    neg_sim = compute_similarity(img_emb, neg_txt)
    return pos_sim - neg_sim

def main():
    print("Connecting to LanceDB...")
    db_img = lancedb.connect(URI_IMG)
    db_txt = lancedb.connect(URI_TXT)
    
    tbl_img = db_img.open_table(TABLE_IMG)
    tbl_txt = db_txt.open_table(TABLE_TXT)
    
    print("Fetching text embeddings...")
    # Fetch all text embeddings into a dataframe
    df_txt = tbl_txt.search().to_pandas()
    # Map disease names to rows for quick lookup
    txt_lookup = {row["disease"]: row for _, row in df_txt.iterrows()}
    
    print("Fetching image embeddings (test split, ignore=0)...")
    # Pushdown filters to LanceDB to only load necessary images into memory
    df_img = tbl_img.search().where("split = 'test' AND ignore = 0").to_pandas()
    
    # Initialize the results container
    results = {
        "dicom_id": df_img["dicom_id"].tolist(),
        "study_id": df_img["study_id"].tolist(),
        "subject_id": df_img["subject_id"].tolist(),
        "view_position": df_img["ViewCodeSequence_CodeMeaning"].tolist()
    }
    
    print("Parsing NLP Labels...")
    for disease in DISEASES:
        # Convert spaced names to underscore names for the struct keys
        struct_key = disease.replace(" ", "_")
        
        # Extract CheXpert and NegBio labels, replacing Nones with np.nan for UI toggling
        results[f"label_chexpert_{struct_key}"] = [
            row.get(struct_key, np.nan) if row is not None else np.nan 
            for row in df_img["CheXpert_labels"]
        ]
        results[f"label_negbio_{struct_key}"] = [
            row.get(struct_key, np.nan) if row is not None else np.nan 
            for row in df_img["NegBio_labels"]
        ]
    
    print("Computing zero-shot scores...")
    for model in MODELS:
        print(f"  -> Processing {model}...")
        
        img_col = f"{model}_l2" if model != "BioViL-T" else "BioViL-T_l2" 
        
        # Determine the text columns based on schema naming quirks
        txt_pos_col, txt_neg_col = f"{model}_positive_embedding", f"{model}_negative_embedding"
            
        for disease in DISEASES:
            struct_key = disease.replace(" ", "_")
            scores = []
            
            # Fetch text vectors for this specific disease and model
            pos_txt = np.array(txt_lookup[disease][txt_pos_col])
            neg_txt = np.array(txt_lookup[disease][txt_neg_col])
            
            # Compute score for every image
            for img_emb in df_img[img_col]:
                img_emb_np = np.array(img_emb)
                
                if model == "CXR_Foundation":
                    score = compute_cxr_foundation_score(img_emb_np, pos_txt, neg_txt)
                else:
                    score = compute_standard_score(img_emb_np, pos_txt, neg_txt)
                    
                scores.append(score)
                
            results[f"score_{model}_{struct_key}"] = scores

    print("Compiling final dataset...")
    df_results = pd.DataFrame(results)
    
    # Save as parquet for highly efficient loading in Streamlit
    output_path = "zeroshot_evaluation_results.parquet"
    df_results.to_parquet(output_path, index=False)
    print(f"Done! Results saved to {output_path}")

if __name__ == "__main__":
    main()