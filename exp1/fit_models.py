import joblib
import lancedb
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DB_URI = "../embeddings/MIMIC-CXR-JPG"
TABLE_NAME = "complete_embeddings_MIMIC-CXR-JPG"
OUTPUT_DIR = "evaluation_artifacts"

MODELS = ["MedSigLIP", "BioViL-T", "EVA-X", "CheXFound", "CheXagent", "CXR_Foundation"]
VARIANTS = ["raw", "l2"]

CHEXPERT_DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged_Cardiomediastinum", "Fracture", "Lung_Lesion",
    "Lung_Opacity", "Pleural_Effusion", "Pneumonia",
    "Pneumothorax", "Pleural_Other", "Support_Devices", "No_Finding",
]

def process_labels(df_labels, strategy="zeros"):
    labels_matrix = np.zeros((len(df_labels), len(CHEXPERT_DISEASES)))
    for idx, row in enumerate(df_labels):
        for d_idx, disease in enumerate(CHEXPERT_DISEASES):
            val = row[disease]
            if val == 1.0:
                labels_matrix[idx, d_idx] = 1.0
            elif val == -1.0:
                labels_matrix[idx, d_idx] = 1.0 if strategy == "ones" else 0.0
            else:
                labels_matrix[idx, d_idx] = 0.0
    return labels_matrix

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    db = lancedb.connect(DB_URI)
    tbl = db.open_table(TABLE_NAME)
    df = tbl.to_pandas()

    # Filter data
    valid_data_mask = (df["ignore"] != 1)
    view_data_mask = (df["ViewCodeSequence_CodeMeaning"] == "postero-anterior")
    base_mask = valid_data_mask & view_data_mask

    train_mask = base_mask & (df["split"] == "train")
    val_mask = base_mask & (df["split"].isin(["val", "validate"]))
    test_mask = base_mask & (df["split"] == "test")

    # Ground truths
    y_train = process_labels(df.loc[train_mask, "CheXpert_labels"])
    y_val = process_labels(df.loc[val_mask, "CheXpert_labels"])
    y_test = process_labels(df.loc[test_mask, "CheXpert_labels"])

    # Save ground truths for the Streamlit app
    np.save(os.path.join(OUTPUT_DIR, "y_val_true.npy"), y_val)
    np.save(os.path.join(OUTPUT_DIR, "y_test_true.npy"), y_test)

    results = []
    
    def train_and_eval(model_id, X_tr, y_tr, X_v, y_v, X_te, y_te, c_val):
        print(f"--- Training: {model_id} ---")
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", MultiOutputClassifier(
                LogisticRegression(C=c_val, max_iter=5000, class_weight="balanced", solver="saga", n_jobs=1),
                n_jobs=-1,
            )),
        ])
        clf.fit(X_tr, y_tr)

        # Predict probabilities (better for ROC/Thresholding than raw logits)
        val_probs = np.array([p[:, 1] for p in clf.predict_proba(X_v)]).T
        test_probs = np.array([p[:, 1] for p in clf.predict_proba(X_te)]).T

        # Save probabilities
        np.save(os.path.join(OUTPUT_DIR, f"{model_id}_val_probs.npy"), val_probs)
        np.save(os.path.join(OUTPUT_DIR, f"{model_id}_test_probs.npy"), test_probs)
        joblib.dump(clf, os.path.join(OUTPUT_DIR, f"lr_{model_id}.joblib"))

        # Calculate AUCs
        def calc_auc(y_true, y_prob):
            aucs = [roc_auc_score(y_true[:, i], y_prob[:, i]) if len(np.unique(y_true[:, i])) > 1 else np.nan for i in range(len(CHEXPERT_DISEASES))]
            return np.nanmean(aucs)

        val_auc, test_auc = calc_auc(y_v, val_probs), calc_auc(y_te, test_probs)
        print(f"{model_id} | Val AUC: {val_auc:.4f} | Test AUC: {test_auc:.4f}\n")
        
        return val_auc, test_auc

    # 1. Foundation Models
    for model_name in MODELS:
        for variant in VARIANTS:
            col_key = f"{model_name}_{variant}"
            X_train = np.vstack(df.loc[train_mask, col_key].values)
            X_val = np.vstack(df.loc[val_mask, col_key].values)
            X_test = np.vstack(df.loc[test_mask, col_key].values)
            
            c_val = 0.1 if model_name == "CXR_Foundation" else 1.0
            v_auc, t_auc = train_and_eval(col_key, X_train, y_train, X_val, y_val, X_test, y_test, c_val)
            results.append({"Model": model_name, "Variant": variant, "Val_AUC": v_auc, "Test_AUC": t_auc})

    # 2. Early Fusion
    for variant in VARIANTS:
        X_train_fusion = np.hstack([np.vstack(df.loc[train_mask, f"{m}_{variant}"].values) for m in MODELS])
        X_val_fusion = np.hstack([np.vstack(df.loc[val_mask, f"{m}_{variant}"].values) for m in MODELS])
        X_test_fusion = np.hstack([np.vstack(df.loc[test_mask, f"{m}_{variant}"].values) for m in MODELS])
        
        v_auc, t_auc = train_and_eval(f"Early_Fusion_{variant}", X_train_fusion, y_train, X_val_fusion, y_val, X_test_fusion, y_test, c_val=0.01)
        results.append({"Model": "Early_Fusion", "Variant": variant, "Val_AUC": v_auc, "Test_AUC": t_auc})

    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, "roc_auc_summary.csv"), index=False)

if __name__ == "__main__":
    main()