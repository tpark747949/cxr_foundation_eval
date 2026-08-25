import os
import lancedb
import duckdb

def append_latecomer():
    uri = os.path.expanduser("~/cxr_foundation_eval/embeddings/MIMIC-CXR-JPG/")
    print(f"Connecting to LanceDB at: {uri}")
    db = lancedb.connect(uri)

    print("Loading master table and latecomer table into PyArrow...")
    # 1. Load the existing master table and the new latecomer table
    t_master = db.open_table("complete_embeddings_MIMIC-CXR-JPG").to_arrow()
    
    # REPLACE with the actual name of your latecomer table
    t_late = db.open_table("CheXfound_MIMIC").to_arrow() 

    print("Executing DuckDB JOIN to append new columns...")
    # 2. Join them, keeping all existing master columns (m.*) and adding the new ones
    query = """
        SELECT 
            m.*,
            
            -- Rename and select the new model's embeddings
            -- Update 'LateModel' to whatever your new model is named
            l.embedding_raw AS CheXFound_raw,
            l.embedding_l2  AS CheXFound_l2
            
        FROM t_master m
        JOIN t_late l ON m.dicom_id = l.dicom_id
    """

    merged_arrow = duckdb.query(query).to_arrow_table()
    
    print(f"Merge complete. New shape: {merged_arrow.shape}")
    print("Overwriting master table with new schema...")

    # 3. Overwrite the existing master table with the updated schema
    db.create_table(
        "complete_embeddings_MIMIC-CXR-JPG", 
        data=merged_arrow, 
        mode="overwrite"
    )
    
    print("Success! Latecomer embeddings have been appended.")

if __name__ == "__main__":
    append_latecomer()