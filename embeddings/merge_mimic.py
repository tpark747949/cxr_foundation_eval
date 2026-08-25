import os
import lancedb
import duckdb


def merge_embeddings():
    # 1. Connect to the existing LanceDB URI
    # Using os.path.expanduser to safely resolve the '~' in your path
    uri = "MIMIC-CXR-JPG"
    print(f"Connecting to LanceDB at: {uri}")
    db = lancedb.connect(uri)

    # 2. Define your table names (Update these to match your actual 6 tables)
    table_names = {
        "MedSigLIP": "MedSigLIP_embeddings_MIMIC-CXR-JPG",
        "CXR_Foundation": "CXR_Foundation_embeddings_MIMIC-CXR-JPG", # REPLACE with actual name
        "BioViL-T": "BioViL-T_embeddings_MIMIC-CXR-JPG", # REPLACE with actual name
        "EVA-X": "EVA-X_embeddings_MIMIC-CXR-JPG", # REPLACE with actual name
        "CheXagent": "CheXagent_MIMIC_Part_0", # REPLACE with actual name
        # "model6": "Model6_embeddings_MIMIC-CXR-JPG"  # REPLACE with actual name
    }

    print("Opening tables and converting to PyArrow...")
    # 3. Load tables into local variables. 
    # DuckDB will automatically find these variables by name during the query.
    t_m1 = db.open_table(table_names["MedSigLIP"]).to_arrow()
    t_m2 = db.open_table(table_names["CXR_Foundation"]).to_arrow()
    t_m3 = db.open_table(table_names["BioViL-T"]).to_arrow()
    t_m4 = db.open_table(table_names["EVA-X"]).to_arrow()
    t_m5 = db.open_table(table_names["CheXagent"]).to_arrow()
    # t_m6 = db.open_table(table_names["model6"]).to_arrow()

    print("Executing DuckDB JOIN query...")
    # 4. Write the SQL query to join on dicom_id and consolidate metadata
    query = """
        SELECT 
            -- Keep all shared metadata from Model 1, drop its original embedding columns
            m1.* EXCLUDE (embedding_raw, embedding_l2),
            
            -- Rename and select embeddings for all 6 models
            m1.embedding_raw AS MedSigLIP_raw,
            m1.embedding_l2  AS MedSigLIP_l2,
            
            m2.embedding_raw AS CXR_Foundation_raw,
            m2.embedding_l2  AS CXR_Foundation_l2,
            
            -- Enclosed in double quotes to escape the hyphen
            m3.embedding_raw AS "BioViL-T_raw",
            m3.embedding_l2  AS "BioViL-T_l2",
            
            -- Enclosed in double quotes to escape the hyphen
            m4.embedding_raw AS "EVA-X_raw",
            m4.embedding_l2  AS "EVA-X_l2",
            
            m5.embedding_raw AS CheXagent_raw,
            m5.embedding_l2  AS CheXagent_l2,
            
            -- m6.embedding_raw AS model6_raw,
            -- m6.embedding_l2  AS model6_l2
            
        FROM t_m1 m1
        JOIN t_m2 m2 ON m1.dicom_id = m2.dicom_id
        JOIN t_m3 m3 ON m1.dicom_id = m3.dicom_id
        JOIN t_m4 m4 ON m1.dicom_id = m4.dicom_id
        JOIN t_m5 m5 ON m1.dicom_id = m5.dicom_id
        -- JOIN t_m6 m6 ON m1.dicom_id = m6.dicom_id
    """

    # 5. Run the query and convert the output to an Arrow table
    merged_arrow = duckdb.query(query).to_arrow_table()
    
    print(f"Merge complete. New shape: {merged_arrow.shape}")
    print("Writing master table to LanceDB...")

    # 6. Save the master table back to the same LanceDB URI
    master_table_name = "fixed_embeddings_MIMIC-CXR-JPG"
    db.create_table(master_table_name, data=merged_arrow, mode="overwrite")
    
    print(f"Success! Master table '{master_table_name}' has been created.")

if __name__ == "__main__":
    merge_embeddings()