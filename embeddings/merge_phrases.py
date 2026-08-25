import os
import lancedb
import duckdb

URI = "phrases"
OUTPUT = "complete_phrases"
TABLE_NAMES = ["MedSigLIP", "CXR_Foundation"]

def merge_embeddings():
    print(f"Connecting to LanceDB at: {URI}")
    db = lancedb.connect(URI)

    table_names = TABLE_NAMES
    print(table_names)

    print("Opening tables and converting to PyArrow...")
    t_m1 = db.open_table(table_names[0]).to_arrow()
    t_m2 = db.open_table(table_names[1]).to_arrow()

    print("Executing DuckDB JOIN query...")
    query = """
        SELECT
            m1.* EXCLUDE (positive_embedding, negative_embedding),

            m1.positive_embedding AS MedSigLIP_positive_embedding,
            m1.negative_embedding AS MedSigLIP_negative_embedding,
            m2.positive_embedding AS CXR_Foundation_positive_embedding,
            m2.negative_embedding AS CXR_Foundation_negative_embedding,
        FROM t_m1 AS m1
        JOIN t_m2 AS m2 ON m1.disease = m2.disease
    """    

    merged_arrow = duckdb.query(query).to_arrow_table()

    print(f"Merge complete. New shape: {merged_arrow.shape[0]} rows, {merged_arrow.shape[1]} columns")
    print("Writing master table to LanceDB...")

    master_table_name = OUTPUT
    db.create_table(master_table_name, data=merged_arrow, mode="overwrite")

    print(f"Successfully merged embeddings and saved to LanceDB table: {master_table_name}")

if __name__ == "__main__":
    merge_embeddings()