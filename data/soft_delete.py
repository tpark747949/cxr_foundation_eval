import os
import lancedb
import pyarrow as pa
import pyarrow.compute as pc

# ==========================================
# CONFIGURATION
# ==========================================
TXT_FILE = os.path.expanduser("~/cxr_foundation_eval/data/mimic_exclude_list.txt")
LANCEDB_URI = os.path.expanduser("~/cxr_foundation_eval/embeddings/MIMIC-CXR-JPG")
TABLE_NAME = "complete_embeddings_MIMIC-CXR-JPG" # Update if your table name differs

def flag_ignored_images():
    # 1. Read the DICOM IDs from the text file
    print(f"Reading exclude list from {TXT_FILE}...")
    with open(TXT_FILE, "r") as f:
        bad_dicoms = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(bad_dicoms)} DICOM IDs to flag.")

    # 2. Connect to LanceDB and load the dataset into PyArrow
    print(f"Connecting to LanceDB table: {TABLE_NAME}...")
    db = lancedb.connect(LANCEDB_URI)
    tbl = db.open_table(TABLE_NAME)
    
    # Read the entire table into memory (PyArrow handles this efficiently)
    arrow_tbl = tbl.to_arrow()
    
    # 3. Create the new 'ignore' column using highly optimized Arrow compute
    print("Computing the 'ignore' vector...")
    dicom_array = arrow_tbl["dicom_id"]
    
    # pc.is_in returns a boolean array (True if in list, False otherwise)
    is_bad = pc.is_in(dicom_array, value_set=pa.array(bad_dicoms))
    
    # Cast the booleans to int8 (1 for True, 0 for False)
    ignore_array = pc.cast(is_bad, pa.int8())
    
    # Append the new column to our PyArrow table
    arrow_tbl_updated = arrow_tbl.append_column("ignore", ignore_array)
    
    # 4. Overwrite the LanceDB table with the new schema
    print("Overwriting LanceDB table with new 'ignore' column...")
    db.create_table(TABLE_NAME, data=arrow_tbl_updated, mode="overwrite")
    
    # Verify the results
    tbl_new = db.open_table(TABLE_NAME)
    flagged_count = len(tbl_new.search().where("ignore = 1").to_pandas())
    print(f"Success! {flagged_count} rows are now flagged with ignore=1.")

if __name__ == "__main__":
    flag_ignored_images()