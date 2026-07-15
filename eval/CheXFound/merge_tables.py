import lancedb

URI = "../../embeddings/MIMIC-CXR-JPG"

def merge_lancedb_partitions():
    print("Connecting to LanceDB...")
    db = lancedb.connect(URI)
    
    master_table_name = "CheXfound_MIMIC_Part_0"
    master_table = db.open_table(master_table_name)
    
    # Append the remaining partitions (2 and 3)
    for i in [1, 2, 3]:
        part_name = f"CheXfound_MIMIC_Part_{i}"
        # ... rest of the merge logic is the same ...
        
        try:
            part_table = db.open_table(part_name)
            # Read the partition as a PyArrow table and append it instantly
            master_table.add(part_table.to_arrow())
            print(f"Successfully merged {part_name}. Master table now has {len(master_table)} rows.")
            
            # Optional: Drop the partition table to save disk space after merging
            # db.drop_table(part_name)
            
        except Exception as e:
            print(f"Error merging {part_name}: {e}")

    # Optional: Rename the final merged table to something clean
    # db.rename_table(master_table_name, "CXR_Foundation_MIMIC_Complete")
    
    print(f"\nMerge Complete! Final row count: {len(master_table)}")

if __name__ == "__main__":
    merge_lancedb_partitions()