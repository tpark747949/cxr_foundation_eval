import torch
import lancedb
import pyarrow as pa
import numpy as np

# 1. Connect and Setup
uri = "../embeddings/MIMIC-CXR-JPG"
db = lancedb.connect(uri)
device = "cuda" if torch.cuda.is_available() else "cpu"
table_name = "complete_embeddings_MIMIC-CXR-JPG"

# 2. Load Reference Embedding
text_embedding = torch.load('../eval/MedSigLIP/text_embedding.pt').to(device)
if text_embedding.dim() > 1:
    text_embedding = text_embedding.squeeze()

# 3. Pull the ENTIRE table to PyArrow
# This pulls all vectors and all metadata into memory, preserving row order exactly.
table = db.open_table(table_name)
arrow_table = table.to_arrow()

# 4. Extract vectors instantly using Arrow's flat buffer
embedding_dim = 1152
flat_numpy = arrow_table["MedSigLIP_l2"].combine_chunks().values.to_numpy()
reshaped_numpy = flat_numpy.reshape(-1, embedding_dim)

# Move to GPU and compute
vision_embeddings = torch.from_numpy(reshaped_numpy).to(device)
similarities = torch.mv(vision_embeddings, text_embedding)

# 5. Append the new column to the PyArrow table
# PyArrow arrays perfectly align with the row order of the table we just pulled
similarities_cpu = similarities.cpu().numpy()
similarity_array = pa.array(similarities_cpu)

# append_column creates a new PyArrow table with the appended data
arrow_table = arrow_table.append_column("cxr_similarity1", similarity_array)

# 6. Overwrite the LanceDB table
# This replaces the old table with your new one, safely locking in the new column
db.create_table(table_name, data=arrow_table, mode="overwrite")

print(f"Successfully added 'cxr_similarity' to {table_name}.")
print(f"Total rows updated: {arrow_table.num_rows}")