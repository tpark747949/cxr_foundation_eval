import lancedb
import matplotlib.pyplot as plt

# 1. Connect to your database and open the table
db = lancedb.connect("../embeddings/MIMIC-CXR-JPG")
table = db.open_table("complete_embeddings_MIMIC-CXR-JPG")

# 2. Select only the target column and convert it to a NumPy array
# This avoids loading unnecessary columns into memory
column_name = "cxr_similarity1"
values = table.to_arrow().column(column_name).to_numpy()

# 3. Create and display the histogram
plt.figure(figsize=(10, 6))
plt.hist(values, bins=50, edgecolor="black", alpha=0.75)

# 4. Add labels and title
plt.xlabel(column_name)
plt.ylabel("Frequency")
plt.title(f"Histogram of {column_name}")
plt.grid(axis="y", linestyle="--", alpha=0.7)

output_path = "cosine_similarity.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Histogram successfully saved to {output_path}")
