from transformers import AutoProcessor, AutoModel
import torch
import numpy as np
import lancedb
import pyarrow as pa

DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
    "No Finding",]
URI = "../../embeddings/phrases"
TABLE_NAME = "MedSigLIP"
SCHEMA = pa.schema([
    pa.field("disease", pa.string()),
    pa.field("positive_embedding", pa.list_(pa.float32(), 1152)),
    pa.field("negative_embedding", pa.list_(pa.float32(), 1152)),
])
# this should be like each row is a disease, and the columns are the embeddings for the positive and negative phrases from each model

text_model = AutoModel.from_pretrained("google/medsiglip-448")
processor = AutoProcessor.from_pretrained("google/medsiglip-448")

texts = []
for disease in DISEASES:
    disease = disease.lower()
    pos_phrase = f"{disease} is present"
    neg_phrase = f"no {disease}"
    texts.append(pos_phrase)
    texts.append(neg_phrase)
print("Number of phrases:", len(texts))

inputs = processor(text=texts, padding="max_length", return_tensors="pt")

with torch.no_grad():
    outputs = text_model.get_text_features(**inputs)

output_embeddings = outputs["pooler_output"] / outputs["pooler_output"].norm(p=2, dim=-1, keepdim=True)

print("Connecting to LanceDB...")
db = lancedb.connect(URI)
table = db.create_table(TABLE_NAME, schema=SCHEMA, mode="overwrite")

i = 0
records = []
for disease in DISEASES:
    post_embedding = output_embeddings[i]
    i += 1
    neg_embedding = output_embeddings[i]
    i += 1
    record = {
        "disease": disease,
        "positive_embedding": post_embedding.numpy(),
        "negative_embedding": neg_embedding.numpy(),
    }
    records.append(record)

table.add(records)