from transformers import AutoTokenizer, AutoModelForZeroShotImageClassification
import torch
import numpy as np
import lancedb
import pyarrow as pa
import torch.nn.functional as F

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
TABLE_NAME = "CheXagent"
SCHEMA = pa.schema([
    pa.field("disease", pa.string()),
    pa.field("positive_embedding", pa.list_(pa.float32(), 1024)),
    pa.field("negative_embedding", pa.list_(pa.float32(), 1024)),
])
# this should be like each row is a disease, and the columns are the embeddings for the positive and negative phrases from each model

text_model = AutoModelForZeroShotImageClassification.from_pretrained("StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli")

texts = []
for disease in DISEASES:
    disease = disease.lower()
    # Distinct positive phrase
    pos_phrase = f"radiographic findings consistent with {disease}"
    
    # Universal negative phrase (NO lexical overlap with the disease)
    neg_phrase = "normal chest x-ray, clear lungs, no abnormalities"
    
    texts.append(pos_phrase)
    texts.append(neg_phrase)
print("Number of phrases:", len(texts))

inputs = tokenizer(text=texts, padding="max_length", return_tensors="pt")

with torch.no_grad():
    outputs = text_model.get_text_features(**inputs)


if "pooler_output" in outputs:
    text_embeddings_tensor = outputs["pooler_output"]
elif "text_embeds" in outputs:
    text_embeddings_tensor = outputs["text_embeds"]
else:
    # Fallback if it returned a raw tensor directly
    text_embeddings_tensor = outputs

output_embeddings = text_embeddings_tensor

print("Connecting to LanceDB...")
db = lancedb.connect(URI)
table = db.create_table(TABLE_NAME, schema=SCHEMA, mode="overwrite")

i = 0
records = []
for disease in DISEASES:
    # 3. CHANGED: Fixed typo 'post_embedding' (was 'post_embedding' vs text variable names)
    pos_embedding = output_embeddings[i].cpu()
    i += 1
    neg_embedding = output_embeddings[i].cpu()
    i += 1
    record = {
        "disease": disease,
        "positive_embedding": pos_embedding.numpy(),
        "negative_embedding": neg_embedding.numpy(),
    }
    records.append(record)

table.add(records)