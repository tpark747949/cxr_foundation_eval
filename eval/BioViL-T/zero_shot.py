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
TABLE_NAME = "BioViL-T"
SCHEMA = pa.schema([
    pa.field("disease", pa.string()),
    pa.field("positive_embedding", pa.list_(pa.float32(), 128)),
    pa.field("negative_embedding", pa.list_(pa.float32(), 128)),
])
# this should be like each row is a disease, and the columns are the embeddings for the positive and negative phrases from each model

from health_multimodal.text import get_bert_inference
from health_multimodal.text.utils import BertEncoderType

text_model = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)
# processor = AutoProcessor.from_pretrained("google/medsiglip-448")

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


with torch.no_grad():
    # text_embedding = self.text_inference_engine.get_embeddings_from_prompt(query_text)
    outputs = text_model.get_embeddings_from_prompt(texts)

print(outputs.shape)

output_embeddings = outputs / outputs.norm(p=2, dim=-1, keepdim=True)

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