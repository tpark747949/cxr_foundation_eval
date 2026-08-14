from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import torch
import numpy as np
import lancedb
import pyarrow as pa

import os
import shutil
from pathlib import Path

REPORTS_PATH = "../../data/MIMIC-CXR"

URI2 = "../../embeddings/MIMIC-CXR-JPG"

URI = "../../embeddings/reports"
TABLE_NAME = "CheXagent"
SCHEMA = pa.schema([
    pa.field("disease", pa.string()),
    pa.field("study_id", pa.uint32()),   
    pa.field("subject_id", pa.uint32()),
    pa.field("embedding", pa.list_(pa.float32(), 1024)),
])
# this should be like each row is a disease, and the columns are the embeddings for the positive and negative phrases from each model

print("Connecting to reference LanceDB...")
db2 = lancedb.connect(URI2)
table2 = db2.open_table("complete_embeddings_MIMIC-CXR-JPG")
df_filtered = table2.search().where("split = 'test'").to_pandas()
subject_list = df_filtered["subject_id"].unique().tolist()
study_list = df_filtered["study_id"].unique().tolist()


print("Scanning report directory...")

all_files = list(Path(REPORTS_PATH).resolve().rglob("*.txt"))

reports = []
for path in all_files:
    split_path = path.parts
    patient = int("".join(filter(str.isdigit, split_path[6])))
    study = int("".join(filter(str.isdigit, split_path[7])))

    if patient not in subject_list or study not in study_list:
        continue

    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    cleaned_text = "\n".join(lines)
    report = [patient, study, cleaned_text]
    reports.append(report)

print(f"Found {len(reports)} reports.")


text_model = AutoModelForZeroShotImageClassification.from_pretrained("StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli", device_map="auto")
processor = AutoProcessor.from_pretrained("StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli")

texts = []
for _, _, text in reports:
    texts.append(text)

processor.tokenizer.truncation_side = "left"
inputs = processor(text=texts, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    outputs = text_model.get_text_features(**inputs)

output_embeddings = outputs["pooler_output"] / outputs["pooler_output"].norm(p=2, dim=-1, keepdim=True)

print(output_embeddings.shape)

print("Connecting to LanceDB...")
db = lancedb.connect(URI)
table = db.create_table(TABLE_NAME, schema=SCHEMA, mode="overwrite")

i = 0
records = []
for patient, study, text in reports:
    embedding = output_embeddings[i].cpu().numpy().tolist()
    record = {
        "study_id": study,
        "subject_id": patient,
        "embedding": embedding
    }
    records.append(record)
    i += 1

print(f"Writing {len(records)} records to LanceDB...")
table.add(records)