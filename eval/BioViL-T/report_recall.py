from transformers import AutoProcessor, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
import lancedb
import pyarrow as pa
import re
from pathlib import Path

REPORTS_PATH = "../../data/MIMIC-CXR"
URI2 = "../../embeddings/MIMIC-CXR-JPG"
URI = "../../embeddings/reports"
TABLE_NAME = "BioViL-T"

# Removed the 'disease' column
SCHEMA = pa.schema([
    pa.field("study_id", pa.uint32()),   
    pa.field("subject_id", pa.uint32()),
    pa.field("findings_embedding", pa.list_(pa.float32(), 128)),
    pa.field("impression_embedding", pa.list_(pa.float32(), 128)),
])

def extract_sections(report_text):
    """
    Parses a MIMIC-CXR report and groups inconsistent headers into standard 
    'findings' and 'impression' blocks.
    """
    extracted = {"findings": "", "impression": ""}
    
    heading_pattern = re.compile(r'^\s*([A-Z][A-Z0-9\s,\/\-]*?):', re.MULTILINE)
    matches = list(heading_pattern.finditer(report_text))
    
    if not matches:
        return extracted

    sections = {}
    for i, match in enumerate(matches):
        heading = match.group(1).strip()
        start = match.end()
        end = matches[i+1].start() if i + 1 < len(matches) else len(report_text)
        sections[heading] = report_text[start:end].strip()

    findings_headings = [
        "FINDINGS", "PA AND LATERAL VIEWS OF THE CHEST", "CHEST", 
        "FRONTAL AND LATERAL VIEWS OF THE CHEST", "CT CHEST", 
        "EXAMINATION", "PORTABLE RADIOGRAPH OF THE CHEST"
    ]
    impression_headings = ["IMPRESSION", "IMPRESSIONS", "CONCLUSION", "CONCLUSIONS", "SUMMARY"]

    findings_texts = []
    impression_texts = []

    for heading, content in sections.items():
        heading_upper = heading.upper()
        if any(h == heading_upper for h in impression_headings) or "IMPRESSION" in heading_upper:
            impression_texts.append(content)
        elif any(h == heading_upper for h in findings_headings) or "FINDINGS" in heading_upper or "VIEWS" in heading_upper:
            findings_texts.append(content)

    if findings_texts:
        extracted["findings"] = " ".join(findings_texts)
    if impression_texts:
        extracted["impression"] = " ".join(impression_texts)

    return extracted


print("Connecting to reference LanceDB...")
db2 = lancedb.connect(URI2)
table2 = db2.open_table("complete_embeddings_MIMIC-CXR-JPG")
df_filtered = table2.search().where("split = 'test'").to_pandas()
subject_list = set(df_filtered["subject_id"].unique().tolist())
study_list = set(df_filtered["study_id"].unique().tolist())

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
        full_text = f.read()

    sections = extract_sections(full_text)
    
    reports.append({
        "patient": patient,
        "study": study,
        "findings": sections["findings"],
        "impression": sections["impression"]
    })

print(f"Found {len(reports)} reports to process.")

# Removed device_map="auto" to ensure it loads on CPU
from health_multimodal.text import get_bert_inference
from health_multimodal.text.utils import BertEncoderType

text_model = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)

# Feed a blank space " " to the tokenizer if a section is missing so it doesn't crash
findings_texts = [r["findings"] if r["findings"] else " " for r in reports]
impression_texts = [r["impression"] if r["impression"] else " " for r in reports]


print("Generating Findings embeddings...")
with torch.no_grad():
    inputs_f = findings_texts
    # get_text_features returns a tensor, not a dictionary, so we normalize it directly
    out_f = text_model.get_embeddings_from_prompt(inputs_f)
    out_f = out_f / out_f.norm(p=2, dim=-1, keepdim=True)
    f_embeddings_np = out_f.numpy().tolist()

print("Generating Impression embeddings...")
with torch.no_grad():
    inputs_i = impression_texts
    out_i = text_model.get_embeddings_from_prompt(inputs_i)
    out_i = out_i / out_i.norm(p=2, dim=-1, keepdim=True)
    i_embeddings_np = out_i.numpy().tolist()

print("Connecting to LanceDB...")
db = lancedb.connect(URI)
table = db.create_table(TABLE_NAME, schema=SCHEMA, mode="overwrite")

records = []
for idx, r in enumerate(reports):
    record = {
        "study_id": r["study"],
        "subject_id": r["patient"],
        "findings_embedding": f_embeddings_np[idx] if r["findings"] else None,
        "impression_embedding": i_embeddings_np[idx] if r["impression"] else None
    }
    records.append(record)

print(f"Writing {len(records)} records to LanceDB...")
table.add(records)
print("Done!")