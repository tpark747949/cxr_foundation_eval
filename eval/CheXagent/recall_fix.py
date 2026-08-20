import os
import re
import torch
import lancedb
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

# --- Configuration ---
URI_IMAGES = os.path.expanduser("~/cxr_foundation_eval/embeddings/MIMIC-CXR-JPG")
URI_REPORTS = os.path.expanduser("~/cxr_foundation_eval/embeddings/reports")
REPORTS_PATH = os.path.expanduser("~/cxr_foundation_eval/data/MIMIC-CXR") # Adjust if needed
CHECKPOINT = "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli"
BATCH_SIZE = 32

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def extract_sections(report_text):
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

# --- 1. Identify Target Studies ---
db_img = lancedb.connect(URI_IMAGES)
table_img = db_img.open_table("complete_embeddings_MIMIC-CXR-JPG")
df_img = table_img.search().where("split = 'test'").to_pandas()

subject_list = set(df_img["subject_id"].unique().tolist())
study_list = set(df_img["study_id"].unique().tolist())

# --- 2. Parse Raw Texts ---
print("Scanning report directory...")
all_files = list(Path(REPORTS_PATH).resolve().rglob("*.txt"))

reports = []
for path in all_files:
    split_path = path.parts
    # Adjust indexing [6] and [7] if your path depth differs from the MedSigLIP script!
    try:
        patient = int("".join(filter(str.isdigit, split_path[-2]))) # Safer relative indexing
        study = int("".join(filter(str.isdigit, split_path[-1])))
    except ValueError:
        continue

    if patient not in subject_list or study not in study_list:
        continue

    with open(path, "r") as f:
        full_text = f.read()

    sections = extract_sections(full_text)
    
    reports.append({
        "study_id": study,
        "findings": sections["findings"],
        "impression": sections["impression"]
    })

print(f"Found {len(reports)} reports to process.")

# --- 3. Extract Joint Text Embeddings ---
processor = AutoProcessor.from_pretrained(CHECKPOINT)
model = AutoModelForZeroShotImageClassification.from_pretrained(CHECKPOINT).to(device)
model.eval()

# Dummy image to force the joint forward pass
dummy_image_template = Image.new("RGB", (224, 224))

def extract_text_embeds(texts):
    embeds = []
    # Feed a blank space " " to the tokenizer if a section is missing so it doesn't crash
    clean_texts = [t if t and len(t.strip()) > 0 else " " for t in texts]
    
    for i in tqdm(range(0, len(clean_texts), BATCH_SIZE)):
        batch_txt = clean_texts[i:i + BATCH_SIZE]
        dummy_imgs = [dummy_image_template] * len(batch_txt)
        
        inputs = processor(
            images=dummy_imgs, 
            text=batch_txt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            t_feats = outputs.text_embeds.cpu()
            # L2 Normalization
            t_feats = t_feats / t_feats.norm(dim=-1, keepdim=True)
            embeds.extend(t_feats.numpy().tolist())
            
    return embeds

print("\nExtracting CheXagent joint findings embeddings...")
findings_list = [r["findings"] for r in reports]
findings_embeds = extract_text_embeds(findings_list)

print("\nExtracting CheXagent joint impression embeddings...")
impression_list = [r["impression"] for r in reports]
impression_embeds = extract_text_embeds(impression_list)

# --- 4. Save to LanceDB ---
df_chex_rpt = pd.DataFrame({
    "study_id": [r["study_id"] for r in reports],
    "findings_embedding": findings_embeds,
    "impression_embedding": impression_embeds
})

db_rpt = lancedb.connect(URI_REPORTS)
db_rpt.create_table("CheXagent", df_chex_rpt, mode="overwrite")
print("Successfully overwrote CheXagent report embeddings table!")