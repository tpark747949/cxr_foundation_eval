import os
import torch
import lancedb
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

# --- Configuration ---
URI_IMAGES = os.path.expanduser("~/cxr_foundation_eval/embeddings/MIMIC-CXR-JPG")
URI_REPORTS = os.path.expanduser("~/cxr_foundation_eval/embeddings/reports")
DATA_DIR = os.path.expanduser("~/cxr_foundation_eval/data/MIMIC-CXR-JPG/2.1.0/")
CHECKPOINT = "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli"
BATCH_SIZE = 32

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Load Model & Processor ---
processor = AutoProcessor.from_pretrained(CHECKPOINT)
model = AutoModelForZeroShotImageClassification.from_pretrained(CHECKPOINT).to(device)
model.eval()

# --- 1. Process & Save Image Embeddings ---
db_img = lancedb.connect(URI_IMAGES)
table_img = db_img.open_table("complete_embeddings_MIMIC-CXR-JPG")
df_img = table_img.search().where("split = 'test'").to_pandas()

image_paths = [os.path.join(DATA_DIR, path) for path in df_img["path"].tolist()]
dicom_ids = df_img["dicom_id"].tolist()

img_embeddings = []
print(f"Extracting CheXagent joint image embeddings for {len(image_paths)} images...")

for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
    batch_paths = image_paths[i:i + BATCH_SIZE]
    batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]
    
    # Pass dummy text to force the joint forward pass
    dummy_text = [""] * len(batch_imgs)
    inputs = processor(
        images=batch_imgs, 
        text=dummy_text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
        # Extract from the joint space outputs
        feats = outputs.image_embeds.cpu()
        
        # Ensure L2 Normalization (model(**inputs) usually does this, but this is a safe fallback guarantee)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        img_embeddings.extend(feats.numpy().tolist())

# Update LanceDB image table
df_img["CheXagent_l2"] = img_embeddings
db_img.create_table("complete_embeddings_MIMIC-CXR-JPG", df_img, mode="overwrite")
print("Saved joint image embeddings to LanceDB.")


# --- 2. Process & Save Report Text Embeddings ---
db_rpt = lancedb.connect(URI_REPORTS)
table_rpt = db_rpt.open_table("MedSigLIP") # source for report texts
df_rpt = table_rpt.to_pandas()

study_ids = df_rpt["study_id"].tolist()
findings_list = df_rpt["findings"].tolist() if "findings" in df_rpt.columns else [""] * len(study_ids)
impression_list = df_rpt["impression"].tolist() if "impression" in df_rpt.columns else [""] * len(study_ids)

# Create a small, blank dummy image to pair with text
dummy_image_template = Image.new("RGB", (224, 224))

def extract_text_embeds(texts):
    embeds = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_txt = [t if isinstance(t, str) and len(t.strip()) > 0 else "None" for t in texts[i:i + BATCH_SIZE]]
        
        # Pass dummy images to force the joint forward pass
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
            
            # Extract from the joint space outputs
            t_feats = outputs.text_embeds.cpu()
            
            # L2 Normalization
            t_feats = t_feats / t_feats.norm(dim=-1, keepdim=True)
            embeds.extend(t_feats.numpy().tolist())
            
    return embeds

print("\nExtracting CheXagent joint report text embeddings...")
findings_embeds = extract_text_embeds(findings_list)
impression_embeds = extract_text_embeds(impression_list)

df_chex_rpt = pd.DataFrame({
    "study_id": study_ids,
    "findings_embedding": findings_embeds,
    "impression_embedding": impression_embeds
})

db_rpt.create_table("CheXagent", df_chex_rpt, mode="overwrite")
print("Saved CheXagent report embeddings table to LanceDB.")