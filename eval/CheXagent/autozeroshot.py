from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import torch
import lancedb
import pyarrow as pa
import torch.nn.functional as F

REF_URI = "../../embeddings/MIMIC-CXR-JPG"
URI = "../../embeddings/phrases"

DATA_DIR = "../../data/MIMIC-CXR-JPG/2.1.0/"

CHECKPOINT = "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli"

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

# Exact 13 CheXpert categories mapped to structurally balanced pairs
DISEASE_PAIRS = {
    "Atelectasis": [
        "radiographic findings consistent with atelectasis",
        "no evidence of atelectasis, lungs are fully inflated"
    ],
    "Cardiomegaly": [
        "radiographic findings consistent with cardiomegaly, enlarged heart silhouette",
        "normal cardiac silhouette size, normal heart size"
    ],
    "Consolidation": [
        "radiographic findings consistent with lung consolidation",
        "no evidence of consolidation, lungs are clear"
    ],
    "Edema": [
        "radiographic findings consistent with pulmonary edema, vascular congestion",
        "no evidence of pulmonary edema, lung fields are clear"
    ],
    "Enlarged Cardiomediastinum": [
        "radiographic findings consistent with an enlarged cardiomediastinum",
        "normal cardiomediastinal contour, no mediastinal widening"
    ],
    "Fracture": [
        "radiographic findings consistent with a fracture",
        "no evidence of acute fracture, intact bony structures"
    ],
    "Lung Lesion": [
        "radiographic findings consistent with a lung lesion, nodule, or mass",
        "no evidence of pulmonary nodules or lung masses"
    ],
    "Lung Opacity": [
        "radiographic findings consistent with lung opacity",
        "no evidence of abnormal lung opacities, lung fields are lucent"
    ],
    "Pleural Effusion": [
        "radiographic findings consistent with pleural effusion",
        "sharp costophrenic angles, no evidence of pleural effusion"
    ],
    "Pleural Other": [
        "radiographic findings consistent with pleural thickening or fibrocalcific scarring",
        "normal pleural space, no pleural thickening"
    ],
    "Pneumonia": [
        "radiographic findings consistent with pneumonia, infectious infiltrate",
        "no evidence of focal pneumonia or infectious infiltrate"
    ],
    "Pneumothorax": [
        "radiographic findings consistent with pneumothorax, pleural air",
        "no evidence of a pneumothorax, lung apex is intact"
    ],
    "Support Devices": [
        "presence of support devices, tubes, lines, or hardware",
        "no support devices, medical lines, or tubes present"
    ],
    "No Finding": [
        "normal chest x-ray, clear lung fields, no acute cardiopulmonary abnormalities",
        "abnormal radiographic findings, presence of acute cardiopulmonary disease"
    ],
}


texts = []
for disease in DISEASES:
    pos_phrase, neg_phrase = DISEASE_PAIRS[disease]
    texts.append(pos_phrase)
    texts.append(neg_phrase)

# Should be 28 phrases long
print("Number of phrases:", len(texts))

ref_db = lancedb.connect(REF_URI)
ref = ref_db.open_table("complete_embeddings_MIMIC-CXR-JPG")
# Filter rows and select only the two columns
ref_df = ref.search() \
        .where("split = 'test' AND ignore = 0") \
        .select(["dicom_id", "path"]) \
        .to_pandas()

dicom_ids = ref_df["dicom_id"].tolist()
paths = ref_df["path"].tolist()
paths = [f"{DATA_DIR}{path}" for path in paths]
print(f"Loaded {len(paths)} image paths")


clean_label_cols = [col.replace(' ', '_') for col in DISEASES]
schema = pa.schema([
    pa.field("path", pa.string()),
    pa.field("dicom_id", pa.string()),
    # CHANGED: int8() to float32() to store continuous scores
    pa.field("prediction", pa.struct([pa.field(col, pa.float32()) for col in clean_label_cols])),
])

device = "cpu"

model = AutoModelForZeroShotImageClassification.from_pretrained(CHECKPOINT)
model = model.to(device)

processor = AutoProcessor.from_pretrained(CHECKPOINT)

images = paths
candidate_labels = texts

inputs = processor(images=images, text=candidate_labels, return_tensors="pt", padding=True)


with torch.no_grad():
    outputs = model(**inputs)

print("Finished inference")

# 1. Bypass any automatic wrapper scaling by calculating similarities directly
# This gives you the pure, untampered logit matrix for SigLIP
image_embeds = outputs.image_embeds.cpu()
text_embeds = outputs.text_embeds.cpu()

# 2. Re-apply SigLIP's native scale factor and bias parameters
logit_scale = model.config.logit_scale if hasattr(model.config, 'logit_scale') else 1.0
logit_bias = model.config.logit_bias if hasattr(model.config, 'logit_bias') else 0.0

image_embeds = F.normalize(image_embeds, p=2, dim=-1)
text_embeds = F.normalize(text_embeds, p=2, dim=-1)

# Calculate raw pairwise similarities
logits = (image_embeds @ text_embeds.T) * logit_scale + logit_bias

db = lancedb.connect(URI)
table_name = "CheXagent"
table = db.create_table(table_name, schema=schema, mode="overwrite")

batch_record = []

for i, logits in enumerate(logits):

    # logits = logits[i]  # Grab the first image's array (length 28)

    # 3. Reshape the RAW logits into pairs first (14 diseases, 2 prompts each)
    # This keeps the positive and negative prompts correctly aligned 
    paired_logits = logits.reshape(len(DISEASES), 2)

    # 4. Use Softmax here if you want them to directly compete (adds up to 1.0)
    # Or use torch.sigmoid(paired_logits) if you want independent true/false probabilities
    # Change Option 4 from Softmax to Sigmoid
    # probs = torch.sigmoid(paired_logits).cpu().numpy()
    probs = paired_logits.softmax(dim=-1).numpy()

    pred = {}
    for j, disease in enumerate(clean_label_cols):
        pos_score, neg_score = probs[j]
        
        # Save the differential continuous score (ranges from -1.0 to 1.0)
        pred[f"{disease}"] = float(pos_score - neg_score)

    record = {
        "path": paths[i],
        "dicom_id": dicom_ids[i],
        "prediction": pred
    }

    batch_record.append(record)
    record = {}

table.add(batch_record)