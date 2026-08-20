from typing import List
from typing import Tuple
from pathlib import Path

import torch
from health_multimodal.common.visualization import plot_phrase_grounding_similarity_map
from health_multimodal.text import get_bert_inference
from health_multimodal.text.utils import BertEncoderType
from health_multimodal.image import get_image_inference
from health_multimodal.image.utils import ImageModelType
from health_multimodal.vlp import ImageTextInferenceEngine

import lancedb
import pyarrow as pa
from tqdm import tqdm

REF_URI = "../../embeddings/MIMIC-CXR-JPG"
URI = "../../embeddings/phrases"

DATA_DIR = "../../data/MIMIC-CXR-JPG/2.1.0/"

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
        "radiographic findings consistent with pleural thickening or other pleural abnormalities",
        "normal pleural space, no pleural thickening"
    ],
    "Pneumonia": [
        "radiographic findings consistent with pneumonia, infectious infiltrate",
        "no evidence of pneumonia or consolidation"
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

device = "cuda" if torch.cuda.is_available() else "cpu"

text_inference = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)
text_inference.model = text_inference.model.to(device)

text = texts

txt_output = text_inference.get_embeddings_from_prompt(texts, normalize=True)

txt_output = txt_output.cpu()

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

db = lancedb.connect(URI)
table_name = "BioViL-T"
table = db.create_table(table_name, schema=schema, mode="overwrite")

image_inference = get_image_inference(ImageModelType.BIOVIL_T)
image_inference.model = image_inference.model.to(device) 

print(f"Performing image inference on {len(paths)} images.")
img_output = []
for path in tqdm(paths, desc="Extracting embeddings"):
    image = Path(path)
    emb = image_inference.get_projected_global_embedding(image)
    img_output.append(emb.cpu())

batch_record = []


for i, img_emb in enumerate(img_output):
    similarities = img_emb @ txt_output.t()
    paired_logits = similarities.reshape(len(DISEASES), 2)
    probs = paired_logits.softmax(dim=-1).detach().numpy()

    pred = {}
    for u, disease in enumerate(clean_label_cols):
        pos_score, neg_score = probs[u]
        
        # Save the continuous probability of the positive phrase directly
        pred[f"{disease}"] = float(pos_score)

    record = {
        "path": paths[i],
        "dicom_id": dicom_ids[i],
        "prediction": pred
    }

    batch_record.append(record)
    record = {}

table.add(batch_record)