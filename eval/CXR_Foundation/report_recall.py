import tensorflow as tf
import tensorflow_hub as tf_hub
import numpy as np
import lancedb
import pyarrow as pa
import re
from pathlib import Path
import tensorflow_text as tf_text

# Force TensorFlow to use CPU only
tf.config.set_visible_devices([], 'GPU')

REPORTS_PATH = "../../data/MIMIC-CXR"
URI2 = "../../embeddings/MIMIC-CXR-JPG"
URI = "../../embeddings/reports"
TABLE_NAME = "CXR_Foundation"

# IMPORTANT: Update '1024' to match ELIXR's output dimension (e.g., 128 or 768)
EMBEDDING_DIM = 128 

SCHEMA = pa.schema([
    pa.field("study_id", pa.uint32()),   
    pa.field("subject_id", pa.uint32()),
    pa.field("findings_embedding", pa.list_(pa.float32(), EMBEDDING_DIM)),
    pa.field("impression_embedding", pa.list_(pa.float32(), EMBEDDING_DIM)),
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

# --- TENSORFLOW ELIXR INFERENCE SETUP ---
print("Loading TF Hub Preprocessor and ELIXR Model...")
preprocessor = tf_hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
qformer_model = tf.saved_model.load("./checkpoints/hf/pax-elixr-b-text")

def bert_tokenize(text):
    """Tokenizes input text and returns token IDs and padding masks."""
    out = preprocessor(tf.constant([text.lower()]))
    ids = out['input_word_ids'].numpy().astype(np.int32)
    masks = out['input_mask'].numpy().astype(np.float32)
    
    paddings = 1.0 - masks
    end_token_idx = ids == 102
    ids[end_token_idx] = 0
    paddings[end_token_idx] = 1.0
    
    ids = np.expand_dims(ids, axis=1)
    paddings = np.expand_dims(paddings, axis=1)
    return ids, paddings

def get_elixr_embeddings(text_list):
    """Helper function to loop through texts and extract normalized ELIXR embeddings."""
    embeddings = []
    for text in text_list:
        tokens, paddings = bert_tokenize(text)
        
        qformer_input = {
            'image_feature': np.zeros([1, 8, 8, 1376], dtype=np.float32).tolist(),
            'ids': tokens.tolist(),
            'paddings': paddings.tolist(),
        }
        
        qformer_output = qformer_model.signatures['serving_default'](**qformer_input)
        emb = qformer_output['contrastive_txt_emb'].numpy()[0]
        
        # L2 Normalize the embedding (standard practice for contrastive models)
        emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)
        embeddings.append(emb.tolist())
        
    return embeddings

# Pass a dummy blank string " " to the tokenizer if a section is missing so it doesn't crash
findings_texts = [r["findings"] if r["findings"] else " " for r in reports]
impression_texts = [r["impression"] if r["impression"] else " " for r in reports]

print("Generating Findings embeddings...")
f_embeddings_np = get_elixr_embeddings(findings_texts)

print("Generating Impression embeddings...")
i_embeddings_np = get_elixr_embeddings(impression_texts)

print("Connecting to LanceDB...")
db = lancedb.connect(URI)
table = db.create_table(TABLE_NAME, schema=SCHEMA, mode="overwrite")

records = []
for idx, r in enumerate(reports):
    record = {
        "study_id": r["study"],
        "subject_id": r["patient"],
        # Insert actual embedding if text exists, otherwise None
        "findings_embedding": f_embeddings_np[idx] if r["findings"] else None,
        "impression_embedding": i_embeddings_np[idx] if r["impression"] else None
    }
    records.append(record)

print(f"Writing {len(records)} records to LanceDB...")
table.add(records)
print("Done!")