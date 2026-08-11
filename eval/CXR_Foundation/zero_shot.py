import torch
import numpy as np
import lancedb
import pyarrow as pa
import tensorflow as tf
import tensorflow_text as tf_text
import tensorflow_hub as tf_hub

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
TABLE_NAME = "CXR_Foundation"
SCHEMA = pa.schema([
    pa.field("disease", pa.string()),
    pa.field("positive_embedding", pa.list_(pa.float32(), 128)),
    pa.field("negative_embedding", pa.list_(pa.float32(), 128)),
])
# this should be like each row is a disease, and the columns are the embeddings for the positive and negative phrases from each model

# Download the model repository files
from huggingface_hub import snapshot_download
snapshot_download(repo_id="google/cxr-foundation",local_dir='./checkpoints/hf',
                  allow_patterns=['elixr-c-v2-pooled/*', 'pax-elixr-b-text/*'])

if 'qformer_model' not in locals():
  qformer_model = tf.saved_model.load("./checkpoints/hf/pax-elixr-b-text")

texts = []
for disease in DISEASES:
    disease = disease.lower()
    pos_phrase = f"{disease} is present"
    neg_phrase = f"no {disease}"
    texts.append(pos_phrase)
    texts.append(neg_phrase)
print("Number of phrases:", len(texts))

# Helper function for tokenizing text input
def bert_tokenize(text):
    """Tokenizes input text and returns token IDs and padding masks."""
    preprocessor = tf_hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    out = preprocessor(tf.constant([text.lower()]))
    ids = out['input_word_ids'].numpy().astype(np.int32)
    masks = out['input_mask'].numpy().astype(np.float32)
    paddings = 1.0 - masks
    end_token_idx = ids == 102
    ids[end_token_idx] = 0
    paddings[end_token_idx] = 1.0
    ids = np.expand_dims(ids, axis=1)
    paddings = np.expand_dims(paddings, axis=1)
    assert ids.shape == (1, 1, 128)
    assert paddings.shape == (1, 1, 128)
    return ids, paddings

output_embeddings = []
for text in texts:
    tokens, paddings = bert_tokenize(text)
    qformer_input = {
        'image_feature': np.zeros([1, 8, 8, 1376], dtype=np.float32).tolist(),
        'ids': tokens.tolist(),
        'paddings': paddings.tolist(),
    }
    qformer_output = qformer_model.signatures['serving_default'](**qformer_input)
    text_embeddings = qformer_output['contrastive_txt_emb'].numpy()
    # h_embeddings = text_embeddings.flatten().numpy()
    output_embeddings.append(text_embeddings[0])
    print(text_embeddings[0].shape)
    # print(text, text_embeddings, np.linalg.norm(text_embeddings, axis=-1), text_embeddings.shape)


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
        "positive_embedding": post_embedding,
        "negative_embedding": neg_embedding,
    }
    records.append(record)

print(f"Adding {len(records)} records to the table...")
table.add(records)