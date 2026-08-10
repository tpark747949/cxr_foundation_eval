# ~/cxr_foundation_eval/MedSigLIP/extract.py
import sys
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, SiglipVisionModel
from tensorflow.image import resize as tf_resize
import tensorflow as tf

# Command line arguments
img_path = sys.argv[1]
out_path = sys.argv[2]

INPUT_DIMENSION = [448, 448]

def resize(image):
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_array = np.expand_dims(img_array, axis=-1)
    resized_arr = tf_resize(
        images=img_array, size=INPUT_DIMENSION, method='bilinear', antialias=False
    ).numpy().astype(np.uint8)
    if resized_arr.shape[-1] == 1:
        resized_arr = np.squeeze(resized_arr, axis=-1)
    return Image.fromarray(resized_arr)

# Load and process
img = resize(Image.open(img_path))
model = SiglipVisionModel.from_pretrained("google/medsiglip-448")
processor = AutoProcessor.from_pretrained("google/medsiglip-448")

inputs = processor(images=img, padding="max_length", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Extract, normalize, and save
output_embeddings = outputs["pooler_output"] / outputs["pooler_output"].norm(p=2, dim=-1, keepdim=True)
np.save(out_path, output_embeddings.numpy())