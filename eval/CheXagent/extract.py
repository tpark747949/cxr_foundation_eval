import sys
import torch
from transformers import AutoProcessor, SiglipVisionModel
from PIL import Image
import numpy as np

img_path = sys.argv[1]
out_path = sys.argv[2]

img = Image.open(img_path)

model = SiglipVisionModel.from_pretrained("StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli")
processor = AutoProcessor.from_pretrained("StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli")

inputs = processor(images=img, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

output_embeddings = outputs["pooler_output"] / outputs["pooler_output"].norm(p=2, dim=-1, keepdim=True)
np.save(out_path, output_embeddings.numpy())