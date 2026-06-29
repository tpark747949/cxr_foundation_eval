from transformers import AutoProcessor, SiglipVisionModel, AutoModel
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.image import resize as tf_resize
import tensorflow as tf
import numpy as np

tf.config.set_visible_devices([], 'GPU')

img = Image.open("../../data/sample1.jpg")
print(img.height, img.width)

INPUT_DIMENSION = [448, 448]


def resize(image):
    # 1. Convert to numpy array in case it's currently a PIL Image
    img_array = np.array(image)
    
    # 2. If the image is 2D (H, W), add a channel dimension -> (H, W, 1)
    if len(img_array.shape) == 2:
        img_array = np.expand_dims(img_array, axis=-1)
        
    # 3. Perform the TensorFlow resize (it now sees 3 dimensions)
    resized_arr = tf_resize(
        images=img_array, size=INPUT_DIMENSION, method='bilinear', antialias=False
    ).numpy().astype(np.uint8)
    
    # 4. Strip the channel dimension back out so PIL accepts it as grayscale -> (H, W)
    if resized_arr.shape[-1] == 1:
        resized_arr = np.squeeze(resized_arr, axis=-1)
        
    return Image.fromarray(resized_arr)

img = resize(img)

model = SiglipVisionModel.from_pretrained("google/medsiglip-448")
processor = AutoProcessor.from_pretrained("google/medsiglip-448")

inputs = processor(images=img, padding="max_length", return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

output_embeddings = outputs["pooler_output"] / outputs["pooler_output"].norm(p=2, dim=-1, keepdim=True)
print("Size of embedding vector:", output_embeddings.size()[1])

plt.figure(figsize=(12, 4))
for vector in output_embeddings.numpy():
    plt.plot(vector)
plt.title('Embedding Vectors')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)

# Saves the plot as an image file in your current folder
plt.savefig('embedding_plot.png', dpi=300, bbox_inches='tight')
plt.close()  # Cleans up memory