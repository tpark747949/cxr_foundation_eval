import sys
from pathlib import Path

# 1. Point Python to the cloned EVA-X subfolder
eva_x_dir = Path(__file__).parent / "EVA-X"
sys.path.append(str(eva_x_dir))

from eva_x import eva_x_base_patch16
import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

model = eva_x_base_patch16(pretrained="checkpoints/eva_x_base_patch16_merged520k_mim.pt")

# 1. Define the exact input transformation pipeline
transform = T.Compose([
    T.Resize((224, 224)),                   # Resize the X-ray to match ViT patch limits
    T.ToTensor(),                           # Convert 0-255 integers to 0.0-1.0 float32
    T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x[:3, :, :]), 
                                            # Essential: Force Grayscale X-rays to 3 Channels (RGB)
    T.Normalize(
        mean=[0.485, 0.456, 0.406],         # Standard ImageNet mean
        std=[0.229, 0.224, 0.225]           # Standard ImageNet standard deviation
    )
])

# 2. Load your raw X-ray image file
image_path = "../../data/sample_images/sample1.jpg"
raw_image = Image.open(image_path)

# 3. Transform the image and prepare the dimensions
input_tensor = transform(raw_image)          # Gives a shape of [3, 224, 224]
input_tensor = input_tensor.unsqueeze(0)    # Add batch dimension -> [1, 3, 224, 224]

# 4. Migrate the image data to your NVCC 12.6 GPU
input_tensor = input_tensor.to("cuda")

# Ensure the model is evaluating, not training
model = model.to("cuda")
model.eval()

with torch.no_grad(): # Bypasses backpropagation calculation to save VRAM
    # Extract the foundational visual embeddings
    features = model(input_tensor)
    
print("Output feature representation shape:", features.shape)

features = features.cpu()
image_embedding = features / features.norm(p=2, dim=-1, keepdim=True)  # Normalize the embedding

plt.figure(figsize=(12, 4))
for vector in image_embedding.numpy():
    plt.plot(vector)
plt.title('Embedding Vectors')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)

# Saves the plot as an image file in your current folder
plt.savefig('embedding_plot.png', dpi=300, bbox_inches='tight')
plt.close()  # Cleans up memory