import sys
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np

img_path = sys.argv[1]
out_path = sys.argv[2]

# 1. Point Python to the cloned EVA-X subfolder
eva_x_dir = Path(__file__).parent / "EVA-X"
sys.path.append(str(eva_x_dir))

from eva_x import eva_x_base_patch16

model = eva_x_base_patch16(pretrained="checkpoints/eva_x_base_patch16_merged520k_mim.pt")

# 2. Define the exact input transformation pipeline
transform = T.Compose([
    T.Resize((224, 224)),                   
    T.ToTensor(),                           
    T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x[:3, :, :]), 
    T.Normalize(
        mean=[0.485, 0.456, 0.406],         
        std=[0.229, 0.224, 0.225]           
    )
])

# 3. Transform the image and prepare the dimensions
raw_image = Image.open(img_path)
input_tensor = transform(raw_image).unsqueeze(0)   
input_tensor = input_tensor.to("cuda")

model = model.to("cuda")
model.eval()

with torch.no_grad(): 
    features = model(input_tensor)
    
features = features.cpu()
image_embedding = features / features.norm(p=2, dim=-1, keepdim=True)  

np.save(out_path, image_embedding.numpy())