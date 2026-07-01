import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from huggingface_hub import hf_hub_download
from PIL import Image
import matplotlib.pyplot as plt


# 1. Download the raw .pt checkpoint directly from the Hugging Face repo
checkpoint_path = hf_hub_download(
    repo_id="microsoft/BiomedVLP-BioViL-T", 
    filename="biovil_t_image_model_proj_size_128.pt"
)

# 2. Build the underlying ResNet50 vision encoder block
class BioViLTVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # BioViL-T uses a standard ResNet50 base structure
        self.backbone = models.resnet50()
        num_features = self.backbone.fc.in_features
        
        # Replace classification layer with BioViL-T's custom 128-dim projector
        self.backbone.fc = nn.Identity() 
        self.projector = nn.Linear(num_features, 128)

    def forward(self, x):
        # Emulate the expected custom outputs object schema 
        features = self.backbone(x)
        projected = self.projector(features)
        
        # Create an anonymous structural container to match .projected_global_embedding
        class OutputContainer:
            def __init__(self, embedding):
                self.projected_global_embedding = embedding
        return OutputContainer(projected)

# 3. Initialize and load the model weights cleanly
image_model = BioViLTVisionModel()
state_dict = torch.load(checkpoint_path, map_location="cpu")

# Strip out nesting prefixes if present in checkpoint
clean_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
image_model.load_state_dict(clean_state_dict, strict=False)
image_model.eval()

# 4. Standard preprocessing (required 480x480 crop)
transform = T.Compose([
    T.Resize(512),
    T.CenterCrop(480),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 5. Extract image embedding
image = Image.open("../../data/sample_images/sample1.jpg").convert("RGB")
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    outputs = image_model(image_tensor)

image_embedding = outputs.projected_global_embedding
image_embedding = image_embedding / image_embedding.norm(p=2, dim=-1, keepdim=True)
print("Embedding Shape:", image_embedding.shape)
print(image_embedding)  # Optional: print the actual embedding values
# Output: torch.Size([1, 128])

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