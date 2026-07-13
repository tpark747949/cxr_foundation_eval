import os
os.chdir('.')  # Change this path to the repository path
import json
import torch
import argparse
import pandas as pd
from PIL import Image
from chexfound.eval.setup import setup_and_build_model
from chexfound.data.transforms import make_classification_eval_transform
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from chexfound.eval.utils import extract_hyperparameters_from_model
from chexfound.eval.classification.utils import setup_glori
from fvcore.common.checkpoint import Checkpointer
from matplotlib.colors import ListedColormap

target_class = 'Cardiomegaly'
assert target_class in ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

base_dir = './checkpoints/'  # Change to the directory storing checkpoints and configuration files

config_file = base_dir + 'config.yaml'
pretrained_weights = base_dir + 'teacher_checkpoint.pth'
classifier_fpath = base_dir + 'glori.pth'
classifier_json = base_dir + 'results_eval_linear.json'
output_dir = base_dir + 'example'
os.makedirs(output_dir, exist_ok=True)

parser = argparse.ArgumentParser()

parser.set_defaults(
    config_file=config_file,  # path to architecture configuration files
    pretrained_weights=None,
    output_dir=output_dir,
    opts=[],
    image_size=512,
    patch_size=16,
    n_register_tokens=4,
    n_last_blocks=4,
    return_class_token=True,
    num_classes=40,
    num_heads=8,
)
args, unknown = parser.parse_known_args()

# set up foundation model
model, autocast_dtype = setup_and_build_model(args)

# load checkpoint for foundation model
state_dict = torch.load(pretrained_weights)['teacher']
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('backbone'):
        ls = k.split('.')
        if 'blocks' in k:
            new_k = '.'.join([ls[1], *ls[3:]])
        else:
            new_k = '.'.join(ls[1:])
    else:
        new_k = k
    new_state_dict.update({new_k: v})

model.load_state_dict(new_state_dict, strict=False)


# heads and size
nh = args.num_heads
h_map = args.image_size // args.patch_size
w_map = h_map

# set up data transform
eval_transform = make_classification_eval_transform(resize_size=args.image_size, crop_size=args.image_size)

# Read images        
img_path = '../../data/sample_images/sample1.jpg'

img = Image.open(img_path)
img = img.convert(mode="RGB")
img = eval_transform(img)

# Get patch tokens from the backbone
with torch.no_grad():
    features = model.get_intermediate_layers(
        img.cuda().unsqueeze(0),
        n=args.n_last_blocks,
        return_class_token=args.return_class_token,
    )

# 1. Get the output from the final Transformer block (the last item in the outer tuple)
final_block_output = features[-1]

# 2. Extract the [CLS] token (the second item in the inner tuple)
image_embedding = final_block_output[1]

print(image_embedding.shape) 
print(image_embedding)
# Expected output: torch.Size([1, 1024])