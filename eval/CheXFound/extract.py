import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from chexfound.eval.setup import setup_and_build_model
from chexfound.data.transforms import make_classification_eval_transform

os.chdir('.') 

img_path = sys.argv[1]
out_path = sys.argv[2]

base_dir = './checkpoints/' 
config_file = base_dir + 'config.yaml'
pretrained_weights = base_dir + 'teacher_checkpoint.pth'
output_dir = base_dir + 'example'
os.makedirs(output_dir, exist_ok=True)

parser = argparse.ArgumentParser()
parser.set_defaults(
    config_file=config_file,  
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
model.eval()

# set up data transform
eval_transform = make_classification_eval_transform(resize_size=args.image_size, crop_size=args.image_size)

# Read images        
img = Image.open(img_path).convert(mode="RGB")
img = eval_transform(img)

# Get patch tokens from the backbone
with torch.no_grad():
    features = model.get_intermediate_layers(
        img.cuda().unsqueeze(0),
        n=args.n_last_blocks,
        return_class_token=args.return_class_token,
    )

final_block_output = features[-1]
image_embedding = final_block_output[1]

np.save(out_path, image_embedding.cpu().numpy())