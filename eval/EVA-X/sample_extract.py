import sys
from pathlib import Path

# 1. Point Python to the cloned EVA-X subfolder
eva_x_dir = Path(__file__).parent / "EVA-X"
sys.path.append(str(eva_x_dir))

from eva_x import eva_x_tiny_patch16, eva_x_small_patch16, eva_x_base_patch16

print("all good.")

# model = eva_x_small_patch16(pretrained=/path/to/pre-trained)