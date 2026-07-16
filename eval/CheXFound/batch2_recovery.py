import os
import torch
import lancedb
from PIL import Image
from argparse import Namespace

# Import your custom modules
from chexfound.eval.setup import setup_and_build_model
from chexfound.data.transforms import make_classification_eval_transform

# ==========================================
# 0. CONFIGURATION
# ==========================================
DATA_DIR = "../../data/MIMIC-CXR-JPG/2.1.0"
URI = "../../embeddings/MIMIC-CXR-JPG"
BASE_DIR = './checkpoints/'

TARGET_GPU = 0

CHEXFOUND_ARGS = Namespace(
    config_file=os.path.join(BASE_DIR, 'config.yaml'),
    pretrained_weights=os.path.join(BASE_DIR, 'teacher_checkpoint.pth'),
    output_dir=os.path.join(BASE_DIR, 'example'),
    opts=[],
    image_size=512,
    patch_size=16,
    n_register_tokens=4,
    n_last_blocks=4,
    return_class_token=True,
    num_classes=40,
    num_heads=8,
)

BAD_DICOMS = [
    "0539ee33-9d402e49-a9cc6d36-7aabc539-3d80a62b",
    "14a5423b-9989fc33-123ce6f1-4cc7ca9a-9a3d2179"
]

# ==========================================
# 1. PATCH OPERATION
# ==========================================
if __name__ == '__main__':
    # 1. Connect to Database and Fetch Records
    db = lancedb.connect(URI)
    table = db.open_table("CheXfound_MIMIC")
    
    # Format for SQL-like query
    dicom_list_str = ", ".join([f"'{d}'" for d in BAD_DICOMS])
    query = f"dicom_id IN ({dicom_list_str})"
    
    print(f"Querying LanceDB for DICOM IDs...")
    # Extracting as PyArrow and converting to standard Python dicts
    records = table.search().where(query).to_arrow().to_pylist()
    
    if not records:
        print("No matching records found. Double check the DICOM IDs or database URI.")
        exit()
        
    print(f"Found {len(records)} records. Initializing model in pure float32...")

    # 2. Setup Device and Model
    torch.cuda.set_device(TARGET_GPU)
    device = torch.device(f"cuda:{TARGET_GPU}")
    
    eval_transform = make_classification_eval_transform(
        resize_size=CHEXFOUND_ARGS.image_size, crop_size=CHEXFOUND_ARGS.image_size
    )

    model, _ = setup_and_build_model(CHEXFOUND_ARGS)
    state_dict = torch.load(CHEXFOUND_ARGS.pretrained_weights, map_location="cpu")['teacher']
    
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
    # Force model to float32 just to be absolutely certain
    model = model.to(device, dtype=torch.float32)
    model.eval()

    updated_records = []
    
    # 3. Inference Loop (No Autocast)
    with torch.no_grad():
        for record in records:
            dicom_id = record["dicom_id"]
            img_path = os.path.join(DATA_DIR, record["path"])
            split = record["split"]
            
            print(f"\nProcessing {dicom_id}...")
            print(f"-> [INFO] Loading image from: {img_path}")
            print(f"-> [INFO] Split: {split}")
            
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = eval_transform(img).unsqueeze(0).to(device, dtype=torch.float32)
            except Exception as e:
                print(f"Failed to load image file {img_path}: {e}")
                continue

            # Run inference
            features = model.get_intermediate_layers(
                img_tensor, n=CHEXFOUND_ARGS.n_last_blocks, return_class_token=CHEXFOUND_ARGS.return_class_token
            )
            
            raw_outs = features[-1][1]
            
            # Check for NaN again
            if torch.isnan(raw_outs).any() or torch.isinf(raw_outs).any():
                print(f"-> [RESULT] Even pure float32 produced a NaN. This image file is mathematically unprocessable. Leaving as zero-vector.")
                raw_outs = torch.zeros_like(raw_outs)
            else:
                print(f"-> [RESULT] Success! float32 bypassed the overflow.")
                
            l2_outs = raw_outs / (raw_outs.norm(p=2, dim=-1, keepdim=True) + 1e-8)
            
            # Update the dictionary with the new arrays
            record["embedding_raw"] = raw_outs.cpu().numpy()[0].tolist()
            record["embedding_l2"] = l2_outs.cpu().numpy()[0].tolist()
            updated_records.append(record)

    # 4. Database Overwrite
    if updated_records:
        print("\nDeleting old corrupted rows from LanceDB...")
        table.delete(query)
        
        print("Inserting recalculated rows...")
        table.add(updated_records)
        
        print("Patch complete! Database is clean.")