import os
import shutil
import numpy as np
from pathlib import Path

# --- Configuration Lists ---
MODELS = ["MedSigLIP", "BioViL-T", "EVA-X", "CheXFound", "CheXagent", "CXR_Foundation", "Early_Fusion"]
HEADS = ["LR", "XGB", "s2", "s4", "i2", "i4"]
LABELS = ["CheXpert", "NegBio", "1pct", "5pct", "10pct"]
VARS = ["raw", "l2", "pca95"]

DEST_DIR = Path("test_probs")

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def guess_attributes(path_str, filename):
    """Attempts to guess the parameters based on the filepath to speed up user input."""
    path_str = path_str.lower()
    
    # Guess Model
    guess_model = next((m for m in MODELS if m.lower() in filename.lower()), MODELS[0])
    
    # Guess Labeller
    guess_label = "NegBio" if "negbio" in path_str else "CheXpert"
    if "1p" in path_str: guess_label = "1pct"
    elif "5p" in path_str: guess_label = "5pct"
    elif "10p" in path_str: guess_label = "10pct"
    
    # Guess Var
    guess_var = "raw"  # Default to raw
    if "raw" in path_str: guess_var = "raw"
    elif "l2" in path_str: guess_var = "l2"
    elif "pca95" in path_str: guess_var = "pca95"
    elif "pca_95" in path_str: guess_var = "pca95"
    
    # Guess Head
    guess_head = HEADS[0]
    if any(x in path_str for x in ("xgboost", "extreme_gradient")): guess_head = "XGB"
    elif any(x in path_str for x in ("lr", "logistic")): guess_head = "LR"
    elif "shared_4class" in path_str: guess_head = "s4"
    elif any(x in path_str for x in ("shared_mlp", "shared_binary", "mlp_grid")): guess_head = "s2"
    elif "independent_4class" in path_str: guess_head = "i4"
    elif any(x in path_str for x in ("independent_mlp", "independent_binary")): guess_head = "i2"

        
    return guess_model, guess_head, guess_label, guess_var

def get_menu_choice(prompt, options, default_val):
    """Displays a numbered menu and returns the selected string."""
    default_idx = options.index(default_val) + 1 if default_val in options else 1
    
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        indicator = "*" if i == default_idx else " "
        print(f"  [{i}] {indicator} {opt}")
        
    while True:
        choice = input(f"Select 1-{len(options)} (Press Enter for default [{default_idx}]): ").strip()
        if choice == "":
            return options[default_idx - 1]
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid choice, please try again.")

def main():
    DEST_DIR.mkdir(exist_ok=True)
    
    print("Scanning exp1 and exp2 for numpy arrays...")
    all_files = list(Path("../exp1").resolve().rglob("*.npy")) + list(Path("../exp2/artifacts").resolve().rglob("*.npy"))    

    # Filter out ground truths, validation arrays, and hidden virtualenvs
    valid_files = []
    for f in all_files:
        if "y_" in f.name or "val" in f.name: continue
        if any(p.startswith(".") for p in f.parts): continue
        valid_files.append(f)
        
    print(f"Found {len(valid_files)} potential prediction files.\n")
    input("Press Enter to begin interactive renaming...")

    processed_count = 0
    skipped_count = 0

    for path in valid_files:
        clear_screen()
        
        try:
            # We use mmap_mode='r' to read the shape quickly without loading giant arrays into RAM
            arr = np.load(path, mmap_mode='r')
            shape = arr.shape
        except Exception as e:
            print(f"Error reading {path.name}: {e}. Skipping.")
            skipped_count += 1
            continue
            
        # Hard fail-safe: The array MUST have 996 rows to be the test set. 
        if shape[0] != 996:
            print(f"Skipping {path.name} (Shape {shape} does not match 996 test samples).")
            skipped_count += 1
            continue

        print("="*60)
        print(f"FILE: {path}")
        print(f"SHAPE: {shape}")
        print("="*60)
        
        action = input("\n[1] Process this file\n[2] Skip this file\nSelect (1/2): ").strip()
        if action == "2":
            skipped_count += 1
            continue

        # Generate intelligent defaults
        g_mod, g_head, g_lab, g_var = guess_attributes(str(path), path.name)

        model = get_menu_choice("Foundation Model:", MODELS, g_mod)
        head = get_menu_choice("Classifier Head:", HEADS, g_head)
        labeler = get_menu_choice("Labeller:", LABELS, g_lab)
        var = get_menu_choice("Variable (Data %):", VARS, g_var)

        new_filename = f"{model}_{head}_{labeler}_{var}.npy"
        dest_path = DEST_DIR / new_filename

        print("\n" + "-"*40)
        print(f"SOURCE:      {path.name}")
        print(f"DESTINATION: {new_filename}")
        print("-"*40)
        
        confirm = input("\nCopy with this name? (Y/n/edit): ").strip().lower()
        
        if confirm == 'n':
            print("Skipped.")
            skipped_count += 1
        elif confirm == 'edit':
            custom_name = input("Enter custom filename (including .npy): ").strip()
            shutil.copy2(path, DEST_DIR / custom_name)
            processed_count += 1
            print(f"Copied as {custom_name}")
        else:
            shutil.copy2(path, dest_path)
            processed_count += 1
            print("Copied successfully.")
            
    clear_screen()
    print("=== FINISHED ===")
    print(f"Successfully mapped & copied: {processed_count}")
    print(f"Skipped/Ignored: {skipped_count}")
    print(f"Files are located in: {DEST_DIR.resolve()}")

if __name__ == "__main__":
    main()