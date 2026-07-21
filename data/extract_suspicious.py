import pandas as pd
import argparse

def export_suspicious_paths(input_csv, output_txt, n_per_category):
    print(f"Loading metrics from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Filter out files that were physically corrupted and couldn't be opened at all
    if "corrupted" in df.columns:
        df = df[df["corrupted"] == False]

    suspicious_paths = set()

    # 1. Collimation Errors / Half-black (Lowest min_half_mean)
    suspicious_paths.update(df.nsmallest(n_per_category, "min_half_mean")["path"].tolist())

    # 2. Blank Scans / Dead Pixels (Lowest overall_std)
    suspicious_paths.update(df.nsmallest(n_per_category, "overall_std")["path"].tolist())

    # 3. Underexposed (Lowest overall_mean)
    suspicious_paths.update(df.nsmallest(n_per_category, "overall_mean")["path"].tolist())

    # 4. Overexposed (Highest overall_mean)
    suspicious_paths.update(df.nlargest(n_per_category, "overall_mean")["path"].tolist())

    # 5. Photometric Inversions (Highest corner_mean)
    suspicious_paths.update(df.nlargest(n_per_category, "corner_mean")["path"].tolist())

    # Write the unique paths to a text file
    with open(output_txt, "w") as f:
        for path in suspicious_paths:
            f.write(f"{path}\n")

    print(f"Successfully exported {len(suspicious_paths)} unique paths to {output_txt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export suspicious MIMIC-CXR image paths.")
    parser.add_argument("--input", type=str, default="./qc_metrics.csv", help="Path to the QC metrics CSV")
    parser.add_argument("--output", type=str, default="./suspicious_images.txt", help="Output text file")
    parser.add_argument("--n", type=int, default=100, help="Number of outliers to pull per category")
    
    args = parser.parse_args()
    
    export_suspicious_paths(args.input, args.output, args.n)