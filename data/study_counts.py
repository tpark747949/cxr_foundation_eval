import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset directly from the gzipped CSV
# pandas handles gzip decompression automatically
df = pd.read_csv('MIMIC-CXR-JPG/2.1.0/mimic-cxr-2.0.0-split.csv.gz')

# 2. Calculate Counts
# Images per study: Count how many dicom_ids exist for each study_id
images_per_study = df.groupby('study_id').size()

# Studies per patient: Count unique study_ids for each subject_id
studies_per_patient = df.groupby('subject_id')['study_id'].nunique()

# Images per patient: Count how many dicom_ids exist for each subject_id
images_per_patient = df.groupby('subject_id').size()

print(f"Total Patients: {len(studies_per_patient)}")
print(f"Total Studies: {len(images_per_study)}")
print(f"Total Images: {len(df)}")

# 3. Visualization
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(21, 5))

# Plot A: Images per Study
sns.histplot(images_per_study, bins=range(1, images_per_study.max() + 2), 
             ax=axes[0], discrete=True, color='steelblue')
axes[0].set_title('Distribution of Images per Study')
axes[0].set_xlabel('Number of Images')
axes[0].set_ylabel('Count of Studies (Log Scale)')
axes[0].set_yscale('log') # Log scale because most studies have exactly 1 or 2 images

# Plot B: Studies per Patient
sns.histplot(studies_per_patient, bins=50, ax=axes[1], color='darkorange')
axes[1].set_title('Distribution of Studies per Patient')
axes[1].set_xlabel('Number of Studies')
axes[1].set_ylabel('Count of Patients (Log Scale)')
axes[1].set_yscale('log') # Log scale for the "frequent flyer" long tail

# Plot C: Images per Patient
sns.histplot(images_per_patient, bins=50, ax=axes[2], color='green')
axes[2].set_title('Distribution of Images per Patient')
axes[2].set_xlabel('Number of Images')
axes[2].set_ylabel('Count of Patients (Log Scale)')
axes[2].set_yscale('log') # Log scale for the "frequent flyer" long tail

output_path = "images_per_study_and_patient.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Histogram successfully saved to {output_path}")