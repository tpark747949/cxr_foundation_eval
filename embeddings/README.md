IMPORTANT:
THERE ARE PROBLEMATIC IMAGES IN THE MIMIC DATASET
THEY ARE 90% BLACK, POSSIBLY DUE TO SCANNING ISSUES
MOST MODELS (EXCEPT FOR CHEXFOUND) 'SUCCESSFULLY' EXTRACTED EMBEDDINGS
THEY ARE THE FOLLOWING

dicom_id: 0539ee33-9d402e49-a9cc6d36-7aabc539-3d80a62b...
path: MIMIC-CXR-JPG/2.1.0/files/p10/p10291098/s57194260/0539ee33-9d402e49-a9cc6d36-7aabc539-3d80a62b.jpg
split: train


dicom_id: 14a5423b-9989fc33-123ce6f1-4cc7ca9a-9a3d2179...
path: MIMIC-CXR-JPG/2.1.0/files/p13/p13579794/s51003958/14a5423b-9989fc33-123ce6f1-4cc7ca9a-9a3d2179.jpg
split: train

Problematic images have been identified, 85 in fact. Ignore=1 for these images.

MIMIC-CXR-JPG complete embeddings schema
path: string
dicom_id: string
PerformedProcedureStepDescription: string
ViewPosition: string
image_size: struct<Rows: uint16, Columns: uint16>
  child 0, Rows: uint16
  child 1, Columns: uint16
StudyDate: string
StudyTime: string
ProcedureCodeSequence_CodeMeaning: string
ViewCodeSequence_CodeMeaning: string
PatientOrientationCodeSequence_CodeMeaning: string
study_id: uint32
subject_id: uint32
split: string
CheXpert_labels: struct<Atelectasis: int8, Cardiomegaly: int8, Consolidation: int8, Edema: int8, Enlarged_Cardiomedia (... 191 chars omitted)
  child 0, Atelectasis: int8
  child 1, Cardiomegaly: int8
  child 2, Consolidation: int8
  child 3, Edema: int8
  child 4, Enlarged_Cardiomediastinum: int8
  child 5, Fracture: int8
  child 6, Lung_Lesion: int8
  child 7, Lung_Opacity: int8
  child 8, Pleural_Effusion: int8
  child 9, Pneumonia: int8
  child 10, Pneumothorax: int8
  child 11, Pleural_Other: int8
  child 12, Support_Devices: int8
  child 13, No_Finding: int8
NegBio_labels: struct<Atelectasis: int8, Cardiomegaly: int8, Consolidation: int8, Edema: int8, Enlarged_Cardiomedia (... 191 chars omitted)
  child 0, Atelectasis: int8
  child 1, Cardiomegaly: int8
  child 2, Consolidation: int8
  child 3, Edema: int8
  child 4, Enlarged_Cardiomediastinum: int8
  child 5, Fracture: int8
  child 6, Lung_Lesion: int8
  child 7, Lung_Opacity: int8
  child 8, Pleural_Effusion: int8
  child 9, Pneumonia: int8
  child 10, Pneumothorax: int8
  child 11, Pleural_Other: int8
  child 12, Support_Devices: int8
  child 13, No_Finding: int8
MedSigLIP_raw: fixed_size_list<item: float>[1152]
  child 0, item: float
MedSigLIP_l2: fixed_size_list<item: float>[1152]
  child 0, item: float
CXR_Foundation_raw: fixed_size_list<item: float>[4096]
  child 0, item: float
CXR_Foundation_l2: fixed_size_list<item: float>[4096]
  child 0, item: float
BioViL-T_raw: fixed_size_list<item: float>[128]
  child 0, item: float
BioViL-T_l2: fixed_size_list<item: float>[128]
  child 0, item: float
EVA-X_raw: fixed_size_list<item: float>[768]
  child 0, item: float
EVA-X_l2: fixed_size_list<item: float>[768]
  child 0, item: float
CheXagent_raw: fixed_size_list<item: float>[1024]
  child 0, item: float
CheXagent_l2: fixed_size_list<item: float>[1024]
  child 0, item: float
CheXFound_raw: fixed_size_list<item: float>[1024]
  child 0, item: float
CheXFound_l2: fixed_size_list<item: float>[1024]
  child 0, item: float
cxr_similarity: float
cxr_similarity1: float
ignore: int8



exp3 pos/neg disease phrases embeddings schema

disease: string
MedSigLIP_positive_embedding: fixed_size_list<item: float>[1152]
  child 0, item: float
MedSigLIP_negative_embedding: fixed_size_list<item: float>[1152]
  child 0, item: float
CXR_Foundation_positive_embedding: fixed_size_list<item: float>[128]
  child 0, item: float
CXR_Foundation_negative_embedding: fixed_size_list<item: float>[128]
  child 0, item: float
BioViL_T_positive_embedding: fixed_size_list<item: float>[128]
  child 0, item: float
BioViL_T_negative_embedding: fixed_size_list<item: float>[128]
  child 0, item: float
CheXagent_positive_embedding: fixed_size_list<item: float>[1024]
  child 0, item: float
CheXagent_negative_embedding: fixed_size_list<item: float>[1024]
  child 0, item: float


exp BioViL-T/CheXagent joint zero shot classification schema
path: string
dicom_id: string
prediction: struct<Atelectasis: int8, Cardiomegaly: int8, Consolidation: int8, Edema: int8, Enlarged_Cardiomedia (... 191 chars omitted)
  child 0, Atelectasis: int8
  child 1, Cardiomegaly: int8
  child 2, Consolidation: int8
  child 3, Edema: int8
  child 4, Enlarged_Cardiomediastinum: int8
  child 5, Fracture: int8
  child 6, Lung_Lesion: int8
  child 7, Lung_Opacity: int8
  child 8, Pleural_Effusion: int8
  child 9, Pleural_Other: int8
  child 10, Pneumonia: int8
  child 11, Pneumothorax: int8
  child 12, Support_Devices: int8
  child 13, No_Finding: int8