# setting up lance stuff
import lancedb
import pyarrow as pa
uri = "../../embeddings/MIMIC-CXR-JPG"

db = lancedb.connect(uri)

schema = pa.schema(
    [
        pa.field("path", pa.string()),
        pa.field("dicom_id", pa.string()),
        pa.field("PerformedProcedureStepDescription", pa.string()),
        pa.field("ViewPosition", pa.string()),
        pa.field("image_size", pa.struct([pa.field("Rows", pa.uint16()), pa.field("Columns", pa.uint16())])),
        pa.field("StudyDate", pa.string()),
        pa.field("StudyTime", pa.string()),
        pa.field("ProcedureCodeSequence_CodeMeaning", pa.string()),
        pa.field("ViewCodeSequence_CodeMeaning", pa.string()),
        pa.field("PatientOrientationCodeSequence_CodeMeaning", pa.string()),
        pa.field("study_id", pa.uint16()),
        pa.field("subject_id", pa.uint16()),
        pa.field("split", pa.string()),
        pa.field(
            "CheXpert_labels", 
            pa.struct(
                [
                    pa.field("Atelectasis", pa.int8()),
                    pa.field("Cardiomegaly", pa.int8()),
                    pa.field("Consolidation", pa.int8()),
                    pa.field("Edema", pa.int8()),
                    pa.field("Enlarged_Cardiomediastinum", pa.int8()),
                    pa.field("Fracture", pa.int8()),
                    pa.field("Lung_Lesion", pa.int8()),
                    pa.field("Lung_Opacity", pa.int8()),
                    pa.field("Pleural_Effusion", pa.int8()),
                    pa.field("Pneumonia", pa.int8()),
                    pa.field("Pneumothorax", pa.int8()),
                    pa.field("Pleural_Other", pa.int8()),
                    pa.field("Support_Devices", pa.int8()),
                    pa.field("No_Finding", pa.int8()),
                ]
            ),
        ),
        pa.field("embedding_raw", pa.list_(pa.float32(), 1152)),
        pa.field("embedding_l2", pa.list_(pa.float32(), 1152)),
    ]
)

table = db.create_table("MedSigLIP_embeddings_MIMIC-CXR-JPG", schema=schema, mode="overwrite")

print(table.schema)