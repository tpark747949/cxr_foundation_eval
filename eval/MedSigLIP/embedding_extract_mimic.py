# setting up lance stuff
import lancedb
import pyarrow as pa
uri = "../../embeddings/MIMIC-CXR-JPG"

db = lancedb.connect(uri)

schema = pa.schema(
    [
        pa.field("path", pa.string()),
        pa.field("dicom_id", pa.string()),
        pa.field("study_id", pa.uint16()),
        pa.field("subject_id", pa.uint16()),
        pa.field("split", pa.string()),
        pa.field("")
        # to do: add the remaining fields
        # importantly, set the chexpert and negbio labels as defined vectors with the -1, 0, 1, and NaN scheme
        # keep in mind: medsiglip embeddings are 1152 torch float tensors
    ]
)

# table = db.create_table("MedSigLIP embeddings", data=data, mode="overwrite")