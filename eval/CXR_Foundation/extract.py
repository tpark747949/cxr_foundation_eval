import io
import sys
import png
import tensorflow as tf
import tensorflow_text as tf_text
import numpy as np
from PIL import Image
import logging
from huggingface_hub import snapshot_download

tf.get_logger().setLevel(logging.ERROR)

img_path = sys.argv[1]
out_path = sys.argv[2]

snapshot_download(repo_id="google/cxr-foundation", local_dir='./checkpoints/hf',
                  allow_patterns=['elixr-c-v2-pooled/*', 'pax-elixr-b-text/*'])

# Helper function for processing image data
def png_to_tfexample(image_array: np.ndarray) -> tf.train.Example:
    image = image_array.astype(np.float32)
    image -= image.min()

    if image_array.dtype == np.uint8:
        pixel_array = image.astype(np.uint8)
        bitdepth = 8
    else:
        max_val = image.max()
        if max_val > 0:
            image *= 65535 / max_val 
        pixel_array = image.astype(np.uint16)
        bitdepth = 16

    output = io.BytesIO()
    png.Writer(
        width=pixel_array.shape[1],
        height=pixel_array.shape[0],
        greyscale=True,
        bitdepth=bitdepth
    ).write(output, pixel_array.tolist())
    png_bytes = output.getvalue()

    example = tf.train.Example()
    features = example.features.feature
    features['image/encoded'].bytes_list.value.append(png_bytes)
    features['image/format'].bytes_list.value.append(b'png')

    return example

img = Image.open(img_path).convert("L")  

# Step 1 - ELIXR C
serialized_img_tf_example = png_to_tfexample(np.array(img)).SerializeToString()

elixrc_model = tf.saved_model.load('./checkpoints/hf/elixr-c-v2-pooled')
elixrc_infer = elixrc_model.signatures['serving_default']

elixrc_output = elixrc_infer(input_example=tf.constant([serialized_img_tf_example]))
elixrc_embedding = elixrc_output['feature_maps_0'].numpy()

# Step 2 - Invoke QFormer
qformer_input = {
    'image_feature': elixrc_embedding.tolist(),
    'ids': np.zeros((1, 1, 128), dtype=np.int32).tolist(),
    'paddings': np.zeros((1, 1, 128), dtype=np.float32).tolist(),
}

qformer_model = tf.saved_model.load("./checkpoints/hf/pax-elixr-b-text")
qformer_output = qformer_model.signatures['serving_default'](**qformer_input)
elixrb_embeddings = qformer_output['all_contrastive_img_emb']

np.save(out_path, elixrb_embeddings.numpy())