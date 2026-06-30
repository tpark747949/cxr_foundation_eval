import io
import png
import tensorflow as tf
import tensorflow_text as tf_text
import numpy as np
from PIL import Image
import logging
import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download

snapshot_download(repo_id="google/cxr-foundation", local_dir='./checkpoints/hf',
                  allow_patterns=['elixr-c-v2-pooled/*', 'pax-elixr-b-text/*'])

tf.get_logger().setLevel(logging.ERROR)

# Helper function for processing image data
def png_to_tfexample(image_array: np.ndarray) -> tf.train.Example:
    """Creates a tf.train.Example from a NumPy array."""
    # Convert the image to float32 and shift the minimum value to zero
    image = image_array.astype(np.float32)
    image -= image.min()

    if image_array.dtype == np.uint8:
        # For uint8 images, no rescaling is needed
        pixel_array = image.astype(np.uint8)
        bitdepth = 8
    else:
        # For other data types, scale image to use the full 16-bit range
        max_val = image.max()
        if max_val > 0:
            image *= 65535 / max_val  # Scale to 16-bit range
        pixel_array = image.astype(np.uint16)
        bitdepth = 16

    # Ensure the array is 2-D (grayscale image)
    if pixel_array.ndim != 2:
        raise ValueError(f'Array must be 2-D. Actual dimensions: {pixel_array.ndim}')

    # Encode the array as a PNG image
    output = io.BytesIO()
    png.Writer(
        width=pixel_array.shape[1],
        height=pixel_array.shape[0],
        greyscale=True,
        bitdepth=bitdepth
    ).write(output, pixel_array.tolist())
    png_bytes = output.getvalue()

    # Create a tf.train.Example and assign the features
    example = tf.train.Example()
    features = example.features.feature
    features['image/encoded'].bytes_list.value.append(png_bytes)
    features['image/format'].bytes_list.value.append(b'png')

    return example

img = Image.open("../../data/samples/sample2.png").convert("L")  # Convert to greyscale

# Step 1 - ELIXR C (image to elixr C embeddings)
serialized_img_tf_example = png_to_tfexample(np.array(img)).SerializeToString()

if 'elixrc_model' not in locals():
  elixrc_model = tf.saved_model.load('./checkpoints/hf/elixr-c-v2-pooled')
  elixrc_infer = elixrc_model.signatures['serving_default']

elixrc_output = elixrc_infer(input_example=tf.constant([serialized_img_tf_example]))
elixrc_embedding = elixrc_output['feature_maps_0'].numpy()

print("ELIXR-C - interim embedding shape: ", elixrc_embedding.shape)

# Step 2 - Invoke QFormer with Elixr-C embeddings
# Initialize text inputs with zeros
qformer_input = {
    'image_feature': elixrc_embedding.tolist(),
    'ids': np.zeros((1, 1, 128), dtype=np.int32).tolist(),
    'paddings':np.zeros((1, 1, 128), dtype=np.float32).tolist(),
}

if 'qformer_model' not in locals():
  qformer_model = tf.saved_model.load("./checkpoints/hf/pax-elixr-b-text")

qformer_output = qformer_model.signatures['serving_default'](**qformer_input)
elixrb_embeddings = qformer_output['all_contrastive_img_emb']

print("ELIXR-B - embedding shape: ", elixrb_embeddings.shape)

# 2. Create the figure and axes directly (bypasses the interactive GUI layer)
fig = plt.Figure()
ax = fig.subplots()

# 3. Plot exactly like before
im = ax.imshow(elixrb_embeddings[0], cmap='gray')
fig.colorbar(im, ax=ax)
ax.set_title('Visualization of ELIXR-B embedding output')

# 4. Bake it directly to disk
fig.savefig('elixrb_embedding.png', bbox_inches='tight', dpi=300)