#| default_exp core


!pip install -q -U keras-nlp tensorflow-text
# Install tensorflow-cpu so tensorflow does not attempt to access the TPU.
!pip install -q -U tensorflow-cpu
!pip install polars


import logging
import tensorflow as tf
import re
import kaggle_evaluation


import jax

jax.devices()


import os

# The Keras 3 distribution API is only implemented for the JAX backend for now
os.environ["KERAS_BACKEND"] = "jax"
# Pre-allocate all TPU memory to minimize memory fragmentation and allocation overhead.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"


import keras
import keras_nlp


# Create a device mesh with (1, 8) shape so that the weights are sharded across
# all 8 TPUs.
device_mesh = keras.distribution.DeviceMesh(
    (1, 4),
    ["batch", "model"],
    devices=keras.distribution.list_devices()[:4],
)


model_dim = "model"

layout_map = keras.distribution.LayoutMap(device_mesh)

# Weights that match 'token_embedding/embeddings' will be sharded on 8 TPUs
layout_map["token_embedding/embeddings"] = (model_dim, None)
# Regex to match against the query, key and value matrices in attention layers
layout_map["decoder_block.*attention.*(query|key|value)/kernel"] = (model_dim, None, None)
layout_map["decoder_block.*attention_output/kernel"] = (model_dim, None, None)
layout_map["decoder_block.*ffw_gating.*/kernel"] = (None, model_dim)
layout_map["decoder_block.*ffw_linear/kernel"] = (model_dim, None)


%%time
model_parallel = keras.distribution.ModelParallel(
    layout_map=layout_map,
    batch_dim_name="batch",
)

keras.distribution.set_distribution(model_parallel)

model_path = "/kaggle/input/gemma-2-tpu-fine-tuning-keras-2b-databricks-15k/gemma_2b_databricks_15000.keras"
#model_path = "/kaggle/input/gemma2/keras/gemma2_9b_en/2" # Is it better with 9B, not trained on databricks?

gemma_lm = keras.saving.load_model(model_path, custom_objects=None, compile=True, safe_mode=True)


import keras_hub

if isinstance(gemma_lm, keras_hub.src.models.gemma.gemma_backbone.GemmaBackbone):
    decoder_block_1 = gemma_lm.get_layer('decoder_block_1')
else:
    decoder_block_1 = gemma_lm.backbone.get_layer('decoder_block_1')

print(type(decoder_block_1))
for variable in decoder_block_1.weights:
  print(f'{variable.path:<48}  {str(variable.shape):<14}  {str(variable.value.sharding.spec)}')


%%time
print(gemma_lm.generate("Describe a goose wearing a gold medal in terms of simple SVG forms", max_length=512))


import json

# Load the JSON file from the specified path.
with open("/kaggle/input/visual-scene-instructions-for-generative-llms/combined_train.json", "r") as f:
    raw_data = json.load(f)

# If the data is nested under a "root" key, use that dictionary.
if "root" in raw_data:
    data_dict = raw_data["root"]
else:
    data_dict = raw_data

inputs = [data_dict[k]["concept"] for k in range(len(data_dict))]
outputs = [data_dict[k]["svg"] for k in range(len(data_dict))]

# Build a list of training examples.
data = []
for key in range(len(inputs)):
    input_text = inputs[key]
    input_output = outputs[key]
    prompt = f"Describe {input_text} in terms of simple SVG forms: {input_output}"
    data.append(prompt)

# Optionally, truncate the dataset to speed up training.
trunc = 10_000 # Not truncated as the dataset has ~2k instances
data = data[:trunc]

# 'data' is now a list of strings (each containing the essay and its score)
# and can be used to train gemma_lm with Keras/KerasNLP.


data[0]


# Enable LoRA for the model and set the LoRA rank to 8.
gemma_lm.backbone.enable_lora(rank=4)


# Limit the input sequence length to 512 to control memory usage.
gemma_lm.preprocessor.sequence_length = 512
gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=5e-5),
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
gemma_lm.summary()


%%time
history = gemma_lm.fit(data, epochs=10, batch_size=4)


import matplotlib.pyplot as plt

# Plotting Loss curve
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Loss', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.show()

# Plot Sparse Categorical Accuracy curve
if 'sparse_categorical_accuracy' in history.history:
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['sparse_categorical_accuracy'], label='Sparse Categorical Accuracy', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Sparse Categorical Accuracy Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


%%time
svg = gemma_lm.generate("""
Describe a goose wearing a gold medal in terms of simple SVG forms
""", max_length=512)
print(svg)


from IPython.display import SVG, display

def add_background_if_white(svg_str, bg_color="lightgray", opacity=0.5):
    # Check for white fill attributes
    if "fill='white'" in svg_str or 'fill="white"' in svg_str:
        # Find the end of the opening <svg> tag
        idx = svg_str.find('>')
        if idx != -1:
            background_rect = f"<rect x='0' y='0' width='500' height='300' fill='{bg_color}' opacity='{opacity}'/>"
            # Insert the background rectangle immediately after the <svg> tag
            svg_str = svg_str[:idx+1] + background_rect + svg_str[idx+1:]
    return svg_str

def segment_svg(input_string):
    # Find the start and end of the <svg> block
    start = input_string.find("<svg")
    end = input_string.find("</svg>") + len("</svg>")
    
    # Extract the <svg> block
    svg_block = input_string[start:end]
    return svg_block
    

svg_modified = add_background_if_white(segment_svg(svg))
display(SVG(svg_modified))


from time import time

def svg_from_scene(scene):
    t0 = time()
    prompt = f"Describe {scene} in terms of simple SVG forms"
    svg = gemma_lm.generate(prompt, max_length=2000)
    svg_modified = add_background_if_white(segment_svg(svg))
    print(scene, "(took", int(time()-t0), "sec. to generate).")
    display(SVG(svg_modified))

generate = False

if generate:
    svg_from_scene("apple on a tree branch")
    svg_from_scene("blue cookie")
    svg_from_scene("green moutains under the moon")
    svg_from_scene("red boat sailing on calm water at sunset")
    svg_from_scene("orange fox sleeping under a starry sky")
    svg_from_scene("cup of steaming coffee on a wooden table")
    svg_from_scene("snowy mountain peak with sunlight hitting the summit")
    svg_from_scene("butterfly landing on a sunflower")
    svg_from_scene("hot air balloon flying over green fields")
    svg_from_scene("purple whale swimming beneath ocean waves")
    svg_from_scene("white owl perched on a pine branch at night")
    svg_from_scene("cat sitting by a window looking at raindrops")
    svg_from_scene("a small cottage surrounded by cherry blossom trees")
    svg_from_scene("campfire burning near a tent in the forest")
    svg_from_scene("yellow rubber duck floating in a bathtub")
    svg_from_scene("lighthouse casting beams over stormy seas")
    svg_from_scene("astronaut floating near a satellite in space")
    svg_from_scene("desert cactus under a glowing full moon")
    svg_from_scene("ice cream cone melting under the summer sun")
    svg_from_scene("bicycle leaning against a brick wall covered in vines")
    svg_from_scene("bluebird perched on a mailbox in the countryside")
    svg_from_scene("slice of pizza topped with mushrooms and peppers")
    svg_from_scene("rainbow arching across two green hills")


!df -h


import numpy as np

num_params = gemma_lm.count_params()  # Total number of parameters
size_in_gb = (num_params * 4) / (1024**3)  # Convert bytes to GB

print(f"Estimated model size: {size_in_gb:.2f} GB ({int(num_params/100_000_000)/10}B parameters)")


!du -sh /kaggle/working/


gemma_lm.save(f"/kaggle/working/gemma_2b_SVG_artist.keras")


#| export

class Model:
    def __init__(self):
        self.gemma_lm = gemma_lm
        self.default_svg = """<svg width="256" height="256" viewBox="0 0 256 256">
  <circle cx="50" cy="50" r="40" fill="red" />
</svg>"""

    def predict(self, description: str, max_new_tokens: int = 512) -> str:
        try:
            prompt = f"Describe {description} in terms of simple SVG forms"
            raw_output = self.gemma_lm.generate(prompt, max_length=max_new_tokens)
            matches = re.findall(r"<svg.*?</svg>", raw_output, re.DOTALL | re.IGNORECASE)
            if not matches:
                return self.default_svg
            
            final_svg = matches[-1]
            return final_svg
        
        except Exception as e:
            logging.error(f"Erreur lors de la génération: {e}")
            return self.default_svg


%%time
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    kaggle_evaluation.test(Model)

