# Install Keras 3 last. See https://keras.io/getting_started/ for more details.
!pip install -q -U keras-nlp datasets
!pip install -q -U keras

import os

# Set the backbend before importing Keras
os.environ["KERAS_BACKEND"] = "jax"
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

import keras_nlp
import keras

# Run at half precision.
#keras.config.set_floatx("bfloat16")

# Training Configurations
token_limit = 256
num_data_limit = 100
lora_name = "cakeboss"
lora_rank = 4
lr_value = 1e-4
train_epoch = 20
model_id = "gemma2_instruct_2b_en"


import keras
import keras_nlp

import time

gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(model_id)
gemma_lm.summary()

tick_start = 0

def tick():
    global tick_start
    tick_start = time.time()

def tock():
    print(f"TOTAL TIME ELAPSED: {time.time() - tick_start:.2f}s")

def text_gen(prompt):
    tick()
    input = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    output = gemma_lm.generate(input, max_length=token_limit)
    print("\nGemma output:")
    print(output)
    tock()

# inference before fine-tuning
text_gen("Thơ lục bát về mùa xuân")


gemma_lm.backbone.enable_lora(rank=4)
gemma_lm.summary()


gemma_lm.preprocessor.sequence_length = 128
optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
)
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])
gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)


import os
# from google.colab import userdata

GOOGLE_API_KEY = "AIzaSyDgaUm3zQFixmlGm74cBtHLmeIbiRqdSsI"

os.environ["KAGGLE_USERNAME"] = "dedquoc"
os.environ["KAGGLE_KEY"] = "4e52ef96550ed982cdf4d8e44beb0e33"


!wget -O databricks-dolly-15k.jsonl https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl


import json
data = []
# Define the input and output file paths
input_file_path = 'databricks-dolly-15k.jsonl'
output_file_path = 'filtered_databricks-dolly-15k.jsonl'

# Open the input file
with open(input_file_path, 'r') as input_file:
    # Open the output file
    with open(output_file_path, 'w') as output_file:
        # Iterate over each line in the input file
        for line in input_file:
            # Parse the JSON object
            example = json.loads(line)
            # Check if the example has context
            if 'context' not in example:
                # Write the example to the output file
                output_file.write(json.dumps(example) + '\n')
        template = "Instruction:\n{instruction}\n\nResponse:\n{response}"
        data.append(template.format(**example))
       
data = data[:100]


gemma_lm.fit(data, epochs=1, batch_size=1)


prompt = template.format(
    instruction="Write a poem about the beauty of the Vietnamese countryside",
    response="",
)
sampler = keras_nlp.samplers.TopKSampler(k=5, seed=2)
gemma_lm.compile(sampler=sampler)
print(gemma_lm.generate(prompt, max_length=256))

