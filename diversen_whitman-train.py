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
token_limit = 512
num_data_limit = 2000
lora_name = "whitman"
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
    input = f"<start_of_turn>instruct\n{prompt}<end_of_turn>\n<start_of_turn>completion\n"
    output = gemma_lm.generate(input, max_length=token_limit)
    print("\nGemma output:")
    print(output)
    tock()

# inference before fine-tuning
text_gen("Come, said my soul,")


import keras
import keras_nlp
import datasets

tokenizer = keras_nlp.models.GemmaTokenizer.from_preset(model_id)

# prompt structure
# <start_of_turn>user
# {input}
# <end_of_turn>
# <start_of_turn>model
# {output}<end_of_turn>

# input, output
from datasets import load_dataset

ds = load_dataset(
    "diversen/leaves-of-grass",
    split="train",
)
print(ds)
data = ds.with_format("np", columns=["input", "output"], output_all_columns=False)
train = []

for x in data:
    item = f"<start_of_turn>instruct\n{x['input']}<end_of_turn>\n<start_of_turn>completion\n{x['output']}<end_of_turn>"
    length = len(tokenizer(item))
    # skip data if the token length is longer than our limit
    if length < token_limit:
        train.append(item)
        if len(train) >= num_data_limit:
            break
    else:
        print(f"Training item has to many tokens. Skipping.")


print(train[0])
print(train[1])
print(train[2])


# Enable LoRA for the model and set the LoRA rank to 4.
gemma_lm.backbone.enable_lora(rank=lora_rank)
gemma_lm.summary()

# Limit the input sequence length (to control memory usage).
gemma_lm.preprocessor.sequence_length = token_limit
# Use AdamW (a common optimizer for transformer models).
optimizer = keras.optimizers.AdamW(
    learning_rate=lr_value,
    weight_decay=0.01,
)
# Exclude layernorm and bias terms from decay.
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)


class CustomCallback(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    model_name = f"/kaggle/working/{lora_name}_{lora_rank}_epoch_{epoch+1}.lora.h5"
    gemma_lm.backbone.save_lora_weights(model_name)

    # Evaluate
    text_gen("Come, said my soul,")

    if logs.get("sparse_categorical_accuracy"):
        print(logs.get("sparse_categorical_accuracy"))

    target_accuracy = 0.97
    if logs.get("sparse_categorical_accuracy") >= target_accuracy:
        print(f"\nReached target accuracy of {target_accuracy:.2f}. Stopping training.")
        self.model.stop_training = True


history = gemma_lm.fit(
    train,
    epochs=train_epoch,
    batch_size=1,
    callbacks=[CustomCallback()],
)

# StopOnAccuracy(target_accuracy=target_accuracy)

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.show()


# import os
# import keras
# import keras_nlp

# gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma2_instruct_2b_en")
# # Use the same LoRA rank that you trained
# gemma_lm.backbone.enable_lora(rank=4)

# # Load pre-trained LoRA weights
# gemma_lm.backbone.load_lora_weights("/kaggle/input/whitman_test/keras/default/1/whitman_4_epoch1.lora.h5")



gemma_lm.compile(sampler="top_k")
text_gen("Come, said my soul,")
text_gen("Come, said my soul,")
text_gen("Come, said my soul,")


text_gen("That should I after return,")
text_gen("That should I after return,")
text_gen("That should I after return,")

