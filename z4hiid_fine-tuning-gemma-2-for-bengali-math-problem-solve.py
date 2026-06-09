!pip install -q -U keras-nlp
!pip install -q -U keras
!pip install -q -U kagglehub --upgrade


import os

# Set the backbend before importing Keras
os.environ["KERAS_BACKEND"] = "jax"
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

import keras_nlp
import keras

import pandas as pd


# Training Configurations

# Model Parameters
token_limit = 512         # Maximum sequence length
num_data_limit = 100      # Dataset size limit
lora_name = "gemma-bn-math" # LoRA Name
lora_rank = 4            # LoRA adaptation rank
lr_value = 1e-4          # Learning rate
train_epoch = 30         # Number of training epochs
model_id = "gemma2_instruct_2b_en"  # Base model identifier


# df = pd.read_csv(Config.dataset_path)
df = pd.read_csv('/kaggle/input/bengali-math-cot-dataset/bangla-math-cot-dataset.csv')

print(f"Total examples in dataset {len(df)}")


df.head()


# -------------------  LOAD TOKENIZER ---------------------
tokenizer = keras_nlp.models.GemmaTokenizer.from_preset(model_id)

# -------------------  DATA PREPROCESSING  ---------------------
                                                        
train = [] # empty list to store processed items
# Iterate through rows of the pandas DataFrame
for index, row in df.iterrows():
  
    # Ensure 'input' and 'output' columns exist in your DataFrame
    if 'problem' in row and 'solution' in row:
      input_text = row['problem']
      output_text = row['solution']
    else:
      print("ERROR: Dataset must contain columns named 'problem' and 'solution'")
      break
    # Formulate the instruction format
    item = (
      f"<start_of_turn>user\n"
      f"গণিতের সমস্যাটি ধাপে ধাপে সমাধান করো :\n\"{input_text}\"<end_of_turn>\n"
      f"<start_of_turn>model\n{output_text}<end_of_turn>"
    )

    # Tokenize the text
    length = len(tokenizer(item))

    # Skip data if the token length is longer than our limit
    if length < token_limit:
      train.append(item)
      if len(train) >= num_data_limit:
        break


# -------------------  OUTPUT DATA  ---------------------
# Output the processed data
print(len(train))
print(train[0])


import time

# Load the pre-trained Gemma language model from a preset
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(model_id)
gemma_lm.summary()

# Initialize a variable to store the start time for timing operations
tick_start = 0

# Function to record the start time of an operation
def tick():
    """Start timing an operation."""
    global tick_start
    tick_start = time.time()

# Function to calculate and display the elapsed time of an operation
def tock():
    """Display the total time elapsed since tick() was called."""
    elapsed_time = time.time() - tick_start
    print(f"TOTAL TIME ELAPSED: {elapsed_time:.2f}s")

# Function to generate text using the Gemma language model
def generate_text(prompt, token_limit=512):
    """
    Generate text using the Gemma language model with a Bengali prompt.

    Args:
        prompt (str): The input prompt for text generation.
        token_limit (int): Maximum token length for the generated output.

    Returns:
        None: Outputs generated text and timing directly.
    """
    print("\nGenerating text for the given Bengali prompt...")
    tick()  # Start timing

    # Preprocess the prompt to include necessary formatting for the model
    input_text = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

    # Generate text using the model
    output = gemma_lm.generate(input_text, max_length=token_limit)

    # Display the output
    print("\nGemma output:")
    print(output)

    # End timing and display elapsed time
    tock()

# Bengali prompt for text generation
bengali_prompt = (
    "আপনি কে?"
)

# Generate text using the Bengali prompt
generate_text(bengali_prompt)


# Enable LoRA for the model and set the LoRA rank to 4.
gemma_lm.backbone.enable_lora(rank=lora_rank)
gemma_lm.summary()

# Limit the input sequence length (to control memory usage).
gemma_lm.preprocessor.sequence_length = token_limit
# Use AdamW (a common optimizer for transformer models).
optimizer = keras.optimizers.Adam(
    learning_rate=lr_value,
)


gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)


# Define the callback for saving LoRA weights and evaluating
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        model_name = f"/kaggle/working/{lora_name}_{lora_rank}_epoch{epoch+1}.lora.h5"
        gemma_lm.backbone.save_lora_weights(model_name)

        # Evaluate with a sample prompt
        print("\nSample Evaluation:")
        sample_prompt = (
            "গণিতের সমস্যাটি ধাপে ধাপে সমাধান করো :\n"
            "\"72 এবং 108 এর গরিষ্ঠ সাধারণ গুণনীয়ক কী?\""
        )
        input_text = (
            f"<start_of_turn>user\n{sample_prompt}<end_of_turn>\n<start_of_turn>model\n"
        )
        output = gemma_lm.generate(input_text, max_length=token_limit)
        print(f"Model Output:\n{output}")



# Fine-tune the model
history = gemma_lm.fit(
    train,
    epochs=train_epoch,
    batch_size=1,
    callbacks=[CustomCallback()]
)



# Bengali prompt for text generation
bengali_prompt = (
    "এক প্যাকেটে 12টি চকোলেট আছে। রাহিম 5টি প্যাকেট কিনল। রাহিমের কাছে মোট কতটি চকোলেট আছে?"
)

# Generate text using the Bengali prompt
generate_text(bengali_prompt)


# Save the finetuned model as a KerasNLP preset.
gemma_lm.save_to_preset("./gemma-2-math-cot-bn")


import kagglehub

kagglehub.login()

# Replace with path to directory containing model files.
LOCAL_MODEL_DIR = '/kaggle/working/gemma-2-math-cot-bn'

MODEL_SLUG = 'gemma-2-math-cot-bn' # Replace with model slug.

# Learn more about naming model variations at
# https://www.kaggle.com/docs/models#name-model.
VARIATION_SLUG = 'default' # Replace with variation slug.

kagglehub.model_upload(
  handle = f"z4hiid/{MODEL_SLUG}/keras/{VARIATION_SLUG}",
  local_model_dir = LOCAL_MODEL_DIR,
  version_notes = 'Update 2025-01-14')

