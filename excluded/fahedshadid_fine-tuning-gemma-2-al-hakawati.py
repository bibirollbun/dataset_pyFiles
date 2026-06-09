!pip install -q -U keras keras-nlp 
!pip install -q -U datasets
!pip install --upgrade kagglehub


import os
import pandas as pd
import keras
import keras_nlp
import re
import matplotlib.pyplot as plt
from IPython.display import HTML, display
import time


# The Keras 3 distribution API is only implemented for the JAX backend for now
os.environ["KERAS_BACKEND"] = "jax"
# Pre-allocate all TPU memory to minimize memory fragmentation and allocation overhead.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"


# Helper functions
def convert_message_to_prompt(message: str, model_prefix: str = "") -> str:
    """Converts a message to a prompt for instruction tuning."""
    user_part = f"<start_of_turn>user\n{message}<end_of_turn>"
    if model_prefix:
        model_part = f"<start_of_turn>model\n{model_prefix}<end_of_turn>"
    else:
        model_part = f"<start_of_turn>model\n"
    return f"{user_part}\n{model_part}"

def strip_tokens(response: str) -> str:
    """Removes control tokens from the model's response."""
    cleaned_response = re.sub(r"<start_of_turn>|<end_of_turn>|user|model", "", response)
    return cleaned_response.strip()

def extract_model_response(response: str) -> str:
    """Extracts the model's response from the generated text."""
    match = re.search(r"<start_of_turn>model\n(.*?)<end_of_turn>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return response

def prepare_dataset(df, tokenizer, max_length=1024):
    """Prepares and filters the dataset for instruction tuning."""
    formatted_data = []
    filtered_count = 0
    total_count = len(df)
    
    for _, row in df.iterrows():
        prompt = convert_message_to_prompt(row['instruction'], row['output'])
        tokens = tokenizer(prompt)
        if len(tokens) <= max_length:
            formatted_data.append(prompt)
        else:
            filtered_count += 1
    
    print(f"Removed {filtered_count}/{total_count} examples exceeding {max_length} tokens")
    return formatted_data


# Load your instruction dataset
train_df = pd.read_csv('/kaggle/input/al-hakawati-dataset-gemma2-2b/al-hakawati-dataset.csv')


gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("/kaggle/input/gemma2/keras/gemma2_instruct_2b_en/2")


# Enable LoRA
gemma_lm.backbone.enable_lora(rank=8)
gemma_lm.preprocessor.sequence_length = 128


tokenizer = gemma_lm.preprocessor.tokenizer


train_data = prepare_dataset(train_df, tokenizer, max_length=128)


# Use AdamW optimizer and decay
initial_lr = 2e-5
optimizer = keras.optimizers.AdamW(
    learning_rate=keras.optimizers.schedules.CosineDecay(
        initial_lr,
        decay_steps=15 * len(train_data),
        alpha=0.1
    ),
    weight_decay=0.01,
)
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])


# Compile the model
gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    jit_compile=True
)


# Define the inference function
def generate_answer(question, max_length=1024):
    prompt = convert_message_to_prompt(question)
    generated_answer = gemma_lm.generate(prompt, max_length=max_length)
    return extract_model_response(generated_answer)

# Define callbacks
class CustomCallback(keras.callbacks.Callback):
    def __init__(self, lora_name, lora_rank, prompt_example):
        super(CustomCallback, self).__init__()
        self.lora_name = lora_name
        self.lora_rank = lora_rank
        self.prompt_example = prompt_example

    def on_epoch_end(self, epoch, logs=None):
        model_name = f"fine-tuned-gemma_{self.lora_rank}_epoch{epoch+1}.lora.h5"
        gemma_lm.backbone.save_lora_weights(model_name)
        print(f"Saved LoRA weights to {model_name}\n")
        # Perform inference with a sample prompt
        generated_answer = generate_answer(self.prompt_example)
        print(f"\nSample Generated Answer after Epoch {epoch+1}:\n{self.prompt_example}\n")
        html = f"<div style='text-align: right; direction: rtl;'>{generated_answer}</div>"
        display(HTML(html))


prompt_example = "Ø¹Ø±Ù� Ø¹Ù† Ù†Ù�Ø³Ùƒ"


lora_rank = 8
token_limit = 128
lora_name = "gemma2_ar_hakawati" 


custom_callback = CustomCallback(
    lora_name=lora_name,
    lora_rank=lora_rank,
    prompt_example=prompt_example
)


# Train the model
history = gemma_lm.fit(
    train_data,
    epochs=15,
    batch_size=1,
    callbacks=[custom_callback]
)


gemma_lm.backbone.load_lora_weights("/kaggle/working/fine-tuned-gemma_8_epoch9.lora.h5")


# Define the input prompt or question
prompt = "Ø£Ø­ÙƒÙŠ Ù„ÙŠ Ù‚ØµØ©" 

# Generate the answer using the generate_answer function
generated_answer = generate_answer(prompt)

# Print the generated answer
print(generated_answer)


# Define the input prompt or question
prompt = " Ø£Ø­ÙƒÙŠ Ù„ÙŠ Ù‚ØµØ© Ø¹Ù† Ø¬Ø­Ø§" 

# Generate the answer using the generate_answer function
generated_answer = generate_answer(prompt)

# Print the generated answer
print(generated_answer)


# Step: Merge the LoRA weights into the base model and save it

# Define the directory to save the fine-tuned model
preset_dir = "./fine_tuned_gemma_hakawati"

# Save the fine-tuned model. Uncomment to save
gemma_lm.save_to_preset(preset_dir)

print(f"Fine-tuned Gemma model saved to {preset_dir}")

