# 1. Update installation for relevant libraries (optional).
# If you plan to parse or preprocess recipe data, you might need additional packages.
!pip install -q -U keras-nlp datasets
!pip install -q -U keras
# !pip install -q -U pandas  # e.g., for tabular data of ingredients
# !pip install -q -U numpy   # e.g., for advanced numeric manipulations


# ============================= WARNING =============================
# The kernel may lost the connection during long trainings OR you may want to make additional training
#the model check point are saved in the output
# folder in the format /kaggle/working/{lora_name}_{lora_rank}_epoch{LAST_CHECK_POINT}.lora.h5  
# You can then set the training to start at the last check point
START_FROM_CHECK_POINT = True # Or False to start new training
LAST_CHECK_POINT = 2 # the last saved LoRA checkpoint (epoch 2 in this example).
LAST_CHECK_POINT_EPOCH = 5 # a small number of additional epochs or not for additional fine tunning


import os

# 2. (Optional) Keep or change the backend based on hardware and preference.
# JAX is efficient for some large models, but TensorFlow or PyTorch might be more
# familiar or better-supported for certain text tasks. If you *do* use JAX:
os.environ["KERAS_BACKEND"] = "jax"

# 3. Manage memory usage on JAX if needed.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

import keras_nlp
import keras

# 4. Decide on precision.
# If you have limited GPU/TPU memory, half precision can help fit larger models or bigger batches.
# If you need more stable training for text-based tasks or small data, consider float32.
keras.config.set_floatx("float32")

# 5. Adjust hyperparameters and naming to reflect your *use cases*.
# For instance, rename variables to emphasize you’re building a model.

# Choose a maximum token limit. Instructions can vary in length, 
# but recipes often have multiple steps. 512 tokens might be more comfortable 
# for detailed instructions than 256, if memory permits.
token_limit = 512  # was 256

# LORA (Low-Rank Adaptation) parameters for fine-tuning.
# You can rename it to something more descriptive to you trainin
lora_name = "sustainable_fashion"
lora_rank = 8

# Learning rate might be fine as 1e-4, but you could tweak it if the model
# either underfits or overfits your dataset.
lr_value = 2e-5  # Initial learning rate

# Depending on how large and complex your dataset is, you might need
# more or fewer epochs. Start with 3-5, then adjust after initial experiments.
train_epoch = 5 # original train epoch

# 6. Pick a model ID consistent with your project.
# You could choose or create a model name indicating it's specialized for instructions.
model_id = "gemma2_instruct_2b_en"

print("Configuration ready for model fine-tuning.")
print(f"Token Limit: {token_limit}")
print(f"LoRA Name: {lora_name}")
print(f"LoRA Rank: {lora_rank}")
print(f"Learning Rate: {lr_value}")
print(f"Train Epoch: {train_epoch}")
print(f"Model ID: {model_id}")


##############################################################################
# Model & Basic Inference Test (BEFORE Fine-Tuning)
##############################################################################

import time
import keras
import keras_nlp

# Load your model by preset. If "GemmaCausalLM" is the correct class for your model:
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(model_id)
gemma_lm.summary()

# Track elapsed time for generation
the_start_time = 0

def start_time():
    """Mark the start time for generation."""
    global the_start_time
    the_start_time = time.time()

def end_time():
    """Print the total elapsed time since 'start_time' was called."""
    print(f"TOTAL TIME ELAPSED: {time.time() - the_start_time:.2f}s")

def text_gen(prompt):
    """
    Generate text using the gemma_lm model. 
    Prepends conversation tokens around 'prompt' and times the generation.
    """
    start_time()
    # Format the prompt to match your conversation style
    model_input = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    
    # Make sure 'token_limit' is defined in your notebook or set a default (e.g., 512).
    generated = gemma_lm.generate(model_input, max_length=token_limit)
    
    print("\nGemma output:")
    print(generated)
    end_time()

# Example: Testing the model before fine-tuning. 
# You can replace the text with or general prompt to see the model's baseline behavior.
text_gen("How do I figure out which silhouettes work best for my body type without resorting to restrictive style rules?")


from datasets import load_dataset
import keras_nlp

# 2. Instantiate the tokenizer (assuming you have a GemmaTokenizer or an equivalent tokenizer).
tokenizer = keras_nlp.models.GemmaTokenizer.from_preset(model_id)

# 3. Load your  dataset.

ds = load_dataset(
    "csv", 
    data_files="/kaggle/input/sustainable-fashion/sustainable_fashion.csv", 
    split="train"
)

# 4. Configure how you want to handle the data. For example, if your dataset has two columns:
#    "instruction" and "response", or "user_prompt" and "model_reply", etc.
#    Adjust these names to match your actual dataset.
ds = ds.with_format(
    "np",  # or 'numpy'
    columns=["instruction", "response"],  # adapt to your column names
    output_all_columns=False
)

# print(ds)

# 5. Prepare a container for training data strings.
train_data = []

# 6. Loop over each item in the dataset.
#    Adapt the prompt format to how you want the model to see user instructions vs. model responses.
for example in ds:
    # This format can mimic a conversation turn if desired:
    # <start_of_turn>user\n<INSTRUCTION>\n<end_of_turn>\n<start_of_turn>model\n<RESPONSE>\n<end_of_turn>
    # You can rename tokens or keep them as is.
    item = (
        f"<start_of_turn>user\n{example['instruction']}<end_of_turn>\n"
        f"<start_of_turn>model\n{example['response']}<end_of_turn>"
    )

    # 7. Tokenize the combined string and check length. 
    #    Adjust 'token_limit' as needed for your instructions.
    token_limit = 512
    length = len(tokenizer(item))
    if length < token_limit:
        train_data.append(item)

# 8. Print some stats and samples to verify everything looks right.
print("Number of samples kept:", len(train_data))
print("Sample 1:", train_data[0])
print("Sample 2:", train_data[1])
print("Sample 3:", train_data[2])


##############################################################################
# 8. Prepare Model for LoRA Fine-Tuning
##############################################################################

# 8.1 Enable LoRA layers on the model's backbone.
#     This drastically reduces trainable parameters while still adapting the model.
#gemma_lm.backbone.enable_lora(rank=lora_rank)

if START_FROM_CHECK_POINT == True:
    # 8.2 Load the last saved LoRA checkpoint (epoch 2 for example).
    last_checkpoint_path = f"/kaggle/input/sustainable-fashion/{lora_name}_{lora_rank}_epoch{LAST_CHECK_POINT}.lora.h5"
    gemma_lm.backbone.load_lora_weights(last_checkpoint_path)
    
    print(f">>>>>>> Checkpoint  {LAST_CHECK_POINT} loaded successfully!")


# 8.3 Review the updated model summary to see LoRA parameters.
gemma_lm.summary()

# 8.4 Limit the input sequence length (controls memory usage).
gemma_lm.preprocessor.sequence_length = token_limit


##############################################################################
# 9. Compile the Model with an Optimizer & Loss
##############################################################################

# 9.1 Choose AdamW as the optimizer with weight decay (commonly used for Transformers).
optimizer = keras.optimizers.AdamW(
    learning_rate=lr_value,
    weight_decay=0.01
)

# 9.2 Exclude LayerNorm and bias from weight decay (best practice in many LLM fine-tunes).
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

# 9.3 Compile the model with a typical LM loss: SparseCategoricalCrossentropy.
#     Weighted metrics let you see how well the model predicts tokens across the entire dataset.
gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

print("Model compiled and ready to train.")


##############################################################################
# 10. Define the Custom Callback
##############################################################################
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Save the LoRA weights after each epoch
        model_name = f"/kaggle/working/{lora_name}_{lora_rank}_epoch{epoch+1}.lora.h5"
        gemma_lm.backbone.save_lora_weights(model_name)

        # Generate a sample text to see how the model is doing
        # (Replace with a your model related prompt if you want a domain-specific test)
        text_gen("How do I figure out which silhouettes work best for my body type without resorting to restrictive style rules?")

##############################################################################
# 11. Train the Model & Plot Loss
##############################################################################
# Suppose you have X, Y as your final tokenized/padded arrays. 
# Or you might rely on a KerasNLP text processing pipeline that takes raw strings.
# For demonstration, let's just call gemma_lm.fit() on 'train_data' if it supports raw strings.


if START_FROM_CHECK_POINT == True:
    print(f">>>>>>> Starting training from a check point {LAST_CHECK_POINT} !")
    # If the training must restart from a check point
    history = gemma_lm.fit(
        train_data,   
        batch_size=1,
        epochs=LAST_CHECK_POINT_EPOCH, # total epochs you want or additional epochs
        initial_epoch=LAST_CHECK_POINT, # start counting from LAST_CHECK_POINT
        callbacks=[CustomCallback()]
    )    
else:
    print(">>>>>>> Starting training from beginning !")
    # If the training must start from beginning
    history = gemma_lm.fit(
        train_data,        # If your model supports direct string input. (Check gemma_lm docs.)
        epochs=train_epoch,
        batch_size=1,      # Adjust batch size based on memory constraints
        callbacks=[CustomCallback()]
    )

print(">>>>>>> Fine-tuning complete!")


# After training, plot the training loss
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Train Loss')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

