# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


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
token_limit = 1024
lora_name = "arya"
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


text_gen("వెళ్ళిపోతూ మళ్లీ వస్తానని అన్నాడు. కానీ, అతను తిరిగి రాలేదు. అతని గురించి ఏం అనిపిస్తోంది?")


text_gen("उसने कहा था कि वो लौटकर आएगा, लेकिन वो वापस नहीं आया। उसके बारे में आपको क्या लगता है?")


text_gen("उहाँले फर्किन्छु भन्नु भयो, तर उहाँ फर्किनु भएन। तपाईंलाई उहाँबारे के लाग्छ?")


import keras
import keras_nlp
from datasets import load_dataset

# Load Gemma tokenizer
model_id = "gemma2_instruct_2b_en"
tokenizer = keras_nlp.models.GemmaTokenizer.from_preset(model_id)

# Configuration
token_limit = 256  # Maximum token length
num_data_limit = 1000  # Limit on the number of examples to process

# Language tags mapping
language_tags = {
     "san":"sanskrit",
    "tel":"telugu",
    "hin":"hindi",
    "npi":"nepali"
}

# Load dataset
dataset_path = "/kaggle/input/multilingual-text-corpus/multilingual_corpus_with_tags_reordered.txt"
raw_dataset = load_dataset("text", data_files={"train": dataset_path})

# Prepare dataset for fine-tuning
train_data = []

# Loop through the dataset and tokenize
for example in raw_dataset["train"]:
    text = example["text"]
    
    # Extract the language tag (example assumes the language is in the first part of the text)
    # Example: "<tel> This is a Telugu sentence."
    language = text.split(">")[0][1:]  # Extract "tel" from "<tel>"
    tag = language_tags.get(language, "<unk>")  # Use <unk> for unknown languages
    #print(language)
    # Add language tag explicitly
    tagged_text = f"{tag} {text}"

    # Tokenize the text
    tokenized = tokenizer.tokenize(tagged_text)  # Tokenize the tagged text
    token_length = len(tokenized)  # Get the length of the tokenized sequence
    
    # Filter long sequences and add to training data
    if token_length < token_limit:
        train_data.append(tagged_text)
    


# Output dataset stats and examples
print(f"Number of training examples: {len(train_data)}")
print(f"First example:\n{train_data[0]}")
print(f"Second example:\n{train_data[1]}")



# Enable LoRA (Low-Rank Adaptation)
lora_rank = 4  # LoRA rank
gemma_lm.backbone.enable_lora(rank=lora_rank)
gemma_lm.preprocessor.sequence_length = token_limit

# Configure the optimizer
optimizer = keras.optimizers.AdamW(
    learning_rate=lr_value,
    weight_decay=0.01,
)
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)



class SaveLoRAWeightsCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lora_weights_path = f"/kaggle/working/lora_weights_epoch{epoch + 1}.lora.h5"
        gemma_lm.backbone.save_lora_weights(lora_weights_path)
        print(f"Saved LoRA weights to: {lora_weights_path}")



# Fine-tune the model
epochs = 10
batch_size = 8

gemma_lm.fit(
    train_data,  # The tokenized dataset
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[SaveLoRAWeightsCallback()],
)




import keras_nlp
import keras


# Load Gemma 2 model
model_id = "gemma2_instruct_2b_en"
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(model_id)

# Enable LoRA and load weights
lora_rank = 4
gemma_lm.backbone.enable_lora(rank=lora_rank)
lora_weights_path = "/kaggle/input/kaggleinputsupervised-finetuned-weightsjax/jax/default/1/lora_weights_epoch1.lora.h5"
gemma_lm.backbone.load_lora_weights(lora_weights_path)

print("LoRA weights loaded successfully.")
gemma_lm.summary()



import json

# Load translation dataset
json_path = "/content/drive/MyDrive/Translations_Multilingual.json"
with open(json_path, "r", encoding="utf-8") as f:
    translation_data = json.load(f)

# Configuration
token_limit = 1024
train_data = []

# Prepare data for fine-tuning
for example in translation_data[0]: 
    prompt = example["prompt"]
    response = example["response"]
    #print(f"Prompt: {prompt}")

    # Prepare input-output text format for supervised learning
    input_text = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"

    # Tokenize the text using the preprocessor
    tokenized = gemma_lm.preprocessor(input_text)  # Returns a tuple

    # Extract token_ids and attention mask
    token_ids = tokenized[0]["token_ids"]
    #print(f"Token IDs: {token_ids.numpy()}")
    #print(len(token_ids))
    # Filter long sequences based on token length
    if len(token_ids) <= token_limit:
        train_data.append(input_text)

# Display dataset stats
print(f"Number of training examples: {len(train_data)}")
if len(train_data) > 0:
    print(f"First example:\n{train_data[0]}")



import json

# Save tokenized training data
tokenized_data_path = "/content/drive/MyDrive/Tokenized_Translations.json"
with open(tokenized_data_path, "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

print(f"Tokenized data saved to: {tokenized_data_path}")


import json
tokenized_data_path = "/content/drive/MyDrive/Tokenized_Translations.json"
with open(tokenized_data_path, "r", encoding="utf-8") as f:
    train_data = json.load(f)

print(f"Tokenized data loaded successfully. Number of examples: {len(train_data)}")



from keras.mixed_precision import set_global_policy
set_global_policy("mixed_float16")


# Fine-tuning configuration
epochs = 4 
batch_size = 2  

# Train the model
gemma_lm.fit(
    train_data,  # Tokenized supervised dataset
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[save_callback],
)



#Saving the final model
save_path = "/kaggle/working/Gemma2_Arya.h5"  # Specify the path where you want to save
gemma_lm.save(save_path) 

print(f"Model saved successfully to {save_path}")



import keras_nlp
import keras


# Load Gemma 2 model
model_id = "gemma2_instruct_2b_en"
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(model_id)

# Enable LoRA and load weights
lora_rank = 4
gemma_lm.backbone.enable_lora(rank=lora_rank)
lora_weights_path = "/kaggle/input/kaggleinputsupervised-finetuned-weightsjax/jax/default/3/lora_weights_epoch4.lora.h5"
gemma_lm.backbone.load_lora_weights(lora_weights_path)

print("LoRA weights loaded successfully.")
gemma_lm.summary()



import keras
import keras_nlp

import time


tick_start = 0
token_limit=1024
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


text_gen("Translate to Hindi: The ship rested on the mountains of Ararat on the seventeenth day of the seventh month.")
text_gen("Translate to Hindi: The waters increased greatly on the earth, so that all the high mountains under the whole sky were covered.")
text_gen("Translate to Hindi: All who had the breath of life in their nostrils, of all that was on the dry land, died.")
text_gen("Translate to Hindi: It happened at the end of forty days, that Noah opened the window of the ship which he had made and sent out a raven.")
text_gen("Translate to Hindi: Then the dove came back to him in the evening, and behold, in her mouth was an olive leaf.")


text_gen("Translate to English: और शुद्ध, और अशुद्ध दोनो प्रकार के पशुओं में से, पक्षियों,")
text_gen("Translate to English: वह लड़का अपनी किताबें पढ़ रहा है।")
text_gen("Translate to English: मुझे तुम्हारी मदद की ज़रूरत है।")
text_gen("Translate to English: वह हर सुबह दौड़ने जाता है।")
text_gen("Translate to English: यह एक सुंदर दिन है।")



import keras_nlp
import keras


# Load Gemma 2 model
model_id = "gemma2_instruct_2b_en"
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(model_id)

# Enable LoRA and load weights
lora_rank = 4
gemma_lm.backbone.enable_lora(rank=lora_rank)
lora_weights_path = "/kaggle/input/kaggleinputsupervised-finetuned-weightsjax/jax/default/3/lora_weights_epoch4.lora.h5"
gemma_lm.backbone.load_lora_weights(lora_weights_path)

print("LoRA weights loaded successfully.")
gemma_lm.summary()



import json

# Load tokenized data
tokenized_data_path = "/kaggle/input/tokenized-data-for-finetuning/Tokenized_Translations.json"
with open(tokenized_data_path, "r", encoding="utf-8") as f:
    tokenized_data = json.load(f)

print(f"Loaded {len(tokenized_data)} tokenized examples.")



from nltk.translate.bleu_score import sentence_bleu

def evaluate_model(model, tokenized_data, tokenizer):
    bleu_scores = []
    
    for example in tokenized_data[:10000]:
        # Prepare the input text
        input_text = example  # The tokenized input
        expected_response = example.split("<start_of_turn>model\n")[-1].strip("<end_of_turn>")
        
        # Generate model prediction
        prediction = model.generate(input_text, max_length=1024)
        
        # Post-process prediction
        prediction_text = prediction.replace("<start_of_turn>model\n", "").strip("<end_of_turn>")
        
        # Calculate BLEU score
        bleu_score = sentence_bleu([expected_response.split()], prediction_text.split())
        bleu_scores.append(bleu_score)
    
    # Average BLEU score across examples
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    return avg_bleu, bleu_scores

avg_bleu, bleu_scores = evaluate_model(gemma_lm, tokenized_data, gemma_lm.preprocessor.tokenizer)

print(f"Average BLEU Score: {avg_bleu}")



import tensorflow as tf
import numpy as np

def calculate_perplexity(model, data, tokenizer):
    total_loss = 0
    total_tokens = 0

    for example in data:
        input_text = example  # tokenized input text
        target_text = example.split("<start_of_turn>model\n")[-1].strip("<end_of_turn>")

        # Tokenize the input and target text using Gemma's tokenizer (adjust this based on the tokenizer method)
        input_ids = tokenizer.tokenize(input_text)  # Adjust this if needed
        target_ids = tokenizer.tokenize(target_text)  # Adjust this if needed

        # Create the attention mask (1 for valid tokens, 0 for padding tokens)
        attention_mask = [1] * len(input_ids)  # Assuming no padding, you can add logic for padding if needed
        target_attention_mask = [1] * len(target_ids)  # Similarly for target sequence

        # Convert to tensors
        input_tensor = tf.convert_to_tensor([input_ids])
        target_tensor = tf.convert_to_tensor([target_ids])
        attention_mask_tensor = tf.convert_to_tensor([attention_mask])

        # Generate predictions (log-probabilities)
        logits = model([input_tensor, attention_mask_tensor], training=False)  # Provide input and attention mask

        # Check the shape of logits and target tensor
        #print(f"Logits shape: {logits.shape}, Target shape: {target_tensor.shape}")

        # Padding target tensor to match logits shape (model expects logits with seq_len of 46)
        target_len = logits.shape[1]  # Get model's output sequence length
        target_tensor_padded = tf.pad(target_tensor, [[0, 0], [0, target_len - target_tensor.shape[1]]], constant_values=0)

        # Compute the loss (negative log likelihood)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='sum_over_batch_size')  # Use sum_over_batch_size for averaging loss
        loss = loss_fn(target_tensor_padded, logits)

        # Accumulate loss and token count
        total_loss += loss.numpy()  # Loss will be averaged across the sequence length
        total_tokens += target_len

    # Calculate Perplexity (average loss per token)
    perplexity = np.exp(total_loss / total_tokens)  # Perplexity is calculated as exp(loss per token)
    return perplexity

# Assuming you have your tokenized data and model ready
perplexity = calculate_perplexity(gemma_lm, tokenized_data[:10000], gemma_lm.preprocessor.tokenizer)
print(f"Perplexity: {perplexity}")





