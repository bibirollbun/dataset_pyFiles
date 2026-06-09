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


from datasets import load_dataset
import pandas as pd

# Load the HindiChat dataset
dataset = load_dataset("rishiraj/hindichat")

# Convert the 'train' split to a DataFrame
df = pd.DataFrame(dataset['train'])

# Save the DataFrame as a JSON file locally
output_path = "hindichat_train.json"
df.to_json(output_path, orient='records', lines=True)

print(f"Dataset saved as {output_path}")



print("First 5 entries of the dataset:")
df.head()


# Check the 'category' column 
if 'category' in df.columns:
    # Get the distribution of the 'category' column
    category_distribution = df['category'].value_counts()

    # Print the distribution
    print("Distribution of Categories:")
    print(category_distribution)

    # Plot the distribution as a bar chart (optional)
    import matplotlib.pyplot as plt

    category_distribution.plot(kind='bar', figsize=(10, 6), color='skyblue')
    plt.title("Distribution of Categories in the Dataset")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("The 'category' column is not found in the dataset.")



# Function to display one example of all column contents for a specific category
def display_example_by_category(category):
    example = df[df['category'] == category].iloc[0]  # Get the first entry for the specified category
    print(f"\nExample for category: {category}")
    for column in df.columns:
        print(f"{column}:\n{example[column]}\n")

# List of unique categories
categories = ["Generation", "Open QA", "Brainstorm", "Chat", "Rewrite", "Summarize", "Coding", "Classify", "Closed QA", "Extract"]

# Iterate through each category and display one example
for category in categories:
    display_example_by_category(category)



import keras
import keras_nlp
import datasets
from datasets import load_dataset
import time

# Set the backbend before importing Keras
os.environ["KERAS_BACKEND"] = "jax"
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

# Run at half precision.
#keras.config.set_floatx("bfloat16")

# Training Configurations
token_limit = 512
lora_name = "HindiBot"
lora_rank = 16
lr_value = 5e-5
train_epoch = 10
model_id = "gemma2_instruct_2b_en"


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
text_gen("कृपया निम्नलिखित के लिए एक ईमेल हिंदी में उत्तर लिखें:\n\"नमस्ते, मैं अपनी शादी की सालगिरह के लिए एक 3 नंबर का केक ऑर्डर करना चाहता हूँ, क्या यह संभव है? \"")



# Define test prompts
test_prompts = [
    # 1. Instruction-following test with an email response request
    "कृपया निम्नलिखित के लिए एक ईमेल उत्तर हिंदी में लिखें:\n\"प्रिय टीम, मुझे अपनी बैठक का समय बदलने की आवश्यकता है। कृपया मुझे अपनी उपलब्धता बताएं।\"",

    # 2. Conversational test with a request to provide advice
    "हिंदी में उत्तर लिखें:\n\"क्या मुझे स्वास्थ्य के लिए योग या व्यायाम में से किसे प्राथमिकता देनी चाहिए?\"",

    # 3. Story generation prompt for creativity testing
    "कृपया निम्नलिखित के लिए एक उत्तर हिंदी में लिखें:\n\"एक छोटे से गाँव में एक बहादुर लड़की की कहानी लिखें जो अपने गाँव को संकट से बचाती है।\""
]

# Run model inference on each prompt and print the output
for i, prompt in enumerate(test_prompts, 1):
    print(f"\nPrompt {i}: {prompt}")
    output = text_gen(prompt)
    print(f"Model Response {i}:\n{output}\n")






# Initialize the tokenizer for a Hindi-compatible model
tokenizer = keras_nlp.models.GemmaTokenizer.from_preset(model_id)

# Load the dataset
ds = load_dataset("rishiraj/hindichat", split="train")

# Format data for the model
train = []
token_limit = 256  # Example token limit
num_data_limit = 9000  # Example data limit

# Loop through the dataset to create structured input-output pairs
for x in ds:
    category = x['category']
    prompt = x['prompt']
    text = x['text']

    # Construct prompt based on category
    if category == "Generation":
        item = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>assistant\n{text}<end_of_turn>"
    elif category == "Open QA":
        item = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>assistant\n{text}<end_of_turn>"
    elif category == "Brainstorm":
        item = f"<start_of_turn>user\nGenerate ideas for the following:\n{prompt}<end_of_turn>\n<start_of_turn>assistant\n{text}<end_of_turn>"
    elif category == "Chat":
        item = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>assistant\n{text}<end_of_turn>"
    elif category == "Rewrite":
        item = f"<start_of_turn>user\nRewrite the following:\n{prompt}<end_of_turn>\n<start_of_turn>assistant\n{text}<end_of_turn>"
    elif category == "Summarize":
        item = f"<start_of_turn>user\nSummarize the following:\n{prompt}<end_of_turn>\n<start_of_turn>assistant\n{text}<end_of_turn>"
    elif category == "Coding":
        item = f"<start_of_turn>user\nWrite python code for the following:\n{prompt}<end_of_turn>\n<start_of_turn>assistant\n{text}<end_of_turn>"
    elif category == "Classify":
        item = f"<start_of_turn>user\nClassify the following:\n{prompt}<end_of_turn>\n<start_of_turn>assistant\n{text}<end_of_turn>"
    elif category == "Closed QA":
        item = f"<start_of_turn>user\nAnswer the following:\n{prompt}<end_of_turn>\n<start_of_turn>assistant\n{text}<end_of_turn>"
    elif category == "Extract":
        item = f"<start_of_turn>user\nExtract text as per the instruction from the following:\n{prompt}<end_of_turn>\n<start_of_turn>assistant\n{text}<end_of_turn>"
    else:
        # Default structure for unknown categories
        item = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>assistant\n{text}<end_of_turn>"
    
    # Tokenize and check length
    length = len(tokenizer(item))
    if length < token_limit:
        train.append(item)
        if len(train) >= num_data_limit:
            break

# Print sample structured data
print(f"Total examples in training set: {len(train)}")
print("Sample data:")
for i in range(min(3, len(train))):
    print(train[i])



# Enable LoRA for the model and set the LoRA rank to 16.
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
        # Save the model's LoRA weights at the end of each epoch
        model_name = f"/kaggle/working/{lora_name}_{lora_rank}_epoch{epoch+1}.lora.h5"
        gemma_lm.backbone.save_lora_weights(model_name)

        # Evaluate the model on a sample Hindi prompt
        text_gen("कृपया निम्नलिखित के लिए एक ईमेल उत्तर लिखें:\n\"नमस्ते, मैं अपनी शादी की सालगिरह के लिए 3 नंबर का केक ऑर्डर करना चाहता हूँ, क्या यह संभव है?\"")

# Train the model with the revised Hindi dataset
history = gemma_lm.fit(train, epochs=train_epoch, batch_size=1, callbacks=[CustomCallback()])

# Plot the training loss over epochs
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.show()



# Define test prompts
test_prompts = [
    # 1. Instruction-following test with an email response request
    "कृपया निम्नलिखित के लिए एक ईमेल उत्तर लिखें:\n\"प्रिय टीम, मुझे अपनी बैठक का समय बदलने की आवश्यकता है। कृपया मुझे अपनी उपलब्धता बताएं।\"",

    # 2. Conversational test with a request to provide advice
    "कृपया निम्नलिखित के लिए एक उत्तर लिखें:\n\"क्या मुझे स्वास्थ्य के लिए योग या व्यायाम में से किसे प्राथमिकता देनी चाहिए?\"",
    
    "कृपया निम्नलिखित के लिए एक उत्तर लिखें:\n\"कृपया बताएं कि नए व्यवसाय शुरू करने के लिए क्या कदम उठाने चाहिए।\"",
      
    
]

# Run model inference on each prompt and print the output
for i, prompt in enumerate(test_prompts, 1):
    print(f"\nPrompt {i}: {prompt}")
    output = text_gen(prompt)
    





