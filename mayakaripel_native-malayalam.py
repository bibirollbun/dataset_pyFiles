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


!pip install sentence-transformers
!pip install nltk
!pip install gutenbergpy
!pip install datasets --quiet
!pip install transformers --quiet
!pip install tensorboard --quiet
!pip install trl --quiet
!pip install peft --quiet
!pip install kagglehub --quiet --upgrade 
!pip install gutenbergpy --quiet
!pip install accelerate --quiet
!pip install bitsandbytes --quiet


!pip install --upgrade transformers


import huggingface_hub
import peft

print(f"huggingface_hub version: {huggingface_hub.__version__}")
print(f"peft version: {peft.__version__}")


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
import os
import re
import gutenbergpy.textget
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import kagglehub
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import tensorflow as tf
import gc
from tqdm import tqdm

# Check for GPU availability
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")
        device = 0
    except RuntimeError as e:
        print(e)
        device = -1
else:
    print("No GPU found, using CPU.")
    device = -1

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')
# Configuration
model_name = "/kaggle/input/gemma/transformers/2b-it/3"
output_dir = "/kaggle/working/gemma-2/transformers/gemma-2-2b-ml-en/1"

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'right'

print(model)
model.get_memory_footprint()

# Using transformers pipeline, lets generate a few outputs
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)


messages = [
    {"role": "user", "content": "You are a malayalam to english translator, \
    write me a short text in english about the importance of rain in India"},
]

outputs = pipe(
    messages,
    max_new_tokens=256,
    do_sample=True,
    num_return_sequences=5
)
# Print each generated response
for i, output in enumerate(outputs):
    assistant_response = output["generated_text"][-1]["content"]
    print(f"Response {i+1}:")
    print(assistant_response)
    print("---")

messages = [
    {
      "role": "user",
      "content": 
"""You are a malayalam to english translator. 
Translate this malayalam sentence to english: "ഇന്ത്യയിലെ മഴയുടെ പ്രാധാന്യം" """
    }
  ]

outputs = pipe(
    messages,
    max_new_tokens=256,
    do_sample=True,
    num_return_sequences=5
)

# Print each generated response
for i, output in enumerate(outputs):
    assistant_response = output["generated_text"][-1]["content"]
    print(f"Response {i+1}:")
    print(assistant_response)
    print("---")


# ----------------------------------------------------------
#   Data Preparation
# ----------------------------------------------------------

def create_tatoeba_csv(eng_sentences_file, mal_sentences_file, links_file, output_file="tatoeba_combined_dataset.csv"):
    """
    Creates a combined CSV file from Tatoeba sentences and links using specific language files.
    """
    # Load English sentences
    eng_sentences = pd.read_csv(eng_sentences_file, sep="\t", header=None, names=["sentence_id", "language", "sentence"])
    eng_sentences = eng_sentences[eng_sentences['language'] == 'eng']

    # Load Malayalam sentences
    mal_sentences = pd.read_csv(mal_sentences_file, sep="\t", header=None, names=["sentence_id", "language", "sentence"])
    mal_sentences = mal_sentences[mal_sentences['language'] == 'mal']

    # Load links
    links = pd.read_csv(links_file, sep="\t", header=None, names=["sentence_id_1", "sentence_id_2"])

    # Merge sentences and links for the specific languages
    merged = pd.merge(links, eng_sentences, left_on="sentence_id_1", right_on="sentence_id", how="inner")
    merged = pd.merge(merged, mal_sentences, left_on="sentence_id_2", right_on="sentence_id", how="inner", suffixes=("_eng", "_mal"))

    # Prepare the combined DataFrame
    combined = merged[["sentence_id_1", "language_eng", "sentence_eng", "language_mal", "sentence_mal"]]
    combined = combined.rename(columns={
        "sentence_id_1": "sentence_id",
        "language_eng": 'language_1',
        "language_mal": 'language_2',
        "sentence_eng": 'sentence_1',
        "sentence_mal": 'sentence_2',
    })

    # Save as CSV
    combined.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Created combined CSV: {output_file}")


# Example of how to use create_tatoeba_csv
eng_sentences_file = "/kaggle/input/eng-sentences/eng_sentences.tsv"
mal_sentences_file = "/kaggle/input/mal-sentences/mal_sentences.tsv"
links_file = "/kaggle/input/links-file/links.csv"
create_tatoeba_csv(eng_sentences_file, mal_sentences_file, links_file)

# Load the generated data
DATASET_PATH = "tatoeba_combined_dataset.csv"
try:
    df = pd.read_csv(DATASET_PATH, encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv(DATASET_PATH, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv(DATASET_PATH, encoding='latin1')

# Assuming your columns are named 'english' and 'malayalam'
train_data = df.dropna(subset=['sentence_1', 'sentence_2'])
train_data = train_data.rename(columns={'sentence_1': 'english', 'sentence_2': 'malayalam'})

class Config:
    token_limit = 64
    lora_name = "eng_mala_translator"
    lora_rank = 4
    lr_value = 1e-6
    train_epoch = 2
    model_id = "/kaggle/input/gemma/transformers/2b-it/3"
    save_frequency = 1
    gradient_accumulation_steps = 4
    
tokenizer = AutoTokenizer.from_pretrained(Config.model_id)
train_data = train_data[train_data['english'].apply(lambda x: len(tokenizer(x).input_ids) < Config.token_limit) & \
                        train_data['malayalam'].apply(lambda x: len(tokenizer(x).input_ids) < Config.token_limit)]

# Creating a smaller test dataset for evaluation, you may need to split data properly
test_data = train_data.sample(10)
train_data = train_data.drop(test_data.index)

def restore_width(text):
    paragraphs = text.split('\n\n')
    paragraphs = [' '.join(paragraph.split('\n')) for paragraph in paragraphs]
    text = '\n\n'.join(paragraphs)
    return text

def clean_text(text):
    text = text.lstrip()
    text = re.sub(r'([a-zA-Z]+)’([a-zA-Z]+)', r'\1\'\2', text)
    text = re.sub(r'[_\*]', '', text)
    text = re.sub(r'[«»]', '', text)
    text = re.sub('--', '', text)
    text = re.sub(r'^-+', '', text)
    text = text.replace('...', ',')
    text = re.sub(r',{2,}', ',', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)
    text = text.rstrip(',').rstrip()
    text = re.sub(r',(?=\n|$)', '.', text)
    text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)
    return text

def add_model_format(text):
    return "<start_of_turn>user\n" + text + "<end_of_turn>\n<start_of_turn>model\n"

# Disable W&B
os.environ["WANDB_MODE"] = "disabled"
torch.cuda.empty_cache()


# Ensure LoRA layers target instruction-related layers
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Load sentence transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load sentiment analysis and toxicity classifiers (Load once outside loop)
sentiment_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device if device!=-1 else "cpu")
toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert", device=-1)


def get_reward(response, user_input):
    """
    Calculates a reward for the chatbot's response based on various criteria.
    """
    try:
        response_embedding = sentence_model.encode(response)
        input_embedding = sentence_model.encode(user_input)
        similarity = util.cos_sim(response_embedding, input_embedding)[0][0].item()
        distinctness_score = 1.0 - similarity

        conciseness_score = 1.0 / len(response.split())
        
        sentiment_result = sentiment_classifier(response)
        empathy_score = sentiment_result[0]['score']

        toxicity_result = toxicity_classifier(response)
        safety_score = 1.0 - toxicity_result[0]['score']

        overall_reward = (
            distinctness_score * 0.3 +
            conciseness_score * 0.2 +
            empathy_score * 0.3 +
            safety_score * 0.2
        )

        return overall_reward
    except Exception as e:
      print(f"Error in reward: {e}")
      return 0 # Handle exceptions and make it robust

def rl_update(model, optimizer, user_input_ids, response_ids, reward, padding_mask, accumulation_steps):
    """
    Updates the model weights using the calculated reward and performs backpropagation.
    """
    with tf.GradientTape() as tape:
      output = model(input_ids = user_input_ids, attention_mask = padding_mask, training=True)
      predicted_token_logits = output.logits[:, :-1, :]

      target_ids = response_ids[:, 1:]

      loss = tf.keras.losses.sparse_categorical_crossentropy(
          y_true=target_ids,
          y_pred=predicted_token_logits,
          from_logits=True
      )
      loss = tf.reduce_mean(loss * -reward)

    grads = tape.gradient(loss, model.trainable_variables)
    grads = [g / accumulation_steps for g in grads]  # Scale gradients by accumulation steps
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def gen(prompt, model=model):
    """
    Generates a response for a given prompt using the model.
    """
    try:
        input_text = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)
        output = model.generate(input_ids, max_length=Config.token_limit)
        malayalam_output = re.search(r'<start_of_turn>model\n(.*?)<end_of_turn>', tokenizer.decode(output[0], skip_special_tokens=True), re.DOTALL)
        if malayalam_output:
           return malayalam_output.group(1).strip()
        else:
           return ""
    except Exception as e:
       print(f"Error in text generation {e}")
       return ""

# Training Loop with RL
def train_with_rl(model, data, optimizer, epochs, batch_size):
    """
    Trains the model using reinforcement learning.
    """
    def prepare_batch(batch, max_sequence_length=Config.token_limit):
        batch_inputs = []
        batch_responses = []
        batch_rewards = []

        for item in batch:
            # `item` is a tensor of shape () containing a single string
            formatted_input = item.numpy().decode('utf-8')
            response = gen(formatted_input, model)
            if response:
                reward = get_reward(response, formatted_input)
                batch_inputs.append(formatted_input)
                batch_responses.append(response)
                batch_rewards.append(reward)
        if not batch_inputs: return None, None, None, None

        # Tokenize input and response
        input_ids = tokenizer(batch_inputs, padding=True, return_tensors="tf", max_length=max_sequence_length).input_ids
        response_ids = tokenizer(batch_responses, padding=True, return_tensors="tf", max_length=max_sequence_length).input_ids

        # Create padding masks manually
        padding_mask = tf.cast(input_ids != tokenizer.pad_token_id, dtype=tf.int32)
        
        rewards_tensor = tf.constant(batch_rewards, dtype=tf.float32)

        return input_ids, response_ids, rewards_tensor, padding_mask
        
    dataset = tf.data.Dataset.from_tensor_slices(data['english'].tolist())
    dataset = dataset.batch(batch_size)
    
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}")
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(dataset)):
            input_ids, response_ids, rewards_tensor, padding_mask = prepare_batch(batch)
            if input_ids is not None:
                try:
                    loss = rl_update(model, optimizer, input_ids, response_ids, tf.reduce_mean(rewards_tensor), padding_mask, Config.gradient_accumulation_steps) # Average the reward for batch
                    print(f"Loss: {loss:.4f}")
                    if (step + 1) % Config.gradient_accumulation_steps == 0:
                       optimizer.step()
                       optimizer.zero_grad()
                except tf.errors.ResourceExhaustedError as e:
                   print(f"Resource exhausted: {e}")
                   # Clear GPU memory
                   tf.keras.backend.clear_session()
                   gc.collect()
                   tf.config.experimental.reset_memory_stats(device)
                   continue


optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr_value)

# Begin Fine Tuning
#history = gemma.fit(data, epochs=Config.train_epoch, batch_size=1, callbacks=[CustomCallback()])
# SFT training is not used.
# Call RL method
train_with_rl(model, train_data, optimizer, Config.train_epoch, batch_size=1)


messages = [
    {
      "role": "user",
      "content": 
"""You are a malayalam to english translator. 
Translate this malayalam sentence to english: "ഇന്ത്യയിലെ മഴയുടെ പ്രാധാന്യം" """
    }
  ]

outputs = pipe(
    messages,
    max_new_tokens=256,
    do_sample=True,
    num_return_sequences=5
)

# Print each generated response
for i, output in enumerate(outputs):
    assistant_response = output["generated_text"][-1]["content"]
    print(f"Response {i+1}:")
    print(assistant_response)
    print("---")


word = "pull"

# Encode the word and get the number of tokens
encoded_word = tokenizer.encode(word, return_tensors='pt')
num_tokens = len(encoded_word[0])

print(f"The word '{word}' is tokenized into {num_tokens} tokens.")

decoded_word = [tokenizer.decode(w) for w in encoded_word[0]]
print(f"The decoded word is: {decoded_word}")


messages = [
    {
      "role": "user",
      "content": 
"""You are a malayalam to english translator. 
Translate this malayalam sentence to english: "ഇന്ത്യയിലെ മഴയുടെ പ്രാധാന്യം" """
    }
  ]

outputs = pipe(
    messages,
    max_new_tokens=256,
    do_sample=True,
    num_return_sequences=5
)

# Print each generated response
for i, output in enumerate(outputs):
    assistant_response = output["generated_text"][-1]["content"]
    print(f"Response {i+1}:")
    print(assistant_response)
    print("---")


model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)


import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import tensorflow as tf

# Set up the paths
adapter_model_path = "/kaggle/input/gemma2/keras/gemma2_instruct_2b_en/1"  # Path to your adapter
base_model_path = "/kaggle/input/gemma/transformers/2b-it/3"  # Path to your base model
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the base model configuration
config = AutoConfig.from_pretrained(base_model_path)

# Manually define the TensorFlow model
class TFGemmaModel(tf.keras.Model):
    def __init__(self, config):
        super(TFGemmaModel, self).__init__()
        # Define layers that match the Keras model architecture
        self.embed_tokens = tf.keras.layers.Embedding(config.vocab_size, config.hidden_size)
        self.transformer_layers = [tf.keras.layers.Dense(config.hidden_size, activation='relu') for _ in range(config.num_hidden_layers)]
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.lm_head = tf.keras.layers.Dense(config.vocab_size, use_bias=False)

    def call(self, inputs):
        x = self.embed_tokens(inputs)
        for layer in self.transformer_layers:
            x = layer(x)
        x = self.norm(x)
        x = self.lm_head(x)
        return x

# Instantiate the TensorFlow model
tf_base_model = TFGemmaModel(config)

# Load the keras model weights
adapter_weights_path = os.path.join(adapter_model_path, "model.weights.h5")
tf_base_model.load_weights(adapter_weights_path)

# Load the base model in PyTorch
pytorch_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, device_map="auto")

# Transfer weights from Keras to PyTorch
state_dict = {}
for layer in tf_base_model.layers:
    weights = layer.get_weights()
    if isinstance(layer, tf.keras.layers.Embedding) and weights:
        state_dict['transformer.embed_tokens.weight'] = torch.tensor(weights[0]).to(device)
    elif isinstance(layer, tf.keras.layers.Dense) and weights:
        if len(weights) > 0:
            weight_name = f'transformer.h.{tf_base_model.layers.index(layer)}.mlp.fc_1.weight'
            state_dict[weight_name] = torch.tensor(weights[0].T).to(device)
        if len(weights) > 1:
            bias_name = f'transformer.h.{tf_base_model.layers.index(layer)}.mlp.fc_1.bias'
            state_dict[bias_name] = torch.tensor(weights[1]).to(device)
    elif isinstance(layer, tf.keras.layers.LayerNormalization) and weights:
        if len(weights) > 0:
            gamma_name = f'transformer.h.{tf_base_model.layers.index(layer)}.ln_1.weight'
            state_dict[gamma_name] = torch.tensor(weights[0]).to(device)
        if len(weights) > 1:
            beta_name = f'transformer.h.{tf_base_model.layers.index(layer)}.ln_1.bias'
            state_dict[beta_name] = torch.tensor(weights[1]).to(device)
    elif isinstance(layer, tf.keras.layers.Dense) and layer.name == 'lm_head' and weights:
        state_dict['lm_head.weight'] = torch.tensor(weights[0].T).to(device)

# Save the PyTorch model weights to a writable directory
converted_weights_path = "/kaggle/working/adapter_model.bin"
torch.save(state_dict, converted_weights_path)

# Load the converted adapter weights into the PyTorch model
pytorch_model.load_state_dict(torch.load(converted_weights_path, map_location=device), strict=False)

# Move the model to the appropriate device
pytorch_model.to(device)

# Load the tokenizer from the base model path
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# Prepare the input prompt
input_text = "ഇന്ത്യയിലെ മഴയുടെ പ്രാധാന്യം"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

# Generate output
outputs = pytorch_model.generate(input_ids=input_ids, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=0.8)

# Decode and print the output text
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output_text)

