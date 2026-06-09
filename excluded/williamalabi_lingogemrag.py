!pip install git+https://github.com/stanfordnlp/pyreft.git -qq
!pip install peft -qq
!pip install rouge_score -qq
!pip install sentence-transformers faiss-cpu accelerate -qq

import os
import csv
import ast
import time
import torch
import faiss
import kagglehub
import numpy as np
import torchvision
import pandas as pd
import transformers
import torch.nn as nn
from tqdm import tqdm
from typing import List
import torch.optim as optim
from torch.nn import Softplus
import torch.nn.functional as F
from rouge_score import rouge_scorer
from IPython.display import display, Markdown
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from peft import (get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, prepare_model_for_kbit_training)
from pyreft import (get_reft_model, ReftModel, ReftConfig, ConsreftIntervention, TaskType, LoreftIntervention, ReftDataCollator, make_last_position_supervised_data_module, ReftSupervisedDataset, ReftTrainerForCausalLM)


# --- Model and Tokenizer Configuration ---
MODEL_PATH = "google/gemma-2-2b-it" # Specifies the path to the pre-trained Gemma 2 2B-it model on Hugging Face Hub.

# --------------------------------------------------
#  Load model configuration
# --------------------------------------------------
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, token="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") # Load the configuration of the pretrained model. Trust remote code allows the code to load the model definition from the web

# --------------------------------------------------
# Enable gradient checkpointing
# --------------------------------------------------
config.gradient_checkpointing = True # Activate gradient checkpointing to reduce memory consumption during training
# Load the tokenizer associated with the specified model path.
# It uses the provided access token to log in into Hugging Face, in order to use the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, config=config, token="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")

# --- Define the end-of-sequence (EOS) token as a stopping point for the generator.
stop = tokenizer.eos_token

# --- Add a pad token to the tokenizer
# A padding token is added to the tokenizer's vocabulary for handling sequences of variable lengths.
tokenizer.add_special_tokens({"pad_token": "<pad>"})

#--- Setting the padding token and the eos token to be the same
tokenizer.pad_token = tokenizer.eos_token

#--- Set the tokenizer's maximum sequence length to MAX_LENGTH to ensure consistency with the model and previous configurations.
tokenizer.model_max_length = 256 #Very important


# --- Configuration Parameters ---
batch_size = 4  # Batch size for training and inference. Set to 4 as we are processing four sequences at a time.
seed_value = 42 # Set random seed for reproducibility.
MAX_LENGTH = 256 # Maximum length of the generated sequence.
device = torch.device("cuda") # Set the device to GPU if available; otherwise, use CPU.
# Disable Flash and memory efficient SDP for debugging purposes and compatibility.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

# --- Helper Functions for Model Checkpointing ---
def checkpoint(model, filename):
    """
    Saves the state dictionary of the given model to the specified file.

    Args:
        model (torch.nn.Module): The model whose state dict needs to be saved.
        filename (str): The name of the file where the state dict should be saved.
    """
    torch.save(model.state_dict(), filename)

def resume(model, filename):
    """
    Loads the state dictionary from the specified file into the given model.

    Args:
        model (torch.nn.Module): The model into which the state dict should be loaded.
        filename (str): The name of the file from which the state dict should be loaded.
    """
    model.load_state_dict(torch.load(filename))

# --- Inference Function ---
def inference(text):
    """
    Generates a translation based on the given text using the provided model and tokenizer.

    Args:
        text (str): The input text to be translated.
        tokenizer: The tokenizer used to tokenize the input and output sequences.
        model: The trained model used for translation.
        stop: stop word for generating sequences

    Returns:
        str: The generated translated text.
    """
    # Tokenize the input text and move to the CUDA device.
    prompt = tokenizer(text, return_tensors="pt").to("cuda")

    # Calculate the base unit location for LoReFT. It refers to the last token of the input text.
    base_unit_location = prompt["input_ids"].shape[-1] - 1

    # Generate translated text using the model and apply LoReFT intervention on prompt.
    # unit_locations are used to define where the intervention should be applied.
    # intervene_on_prompt=True specifies that we want to apply the intervention in the prompt
    _, reft_response = model.generate(
        prompt,
        unit_locations={"sources->base": (None, [[[base_unit_location]]])},
        intervene_on_prompt=True,
        max_new_tokens=MAX_LENGTH, # Generates up to MAX_LENGTH tokens.
        do_sample=True,  # Enables sampling for a diversity in the output.
        remove_invalid_values=True, # Removes tokens considered invalid.
        eos_token_id=tokenizer.eos_token_id, # The end of sequence token ID.
        early_stopping=True,  # Stops generation when eos token is generated.
        num_beams=5 # Uses beam search for improved outputs.
    )

    # Decode the generated tokens back into text
    gen_text = tokenizer.decode(reft_response[:, prompt.input_ids.shape[1]:][0],
                              skip_special_tokens=True,  # Skips special tokens like [PAD].
                              return_full_text=False, # Returns only the generated portion without the prompt.
                              stop_token=stop) # Stops generating tokens if the given stop word is found

    return gen_text # Returns the generated translated text.
    


np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)


# --- Prompt Template ---
# Defines a template to format input text, guiding the model for translation.
prompt_no_input_template = """You are a helpful assistant that translates text from English to Hebrew without saying anything else. Translate the text below:\n
%s
"""

# --- Load the Dataset ---
# Loads the English-Hebrew text dataset from a CSV file into a Pandas DataFrame.
df = pd.read_csv("/kaggle/input/english-to-hebrew-bible-translations/English_Hebrew_Text.csv")

# --- Data Cleaning ---
# Removes rows containing any missing or NaN values and drops duplicates from the DataFrame.
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# --- Apply Prompt Template ---
# Formats the English text using the prompt template created previously, for all English texts.
df['English_Text'] = df['English_Text'].apply(lambda x: prompt_no_input_template % x)
df['English_Text'] = df['English_Text'] + " " + stop
df['Hebrew_Text'] = df['Hebrew_Text'] + " " + stop

# --- Print completion message ---
# A print message saying that all English text have been formatted, useful for debugging.
print("done")

# --- Train-Test Split ---
# Splits the dataset into training and testing sets with a 50-50 split, using a fixed random seed, because the data is about 31101 rows.
X_train, X_test, Y_train, Y_test = train_test_split(df["English_Text"].values,
                                                    df["Hebrew_Text"].values,
                                                    test_size=0.5,
                                                    random_state=42)

# --- Print Training Data Shape ---
# Prints the shape of the training data, useful for verifying the split.
print("X_train shape is:", X_train.shape)


def colorize_text(text):
    for word, color in zip(["English", "Hebrew Translation", "Original Translation"],
                           ["red", "green", "yellow"]):
        # Replace with proper HTML
        text = text.replace(f"{word}:", f"<br><br><span style='color:{color}; font-weight:bold;'>{word}:</span>")
    return text


# Take a random sample
sample = df.iloc[10]
sample = "English: " + str(sample['English_Text']) + "\n" + "Original Translation: " + str(sample['Hebrew_Text'])
# Give colors to English Text and Hebrew Translation
sample = colorize_text(sample)

# Show sample in markdown
display(Markdown(str(sample)))


model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, config=config, low_cpu_mem_usage=True, token="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")
with torch.no_grad():
	model.to(device)


# --------------------------------------------------
#  Configuration for LoRA (Low-Rank Adaptation)
# --------------------------------------------------
peft_config = LoraConfig(
    r=16,  # Rank of the low-rank matrices
    lora_alpha=64,  # Scaling factor for the LoRA matrices
    target_modules=["lm_head"], # Modules to apply LoRA to
    layers_to_transform=[15], # The specific layer(s) on which to apply lora transformations.
    use_rslora=True,  # Use Rank-Stabilized LoRA for better training stability
    lora_dropout=0.05,  # Dropout rate for LoRA layers
    bias="none",  # No bias in LoRA layers
    task_type="CAUSAL_LM" # Task type is Causal language modeling
)

# Apply LoRA to the base model using the config
model = get_peft_model(model, peft_config)


# --------------------------------------------------
# Configuration for ReFT (including LoreftIntervention)
# --------------------------------------------------
reft_config = ReftConfig(representations=[{
    # string component access is enforced for customized model such as a peft model!
    "layer": l,  # Layer to apply the intervention on
    "component": f"base_model.model.model.layers[{l}].output",  # Specific output of the layer to be modified
    "low_rank_dimension": 4,  # Low-rank dimension of the LoreftIntervention
    "intervention": LoreftIntervention(embed_dim=model.config.hidden_size, # The intervention itself: Low-Rank ReFT
    low_rank_dimension=4)} for l in [15]])

# Wrap the base model (with LoRA) with ReFT
model = get_reft_model(model, reft_config)

# Activate the adapter layers (LoRA and ReFT)
model.model.enable_adapter_layers()

# Print the number of trainable parameters in the model
model.print_trainable_parameters()


prompt = "Heaven helps those who help themselves."
prompt = prompt_no_input_template % prompt
response = inference(str(prompt))

sample = "English: " + str(prompt) + "\nHebrew Translation: " + str(response) + "\nOriginal Translation: " + str("השמיים עוזרים לאלה שעוזרים לעצמם.")
# Give colors to English text and Translation
sample = colorize_text(sample)

# Show sample in markdown
display(Markdown(str(sample)))


prompt = "And remember, that the greatest gift of all is the compassion and grace we share with each other."
prompt = prompt_no_input_template % prompt
response = inference(str(prompt))

sample = "English: " + str(prompt) + "\nHebrew Translation: " + str(response) + "\nOriginal Translation: " + str("וזכור שהמתנה הגדולה מכולן היא החמלה והחסד שאנו חולקים זה עם זה.")
# Give colors to English text and Translation
sample = colorize_text(sample)

# Show sample in markdown
display(Markdown(str(sample)))


n_epochs = 10 # Number of training epochs

data_module = make_last_position_supervised_data_module(
	tokenizer, model, X_train, Y_train)


# --------------------------------------------------
# Configuration for training
# --------------------------------------------------
training_args = transformers.TrainingArguments(
	num_train_epochs=n_epochs, # Total number of training epochs to perform
	output_dir="./tmp", # Directory where to save model outputs and checkpoints
    per_device_train_batch_size=batch_size, # Batch size per GPU during training
    report_to=[], # Disable reporting during training
	learning_rate=2e-3, # Learning rate for the AdamW optimizer
    logging_steps=50 # Number of steps to perform logging
)


# --------------------------------------------------
#  Initialize and start the ReFT trainer
# --------------------------------------------------
trainer = ReftTrainerForCausalLM( # Initialize the ReFT trainer for Causal Language Modeling
	model=model,   # The model to be trained, already with LoRA and ReFT applied
	tokenizer=tokenizer, # The tokenizer to use
    args=training_args,   # The training configuration
    **data_module # Dataset to be used for training
)

# Start the training loop and store the output for further use
_ = trainer.train()


lingogem_path = "./LingoGem"
model.save(lingogem_path)

#upload to Kaggle Models
USERNAME = "williamalabi"
MODEL_NAME = "LingoGem"
FRAMEWORK = "transformers"
VARIATION = "2b-it"

KAGGLE_URI = f"{USERNAME}/{MODEL_NAME}/{FRAMEWORK}/{VARIATION}"
kagglehub.model_upload(KAGGLE_URI, lingogem_path, 'Apache 2.0')


prompt = "Heaven helps those who help themselves."
prompt = prompt_no_input_template % prompt
response = inference(str(prompt))

sample = "English: " + str(prompt) + "\nHebrew Translation: " + str(response) + "\nOriginal Translation: " + str("השמיים עוזרים לאלה שעוזרים לעצמם.")
# Give colors to English text and Translation
sample = colorize_text(sample)

# Show sample in markdown
display(Markdown(str(sample)))


prompt = "And remember, that the greatest gift of all is the compassion and grace we share with each other."
prompt = prompt_no_input_template % prompt
response = inference(str(prompt))

sample = "English: " + str(prompt) + "\nHebrew Translation: " + str(response) + "\nOriginal Translation: " + str("וזכור שהמתנה הגדולה מכולן היא החמלה והחסד שאנו חולקים זה עם זה.")
# Give colors to English text and Translation
sample = colorize_text(sample)

# Show sample in markdown
display(Markdown(str(sample)))



# --- Configuration ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # Replace with your embedding model name
INDEX_DIR = "faiss_index"  # Directory to store FAISS index
CSV_FILES = ["/kaggle/input/opus-english-to-hebrew-csv/en2he_train.csv", "/kaggle/input/opus-english-to-hebrew-csv/en2he_test.csv"]  # List of CSV files

# Global variables to store loaded index, chunks and embedding model
_GLOBAL_INDEX = None
_GLOBAL_CHUNKS = None
_GLOBAL_EMBEDDING_MODEL = None
_GLOBAL_TOKENIZER = None
_GLOBAL_MODEL = None

# --- 1. Indexing ---
def load_and_chunk_documents_from_csv(csv_files: List[str], chunk_size=256):
    """Loads text from multiple CSV files and splits into chunks."""
    all_chunks = []
    for csv_file in csv_files:
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                english_text = row.get("en", "")
                hebrew_text = row.get("he", "")
                full_text = f"English: {english_text} Hebrew: {hebrew_text}"
                for i in range(0, len(full_text), chunk_size):
                    all_chunks.append(full_text[i:i + chunk_size])
    return all_chunks

def create_embeddings(texts: List[str], embedding_model):
    """Generates embeddings for the given texts."""
    return embedding_model.encode(texts)

def build_faiss_index(embeddings, index_dir="faiss_index", overwrite=False):
    """Builds a FAISS index from embeddings."""
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, "faiss_index.bin")

    if os.path.exists(index_path) and not overwrite:
        print(f"FAISS index found at {index_path}. Loading existing index.")
        index = faiss.read_index(index_path)
        return index
    else:
        print(f"Building FAISS index and saving to {index_path}.")
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)  # Or other suitable FAISS index
        index.add(embeddings)
        faiss.write_index(index, index_path)
        return index

def build_index(csv_files, embedding_model, overwrite=False):
    """Builds the vector index from multiple csv files."""
    chunks = load_and_chunk_documents_from_csv(csv_files)
    embeddings = create_embeddings(chunks, embedding_model)
    index = build_faiss_index(np.array(embeddings).astype('float32'), index_dir=INDEX_DIR, overwrite=overwrite)
    return index, chunks

# --- 2. Retrieval ---
def retrieve_context(query: str, embedding_model, index, chunks, top_k=3):
    """Retrieves the top-k context chunks based on query similarity."""
    query_embedding = embedding_model.encode(query).reshape(1, -1).astype("float32")
    _, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]


def generate_with_rag(query: str, embedding_model, index, chunks, tokenizer, model, top_k=3):
    """Generates text using RAG."""
    context_chunks = retrieve_context(query, embedding_model, index, chunks, top_k=top_k)
    augmented_prompt = f"Context: {' '.join(context_chunks)}\n\nQuery: {query}"
    response = inference(augmented_prompt)
    return response

# --- Main ---
def main(csv_files, query, overwrite_index=False):
    global _GLOBAL_INDEX, _GLOBAL_CHUNKS, _GLOBAL_EMBEDDING_MODEL, _GLOBAL_TOKENIZER, _GLOBAL_MODEL

    # Initialize embedding model, tokenizer and model only once
    if _GLOBAL_EMBEDDING_MODEL is None:
        _GLOBAL_EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    if _GLOBAL_TOKENIZER is None:
        _GLOBAL_TOKENIZER = tokenizer
    if _GLOBAL_MODEL is None:
         _GLOBAL_MODEL = model
         _GLOBAL_MODEL = _GLOBAL_MODEL.to("cuda")


    # Build index only once, if it doesn't exist or overwrite is True
    if _GLOBAL_INDEX is None or overwrite_index:
        start = time.time()
        _GLOBAL_INDEX, _GLOBAL_CHUNKS = build_index(csv_files, _GLOBAL_EMBEDDING_MODEL, overwrite=overwrite_index)
        end = time.time()
        print(f"Index creation time : {end-start} s")


    # Generate response using RAG
    start = time.time()
    rag_response = generate_with_rag(query, _GLOBAL_EMBEDDING_MODEL, _GLOBAL_INDEX, _GLOBAL_CHUNKS, _GLOBAL_TOKENIZER, _GLOBAL_MODEL, top_k=3)
    end = time.time()
    print(f"RAG response generation time : {end-start} s")
    print("RAG Response:", rag_response)
    return rag_response


# Initialize a RougeScorer object to calculate ROUGE scores.
# We specify the ROUGE types we want to compute ('rouge1', 'rouge2', 'rougeL')
# and set use_stemmer to True for more accurate text matching.
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Initialize lists to store generated responses and ROUGE scores.
response_lst = []
rouge_scores = []

# Iterate through each example in the test dataset.
# We zip X_test and Y_test to iterate through input-output pairs.
num_iterations = 5 # Set the number of iterations

for idx, (row_x, row_y) in enumerate(zip(X_test.tolist(), Y_test.tolist())):
    if idx >= num_iterations: # Break when the number of iterations has been reached
        break
	# Generate a prediction using the 'inference' function, converting input row_x to a string.
    response = main(CSV_FILES, row_x, overwrite_index=False)

	# Output a separator for readability, the input, and the model output
    print("----------------------------------------------------------------------------")
    print("Query is: ", row_x)
    print("Hebrew Translation is: ", response)

	# Store the generated response in response_lst.
    response_lst.append(response)

    # Calculate the ROUGE score by comparing the generated response to the true target translation
    # Store the score for calculating averages.
    scores = scorer.score(row_y, response)
    print(scores)
    rouge_scores.append(scores)


# Calculate the average ROUGE-1 score by extracting f-measure from all stored scores
avg_rouge1 = np.mean([score['rouge1'].fmeasure for score in rouge_scores])
# Calculate the average ROUGE-2 score by extracting f-measure from all stored scores
avg_rouge2 = np.mean([score['rouge2'].fmeasure for score in rouge_scores])
# Calculate the average ROUGE-L score by extracting f-measure from all stored scores
avg_rougeL = np.mean([score['rougeL'].fmeasure for score in rouge_scores])


# Print the average ROUGE scores to evaluate model performance.
print(f"Average ROUGE-1: {avg_rouge1:.4f}")
print(f"Average ROUGE-2: {avg_rouge2:.4f}")
print(f"Average ROUGE-L: {avg_rougeL:.4f}")


prompt = "Heaven helps those who help themselves."
prompt = prompt_no_input_template % prompt
response = main(CSV_FILES, prompt, overwrite_index=False)

sample = "English: " + str(prompt) + "\nHebrew Translation: " + str(response) + "\nOriginal Translation: " + str("השמיים עוזרים לאלה שעוזרים לעצמם.")
# Give colors to English text and Translation
sample = colorize_text(sample)

# Show sample in markdown
display(Markdown(str(sample)))


prompt = "And remember, that the greatest gift of all is the compassion and grace we share with each other."
prompt = prompt_no_input_template % prompt
response = main(CSV_FILES, prompt, overwrite_index=False)

sample = "English: " + str(prompt) + "\nHebrew Translation: " + str(response) + "\nOriginal Translation: " + str("השמיים עוזרים לאלה שעוזרים לעצמם.")
# Give colors to English text and Translation
sample = colorize_text(sample)

# Show sample in markdown
display(Markdown(str(sample)))

