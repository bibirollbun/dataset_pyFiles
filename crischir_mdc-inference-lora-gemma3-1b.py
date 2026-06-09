# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# !pip install -q -U transformers
# !pip install -q -U accelerate
# !pip install -q -U datasets
# !pip install -q -U peft
# !pip install -q -i https://pypi.org/simple/ bitsandbytes
# !pip install -q -U trl
# !pip install pymupdf


# # Setați calea către dataset-ul cu pachetele descărcate
# package_path = '/kaggle/input/trl-package'

# # Instalați pachetele
# # !pip install --no-index --find-links=$package_path trl


import kagglehub
import re


os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import warnings
warnings.filterwarnings("ignore")


max_tokens=1000
max_tokens_and_prompt=max_tokens+100


# import numpy as np
# import pandas as pd
# import os
from tqdm import tqdm

import torch
import torch.nn as nn

import transformers
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          TrainingArguments, # Note: SFTConfig from TRL is used later
                          pipeline,
                          logging)

# Explicitly import Gemma3ForCausalLM
from transformers.models.gemma3 import Gemma3ForCausalLM

from datasets import Dataset
from peft import LoraConfig, PeftConfig, PeftModel
from trl import SFTTrainer, SFTConfig # Use SFTConfig from TRL
import bitsandbytes as bnb

from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix)

from sklearn.model_selection import train_test_split

# Check transformers version
print(f"transformers=={transformers.__version__}")


def define_device():
    """Determine and return the optimal PyTorch device based on availability."""
    print(f"PyTorch version: {torch.__version__}", end=" -- ")

    # Check for CUDA availability first
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("using CUDA device")
        return device

    # Check for MPS (Metal Performance Shaders) for macOS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("using MPS device on macOS")
        return torch.device("mps")

    # Fallback to CPU if no GPU or MPS is available
    print("using CPU")
    return torch.device("cpu")

# Select the best available device (CPU, CUDA, or MPS)
device = define_device()
print(f"Operating on {device}")



# Determine optimal computation dtype based on the selected device
if device.type == 'cuda':
    # Use bfloat16 for GPUs with Compute Capability >= 8.0, otherwise float16
    compute_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"Using compute dtype {compute_dtype}")
else:
    # Use float32 for CPU and MPS
    compute_dtype = torch.float32
    print(f"Using compute dtype {compute_dtype}")


GEMMA_PATH = kagglehub.model_download("google/gemma-3/transformers/gemma-3-1b-it")

# Load the model with optimized settings
model = Gemma3ForCausalLM.from_pretrained(
    GEMMA_PATH,
    torch_dtype=compute_dtype,           # Use the PyTorch dtype
    attn_implementation="eager",         # Specify attention implementation
    low_cpu_mem_usage=True,              # Reduces CPU RAM usage during loading
    device_map=device                    # Automatically map model layers to the device
)

# Define maximum sequence length for the tokenizer
max_seq_length = 8192

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    GEMMA_PATH,
    max_seq_length=max_seq_length
)

# Store the EOS token for later use in prompts
EOS_TOKEN = tokenizer.eos_token


# # Determine optimal computation dtype based on GPU capability
# # Use bfloat16 if Compute Capability >= 8.0, otherwise float16
# compute_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
# print(f"Using compute dtype {compute_dtype}")

# # Select the best available device (CPU, CUDA, or MPS)
# device = define_device()
# print(f"Operating on {device}")

# #====
# from transformers import AutoTokenizer
# from transformers.models.gemma3 import Gemma3ForCausalLM

# # Download the model files to a local directory and get the path.
# GEMMA_PATH = kagglehub.model_download("google/gemma-3/transformers/gemma-3-1b-it")

# # Load the tokenizer from the local path.
# # The `local_files_only=True` flag is crucial.
# tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH, local_files_only=True)

# # Load the model from the local path.
# # `Gemma3ForCausalLM` doesn't need the flag if you provide the local path.
# # This should work as is with the local directory.
# # model = Gemma3ForCausalLM.from_pretrained(GEMMA_PATH)



# #======

# # Path to the pre-trained model (adjust if necessary)
# # GEMMA_PATH = "/kaggle/input/gemma-3/transformers/gemma-3-1b-it/1"

# # Load the model with optimized settings
# model = Gemma3ForCausalLM.from_pretrained(
#     GEMMA_PATH,
#     dtype=compute_dtype,  # Use the new argument name
#     attn_implementation="eager", # Specify attention implementation
#     low_cpu_mem_usage=True,      # Reduces CPU RAM usage during loading
#     device_map=device            # Automatically map model layers to the device
# )

# # Define maximum sequence length for the tokenizer
# max_seq_length = 8192 # Gemma 3 supports long contexts

# # Load the tokenizer
# tokenizer = AutoTokenizer.from_pretrained(
#     GEMMA_PATH,
#     max_seq_length=max_seq_length,
#     device_map=device # Map tokenizer operations if relevant (less common)
# )

# # Store the EOS token for later use in prompts
# EOS_TOKEN = tokenizer.eos_token


# Check if all model parameters are on the CUDA device
is_on_gpu = all(param.device.type == 'cuda' for param in model.parameters())
print("Model is on GPU:", is_on_gpu)


import fitz # PyMuPDF


from typing import Callable, Any


def process_pdfs_for_inference(directory_path: str, max_tokens: int, tokenizer_obj: Any) -> pd.DataFrame:
    """
    Reads all PDF files from a directory, extracts text, and splits it into chunks
    ready for model inference.

    Args:
        directory_path (str): The path to the directory containing PDF files.
        max_tokens (int): The maximum number of tokens for each text chunk.
        tokenizer_obj (Any): An object with `encode` and `decode` methods.
        
    Returns:
        pd.DataFrame: A DataFrame with text chunks and their corresponding source file.
    """
    inference_chunks = []
    
    # Check if the directory exists.
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        return pd.DataFrame(columns=['file_name', 'text_chunk'])
        
    print(f"Processing PDF files in directory: {directory_path}")

    for file_name in os.listdir(directory_path):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(directory_path, file_name)
            full_text = ""
            try:
                # Open the PDF file
                doc = fitz.open(file_path)
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    full_text += page.get_text()
                doc.close()

                # Chunk the extracted text
                if full_text:
                    # Correct: Use tokenizer.encode() to get a list of integer IDs
                    tokens_list = tokenizer_obj.encode(full_text)
                    num_chunks = (len(tokens_list) + max_tokens - 1) // max_tokens

                    for i in range(num_chunks):
                        start_index = i * max_tokens
                        end_index = start_index + max_tokens
                        chunk_tokens = tokens_list[start_index:end_index]
                        
                        # Correct: Now tokenizer.decode() receives a list of integers
                        chunk_text = tokenizer_obj.decode(chunk_tokens)
                        
                        # Store the chunk with a reference to the source file
                        inference_chunks.append({
                            'file_name': file_name,
                            'text_chunk': chunk_text
                        })

            except Exception as e:
                print(f"Could not process file {file_name}: {e}")

    return pd.DataFrame(inference_chunks)


max_tokens=1000
max_tokens_and_prompt=max_tokens+100


# Now, we process a directory of PDFs for inference
pdf_directory = "/kaggle/input/make-data-count-finding-data-references/test/PDF/"
inference_df = process_pdfs_for_inference(pdf_directory, max_tokens, tokenizer)
print("\nInference DataFrame (created from PDF files):")
# print(inference_df)


inference_df.columns


inference_df





# -- New Prompt Generation Functions --
# Function to generate training prompts (with labels for type and dataset_id)
EOS_TOKEN = "</s>"

# Function to generate training prompts
def generate_train_prompt(data_point):
    return f"""<|system|>
You are an expert at extracting data citations from text. Your task is to identify the 'type'of citation  and 'dataset_id' (doi or accession number) and return a JSON object.
The 'type' of citation can be: Primary - raw or processed data cited was generated by text authors, specifically for the study; Secondary - cited raw or processed data is derived or reused from existing records or published data; The other value is Missing
<|user|>
Text: {data_point["text_chunk"]}
<|assistant|>
{{"type": "{data_point["type"]}", "dataset_id": "{data_point["dataset_id"]}"}}</s>"""

# Function to generate test/evaluation prompts
def generate_test_prompt(data_point):
    return f"""<|system|>
You are an expert at extracting data citations from text. Your task is to identify the 'type'of citation  and 'dataset_id' (doi or accession number) and return a JSON object.
The 'type' of citation can be: Primary - raw or processed data cited was generated by text authors, specifically for the study; Secondary - cited raw or processed data is derived or reused from existing records or published data; The other value is Missing
<|user|>
Text: {data_point["text_chunk"]}
<|assistant|>
"""
# Function to generate training prompts
def generate_eval_prompt(data_point):
    return f"""<|system|>
You are an expert at extracting data citations from text. Your task is to identify the 'type'of citation  and 'dataset_id' (doi or accession number) and return a JSON object.
The 'type' of citation can be: Primary - raw or processed data cited was generated by text authors, specifically for the study; Secondary - cited raw or processed data is derived or reused from existing records or published data; The other value is Missing
<|user|>
Text: {data_point["text"]}
<|assistant|>
"""


X_test_prompts = pd.DataFrame(inference_df.apply(generate_test_prompt, axis=1), columns=["text_chunk"])


X_test_prompts





test_data = Dataset.from_pandas(X_test_prompts)





def predict(X_test_df, model_to_use, tokenizer_to_use, device_to_use=None, max_new_tokens=100, temperature=0.0):
    """Predict the type and dataset_id using the provided model and tokenizer with robust parsing.

    Args:
        X_test_df (pd.DataFrame): DataFrame containing the text chunks to predict.
        model_to_use: The model for prediction.
        tokenizer_to_use: The tokenizer for the model.
        device_to_use (str, optional): The device to run the model on (e.g., 'cuda' or 'cpu').
        max_new_tokens (int): The maximum number of tokens to generate.
        temperature (float): The temperature for generation.
    
    Returns:
        list: A list of dictionaries with 'type' and 'dataset_id' keys for each prediction.
    """
    if device_to_use is None:
        device_to_use = "cuda" if torch.cuda.is_available() else "cpu"
    
    y_pred = []
    model_to_use.eval()

    # Define the pattern to find the JSON object, which might be in a markdown code block
    json_pattern = re.compile(r'```json\s*(\{.*?\})\s*```|(\{.*?\})', re.DOTALL)

    for i in tqdm(range(len(X_test_df)), desc="Predicting Outputs"):
        prompt = X_test_df.iloc[i]["text_chunk"]

        input_ids = tokenizer_to_use(prompt, return_tensors="pt").to(device_to_use)

        with torch.no_grad():
            outputs = model_to_use.generate(
                **input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                pad_token_id=tokenizer_to_use.eos_token_id
            )

        full_decoded_text = tokenizer_to_use.decode(outputs[0], skip_special_tokens=True)
        
        # --- Robust JSON Parsing ---
        parsed_prediction = {'type': 'unknown', 'dataset_id': 'unknown'}
        
        # Search for a JSON object in the generated text
        match = json_pattern.search(full_decoded_text)
        
        if match:
            # The regex returns a tuple of groups; the JSON is in the first or second group
            json_str = match.group(1) or match.group(2)
            try:
                # Replace single quotes with double quotes for valid JSON parsing
                json_str = json_str.replace("'", '"')
                prediction = json.loads(json_str)
                # Check if the parsed object has the expected keys
                if 'type' in prediction and 'dataset_id' in prediction:
                    parsed_prediction = prediction
            except (json.JSONDecodeError, KeyError) as e:
                # Log a warning if the found JSON is malformed
                print(f"Warning: Found a potential JSON, but parsing failed. Error: {e}")
                
        else:
            # Fallback for when no JSON pattern is found
            print(f"Warning: No valid JSON pattern found for text chunk at index {i}.")

        y_pred.append(parsed_prediction)
        
    return y_pred


def predict_single_value(input_text, model_to_use, tokenizer_to_use, device_to_use=None, max_new_tokens=100, temperature=0.0):
    """Predict the type and dataset_id for a single text value with robust parsing.
    
    Args:
        input_text (str): The single text string to analyze.
        model_to_use: The model for prediction.
        tokenizer_to_use: The tokenizer for the model.
        device_to_use (str): The device to run the model on (e.g., 'cuda' or 'cpu').
        max_new_tokens (int): The maximum number of tokens to generate.
        temperature (float): The temperature for generation.
        
    Returns:
        dict: A dictionary with 'type' and 'dataset_id' keys.
    """
    if device_to_use is None:
        device_to_use = "cuda" if torch.cuda.is_available() else "cpu"

    model_to_use.eval()
    
    # Define the pattern to find the JSON object
    json_pattern = re.compile(r'```json\s*(\{.*?\})\s*```|(\{.*?\})', re.DOTALL)

    # Convert the single input text into a prompt
    prompt = f"Text: {input_text}\n\nTask: Analyze the text and provide the 'type' and 'dataset_id' in a JSON object.\n\n<|assistant|>"

    input_ids = tokenizer_to_use(prompt, return_tensors="pt").to(device_to_use)

    with torch.no_grad():
        outputs = model_to_use.generate(
            **input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            pad_token_id=tokenizer_to_use.eos_token_id
        )

    full_decoded_text = tokenizer_to_use.decode(outputs[0], skip_special_tokens=True)

    # --- Robust JSON Parsing ---
    parsed_prediction = {'type': 'unknown', 'dataset_id': 'unknown'}
    
    # Search for a JSON object in the generated text
    match = json_pattern.search(full_decoded_text)
    
    if match:
        json_str = match.group(1) or match.group(2)
        try:
            prediction = json.loads(json_str)
            if 'type' in prediction and 'dataset_id' in prediction:
                parsed_prediction = prediction
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Found a potential JSON, but parsing failed. Error: {e}")
            
    else:
        print(f"Warning: No valid JSON pattern found in generated text.")

    return parsed_prediction


# import shutil
# # Define the source and destination directories
# # Define the source and destination directories
# source_dir = "/kaggle/input/lora-gemma3-mdc/transformers/default/1/LoRA-Gemma3-1B-MDC"
# destination_dir = "/kaggle/working/LoRA-Gemma3-1B-MDC"

# print(f"Checking for source directory: {source_dir}")

# # Check if the source directory exists
# if not os.path.isdir(source_dir):
#     print(f"Error: Source directory '{source_dir}' not found.")
# else:
#     print(f"Source directory found. Copying contents to {destination_dir}...")
    
#     # Check if the destination directory already exists. If it does, we assume the copy is complete.
#     if os.path.isdir(destination_dir):
#         print("Destination directory already exists. Skipping copy.")
#     else:
#         try:
#             # Copy the entire directory tree from source to destination
#             shutil.copytree(source_dir, destination_dir)
#             print("Successfully copied files.")
#         except Exception as e:
#             print(f"An error occurred during copying: {e}")

# lora_directory = "/kaggle/working/LoRA-Gemma3-1B-MDC"
# # Verify that the directory exists before trying to load it
# if not os.path.isdir(lora_directory):
#     raise FileNotFoundError(f"The directory '{lora_directory}' does not exist. Please check the path where you saved your model.")


# def copy_model_files():
#     """Copies model files from the Kaggle input directory to the working directory."""
    
#     # Define the source and destination directories
#     source_dir_main = "/kaggle/input/lora-gemma3-mdc/transformers/default/1/LoRA-Gemma3-1B-MDC"
#     source_dir_alt = "/kaggle/input/lora-gemma3-mdc/LoRA-Gemma3-1B-MDC"
#     destination_dir = "/kaggle/working/LoRA-Gemma3-1B-MDC"
    
#     # Determine the correct source directory to use
#     if os.path.isdir(source_dir_main):
#         source_dir = source_dir_main
#     elif os.path.isdir(source_dir_alt):
#         source_dir = source_dir_alt
#     else:
#         print(f"Error: Neither of the source directories were found.")
#         print(f"Please check the path to your LoRA model in the Kaggle input data.")
#         return

#     print(f"Source directory found: {source_dir}")

#     # Remove destination directory if it exists to ensure a clean copy
#     if os.path.isdir(destination_dir):
#         print("Destination directory already exists. Removing it to ensure a clean copy.")
#         try:
#             shutil.rmtree(destination_dir)
#         except Exception as e:
#             print(f"An error occurred while removing the directory: {e}")

#     # Now, copy the entire directory tree from source to destination
#     try:
#         shutil.copytree(source_dir, destination_dir)
#         print("Successfully copied files.")

#         # Final check to confirm the adapter config file is in place
#         if os.path.exists(os.path.join(destination_dir, 'adapter_config.json')):
#             print("Verified: 'adapter_config.json' exists in the destination directory.")
#         else:
#             print("Warning: 'adapter_config.json' was not found after copying.")

#     except Exception as e:
#         print(f"An error occurred during copying: {e}")

# # Execute the function to perform the copy
# copy_model_files()


# # Load the base model again (ensure enough RAM/VRAM)
# base_model = AutoModelForCausalLM.from_pretrained(
#     GEMMA_PATH,
#     torch_dtype=compute_dtype, # Use the same dtype as training
#     low_cpu_mem_usage=True,
#     device_map='auto' # Let transformers handle device mapping
# )
# # lora_directory="/kaggle/input/lora-gemma3-mdc/transformers/default/1/LoRA-Gemma3-1B-MDC/"
# print("Loading PeftModel and merging...")
# # Load the PeftModel by combining base model and LoRA adapter
# # peft_model = PeftModel.from_pretrained(base_model, lora_directory)

# # Merge the LoRA weights into the base model
# # merged_model = peft_model.merge_and_unload()
# print("Merging complete.")
# base_model.load_lora_adapter(lora_directory, weight_name="model.safetensors")

# # --- Save Merged Model and Tokenizer ---
# # merged_model_directory = "merged-Gemma3-1B-MDC"
# # print(f"Saving merged model to {merged_model_directory}...")
# # merged_model.save_pretrained(merged_model_directory,
# #                              safe_serialization=True, # Recommended format
# #                              max_shard_size="2GB")    # Shard large models if needed
# # print("Merged model saved.")

# # print(f"Saving tokenizer to {merged_model_directory}...")
# # Load the tokenizer from the LoRA directory and save it with the merged model
# tokenizer_for_merged = AutoTokenizer.from_pretrained(lora_directory)
# tokenizer_for_merged.save_pretrained(merged_model_directory)
# print("Tokenizer saved.")


# Path to your saved LoRA adapter
lora_directory = "LoRA-Gemma3-1B-MDC-peft"
#######
# # Load the base model  #####
# base_model_path = GEMMA_PATH
# base_model = AutoModelForCausalLM.from_pretrained(
#     base_model_path,
#     torch_dtype=torch.bfloat16,
#     device_map="auto"
# )
# ###############
# Load the LoRA adapter onto the base model
lora_directory="/kaggle/input/lora-gemma3-mdc/transformers/default/2/LoRA-Gemma3-1B-MDC-peft"
peft_model = PeftModel.from_pretrained(model, lora_directory)

# Merge the LoRA weights into the base model
merged_model = peft_model.merge_and_unload()

# Save the full, merged model to a new directory
# merged_model.save_pretrained("LoRA-Gemma3-1B-MDC-full-model")

# Also save the tokenizer
tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH)
# tokenizer.save_pretrained("LoRA-Gemma3-1B-MDC-full-model")


# from peft import PeftModel
# from peft import get_peft_model
# lora_directory2 = "LoRA-Gemma3-1B-MDC-peft"

# peft_config = LoraConfig(
#     lora_alpha=32,        # Scaling factor for LoRA weights
#     lora_dropout=0.05,    # Dropout probability for LoRA layers
#     r=64,                 # Rank of the LoRA decomposition (higher r = more parameters)
#     bias="none",          # Whether to train bias parameters ('none', 'all', or 'lora_only')
#     task_type="CAUSAL_LM", # Task type is Causal Language Modeling
#     target_modules="all-linear", # Apply LoRA to all linear layers
# )
# # Your base model and LoRA config
# # model = ...
# # lora_config = ...

# # Wrap the model with the PEFT config
# peft_model = get_peft_model(model, peft_config)

# # Save the adapter
# peft_model.save_pretrained(lora_directory2)


import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def evaluate_classification(y_true_list, y_pred_list):
    """
    Evaluates the performance of a multi-class classification model.

    This function provides a comprehensive report including precision, recall,
    F1-score, and a confusion matrix, which are more meaningful than
    simple accuracy for imbalanced datasets.

    Args:
        y_true_list (list): The list of true string labels.
        y_pred_list (list): The list of predicted string labels.

    Returns:
        A dictionary containing:
        - 'report': A string of the detailed classification report.
        - 'accuracy': The overall accuracy score (float).
        - 'confusion_matrix': A numpy array of the confusion matrix.
    """
    accuracy = accuracy_score(y_true_list, y_pred_list)
    report = classification_report(y_true_list, y_pred_list, zero_division=0)
    cm = confusion_matrix(y_true_list, y_pred_list)

    return {
        'report': report,
        'accuracy': accuracy,
        'confusion_matrix': cm
    }

def evaluate_information_extraction(y_true_list, y_pred_list):
    """
    Evaluates the performance of an information extraction model for dataset_id values.

    This function is suitable for problems with "unlimited" unique values.
    It calculates precision, recall, and F1-score based on true positives,
    false positives, and false negatives from the two lists.

    Args:
        y_true_list (list): The list of true dataset_id values.
        y_pred_list (list): The list of predicted dataset_id values.

    Returns:
        A dictionary containing the precision, recall, and F1-score.
    """
    # Convert lists to sets to find unique items and perform set operations
    y_true_set = set(y_true_list)
    y_pred_set = set(y_pred_list)

    # True Positives (TP): items that are correctly identified in both sets
    true_positives = len(y_true_set.intersection(y_pred_set))

    # False Positives (FP): items that are predicted but are not in the true set
    false_positives = len(y_pred_set.difference(y_true_set))

    # False Negatives (FN): items that are in the true set but are not predicted
    false_negatives = len(y_true_set.difference(y_pred_set))

    # Calculate Precision: TP / (TP + FP)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

    # Calculate Recall: TP / (TP + FN)
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    # Calculate F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

# --- Example Usage for Information Extraction ---

# Example lists representing true and predicted dataset_id values
y_true_info_ext = [
    'GSE67047',
    'https://doi.org/10.6096/aeroclo.1754',
    'SAMN16233664',
    'https://doi.org/10.5517/cc1k2lx4'
]

y_pred_info_ext = [
    'GSE67047',  # Correctly identified (TP)
    'https://doi.org/10.6096/aeroclo.1754',  # Correctly identified (TP)
    'http://data.xyz.org/dataset/new-dataset',  # Incorrectly identified (FP)
    'SAMN16233664', # Correctly identified (TP)
    'https://doi.org/10.5517/cc1k2lx4'
]


# Evaluate the information extraction results
info_ext_results = evaluate_information_extraction(y_true_info_ext, y_pred_info_ext)

# Print the results
print("--- Information Extraction Results ---")
print(f"Precision: {info_ext_results['precision']:.4f}")
print(f"Recall: {info_ext_results['recall']:.4f}")
print(f"F1-Score: {info_ext_results['f1_score']:.4f}")

print("\nBreakdown:")
print(f"True Positives: {info_ext_results['true_positives']}")
print(f"False Positives: {info_ext_results['false_positives']}")
print(f"False Negatives: {info_ext_results['false_negatives']}")



y_pred_info_ext


def predict(X_test_df, model_to_use, tokenizer_to_use, device_to_use=None, max_new_tokens=100, temperature=0.0):
    """Predict the type and dataset_id using the provided model and tokenizer with robust parsing.

    Args:
        X_test_df (pd.DataFrame): DataFrame containing the text chunks to predict.
        model_to_use: The model for prediction.
        tokenizer_to_use: The tokenizer for the model.
        device_to_use (str, optional): The device to run the model on (e.g., 'cuda' or 'cpu').
        max_new_tokens (int): The maximum number of tokens to generate.
        temperature (float): The temperature for generation.
    
    Returns:
        list: A list of dictionaries with 'type' and 'dataset_id' keys for each prediction.
    """
    if device_to_use is None:
        device_to_use = "cuda" if torch.cuda.is_available() else "cpu"
    
    y_pred = []
    model_to_use.eval()

    # Define the pattern to find the JSON object, which might be in a markdown code block
    json_pattern = re.compile(r'```json\s*(\{.*?\})\s*```|(\{.*?\})', re.DOTALL)

    for i in tqdm(range(len(X_test_df)), desc="Predicting Outputs"):
        # Use a structured prompt to guide the model towards the desired JSON format
        prompt = f"Text: {X_test_df.iloc[i]['text_chunk']}\n\nTask: Analyze the text and provide the 'type' and 'dataset_id' in a JSON object.\n\n<|assistant|>"

        input_ids = tokenizer_to_use(prompt, return_tensors="pt").to(device_to_use)

        with torch.no_grad():
            outputs = model_to_use.generate(
                **input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                pad_token_id=tokenizer_to_use.eos_token_id
            )

        full_decoded_text = tokenizer_to_use.decode(outputs[0], skip_special_tokens=True)
        
        # --- Robust JSON Parsing ---
        parsed_prediction = {'type': 'unknown', 'dataset_id': 'unknown'}
        
        # Search for a JSON object in the generated text
        match = json_pattern.search(full_decoded_text)
        
        if match:
            # The regex returns a tuple of groups; the JSON is in the first or second group
            json_str = match.group(1) or match.group(2)
            try:
                # Replace single quotes with double quotes for valid JSON parsing
                json_str = json_str.replace("'", '"')
                prediction = json.loads(json_str)
                # Check if the parsed object has the expected keys
                if 'type' in prediction and 'dataset_id' in prediction:
                    parsed_prediction = prediction
            except (json.JSONDecodeError, KeyError) as e:
                # Log a warning if the found JSON is malformed
                print(f"Warning: Found a potential JSON, but parsing failed. Error: {e}")
                
        else:
            # Fallback for when no JSON pattern is found
            print(f"Warning: No valid JSON pattern found for text chunk at index {i}.")
            parsed_prediction = {'type': 'Missing', 'dataset_id': 'Missing'}

        y_pred.append(parsed_prediction)
        
    return y_pred


# Define the path for the training and validation data
training_file_path = "/kaggle/input/dummy-train/training.csv"
validation_file_path = "/kaggle/input/dummy-train/validation.csv"



# Load the training dataset with the new column names
df_train = pd.read_csv(training_file_path, 
                       encoding="utf-8", 
                       encoding_errors="replace")
df_train=df_train[["article_id", "dataset_id", "type", "text_chunk"]]
# Load the validation dataset
df_eval = pd.read_csv(validation_file_path, 
                      encoding="utf-8", 
 
                      encoding_errors="replace")
df_eval=df_eval[["article_id", "dataset_id", "type", "text_chunk"]]
# if test:
#     df_eval=df_eval[:20].copy()
# The columns to be predicted are 'type' and 'dataset_id'.
# 'type' is the primary stratification column.

# Stratified train-test split for the training data
# Note: The split is based on the 'type' column.
# X_train_full, X_test_full = train_test_split(df_train,
#                                              test_size=0.2, # Adjust the size as needed
#                                              random_state=42,
#                                              stratify=df_train['type'])
X_train_full, X_test_full = train_test_split(df_train,
                                             test_size=0.1,
                                             random_state=42)
# We now have the training data (X_train_full) and test data (X_test_full) from the original training file.
# The validation data is already in df_eval.
y_true_test = X_test_full[['type', 'dataset_id']].copy()
y_true_eval = df_eval[['type', 'dataset_id']].copy()


 #Apply prompt generation to create the final text column for training and evaluation
X_train_prompts = pd.DataFrame(X_train_full.apply(generate_train_prompt, axis=1), columns=["text"])
# X_test_prompts = pd.DataFrame(X_test_full.apply(generate_test_prompt, axis=1), columns=["text"])
X_test_prompts = pd.DataFrame(X_test_full.apply(generate_train_prompt, axis=1), columns=["text"])
X_eval_prompts = pd.DataFrame(df_eval.apply(generate_train_prompt, axis=1), columns=["text"])

# Convert pandas DataFrames to Hugging Face Dataset objects
train_data = Dataset.from_pandas(X_train_prompts)
test_data = Dataset.from_pandas(X_test_prompts)
eval_data = Dataset.from_pandas(X_eval_prompts)


X_eval_prompts


X_eval_prompts = pd.DataFrame(X_eval_prompts.apply(generate_eval_prompt, axis=1), columns=["text"])


X_eval_prompts=X_eval_prompts.rename(columns={"text": "text_chunk"})


X_eval_prompts


y_true_eval


import json


y_pred_eval = predict(X_eval_prompts , merged_model, tokenizer)


y_pred_df=pd.DataFrame(y_pred_eval)


y_pred_df



def evaluate_information_extraction(y_true_list, y_pred_list):
    """
    Evaluates the performance of an information extraction model for dataset_id values.

    This function is suitable for problems with "unlimited" unique values.
    It calculates precision, recall, and F1-score based on true positives,
    false positives, and false negatives from the two lists.

    Args:
        y_true_list (list): The list of true dataset_id values.
        y_pred_list (list): The list of predicted dataset_id values.

    Returns:
        A dictionary containing the precision, recall, and F1-score.
    """
    # Convert lists to sets to find unique items and perform set operations
    y_true_set = set(y_true_list)
    y_pred_set = set(y_pred_list)

    # True Positives (TP): items that are correctly identified in both sets
    true_positives = len(y_true_set.intersection(y_pred_set))

    # False Positives (FP): items that are predicted but are not in the true set
    false_positives = len(y_pred_set.difference(y_true_set))

    # False Negatives (FN): items that are in the true set but are not predicted
    false_negatives = len(y_true_set.difference(y_pred_set))

    # Calculate Precision: TP / (TP + FP)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

    # Calculate Recall: TP / (TP + FN)
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    # Calculate F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


y_true_list = y_true_eval['dataset_id'].tolist()
y_pred_list = y_pred_df['dataset_id'].tolist()
# Now, you can pass these lists to the evaluation function.
info_ext_results = evaluate_information_extraction(y_true_list, y_pred_list)

# Print the results
print("--- Information Extraction Results (from DataFrames) ---")
print(f"Precision: {info_ext_results['precision']:.4f}")
print(f"Recall: {info_ext_results['recall']:.4f}")
print(f"F1-Score: {info_ext_results['f1_score']:.4f}")

print("\nBreakdown:")
print(f"True Positives: {info_ext_results['true_positives']}")
print(f"False Positives: {info_ext_results['false_positives']}")
print(f"False Negatives: {info_ext_results['false_negatives']}")


y_true_list = y_true_eval['type'].tolist()
y_pred_list = y_pred_df['type'].tolist()
# Now, you can pass these lists to the evaluation function.
info_ext_results = evaluate_information_extraction(y_true_list, y_pred_list)

# Print the results
print("--- Information Extraction Results (from DataFrames) ---")
print(f"Precision: {info_ext_results['precision']:.4f}")
print(f"Recall: {info_ext_results['recall']:.4f}")
print(f"F1-Score: {info_ext_results['f1_score']:.4f}")

print("\nBreakdown:")
print(f"True Positives: {info_ext_results['true_positives']}")
print(f"False Positives: {info_ext_results['false_positives']}")
print(f"False Negatives: {info_ext_results['false_negatives']}")


y_true_list = y_true_eval['type'].tolist()
y_pred_list = y_pred_df['type'].tolist()
classification_results = evaluate_classification(y_true_list, y_pred_list)

# Print the results
print(f"\nOverall Accuracy: {classification_results['accuracy']:.2f}")
print("\nClassification Report:")
print(classification_results['report'])
print("Confusion Matrix:")
print(classification_results['confusion_matrix'])


import json


X_test_prompts.text[427]


predict_single_value(X_test_prompts.text[427], merged_model, tokenizer)


predict_single_value(X_test_prompts.text[427],model, tokenizer)


def predict(X_test_df, model_to_use, tokenizer_to_use, device_to_use=None, max_new_tokens=100, temperature=0.0):
    """Predict the type and dataset_id using the provided model and tokenizer with robust parsing.

    Args:
        X_test_df (pd.DataFrame): DataFrame containing the text chunks to predict.
        model_to_use: The model for prediction.
        tokenizer_to_use: The tokenizer for the model.
        device_to_use (str, optional): The device to run the model on (e.g., 'cuda' or 'cpu').
        max_new_tokens (int): The maximum number of tokens to generate.
        temperature (float): The temperature for generation.
    
    Returns:
        list: A list of dictionaries with 'type' and 'dataset_id' keys for each prediction.
    """
    if device_to_use is None:
        device_to_use = "cuda" if torch.cuda.is_available() else "cpu"
    
    y_pred = []
    model_to_use.eval()

    # Define the pattern to find the JSON object, which might be in a markdown code block
    json_pattern = re.compile(r'```json\s*(\{.*?\})\s*```|(\{.*?\})', re.DOTALL)

    for i in tqdm(range(len(X_test_df)), desc="Predicting Outputs"):
        # Use a structured prompt to guide the model towards the desired JSON format
        prompt = f"Text: {X_test_df.iloc[i]['text']}\n\nTask: Analyze the text and provide the 'type' and 'dataset_id' in a JSON object.\n\n<|assistant|>"

        input_ids = tokenizer_to_use(prompt, return_tensors="pt").to(device_to_use)

        with torch.no_grad():
            outputs = model_to_use.generate(
                **input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                pad_token_id=tokenizer_to_use.eos_token_id
            )

        full_decoded_text = tokenizer_to_use.decode(outputs[0], skip_special_tokens=True)
        
        # --- Robust JSON Parsing ---
        parsed_prediction = {'type': 'unknown', 'dataset_id': 'unknown'}
        
        # Search for a JSON object in the generated text
        match = json_pattern.search(full_decoded_text)
        
        if match:
            # The regex returns a tuple of groups; the JSON is in the first or second group
            json_str = match.group(1) or match.group(2)
            try:
                # Replace single quotes with double quotes for valid JSON parsing
                json_str = json_str.replace("'", '"')
                prediction = json.loads(json_str)
                # Check if the parsed object has the expected keys
                if 'type' in prediction and 'dataset_id' in prediction:
                    parsed_prediction = prediction
            except (json.JSONDecodeError, KeyError) as e:
                # Log a warning if the found JSON is malformed
                print(f"Warning: Found a potential JSON, but parsing failed. Error: {e}")
                parsed_prediction = {'type': 'Missing', 'dataset_id': 'Missing'}
                
        else:
            # Fallback for when no JSON pattern is found
            print(f"Warning: No valid JSON pattern found for text chunk at index {i}.")
            parsed_prediction = {'type': 'Missing', 'dataset_id': 'Missing'}

        y_pred.append(parsed_prediction)
        
    return y_pred


result = predict(X_test_prompts, merged_model, tokenizer)


result


predictions_df = pd.DataFrame(result)

# Display the new DataFrame
print(predictions_df)


predictions_df.head(10)


submission_df=pd.DataFrame()


submission_df['row_id']=inference_df.index


submission_df['article_id']=inference_df["file_name"]


submission_df['article_id'] = submission_df['article_id'].str.replace(r'\.[^.]*$', '', regex=True)


submission_df['dataset_id']=predictions_df['dataset_id']


submission_df['type']=predictions_df['type']


submission_df


submission_df.tail(20)


submission_df = submission_df.dropna(subset=['dataset_id', 'type'])


# Remove rows where the dataset_id or type is "Missing"
submission_df = submission_df[~submission_df['dataset_id'].isin(['Missing', None])]
submission_df = submission_df[~submission_df['type'].isin(['Missing', None])]


# NEW: Filter to keep only 'Primary' and 'Secondary' types
submission_df = submission_df[submission_df['type'].isin(['Primary', 'Secondary'])]


# # NEW: Remove ":" from the 'dataset_id' column
# submission_df['dataset_id'] = submission_df['dataset_id'].str.replace(':', '')


# The new, robust pattern for capturing just the DOI string
doi_pattern = r'10\.\d{4,9}/[^\s]+'

# A function to clean and normalize a single DOI string
def normalize_doi(doi_string):
    if not isinstance(doi_string, str):
        return doi_string

    # Search for the DOI pattern within the string
    match = re.search(doi_pattern, doi_string)
    
    # If a DOI is found, format it as a full URL
    if match:
        return f"https://doi.org/{match.group(0)}"
    # Otherwise, return the original string
    return doi_string
submission_df['dataset_id'] = submission_df['dataset_id'].apply(normalize_doi)


# # NEW: Convert non-URL DOIs to full URL format
# doi_pattern = r"10\.\d{4,9}/[^\s]+"
# submission_df['dataset_id'] = submission_df['dataset_id'].apply(lambda x: f"https://doi.org/{x}" if re.match(doi_pattern, str(x)) and not str(x).startswith("https://doi.org/") else x)


# NEW: Filter out rows where dataset_id does not match the desired format
# Define the regex pattern
REGEX_IDS = (
        r"(?i)\b(?:"
        r"CHEMBL\d+|"
        r"E-GEOD-\d+|E-PROT-\d+|E-MTAB-\d+|E-MEXP-\d+|EMPIAR-\d+|"
        r"ENSBTAG\d+|ENSOARG\d+|"
        r"EPI_ISL_\d{5,}|EPI\d{6,7}|"
        r"HPA\d+|CP\d{6}|IPR\d{6}|PF\d{5}|BX\d{6}|KX\d{6}|K0\d{4}|CAB\d{6}|"
        r"NC_\d{6}\.\d{1}|NM_\d{9}|"
        r"PRJNA\d+|PRJEB\d+|PRJDB\d+|PXD\d+|SAMN\d+|"
        r"GSE\d+|GSM\d+|GPL\d+|"
        r"PDB\s?[1-9][A-Z0-9]{3}|HMDB\d+|"
        r"dryad\.[^\s\"<>]+|pasta\/[^\s\"<>]+|"
        r"(?:SR[RPAX]|STH|ERR|DRR|DRP|ERP|ERX)\d+|"
        r"CVCL_[A-Z0-9]{4}"
        r")"
    )

submission_df = submission_df[submission_df['dataset_id'].str.contains(doi_pattern, regex=True) | submission_df['dataset_id'].str.contains(REGEX_IDS, regex=True)]


submission_df = submission_df.reset_index(drop=True)
submission_df['row_id'] = submission_df.index


# Define the category order for 'type'
category_order = pd.CategoricalDtype(['Primary', 'Secondary', 'Missing'], ordered=True)


# Convert the 'type' column to a categorical data type
submission_df['type'] = submission_df['type'].astype(category_order)
# Sort the DataFrame based on the desired priority.
# The `type` column is now sorted by the order defined above.
submission_df = submission_df.sort_values(by=['article_id', 'dataset_id', 'type'])
# Drop duplicates based on 'article_id' and 'dataset_id',
# keeping the first occurrence (which is now the highest priority type).
submission_df = submission_df.drop_duplicates(subset=['article_id', 'dataset_id'], keep='first')


submission_df.tail(20)


submission_df.head(20)



# Save the final DataFrame to a CSV file
submission_df['row_id'] = submission_df.index
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission DataFrame with Inference Results:")
print(submission_df)
print("\nDataFrame saved to submission.csv")


# submission_df.write_csv('/kaggle/working/submission.csv')

