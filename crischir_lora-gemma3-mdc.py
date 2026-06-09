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


# Setting Up the Environment


!pip install -q -U transformers
!pip install -q -U accelerate
!pip install -q -U datasets
!pip install -q -U peft
!pip install -q -i https://pypi.org/simple/ bitsandbytes
!pip install -q -U trl


import kagglehub
import re


max_seq_length=1200


test=False





import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import warnings
warnings.filterwarnings("ignore")


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

    # Check if MPS (Metal Performance Shaders) is available for macOS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("using MPS device on macOS")
        return torch.device("mps")

    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device}")
    return device


# Determine optimal computation dtype based on GPU capability
# Use bfloat16 if Compute Capability >= 8.0, otherwise float16
compute_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
print(f"Using compute dtype {compute_dtype}")

# Select the best available device (CPU, CUDA, or MPS)
device = define_device()
print(f"Operating on {device}")

#====
from transformers import AutoTokenizer
from transformers.models.gemma3 import Gemma3ForCausalLM

# Download the model files to a local directory and get the path.
GEMMA_PATH = kagglehub.model_download("google/gemma-3/transformers/gemma-3-1b-it")

# Load the tokenizer from the local path.
# The `local_files_only=True` flag is crucial.
tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH, local_files_only=True)

# Load the model from the local path.
# `Gemma3ForCausalLM` doesn't need the flag if you provide the local path.
# This should work as is with the local directory.
# model = Gemma3ForCausalLM.from_pretrained(GEMMA_PATH)



#======

# Path to the pre-trained model (adjust if necessary)
# GEMMA_PATH = "/kaggle/input/gemma-3/transformers/gemma-3-1b-it/1"

# Load the model with optimized settings
model = Gemma3ForCausalLM.from_pretrained(
    GEMMA_PATH,
    dtype=compute_dtype,  # Use the new argument name
    attn_implementation="eager", # Specify attention implementation
    low_cpu_mem_usage=True,      # Reduces CPU RAM usage during loading
    device_map=device            # Automatically map model layers to the device
)

# Define maximum sequence length for the tokenizer
max_seq_length = 8192 # Gemma 3 supports long contexts

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    GEMMA_PATH,
    max_seq_length=max_seq_length,
    device_map=device # Map tokenizer operations if relevant (less common)
)

# Store the EOS token for later use in prompts
EOS_TOKEN = tokenizer.eos_token


# Check if all model parameters are on the CUDA device
is_on_gpu = all(param.device.type == 'cuda' for param in model.parameters())
print("Model is on GPU:", is_on_gpu)


# Data Preparation


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
if test:
    df_eval=df_eval[:20].copy()
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


y_true_test


y_true_eval


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


# -- Apply Prompts and Convert to Dataset --

# Apply prompt generation to create the final text column for training and evaluation
X_train_prompts = pd.DataFrame(X_train_full.apply(generate_train_prompt, axis=1), columns=["text"])
# X_test_prompts = pd.DataFrame(X_test_full.apply(generate_test_prompt, axis=1), columns=["text"])
X_test_prompts = pd.DataFrame(X_test_full.apply(generate_train_prompt, axis=1), columns=["text"])
X_eval_prompts = pd.DataFrame(df_eval.apply(generate_train_prompt, axis=1), columns=["text"])

# Convert pandas DataFrames to Hugging Face Dataset objects
train_data = Dataset.from_pandas(X_train_prompts)
test_data = Dataset.from_pandas(X_test_prompts)
eval_data = Dataset.from_pandas(X_eval_prompts)


X_train_prompts.head()


train_data


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json



# def evaluate(y_true, y_pred):
#     """
#     Evaluate the fine-tuned model's performance on the new dataset.
#     This function handles multi-label predictions for 'type' and 'dataset_id'.
#     """

#     # --- 1. Extracting Values ---
#     # The predictions are already parsed into a list of dictionaries.
#     # We now directly extract true and predicted values for 'type' and 'dataset_id'
#     y_true_type = [item['type'] for item in y_true]
#     y_pred_type = [item.get('type', 'unknown') for item in y_pred]
    
#     y_true_dataset_id = [item['dataset_id'] for item in y_true]
#     y_pred_dataset_id = [item.get('dataset_id', 'unknown') for item in y_pred]
    
#     # --- 2. Evaluation of 'type' Predictions ---
#     print("\n--- Evaluation of 'type' Predictions ---")
    
#     # Define a label mapping for a clean report
#     type_labels = ["Missing", "Secondary", "Primary"]
    
#     # Calculate and print overall accuracy for 'type'
#     type_accuracy = accuracy_score(y_true_type, y_pred_type)
#     print(f'Overall Accuracy for type: {type_accuracy:.3f}')
    
#     # Compute and display accuracy for each type label
#     unique_true_types = np.unique(y_true_type)
#     for label in unique_true_types:
#         label_mask = np.array(y_true_type) == label
#         label_accuracy = accuracy_score(np.array(y_true_type)[label_mask], np.array(y_pred_type)[label_mask])
#         print(f'Accuracy for type "{label}": {label_accuracy:.3f}')
        
#     # Generate and print classification report for 'type'
#     try:
#         class_report_type = classification_report(y_true_type, y_pred_type, labels=type_labels, zero_division=0)
#         print('\nClassification Report (Type):\n', class_report_type)
#     except ValueError:
#         print("Could not generate classification report for 'type' due to mismatched labels.")
        
#     # Compute and display confusion matrix for 'type'
#     try:
#         conf_matrix_type = confusion_matrix(y_true_type, y_pred_type, labels=type_labels)
#         print('\nConfusion Matrix (Type - Rows: True, Cols: Pred) [Missing, Secondary, Primary]:\n', conf_matrix_type)
#     except ValueError:
#         print("Could not generate confusion matrix for 'type' due to mismatched labels.")

#     # --- 3. Evaluation of 'dataset_id' Predictions ---
#     print("\n--- Evaluation of 'dataset_id' Predictions ---")
    
#     # We'll calculate a simple exact match accuracy for dataset_id
#     dataset_id_match_count = sum(1 for true_id, pred_id in zip(y_true_dataset_id, y_pred_dataset_id) if true_id == pred_id)
#     dataset_id_accuracy = dataset_id_match_count / len(y_true_dataset_id)
    
#     print(f'Overall Exact Match Accuracy for dataset_id: {dataset_id_accuracy:.3f}')
#     print(f'Total correct dataset_id predictions: {dataset_id_match_count} out of {len(y_true_dataset_id)}')


def evaluate(y_true, y_pred):
    """
    Evaluate the fine-tuned model's performance on a dataset.
    This function dynamically handles labels for 'type' and 'dataset_id'.
    """

    # --- 1. Extracting Values ---
    # Ensure y_true and y_pred are lists of dictionaries
    if not isinstance(y_true, list) or not isinstance(y_pred, list):
        print("Input must be a list of dictionaries.")
        return

    # Extract true and predicted values for 'type' and 'dataset_id'
    y_true_type = [item['type'] for item in y_true]
    y_pred_type = [item.get('type', 'Unknown') for item in y_pred]

    y_true_dataset_id = [item['dataset_id'] for item in y_true]
    y_pred_dataset_id = [item.get('dataset_id', 'Unknown') for item in y_pred]
    
    # --- 2. Evaluation of 'type' Predictions ---
    print("\n--- Evaluation of 'type' Predictions ---")

    # Dynamically determine all unique labels from both true and predicted values
    # This prevents errors from unknown predicted labels
    all_type_labels = sorted(list(set(y_true_type + y_pred_type)))
    
    try:
        # Calculate and print overall accuracy for 'type'
        type_accuracy = accuracy_score(y_true_type, y_pred_type)
        print(f'Overall Accuracy for type: {type_accuracy:.3f}')
        
        # Compute and display classification report for 'type'
        class_report_type = classification_report(y_true_type, y_pred_type, labels=all_type_labels, zero_division=0)
        print('\nClassification Report (Type):\n', class_report_type)
        
        # Compute and display confusion matrix for 'type'
        conf_matrix_type = confusion_matrix(y_true_type, y_pred_type, labels=all_type_labels)
        print(f'\nConfusion Matrix (Type - Rows: True, Cols: Pred):\n\nLabels: {all_type_labels}\n{conf_matrix_type}')
        
    except Exception as e:
        print(f"An error occurred during 'type' evaluation: {e}")

    # --- 3. Evaluation of 'dataset_id' Predictions ---
    print("\n--- Evaluation of 'dataset_id' Predictions ---")
    
    try:
        # Calculate a simple exact match accuracy for dataset_id
        # This is a good metric for string-based identifiers
        dataset_id_match_count = sum(1 for true_id, pred_id in zip(y_true_dataset_id, y_pred_dataset_id) if true_id == pred_id)
        dataset_id_accuracy = dataset_id_match_count / len(y_true_dataset_id)
        
        print(f'Overall Exact Match Accuracy for dataset_id: {dataset_id_accuracy:.3f}')
        print(f'Total correct dataset_id predictions: {dataset_id_match_count} out of {len(y_true_dataset_id)}')
        
    except Exception as e:
        print(f"An error occurred during 'dataset_id' evaluation: {e}")


import torch
import json
from tqdm import tqdm




# def predict(X_test_df, model_to_use, tokenizer_to_use, device_to_use=device, max_new_tokens=100, temperature=0.0):
#     """Predict the type and dataset_id using the provided model and tokenizer with robust parsing."""

#     y_pred = []
#     model_to_use.eval()

#     # Define the pattern to find the JSON object, which might be in a markdown code block
#     json_pattern = re.compile(r'```json\s*(\{.*?\})\s*```|(\{.*?\})', re.DOTALL)

#     for i in tqdm(range(len(X_test_df)), desc="Predicting Outputs"):
#         prompt = X_test_df.iloc[i]["text"]

#         input_ids = tokenizer_to_use(prompt, return_tensors="pt").to(device_to_use)

#         with torch.no_grad():
#             outputs = model_to_use.generate(
#                 **input_ids,
#                 max_new_tokens=max_new_tokens,
#                 temperature=temperature,
#                 pad_token_id=tokenizer_to_use.eos_token_id
#             )

#         full_decoded_text = tokenizer_to_use.decode(outputs[0], skip_special_tokens=True)
        
#         # --- Robust JSON Parsing ---
#         parsed_prediction = {'type': 'unknown', 'dataset_id': 'unknown'}
        
#         # Search for a JSON object in the generated text
#         match = json_pattern.search(full_decoded_text)
        
#         if match:
#             # The regex returns a tuple of groups; the JSON is in the first or second group
#             json_str = match.group(1) or match.group(2)
            
#             # Pre-process the string to handle common errors like single quotes
#             # This is a simple fix for the most frequent issues.
#             json_str = json_str.replace("'", '"')

#             try:
#                 prediction = json.loads(json_str)
#                 # Check if the parsed object has the expected keys
#                 if 'type' in prediction and 'dataset_id' in prediction:
#                     parsed_prediction = prediction
#             except (json.JSONDecodeError, KeyError) as e:
#                 # Log a warning if the found JSON is malformed
#                 print(f"Warning: Found a potential JSON, but parsing failed. Error: {e}")
                
#         else:
#             # Fallback for when no JSON pattern is found
#             print(f"Warning: No valid JSON pattern found in generated text.")

#         y_pred.append(parsed_prediction)
        
#     return y_pred


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
        prompt = X_test_df.iloc[i]["text"]

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


X_test_prompts


train_data


# Generate predictions using the base model
y_pred_base = predict(X_test_prompts , model, tokenizer)
# y_pred_base = predict(X_test_prompts , model, tokenizer)
# y_pred_base = predict(X_test_prompts, model, tokenizer)


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


X_test_prompts


# predict_single_value(X_test_prompts.text[427], model, tokenizer)


# X_test_prompts.text[427]


# df_eval.columns





# input_text=df_eval.text_chunk[1]


# predict_single_value(input_text, model, tokenizer):



# The original script would have used this, which now causes the error:
# y_pred_base = predict(X_test, model, tokenizer)

# The corrected code should reference the DataFrame that contains the test prompts
# y_pred_base = predict(X_test_prompts, model, tokenizer)


# Evaluate the baseline predictions
print("--- Baseline Model Evaluation ---")
# evaluate(y_true, y_pred_base)

# Assuming y_true_test and y_pred_base are already defined

# Ensure y_true is a list of dictionaries, not a dict_values object
# y_true_list = list(y_true_test.values)
y_true_list = y_true_test.to_dict('records')

print("--- Baseline Model Evaluation ---")
evaluate(y_true_list, y_pred_base)


peft_config = LoraConfig(
    lora_alpha=32,        # Scaling factor for LoRA weights
    lora_dropout=0.05,    # Dropout probability for LoRA layers
    r=64,                 # Rank of the LoRA decomposition (higher r = more parameters)
    bias="none",          # Whether to train bias parameters ('none', 'all', or 'lora_only')
    task_type="CAUSAL_LM", # Task type is Causal Language Modeling
    target_modules="all-linear", # Apply LoRA to all linear layers
)



# # Define TrainingArguments separately to handle evaluation and logging
# Define TrainingArguments to handle all training parameters
training_args = TrainingArguments(
    output_dir="logs",
    num_train_epochs=6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # Increased to further reduce memory
    optim="adamw_torch_fused",
    save_strategy="no",  # Prevents saving any model checkpoints
    logging_steps=300,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7 else False,
    bf16=True if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    eval_strategy="no",  # Disables evaluation during training to prevent OOM
    eval_steps=112,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)



# training_arguments = SFTConfig(
#     output_dir="logs",                     # Directory to save logs and checkpoints
#     num_train_epochs=4,                    # Number of training epochs
#     per_device_train_batch_size=1,         # Batch size per GPU (keep small for large models/limited VRAM)
#     gradient_accumulation_steps=8,         # Accumulate gradients over 8 steps (effective batch size = 1*8=8)
#     optim="adamw_torch_fused",             # Use fused AdamW optimizer (efficient)
#     save_steps=112,                        # Save a checkpoint every 112 steps
#     logging_steps=25,                      # Log training metrics every 25 steps
#     learning_rate=2e-4,                    # Learning rate
#     weight_decay=0.001,                    # Weight decay for regularization
#     fp16=True if compute_dtype == torch.float16 else False,  # Enable mixed-precision (FP16) if available
#     bf16=True if compute_dtype == torch.bfloat16 else False, # Enable mixed-precision (BF16) if available
#     max_grad_norm=0.3,                     # Gradient clipping threshold
#     max_steps=-1,                          # Max training steps (-1 means use num_train_epochs)
#     warmup_ratio=0.03,                     # Proportion of training steps for learning rate warmup
#     group_by_length=False,                 # Don't group sequences by length (can sometimes speed up)
#     lr_scheduler_type="constant",          # Learning rate scheduler type
#     report_to="tensorboard",               # Report metrics to TensorBoard
#     evaluation_strategy="steps",           # Evaluate during training at specified step intervals
#     eval_steps=112,                        # Evaluate every 112 steps
#     load_best_model_at_end=True,           # Load the best model checkpoint at the end of training
#     gradient_checkpointing=True,           # Enable gradient checkpointing to save memory
#     gradient_checkpointing_kwargs={"use_reentrant": False}, # Recommended setting for new PyTorch versions

#     # SFTTrainer specific arguments
#     dataset_text_field="text",             # Name of the text field in the dataset
#     max_seq_length=max_seq_length,         # Maximum sequence length
#     packing=False,                         # Don't pack multiple sequences into one input
#     dataset_kwargs={                       # Arguments for dataset processing
#         "add_special_tokens": False,       # Don't add special tokens automatically (handled in prompt)
#         "append_concat_token": False,      # Don't append concat token (EOS is in our prompt)
#     }
# )


# Disable caching for training, re-enable for inference later
model.config.use_cache = False

# Set pretraining_tp if relevant for distributed training (usually 1 for single GPU)
model.config.pretraining_tp = 1
# Initialize the SFTTrainer with the corrected arguments
trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    processing_class=tokenizer, # Use the new argument
    args=training_args,
    peft_config=peft_config, # You need to pass the LoRA config to the trainer
)

# Begin training
# trainer.train()



# Train the model
print("Starting fine-tuning...")
train_result = trainer.train()
print("Fine-tuning finished.")

# Optionally, print training metrics
metrics = train_result.metrics
print("Training Metrics:", metrics)


# Define directory to save LoRA adapter and tokenizer
lora_directory = "LoRA-Gemma3-1B-MDC"

# Save the LoRA adapter weights
trainer.model.save_pretrained(lora_directory)
print(f"LoRA adapter saved to {lora_directory}")

# Save the tokenizer associated with the training
trainer.tokenizer.save_pretrained(lora_directory)
print(f"Tokenizer saved to {lora_directory}")


from peft import PeftModel
from peft import get_peft_model
lora_directory2 = "LoRA-Gemma3-1B-MDC-peft"
# Your base model and LoRA config
# model = ...
# lora_config = ...

# Wrap the model with the PEFT config
peft_model = get_peft_model(model, peft_config)

# Save the adapter
peft_model.save_pretrained(lora_directory2)


# Path to your saved LoRA adapter
lora_directory = "LoRA-Gemma3-1B-MDC-peft"

# Load the base model
base_model_path = GEMMA_PATH
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load the LoRA adapter onto the base model
peft_model = PeftModel.from_pretrained(base_model, lora_directory)

# Merge the LoRA weights into the base model
merged_model = peft_model.merge_and_unload()

# Save the full, merged model to a new directory
merged_model.save_pretrained("LoRA-Gemma3-1B-MDC-full-model")

# Also save the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained("LoRA-Gemma3-1B-MDC-full-model")


# lora_directory2 = "LoRA-Gemma3-1B-MDC-full-model"
# # Merge the LoRA adapter weights into the base model
# merged_model = trainer.model.merge_and_unload()

# # Save the full, merged model
# merged_model.save_pretrained(lora_directory)
# print(f"Merged model saved to {lora_directory}")


# Ensure the model is in evaluation mode
# trainer.model.eval() # Already handled by predict function, but good practice

# Generate predictions using the fine-tuned model from the trainer
print("Predicting with fine-tuned model...")
y_pred_tuned = predict(X_eval_prompts, trainer.model, tokenizer) # Use trainer.model

# Evaluate the fine-tuned predictions
# y_true_eval = df_eval[['type', 'dataset_id']].copy()
print("\n--- Fine-Tuned Model Evaluation ---")
# evaluate(y_true_eval, y_pred_tuned)
y_true_eval = df_eval[['type', 'dataset_id']].to_dict('records') # <-- Corrected line
print("\n--- Fine-Tuned Model Evaluation ---")
evaluate(y_true_eval, y_pred_tuned)


# Create DataFrame with test texts, true labels, and predicted labels
evaluation = pd.DataFrame({'text': X_eval_prompts["text"], # Use the prompts used for prediction
                           'y_true':y_true_eval,
                           'y_pred': y_pred_tuned},
                         )

# Save the evaluation DataFrame to a CSV file
output_predictions_file = "test_predictions_gemma3_1b_tuned.csv"
evaluation.to_csv(output_predictions_file, index=False)
print(f"Test predictions saved to {output_predictions_file}")


# --- Reload and Merge ---
# Ensure the trainer and original model are not needed anymore to free memory if necessary
# import gc
# del trainer, model
# gc.collect()
# torch.cuda.empty_cache()

print("Reloading base model...")
# Load the base model again (ensure enough RAM/VRAM)
base_model = AutoModelForCausalLM.from_pretrained(
    GEMMA_PATH,
    torch_dtype=compute_dtype, # Use the same dtype as training
    low_cpu_mem_usage=True,
    device_map='auto' # Let transformers handle device mapping
)

print("Loading PeftModel and merging...")
# Load the PeftModel by combining base model and LoRA adapter
# peft_model = PeftModel.from_pretrained(base_model, lora_directory)
peft_model = PeftModel.from_pretrained(base_model, lora_directory)


# Merge the LoRA weights into the base model
merged_model = peft_model.merge_and_unload()
print("Merging complete.")

# --- Save Merged Model and Tokenizer ---
merged_model_directory = "merged-Gemma3-1B-MDC"
print(f"Saving merged model to {merged_model_directory}...")
merged_model.save_pretrained(
    merged_model_directory,
    safe_serialization=True,
    max_shard_size="2GB"
)
print("Merged model saved.")

print(f"Saving tokenizer to {merged_model_directory}...")
# Load the tokenizer from the BASE MODEL path, not the LoRA adapter path
tokenizer_for_merged = AutoTokenizer.from_pretrained(GEMMA_PATH)
tokenizer_for_merged.save_pretrained(merged_model_directory)
print("Tokenizer saved.")




