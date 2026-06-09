# !pip install PyMuPDF==1.26.3
# !pip install pymupdf4llm==0.0.26
# !pip install bitsandbytes==0.46.1
# !pip install trl==0.19.1
# !pip install transformers==4.52.4
# !pip install peft==0.15.2
# !pip install accelerate==1.8.1
# !pip install torch==2.6.0
# !pip install torchvision==0.21.0
# !pip install torchaudio==2.6.0


# 1.1. Install necessary libraries
# Use !pip install for notebook environment
# !pip install transformers trl accelerate bitsandbytes sentencepiece lxml PyMuPDF spacy peft
# !python -m spacy download en_core_web_sm # Download a small spaCy model

# 1.2. Import Libraries
import os
import re
import json
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Set, List, Optional, Dict, Any

import fitz # PyMuPDF
from lxml import etree # For XML parsing
import spacy
import kagglehub

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.trainer_callback import EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training

from tqdm.auto import tqdm
# import gc # For garbage collection

# For KaggleHub integration (assuming it's set up or models are downloaded)
# You might need to install kagglehub if you plan to use it directly for model download
# !pip install kagglehub

# 1.3. Configure CUDA for local GPU
if torch.cuda.is_available():
    print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
    torch.cuda.empty_cache() # Clear GPU memory
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")




# Import classes from local utility file
import mdc_data_processing_utils

# If mdc_data_processing_utils.py has been changed and saved.
# To load the changes without restarting the kernel:
import importlib
importlib.reload(mdc_data_processing_utils)

# Now, any calls to functions from mdc_data_processing_utils
# will use the newly reloaded code.
from mdc_data_processing_utils import (
    ArticleData,
    DatasetCitation,
    LlmTrainingData,
    SubmissionData,
    MdcFileTextExtractor,
)



# Define constants for file paths and model configurations
BASE_INPUT_DIR = '/kaggle/input/make-data-count-finding-data-references'
BASE_OUTPUT_DIR = '/kaggle/working'

# Define directories for articles in train and test sets
TRAIN_DATA_DIR = os.path.join(BASE_INPUT_DIR, 'train')
TEST_DATA_DIR = os.path.join(BASE_INPUT_DIR, 'test')
TRAIN_LABELS_PATH = os.path.join(BASE_INPUT_DIR, 'train_labels.csv')
SAVED_TRAINING_DATA_CSV_PATH = '/kaggle/input/mdc-training-data-for-llm-2/training_data_for_llm.csv'

# Define the path to the few-shot examples CSV
FEW_SHOT_CSV_PATH = os.path.join("/kaggle/input/mdc-few-shot-examples", "few_shot_examples.csv")

# Define the base model path
QWEN_BASE_MODEL_PATH = kagglehub.model_download("qwen-lm/qwen-3/transformers/0.6b")

# Output directory for the fine-tuned model and results
TRAINED_MODEL_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "results")
TRAINED_MODEL_FINAL_OUTPUT_DIR = os.path.join(TRAINED_MODEL_OUTPUT_DIR, "final_model")
TRAINED_MODEL_DIR = '/kaggle/input/m/jimgile/qwen-3/transformers/.06b-mdc-fine-tuned-classifier/1/results/final_model'
SUBMISSION_FILE_PATH = os.path.join(BASE_OUTPUT_DIR, "submission.csv")

# Load spaCy model for sentence segmentation and potentially other NLP tasks
# python -m spacy download en_core_web_sm 
NLP_SPACY = spacy.load("en_core_web_sm")



def load_file_paths(dataset_type_dir: str) -> pd.DataFrame: 
    pdf_path = os.path.join(dataset_type_dir, 'PDF')
    xml_path = os.path.join(dataset_type_dir, 'XML')
    dataset_type = os.path.basename(dataset_type_dir)
    pdf_files = [f for f in os.listdir(pdf_path) if f.endswith('.pdf')]
    xml_files = [f for f in os.listdir(xml_path) if f.endswith('.xml')]
    df_pdf = pd.DataFrame({
        'article_id': [f.replace('.pdf', '') for f in pdf_files],
        'pdf_file_path': [os.path.join(pdf_path, f) for f in pdf_files]
    })
    df_xml = pd.DataFrame({
        'article_id': [f.replace('.xml', '') for f in xml_files],
        'xml_file_path': [os.path.join(xml_path, f) for f in xml_files]
    })
    merge_df = pd.merge(df_pdf, df_xml, on='article_id', how='outer', suffixes=('_pdf', '_xml'), validate="one_to_many")
    merge_df['dataset_type'] = dataset_type
    return merge_df


# # Load the labeled training data CSV file
# print(f"Loading labeled training data from: {TRAIN_LABELS_PATH}")
# train_labels_df = pd.read_csv(TRAIN_LABELS_PATH)
# print(f"Training labels shape: {train_labels_df.shape}")

# # Group training data by article_id to get all datasets for each article
# # This creates a dictionary where keys are article_ids and values are lists of dataset dicts
# grouped_training_data = {}
# for article_id, group_df in train_labels_df.groupby('article_id'):
#     grouped_training_data[article_id] = group_df[['dataset_id', 'type']].to_dict('records')

# # Example usage of grouped_training_data
# print(f"Example grouped training data for article_id '10.1002_2017jc013030': {grouped_training_data['10.1002_2017jc013030']}")

# # Just for testing, always set to the TEST_DATA_DIR
# base_file_dir = TRAIN_DATA_DIR

# # Load file paths for base directory
# file_paths_df = load_file_paths(base_file_dir)
# file_paths_df['pdf_file_path'] = file_paths_df['pdf_file_path'].fillna('')
# file_paths_df['xml_file_path'] = file_paths_df['xml_file_path'].fillna('')

# # Merge the file paths with the grouped_training_data
# file_paths_df['ground_truth_dataset_info'] = file_paths_df['article_id'].map(grouped_training_data)
# file_paths_df['ground_truth_dataset_info'] = file_paths_df['ground_truth_dataset_info'].fillna('')

# # Reduce the file paths DataFrame to only those with ground truth dataset info and get a sample
# # This is to ensure we have a manageable dataset for training
# file_paths_df = file_paths_df[file_paths_df['ground_truth_dataset_info'].astype(bool)]
# file_paths_df = file_paths_df.reset_index(drop=True)
# file_paths_df = file_paths_df.sample(frac=.65, random_state=42).reset_index(drop=True)  # Shuffle the DataFrame
# print(f"Files paths shape: {file_paths_df.shape}")
# display(file_paths_df.sample(3))


# def extract_training_data_for_llm(file_paths_df: pd.DataFrame) -> list[dict[str, str]]:
#     """
#     Extracts article data for training set with ground truth.
    
#     Args:
#         file_paths_df (pd.DataFrame): DataFrame containing file paths and ground truth info.
        
#     Returns:
#         Dict[str, ArticleData]: Dictionary mapping article IDs to ArticleData objects.
#     """
#     training_data_for_llm: list[dict[str, str]] = [] # This will be a list of LlmTrainingData for the LLM training dataset
#     for i, row in tqdm(file_paths_df.iterrows(), total=len(file_paths_df)):
#         article_id = row['article_id']
#         filepath = row['pdf_file_path'] if row['pdf_file_path'] else row['xml_file_path']
#         ground_truth_list = row['ground_truth_dataset_info'] if 'ground_truth_dataset_info' in row else []

#         file_extractor = MdcFileTextExtractor(article_id, filepath)
#         article_data = file_extractor.extract_article_data_for_training(NLP_SPACY, ground_truth_list)
#         training_data_for_llm.extend(article_data.get_data_for_llm())

#     print(f"Loaded training data for {len(training_data_for_llm)} articles.")
#     return training_data_for_llm



# # For testing, let's extract training data for a specific article
# sample_file_paths_df = file_paths_df.loc[file_paths_df['article_id'] == '10.1002_esp.5058']
# sample_file_paths_df


# # This takes 15+ minutes

# # Remove bad pdf file:
# file_paths_df = file_paths_df[file_paths_df['article_id'] != '10.20944_preprints202009.0353.v1']
# print(f"File paths shape: {file_paths_df.shape}")

# # 4.3. Populate ArticleData with DatasetCitation objects and ground truth
# training_data_for_llm = extract_training_data_for_llm(file_paths_df)
# print(f"Prepared {len(training_data_for_llm)} training examples for the LLM.")

# # Convert the list of LlmTrainingData to a DataFrame and save it
# training_data_for_llm_df = pd.DataFrame(training_data_for_llm)
# training_data_for_llm_df.to_csv(os.path.join(BASE_OUTPUT_DIR, "training_data_for_llm.csv"), index=False)



def last_of_string(s: str, length: int = 400) -> str:
    return s[-length:]

# Load saved training data from the csv file
training_data_for_llm_df = pd.read_csv(SAVED_TRAINING_DATA_CSV_PATH)
training_data_for_llm_df['citation_context'] = training_data_for_llm_df['citation_context'].apply(last_of_string)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(training_data_for_llm_df)

# Split into train/validation
train_test_split = train_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']
print(f"Training set size: {len(train_dataset)} examples")
print(f"Validation set size: {len(eval_dataset)} examples")

# Clean up
del training_data_for_llm_df
del train_test_split



# Concatenate the few-shot examples with the main training dataset
# This will add the few-shot examples as additional rows to the training data
few_shot_dataset = Dataset.from_csv(FEW_SHOT_CSV_PATH)
print(f"Loaded {len(few_shot_dataset)} few-shot examples.")
train_dataset = concatenate_datasets([train_dataset, few_shot_dataset])
print(f"New training set size: {len(train_dataset)} examples")



# 5.1. Choose a Model from KaggleHub
# Example: Qwen/Qwen1.5-0.5B-Chat (or 1.8B-Chat if 0.5B is too small/performs poorly)
# You can find these on KaggleHub or Hugging Face Hub.
model_name = QWEN_BASE_MODEL_PATH

# 5.2. Load Training Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token # Qwen uses EOS for padding


# 5.3. Load Model with Quantization (4-bit)
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 # Or torch.float16 if bfloat16 is not supported by your GPU
)


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16, # Match compute_dtype
    device_map="auto", # Automatically maps model to available devices
    trust_remote_code=True # Required for some models like Qwen
)

# Prepare model for k-bit training (LoRA compatible)
model.config.use_cache = False
model.config.pretraining_tp = 1
model = prepare_model_for_kbit_training(model)

print(f"Model {model_name} loaded with 4-bit quantization for training.")


# 6.1. Define the formatting function for ChatML (Corrected for trl 0.19.1)
def format_example(example):
    messages = [
        {"role": "system", "content": "You are an expert assistant for classifying research data citations. /no_think"},
        {"role": "user", "content": (
            f"""
Given the following 'Article Abstract' and a specific data citation ('Dataset ID' and 'Data Citation Context' combination), classify the data citation as either: 
'Primary' (if the data citation refers to raw or processed **data created/generated as part of the paper**, specifically for this study), 
'Secondary' (if the data citation refers to raw or processed **data derived/reused from existing records** or previously published data), or 
'Missing' (if the data citation refers to another **article/paper/journal**, a **figure, software, or other non-data entity**, or the 'Data Citation Context' is **empty or irrelevant**).\n\n"""
            f"Now, classify the following:\n\n" # Add a clear separator            
            f"Article Abstract: {example['article_abstract']}\n" 
            f"Dataset ID: {example['dataset_id']}\n"
            f"Data Citation Context: {example['citation_context']}\n\n"
            f"Classification:"
        )}
    ]
    # The target output for the model is just "Primary", "Secondary, or "Missing"
    messages.append({"role": "assistant", "content": example['label']})
    
    # Apply chat template and return the string directly
    # <--- IMPORTANT CHANGE: Directly return the string, not a dictionary
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)

# Apply the formatting to the dataset
# IMPORTANT: When formatting_func returns a string directly, you typically don't
# need to call .map() on the dataset beforehand if SFTTrainer handles it internally.
# However, if you want to inspect the formatted text, you can still do this:
# formatted_train_dataset = train_dataset.map(format_example)
# But for SFTTrainer, you pass the original `train_dataset` and the `formatting_func`
# and `dataset_text_field` (which will be ignored if formatting_func is used to generate the text).

# Print an example to verify (you'll need to call format_example directly for this)
print("\nExample of formatted training data (string output):")
# You can't directly print from formatted_train_dataset if you don't map it first.
# Let's print by calling the function on a sample:
if len(train_dataset) > 0:
    sample_formatted_text = format_example(train_dataset[17])
    tokenized_input = tokenizer(sample_formatted_text, return_tensors="pt")
    prompt_length = tokenized_input.input_ids.shape[1]
    print(f"Length of the full prompt in tokens: {prompt_length}")    
    print(sample_formatted_text)
else:
    print("No training data to display example.")


# ---------------------------------------------------------
# Use the evaluation dataset in the SFTTrainer
# ---------------------------------------------------------

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers.trainer_utils import EvalPrediction # Import this for type hinting if desired

# Assuming your model outputs logits for 3 classes (Primary, Secondary, Missing)
# And your labels are integers (e.g., 0, 1, 2) corresponding to these classes.

def compute_classification_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    
    # For classification, you typically take the argmax of the logits to get predicted class IDs
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate desired metrics
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro') # Macro F1 treats all classes equally
    f1_weighted = f1_score(labels, predictions, average='weighted') # Weighted F1 accounts for class imbalance
    
    # You might also want precision and recall per class, or just overall
    # precision_macro = precision_score(labels, predictions, average='macro')
    # recall_macro = recall_score(labels, predictions, average='macro')

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        # You can also include 'eval_loss' if you want, but Trainer computes it by default
        # "eval_loss": eval_pred.metrics.get("eval_loss", None) # Access it if Trainer provides it
    }

# 7.1. Configure LoRA
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear", # Adjust based on model architecture if needed
)

# 7.2. Configure Training Arguments (now using SFTConfig)
training_args = SFTConfig(
    output_dir=TRAINED_MODEL_OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=5,
    logging_steps=10,
    save_steps=20,
    optim="paged_adamw_8bit",
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    report_to="none",
    disable_tqdm=False,
    remove_unused_columns=False,
    label_names=['labels'],
    
    # SFTTrainer-specific parameters moved into SFTConfig
    max_seq_length=512,
    packing=False,
    dataset_text_field="text",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False},

    # --- NEW: Evaluation Parameters ---
    eval_strategy="steps", # Evaluate every 'eval_steps'. You can also use "epoch" for evaluation_strategy
    eval_steps=20,               # How often to run evaluation (e.g., every 25 steps)
    save_strategy="steps",       # How often to save checkpoints
    save_total_limit=1,          # Only keep the best model checkpoint
    load_best_model_at_end=True, # Load the model with the best validation metric at the end of training
    greater_is_better=False,     # For loss, lower is better
)

# 7.3. Initialize SFTTrainer
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset, # <--- Pass the evaluation dataset here
    # compute_metrics=compute_classification_metrics,
    peft_config=peft_config,
    args=training_args,
    formatting_func=format_example,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] # Stop if eval_loss doesn't improve for 3 evaluations
)



# 7.4. Start Training
print("\nStarting model training...")
trainer.train()
print("Training complete!")

# Save the fine-tuned model (LoRA adapters)
trainer.save_model(TRAINED_MODEL_FINAL_OUTPUT_DIR)
print(f"Fine-tuned model saved to {TRAINED_MODEL_FINAL_OUTPUT_DIR}")


# Load Tokenizer from trained model
tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_FINAL_OUTPUT_DIR)
tokenizer.pad_token = tokenizer.eos_token # Qwen uses EOS for padding
print("Loaded trained tokenizer for inference.")

# 8.1. Load the Trained Model
# Load the base model and then the LoRA adapters for inference:
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, TRAINED_MODEL_FINAL_OUTPUT_DIR)
model.eval() # Set to evaluation mode

print("Loaded trained model for inference.")




def invoke_model_for_inference(tokenizer, article_data: ArticleData) -> list[SubmissionData]:
    submission_data_list = []
    article_id = article_data.article_id
    dataset_citations = article_data.dataset_citations
    if not dataset_citations:
        submission_data_list.append(SubmissionData(article_id, dataset_id="Missing", type="Missing"))
        return submission_data_list

    print(f"Found {len(dataset_citations)} citations for {article_id}")
    for dc in dataset_citations:
        # Create the prompt for inference
        messages = [
            {"role": "system", "content": "You are an expert assistant for classifying research data citations. /no_think"},
            {"role": "user", "content": (
                f"""
Given the following 'Article Abstract' and a specific data citation ('Dataset ID' and 'Data Citation Context' combination), classify the data citation as either: 
'Primary' (if the data citation refers to raw or processed **data created/generated as part of the paper**, specifically for this study), 
'Secondary' (if the data citation refers to raw or processed **data derived/reused from existing records** or previously published data), or 
'Missing' (if the data citation refers to another **article/paper/journal**, a **figure, software, or other non-data entity**, or the 'Data Citation Context' is **empty or irrelevant**).\n\n"""
                f"Now, classify the following:\n\n" # Add a clear separator            
                f"Article Abstract: {article_data.abstract}\n"
                f"Dataset ID: {dc.dataset_id}\n"                
                f"Data Citation Context: {dc.citation_context}\n\n"
                f"Classification:"
            )}
        ]

        # --- CHANGE STARTS HERE ---
        # Tokenize and get both input_ids and attention_mask
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs, # <--- Pass the entire dictionary (includes input_ids and attention_mask)
                max_new_tokens=10, # Expecting "Primary" or "Secondary"
                do_sample=True,    # <--- Enable sampling
                temperature=0.7,   # <--- Adjust temperature (0.7-0.9 is common)
                top_p=0.9,         # <--- Top-p sampling (consider tokens that sum to 90% probability)
                top_k=50,          # <--- Top-k sampling (consider only the top 50 most probable tokens)                
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        # --- CHANGE ENDS HERE ---        

        generated_text = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip() # Use inputs['input_ids']
        # print(f"LLM Resp: {generated_text}")        
        
        # Post-process the generated text to get the classification
        predicted_type = "Missing"
        if "Primary" in generated_text:
            predicted_type = "Primary"
        elif "Secondary" in generated_text:
            predicted_type = "Secondary"
        
        submission_data_list.append(SubmissionData(article_id, dataset_id=dc.dataset_id, type=predicted_type, context=dc.citation_context))

    return submission_data_list

def process_test_articles(tokenizer, file_paths_df: pd.DataFrame) -> list[SubmissionData]:
    """
    Extracts article data for testing set without ground truth.
    
    Args:
        file_paths_df (pd.DataFrame): DataFrame containing file paths and ground truth info.
        
    Returns:
        Dict[str, ArticleData]: Dictionary mapping article IDs to ArticleData objects.
    """
    submission_data_list = []
    for i, row in tqdm(file_paths_df.iterrows(), total=len(file_paths_df)):
        article_id = row['article_id']
        filepath = row['pdf_file_path'] if row['pdf_file_path'] else row['xml_file_path']
        file_extractor = MdcFileTextExtractor(article_id, filepath)
        
        # Extract article data
        article_data = file_extractor.extract_article_data_for_inference(NLP_SPACY)

        # Invoke the model with the collected article_data
        submission_data_list.extend(invoke_model_for_inference(tokenizer, article_data))

    print(f"Processed testing data for {len(submission_data_list)} article and dataset_id combos.")
    return submission_data_list


# For testing, always set to the TEST_DATA_DIR
base_file_dir = TEST_DATA_DIR

# Load file paths for base directory
test_file_paths_df = load_file_paths(base_file_dir)
test_file_paths_df['xml_file_path'] = test_file_paths_df['xml_file_path'].fillna('')

print(f"Files paths shape: {test_file_paths_df.shape}")
display(test_file_paths_df.sample(3))


# sample_test_file_paths_df = test_file_paths_df.sample(2, random_state=42)
# sample_test_file_paths_df = test_file_paths_df.loc[test_file_paths_df['article_id']=='10.1002_mp.14424']
# sample_test_file_paths_df = test_file_paths_df.loc[test_file_paths_df['article_id']=='10.1002_cssc.202201821']
# sample_test_file_paths_df = test_file_paths_df.loc[test_file_paths_df['article_id']=='10.1002_ecs2.1280']
# # sample_test_file_paths_df = test_file_paths_df.loc[test_file_paths_df['article_id']=='10.1002_esp.5090']
# sample_test_file_paths_df


sample_sub = process_test_articles(tokenizer, test_file_paths_df)
# display(sample_sub)


def format_dataset_id(dataset_id: str) -> str:
    """
    Formats the dataset_id by removing any leading/trailing whitespace and ensuring it is a string.
    
    Args:
        dataset_id (str): The dataset identifier to format.
        
    Returns:
        str: The formatted dataset identifier.
    """
    if dataset_id and dataset_id.startswith("10.") and len(dataset_id) > 10:
        # If the dataset_id starts with "10." and is longer than 10 characters, it's likely a DOI
        dataset_id = "https://doi.org/" + dataset_id.lower().strip()
    return dataset_id

def prepare_for_submission(submission_list: list[SubmissionData]) -> pd.DataFrame:
    """
    Prepares the submission_list for submission by ensuring the correct columns and formatting.
    
    Args:
        expanded_df (pd.DataFrame): The DataFrame containing expanded dataset information.
        
    Returns:
        pd.DataFrame: A DataFrame ready for submission with 'article_id', 'dataset_id', and 'type' columns.
    """
    submission_df = pd.DataFrame(sample_sub)
    # Ensure the DataFrame has the correct columns
    submission_df = submission_df[['article_id', 'dataset_id', 'type']].copy()

    # Format dataset_id
    submission_df['dataset_id'] = submission_df['dataset_id'].apply(format_dataset_id)  

    # Remove rows where type is 'Missing' and reset index
    submission_df = submission_df[submission_df['type'] != 'Missing'].reset_index(drop=True)
    submission_df['row_id'] = range(len(submission_df))

    # Reorder columns to match the submission format
    submission_df = submission_df[['row_id', 'article_id', 'dataset_id', 'type']]
    
    return submission_df



# 9.1. Create Submission DataFrame

submission_df = prepare_for_submission(sample_sub)
submission_df.to_csv(SUBMISSION_FILE_PATH, index=False)
print(f"Submission file {SUBMISSION_FILE_PATH} created successfully!")
display(submission_df.head())


def f1_score(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0.0
    
    
# if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
pred_df = submission_df.copy()
label_df = pd.read_csv("/kaggle/input/make-data-count-finding-data-references/sample_submission.csv")
label_df = label_df[label_df['type'] != 'Missing'].reset_index(drop=True)

hits_df = label_df.merge(pred_df, on=["article_id", "dataset_id", "type"])

tp = hits_df.shape[0]
fp = pred_df.shape[0] - tp
fn = label_df.shape[0] - tp


print("TP:", tp)
print("FP:", fp)
print("FN:", fn)
print("F1 Score:", round(f1_score(tp, fp, fn), 3))

