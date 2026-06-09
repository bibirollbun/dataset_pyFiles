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


# !pip show PyMuPDF
# !pip show pymupdf4llm
# !pip show bitsandbytes 
# !pip show trl
# !pip show transformers
# !pip show peft

# !pip show accelerate 
# !pip show numpy
# !pip show sentencepiece 
# !pip show spacy 


# 1.1. Install necessary libraries
# Use !pip install for notebook environment
# !pip install transformers trl accelerate bitsandbytes sentencepiece lxml PyMuPDF spacy peft
# !python -m spacy download en_core_web_sm # Download a small spaCy model

# 1.2. Import Libraries
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
from datasets import Dataset, concatenate_datasets
from peft import PeftModel

from tqdm.auto import tqdm

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
# TRAINED_MODEL_DIR = '/kaggle/input/m/jimgile/qwen-3/transformers/.06b-mdc-fine-tuned-classifier/1/results/final_model'
TRAINED_MODEL_DIR = '/kaggle/input/m/jimgile/qwen-3/transformers/.06b-mdc-fine-tuned-classifier/6/results/final_model'
SUBMISSION_FILE_PATH = os.path.join(BASE_OUTPUT_DIR, "submission.csv")

# Load spaCy model for sentence segmentation and potentially other NLP tasks
# python -m spacy download en_core_web_sm 
NLP_SPACY = spacy.load("en_core_web_sm")



# Set the base model name
model_name = QWEN_BASE_MODEL_PATH

# Define 4-bit Quantization Config
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 # Or torch.float16 if bfloat16 is not supported by your GPU
)

# Load Tokenizer from trained model
tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token # Qwen uses EOS for padding
print("Loaded trained tokenizer for inference.")

# Load the Trained Model
# Load the base model and then the LoRA adapters for inference:
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, TRAINED_MODEL_DIR)
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


# For testing, always set to the TEST_DATA_DIR
base_file_dir = TEST_DATA_DIR

# Load file paths for base directory
test_file_paths_df = load_file_paths(base_file_dir)
test_file_paths_df['pdf_file_path'] = test_file_paths_df['pdf_file_path'].fillna('')
test_file_paths_df['xml_file_path'] = test_file_paths_df['xml_file_path'].fillna('')

print(f"Files paths shape: {test_file_paths_df.shape}")
display(test_file_paths_df.sample(3))


# sample_test_file_paths_df = test_file_paths_df.sample(2, random_state=42)
# sample_test_file_paths_df = test_file_paths_df.loc[test_file_paths_df['article_id']=='10.1002_mp.14424']
# sample_test_file_paths_df = test_file_paths_df.loc[test_file_paths_df['article_id']=='10.1002_cssc.202201821']
# sample_test_file_paths_df = test_file_paths_df.loc[test_file_paths_df['article_id']=='10.1002_ecs2.1280']
# # sample_test_file_paths_df = test_file_paths_df.loc[test_file_paths_df['article_id']=='10.1002_esp.5090']
# sample_test_file_paths_df


submission_data_list = process_test_articles(tokenizer, test_file_paths_df)
# display(submission_data_list)


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
    submission_df = pd.DataFrame(submission_list)
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

submission_df = prepare_for_submission(submission_data_list)
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

