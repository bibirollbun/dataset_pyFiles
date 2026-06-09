!pip install --quiet transformers accelerate datasets bitsandbytes evaluate peft sentencepiece


import os
import torch
import random
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
   AutoTokenizer,
   AutoModelForCausalLM,
   TrainingArguments,
   Trainer,
   DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# Set random seed
def set_seed(seed: int):
   random.seed(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
set_seed(42)


# Read samples from Inkuba Mono Swahili dataset
def read_dataset_samples(file_path, num_samples=10):
    """
    Read a specified number of samples from the dataset
    
    Parameters:
    - file_path: Path to the dataset text file
    - num_samples: Number of samples to read (default 10)
    
    Returns:
    - List of text samples
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read all lines
            lines = file.readlines()
            
            # Print total number of lines
            print(f"Total lines in dataset: {len(lines)}")
            
            # Select and print samples
            print("\nDataset Samples:")
            print("-" * 50)
            
            for i in range(min(num_samples, len(lines))):
                print(f"\nSample {i+1}:")
                print("-" * 30)
                print(lines[i].strip())
                print("-" * 30)
    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Path to the dataset
inkuba_mono_dataset_path = '/kaggle/input/inkuba-mono-swahili/data.txt'

# Read and display samples
read_dataset_samples(inkuba_mono_dataset_path)


import json

def read_instruction_dataset_samples(file_path, num_samples=4):
    """
    Read and display samples from the Swahili instructions JSON dataset
    
    Parameters:
    - file_path: Path to the JSON dataset file
    - num_samples: Number of samples to read (default 10)
    
    Returns:
    - Displays dataset samples
    """
    try:
        # Read the JSON file
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Print total number of samples
        print(f"Total samples in dataset: {len(data)}")
        
        # Select and print samples
        print("\nDataset Samples:")
        print("-" * 50)
        
        for i in range(min(num_samples, len(data))):
            print(f"\nSample {i+1}:")
            print("-" * 30)
            print("Instruction:")
            print(data[i].get('instruction', 'No instruction'))
            
            if data[i].get('input'):
                print("\nInput:")
                print(data[i]['input'])
            
            print("\nOutput:")
            print(data[i].get('output', 'No output'))
            print("-" * 30)
    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Path to the dataset
dataset_path = '/kaggle/input/swahili-instructions/swahili-instructions-response.json'

# Read and display samples
read_instruction_dataset_samples(dataset_path)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Swahili sentiment dataset
def load_swahili_sentiment_dataset(file_path):
    """
    Load and analyze the Swahili sentiment dataset
    
    Parameters:
    - file_path: Path to the CSV file
    
    Returns:
    - Pandas DataFrame with dataset
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Basic dataset information
        print("Dataset Overview:")
        print("-" * 50)
        print(f"Total number of samples: {len(df)}")
        
        # Display column names
        print("\nColumns:")
        print(df.columns.tolist())
        
        # Label distribution
        print("\nLabel Distribution:")
        label_counts = df['labels'].value_counts()
        print(label_counts)
        print("\nLabel Percentages:")
        print(label_counts / len(df) * 100)
        
        # Text length analysis
        df['text_length'] = df['text'].str.len()
        print("\nText Length Statistics:")
        print(df['text_length'].describe())
        
        # Visualization of label distribution
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x='labels')
        plt.title('Distribution of Sentiment Labels')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()
        
        # Boxplot of text lengths by sentiment
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x='labels', y='text_length')
        plt.title('Text Length Distribution by Sentiment')
        plt.xlabel('Sentiment')
        plt.ylabel('Text Length')
        plt.tight_layout()
        plt.show()
        
        # Display a few sample texts
        print("\nSample Texts:")
        print("-" * 50)
        for i in range(min(5, len(df))):
            print(f"\nSample {i+1}:")
            print(f"Text: {df.iloc[i]['text']}")
            print(f"Label: {df.iloc[i]['labels']}")
        
        return df
    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Path to the dataset
dataset_path = '/kaggle/input/swahili-sentiment-dataset/swahili-sentiment.csv'

# Load and analyze the dataset
sentiment_df = load_swahili_sentiment_dataset(dataset_path)


from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_swahili_mmlu_dataset():
    """
    Load and analyze the Swahili MMLU dataset from Hugging Face
    
    Returns:
    - Loaded dataset
    """
    try:
        # Load the dataset
        sw_mmlu = load_dataset("Svngoku/swahili-mmmlu")
        
        # Convert to pandas for easier analysis
        df = sw_mmlu['train'].to_pandas()
        
        # Basic dataset information
        print("Dataset Overview:")
        print("-" * 50)
        print(f"Total number of examples: {len(df)}")
        
        # Display column names
        print("\nColumns:")
        print(df.columns.tolist())
        
        # Subject distribution
        print("\nSubject Distribution:")
        subject_counts = df['subject'].value_counts()
        print(subject_counts.head(10))  # Top 10 subjects
        
        # Percentage of top subjects
        print("\nTop Subject Percentages:")
        subject_percentages = (subject_counts / len(df) * 100).head(10)
        print(subject_percentages)
        
        # Visualize top subjects
        plt.figure(figsize=(12, 6))
        subject_counts.head(10).plot(kind='bar')
        plt.title('Top 10 Subjects in Swahili MMLU Dataset')
        plt.xlabel('Subject')
        plt.ylabel('Number of Examples')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        # Display a few sample questions
        print("\nSample Questions:")
        print("-" * 50)
        for i in range(min(5, len(df))):
            print(f"\nSample {i+1}:")
            print(f"Subject: {df.iloc[i]['subject']}")
            print(f"Question: {df.iloc[i]['question']}")
            print("Options:")
            options = df.iloc[i]['options']
            for key, value in options.items():
                print(f"{key}: {value}")
            print(f"Correct Answer: {df.iloc[i]['answer']}")
        
        return sw_mmlu
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Load and analyze the Swahili MMLU dataset
swahili_mmlu_dataset = load_swahili_mmlu_dataset()

# Optional: Additional analysis of subject distribution
if swahili_mmlu_dataset is not None:
    # Convert to pandas for further analysis
    df = swahili_mmlu_dataset['train'].to_pandas()
    
    # Pie chart of top 10 subjects
    plt.figure(figsize=(12, 8))
    subject_counts = df['subject'].value_counts()
    top_subjects = subject_counts.head(10)
    others = pd.Series({'Others': subject_counts[10:].sum()})
    plot_data = pd.concat([top_subjects, others])
    
    plt.pie(plot_data, labels=plot_data.index, autopct='%1.1f%%')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def load_translation_dataset(english_path, swahili_path, num_samples=5):
    """
    Load English-Swahili translation dataset from Kaggle
    
    Parameters:
    - english_path: Path to English source texts
    - swahili_path: Path to Swahili translation texts
    - num_samples: Number of samples to display
    
    Returns:
    - List of translation pairs
    """
    try:
        # Read English texts
        with open(english_path, 'r', encoding='utf-8') as f:
            english_texts = f.readlines()
        
        # Read Swahili texts
        with open(swahili_path, 'r', encoding='utf-8') as f:
            swahili_texts = f.readlines()
        
        # Validate dataset
        assert len(english_texts) == len(swahili_texts), "Mismatched number of texts"
        
        # Print dataset size
        print(f"Total number of translation pairs: {len(english_texts)}")
        
        # Display samples
        print("\nTranslation Samples:")
        print("-" * 50)
        
        for i in range(min(num_samples, len(english_texts))):
            print(f"\nSample {i+1}:")
            print("English:")
            print(english_texts[i].strip())
            print("\nSwahili:")
            print(swahili_texts[i].strip())
            print("-" * 50)
        
        # Create list of translation pairs
        translation_pairs = list(zip(
            [text.strip() for text in english_texts],
            [text.strip() for text in swahili_texts]
        ))
        
        return translation_pairs
    
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Paths to the dataset
english_path = '/kaggle/input/wikimedia-english-swahili-dataset/wikimedia.en-sw.en'
swahili_path = '/kaggle/input/wikimedia-english-swahili-dataset/wikimedia.en-sw.sw'

# Load and display translation dataset
translation_pairs = load_translation_dataset(english_path, swahili_path)


def evaluate_model(prompt, model_path):
    """
    Generate Swahili text using Gemma2-2B-Swahili-IT model.
    
    Args:
        prompt (str): Input text prompt
        model_path (str): Path to model directory
    
    Returns:
        str: Generated response
    """
    try:
        # Initialize tokenizer with trust_remote_code
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False  # Try using slow tokenizer
        )
        
        # Initialize model with trust_remote_code
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Ensure model is in evaluation mode
        model.eval()
        
        # Prepare input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=True,
                temperature=0.7,
                top_p=0.95
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up
        del model, inputs, outputs
        torch.cuda.empty_cache()
        
        return response
        
    except Exception as e:
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        raise


gemma2_2b_it = "/kaggle/input/gemma-2/transformers/gemma-2-2b-it/2"
print("Gemma2 2B IT response on digitial economy prompt")
prompt = "Eleza dhana ya uchumi wa kidijitali na umuhimu wake katika ulimwengu wa leo, maelezo yako yasizidi maneno 200"
# Define the concept of the digital economy and its importance in today's world. Your explanation should not exceed 200 words.
response = generate_swahili_text(prompt, gemma2_2b_it)
print(response)


swahili_gemma2_2b_it = "/kaggle/input/gemma-2-swahili/transformers/gemma2-2b-swahili-it/1"
print("Gemma2 2B Swahili IT response on digitial economy prompt")
prompt = "Eleza dhana ya uchumi wa kidijitali na umuhimu wake katika ulimwengu wa leo, maelezo yako yasizidi maneno 200"
# Define the concept of the digital economy and its importance in today's world. Your explanation should not exceed 200 words.
response = generate_swahili_text(prompt, swahili_gemma2_2b_it)
print(response)


print("Gemma2 2B IT Creative Writing")
prompt = "Tunga hadithi fupi kuhusu Twiga, isizidi maneno 200"
# Write a short story about a giraffe, under 200 words
response = generate_swahili_text(prompt, gemma2_2b_it)
print(response)



print("Gemma2 2B Swahili IT  Creative Writing")
prompt = "Tunga hadithi fupi kuhusu Twiga, isizidi maneno 200"
# Write a short story about a giraffe, Under 200 words
response = generate_swahili_text(prompt, swahili_gemma2_2b_it)
print(response)

