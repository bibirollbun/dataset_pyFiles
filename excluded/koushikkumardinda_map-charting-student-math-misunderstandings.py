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


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the paths to your dataset files
# In a Kaggle notebook, these paths are typically standardized
train_csv_path = '/kaggle/input/map-charting-student-math-misunderstandings/train.csv'
test_csv_path = '/kaggle/input/map-charting-student-math-misunderstandings/test.csv'

# Load the datasets
print("Loading datasets...")
try:
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    print("Datasets loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure the files are in the correct directory.")
    # Exit or handle the error appropriately if files are not found
    exit()

# --- Initial Inspection ---
print("\n--- Training Data Info ---")
train_df.info()
print("\nFirst 5 rows of the training data:")
print(train_df.head())

print("\n--- Test Data Info ---")
test_df.info()
print("\nFirst 5 rows of the test data:")
print(test_df.head())

# --- Analyze 'Category' and 'Misconception' Columns ---
print("\n--- Category Distribution in Training Data ---")
category_counts = train_df['Category'].value_counts()
print(category_counts)
plt.figure(figsize=(10, 6))
sns.barplot(x=category_counts.index, y=category_counts.values, palette="viridis")
plt.title('Distribution of Categories')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Count')
plt.xlabel('Category')
plt.tight_layout()
plt.show()

print("\n--- Misconception Distribution in Training Data ---")
misconception_df = train_df[train_df['Misconception'] != 'NA']
misconception_counts = misconception_df['Misconception'].value_counts()
print(misconception_counts)
plt.figure(figsize=(12, 8))
sns.barplot(x=misconception_counts.index, y=misconception_counts.values, palette="rocket")
plt.title('Distribution of Misconceptions')
plt.xticks(rotation=90, ha='right')
plt.ylabel('Count')
plt.xlabel('Misconception')
plt.tight_layout()
plt.show()

# --- Analyze Text Data ---
# A simple check for text length can provide valuable insights
print("\n--- Text Length Analysis ---")
train_df['QuestionText_len'] = train_df['QuestionText'].apply(lambda x: len(str(x)))
train_df['StudentExplanation_len'] = train_df['StudentExplanation'].apply(lambda x: len(str(x)))

print("\nQuestionText length statistics:")
print(train_df['QuestionText_len'].describe())
print("\nStudentExplanation length statistics:")
print(train_df['StudentExplanation_len'].describe())

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(train_df['QuestionText_len'], bins=50, kde=True)
plt.title('Distribution of QuestionText Length')
plt.xlabel('Length')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(train_df['StudentExplanation_len'], bins=50, kde=True, color='orange')
plt.title('Distribution of StudentExplanation Length')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# --- Check for Missing Values ---
print("\n--- Missing Values Check ---")
print("Training Data Missing Values:")
print(train_df.isnull().sum())
print("\nTest Data Missing Values:")
print(test_df.isnull().sum())


import pandas as pd
import re

# Define the paths to your dataset files
train_csv_path = '/kaggle/input/map-charting-student-math-misunderstandings/train.csv'
test_csv_path = '/kaggle/input/map-charting-student-math-misunderstandings/test.csv'

# Load the datasets
try:
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    print("Datasets loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure the files are in the correct directory.")
    exit()

# --- Text Cleaning Function ---
def clean_text(text):
    """
    Cleans and preprocesses a string of text.
    Handles mathematical expressions and general text cleaning.
    """
    if not isinstance(text, str):
        return ""  # Return an empty string for non-string types

    # 1. Lowercase the text
    text = text.lower()

    # 2. Handle mathematical expressions
    # This regex looks for patterns of numbers, operators, and common math symbols.
    # It's a simple approach; a more advanced method might use a specialized library.
    # For this example, we'll replace math with a special token.
    math_pattern = r'(\d+(\.\d+)?[+\-*/=^xX]|sqrt|pi|cm|m|km|g|kg|ml|l|\(|\)|\{|\}|\[|\]|<|>)'
    text = re.sub(math_pattern, r' <MATH_EXP> ', text)

    # 3. Remove punctuation and special characters
    # Keep only letters, numbers, and whitespace.
    text = re.sub(r'[^a-z0-9\s<>]', '', text)
    
    # 4. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# --- Feature Engineering Function ---
def create_features(df):
    """
    Combines relevant columns into a single text feature.
    """
    df['clean_question'] = df['QuestionText'].apply(clean_text)
    df['clean_explanation'] = df['StudentExplanation'].apply(clean_text)
    df['clean_mc_answer'] = df['MC_Answer'].apply(clean_text)
    
    # Combine the cleaned text fields with a separator token for the model.
    # The '[SEP]' token is a common practice in transformer models to distinguish different segments of text.
    df['combined_features'] = (df['clean_question'] + " [SEP] " + 
                               df['clean_mc_answer'] + " [SEP] " + 
                               df['clean_explanation'])
    
    return df

# --- Apply Preprocessing to Datasets ---
print("\nPreprocessing training data...")
train_df = create_features(train_df)
print("Training data preprocessed.")
print("Example of combined feature from training data:")
print(train_df['combined_features'].iloc[0])

print("\nPreprocessing test data...")
test_df = create_features(test_df)
print("Test data preprocessed.")
print("Example of combined feature from test data:")
print(test_df['combined_features'].iloc[0])

# --- Final Check on the Processed DataFrames ---
print("\n--- Processed Training Data Info ---")
train_df.info()
print("\n--- Processed Test Data Info ---")
test_df.info()

# You can now save these processed dataframes or pass them directly to the next step (model training).
# Example of saving:
train_df.to_csv('processed_train.csv', index=False)
test_df.to_csv('processed_test.csv', index=False)


import pandas as pd
import torch
import os
import json
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# --- Configuration and Data Loading ---
# Adjust these paths to your specific Kaggle dataset paths
# Assuming you've uploaded the processed CSV and a transformer model like DistilBERT
PROCESSED_TRAIN_CSV_PATH = '/kaggle/working/processed_train.csv'
PRETRAINED_MODEL_PATH = '/kaggle/input/pretrain_model.torch/pytorch/default/5'
OUTPUT_DIR = '/kaggle/working/' 

# Create the output directory to ensure models can be saved
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the processed training data from Step 2
try:
    train_df = pd.read_csv(PROCESSED_TRAIN_CSV_PATH)
    print("Processed training data loaded.")
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure you ran Step 2 and uploaded the processed file.")
    exit()

# Load the pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)

# --- Stage 1: Category Prediction Model ---
print("\n--- Stage 1: Training the Category Classifier ---")

# Prepare labels for the Category model
category_labels = sorted(train_df['Category'].unique())
category_label_to_id = {label: i for i, label in enumerate(category_labels)}
train_df['category_label_id'] = train_df['Category'].map(category_label_to_id)

# Split data for training and validation, ensuring reproducibility
train_cat_df, val_cat_df = train_test_split(
    train_df, test_size=0.2, random_state=42, stratify=train_df['category_label_id']
)

# Convert to Hugging Face Dataset format
train_cat_dataset = Dataset.from_pandas(train_cat_df.reset_index(drop=True))
val_cat_dataset = Dataset.from_pandas(val_cat_df.reset_index(drop=True))

# Tokenize the datasets
def tokenize_function(examples):
    return tokenizer(examples['combined_features'], padding="max_length", truncation=True, max_length=512)

tokenized_train_cat = train_cat_dataset.map(tokenize_function, batched=True)
tokenized_val_cat = val_cat_dataset.map(tokenize_function, batched=True)

# Rename the label column to 'labels' as required by the Trainer API
tokenized_train_cat = tokenized_train_cat.rename_column("category_label_id", "labels")
tokenized_val_cat = tokenized_val_cat.rename_column("category_label_id", "labels")
tokenized_train_cat.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_val_cat.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Load the model with the correct number of labels and ignore any size mismatch
model_category = AutoModelForSequenceClassification.from_pretrained(
    PRETRAINED_MODEL_PATH, 
    num_labels=len(category_labels),
    ignore_mismatched_sizes=True  
)

print("Category labels:", category_labels)
print("Number of categories:", len(category_labels))
print("Category label map:", category_label_to_id)
print("Max category label ID:", max(train_df['category_label_id'].unique()))

# Define training arguments
training_args_category = TrainingArguments(
    output_dir=f'{OUTPUT_DIR}/category_model',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=100,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
    save_safetensors=True,  # Crucial fix: uses the more reliable safetensors format
)

# Initialize and start training
trainer_category = Trainer(
    model=model_category,
    args=training_args_category,
    train_dataset=tokenized_train_cat,
    eval_dataset=tokenized_val_cat,
)

trainer_category.train()
trainer_category.save_model(f'{OUTPUT_DIR}/category_model_final')
print("Category model training complete and saved.")

# --- Stage 2: Misconception Prediction Model ---
print("\n--- Stage 2: Training the Misconception Classifier ---")

# Filter data to include only rows with a misconception
misconception_df = train_df[train_df['Misconception'] != 'NA'].copy()

# Prepare labels for the Misconception model
misconception_labels = sorted(misconception_df['Misconception'].unique())
misconception_label_to_id = {label: i for i, label in enumerate(misconception_labels)}
misconception_df['misconception_label_id'] = misconception_df['Misconception'].map(misconception_label_to_id)

# Split data
train_mis_df, val_mis_df = train_test_split(
    misconception_df, test_size=0.2, random_state=42, stratify=misconception_df['misconception_label_id']
)

# Convert to Hugging Face Dataset format
train_mis_dataset = Dataset.from_pandas(train_mis_df.reset_index(drop=True))
val_mis_dataset = Dataset.from_pandas(val_mis_df.reset_index(drop=True))
tokenized_train_mis = train_mis_dataset.map(tokenize_function, batched=True)
tokenized_val_mis = val_mis_dataset.map(tokenize_function, batched=True)

# Rename the label column
tokenized_train_mis = tokenized_train_mis.rename_column("misconception_label_id", "labels")
tokenized_val_mis = tokenized_val_mis.rename_column("misconception_label_id", "labels")
tokenized_train_mis.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_val_mis.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Load the model with the correct number of labels and ignore any size mismatch
model_misconception = AutoModelForSequenceClassification.from_pretrained(
    PRETRAINED_MODEL_PATH, 
    num_labels=len(misconception_labels),
    ignore_mismatched_sizes=True  
)

print("Misconception labels:", misconception_labels)
print("Number of misconceptions:", len(misconception_labels))
print("Misconception label map:", misconception_label_to_id)
print("Max misconception label ID:", max(misconception_df['misconception_label_id'].unique()))

# Define training arguments
training_args_misconception = TrainingArguments(
    output_dir=f'{OUTPUT_DIR}/misconception_model',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=100,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
    save_safetensors=True, # Critical fix
)

# Initialize and start training
trainer_misconception = Trainer(
    model=model_misconception,
    args=training_args_misconception,
    train_dataset=tokenized_train_mis,
    eval_dataset=tokenized_val_mis,
)

trainer_misconception.train()
trainer_misconception.save_model(f'{OUTPUT_DIR}/misconception_model_final')
print("Misconception model training complete and saved.")

# Save label mappings to be used in the final submission step
with open(os.path.join(OUTPUT_DIR, 'category_labels.json'), 'w') as f:
    json.dump(category_labels, f)

with open(os.path.join(OUTPUT_DIR, 'misconception_labels.json'), 'w') as f:
    json.dump(misconception_labels, f)

print("\nLabel mappings saved.")


import pandas as pd
import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Configuration and File Paths ---
# Adjust these paths to where your files are located in the Kaggle environment
TEST_CSV_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/test.csv'
PROCESSED_TEST_CSV_PATH = '/kaggle/working/processed_test.csv' # Path to your preprocessed test data
MODEL_DIR = '/kaggle/working/'
MODEL_PATH_CATEGORY = os.path.join(MODEL_DIR, 'category_model_final')
MODEL_PATH_MISCONCEPTION = os.path.join(MODEL_DIR, 'misconception_model_final')
LABEL_MAPPINGS_PATH_CATEGORY = os.path.join(MODEL_DIR, 'category_labels.json')
LABEL_MAPPINGS_PATH_MISCONCEPTION = os.path.join(MODEL_DIR, 'misconception_labels.json')
PRETRAINED_MODEL_PATH = '/kaggle/input/pretrain_model.torch/pytorch/default/5'

# --- Load Necessary Components ---
print("Loading tokenizer, models, and label mappings...")
try:
    # Load the tokenizer used during training
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)
    
    # Load the trained models
    model_category = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH_CATEGORY)
    model_misconception = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH_MISCONCEPTION)
    
    # Load the label mappings
    with open(LABEL_MAPPINGS_PATH_CATEGORY, 'r') as f:
        category_labels = json.load(f)
    with open(LABEL_MAPPINGS_PATH_MISCONCEPTION, 'r') as f:
        misconception_labels = json.load(f)
        
    print("All components loaded successfully!")

except Exception as e:
    print(f"Error loading files: {e}")
    print("Please ensure your Step 3 notebook ran successfully and saved all outputs to a dataset.")
    exit()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_category.to(device)
model_misconception.to(device)
model_category.eval()
model_misconception.eval()

# --- Preprocessing Function (same as Step 2) ---
def clean_text(text):
    import re
    if not isinstance(text, str): return ""
    text = text.lower()
    math_pattern = r'(\d+(\.\d+)?[+\-*/=^xX]|sqrt|pi|cm|m|km|g|kg|ml|l|\(|\)|\{|\}|\[|\]|<|>)'
    text = re.sub(math_pattern, r' <MATH_EXP> ', text)
    text = re.sub(r'[^a-z0-9\s<>]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_features(df):
    df['combined_features'] = (df['QuestionText'].apply(clean_text) + " [SEP] " +
                               df['MC_Answer'].apply(clean_text) + " [SEP] " +
                               df['StudentExplanation'].apply(clean_text))
    return df

# --- Main Prediction Function ---
def generate_submission(test_df):
    """
    Generates predictions for each row in the test set.
    """
    submission_rows = []

    # Apply preprocessing to the test data
    test_df = create_features(test_df)

    for _, row in test_df.iterrows():
        input_text = row['combined_features']

        # Tokenize the input text
        encoded_input = tokenizer(input_text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
        predictions = []

        # --- Stage 1: Predict Category and get top 3 candidates ---
        with torch.no_grad():
            category_outputs = model_category(**encoded_input)
            category_logits = category_outputs.logits
        
        category_probs = torch.softmax(category_logits, dim=-1).squeeze(0)
        top3_cat_probs, top3_cat_indices = torch.topk(category_probs, k=3)
        
        for i in range(3):
            predicted_category = category_labels[top3_cat_indices[i].item()]
            
            # --- Stage 2: Predict Misconception if applicable ---
            if 'Misconception' in predicted_category:
                with torch.no_grad():
                    misconception_outputs = model_misconception(**encoded_input)
                    misconception_logits = misconception_outputs.logits
                
                misconception_idx = torch.argmax(misconception_logits, dim=-1).item()
                predicted_misconception = misconception_labels[misconception_idx]
                
                final_prediction = f"{predicted_category}:{predicted_misconception}"
            else:
                final_prediction = predicted_category
            
            predictions.append(final_prediction)

        # Join the top 3 predictions with a space
        submission_string = " ".join(predictions)
        
        submission_rows.append({
            'QuestionId': row['QuestionId'], 
            'Category:Misconception': submission_string
        })

    # Create the final submission dataframe
    submission_df = pd.DataFrame(submission_rows)
    
    # Save the file
    submission_df.to_csv('submission.csv', index=False)
    print("Submission file generated successfully!")

# --- Main Execution ---
if __name__ == "__main__":
    # Load the raw test data
    test_df = pd.read_csv(TEST_CSV_PATH)
    generate_submission(test_df)

