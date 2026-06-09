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

# Load the datasets
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    sample_submission_df = pd.read_csv('sample_submission.csv')
except FileNotFoundError:
    print("Make sure train.csv, test.csv, and sample_submission.csv are in the same directory.")
    # As a fallback for environments like Google Colab, let's try a common Kaggle path
    try:
        KAGGLE_INPUT_PATH = '/kaggle/input/jigsaw-agile-community-rules/'
        train_df = pd.read_csv(KAGGLE_INPUT_PATH + 'train.csv')
        test_df = pd.read_csv(KAGGLE_INPUT_PATH + 'test.csv')
        sample_submission_df = pd.read_csv(KAGGLE_INPUT_PATH + 'sample_submission.csv')
    except FileNotFoundError:
        print("Could not find data files. Please check your file paths.")
        # Exit or create dummy dataframes if you want the script to continue
        train_df, test_df = pd.DataFrame(), pd.DataFrame()


# Let's inspect the training data
print("--- Training Data Info ---")
if not train_df.empty:
    print(train_df.info())
    print("\n--- First 5 Rows of Training Data ---")
    # Display all columns to see the full text
    with pd.option_context('display.max_colwidth', None):
        display(train_df.head())

# Inspect the test data
print("\n--- Test Data Info ---")
if not test_df.empty:
    print(test_df.info())


import pandas as pd

# (Assuming train_df is already loaded from the previous step)

# Create a list to hold our new, augmented data
augmented_data = []

# Iterate over each row of the original training dataframe
for _, row in train_df.iterrows():
    rule = row['rule']
    
    # 1. The original comment
    augmented_data.append({
        'text': row['body'],
        'rule': rule,
        'label': row['rule_violation']
    })
    
    # 2. Positive Example 1
    augmented_data.append({
        'text': row['positive_example_1'],
        'rule': rule,
        'label': 1  # This is a confirmed violation
    })
    
    # 3. Positive Example 2
    augmented_data.append({
        'text': row['positive_example_2'],
        'rule': rule,
        'label': 1  # This is a confirmed violation
    })
    
    # 4. Negative Example 1
    augmented_data.append({
        'text': row['negative_example_1'],
        'rule': rule,
        'label': 0  # This is a confirmed non-violation
    })
    
    # 5. Negative Example 2
    augmented_data.append({
        'text': row['negative_example_2'],
        'rule': rule,
        'label': 0  # This is a confirmed non-violation
    })

# Convert the list of dictionaries into a new DataFrame
augmented_train_df = pd.DataFrame(augmented_data)

# Let's see the result
print(f"Original training data size: {len(train_df)} rows")
print(f"Augmented training data size: {len(augmented_train_df)} rows")
print(f"({len(train_df)} * 5 = {len(train_df) * 5})")

print("\n--- First 10 Rows of Augmented Data ---")
# Shuffle the dataframe to see a mix of examples, then show the head
augmented_train_df = augmented_train_df.sample(frac=1).reset_index(drop=True)
with pd.option_context('display.max_colwidth', None):
    display(augmented_train_df.head(10))


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define the model we want to use
MODEL_NAME = 'microsoft/deberta-v3-base'

# A device to run the model on. Use GPU if available, otherwise CPU.
# This is important for training later.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load the Tokenizer
# This will download the tokenizer configuration for our chosen model.
print(f"Loading tokenizer for '{MODEL_NAME}'...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Tokenizer loaded successfully.")

# 2. Load the Model
# We use `AutoModelForSequenceClassification`.
# We specify `num_labels=2` to tell the model we are doing binary classification 
# (Label 0: No Violation, Label 1: Violation).
# This automatically creates the fresh classification head we designed in our strategy.
print(f"Loading model '{MODEL_NAME}' for sequence classification...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Move the model to the selected device (GPU or CPU)
model.to(device)

print("Model loaded successfully and moved to device.")
print("\n--- Model Architecture ---")
print(model)


import torch
from torch.utils.data import Dataset, DataLoader

# (Assuming augmented_train_df, tokenizer are already defined)

class JigsawDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=256):
        """
        Args:
            dataframe (pd.DataFrame): The dataframe with 'text', 'rule', and 'label' columns.
            tokenizer: The tokenizer for the model.
            max_length (int): The maximum sequence length for padding/truncation.
        """
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Fetches and processes one sample from the dataframe.
        This is where the magic happens.
        """
        # Get the data at the given index
        row = self.dataframe.iloc[idx]
        text = row['text']
        rule = row['rule']
        label = row['label']

        # Format the input as "RULE [SEP] TEXT"
        # The tokenizer.sep_token is the special separator token for the model (e.g., [SEP])
        combined_text = rule + self.tokenizer.sep_token + text

        # Tokenize the combined text
        inputs = self.tokenizer.encode_plus(
            combined_text,
            add_special_tokens=True,  # Adds [CLS] and [SEP] tokens
            max_length=self.max_length,
            padding='max_length',     # Pads short sequences to max_length
            truncation=True,          # Truncates long sequences to max_length
            return_tensors='pt'       # Returns PyTorch tensors
        )

        # The tokenizer returns a dictionary. We need to get the actual tensors
        # and remove the extra dimension (since we're processing one sample)
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        
        # DeBERTa-v3 models don't use token_type_ids, but some others do.
        # We can safely ignore it for DeBERTa.

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- Let's create an instance of our dataset ---
print("Creating the training dataset...")
train_dataset = JigsawDataset(augmented_train_df, tokenizer)
print("Dataset created successfully.")

# --- Let's inspect one sample to see what the model will receive ---
print("\n--- Inspecting a single sample from the dataset ---")
sample = train_dataset[0]
print("Keys:", sample.keys())
print("\nShape of input_ids:", sample['input_ids'].shape)
print("Shape of attention_mask:", sample['attention_mask'].shape)
print("Label:", sample['labels'])

# You can also decode the input_ids back to text to verify
print("\nDecoded input_ids:")
print(tokenizer.decode(sample['input_ids']))


from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments

# (Assuming augmented_train_df, model, tokenizer are already defined)

# --- 1. Split the data into training and validation sets ---
# We'll use 90% for training and 10% for validation.
train_subset_df, eval_subset_df = train_test_split(
    augmented_train_df,
    test_size=0.1,
    random_state=42,
    stratify=augmented_train_df['label'] # Ensures both sets have a similar label distribution
)

# Create Dataset objects for our new splits
train_subset_dataset = JigsawDataset(train_subset_df, tokenizer)
eval_subset_dataset = JigsawDataset(eval_subset_df, tokenizer)

print(f"Training on {len(train_subset_dataset)} samples.")
print(f"Validating on {len(eval_subset_dataset)} samples.")

# --- 2. Define Training Arguments (Final Workaround) ---
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=200,

    # --- FIX ---
    # To resolve the strategy mismatch in your old library version,
    # we will disable loading the best model at the end.
    # The model will still be saved, but it will be the final checkpoint.
    load_best_model_at_end=False,
    # --- END FIX ---
    
    # We still want to see evaluation metrics, so we need to
    # explicitly tell the trainer to evaluate.
    # We will try adding back the older flag.
    evaluate_during_training=True,
    eval_steps=200,
    save_steps=200,
    report_to="none"
)

# --- The rest of the code is the same ---

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_subset_dataset,
    eval_dataset=eval_subset_dataset
)

print("\nStarting the fine-tuning process...")
trainer.train()
print("Training complete!")




