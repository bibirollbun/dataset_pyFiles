# --- 1. Import Libraries ---
import pandas as pd
from datasets import Dataset, ClassLabel
from transformers import AutoTokenizer


# --- 2. Define Configuration ---
class CFG:
    # We are preparing data for our future powerful model, DeBERTa
    TOKENIZER_NAME = 'microsoft/deberta-v3-large'
    MAX_LENGTH = 128
    OUTPUT_DIR = "pretokenized_deberta_large_competition_only" # Clearer name


# --- 3. Load Data ---
# We will use ONLY the original competition data for this notebook.
# This keeps the process clean and 100% your own work.
train_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')


# --- 4. Create the Correct Combined Target Label ---
print("Creating the combined 'Category:Misconception' target label...")

# First, fill the NaN values in 'Misconception' with a placeholder string 'NA'
train_df['Misconception'] = train_df['Misconception'].fillna('NA')

# Create the new target column by concatenating 'Category' and 'Misconception'
train_df['target'] = train_df['Category'] + ':' + train_df['Misconception']

# Select only the columns we need for training and drop the old ones
final_df = train_df[['StudentExplanation', 'target']].copy()

print(f"Using {final_df.shape[0]} samples for processing.")
print(f"Number of unique combined labels: {final_df['target'].nunique()}")
print("-" * 30)


# Display a few rows to verify our new target
print("Sample of the data with the new 'target' column:")
display(final_df.head())


# --- 1. Create Label Mappings for the New Target ---
unique_labels = final_df['target'].unique().tolist()
label2id = {label: i for i, label in enumerate(unique_labels)}

# Rename 'target' to 'label' for compatibility with Hugging Face Trainer
final_df['label'] = final_df['target'].map(label2id)
final_df = final_df.drop(columns=['target']) # Drop the old text-based target


# --- 2. Load the Tokenizer ---
print(f"Loading tokenizer: {CFG.TOKENIZER_NAME}")
tokenizer = AutoTokenizer.from_pretrained(CFG.TOKENIZER_NAME)
print("Tokenizer loaded successfully.")
print("-" * 30)


# --- 3. Create and Tokenize the Hugging Face Dataset ---
dataset = Dataset.from_pandas(final_df)

def tokenize_function(examples):
    """Applies the tokenizer to a batch of examples."""
    return tokenizer(
        examples["StudentExplanation"],
        padding="max_length",
        truncation=True,
        max_length=CFG.MAX_LENGTH
    )

print("Starting tokenization...")
# Remove the original text column after tokenization
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['StudentExplanation']
)
print("Tokenization complete!")
print("-" * 30)

print("Structure of the final tokenized dataset:")
print(tokenized_dataset)
print("-" * 30)


# --- 4. Save the Processed Dataset to Disk ---
print(f"Saving dataset to directory: ./{CFG.OUTPUT_DIR}")
tokenized_dataset.save_to_disk(CFG.OUTPUT_DIR)
print("Dataset saved successfully!")




