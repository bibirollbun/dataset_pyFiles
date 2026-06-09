!pip install -U seaborn > /dev/null
!pip install tqdm > /dev/null


# Basics
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import string

from datasets import Dataset

# Data Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score

# NLP and Transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoModelForSequenceClassification, logging, EarlyStoppingCallback
import torch

# Utilities
from tqdm import tqdm
import re
import os
import random


print(sns.__version__)


# Loading the Competition paths
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


class CFG:

    # DEBUG = True  
    DEBUG = False  
    debug_rate = 0.05 if DEBUG else 1
    
    # Data and model paths
    train_data_path = "/kaggle/input/TweetSentimentBR/Train.csv"
    test_data_path = "/kaggle/input/TweetSentimentBR/Test.csv"
    submission_sample_path = '/kaggle/input/TweetSentimentBR/zSample_Submission.csv'
    model_name = "neuralmind/bert-base-portuguese-cased"  # Or 'pierreguillou/bert-base-cased-sentiment'
    submission_path = "/kaggle/working/submission.csv"
    
    # Training parameters
    max_len = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Seed for reproducibility
    seed = 42

    # training args
    training_args = {
        'output_dir' : './results',         # Directory to save model and checkpoints
        'run_name' : "bertimbau_sentiment_analysis",  # Unique run name
        'eval_strategy' : "epoch",   # Evaluate at the end of each epoch
        'save_strategy' : "epoch",         # Save model at the end of each epoch
        'learning_rate' : 3e-5,
        'per_device_train_batch_size' : 16,
        'per_device_eval_batch_size' : 16,
        'gradient_accumulation_steps': 2,
        'num_train_epochs' : 10,
        'logging_dir' : './logs',          # Directory for logs
        'logging_steps' : 10,
        'load_best_model_at_end' : True,   # Load the best model at the end of training
        'metric_for_best_model' : "accuracy",  # Metric to evaluate the best model
        'save_total_limit' : 1,             # Limit saved checkpoints
        'seed': seed,
        'fp16': True,  # Enables mixed precision for faster training on GPU
        'optim': "adamw_torch"  # More optimized AdamW implementation
    }

    early_stopping_patience = 2


def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    # Remove hashtags (and the text attached until the first space)
    text = re.sub(r'#\S+', '', text)
    # Remove mentions (@ and the text attached until the first space)
    text = re.sub(r'@\S+', '', text)
    # Remove repeated spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove leading and trailing spaces
    text = text.strip()
    return text

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average='weighted')
    }

def prepare_dataset(train_df, val_df, test_df, model_name, max_length):
    """
    Prepares and tokenizes the dataset for training, validation, and testing.

    Parameters:
    train_df (pd.DataFrame): Training DataFrame containing 'cleaned_text' and 'sentiment'.
    val_df (pd.DataFrame): Validation DataFrame containing 'cleaned_text' and 'sentiment'.
    test_df (pd.DataFrame): Test DataFrame containing 'id' and 'cleaned_text'.
    model_name (str): Pretrained tokenizer model name.
    max_length (int): Maximum token sequence length.

    Returns:
    tuple: Train dataset, validation dataset, test dataset, and tokenizer.
    """

    # Rename the 'sentiment' column to 'labels'
    train_df = train_df.rename(columns={'sentiment': 'labels'})
    val_df = val_df.rename(columns={'sentiment': 'labels'})

    # Convert DataFrames to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples['cleaned_text'], 
                         truncation = True, 
                         padding = "max_length", 
                         max_length = max_length)

    # Apply tokenization
    train_dataset = train_dataset.map(tokenize_function, batched = True)
    val_dataset = val_dataset.map(tokenize_function, batched = True)
    test_dataset = test_dataset.map(tokenize_function, batched = True)

    # Remove unnecessary columns
    train_dataset = train_dataset.remove_columns(['cleaned_text'])
    val_dataset = val_dataset.remove_columns(['cleaned_text'])
    test_dataset = test_dataset.remove_columns(['cleaned_text'])

    print("Dataset preparation completed.")
    
    return train_dataset, val_dataset, test_dataset, tokenizer

def setup_and_train(model_name, num_labels, training_args, train_dataset, val_dataset, tokenizer, compute_metrics):
    """
    Sets up the model and trainer, then starts training.

    Parameters:
    model_name (str): Pretrained model name.
    num_labels (int): Number of labels for classification.
    training_args (dict): Training configuration.
    train_dataset (Dataset): Training dataset.
    val_dataset (Dataset): Validation dataset.
    tokenizer: Tokenizer instance.
    compute_metrics (function): Function to compute evaluation metrics.

    Returns:
    Trainer: Trained model's trainer instance.
    """
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels = num_labels
    )
    
    # Set training arguments
    args = TrainingArguments(**training_args)
    
    # Initialize Trainer
    trainer = Trainer(
        model = model,
        args = args,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        processing_class = tokenizer,
        compute_metrics = compute_metrics
    )

    early_stopping = EarlyStoppingCallback(early_stopping_patience = CFG.early_stopping_patience)
    trainer.add_callback(early_stopping)
    
    # Train the model
    trainer.train()
    print(f"Best model loaded with accuracy: {trainer.state.best_metric}")

    return trainer


logging.set_verbosity_error()

os.environ["WANDB_DISABLED"] = "true"


# Load the datasets
train_df = pd.read_csv(CFG.train_data_path)
test_df = pd.read_csv(CFG.test_data_path)

# Quick view of the data
print("Training data preview:")
print(train_df.head())
print("\nTesting data preview:")
print(test_df.head())


print('Train:\n')
print(train_df.columns)
print(train_df.shape)
print('\nTest:\n')
print(test_df.columns)
print(test_df.shape)


columns_to_keep = ['id', 'tweet_text', 'sentiment']


# Check for missing values
print('Train:\n')
print(train_df.isnull().sum())
print('\nTest:\n')
print(test_df.isnull().sum())

# # Heatmap of missing values
# import seaborn as sns
# plt.figure(figsize=(10, 6))
# sns.heatmap(train_df.isnull(), cbar=False, cmap='viridis')
# plt.title('Missing Values Heatmap')
# plt.show()


# Frequency plot
sentiment_counts = train_df['sentiment'].value_counts()
plt.figure(figsize=(4, 3))
sentiment_counts.plot(kind='bar', color=['skyblue', 'salmon', 'lightgreen'])
plt.title('Frequency of Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()


NUM_LABELS = 2


# Add a new column with text lengths
train_df['text_length'] = train_df['tweet_text'].apply(len)

# Plot distribution
plt.figure(figsize=(10, 6))
train_df['text_length'].plot(kind='hist', bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Train Tweet Text Length')
plt.xlabel('Length of Tweet')
plt.ylabel('Frequency')
plt.show()


# Filter rows where 'tweet_text' has less than 10 characters
short_tweets = train_df[train_df['tweet_text'].str.len() < 10]
print('Train:\n')
# Display the first rows
print(short_tweets[['tweet_text', 'sentiment']].head(10))


# Add a new column with text lengths
test_df['text_length'] = test_df['tweet_text'].apply(len)

# Plot distribution
plt.figure(figsize=(10, 6))
test_df['text_length'].plot(kind='hist', bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Test Tweet Text Length')
plt.xlabel('Length of Tweet')
plt.ylabel('Frequency')
plt.show()


# Filter rows where 'tweet_text' has less than 10 characters
test_short_tweets = test_df[test_df['tweet_text'].str.len() < 10]
print('\nTest:\n')
# Display the first rows
print(test_short_tweets[['tweet_text']].head(10))


# Check for non-printable characters
non_printable = set(string.printable)
train_df['has_nonprintable'] = train_df['tweet_text'].apply(
    lambda x: any(char not in non_printable for char in x)
)


print('Train:\n')

non_printable = set(string.printable)
train_df['has_nonprintable'] = train_df['tweet_text'].apply(
    lambda x: any(char not in non_printable for char in x)
)
# Display rows with non-printable characters
non_printable_rows = train_df[train_df['has_nonprintable']]
print(non_printable_rows)


print('\nTest:\n')
non_printable = set(string.printable)
test_df['has_nonprintable'] = test_df['tweet_text'].apply(
    lambda x: any(char not in non_printable for char in x)
)
# Display rows with non-printable characters
non_printable_rows = test_df[test_df['has_nonprintable']]
print(non_printable_rows)



print('Train:\n')

# Check for HTML tags
train_df['has_html'] = train_df['tweet_text'].str.contains(r'<[^>]*>', regex=True)

# Count rows with HTML
html_count = train_df['has_html'].sum()
print(f"Tweets with HTML tags: {html_count}")

print('\nTest:\n')
# Check for HTML tags
test_df['has_html'] = test_df['tweet_text'].str.contains(r'<[^>]*>', regex=True)

# Count rows with HTML
html_count = test_df['has_html'].sum()
print(f"Tweets with HTML tags: {html_count}")


# Check for URLs
print('Train:\n')
train_df['has_url'] = train_df['tweet_text'].str.contains(r'http[s]?://', regex=True)
print(f"Tweets with URLs: {train_df['has_url'].sum()}")

# Check for URLs
print('\nTest:\n')

test_df['has_url'] = test_df['tweet_text'].str.contains(r'http[s]?://', regex=True)
print(f"Tweets with URLs: {test_df['has_url'].sum()}")




# Check for mentions
print('Train:\n')
train_df['has_mentions'] = train_df['tweet_text'].str.contains(r'@\w+', regex=True)
print(f"Tweets with mentions: {train_df['has_mentions'].sum()}")

# Check for mentions
print('\nTest:\n')
test_df['has_mentions'] = test_df['tweet_text'].str.contains(r'@\w+', regex=True)
print(f"Tweets with mentions: {test_df['has_mentions'].sum()}")


# Check for hashtags
print('Train:\n')
train_df['has_hashtags'] = train_df['tweet_text'].str.contains(r'#\w+', regex=True)
print(f"Tweets with hashtags: {train_df['has_hashtags'].sum()}")

# Check for hashtags
print('\nTest:\n')
test_df['has_hashtags'] = test_df['tweet_text'].str.contains(r'#\w+', regex=True)
print(f"Tweets with hashtags: {test_df['has_hashtags'].sum()}")


print('Train:\n')
# Apply preprocessing to the 'tweet_text' column
train_df['cleaned_text'] = train_df['tweet_text'].apply(preprocess_text)

# Check the result
print(train_df[['tweet_text', 'cleaned_text']].head())



print('\nTest:\n')
# Apply preprocessing to the 'tweet_text' column
test_df['cleaned_text'] = test_df['tweet_text'].apply(preprocess_text)

# Check the result
print(test_df[['tweet_text', 'cleaned_text']].head())


# Check for missing values
print('Train:\n')
print(train_df.isnull().sum())
print('\nTest:\n')
print(test_df.isnull().sum())


# Filter rows where 'tweet_text' has less than 10 characters
min_len = 5
short_tweets = train_df[train_df['tweet_text'].str.len() < min_len]
print('Train:\n')
# Print the number of rows with less than min_len characters
print(f"Number of tweets with less than {min_len} characters: {len(short_tweets)}")

# Display the first rows
print(short_tweets[['tweet_text', 'sentiment']].head(10))


# Filter rows where 'tweet_text' has less than 10 characters
test_short_tweets = test_df[test_df['tweet_text'].str.len() < min_len]
print('\nTest:\n')
# Print the number of rows with less than min_len characters
print(f"Number of tweets with less than {min_len} characters: {len(test_short_tweets)}")
# Display the first rows
print(test_short_tweets[['tweet_text']].head(10))


# Tokenize the train and test texts to get token lengths
tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

train_token_lengths = train_df['cleaned_text'].apply(lambda x: len(tokenizer.tokenize(x)))
test_token_lengths = test_df['cleaned_text'].apply(lambda x: len(tokenizer.tokenize(x)))

# Replace infinite values with NaN, then drop them
train_token_lengths.replace([np.inf, -np.inf], np.nan, inplace=True)
test_token_lengths.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop NaN values
train_token_lengths.dropna(inplace=True)
test_token_lengths.dropna(inplace=True)

# Count rows where token length > 128
train_above_128 = (train_token_lengths > 128).sum()
test_above_128 = (test_token_lengths > 128).sum()

# Print the counts
print(f"Number of train samples with >128 tokens: {train_above_128}")
print(f"Number of test samples with >128 tokens: {test_above_128}")

# Create bins for visualization
max_length = max(train_token_lengths.max(), test_token_lengths.max())
bins = np.linspace(0, max_length, 30)
bins_above_128 = np.linspace(128, max_length, 20)

# Plot the first histogram (0 to max)
plt.figure(figsize=(10, 6))
sns.histplot(train_token_lengths, bins=bins, color="blue", alpha=0.6, label="Train", kde=True)
sns.histplot(test_token_lengths, bins=bins, color="lightblue", alpha=0.6, label="Test", kde=True)

plt.xlabel("Number of Tokens")
plt.ylabel("Frequency")
plt.title("Token Length Distribution (Train vs Test) - Full Range")
plt.legend()
plt.show()

# Plot the second histogram (128 to max)
plt.figure(figsize=(10, 6))
sns.histplot(train_token_lengths[train_token_lengths > 128], bins=bins_above_128, color="red", alpha=0.6, label="Train", kde=True)
sns.histplot(test_token_lengths[test_token_lengths > 128], bins=bins_above_128, color="orange", alpha=0.6, label="Test", kde=True)

plt.xlabel("Number of Tokens")
plt.ylabel("Frequency")
plt.title("Token Length Distribution (Train vs Test) - Above 128 Tokens")
plt.legend()
plt.show()



print(train_df.columns)
print(train_df.shape)


print(test_df.columns)
print(test_df.shape)


# Apply sampling if debug_rate < 1.0
if CFG.debug_rate < 1.0:
    # Use train_test_split for stratified downsampling
    train_df, _ = train_test_split(
        train_df,
        train_size = CFG.debug_rate,  # Keep only the required fraction
        stratify = train_df['sentiment'],  # Ensure class balance
        random_state = CFG.seed
    )
    test_df = test_df.sample(frac=CFG.debug_rate, random_state=CFG.seed).reset_index(drop=True)

# Check the sizes of the datasets after sampling
print(f"After debug sampling (if applied):")
print(f"Sampled Train Size: {train_df.shape}")
print(f"Sampled Test Size: {test_df.shape}")

train_df, val_df = train_test_split(
    train_df,
    test_size = 0.3,
    stratify = train_df['sentiment'],
    random_state = CFG.seed
)

# Display the new dataset sizes
print(f"Train Size: {train_df.shape}")
print(f"Validation Size: {val_df.shape}")


# Required columns for training and inference
required_train_columns = ['cleaned_text', 'sentiment']
required_test_columns = ['id', 'cleaned_text']


# Filter the columns
train_df = train_df[required_train_columns]
val_df = val_df[required_train_columns]
test_df = test_df[required_test_columns]


# Display the new dataset sizes
print(f"Train Size: {train_df.shape}")
print(f"Validation Size: {val_df.shape}")
print(f"Test Size: {test_df.shape}")


# Prepare datasets
train_dataset, val_dataset, test_dataset, tokenizer = prepare_dataset(
    train_df = train_df, 
    val_df = val_df, 
    test_df = test_df, 
    model_name = CFG.model_name, 
    max_length = CFG.max_len
)


CFG.training_args


# Train the model
trainer = setup_and_train(
    model_name = CFG.model_name, 
    num_labels = NUM_LABELS,  # Binary classification
    training_args = CFG.training_args, 
    train_dataset = train_dataset, 
    val_dataset = val_dataset, 
    tokenizer = tokenizer, 
    compute_metrics = compute_metrics
)


# Generate predictions
raw_predictions = trainer.predict(test_dataset)


# Extract logits and convert to labels
predictions = raw_predictions.predictions.argmax(axis=1)

# Ensure 'id' column is present in test_df
submission = test_df[['id']].copy()
submission['sentiment'] = predictions  # Add predictions


submission.to_csv(CFG.submission_path, index=False)
print(f"Submission file saved as {CFG.submission_path}.")


sub = pd.read_csv(CFG.submission_path)
print(sub.head())

