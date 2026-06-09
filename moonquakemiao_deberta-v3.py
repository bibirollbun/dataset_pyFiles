!pip install transformers datasets accelerate -U -q


"""
DeBERTa-v3 Movie Review Sentiment Classification - Full Code for Kaggle Competition
Suitable for: https://www.kaggle.com/competitions/py-sphere-movie-review-sentiment-challenge
Current baseline: 0.81368 â†’ Target: 0.90+
"""

# ===== 1. Install Dependencies =====
# Run in the first cell of Kaggle Notebook:
# !pip install transformers datasets accelerate -U -q

# ===== 2. Import Libraries =====
import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, # Used to load the tokenizer for the pre-trained model
    AutoModelForSequenceClassification, # Used to load the pre-trained model for sequence classification
    TrainingArguments, # Defines the training configuration for the Trainer
    Trainer # A class for training PyTorch models with ğŸ¤— Transformers
)
from datasets import Dataset # Hugging Face's Dataset object for efficient data handling
from sklearn.model_selection import train_test_split # Utility for splitting datasets into train and test sets
from sklearn.metrics import accuracy_score, f1_score # Metrics to evaluate model performance
import warnings # Used to manage warnings
warnings.filterwarnings('ignore') # Ignores all warning messages for cleaner output

# ===== 3. Set Random Seed (for reproducibility) =====
def set_seed(seed=42):
    """
    Sets the random seed for NumPy, PyTorch, and CUDA to ensure reproducibility
    of experiments.
    """
    np.random.seed(seed) # Seed for NumPy random number generator
    torch.manual_seed(seed) # Seed for PyTorch on CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # Seed for PyTorch on all CUDA GPUs

set_seed(42) # Call the function to set the global random seed

# ===== 4. Load Data =====
# Modify to your Kaggle data path if different
# Loads the training, testing, and sample submission data from CSV files.
train_df = pd.read_csv('/kaggle/input/py-sphere-movie-review-sentiment-challenge/train.csv')
test_df = pd.read_csv('/kaggle/input/py-sphere-movie-review-sentiment-challenge/test.csv')
sample_submission = pd.read_csv('/kaggle/input/py-sphere-movie-review-sentiment-challenge/sample_submission.csv')

# Prints the size of the datasets and the distribution of sentiment labels in the training set.
print(f"Training set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")
print(f"Positive/Negative sample distribution:\n{train_df['sentiment'].value_counts()}")

# ===== 5. Data Preprocessing =====
# Clean text data
def clean_text(text):
    """
    Simple text cleaning function: converts text to string and removes leading/trailing whitespace.
    Further cleaning (e.g., removing punctuation, special characters) might be added here.
    """
    text = str(text).strip() # Ensure text is a string and remove whitespace
    return text

# Apply the cleaning function to the 'review' columns of both training and testing DataFrames.
train_df['review'] = train_df['review'].apply(clean_text)
test_df['review'] = test_df['review'].apply(clean_text)

# Split training set into training and validation sets (Crucial for realistic performance evaluation)
# Splits `train_df` into `train_data` (90%) and `val_data` (10%).
# `test_size=0.1`: 10% of the original training data will be used for validation.
# `random_state=42`: Ensures the split is reproducible.
# `stratify=train_df['sentiment']`: Maintains the same proportion of sentiment classes
#                                   in both the training and validation splits as in the original `train_df`.
train_data, val_data = train_test_split(
    train_df,
    test_size=0.1,
    random_state=42,
    stratify=train_df['sentiment'] # Keep category distribution consistent
)

# Prints the sizes and sentiment distributions of the newly created training and validation sets.
print(f"\nTraining data: {len(train_data)}, Validation data: {len(val_data)}")
print(f"Training data distribution: {train_data['sentiment'].value_counts().to_dict()}")
print(f"Validation data distribution: {val_data['sentiment'].value_counts().to_dict()}")

# ===== 6. Load Model and Tokenizer =====
MODEL_NAME = "microsoft/deberta-v3-base" # Specifies the name of the pre-trained DeBERTa-v3 base model
# If GPU memory is insufficient, you can use the small version:
# MODEL_NAME = "microsoft/deberta-v3-small"

print(f"\nLoading model: {MODEL_NAME}")
# Loads the tokenizer associated with the specified pre-trained model.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Loads the pre-trained DeBERTa-v3 model for sequence classification.
# `num_labels=2`: Configures the model for binary classification (positive/negative sentiment).
# `ignore_mismatched_sizes=True`: Allows loading weights even if some layers' sizes don't perfectly match,
#                                  useful if the pre-trained model had a different number of output labels.
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    ignore_mismatched_sizes=True
)

# ===== 7. Data Encoding =====
def tokenize_function(examples):
    """
    Tokenizes a batch of text examples using the pre-trained tokenizer.
    `padding='max_length'`: Pads shorter sequences to `max_length`.
    `truncation=True`: Truncates longer sequences to `max_length`.
    `max_length=256`: Sets the maximum sequence length for tokenization.
                      Can be adjusted; longer sequences might yield better results but are slower.
    """
    return tokenizer(
        examples['review'],
        padding='max_length',
        truncation=True,
        max_length=256 # Adjustable, longer might be slower but better results
    )

# Convert Pandas DataFrames to Hugging Face Dataset format
# `from_pandas` creates a Dataset object from a DataFrame, selecting specified columns.
train_dataset = Dataset.from_pandas(train_data[['review', 'sentiment']])
val_dataset = Dataset.from_pandas(val_data[['review', 'sentiment']])
test_dataset = Dataset.from_pandas(test_df[['review']])

# Rename column (Trainer requires 'label' column)
# The `Trainer` expects the target column to be named 'labels', so we rename 'sentiment'.
train_dataset = train_dataset.rename_column('sentiment', 'labels')
val_dataset = val_dataset.rename_column('sentiment', 'labels')

# Batch encoding
print("\nEncoding data...")
# Apply the `tokenize_function` to each dataset in batches.
# `batched=True`: Processes multiple examples at once for efficiency.
# `remove_columns=['review']`: Removes the original 'review' text column after tokenization
#                               to save memory, as the tokenized IDs are now the features.
train_tokenized = train_dataset.map(tokenize_function, batched=True, remove_columns=['review'])
val_tokenized = val_dataset.map(tokenize_function, batched=True, remove_columns=['review'])
test_tokenized = test_dataset.map(tokenize_function, batched=True, remove_columns=['review'])

# ===== 8. Define Evaluation Metrics =====
def compute_metrics(eval_pred):
    """
    Computes accuracy and F1-score for evaluation.
    `eval_pred` is a tuple containing predictions (logits) and true labels.
    """
    logits, labels = eval_pred # Unpack predictions (raw model outputs) and true labels
    predictions = np.argmax(logits, axis=-1) # Convert logits to class predictions (0 or 1)
    acc = accuracy_score(labels, predictions) # Calculate accuracy
    f1 = f1_score(labels, predictions, average='binary') # Calculate F1-score for binary classification
    return {
        'accuracy': acc,
        'f1': f1
    }

# ===== 9. Training Arguments Setup =====
training_args = TrainingArguments(
    output_dir='./results', # Directory where model checkpoints and outputs will be saved

    # â­� Fixed parameter names for consistency with Hugging Face Transformers updates
    eval_strategy='epoch', # Evaluate every epoch
    save_strategy='epoch', # Save model checkpoints every epoch

    # Learning rate and batch size
    learning_rate=2e-5, # Initial learning rate for the optimizer
    per_device_train_batch_size=16, # Batch size per GPU/CPU for training (reduce if Out Of Memory)
    per_device_eval_batch_size=32, # Batch size per GPU/CPU for evaluation

    # Number of training epochs
    num_train_epochs=3, # Total number of training epochs (can try 4-5 for potentially better results)

    # Regularization
    weight_decay=0.01, # L2 regularization applied to all layers except bias and layer normalization weights

    # Early stopping and model saving
    load_best_model_at_end=True, # Loads the best model (based on `metric_for_best_model`) at the end of training
    metric_for_best_model='accuracy', # Metric used to determine the "best" model to load

    # Logging
    logging_dir='./logs', # Directory for storing logs
    logging_steps=50, # Log training progress every 50 steps

    # Other settings
    report_to='none', # Disables reporting to experiment tracking platforms like Weights & Biases
    seed=42, # Random seed for reproducibility during training
    fp16=torch.cuda.is_available(), # Automatically enables mixed precision training if a CUDA GPU is available
)

# ===== 10. Create Trainer and Train =====
# Initializes the Hugging Face Trainer with the model, arguments, datasets, tokenizer, and metrics.
trainer = Trainer(
    model=model, # The ğŸ¤— Transformers model to train
    args=training_args, # The training arguments
    train_dataset=train_tokenized, # The tokenized training dataset
    eval_dataset=val_tokenized, # The tokenized validation dataset
    tokenizer=tokenizer, # The tokenizer used for encoding (optional, but good practice to pass)
    compute_metrics=compute_metrics, # The function to compute metrics during evaluation
)

print("\nStarting training...")
trainer.train() # Starts the training process

# ===== 11. Validation Set Evaluation =====
print("\nEvaluating on validation set...")
# Evaluates the trained model on the validation set.
eval_results = trainer.evaluate()
# Prints the accuracy and F1-score on the validation set.
print(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"Validation F1 Score: {eval_results['eval_f1']:.4f}")

# ===== 12. Retrain on Full Training Data (Optional, usually improves by 1-2%) =====
print("\nRetraining with full training data...")
# Creates a full training dataset from the original `train_df`.
full_train_dataset = Dataset.from_pandas(train_df[['review', 'sentiment']])
full_train_dataset = full_train_dataset.rename_column('sentiment', 'labels') # Rename 'sentiment' to 'labels'
full_train_tokenized = full_train_dataset.map(tokenize_function, batched=True, remove_columns=['review']) # Tokenize

# Re-create Trainer (without validation set for final training)
# A new Trainer is created for training on the *entire* original training dataset.
# Validation is usually skipped in this final stage as the goal is to maximize performance on test data.
final_trainer = Trainer(
    model=model, # The same model (potentially fine-tuned from previous training)
    args=TrainingArguments(
        output_dir='./final_results', # Output directory for this final training run
        per_device_train_batch_size=16, # Training batch size
        num_train_epochs=3, # Number of epochs for this final training
        learning_rate=2e-5, # Learning rate
        weight_decay=0.01, # Weight decay
        save_strategy='no', # No checkpoints saved during this final training
        fp16=torch.cuda.is_available(), # Mixed precision
        report_to='none', # No reporting
    ),
    train_dataset=full_train_tokenized, # The tokenized full training dataset
    tokenizer=tokenizer, # The tokenizer
)

final_trainer.train() # Starts the final training process

# ===== 13. Predict Test Set =====
print("\nPredicting test set...")
# Makes predictions on the tokenized test dataset using the finally trained model.
predictions = final_trainer.predict(test_tokenized)
# Extracts the predicted class labels (0 or 1) from the raw prediction scores (logits).
pred_labels = np.argmax(predictions.predictions, axis=-1)

# ===== 14. Generate Submission File =====
# Creates a Pandas DataFrame for the submission file.
# It includes the 'id' from the original test data and the predicted 'sentiment' labels.
submission = pd.DataFrame({
    'id': test_df['id'],
    'sentiment': pred_labels
})

# Saves the submission DataFrame to a CSV file.
# `index=False` prevents Pandas from writing the DataFrame index as a column in the CSV,
# which is usually not required for Kaggle submissions.
submission.to_csv('deberta_submission.csv', index=False)
print("\nâœ… Submission file saved: deberta_submission.csv")
# Prints the distribution of the predicted sentiment labels in the submission file.
print(f"Prediction distribution: {pd.Series(pred_labels).value_counts().to_dict()}")




