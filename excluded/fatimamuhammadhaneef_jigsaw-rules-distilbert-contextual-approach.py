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
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import more_itertools
import logging
import numpy as np
from sklearn.metrics import roc_auc_score

# Configure logging for monitoring training progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Optimize device usage for maximum performance
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Load pre-trained DistilBERT model and tokenizer
try:
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2).to(device)
    logger.info("Model and tokenizer loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or tokenizer: {e}")
    raise


def create_input(row: pd.Series) -> str:
    """
    Create structured input format combining rule context with comment.
    This format provides the model with comprehensive context for better classification.
    """
    return f"""[Subreddit]: {row['subreddit']}
[Rule]: {row['rule']}
[Positive Example 1 (Violates)]: {row['positive_example_1']}
[Positive Example 2 (Violates)]: {row['positive_example_2']}
[Negative Example 1 (Does Not Violate)]: {row['negative_example_1']}
[Negative Example 2 (Does Not Violate)]: {row['negative_example_2']}
[Comment]: {row['body']}"""


# Load and preprocess training dataset
try:
    train_data = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
    logger.info(f"Training data loaded successfully - Shape: {train_data.shape}")
except Exception as e:
    logger.error(f"Error loading training data: {e}")
    raise

# Create formatted inputs and ensure proper label encoding
train_data['text'] = train_data.apply(create_input, axis=1)
train_data['label'] = train_data['rule_violation'].astype(int)
train_dataset = Dataset.from_pandas(train_data[['text', 'label']])

logger.info(f"Training samples prepared: {len(train_dataset)}")


def tokenize_function(examples):
    """Tokenize text inputs with optimal parameters for DistilBERT"""
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

try:
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    logger.info("Training data tokenized successfully")
except Exception as e:
    logger.error(f"Error tokenizing training data: {e}")
    raise


# Optimized training arguments for performance
training_args = TrainingArguments(
    output_dir='/kaggle/working/results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='/kaggle/working/logs',
    logging_steps=10,
    eval_strategy="no",
    save_strategy="epoch",
    load_best_model_at_end=False,
    report_to="none",
    dataloader_pin_memory=False,
    remove_unused_columns=False,
)

def compute_metrics(eval_pred):
    """Calculate AUC-ROC for performance monitoring"""
    logits, labels = eval_pred
    probabilities = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    return {'auc_roc': roc_auc_score(labels, probabilities)}

# Initialize trainer with optimized configuration
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    compute_metrics=compute_metrics
)


# Execute fine-tuning process
try:
    trainer.train()
    logger.info("Model fine-tuning completed successfully")
    
    # Save fine-tuned model for potential future use
    model.save_pretrained('/kaggle/working/fine_tuned_distilbert')
    tokenizer.save_pretrained('/kaggle/working/fine_tuned_distilbert')
    logger.info("Fine-tuned model and tokenizer saved")
except Exception as e:
    logger.error(f"Error during fine-tuning: {e}")
    raise


# Load test dataset for inference
try:
    test_data = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
    logger.info(f"Test data loaded successfully - Shape: {test_data.shape}")
except Exception as e:
    logger.error(f"Error loading test data: {e}")
    raise


# Process test data in optimized batches
responses = []
batch_size = 8

# Set model to evaluation mode for optimal inference
model.eval()

for batch in more_itertools.batched(test_data.iterrows(), batch_size):
    inputs = [create_input(row[1]) for row in batch]
    try:
        # Tokenize batch inputs
        encodings = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        
        # Generate predictions with gradient computation disabled
        with torch.no_grad():
            outputs = model(**encodings)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            true_probs = probabilities[:, 1].cpu().numpy().tolist()
            responses.extend(true_probs)
            logger.info(f"Processed batch of size {len(inputs)}")
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        # Use neutral probability as fallback
        responses.extend([0.5] * len(inputs))


# Apply probability clipping to prevent extreme values
responses = np.clip(responses, 0.01, 0.99)

# Create competition submission format
submission = pd.DataFrame({
    'row_id': test_data['row_id'],
    'rule_violation': responses
})

# Save final submission file
try:
    submission.to_csv('/kaggle/working/submission.csv', index=False)
    logger.info("Submission file saved successfully")
    logger.info(f"Submission shape: {submission.shape}")
    logger.info(f"Submission preview:\n{submission.head()}")
    logger.info(f"Prediction statistics - Mean: {np.mean(responses):.4f}, Std: {np.std(responses):.4f}")
except Exception as e:
    logger.error(f"Error saving submission: {e}")
    raise

print(" Pipeline completed successfully!")

