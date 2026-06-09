# надо сделать чтобы числа были в диапазоне [0, 1]

import pandas as pd
import numpy as np
import torch
import time
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

# Configuration
ROBERTA_PATH = '/kaggle/input/berta-and-stella/pytorch/default/1/bert_models/roberta-large'
BGE_PATH = '/kaggle/input/berta-and-stella/pytorch/default/1/stella_bge_models/bge-large-en-v1.5'
BATCH_SIZE = 8
MAX_LENGTH = 256
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
SEED = 42

# Set seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# Disable wandb completely
os.environ['WANDB_DISABLED'] = 'true'

class RedditCommentDataset(Dataset):
    def __init__(self, texts, rules, subreddits, labels=None, tokenizer=None, model_type='roberta'):
        self.texts = texts
        self.rules = rules
        self.subreddits = subreddits
        self.labels = labels
        self.tokenizer = tokenizer
        self.model_type = model_type
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        rule = str(self.rules[idx])
        subreddit = str(self.subreddits[idx])
        
        # Create input text
        input_text = f"Rule: {rule}. Subreddit: {subreddit}. Comment: {text}"
        
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
            
        return item

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
    auc = roc_auc_score(labels, predictions)
    return {'auc': auc}

# Custom callback for time tracking
class TimeTrackingCallback(EarlyStoppingCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.time()
        self.epoch_start_time = None
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        current_epoch = state.epoch if hasattr(state, 'epoch') else 0
        total_epochs = args.num_train_epochs
        elapsed = time.time() - self.start_time
        estimated_total = elapsed * total_epochs / max(current_epoch, 1)
        remaining = estimated_total - elapsed
        completion_time = datetime.now() + timedelta(seconds=remaining)
        
        print(f"\nEpoch {int(current_epoch) + 1}/{int(total_epochs)}")
        print(f"Elapsed: {timedelta(seconds=int(elapsed))}")
        print(f"Estimated remaining: {timedelta(seconds=int(remaining))}")
        print(f"Estimated completion: {completion_time.strftime('%H:%M:%S')}")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            current_step = state.global_step if hasattr(state, 'global_step') else 0
            total_steps = state.max_steps if hasattr(state, 'max_steps') else 0
            if total_steps > 0 and current_step > 0:
                progress = current_step / total_steps * 100
                elapsed = time.time() - self.start_time
                time_per_step = elapsed / current_step
                remaining_steps = total_steps - current_step
                remaining_time = remaining_steps * time_per_step
                print(f"Progress: {progress:.1f}% | Remaining: {timedelta(seconds=int(remaining_time))}", end='\r')

def train_roberta_model(train_dataset, val_dataset):
    """Train RoBERTa model with correct output layer"""
    print("Training RoBERTa model...")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load tokenizer and model from local path
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_PATH, local_files_only=True)
    
    # Load base model
    base_model = AutoModel.from_pretrained(ROBERTA_PATH, local_files_only=True)
    
    # Create custom model
    class CustomRobertaForSequenceClassification(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.roberta = base_model
            self.dropout = torch.nn.Dropout(0.1)
            self.classifier = torch.nn.Linear(base_model.config.hidden_size, 1)
            
        def forward(self, input_ids, attention_mask, labels=None):
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            
            loss = None
            if labels is not None:
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            
            return (loss, logits) if loss is not None else logits
    
    model = CustomRobertaForSequenceClassification(base_model)
    
    # Calculate estimated training time
    total_steps = len(train_dataset) * NUM_EPOCHS / BATCH_SIZE
    estimated_time_per_step = 0.5
    estimated_total_seconds = total_steps * estimated_time_per_step
    estimated_end_time = datetime.now() + timedelta(seconds=estimated_total_seconds)
    
    print(f"Estimated training time: {timedelta(seconds=int(estimated_total_seconds))}")
    print(f"Estimated completion time: {estimated_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)
    
    training_args = TrainingArguments(
        output_dir='./roberta_results',
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",  # Исправлено на старую версию
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        greater_is_better=True,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        report_to=None,  # Отключаем все логгеры
        logging_steps=100,
        save_total_limit=2,
        logging_dir='./roberta_logs',
        logging_first_step=True,
        prediction_loss_only=False,
        dataloader_pin_memory=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[TimeTrackingCallback(early_stopping_patience=1)]
    )
    
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    total_time = end_time - start_time
    print(f"\n{'='*50}")
    print(f"RoBERTa training completed!")
    print(f"Total training time: {timedelta(seconds=int(total_time))}")
    print(f"Average time per epoch: {timedelta(seconds=int(total_time/NUM_EPOCHS))}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")
    
    return trainer, tokenizer

def train_bge_model(train_dataset, val_dataset):
    """Train BGE model"""
    print("Training BGE model...")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load tokenizer and model from local path
    tokenizer = AutoTokenizer.from_pretrained(BGE_PATH, local_files_only=True)
    
    # Load base model
    base_model = AutoModel.from_pretrained(BGE_PATH, local_files_only=True)
    
    # Create custom model
    class CustomBGEForSequenceClassification(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.bge = base_model
            self.dropout = torch.nn.Dropout(0.1)
            self.classifier = torch.nn.Linear(base_model.config.hidden_size, 1)
            
        def forward(self, input_ids, attention_mask, labels=None):
            outputs = self.bge(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            
            loss = None
            if labels is not None:
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            
            return (loss, logits) if loss is not None else logits
    
    model = CustomBGEForSequenceClassification(base_model)
    
    # Calculate estimated training time
    total_steps = len(train_dataset) * NUM_EPOCHS / BATCH_SIZE
    estimated_time_per_step = 0.5
    estimated_total_seconds = total_steps * estimated_time_per_step
    estimated_end_time = datetime.now() + timedelta(seconds=estimated_total_seconds)
    
    print(f"Estimated training time: {timedelta(seconds=int(estimated_total_seconds))}")
    print(f"Estimated completion time: {estimated_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)
    
    training_args = TrainingArguments(
        output_dir='./bge_results',
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",  # Исправлено на старую версию
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        greater_is_better=True,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        report_to=None,  # Отключаем все логгеры
        logging_steps=100,
        save_total_limit=2,
        logging_dir='./bge_logs',
        logging_first_step=True,
        prediction_loss_only=False,
        dataloader_pin_memory=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[TimeTrackingCallback(early_stopping_patience=1)]
    )
    
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    total_time = end_time - start_time
    print(f"\n{'='*50}")
    print(f"BGE training completed!")
    print(f"Total training time: {timedelta(seconds=int(total_time))}")
    print(f"Average time per epoch: {timedelta(seconds=int(total_time/NUM_EPOCHS))}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")
    
    return trainer, tokenizer

def predict_ensemble(models_tokenizers, test_datasets):
    """Make predictions using ensemble of models"""
    all_predictions = []
    
    for (trainer, tokenizer), test_dataset in zip(models_tokenizers, test_datasets):
        predictions = trainer.predict(test_dataset)
        pred_probs = torch.sigmoid(torch.tensor(predictions.predictions)).numpy()
        all_predictions.append(pred_probs)
    
    # Simple average ensemble
    ensemble_predictions = np.mean(all_predictions, axis=0)
    return ensemble_predictions.flatten()

def augment_with_examples(train_df):
    """Augment training data with positive and negative examples"""
    augmented_rows = []
    
    for _, row in train_df.iterrows():
        # Original row
        augmented_rows.append(row.to_dict())
        
        # Add positive examples
        for i in range(1, 3):
            example_col = f'positive_example_{i}'
            if example_col in row and pd.notna(row[example_col]):
                new_row = row.copy()
                new_row['body'] = row[example_col]
                new_row['rule_violation'] = 1.0
                augmented_rows.append(new_row)
        
        # Add negative examples
        for i in range(1, 3):
            example_col = f'negative_example_{i}'
            if example_col in row and pd.notna(row[example_col]):
                new_row = row.copy()
                new_row['body'] = row[example_col]
                new_row['rule_violation'] = 0.0
                augmented_rows.append(new_row)
    
    return pd.DataFrame(augmented_rows)

def main():
    # Disable wandb completely
    os.environ['WANDB_DISABLED'] = 'true'
    
    # Load data
    train_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
    test_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Augment training data with examples
    print("Augmenting training data...")
    train_df_augmented = augment_with_examples(train_df)
    print(f"Augmented training data shape: {train_df_augmented.shape}")
    
    # Prepare training and validation split
    train_data, val_data = train_test_split(
        train_df_augmented, 
        test_size=0.2, 
        random_state=SEED, 
        stratify=train_df_augmented['rule_violation']
    )
    
    print(f"Train data: {train_data.shape}, Validation data: {val_data.shape}")
    
    models_tokenizers = []
    test_datasets = []
    
    # Train RoBERTa model
    print("\n" + "="*50)
    roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_PATH, local_files_only=True)
    roberta_train_dataset = RedditCommentDataset(
        train_data['body'].values,
        train_data['rule'].values,
        train_data['subreddit'].values,
        train_data['rule_violation'].values,
        roberta_tokenizer,
        'roberta'
    )
    roberta_val_dataset = RedditCommentDataset(
        val_data['body'].values,
        val_data['rule'].values,
        val_data['subreddit'].values,
        val_data['rule_violation'].values,
        roberta_tokenizer,
        'roberta'
    )
    roberta_trainer, roberta_tokenizer = train_roberta_model(roberta_train_dataset, roberta_val_dataset)
    models_tokenizers.append((roberta_trainer, roberta_tokenizer))
    
    # Prepare RoBERTa test dataset
    roberta_test_dataset = RedditCommentDataset(
        test_df['body'].values,
        test_df['rule'].values,
        test_df['subreddit'].values,
        None,
        roberta_tokenizer,
        'roberta'
    )
    test_datasets.append(roberta_test_dataset)
    
    # Train BGE model
    print("\n" + "="*50)
    bge_tokenizer = AutoTokenizer.from_pretrained(BGE_PATH, local_files_only=True)
    bge_train_dataset = RedditCommentDataset(
        train_data['body'].values,
        train_data['rule'].values,
        train_data['subreddit'].values,
        train_data['rule_violation'].values,
        bge_tokenizer,
        'bge'
    )
    bge_val_dataset = RedditCommentDataset(
        val_data['body'].values,
        val_data['rule'].values,
        val_data['subreddit'].values,
        val_data['rule_violation'].values,
        bge_tokenizer,
        'bge'
    )
    bge_trainer, bge_tokenizer = train_bge_model(bge_train_dataset, bge_val_dataset)
    models_tokenizers.append((bge_trainer, bge_tokenizer))
    
    # Prepare BGE test dataset
    bge_test_dataset = RedditCommentDataset(
        test_df['body'].values,
        test_df['rule'].values,
        test_df['subreddit'].values,
        None,
        bge_tokenizer,
        'bge'
    )
    test_datasets.append(bge_test_dataset)
    
    # Make ensemble predictions
    print("\nMaking ensemble predictions...")
    predictions = predict_ensemble(models_tokenizers, test_datasets)
    
    # Create submission
    submission = pd.DataFrame({
        'row_id': test_df['row_id'],
        'rule_violation': predictions
    })
    
    submission.to_csv('submission.csv', index=False)
    print("Submission file created!")
    
    # Show some predictions
    print("\nSample predictions:")
    print(submission.head(10))

if __name__ == "__main__":
    main()







