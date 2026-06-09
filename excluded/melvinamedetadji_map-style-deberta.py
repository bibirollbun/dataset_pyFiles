import os, sys, math, random, gc, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Any
import re
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    DataCollatorWithPadding, Trainer, TrainingArguments,
    EarlyStoppingCallback, get_cosine_schedule_with_warmup
)

warnings.filterwarnings('ignore')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Improved model selection with fallback options
MODEL_CANDIDATES = [
    "/kaggle/input/huggingfacedebartav3variants/transformers/default/1/huggingfacedebartav3variants/deberta-v3-small",
    "/kaggle/input/huggingfacedebartav3variants/transformers/default/1/huggingfacedebartav3variants/deberta-v3-xsmall",
    "/kaggle/input/huggingfacedebartav3variants/transformers/default/1/huggingfacedebartav3variants/deberta-v3-base",
    
]

def pick_first_existing(paths: List[str]):
    for p in paths:
        if Path(p).exists() or (not str(p).startswith("/kaggle") and not str(p).startswith("/")):
            return p
    return None

MODEL_NAME_OR_PATH = pick_first_existing(MODEL_CANDIDATES)
print(f"Using model: {MODEL_NAME_OR_PATH}")

# Enhanced hyperparameters
CONFIG = {
    'N_FOLDS': 3,
    'EPOCHS': 4,  # Increased for better convergence
    'LR': 1.5e-5,  # Slightly lower for stability
    'WARMUP_RATIO': 0.15,  # Increased warmup
    'BATCH_SIZE': 12,  # Optimized for memory
    'GRAD_ACCUM': 3,  # Effective batch size = 36
    'MAX_LEN': 320,  # Increased for better context
    'EVAL_STEPS': 150,
    'SAVE_STEPS': 150,
    'EARLY_STOP': 3,
    'WEIGHT_DECAY': 0.01,
    'DROPOUT': 0.15,  # Added regularization
    'LABEL_SMOOTHING': 0.1,  # Help with class imbalance
    'SCHEDULER': 'cosine',
    'FREEZE_LAYERS': 2,  # Freeze early layers for stability
}


def load_data():
    train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
    test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
    return train, test

def advanced_text_preprocessing(text):
    """Enhanced text preprocessing for math education domain"""
    if pd.isna(text):
        return ""
    
    # Convert to string and normalize
    text = str(text)
    
    # Fix common math notation issues
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', text)  # LaTeX fractions
    text = re.sub(r'\\left\(([^)]+)\\right\)', r'(\1)', text)  # LaTeX parentheses
    text = re.sub(r'\\([a-zA-Z]+)', r'\1', text)  # Remove LaTeX commands
    
    # Normalize mathematical expressions
    text = re.sub(r'\s*\/\s*', '/', text)  # Normalize fractions
    text = re.sub(r'\s*\+\s*', ' + ', text)  # Normalize addition
    text = re.sub(r'\s*-\s*', ' - ', text)  # Normalize subtraction
    text = re.sub(r'\s*\*\s*', ' * ', text)  # Normalize multiplication
    
    # Fix common student writing issues
    text = re.sub(r'\b(\d+)st\b', r'\1', text)  # Remove ordinal suffixes
    text = re.sub(r'\b(\d+)nd\b', r'\1', text)
    text = re.sub(r'\b(\d+)rd\b', r'\1', text)
    text = re.sub(r'\b(\d+)th\b', r'\1', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def enhanced_format_text(row):
    """Improved text formatting with better structure"""
    question = advanced_text_preprocessing(row['QuestionText'])
    answer = advanced_text_preprocessing(row['MC_Answer']) 
    explanation = advanced_text_preprocessing(row['StudentExplanation'])
    
    # More structured prompt for better model understanding
    formatted = (
        f"Math Question: {question}\n"
        f"Correct Answer: {answer}\n" 
        f"Student Response: {explanation}\n"
        f"Task: Classify the student's understanding and identify any misconceptions."
    )
    
    return formatted

def prepare_data():
    train, test = load_data()
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    
    # Handle missing values and create target
    train["Misconception"] = train["Misconception"].fillna("NA")
    train["target"] = train["Category"].astype(str) + ":" + train["Misconception"].astype(str)
    
    # Create label mappings
    labels = sorted(train["target"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    train["label"] = train["target"].map(label2id)
    
    n_classes = len(labels)
    print(f"Number of classes: {n_classes}")
    
    # Enhanced text formatting
    train["text"] = train.apply(enhanced_format_text, axis=1)
    test["text"] = test.apply(enhanced_format_text, axis=1)
    
    # Class distribution analysis
    class_counts = train["label"].value_counts().sort_index()
    print(f"Class distribution - Min: {class_counts.min()}, Max: {class_counts.max()}, Mean: {class_counts.mean():.1f}")
    
    return train, test, labels, label2id, id2label, n_classes


class EnhancedTextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, has_label=True, augment=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.has_label = has_label
        self.augment = augment
        
    def __len__(self):
        return len(self.df)
    
    def simple_augment(self, text):
        """Simple text augmentation for training"""
        if random.random() < 0.3:  # 30% chance
            # Randomly swap adjacent words occasionally
            words = text.split()
            if len(words) > 3:
                idx = random.randint(0, len(words) - 2)
                words[idx], words[idx + 1] = words[idx + 1], words[idx]
                text = " ".join(words)
        return text
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["text"]
        
        if self.augment and self.has_label:
            text = self.simple_augment(text)
            
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors=None,
            add_special_tokens=True,
        )
        
        if self.has_label:
            encoding["labels"] = row["label"]
            
        return encoding

class ImprovedDebertaModel(nn.Module):
    """Enhanced model with better regularization"""
    def __init__(self, model_name, num_labels, config):
        super().__init__()
        self.config = config
        self.deberta = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            hidden_dropout_prob=CONFIG['DROPOUT'],
            attention_probs_dropout_prob=CONFIG['DROPOUT'],
        )
        
        # Freeze early layers for stability
        if CONFIG['FREEZE_LAYERS'] > 0:
            for i, layer in enumerate(self.deberta.deberta.encoder.layer):
                if i < CONFIG['FREEZE_LAYERS']:
                    for param in layer.parameters():
                        param.requires_grad = False
    
    def forward(self, input_ids, attention_mask, labels=None):
        return self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )


def map_at_k_from_logits(logits: np.ndarray, labels: np.ndarray, k: int = 3) -> float:
    """Optimized MAP@K calculation"""
    if len(logits.shape) == 1:
        logits = logits.reshape(1, -1)
    if len(labels.shape) == 0:
        labels = np.array([labels])
        
    top_k = np.argsort(-logits, axis=1)[:, :k]
    scores = np.zeros(len(labels))
    
    for i in range(len(labels)):
        true_label = labels[i]
        predictions = top_k[i]
        
        for rank, pred in enumerate(predictions):
            if pred == true_label:
                scores[i] = 1.0 / (rank + 1)
                break
                
    return np.mean(scores)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
        
    logits = np.array(logits)
    labels = np.array(labels)
    
    map3 = map_at_k_from_logits(logits, labels, k=3)
    predictions = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, predictions)
    
    return {
        "map@3": map3,
        "accuracy": accuracy,
        "eval_map3": map3  # For early stopping
    }

class CustomTrainer(Trainer):
    """Enhanced trainer with label smoothing and custom loss"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        # Apply label smoothing
        if CONFIG['LABEL_SMOOTHING'] > 0 and self.model.training:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=CONFIG['LABEL_SMOOTHING'])
            loss = loss_fct(logits, labels)
        else:
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            
        return (loss, outputs) if return_outputs else loss

def create_enhanced_trainer(model, tokenizer, train_dataset, val_dataset, output_dir):
    """Create trainer with enhanced configuration"""
    
    steps_per_epoch = len(train_dataset) // (CONFIG['BATCH_SIZE'] * CONFIG['GRAD_ACCUM'])
    total_steps = steps_per_epoch * CONFIG['EPOCHS']
    warmup_steps = int(total_steps * CONFIG['WARMUP_RATIO'])
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=CONFIG['EPOCHS'],
        per_device_train_batch_size=CONFIG['BATCH_SIZE'],
        per_device_eval_batch_size=CONFIG['BATCH_SIZE'] * 2,
        gradient_accumulation_steps=CONFIG['GRAD_ACCUM'],
        learning_rate=CONFIG['LR'],
        weight_decay=CONFIG['WEIGHT_DECAY'],
        warmup_steps=warmup_steps,
        
        # Evaluation & saving
        eval_strategy="steps",
        eval_steps=CONFIG['EVAL_STEPS'],
        save_strategy="steps", 
        save_steps=CONFIG['SAVE_STEPS'],
        save_total_limit=2,
        
        # Optimization
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        group_by_length=True,
        
        # Logging & monitoring
        logging_steps=50,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_map3",
        greater_is_better=True,
        
        # Regularization
        label_smoothing_factor=CONFIG['LABEL_SMOOTHING'],
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,  # Updated for newer transformers
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=CONFIG['EARLY_STOP'])]
    )
    
    return trainer


import gc
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from sklearn.model_selection import StratifiedKFold

def main():
    # Load and prepare data
    train, test, labels, label2id, id2label, n_classes = prepare_data()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Enhanced cross-validation with stratification
    min_count = train["label"].value_counts().min()
    n_folds = min(CONFIG['N_FOLDS'], max(2, min_count))
    
    if n_folds != CONFIG['N_FOLDS']:
        print(f"[INFO] Reduced n_splits from {CONFIG['N_FOLDS']} to {n_folds} due to rare classes")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Storage for predictions
    oof_predictions = np.zeros((len(train), n_classes), dtype=np.float32)
    test_predictions = []
    fold_scores = []
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(train, train["label"])):
        print(f"\n{'='*20} FOLD {fold+1}/{n_folds} {'='*20}")
        
        # Split data
        train_fold = train.iloc[train_idx].reset_index(drop=True)
        val_fold = train.iloc[val_idx].reset_index(drop=True)
        
        # Create datasets with augmentation for training
        train_dataset = EnhancedTextDataset(
            train_fold, tokenizer, CONFIG['MAX_LEN'], 
            has_label=True, augment=True
        )
        val_dataset = EnhancedTextDataset(
            val_fold, tokenizer, CONFIG['MAX_LEN'],
            has_label=True, augment=False
        )
        
        # Model configuration
        config = AutoConfig.from_pretrained(
            MODEL_NAME_OR_PATH,
            num_labels=n_classes,
            id2label=id2label,
            label2id=label2id,
            hidden_dropout_prob=CONFIG['DROPOUT'],
            attention_probs_dropout_prob=CONFIG['DROPOUT'],
        )
        
        # Initialize model
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME_OR_PATH,
            config=config,
            ignore_mismatched_sizes=True
        )
        
        # Apply layer freezing if specified
        if CONFIG.get('FREEZE_LAYERS', 0) > 0:
            if hasattr(model, 'deberta') and hasattr(model.deberta, 'encoder'):
                for i, layer in enumerate(model.deberta.encoder.layer[:CONFIG['FREEZE_LAYERS']]):
                    for param in layer.parameters():
                        param.requires_grad = False
                    
        model.to(device)
        
        # Create trainer
        trainer = create_enhanced_trainer(
            model, tokenizer, train_dataset, val_dataset, f"./fold_{fold}"
        )
        
        # Train model
        print(f"Training fold {fold+1}...")
        trainer.train()
        
        # Evaluate and get OOF predictions
        val_predictions = trainer.predict(val_dataset)
        oof_predictions[val_idx] = val_predictions.predictions
        
        # Calculate fold score
        fold_score = map_at_k_from_logits(
            val_predictions.predictions, 
            val_fold["label"].values
        )
        fold_scores.append(fold_score)
        print(f"Fold {fold+1} MAP@3: {fold_score:.6f}")
        
        # Test predictions
        test_dataset = EnhancedTextDataset(
            test, tokenizer, CONFIG['MAX_LEN'], 
            has_label=False, augment=False
        )
        test_pred = trainer.predict(test_dataset)
        test_predictions.append(test_pred.predictions)
        
        # Cleanup
        del model, trainer, train_dataset, val_dataset, test_dataset
        gc.collect()
        torch.cuda.empty_cache()
    
    # Calculate final scores
    oof_score = map_at_k_from_logits(oof_predictions, train["label"].values)
    mean_cv_score = np.mean(fold_scores)
    std_cv_score = np.std(fold_scores)
    
    print(f"\n{'='*50}")
    print(f"OOF MAP@3: {oof_score:.6f}")
    print(f"CV MAP@3: {mean_cv_score:.6f} ± {std_cv_score:.6f}")
    print(f"Individual fold scores: {[f'{s:.6f}' for s in fold_scores]}")
    
    # Generate submission
    final_test_predictions = np.mean(test_predictions, axis=0)
    top3_indices = np.argsort(-final_test_predictions, axis=1)[:, :3]
    top3_labels = [[id2label[idx] for idx in indices] for indices in top3_indices]
    
    submission = pd.DataFrame({
        "row_id": test["row_id"].values,
        "Category:Misconception": [" ".join(labels) for labels in top3_labels]
    })
    
    submission.to_csv("submission.csv", index=False)
    print(f"\nSubmission saved to submission.csv")
    print(f"Shape: {submission.shape}")
    print(submission.head())
    
    return submission, oof_score, fold_scores

if __name__ == "__main__":
    submission, oof_score, fold_scores = main()

