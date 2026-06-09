import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import torch
from torch.utils.data import Dataset
from scipy.stats import mode

MODEL_NAME = "tbs17/MathBERT"  
MAX_LENGTH = 256
BATCH_SIZE = 16


train_df = pd.read_csv('/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv')
test_df = pd.read_csv('/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv')
train_df['label'].value_counts()


CLASS_NAMES = [
    "Algebra",
    "Geometry and Trigonometry",
    "Calculus and Analysis",
    "Probability and Statistics",
    "Number Theory",
    "Combinatorics and Discrete Math",
    "Linear Algebra",
    "Abstract Algebra and Topology"
]

MATH_STOP_WORDS = {'find', 'prove', 'show', 'calculate', 'determine', 'let', 'given', 'solve'}


def math_text_preprocessor(text):
    """Preprocessing for TF-IDF model"""
    text = re.sub(r'\$(.*?)\$', r' MATH_EXPR \1 MATH_EXPR ', text)
    text = re.sub(r'\\\w+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' NUM ', text)
    return text.lower().strip()


def load_data():
    """Load and preprocess data for both models"""
    train_df = pd.read_csv('/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv')  
    test_df = pd.read_csv('/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv')    
    
    # BERT preprocessing
    def clean_math_text(text):
        text = re.sub(r'\$(.*?)\$', r' [MATH] \1 [MATH] ', text)
        text = re.sub(r'\\\w+', lambda m: ' ' + m.group(0) + ' ', text)
        return text.strip()
    
    # TF-IDF preprocessing
    train_df['cleaned_bert'] = train_df['Question'].apply(clean_math_text)
    test_df['cleaned_bert'] = test_df['Question'].apply(clean_math_text)
    train_df['cleaned_tfidf'] = train_df['Question'].apply(math_text_preprocessor)
    test_df['cleaned_tfidf'] = test_df['Question'].apply(math_text_preprocessor)
    
    train_df.drop(columns=['Question'], inplace=True)
    test_df.drop(columns=['Question'], inplace=True)
    
    return train_df, test_df


class MathDataset(Dataset):
    """Dataset class for MathBERT"""
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx])
        }


def train_mathbert():
    """Main training function with ensemble prediction"""
    train_df, test_df = load_data()
    
    # ========== TF-IDF Model Training ==========
    print("\nTraining TF-IDF classifier...")
    tfidf_pipe = make_pipeline(
        TfidfVectorizer(
            stop_words=list(MATH_STOP_WORDS),
            ngram_range=(1, 2),
            max_features=25000,
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
            analyzer='word',
            token_pattern=r'\b[^\d\W]+\b'
        ),
        FunctionTransformer(lambda x: x.tocsc()),
        LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    )
    
    tfidf_pipe.fit(train_df['cleaned_tfidf'], train_df['label'])
    tfidf_preds = tfidf_pipe.predict(test_df['cleaned_tfidf'])
    print("TF-IDF predictions sample:", tfidf_preds[:5])
    print("TF-IDF class distribution:", np.bincount(tfidf_preds))

    # ========== MathBERT Training ==========
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[MATH]']})
    
    test_dataset = MathDataset(test_df['cleaned_bert'].tolist(), 
                              [0]*len(test_df), tokenizer)
    
    # Cross-validation setup
    N_SPLITS = 3
    skf = StratifiedKFold(n_splits=N_SPLITS)
    all_bert_preds = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(
        train_df['cleaned_bert'], train_df['label']
    )):
        print(f"\nTraining Fold {fold+1}/{N_SPLITS}")
        
        # Model initialization
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=8,
            ignore_mismatched_sizes=True
        )
        model.resize_token_embeddings(len(tokenizer))
        
        # Training arguments
        args = TrainingArguments(
            output_dir=f'./fold_{fold}',
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            logging_strategy='steps',  # Add this
            logging_steps=50,         # More frequent logging
            learning_rate=2e-5,
            fp16=True,
            warmup_ratio=0.1,
            weight_decay=0.01,
            seed=42,
            load_best_model_at_end=True,
            metric_for_best_model='f1_micro',
            report_to='none',
            disable_tqdm=False,  # Ensure progress bars are enabled
        )
        
        # Trainer setup
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=MathDataset(
                train_df.iloc[train_idx]['cleaned_bert'].tolist(),
                train_df.iloc[train_idx]['label'].values,
                tokenizer
            ),
            eval_dataset=MathDataset(
                train_df.iloc[val_idx]['cleaned_bert'].tolist(),
                train_df.iloc[val_idx]['label'].values,
                tokenizer
            ),
            compute_metrics=lambda p: {
                'f1_micro': f1_score(p.label_ids, p.predictions.argmax(-1), average='micro')
            }
        )
        
        # Training and prediction
        trainer.train()
        fold_preds = trainer.predict(test_dataset).predictions.argmax(-1)
        all_bert_preds.append(fold_preds)
        
        del model
        torch.cuda.empty_cache()

    # ========== Ensemble Predictions ==========
    # Combine BERT predictions with TF-IDF predictions
    all_preds = np.vstack([np.array(all_bert_preds), tfidf_preds])
    
    # Calculate mode across all predictions
    final_preds, _ = mode(all_preds, axis=0)
    final_preds = final_preds.flatten().astype(int)
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_df['id'].values,
        'label': final_preds
    })
    submission.to_csv('submission.csv', index=False)
    print("\nEnsemble submission saved!")


train_mathbert()


submission = pd.read_csv('/kaggle/working/ensemble_submission.csv')
submission




