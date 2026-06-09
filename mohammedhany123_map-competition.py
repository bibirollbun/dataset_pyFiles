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
import numpy as np
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    Trainer, 
    logging as hf_logging,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score, accuracy_score
import lightgbm as lgb
import warnings
import logging
import sys
import os
import joblib
from datetime import datetime
import json
warnings.filterwarnings('ignore')


hf_logging.set_verbosity_info()
hf_logging.enable_default_handler()
hf_logging.enable_explicit_format()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
category_encoder = LabelEncoder()
misconception_encoder = LabelEncoder()
feature_names = []


def apk(actual, predicted, k=10):
    """Computes the average precision at k."""
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """Computes the mean average precision at k."""
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def load_data(train_path):
    """Load and preprocess training data"""
    print("=== Loading Data ===")
    train_df = pd.read_csv(train_path)
    print(f"Loaded {len(train_df)} training samples")
    
    # Handle missing values
    train_df['StudentExplanation'] = train_df['StudentExplanation'].fillna('')
    train_df['QuestionText'] = train_df['QuestionText'].fillna('')
    train_df['MC_Answer'] = train_df['MC_Answer'].fillna('')
    train_df['Misconception'] = train_df['Misconception'].fillna('NA')
    
    return train_df


def extract_mathematical_features(text): 
    """Extract mathematical and linguistic features from text""" 
    features = {} 
     
    # Basic text features 
    features['length'] = len(str(text)) 
    features['word_count'] = len(str(text).split()) 
    features['sentence_count'] = len(str(text).split('.')) 
     
    # Mathematical content features 
    features['has_fraction'] = int(bool(re.search(r'\\frac|/', str(text)))) 
    features['has_decimal'] = int(bool(re.search(r'\d+\.\d+', str(text)))) 
    features['has_negative'] = int(bool(re.search(r'-\d+', str(text)))) 
    features['has_multiplication'] = int(bool(re.search(r'\*|Ã—|times', str(text)))) 
    features['has_division'] = int(bool(re.search(r'Ã·|div|/', str(text)))) 
    features['has_addition'] = int(bool(re.search(r'\+|plus|add', str(text)))) 
    features['has_subtraction'] = int(bool(re.search(r'-|minus|subtract', str(text)))) 
    features['has_equals'] = int(bool(re.search(r'=|equals', str(text)))) 
     
    # Count numbers 
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', str(text)) 
    features['number_count'] = len(numbers) 
     
    # Reasoning indicators 
    features['has_because'] = int('because' in str(text).lower()) 
    features['has_so'] = int('so' in str(text).lower()) 
    features['has_therefore'] = int('therefore' in str(text).lower()) 
    features['has_since'] = int('since' in str(text).lower()) 
     
    # Uncertainty indicators 
    features['has_think'] = int('think' in str(text).lower()) 
    features['has_maybe'] = int('maybe' in str(text).lower()) 
    features['has_probably'] = int('probably' in str(text).lower()) 
    features['has_guess'] = int('guess' in str(text).lower()) 
     
    # Common misconception keywords 
    features['mentions_bigger'] = int('bigger' in str(text).lower()) 
    features['mentions_smaller'] = int('smaller' in str(text).lower()) 
    features['mentions_more'] = int('more' in str(text).lower()) 
    features['mentions_less'] = int('less' in str(text).lower()) 
     
    return features 


def prepare_features(df, tfidf_vectorizer, question_encoder, answer_encoder):
    """Prepare all features for modeling""" 
    print("=== Feature Engineering ===") 
     
    # Extract mathematical features 
    math_features = [] 
    for explanation in train_df['StudentExplanation']: 
        math_features.append(extract_mathematical_features(explanation)) 
     
    math_features_df = pd.DataFrame(math_features) 
    print(f"Mathematical features created: {math_features_df.shape[1]}") 
     
    # TF-IDF features for text 
    tfidf_matrix = tfidf_vectorizer.transform(df['StudentExplanation'].astype(str))
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])])
    print(f"TF-IDF features created: {tfidf_df.shape[1]}") 
          
    question_features = pd.DataFrame({ 
        'question_encoded': question_encoder.transform(df['QuestionText'].astype(str)),
        'answer_encoded': answer_encoder.transform(df['MC_Answer'].astype(str)), 
        'question_length': df['QuestionText'].str.len(), 
        'answer_length': df['MC_Answer'].str.len() 
    }) 
    print(f"Question/Answer features created: {question_features.shape[1]}") 
     
    # Combine all features 
    features_df = pd.concat([math_features_df, tfidf_df, question_features], axis=1) 
    feature_names.extend(features_df.columns.tolist()) 
    print(f"Total features: {features_df.shape[1]}") 
     
    return features_df


def prepare_targets(train_df): 
    """Prepare target variables""" 
    print("=== Target Engineering ===") 
     
    # Parse category into correctness and misconception type 
    categories = train_df['Category'].str.split('_', expand=True) 
    correctness = categories[0].map({'True': 1, 'False': 0}) 
    category_type = categories[1]  # Correct, Misconception, Neither 
     
    # Encode targets 
    category_type_encoded = category_encoder.fit_transform(category_type) 
     
    # Handle misconceptions 
    misconceptions = train_df['Misconception'].fillna('NA') 
    misconception_encoded = misconception_encoder.fit_transform(misconceptions) 
     
    print(f"Correctness classes: {correctness.value_counts().to_dict()}") 
    print(f"Category type classes: {dict(zip(category_encoder.classes_, range(len(category_encoder.classes_))))}") 
    print(f"Misconception classes: {len(misconception_encoder.classes_)}") 
     
    return correctness, category_type_encoded, misconception_encoded


class ExplanationDataset(Dataset):
    """Custom dataset for student explanations"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }


def compute_metrics_with_map3(eval_pred, validation_data=None, encoders=None, task_type='single'):
    """
    Compute metrics including MAP@3 for evaluation
    
    Parameters
    ----------
    eval_pred : tuple
        Predictions and labels from the model
    validation_data : dict, optional
        Additional validation data needed for MAP@3 calculation
    encoders : dict, optional
        Label encoders for converting predictions back to original format
    task_type : str
        Type of task ('single' for individual classification, 'combined' for full pipeline)
    """
    predictions, labels = eval_pred
    
    # Standard metrics
    pred_classes = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, pred_classes)
    f1 = f1_score(labels, pred_classes, average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'f1': f1
    }
    
    # Add MAP@3 if we have the necessary data for combined evaluation
    if task_type == 'combined' and validation_data is not None and encoders is not None:
        # Get top-3 predictions
        top3_indices = np.argsort(predictions, axis=1)[:, -3:][:, ::-1]
        
        # Convert to MAP@3 format (this is simplified - you'd need full pipeline)
        predicted_for_map = []
        actual_for_map = []
        
        for i in range(len(labels)):
            # Convert predictions to format expected by MAP@3
            pred_list = [encoders['current_task'].inverse_transform([idx])[0] for idx in top3_indices[i]]
            actual_list = [encoders['current_task'].inverse_transform([labels[i]])[0]]
            
            predicted_for_map.append(pred_list)
            actual_for_map.append(actual_list)
        
        map3_score = mapk(actual_for_map, predicted_for_map, k=3)
        metrics['map3'] = map3_score
    
    return metrics


def train_roberta(texts, labels, num_classes, model_name='roberta-base', max_length=512):
    """Train RoBERTa model for classification"""
    
    print(f"Training RoBERTa with {len(texts)} samples, {num_classes} classes")
    
    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes
    ).to("cuda")
    
    # Create dataset
    dataset = ExplanationDataset(texts, labels, tokenizer, max_length)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Training arguments optimized for MAP@3
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5, 
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16, 
        warmup_steps=100, 
        weight_decay=0.01,
        learning_rate=1e-5,
        eval_strategy="steps", 
        eval_steps=100,
        logging_dir='./logs', 
        logging_steps=25,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        seed=42,
        report_to="none",
        disable_tqdm=False,
        log_level="info",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Evaluate the model
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"Validation Results: {eval_results}")
    
    return model, tokenizer


def predict_with_roberta(model, tokenizer, texts, device='cuda'):
    """Make predictions with trained RoBERTa model"""
    
    model.eval()
    model.to(device)
    
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                str(text),
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            ).to(device)
            
            outputs = model(**inputs)
            logits = outputs.logits
            
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(logits, dim=-1)
            
            predictions.append(pred.cpu().numpy()[0])
            probabilities.append(probs.cpu().numpy()[0])
    
    return np.array(predictions), np.array(probabilities)


def get_top_k_predictions(probabilities, k=3):
    """Get top-k predictions from probability arrays"""
    return np.argsort(probabilities, axis=1)[:, -k:][:, ::-1]


def predict_top3_roberta(models, encoders, val_df, X):
    """Generate top-3 predictions using trained models"""
    
    preds = []
    texts = val_df['StudentExplanation'].tolist()
    
    # Get predictions from each model
    print("Predicting lightGBM...")
    X_val = X.loc[val_df.index]
    corr_preds = models['correctness'].predict(X_val)

    print("Predicting Roberta (category)...")
    cat_preds, cat_probs = predict_with_roberta(
        RobertaForSequenceClassification.from_pretrained(models['category']), RobertaTokenizer.from_pretrained(models['category_tokenizer']), texts
    )

    print("Predicting Roberta (misconception)...")
    misc_preds, misc_probs = predict_with_roberta(
        RobertaForSequenceClassification.from_pretrained(models['misconception']), RobertaTokenizer.from_pretrained(models['misconception_tokenizer']), texts
    )
    
    # Get top-3 for each component
    top3_cat = get_top_k_predictions(cat_probs, k=3)
    top3_misc = get_top_k_predictions(misc_probs, k=3)
    
    for i in range(len(texts)):
        sample_preds = []
        
        # Combine predictions to create top-3 overall predictions
        correctness = 'True' if corr_preds[i] == 1 else 'False'
        
        # Create combinations of top predictions
        for cat_idx in top3_cat[i]:
            for misc_idx in top3_misc[i]:
                if len(sample_preds) >= 3:
                    break
                
                cat_name = encoders['category'].inverse_transform([cat_idx])[0]
                misc_name = encoders['misconception'].inverse_transform([misc_idx])[0]
                
                prediction = f"{correctness}_{cat_name}:{misc_name}"
                if prediction not in sample_preds:
                    sample_preds.append(prediction)
            
            if len(sample_preds) >= 3:
                break
        
        # Ensure we always have 3 predictions
        while len(sample_preds) < 3:
            sample_preds.append(sample_preds[0] if sample_preds else "False_Neither:NA")
        
        preds.append(sample_preds[:3])
    
    return preds


def save_category_model(model, tokenizer, encoder, output_dir="output", model_name="category_model"):
    """
    Save category model, tokenizer, and encoder to output directory
    
    Args:
        model: Trained category model (RoBERTa or similar)
        tokenizer: Model tokenizer
        encoder: Label encoder for categories
        output_dir: Directory to save files
        model_name: Base name for the model files
    
    Returns:
        dict: Paths of saved files
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = os.path.join(output_dir, f"{model_name}_{timestamp}")
    
    saved_files = {}
    
    try:
        # Save the model
        model_path = f"{base_path}_model"
        if hasattr(model, 'save_pretrained'):
            # For Hugging Face models
            model.save_pretrained(model_path)
            print(f"âœ… Category model saved to: {model_path}")
        else:
            # For other models (sklearn, etc.)
            model_file = f"{model_path}.joblib"
            joblib.dump(model, model_file)
            model_path = model_file
            print(f"âœ… Category model saved to: {model_path}")
        
        saved_files['model'] = model_path
        
        # Save the tokenizer
        tokenizer_path = f"{base_path}_tokenizer"
        if hasattr(tokenizer, 'save_pretrained'):
            tokenizer.save_pretrained(tokenizer_path)
            print(f"âœ… Category tokenizer saved to: {tokenizer_path}")
        else:
            tokenizer_file = f"{tokenizer_path}.joblib"
            joblib.dump(tokenizer, tokenizer_file)
            tokenizer_path = tokenizer_file
            print(f"âœ… Category tokenizer saved to: {tokenizer_path}")
        
        saved_files['tokenizer'] = tokenizer_path
        
        # Save the label encoder
        encoder_path = f"{base_path}_encoder.joblib"
        joblib.dump(encoder, encoder_path)
        saved_files['encoder'] = encoder_path
        print(f"âœ… Category encoder saved to: {encoder_path}")
        
        # Save metadata
        metadata = {
            'model_type': 'category',
            'timestamp': timestamp,
            'model_path': model_path,
            'tokenizer_path': tokenizer_path,
            'encoder_path': encoder_path,
            'classes': encoder.classes_.tolist() if hasattr(encoder, 'classes_') else None
        }
        
        metadata_path = f"{base_path}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_files['metadata'] = metadata_path
        print(f"âœ… Category metadata saved to: {metadata_path}")
        
        print(f"\nğŸ�‰ Category model successfully saved with timestamp: {timestamp}")
        return saved_files
        
    except Exception as e:
        print(f"â�Œ Error saving category model: {str(e)}")
        return None


def save_misconception_model(model, tokenizer, encoder, output_dir="output", model_name="misconception_model"):
    """
    Save misconception model, tokenizer, and encoder to output directory
    
    Args:
        model: Trained misconception model (RoBERTa or similar)
        tokenizer: Model tokenizer
        encoder: Label encoder for misconceptions
        output_dir: Directory to save files
        model_name: Base name for the model files
    
    Returns:
        dict: Paths of saved files
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = os.path.join(output_dir, f"{model_name}_{timestamp}")
    
    saved_files = {}
    
    try:
        # Save the model
        model_path = f"{base_path}_model"
        if hasattr(model, 'save_pretrained'):
            # For Hugging Face models
            model.save_pretrained(model_path)
            print(f"âœ… Misconception model saved to: {model_path}")
        else:
            # For other models (sklearn, etc.)
            model_file = f"{model_path}.joblib"
            joblib.dump(model, model_file)
            model_path = model_file
            print(f"âœ… Misconception model saved to: {model_path}")
        
        saved_files['model'] = model_path
        
        # Save the tokenizer
        tokenizer_path = f"{base_path}_tokenizer"
        if hasattr(tokenizer, 'save_pretrained'):
            tokenizer.save_pretrained(tokenizer_path)
            print(f"âœ… Misconception tokenizer saved to: {tokenizer_path}")
        else:
            tokenizer_file = f"{tokenizer_path}.joblib"
            joblib.dump(tokenizer, tokenizer_file)
            tokenizer_path = tokenizer_file
            print(f"âœ… Misconception tokenizer saved to: {tokenizer_path}")
        
        saved_files['tokenizer'] = tokenizer_path
        
        # Save the label encoder
        encoder_path = f"{base_path}_encoder.joblib"
        joblib.dump(encoder, encoder_path)
        saved_files['encoder'] = encoder_path
        print(f"âœ… Misconception encoder saved to: {encoder_path}")
        
        # Save metadata
        metadata = {
            'model_type': 'misconception',
            'timestamp': timestamp,
            'model_path': model_path,
            'tokenizer_path': tokenizer_path,
            'encoder_path': encoder_path,
            'classes': encoder.classes_.tolist() if hasattr(encoder, 'classes_') else None
        }
        
        metadata_path = f"{base_path}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_files['metadata'] = metadata_path
        print(f"âœ… Misconception metadata saved to: {metadata_path}")
        
        print(f"\nğŸ�‰ Misconception model successfully saved with timestamp: {timestamp}")
        return saved_files
        
    except Exception as e:
        print(f"â�Œ Error saving misconception model: {str(e)}")
        return None



def train_pipeline_with_map3_optimization(train_df):
    """Enhanced training pipeline optimized for MAP@3 performance"""
    
    correctness, category_type, misconceptions = prepare_targets(train_df)
    
    tfidf_vectorizer.fit(train_df['StudentExplanation'])
    question_encoder = LabelEncoder()
    answer_encoder = LabelEncoder()
    question_encoder.fit(train_df['QuestionText'])
    answer_encoder.fit(train_df['MC_Answer'])
    
    X = prepare_features(train_df, tfidf_vectorizer, question_encoder, answer_encoder)

    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
    joblib.dump(question_encoder, 'question_encoder.pkl')
    joblib.dump(answer_encoder, 'answer_encoder.pkl')
    
    # Split data
    train_df_split, val_df_split, y_corr_train, y_corr_val, y_cat_train, y_cat_val, y_misc_train, y_misc_val = train_test_split(
        train_df,
        correctness,
        category_type,
        misconceptions,
        test_size=0.2,
        random_state=42,
        stratify=correctness
    )
    
    # Initialize models and encoders
    models = {}
    encoders = {
        'category': category_encoder,
        'misconception': misconception_encoder
    }
    
    print("=== Training Models with MAP@3 Optimization ===")
    
    # 1. Train correctness model (LightGBM)
    print("Training correctness classifier...")
    X_train_corr = X.loc[train_df_split.index]
    X_val_corr = X.loc[val_df_split.index]
    
    lgb_correct = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    lgb_correct.fit(X_train_corr, y_corr_train)
    joblib.dump(lgb_correct, 'lgb_correct.pkl')
    print("Saved lightGBM model")
    models['correctness'] = lgb_correct

    models['category'] = "/kaggle/input/map-competition/output/category_model_20250724_171821_model/"
    models['category_tokenizer'] = "/kaggle/input/map-competition/output/category_model_20250724_171821_tokenizer/"

    models['misconception'] = "/kaggle/input/map-competition/output/misconception_model_20250724_195815_model/"
    models['misconception_tokenizer'] = "/kaggle/input/map-competition/output/misconception_model_20250724_195815_tokenizer/"
    
    # # 2. Train category classification
    # print("Training category classifier...")
    # models['category'], models['category_tokenizer'] = train_roberta(
    #     train_df_split['StudentExplanation'].tolist(),
    #     y_cat_train.tolist(),
    #     len(category_encoder.classes_)
    # )

    # category_files = save_category_model(
    #     model=models['category'],
    #     tokenizer=models['category_tokenizer'], 
    #     encoder=encoders['category']
    # )
    
    # # 3. Train misconception classification
    # print("Training misconception classifier...")
    # models['misconception'], models['misconception_tokenizer'] = train_roberta(
    #     train_df_split['StudentExplanation'].tolist(),
    #     y_misc_train.tolist(),
    #     len(misconception_encoder.classes_)
    # )

    # misconception_files = save_misconception_model(
    #     model=models['misconception'],
    #     tokenizer=models['misconception_tokenizer'],
    #     encoder=encoders['misconception']
    # )
    
    # print("=== Comprehensive Evaluation ===")
    
    # # Evaluate pipeline
    # evaluate_pipeline_map3(models, encoders, val_df_split, y_corr_val, y_cat_val, y_misc_val, X)
    
    return models, encoders


def evaluate_pipeline_map3(models, encoders, val_df, y_corr_val, y_cat_val, y_misc_val, X):
    """Comprehensive evaluation focused on MAP@3 performance"""
    
    texts = val_df['StudentExplanation'].tolist()
    
    print("\n--- Individual Model Performance ---")
    
    # Category model
    cat_preds, cat_probs = predict_with_roberta(
        models['category'], models['category_tokenizer'], texts
    )
    cat_f1 = f1_score(y_cat_val, cat_preds, average='weighted')
    print(f"Category F1: {cat_f1:.4f}")
    
    # Misconception model  
    misc_preds, misc_probs = predict_with_roberta(
        models['misconception'], models['misconception_tokenizer'], texts
    )
    misc_f1 = f1_score(y_misc_val, misc_preds, average='weighted')
    print(f"Misconception F1: {misc_f1:.4f}")

    y_corr_val = pd.Series(y_corr_val, index=val_df.index)
    y_cat_val = pd.Series(y_cat_val, index=val_df.index)
    y_misc_val = pd.Series(y_misc_val, index=val_df.index)
    
    # Combined MAP@3 evaluation
    print("\n--- MAP@3 Performance ---")
    
    # Generate true labels
    true_labels = []
    for i in val_df.index:
        c = 'True' if y_corr_val.loc[i] == 1 else 'False'
        cat = encoders['category'].inverse_transform([y_cat_val.loc[i]])[0]
        misc = encoders['misconception'].inverse_transform([y_misc_val.loc[i]])[0]
        true_labels.append([f"{c}_{cat}:{misc}"])
    
    # Generate top-3 predictions
    val_preds_top3 = predict_top3_roberta(models, encoders, val_df, X)
    
    # Calculate MAP@3
    map3_score = mapk(true_labels, val_preds_top3, k=3)
    print(f"ğŸ�¯ Final MAP@3 Score: {map3_score:.4f}")
    
    # Additional analysis
    print("\n--- Prediction Analysis ---")
    correct_at_1 = sum(1 for true, pred in zip(true_labels, val_preds_top3) if true[0] == pred[0])
    correct_at_3 = sum(1 for true, pred in zip(true_labels, val_preds_top3) if true[0] in pred)
    
    print(f"Accuracy @ 1: {correct_at_1/len(true_labels):.4f}")
    print(f"Accuracy @ 3: {correct_at_3/len(true_labels):.4f}")
    
    return map3_score


def make_test_predictions(models, encoders, test_df):
    """Make predictions on test data"""
    print("=== Making Test Predictions ===")
    
    # Prepare test features (same as training)

    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    question_encoder = joblib.load('question_encoder.pkl')
    answer_encoder = joblib.load('answer_encoder.pkl')

    X = prepare_features(test_df, tfidf_vectorizer, question_encoder, answer_encoder)
    
    test_preds_top3 = predict_top3_roberta(models, encoders, test_df, X)
    
    return test_preds_top3


def save_submission(predictions, df, output_path='submission.csv'):
    """Save predictions in submission format"""
    print(f"=== Saving Submission to {output_path} ===")
    
    submission_data = []
    for i, preds in enumerate(predictions):
        row_id = df.iloc[i]['row_id']
        # Join all 3 predictions into one string (space-separated)
        pred_str = " ".join(preds)
        category = f"{pred_str}"
        
        submission_data.append({
            'row_id': row_id,
            'Category': category
        })
    
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved with {len(submission_data)} predictions")


print("ğŸš€ Student Explanation Classification with MAP@3")
print("=" * 60)

# 1. Load data
train_df = load_data('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')

# 2. Train models
models, encoders = train_pipeline_with_map3_optimization(train_df)

# 3. Make test predictions (optional)
try:
    test_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
    test_df = test_df.fillna('')
    test_predictions = make_test_predictions(models, encoders, test_df)
    save_submission(test_predictions, test_df)
except FileNotFoundError:
    print("No test file found, skipping predictions")

print("\nâœ… Pipeline completed!")

