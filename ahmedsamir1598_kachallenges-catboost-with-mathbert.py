import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.decomposition import TruncatedSVD
from catboost import CatBoostClassifier, Pool
from scipy.sparse import hstack, csr_matrix


def math_text_preprocessor(text):
    # Preserve mathematical expressions
    text = re.sub(r'\$(.*?)\$', r' MATH_EXPR \1 MATH_EXPR ', text)
    # Capture LaTeX commands
    text = re.sub(r'\\\w+', lambda m: ' LATEX_' + m.group(0)[1:] + ' ', text)
    # Normalize numbers
    text = re.sub(r'\d+', ' NUM ', text)
    # Remove non-essential punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.lower().strip()

def create_meta_features(texts):
    features = []
    for text in texts:
        # Math-specific features
        num_math_expr = text.count('MATH_EXPR')
        num_latex_cmds = text.count('LATEX_')
        num_numbers = text.count('NUM')
        text_length = len(text.split())
        
        features.append([
            num_math_expr,
            num_latex_cmds,
            num_numbers,
            text_length
        ])
    return np.array(features)


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

train_df = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv")  
test_df = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv")    

# Preprocess text
X_train = train_df["Question"].apply(math_text_preprocessor).tolist() 
X_test = test_df["Question"].apply(math_text_preprocessor).tolist() 
y_train = train_df["label"].values

def generate_mathbert_embeddings(texts, batch_size=16):
    """Generate contextual embeddings using MathBERT with GPU acceleration"""
    model_name = "tbs17/MathBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to('cuda')
    
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch, 
            padding=True, 
            truncation=True, 
            max_length=256,
            return_tensors="pt"
        ).to('cuda')
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        embeddings.append(batch_embeddings)
    
    return np.concatenate(embeddings)

# Generate embeddings for all data
train_embeddings = generate_mathbert_embeddings(X_train)
test_embeddings = generate_mathbert_embeddings(X_test)

# TF-IDF with math-aware parameters
tfidf = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=25000,
    min_df=2,
    max_df=0.85,
    stop_words=['find', 'prove', 'show', 'calculate', 'determine'],
    token_pattern=r'\b[^\d\W]+\b'
)

# Generate features
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

def reduce_dimensionality(train_vec, test_vec, n_components=64):
    """Efficient dimensionality reduction preserving 95% variance"""
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    train_reduced = svd.fit_transform(train_vec)
    test_reduced = svd.transform(test_vec)
    return train_reduced, test_reduced

# Reduce TF-IDF dimensions
#X_train_tfidf_reduced, X_test_tfidf_reduced = reduce_dimensionality(
#    X_train_tfidf, X_test_tfidf, n_components=128
#)

# Reduce MathBERT dimensions
#X_train_mathbert_reduced, X_test_mathbert_reduced = reduce_dimensionality(
#    train_embeddings, test_embeddings, n_components=64
#)

# Add meta features
X_train_meta = create_meta_features(X_train)
X_test_meta = create_meta_features(X_test)

X_train_full = hstack([
    X_train_tfidf,
    csr_matrix(train_embeddings),
    csr_matrix(X_train_meta)
]).tocsr()

X_test_full = hstack([
    X_test_tfidf,
    csr_matrix(test_embeddings),
    csr_matrix(X_test_meta)
]).tocsr()



tfidf_feature_names = [f'tfidf_{i}' for i in range(X_train_tfidf.shape[1])] 
mathbert_feature_names = [f'mathbert_{i}' for i in range(train_embeddings.shape[1])]  # 64 features
meta_feature_names = ['math_expr', 'latex_cmds', 'numbers', 'length']
all_feature_names = tfidf_feature_names + mathbert_feature_names + meta_feature_names


NUM_FOLDS = 5
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_train))
test_preds = np.zeros((len(X_test), NUM_FOLDS))

class_weights = {
    0: 1.0,  # Algebra
    1: 1.0,  # Geometry
    2: 1.2,  # Calculus
    3: 1.5,  # Probability
    4: 1.2,  # Number Theory
    5: 1.3,  # Combinatorics
    6: 5.0,  # Linear Algebra 
    7: 3.0   # Abstract Algebra
}

params = {
    'iterations': 2000,
    'learning_rate': 0.05,
    'depth': 8,
    'l2_leaf_reg': 3,
    'random_strength': 0.5,
    'border_count': 128,
    'class_weights': class_weights,
    'auto_class_weights': None,
    'loss_function': 'MultiClass',
    'eval_metric': 'TotalF1:average=Micro',
    'task_type': 'GPU',  # Enable GPU acceleration
    'early_stopping_rounds': 100,
    'verbose': 200
}

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train_full, y_train)):
    print(f"\nFold {fold+1}/{NUM_FOLDS}")
    
    # Split data using indices on CSR matrix
    X_trn = X_train_full[trn_idx]
    X_val = X_train_full[val_idx]
    y_trn, y_val = y_train[trn_idx], y_train[val_idx]
    
    # Create CatBoost pools without text_features
    train_pool = Pool(
        data=X_trn,
        label=y_trn,
        feature_names=all_feature_names
    )
    
    val_pool = Pool(
        data=X_val,
        label=y_val,
        feature_names=all_feature_names
    )
    
    model = CatBoostClassifier(**params)
    
    # Train with early stopping
    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True
    )
    
    # Predictions
    oof_preds[val_idx] = model.predict(X_val).flatten()  
    test_preds[:, fold] = model.predict(X_test_full).flatten()  
    
    # Fold evaluation
    fold_f1 = f1_score(y_val, oof_preds[val_idx], average="micro")
    print(classification_report(y_val, oof_preds[val_idx], target_names=CLASS_NAMES))
    print(f"Fold {fold+1} F1 (micro): {fold_f1:.4f}")


def majority_vote(predictions):
    final_preds = []
    for sample in predictions:
        counts = np.bincount(sample.astype(int))  
        max_count = np.max(counts)
        candidates = np.where(counts == max_count)[0]
        
        if len(candidates) > 1:
            # Tie-break: Use prediction from first fold
            final_preds.append(sample[0])
        else:
            final_preds.append(candidates[0])
    return np.array(final_preds)

# Apply voting
final_preds = majority_vote(test_preds)

# OOF evaluation
oof_f1 = f1_score(y_train, oof_preds, average="micro")
print(f"\nOverall OOF F1 (micro): {oof_f1:.4f}")

# Save submission
submission = pd.DataFrame({"id": test_df["id"], "label": final_preds.astype(int)})
submission.to_csv("submission.csv", index=False)
submission




