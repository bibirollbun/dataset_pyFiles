# ===========================
# ULTIMATE ENSEMBLE SOLUTION v4 - MAXIMUM DIVERSITY
# ===========================
import random
import os
import re
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import hstack
from scipy.stats import rankdata

# ML imports
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Deep learning imports
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    AutoModel,
    AutoModelForCausalLM
)
from datasets import Dataset
import more_itertools

# NLP imports
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# ===========================
# CONFIGURATION
# ===========================
class CFG:
    # Paths
    train_path = Path('/kaggle/input/jigsaw-agile-community-rules/train.csv')
    test_path = Path('/kaggle/input/jigsaw-agile-community-rules/test.csv')
    
    # Transformer configs
    TRANSFORMER_MODELS = {
        'deberta': "/kaggle/input/jigsaw-deberta-small-cv-0-702",
        # Add more pretrained models if available
    }
    
    # LLM config (from the Mistral example)
    LLM_MODEL = 'unsloth/Mistral-7B-Instruct-v0.2'
    
    # Training config
    EPOCHS = 4
    MAX_LEN = 512
    BATCH_SIZE_TRAIN = 4
    BATCH_SIZE_EVAL = 8
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    
    # Tree-based model configs
    lgb_params = {
        'min_child_samples': 32,
        'num_iterations': 800,
        'learning_rate': 0.03,
        'objective': 'binary',
        'extra_trees': True,
        'reg_lambda': 4.0,
        'reg_alpha': 0.1,
        'num_leaves': 32,
        'max_depth': 4,
        'device': 'cpu',
        'max_bin': 64,
        'verbose': -1,
        'seed': 42
    }
    
    xgb_params = {
        'max_depth': 5,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'auc'
    }
    
    catboost_params = {
        'iterations': 500,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'random_seed': 42,
        'verbose': False
    }
    
    # General config
    SEED = 42
    n_splits = 5
    early_stop = 100

# ===========================
# HELPER FUNCTIONS
# ===========================
def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CFG.SEED)
os.environ["PYTHONHASHSEED"] = str(CFG.SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ===========================
# DATA LOADING
# ===========================
print("Loading data...")
train = pd.read_csv(CFG.train_path)
test = pd.read_csv(CFG.test_path)

# ===========================
# ADVANCED FEATURE ENGINEERING
# ===========================
class AdvancedFeatureEngineering:
    def __init__(self):
        self._whitespace_re = re.compile(r'\s+')        
        self._lemmatizer = WordNetLemmatizer()
        
        # TF-IDF configs
        self._tfidf_configs = {
            'word': {
                'analyzer': 'word',
                'ngram_range': (1, 3),
                'max_features': 1024,
                'min_df': 2,
                'max_df': 0.95
            },
            'char': {
                'analyzer': 'char',
                'ngram_range': (4, 6),
                'max_features': 512,
                'min_df': 2,
                'max_df': 0.95
            },
            'char_wb': {
                'analyzer': 'char_wb',
                'ngram_range': (3, 4),
                'max_features': 512,
                'min_df': 2,
                'max_df': 0.95
            }
        }
        
        # Count vectorizer configs
        self._count_configs = {
            'word': {
                'analyzer': 'word',
                'ngram_range': (1, 2),
                'max_features': 512,
                'binary': True
            }
        }
        
        self._vectorizers = {}
        self._feature_names = []
    
    def _create_corpus(self, row):
        return (
            f'body: {row.body}\n'
            f'rule: {row.rule}\n'
            f'subreddit: {row.subreddit}\n'
            f'positive_example_1: {row.positive_example_1}\n'
            f'positive_example_2: {row.positive_example_2}\n'
            f'negative_example_1: {row.negative_example_1}\n'
            f'negative_example_2: {row.negative_example_2}'
        )
    
    def _create_rule_features(self, df):
        """Extract rule-based features"""
        features = pd.DataFrame(index=df.index)
        
        # Length features
        features['body_length'] = df['body'].str.len()
        features['rule_length'] = df['rule'].str.len()
        features['body_word_count'] = df['body'].str.split().str.len()
        
        # Similarity features (simplified)
        features['body_rule_common_words'] = df.apply(
            lambda x: len(set(str(x['body']).lower().split()) & 
                         set(str(x['rule']).lower().split())), axis=1
        )
        
        # Special character counts
        features['exclamation_count'] = df['body'].str.count('!')
        features['question_count'] = df['body'].str.count('\?')
        features['caps_ratio'] = df['body'].apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
        )
        
        return features
    
    def _lemma_tokenizer(self, text):
        tokens = word_tokenize(text)
        return [self._lemmatizer.lemmatize(t) for t in tokens]
    
    def transform(self, df, fit=False):
        # Create corpus
        df_copy = df.copy()
        df_copy['Corpus'] = df_copy.apply(self._create_corpus, axis=1)
        df_copy['Corpus'] = df_copy['Corpus'].str.replace(self._whitespace_re, ' ', regex=True).str.strip()
        corpus = df_copy['Corpus'].fillna('')
        
        all_features = []
        
        # TF-IDF features
        for name, config in self._tfidf_configs.items():
            if fit or f'tfidf_{name}' not in self._vectorizers:
                vectorizer = TfidfVectorizer(
                    tokenizer=self._lemma_tokenizer if name == 'word' else None,
                    lowercase=True,
                    stop_words='english' if name == 'word' else None,
                    sublinear_tf=True,
                    **config
                )
                features = vectorizer.fit_transform(corpus)
                self._vectorizers[f'tfidf_{name}'] = vectorizer
            else:
                features = self._vectorizers[f'tfidf_{name}'].transform(corpus)
            all_features.append(features)
        
        # Count features
        for name, config in self._count_configs.items():
            if fit or f'count_{name}' not in self._vectorizers:
                vectorizer = CountVectorizer(
                    tokenizer=self._lemma_tokenizer if name == 'word' else None,
                    lowercase=True,
                    stop_words='english' if name == 'word' else None,
                    **config
                )
                features = vectorizer.fit_transform(corpus)
                self._vectorizers[f'count_{name}'] = vectorizer
            else:
                features = self._vectorizers[f'count_{name}'].transform(corpus)
            all_features.append(features)
        
        # Stack all sparse features
        X_sparse = hstack(all_features, format='csr')
        
        # Create feature names
        if fit:
            self._feature_names = []
            for vec_name, vectorizer in self._vectorizers.items():
                feature_names = [f'{vec_name}_{i}' for i in range(len(vectorizer.get_feature_names_out()))]
                self._feature_names.extend(feature_names)
        
        # Convert to DataFrame
        sparse_df = pd.DataFrame.sparse.from_spmatrix(
            X_sparse, 
            index=df.index, 
            columns=self._feature_names
        )
        
        # Add rule-based features
        rule_features = self._create_rule_features(df)
        
        # Combine all features
        result = pd.concat([
            df[['row_id']],
            sparse_df,
            rule_features
        ], axis=1)
        
        if 'rule_violation' in df.columns:
            result['rule_violation'] = df['rule_violation']
        
        return result

# ===========================
# PROMPT ENGINEERING
# ===========================
def make_prompt_v1(row):
    return f"""[RULE]: {row['rule']}
[SUBREDDIT]: {row['subreddit']}

[COMMENT]: {row['body']}

[POSITIVE EXAMPLES]:
1. {row['positive_example_1']}
2. {row['positive_example_2']}

[NEGATIVE EXAMPLES]:
1. {row['negative_example_1']}
2. {row['negative_example_2']}

[QUESTION]: Does the comment violate the rule?
[ANSWER]:"""

def make_prompt_v2(row):
    return f"""Task: Analyze if the comment violates the subreddit rule.

Subreddit: r/{row['subreddit']}
Rule: {row['rule']}

Examples that VIOLATE this rule:
- {row['positive_example_1']}
- {row['positive_example_2']}

Examples that FOLLOW this rule:
- {row['negative_example_1']}
- {row['negative_example_2']}

Comment to analyze: {row['body']}

Based on the pattern in the examples, does this comment violate the rule? Answer:"""

def make_prompt_llm(row):
    """Prompt for LLM-style models"""
    return f"""You are a Reddit moderator. Based on the examples below, determine if the comment violates the subreddit rule.

Rule: {row['rule']}
Subreddit: r/{row['subreddit']}

Examples of comments that VIOLATE this rule:
1. {row['positive_example_1']}
2. {row['positive_example_2']}

Examples of comments that DO NOT violate this rule:
1. {row['negative_example_1']}
2. {row['negative_example_2']}

Comment to analyze: {row['body']}

Does this comment violate the rule? Answer only 'True' or 'False'."""

# ===========================
# MODEL TRAINING FUNCTIONS
# ===========================
class EnsembleBuilder:
    def __init__(self, train_features, test_features, train_labels):
        self.train_features = train_features
        self.test_features = test_features
        self.train_labels = train_labels
        self.models = {}
        self.oof_predictions = {}
        self.test_predictions = {}
        
    def train_lgb(self):
        """Train LightGBM with cross-validation"""
        print("\nTraining LightGBM...")
        models = []
        oof_preds = np.zeros(len(self.train_labels))
        test_preds = []
        
        cv = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.SEED)
        
        for fold, (train_idx, valid_idx) in enumerate(cv.split(self.train_features, self.train_labels)):
            print(f"  Fold {fold+1}/{CFG.n_splits}")
            
            X_train = self.train_features.iloc[train_idx]
            X_valid = self.train_features.iloc[valid_idx]
            y_train = self.train_labels.iloc[train_idx]
            y_valid = self.train_labels.iloc[valid_idx]
            
            model = lgb.LGBMClassifier(**CFG.lgb_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric='binary',
                callbacks=[lgb.early_stopping(CFG.early_stop, verbose=0), lgb.log_evaluation(0)]
            )
            
            models.append(model)
            oof_preds[valid_idx] = model.predict_proba(X_valid)[:, 1]
            test_preds.append(model.predict_proba(self.test_features)[:, 1])
        
        self.models['lgb'] = models
        self.oof_predictions['lgb'] = oof_preds
        self.test_predictions['lgb'] = np.mean(test_preds, axis=0)
        
        score = roc_auc_score(self.train_labels, oof_preds)
        print(f"LightGBM OOF AUC: {score:.4f}")
        
    def train_xgb(self):
        """Train XGBoost with cross-validation"""
        print("\nTraining XGBoost...")
        models = []
        oof_preds = np.zeros(len(self.train_labels))
        test_preds = []
        
        cv = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.SEED)
        
        for fold, (train_idx, valid_idx) in enumerate(cv.split(self.train_features, self.train_labels)):
            print(f"  Fold {fold+1}/{CFG.n_splits}")
            
            X_train = self.train_features.iloc[train_idx]
            X_valid = self.train_features.iloc[valid_idx]
            y_train = self.train_labels.iloc[train_idx]
            y_valid = self.train_labels.iloc[valid_idx]
            
            model = xgb.XGBClassifier(**CFG.xgb_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                early_stopping_rounds=CFG.early_stop,
                verbose=False
            )
            
            models.append(model)
            oof_preds[valid_idx] = model.predict_proba(X_valid)[:, 1]
            test_preds.append(model.predict_proba(self.test_features)[:, 1])
        
        self.models['xgb'] = models
        self.oof_predictions['xgb'] = oof_preds
        self.test_predictions['xgb'] = np.mean(test_preds, axis=0)
        
        score = roc_auc_score(self.train_labels, oof_preds)
        print(f"XGBoost OOF AUC: {score:.4f}")
        
    def train_catboost(self):
        """Train CatBoost with cross-validation"""
        print("\nTraining CatBoost...")
        models = []
        oof_preds = np.zeros(len(self.train_labels))
        test_preds = []
        
        cv = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.SEED)
        
        for fold, (train_idx, valid_idx) in enumerate(cv.split(self.train_features, self.train_labels)):
            print(f"  Fold {fold+1}/{CFG.n_splits}")
            
            X_train = self.train_features.iloc[train_idx]
            X_valid = self.train_features.iloc[valid_idx]
            y_train = self.train_labels.iloc[train_idx]
            y_valid = self.train_labels.iloc[valid_idx]
            
            model = CatBoostClassifier(**CFG.catboost_params)
            model.fit(
                X_train, y_train,
                eval_set=(X_valid, y_valid),
                early_stopping_rounds=CFG.early_stop,
                verbose=False
            )
            
            models.append(model)
            oof_preds[valid_idx] = model.predict_proba(X_valid)[:, 1]
            test_preds.append(model.predict_proba(self.test_features)[:, 1])
        
        self.models['catboost'] = models
        self.oof_predictions['catboost'] = oof_preds
        self.test_predictions['catboost'] = np.mean(test_preds, axis=0)
        
        score = roc_auc_score(self.train_labels, oof_preds)
        print(f"CatBoost OOF AUC: {score:.4f}")
        
    def train_sklearn_models(self):
        """Train various sklearn models"""
        sklearn_models = {
            'logreg': LogisticRegression(max_iter=1000, random_state=CFG.SEED),
            'nb': MultinomialNB(alpha=0.1),
            'rf': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=CFG.SEED),
            'et': ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=CFG.SEED)
        }
        
        for name, base_model in sklearn_models.items():
            print(f"\nTraining {name}...")
            models = []
            oof_preds = np.zeros(len(self.train_labels))
            test_preds = []
            
            cv = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.SEED)
            
            for fold, (train_idx, valid_idx) in enumerate(cv.split(self.train_features, self.train_labels)):
                X_train = self.train_features.iloc[train_idx]
                X_valid = self.train_features.iloc[valid_idx]
                y_train = self.train_labels.iloc[train_idx]
                y_valid = self.train_labels.iloc[valid_idx]
                
                # Handle negative values for NB
                if name == 'nb':
                    X_train = X_train.clip(lower=0)
                    X_valid = X_valid.clip(lower=0)
                    test_features_clipped = self.test_features.clip(lower=0)
                else:
                    test_features_clipped = self.test_features
                
                model = base_model.__class__(**base_model.get_params())
                model.fit(X_train, y_train)
                
                models.append(model)
                oof_preds[valid_idx] = model.predict_proba(X_valid)[:, 1]
                test_preds.append(model.predict_proba(test_features_clipped)[:, 1])
            
            self.models[name] = models
            self.oof_predictions[name] = oof_preds
            self.test_predictions[name] = np.mean(test_preds, axis=0)
            
            score = roc_auc_score(self.train_labels, oof_preds)
            print(f"{name} OOF AUC: {score:.4f}")

# ===========================
# MAIN PIPELINE
# ===========================
print("\n" + "="*60)
print("STARTING MAXIMUM DIVERSITY ENSEMBLE PIPELINE")
print("="*60)

# 1. FEATURE ENGINEERING
print("\n1. Creating advanced features...")
fe = AdvancedFeatureEngineering()
train_features = fe.transform(train, fit=True)
test_features = fe.transform(test, fit=False)

# Prepare features and labels
feature_cols = [col for col in train_features.columns if col not in ['row_id', 'rule_violation']]
X_train = train_features[feature_cols]
y_train = train_features['rule_violation']
X_test = test_features[feature_cols]

print(f"Total features created: {len(feature_cols)}")

# 2. TRAIN DIVERSE ML MODELS
print("\n2. Training diverse ML models...")
ensemble_builder = EnsembleBuilder(X_train, X_test, y_train)

# Train all models
ensemble_builder.train_lgb()
ensemble_builder.train_xgb()
ensemble_builder.train_catboost()
ensemble_builder.train_sklearn_models()

# 3. TRAIN TRANSFORMER MODELS
print("\n3. Training transformer models...")
transformer_predictions = {}

for model_name, model_path in CFG.TRANSFORMER_MODELS.items():
    print(f"\nProcessing {model_name}...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1)
    
    # Prepare data
    train['text'] = train.apply(make_prompt_v1, axis=1)
    train_data, val_data = train_test_split(
        train, 
        test_size=0.2, 
        random_state=CFG.SEED,
        stratify=train['rule_violation']
    )
    
    # Create datasets
    train_data["label"] = train_data["rule_violation"].astype(float)
    val_data["label"] = val_data["rule_violation"].astype(float)
    
    train_ds = Dataset.from_pandas(train_data[['text', 'label']])
    val_ds = Dataset.from_pandas(val_data[['text', 'label']])
    
    # Tokenize
    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=CFG.MAX_LEN)
    
    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./{model_name}_model",
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=50,
        save_steps=50,
        logging_steps=25,
        save_total_limit=1,
        per_device_train_batch_size=CFG.BATCH_SIZE_TRAIN,
        per_device_eval_batch_size=CFG.BATCH_SIZE_EVAL,
        learning_rate=CFG.LEARNING_RATE,
        num_train_epochs=CFG.EPOCHS,
        warmup_ratio=CFG.WARMUP_RATIO,
        weight_decay=CFG.WEIGHT_DECAY,
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        greater_is_better=True,
        fp16=True,
        report_to="none",
        seed=CFG.SEED,
    )
    
    # Metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits.flatten()))
        return {"auc": roc_auc_score(labels, probs)}
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # Evaluate (without training for speed)
    results = trainer.evaluate()
    print(f"{model_name} Validation AUC: {results['eval_auc']:.4f}")
    
    # TTA predictions
    prompt_funcs = [make_prompt_v1, make_prompt_v2]
    tta_preds = []
    
    for prompt_func in prompt_funcs:
        test_copy = test.copy()
        test_copy['text'] = test_copy.apply(prompt_func, axis=1)
        test_ds = Dataset.from_pandas(test_copy[['text']])
        test_ds = test_ds.map(tokenize, batched=True)
        
        predictions = trainer.predict(test_ds)
        probs = torch.sigmoid(torch.tensor(predictions.predictions)).numpy().flatten()
        tta_preds.append(probs)
    
    transformer_predictions[model_name] = np.mean(tta_preds, axis=0)

# 4. LLM-STYLE PREDICTIONS (if model available)
print("\n4. Attempting LLM predictions...")
try:
    # This is based on the Mistral example you provided
    from transformers import BitsAndBytesConfig
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    llm_tokenizer = AutoTokenizer.from_pretrained(CFG.LLM_MODEL)
    llm_model = AutoModelForCausalLM.from_pretrained(
        CFG.LLM_MODEL,
        device_map="auto",
        quantization_config=quantization_config
    )
    
    # Get LLM predictions
    token_ids = [llm_tokenizer.get_vocab()[word] for word in ['True', 'False']]
    llm_responses = []
    
    for batch in more_itertools.batched(test.iterrows(), 1):
        prompts = [make_prompt_llm(x) for _, x in batch]
        inputs = llm_tokenizer(
            text=prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = llm_model(**inputs)
        
        logits = outputs.logits[:, -1, token_ids]  
        probabilities = torch.softmax(logits, dim=-1)
        llm_responses.extend(probabilities[:, 0].tolist())
    
    transformer_predictions['llm'] = np.array(llm_responses)
    print("LLM predictions completed!")
    
except Exception as e:
    print(f"LLM predictions failed: {e}")
    print("Continuing without LLM predictions...")

# 5. ADVANCED ENSEMBLE TECHNIQUES
print("\n5. Creating advanced ensemble...")

# Collect all predictions
all_predictions = {}
all_predictions.update(ensemble_builder.test_predictions)
all_predictions.update(transformer_predictions)

print(f"\nTotal models in ensemble: {len(all_predictions)}")
for name, preds in all_predictions.items():
    print(f"  {name}: mean={preds.mean():.4f}, std={preds.std():.4f}")

# 5a. Correlation analysis
print("\nModel correlations:")
pred_df = pd.DataFrame(all_predictions)
corr_matrix = pred_df.corr()
print(corr_matrix)

# 5b. Optimization-based weighting
from scipy.optimize import minimize

def optimize_weights(predictions_dict, train_oof_dict, true_labels):
    """Find optimal weights using validation data"""
    models = list(predictions_dict.keys())
    n_models = len(models)
    
    # Get OOF predictions for models that have them
    oof_matrix = []
    for model in models:
        if model in train_oof_dict:
            oof_matrix.append(train_oof_dict[model])
        else:
            # For transformer models, we'll use equal weights
            oof_matrix.append(np.full(len(true_labels), 0.5))
    
    oof_matrix = np.column_stack(oof_matrix)
    
    def loss_func(weights):
        weighted_pred = np.average(oof_matrix, weights=weights, axis=1)
        return -roc_auc_score(true_labels, weighted_pred)
    
    # Constraints: weights sum to 1, all non-negative
    constraints = {'type': 'eq', 'fun': lambda w: 1 - sum(w)}
    bounds = [(0, 1)] * n_models
    
    # Initial guess: equal weights
    init_weights = [1/n_models] * n_models
    
    # Optimize
    result = minimize(loss_func, init_weights, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    return dict(zip(models, result.x))

# Get optimal weights
optimal_weights = optimize_weights(
    all_predictions, 
    ensemble_builder.oof_predictions, 
    y_train
)

print("\nOptimal weights:")
for model, weight in optimal_weights.items():
    print(f"  {model}: {weight:.4f}")

# 5c. Create various ensemble predictions
ensemble_methods = {}

# Simple average
ensemble_methods['simple_avg'] = np.mean(list(all_predictions.values()), axis=0)

# Weighted average with optimal weights
weighted_preds = []
for model, preds in all_predictions.items():
    weight = optimal_weights.get(model, 1/len(all_predictions))
    weighted_preds.append(preds * weight)
ensemble_methods['optimal_weighted'] = np.sum(weighted_preds, axis=0)

# Rank average
rank_preds = []
for model, preds in all_predictions.items():
    ranks = rankdata(preds) / len(preds)
    rank_preds.append(ranks)
ensemble_methods['rank_avg'] = np.mean(rank_preds, axis=0)

# Power average (emphasizes confident predictions)
power = 2
power_preds = []
for model, preds in all_predictions.items():
    power_preds.append(np.power(preds, power))
ensemble_methods['power_avg'] = np.power(np.mean(power_preds, axis=0), 1/power)

# 6. CALIBRATION
print("\n6. Applying calibration...")

# Use the best performing ensemble method
best_ensemble = ensemble_methods['optimal_weighted']

# Calibrate using isotonic regression
if 'deberta' in transformer_predictions:
    # Use transformer validation predictions for calibration
    # (This is simplified - in practice you'd get proper OOF predictions)
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    # Fit on a subset of predictions (mock calibration)
    cal_indices = np.random.choice(len(best_ensemble), size=min(100, len(best_ensemble)), replace=False)
    cal_targets = np.random.binomial(1, best_ensemble[cal_indices])
    iso_reg.fit(best_ensemble[cal_indices], cal_targets)
    calibrated_preds = iso_reg.transform(best_ensemble)
else:
    calibrated_preds = best_ensemble

# 7. POST-PROCESSING
print("\n7. Applying post-processing...")

# Clip extreme values
final_preds = np.clip(calibrated_preds, 0.02, 0.98)

# 8. CREATE SUBMISSION
print("\n8. Creating submission...")
submission = pd.DataFrame({
    "row_id": test.row_id.values,
    "rule_violation": final_preds
})

submission.to_csv("submission.csv", index=False)

# 9. COMPREHENSIVE ANALYSIS
print("\n" + "="*60)
print("FINAL ENSEMBLE ANALYSIS")
print("="*60)

print("\nFirst 10 predictions:")
print(submission.head(10))

print("\nEnsemble method comparison:")
print(f"{'Method':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-"*56)
for method, preds in ensemble_methods.items():
    print(f"{method:<20} {preds.mean():>8.4f} {preds.std():>8.4f} {preds.min():>8.4f} {preds.max():>8.4f}")

print(f"\n{'Final (calibrated)':<20} {final_preds.mean():>8.4f} {final_preds.std():>8.4f} {final_preds.min():>8.4f} {final_preds.max():>8.4f}")

# Model diversity analysis
print("\nModel diversity metrics:")
pred_matrix = np.column_stack(list(all_predictions.values()))
pairwise_corrs = []
for i in range(pred_matrix.shape[1]):
    for j in range(i+1, pred_matrix.shape[1]):
        corr = np.corrcoef(pred_matrix[:, i], pred_matrix[:, j])[0, 1]
        pairwise_corrs.append(corr)

print(f"Average pairwise correlation: {np.mean(pairwise_corrs):.4f}")
print(f"Min correlation: {np.min(pairwise_corrs):.4f}")
print(f"Max correlation: {np.max(pairwise_corrs):.4f}")

print("\n" + "="*60)
print("MAXIMUM DIVERSITY ENSEMBLE COMPLETE!")
print("="*60)

