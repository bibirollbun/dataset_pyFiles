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


# -*- coding: utf-8 -*-
"""
Final Ensemble Pipeline for Math Problem Classification (Advanced Training & Stacking)

This notebook implements a multi-stage ensemble approach to classify math problems
into one of eight predefined categories. Key features:
1. Advanced math-aware text preprocessing.
2. (Optional) Contrastive pre-training of a Transformer model.
3. Fine-tuning a Transformer model (e.g., DeBERTa-v3-base) using Layer-wise 
   Learning Rate Decay (LLRD), (Optional) Adversarial Weight Perturbation (AWP),
   and K-Fold Cross-Validation to generate Out-Of-Fold (OOF) predictions.
4. Training a LightGBM model on engineered symbolic/TF-IDF features, also using 
   K-Fold Cross-Validation to generate OOF predictions.
5. Stacking: Training a meta-learner (e.g., Logistic Regression or LGBM) on the OOF 
   predictions from the Transformer and symbolic models.

The evaluation metric is F1-micro.
"""

# Core Libraries
import os
import gc
import numpy as np
import pandas as pd
import re
import random
import time
import shutil
import logging
from typing import Optional, Tuple, Dict, Any, List

# Scikit-learn for modeling and metrics
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression # For stacking meta-learner
from scipy.special import softmax 

# PyTorch
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader as TorchDataLoader 

# Hugging Face Transformers & Datasets
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainerCallback,
    logging as hf_logging,
    get_scheduler
)
from datasets import Dataset as HFDataset 
from datasets import DatasetDict

# Sentence Transformers for optional contrastive learning stage
try:
    from sentence_transformers import SentenceTransformer, InputExample, losses, models
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers library not found. Contrastive pre-training stage will be disabled.")

# LightGBM for the symbolic model
import lightgbm as lgb

# ==============================================================================
# Configuration
# ==============================================================================
CONFIG = {
    # General Settings
    "seed": 42,
    "num_labels": 8,
    "base_model_name": "microsoft/deberta-v3-base",
    "max_length": 384,
    "output_dir_base": "./math_problem_classifier_advanced",
    "logging_steps": 100, # Frequency of logging by HuggingFace Trainer for Transformer part
    "report_to": "none",

    # Preprocessing Settings
    "transformer_preprocess_strategy": "linearize", # "linearize" or "special_tokens"
    "math_special_tokens_list": [
        "[MATH]", "[FRAC]", "[SQRT]", "[SUM]", "[INT]", "[LIM]", "[SUP]", "[SUB]", "[VEC]",
        "[MAT]", "[EQ]", "[APPROX]", "[NEQ]", "[LT]", "[GT]", "[LEQ]", "[GEQ]",
        "[TIMES]", "[DIV]", "[PM]", "[MP]", "[SIN]", "[COS]", "[TAN]", "[LOG]", "[LN]"
    ],
    "tfidf_max_features": 500,

    # Stage 0: Contrastive Pre-training (Optional)
    "use_contrastive_pretraining": False,
    "contrastive_epochs": 1,
    "contrastive_batch_size": 8,
    "contrastive_warmup_steps_ratio": 0.1,
    "contrastive_output_path": "./math_problem_classifier_advanced/stage0_contrastive_model",
    "contrastive_num_pairs_per_sample": 1,

    # Stage 1A: Transformer Classification Fine-tuning
    "transformer_epochs": 4, 
    "transformer_train_batch_size": 8,
    "transformer_eval_batch_size": 16,
    "transformer_base_learning_rate": 2e-5,
    "transformer_weight_decay": 0.01,
    "transformer_warmup_ratio": 0.1,
    "transformer_grad_accum_steps": 2,
    "transformer_fp16": torch.cuda.is_available(),
    "transformer_early_stopping_patience": 3, 
    "transformer_n_splits": 5, # For K-Fold OOF generation
    "llrd_decay_factor": 0.90,
    "llrd_num_groups": 6,
    "llrd_debug": False,
    # AWP Config
    "use_awp": False, # <<< START WITH THIS AS FALSE TO DEBUG OTHER PARTS FIRST >>>
    "awp_lr": 1e-4,   
    "awp_eps": 1e-3,  
    "awp_start_epoch": 1, 
    "awp_param_name_substring": "weight", # Substring in parameter names to apply AWP (e.g., "weight", "attention", "encoder.layer.11")

    # Stage 1B: Symbolic Model (LightGBM)
    "use_symbolic_model": True,
    "symbolic_model_n_splits": 5, # K-Fold for LGBM OOF generation
    "lgbm_params": {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': 8,
        'n_estimators': 2000, 
        'learning_rate': 0.02, 
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'num_leaves': 31, 
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42, # LGBM's own seed
        'boosting_type': 'gbdt',
    },
    "lgbm_early_stopping_rounds": 100,

    # Stage 2: Stacking Meta-Learner
    "meta_learner_type": "logistic_regression", # "logistic_regression" or "lgbm"
    "meta_lgbm_params": { # If meta_learner_type is "lgbm"
        'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': 8,
        'n_estimators': 500, 'learning_rate': 0.05, 'num_leaves': 20,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'verbose': -1, 'n_jobs': -1, 'seed': 123
    },
    "meta_lr_C": 1.0, # Regularization strength for Logistic Regression

    # Kaggle Specific Paths
    "input_dir": "/kaggle/input/classification-of-math-problems-by-kasut-academy",
    "train_csv": "train.csv",
    "test_csv": "test.csv",
    "submission_filename_prefix": "submission_advanced_stacking",
}

if not SENTENCE_TRANSFORMERS_AVAILABLE:
    CONFIG["use_contrastive_pretraining"] = False
    print("INFO: Contrastive pre-training disabled as sentence-transformers is not available.")
if not torch.cuda.is_available(): # Disable AWP if no CUDA, as it's often tuned for GPU training
    CONFIG["use_awp"] = False
    print("INFO: AWP disabled as CUDA is not available.")


# ==============================================================================
# Setup Logging
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MathProblemClassifier")
hf_logging.set_verbosity_warning()
logger.info("--- Logger Initialized ---")
if CONFIG["report_to"] == "none":
    os.environ['WANDB_DISABLED'] = 'true'
    logger.info("Weights & Biases logging explicitly disabled via WANDB_DISABLED=true")

# ==============================================================================
# Seed for Reproducibility
# ==============================================================================
def seed_everything(seed_value):
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Global seed set to {seed_value}")

seed_everything(CONFIG["seed"])

# ==============================================================================
# Custom Callbacks & Utility Functions (Trainer Base Classes FIRST)
# ==============================================================================

class TrainingProgressCallback(TrainerCallback):
    """A custom Hugging Face TrainerCallback to log training progress."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs: 
            _log_str = ""
            loss = logs.get('loss'); lr = logs.get('learning_rate')
            if loss is not None: _log_str += f"Loss={loss:.4f} | "
            if lr is not None: _log_str += f"LR={lr:.2e} | "
            if 'epoch' in logs: _log_str += f"Epoch={logs['epoch']:.2f} | "
            for k, v in logs.items():
                if k.startswith("eval_"): _log_str += f"{k}={v:.4f} | "
            if _log_str: 
                logger.info(f"Logs Step {state.global_step}: {_log_str.rstrip(' | ')}")

def load_dataframes():
    """Loads train and test datasets from CSV files specified in CONFIG."""
    logger.info("--- Starting Data Loading ---")
    start_time = time.time()
    train_df = pd.read_csv(os.path.join(CONFIG['input_dir'], CONFIG['train_csv']))
    test_df = pd.read_csv(os.path.join(CONFIG['input_dir'], CONFIG['test_csv']))
    train_df['Question'] = train_df['Question'].fillna('')
    test_df['Question'] = test_df['Question'].fillna('')
    logger.info(f"Train: {train_df.shape}, Test: {test_df.shape}. Loaded in {time.time() - start_time:.2f}s.")
    return train_df, test_df

def preprocess_math_text_for_transformer(text: str) -> str:
    text = str(text); text = re.sub(r'\s+', ' ', text).strip()
    strategy = CONFIG['transformer_preprocess_strategy']
    if strategy == 'linearize':
        replacements = [
            (r"\\frac\{([^}]+)\}\{([^}]+)\}", r" fraction \1 over \2 "), (r"\\sqrt\{([^}]+)\}", r" square root of \1 "),
            (r"\\sum_\{([^}]+)\}\^\{([^}]+)\}", r" summation from \1 to \2 of "), (r"\\sum", " summation "),
            (r"\\int_\{([^}]+)\}\^\{([^}]+)\}", r" integral from \1 to \2 of "), (r"\\int", " integral "),
            (r"\\lim_\{([^}]+)\}", r" limit as approaches \1 of "), (r"\\lim", " limit "),
            (r"\^\{([^}]+)\}", r" superscript \1 "), (r"_\{([^}]+)\}", r" subscript \1 "),
            (r"\\vec\{([^}]+)\}", r" vector \1 "), (r"\\mathbf\{([^}]+)\}", r" matrix \1 "),
            (r"\\begin\{(?:pmatrix|bmatrix|matrix)\}.*?\\end\{(?:pmatrix|bmatrix|matrix)\}", " matrix expression "),
            (r"\\[Ss]in", " sine "), (r"\\[Cc]os", " cosine "), (r"\\[Tt]an", " tangent "),
            (r"\\[Ll]og", " log "), (r"\\[Ll]n", " natural log "), (r"\\approx", " approximately equal "), 
            (r"\\neq", " not equal "), (r"\\leq", " less than or equal "), (r"\\geq", " greater than or equal "),
            (r"\\times", " times "), (r"\\cdot", " times "), (r"\\div", " divided by "),
            (r"\\pm", " plus minus "), (r"\\mp", " minus plus "), (r"<", " less than "), 
            (r">", " greater than "), (r"=", " equals "), (r"\$(.*?)\$", r" math expression \1 math expression "),
            (r"\{", ""), (r"\}", ""), (r"\\", " ")
        ]
        for pattern, replacement in replacements: text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    elif strategy == 'special_tokens':
        # (Same replacement list as before for special_tokens strategy)
        replacements = [
            (r"\\frac\{.*?\}\{.*?\}", " [FRAC] "), (r"\\sqrt\{.*?\}", " [SQRT] "), (r"\\sum", " [SUM] "),
            (r"\\int", " [INT] "), (r"\\lim", " [LIM] "), (r"\^\{.*?\}", " [SUP] "), (r"_\{.*?\}", " [SUB] "),
            (r"\\vec\{.*?\}", " [VEC] "), (r"\\mathbf\{.*?\}", " [MAT] "),
            (r"\\begin\{(?:pmatrix|bmatrix|matrix)\}.*?\\end\{(?:pmatrix|bmatrix|matrix)\}", " [MAT] "),
            (r"=", " [EQ] "), (r"\\approx", " [APPROX] "), (r"\\neq", " [NEQ] "), (r"\\leq", " [LEQ] "), 
            (r"<", " [LT] "), (r"\\geq", " [GEQ] "), (r">", " [GT] "), (r"\\times", " [TIMES] "), 
            (r"\\cdot", " [TIMES] "), (r"\\div", " [DIV] "), (r"\\pm", " [PM] "), (r"\\mp", " [MP] "), 
            (r"\\[Ss]in", " [SIN] "), (r"\\[Cc]os", " [COS] "), (r"\\[Tt]an", " [TAN] "), 
            (r"\\[Ll]og", " [LOG] "), (r"\\[Ll]n", " [LN] "), (r"\$(.*?)\$", r" [MATH] \1 [MATH] "),
            (r"\{|\}", ""), (r"\\", " ")
        ]
        for pattern, replacement in replacements: text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_symbolic_features(text_series: pd.Series) -> pd.DataFrame:
    logger.info("Extracting symbolic features...")
    all_features_list = []
    for text_content in text_series:
        text_lower = str(text_content).lower(); current_features = {}
        current_features['n_math_dollar'] = text_lower.count('$')
        current_features['n_math_display'] = len(re.findall(r'\$\$', text_lower))
        latex_commands_map = {
            'n_frac': r'\\frac', 'n_int': r'\\int', 'n_sum': r'\\sum', 'n_lim': r'\\lim',
            'n_matrix': r'\\begin\{(?:matrix|pmatrix|bmatrix)\}|\\mathbf', 'n_vec': r'\\vec',
            'n_sqrt': r'\\sqrt', 'n_partial': r'\\partial', 'n_sin': r'\\[sS]in', 'n_cos': r'\\[cC]os', 
            'n_tan': r'\\[tT]an', 'n_log': r'\\[lL]og', 'n_ln': r'\\[lL]n',
            'n_greek': r'\\(?:alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega)',
            'n_mathrm': r'\\mathrm', 'n_mathcal': r'\\mathcal', 'n_mathbb': r'\\mathbb', 'n_operatorname': r'\\operatorname'
        }
        for name, pattern in latex_commands_map.items(): current_features[name] = len(re.findall(pattern, str(text_content)))
        current_features['n_pow'] = str(text_content).count('^'); current_features['n_sub'] = str(text_content).count('_')
        current_features['n_prime'] = str(text_content).count("'"); current_features['n_equals'] = str(text_content).count("=")
        current_features['n_plus_minus'] = len(re.findall(r'\\pm', str(text_content)))
        current_features['n_times_cdot'] = len(re.findall(r'\\times|\\cdot', str(text_content)))
        current_features['n_div'] = len(re.findall(r'\\div', str(text_content)))
        current_features['n_braces'] = str(text_content).count("{") + str(text_content).count("}")
        current_features['n_brackets'] = str(text_content).count("[") + str(text_content).count("]")
        current_features['n_parentheses'] = str(text_content).count("(") + str(text_content).count(")")
        keywords_list = ['prove', 'solve', 'find', 'calculate', 'compute', 'determine', 'show', 'express', 'integral', 'derivative', 'limit', 'matrix', 'vector', 'eigenvalue', 'eigenvector', 'algebra', 'calculus', 'probability', 'statistics', 'geometry', 'topology', 'combinatorics', 'logic', 'linear', 'differential', 'equation', 'function', 'set', 'space', 'group', 'ring', 'field', 'theorem', 'proof', 'lemma', 'corollary', 'series', 'sequence', 'graph', 'polynomial', 'real', 'complex', 'number', 'integer', 'rational', 'irrational', 'if and only if', 'such that', 'let', 'assume', 'suppose', 'given', 'then', 'hence', 'thus']
        for kw in keywords_list: current_features[f'kw_{kw.replace(" ", "_")}'] = len(re.findall(r'\b' + re.escape(kw) + r'\b', text_lower))
        current_features['text_len'] = len(str(text_content)); current_features['n_words'] = len(str(text_content).split())
        current_features['n_unique_words'] = len(set(text_lower.split())); current_features['n_digits'] = sum(c.isdigit() for c in str(text_content))
        current_features['n_uppercase'] = sum(c.isupper() for c in str(text_content))
        epsilon = 1e-6
        current_features['ratio_math_delimiters_len'] = (current_features['n_math_dollar'] + current_features['n_math_display']*2) / (current_features['text_len'] + epsilon)
        total_latex_symbols = sum(v for k, v in current_features.items() if k.startswith('n_') and k not in ['n_math_dollar', 'n_math_display', 'n_words', 'n_unique_words', 'n_digits', 'n_uppercase', 'text_len'])
        current_features['ratio_latex_symbols_len'] = total_latex_symbols / (current_features['text_len'] + epsilon)
        total_keywords = sum(v for k, v in current_features.items() if k.startswith('kw_'))
        current_features['ratio_keywords_words'] = total_keywords / (current_features['n_words'] + epsilon)
        current_features['ratio_digits_len'] = current_features['n_digits'] / (current_features['text_len'] + epsilon)
        current_features['avg_word_len'] = sum(len(w) for w in text_lower.split()) / (current_features['n_words'] + epsilon)
        all_features_list.append(current_features)
    return pd.DataFrame(all_features_list)

def combine_symbolic_and_tfidf(df_sym_features: pd.DataFrame, text_series_raw: pd.Series, fit_vectorizer: bool = True, vectorizer: Optional[TfidfVectorizer] = None) -> Tuple[pd.DataFrame, TfidfVectorizer]:
    logger.info(f"Combining symbolic features with TF-IDF. Fit new vectorizer: {fit_vectorizer}")
    if fit_vectorizer:
        tfidf_vectorizer = TfidfVectorizer(max_features=CONFIG['tfidf_max_features'], token_pattern=r'(?u)\b\w+\b|\$|\$\$|\\(?:[a-zA-Z]+|\W)', ngram_range=(1,2), min_df=3, max_df=0.9)
        X_tfidf = tfidf_vectorizer.fit_transform(text_series_raw.astype(str))
    else:
        if vectorizer is None: raise ValueError("Vectorizer must be provided if not fitting.")
        tfidf_vectorizer = vectorizer
        X_tfidf = tfidf_vectorizer.transform(text_series_raw.astype(str))
    df_tfidf = pd.DataFrame(X_tfidf.toarray(), index=df_sym_features.index).add_prefix('tfidf_')
    X_combined = pd.concat([df_sym_features, df_tfidf], axis=1)
    X_combined.columns = [str(col) for col in X_combined.columns]
    logger.info(f"Combined symbolic and TF-IDF features shape: {X_combined.shape}")
    return X_combined, tfidf_vectorizer

def tokenize_function_transformer(examples: Dict, tokenizer: AutoTokenizer) -> Dict:
    return tokenizer(examples["processed_text_transformer"], truncation=True, max_length=CONFIG["max_length"], padding=False)

def compute_f1_micro(eval_pred: Tuple) -> Dict[str, float]:
    logits, labels = eval_pred
    if isinstance(logits, tuple): logits = logits[0]
    if not np.all(np.isfinite(logits)): logits = np.nan_to_num(logits)
    predictions = np.argmax(logits, axis=1)
    try: labels = labels.astype(int)
    except ValueError: return {"f1_micro": 0.0}
    if labels.shape != predictions.shape: return {"f1_micro": 0.0}
    try: f1 = f1_score(labels, predictions, average="micro", zero_division=0)
    except Exception: return {"f1_micro": 0.0}
    return {"f1_micro": f1}

def get_llrd_parameter_groups(model: nn.Module, base_lr: float, decay_factor: float, weight_decay: float, num_groups: int) -> List[Dict]:
    logger.info(f"LLRD: BaseLR={base_lr:.2e}, Decay={decay_factor}, WD={weight_decay}, Groups={num_groups}")
    model_type_str = model.config.model_type.split('-')[0]
    base_model = getattr(model, model_type_str, getattr(model, 'base_model', None))
    if base_model is None:
        for attr_name in ['bert', 'roberta', 'deberta', 'electra', 'albert']:
            if hasattr(model, attr_name): base_model = getattr(model, attr_name); break
        if base_model is None: raise ValueError(f"Cannot find base model for LLRD in {model.config.model_type}")

    encoder = getattr(base_model, "encoder", base_model) # Handle models where layers are not under 'encoder'
    num_encoder_layers = len(getattr(encoder, "layer", []))
    if num_encoder_layers == 0: # Fallback for models like RoBERTa that might have different structure for layers
        if hasattr(encoder,'block'): num_encoder_layers = len(encoder.block) # T5
        elif hasattr(model.config, 'num_hidden_layers'): num_encoder_layers = model.config.num_hidden_layers
        else: num_encoder_layers = 12; logger.warning("LLRD: Defaulting to 12 encoder layers.")
    logger.info(f"LLRD: Detected {num_encoder_layers} encoder layers.")
    
    num_enc_groups = num_groups - 2
    layers_per_group = num_encoder_layers // num_enc_groups if num_enc_groups > 0 else num_encoder_layers
    if layers_per_group == 0 and num_encoder_layers > 0 : layers_per_group = 1
    
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    opt_params = []; assigned_ids = set()

    # Embeddings
    lr_emb = base_lr * (decay_factor ** (num_groups - 1))
    emb_wd, emb_nd = [], []
    for n, p in base_model.named_parameters():
        if any(kword in n.lower() for kword in ["embeddings", "Embedding"]) and "encoder.layer" not in n and p.requires_grad:
            (emb_nd if any(nd_name in n for nd_name in no_decay) else emb_wd).append(p)
            assigned_ids.add(id(p))
    opt_params.extend([{'params': emb_wd, 'lr': lr_emb, 'weight_decay': weight_decay}, {'params': emb_nd, 'lr': lr_emb, 'weight_decay': 0.0}])
    logger.info(f"LLRD Group 0 (Embeddings): LR={lr_emb:.2e}, #WD={len(emb_wd)}, #NoWD={len(emb_nd)}")

    # Encoder Layers
    for i in range(num_enc_groups if num_enc_groups > 0 else (1 if num_encoder_layers > 0 else 0)):
        lr_grp = base_lr * (decay_factor ** (num_groups - 1 - (i + 1)))
        start_l = i * layers_per_group; end_l = start_l + layers_per_group
        if i == (num_enc_groups if num_enc_groups > 0 else (1 if num_encoder_layers > 0 else 0)) - 1: end_l = num_encoder_layers
        layer_wd, layer_nd = [], []
        for l_idx in range(start_l, end_l):
            if l_idx >= num_encoder_layers: break
            prefixes = [f"encoder.layer.{l_idx}.", f"layer.{l_idx}.", f"block.{l_idx}."] # Common prefixes
            for n, p in base_model.named_parameters():
                if any(n.startswith(prfx) for prfx in prefixes) and p.requires_grad and id(p) not in assigned_ids:
                    (layer_nd if any(nd_name in n for nd_name in no_decay) else layer_wd).append(p)
                    assigned_ids.add(id(p))
        opt_params.extend([{'params': layer_wd, 'lr': lr_grp, 'weight_decay': weight_decay}, {'params': layer_nd, 'lr': lr_grp, 'weight_decay': 0.0}])
        logger.info(f"LLRD Group {i+1} (Layers {start_l}-{end_l-1}): LR={lr_grp:.2e}, #WD={len(layer_wd)}, #NoWD={len(layer_nd)}")
    
    # Head/Pooler
    head_wd, head_nd = [], []
    for n, p in model.named_parameters():
        if p.requires_grad and id(p) not in assigned_ids:
            (head_nd if any(nd_name in n for nd_name in no_decay) else head_wd).append(p)
    opt_params.extend([{'params': head_wd, 'lr': base_lr, 'weight_decay': weight_decay}, {'params': head_nd, 'lr': base_lr, 'weight_decay': 0.0}])
    logger.info(f"LLRD Group Head: LR={base_lr:.2e}, #WD={len(head_wd)}, #NoWD={len(head_nd)}")
    return opt_params

class LLRDTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.optimizer is None:
            logger.info("LLRDTrainer: Creating LLRD optimizer.")
            param_groups = get_llrd_parameter_groups(self.model, self.args.learning_rate, CONFIG['llrd_decay_factor'], self.args.weight_decay, CONFIG['llrd_num_groups'])
            self.optimizer = AdamW(param_groups, lr=self.args.learning_rate, eps=self.args.adam_epsilon, betas=(self.args.adam_beta1, self.args.adam_beta2))
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(self.args.lr_scheduler_type, self.optimizer, self.args.get_warmup_steps(num_training_steps), num_training_steps)
            logger.info(f"LLRDTrainer: Scheduler created. Type: {self.args.lr_scheduler_type}")

class AWP:
    def __init__(self, model, optimizer, adv_param_name_substring="weight", adv_lr=1e-3, adv_eps=1e-2, start_epoch=0):
        self.model = model; self.optimizer = optimizer
        self.adv_param_name_substring = adv_param_name_substring
        self.adv_lr = adv_lr; self.adv_eps = adv_eps; self.start_epoch = start_epoch
        self.current_epoch = 0; self.backup = {}; self.backup_eps = {}
        logger.info(f"AWP Initialized: lr={adv_lr}, eps={adv_eps}, start_epoch={start_epoch}, param_substr='{adv_param_name_substring}'")

    def set_epoch(self, epoch): self.current_epoch = epoch

    def attack_step(self):
        if self.adv_lr == 0 or self.current_epoch < self.start_epoch: return
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param_name_substring in name:
                # Simplified perturbation using gradient sign and fixed epsilon
                # This makes perturbation magnitude independent of grad norm, controlled by adv_eps * adv_lr
                perturb = (self.adv_lr * param.grad.sign() * self.adv_eps).detach()
                self.backup_eps[name] = perturb # Store the actual perturbation
                param.data = param.data + perturb

    def save_backup(self):
        if self.adv_lr == 0 or self.current_epoch < self.start_epoch: return
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.adv_param_name_substring in name:
                if name not in self.backup: self.backup[name] = param.data.clone()

    def restore_from_backup(self):
        if self.adv_lr == 0 or self.current_epoch < self.start_epoch or not self.backup: return
        for name, param_backup_val in self.backup.items():
            try: self.model.get_parameter(name).data = param_backup_val
            except AttributeError: logger.warning(f"AWP: Param {name} not found for restore.")
        self.backup = {}; self.backup_eps = {}

class AWPTrainer(LLRDTrainer):
    def __init__(self, *args, awp_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.awp_config = awp_config; self.awp_adversary = None

    def train(self, *args, **kwargs):
        if self.awp_config and self.awp_config.get("use_awp", False) and self.awp_adversary is None:
            if self.optimizer is None: logger.error("AWP Error: Optimizer not ready for AWP.")
            else:
                self.awp_adversary = AWP(model=self.model, optimizer=self.optimizer,
                    adv_lr=self.awp_config.get("awp_lr",1e-4), adv_eps=self.awp_config.get("awp_eps",1e-3),
                    start_epoch=self.awp_config.get("awp_start_epoch",0), 
                    adv_param_name_substring=self.awp_config.get("awp_param_name_substring","weight"))
                logger.info("AWP adversary initialized in AWPTrainer.train()")
        return super().train(*args, **kwargs)

    def training_step(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        model.train(); inputs = self._prepare_inputs(inputs)
        if self.awp_adversary: self.awp_adversary.set_epoch(int(self.state.epoch))

        with self.compute_loss_context_manager(): loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1: loss = loss.mean()

        # Regular backward pass
        if self.use_amp and self.scaler is not None: self.scaler.scale(loss).backward()
        elif self.use_apex:
            with self.accelerator.amp.scale_loss(loss, self.optimizer) as scaled_loss: scaled_loss.backward() # type: ignore
        else: self.accelerator.backward(loss)
        
        # AWP Attack
        if self.awp_adversary and self.awp_adversary.current_epoch >= self.awp_adversary.start_epoch:
            self.awp_adversary.save_backup()   
            self.awp_adversary.attack_step()   
            with self.compute_loss_context_manager(): adv_loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1: adv_loss = adv_loss.mean()
            # Adversarial backward pass
            if self.use_amp and self.scaler is not None: self.scaler.scale(adv_loss).backward()
            elif self.use_apex:
                with self.accelerator.amp.scale_loss(adv_loss, self.optimizer) as scaled_adv_loss: scaled_adv_loss.backward() # type: ignore
            else: self.accelerator.backward(adv_loss)
            self.awp_adversary.restore_from_backup()
        return loss.detach() / self.args.gradient_accumulation_steps

# ==============================================================================
# Main Execution Pipeline
# ==============================================================================
def main():
    os.makedirs(CONFIG["output_dir_base"], exist_ok=True)
    train_df, test_df = load_dataframes()

    logger.info(f"Preprocessing text for Transformer with strategy: {CONFIG['transformer_preprocess_strategy']}")
    train_df['processed_text_transformer'] = train_df['Question'].apply(preprocess_math_text_for_transformer)
    test_df['processed_text_transformer'] = test_df['Question'].apply(preprocess_math_text_for_transformer)

    transformer_model_path = CONFIG['base_model_name']
    if CONFIG['use_contrastive_pretraining']: # Stage 0: Contrastive Pre-training
        if not SENTENCE_TRANSFORMERS_AVAILABLE: logger.warning("Contrastive pre-training skipped: sentence-transformers not found.")
        else:
            logger.info("\n===== Stage 0: Contrastive Pre-training =====")
            os.makedirs(CONFIG['contrastive_output_path'], exist_ok=True)
            ct_samples = []
            for _, group in train_df.groupby('label'):
                texts = group['processed_text_transformer'].tolist()
                if len(texts) < 2: continue
                for i in range(len(texts)):
                    for _ in range(CONFIG['contrastive_num_pairs_per_sample']):
                        pos_texts = texts[:i] + texts[i+1:]
                        if pos_texts: ct_samples.append(InputExample(texts=[texts[i], random.choice(pos_texts)]))
            logger.info(f"Generated {len(ct_samples)} contrastive pairs.")
            if ct_samples:
                word_emb = models.Transformer(CONFIG['base_model_name'], max_seq_length=CONFIG['max_length'])
                pooler = models.Pooling(word_emb.get_word_embedding_dimension())
                sbert_model = SentenceTransformer(modules=[word_emb, pooler])
                ct_loss = losses.MultipleNegativesRankingLoss(model=sbert_model)
                ct_loader = TorchDataLoader(ct_samples, shuffle=True, batch_size=CONFIG['contrastive_batch_size'])
                total_steps_ct = len(ct_loader) * CONFIG['contrastive_epochs']
                sbert_model.fit(train_objectives=[(ct_loader, ct_loss)], epochs=CONFIG['contrastive_epochs'],
                                warmup_steps=int(total_steps_ct * CONFIG['contrastive_warmup_steps_ratio']),
                                output_path=CONFIG['contrastive_output_path'], show_progress_bar=True)
                transformer_model_path = CONFIG['contrastive_output_path']
                logger.info(f"Contrastive pre-training finished. Model at: {transformer_model_path}")
                del sbert_model, ct_loader, ct_samples, word_emb, pooler; gc.collect(); torch.cuda.empty_cache()
            else: logger.warning("No contrastive pairs. Skipping contrastive pre-training.")
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['base_model_name']) # Always load tokenizer from base
    if CONFIG['transformer_preprocess_strategy'] == 'special_tokens':
        tokenizer.add_tokens(CONFIG['math_special_tokens_list'], special_tokens=True)

    hf_train_ds = HFDataset.from_pandas(train_df[['processed_text_transformer', 'label']])
    hf_test_ds = HFDataset.from_pandas(test_df[['processed_text_transformer']])
    tokenized_train = hf_train_ds.map(tokenize_function_transformer, batched=True, fn_kwargs={"tokenizer": tokenizer}, remove_columns=['processed_text_transformer'])
    tokenized_test = hf_test_ds.map(tokenize_function_transformer, batched=True, fn_kwargs={"tokenizer": tokenizer}, remove_columns=['processed_text_transformer'])
    tokenized_train.set_format("torch")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Stage 1A: Transformer K-Fold OOF
    logger.info(f"\n===== Stage 1A: {CONFIG['transformer_n_splits']}-Fold Transformer OOF Generation =====")
    skf_tf = StratifiedKFold(n_splits=CONFIG['transformer_n_splits'], shuffle=True, random_state=CONFIG['seed'])
    oof_tf_probs = np.zeros((len(train_df), CONFIG['num_labels']))
    test_tf_probs_folds = np.zeros((len(test_df), CONFIG['transformer_n_splits'], CONFIG['num_labels']))

    for fold, (train_idx, val_idx) in enumerate(skf_tf.split(X=np.zeros(len(train_df)), y=train_df['label'])):
        logger.info(f"\n--- Transformer Fold {fold+1}/{CONFIG['transformer_n_splits']} ---")
        seed_everything(CONFIG['seed'] + fold)
        output_dir_fold = os.path.join(CONFIG['output_dir_base'], f"s1a_tf_fold_{fold}")
        os.makedirs(output_dir_fold, exist_ok=True)

        train_fold_ds = tokenized_train.select(train_idx); val_fold_ds = tokenized_train.select(val_idx)
        model_cfg = AutoConfig.from_pretrained(transformer_model_path, num_labels=CONFIG['num_labels'])
        model = AutoModelForSequenceClassification.from_pretrained(transformer_model_path, config=model_cfg)
        if CONFIG['transformer_preprocess_strategy'] == 'special_tokens': model.resize_token_embeddings(len(tokenizer))

        args = TrainingArguments(
            output_dir=output_dir_fold, num_train_epochs=CONFIG['transformer_epochs'],
            per_device_train_batch_size=CONFIG['transformer_train_batch_size'],
            per_device_eval_batch_size=CONFIG['transformer_eval_batch_size'],
            gradient_accumulation_steps=CONFIG['transformer_grad_accum_steps'],
            learning_rate=CONFIG['transformer_base_learning_rate'], weight_decay=CONFIG['transformer_weight_decay'],
            warmup_ratio=CONFIG['transformer_warmup_ratio'], logging_strategy="epoch",
            eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
            metric_for_best_model="f1_micro", greater_is_better=True, save_total_limit=1,
            report_to=CONFIG["report_to"], fp16=CONFIG['transformer_fp16'], seed=CONFIG['seed'] + fold,
            dataloader_num_workers=2, dataloader_pin_memory=True, remove_unused_columns=True)
        
        TrainerImpl = AWPTrainer if CONFIG["use_awp"] else LLRDTrainer
        awp_kwargs = {"awp_config": CONFIG} if CONFIG["use_awp"] else {}
        logger.info(f"Using Trainer: {TrainerImpl.__name__}")

        trainer = TrainerImpl(model=model, args=args, train_dataset=train_fold_ds, eval_dataset=val_fold_ds,
                              tokenizer=tokenizer, data_collator=data_collator, compute_metrics=compute_f1_micro,
                              callbacks=[EarlyStoppingCallback(CONFIG['transformer_early_stopping_patience']), TrainingProgressCallback()],
                              **awp_kwargs)
        trainer.train()
        oof_logits = trainer.predict(val_fold_ds).predictions
        oof_tf_probs[val_idx] = softmax(oof_logits, axis=1)
        test_logits = trainer.predict(tokenized_test).predictions
        test_tf_probs_folds[:, fold, :] = softmax(test_logits, axis=1)
        fold_f1 = f1_score(val_fold_ds['label'], np.argmax(oof_logits, axis=1), average='micro')
        logger.info(f"Fold {fold+1} OOF F1: {fold_f1:.4f}")
        del model, trainer, train_fold_ds, val_fold_ds; gc.collect(); torch.cuda.empty_cache()
    
    avg_test_tf_probs = np.mean(test_tf_probs_folds, axis=1)
    overall_tf_oof_f1 = f1_score(train_df['label'], np.argmax(oof_tf_probs, axis=1), average='micro')
    logger.info(f"Overall Transformer OOF F1: {overall_tf_oof_f1:.5f}")

    # Stage 1B: LightGBM K-Fold OOF
    oof_lgbm_probs = np.full((len(train_df), CONFIG['num_labels']), 1/CONFIG['num_labels'])
    avg_test_lgbm_probs = np.full((len(test_df), CONFIG['num_labels']), 1/CONFIG['num_labels'])
    if CONFIG['use_symbolic_model']:
        logger.info(f"\n===== Stage 1B: {CONFIG['symbolic_model_n_splits']}-Fold LGBM OOF Generation =====")
        train_sym_feats = extract_symbolic_features(train_df['Question'])
        test_sym_feats = extract_symbolic_features(test_df['Question'])
        
        tfidf_vec = TfidfVectorizer(max_features=CONFIG['tfidf_max_features'], token_pattern=r'(?u)\b\w+\b|\$|\$\$|\\(?:[a-zA-Z]+|\W)', ngram_range=(1,2), min_df=3, max_df=0.9)
        tfidf_vec.fit(train_df['Question'].astype(str)) # Fit on all train text
        
        X_test_comb, _ = combine_symbolic_and_tfidf(test_sym_feats.copy(), test_df['Question'].astype(str), False, tfidf_vec)
        test_lgbm_probs_folds = np.zeros((len(test_df), CONFIG['symbolic_model_n_splits'], CONFIG['num_labels']))

        skf_lgbm = StratifiedKFold(n_splits=CONFIG['symbolic_model_n_splits'], shuffle=True, random_state=CONFIG['seed'] + 100)
        y_lgbm = train_df['label'].values
        for fold, (train_idx, val_idx) in enumerate(skf_lgbm.split(X=train_sym_feats, y=y_lgbm)):
            logger.info(f"\n--- LGBM Fold {fold+1}/{CONFIG['symbolic_model_n_splits']} ---")
            X_tr_sym, X_val_sym = train_sym_feats.iloc[train_idx], train_sym_feats.iloc[val_idx]
            X_tr_txt, X_val_txt = train_df['Question'].iloc[train_idx].astype(str), train_df['Question'].iloc[val_idx].astype(str)
            
            X_tr_comb, _ = combine_symbolic_and_tfidf(X_tr_sym.copy(), X_tr_txt, False, tfidf_vec)
            X_val_comb, _ = combine_symbolic_and_tfidf(X_val_sym.copy(), X_val_txt, False, tfidf_vec)
            y_tr, y_val = y_lgbm[train_idx], y_lgbm[val_idx]

            lgbm = lgb.LGBMClassifier(**CONFIG["lgbm_params"])
            lgbm.fit(X_tr_comb, y_tr, eval_set=[(X_val_comb, y_val)], eval_metric=CONFIG["lgbm_params"]['metric'],
                       callbacks=[lgb.early_stopping(CONFIG["lgbm_early_stopping_rounds"], verbose=False)])
            oof_lgbm_probs[val_idx] = lgbm.predict_proba(X_val_comb)
            test_lgbm_probs_folds[:, fold, :] = lgbm.predict_proba(X_test_comb)
            logger.info(f"LGBM Fold {fold+1} OOF F1: {f1_score(y_val, np.argmax(oof_lgbm_probs[val_idx], axis=1), average='micro'):.4f}")
        avg_test_lgbm_probs = np.mean(test_lgbm_probs_folds, axis=1)
        overall_lgbm_oof_f1 = f1_score(y_lgbm, np.argmax(oof_lgbm_probs, axis=1), average='micro')
        logger.info(f"Overall LGBM OOF F1: {overall_lgbm_oof_f1:.5f}")
        del train_sym_feats, test_sym_feats, X_test_comb, tfidf_vec; gc.collect()

    # Stage 2: Meta-Learner (Stacking)
    logger.info("\n===== Stage 2: Stacking Meta-Learner Training =====")
    meta_feats_train = np.concatenate([oof_tf_probs, oof_lgbm_probs], axis=1)
    meta_feats_test = np.concatenate([avg_test_tf_probs, avg_test_lgbm_probs], axis=1)
    logger.info(f"Meta-learner features - Train: {meta_feats_train.shape}, Test: {meta_feats_test.shape}")
    y_meta_train = train_df['label'].values

    if CONFIG['meta_learner_type'] == 'logistic_regression':
        meta_model = LogisticRegression(solver='liblinear', random_state=CONFIG['seed'], C=CONFIG['meta_lr_C'], max_iter=1000)
        meta_model.fit(meta_feats_train, y_meta_train)
        logger.info("Trained Logistic Regression meta-learner.")
    elif CONFIG['meta_learner_type'] == 'lgbm':
        meta_model = lgb.LGBMClassifier(**CONFIG['meta_lgbm_params'])
        meta_model.fit(meta_feats_train, y_meta_train) # Can add early stopping for meta-LGBM if desired
        logger.info("Trained LGBM meta-learner.")
    else: raise ValueError(f"Unsupported meta_learner_type: {CONFIG['meta_learner_type']}")

    stacked_oof_preds = meta_model.predict(meta_feats_train)
    stacked_oof_f1 = f1_score(y_meta_train, stacked_oof_preds, average="micro")
    logger.info(f"Overall STACKED OOF F1 (Meta-learner on L1 OOFs): {stacked_oof_f1:.5f}")

    # Stage 3: Final Prediction
    logger.info("\n===== Stage 3: Final Prediction with Meta-Learner =====")
    final_preds = meta_model.predict(meta_feats_test)

    # Create Submission File
    logger.info("Creating submission file...")
    sub_df = pd.DataFrame({'id': test_df['id'], 'label': final_preds})
    fname_parts = [CONFIG["submission_filename_prefix"], CONFIG['base_model_name'].split('/')[-1]]
    if CONFIG['use_awp']: fname_parts.append("AWP")
    if CONFIG['use_symbolic_model']: fname_parts.append("SymLGBM")
    fname_parts.append(f"Meta{CONFIG['meta_learner_type'].replace('_','').title()}")
    fname_parts.append(f"StackOOF{stacked_oof_f1:.4f}.csv")
    detailed_fname = "_".join(fname_parts)
    
    final_sub_path = "submission.csv"
    detailed_sub_path = os.path.join(CONFIG['output_dir_base'], detailed_fname)
    try:
        sub_df.to_csv(final_sub_path, index=False); logger.info(f"Submission: {final_sub_path}")
        if os.path.abspath(final_sub_path) != os.path.abspath(detailed_sub_path):
            os.makedirs(os.path.dirname(detailed_sub_path), exist_ok=True)
            shutil.copyfile(final_sub_path, detailed_sub_path); logger.info(f"Also saved as: {detailed_sub_path}")
    except Exception as e:
        logger.error(f"Saving submission '{final_sub_path}' failed: {e}")
        try:
            os.makedirs(os.path.dirname(detailed_sub_path), exist_ok=True)
            sub_df.to_csv(detailed_sub_path, index=False); logger.info(f"Saved submission to: {detailed_sub_path}")
        except Exception as e2: logger.error(f"Saving detailed submission '{detailed_sub_path}' also failed: {e2}")

    # Save OOF features and predictions for analysis
    oof_analysis_df = pd.DataFrame({'id': train_df.get('id', train_df.index), 'true_label': y_meta_train})
    for i in range(CONFIG['num_labels']): oof_analysis_df[f'tf_oof_prob_{i}'] = oof_tf_probs[:, i]
    if CONFIG['use_symbolic_model']:
        for i in range(CONFIG['num_labels']): oof_analysis_df[f'lgbm_oof_prob_{i}'] = oof_lgbm_probs[:, i]
    for i in range(meta_feats_train.shape[1]): oof_analysis_df[f'meta_feat_{i}'] = meta_feats_train[:, i]
    oof_analysis_df['stacked_oof_pred'] = stacked_oof_preds
    oof_analysis_fname = detailed_fname.replace(".csv", "_OOF_Analysis.csv").replace("submission_", "oof_")
    oof_analysis_path = os.path.join(CONFIG['output_dir_base'], oof_analysis_fname)
    try:
        os.makedirs(os.path.dirname(oof_analysis_path), exist_ok=True)
        oof_analysis_df.to_csv(oof_analysis_path, index=False); logger.info(f"OOF analysis saved: {oof_analysis_path}")
    except Exception as e: logger.error(f"Saving OOF analysis failed: {e}")

    logger.info("--- Advanced Math Problem Classification Pipeline Finished Successfully ---")

if __name__ == "__main__":
    main()




