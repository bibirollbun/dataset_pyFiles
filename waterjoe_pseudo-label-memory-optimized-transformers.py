# Install core libraries
!pip install -q tqdm textstat langdetect lightgbm scikit-learn

# Install HuggingFace Transformers and PyTorch optimization libraries
# 'accelerate' is crucial for mixed precision training (amp)
!pip install -q transformers sentence-transformers accelerate

# Install WordCloud for visualizations
!pip install -q wordcloud

# Download NLTK data (punkt and stopwords) - also handled by Python code
import nltk
nltk.download('punkt')
nltk.download('stopwords')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

probability_submission = pd.read_csv('/kaggle/input/submis/0.9probabilities.csv')
# 可视化概率分布
plt.figure(figsize=(12, 6))
sns.histplot(data=probability_submission, x='prob0', kde=True, color='blue', label='prob0')
sns.histplot(data=probability_submission, x='prob1', kde=True, color='orange', label='prob1')
plt.title('Probability Distribution')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.legend()

# 设置x轴的刻度为0.1
plt.xticks(np.arange(0.0, 1.1, 0.05))  # 从0.0到1.0，间隔为0.1

plt.show()

# 定义高置信度阈值
high_confidence_threshold = 0.95  # 你可以根据需要调整这个阈值

# 筛选出高置信度的行
high_confidence_df = probability_submission[
    (probability_submission['prob0'] >= high_confidence_threshold) | 
    (probability_submission['prob1'] >= high_confidence_threshold)
]

# 打印筛选后的数据
print("\n高置信度数据：")
print(high_confidence_df)


# ==================================================================================
#
#   Project: Fake or Real: The Impostor Hunt in Texts (Kaggle Competition)
#
#   Description:
#   This script implements a complete, end-to-end pipeline for the "Fake or Real"
#   competition, embodying a sophisticated strategy including:
#   1. Robust cross-validation protocol.
#   2. A twin-pillar modeling approach:
#      - A Deep Learning "Scalpel" (Conceptual Siamese DeBERTa-v3) for semantic nuance.
#      - A Feature Engineering "Hammer" (LightGBM) with advanced forensic features.
#   3. Implementation of the "LLM Judge" feature using the Gemini API (offline-safe for submission).
#   4. Advanced multi-level ensembling of the model outputs, including a 'Nemesis' model.
#   5. Conceptual implementation of Cascaded Inference for optimized execution.
#   6. Advanced visualizations for deep insights into model behavior and interpretability.
#   7. Ultra-optimization for time and space complexity, including GPU efficiency.
#
#   This version ensures error-free execution in Kaggle notebooks by making
#   API-dependent and large model features offline-safe, with clear guidance
#   on how to pre-compute them for competitive performance.
#
# ==================================================================================

# --- 0. Environment Setup and Imports ---
print("Step 0: Setting up the environment and importing libraries...")

import numpy as np
import pandas as pd
import os
import glob
from tqdm.auto import tqdm
import textstat
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string
import re
import unicodedata
from langdetect import detect, DetectorFactory, LangDetectException # For English/Latin ratio features
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, brier_score_loss # For calibration
from sklearn.preprocessing import StandardScaler # For feature scaling (optional but good practice)
from sklearn.linear_model import LogisticRegression # For Level 2 Stacking meta-model
from sklearn.calibration import calibration_curve # For reliability diagram
import warnings
import json
import requests
import time
import gc # For garbage collection

# Deep Learning specific imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig # AutoModel for custom Siamese backbone
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW # Corrected import for AdamW
import torch.cuda.amp as amp # For Mixed Precision Training

# For Tier 2 & 3 Deep Forensic Features (made offline-safe)
from sentence_transformers import SentenceTransformer, util 
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud # For LLM Judge explanation visualization
sns.set_style("whitegrid") # Set seaborn style for better aesthetics
plt.rcParams['figure.dpi'] = 150 # High resolution plots
plt.rcParams['savefig.dpi'] = 150 # High resolution saved plots

# Kaggle Secrets import for API Key
from kaggle_secrets import UserSecretsClient 

warnings.filterwarnings('ignore')

# Set seed for reproducibility for langdetect and overall pipeline
DetectorFactory.seed = 42
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Use current_device for multi-GPU setups and consistent seeding on GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(torch.cuda.current_device()) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(42)

# Download NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    print("Downloading NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("NLTK data downloaded.")

# --- Configuration ---
print("Step 1: Configuring the pipeline...")

class CFG:
    # --- General ---
    competition_name = 'fake-or-real-the-impostor-hunt'
    seed = 42
    n_folds = 5 # Reduced for demonstration; 10 is recommended for final submission as per Master Plan.
    target_col = 'label'

    # --- Paths ---
    data_path = f'/kaggle/input/{competition_name}/data/' 
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    train_csv_path = os.path.join(data_path, 'train.csv')
    output_path = './' # Output will be in the notebook's working directory

    # --- Transformer Model (DeBERTa) Config ---
    # Note: 'large' is preferred for performance, 'base' for speed/demonstration
    deberta_model_name = 'microsoft/deberta-v3-base' 
    roberta_model_name = 'roberta-base' # Secondary model
    
    max_length = 512 # Max sequence length for Transformer inputs
    batch_size = 4 # REDUCED BATCH SIZE FOR MEMORY OPTIMIZATION - CRITICAL FOR GPU
    n_epochs = 25 # Number of training epochs for Transformers
    learning_rate = 2e-5
    weight_decay = 0.01
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- LightGBM Model Config ---
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': 2000, # Increased estimators, will use early stopping
        'learning_rate': 0.01,
        'num_leaves': 31,
        'max_depth': 7,
        'seed': seed,
        'n_jobs': -1,
        'verbose': -1,
        'colsample_bytree': 0.7,
        'subsample': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'max_bin': 255, # Max bins for feature discretization
    }

    # --- LLM Judge Feature Config ---
    MAX_CHARS_FOR_LLM_JUDGE = 2000 # Limit text length for LLM API calls to avoid token limits
    LLM_API_MODEL = "gemini-2.0-flash" 
    # Fetch API Key from Kaggle Secrets (will be empty string if not configured)
    
    # user_secrets = UserSecretsClient()
    # LLM_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")# Fetches the secret named "GOOGLE_API_KEY"
    LLM_API_KEY = None  # Handle this case in your code
    
    # --- Cascaded Inference Config ---
    # Threshold for LightGBM confidence to classify as "easy" case.
    # If abs(prob - 0.5) > CASCADED_HIGH_CONFIDENCE_THRESHOLD, use LGBM.
    
    CASCADED_HIGH_CONFIDENCE_THRESHOLD = 0.45 # e.g., if prob < 0.05 or prob > 0.95, LGBM is confident.

config = CFG()
stop_words = set(stopwords.words("english"))

# ===============================================================
# 2. The Foundational Layer: Protocol & Data Integrity
# ===============================================================
print("Step 2: Implementing foundational data loading and preprocessing...")

def read_text_files_robust(df, base_path):
    """
    Reads text content from file_1.txt and file_2.txt for each article ID.
    Handles missing files gracefully.
    """
    texts_1, texts_2 = [], []
    all_dirs = glob.glob(os.path.join(base_path, 'article_*')) 
    dir_map = {int(os.path.basename(p).replace('article_', '')): p for p in all_dirs} 

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Reading files from {os.path.basename(base_path)}"): 
        article_id = row['id']
        dir_path = dir_map.get(article_id)

        text1_content = ""
        text2_content = ""

        if dir_path:
            try:
                with open(os.path.join(dir_path, 'file_1.txt'), 'r', encoding='utf-8') as f: 
                    text1_content = f.read()
            except FileNotFoundError:
                pass 
            
            try:
                with open(os.path.join(dir_path, 'file_2.txt'), 'r', encoding='utf-8') as f: 
                    text2_content = f.read()
            except FileNotFoundError:
                pass 
        
        texts_1.append(text1_content)
        texts_2.append(text2_content)

    df['text_1'] = texts_1
    df['text_2'] = texts_2
    return df

# def load_competition_data(config):
#     """Loads train.csv and text files for train and test sets."""
#     train_df = pd.read_csv(config.train_csv_path)
    
#     train_df = read_text_files_robust(train_df, config.train_path)
    
#     test_dirs = glob.glob(os.path.join(config.test_path, 'article_*')) 
#     if not test_dirs:
#         raise FileNotFoundError(f"No 'article_*' directories found in {config.test_path}")
#     test_ids = [int(os.path.basename(p).replace('article_', '')) for p in test_dirs] 
#     test_df = pd.DataFrame(sorted(test_ids), columns=['id'])

#     test_df = read_text_files_robust(test_df, config.test_path)
    
#     train_df[config.target_col] = train_df['real_text_id'].apply(lambda x: 0 if x == 1 else 1)

#     print(f"Train data shape: {train_df.shape}")
#     print(f"Test data shape: {test_df.shape}")
#     print("Train data head (with loaded texts):")
#     print(train_df.head())
#     return train_df, test_df

def load_competition_data(config):
    """Loads train.csv and text files for train and test sets, then adds pseudo-labels."""
    # 1. 加载原始训练集和测试集
    train_df = pd.read_csv(config.train_csv_path)
    train_df = read_text_files_robust(train_df, config.train_path)  # 加载训练集文本
    
    # 加载测试集（无标签，仅id和文本）
    test_dirs = glob.glob(os.path.join(config.test_path, 'article_*')) 
    if not test_dirs:
        raise FileNotFoundError(f"No 'article_*' directories found in {config.test_path}")
    test_ids = [int(os.path.basename(p).replace('article_', '')) for p in test_dirs] 
    test_df = pd.DataFrame(sorted(test_ids), columns=['id'])
    test_df = read_text_files_robust(test_df, config.test_path)  # 加载测试集文本
    
    # 2. 处理原始训练集标签（保持原有逻辑）
    train_df[config.target_col] = train_df['real_text_id'].apply(lambda x: 0 if x == 1 else 1)
    # SWAP DATA
    df_swap = train_df.copy()
    df_swap['text_1'], df_swap['text_2'] = df_swap['text_2'], df_swap['text_1']
    df_swap['label'] = 1 - df_swap['label']
    # CONCAT AUGMENTED DATA TO REAL DATA
    train_df = pd.concat((train_df, df_swap), axis = 0).reset_index(drop = True)
    
    # 3. 加载伪标签文件并与测试集合并
    # 假设伪标签文件路径为 '/kaggle/input/submis/0.9.csv'，列名为 'id' 和 'real_text_id'
    # pseudo_label_path = '/kaggle/input/submis/0.9.csv'
    # pseudo_labels = pd.read_csv(pseudo_label_path)

    # 假设你已经有一个包含概率的 CSV 文件
    probability_submission = pd.read_csv('/kaggle/input/submis/0.9probabilities.csv')
    # 定义高置信度阈值
    high_confidence_threshold = 0.95  # 可根据需要调整
    # 筛选出高置信度的行：prob0 或 prob1 至少有一个 >= 阈值
    high_confidence_df = probability_submission[
        (probability_submission['prob0'] >= high_confidence_threshold) | 
        (probability_submission['prob1'] >= high_confidence_threshold)
    ].copy()
    high_confidence_df['real_text_id'] = high_confidence_df[['prob0', 'prob1']].idxmax(axis=1).map({'prob0': 1, 'prob1': 2})
    high_confidence_submission = high_confidence_df[['id', 'real_text_id']].copy()
    print("高置信度样本的预测结果:")
    print(high_confidence_submission)

    pseudo_labels = high_confidence_submission

    # 按id合并测试集与伪标签（确保每个测试样本都能匹配到伪标签）
    test_with_pseudo = test_df.merge(pseudo_labels, on='id', how='left')
    # 检查是否有未匹配的伪标签（可选，用于验证数据完整性）
    if test_with_pseudo['real_text_id'].isnull().any():
        missing_ids = test_with_pseudo[test_with_pseudo['real_text_id'].isnull()]['id'].tolist()
        print(f"Warning: {len(missing_ids)} test samples have no pseudo-labels. IDs: {missing_ids[:5]}...")


    # 4. 为带伪标签的测试集生成目标列（与训练集标签逻辑一致）
    test_with_pseudo[config.target_col] = test_with_pseudo['real_text_id'].apply(
        lambda x: 0 if x == 1 else 1
    )
    
    # 5. 将带伪标签的测试集拼接至训练集
    combined_train = pd.concat([train_df, test_with_pseudo], ignore_index=True)
    
    # 打印信息验证
    print(f"Original train data shape: {train_df.shape}")
    print(f"Test data with pseudo-labels shape: {test_with_pseudo.shape}")
    print(f"Combined train data shape (original + pseudo): {combined_train.shape}")
    print("Combined train data head:")
    print(combined_train.head())
    
    return combined_train, test_with_pseudo  # 返回合并后的训练集和带伪标签的测试集

# Load the data
train_df, test_df = load_competition_data(config)

# ===============================================================
# 3. The Modeling Core: A Hybrid, Forensically-Driven Architecture
# ===============================================================
print("\nStep 3: Building the Hybrid Modeling Core...")

# --- 3.1. The Feature Engineering Vector: Gradient-Boosted Forensic Analysis ---

def preprocess_text_for_features(text):
    """
    Standardized Data Preprocessing (Section 2.2 of Master Plan).
    Applies NFC Unicode normalization, handles whitespace and line breaks,
    and attempts to remove common HTML/Markdown remnants.
    """
    if not isinstance(text, str):
        return ""
    # NFC Unicode normalization
    text = unicodedata.normalize('NFC', text)
    # Consistent handling of whitespace and line breaks
    text = re.sub(r'\s+', ' ', text).strip()
    # Basic HTML/Markdown remnants sanitization (can be expanded)
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = text.replace('##', '').replace('**', '') # Remove common markdown bold/header
    return text

def generate_stylometric_features(text):
    """
    Extracts a comprehensive set of stylometric and complexity features for a single text.
    Includes robust error handling for textstat.
    """
    text = preprocess_text_for_features(text)

    # Initialize all features to 0 for robustness
    features = {
        'char_count': 0, 'word_count': 0, 'sentence_count': 0, 'avg_word_length': 0,
        'avg_sentence_length': 0, 'unique_word_count': 0, 'ttr': 0, 'stopword_count': 0,
        'stopword_ratio': 0, 'punctuation_count': 0, 'flesch_reading_ease': 0,
        'flesch_kincaid_grade': 0, 'gunning_fog': 0, 'smog_index': 0, 'coleman_liau_index': 0,
        'automated_readability_index': 0, 'dale_chall_readability_score': 0,
        'linsear_write_formula': 0, 'english_ratio': 0, 'latin_ratio': 0,
        'digit_count': 0, 'uppercase_ratio': 0, 'long_word_count': 0, 'short_word_count': 0,
        'avg_syllables_per_word': 0, 'type_token_ratio_sqrt': 0, 'readability_avg': 0
    }
    
    if not text.strip():
        return features

    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    word_count = len(words)
    
    if word_count == 0:
        return features

    char_count = len(text)
    sentence_count = len(sentences)
    avg_word_length = np.mean([len(w) for w in words]) if words else 0
    avg_sentence_length = np.mean([len(word_tokenize(s)) for s in sentences]) if sentences else 0
    unique_word_count = len(set(w.lower() for w in words if w.isalpha()))
    ttr = unique_word_count / word_count if word_count > 0 else 0
    stopword_count = sum(1 for w in words if w.lower() in stop_words)
    stopword_ratio = stopword_count / word_count if word_count > 0 else 0
    punctuation_count = sum(1 for char in text if char in string.punctuation)
    
    # Readability scores with robust error handling for each call
    flesch_reading_ease = 0
    try: flesch_reading_ease = textstat.flesch_reading_ease(text)
    except Exception: pass
    
    flesch_kincaid_grade = 0
    try: flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
    except Exception: pass
    
    gunning_fog = 0
    try: gunning_fog = textstat.gunning_fog(text)
    except Exception: pass
    
    smog_index = 0
    try: smog_index = textstat.smog_index(text)
    except Exception: pass
    
    coleman_liau_index = 0
    try: coleman_liau_index = textstat.coleman_liau_index(text)
    except Exception: pass
    
    automated_readability_index = 0
    try: automated_readability_index = textstat.automated_readability_index(text)
    except Exception: pass
    
    dale_chall_readability_score = 0
    try: dale_chall_readability_score = textstat.dale_chall_readability_score(text)
    except Exception: pass
    
    linsear_write_formula = 0
    try: linsear_write_formula = textstat.linsear_write_formula(text)
    except Exception: pass

    # Language Ratios (useful for LLM artifacts)
    english_ratio = 0
    try:
        chunks = [' '.join(words[i:i+10]) for i in range(0, len(words), 10)]
        if chunks:
            english_count = 0
            for chunk in chunks:
                try:
                    if detect(chunk) == 'en':
                        english_count += 1
                except LangDetectException:
                    pass
            english_ratio = english_count / len(chunks)
    except Exception:
        pass # Keep as 0

    latin_ratio = 0
    non_space_chars = [c for c in text if c != ' ']
    if non_space_chars:
        latin_chars = [c for c in non_space_chars if 'LATIN' in unicodedata.name(c, '')]
        latin_ratio = len(latin_chars) / len(non_space_chars)

    # Additional Features
    digit_count = sum(c.isdigit() for c in text)
    uppercase_ratio = sum(1 for c in text if c.isupper()) / char_count if char_count > 0 else 0
    long_word_count = sum(1 for w in words if len(w) >= 7) # Example threshold
    short_word_count = sum(1 for w in words if len(w) <= 3) # Example threshold
    
    avg_syllables_per_word = 0
    try:
        avg_syllables_per_word = textstat.avg_syllables_per_word(text)
    except Exception: pass

    type_token_ratio_sqrt = unique_word_count / np.sqrt(word_count) if word_count > 0 else 0

    # Calculate readability_avg from the handled scores
    readability_scores_list = [flesch_reading_ease, flesch_kincaid_grade, gunning_fog, smog_index, 
                               coleman_liau_index, automated_readability_index, 
                               dale_chall_readability_score, linsear_write_formula]
    readability_avg = np.mean([s for s in readability_scores_list if s is not None]) if readability_scores_list else 0

    features = {
        'char_count': char_count, 'word_count': word_count, 'sentence_count': sentence_count,
        'avg_word_length': avg_word_length, 'avg_sentence_length': avg_sentence_length,
        'unique_word_count': unique_word_count, 'ttr': ttr, 'stopword_count': stopword_count,
        'stopword_ratio': stopword_ratio, 'punctuation_count': punctuation_count,
        'flesch_reading_ease': flesch_reading_ease, 'flesch_kincaid_grade': flesch_kincaid_grade,
        'gunning_fog': gunning_fog, 'smog_index': smog_index, 'coleman_liau_index': coleman_liau_index,
        'automated_readability_index': automated_readability_index,
        'dale_chall_readability_score': dale_chall_readability_score,
        'linsear_write_formula': linsear_write_formula,
        'english_ratio': english_ratio, 
        'latin_ratio': latin_ratio,     
        'digit_count': digit_count,
        'uppercase_ratio': uppercase_ratio,
        'long_word_count': long_word_count, 'short_word_count': short_word_count,
        'avg_syllables_per_word': avg_syllables_per_word, 'type_token_ratio_sqrt': type_token_ratio_sqrt,
        'readability_avg': readability_avg 
    }
    return features


# Initialize SentenceTransformer for semantic similarity (Offline-safe)
sbert_model = None
try:
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("SentenceTransformer loaded for semantic similarity.")
except Exception as e:
    print(f"Could not load SentenceTransformer: {e}. Semantic similarity features will be skipped (requires internet/offline model).")

# Initialize GPT2 for perplexity calculation (Offline-safe)
gpt2_tokenizer = None
gpt2_model = None
try:
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt2_model.eval()
    gpt2_model.to(config.device)
    print("GPT2 model loaded for perplexity calculation.")
except Exception as e:
    print(f"Could not load GPT2 model: {e}. Perplexity features will be skipped (requires internet/offline model).")


def calculate_semantic_similarity(text1, text2):
    """Tier 2: Calculates semantic similarity using SentenceTransformer."""
    if sbert_model is None: return 0.0
    try:
        embeddings1 = sbert_model.encode(text1, convert_to_tensor=True, show_progress_bar=False)
        embeddings2 = sbert_model.encode(text2, convert_to_tensor=True, show_progress_bar=False)
        similarity = util.pytorch_cos_sim(embeddings1, embeddings2).item()
        return similarity
    except Exception as e:
        # print(f"Error calculating semantic similarity: {e}") # Uncomment for debugging
        return 0.0

def calculate_perplexity(text):
    """Tier 3: Calculates perplexity using GPT2."""
    if gpt2_model is None or not text.strip(): return 1e6 # Return high perplexity for empty text
    try:
        encodings = gpt2_tokenizer(text, return_tensors='pt', truncation=True, max_length=config.max_length)
        input_ids = encodings.input_ids.to(config.device)
        attention_mask = encodings.attention_mask.to(config.device)

        with torch.no_grad():
            outputs = gpt2_model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        return perplexity
    except Exception as e:
        # print(f"Error calculating perplexity for text: {text[:50]}... Error: {e}") # Uncomment for debugging
        return 1e6 # Return a very high perplexity on error

sbert_model = None
try:
    # 使用您指定的 multilingual-e5-small 模型
    sbert_model = SentenceTransformer('intfloat/multilingual-e5-small')
    # 将模型移动到 GPU (如果可用) 以加速推理
    sbert_model = sbert_model.to(config.device)
    print("SentenceTransformer (multilingual-e5-small) loaded for semantic similarity and moved to device.")
except Exception as e:
    print(f"Could not load SentenceTransformer: {e}. Semantic similarity features will be skipped (requires internet/offline model).")
def calculate_semantic_similarity_1(text1, text2):
    """
    Tier 2: Calculates semantic similarity using the multilingual-e5-small model.
    IMPORTANT: The multilingual-e5 models require a prefix for the text.
    For retrieval tasks, use 'query: ' for the query and 'passage: ' for the passage.
    Since we are doing similarity between two passages, using 'passage: ' for both is appropriate.
    """
    if sbert_model is None: 
        return 0.0
    
    try:
        # 为 multilingual-e5 模型添加前缀
        # 根据官方文档，对于段落-段落相似度，使用 'passage: ' 前缀
        prefixed_text1 = f"passage: {text1}"
        prefixed_text2 = f"passage: {text2}"
        
        # 生成嵌入
        embeddings1 = sbert_model.encode(prefixed_text1, convert_to_tensor=True, show_progress_bar=False)
        embeddings2 = sbert_model.encode(prefixed_text2, convert_to_tensor=True, show_progress_bar=False)
        
        # 计算余弦相似度
        similarity = util.pytorch_cos_sim(embeddings1, embeddings2).item()
        
        return similarity
        
    except Exception as e:
        # print(f"Error calculating semantic similarity: {e}") # Uncomment for debugging
        return 0.0
def get_llm_judge_feature(text_a, text_b):
    """
    Tier 4: The Zero-Shot LLM Judge Feature.
    Uses Gemini API to ask an LLM to judge which text is more likely real.
    Returns (verdict: 1/2/0).
    NOTE: This function will return 0 if LLM_API_KEY is not configured (e.g., in Kaggle submission).
    For competitive performance, pre-compute this feature offline.
    """
    if not config.LLM_API_KEY:
        # print("LLM Judge skipped: GOOGLE_API_KEY not found. Pre-compute this feature offline for competitive use.")
        return 0 # Return 0 if API key is not available

    # Truncate texts to avoid token limits for LLM API
    text_a_truncated = text_a[:config.MAX_CHARS_FOR_LLM_JUDGE]
    text_b_truncated = text_b[:config.MAX_CHARS_FOR_LLM_JUDGE]

    prompt = f"""You are a highly discerning forensic editor. You are given two texts, Text A and Text B. One is a real, human-written document from a scientific/technical domain, and the other has been subtly modified or generated by an AI (LLM) to appear real, but it might contain subtle inconsistencies, unnatural phrasing, or factual deviations.

Your task is to determine which text is more likely to be the REAL, human-written one. Consider style, coherence, factual consistency (if implied), and any subtle linguistic artifacts.

Text A:
---
{text_a_truncated}
---

Text B:
---
{text_b_truncated}
---

Based on your expert analysis, state ONLY 'A' if Text A is more likely real, or 'B' if Text B is more likely real. If you genuinely cannot tell, state 'UNCLEAR'.
"""
    
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.LLM_API_MODEL}:generateContent?key={config.LLM_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1, # Keep temperature low for deterministic output
            "maxOutputTokens": 10 # Expecting 'A', 'B', or 'UNCLEAR'
        }
    }

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Raise an exception for HTTP errors
        result = response.json()

        if result.get('candidates') and result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts'):
            llm_response = result['candidates'][0]['content']['parts'][0]['text'].strip().upper()
            if 'A' in llm_response:
                return 1
            elif 'B' in llm_response:
                return 2
            else:
                return 0 # UNCLEAR or other unexpected response
        else:
            return 0
    except Exception as e:
        return 0



def load_precomputed_features(df, feature_name, path_prefix=""):
    """
    Conceptual function to load pre-computed features from a CSV.
    For competitive performance, these features (LLM Judge, Semantic Sim, Perplexity)
    should be generated offline and loaded here.
    """
    file_path = os.path.join(path_prefix, f'{feature_name}_features.csv')
    if os.path.exists(file_path):
        print(f"Loading pre-computed {feature_name} features from {file_path}...")
        precomputed_df = pd.read_csv(file_path)
        # Assuming precomputed_df has 'id' and the feature_name column
        df = df.merge(precomputed_df[['id', feature_name]], on='id', how='left')
        return df[feature_name].values
    else:
        print(f"Pre-computed {feature_name} features not found at {file_path}. Generating/skipping dynamically.")
        return None # Indicate that pre-computed features were not loaded


def create_differential_features(df):
    """
    Creates comprehensive differential features for each pair of texts.
    Includes Tier 1, Tier 2 (Semantic Similarity), Tier 3 (Perplexity), and Tier 4 (LLM Judge).
    Prioritizes loading pre-computed features if available.
    """
    print("Extracting base features for text_1 and text_2...")
    features_1 = df['text_1'].apply(generate_stylometric_features).apply(pd.Series)
    features_2 = df['text_2'].apply(generate_stylometric_features).apply(pd.Series)
    
    feature_cols = list(features_1.columns)
    
    # Create difference and ratio features (Tier 1: Differential Stylometrics)
    print("Creating differential features (diff and ratio)...")
    for col in tqdm(feature_cols, desc="Creating comparison features"): # tqdm wraps the loop
        df[f'{col}_diff'] = features_1[col].astype(float) - features_2[col].astype(float)
        df[f'{col}_ratio'] = features_1[col].astype(float) / (features_2[col].astype(float) + 1e-9)
        
    # --- Tier 2: Cross-Text Relational Features (Semantic Similarity) ---
    # Attempt to load pre-computed, else calculate dynamically (offline-safe)
    precomputed_sem_sim = load_precomputed_features(df, 'semantic_similarity', config.data_path)
    if precomputed_sem_sim is not None:
        df['semantic_similarity'] = precomputed_sem_sim
    else:
        print("Calculating Semantic Similarity (Tier 2) dynamically...")
        if sbert_model is not None:
            df['semantic_similarity'] = df.apply(lambda row: calculate_semantic_similarity(row['text_1'], row['text_2']), axis=1)
            print("Semantic Similarity feature created dynamically.")
        else:
            df['semantic_similarity'] = 0.0
            print("Semantic Similarity skipped (model not loaded/internet restricted).")
            
    precomputed_sem_sim = load_precomputed_features(df, 'semantic_similarity_1', config.data_path)
    if precomputed_sem_sim is not None:
        df['semantic_similarity_1'] = precomputed_sem_sim
    else:
        print("Calculating Semantic Similarity (Tier 2) dynamically...")
        if sbert_model is not None:
            df['semantic_similarity_1'] = df.apply(lambda row: calculate_semantic_similarity_1(row['text_1'], row['text_2']), axis=1)
            print("Semantic Similarity feature created dynamically.")
        else:
            df['semantic_similarity_1'] = 0.0
            print("Semantic Similarity skipped (model not loaded/internet restricted).")

    # --- Tier 3: Deep Forensic Features (Perplexity) ---
    # Attempt to load pre-computed, else calculate dynamically (offline-safe)
    precomputed_perplexity_diff = load_precomputed_features(df, 'perplexity_diff', config.data_path)
    if precomputed_perplexity_diff is not None:
        df['perplexity_diff'] = precomputed_perplexity_diff
        df['perplexity_ratio'] = load_precomputed_features(df, 'perplexity_ratio', config.data_path) # Assuming ratio also precomputed
    else:
        print("Calculating Perplexity Scores (Tier 3) dynamically...")
        if gpt2_model is not None:
            df['perplexity_1'] = df['text_1'].apply(calculate_perplexity)
            df['perplexity_2'] = df['text_2'].apply(calculate_perplexity)
            df['perplexity_diff'] = df['perplexity_1'] - df['perplexity_2']
            df['perplexity_ratio'] = df['perplexity_1'] / (df['perplexity_2'] + 1e-9)
            print("Perplexity features created dynamically.")
        else:
            df['perplexity_1'] = 0.0
            df['perplexity_2'] = 0.0
            df['perplexity_diff'] = 0.0
            df['perplexity_ratio'] = 0.0
            print("Perplexity skipped (model not loaded/internet restricted).")

    # QA-based Factuality Score & LLM Fingerprinting remain conceptual due to complexity
    # print("Conceptual: QA-based Factuality and LLM Fingerprinting features would be added here.")

    # --- Tier 4: The Zero-Shot LLM Judge Feature ---
    # Attempt to load pre-computed, else calculate dynamically (offline-safe)
    precomputed_llm_judge = load_precomputed_features(df, 'llm_judge_verdict', config.data_path)
    if precomputed_llm_judge is not None:
        df['llm_judge_verdict'] = precomputed_llm_judge
    else:
        print("Generating Zero-Shot LLM Judge Feature dynamically (offline-safe fallback)...")
        llm_judge_verdicts = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="LLM Judging"):
            verdict = get_llm_judge_feature(row['text_1'], row['text_2'])
            llm_judge_verdicts.append(verdict)
            time.sleep(0.05) # Small delay to avoid hitting API rate limits if key is present
        df['llm_judge_verdict'] = llm_judge_verdicts
        print("LLM Judge Feature created dynamically (or skipped if no API key).")


    # Collect all feature columns for LightGBM
    final_feature_cols = [f'{col}_diff' for col in feature_cols] + \
                         [f'{col}_ratio' for col in feature_cols] + \
                         ['llm_judge_verdict'] 
    
    if 'semantic_similarity' in df.columns: final_feature_cols.append('semantic_similarity')
    if 'perplexity_diff' in df.columns: final_feature_cols.extend(['perplexity_diff', 'perplexity_ratio'])
    
    return df, final_feature_cols

# --- 3.2. The Deep Learning Vector: Bespoke Semantic Analysis 分类头---
# NEW: True Siamese Network with Cross-Attention (More Explicit Conceptual Outline)
# This is a significant architectural innovation (Master Plan 4.1.1).
# A full implementation would involve defining a custom PyTorch nn.Module.
class SiameseCrossAttentionNetwork(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        # Load the base transformer model (weight-shared backbone)
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Conceptual Cross-Attention Layer
        # This is a simplified representation of cross-attention.
        # A true research-grade cross-attention might involve:
        #   nn.MultiheadAttention(embed_dim, num_heads)
        #   followed by LayerNorm, Dropout, and FeedForward layers.
        # Here, we use MultiheadAttention on the [CLS] tokens, which is a form of interaction.
        hidden_size = self.backbone.config.hidden_size
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=self.backbone.config.num_attention_heads, batch_first=True)
        
        # self.interaction_head = nn.Sequential(
        #     nn.Linear(hidden_size * 2, hidden_size), # Concatenate [CLS] tokens for initial input
        #     nn.ReLU(),
        #     # nn.Dropout(0.1),
        #     nn.Linear(hidden_size, hidden_size // 2),
        #     nn.ReLU(),
        #     # nn.Dropout(0.1)
        # )
        # 输入维度现在是 5 * hidden_size (5种交互特征)
        self.interaction_head = nn.Sequential(
            nn.Linear(hidden_size * 5, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),  # 比ReLU更适合Transformer架构
            nn.Dropout(0.1), 
            
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU()
        )
        self.classifier = nn.Linear(hidden_size // 2, num_labels)

    def forward_one(self, input_ids, attention_mask):
        # Process one text through the shared backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token representation (first token)
        return outputs.last_hidden_state[:, 0, :] # Shape: (batch_size, hidden_size)

    def forward(self, input_ids_A, attention_mask_A, input_ids_B, attention_mask_B, labels=None):
        vec_A = self.forward_one(input_ids_A, attention_mask_A)
        vec_B = self.forward_one(input_ids_B, attention_mask_B)

        # Apply conceptual cross-attention on the [CLS] tokens
        # A more direct cross-attention example (requires unsqueeze/squeeze for MultiheadAttention):
        # query_A = vec_A.unsqueeze(1) # (batch_size, 1, hidden_size)
        # key_value_B = vec_B.unsqueeze(1) # (batch_size, 1, hidden_size)
        # attn_output_A_to_B, _ = self.cross_attn(query=query_A, key=key_value_B, value=key_value_B)
        # # Similarly for B to A
        # query_B = vec_B.unsqueeze(1)
        # key_value_A = vec_A.unsqueeze(1)
        # attn_output_B_to_A, _ = self.cross_attn(query=query_B, key=key_value_A, value=key_value_A)
        # # Combine the attention outputs with original vectors or each other
        # combined_attn_output = torch.cat((attn_output_A_to_B.squeeze(1), attn_output_B_to_A.squeeze(1)), dim=1)
        # interaction_output = self.interaction_head(combined_attn_output)

        # For this simplified conceptual cross-attention, we'll concatenate and pass through a head
        # 对于这个简化的概念性交叉注意，我们将连接并通过一个头部
        # combined_vec_for_head = torch.cat((vec_A, vec_B), dim=1)
        # 丰富的特征交互（关键改进）

        # 计算余弦相似度 (batch_size, 1)
        # 先对向量进行归一化
        vec_A_norm = torch.nn.functional.normalize(vec_A, p=2, dim=1)
        vec_B_norm = torch.nn.functional.normalize(vec_B, p=2, dim=1)
        # 点积得到余弦相似度
        cos_sim = torch.sum(vec_A_norm * vec_B_norm, dim=1, keepdim=True)
        combined_vec_for_head = torch.cat([
            vec_A,
            vec_B,
            vec_A * vec_B,          # 元素级乘积 - 捕捉共同特征
            torch.abs(vec_A - vec_B), # 绝对差值 - 捕捉差异特征
            cos_sim
        ], dim=1)        
        
        interaction_output = self.interaction_head(combined_vec_for_head)
        logits = self.classifier(interaction_output)
        
        loss = None
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, labels)
        
        return type('Outputs', (object,), {'loss': loss, 'logits': logits})()


class TextPairDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts1 = df['text_1'].values
        self.texts2 = df['text_2'].values
        self.labels = df[CFG.target_col].values if CFG.target_col in df.columns else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # For Siamese network, tokenize texts separately
        encoding_A = self.tokenizer(
            self.texts1[idx],
            add_special_tokens=True, truncation=True,
            max_length=self.max_len, padding='max_length',
            return_tensors='pt'
        )
        encoding_B = self.tokenizer(
            self.texts2[idx],
            add_special_tokens=True, truncation=True,
            max_length=self.max_len, padding='max_length',
            return_tensors='pt'
        )
        
        item = {
            'input_ids_A': encoding_A['input_ids'].flatten(),
            'attention_mask_A': encoding_A['attention_mask'].flatten(),
            'input_ids_B': encoding_B['input_ids'].flatten(),
            'attention_mask_B': encoding_B['attention_mask'].flatten(),
        }
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def train_fn(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    scaler = amp.GradScaler() # Initialize GradScaler for mixed precision
    for batch in tqdm(dataloader, desc="Training Batches"): # Generic desc for both DeBERTa/RoBERTa
        input_ids_A = batch['input_ids_A'].to(device)
        attention_mask_A = batch['attention_mask_A'].to(device)
        input_ids_B = batch['input_ids_B'].to(device)
        attention_mask_B = batch['attention_mask_B'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        with amp.autocast(): # Mixed precision context
            outputs = model(input_ids_A, attention_mask_A, input_ids_B, attention_mask_B, labels=labels)
            loss = outputs.loss
        
        scaler.scale(loss).backward() # Scale loss for mixed precision
        scaler.step(optimizer) # Update optimizer
        scaler.update() # Update scaler

        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def eval_fn(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation/Test Batches"): # Generic desc
            input_ids_A = batch['input_ids_A'].to(device)
            attention_mask_A = batch['attention_mask_A'].to(device)
            input_ids_B = batch['input_ids_B'].to(device)
            attention_mask_B = batch['attention_mask_B'].to(device)
            
            with amp.autocast(): # Mixed precision context for evaluation
                outputs = model(input_ids_A, attention_mask_A, input_ids_B, attention_mask_B)
            # Access the logits tensor from the Outputs object
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy() 
            predictions.extend(probs)
    return np.array(predictions)

# --- NEW: Specialist 'Nemesis' Model ---
def train_nemesis_model(oof_preds, true_labels, feature_data, config):
    """
    Trains a specialist 'Nemesis' model on hard-to-classify samples (Master Plan 5.3).
    """
    print("\nTraining Specialist 'Nemesis' Model on hard cases...")
    # Identify hard cases: where the main ensemble's OOF prediction is wrong or very uncertain
    wrong_preds_idx = np.where(np.round(oof_preds) != true_labels)[0]
    
    if len(wrong_preds_idx) == 0:
        print("No wrong predictions identified for Nemesis model. Skipping.")
        class DummyNemesisModel:
            def predict_proba(self, X): return np.zeros((len(X), 2))
        return DummyNemesisModel(), np.zeros(len(feature_data))

    hard_X = feature_data.iloc[wrong_preds_idx]
    hard_y = true_labels[wrong_preds_idx]

    nemesis_model = lgb.LGBMClassifier(**config.lgb_params)
    nemesis_model.fit(hard_X, hard_y)
    print(f"Nemesis model trained on {len(wrong_preds_idx)} hard cases.")
    
    return nemesis_model, nemesis_model.predict_proba(feature_data)[:, 1]

# ===============================================================
# 4. The Synthesis Layer & Main Execution Flow
# ===============================================================
print("\nStep 4: The Synthesis Layer & Main Execution Flow...")

if __name__ == "__main__":
    # Diagnostic print for device usage
    print(f"PyTorch is configured to use device: {config.device}")
    print("\nStarting 'Fake or Real' Competition Solution Execution...")

    # 1. Create Differential Features (including Tier 2/3/4)
    train_df, feature_cols = create_differential_features(train_df)
    test_df, _ = create_differential_features(test_df)
    
    # Ensure all feature columns are numeric and handle NaNs/Infs
    for col in feature_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)

    # 2. Train and Predict with all base models (LightGBM, DeBERTa, RoBERTa)
    oof_lgbm = np.zeros(len(train_df))
    test_lgbm = np.zeros(len(test_df))
    oof_deberta = np.zeros(len(train_df))
    test_deberta = np.zeros(len(test_df))
    oof_roberta = np.zeros(len(train_df)) # For secondary RoBERTa model
    test_roberta = np.zeros(len(test_df)) # For secondary RoBERTa model
    
    deberta_tokenizer = AutoTokenizer.from_pretrained(config.deberta_model_name)
    roberta_tokenizer = AutoTokenizer.from_pretrained(config.roberta_model_name) # For secondary RoBERTa
    
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)

    # Store LGBM models for Cascaded Inference
    lgbm_fold_models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df[config.target_col])):
        print(f"\n===== FOLD {fold+1}/{config.n_folds} =====")
        train_fold_df, val_fold_df = train_df.iloc[train_idx], train_df.iloc[val_idx]

        # --- LightGBM Training & Inference ---
        print(f"--- Training LightGBM for Fold {fold+1} ---")
        lgb_model = lgb.LGBMClassifier(**config.lgb_params)
        lgb_model.fit(train_fold_df[feature_cols], train_fold_df[config.target_col],
                      eval_set=[(val_fold_df[feature_cols], val_fold_df[config.target_col])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_lgbm[val_idx] = lgb_model.predict_proba(val_fold_df[feature_cols])[:, 1]
        test_lgbm += lgb_model.predict_proba(test_df[feature_cols])[:, 1] / config.n_folds
        lgbm_fold_models.append(lgb_model) # Store model for cascaded inference
        del lgb_model
        gc.collect()

        # --- DeBERTa Training & Inference (Conceptual Siamese Network) ---
        print(f"--- Training DeBERTa (Conceptual Siamese) for Fold {fold+1} ---")
        train_dataset_deberta = TextPairDataset(train_fold_df, deberta_tokenizer, config.max_length)
        val_dataset_deberta = TextPairDataset(val_fold_df, deberta_tokenizer, config.max_length)
        train_loader_deberta = DataLoader(train_dataset_deberta, batch_size=config.batch_size, shuffle=True)
        val_loader_deberta = DataLoader(val_dataset_deberta, batch_size=config.batch_size, shuffle=False)
        
        deberta_model = SiameseCrossAttentionNetwork(config.deberta_model_name, num_labels=2)
        deberta_model.to(config.device)
        
        optimizer_deberta = AdamW(deberta_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        num_training_steps_deberta = len(train_loader_deberta) * config.n_epochs
        scheduler_deberta = get_linear_schedule_with_warmup(optimizer_deberta, num_warmup_steps=0, num_training_steps=num_training_steps_deberta)
        
        for epoch in range(config.n_epochs):
            train_loss_deberta = train_fn(deberta_model, train_loader_deberta, optimizer_deberta, scheduler_deberta, config.device)
            print(f"DeBERTa Epoch {epoch+1}, Train Loss: {train_loss_deberta:.4f}")
        
        oof_deberta[val_idx] = eval_fn(deberta_model, val_loader_deberta, config.device)
        
        test_dataset_deberta_full = TextPairDataset(test_df, deberta_tokenizer, config.max_length)
        test_loader_deberta_full = DataLoader(test_dataset_deberta_full, batch_size=config.batch_size, shuffle=False)
        test_deberta += eval_fn(deberta_model, test_loader_deberta_full, config.device) / config.n_folds
        
        del deberta_model, train_loader_deberta, val_loader_deberta, test_loader_deberta_full, test_dataset_deberta_full
        gc.collect()
        torch.cuda.empty_cache()

        # --- Secondary RoBERTa Model Training & Inference ---
        print(f"--- Training RoBERTa (Secondary Model) for Fold {fold+1} ---")
        train_dataset_roberta = TextPairDataset(train_fold_df, roberta_tokenizer, config.max_length)
        val_dataset_roberta = TextPairDataset(val_fold_df, roberta_tokenizer, config.max_length)
        train_loader_roberta = DataLoader(train_dataset_roberta, batch_size=config.batch_size, shuffle=True)
        val_loader_roberta = DataLoader(val_dataset_roberta, batch_size=config.batch_size, shuffle=False)

        # Use SiameseCrossAttentionNetwork for RoBERTa as well for consistency with conceptual Siamese approach
        roberta_model = SiameseCrossAttentionNetwork(config.roberta_model_name, num_labels=2) 
        roberta_model.to(config.device)

        optimizer_roberta = AdamW(roberta_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        num_training_steps_roberta = len(train_loader_roberta) * config.n_epochs
        scheduler_roberta = get_linear_schedule_with_warmup(optimizer_roberta, num_warmup_steps=0, num_training_steps=num_training_steps_roberta)

        for epoch in range(config.n_epochs):
            train_loss_roberta = train_fn(roberta_model, train_loader_roberta, optimizer_roberta, scheduler_roberta, config.device)
            print(f"RoBERTa Epoch {epoch+1}, Train Loss: {train_loss_roberta:.4f}")

        oof_roberta[val_idx] = eval_fn(roberta_model, val_loader_roberta, config.device)
        
        test_dataset_roberta_full = TextPairDataset(test_df, roberta_tokenizer, config.max_length)
        test_loader_roberta_full = DataLoader(test_dataset_roberta_full, batch_size=config.batch_size, shuffle=False)
        test_roberta += eval_fn(roberta_model, test_loader_roberta_full, config.device) / config.n_folds

        del roberta_model, train_loader_roberta, val_loader_roberta, test_loader_roberta_full, test_dataset_roberta_full
        gc.collect()
        torch.cuda.empty_cache()


    # --- 3.3. Train Nemesis Model ---
    # This is trained on the combined OOF of LGBM and DeBERTa to identify hard cases.
    print("\nTraining Nemesis Model...")
    # Using a simple average of OOFs to identify hard cases for Nemesis
    # Include RoBERTa OOF in combined for Nemesis training
    combined_oof_for_nemesis = (oof_deberta + oof_lgbm + oof_roberta) / 3 
    nemesis_model, nemesis_oof_preds = train_nemesis_model(combined_oof_for_nemesis, train_df[config.target_col].values, train_df[feature_cols], config) 
    
    # If Nemesis model was trained, get its test predictions
    nemesis_test_preds = np.zeros(len(test_df))
    if nemesis_model:
        nemesis_test_preds = nemesis_model.predict_proba(test_df[feature_cols])[:, 1]
    
    # --- 3.4. Level 1 Blending & Level 2 Stacking ---
    print("\nEnsembling models and creating submission file...")
    
    # Evaluate OOF Scores
    deberta_oof_accuracy = accuracy_score(train_df[config.target_col], np.round(oof_deberta))
    lgb_oof_accuracy = accuracy_score(train_df[config.target_col], np.round(oof_lgbm))
    roberta_oof_accuracy = accuracy_score(train_df[config.target_col], np.round(oof_roberta))
    print(f"DeBERTa OOF Accuracy: {deberta_oof_accuracy:.5f}")
    print(f"LightGBM OOF Accuracy: {lgb_oof_accuracy:.5f}")
    print(f"RoBERTa OOF Accuracy: {roberta_oof_accuracy:.5f}")

    # Level 1 Blending (simple average for demonstration, weights can be optimized)
    ensemble_oof_level1_avg = (oof_deberta + oof_lgbm + oof_roberta) / 3
    best_level1_acc = accuracy_score(train_df[config.target_col], np.round(ensemble_oof_level1_avg))
    print(f"Level 1 Ensemble (Avg of DeBERTa, LGBM, RoBERTa) OOF Accuracy: {best_level1_acc:.5f}")


    # --- Level 2 Stacking ---
    # Prepare meta-features from OOF predictions of base models (and Nemesis if available)
    meta_X_train = np.column_stack([oof_deberta, oof_lgbm, oof_roberta])
    meta_X_test = np.column_stack([test_deberta, test_lgbm, test_roberta])

    if nemesis_model:
        meta_X_train = np.column_stack([meta_X_train, nemesis_oof_preds])
        meta_X_test = np.column_stack([meta_X_test, nemesis_test_preds])
        print("Nemesis model predictions included in Level 2 stacking.")

    # Train a meta-model (LightGBM as per Master Plan optimization)
    # Using a simplified LGBM for meta-model, can be tuned further
    meta_model = lgb.LGBMClassifier(objective='binary', metric='binary_logloss', n_estimators=200, learning_rate=0.05, num_leaves=10, random_state=config.seed, verbose=-1)
    meta_model.fit(meta_X_train, train_df[config.target_col])
    final_ensemble_preds_proba = meta_model.predict_proba(meta_X_test)[:, 1]
    print("Level 2 Stacking applied with LightGBM meta-model.")
    
    # --- 3.5. Execution Innovation: Cascaded Inference ---
    # Apply Cascaded Inference on the final ensemble predictions将级联推理应用于最终集成预测
    print("\nApplying Cascaded Inference for final submission predictions...")

    # Get LGBM's test predictions for the cascaded inference sieve
    # This is already calculated as test_lgbm
    
    # Identify "easy" cases based on LGBM's confidence
    lgbm_test_confidences = np.abs(test_lgbm - 0.5) # Distance from 0.5
    
    # Initialize final predictions with the full Level 2 ensemble's output
    final_submission_preds_proba = np.copy(final_ensemble_preds_proba)

    # For cases where LGBM is highly confident, use LGBM's prediction
    # This implements the "sieve" where LGBM handles easy cases.
    confident_lgbm_indices = np.where(lgbm_test_confidences > config.CASCADED_HIGH_CONFIDENCE_THRESHOLD)[0]
    final_submission_preds_proba[confident_lgbm_indices] = test_lgbm[confident_lgbm_indices]
    
    print(f"Cascaded Inference: {len(confident_lgbm_indices)} samples handled by LGBM (confident cases).")
    print(f"Remaining {len(test_df) - len(confident_lgbm_indices)} samples handled by full ensemble (hard cases).")

    # Convert probabilities to final class predictions (1 or 2) for the main submission
    main_submission_preds_class = np.round(final_submission_preds_proba).astype(int)
    main_submission_preds = [1 if pred == 0 else 2 for pred in main_submission_preds_class]

    # --- Create Main Submission File ---
    submission_df_main = pd.DataFrame({'id': test_df['id'], 'real_text_id': main_submission_preds})
    submission_df_main.to_csv(os.path.join(config.output_path, 'submission.csv'), index=False)
    print("\nMain Submission file 'submission.csv' created successfully (Full Ensemble + Cascaded Inference)!")
    print(submission_df_main.head())

    # --- Strategic Submission Hedge (Challenger Submission) ---
    # 示例：仅使用 DeBERTa 预测的简单提交
    challenger_preds_class = np.round(test_deberta).astype(int)
    challenger_submission_preds = [1 if pred == 0 else 2 for pred in challenger_preds_class]
    submission_df_challenger = pd.DataFrame({'id': test_df['id'], 'real_text_id': challenger_submission_preds})
    submission_df_challenger.to_csv(os.path.join(config.output_path, 'submission_challenger_deberta_only.csv'), index=False)
    print("\nChallenger Submission file 'submission_challenger_deberta_only.csv' created (DeBERTa Only)!")
    print(submission_df_challenger.head())

    print("\n===== PIPELINE COMPLETE =====")



if __name__ == "__main__":
    # ===============================================================
    # 7. Visualizations and Model Accuracy Results
    # ===============================================================
    print("\nStep 5: Generating Visualizations and Accuracy Results...") # Renumbered step

    # --- 5.1. Overall OOF Accuracy Comparison --- # Renumbered step
    print("\n--- Overall Out-of-Fold Accuracy Comparison ---")
    models_oof_acc = {
        "DeBERTa OOF": deberta_oof_accuracy,
        "LightGBM OOF": lgb_oof_accuracy,
        "RoBERTa OOF": roberta_oof_accuracy, # Added RoBERTa
        "Level 1 Ensemble OOF": best_level1_acc,
        "Level 2 Ensemble OOF": accuracy_score(train_df[config.target_col], np.round(meta_model.predict_proba(meta_X_train)[:,1]))
    }
    
    plt.figure(figsize=(12, 7)) # Adjusted figure size for more bars
    sns.barplot(x=list(models_oof_acc.keys()), y=list(models_oof_acc.values()), palette="viridis")
    plt.title("Model Out-of-Fold Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0.5, 1.0) # Assuming accuracy is above 0.5
    for index, value in enumerate(models_oof_acc.values()):
        plt.text(index, value + 0.01, f'{value:.4f}', ha='center')
    plt.savefig(os.path.join(config.output_path, 'oof_accuracy_comparison.png'))
    plt.show()
    print("Saved: oof_accuracy_comparison.png")

    # --- 5.2. Ensemble Weight vs. Accuracy Plot (Level 1) --- # Renumbered step
    # This plot is now less meaningful with 3 models in Level 1, but we keep it for conceptual Level 1 blending.
    # For a 3-model blend, a ternary plot or more complex visualization would be needed.
    print("\n--- Level 1 Ensemble Weight vs. Accuracy (Conceptual for 3 models) ---")
    # We'll just show the best 3-model average here.
    plt.figure(figsize=(10, 6))
    plt.bar(["Avg of DeBERTa, LGBM, RoBERTa"], [best_level1_acc], color='skyblue')
    plt.title("Level 1 Ensemble (Simple Average) OOF Accuracy")
    plt.ylabel("OOF Accuracy")
    plt.ylim(0.5, 1.0)
    plt.text(0, best_level1_acc + 0.01, f'{best_level1_acc:.4f}', ha='center')
    plt.savefig(os.path.join(config.output_path, 'ensemble_level1_accuracy.png'))
    plt.show()
    print("Saved: ensemble_level1_accuracy.png")


    # --- 5.3. LightGBM Feature Importance Plot --- # Renumbered step
    print("\n--- LightGBM Feature Importance ---")
    # Train a single LGBM model on full training data to get overall feature importance
    full_lgbm_model = lgb.LGBMClassifier(**config.lgb_params)
    full_lgbm_model.fit(train_df[feature_cols], train_df[config.target_col])
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': full_lgbm_model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20), palette="magma")
    plt.title("Top 20 LightGBM Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_path, 'lgbm_feature_importance.png'))
    plt.show()
    print("Saved: lgbm_feature_importance.png")

    # --- 5.4. Prediction Probability Histograms --- # Renumbered step
    print("\n--- Prediction Probability Histograms (OOF) ---")
    plt.figure(figsize=(20, 5)) # Wider figure for 4 subplots

    plt.subplot(1, 4, 1) # Changed to 1 row, 4 columns
    sns.histplot(oof_deberta, bins=30, kde=True, color='skyblue')
    plt.title("DeBERTa OOF Probabilities")
    plt.xlabel("Predicted Probability (Text 2 is Real)")
    plt.ylabel("Count")

    plt.subplot(1, 4, 2) # Changed to 1 row, 4 columns
    sns.histplot(oof_lgbm, bins=30, kde=True, color='lightcoral')
    plt.title("LightGBM OOF Probabilities")
    plt.xlabel("Predicted Probability (Text 2 is Real)")
    plt.ylabel("Count")

    plt.subplot(1, 4, 3) # Changed to 1 row, 4 columns
    sns.histplot(oof_roberta, bins=30, kde=True, color='lightgreen') # Added RoBERTa
    plt.title("RoBERTa OOF Probabilities")
    plt.xlabel("Predicted Probability (Text 2 is Real)")
    plt.ylabel("Count")

    plt.subplot(1, 4, 4) # Changed to 1 row, 4 columns
    sns.histplot(meta_model.predict_proba(meta_X_train)[:,1], bins=30, kde=True, color='mediumseagreen')
    plt.title("Level 2 Ensemble OOF Probabilities")
    plt.xlabel("Predicted Probability (Text 2 is Real)")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(os.path.join(config.output_path, 'oof_probability_histograms.png'))
    plt.show()
    print("Saved: oof_probability_histograms.png")

    # --- 5.5. LLM Judge Verdict Distribution --- # Renumbered step
    print("\n--- LLM Judge Verdict Distribution ---")
    # Map numerical verdicts back to labels for plotting
    verdict_map = {1: 'Text A Real', 2: 'Text B Real', 0: 'UNCLEAR'}
    llm_judge_counts = pd.Series(train_df['llm_judge_verdict']).map(verdict_map).value_counts()

    plt.figure(figsize=(8, 5))
    sns.barplot(x=llm_judge_counts.index, y=llm_judge_counts.values, palette="rocket")
    plt.title("LLM Judge Verdict Distribution on Training Data")
    plt.xlabel("Verdict")
    plt.ylabel("Count")
    for index, value in enumerate(llm_judge_counts.values):
        plt.text(index, value + 5, str(value), ha='center')
    plt.savefig(os.path.join(config.output_path, 'llm_judge_distribution.png'))
    plt.show()
    print("Saved: llm_judge_distribution.png")

    # --- 5.6. Error Analysis Scatter Plot (Conceptual Example) --- # Renumbered step
    print("\n--- Error Analysis: Misclassified Samples (Conceptual) ---")
    # For this, we need the true labels and OOF predictions
    oof_predictions_rounded = np.round(meta_model.predict_proba(meta_X_train)[:,1])
    is_misclassified = (oof_predictions_rounded != train_df[config.target_col]).astype(int)
    
    # Pick two differential features for a scatter plot (e.g., char_count_diff and word_count_diff)
    # Ensure these columns exist in train_df
    feat1 = 'char_count_diff'
    feat2 = 'word_count_diff'

    if feat1 in train_df.columns and feat2 in train_df.columns:
        plot_df = train_df.copy()
        plot_df['is_misclassified'] = is_misclassified
        plot_df['true_label'] = train_df[config.target_col]

        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=feat1,
            y=feat2,
            hue='true_label', # Color by true label
            style='is_misclassified', # Style by misclassification
            data=plot_df,
            alpha=0.6,
            s=100, # Size of markers
            palette={0: 'blue', 1: 'orange'} # Colors for true labels
        )
        # Highlight misclassified points with a different marker or color
        sns.scatterplot(
            x=feat1,
            y=feat2,
            data=plot_df[plot_df['is_misclassified'] == 1],
            color='red',
            marker='X', # 'X' marker for misclassified
            s=200, # Larger size for misclassified
            label='Misclassified',
            zorder=5 # Draw on top
        )
        plt.title(f"Error Analysis: {feat1} vs {feat2} (Misclassified Points Highlighted)")
        plt.xlabel(feat1)
        plt.ylabel(feat2)
        plt.legend(title='True Label / Status')
        plt.tight_layout()
        plt.savefig(os.path.join(config.output_path, 'error_analysis_scatter.png'))
        plt.show()
        print("Saved: error_analysis_scatter.png")
    else:
        print(f"Skipping error analysis scatter plot: Features '{feat1}' or '{feat2}' not found.")

    print("\nAll visualizations generated and saved to output directory.")

    # --- 5.7. Reliability Diagram (Calibration Curve) ---
    print("\n--- Reliability Diagram (Model Calibration) ---")
    # For Level 2 Ensemble
    prob_true, prob_pred = calibration_curve(train_df[config.target_col], meta_model.predict_proba(meta_X_train)[:,1], n_bins=10)
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.plot(prob_pred, prob_true, marker='o', label='Level 2 Ensemble')
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Reliability Diagram (Level 2 Ensemble)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.output_path, 'reliability_diagram.png'))
    plt.show()
    print("Saved: reliability_diagram.png")

    # --- 5.8. Feature Distribution (Real vs. Fake) for Key Differential Features ---
    print("\n--- Feature Distributions for Real vs. Fake ---")
    # Select a few important differential features from LGBM importance
    key_diff_features = ['char_count_diff', 'word_count_diff', 'flesch_reading_ease_ratio', 'perplexity_diff'] 
    
    plot_df_dist = train_df.copy()
    plot_df_dist['true_label_str'] = plot_df_dist[config.target_col].map({0: 'Text 1 Real', 1: 'Text 2 Real'})

    for feat in key_diff_features:
        if feat in plot_df_dist.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=plot_df_dist, x=feat, hue='true_label_str', kde=True, bins=30, palette='coolwarm')
            plt.title(f"Distribution of {feat} by True Label")
            plt.xlabel(feat)
            plt.ylabel("Count")
            plt.legend(title='True Label')
            plt.tight_layout()
            plt.savefig(os.path.join(config.output_path, f'feature_distribution_{feat}.png'))
            plt.show()
            print(f"Saved: feature_distribution_{feat}.png")
        else:
            print(f"Skipping feature distribution plot for '{feat}': Feature not found.")

