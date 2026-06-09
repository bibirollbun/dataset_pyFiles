import logging
import warnings
import os

# === SUPPRESS NON-FATAL WARNINGS ===
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # For TensorFlow/CUDA if used
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("pydantic").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Attempting to register.*factory")
warnings.filterwarnings("ignore", message="The 'repr' attribute.*")


# === INSTALL textstat FROM LOCAL WHEEL (NO INTERNET) ===
import sys
import os

WHEEL_PATH = "/kaggle/input/lmsys-deps/textstat-0.7.11-py3-none-any.whl"

if os.path.exists(WHEEL_PATH):
    print(f"Installing textstat from {WHEEL_PATH}...")
    !pip install --no-index --find-links /kaggle/input/lmsys-deps/ textstat
else:
    raise FileNotFoundError(f"Wheel file not found at {WHEEL_PATH}")

# Now import
import textstat
print("✅ textstat installed and imported successfully.")


# === INSTALL bitsandbytes FROM LOCAL WHEEL (WITH METADATA FIX) ===
import os

BITSANDBYTES_WHEEL = "/kaggle/input/bitsbytesdeps/bitsandbytes-0.48.2-py3-none-manylinux_2_24_x86_64.whl"

if not os.path.exists(BITSANDBYTES_WHEEL):
    raise FileNotFoundError(f"bitsandbytes wheel not found at {BITSANDBYTES_WHEEL}")

print("Installing bitsandbytes from local wheel...")
!pip install --no-index --find-links /kaggle/input/bitsbytesdeps/ bitsandbytes

# Verify installation and metadata
try:
    import importlib.metadata
    version = importlib.metadata.version("bitsandbytes")
    print(f"✅ bitsandbytes {version} installed successfully.")
except Exception as e:
    print(f"❌ Failed to load bitsandbytes metadata: {e}")
    # Fallback: manually patch if needed (rare)
    pass


# === FULL OFFLINE XGBOOST TEST PREDICTION SCRIPT ===
# Designed for Kaggle (internet disabled)
# Author: Tan Wee Choon Waylen
# Input: test.csv
# Output: xgboost_test_pred.csv

import os
import pandas as pd
import numpy as np
import torch
import json
import ast
import re
from tqdm import tqdm
import logging
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import textstat
# -----------------------------
# LOGGING SETUP
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Starting offline XGBoost test prediction pipeline.")

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_FILE = "/kaggle/input/lmsys-chatbot-arena/test.csv"  # Provided by Kaggle
JUDGE_MODEL_PATH = "/kaggle/input/qwen15-4b-chat-offline"  # ← Upload Qwen1.5-4B-Chat here
CHECKPOINT_DIR = "checkpoints_llm_judge"
CHUNK_SIZE = 50
BATCH_SIZE_JUDGE = 1
MAX_NEW_TOKENS = 150
MAX_SEQ_LENGTH = 512 # For sentiment tokenizer
PREDICTION_OUTPUT = "submission.csv"
XGB_MODEL_PATH = "xgboost_final_model.json"  # Ensure this is uploaded

os.makedirs(CHECKPOINT_DIR, exist_ok=True)



import os

SIM_MODEL_PATH = "/kaggle/input/all-minilm-l6-v2-offline"
print("\nContents of model directory:")
for root, dirs, files in os.walk(SIM_MODEL_PATH):
    level = root.replace(SIM_MODEL_PATH, '').count(os.sep)
    indent = "  " * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = "  " * (level + 1)
    for f in files[:5]:  # Show first 5 files per dir
        print(f"{subindent}{f}")
    if len(files) > 5:
        print(f"{subindent}... (+{len(files)-5} more)")


# JUDGE_MODEL_PATH = "/kaggle/input/qwen1-5-4b-chat-offline"
SIM_MODEL_PATH = "/kaggle/input/all-minilm-l6-v2-offline"
SENT_MODEL_PATH = "/kaggle/input/twitter-roberta-sentiment-offline"
device = "cuda" if torch.cuda.is_available() else "cpu"


# === LOAD all-MiniLM-L6-v2 USING TRANSFORMERS (OFFLINE-SAFE) ===
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SIM_MODEL_PATH = "/kaggle/input/all-minilm-l6-v2-offline"

# Verify path exists
import os
if not os.path.exists(SIM_MODEL_PATH):
    raise FileNotFoundError(f"Model path not found: {SIM_MODEL_PATH}")

print(f"Loading model from: {SIM_MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(SIM_MODEL_PATH, local_files_only=True)
model = AutoModel.from_pretrained(SIM_MODEL_PATH, local_files_only=True)
model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

print("✅ Model loaded successfully in offline mode.")


import pandas as pd
test_df = pd.read_csv("/kaggle/input/lmsys-chatbot-arena/test.csv")
print("Test shape:", test_df.shape)
test_df.head()


# === INSTALL DEPENDENCIES (OFFLINE) ===
# Note: Upload wheels for textblob and any other missing packages as datasets.
# For example, upload textblob-0.18.0.post0-py3-none-any.whl to /kaggle/input/textblob-wheel/
# Similarly for other packages if needed.
# textstat and bitsandbytes are already handled in the provided code.



# Other imports (assume available)
import pandas as pd
import numpy as np
import textstat
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoModelForCausalLM, pipeline
import torch.nn.functional as F
from tqdm import tqdm
import ast
import logging
import time
import re
import json
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === CONFIGURATION ===
# DATA_FILE = '/kaggle/input/lmsys-chatbot-arena/test.csv'  # Test data
HAS_LABELS = False  # Test data has no labels

# Model paths (upload as datasets)
SENTIMENT_MODEL_PATH = "/kaggle/input/twitter-roberta-sentiment-offline"  # Upload this model
SIMILARITY_MODEL_PATH = "/kaggle/input/all-minilm-l6-v2-offline"  # Upload this
SENTIMENT_LABELS = ['negative', 'neutral', 'positive']
# JUDGE_MODEL_PATH = "/kaggle/input/qwen1-5-4b-chat-offline"  # Upload as dataset
CHECKPOINT_DIR = "checkpoints_llm_judge"

# Processing Parameters
BATCH_SIZE = 4
BATCH_SIZE_JUDGE = 1  # Increased for throughput
MAX_NEW_TOKENS = 150
MAX_SEQ_LENGTH = 512
CHUNK_SIZE = 50  # For checkpointing
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Output
FEATURE_OUTPUT = "lmsys_test_features_final.csv"

# === LOAD DATA ===
logger.info(f"Loading data from '{DATA_FILE}'...")
try:
    df_data = pd.read_csv(DATA_FILE, engine='python', on_bad_lines='skip')
    logger.info(f"Successfully loaded '{DATA_FILE}'. Shape: {df_data.shape}")
except FileNotFoundError:
    logger.error(f"Error: '{DATA_FILE}' not found.")
    raise
except Exception as e:
    logger.error(f"Error loading '{DATA_FILE}': {e}")
    raise

# === INITIALIZE MODELS ===
# --- Sentiment Analysis ---
logger.info(f"Loading Sentiment Analysis Model from: {SENTIMENT_MODEL_PATH}")
sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_PATH, local_files_only=True)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_PATH, local_files_only=True)
sentiment_model.eval()
if torch.cuda.is_available():
    sentiment_model = sentiment_model.to('cuda')
    logger.info("Sentiment model moved to GPU.")

# --- Semantic Similarity (using transformers as in provided code) ---
logger.info(f"Loading Semantic Similarity Model from: {SIMILARITY_MODEL_PATH}")
sim_tokenizer = AutoTokenizer.from_pretrained(SIMILARITY_MODEL_PATH, local_files_only=True)
sim_model = AutoModel.from_pretrained(SIMILARITY_MODEL_PATH, local_files_only=True)
sim_model.eval()
if torch.cuda.is_available():
    sim_model = sim_model.to('cuda')
    logger.info("Similarity model moved to GPU.")

# Mean Pooling Function
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Batched Encoding Function for Similarity
def encode_sentences(sentences, model, tokenizer, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # --- Robust cleaning: ensure every sentence is a clean string ---
    cleaned = []
    for sent in sentences:
        if sent is None or pd.isna(sent):
            cleaned.append("")
        elif isinstance(sent, (int, float)):
            cleaned.append(str(int(sent)) if not pd.isna(sent) else "")
        else:
            try:
                cleaned.append(str(sent).strip())
            except Exception:
                cleaned.append("")
    # --------------------------------------------------------------

    embeddings = []
    dataset = [cleaned[i:i+batch_size] for i in range(0, len(cleaned), batch_size)]
    model.to(device)
    
    with torch.no_grad():
        for batch_texts in tqdm(dataset, desc="Encoding Batches"):
            # Final safety wrap
            batch_texts = [
                txt if isinstance(txt, str) and len(txt.strip()) > 0 else ""
                for txt in batch_texts
            ]
            
            # Tokenize
            try:
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(device)
                out = model(**encoded)
                pooled = mean_pooling(out.last_hidden_state, encoded['attention_mask'])
                pooled = F.normalize(pooled, p=2, dim=1)
                embeddings.append(pooled.cpu().numpy())
            except Exception as e:
                logger.error(f"Tokenization failed on batch: {e}")
                # Fallback: zero embedding
                dummy = np.zeros((len(batch_texts), 384))  # Adjust dim if needed
                embeddings.append(dummy)

    return np.concatenate(embeddings, axis=0)

# === HELPER FUNCTIONS ===
def safe_literal_eval(text):
    if not isinstance(text, str):
        return text
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text

def get_last_turn(prompt_list):
    if not isinstance(prompt_list, list) or not prompt_list:
        return ""  # Always return string
    user_turns = [t for i, t in enumerate(prompt_list) if i % 2 == 0]  # First is user
    return str(user_turns[-1]) if user_turns else ""

def get_textblob_features(text):
    if not isinstance(text, str) or not text.strip():
        return {"textblob_polarity": np.nan, "textblob_subjectivity": np.nan}
    try:
        blob = TextBlob(text)
        return {
            "textblob_polarity": float(blob.sentiment.polarity),
            "textblob_subjectivity": float(blob.sentiment.subjectivity)
        }
    except Exception as e:
        logger.warning(f"TextBlob error: {e}")
        return {"textblob_polarity": np.nan, "textblob_subjectivity": np.nan}

def get_readability_scores(text):
    if not isinstance(text, str) or not text.strip():
        return {"flesch_kincaid_grade": np.nan, "gunning_fog": np.nan}
    try:
        return {
            "flesch_kincaid_grade": float(textstat.flesch_kincaid_grade(text)),
            "gunning_fog": float(textstat.gunning_fog(text)),
        }
    except Exception as e:
        logger.warning(f"Readability error: {e}")
        return {"flesch_kincaid_grade": np.nan, "gunning_fog": np.nan}

class TextListDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        return text

def batch_get_sentiment_scores(texts, tokenizer, model, labels, batch_size=32, max_length=512):
    dataset = TextListDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_scores = []
    device = next(model.parameters()).device
    logger.info(f"Calculating sentiment for {len(texts)} texts...")
    for text_batch in tqdm(dataloader, desc="Sentiment Batches"):
        try:
            inputs = tokenizer(list(text_batch), return_tensors="pt", truncation=True, padding=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            batch_scores = predictions.cpu().numpy()
            for scores in batch_scores:
                score_dict = {f"sentiment_{label}": float(score) for label, score in zip(labels, scores)}
                all_scores.append(score_dict)
        except Exception as e:
            logger.error(f"Sentiment error: {e}")
            for _ in text_batch:
                neutral_scores = {f"sentiment_{label}": 0.0 if label != 'neutral' else 1.0 for label in labels}
                all_scores.append(neutral_scores)
    return all_scores

class TextPairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        prompt, response = self.pairs[idx]
        return str(prompt), str(response)

def batch_get_semantic_similarity(prompt_response_pairs, model, tokenizer, batch_size=32):
    similarities = []
    prompts, responses = zip(*prompt_response_pairs)
    prompt_emb = encode_sentences(list(prompts), model, tokenizer, batch_size)
    resp_emb = encode_sentences(list(responses), model, tokenizer, batch_size)
    cos_sims = np.sum(prompt_emb * resp_emb, axis=1)
    similarities = list(cos_sims)
    return similarities

def clean_text(x):
    return re.sub(r"\s+", " ", str(x)).strip()

def parse_list_first(text):
    if pd.isna(text):
        return ""
    s = str(text).strip()
    try:
        lst = json.loads(s)
        return lst[0] if isinstance(lst, list) and len(lst) else ""
    except Exception:
        pass
    s2 = s.replace(r"\/", "/").replace("null", "''")
    try:
        lst = ast.literal_eval(s2)
        return lst[0] if isinstance(lst, list) and len(lst) else ""
    except Exception:
        return ""

def build_prompt_purpose_strict(df, prompt_col='prompt_clean', proximity_window=60):
    s = df[prompt_col].astype('string').fillna('')
    lo = s.str.lower()
    P = pd.DataFrame(index=df.index)
    P['p_codefences'] = s.str.count(r'```')
    P['p_bullets'] = s.str.count(r'(?m)^\s*[-*•]\s+')
    P['p_numlist'] = s.str.count(r'(?m)^\s*[0-9]{1,2}[.)]\s+')
    P['p_list_lines'] = (P['p_bullets'] + P['p_numlist']).astype('int16')
    tail_stripped = s.str.replace(r'[\s"”’\')\]]+$', '', regex=True)
    P['p_is_question'] = tail_stripped.str.endswith('?').fillna(False).astype('int8')
    steps_kw = r'\b(?:step[- ]?by[- ]?step|steps?|bullet(?:ed)?|checklist|enumerate|numbered list|procedure|instructions?|outline)\b'
    P['p_asks_steps'] = (
        lo.str.contains(steps_kw, regex=True, na=False) |
        (P['p_list_lines'] > 0)
    ).astype('int8')
    langs = r'(?:python|java|javascript|typescript|c\+\+|c#|go|rust|ruby|php|sql|bash|powershell|kotlin|swift|matlab|r|scala|perl|haskell|lua|dart|c)'
    code_nouns = r'(?:code|function|method|class|script|snippet|program|algorithm|regex|query|api|unit test|unit tests|test case|module|package|library|endpoint)'
    code_verbs = r'(?:write|implement|provide|show|give|generate|create|produce|build|define|return|refactor)'
    W = proximity_window
    prox_verb_noun = rf'\b{code_verbs}\b[\s\S]{{0,{W}}}\b{code_nouns}\b'
    prox_lang_noun = rf'\b{langs}\b[\s\S]{{0,{W}}}\b{code_nouns}\b'
    prox_lang_verb = rf'\b{langs}\b[\s\S]{{0,{W}}}\b{code_verbs}\b'
    in_lang_phrase = rf'\b(?:in|using)\s+{langs}\b'
    P['p_asks_code'] = (
        (P['p_codefences'] > 0) |
        lo.str.contains(prox_verb_noun, regex=True, na=False) |
        lo.str.contains(prox_lang_noun, regex=True, na=False) |
        lo.str.contains(prox_lang_verb, regex=True, na=False) |
        lo.str.contains(rf'\b{code_nouns}\b\s+(?:example|sample)\b', regex=True, na=False) |
        lo.str.contains(rf'{in_lang_phrase}[\s\S]{{0,{W}}}\b{code_nouns}\b', regex=True, na=False)
    ).astype('int8')
    math_kw = r'\b(?:equation|solve|solution|derivative|integral|limit|matrix|vector|probability|statistics?|theorem|proof|algebra|calculus|gradient|expectation|variance|distribution)\b'
    latex = r"\$[^\$]+\$|\\\(|\\\)|\\begin\{equation"
    P['p_asks_math'] = (
        lo.str.contains(math_kw, regex=True, na=False) |
        s.str.contains(latex, regex=True, na=False)
    ).astype('int8')
    advice_kw = (
        r'(?:\bwhat should i\b|\bhow should i\b|\bshould (?:i|we)\b|'
        r'\badvice\b|\badvise\b|\brecommend(?:ation)?s?\b|'
        r'\bpros and cons\b|\bis it (?:okay|ok|ethical|right|wrong|bad|good)\b|'
        r'\bmorally\b|\bwhat do you think\b)'
    )
    P['p_asks_advice'] = lo.str.contains(advice_kw, regex=True, na=False).astype('int8')
    compare_kw = r'(?:\bcompare\b|\bcomparison\b|\bdifference between\b|\bversus\b| vs\.? |\bwhich is better\b|\bbetter than\b)'
    P['p_compare'] = lo.str.contains(compare_kw, regex=True, na=False).astype('int8')
    summarize_kw = r'(?:\bsummariz(?:e|ation)\b|\bsummary\b|\btl;dr\b|\bcondense\b|\bbrief overview\b|\bkey points\b|\boutline the main points\b)'
    P['p_summarize'] = lo.str.contains(summarize_kw, regex=True, na=False).astype('int8')
    rewrite_kw = r'(?:\brewrite\b|\brephrase\b|\bparaphrase\b|\bpolish\b|\bedit for clarity\b|\bimprove (?:the )?writing\b|\bmake (?:it )?(?:formal|polite|concise)\b|\bfix grammar\b)'
    P['p_rewrite'] = lo.str.contains(rewrite_kw, regex=True, na=False).astype('int8')
    langs_words = r'(?:spanish|french|german|chinese|japanese|korean|hindi|arabic|portuguese|italian|russian|turkish|vietnamese|thai|indonesian|dutch|swedish|polish|greek)'
    translate_kw = rf'(?:\btranslate\b|\btranslate .* into (?:{langs_words})\b|\bto (?:{langs_words})\b)'
    P['p_translate'] = lo.str.contains(translate_kw, regex=True, na=False).astype('int8')
    classify_kw = r'(?:\bclassif(?:y|ication)\b|\blabel\b|\bcategorize\b|\bdetermine whether\b|\btrue or false\b|\byes or no\b|\bspam\b)'
    P['p_classify'] = lo.str.contains(classify_kw, regex=True, na=False).astype('int8')
    return P.astype({'p_codefences': 'int8', 'p_bullets': 'int8', 'p_numlist': 'int8', 'p_list_lines': 'int8', 'p_is_question': 'int8', 'p_asks_steps': 'int8', 'p_asks_code': 'int8', 'p_asks_math': 'int8', 'p_asks_advice': 'int8', 'p_compare': 'int8', 'p_summarize': 'int8', 'p_rewrite': 'int8', 'p_translate': 'int8', 'p_classify': 'int8'})

def _mk_len_struct_useful(series, prefix):
    s = series.astype('string').fillna('')
    F = pd.DataFrame(index=s.index)
    F[f'{prefix}chars'] = s.str.len()
    F[f'{prefix}words'] = s.str.split().str.len()
    F[f'{prefix}sents'] = s.str.count(r'[.!?]+').clip(lower=1)
    F[f'{prefix}paragraphs'] = (s.str.count(r'\n\s*\n') + 1).where(F[f'{prefix}chars'] > 0, 0)
    F[f'{prefix}codefences'] = s.str.count(r"```")
    F[f'{prefix}headings'] = s.str.count(r"(?m)^\s*#{1,6}\s+")
    F[f'{prefix}bullets'] = s.str.count(r"(?m)^\s*[-*•]\s+")
    F[f'{prefix}numlist'] = s.str.count(r"(?m)^\s*[0-9]{1,2}[.)]\s+")
    F[f'{prefix}list_lines'] = F[f'{prefix}bullets'] + F[f'{prefix}numlist']
    F[f'{prefix}qmarks'] = s.str.count(r"\?")
    F[f'{prefix}exclaims'] = s.str.count(r"!")
    ww = F[f'{prefix}words'].replace(0, np.nan)
    for k in ['qmarks','exclaims','list_lines','codefences','headings']:
        F[f'{prefix}{k}_per100w'] = (F[f'{prefix}{k}'] / ww * 100).fillna(0)
    return F

def build_lenstruct_useful(df, prompt_col='prompt_clean', resp_a_col='response_a_clean', resp_b_col='response_b_clean'):
    A = _mk_len_struct_useful(df[resp_a_col], 'a_')
    B = _mk_len_struct_useful(df[resp_b_col], 'b_')
    X = pd.concat([A, B], axis=1)
    kept_bases = [c[2:] for c in A.columns if c[2:] not in ('bullets','numlist')]
    for k in kept_bases:
        X[f'diff_{k}'] = X[f'a_{k}'] - X[f'b_{k}']
        X[f'ratio_{k}'] = (X[f'a_{k}'] + 1e-6) / (X[f'b_{k}'] + 1e-6)
    p_words = df[prompt_col].astype('string').str.split().str.len().replace(0, np.nan)
    X['a_to_prompt_word_ratio'] = (X['a_words'] / p_words).fillna(0)
    X['b_to_prompt_word_ratio'] = (X['b_words'] / p_words).fillna(0)
    X['a_longer_word'] = (X['a_words'] > X['b_words']).astype('int8')
    X['a_longer_char'] = (X['a_chars'] > X['b_chars']).astype('int8')
    return X.astype('float32', errors='ignore')

# === MAIN PROCESSING ===
start_time = time.time()

# Parse and flatten
df_data['parsed_prompt'] = df_data['prompt'].apply(safe_literal_eval)
df_data['parsed_response_a'] = df_data['response_a'].apply(safe_literal_eval)
df_data['parsed_response_b'] = df_data['response_b'].apply(safe_literal_eval)
df_data['prompt_last_turn'] = df_data['parsed_prompt'].apply(get_last_turn)
def safe_join(x):
    if isinstance(x, list):
        return " ".join([str(item) for item in x if item is not None])
    return str(x) if x is not None else ""
df_data['flat_prompt'] = df_data['parsed_prompt'].apply(safe_join)
df_data['flat_response_a'] = df_data['parsed_response_a'].apply(safe_join)
df_data['flat_response_b'] = df_data['parsed_response_b'].apply(safe_join)

df_data["prompt_clean"] = df_data["prompt"].map(parse_list_first).map(clean_text)
df_data["response_a_clean"] = df_data["response_a"].map(parse_list_first).map(clean_text)
df_data["response_b_clean"] = df_data["response_b"].map(parse_list_first).map(clean_text)

# Collect for batch
texts_a = df_data['flat_response_a'].tolist()
texts_b = df_data['flat_response_b'].tolist()
# Do this:
df_data['prompt_last_turn'] = df_data['prompt_last_turn'].fillna("").astype(str)
df_data['flat_response_a'] = df_data['flat_response_a'].fillna("").astype(str)
df_data['flat_response_b'] = df_data['flat_response_b'].fillna("").astype(str)
pairs_a = list(zip(df_data['prompt_last_turn'], df_data['flat_response_a']))
pairs_b = list(zip(df_data['prompt_last_turn'], df_data['flat_response_b']))

assert all(isinstance(p, str) for p, r in pairs_a), "Non-string prompt found"
assert all(isinstance(r, str) for p, r in pairs_a), "Non-string response found"
print("Sample prompt:", repr(pairs_a[0][0]))
print("Sample response:", repr(pairs_a[0][1]))
print("Types:", type(pairs_a[0][0]), type(pairs_a[0][1]))

# Batched features
sentiment_scores_a = batch_get_sentiment_scores(texts_a, sentiment_tokenizer, sentiment_model, SENTIMENT_LABELS, BATCH_SIZE, MAX_SEQ_LENGTH)
sentiment_scores_b = batch_get_sentiment_scores(texts_b, sentiment_tokenizer, sentiment_model, SENTIMENT_LABELS, BATCH_SIZE, MAX_SEQ_LENGTH)
similarity_scores_a = batch_get_semantic_similarity(pairs_a, sim_model, sim_tokenizer, BATCH_SIZE)
similarity_scores_b = batch_get_semantic_similarity(pairs_b, sim_model, sim_tokenizer, BATCH_SIZE)

# Assign
df_data['semantic_similarity_A'] = similarity_scores_a
df_data['semantic_similarity_B'] = similarity_scores_b
df_sentiment_a = pd.DataFrame(sentiment_scores_a)
df_sentiment_b = pd.DataFrame(sentiment_scores_b)
df_sentiment_a.columns = [f"{col}_A" for col in df_sentiment_a.columns]
df_sentiment_b.columns = [f"{col}_B" for col in df_sentiment_b.columns]

# Features DF
df_features = df_data[['id']].copy()
df_features = pd.concat([df_features, df_sentiment_a, df_sentiment_b], axis=1)
df_features['semantic_similarity_A'] = df_data['semantic_similarity_A']
df_features['semantic_similarity_B'] = df_data['semantic_similarity_B']

# Non-batched
df_features['textblob_polarity_A'] = df_data['flat_response_a'].apply(lambda x: get_textblob_features(x)['textblob_polarity'])
df_features['textblob_subjectivity_A'] = df_data['flat_response_a'].apply(lambda x: get_textblob_features(x)['textblob_subjectivity'])
df_features['textblob_polarity_B'] = df_data['flat_response_b'].apply(lambda x: get_textblob_features(x)['textblob_polarity'])
df_features['textblob_subjectivity_B'] = df_data['flat_response_b'].apply(lambda x: get_textblob_features(x)['textblob_subjectivity'])
df_features['flesch_kincaid_grade_A'] = df_data['flat_response_a'].apply(lambda x: get_readability_scores(x)['flesch_kincaid_grade'])
df_features['gunning_fog_A'] = df_data['flat_response_a'].apply(lambda x: get_readability_scores(x)['gunning_fog'])
df_features['flesch_kincaid_grade_B'] = df_data['flat_response_b'].apply(lambda x: get_readability_scores(x)['flesch_kincaid_grade'])
df_features['gunning_fog_B'] = df_data['flat_response_b'].apply(lambda x: get_readability_scores(x)['gunning_fog'])

# Differences
feature_pairs = [
    ('sentiment_negative', 'sentiment_neutral', 'sentiment_positive'),
    ('textblob_polarity', 'textblob_subjectivity'),
    ('flesch_kincaid_grade', 'gunning_fog'),
    ('semantic_similarity',)
]
for feat_tuple in feature_pairs:
    for feat_base in feat_tuple:
        feat_a = f"{feat_base}_A"
        feat_b = f"{feat_base}_B"
        if feat_a in df_features.columns and feat_b in df_features.columns:
            df_features[f"{feat_base}_diff_A_minus_B"] = df_features[feat_a] - df_features[feat_b]

# Lengths
df_features['len_prompt'] = df_data['parsed_prompt'].apply(lambda x: len(" ".join(x)) if isinstance(x, list) else len(str(x)))
df_features['len_response_a'] = df_data['flat_response_a'].apply(len)
df_features['len_response_b'] = df_data['flat_response_b'].apply(len)
df_features['len_diff_A_minus_B'] = df_features['len_response_a'] - df_features['len_response_b']

# Extended features
pstruct = build_prompt_purpose_strict(df_data)
lenstruct = build_lenstruct_useful(df_data)
df_extended = pd.concat([pstruct, lenstruct], axis=1)

# Merge
# Reset index if needed
if df_features.index.name == 'id' or 'id' not in df_features.columns:
    df_features = df_features.reset_index()

# Create extended features DataFrame with 'id'
df_extended_with_id = pd.concat([df_extended], axis=1)
df_extended_with_id['id'] = df_data['id'].values

# Now merge safely
df_features = df_features.merge(df_extended_with_id, on='id', how='left')


# Merge
# === NO JUDGE FEATURES TO MERGE ===
df_final = df_features.copy()
logger.info(f"No judge features. Final feature shape: {df_final.shape}")

# === FINAL COLUMN REORDERING (EXACT REQUIRED ORDER) ===
logger.info("Reordering columns to match exact required specification...")

# Define the exact desired column order
desired_order = [
    'id',
    # Sentiment scores: A (neg, neu, pos), then B
    'sentiment_negative_A', 'sentiment_neutral_A', 'sentiment_positive_A',
    'sentiment_negative_B', 'sentiment_neutral_B', 'sentiment_positive_B',

    # Semantic similarity
    'semantic_similarity_A', 'semantic_similarity_B',

    # TextBlob features
    'textblob_polarity_A', 'textblob_subjectivity_A',
    'textblob_polarity_B', 'textblob_subjectivity_B',

    # Readability scores
    'flesch_kincaid_grade_A', 'gunning_fog_A',
    'flesch_kincaid_grade_B', 'gunning_fog_B',

    # Difference features (A - B)
    'sentiment_positive_diff_A_minus_B',
    'sentiment_negative_diff_A_minus_B',
    'sentiment_neutral_diff_A_minus_B',
    'textblob_polarity_diff_A_minus_B',
    'textblob_subjectivity_diff_A_minus_B',
    'flesch_kincaid_grade_diff_A_minus_B',
    'gunning_fog_diff_A_minus_B',
    'semantic_similarity_diff_A_minus_B',

    # Length features
    'len_prompt', 'len_response_a', 'len_response_b', 'len_diff_A_minus_B',

    # Prompt purpose flags (from build_prompt_purpose_strict)
    'p_codefences', 'p_bullets', 'p_numlist', 'p_list_lines', 'p_is_question',
    'p_asks_steps', 'p_asks_code', 'p_asks_math', 'p_asks_advice', 'p_compare',
    'p_summarize', 'p_rewrite', 'p_translate', 'p_classify',

    # Response structural features for A
    'a_chars', 'a_words', 'a_sents', 'a_paragraphs',
    'a_codefences', 'a_headings', 'a_bullets', 'a_numlist', 'a_list_lines',
    'a_qmarks', 'a_exclaims',
    'a_qmarks_per100w', 'a_exclaims_per100w', 'a_list_lines_per100w',
    'a_codefences_per100w', 'a_headings_per100w',

    # Response structural features for B
    'b_chars', 'b_words', 'b_sents', 'b_paragraphs',
    'b_codefences', 'b_headings', 'b_bullets', 'b_numlist', 'b_list_lines',
    'b_qmarks', 'b_exclaims',
    'b_qmarks_per100w', 'b_exclaims_per100w', 'b_list_lines_per100w',
    'b_codefences_per100w', 'b_headings_per100w',

    # Differences and ratios between A and B
    'diff_chars', 'ratio_chars',
    'diff_words', 'ratio_words',
    'diff_sents', 'ratio_sents',
    'diff_paragraphs', 'ratio_paragraphs',
    'diff_codefences', 'ratio_codefences',
    'diff_headings', 'ratio_headings',
    'diff_list_lines', 'ratio_list_lines',
    'diff_qmarks', 'ratio_qmarks',
    'diff_exclaims', 'ratio_exclaims',
    'diff_qmarks_per100w', 'ratio_qmarks_per100w',
    'diff_exclaims_per100w', 'ratio_exclaims_per100w',
    'diff_list_lines_per100w', 'ratio_list_lines_per100w',
    'diff_codefences_per100w', 'ratio_codefences_per100w',
    'diff_headings_per100w', 'ratio_headings_per100w',

    # Ratios relative to prompt length
    'a_to_prompt_word_ratio', 'b_to_prompt_word_ratio',

    # Binary indicators
    'a_longer_word', 'a_longer_char'
]

# Verify all required columns exist in df_final
missing_cols = [col for col in desired_order if col not in df_final.columns]
if missing_cols:
    logger.warning(f"Missing columns in df_final: {missing_cols}")
    # Optionally, add missing ones as NaN
    for col in missing_cols:
        df_final[col] = np.nan

# Now reorder
df_final = df_final[desired_order]

logger.info(f"Columns successfully reordered to match required schema. Final shape: {df_final.shape}")

# Final save
df_final.to_csv(FEATURE_OUTPUT, index=False)
logger.info(f"Final merged features saved to {FEATURE_OUTPUT}. Shape: {df_final.shape}")


end_time = time.time()
logger.info(f"Elapsed Time: {end_time - start_time:.2f} seconds")


import os
import logging
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib  # lazy import, available by default on Kaggle
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch.nn.functional as F
from tqdm import tqdm
import ast
import re
import json
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load features and labels
df_features = pd.read_csv(FEATURE_OUTPUT)
X_test = df_features.drop(columns=['id', 'winner_model_a', 'winner_model_b', 'winner_tie'], errors='ignore').values  # Only keep feature columns
ids = df_features['id']  # Use test IDs only

print(f"Features shape: {X_test.shape}")


# # === CONFIG ===
# TEACHER_PATHS = {
#     'qwen': '/kaggle/input/train-test-prob/qwen_oof_predictions.csv',
#     'llama': '/kaggle/input/train-test-prob/llama_oof_predictions.csv',
#     'deberta': '/kaggle/input/train-test-prob/deberta_oof_predictions.csv',
#     'xgboost': '/kaggle/input/xgboof/xgboost_oof_predictions.csv',
# }

# SCORES = {'qwen': 11.812, 'llama': 2.313, 'deberta': 1.098, 'xgboost': 1.0375}
# WEIGHTS = {k: 1/v for k, v in SCORES.items()}
# weight_sum = sum(WEIGHTS.values())
# WEIGHTS = {k: v / weight_sum for k, v in WEIGHTS.items()}
# print("Teacher weights:", {k: round(v, 4) for k, v in WEIGHTS.items()})

# # === STEP 1: Load Teacher OOF Predictions Safely ===
# teacher_prob = np.zeros((len(df_features), 3))

# for name, path in TEACHER_PATHS.items():
#     print(f"\nLoading teacher: {name}")
#     try:
#         oof = pd.read_csv(path)
#     except FileNotFoundError:
#         raise FileNotFoundError(f"Teacher file not found: {path}")

#     print(f"  Raw columns: {list(oof.columns)}")

#     if name == 'deberta':
#         oof = oof.rename(columns={
#             'pred_winner_model_a': 'winner_model_a',
#             'pred_winner_model_b': 'winner_model_b',
#             'pred_winner_tie': 'winner_tie'
#         })
#         print("  → Applied deberta 'pred_' prefix rename")

#     required_cols = ['id', 'winner_model_a', 'winner_model_b', 'winner_tie']
#     missing = [c for c in required_cols if c not in oof.columns]
#     if missing:
#         raise ValueError(f"{name} missing columns: {missing}")

#     oof = df_features[['id']].merge(oof[required_cols], on='id', how='left')

#     probs = oof[['winner_model_a', 'winner_model_b', 'winner_tie']].fillna(1/3).values
#     probs = np.clip(probs, 1e-7, 1 - 1e-7)

#     row_sums = probs.sum(axis=1)
#     if not np.allclose(row_sums, 1.0, atol=1e-3):
#         print(f"⚠️ {name} row sums deviate: min={row_sums.min():.6f}, max={row_sums.max():.6f}")
#         probs = probs / (row_sums.reshape(-1, 1) + 1e-8)

#     teacher_prob += WEIGHTS[name] * probs
#     print(f"  ✓ Added with weight {WEIGHTS[name]:.4f}")

# print("\n✅ All teachers loaded and weighted.")

# # === STEP 2: Temperature Sharpening (Stable) ===
# def sharpen(p, T=5.0):
#     p = np.clip(p, 1e-7, 1 - 1e-7)
#     log_p = np.log(p) / T
#     log_p -= log_p.max(axis=1, keepdims=True)
#     p_sharpened = np.exp(log_p)
#     return p_sharpened / p_sharpened.sum(axis=1, keepdims=True)

# teacher_prob = sharpen(teacher_prob, T=5.0)
# assert np.allclose(teacher_prob.sum(axis=1), 1.0, atol=1e-5), "Teacher prob rows don't sum to 1"

# # === STEP 3: Mix Hard and Soft Labels ===
# alpha = 0.5
# soft_targets = alpha * np.eye(3)[y_hard] + (1 - alpha) * teacher_prob
# soft_targets = np.clip(soft_targets, 1e-7, 1 - 1e-7)
# print("Soft targets sample:\n", soft_targets[:5].round(4))

# # === STEP 4: Handle Missing Values & Scale Features Safely ===
# print("Original X stats:")
# print(f"  Shape: {X.shape}")
# print(f"  NaN count: {np.isnan(X).sum()}")
# print(f"  Inf count: {np.isinf(X).sum()}")

# # === IMPUTE MISSING VALUES ===
# from sklearn.impute import SimpleImputer

# imputer = SimpleImputer(strategy='median')  # Robust to outliers; better than mean
# X_imputed = imputer.fit_transform(X)

# print(f"After imputation - NaN count: {np.isnan(X_imputed).sum()}")

# # === HANDLE INFS (±infinity) ===
# # Replace ±inf with finite bounds
# X_imputed = np.where(np.isinf(X_imputed), np.sign(X_imputed) * 1e6, X_imputed)

# # === SCALE FEATURES ===
# scaler = RobustScaler()  # Or StandardScaler if you prefer
# X_scaled = scaler.fit_transform(X_imputed)

# # Final safety check
# assert not np.isnan(X_scaled).any(), "X_scaled contains NaN after scaling!"
# assert not np.isinf(X_scaled).any(), "X_scaled contains Inf after scaling!"
# print(f"✅ Final X_scaled shape: {X_scaled.shape}, no NaN/Inf")

# # === STEP 5: Define Student MLP — With Gradient Initialization & Smaller Width ===
class StudentMLP(nn.Module):
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)
        )
        # Initialize weights carefully
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Clamp input to prevent extreme values entering network
        x = torch.clamp(x, -1e3, 1e3)
        return self.net(x)

# input_dim = X_scaled.shape[1]
# model = StudentMLP(input_dim).to(device)

# # Test forward pass before training
# with torch.no_grad():
#     test_input = torch.from_numpy(X_scaled[:8]).float().to(device)
#     test_logits = model(test_input)
#     assert not torch.isnan(test_logits).any(), "Model produces NaN in initial forward pass!"
#     print("✅ Model passed initial forward pass sanity check.")

# # === STEP 6: Stable Distillation Loss ===
# class DistillationLoss(nn.Module):
#     def __init__(self, alpha=0.5, T=5.0, lambda_cos=0.1):
#         super().__init__()
#         self.alpha = alpha
#         self.T = T
#         self.lambda_cos = lambda_cos
#         self.ce = nn.CrossEntropyLoss()

#     def forward(self, logits, teacher_probs, labels):
#         # Safety checks
#         if torch.isnan(logits).any():
#             print("❌ logits contain NaN!")
#             logits = torch.nan_to_num(logits, 0.0)
#         if torch.isnan(labels).any():
#             raise ValueError("Labels contain NaN")

#         ce = self.ce(logits, labels)

#         if not isinstance(teacher_probs, torch.Tensor):
#             teacher_probs = torch.from_numpy(teacher_probs).float().to(logits.device)
#         teacher_probs = teacher_probs.clamp(1e-7, 1 - 1e-7)

#         log_student = F.log_softmax(logits / self.T, dim=1)
#         teacher_soft = F.softmax(teacher_probs / self.T, dim=1).clamp(1e-7, 1 - 1e-7)

#         kl = F.kl_div(log_student, teacher_soft, reduction='batchmean') * (self.T ** 2)

#         student_soft = F.softmax(logits / self.T, dim=1).clamp(1e-7, 1 - 1e-7)
#         cos_sim = F.cosine_similarity(student_soft, teacher_soft.detach(), dim=1).mean()
#         cos_loss = 1.0 - cos_sim

#         total_loss = self.alpha * ce + (1 - self.alpha) * kl + self.lambda_cos * cos_loss
#         return total_loss

# criterion = DistillationLoss(alpha=0.5, T=5.0, lambda_cos=0.1)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)  # Lower LR

# # === STEP 7: Train Model — With Full Debugging ===
# dataset = TensorDataset(
#     torch.from_numpy(X_scaled).float(),
#     torch.from_numpy(y_hard).long(),
#     torch.from_numpy(soft_targets).float()
# )
# dataloader = DataLoader(dataset, batch_size=256, shuffle=True)  # Reduced batch size

# print("\nStarting training...")

# for epoch in range(300):
#     model.train()
#     total_loss = 0.0
#     step_count = 0

#     for x, y_h, y_t in dataloader:
#         x, y_h, y_t = x.to(device), y_h.to(device), y_t.to(device)

#         # Check input
#         if torch.isnan(x).any() or torch.isinf(x).any():
#             print("❌ Input x contains NaN/Inf")
#             continue

#         optimizer.zero_grad()

#         try:
#             logits = model(x)
#             if torch.isnan(logits).any():
#                 print("❌ Logits are NaN after forward!")
#                 print("Input stats:", x.mean().item(), x.std().item())
#                 print("Logits sample:", logits[0].detach().cpu().numpy())
#                 continue

#             loss = criterion(logits, y_t, y_h)
#             if torch.isnan(loss):
#                 print("❌ Loss is NaN")
#                 continue

#             loss.backward()

#             # Clip gradients
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

#             optimizer.step()

#             total_loss += loss.item()
#             step_count += 1

#         except Exception as e:
#             print(f"❌ Error in training step: {e}")
#             continue

#     if step_count > 0:
#         avg_loss = total_loss / step_count
#     else:
#         avg_loss = float('nan')

#     if epoch % 20 == 0 or True:
#         print(f"Epoch {epoch+1:03d} - Loss: {avg_loss:.4f} | Steps: {step_count}")

# # Save model
# torch.save(model.state_dict(), "distilled_mlp.pth")
# print("\n✅ Training complete. Model saved.")


    # # Inference
    # scaler = joblib.load('/kaggle/input/distillation/scaler.pkl')
    # X_scaled = scaler.transform(X_test)

    # input_dim = X_scaled.shape[1]
    # model = StudentMLP(input_dim).to(device)
    # model.load_state_dict(torch.load('/kaggle/input/distillation/distilled_mlp.pth'))
    # model.eval()

    # preds = []
    # batch_size = 512
    # for i in range(0, len(X_scaled), batch_size):
    #     batch = torch.from_numpy(X_scaled[i:i+batch_size]).float().to(device)
    #     with torch.no_grad():
    #         logits = model(batch)
    #         preds.append(F.softmax(logits, dim=1).cpu().numpy())
    # preds = np.concatenate(preds)

    # submission1 = pd.DataFrame({
    #     'id': ids,
    #     'winner_model_a': preds[:,0],
    #     'winner_model_b': preds[:,1],
    #     'winner_tie': preds[:,2]
    # })
    # submission1.to_csv('submission1.csv', index=False)
    # print("Submission saved - shape:", submission1.shape)
    # display(submission1.head())


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler
import joblib

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD TEST DATA ===
TEST_CSV = FEATURE_OUTPUT
df_test = pd.read_csv(TEST_CSV)

ids = df_test['id'].values
feature_cols = [col for col in df_test.columns if col != 'id']
X_test = df_test[feature_cols].values.astype(np.float32)

print(f"Loaded test data: {X_test.shape}")

# === INLINE IMPUTATION (No imputer.pkl needed) ===
print("Before imputation - NaN:", np.isnan(X_test).sum(), "| Inf:", np.isinf(X_test).sum())

# Fill NaN with column medians
col_medians = np.nanmedian(X_test, axis=0)
nan_mask = np.isnan(X_test)
if nan_mask.sum() > 0:
    X_test[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

# Clip infinities
X_test = np.where(np.isposinf(X_test),  1e6, X_test)
X_test = np.where(np.isneginf(X_test), -1e6, X_test)

print("After imputation - NaN:", np.isnan(X_test).sum(), "| Inf:", np.isinf(X_test).sum())

# === SCALE USING SAVED SCALER ===
scaler = joblib.load('/kaggle/input/distillation/scaler.pkl')
X_scaled = scaler.transform(X_test)

assert not np.isnan(X_scaled).any(), "Scaling introduced NaN"
assert not np.isinf(X_scaled).any(), "Scaling introduced Inf"

# === DEFINE STUDENT MODEL ===
class StudentMLP(nn.Module):
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = torch.clamp(x, -1e3, 1e3)
        return self.net(x)

# === LOAD AND PREDICT ===
model = StudentMLP(input_dim=X_scaled.shape[1]).to(device)
model.load_state_dict(torch.load('/kaggle/input/distillation/distilled_mlp.pth'))
model.eval()

preds = []
batch_size = 512
for i in range(0, len(X_scaled), batch_size):
    x_batch = torch.from_numpy(X_scaled[i:i+batch_size]).float().to(device)
    with torch.no_grad():
        logits = model(x_batch)
        batch_probs = F.softmax(logits, dim=1).cpu().numpy()
        preds.append(batch_probs)

preds = np.concatenate(preds)

# === FINAL VALIDATION ===
assert not np.isnan(preds).any(), "Predictions contain NaN"
row_sums = preds.sum(axis=1)
assert np.allclose(row_sums, 1.0, atol=1e-3), f"Row sums not close to 1: {row_sums.min():.6f} → {row_sums.max():.6f}"

# Re-normalize and clip
preds = np.clip(preds, 1e-7, 1 - 1e-7)
preds = preds / preds.sum(axis=1, keepdims=True)

# === CREATE SUBMISSION ===
submission = pd.DataFrame({
    'id': ids,
    'winner_model_a': preds[:, 0],
    'winner_model_b': preds[:, 1],
    'winner_tie': preds[:, 2]
})[['id', 'winner_model_a', 'winner_model_b', 'winner_tie']]

submission.to_csv('submission.csv', index=False)
print("✅ Submission saved.")
print(submission.head())
print(f"Shape: {submission.shape}")


# df_final.head()
# df_final = df_final[desired_order]

# logger.info(f"Columns successfully reordered to match required schema. Final shape: {df_final.shape}")

# # Final save
# df_final.to_csv(FEATURE_OUTPUT, index=False)
# logger.info(f"Final merged features saved to {FEATURE_OUTPUT}. Shape: {df_final.shape}")


# import lightgbm as lgb


# import pandas as pd
# import numpy as np
# import xgboost as xgb
# import optuna
# from sklearn.metrics import log_loss
# from functools import partial

# # === Load data ===
# FEATURES_FILE = '/kaggle/input/traindata/lmsys_train_features_final.csv'
# FOLDS_FILE = '/kaggle/input/folds-split/folds.csv'
# TARGET_COLS = ['winner_model_a', 'winner_model_b', 'winner_tie']
# ID_COL = 'id'

# df_features = pd.read_csv(FEATURES_FILE)
# folds_df = pd.read_csv(FOLDS_FILE)

# # Prepare labels and features
# y_multiclass = df_features[TARGET_COLS].values.argmax(axis=1)
# feature_cols = [col for col in df_features.columns if col not in [ID_COL] + TARGET_COLS]
# X = df_features[feature_cols]
# y = y_multiclass

# # Merge folds
# df = df_features[[ID_COL]].merge(folds_df[['id', 'fold']], on='id', how='left')
# fold_ids = df['fold'].values

# # === Objective function for Optuna ===
# def objective(trial, X, y, fold_ids):
#     params = {
#         'objective': 'multi:softprob',
#         'num_class': 3,
#         'eval_metric': 'mlogloss',
        
#         # Core tree and boosting parameters
#         'max_depth': trial.suggest_int('max_depth', 4, 12),
#         'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),  # Log-uniform
#         'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
#         'booster': 'gbtree',

#         # Sampling
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
#         'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),

#         # Regularization
#         'gamma': trial.suggest_float('gamma', 1e-8, 10.0, log=True),  # Encourage exploration
#         'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
#         'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),

#         # Tree construction
#         'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
#         'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),

#         # GPU acceleration
#         'tree_method': 'gpu_hist',
#         'predictor': 'gpu_predictor',  # Ensure prediction uses GPU
#         'sampling_method': trial.suggest_categorical('sampling_method', ['uniform', 'gradient_based']),

#         # Random state
#         'random_state': 42,
#         'verbosity': 0
#     }

#     cv_scores = []
#     for fold in range(5):
#         train_idx = np.where(fold_ids != fold)[0]
#         val_idx = np.where(fold_ids == fold)[0]

#         X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#         y_train, y_val = y[train_idx], y[val_idx]

#         dtrain = xgb.DMatrix(X_train, label=y_train)
#         dval = xgb.DMatrix(X_val, label=y_val)

#         # Use params['n_estimators'] as num_boost_round
#         model = xgb.train(
#             {k: v for k, v in params.items() if k != 'n_estimators'},  # Remove n_estimators from params passed
#             dtrain,
#             num_boost_round=params['n_estimators'],
#             evals=[(dval, 'val')],
#             early_stopping_rounds=50,
#             verbose_eval=False
#         )

#         preds = model.predict(dval, iteration_range=(0, model.best_iteration))
#         score = log_loss(y_val, preds)
#         cv_scores.append(score)

#     return np.mean(cv_scores)

# # === Run Optimization ===
# study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=20))
# study.optimize(partial(objective, X=X, y=y, fold_ids=fold_ids), n_trials=150)

# print("Best params:", study.best_params)
# print("Best log loss:", study.best_value)

# # Save best params
# import json
# with open('xgboost_best_params.json', 'w') as f:
#     json.dump(study.best_params, f)


# import json
# with open('xgboost_best_params.json', 'r') as f:
#     PARAMS = json.load(f)
# PARAMS.update({
#     'objective': 'multi:softprob',
#     'num_class': 3,
#     'eval_metric': 'mlogloss',
#     'random_state': 42,
#     'tree_method': 'gpu_hist'
# })


# import pandas as pd
# import numpy as np
# import xgboost as xgb
# from sklearn.metrics import log_loss
# import os

# # Train on full train set
# dtrain_full = xgb.DMatrix(X, label=y)
# final_model = xgb.train(
#     PARAMS,
#     dtrain_full,
#     num_boost_round=1000,
#     evals=[(dtrain_full, 'train')],
#     early_stopping_rounds=50,
#     verbose_eval=False
# )

# # Save model
# model_output_file = 'xgboost_final_model.json'
# final_model.save_model(model_output_file)
# print(f"✅ Final model saved to '{model_output_file}'")

# print("\n--- XGBoost Training Complete ---")


# import json
# import pandas as pd
# import numpy as np
# import xgboost as xgb
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# with open('/kaggle/input/xgboostparams/xgboost_best_params.json', 'r') as f:
#     PARAMS = json.load(f)
# PARAMS.update({
#     'objective': 'multi:softprob',
#     'num_class': 3,
#     'eval_metric': 'mlogloss',
#     'random_state': 42,
#     'tree_method': 'gpu_hist'
# })

# # === INFERENCE ON TEST SET ===

# # Load test features (NO TARGET COLUMNS)
# TEST_FEATURES_FILE = '/kaggle/working/lmsys_test_features_final.csv'
# ID_COL = 'id'

# logger.info("Loading test features...")
# df_test = pd.read_csv(TEST_FEATURES_FILE)
# assert ID_COL in df_test.columns, f"{ID_COL} not found in test data"

# # Ensure 'id' is kept
# test_ids = df_test[ID_COL]

# # Features: exclude ID and any potential target cols (just in case)
# feature_cols = [col for col in df_test.columns if col != ID_COL and not col.startswith('winner')]
# X_test = df_test[feature_cols]

# # Load trained model
# model = xgb.Booster()
# model.load_model('/kaggle/input/xgboost-no-judge/xgboost_final_model.json')

# # Predict
# dtest = xgb.DMatrix(X_test)
# preds = model.predict(dtest)  # shape: (n_samples, 3)

# # Create submission DataFrame
# TARGET_COLS = ['winner_model_a', 'winner_model_b', 'winner_tie']
# submission = pd.DataFrame(preds, columns=TARGET_COLS)
# submission.insert(0, ID_COL, test_ids)

# # Save
# submission.to_csv('/kaggle/working/submission.csv', index=False)
# print(submission.head())
# print(f"✅ Submission saved with shape: {submission.shape}")

