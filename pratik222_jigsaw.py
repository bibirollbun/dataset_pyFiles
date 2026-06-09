import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import sys
import os
import re
import random
import torch
import torch.nn.functional as F
import logging
import time
import nltk
from nltk.corpus import wordnet
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import warnings



# Suppress warnings (including CUDA warnings)
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Download NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Disable W&B logging
os.environ["WANDB_MODE"] = "disabled"

# Enable logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Verify library versions
import transformers
import datasets
logger.info(f"Transformers version: {transformers.__version__}")
logger.info(f"Datasets version: {datasets.__version__}")
logger.info(f"Torch version: {torch.__version__}")

# --- Configuration ---
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")


# Model path (use DistilBERT for Kaggle compatibility)
MODEL_CHECKPOINT = "/kaggle/input/distilbert-base-uncased-offline/distilbert-base-uncased"

# --- Data Loading ---
logger.info("Loading data...")
start_time = time.time()
kaggle_data_path = "/kaggle/input/jigsaw-agile-community-rules/"

try:
    train_df = pd.read_csv(f"{kaggle_data_path}train.csv")
    test_df = pd.read_csv(f"{kaggle_data_path}test.csv")
    sample_submission_df = pd.read_csv(f"{kaggle_data_path}sample_submission.csv")
    logger.info("âœ… Data loaded successfully!")
except FileNotFoundError as e:
    logger.error("â�Œ Data files not found!")
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            logger.error(os.path.join(dirname, filename))
    sys.exit("Exiting: Data files not found.")

logger.info(f"ğŸ“Š Train shape: {train_df.shape}")
logger.info(f"ğŸ“Š Test shape: {test_df.shape}")
logger.info(f"ğŸ“Š Train columns: {train_df.columns.tolist()}")

# Convert row_id to string
train_df['row_id'] = train_df['row_id'].astype(str)
test_df['row_id'] = test_df['row_id'].astype(str)

# --- Preprocessing ---
logger.info("ğŸ§¹ Starting preprocessing...")
start_time = time.time()

def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower().strip()
        promo_words = ['click', 'vote', 'win', 'free', 'download', 'stream', 'link']
        for word in promo_words:
            text = re.sub(rf'\b{word}\b', '', text)
        return text
    return ''

# Apply cleaning to text columns
text_cols = ['body', 'rule', 'subreddit', 'positive_example_1', 'positive_example_2', 
             'negative_example_1', 'negative_example_2']
for col in text_cols:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna('').apply(clean_text)
    if col in test_df.columns:
        test_df[col] = test_df[col].fillna('').apply(clean_text)

# Feature engineering
def extract_rule_features(rule):
    if not isinstance(rule, str):
        return 0, 0
    toxicity_words = ['toxic', 'offensive', 'hate', 'abuse']
    spam_words = ['spam', 'advertising', 'promotion']
    toxicity_score = sum(rule.count(kw) for kw in toxicity_words)
    spam_score = sum(rule.count(kw) for kw in spam_words)
    return toxicity_score, spam_score

for df in [train_df, test_df]:
    df['rule_length'] = df['rule'].apply(len)
    df['comment_length'] = df['body'].apply(len)
    df['toxicity_score'], df['spam_score'] = zip(*df['rule'].apply(extract_rule_features))

# Combine texts
def combine_texts(row):
    core = f"Comment: {row['body']} | Rule: {row['rule']} | Subreddit: {row['subreddit']}"
    examples = []
    for ex_col in ['positive_example_1', 'positive_example_2', 'negative_example_1', 'negative_example_2']:
        if row.get(ex_col, '') and len(row[ex_col]) > 10:
            examples.append(f"{ex_col}: {row[ex_col][:50]}")
    if examples:
        core += f" | {' | '.join(examples)}"
    core += f" | Len:{row['comment_length']}-{row['rule_length']} | Tox:{row['toxicity_score']}"
    return core

try:
    train_df['full_text'] = train_df.apply(combine_texts, axis=1)
    test_df['full_text'] = test_df.apply(combine_texts, axis=1)
    logger.info(f"ğŸ“� Sample text: {train_df['full_text'].iloc[0][:100]}...")
except Exception as e:
    logger.error(f"â�Œ Preprocessing error: {str(e)}")
    raise

logger.info(f"âœ… Preprocessing done in {time.time() - start_time:.2f}s")

# --- Augmentation ---
logger.info("ğŸ”„ Augmenting data...")
start_time = time.time()

def get_synonyms(word):
    synonyms = set()
    try:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != word and len(synonym.split()) == 1:
                    synonyms.add(synonym)
        return list(synonyms)[:2]
    except:
        return []

def simple_synonym_replace(text, n=2):
    if not isinstance(text, str) or len(text.split()) < n:
        return text
    words = text.split()
    indices = random.sample(range(len(words)), min(n, len(words)))
    for i in indices:
        synonyms = get_synonyms(words[i])
        if synonyms:
            words[i] = random.choice(synonyms)
    return ' '.join(words)

# Augment minority class
minority_class = train_df['rule_violation'].value_counts().idxmin()
augmented_rows = []
for _, row in train_df[train_df['rule_violation'] == minority_class].sample(frac=0.5, random_state=SEED).iterrows():
    aug_text = simple_synonym_replace(row['full_text'])
    augmented_rows.append({
        'full_text': aug_text,
        'rule_violation': row['rule_violation'],
        'row_id': f"aug_{row['row_id']}"
    })

if augmented_rows:
    aug_df = pd.DataFrame(augmented_rows)
    train_df = pd.concat([train_df, aug_df], ignore_index=True)

logger.info(f"ğŸ“Š After augmentation: {train_df.shape}")
logger.info(f"ğŸ“Š Class balance: {train_df['rule_violation'].value_counts(normalize=True).to_dict()}")

# --- Tokenization ---
logger.info("ğŸ”¤ Tokenizing...")
start_time = time.time()

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, local_files_only=True)
except Exception as e:
    logger.error(f"â�Œ Tokenizer error: {str(e)}")
    logger.error(f"Listing files in {MODEL_CHECKPOINT}:")
    for dirname, _, filenames in os.walk(MODEL_CHECKPOINT):
        for filename in filenames:
            logger.error(os.path.join(dirname, filename))
    sys.exit("Exiting: Failed to load tokenizer.")

def tokenize_function(examples):
    return tokenizer(examples["full_text"], truncation=True, padding="max_length", max_length=96)

try:
    train_dataset = Dataset.from_pandas(train_df[['full_text', 'rule_violation', 'row_id']])
    test_dataset = Dataset.from_pandas(test_df[['full_text', 'row_id']])
    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=['full_text', 'row_id'])
    tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=['full_text', 'row_id'])
    tokenized_train = tokenized_train.rename_column("rule_violation", "labels")
    tokenized_train.set_format('torch')
    tokenized_test.set_format('torch')
    logger.info("âœ… Tokenization complete")
except Exception as e:
    logger.error(f"â�Œ Tokenization error: {str(e)}")
    raise

logger.info(f"âœ… Tokenization done in {time.time() - start_time:.2f}s")

# --- Model & Training ---
logger.info("ğŸš€ Setting up training...")

# Split
train_idx, val_idx = train_test_split(
    range(len(tokenized_train)), test_size=0.2, 
    stratify=tokenized_train["labels"], random_state=SEED
)
train_split = tokenized_train.select(train_idx)
val_split = tokenized_train.select(val_idx)
logger.info(f"ğŸ“Š Train split: {len(train_split)}, Val split: {len(val_split)}")

# Class weights
classes = np.unique(train_df['rule_violation'])
weights = compute_class_weight('balanced', classes=classes, y=train_df['rule_violation'])
class_weights = torch.tensor(weights, dtype=torch.float).to(device)
logger.info(f"âš–ï¸� Class weights: {class_weights}")

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = F.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    return {"auc": roc_auc_score(labels, probs)}

# Training args
training_args = TrainingArguments(
    output_dir="/kaggle/working/results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    weight_decay=0.01,
    learning_rate=2e-5,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="auc",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    report_to="none",
    save_total_limit=1,
    gradient_checkpointing=True
)

# Initialize model
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=2, local_files_only=True
    ).to(device)
    logger.info("âœ… Model loaded")
except Exception as e:
    logger.error(f"â�Œ Model loading error: {str(e)}")
    sys.exit("Exiting: Failed to load model.")

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_split,
    eval_dataset=val_split,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train
try:
    trainer.train()
    eval_result = trainer.evaluate()
    logger.info(f"âœ… Training complete! Val AUC: {eval_result['eval_auc']:.4f}")
except Exception as e:
    logger.error(f"â�Œ Training error: {str(e)}")
    raise

# Save model
MODEL_SAVE_PATH = "/kaggle/working/best_model"
trainer.save_model(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
logger.info(f"âœ… Model saved to {MODEL_SAVE_PATH}")

# --- Calibration ---
logger.info("âš–ï¸� Calibrating...")
val_preds = trainer.predict(val_split)
val_probs = F.softmax(torch.tensor(val_preds.predictions), dim=-1)[:, 1].numpy()
calibrator = CalibratedClassifierCV(LogisticRegression(), method='sigmoid', cv=3)
calibrator.fit(val_probs.reshape(-1, 1), val_preds.label_ids)
logger.info("âœ… Calibration complete")

# --- Predictions ---
logger.info("ğŸ”® Predicting...")
model.eval()
test_loader = torch.utils.data.DataLoader(tokenized_test, batch_size=32)

probs = []
for batch in tqdm(test_loader, desc="Predict"):
    batch = {k: v.to(device) for k, v in batch.items() if k != 'row_id'}
    with torch.no_grad():
        outputs = model(**batch)
    batch_probs = F.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
    probs.extend(batch_probs)

# Calibrate
probs = calibrator.predict_proba(np.array(probs).reshape(-1, 1))[:, 1]
probs = np.clip(probs, 0.01, 0.99)

# Submission
submission_df = pd.DataFrame({
    'row_id': test_df['row_id'],
    'rule_violation': probs
})
submission_df.to_csv("/kaggle/working/submission.csv", index=False)

logger.info("ğŸ�‰ Submission created!")
logger.info(f"ğŸ“ˆ Submission head:\n{submission_df.head().to_string()}")
logger.info(f"ğŸ“Š Prob range: [{probs.min():.3f}, {probs.max():.3f}]")
logger.info(f"ğŸ“Š Mean prob: {probs.mean():.3f}")

