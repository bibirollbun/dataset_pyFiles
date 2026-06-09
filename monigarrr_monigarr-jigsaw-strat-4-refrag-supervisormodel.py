# --- Imports and Configuration ---
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sentence_transformers import CrossEncoder
from torch.utils.data import Dataset
from nltk.tokenize import sent_tokenize
import nltk
import os
import gc
from tqdm.auto import tqdm

# Ensure the NLTK data is available
nltk.download('punkt', quiet=True)

class CFG:
    # --- Paths to Models & Data ---
    SUPERVISOR_PATH = "/kaggle/input/jigsaw-2025-strategy4-supervisor-model-monigarr/supervisor-model-refrag-monigarr"
    CLASSIFIER_PATH = "/kaggle/input/monigarr-jigsaw-competition-models-aug-9-2025/final_models/roberta-base-reddit-pretrained"
    COMPETITION_DATA_DIR = "/kaggle/input/jigsaw-agile-community-rules/"
    
    # --- Inference Hyperparameters ---
    MAX_LEN = 256
    BATCH_SIZE = 32
    TOP_K_SENTENCES = 3 # How many relevant sentences to extract


# --- Helper Classes & Functions ---

def softmax(x):
    """Computes softmax probabilities from logits."""
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

class JigsawDataset(Dataset):
    """Custom PyTorch Dataset for inference."""
    def __init__(self, tokenized_data):
        self.input_ids = torch.tensor(tokenized_data['input_ids'])
        self.attention_mask = torch.tensor(tokenized_data['attention_mask'])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
        }


# --- Load Models & Data ---
print("Loading models...")
supervisor_model = CrossEncoder(CFG.SUPERVISOR_PATH)
tokenizer_path = os.path.join(CFG.CLASSIFIER_PATH, "fold0")
classifier_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
print("Supervisor model and classifier tokenizer loaded successfully.")

test_df = pd.read_csv(f"{CFG.COMPETITION_DATA_DIR}test.csv")
print("Test data loaded successfully.")


# --- STAGE 1: SUPERVISOR (Filter) ---
print("Starting Stage 1: Filtering with Supervisor Model...")

all_sentence_pairs = []
original_indices = [] # To map sentences back to their original comment
for i, row in test_df.iterrows():
    sentences = sent_tokenize(row['body'])
    if sentences:
        for s in sentences:
            all_sentence_pairs.append([row['rule'], s])
            original_indices.append(i)

print(f"Scoring {len(all_sentence_pairs)} sentence pairs...")
all_relevance_scores = supervisor_model.predict(all_sentence_pairs, show_progress_bar=True)

test_df['relevance_scores'] = [[] for _ in range(len(test_df))]
for i, score in zip(original_indices, all_relevance_scores):
    test_df.loc[i, 'relevance_scores'].append(score)

focused_texts = []
for i, row in test_df.iterrows():
    sentences = sent_tokenize(row['body'])
    scores = row['relevance_scores']
    if not sentences or not scores:
        focused_texts.append("")
        continue
    
    top_k_indices = np.argsort(scores)[-CFG.TOP_K_SENTENCES:]
    focused_body = " ".join([sentences[i] for i in top_k_indices])
    focused_texts.append(focused_body)

test_df['focused_body'] = focused_texts
print("Stage 1 complete. All texts have been filtered.")


# --- STAGE 2: CLASSIFIER (Predict) ---
all_fold_predictions = []
print("\nStarting Stage 2: Classifying with Main Model...")

print("Tokenizing filtered texts...")
tokenized_texts = classifier_tokenizer(
    test_df['rule'].tolist(),
    test_df['focused_body'].tolist(),
    max_length=CFG.MAX_LEN,
    padding='max_length',
    truncation=True
)
test_dataset = JigsawDataset(tokenized_texts)
print("Tokenization complete.")

for fold in range(5):
    print(f"  > Processing Fold {fold}...")
    model_path = os.path.join(CFG.CLASSIFIER_PATH, f"fold{fold}")
    classifier_model = AutoModelForSequenceClassification.from_pretrained(model_path, use_safetensors=True)
    
    trainer = Trainer(
        model=classifier_model,
        args=TrainingArguments(
            output_dir="./temp_output",
            per_device_eval_batch_size=CFG.BATCH_SIZE,
            fp16=True,
            report_to="none",
        ),
    )
    
    test_output = trainer.predict(test_dataset)
    test_probs = softmax(test_output.predictions)
    all_fold_predictions.append(test_probs[:, 1])
    
    del classifier_model, trainer
    gc.collect()
    torch.cuda.empty_cache()

print("Stage 2 complete. All folds have made predictions.")


# --- Create Final Submission ---
final_predictions = np.mean(all_fold_predictions, axis=0)

submission_df = pd.DataFrame({
    'row_id': test_df['row_id'],
    'rule_violation': final_predictions
})

assert not submission_df.isnull().values.any(), "Submission contains NaN values!"
print("\nFinal predictions validated.")

submission_df.to_csv('submission.csv', index=False)
print("Strategy 4 submission.csv created successfully!")
display(submission_df.head())

