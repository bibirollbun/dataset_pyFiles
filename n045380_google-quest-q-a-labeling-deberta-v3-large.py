import os
import html
import re
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from huggingface_hub import snapshot_download

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")


DATA_DIR = "/kaggle/input/google-quest-challenge"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

# Model configuration
MODEL_NAME = "/kaggle/input/deberta-v3-large-fold3-val-spearman0-4392/pytorch/default/2/deberta-v3-large-model/deberta-v3-large-model"
MAX_LENGTH = 512
BATCH_SIZE = 8  # Adjust based on your GPU memory
SEED = 42

# Best checkpoint path (highest validation score: fold2 with val_spearman=0.4418)
CHECKPOINT_PATH = "/kaggle/input/deberta-v3-large-fold3-val-spearman0-4392/pytorch/default/2/deberta_v3_large_fold3_val_spearman0.4392.ckpt"

# Target columns (30 total: 21 question + 9 answer)
QUESTION_TARGET_COLS = [
    "question_asker_intent_understanding",
    "question_body_critical",
    "question_conversational",
    "question_expect_short_answer",
    "question_fact_seeking",
    "question_has_commonly_accepted_answer",
    "question_interestingness_others",
    "question_interestingness_self",
    "question_multi_intent",
    "question_not_really_a_question",
    "question_opinion_seeking",
    "question_type_choice",
    "question_type_compare",
    "question_type_consequence",
    "question_type_definition",
    "question_type_entity",
    "question_type_instructions",
    "question_type_procedure",
    "question_type_reason_explanation",
    "question_type_spelling",
    "question_well_written",
]

ANSWER_TARGET_COLS = [
    "answer_helpful",
    "answer_level_of_information",
    "answer_plausible",
    "answer_relevance",
    "answer_satisfaction",
    "answer_type_instructions",
    "answer_type_procedure",
    "answer_type_reason_explanation",
    "answer_well_written",
]

TARGET_COLS = QUESTION_TARGET_COLS + ANSWER_TARGET_COLS
NUM_QUESTION_TARGETS = len(QUESTION_TARGET_COLS)
NUM_ANSWER_TARGETS = len(ANSWER_TARGET_COLS)
NUM_TARGETS = len(TARGET_COLS)

# Multi-sample dropout rates
DROPOUT_RATES = [0.1, 0.15, 0.2, 0.25, 0.3]

print(f"Number of targets: {NUM_TARGETS}")
print(f"Question targets: {NUM_QUESTION_TARGETS}")
print(f"Answer targets: {NUM_ANSWER_TARGETS}")
print(f"Checkpoint: {CHECKPOINT_PATH}")


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clean_text(text: str) -> str:
    """Clean and preprocess text."""
    if pd.isna(text):
        return ""
    text = html.unescape(str(text))
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def compute_spearman(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute mean Spearman correlation."""
    scores = []
    for i in range(predictions.shape[1]):
        score, _ = spearmanr(predictions[:, i], targets[:, i])
        if not np.isnan(score):
            scores.append(score)
    return np.mean(scores) if scores else 0.0

set_seed(SEED)


class QuestDataset(Dataset):
    """
    Dataset for Google QUEST Q&A Labeling.
    Creates two inputs for Siamese architecture:
    - Question input: [CLS] question_title [SEP] question_body [SEP]
    - Answer input: [CLS] question_title + body [SEP] answer [SEP]
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int = 512,
        is_test: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        
        # Preprocess text
        self.question_titles = [clean_text(t) for t in self.df["question_title"].values]
        self.question_bodies = [clean_text(t) for t in self.df["question_body"].values]
        self.answers = [clean_text(t) for t in self.df["answer"].values]
        
        # Get targets if not test
        if not is_test:
            self.targets = self.df[TARGET_COLS].values.astype(np.float32)
        
        self.qa_ids = self.df["qa_id"].values
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        title = self.question_titles[idx]
        body = self.question_bodies[idx]
        answer = self.answers[idx]
        
        # Question encoding
        q_encoding = self.tokenizer(
            title, body,
            max_length=self.max_length,
            padding="max_length",
            truncation="longest_first",
            return_tensors="pt",
        )
        
        # Answer encoding
        question_text = f"{title} {body}"
        a_encoding = self.tokenizer(
            question_text, answer,
            max_length=self.max_length,
            padding="max_length",
            truncation="longest_first",
            return_tensors="pt",
        )
        
        item = {
            "q_input_ids": q_encoding["input_ids"].squeeze(0),
            "q_attention_mask": q_encoding["attention_mask"].squeeze(0),
            "a_input_ids": a_encoding["input_ids"].squeeze(0),
            "a_attention_mask": a_encoding["attention_mask"].squeeze(0),
            "qa_id": self.qa_ids[idx],
        }
        
        if not self.is_test:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        return item


class AttentionPooling(nn.Module):
    """Attention-weighted pooling over sequence dimension."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1),
        )
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        weights = self.attention(hidden_states).squeeze(-1)
        weights = weights.masked_fill(~attention_mask.bool(), float('-inf'))
        weights = F.softmax(weights, dim=1)
        pooled = (hidden_states * weights.unsqueeze(-1)).sum(dim=1)
        return pooled


class MultiSampleDropout(nn.Module):
    """Multi-sample dropout for better generalization."""
    
    def __init__(self, dropout_rates: List[float] = None):
        super().__init__()
        if dropout_rates is None:
            dropout_rates = DROPOUT_RATES
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropout_rates])
    
    def forward(self, x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
        outputs = torch.stack([linear(dropout(x)) for dropout in self.dropouts])
        return outputs.mean(dim=0)


class SiameseQuestModel(nn.Module):
    """
    Siamese Dual-Transformer for Google QUEST Q&A Labeling.
    
    Architecture:
    - Shared transformer encoder (RoBERTa-Large)
    - Two branches: Question and Answer
    - Weighted layer aggregation
    - Attention pooling
    - Multi-sample dropout
    """
    
    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        
        # Load transformer
        if pretrained:
            self.transformer = AutoModel.from_pretrained(
                model_name,
                output_hidden_states=True,
            )
        else:
            config = AutoConfig.from_pretrained(model_name)
            config.output_hidden_states = True
            self.transformer = AutoModel.from_config(config)
        
        hidden_size = self.transformer.config.hidden_size
        num_layers = self.transformer.config.num_hidden_layers
        
        # Learnable layer weights (ELMO-style)
        self.layer_weights = nn.Parameter(torch.ones(num_layers + 1))
        
        # Attention pooling
        self.q_attention = AttentionPooling(hidden_size)
        self.a_attention = AttentionPooling(hidden_size)
        
        # Multi-sample dropout
        self.multi_dropout = MultiSampleDropout()
        
        # Prediction heads
        self.question_head = nn.Linear(hidden_size, NUM_QUESTION_TARGETS)
        self.answer_head = nn.Linear(hidden_size, NUM_ANSWER_TARGETS)
        self.combined_head = nn.Linear(hidden_size * 2, NUM_TARGETS)
    
    def weighted_layer_pooling(self, hidden_states: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        stacked = torch.stack(hidden_states, dim=0)
        weights = F.softmax(self.layer_weights, dim=0)
        weighted = (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)
        return weighted
    
    def encode_branch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                      attention_pooling: AttentionPooling) -> torch.Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden = self.weighted_layer_pooling(outputs.hidden_states)
        pooled = attention_pooling(hidden, attention_mask)
        return pooled
    
    def forward(
        self,
        q_input_ids: torch.Tensor,
        q_attention_mask: torch.Tensor,
        a_input_ids: torch.Tensor,
        a_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Encode question branch
        q_pooled = self.encode_branch(q_input_ids, q_attention_mask, self.q_attention)
        
        # Encode answer branch
        a_pooled = self.encode_branch(a_input_ids, a_attention_mask, self.a_attention)
        
        # Predictions with multi-sample dropout
        q_preds = self.multi_dropout(q_pooled, self.question_head)
        a_preds = self.multi_dropout(a_pooled, self.answer_head)
        
        # Combined predictions
        combined = torch.cat([q_pooled, a_pooled], dim=-1)
        combined_preds = self.multi_dropout(combined, self.combined_head)
        
        # Blend specialized and combined predictions
        specialized_preds = torch.cat([q_preds, a_preds], dim=-1)
        final_logits = specialized_preds * 0.5 + combined_preds * 0.5
        
        return final_logits


def load_model_from_checkpoint(checkpoint_path: str, model_name: str, device: torch.device) -> nn.Module:
    """
    Load model from PyTorch Lightning checkpoint.
    
    Args:
        checkpoint_path: Path to .ckpt file
        model_name: HuggingFace model name
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create model
    model = SiameseQuestModel(model_name, pretrained=False)
    
    # Extract state dict from PyTorch Lightning checkpoint
    state_dict = checkpoint['state_dict']
    
    # Remove 'model.' prefix if present (from Lightning)
    # Load weights with strict=False to handle any minor mismatches
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
 
    return model


# Load data
print("Loading data...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")


# Initialize tokenizer
print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Vocab size: {tokenizer.vocab_size}")


# Create test dataset and dataloader
test_dataset = QuestDataset(
    df=test_df,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH,
    is_test=True,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

print(f"Test dataset size: {len(test_dataset)}")
print(f"Test batches: {len(test_dataloader)}")


# Verify checkpoint path exists
exists = os.path.exists(CHECKPOINT_PATH)
status = "OK" if exists else "NOT FOUND"
print(f"Checkpoint: {CHECKPOINT_PATH} [{status}]")


# Generate predictions using the best model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model from checkpoint
model = load_model_from_checkpoint(CHECKPOINT_PATH, MODEL_NAME, device)

# Generate predictions
all_predictions = []
all_qa_ids = []

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Predicting"):
        q_input_ids = batch["q_input_ids"].to(device)
        q_attention_mask = batch["q_attention_mask"].to(device)
        a_input_ids = batch["a_input_ids"].to(device)
        a_attention_mask = batch["a_attention_mask"].to(device)
        
        logits = model(
            q_input_ids=q_input_ids,
            q_attention_mask=q_attention_mask,
            a_input_ids=a_input_ids,
            a_attention_mask=a_attention_mask,
        )
        
        # Apply sigmoid to get predictions in [0, 1]
        preds = torch.sigmoid(logits)
        
        all_predictions.append(preds.cpu().numpy())
        all_qa_ids.extend(batch["qa_id"])

predictions = np.concatenate(all_predictions, axis=0)
print(f"Predictions shape: {predictions.shape}")


# Create submission DataFrame
submission = pd.DataFrame(predictions, columns=TARGET_COLS)
submission.insert(0, "qa_id", [x.item() for x in all_qa_ids])

# Clip predictions to [0, 1]
for col in TARGET_COLS:
    submission[col] = submission[col].clip(0, 1)

print(f"Submission shape: {submission.shape}")
submission.to_csv('submission.csv', index=False)
print('submission.csv created!')
submission.head()

