import os
import re
import pandas as pd
import numpy as np
from datasets import Dataset
import warnings
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.nn.functional as F
import random
import nltk
from functools import partial
import joblib

# --- 0. Setup and Configuration ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
warnings.filterwarnings("ignore", category=FutureWarning)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
set_seed(42)

# --- Configuration ---
MODEL_NAME = "microsoft/deberta-v3-base"
TEST_DIR = "/kaggle/input/fake-or-real-the-impostor-hunt/data/test"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8 # Larger batch size for faster inference
MC_PASSES = 20 # Use 30 passes for stable predictions

# --- 1. Model Class Definitions (Include all three) ---

# Model 1: Your original, best-performing Siamese model
class CrossEncoderClassifier(nn.Module):
    """
    A cross-encoder that fuses the concept of weighted layer pooling.

    Instead of creating separate embeddings for each text, this model concatenates
    them and processes them in a single pass. It then creates a weighted average 
    of the [CLS] token's embedding across all hidden layers to make a final prediction.
    """
    def __init__(self, model_name="microsoft/deberta-v3-large", num_labels=2):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load backbone model, ensuring it outputs all hidden states
        self.backbone = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone.to(self.device)

        self.hidden_size = self.backbone.config.hidden_size
        # +1 for the initial word embedding layer
        self.num_hidden_layers = self.backbone.config.num_hidden_layers + 1 

        # ⭐ Your trainable weights for each layer are preserved here!
        self.layer_weights = nn.Parameter(torch.ones(self.num_hidden_layers) / self.num_hidden_layers)

        # The classifier is simpler as it only takes one final vector
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size // 2, num_labels)
        )

    def forward(self, text1_list, text2_list, labels=None):
        # Tokenize the pair together. The tokenizer handles the [CLS] and [SEP] tokens.
        # Format: [CLS] text1 [SEP] text2 [SEP]
        inputs = self.tokenizer(
            text1_list,
            text2_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512  # Note: You may need to adjust this
        ).to(self.device)

        # Get model outputs
        outputs = self.backbone(**inputs)
        
        # hidden_states is a tuple of (L+1) tensors of shape [B, T, H]
        hidden_states = outputs.hidden_states

        # Extract the [CLS] token's representation from each layer.
        # The [CLS] token is always at the first position (index 0).
        # This creates a list of L+1 tensors, each of shape [B, H]
        cls_embeddings_per_layer = [layer[:, 0, :] for layer in hidden_states]
        
        # Stack them into a single tensor of shape [L+1, B, H]
        stacked_cls_embeddings = torch.stack(cls_embeddings_per_layer, dim=0)

        # Apply softmax to the layer weights to get a probability distribution
        norm_weights = F.softmax(self.layer_weights, dim=0)

        # Create the final weighted embedding for the [CLS] token
        # Reshape weights to [L, 1, 1] for broadcasting and sum over the layer dimension
        weighted_cls_embedding = (norm_weights.view(-1, 1, 1) * stacked_cls_embeddings).sum(dim=0)
        
        # Pass the final embedding through the classifier
        logits = self.classifier(weighted_cls_embedding)

        loss = None
        if labels is not None:
            labels = labels.to(self.device).long()
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return type('Output', (object,), {'loss': loss, 'logits': logits})()

# Model 2: The Statistical Feature model
class SiameseWithFeaturesClassifier(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base", num_labels=2, freeze_backbone=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone.to(self.device)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.hidden_size = self.backbone.config.hidden_size
        self.num_hidden_layers = self.backbone.config.num_hidden_layers + 1  # +1 for embeddings
        # We are adding 5 features per text, so 10 total statistical features
        self.num_stat_features = 5
        
        # ⭐ Redefine the classifier to accept the new, wider feature vector
        # Original size: self.hidden_size * 4
        # New size: (self.hidden_size * 4) + (num_stat_features * 2)
        new_classifier_input_size = (self.hidden_size * 4) + (self.num_stat_features * 2)
        
        # Trainable weights for each layer (L+1 layers total)
        self.layer_weights = nn.Parameter(torch.ones(self.num_hidden_layers) / self.num_hidden_layers)

        # Interaction through concatenation, absolute difference, element-wise multiplication
        self.classifier = nn.Sequential(
            nn.Linear(new_classifier_input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, num_labels)
        )

    def weighted_pooling(self, hidden_states, attention_mask):
        """
        Apply attention-masked mean pooling across sequence, then weighted sum over layers.
        hidden_states: tuple of [B, T, H] tensors, one per layer
        attention_mask: [B, T]
        """
        # Stack: [L, B, T, H]
        all_layers = torch.stack(hidden_states, dim=0)

        # Expand mask: [B, T] → [1, B, T, 1]
        mask = attention_mask.unsqueeze(0).unsqueeze(-1).float()  # [1, B, T, 1]

        # Masked sum: zero out pad tokens, then mean over tokens → [L, B, H]
        summed = (all_layers * mask).sum(dim=2)  # sum over token dim
        lengths = torch.clamp(mask.sum(dim=2), min=1e-5)  # avoid divide by zero
        mean_pooled = summed / lengths  # [L, B, H]

        # Weighted sum over layers
        norm_weights = torch.nn.functional.softmax(self.layer_weights, dim=0)  # [L]
        pooled = (norm_weights[:, None, None] * mean_pooled).sum(dim=0)  # [B, H]
        return pooled

    def extract_weighted_embedding(self, texts):
        """
        Get pooled token embeddings from all layers using attention-weighted and layer-weighted pooling.
        """
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        outputs = self.backbone(**encoded)
        hidden_states = outputs.hidden_states  # tuple of [B, T, H], len = num_hidden_layers
        pooled = self.weighted_pooling(hidden_states, encoded['attention_mask'])  # [B, H]
        return pooled

    def forward(self, text1_list, text2_list, stats1, stats2, labels=None):
        vec_a = self.extract_weighted_embedding(text1_list)  # [B, H]
        vec_b = self.extract_weighted_embedding(text2_list)  # [B, H]
    
        # Feature interactions
        diff = torch.abs(vec_a - vec_b)
        mul = vec_a * vec_b
        stats1 = stats1.to(self.device)
        stats2 = stats2.to(self.device)
        
        concat = torch.cat([vec_a, vec_b, diff, mul, stats1, stats2], dim=-1)  # [B, 4H]
        
        logits = self.classifier(concat)

        loss = None
        if labels is not None:
            labels = labels.to(self.device).long()
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return type('Output', (object,), {'loss': loss, 'logits': logits})()

# --- 2. Helper Functions for Data and Inference ---
def extract_statistical_features(text_list):
    """
    Calculates statistical features for a list of texts.
    
    Returns:
        np.array: A numpy array of shape [len(text_list), 5]
    """
    feature_list = []
    for text in text_list:
        # Tokenize
        words = nltk.word_tokenize(text)
        sentences = nltk.sent_tokenize(text)
        
        # Basic counts
        word_count = len(words)
        sentence_count = len(sentences)
        char_count = len(text)
        
        # Handle potential division by zero for empty texts
        if word_count == 0 or sentence_count == 0:
            avg_sentence_length = 0
            type_token_ratio = 0
        else:
            # Derived features
            avg_sentence_length = word_count / sentence_count
            # Vocabulary richness
            type_token_ratio = len(set(words)) / word_count
        
        features = [
            char_count, 
            word_count, 
            sentence_count, 
            avg_sentence_length, 
            type_token_ratio
        ]
        feature_list.append(features)
        
    return np.array(feature_list, dtype=np.float32)

def test_data_generator(data_dir):
    folders = sorted([f for f in os.listdir(data_dir) if f.startswith('article_')])
    for folder in folders:
        folder_id = int(folder.split('_')[1]); folder_path = os.path.join(data_dir, folder)
        with open(os.path.join(folder_path, "file_1.txt"), encoding="utf-8") as f1: text1 = f1.read()
        with open(os.path.join(folder_path, "file_2.txt"), encoding="utf-8") as f2: text2 = f2.read()
        yield {"id": folder_id, "text1": text1, "text2": text2}

def eval_collate_fn(batch, scaler):
    # Standard collation
    text1_batch = [item["text1"] for item in batch]
    text2_batch = [item["text2"] for item in batch]
    id_batch = [item["id"] for item in batch]
    
    # Extract features
    stats1_batch_raw = extract_statistical_features(text1_batch)
    stats2_batch_raw = extract_statistical_features(text2_batch)
    
    # ⭐ Apply the fitted scaler to transform the features
    stats1_batch_scaled = torch.from_numpy(scaler.transform(stats1_batch_raw)).float()
    stats2_batch_scaled = torch.from_numpy(scaler.transform(stats2_batch_raw)).float()
    
    return {
        "text1": text1_batch,
        "text2": text2_batch,
        "id": id_batch,
        "stats1": stats1_batch_scaled,
        "stats2": stats2_batch_scaled
    }

# Generic MC Dropout Inference function
def get_mc_predictions(model, dataloader, model_type="cross"):
    model.train(); final_probs = {}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"MC Inference ({model_type})"):
            all_probs = torch.zeros(len(batch["id"]), device=DEVICE)
            for _ in range(MC_PASSES):
                if model_type == "cross":
                    logits1 = model(batch["text1"], batch["text2"]).logits; prob1 = torch.softmax(logits1, dim=1)[:, 1]
                    logits2 = model(batch["text2"], batch["text1"]).logits; prob2 = torch.softmax(logits2, dim=1)[:, 0]
                elif model_type == "feature":
                    logits1 = model(batch["text1"], batch["text2"], batch["stats1"], batch["stats2"]).logits; prob1 = torch.softmax(logits1, dim=1)[:, 1]
                    logits2 = model(batch["text2"], batch["text1"], batch["stats2"], batch["stats1"]).logits; prob2 = torch.softmax(logits2, dim=1)[:, 0]
                all_probs += (prob1 + prob2) / 2
            all_probs /= MC_PASSES
            for a_id, avg_prob in zip(batch["id"], all_probs):
                final_probs[int(a_id)] = avg_prob.item()
    return final_probs

# --- 3. Prediction Generation for All Models ---
scaler = joblib.load("/kaggle/input/fake-or-real-statistical-model-cross-val-17-e/statistical_feature_scaler.pkl")
eval_collate_with_scaler = partial(eval_collate_fn, scaler=scaler)
test_dataset = Dataset.from_generator(lambda: test_data_generator(TEST_DIR))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=eval_collate_with_scaler)

# == Model 1: K-Fold Cross Ensemble ==
print("--- Generating predictions for Model 1: Cross-encoder K-Fold Ensemble ---")
cross_fold_probs = []
for fold in range(5):
    model = CrossEncoderClassifier().to(DEVICE)
    model.load_state_dict(torch.load(f"/kaggle/input/fake-or-real-cross-encoder-cross-val-17-e/model_fold_{fold+1}.pth"))
    cross_fold_probs.append(get_mc_predictions(model, test_loader, "cross"))
# Average the fold predictions
prob_cross_ensemble = {k: np.mean([d[k] for d in cross_fold_probs]) for k in cross_fold_probs[0]}

# == Model 2: Feature Model ==
print("\n--- Generating predictions for Model 2: Feature Model ---")
feature_fold_probs = []
for fold in range(5):
    model_feature = SiameseWithFeaturesClassifier().to(DEVICE)
    model_feature.load_state_dict(torch.load(f"/kaggle/input/fake-or-real-statistical-model-cross-val-17-e/model_fold_{fold+1}.pth"))
    feature_fold_probs.append(get_mc_predictions(model_feature, test_loader, "feature"))
# Average the fold predictions
prob_feature_ensemble = {k: np.mean([d[k] for d in feature_fold_probs]) for k in feature_fold_probs[0]}

# --- 4. Final Weighted Averaging and Submission ---
print("\n--- Combining predictions with weighted average ---")
final_preds = {}
article_ids = sorted(prob_cross_ensemble.keys())

for a_id in article_ids:
    final_prob = (0.2 * prob_cross_ensemble[a_id]) + (0.8 * prob_feature_ensemble[a_id])    
    final_preds[a_id] = 1 if final_prob >= 0.5 else 2

# --- Create Submission File ---
submission = pd.DataFrame({'id': list(final_preds.keys()), 'real_text_id': list(final_preds.values())})
submission.to_csv("submission.csv", index=False)

print("\n✅ Final weighted ensemble submission file saved as 'submission.csv'")
print(submission.head())




