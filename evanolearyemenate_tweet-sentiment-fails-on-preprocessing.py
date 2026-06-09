# ⚠️ ALEXANDRIA MARKER - DO NOT DELETE (used for syncing outputs from Kaggle)
print("===ALEXANDRIA_CELL_1_START===")

# Setup & Imports: Install and import all required libraries for NLP and deep learning

# Install required packages (transformers, tokenizers)
try:
    import transformers
except ImportError:
    print("Installing transformers...")
    !pip install -q transformers

try:
    import tokenizers
except ImportError:
    print("Installing tokenizers...")
    !pip install -q tokenizers

# Import core libraries
import os
import random
import numpy as np
import pandas as pd
import torch

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Progress bar
from tqdm.auto import tqdm

# NLP & Deep Learning
from transformers import AutoTokenizer, AutoModel, AutoConfig

# Sklearn for metrics and preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Set data path
DATA_PATH = "/kaggle/input/tweet-sentiment-extraction"

# Set random seeds for reproducibility
def set_seed(seed=42):
    print(f"Setting random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Configure device (CPU/GPU)
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
except Exception as e:
    print(f"Error configuring device: {e}")
    device = torch.device("cpu")

# Check dataset files
try:
    files = os.listdir(DATA_PATH)
    print(f"Files in dataset directory ({DATA_PATH}): {files}")
except Exception as e:
    print(f"Error accessing dataset directory: {e}")


# ⚠️ ALEXANDRIA MARKER - DO NOT DELETE (used for syncing outputs from Kaggle)
print("===ALEXANDRIA_CELL_2_START===")

# ===ALEXANDRIA_CELL_2_START===
# Data Loading

# Load train, test, and sample submission CSV files into pandas DataFrames

try:
    # Define file paths
    train_path = os.path.join(DATA_PATH, "train.csv")
    test_path = os.path.join(DATA_PATH, "test.csv")
    sample_submission_path = os.path.join(DATA_PATH, "sample_submission.csv")
    
    # Read CSV files into DataFrames
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_submission_df = pd.read_csv(sample_submission_path)
    
    # Display basic info and shapes
    print("=== Data Loading Complete ===")
    print(f"Train DataFrame shape: {train_df.shape}")
    print(f"Test DataFrame shape: {test_df.shape}")
    print(f"Sample Submission DataFrame shape: {sample_submission_df.shape}")
    
    # Display first few rows of each DataFrame
    print("\n--- Train DataFrame (first 5 rows) ---")
    print(train_df.head())
    
    print("\n--- Test DataFrame (first 5 rows) ---")
    print(test_df.head())
    
    print("\n--- Sample Submission DataFrame (first 5 rows) ---")
    print(sample_submission_df.head())
    
except Exception as e:
    print(f"Error loading CSV files: {e}")


# ⚠️ ALEXANDRIA MARKER - DO NOT DELETE (used for syncing outputs from Kaggle)
print("===ALEXANDRIA_CELL_3_START===")

# ===ALEXANDRIA_CELL_3_START===
# Exploratory Data Analysis (EDA) for Tweet Sentiment Extraction

try:
    print("=== Exploratory Data Analysis (EDA) ===")
    
    # 1. Check for missing values and data quality issues
    print("\n--- Missing Values in Train DataFrame ---")
    missing_train = train_df.isnull().sum()
    print(missing_train)
    
    print("\n--- Missing Values in Test DataFrame ---")
    missing_test = test_df.isnull().sum()
    print(missing_test)
    
    # 2. Visualize sentiment distribution in train set
    plt.figure(figsize=(6,4))
    sns.countplot(data=train_df, x="sentiment", order=["positive", "neutral", "negative"], palette="Set2")
    plt.title("Sentiment Distribution (Train Set)")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()
    
    # 3. Examine tweet length distributions (characters and words)
    train_df["tweet_length_char"] = train_df["text"].astype(str).apply(len)
    train_df["tweet_length_word"] = train_df["text"].astype(str).apply(lambda x: len(x.split()))
    
    plt.figure(figsize=(10,4))
    sns.histplot(train_df["tweet_length_char"], bins=40, kde=True, color="skyblue")
    plt.title("Tweet Length Distribution (Characters)")
    plt.xlabel("Number of Characters")
    plt.ylabel("Frequency")
    plt.show()
    
    plt.figure(figsize=(10,4))
    sns.histplot(train_df["tweet_length_word"], bins=30, kde=True, color="salmon")
    plt.title("Tweet Length Distribution (Words)")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.show()
    
    # Tweet length by sentiment
    plt.figure(figsize=(8,4))
    sns.boxplot(data=train_df, x="sentiment", y="tweet_length_char", order=["positive", "neutral", "negative"], palette="Set2")
    plt.title("Tweet Length (Characters) by Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Tweet Length (Characters)")
    plt.show()
    
    # 4. Show random examples of tweets and their selected_text
    print("\n--- Random Examples of Tweets and Selected Text ---")
    sample_examples = train_df.sample(5, random_state=42)[["text", "selected_text", "sentiment"]]
    for idx, row in sample_examples.iterrows():
        print(f"\nSentiment: {row['sentiment']}")
        print(f"Tweet: {row['text']}")
        print(f"Selected Text: {row['selected_text']}")
    
    # 5. Check for empty or problematic selected_text
    empty_selected = train_df["selected_text"].isnull().sum()
    print(f"\nNumber of missing selected_text in train: {empty_selected}")
    empty_text = train_df["text"].isnull().sum()
    print(f"Number of missing text in train: {empty_text}")
    
    # 6. Check for duplicates
    num_duplicates = train_df.duplicated(subset=["text", "sentiment", "selected_text"]).sum()
    print(f"\nNumber of duplicate rows in train: {num_duplicates}")
    
    print("\n=== EDA Complete ===")
    
except Exception as e:
    print(f"Error during EDA: {e}")


# ⚠️ ALEXANDRIA MARKER - DO NOT DELETE (used for syncing outputs from Kaggle)
print("===ALEXANDRIA_CELL_4_START===")

print("===ALEXANDRIA_CELL_4_START===")

# Preprocessing: Clean and prepare tweet data for modeling

try:
    print("=== Preprocessing: Start ===")
    
    # 1. Text normalization function
    import re

    def clean_tweet(text):
        if pd.isnull(text):
            return ""
        # Lowercase
        text = text.lower()
        # Remove extra spaces
        text = re.sub(r"\s+", " ", text)
        # Remove leading/trailing spaces
        text = text.strip()
        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        # Remove HTML entities
        text = re.sub(r"&[a-z]+;", "", text)
        # Remove special characters except basic punctuation
        text = re.sub(r"[^a-z0-9\s\.,!?'\"]", "", text)
        return text

    # Apply cleaning to train and test text columns
    train_df["text_clean"] = train_df["text"].astype(str).apply(clean_tweet)
    test_df["text_clean"] = test_df["text"].astype(str).apply(clean_tweet)
    print("Text normalization complete.")

    # 2. Handle missing selected_text (train only)
    missing_selected = train_df["selected_text"].isnull().sum()
    if missing_selected > 0:
        print(f"Filling {missing_selected} missing selected_text with empty string.")
        train_df["selected_text"] = train_df["selected_text"].fillna("")

    # 3. Normalize selected_text for alignment
    train_df["selected_text_clean"] = train_df["selected_text"].astype(str).apply(clean_tweet)

    # 4. Load transformer tokenizer (RoBERTa-base)
    MODEL_NAME = "roberta-base"
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 5. Tokenization and alignment utilities
    def tokenize_and_align(row):
        text = row["text_clean"]
        sel_text = row["selected_text_clean"]
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_attention_mask=True,
            truncation=True,
            max_length=128,
        )
        offsets = encoding["offset_mapping"]
        # Find selected_text span in text
        start_idx = text.find(sel_text)
        if (start_idx == -1) or (sel_text.strip() == ""):
            # Fallback: use full text if alignment fails
            start_idx, end_idx = 0, len(text)
        else:
            end_idx = start_idx + len(sel_text)
        # Map character indices to token indices
        token_start, token_end = None, None
        for idx, (o_start, o_end) in enumerate(offsets):
            if o_start <= start_idx < o_end:
                token_start = idx
            if o_start < end_idx <= o_end:
                token_end = idx
                break
        # If not found, fallback to first/last non-special token
        if token_start is None:
            token_start = 1  # after <s>
        if token_end is None:
            token_end = sum([1 for o in offsets if o != (0,0)]) - 2  # before </s>
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "token_type_ids": encoding.get("token_type_ids", *len(encoding["input_ids"])),
            "offset_mapping": encoding["offset_mapping"],
            "token_start": token_start,
            "token_end": token_end,
        }

    # 6. Apply tokenization and alignment to train set
    print("Tokenizing and aligning train set...")
    train_features = []
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
        features = tokenize_and_align(row)
        train_features.append(features)
    print("Train set tokenization complete.")

    # 7. Apply tokenization to test set (no selected_text)
    print("Tokenizing test set...")
    def tokenize_test(row):
        text = row["text_clean"]
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_attention_mask=True,
            truncation=True,
            max_length=128,
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "token_type_ids": encoding.get("token_type_ids", *len(encoding["input_ids"])),
            "offset_mapping": encoding["offset_mapping"],
        }
    test_features = []
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        features = tokenize_test(row)
        test_features.append(features)
    print("Test set tokenization complete.")

    # 8. Attach features to DataFrames for downstream use
    train_df["features"] = train_features
    test_df["features"] = test_features

    print("=== Preprocessing: Complete ===")
    print(f"Sample train features:\n{train_df['features'].iloc}")
    print(f"Sample test features:\n{test_df['features'].iloc}")

except Exception as e:
    print(f"Error during preprocessing: {e}")


# ⚠️ ALEXANDRIA MARKER - DO NOT DELETE (used for syncing outputs from Kaggle)
print("===ALEXANDRIA_CELL_5_START===")

# ⚠️ ALEXANDRIA MARKER - DO NOT DELETE (used for syncing outputs from Kaggle)
print("===ALEXANDRIA_CELL_5_START===")

# Feature Engineering for Span Extraction with Transformers

try:
    print("=== Feature Engineering: Start ===")
    
    # 1. Model input formatting: [CLS] sentiment [SEP] tweet [SEP]
    # For RoBERTa, [CLS] is <s>, [SEP] is </s>
    def build_model_inputs(row, tokenizer, max_length=128):
        sentiment = row["sentiment"]
        text = row["text_clean"]
        # RoBERTa does not use token_type_ids, but we keep for compatibility
        # Format: <s> sentiment </s> </s> tweet </s>
        # Double </s> between sentiment and tweet
        encoded = tokenizer(
            sentiment,
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
            return_attention_mask=True,
        )
        return encoded

    # 2. Encode sentiment as an additional input feature (one-hot)
    sentiment_map = {"positive": 0, "neutral": 1, "negative": 2}
    def encode_sentiment(sentiment):
        arr = np.zeros(3, dtype=np.float32)
        idx = sentiment_map.get(sentiment, 1)
        arr[idx] = 1.0
        return arr

    # 3. Generate start and end token labels for selected_text (train only)
    def get_span_labels(row, tokenizer, max_length=128):
        text = row["text_clean"]
        sel_text = row["selected_text_clean"]
        sentiment = row["sentiment"]
        encoded = build_model_inputs(row, tokenizer, max_length)
        offsets = encoded["offset_mapping"]
        # Find selected_text span in text
        start_idx = text.find(sel_text)
        if (start_idx == -1) or (sel_text.strip() == ""):
            # Fallback: use full text if alignment fails
            start_idx, end_idx = 0, len(text)
        else:
            end_idx = start_idx + len(sel_text)
        # Map char indices to token indices (tweet part only)
        token_start, token_end = None, None
        for idx, (o_start, o_end) in enumerate(offsets):
            if o_start <= start_idx < o_end:
                token_start = idx
            if o_start < end_idx <= o_end:
                token_end = idx
                break
        # Fallbacks
        if token_start is None:
            token_start = 1
        if token_end is None:
            token_end = sum([1 for o in offsets if o != (0,0)]) - 2
        # Create start/end label arrays
        start_labels = np.zeros(max_length, dtype=np.float32)
        end_labels = np.zeros(max_length, dtype=np.float32)
        if 0 <= token_start < max_length:
            start_labels[token_start] = 1.0
        if 0 <= token_end < max_length:
            end_labels[token_end] = 1.0
        return start_labels, end_labels

    # 4. Optionally add auxiliary features (tweet length, sentiment markers)
    def get_aux_features(row):
        # Tweet length (normalized)
        length = len(row["text_clean"])
        length_norm = length / 128.0  # max_length normalization
        # Sentiment marker (one-hot)
        sentiment_feat = encode_sentiment(row["sentiment"])
        # Concatenate features
        aux = np.concatenate([[length_norm], sentiment_feat])
        return aux

    # 5. Build feature dicts for train set
    print("Building features for train set...")
    train_inputs = []
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
        encoded = build_model_inputs(row, tokenizer)
        start_labels, end_labels = get_span_labels(row, tokenizer)
        aux_features = get_aux_features(row)
        train_inputs.append({
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "offset_mapping": encoded["offset_mapping"],
            "start_labels": start_labels,
            "end_labels": end_labels,
            "aux_features": aux_features,
        })
    train_df["model_inputs"] = train_inputs

    # 6. Build feature dicts for test set (no labels)
    print("Building features for test set...")
    test_inputs = []
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        encoded = build_model_inputs(row, tokenizer)
        aux_features = get_aux_features(row)
        test_inputs.append({
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "offset_mapping": encoded["offset_mapping"],
            "aux_features": aux_features,
        })
    test_df["model_inputs"] = test_inputs

    print("=== Feature Engineering: Complete ===")
    print(f"Sample train model_inputs:\n{train_df['model_inputs'].iloc[:2]}")
    print(f"Sample test model_inputs:\n{test_df['model_inputs'].iloc[:2]}")

except Exception as e:
    print(f"Error during feature engineering: {e}")


# ⚠️ ALEXANDRIA MARKER - DO NOT DELETE (used for syncing outputs from Kaggle)
print("===ALEXANDRIA_CELL_6_START===")

print("===ALEXANDRIA_CELL_6_START===")

# Model Training: Transformer-based Span Extraction (RoBERTa)

try:
    print("=== Model Training: Start ===")

    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup

    # 1. Dataset class for span extraction
    class TweetSpanDataset(Dataset):
        def __init__(self, df, max_length=128, is_train=True):
            self.df = df
            self.max_length = max_length
            self.is_train = is_train

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            item = self.df.iloc[idx]["model_inputs"]
            input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(item["attention_mask"], dtype=torch.long)
            aux_features = torch.tensor(item["aux_features"], dtype=torch.float)
            if self.is_train:
                start_labels = torch.tensor(item["start_labels"], dtype=torch.float)
                end_labels = torch.tensor(item["end_labels"], dtype=torch.float)
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "aux_features": aux_features,
                    "start_labels": start_labels,
                    "end_labels": end_labels,
                }
            else:
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "aux_features": aux_features,
                }

    # 2. Model architecture: RoBERTa + span prediction heads
    class SpanExtractionModel(nn.Module):
        def __init__(self, model_name="roberta-base", aux_feat_dim=4, dropout_prob=0.2):
            super().__init__()
            self.transformer = AutoModel.from_pretrained(model_name)
            hidden_size = self.transformer.config.hidden_size
            self.dropout = nn.Dropout(dropout_prob)
            # Optionally concatenate auxiliary features
            self.aux_proj = nn.Linear(aux_feat_dim, hidden_size)
            self.start_fc = nn.Linear(hidden_size, 1)
            self.end_fc = nn.Linear(hidden_size, 1)

        def forward(self, input_ids, attention_mask, aux_features):
            outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = outputs  # (batch, seq_len, hidden)
            # Broadcast aux features to each token position
            aux_emb = self.aux_proj(aux_features).unsqueeze(1)  # (batch, 1, hidden)
            sequence_output = sequence_output + aux_emb
            sequence_output = self.dropout(sequence_output)
            start_logits = self.start_fc(sequence_output).squeeze(-1)  # (batch, seq_len)
            end_logits = self.end_fc(sequence_output).squeeze(-1)
            return start_logits, end_logits

    # 3. Loss function: CrossEntropyLoss for start/end positions
    def compute_loss(start_logits, end_logits, start_labels, end_labels):
        loss_fn = nn.BCEWithLogitsLoss()
        start_loss = loss_fn(start_logits, start_labels)
        end_loss = loss_fn(end_logits, end_labels)
        return (start_loss + end_loss) / 2

    # 4. Prepare DataLoaders
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    EPOCHS = 3

    # Split train/val
    train_split, val_split = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df["sentiment"])
    train_dataset = TweetSpanDataset(train_split, max_length=MAX_LENGTH, is_train=True)
    val_dataset = TweetSpanDataset(val_split, max_length=MAX_LENGTH, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 5. Initialize model, optimizer, scheduler
    model = SpanExtractionModel(model_name=MODEL_NAME, aux_feat_dim=4, dropout_prob=0.2)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # 6. Training loop with validation and checkpointing
    best_val_loss = float("inf")
    checkpoint_path = "best_model.pt"

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            aux_features = batch["aux_features"].to(device)
            start_labels = batch["start_labels"].to(device)
            end_labels = batch["end_labels"].to(device)

            start_logits, end_logits = model(input_ids, attention_mask, aux_features)
            loss = compute_loss(start_logits, end_logits, start_labels, end_labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                aux_features = batch["aux_features"].to(device)
                start_labels = batch["start_labels"].to(device)
                end_labels = batch["end_labels"].to(device)

                start_logits, end_logits = model(input_ids, attention_mask, aux_features)
                loss = compute_loss(start_logits, end_logits, start_labels, end_labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model saved at epoch {epoch+1} with val loss {best_val_loss:.4f}")

    print("=== Model Training: Complete ===")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model checkpoint saved to: {checkpoint_path}")

except Exception as e:
    print(f"Error during model training: {e}")


# ⚠️ ALEXANDRIA MARKER - DO NOT DELETE (used for syncing outputs from Kaggle)
print("===ALEXANDRIA_CELL_7_START===")

print("===ALEXANDRIA_CELL_7_START===")

# Evaluation: Jaccard Similarity Metric for Tweet Sentiment Extraction

try:
    print("=== Evaluation: Start ===")

    # 1. Jaccard similarity function
    def jaccard(str1, str2):
        a = set(str1.lower().split())
        b = set(str2.lower().split())
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return float(len(a & b)) / float(len(a | b))

    # 2. Span extraction utility: from logits to text span
    def get_selected_text(text, offsets, start_idx, end_idx):
        # Ensure valid indices
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        char_start = offsets[start_idx]
        char_end = offsets[end_idx][1]
        return text[char_start:char_end]

    # 3. Prepare validation DataLoader
    val_dataset = TweetSpanDataset(val_split, max_length=128, is_train=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    # 4. Load best model weights
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.eval()

    # 5. Predict selected_text on validation set
    val_preds = []
    val_truths = []
    val_texts = []
    val_sentiments = []
    val_offsets = []
    val_ids = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            aux_features = batch["aux_features"].to(device)
            start_logits, end_logits = model(input_ids, attention_mask, aux_features)
            start_probs = torch.sigmoid(start_logits).cpu().numpy()
            end_probs = torch.sigmoid(end_logits).cpu().numpy()
            batch_size = input_ids.size(0)
            for j in range(batch_size):
                # Get offsets and original text
                idx = i * val_loader.batch_size + j
                row = val_split.iloc[idx]
                offsets = row["model_inputs"]["offset_mapping"]
                text = row["text"]
                # Find most probable start/end
                start_idx = np.argmax(start_probs[j])
                end_idx = np.argmax(end_probs[j])
                # Extract predicted span
                pred_span = get_selected_text(text, offsets, start_idx, end_idx)
                val_preds.append(pred_span)
                val_truths.append(row["selected_text"])
                val_texts.append(text)
                val_sentiments.append(row["sentiment"])
                val_offsets.append(offsets)
                val_ids.append(row["textID"])

    # 6. Compute Jaccard similarity for each prediction
    jaccard_scores = [jaccard(p, t) for p, t in zip(val_preds, val_truths)]
    avg_jaccard = np.mean(jaccard_scores)
    print(f"Validation Jaccard Similarity: {avg_jaccard:.4f}")

    # 7. Error analysis: show worst predictions
    val_results = pd.DataFrame({
        "textID": val_ids,
        "text": val_texts,
        "sentiment": val_sentiments,
        "selected_text_true": val_truths,
        "selected_text_pred": val_preds,
        "jaccard": jaccard_scores,
    })
    print("\n--- Worst 10 Jaccard Cases ---")
    display(val_results.sort_values("jaccard").head(10))

    # 8. Visualize Jaccard distribution
    plt.figure(figsize=(8,4))
    sns.histplot(val_results["jaccard"], bins=30, kde=True, color="purple")
    plt.title("Validation Jaccard Similarity Distribution")
    plt.xlabel("Jaccard Similarity")
    plt.ylabel("Frequency")
    plt.show()

    # 9. Baseline: always select full tweet for neutral sentiment
    neutral_mask = val_results["sentiment"] == "neutral"
    baseline_preds = []
    for i, row in val_results.iterrows():
        if row["sentiment"] == "neutral":
            baseline_preds.append(row["text"])
        else:
            baseline_preds.append(row["selected_text_pred"])
    baseline_jaccard = np.mean([jaccard(p, t) for p, t in zip(baseline_preds, val_results["selected_text_true"])])
    print(f"Baseline (full tweet for neutral) Jaccard: {baseline_jaccard:.4f}")

    print("=== Evaluation: Complete ===")

except Exception as e:
    print(f"Error during evaluation: {e}")


# ⚠️ ALEXANDRIA MARKER - DO NOT DELETE (used for syncing outputs from Kaggle)
print("===ALEXANDRIA_CELL_8_START===")

print("===ALEXANDRIA_CELL_8_START===")

# Submission & Results: Generate predictions for test set, format for submission, and display summary

try:
    print("=== Submission & Results: Start ===")
    import torch
    import numpy as np
    import pandas as pd

    # 1. Load best model weights
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.eval()

    # 2. Prepare test DataLoader
    class TweetSpanTestDataset(torch.utils.data.Dataset):
        def __init__(self, df, max_length=128):
            self.df = df
            self.max_length = max_length

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            item = self.df.iloc[idx]["model_inputs"]
            input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(item["attention_mask"], dtype=torch.long)
            aux_features = torch.tensor(item["aux_features"], dtype=torch.float)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "aux_features": aux_features,
            }

    test_dataset = TweetSpanTestDataset(test_df, max_length=128)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    # 3. Span extraction utility for test set
    def get_selected_text_test(text, offsets, start_idx, end_idx):
        # Ensure valid indices
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        char_start = offsets[start_idx]
        char_end = offsets[end_idx][1]
        return text[char_start:char_end]

    # 4. Generate predictions for test set
    test_preds = []
    test_ids = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Predicting Test")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            aux_features = batch["aux_features"].to(device)
            start_logits, end_logits = model(input_ids, attention_mask, aux_features)
            start_probs = torch.sigmoid(start_logits).cpu().numpy()
            end_probs = torch.sigmoid(end_logits).cpu().numpy()
            batch_size = input_ids.size(0)
            for j in range(batch_size):
                idx = i * test_loader.batch_size + j
                if idx >= len(test_df):
                    continue
                row = test_df.iloc[idx]
                offsets = row["model_inputs"]["offset_mapping"]
                text = row["text"]
                sentiment = row["sentiment"]
                # For neutral sentiment, select full tweet (per competition baseline)
                if sentiment == "neutral" or len(text.strip()) == 0:
                    pred_span = text
                else:
                    start_idx = np.argmax(start_probs[j])
                    end_idx = np.argmax(end_probs[j])
                    # Clamp indices to valid range
                    start_idx = max(0, min(start_idx, len(offsets) - 1))
                    end_idx = max(0, min(end_idx, len(offsets) - 1))
                    pred_span = get_selected_text_test(text, offsets, start_idx, end_idx)
                    # Fallback: if empty, use full tweet
                    if not pred_span or pred_span.strip() == "":
                        pred_span = text
                test_preds.append(pred_span)
                test_ids.append(row["textID"])

    # 5. Format predictions for submission
    submission = sample_submission_df.copy()
    submission["selected_text"] = ""
    id2pred = dict(zip(test_ids, test_preds))
    for i, row in submission.iterrows():
        tid = row["textID"]
        if tid in id2pred:
            submission.at[i, "selected_text"] = id2pred[tid]
        else:
            # Fallback: empty string if not found
            submission.at[i, "selected_text"] = ""

    # 6. Save submission file
    submission_file = "submission.csv"
    submission.to_csv(submission_file, index=False)
    print(f"Submission file saved: {submission_file}")

    # 7. Display sample predictions and confirm format
    print("\n--- Sample Submission (first 10 rows) ---")
    display(submission.head(10))

    print("\nSubmission columns:", submission.columns.tolist())
    print("Submission shape:", submission.shape)
    print("Unique textID count:", submission['textID'].nunique())
    print("Any missing selected_text:", submission['selected_text'].isnull().sum())

    print("=== Submission & Results: Complete ===")

except Exception as e:
    print(f"Error during submission & results: {e}")

