!pip install -q --no-deps transformers==4.30.0
# !pip install -q tokenizers


"""
Tweet Sentiment Extraction - Kaggle Production Version
PhiÃªn báº£n tá»‘i Æ°u, Ä‘Ã£ test vÃ  Ä‘áº£m báº£o cháº¡y Ä‘Æ°á»£c trÃªn Kaggle

TÃ¡c giáº£: Optimized for Kaggle Competition
Cuá»™c thi: https://www.kaggle.com/c/tweet-sentiment-extraction
"""

# =====================================================
# BÆ¯á»šC 1: UPGRADE TRANSFORMERS
# =====================================================

# Upgrade transformers Ä‘á»ƒ tÆ°Æ¡ng thÃ­ch vá»›i tokenizers má»›i
!pip install -q transformers --upgrade

import os
import gc
import re
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Machine Learning
from sklearn.model_selection import train_test_split

# Deep Learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm.auto import tqdm

print("âœ… Import thÃ nh cÃ´ng!")
print(f"ğŸ”¥ PyTorch version: {torch.__version__}")
print(f"ğŸ’» CUDA available: {torch.cuda.is_available()}")

# =====================================================
# BÆ¯á»šC 2: Cáº¤U HÃŒNH
# =====================================================

class Config:
    """Cáº¥u hÃ¬nh Ä‘Ã£ Ä‘Æ°á»£c tá»‘i Æ°u cho Kaggle"""
    
    # Paths
    DATA_PATH = '/kaggle/input/tweet-sentiment-extraction'
    OUTPUT_PATH = '/kaggle/working'
    
    # Model - DÃ¹ng model Ä‘Ã£ verify
    MODEL_NAME = 'distilbert-base-uncased'
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 32
    
    # Training
    EPOCHS = 3
    LEARNING_RATE = 3e-5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    
    # Settings
    SEED = 42
    USE_SUBSET = False  # Set True Ä‘á»ƒ test nhanh
    SUBSET_SIZE = 1000
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed=42):
    """Set seed cho reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(Config.SEED)
print(f"âœ… Device: {Config.DEVICE}")
print(f"âœ… Seed: {Config.SEED}")

# =====================================================
# BÆ¯á»šC 3: LOAD DATA
# =====================================================

print("\n" + "="*70)
print("ğŸ“Š LOAD DATA")
print("="*70)

train_df = pd.read_csv(f'{Config.DATA_PATH}/train.csv')
test_df = pd.read_csv(f'{Config.DATA_PATH}/test.csv')
sample_submission = pd.read_csv(f'{Config.DATA_PATH}/sample_submission.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Xá»­ lÃ½ missing values
train_df['text'] = train_df['text'].astype(str)
train_df['selected_text'] = train_df['selected_text'].fillna(train_df['text'])
test_df['text'] = test_df['text'].astype(str)

if Config.USE_SUBSET:
    train_df = train_df.sample(n=Config.SUBSET_SIZE, random_state=Config.SEED).reset_index(drop=True)
    print(f"âš ï¸� Using subset: {len(train_df)} samples")

print("\nâœ… Data loaded successfully!")
print(train_df.head())

# Quick EDA
print("\nğŸ“Š Sentiment Distribution:")
print(train_df['sentiment'].value_counts())

# =====================================================
# BÆ¯á»šC 4: VISUALIZATION (OPTIONAL)
# =====================================================

print("\nğŸ“Š Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Sentiment Distribution
train_df['sentiment'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['#FF6B6B', '#4ECDC4', '#FFE66D'])
axes[0, 0].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Sentiment')
axes[0, 0].set_ylabel('Count')

# 2. Text Length
train_df['text_len'] = train_df['text'].str.len()
axes[0, 1].hist(train_df['text_len'], bins=50, color='#95E1D3', edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Text Length Distribution', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Character Count')
axes[0, 1].set_ylabel('Frequency')

# 3. Selected Text Length
train_df['selected_len'] = train_df['selected_text'].str.len()
axes[1, 0].hist(train_df['selected_len'], bins=50, color='#F38181', edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Selected Text Length Distribution', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Character Count')
axes[1, 0].set_ylabel('Frequency')

# 4. Selection Ratio by Sentiment
train_df['ratio'] = train_df['selected_len'] / train_df['text_len']
train_df.boxplot(column='ratio', by='sentiment', ax=axes[1, 1])
axes[1, 1].set_title('Selection Ratio by Sentiment', fontsize=14, fontweight='bold')
plt.sca(axes[1, 1])
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(f'{Config.OUTPUT_PATH}/eda_visualization.png', dpi=200, bbox_inches='tight')
plt.show()

print("âœ… Visualizations saved!")

# =====================================================
# BÆ¯á»šC 5: LOAD TOKENIZER
# =====================================================

print("\n" + "="*70)
print("ğŸ“¥ LOADING TOKENIZER")
print("="*70)

try:
    # Táº£i tokenizer trá»±c tiáº¿p, khÃ´ng dÃ¹ng offline mode
    tokenizer = AutoTokenizer.from_pretrained(
        Config.MODEL_NAME,
        use_fast=True
    )
    print(f"âœ… Tokenizer loaded: {Config.MODEL_NAME}")
except Exception as e:
    print(f"âš ï¸� Error: {e}")
    print("ğŸ”„ Trying bert-base-uncased...")
    
    # Fallback sang model khÃ¡c
    Config.MODEL_NAME = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, use_fast=True)
    print(f"âœ… Tokenizer loaded: {Config.MODEL_NAME}")

# =====================================================
# BÆ¯á»šC 6: DATASET CLASS
# =====================================================

class TweetDataset(Dataset):
    """Dataset cho Tweet Sentiment Extraction"""
    
    def __init__(self, texts, sentiments, selected_texts, tokenizer, max_len):
        self.texts = texts
        self.sentiments = sentiments
        self.selected_texts = selected_texts
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        sentiment = str(self.sentiments[idx])
        selected = str(self.selected_texts[idx])
        
        # Táº¡o question-answering format
        question = f"What is the {sentiment} part?"
        
        # Tokenize
        encoding = self.tokenizer.encode_plus(
            question,
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # TÃ¬m vá»‹ trÃ­ cá»§a selected_text trong text
        start_char = text.find(selected)
        end_char = start_char + len(selected) if start_char != -1 else 0
        
        # Convert sang token positions (simplified)
        start_token = 0
        end_token = 0
        
        if start_char != -1:
            # Encode láº¡i Ä‘á»ƒ tÃ¬m offset
            char_to_token = self.tokenizer.encode_plus(
                question,
                text,
                return_offsets_mapping=True,
                add_special_tokens=True,
                max_length=self.max_len,
                truncation=True
            )
            
            offsets = char_to_token['offset_mapping']
            
            # TÃ¬m token position
            for i, (offset_start, offset_end) in enumerate(offsets):
                # Skip special tokens vÃ  question part
                if offset_start == 0 and offset_end == 0:
                    continue
                    
                if offset_start <= start_char < offset_end:
                    start_token = i
                if offset_start < end_char <= offset_end:
                    end_token = i
                    break
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'start_positions': torch.tensor(start_token, dtype=torch.long),
            'end_positions': torch.tensor(end_token, dtype=torch.long)
        }

# =====================================================
# BÆ¯á»šC 7: MODEL
# =====================================================

class SentimentExtractor(nn.Module):
    """Model cho extraction task"""
    
    def __init__(self, model_name):
        super(SentimentExtractor, self).__init__()
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )
        return outputs

# =====================================================
# BÆ¯á»šC 8: TRAINING FUNCTIONS
# =====================================================

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train má»™t epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)

def eval_epoch(model, dataloader, device):
    """Evaluate má»™t epoch"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions
            )
            
            total_loss += outputs.loss.item()
    
    return total_loss / len(dataloader)

# =====================================================
# BÆ¯á»šC 9: JACCARD METRIC
# =====================================================

def jaccard_score(str1, str2):
    """TÃ­nh Jaccard similarity"""
    set1 = set(str(str1).lower().split())
    set2 = set(str(str2).lower().split())
    
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0

# =====================================================
# BÆ¯á»šC 10: MAIN TRAINING
# =====================================================

print("\n" + "="*70)
print("ğŸš€ TRAINING MODEL")
print("="*70)

# Split data
train_data, val_data = train_test_split(
    train_df,
    test_size=0.1,
    random_state=Config.SEED,
    stratify=train_df['sentiment']
)

print(f"\nTrain size: {len(train_data)}")
print(f"Val size: {len(val_data)}")

# Create datasets
train_dataset = TweetDataset(
    texts=train_data['text'].values,
    sentiments=train_data['sentiment'].values,
    selected_texts=train_data['selected_text'].values,
    tokenizer=tokenizer,
    max_len=Config.MAX_LEN
)

val_dataset = TweetDataset(
    texts=val_data['text'].values,
    sentiments=val_data['sentiment'].values,
    selected_texts=val_data['selected_text'].values,
    tokenizer=tokenizer,
    max_len=Config.MAX_LEN
)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=Config.TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=Config.VALID_BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print("âœ… DataLoaders created!")

# Initialize model
print(f"\nğŸ¤– Loading model: {Config.MODEL_NAME}")
model = SentimentExtractor(Config.MODEL_NAME)
model.to(Config.DEVICE)
print(f"âœ… Model loaded on {Config.DEVICE}!")

# Optimizer & Scheduler
optimizer = AdamW(
    model.parameters(),
    lr=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY
)

total_steps = len(train_loader) * Config.EPOCHS
warmup_steps = int(Config.WARMUP_RATIO * total_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

print(f"\nâš™ï¸� Total steps: {total_steps}")
print(f"âš™ï¸� Warmup steps: {warmup_steps}")

# Training loop
best_val_loss = float('inf')
history = {'train_loss': [], 'val_loss': []}

for epoch in range(Config.EPOCHS):
    print(f"\n{'='*70}")
    print(f"ğŸ“š Epoch {epoch + 1}/{Config.EPOCHS}")
    print(f"{'='*70}")
    
    # Train
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, Config.DEVICE)
    
    # Validate
    val_loss = eval_epoch(model, val_loader, Config.DEVICE)
    
    # Save history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    
    print(f"\nğŸ“Š Results:")
    print(f"   Train Loss: {train_loss:.4f}")
    print(f"   Val Loss: {val_loss:.4f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f'{Config.OUTPUT_PATH}/best_model.pth')
        print(f"   ğŸ’¾ Best model saved!")
    
    # Clear cache
    gc.collect()
    torch.cuda.empty_cache()

print("\nâœ… Training completed!")

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history['train_loss'], label='Train Loss', marker='o', linewidth=2)
plt.plot(history['val_loss'], label='Val Loss', marker='s', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training History', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig(f'{Config.OUTPUT_PATH}/training_history.png', dpi=200, bbox_inches='tight')
plt.show()

# =====================================================
# BÆ¯á»šC 11: PREDICTION
# =====================================================

print("\n" + "="*70)
print("ğŸ”® MAKING PREDICTIONS")
print("="*70)

def predict_selected_text(model, text, sentiment, tokenizer, max_len, device):
    """Predict selected text"""
    model.eval()
    
    question = f"What is the {sentiment} part?"
    
    encoding = tokenizer.encode_plus(
        question,
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    start_idx = torch.argmax(outputs.start_logits, dim=1).item()
    end_idx = torch.argmax(outputs.end_logits, dim=1).item()
    
    if end_idx < start_idx:
        end_idx = start_idx
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    predicted_text = tokenizer.convert_tokens_to_string(tokens[start_idx:end_idx+1])
    
    # Clean prediction
    predicted_text = predicted_text.strip()
    if not predicted_text or predicted_text == '[CLS]' or predicted_text == '[SEP]':
        predicted_text = text
    
    return predicted_text

# Load best model
model.load_state_dict(torch.load(f'{Config.OUTPUT_PATH}/best_model.pth'))
print("âœ… Best model loaded!")

# Validate on some examples
print("\nğŸ”� Sample Predictions:")
print("="*70)

samples = val_data.sample(n=min(5, len(val_data)), random_state=Config.SEED)

total_jaccard = 0
for idx, row in samples.iterrows():
    text = row['text']
    sentiment = row['sentiment']
    true_selected = row['selected_text']
    
    predicted = predict_selected_text(model, text, sentiment, tokenizer, Config.MAX_LEN, Config.DEVICE)
    score = jaccard_score(true_selected, predicted)
    total_jaccard += score
    
    print(f"\nText: {text}")
    print(f"Sentiment: {sentiment}")
    print(f"True: {true_selected}")
    print(f"Predicted: {predicted}")
    print(f"Jaccard: {score:.4f}")
    print("-"*70)

print(f"\nğŸ“Š Average Jaccard Score: {total_jaccard / len(samples):.4f}")

# =====================================================
# BÆ¯á»šC 12: GENERATE SUBMISSION
# =====================================================

print("\n" + "="*70)
print("ğŸ“� GENERATING SUBMISSION")
print("="*70)

predictions = []

for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc='Predicting'):
    text = row['text']
    sentiment = row['sentiment']
    
    predicted = predict_selected_text(model, text, sentiment, tokenizer, Config.MAX_LEN, Config.DEVICE)
    predictions.append(predicted)

# Create submission
submission = pd.DataFrame({
    'textID': test_df['textID'],
    'selected_text': predictions
})

submission.to_csv(f'{Config.OUTPUT_PATH}/submission.csv', index=False)

print("âœ… Submission file created!")
print(f"\nğŸ“� Files saved:")
print(f"   - best_model.pth")
print(f"   - submission.csv")
print(f"   - training_history.png")
print(f"   - eda_visualization.png")

print("\n" + "="*70)
print("ğŸ�‰ ALL DONE!")
print("="*70)
print(f"Final validation loss: {best_val_loss:.4f}")
print("\nğŸ’¡ Next steps:")
print("   1. Download submission.csv")
print("   2. Submit to Kaggle competition")
print("   3. Check leaderboard score!")


"""
Tweet Sentiment Extraction - Kaggle Notebook Version
PhiÃªn báº£n tá»‘i Æ°u cho Kaggle vá»›i GPU support vÃ  data paths

TÃ¡c giáº£: [TÃªn báº¡n]
Dá»±a trÃªn cuá»™c thi: https://www.kaggle.com/c/tweet-sentiment-extraction
Tham kháº£o: https://www.kaggle.com/code/tanulsingh077/twitter-sentiment-extaction-analysis-eda-and-model
"""

# =====================================================
# BÆ¯á»šC 1: CÃ€I Ä�áº¶T VÃ€ IMPORT THÆ¯ VIá»†N
# =====================================================

import os
import gc
import re
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score

# Deep Learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm.auto import tqdm

# Cáº¥u hÃ¬nh
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("âœ… Import thÃ nh cÃ´ng!")
print(f"ğŸ”¥ PyTorch version: {torch.__version__}")
print(f"ğŸ’» CUDA available: {torch.cuda.is_available()}")

# =====================================================
# BÆ¯á»šC 2: Cáº¤U HÃŒNH VÃ€ THIáº¾T Láº¬P
# =====================================================

class Config:
    """Cáº¥u hÃ¬nh toÃ n bá»™ hyperparameters"""
    
    # Paths - Kaggle specific
    DATA_PATH = '/kaggle/input/tweet-sentiment-extraction'
    OUTPUT_PATH = '/kaggle/working'
    
    # Model settings - Sá»¬A TÃŠN MODEL
    MODEL_NAME = 'bert-base-uncased'  # KHÃ”NG PHáº¢I 'distilbert/distilbert-base-uncased'
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 32
    
    # Training settings
    EPOCHS = 3
    LEARNING_RATE = 3e-5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    
    # Other settings
    SEED = 42
    N_SPLITS = 5  # Cho K-Fold Cross Validation
    USE_SUBSET = False  # Set True náº¿u muá»‘n test vá»›i dá»¯ liá»‡u nhá»�
    SUBSET_SIZE = 5000
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set seeds
def seed_everything(seed=42):
    """Set seed cho táº¥t cáº£ random generators"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(Config.SEED)
print(f"\nâœ… Ä�Ã£ thiáº¿t láº­p seed: {Config.SEED}")
print(f"ğŸ’» Device: {Config.DEVICE}")

# =====================================================
# BÆ¯á»šC 3: LOAD VÃ€ PHÃ‚N TÃ�CH Dá»® LIá»†U (EDA)
# =====================================================

print("\n" + "="*70)
print("ğŸ“Š PHáº¦N 1: EXPLORATORY DATA ANALYSIS (EDA)")
print("="*70)

# Load data
train_df = pd.read_csv(f'{Config.DATA_PATH}/train.csv')
test_df = pd.read_csv(f'{Config.DATA_PATH}/test.csv')

print(f"\nğŸ“ˆ KÃ­ch thÆ°á»›c dá»¯ liá»‡u:")
print(f"   Train: {train_df.shape}")
print(f"   Test: {test_df.shape}")

# Sá»­ dá»¥ng subset náº¿u cáº§n
if Config.USE_SUBSET:
    train_df = train_df.sample(n=Config.SUBSET_SIZE, random_state=Config.SEED).reset_index(drop=True)
    print(f"âš ï¸�  Ä�ang dÃ¹ng subset: {len(train_df)} samples")

# Xem dá»¯ liá»‡u máº«u
print(f"\nğŸ‘€ 5 dÃ²ng Ä‘áº§u tiÃªn cá»§a train data:")
display(train_df.head())

print(f"\nğŸ‘€ 5 dÃ²ng Ä‘áº§u tiÃªn cá»§a test data:")
display(test_df.head())

# Kiá»ƒm tra missing values
print(f"\nâ�Œ Missing values trong train:")
print(train_df.isnull().sum())

# Xá»­ lÃ½ missing values
train_df['selected_text'] = train_df['selected_text'].fillna(train_df['text'])

# PhÃ¢n tÃ­ch sentiment distribution
print(f"\nğŸ“Š PhÃ¢n bá»‘ Sentiment:")
sentiment_counts = train_df['sentiment'].value_counts()
print(sentiment_counts)
print(f"\nTá»· lá»‡ (%):")
print(train_df['sentiment'].value_counts(normalize=True) * 100)

# ThÃªm cÃ¡c features phÃ¢n tÃ­ch
train_df['text_len'] = train_df['text'].apply(lambda x: len(str(x)))
train_df['text_word_count'] = train_df['text'].apply(lambda x: len(str(x).split()))
train_df['selected_text_len'] = train_df['selected_text'].apply(lambda x: len(str(x)))
train_df['selected_word_count'] = train_df['selected_text'].apply(lambda x: len(str(x).split()))

print(f"\nğŸ“� Thá»‘ng kÃª Ä‘á»™ dÃ i:")
print(train_df[['text_len', 'text_word_count', 'selected_text_len', 'selected_word_count']].describe())

# =====================================================
# BÆ¯á»šC 4: VISUALIZATION
# =====================================================

print(f"\nğŸ“Š Táº¡o visualizations...")

fig, axes = plt.subplots(3, 2, figsize=(16, 14))

# 1. Sentiment Distribution
sentiment_counts.plot(kind='bar', ax=axes[0, 0], color=['#FF6B6B', '#4ECDC4', '#FFE66D'])
axes[0, 0].set_title('PhÃ¢n bá»‘ Sentiment', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Sentiment')
axes[0, 0].set_ylabel('Sá»‘ lÆ°á»£ng')
axes[0, 0].tick_params(axis='x', rotation=0)

# 2. Text Length Distribution
axes[0, 1].hist(train_df['text_word_count'], bins=50, color='#95E1D3', edgecolor='black', alpha=0.7)
axes[0, 1].set_title('PhÃ¢n bá»‘ sá»‘ tá»« trong Text', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Sá»‘ tá»«')
axes[0, 1].set_ylabel('Táº§n suáº¥t')
axes[0, 1].axvline(train_df['text_word_count'].mean(), color='red', linestyle='--', label=f'Mean: {train_df["text_word_count"].mean():.1f}')
axes[0, 1].legend()

# 3. Selected Text Length Distribution
axes[1, 0].hist(train_df['selected_word_count'], bins=50, color='#F38181', edgecolor='black', alpha=0.7)
axes[1, 0].set_title('PhÃ¢n bá»‘ sá»‘ tá»« trong Selected Text', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Sá»‘ tá»«')
axes[1, 0].set_ylabel('Táº§n suáº¥t')
axes[1, 0].axvline(train_df['selected_word_count'].mean(), color='red', linestyle='--', label=f'Mean: {train_df["selected_word_count"].mean():.1f}')
axes[1, 0].legend()

# 4. Scatter: Text vs Selected Length by Sentiment
for sentiment, color in zip(['positive', 'negative', 'neutral'], ['#4ECDC4', '#FF6B6B', '#FFE66D']):
    data = train_df[train_df['sentiment'] == sentiment]
    axes[1, 1].scatter(data['text_word_count'], data['selected_word_count'], 
                      alpha=0.5, label=sentiment, color=color, s=10)
axes[1, 1].set_title('Text Length vs Selected Length theo Sentiment', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Text Word Count')
axes[1, 1].set_ylabel('Selected Text Word Count')
axes[1, 1].legend()
axes[1, 1].plot([0, train_df['text_word_count'].max()], [0, train_df['text_word_count'].max()], 
               'r--', alpha=0.5, label='y=x')

# 5. Box plot: Word count by sentiment
train_df.boxplot(column='selected_word_count', by='sentiment', ax=axes[2, 0])
axes[2, 0].set_title('PhÃ¢n bá»‘ Ä‘á»™ dÃ i Selected Text theo Sentiment', fontsize=14, fontweight='bold')
axes[2, 0].set_xlabel('Sentiment')
axes[2, 0].set_ylabel('Selected Word Count')
plt.sca(axes[2, 0])
plt.xticks(rotation=0)

# 6. Ratio of selected text to full text
train_df['selection_ratio'] = train_df['selected_word_count'] / train_df['text_word_count']
train_df.boxplot(column='selection_ratio', by='sentiment', ax=axes[2, 1])
axes[2, 1].set_title('Tá»· lá»‡ Selected/Total theo Sentiment', fontsize=14, fontweight='bold')
axes[2, 1].set_xlabel('Sentiment')
axes[2, 1].set_ylabel('Selection Ratio')
plt.sca(axes[2, 1])
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(f'{Config.OUTPUT_PATH}/eda_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("âœ… Ä�Ã£ lÆ°u biá»ƒu Ä‘á»“ phÃ¢n tÃ­ch!")

# Word Cloud cho má»—i sentiment
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, sentiment in enumerate(['positive', 'negative', 'neutral']):
    text = ' '.join(train_df[train_df['sentiment'] == sentiment]['selected_text'].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                         colormap='Set2', max_words=100).generate(text)
    
    axes[idx].imshow(wordcloud, interpolation='bilinear')
    axes[idx].set_title(f'Word Cloud - {sentiment.upper()}', fontsize=16, fontweight='bold')
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(f'{Config.OUTPUT_PATH}/wordcloud_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("âœ… Ä�Ã£ táº¡o Word Clouds!")

# =====================================================
# BÆ¯á»šC 5: DATA PREPROCESSING
# =====================================================

print("\n" + "="*70)
print("ğŸ”§ PHáº¦N 2: DATA PREPROCESSING")
print("="*70)

def clean_text(text):
    """LÃ m sáº¡ch text (optional - cÃ³ thá»ƒ bá»� qua cho BERT models)"""
    text = str(text)
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    return text

def find_all(text, selected):
    """TÃ¬m vá»‹ trÃ­ start vÃ  end cá»§a selected text trong text"""
    # Chuyá»ƒn sang string vÃ  xá»­ lÃ½ NaN
    text = str(text) if pd.notna(text) else ""
    selected = str(selected) if pd.notna(selected) else ""
    
    if not text or not selected:
        return 0, 0
    
    start_idx = text.find(selected)
    if start_idx == -1:
        return 0, 0
    end_idx = start_idx + len(selected)
    return start_idx, end_idx

# TÃ¬m vá»‹ trÃ­ char-level cá»§a selected_text
print("\nğŸ”� Ä�ang tÃ¬m vá»‹ trÃ­ cá»§a selected_text trong text...")
train_df['start_idx'] = 0
train_df['end_idx'] = 0

for idx in tqdm(range(len(train_df))):
    text = train_df.loc[idx, 'text']
    selected = train_df.loc[idx, 'selected_text']
    start, end = find_all(text, selected)
    train_df.loc[idx, 'start_idx'] = start
    train_df.loc[idx, 'end_idx'] = end

print("âœ… HoÃ n thÃ nh preprocessing!")

# =====================================================
# BÆ¯á»šC 6: DATASET VÃ€ DATALOADER
# =====================================================

print("\n" + "="*70)
print("ğŸ�¯ PHáº¦N 3: Táº O DATASET VÃ€ DATALOADER")
print("="*70)

class TweetDataset(Dataset):
    """
    Custom Dataset cho Tweet Sentiment Extraction
    Sá»­ dá»¥ng Question-Answering approach
    """
    
    def __init__(self, texts, sentiments, selected_texts, tokenizer, max_len):
        self.texts = texts
        self.sentiments = sentiments
        self.selected_texts = selected_texts
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        sentiment = str(self.sentiments[idx])
        selected = str(self.selected_texts[idx])
        
        # Táº¡o question based on sentiment
        question = f"What is the {sentiment} part of this text?"
        
        # Tokenize vá»›i offset mapping Ä‘á»ƒ tÃ¬m vá»‹ trÃ­ chÃ­nh xÃ¡c
        encoding = self.tokenizer(
            question,
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # TÃ¬m vá»‹ trÃ­ cá»§a selected text trong tokenized sequence
        offset_mapping = encoding['offset_mapping'][0]
        
        # TÃ¬m char position cá»§a selected text
        start_char = text.find(selected)
        end_char = start_char + len(selected) if start_char != -1 else 0
        
        # Convert char positions to token positions
        start_token = 0
        end_token = 0
        
        if start_char != -1:
            for i, (start, end) in enumerate(offset_mapping):
                if start <= start_char < end:
                    start_token = i
                if start < end_char <= end:
                    end_token = i
                    break
        
        # Ensure valid positions
        start_token = min(start_token, self.max_len - 1)
        end_token = min(end_token, self.max_len - 1)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'start_positions': torch.tensor(start_token, dtype=torch.long),
            'end_positions': torch.tensor(end_token, dtype=torch.long),
            'offset_mapping': offset_mapping
        }

# Load tokenizer
print(f"\nğŸ“¥ Loading tokenizer: {Config.MODEL_NAME}")
# Workaround cho lá»—i chat_template trong transformers má»›i
import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

tokenizer = AutoTokenizer.from_pretrained(
    Config.MODEL_NAME,
    use_fast=True,
    local_files_only=False
)
print("âœ… Tokenizer loaded!")

# =====================================================
# BÆ¯á»šC 7: MODEL DEFINITION
# =====================================================

print("\n" + "="*70)
print("ğŸ¤– PHáº¦N 4: MODEL DEFINITION")
print("="*70)

class SentimentExtractor(nn.Module):
    """
    Wrapper cho Question-Answering model
    """
    
    def __init__(self, model_name):
        super(SentimentExtractor, self).__init__()
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )
        return outputs

# =====================================================
# BÆ¯á»šC 8: TRAINING FUNCTIONS
# =====================================================

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Training cho 1 epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)

def eval_epoch(model, dataloader, device):
    """Evaluation cho 1 epoch"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions
            )
            
            loss = outputs.loss
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

# =====================================================
# BÆ¯á»šC 9: EVALUATION METRIC
# =====================================================

def jaccard_score(str1, str2):
    """
    TÃ­nh Jaccard Similarity - Metric chÃ­nh thá»©c cá»§a cuá»™c thi
    """
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0

def calculate_jaccard(model, dataloader, tokenizer, device, original_texts):
    """TÃ­nh Jaccard score trÃªn toÃ n bá»™ dataset"""
    model.eval()
    scores = []
    
    idx = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Calculating Jaccard'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            offset_mapping = batch['offset_mapping']
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            
            # Get predictions
            start_preds = torch.argmax(start_logits, dim=1).cpu().numpy()
            end_preds = torch.argmax(end_logits, dim=1).cpu().numpy()
            
            # Decode predictions
            for i in range(len(start_preds)):
                start = start_preds[i]
                end = end_preds[i]
                
                if end < start:
                    end = start
                
                # Get tokens
                tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
                selected_tokens = tokens[start:end+1]
                predicted_text = tokenizer.convert_tokens_to_string(selected_tokens)
                
                # Get true text
                true_text = original_texts[idx]['selected_text']
                
                # Calculate jaccard
                score = jaccard_score(predicted_text, true_text)
                scores.append(score)
                
                idx += 1
    
    return np.mean(scores)

# =====================================================
# BÆ¯á»šC 10: MAIN TRAINING LOOP
# =====================================================

print("\n" + "="*70)
print("ğŸš€ PHáº¦N 5: TRAINING MODEL")
print("="*70)

# TÃ¡ch train/validation
train_data, val_data = train_test_split(
    train_df, 
    test_size=0.1, 
    random_state=Config.SEED,
    stratify=train_df['sentiment']
)

print(f"\nğŸ“Š KÃ­ch thÆ°á»›c datasets:")
print(f"   Train: {len(train_data)}")
print(f"   Validation: {len(val_data)}")

# Táº¡o datasets
train_dataset = TweetDataset(
    texts=train_data['text'].values,
    sentiments=train_data['sentiment'].values,
    selected_texts=train_data['selected_text'].values,
    tokenizer=tokenizer,
    max_len=Config.MAX_LEN
)

val_dataset = TweetDataset(
    texts=val_data['text'].values,
    sentiments=val_data['sentiment'].values,
    selected_texts=val_data['selected_text'].values,
    tokenizer=tokenizer,
    max_len=Config.MAX_LEN
)

# Táº¡o dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=Config.TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=Config.VALID_BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print(f"âœ… Ä�Ã£ táº¡o DataLoaders!")

# Initialize model
print(f"\nğŸ¤– Loading model: {Config.MODEL_NAME}")
model = SentimentExtractor(Config.MODEL_NAME)
model.to(Config.DEVICE)
print(f"âœ… Model loaded vÃ  moved to {Config.DEVICE}!")

# Optimizer vÃ  Scheduler
optimizer = AdamW(
    model.parameters(),
    lr=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY
)

total_steps = len(train_loader) * Config.EPOCHS
warmup_steps = int(Config.WARMUP_RATIO * total_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

print(f"\nâš™ï¸�  Training configuration:")
print(f"   Total steps: {total_steps}")
print(f"   Warmup steps: {warmup_steps}")
print(f"   Learning rate: {Config.LEARNING_RATE}")

# Training loop
best_val_loss = float('inf')
history = {'train_loss': [], 'val_loss': []}

print(f"\nğŸ�¯ Báº¯t Ä‘áº§u training {Config.EPOCHS} epochs...")

for epoch in range(Config.EPOCHS):
    print(f"\n{'='*70}")
    print(f"ğŸ“š Epoch {epoch + 1}/{Config.EPOCHS}")
    print(f"{'='*70}")
    
    # Train
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, Config.DEVICE)
    
    # Validate
    val_loss = eval_epoch(model, val_loader, Config.DEVICE)
    
    # Save history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    
    print(f"\nğŸ“Š Epoch {epoch + 1} Summary:")
    print(f"   Train Loss: {train_loss:.4f}")
    print(f"   Val Loss: {val_loss:.4f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f'{Config.OUTPUT_PATH}/best_model.pth')
        print(f"   ğŸ’¾ Saved best model! (Val Loss: {val_loss:.4f})")
    
    # Clear cache
    gc.collect()
    torch.cuda.empty_cache()

print("\nâœ… Training hoÃ n thÃ nh!")

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history['train_loss'], label='Train Loss', marker='o')
plt.plot(history['val_loss'], label='Validation Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History', fontsize=16, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'{Config.OUTPUT_PATH}/training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# =====================================================
# BÆ¯á»šC 11: EVALUATION Vá»šI JACCARD SCORE
# =====================================================

print("\n" + "="*70)
print("ğŸ“Š PHáº¦N 6: EVALUATION Vá»šI JACCARD SCORE")
print("="*70)

# Load best model
model.load_state_dict(torch.load(f'{Config.OUTPUT_PATH}/best_model.pth'))
print("âœ… Ä�Ã£ load best model!")

# Prepare validation data for Jaccard calculation
val_texts = [{'text': t, 'selected_text': s} for t, s in zip(val_data['text'].values, val_data['selected_text'].values)]

# Calculate Jaccard Score
jaccard = calculate_jaccard(model, val_loader, tokenizer, Config.DEVICE, val_texts)
print(f"\nğŸ�¯ Validation Jaccard Score: {jaccard:.4f}")

# =====================================================
# BÆ¯á»šC 12: PREDICTION EXAMPLES
# =====================================================

print("\n" + "="*70)
print("ğŸ”� PHáº¦N 7: Má»˜T Sá»� VÃ� Dá»¤ Dá»° Ä�OÃ�N")
print("="*70)

def predict_selected_text(model, text, sentiment, tokenizer, max_len, device):
    """Dá»± Ä‘oÃ¡n selected text cho má»™t tweet"""
    model.eval()
    
    question = f"What is the {sentiment} part of this text?"
    
    encoding = tokenizer(
        question,
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits)
    
    if end_idx < start_idx:
        end_idx = start_idx
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    selected_tokens = tokens[start_idx:end_idx+1]
    predicted_text = tokenizer.convert_tokens_to_string(selected_tokens)
    
    return predicted_text

# Show some examples
sample_size = min(10, len(val_data))
samples = val_data.sample(n=sample_size, random_state=Config.SEED)

for idx, row in samples.iterrows():
    text = row['text']
    sentiment = row['sentiment']
    true_selected = row['selected_text']
    
    predicted = predict_selected_text(model, text, sentiment, tokenizer, Config.MAX_LEN, Config.DEVICE)
    score = jaccard_score(true_selected, predicted)
    
    print(f"\n{'='*70}")
    print(f"ğŸ“� Text: {text}")
    print(f"ğŸ˜Š Sentiment: {sentiment}")
    print(f"âœ… True: {true_selected}")
    print(f"ğŸ¤– Predicted: {predicted}")
    print(f"ğŸ“Š Jaccard Score: {score:.4f}")

print("\n" + "="*70)
print("âœ… HOÃ€N THÃ€NH Táº¤T Cáº¢ CÃ�C BÆ¯á»šC!")
print("="*70)

print(f"\nğŸ“� CÃ¡c file Ä‘Ã£ lÆ°u:")
print(f"   - best_model.pth")
print(f"   - eda_analysis.png")
print(f"   - wordcloud_analysis.png")
print(f"   - training_history.png")

print(f"\nğŸ�¯ Káº¿t quáº£ cuá»‘i cÃ¹ng:")
print(f"   Best Validation Loss: {best_val_loss:.4f}")
print(f"   Validation Jaccard Score: {jaccard:.4f}")




