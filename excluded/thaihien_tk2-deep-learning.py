!pip install evaluate
!pip install -U transformers huggingface_hub accelerate
!pip install seqeval
!pip install rouge-score


import os
import random
import re
import warnings
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
)

# Tải tokenizer cho NLTK
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION (Cập nhật cho Model Dự đoán Span)
# ============================================
class Config:
    TRAIN_PATH = '/kaggle/input/tweet-sentiment-extraction/train.csv'
    TEST_PATH = '/kaggle/input/tweet-sentiment-extraction/test.csv'
    SUBMISSION_PATH = '/kaggle/input/tweet-sentiment-extraction/sample_submission.csv'
    OUTPUT_DIR = './best_models_span' # Thư mục mới
    
    # Model
    MODEL_NAME = "roberta-base"
    MAX_LEN = 128
    DROPOUT = 0.3 # Giữ nguyên mức regularization mạnh
    
    # Training
    SEED = 42
    BATCH_SIZE = 16
    NUM_EPOCHS = 25 
    LR = 0.001 # Giữ nguyên LR thấp
    WEIGHT_DECAY = 0.05
    WARMUP_RATIO = 0.1
    VAL_SIZE = 0.12
    PATIENCE = 7 # Early Stopping
    
    # Data Augmentation (Giữ nguyên mức tinh chỉnh)
    AUG_PROB = 0.3 
    AUG_COUNT = 1
    AUG_DELETE_PROB = 0.02 
    AUG_SWAP_PROB = 0.02  
    AUG_MASK_PROB = 0.02  
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# SETUP (Giữ nguyên)
# ============================================
def set_seed(seed=Config.SEED):
    """Set seed cho reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# ============================================
# DATA PREPROCESSING & EDA (Giữ nguyên)
# ============================================
def normalize_text(text: str) -> str:
    """Chuẩn hóa text (giữ nguyên vị trí tương đối)"""
    if not isinstance(text, str):
        return ""
    
    s = text.lower()
    s = s.replace('\r\n', ' ').replace('\n', ' ')
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'(.)\1{2,}', r'\1\1', s)  
    s = re.sub(r'@\w+', '@user', s)
    s = re.sub(r'http\S+|www\.\S+', 'url', s)
    
    return s

def display_eda_stats(df: pd.DataFrame, title: str):
    """Hiển thị thống kê cơ bản và phân tích dữ liệu mở rộng"""
    # [Giữ nguyên code hiển thị EDA]
    if df.empty:
        print(f"\n--- Thống kê Dữ liệu {title} ---")
        print("Tập dữ liệu trống.")
        print("-" * 40)
        return

    print(f"\n--- Thống kê Dữ liệu {title} ({len(df)} samples) ---")
    
    if 'sentiment' in df.columns:
        print(f"Phân phối Sentiment:\n{df['sentiment'].value_counts().to_string()}")
    
    df['text_len'] = df['text_proc'].apply(lambda x: len(x.split()))
    
    print(f"\nChiều dài Tweet (số từ):")
    print(df['text_len'].describe().to_string())
    
    if 'selected_text_proc' in df.columns:
        df['selected_len'] = df['selected_text_proc'].apply(lambda x: len(x.split()))
        print(f"\nChiều dài Selected Text (số từ):")
        print(df['selected_len'].describe().to_string())
        
        df['ratio'] = df['selected_len'] / df['text_len']
        print(f"\nTỷ lệ Selected Text/Tweet:")
        print(df['ratio'].describe().to_string())
        
        words = ' '.join(df['selected_text_proc']).split()
        most_common = Counter(words).most_common(5)
        print(f"\n5 từ phổ biến nhất trong Selected Text: {most_common}")
        
    print("-" * 40)


def load_data():
    """Load và preprocess data, và loại bỏ sentiment 'neutral' khỏi tập train"""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    train = pd.read_csv(Config.TRAIN_PATH)
    test = pd.read_csv(Config.TEST_PATH)
    submission = pd.read_csv(Config.SUBMISSION_PATH)
    
    train = train.dropna().reset_index(drop=True)
    
    # Loại bỏ dữ liệu 'neutral' khỏi tập huấn luyện
    print("Loại bỏ dữ liệu có sentiment 'neutral' khỏi tập huấn luyện.")
    train = train[train['sentiment'].isin(['positive', 'negative'])].reset_index(drop=True)
    
    # Normalize text
    train['text_proc'] = train['text'].astype(str).apply(normalize_text)
    train['selected_text_proc'] = train['selected_text'].astype(str).apply(normalize_text)
    test['text_proc'] = test['text'].astype(str).apply(normalize_text)
    
    print(f"Train shape (sau khi loại bỏ 'neutral'): {train.shape}")
    print(f"Test shape: {test.shape}")
    
    display_eda_stats(train, "TRAIN (Positive/Negative Only)")
    
    return train, test, submission

# ============================================
# DATA AUGMENTATION LOGIC (Giữ nguyên)
# ============================================

def find_span_in_text(text: str, selected: str) -> Tuple[int, int]:
    """Tìm vị trí span (start, end) của selected_text trong text"""
    if not selected: return -1, -1
    start = text.find(selected)
    if start == -1: return -1, -1
    end = start + len(selected)
    return start, end

def get_word_boundaries(text: str) -> List[Tuple[int, int]]:
    """Lấy vị trí ký tự (start, end) của mỗi từ trong text"""
    words = []
    for match in re.finditer(r'\S+', text):
        words.append((match.start(), match.end()))
    return words

def augment_text(text: str, selected: str, tokenizer,
                 prob_delete=Config.AUG_DELETE_PROB,
                 prob_swap=Config.AUG_SWAP_PROB,
                 prob_mask=Config.AUG_MASK_PROB) -> str:
    """Thực hiện Data Augmentation chỉ áp dụng cho các từ nằm ngoài Selected Text."""
    start_char, end_char = find_span_in_text(text, selected)
    word_positions = get_word_boundaries(text)
    words = [text[s:e] for s, e in word_positions]
    
    outside_indices = []
    
    for i, (s, e) in enumerate(word_positions):
        is_inside = (start_char != -1 and s >= start_char and e <= end_char)
        if not is_inside:
            outside_indices.append(i)
            
    augmented_words = words.copy()
    
    # 1. Random deletion 
    for i in outside_indices:
        if random.random() < prob_delete:
            augmented_words[i] = ""
    
    # 2. Random swap 
    if len(outside_indices) >= 2 and random.random() < prob_swap:
        i, j = random.sample(outside_indices, 2)  
        augmented_words[i], augmented_words[j] = augmented_words[j], augmented_words[i]
    
    # 3. Random masking
    for i in outside_indices:
        if random.random() < prob_mask and augmented_words[i]:
            augmented_words[i] = tokenizer.mask_token
    
    augmented_words = [w for w in augmented_words if w]
    augmented_text = " ".join(augmented_words)
    
    if selected and selected not in normalize_text(augmented_text):
        return text  
    
    return augmented_text

# ============================================
# TOKENIZATION & MODEL & DATASET (CẬP NHẬT GÁN NHÃN SPAN)
# ============================================
def create_token_labels(texts: List[str], 
                        selected_texts: List[str],
                        tokenizer,
                        max_len=Config.MAX_LEN) -> Tuple[Dict, List[int], List[int]]:
    """Tạo encodings và gán nhãn token cho vị trí Bắt đầu (start_labels) và Kết thúc (end_labels)"""
    encodings = tokenizer(
        texts,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_offsets_mapping=True,
        return_tensors='pt'
    )
    
    start_labels = []
    end_labels = []
    
    for i in range(len(texts)):
        offsets = encodings['offset_mapping'][i].cpu().tolist()
        text = texts[i]
        selected = selected_texts[i]
        
        start_char, end_char = find_span_in_text(text, selected)
        
        start_index = 0
        end_index = 0
        
        # Nếu không tìm thấy selected text, gán nhãn start/end = 0 (CLS token)
        if start_char != -1:
            for j, (offset_start, offset_end) in enumerate(offsets):
                if offset_start == offset_end: continue # Special token/Padding
                
                # Tìm token chứa ký tự Bắt đầu của selected_text
                if offset_start <= start_char < offset_end:
                    start_index = j
                
                # Tìm token chứa ký tự Kết thúc của selected_text
                if offset_start < end_char <= offset_end:
                    end_index = j
                    # Đảm bảo end_index không nhỏ hơn start_index
                    if end_index < start_index:
                        end_index = start_index 
                    break # Tìm thấy end_index thì dừng
        
        start_labels.append(start_index)
        end_labels.append(end_index)
    
    encodings.pop("offset_mapping")
    
    return encodings, start_labels, end_labels

class TweetDataset(Dataset):
    
    # Cập nhật __init__ để nhận start_labels và end_labels
    def __init__(self, encodings, start_labels, end_labels, texts, selected_texts):
        self.encodings = encodings
        self.start_labels = start_labels
        self.end_labels = end_labels
        self.texts = texts
        self.selected_texts = selected_texts
    
    def __len__(self):
        return len(self.start_labels)
    
    def __getitem__(self, idx):
        item = {key: self.encodings[key][idx].clone().detach() 
                for key in self.encodings}
        # Cập nhật nhãn trả về
        item['start_labels'] = torch.tensor(self.start_labels[idx], dtype=torch.long)
        item['end_labels'] = torch.tensor(self.end_labels[idx], dtype=torch.long)
        item['text'] = self.texts[idx]
        item['selected_text'] = self.selected_texts[idx]
        return item

def prepare_dataset(df: pd.DataFrame, 
                    tokenizer,
                    augment=False) -> TweetDataset:
    
    texts = []
    selected_texts = []
    
    for _, row in df.iterrows():
        text = row['text_proc']
        selected = row['selected_text_proc']
        
        texts.append(text)
        selected_texts.append(selected)
        
        if augment and random.random() < Config.AUG_PROB:
            for _ in range(Config.AUG_COUNT):
                aug_text = augment_text(text, selected, tokenizer)
                if selected in aug_text or not selected:
                    texts.append(aug_text)
                    selected_texts.append(selected)
    
    # Cập nhật hàm gọi gán nhãn
    encodings, start_labels, end_labels = create_token_labels(texts, selected_texts, tokenizer)
    
    return TweetDataset(encodings, start_labels, end_labels, texts, selected_texts)

class TokenClassificationModel(nn.Module):
    
    # Cập nhật model thành Span Prediction (2 nhãn: start_logits và end_logits)
    def __init__(self, model_name: str, dropout: float = Config.DROPOUT):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.base = AutoModel.from_pretrained(model_name, config=self.config)
        
        hidden_size = self.base.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        
        # Output layer dự đoán Start và End (mỗi cái 1 logit)
        self.start_classifier = nn.Linear(hidden_size, 1)
        self.end_classifier = nn.Linear(hidden_size, 1)
        
    def forward(self, input_ids=None, attention_mask=None, start_labels=None, end_labels=None):
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        # Tính logits
        start_logits = self.start_classifier(sequence_output).squeeze(-1)
        end_logits = self.end_classifier(sequence_output).squeeze(-1)
        
        total_loss = None
        if start_labels is not None and end_labels is not None:
            # Loss Function cho Span Prediction (CrossEntropyLoss)
            loss_fct = nn.CrossEntropyLoss()  
            start_loss = loss_fct(start_logits, start_labels)
            end_loss = loss_fct(end_logits, end_labels)
            total_loss = start_loss + end_loss # Tổng Loss
        
        return {"loss": total_loss, "start_logits": start_logits, "end_logits": end_logits}

# ============================================
# EVALUATION METRICS (Giữ nguyên)
# ============================================
def calculate_jaccard(str1: str, str2: str) -> float:
    # [Giữ nguyên code tính Jaccard]
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    
    if not set1 and not set2: return 1.0
    if not set1 or not set2: return 0.0
    
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    return len(intersection) / len(union)

def calculate_extended_metrics(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
    # [Giữ nguyên code tính BLEU/ROUGE]
    # ... (code tính BLEU, ROUGE)
    jaccard_scores = [calculate_jaccard(p, t) for p, t in zip(predictions, ground_truths)]
    
    chencherry = SmoothingFunction()  
    bleu_scores = []
    for pred, true in zip(predictions, ground_truths):
        reference = [nltk.word_tokenize(true.lower())]
        candidate = nltk.word_tokenize(pred.lower())
        
        if not candidate:
            bleu_scores.append(0.0)
            continue
            
        score = sentence_bleu(reference, candidate, smoothing_function=chencherry.method1)  
        bleu_scores.append(score)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_scores = []
    rougeL_scores = []
    
    for pred, true in zip(predictions, ground_truths):
        if not pred or not true:  
            rouge1_scores.append(0.0); rougeL_scores.append(0.0)
            continue
            
        scores = scorer.score(true, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        'Jaccard': np.mean(jaccard_scores),
        'BLEU': np.mean(bleu_scores),
        'ROUGE-L': np.mean(rougeL_scores),
    }

def decode_predictions(start_logits: torch.Tensor, 
                       end_logits: torch.Tensor,
                       text: str,
                       tokenizer) -> str:
    """Chuyển logits thành selected_text bằng cách tìm cặp (start, end) tối ưu"""
    
    encoding = tokenizer(
        text,
        max_length=Config.MAX_LEN,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
        return_offsets_mapping=True
    )
    
    offsets = encoding['offset_mapping'][0].cpu().tolist()  
    
    # Lấy index có score cao nhất
    start_index = torch.argmax(start_logits).item()
    end_index = torch.argmax(end_logits).item()
    
    # Khai báo các ràng buộc:
    # 1. end_index phải lớn hơn hoặc bằng start_index
    # 2. end_index phải nhỏ hơn độ dài của input
    if end_index < start_index:
        # Nếu mô hình dự đoán sai, tìm vị trí end tốt nhất sau start_index
        # Hoặc giữ nguyên start_index
        end_index = start_index 
        
    # Lấy offset ký tự
    start_char, _ = offsets[start_index]
    _, end_char = offsets[end_index]
    
    # Kiểm tra trường hợp đặc biệt: CLS/SEP token
    if start_char == 0 and end_char == 0:
         words = text.split()
         return " ".join(words[:min(3, len(words))])
         
    # Cắt chuỗi
    return text[start_char:end_char].strip()

def calculate_validation_loss(model, dataloader, device):
    """Tính Loss trên validation set"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_labels = batch['start_labels'].to(device)
            end_labels = batch['end_labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_labels=start_labels,
                end_labels=end_labels
            )
            
            loss = outputs['loss']
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, tokenizer, device, calculate_metrics=True):
    """Đánh giá model trên validation set và trả về preds/trues"""
    model.eval()
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            start_logits = outputs['start_logits']
            end_logits = outputs['end_logits']
            
            batch_size = input_ids.shape[0]
            for i in range(batch_size):
                text = batch['text'][i]
                selected_true = batch['selected_text'][i]
                
                # Cập nhật hàm decode_predictions
                pred_text = decode_predictions(start_logits[i], end_logits[i], text, tokenizer)
                
                predictions.append(pred_text)
                ground_truths.append(selected_true)
    
    return predictions, ground_truths

# ============================================
# TRAINING LOGIC (Giữ nguyên Early Stopping)
# ============================================
def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train một epoch"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_labels = batch['start_labels'].to(device)
        end_labels = batch['end_labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_labels=start_labels,
            end_labels=end_labels
        )
        
        loss = outputs['loss']
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)

def train_sentiment_model(sentiment: str,
                          train_df: pd.DataFrame,
                          val_df: pd.DataFrame,
                          tokenizer,
                          output_dir=Config.OUTPUT_DIR):
    """Huấn luyện model cho một sentiment cụ thể và vẽ biểu đồ kết quả"""
    print("\n" + "=" * 60)
    print(f"TRAINING MODEL FOR SENTIMENT: {sentiment.upper()}")
    print("=" * 60)
    
    display_eda_stats(train_df, f"TRAIN ({sentiment.upper()})")
    
    # 1. Prepare datasets
    print("\nPreparing datasets with augmentation...")
    train_dataset = prepare_dataset(train_df, tokenizer, augment=True)
    val_dataset = prepare_dataset(val_df, tokenizer, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # 2. Model & Optimizer
    # Cập nhật model: chỉ cần 2 output cho start/end
    model = TokenClassificationModel(Config.MODEL_NAME, dropout=Config.DROPOUT)
    model.to(Config.DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    total_steps = len(train_loader) * Config.NUM_EPOCHS
    warmup_steps = int(Config.WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    # 3. Training loop
    best_jaccard = -1.0
    patience_counter = 0 
    
    history = {'train_loss': [], 'val_loss': [], 'val_jaccard': [], 'val_bleu': [], 'val_rougeL': []}
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
        print('='*60)
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, Config.DEVICE)
        history['train_loss'].append(train_loss)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate Loss & Metrics
        val_loss = calculate_validation_loss(model, val_loader, Config.DEVICE)
        val_preds, val_trues = evaluate_model(model, val_loader, tokenizer, Config.DEVICE)
        val_metrics = calculate_extended_metrics(val_preds, val_trues)
        val_jaccard = val_metrics['Jaccard']
        
        # Lưu các chỉ số vào history
        history['val_loss'].append(val_loss)
        history['val_jaccard'].append(val_jaccard)
        history['val_bleu'].append(val_metrics['BLEU'])
        history['val_rougeL'].append(val_metrics['ROUGE-L'])

        # Logic Early Stopping và Lưu Model
        if val_jaccard > best_jaccard:
            best_jaccard = val_jaccard
            patience_counter = 0 
            save_path = os.path.join(output_dir, f"best_model_{sentiment}.pt")
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'jaccard': val_jaccard}, save_path)
            print(f"✓ Saved best model to {save_path}")
        else:
            patience_counter += 1 
        
        print(f"Validation Metrics: (Patience: {patience_counter}/{Config.PATIENCE})")
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Jaccard Score: {val_jaccard:.4f} (Best: {best_jaccard:.4f})")
        print(f"  BLEU Score: {val_metrics['BLEU']:.4f}")
        
        if patience_counter >= Config.PATIENCE: 
            print(f"\n!!! Early Stopping triggered after {Config.PATIENCE} epochs without Jaccard improvement.")
            break 
    
    # 4. Vẽ biểu đồ các chỉ số đánh giá theo Epoch
    print("\nGenerating Evaluation Plots...")
    epochs = range(1, len(history['val_jaccard']) + 1)
    
    # Biểu đồ 1: So sánh LOSS (Train vs. Validation)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], marker='o', linestyle='-', label='Training Loss', color='darkred')
    plt.plot(epochs, history['val_loss'], marker='x', linestyle='--', label='Validation Loss', color='darkblue')
    plt.title(f'Loss vs. Epochs ({sentiment.upper()}) - Train vs. Validation (Span)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir, f'loss_comparison_{sentiment}_span.png'))
    
    # Biểu đồ 2: So sánh JACCARD & các Metric khác (Validation Only)
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, history['val_jaccard'], marker='o', linestyle='-', label='Jaccard Score', color='blue')
    plt.plot(epochs, history['val_bleu'], marker='x', linestyle='--', label='BLEU Score', color='green')
    plt.plot(epochs, history['val_rougeL'], marker='s', linestyle='-.', label='ROUGE-L F1', color='red')
    plt.title(f'Validation Scores vs. Epochs ({sentiment.upper()}) (Span)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir, f'val_scores_{sentiment}_span.png'))

    print(f"\n✓ Training completed for {sentiment}")
    print(f"Best Validation Jaccard: {best_jaccard:.4f}")
    
    return os.path.join(output_dir, f"best_model_{sentiment}.pt"), best_jaccard

# ============================================
# INFERENCE LOGIC (SỬ DỤNG MODEL MỚI)
# ============================================
def load_model(model_path: str, device=Config.DEVICE):
    """Load trained model từ checkpoint"""
    # Cập nhật: Load mô hình dự đoán Span
    model = TokenClassificationModel(Config.MODEL_NAME, dropout=Config.DROPOUT)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)  
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def predict_single(model, tokenizer, text: str, device=Config.DEVICE) -> str:
    """Predict selected_text cho một tweet"""
    model.eval()
    
    text_proc = normalize_text(text)
    
    encoding = tokenizer(
        text_proc,
        max_length=Config.MAX_LEN,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
        return_offsets_mapping=True
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        start_logits = outputs['start_logits'][0]
        end_logits = outputs['end_logits'][0]
    
    # Decode sử dụng logic dự đoán span
    return decode_predictions(start_logits, end_logits, text_proc, tokenizer)


def generate_submission(test_df: pd.DataFrame,
                        model_pos,
                        model_neg,
                        tokenizer,
                        submission_df: pd.DataFrame) -> pd.DataFrame:
    """Tạo submission file bằng 2 mô hình sentiment-specific"""
    print("\n" + "=" * 60)
    print("GENERATING PREDICTIONS FOR TEST SET")
    print("=" * 60)
    
    predictions = []
    
    test_df_indexed = test_df.reset_index(drop=True)
    
    for idx, row in tqdm(test_df_indexed.iterrows(), total=len(test_df_indexed)):
        text = row['text']
        sentiment = row.get('sentiment', 'neutral')
        
        if sentiment == 'positive':
            pred = predict_single(model_pos, tokenizer, text, Config.DEVICE)
        elif sentiment == 'negative':
            pred = predict_single(model_neg, tokenizer, text, Config.DEVICE)
        else: # Neutral sentiment (Giữ nguyên)
            pred = text.strip()
        
        predictions.append(pred)
    
    submission_df['selected_text'] = predictions
    return submission_df

# ============================================
# MAIN PIPELINE (Giữ nguyên)
# ============================================
def main():
    """Main training pipeline"""
    
    # 1. Load data
    train_df, test_df, submission_df = load_data()
    
    # 2. Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    # 3. Split data by sentiment & Train/Val split
    print("\n" + "=" * 60)
    print("PREPARING SENTIMENT-SPECIFIC DATASETS (Positive & Negative)")
    print("=" * 60)
    
    df_positive = train_df[train_df['sentiment'] == 'positive'].reset_index(drop=True)
    df_negative = train_df[train_df['sentiment'] == 'negative'].reset_index(drop=True)
    
    train_pos, val_pos = train_test_split(df_positive, test_size=Config.VAL_SIZE, random_state=Config.SEED, shuffle=True)
    train_neg, val_neg = train_test_split(df_negative, test_size=Config.VAL_SIZE, random_state=Config.SEED, shuffle=True)
    
    print(f"\nPositive - Train: {len(train_pos)}, Val: {len(val_pos)}")
    print(f"Negative - Train: {len(train_neg)}, Val: {len(val_neg)}")
    
    # 4. Train models
    model_pos_path, jaccard_pos = train_sentiment_model('positive', train_pos, val_pos, tokenizer)
    model_neg_path, jaccard_neg = train_sentiment_model('negative', train_neg, val_neg, tokenizer)
    
    # 5. Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Positive Model - Best Jaccard: {jaccard_pos:.4f}")
    print(f"Negative Model - Best Jaccard: {jaccard_neg:.4f}")
    print(f"Average Jaccard (Positive/Negative): {(jaccard_pos + jaccard_neg) / 2:.4f}")
    
    # 6. Load best models for inference
    model_pos = load_model(model_pos_path)
    model_neg = load_model(model_neg_path)
    
    # 7. Generate submission
    test_df_reset = test_df.reset_index(drop=True)
    submission_df_reset = submission_df.copy() 
    submission = generate_submission(test_df_reset, model_pos, model_neg, tokenizer, submission_df_reset)
    
    # 8. Save submission
    submission_filename = 'submission_span_optimized_final.csv'
    submission.to_csv(submission_filename, index=False)
    print(f"\n✓ Saved final predictions to {submission_filename}")
    
    # 9. Show sample predictions
    print("\n" + "=" * 60)
    print("SAMPLE TEST PREDICTIONS")
    print("=" * 60)
    
    samples_p = test_df_reset[test_df_reset['sentiment'] == 'positive'].head(5)
    samples_n = test_df_reset[test_df_reset['sentiment'] == 'negative'].head(5)
    samples_o = test_df_reset[test_df_reset['sentiment'] == 'neutral'].head(5)
    
    samples = pd.concat([samples_p, samples_n, samples_o])
    
    for idx in samples.index:
        row = samples.loc[idx]
        predicted_text = submission.loc[idx, 'selected_text']
        
        print(f"\nTweet ID: {row['textID']}")
        print(f"Sentiment: {row.get('sentiment', 'N/A').upper()}")
        print(f"Text: {row['text']}")
        print(f"Predicted: **{predicted_text}**")
        print("-" * 60)
    
    print("\n✓ PIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()


# import os
# import random
# import re
# import warnings
# from typing import List, Tuple, Dict
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split
# from collections import Counter

# import nltk
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from rouge_score import rouge_scorer
# import matplotlib.pyplot as plt

# import torch
# from torch import nn
# from torch.utils.data import Dataset, DataLoader
# from transformers import (
#     AutoTokenizer,
#     AutoModel,
#     AutoConfig,
#     get_linear_schedule_with_warmup,
# )

# # Tải tokenizer cho NLTK (cần thiết cho BLEU/ROUGE)
# try:
#     nltk.data.find('tokenizers/punkt')
# except nltk.downloader.DownloadError:
#     print("Downloading NLTK punkt tokenizer...")
#     nltk.download('punkt')

# warnings.filterwarnings('ignore')

# # ============================================
# # CONFIGURATION
# # ============================================
# class Config:
#     TRAIN_PATH = '/kaggle/input/tweet-sentiment-extraction/train.csv'
#     TEST_PATH = '/kaggle/input/tweet-sentiment-extraction/test.csv'
#     SUBMISSION_PATH = '/kaggle/input/tweet-sentiment-extraction/sample_submission.csv'
#     OUTPUT_DIR = './best_models'
    
#     # Model
#     MODEL_NAME = "roberta-base"
#     MAX_LEN = 128
#     DROPOUT = 0.1
    
#     # Training
#     SEED = 42
#     BATCH_SIZE = 32
#     NUM_EPOCHS = 50
#     LR = 1e-4
#     WEIGHT_DECAY = 0.01
#     WARMUP_RATIO = 0.1
#     VAL_SIZE = 0.12
    
#     # Data Augmentation
#     AUG_PROB = 0.4
#     AUG_COUNT = 1
#     AUG_DELETE_PROB = 0.06
#     AUG_SWAP_PROB = 0.04
#     AUG_MASK_PROB = 0.06
    
#     # Device
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ============================================
# # SETUP
# # ============================================
# def set_seed(seed=Config.SEED):
#     """Set seed cho reproducibility"""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# set_seed()
# os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# # ============================================
# # DATA PREPROCESSING & EDA
# # ============================================
# def normalize_text(text: str) -> str:
#     """Chuẩn hóa text (giữ nguyên vị trí tương đối)"""
#     if not isinstance(text, str):
#         return ""
    
#     s = text.lower()
#     s = s.replace('\r\n', ' ').replace('\n', ' ')
#     s = re.sub(r'\s+', ' ', s).strip()
#     s = re.sub(r'(.)\1{2,}', r'\1\1', s) 
#     s = re.sub(r'@\w+', '@user', s)
#     s = re.sub(r'http\S+|www\.\S+', 'url', s)
    
#     return s

# def display_eda_stats(df: pd.DataFrame, title: str):
#     """Hiển thị thống kê cơ bản và phân tích dữ liệu mở rộng"""
#     if df.empty:
#         print(f"\n--- Thống kê Dữ liệu {title} ---")
#         print("Tập dữ liệu trống.")
#         print("-" * 40)
#         return

#     print(f"\n--- Thống kê Dữ liệu {title} ({len(df)} samples) ---")
    
#     if 'sentiment' in df.columns:
#         print(f"Phân phối Sentiment:\n{df['sentiment'].value_counts().to_string()}")
    
#     df['text_len'] = df['text_proc'].apply(lambda x: len(x.split()))
    
#     print(f"\nChiều dài Tweet (số từ):")
#     print(df['text_len'].describe().to_string())
    
#     if 'selected_text_proc' in df.columns:
#         df['selected_len'] = df['selected_text_proc'].apply(lambda x: len(x.split()))
#         print(f"\nChiều dài Selected Text (số từ):")
#         print(df['selected_len'].describe().to_string())
        
#         df['ratio'] = df['selected_len'] / df['text_len']
#         print(f"\nTỷ lệ Selected Text/Tweet:")
#         print(df['ratio'].describe().to_string())
        
#         words = ' '.join(df['selected_text_proc']).split()
#         most_common = Counter(words).most_common(5)
#         print(f"\n5 từ phổ biến nhất trong Selected Text: {most_common}")
        
#     print("-" * 40)


# def load_data():
#     """Load và preprocess data, và loại bỏ sentiment 'neutral' khỏi tập train"""
#     print("=" * 60)
#     print("LOADING DATA")
#     print("=" * 60)
    
#     train = pd.read_csv(Config.TRAIN_PATH)
#     test = pd.read_csv(Config.TEST_PATH)
#     submission = pd.read_csv(Config.SUBMISSION_PATH)
    
#     train = train.dropna().reset_index(drop=True)
    
#     # Loại bỏ dữ liệu 'neutral' khỏi tập huấn luyện
#     print("Loại bỏ dữ liệu có sentiment 'neutral' khỏi tập huấn luyện.")
#     train = train[train['sentiment'].isin(['positive', 'negative'])].reset_index(drop=True)
    
#     # Normalize text
#     train['text_proc'] = train['text'].astype(str).apply(normalize_text)
#     train['selected_text_proc'] = train['selected_text'].astype(str).apply(normalize_text)
#     test['text_proc'] = test['text'].astype(str).apply(normalize_text)
    
#     print(f"Train shape (sau khi loại bỏ 'neutral'): {train.shape}")
#     print(f"Test shape: {test.shape}")
    
#     display_eda_stats(train, "TRAIN (Positive/Negative Only)")
    
#     return train, test, submission

# # ============================================
# # DATA AUGMENTATION LOGIC
# # ============================================

# def find_span_in_text(text: str, selected: str) -> Tuple[int, int]:
#     """Tìm vị trí span (start, end) của selected_text trong text"""
#     if not selected: return -1, -1
#     start = text.find(selected)
#     if start == -1: return -1, -1
#     end = start + len(selected)
#     return start, end

# def get_word_boundaries(text: str) -> List[Tuple[int, int]]:
#     """Lấy vị trí ký tự (start, end) của mỗi từ trong text"""
#     words = []
#     for match in re.finditer(r'\S+', text):
#         words.append((match.start(), match.end()))
#     return words

# def augment_text(text: str, selected: str, tokenizer,
#                  prob_delete=Config.AUG_DELETE_PROB,
#                  prob_swap=Config.AUG_SWAP_PROB,
#                  prob_mask=Config.AUG_MASK_PROB) -> str:
#     """Thực hiện Data Augmentation chỉ áp dụng cho các từ nằm ngoài Selected Text."""
#     start_char, end_char = find_span_in_text(text, selected)
#     word_positions = get_word_boundaries(text)
#     words = [text[s:e] for s, e in word_positions]
    
#     outside_indices = []
    
#     for i, (s, e) in enumerate(word_positions):
#         is_inside = (start_char != -1 and s >= start_char and e <= end_char)
#         if not is_inside:
#             outside_indices.append(i)
            
#     augmented_words = words.copy()
    
#     # 1. Random deletion 
#     for i in outside_indices:
#         if random.random() < prob_delete:
#             augmented_words[i] = ""
    
#     # 2. Random swap 
#     if len(outside_indices) >= 2 and random.random() < prob_swap:
#         i, j = random.sample(outside_indices, 2) 
#         augmented_words[i], augmented_words[j] = augmented_words[j], augmented_words[i]
    
#     # 3. Random masking
#     for i in outside_indices:
#         if random.random() < prob_mask and augmented_words[i]:
#             augmented_words[i] = tokenizer.mask_token
    
#     augmented_words = [w for w in augmented_words if w]
#     augmented_text = " ".join(augmented_words)
    
#     if selected and selected not in augmented_text:
#         return text 
    
#     return augmented_text

# # ============================================
# # TOKENIZATION & MODEL & DATASET
# # ============================================
# def create_token_labels(texts: List[str], 
#                        selected_texts: List[str],
#                        tokenizer,
#                        max_len=Config.MAX_LEN) -> Tuple[Dict, List[List[int]]]:
#     """Tạo encodings và gán nhãn token (0: outside, 1: inside, -100: special/padding)"""
#     encodings = tokenizer(
#         texts,
#         max_length=max_len,
#         padding='max_length',
#         truncation=True,
#         return_offsets_mapping=True,
#         return_tensors='pt'
#     )
    
#     labels = []
#     for i in range(len(texts)):
#         # Lấy offset_mapping cho mẫu thứ i, chuyển về CPU và list
#         offsets = encodings['offset_mapping'][i].cpu().tolist() 
#         text = texts[i]
#         selected = selected_texts[i]
        
#         start_char, end_char = find_span_in_text(text, selected)
        
#         token_labels = []
#         for s, e in offsets:
#             if s == e:
#                 token_labels.append(-100)
#             else:
#                 if start_char != -1 and not (e <= start_char or s >= end_char):
#                     token_labels.append(1) # Token thuộc Selected Text
#                 else:
#                     token_labels.append(0) # Token KHÔNG thuộc Selected Text
        
#         labels.append(token_labels)
    
#     encodings.pop("offset_mapping")
    
#     return encodings, labels

# class TweetDataset(Dataset):
    
#     def __init__(self, encodings, labels, texts, selected_texts):
#         self.encodings = encodings
#         self.labels = labels
#         self.texts = texts
#         self.selected_texts = selected_texts
    
#     def __len__(self):
#         return len(self.labels)
    
#     def __getitem__(self, idx):
#         item = {key: self.encodings[key][idx].clone().detach() 
#                 for key in self.encodings}
#         item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
#         item['text'] = self.texts[idx]
#         item['selected_text'] = self.selected_texts[idx]
#         return item

# def prepare_dataset(df: pd.DataFrame, 
#                    tokenizer,
#                    augment=False) -> TweetDataset:
    
#     texts = []
#     selected_texts = []
    
#     for _, row in df.iterrows():
#         text = row['text_proc']
#         selected = row['selected_text_proc']
        
#         texts.append(text)
#         selected_texts.append(selected)
        
#         if augment and random.random() < Config.AUG_PROB:
#             for _ in range(Config.AUG_COUNT):
#                 aug_text = augment_text(text, selected, tokenizer)
#                 if selected in aug_text or not selected:
#                     texts.append(aug_text)
#                     selected_texts.append(selected)
    
#     encodings, labels = create_token_labels(texts, selected_texts, tokenizer)
    
#     return TweetDataset(encodings, labels, texts, selected_texts)

# class TokenClassificationModel(nn.Module):
    
#     def __init__(self, model_name: str, num_labels: int = 2, dropout: float = Config.DROPOUT):
#         super().__init__()
#         self.num_labels = num_labels
#         self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
#         self.base = AutoModel.from_pretrained(model_name, config=self.config)
        
#         hidden_size = self.base.config.hidden_size
#         self.dropout = nn.Dropout(dropout)
#         self.classifier = nn.Linear(hidden_size, num_labels)
    
#     def forward(self, input_ids=None, attention_mask=None, labels=None):
#         outputs = self.base(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             return_dict=True
#         )
        
#         sequence_output = outputs.last_hidden_state
#         sequence_output = self.dropout(sequence_output)
#         logits = self.classifier(sequence_output)
        
#         loss = None
#         if labels is not None:
#             loss_fct = nn.CrossEntropyLoss(ignore_index=-100) 
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
#         return {"loss": loss, "logits": logits}

# # ============================================
# # EVALUATION METRICS
# # ============================================
# def calculate_jaccard(str1: str, str2: str) -> float:
#     """Tính Jaccard similarity"""
#     set1 = set(str1.lower().split())
#     set2 = set(str2.lower().split())
    
#     if not set1 and not set2: return 1.0
#     if not set1 or not set2: return 0.0
    
#     intersection = set1.intersection(set2)
#     union = set1.union(set2)
    
#     return len(intersection) / len(union)

# def calculate_extended_metrics(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
#     """Tính Jaccard, BLEU và ROUGE scores"""
    
#     # 1. Jaccard 
#     jaccard_scores = [calculate_jaccard(p, t) for p, t in zip(predictions, ground_truths)]
    
#     # 2. BLEU
#     chencherry = SmoothingFunction() 
#     bleu_scores = []
#     for pred, true in zip(predictions, ground_truths):
#         reference = [nltk.word_tokenize(true.lower())]
#         candidate = nltk.word_tokenize(pred.lower())
        
#         if not candidate:
#             bleu_scores.append(0.0)
#             continue
            
#         score = sentence_bleu(reference, candidate, smoothing_function=chencherry.method1) 
#         bleu_scores.append(score)

#     # 3. ROUGE
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
#     rouge1_scores = []
#     rouge2_scores = []
#     rougeL_scores = []
    
#     for pred, true in zip(predictions, ground_truths):
#         if not pred or not true: 
#             rouge1_scores.append(0.0); rouge2_scores.append(0.0); rougeL_scores.append(0.0)
#             continue
            
#         scores = scorer.score(true, pred)
#         rouge1_scores.append(scores['rouge1'].fmeasure)
#         rouge2_scores.append(scores['rouge2'].fmeasure)
#         rougeL_scores.append(scores['rougeL'].fmeasure)
    
#     return {
#         'Jaccard': np.mean(jaccard_scores),
#         'BLEU': np.mean(bleu_scores),
#         'ROUGE-1': np.mean(rouge1_scores),
#         'ROUGE-2': np.mean(rouge2_scores),
#         'ROUGE-L': np.mean(rougeL_scores),
#     }

# def decode_predictions(logits: torch.Tensor, 
#                       text: str,
#                       tokenizer) -> str:
#     """Convert logits thành selected_text"""
#     preds = torch.argmax(logits, dim=-1).cpu().numpy()
    
#     encoding = tokenizer(
#         text,
#         max_length=Config.MAX_LEN,
#         truncation=True,
#         padding='max_length',
#         return_tensors='pt',
#         return_offsets_mapping=True
#     )
    
#     offsets = encoding['offset_mapping'][0].cpu().tolist() 
    
#     selected_spans = []
#     for i, (pred, (start, end)) in enumerate(zip(preds, offsets)):
#         if start == end:
#             continue
#         if pred == 1: 
#             selected_spans.append((start, end))
    
#     if not selected_spans:
#         words = text.split()
#         return " ".join(words[:min(3, len(words))])
    
#     min_start = min(s for s, e in selected_spans)
#     max_end = max(e for s, e in selected_spans)
    
#     return text[min_start:max_end].strip()

# def calculate_validation_loss(model, dataloader, device):
#     """Tính Loss trên validation set"""
#     model.eval()
#     total_loss = 0
#     with torch.no_grad():
#         for batch in dataloader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)
            
#             outputs = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 labels=labels
#             )
            
#             loss = outputs['loss']
#             total_loss += loss.item()
    
#     return total_loss / len(dataloader)


# def evaluate_model(model, dataloader, tokenizer, device, calculate_metrics=True):
#     """Đánh giá model trên validation set và trả về preds/trues"""
#     model.eval()
#     predictions = []
#     ground_truths = []
    
#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="Evaluating"):
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
            
#             outputs = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask
#             )
            
#             logits = outputs['logits']
            
#             batch_size = input_ids.shape[0]
#             for i in range(batch_size):
#                 text = batch['text'][i]
#                 selected_true = batch['selected_text'][i]
                
#                 pred_text = decode_predictions(logits[i], text, tokenizer)
                
#                 predictions.append(pred_text)
#                 ground_truths.append(selected_true)
    
#     return predictions, ground_truths

# # ============================================
# # TRAINING LOGIC
# # ============================================
# def train_epoch(model, dataloader, optimizer, scheduler, device):
#     """Train một epoch"""
#     model.train()
#     total_loss = 0
#     progress_bar = tqdm(dataloader, desc="Training")
    
#     for batch in progress_bar:
#         optimizer.zero_grad()
        
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
        
#         outputs = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             labels=labels
#         )
        
#         loss = outputs['loss']
#         loss.backward()
        
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
#         optimizer.step()
#         scheduler.step()
        
#         total_loss += loss.item()
#         progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
#     return total_loss / len(dataloader)

# def train_sentiment_model(sentiment: str,
#                          train_df: pd.DataFrame,
#                          val_df: pd.DataFrame,
#                          tokenizer,
#                          output_dir=Config.OUTPUT_DIR):
#     """Huấn luyện model cho một sentiment cụ thể và vẽ biểu đồ kết quả"""
#     print("\n" + "=" * 60)
#     print(f"TRAINING MODEL FOR SENTIMENT: {sentiment.upper()}")
#     print("=" * 60)
    
#     display_eda_stats(train_df, f"TRAIN ({sentiment.upper()})")
    
#     # 1. Prepare datasets
#     print("\nPreparing datasets with augmentation...")
#     train_dataset = prepare_dataset(train_df, tokenizer, augment=True)
#     val_dataset = prepare_dataset(val_df, tokenizer, augment=False)
    
#     train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
#     val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
#     # 2. Model & Optimizer
#     model = TokenClassificationModel(Config.MODEL_NAME, num_labels=2)
#     model.to(Config.DEVICE)
    
#     optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
#     total_steps = len(train_loader) * Config.NUM_EPOCHS
#     warmup_steps = int(Config.WARMUP_RATIO * total_steps)
#     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
#     # 3. Training loop
#     best_jaccard = -1.0
#     history = {'train_loss': [], 'val_loss': [], 'val_jaccard': [], 'val_bleu': [], 'val_rouge1': [], 'val_rouge2': [], 'val_rougeL': []}
    
#     for epoch in range(Config.NUM_EPOCHS):
#         print(f"\n{'='*60}")
#         print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
#         print('='*60)
        
#         train_loss = train_epoch(model, train_loader, optimizer, scheduler, Config.DEVICE)
#         history['train_loss'].append(train_loss)
#         print(f"Train Loss: {train_loss:.4f}")
        
#         # Validate Loss & Metrics
#         val_loss = calculate_validation_loss(model, val_loader, Config.DEVICE)
#         val_preds, val_trues = evaluate_model(model, val_loader, tokenizer, Config.DEVICE)
#         val_metrics = calculate_extended_metrics(val_preds, val_trues)
#         val_jaccard = val_metrics['Jaccard']
        
#         # Lưu các chỉ số vào history
#         history['val_loss'].append(val_loss)
#         history['val_jaccard'].append(val_jaccard)
#         history['val_bleu'].append(val_metrics['BLEU'])
#         history['val_rougeL'].append(val_metrics['ROUGE-L'])

#         print(f"Validation Metrics:")
#         print(f"  Validation Loss: {val_loss:.4f}")
#         print(f"  Jaccard Score: {val_jaccard:.4f} (Best: {best_jaccard:.4f})")
#         print(f"  BLEU Score: {val_metrics['BLEU']:.4f}")
        
#         # Save best model
#         if val_jaccard > best_jaccard:
#             best_jaccard = val_jaccard
#             save_path = os.path.join(output_dir, f"best_model_{sentiment}.pt")
#             torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'jaccard': val_jaccard}, save_path)
#             print(f"✓ Saved best model to {save_path}")
    
#     # 4. Vẽ biểu đồ các chỉ số đánh giá theo Epoch
#     print("\nGenerating Evaluation Plots...")
#     epochs = range(1, len(history['val_jaccard']) + 1)
    
#     # Biểu đồ 1: So sánh LOSS (Train vs. Validation)
#     plt.figure(figsize=(10, 5))
#     plt.plot(epochs, history['train_loss'], marker='o', linestyle='-', label='Training Loss', color='darkred')
#     plt.plot(epochs, history['val_loss'], marker='x', linestyle='--', label='Validation Loss', color='darkblue')
#     plt.title(f'Loss vs. Epochs ({sentiment.upper()}) - Train vs. Validation', fontsize=14)
#     plt.xlabel('Epoch', fontsize=12)
#     plt.ylabel('Loss', fontsize=12)
#     plt.legend(fontsize=10)
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.savefig(os.path.join(output_dir, f'loss_comparison_{sentiment}.png'))
#     plt.show() 
    
#     # Biểu đồ 2: So sánh JACCARD & các Metric khác (Validation Only)
#     plt.figure(figsize=(12, 6))
#     plt.plot(epochs, history['val_jaccard'], marker='o', linestyle='-', label='Jaccard Score', color='blue')
#     plt.plot(epochs, history['val_bleu'], marker='x', linestyle='--', label='BLEU Score', color='green')
#     plt.plot(epochs, history['val_rougeL'], marker='s', linestyle='-.', label='ROUGE-L F1', color='red')
#     plt.title(f'Validation Scores vs. Epochs ({sentiment.upper()})', fontsize=14)
#     plt.xlabel('Epoch', fontsize=12)
#     plt.ylabel('Score', fontsize=12)
#     plt.legend(fontsize=10)
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.savefig(os.path.join(output_dir, f'val_scores_{sentiment}.png'))
#     plt.show() 

#     print(f"\n✓ Training completed for {sentiment}")
#     print(f"Best Validation Jaccard: {best_jaccard:.4f}")
    
#     return os.path.join(output_dir, f"best_model_{sentiment}.pt"), best_jaccard

# # ============================================
# # INFERENCE LOGIC
# # ============================================
# def load_model(model_path: str, device=Config.DEVICE):
#     """Load trained model từ checkpoint"""
#     model = TokenClassificationModel(Config.MODEL_NAME, num_labels=2)
#     checkpoint = torch.load(model_path, map_location=device, weights_only=False) 
    
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.to(device)
#     model.eval()
#     return model

# def predict_single(model, tokenizer, text: str, device=Config.DEVICE) -> str:
#     """Predict selected_text cho một tweet"""
#     model.eval()
    
#     text_proc = normalize_text(text)
    
#     encoding = tokenizer(
#         text_proc,
#         max_length=Config.MAX_LEN,
#         truncation=True,
#         padding='max_length',
#         return_tensors='pt',
#         return_offsets_mapping=True
#     )
    
#     input_ids = encoding['input_ids'].to(device)
#     attention_mask = encoding['attention_mask'].to(device)
    
#     offsets = encoding['offset_mapping'][0].cpu().tolist() 
    
#     # Predict
#     with torch.no_grad():
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         logits = outputs['logits'][0]
    
#     preds = torch.argmax(logits, dim=-1).cpu().numpy()
    
#     # Decode
#     selected_spans = []
#     for pred, (start, end) in zip(preds, offsets):
#         if start == end:
#             continue
#         if pred == 1:
#             selected_spans.append((start, end))
    
#     if not selected_spans:
#         words = text_proc.split()
#         return " ".join(words[:min(3, len(words))])
    
#     min_start = min(s for s, e in selected_spans)
#     max_end = max(e for s, e in selected_spans)
    
#     return text_proc[min_start:max_end].strip()

# def generate_submission(test_df: pd.DataFrame,
#                        model_pos,
#                        model_neg,
#                        tokenizer,
#                        submission_df: pd.DataFrame) -> pd.DataFrame:
#     """Tạo submission file bằng 2 mô hình sentiment-specific"""
#     print("\n" + "=" * 60)
#     print("GENERATING PREDICTIONS FOR TEST SET")
#     print("=" * 60)
    
#     predictions = []
    
#     for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
#         text = row['text']
#         sentiment = row.get('sentiment', 'neutral')
        
#         if sentiment == 'positive':
#             pred = predict_single(model_pos, tokenizer, text, Config.DEVICE)
#         elif sentiment == 'negative':
#             pred = predict_single(model_neg, tokenizer, text, Config.DEVICE)
#         else: # Neutral sentiment (Chỉ áp dụng cho tập TEST)
#             pred = text.strip()
        
#         predictions.append(pred)
    
#     submission_df['selected_text'] = predictions
#     return submission_df

# # ============================================
# # MAIN PIPELINE
# # ============================================
# def main():
#     """Main training pipeline"""
    
#     # 1. Load data (Đã loại bỏ neutral khỏi tập train)
#     train_df, test_df, submission_df = load_data()
    
#     # 2. Initialize tokenizer
#     print("\nInitializing tokenizer...")
#     tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
#     # 3. Split data by sentiment & Train/Val split
#     print("\n" + "=" * 60)
#     print("PREPARING SENTIMENT-SPECIFIC DATASETS (Positive & Negative)")
#     print("=" * 60)
    
#     # train_df giờ chỉ còn positive và negative
#     df_positive = train_df[train_df['sentiment'] == 'positive'].reset_index(drop=True)
#     df_negative = train_df[train_df['sentiment'] == 'negative'].reset_index(drop=True)
    
#     train_pos, val_pos = train_test_split(df_positive, test_size=Config.VAL_SIZE, random_state=Config.SEED, shuffle=True)
#     train_neg, val_neg = train_test_split(df_negative, test_size=Config.VAL_SIZE, random_state=Config.SEED, shuffle=True)
    
#     print(f"\nPositive - Train: {len(train_pos)}, Val: {len(val_pos)}")
#     print(f"Negative - Train: {len(train_neg)}, Val: {len(val_neg)}")
    
#     # 4. Train models
#     model_pos_path, jaccard_pos = train_sentiment_model('positive', train_pos, val_pos, tokenizer)
#     model_neg_path, jaccard_neg = train_sentiment_model('negative', train_neg, val_neg, tokenizer)
    
#     # 5. Summary
#     print("\n" + "=" * 60)
#     print("TRAINING SUMMARY")
#     print("=" * 60)
#     print(f"Positive Model - Best Jaccard: {jaccard_pos:.4f}")
#     print(f"Negative Model - Best Jaccard: {jaccard_neg:.4f}")
#     print(f"Average Jaccard (Positive/Negative): {(jaccard_pos + jaccard_neg) / 2:.4f}")
    
#     # 6. Load best models for inference
#     model_pos = load_model(model_pos_path)
#     model_neg = load_model(model_neg_path)
    
#     # 7. Generate submission
#     submission = generate_submission(test_df, model_pos, model_neg, tokenizer, submission_df)
    
#     # 8. Save submission
#     submission_filename = 'submission_optimized_final.csv'
#     submission.to_csv(submission_filename, index=False)
#     print(f"\n✓ Saved final predictions to {submission_filename}")
    
#     # 9. Show sample predictions
#     print("\n" + "=" * 60)
#     print("SAMPLE TEST PREDICTIONS")
#     print("=" * 60)
    
#     samples_p = test_df[test_df['sentiment'] == 'positive'].head(5)
#     samples_n = test_df[test_df['sentiment'] == 'negative'].head(5)
#     samples_o = test_df[test_df['sentiment'] == 'neutral'].head(5)
    
#     samples = pd.concat([samples_p, samples_n, samples_o])
    
#     for idx in samples.index:
#         row = samples.loc[idx]
#         print(f"\nTweet ID: {row['textID']}")
#         print(f"Sentiment: {row.get('sentiment', 'N/A').upper()}")
#         print(f"Text: {row['text']}")
#         print(f"Predicted: **{submission.loc[idx, 'selected_text']}**")
#         print("-" * 60)
    
#     print("\n✓ PIPELINE COMPLETED SUCCESSFULLY!")

# if __name__ == "__main__":
#     main()




