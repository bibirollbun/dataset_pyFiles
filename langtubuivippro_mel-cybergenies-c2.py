# !pip install transformers bitsandbytes accelerate catboost


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


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("HUB_KEY")



from wordcloud import WordCloud
import matplotlib.pyplot as plt
def perform_eda(X, y):
    """Comprehensive EDA for jailbreak detection"""
    
    df = pd.DataFrame({'text': X, 'label': y})
    
    # Basic stats
    print("\n" + "="*80)
    print("ğŸ“ˆ CLASS DISTRIBUTION")
    print("="*80)
    print(df['label'].value_counts())
    print(f"\nBenign:    {(y==0).sum():>5} ({(y==0).mean()*100:.2f}%)")
    print(f"Jailbreak: {y.sum():>5} ({y.mean()*100:.2f}%)")
    print(f"Imbalance Ratio: {(y==0).sum() / y.sum():.2f}:1")
    
    # Text length analysis
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    
    print("\n" + "="*80)
    print("ğŸ“� TEXT LENGTH STATISTICS")
    print("="*80)
    print("\nCharacter Count:")
    print(df.groupby('label')['text_length'].describe())
    print("\nWord Count:")
    print(df.groupby('label')['word_count'].describe())
    
    # Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Text length distribution
    axes[0, 0].hist([df[df['label']==0]['text_length'], 
                     df[df['label']==1]['text_length']], 
                    bins=50, label=['Benign', 'Jailbreak'], alpha=0.7)
    axes[0, 0].set_xlabel('Text Length')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Text Length Distribution')
    axes[0, 0].legend()
    
    # Word count distribution
    axes[0, 1].hist([df[df['label']==0]['word_count'], 
                     df[df['label']==1]['word_count']], 
                    bins=50, label=['Benign', 'Jailbreak'], alpha=0.7)
    axes[0, 1].set_xlabel('Word Count')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Word Count Distribution')
    axes[0, 1].legend()
    
    # Class balance
    axes[1, 0].bar(['Benign', 'Jailbreak'], df['label'].value_counts().values)
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Class Balance')
    
    # Box plot
    df.boxplot(column='text_length', by='label', ax=axes[1, 1])
    axes[1, 1].set_title('Text Length by Class')
    axes[1, 1].set_xlabel('Class')
    axes[1, 1].set_ylabel('Text Length')
    
    plt.tight_layout()
    plt.savefig('eda_distributions.png', dpi=150, bbox_inches='tight')
    print("\nâœ… Saved: eda_distributions.png")
    plt.close()
    
    # Word clouds
    print("\n" + "="*80)
    print("â˜�ï¸�  GENERATING WORD CLOUDS")
    print("="*80)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Benign word cloud
    benign_text = ' '.join(df[df['label']==0]['text'].values)
    wordcloud_benign = WordCloud(width=800, height=400, 
                                  background_color='white',
                                  colormap='Blues').generate(benign_text)
    axes[0].imshow(wordcloud_benign, interpolation='bilinear')
    axes[0].axis('off')
    axes[0].set_title('Benign Prompts - Common Words', fontsize=16)
    
    # Jailbreak word cloud
    jailbreak_text = ' '.join(df[df['label']==1]['text'].values)
    wordcloud_jailbreak = WordCloud(width=800, height=400,
                                     background_color='white',
                                     colormap='Reds').generate(jailbreak_text)
    axes[1].imshow(wordcloud_jailbreak, interpolation='bilinear')
    axes[1].axis('off')
    axes[1].set_title('Jailbreak Prompts - Common Words', fontsize=16)
    
    plt.tight_layout()
    plt.show()
    # plt.savefig('wordclouds.png', dpi=150, bbox_inches='tight')
    # print("âœ… Saved: wordclouds.png")
    # plt.close()
    
    # Keyword analysis
    print("\n" + "="*80)
    print("ğŸ”‘ KEYWORD ANALYSIS")
    print("="*80)
    
    jailbreak_keywords = [
        'ignore', 'pretend', 'roleplay', 'act as', 'you are',
        'developer mode', 'jailbreak', 'bypass', 'override',
        'instruction', 'system', 'prompt', 'dan', 'evil'
    ]
    
    for keyword in jailbreak_keywords:
        benign_count = df[df['label']==0]['text'].str.lower().str.contains(keyword).sum()
        jailbreak_count = df[df['label']==1]['text'].str.lower().str.contains(keyword).sum()
        print(f"'{keyword}': Benign={benign_count}, Jailbreak={jailbreak_count}, " 
              f"Ratio={jailbreak_count/(benign_count+1):.2f}x")
    
    return df


import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier, Pool
from tqdm import tqdm
import gc

# ğŸ§  Load training and test data
train = pd.read_csv('/kaggle/input/rmit-hackathon-2025/train.csv')
test = pd.read_csv('/kaggle/input/rmit-hackathon-2025/test.csv')

# ğŸ�¯ Target variable (1 = jailbreak, 0 = benign)
y = (train['label'] == 'jailbreak').astype(int)
X = train['text'].astype(str)
X_test = test['text'].astype(str)


y.hist()


df_analysis = perform_eda(X, y)


# !pip install -q --upgrade transformers==4.45.0 huggingface_hub

import os
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

from transformers import AutoTokenizer, AutoModel

# tokenizer = AutoTokenizer.from_pretrained(
#     'jackhhao/jailbreak-classifier',
#     use_fast=True,
#     local_files_only=False
# )
# model = AutoModel.from_pretrained("jackhhao/jailbreak-classifier")



# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
# from torch.optim import AdamW
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import roc_auc_score
# from tqdm.auto import tqdm
# import warnings
# warnings.filterwarnings('ignore')

# # ==================== Configuration ====================
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# MODEL_NAME = 'jackhhao/jailbreak-classifier'
# MAX_LENGTH = 128
# BATCH_SIZE = 16
# LEARNING_RATE = 2e-5
# EPOCHS = 10
# WARMUP_STEPS = 500
# RANDOM_SEED = 42

# torch.manual_seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)

# # ==================== Dataset Class ====================
# class TextDataset(Dataset):
#     def __init__(self, texts, labels, tokenizer, max_length):
#         self.texts = list(texts)
#         self.labels = torch.tensor(labels, dtype=torch.long)
#         self.tokenizer = tokenizer
#         self.max_length = max_length
    
#     def __len__(self):
#         return len(self.labels)
    
#     def __getitem__(self, idx):
#         encoding = self.tokenizer(
#             self.texts[idx],
#             max_length=self.max_length,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )
#         return {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'labels': self.labels[idx]
#         }

# # ==================== Data Loading ====================
# print(f"ğŸš€ Loading data... Device: {DEVICE}")

# train_df = pd.read_csv('/kaggle/input/rmit-hackathon-2025/train.csv')
# text_col = [col for col in train_df.columns if col.lower() in ['text', 'comment', 'review', 'content', 'sentence']][0]
# label_col = [col for col in train_df.columns if col.lower() in ['label', 'target', 'class', 'sentiment']][0]

# print(f"Text column: '{text_col}' | Label column: '{label_col}'")
# print(train_df[label_col].value_counts())

# # Encode labels
# label_encoder = LabelEncoder()
# train_df[label_col] = label_encoder.fit_transform(train_df[label_col])
# num_labels = len(label_encoder.classes_)

# X_train, X_val, y_train, y_val = train_test_split(
#     train_df[text_col].values, train_df[label_col].values,
#     test_size=0.1, random_state=RANDOM_SEED, stratify=train_df[label_col].values
# )

# print(f"Train: {len(X_train)} | Val: {len(X_val)} | Labels: {num_labels}")

# # Calculate class imbalance ratio
# unique, counts = np.unique(y_train, return_counts=True)
# class_distribution = dict(zip(unique, counts))
# print(f"\nğŸ“Š Class Distribution (Train):")
# for class_idx, count in class_distribution.items():
#     print(f"  Class {label_encoder.classes_[class_idx]}: {count}")

# imbalance_ratio = max(counts) / min(counts)
# print(f"\nâš ï¸�  Imbalance Ratio: {imbalance_ratio:.2f}")

# # ==================== Model Setup ====================
# print(f"\nğŸ¤– Loading model: {MODEL_NAME}")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels).to(DEVICE)

# train_dataset = TextDataset(X_train, y_train, tokenizer, MAX_LENGTH)
# val_dataset = TextDataset(X_val, y_val, tokenizer, MAX_LENGTH)

# # ==================== Handle Class Imbalance ====================
# # Calculate class weights (inverse of frequency)
# class_weights = torch.tensor([1.0 / class_distribution[i] for i in range(num_labels)], dtype=torch.float)
# class_weights = class_weights / class_weights.sum() * num_labels
# print(f"\nâš–ï¸�  Class Weights: {class_weights.numpy()}")

# # Create weighted sampler for balanced training
# sample_weights = np.array([class_weights[label].item() for label in y_train])
# sampler = WeightedRandomSampler(
#     weights=sample_weights,
#     num_samples=len(train_dataset),
#     replacement=True
# )

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# # ==================== Optimizer & Loss ====================
# optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
# scheduler = get_linear_schedule_with_warmup(
#     optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=len(train_loader) * EPOCHS
# )

# # Use weighted loss for imbalanced dataset
# criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

# # ==================== Training ====================
# def train_epoch():
#     model.train()
#     total_loss = 0
#     for batch in tqdm(train_loader, desc='Training'):
#         optimizer.zero_grad()
#         input_ids = batch['input_ids'].to(DEVICE)
#         attention_mask = batch['attention_mask'].to(DEVICE)
#         labels = batch['labels'].to(DEVICE)
        
#         outputs = model(input_ids, attention_mask)
#         logits = outputs.logits
        
#         loss = criterion(logits, labels)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()
#         scheduler.step()
#         total_loss += loss.item()
    
#     return total_loss / len(train_loader)

# def eval_model():
#     model.eval()
#     correct, total, total_loss = 0, 0, 0
#     all_preds = []
#     all_probs = []
#     all_labels = []
    
#     with torch.no_grad():
#         for batch in tqdm(val_loader, desc='Validating'):
#             input_ids = batch['input_ids'].to(DEVICE)
#             attention_mask = batch['attention_mask'].to(DEVICE)
#             labels = batch['labels'].to(DEVICE)
            
#             outputs = model(input_ids, attention_mask)
#             logits = outputs.logits
            
#             loss = criterion(logits, labels)
#             preds = torch.argmax(logits, dim=1)
#             probs = torch.softmax(logits, dim=1)
            
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)
#             total_loss += loss.item()
            
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#             all_probs.extend(probs.cpu().numpy())
    
#     accuracy = correct / total
#     avg_loss = total_loss / len(val_loader)
    
#     all_probs = np.array(all_probs)
#     all_labels = np.array(all_labels)
    
#     if num_labels == 2:
#         roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
#     else:
#         roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    
#     return accuracy, avg_loss, roc_auc

# print("\nğŸ�¯ Training Start")
# for epoch in range(EPOCHS):
#     train_loss = train_epoch()
#     val_acc, val_loss, val_auc = eval_model()
#     print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

# print("âœ¨ Training Complete\n")

# # ==================== Test Predictions ====================
# print("ğŸ”® Generating predictions...")

# test_df = pd.read_csv('/kaggle/input/rmit-hackathon-2025/test.csv')
# test_text_col = [col for col in test_df.columns if col.lower() in ['text', 'comment', 'review', 'content', 'sentence']][0]
# test_texts = test_df[test_text_col].values

# predictions = []
# model.eval()

# with torch.no_grad():
#     for i in tqdm(range(0, len(test_texts), BATCH_SIZE), desc='Predicting'):
#         batch = test_texts[i:i+BATCH_SIZE]
#         encodings = tokenizer(batch.tolist(), max_length=MAX_LENGTH, padding='max_length', 
#                              truncation=True, return_tensors='pt')
        
#         input_ids = encodings['input_ids'].to(DEVICE)
#         attention_mask = encodings['attention_mask'].to(DEVICE)
        
#         logits = model(input_ids, attention_mask).logits
#         preds = torch.argmax(logits, dim=1).cpu().numpy()
#         predictions.extend(preds)

# predictions = label_encoder.inverse_transform(predictions)

# submission_df = pd.DataFrame({'id': range(len(predictions)), 'prediction': predictions})
# submission_df.to_csv('submission.csv', index=False)
# print("âœ… Submission saved!")


# !pip install focal_loss_torch


!pip install skorch


import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'madhurjindal/Jailbreak-Detector-Large'
RANDOM_SEED = 42
K_FOLDS = 3
EPOCHS = 10  # âœ… CHANGED: Increased to 10 epochs
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EARLY_STOPPING_PATIENCE = 4  # âœ… NEW: Early stopping patience = 10 epochs

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"âœ… K-Fold Cross-Validation")
print(f"ğŸš€ Device: {DEVICE}")
print(f"ğŸ“Š Configuration: Max Epochs={EPOCHS}, Early Stopping Patience={EARLY_STOPPING_PATIENCE}\n")

# ==================== Dataset Class ====================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=384):
        self.texts = list(texts)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': self.labels[idx]
        }

# ==================== Custom Collate Function ====================
def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    return input_ids, attention_mask, labels

# ==================== Model-Agnostic Layer Freezing ====================
def freeze_early_layers(model, model_name):
    """
    Freeze early layers in a model-agnostic way.
    Works with BERT, DeBERTa, RoBERTa, DistilBERT, etc.
    """
    try:
        frozen = False
        
        # For BERT-based models
        if hasattr(model, 'bert'):
            for param in model.bert.embeddings.parameters():
                param.requires_grad = False
            
            # Check if encoder has layers
            if hasattr(model.bert.encoder, 'layer'):
                for param in model.bert.encoder.layer[:6].parameters():
                    param.requires_grad = False
            
            print("âœ“ Froze BERT embeddings and first 6 encoder layers")
            frozen = True
        
        # For DeBERTa models
        elif hasattr(model, 'deberta'):
            for param in model.deberta.embeddings.parameters():
                param.requires_grad = False
            
            if hasattr(model.deberta.encoder, 'layer'):
                for param in model.deberta.encoder.layer[:6].parameters():
                    param.requires_grad = False
            
            print("âœ“ Froze DeBERTa embeddings and first 6 encoder layers")
            frozen = True
        
        # For RoBERTa models
        elif hasattr(model, 'roberta'):
            for param in model.roberta.embeddings.parameters():
                param.requires_grad = False
            
            if hasattr(model.roberta.encoder, 'layer'):
                for param in model.roberta.encoder.layer[:6].parameters():
                    param.requires_grad = False
            
            print("âœ“ Froze RoBERTa embeddings and first 6 encoder layers")
            frozen = True
        
        # For DistilBERT models
        elif hasattr(model, 'distilbert'):
            for param in model.distilbert.embeddings.parameters():
                param.requires_grad = False
            
            if hasattr(model.distilbert.transformer, 'layer'):
                for param in model.distilbert.transformer.layer[:3].parameters():
                    param.requires_grad = False
            
            print("âœ“ Froze DistilBERT embeddings and first 3 transformer layers")
            frozen = True
        
        if not frozen:
            print("âš ï¸�  Could not identify model architecture for layer freezing")
            print(f"    Model: {model_name}")
            print(f"    Available base attributes: {[attr for attr in dir(model.config) if not attr.startswith('_')][:10]}")
    
    except Exception as e:
        print(f"âš ï¸�  Warning: Could not freeze layers: {e}")
        import traceback
        traceback.print_exc()

# ==================== Data Loading ====================
print(f"ğŸ“Š Loading data...")

train_df = pd.read_csv('/kaggle/input/rmit-hackathon-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/rmit-hackathon-2025/test.csv')

# Auto-detect text and label columns
text_col = [col for col in train_df.columns if col.lower() in ['text', 'comment', 'review', 'content', 'sentence']][0]
label_col = [col for col in train_df.columns if col.lower() in ['label', 'target', 'class', 'sentiment']][0]

print(f"Text column: '{text_col}' | Label column: '{label_col}'")
print(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")
print(f"\nLabel Distribution:")
print(train_df[label_col].value_counts())

# Encode labels
label_encoder = LabelEncoder()
train_df[label_col] = label_encoder.fit_transform(train_df[label_col])
num_labels = len(label_encoder.classes_)

X_train = train_df[text_col].values
y_train = train_df[label_col].values

test_text_col = [col for col in test_df.columns if col.lower() in ['text', 'comment', 'review', 'content', 'sentence']][0]
X_test = test_df[test_text_col].values

print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# Class distribution
unique, counts = np.unique(y_train, return_counts=True)
class_distribution = dict(zip(unique, counts))
print(f"\nğŸ“Š Class Distribution:")
for class_idx, count in class_distribution.items():
    print(f"  {label_encoder.classes_[class_idx]}: {count}")

imbalance_ratio = max(counts) / min(counts)
print(f"âš ï¸�  Imbalance Ratio: {imbalance_ratio:.2f}")

# Calculate class weights
class_weights = torch.tensor([1.0 / class_distribution[i] for i in range(num_labels)], dtype=torch.float)
class_weights = class_weights / class_weights.sum() * num_labels
class_weights = class_weights.to(DEVICE)
print(f"âš–ï¸�  Class Weights: {class_weights.cpu().numpy()}")

# ==================== Tokenizer ====================
print(f"\nğŸ¤– Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ==================== K-Fold Cross-Validation ====================
print(f"\n{'='*60}")
print(f"ğŸ�¯ K-FOLD CROSS-VALIDATION ({K_FOLDS} folds)")
print(f"{'='*60}\n")

skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)

fold_test_predictions = []
fold_accuracies = []
fold_aucs = []
fold_epochs_trained = []  # âœ… NEW: Track epochs trained per fold

fold_idx = 1
for train_idx, val_idx in skf.split(X_train, y_train):
    print(f"\n{'â”€'*60}")
    print(f"Fold {fold_idx}/{K_FOLDS}")
    print(f"{'â”€'*60}")
    
    try:
        X_tr, X_vl = X_train[train_idx], X_train[val_idx]
        y_tr, y_vl = y_train[train_idx], y_train[val_idx]
        
        print(f"Train samples: {len(X_tr)} | Val samples: {len(X_vl)}")
        
        # Create datasets
        train_dataset = TextDataset(X_tr, y_tr, tokenizer)
        val_dataset = TextDataset(X_vl, y_vl, tokenizer)
        test_dataset = TextDataset(X_test, np.zeros(len(X_test)), tokenizer)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Create model
        print(f"ğŸ¤– Loading model: {MODEL_NAME}")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, 
            num_labels=num_labels
        ).to(DEVICE)
        
        print(f"ğŸ“Œ Model architecture: {model.__class__.__name__}")
        
        # Freeze early layers (model-agnostic)
        freeze_early_layers(model, MODEL_NAME)
        
        # Setup optimizer
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        
        # Loss function
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        
        # âœ… NEW: Early stopping variables
        best_accuracy = 0
        best_auc = 0
        best_epoch = 0
        best_state = None
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        print(f"\nğŸ”„ Training (Max {EPOCHS} epochs, Early Stopping Patience: {EARLY_STOPPING_PATIENCE} epochs)...\n")
        
        for epoch in range(EPOCHS):
            # Training
            model.train()
            total_loss = 0
            batch_count = 0
            
            for input_ids, attention_mask, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}', leave=False):
                # Move all tensors to device
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                labels = labels.to(DEVICE)
                
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Compute loss
                loss = criterion(logits, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            avg_train_loss = total_loss / batch_count
            
            # Validation
            model.eval()
            all_preds, all_probs, all_labels = [], [], []
            val_total_loss = 0
            val_batch_count = 0
            
            with torch.no_grad():
                for input_ids, attention_mask, labels in tqdm(val_loader, desc='Validating', leave=False):
                    # Move all tensors to device
                    input_ids = input_ids.to(DEVICE)
                    attention_mask = attention_mask.to(DEVICE)
                    labels = labels.to(DEVICE)
                    
                    # Forward pass
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    
                    # Compute validation loss
                    val_loss = criterion(logits, labels)
                    val_total_loss += val_loss.item()
                    val_batch_count += 1
                    
                    probs = torch.softmax(logits, dim=1)
                    preds = torch.argmax(logits, dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
            
            all_preds = np.array(all_preds)
            all_probs = np.array(all_probs)
            all_labels = np.array(all_labels)
            avg_val_loss = val_total_loss / val_batch_count
            
            val_accuracy = accuracy_score(all_labels, all_preds)
            
            # Handle binary vs multi-class AUC
            if num_labels == 2:
                val_auc = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                val_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
            
            print(f"  Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Val AUC: {val_auc:.4f}")
            
            # âœ… NEW: Early stopping logic with 10 epoch patience
            if val_auc > best_auc:
                best_auc = val_auc
                best_accuracy = val_accuracy
                best_epoch = epoch + 1
                best_state = model.state_dict().copy()
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                print(f"    âœ… Best model updated (AUC: {best_auc:.4f}, Loss: {best_val_loss:.4f})")
            else:
                epochs_without_improvement += 1
                print(f"    âš ï¸�  No improvement for {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE} epochs")
                
                # âœ… Early stopping: stop if no improvement for EARLY_STOPPING_PATIENCE epochs
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    print(f"\n    ğŸ›‘ EARLY STOPPING TRIGGERED!")
                    print(f"    No improvement for {EARLY_STOPPING_PATIENCE} consecutive epochs")
                    print(f"    Stopping at epoch {epoch+1}/{EPOCHS}")
                    break
        
        # Load best model
        if best_state is not None:
            model.load_state_dict(best_state)
        
        print(f"\nâœ… Best Results - Epoch {best_epoch}: Acc={best_accuracy:.4f}, AUC={best_auc:.4f}")
        print(f"ğŸ“Š Training completed: {best_epoch} epochs trained (max {EPOCHS})")
        
        fold_accuracies.append(best_accuracy)
        fold_aucs.append(best_auc)
        fold_epochs_trained.append(best_epoch)  # âœ… NEW: Store epochs trained
        
        # Test predictions
        print(f"ğŸ”® Generating test predictions...")
        model.eval()
        test_preds = []
        
        with torch.no_grad():
            for input_ids, attention_mask, _ in tqdm(test_loader, desc='Predicting', leave=False):
                # Move tensors to device
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                probs = torch.softmax(logits, dim=1)
                test_preds.extend(probs.cpu().numpy())
        
        fold_test_predictions.append(np.array(test_preds))
        
        # Cleanup
        del model, optimizer, criterion
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"â�Œ Error in fold: {str(e)}")
        import traceback
        traceback.print_exc()
    
    fold_idx += 1


print(f"\n{'='*60}")
print(f"ğŸ“Š K-FOLD RESULTS SUMMARY")
print(f"{'='*60}")

if len(fold_accuracies) > 0:
    print(f"\nFold Accuracies: {[f'{acc:.4f}' for acc in fold_accuracies]}")
    print(f"Fold AUCs: {[f'{auc:.4f}' for auc in fold_aucs]}")
    print(f"\nğŸ“ˆ Mean Â± Std:")
    print(f"  Accuracy: {np.mean(fold_accuracies):.4f} Â± {np.std(fold_accuracies):.4f}")
    print(f"  AUC: {np.mean(fold_aucs):.4f} Â± {np.std(fold_aucs):.4f}")
    
    # Average predictions across all folds
    ensemble_preds = np.mean(fold_test_predictions, axis=0)
    
    # Verify predictions are valid
    print(f"\nâœ“ Ensemble predictions range: [{ensemble_preds.min():.6f}, {ensemble_preds.max():.6f}]")
    print(f"âœ“ Ensemble predictions shape: {ensemble_preds.shape}")
    print(f"âœ“ Number of samples: {ensemble_preds.shape[0]}")
    print(f"âœ“ Number of classes: {ensemble_preds.shape[1]}")
    
    # âœ… Extract probability for class 1 (or positive class)
    # For binary classification, we want P(class=1)
    if ensemble_preds.shape[1] == 2:
        final_predictions = ensemble_preds[:, 1]  # Probability of class 1
        print(f"\nâœ“ Using probability of class 1")
    else:
        # For multi-class, you can take max probability or specific class
        final_predictions = ensemble_preds.max(axis=1)
        print(f"\nâœ“ Using max probability (multi-class)")
    
    print(f"âœ“ Final predictions range: [{final_predictions.min():.6f}, {final_predictions.max():.6f}]")
    
    # ==================== Save Submission ====================
    print(f"\n{'='*60}")
    print(f"âœ… SAVING SUBMISSION")
    print(f"{'='*60}\n")
    
    submission_df = pd.DataFrame({
        'Id': test_df['Id'].values,
        'TARGET': final_predictions
    })
    
    print("First 10 rows of submission:")
    print(submission_df.head(10))
    
    submission_df.to_csv('submission.csv', index=False)
    
    print("\nâœ… Submission saved as 'submission.csv'")
    print(f"\nğŸ“Š Final Prediction Distribution:")
    print(f"  Min: {final_predictions.min():.6f}")
    print(f"  Max: {final_predictions.max():.6f}")
    print(f"  Mean: {final_predictions.mean():.6f}")
    print(f"  Median: {np.median(final_predictions):.6f}")
    print(f"  Std: {final_predictions.std():.6f}")
    
    # Distribution of confidence scores
    print(f"\nğŸ“ˆ Confidence Score Distribution:")
    print(f"  >= 0.95: {(final_predictions >= 0.95).sum()} samples ({(final_predictions >= 0.95).sum()/len(final_predictions)*100:.1f}%)")
    print(f"  >= 0.90: {(final_predictions >= 0.90).sum()} samples ({(final_predictions >= 0.90).sum()/len(final_predictions)*100:.1f}%)")
    print(f"  >= 0.80: {(final_predictions >= 0.80).sum()} samples ({(final_predictions >= 0.80).sum()/len(final_predictions)*100:.1f}%)")
    print(f"  >= 0.70: {(final_predictions >= 0.70).sum()} samples ({(final_predictions >= 0.70).sum()/len(final_predictions)*100:.1f}%)")
    print(f"  >= 0.50: {(final_predictions >= 0.50).sum()} samples ({(final_predictions >= 0.50).sum()/len(final_predictions)*100:.1f}%)")
    
    print(f"\nğŸ�† Final Summary:")
    print(f"  â€¢ Model: {MODEL_NAME}")
    print(f"  â€¢ K-Folds: {K_FOLDS}")
    print(f"  â€¢ Final Train AUC: {np.mean(fold_aucs):.4f}")
    print(f"  â€¢ Final Train Accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"  â€¢ Predictions saved: {len(final_predictions)}")
    print(f"  â€¢ Submission file: submission.csv")
    print(f"  â€¢ Output format: Probability scores (0-1)")
    print(f"\nâœ… ALL DONE!")
    
else:
    print("â�Œ No successful folds completed. Please check errors above.")





