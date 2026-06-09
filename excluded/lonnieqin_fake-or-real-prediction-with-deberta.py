import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import warnings
import gc
import os
warnings.filterwarnings('ignore')

# Configuration
class Config:
    model_name = '/kaggle/input/deberta-v3-base-tokenizer/deberta-v3-base'
    max_length = 512
    batch_size = 8
    learning_rate = 3e-5
    num_epochs = 6
    n_folds = 5
    seed = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gradient_accumulation_steps = 2
    warmup_ratio = 0.1
    weight_decay = 0.01
    dropout = 0.2

# Set seeds for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(Config.seed)

# Load and prepare data
print("Loading data...")
train_df = pd.read_csv("/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv")

train_texts = []
train_labels = []
pair_ids = []  # Track which pair each text belongs to

for i in range(len(train_df)):
    identifier = train_df.iloc[i]["id"]
    real_text_id = train_df.iloc[i]["real_text_id"]
    file1_path = f"/kaggle/input/fake-or-real-the-impostor-hunt/data/train/article_{identifier:04d}/file_1.txt"
    file2_path = f"/kaggle/input/fake-or-real-the-impostor-hunt/data/train/article_{identifier:04d}/file_2.txt"
    
    with open(file1_path, "r", encoding='utf-8') as f:
        text1 = f.read()
    with open(file2_path, "r", encoding='utf-8') as f:
        text2 = f.read()
    
    if real_text_id == 1:
        train_texts.append(text1)
        train_labels.append(1)
        pair_ids.append(identifier)
        train_texts.append(text2)
        train_labels.append(0)
        pair_ids.append(identifier)
    elif real_text_id == 2:
        train_texts.append(text1)
        train_labels.append(0)
        pair_ids.append(identifier)
        train_texts.append(text2)
        train_labels.append(1)
        pair_ids.append(identifier)

df = pd.DataFrame({
    "text": train_texts,
    "label": train_labels,
    "pair_id": pair_ids
})

print(f"Total samples: {len(df)}")
print(f"Label distribution: {df['label'].value_counts().to_dict()}")

# Custom Dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Model Definition
class TransformerClassifier(nn.Module):
    def __init__(self, model_name, num_classes=2, dropout=0.2):
        super(TransformerClassifier, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)
        
        # Additional layers for better performance
        self.pre_classifier = nn.Linear(self.transformer.config.hidden_size, 
                                       self.transformer.config.hidden_size)
        self.relu = nn.ReLU()
        
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use pooled output or mean of last hidden states
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            last_hidden_state = outputs.last_hidden_state
            pooled_output = torch.mean(last_hidden_state, dim=1)
        
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = self.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

# Training function
def train_epoch(model, dataloader, optimizer, scheduler, device, accumulation_steps):
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    progress_bar = tqdm(dataloader, desc='Training')
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss = loss / accumulation_steps
        loss.backward()
        
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        
        _, preds = torch.max(outputs, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': total_loss / (step + 1)})
    
    accuracy = accuracy_score(true_labels, predictions)
    return total_loss / len(dataloader), accuracy

# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []
    probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            probabilities.extend(probs[:, 1].cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    try:
        auc = roc_auc_score(true_labels, probabilities)
    except:
        auc = 0.5
    
    return accuracy, auc, predictions, probabilities

# Pairwise accuracy calculation
def calculate_pairwise_accuracy(df_pred, is_oof=False):
    """Calculate accuracy based on pairs of texts"""
    correct = 0
    total = 0
    probability_key = 'oof_probability' if is_oof else 'probability'
    label_key = 'oof_probability' if is_oof else 'label'
    for pair_id in df_pred['pair_id'].unique():
        pair_data = df_pred[df_pred['pair_id'] == pair_id]
        if len(pair_data) == 2:
            probs = pair_data[probability_key].values
            true_labels = pair_data[label_key].values
            
            # Predict the text with higher probability as real (label 1)
            pred_real_idx = np.argmax(probs)
            true_real_idx = np.argmax(true_labels)
            
            if pred_real_idx == true_real_idx:
                correct += 1
            total += 1
    
    return correct / total if total > 0 else 0

# K-Fold Cross Validation Training
print(f"\nStarting {Config.n_folds}-Fold Cross Validation Training...")
print(f"Device: {Config.device}")

tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/deberta-v3-base-tokenizer/deberta-v3-base-tokenizer")
kfold = StratifiedKFold(n_splits=Config.n_folds, shuffle=True, random_state=Config.seed)

cv_scores = []
cv_pairwise_scores = []
all_oof_predictions = np.zeros(len(df))
all_oof_probabilities = np.zeros(len(df))
for fold, (train_idx, val_idx) in enumerate(kfold.split(df['text'], df['label'])):
    print(f"\n{'='*50}")
    print(f"Fold {fold + 1}/{Config.n_folds}")
    print(f"{'='*50}")
    
    # Split data
    train_texts = df.iloc[train_idx]['text'].values
    train_labels = df.iloc[train_idx]['label'].values
    val_texts = df.iloc[val_idx]['text'].values
    val_labels = df.iloc[val_idx]['label'].values
    
    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, Config.max_length)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, Config.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)
    
    # Initialize model
    model = TransformerClassifier(Config.model_name, dropout=Config.dropout)
    model.to(Config.device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    total_steps = len(train_loader) * Config.num_epochs // Config.gradient_accumulation_steps
    warmup_steps = int(total_steps * Config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(Config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{Config.num_epochs}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, 
            Config.device, Config.gradient_accumulation_steps
        )
        val_acc, val_auc, val_preds, val_probs = evaluate(model, val_loader, Config.device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    # Load best model and get final predictions
    model.load_state_dict(best_model_state)
    val_acc, val_auc, val_preds, val_probs = evaluate(model, val_loader, Config.device)
    
    # Store out-of-fold predictions
    all_oof_predictions[val_idx] = val_preds
    all_oof_probabilities[val_idx] = val_probs
    
    # Calculate pairwise accuracy for validation set
    val_df = df.iloc[val_idx].copy()
    val_df['prediction'] = val_preds
    val_df['probability'] = val_probs
    pairwise_acc = calculate_pairwise_accuracy(val_df)
    
    cv_scores.append(val_acc)
    cv_pairwise_scores.append(pairwise_acc)
    
    print(f"\nFold {fold + 1} Results:")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation Pairwise Accuracy: {pairwise_acc:.4f}")
    
    # Save model
    torch.save(model.state_dict(), f'model_fold_{fold}.pth')
    
    # Clean up
    del optimizer, scheduler, train_dataset, val_dataset
    gc.collect()
    torch.cuda.empty_cache()

# Final Results
print("\n" + "="*50)
print("CROSS-VALIDATION RESULTS")
print("="*50)
print(f"CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
print(f"CV Pairwise Accuracy: {np.mean(cv_pairwise_scores):.4f} (+/- {np.std(cv_pairwise_scores):.4f})")

# Overall OOF pairwise accuracy
df['oof_prediction'] = all_oof_predictions
df['oof_probability'] = all_oof_probabilities
overall_pairwise_acc = calculate_pairwise_accuracy(df, True)
print(f"Overall OOF Pairwise Accuracy: {overall_pairwise_acc:.4f}")
print("\nTraining completed successfully!")


def predict_test_data(tokenizer, device, max_length=512, batch_size=8):
    import os
    import time
    
    start_time = time.time()
    
    # Read test files
    test_dir = "/kaggle/input/fake-or-real-the-impostor-hunt/data/test"
    
    # Get all test article folders
    test_folders = sorted([f for f in os.listdir(test_dir) if f.startswith("article_")])
    
    # Initialize model once
    model = TransformerClassifier(Config.model_name, dropout=Config.dropout)
    model.to(Config.device)
    model.eval()
    
    # Store predictions from all folds for ensemble
    all_fold_predictions = []
    
    total_operations = Config.n_folds * len(test_folders)
    current_operation = 0
    
    # Loop through each fold
    for fold in range(Config.n_folds):
        fold_start_time = time.time()
        print(f"Processing fold {fold + 1}/{Config.n_folds}")
        
        # Load the model for this fold
        state_dict = torch.load(f'model_fold_{fold}.pth', map_location=device)
        model.load_state_dict(state_dict)
        
        fold_results = []
        
        # Process each test article
        for folder in test_folders:
            current_operation += 1
            
            # Calculate timing info
            elapsed_time = time.time() - start_time
            progress = current_operation / total_operations
            
            if progress > 0:
                estimated_total_time = elapsed_time / progress
                remaining_time = estimated_total_time - elapsed_time
            else:
                estimated_total_time = 0
                remaining_time = 0
            
            # Format time strings
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            total_str = time.strftime("%H:%M:%S", time.gmtime(estimated_total_time))
            remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
            
            print(f"\rFold {fold+1} - Article {folder} | "
                  f"Elapsed: {elapsed_str} / Est. Total: {total_str} "
                  f"(Remaining: {remaining_str}) [{current_operation}/{total_operations}]", 
                  end='', flush=True)
            
            article_id = int(folder.split("_")[1])
            file1_path = os.path.join(test_dir, folder, "file_1.txt")
            file2_path = os.path.join(test_dir, folder, "file_2.txt")
            
            # Read the two text files
            with open(file1_path, "r", encoding='utf-8') as f:
                text1 = f.read()
            with open(file2_path, "r", encoding='utf-8') as f:
                text2 = f.read()
            
            # Get predictions for both texts
            probs = []
            with torch.no_grad():
                for text in [text1, text2]:
                    encoding = tokenizer(
                        text,
                        truncation=True,
                        padding='max_length',
                        max_length=max_length,
                        return_tensors='pt'
                    )
                    
                    input_ids = encoding['input_ids'].to(device)
                    attention_mask = encoding['attention_mask'].to(device)
                    
                    outputs = model(input_ids, attention_mask)
                    softmax_probs = torch.softmax(outputs, dim=1)
                    # Probability of being real (label=1)
                    real_prob = softmax_probs[0, 1].item()
                    probs.append(real_prob)
            
            fold_results.append({
                'id': article_id,
                'text1_prob': probs[0],
                'text2_prob': probs[1]
            })
        
        print()  # New line after fold completion
        fold_elapsed = time.time() - fold_start_time
        print(f"Fold {fold+1} completed in {time.strftime('%H:%M:%S', time.gmtime(fold_elapsed))}")
        
        all_fold_predictions.append(fold_results)
    
    # Ensemble predictions across all folds
    print("Ensembling predictions across folds...")
    final_results = []
    
    for i in range(len(test_folders)):
        article_id = all_fold_predictions[0][i]['id']
        
        # Average probabilities across all folds
        avg_text1_prob = np.mean([fold_preds[i]['text1_prob'] for fold_preds in all_fold_predictions])
        avg_text2_prob = np.mean([fold_preds[i]['text2_prob'] for fold_preds in all_fold_predictions])
        
        # Determine which text is real based on higher probability
        if avg_text1_prob > avg_text2_prob:
            real_text_id = 1
        else:
            real_text_id = 2
        
        final_results.append({
            'id': article_id,
            'real_text_id': real_text_id
        })
    
    total_elapsed = time.time() - start_time
    print(f"\nTotal processing time: {time.strftime('%H:%M:%S', time.gmtime(total_elapsed))}")
    
    return pd.DataFrame(final_results)


print("\nGenerating predictions for test set...")
test_predictions = predict_test_data(tokenizer, Config.device)
# Save submission
test_predictions.to_csv("submission.csv", index=False)
print("\nSubmission saved to 'submission.csv'")
print(test_predictions.head())

