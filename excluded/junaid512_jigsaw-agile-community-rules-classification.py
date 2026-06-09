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


df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")


df.head()


df.describe()


df.shape


# Install required packages
!pip install -q transformers datasets torch scikit-learn seaborn matplotlib sentencepiece wandb


# Import libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Load train data (replace with your file path)
train = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')  # Ensure you have this

print("Train Shape:", train.shape)
print("Test Shape:", test.shape)
print("\nColumns:", train.columns.tolist())
print("\nTarget Distribution:\n", train['rule_violation'].value_counts(normalize=True))


# Text length analysis
train['body_len'] = train['body'].astype(str).str.len()
train['word_count'] = train['body'].astype(str).str.split().str.len()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(data=train, x='body_len', hue='rule_violation', bins=30)
plt.title("Comment Length by Violation")

plt.subplot(1, 2, 2)
sns.histplot(data=train, x='word_count', hue='rule_violation', bins=30)
plt.title("Word Count by Violation")
plt.tight_layout()
plt.show()


# Punctuation & caps
train['exclaim_count'] = train['body'].str.count('!')
train['question_count'] = train['body'].str.count('\?')
train['caps_ratio'] = train['body'].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0)

plt.figure(figsize=(15, 4))
for i, col in enumerate(['exclaim_count', 'question_count', 'caps_ratio']):
    plt.subplot(1, 3, i+1)
    sns.boxplot(data=train, x='rule_violation', y=col)
    plt.title(f"{col} vs Rule Violation")
plt.tight_layout()
plt.show()


def clean_text(x):
    x = str(x)
    x = x.replace('\n', ' ').replace('\r', ' ')
    x = ' '.join(x.split())  # normalize whitespace
    return x

for col in ['body', 'rule']:
    train[col] = train[col].apply(clean_text)
    test[col] = test[col].apply(clean_text)


# Create input text: Rule + Comment
train['input_text'] = 'RULE: ' + train['rule'] + ' COMMENT: ' + train['body']
test['input_text'] = 'RULE: ' + test['rule'] + ' COMMENT: ' + test['body']


# Handcrafted features
def add_features(df):
    df = df.copy()
    df['body_chars'] = df['body'].astype(str).str.len()
    df['body_words'] = df['body'].astype(str).str.split().str.len()
    df['exclaim'] = df['body'].astype(str).str.count('!')
    df['question'] = df['body'].astype(str).str.count('\?')
    df['url'] = df['body'].astype(str).str.contains(r'http[s]?://|www\.', case=False, na=False).astype(int)
    df['caps_ratio'] = df['body'].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if pd.notna(x) and len(str(x)) > 0 else 0)
    return df

train = add_features(train)
test = add_features(test)


N_SPLITS = 5
skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)


class RuleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }


class RuleClassifier(nn.Module):
    def __init__(self, model_name='microsoft/deberta-v3-base'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 1)  # DeBERTa base hidden size

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(outputs.last_hidden_state[:, 0, :])  # [CLS]
        return self.classifier(x).squeeze()


def train_model(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids, attention_mask)
        loss = nn.BCEWithLogitsLoss()(logits, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, device):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['labels'].numpy()

            logits = model(input_ids, attention_mask).cpu().numpy()
            pred = logits

            preds.extend(pred)
            labels.extend(label)
    return np.array(preds), np.array(labels)


from torch.optim import AdamW
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
model_name = 'microsoft/deberta-v3-base'

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
deberta_test_oof = np.zeros(len(test))  # for ensemble

for fold, (train_idx, val_idx) in enumerate(skf.split(train, train['rule_violation'], groups=train['subreddit'])):
    print(f"\n--- Fold {fold+1} ---")

    # Data
    train_texts = train.iloc[train_idx]['input_text'].values
    train_labels = train.iloc[train_idx]['rule_violation'].values
    val_texts = train.iloc[val_idx]['input_text'].values
    val_labels = train.iloc[val_idx]['rule_violation'].values

    # Datasets
    train_dataset = RuleDataset(train_texts, train_labels, tokenizer)
    val_dataset = RuleDataset(val_texts, val_labels, tokenizer)
    test_dataset = RuleDataset(test['input_text'].values, np.zeros(len(test)), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Model
    model = RuleClassifier(model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * 3
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps*0.1, num_training_steps=total_steps)

    # Train
    best_val_auc = 0
    for epoch in range(3):
        train_loss = train_model(model, train_loader, optimizer, scheduler, device)
        val_pred, val_true = evaluate_model(model, val_loader, device)
        val_pred = torch.sigmoid(torch.from_numpy(val_pred)).numpy()
        val_auc = roc_auc_score(val_true, val_pred)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), f"deberta_fold{fold}.pt")

    # Load best
    model.load_state_dict(torch.load(f"deberta_fold{fold}.pt"))

    # OOF
    val_pred, _ = evaluate_model(model, val_loader, device)
    oof_preds[val_idx] = torch.sigmoid(torch.from_numpy(val_pred)).numpy()

    # Test
    test_pred, _ = evaluate_model(model, test_loader, device)
    deberta_test_oof += torch.sigmoid(torch.from_numpy(test_pred)).numpy() / N_SPLITS


print("Final OOF AUC:", roc_auc_score(train['rule_violation'], oof_preds))


# Use high-confidence test predictions as pseudo-labels
pseudo_preds = deberta_test_oof.copy()
high_conf_idx = (pseudo_preds > 0.95) | (pseudo_preds < 0.05)
pseudo_labels = (pseudo_preds[high_conf_idx] > 0.5).astype(int)

# Create pseudo dataset
pseudo_test = test[high_conf_idx].copy()
pseudo_test['rule_violation'] = pseudo_labels

# Combine with train
extended_train = pd.concat([train, pseudo_test[['input_text', 'rule_violation']]], ignore_index=True)
extended_train = add_features(extended_train)


# Retrain on extended data
final_test_preds = np.zeros(len(test))
# The groups parameter is missing in the split method, which is necessary for StratifiedGroupKFold.
# The 'subreddit' column is not present in the extended_train dataframe after concatenating.
# I'll add the 'subreddit' column to the pseudo_test dataframe before concatenating.
pseudo_test['subreddit'] = 'pseudo'
extended_train = pd.concat([train, pseudo_test[['input_text', 'rule_violation', 'subreddit']]], ignore_index=True)
extended_train = add_features(extended_train)

for fold, (train_idx, val_idx) in enumerate(skf.split(extended_train, extended_train['rule_violation'], groups=extended_train['subreddit'])):
    print(f"\n--- Retraining Fold {fold+1} with Pseudo-Labels ---")

    train_texts = extended_train.iloc[train_idx]['input_text'].values
    train_labels = extended_train.iloc[train_idx]['rule_violation'].values
    val_texts = extended_train.iloc[val_idx]['input_text'].values
    val_labels = extended_train.iloc[val_idx]['rule_violation'].values

    train_dataset = RuleDataset(train_texts, train_labels, tokenizer)
    val_dataset = RuleDataset(val_texts, val_labels, tokenizer)
    test_dataset = RuleDataset(test['input_text'].values, np.zeros(len(test)), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = RuleClassifier(model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * 3
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps*0.1, num_training_steps=total_steps)

    best_val_auc = 0
    for epoch in range(3):
        train_loss = train_model(model, train_loader, optimizer, scheduler, device)
        val_pred, val_true = evaluate_model(model, val_loader, device)
        val_pred = torch.sigmoid(torch.from_numpy(val_pred)).numpy()
        val_auc = roc_auc_score(val_true, val_pred)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), f"deberta_ssl_fold{fold}.pt")

    model.load_state_dict(torch.load(f"deberta_ssl_fold{fold}.pt"))
    test_pred, _ = evaluate_model(model, test_loader, device)
    final_test_preds += torch.sigmoid(torch.from_numpy(test_pred)).numpy() / N_SPLITS


# TF-IDF + Meta Features
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(train['body'])
X_test_tfidf = tfidf.transform(test['body'])

meta_cols = ['body_chars', 'body_words', 'exclaim', 'question', 'url', 'caps_ratio']
X_train_meta = train[meta_cols].values
X_test_meta = test[meta_cols].values

X_train_combined = np.hstack([X_train_tfidf.toarray(), X_train_meta])
X_test_combined = np.hstack([X_test_tfidf.toarray(), X_test_meta])

# Train XGBoost
xgb = XGBClassifier(n_estimators=100, max_depth=6, random_state=42)
xgb.fit(X_train_combined, train['rule_violation'])

xgb_test_preds = xgb.predict_proba(X_test_combined)[:, 1]


# Rank average
deberta_rank = (final_test_preds.argsort().argsort() + 1) / len(final_test_preds)
xgb_rank = (xgb_test_preds.argsort().argsort() + 1) / len(xgb_test_preds)

final_preds = 0.7 * deberta_rank + 0.3 * xgb_rank


iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit((oof_preds.argsort().argsort() + 1) / len(oof_preds), train['rule_violation'])
final_preds_calibrated = iso_reg.predict(final_preds)


iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit((oof_preds.argsort().argsort() + 1) / len(oof_preds), train['rule_violation'])
final_preds_calibrated = iso_reg.predict(final_preds)


# Rank-transform OOF preds (same as in fit)
oof_ranks = (oof_preds.argsort().argsort() + 1) / len(oof_preds)

# Fit calibrator on OOF (train-level)
iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit(oof_ranks, train['rule_violation'])

# Apply to OOF (for evaluation)
oof_calibrated = iso_reg.predict(oof_ranks)

# Apply to test (for submission)
final_preds_calibrated = iso_reg.predict((final_preds.argsort().argsort() + 1) / len(final_preds))


from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import numpy as np

# Convert probabilities to binary predictions (threshold = 0.5)
oof_pred_binary = (oof_calibrated > 0.5).astype(int)

# 1. Accuracy
acc = accuracy_score(train['rule_violation'], oof_pred_binary)
print(f"ðŸŽ¯ Accuracy: {acc:.4f} ({acc*100:.2f}%)")

# 2. AUC (Primary Metric!)
auc = roc_auc_score(train['rule_violation'], oof_calibrated)
print(f"ðŸ“ˆ AUC: {auc:.4f}")

# 3. Full Classification Report
print("\nðŸ“‹ Classification Report:")
print(classification_report(train['rule_violation'], oof_pred_binary, target_names=['No Violation', 'Violation']))


import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# Create calibration plot
fraction_of_positives, mean_predicted_value = calibration_curve(
    train['rule_violation'], oof_calibrated, n_bins=10
)

plt.figure(figsize=(8, 6))
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Calibrated Model")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly Calibrated")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Calibration Plot (Reliability Diagram)")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
for cls in [0, 1]:
    subset = oof_calibrated[train['rule_violation'] == cls]
    plt.hist(subset, bins=50, alpha=0.7, label=f'Class {cls} ({"No Violation" if cls==0 else "Violation"})', density=True)

plt.xlabel('Predicted Probability (Calibrated)')
plt.ylabel('Density')
plt.title('Prediction Distribution by True Class')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(train['rule_violation'], oof_calibrated)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()


cm = confusion_matrix(train['rule_violation'], oof_pred_binary)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(train['rule_violation'], oof_calibrated)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'Precision-Recall AUC')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()


sub = pd.DataFrame({
    'row_id': test['row_id'],
    'rule_violation': final_preds_calibrated
})
sub.to_csv('submission.csv', index=False)
print("Submission saved!")
sub.head()







