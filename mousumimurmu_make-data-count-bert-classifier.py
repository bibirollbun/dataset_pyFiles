# Installation of libraries
!pip uninstall -y fitz
!pip uninstall -y pymupdf
!pip install pymupdf==1.23.7
!pip install --upgrade transformers


# Importing libraries
import os
import re
import gc
import fitz  
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.optim as optim 
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.metrics import f1_score, classification_report, confusion_matrix, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')


nltk.download('punkt')

# Configuration
CONFIG = {
    "max_length": 512,
    "batch_size": 8,
    "learning_rate": 2e-5,
    "epochs": 3,
    "context_window": 3,  # sentences around match
    "min_context_length": 50,
    "confidence_threshold": 0.7,
    "data_paths": {
    "train_csv": "/kaggle/input/make-data-count-finding-data-references/train_labels.csv",
    "train_pdf": "/kaggle/input/make-data-count-finding-data-references/train/PDF",
    "test_pdf": "/kaggle/input/make-data-count-finding-data-references/test/PDF"},
    "eval_metrics": ["macro_f1", "weighted_f1", "precision", "recall"],
    "label_names": ["Primary", "Secondary"]  
}


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("Setup complete.\n")


def load_and_balance_data(filepath, sample_size=None, random_state=42):
    df = pd.read_csv("/kaggle/input/make-data-count-finding-data-references/train_labels.csv")
    df = df[df["type"].isin(["Primary", "Secondary"])]
    if sample_size:
        min_samples = min(len(df[df["type"] == "Primary"]),
                         len(df[df["type"] == "Secondary"]),
                         sample_size)
        primary = df[df["type"] == "Primary"].sample(min_samples, random_state=random_state)
        secondary = df[df["type"] == "Secondary"].sample(min_samples, random_state=random_state)
        df = pd.concat([primary, secondary])

    print(f"Class distribution:\n{df['type'].value_counts()}")
    return df


def extract_context_enhanced(text, dataset_id, num_sentences=3):
    try:
        sentences = sent_tokenize(text)
        for i, sent in enumerate(sentences):
            if dataset_id.lower() in sent.lower():
                start = max(0, i - num_sentences)
                end = min(len(sentences), i + num_sentences + 1)
                context = ' '.join(sentences[start:end])

                # Fallback 1: If context too short, expand window
                if len(context) < CONFIG["min_context_length"] and len(sentences) > end:
                    end = min(len(sentences), end + 2)
                    context = ' '.join(sentences[start:end])

                # Fallback 2: If still too short, return full text
                if len(context) < CONFIG["min_context_length"]:
                    return text[:2000]  # Return first 2000 chars as fallback

                return context
    except:
        pass

    # Final fallback: return text around the match
    match = re.search(re.escape(dataset_id), text, re.IGNORECASE)
    if match:
        start = max(0, match.start() - 300)
        end = min(len(text), match.end() + 300)
        return text[start:end]

    return None


def process_data(df, pdf_dir):
    contexts = []
    missing_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing PDFs"):
        article_id = row["article_id"]
        dataset_id = row["dataset_id"]
        pdf_path = os.path.join(pdf_dir, f"{article_id}.pdf")

        try:
            with fitz.open(pdf_path) as doc:
                full_text = "\n".join([page.get_text() for page in doc])

                # Enhanced context extraction
                context = extract_context_enhanced(full_text, dataset_id, CONFIG["context_window"])

                if context:
                    contexts.append(context)
                else:
                    contexts.append(None)
                    missing_count += 1
        except Exception as e:
            contexts.append(None)
            missing_count += 1

    df["context"] = contexts

    # Analyze missing data
    print(f"\nMissing contexts: {missing_count}/{len(df)} ({missing_count/len(df)*100:.2f}%)")
    print("Missing data by class:")
    print(df[df["context"].isna()]["type"].value_counts())
    df["context"] = df["context"].fillna("")
    return df
    
  


# Loading Data 
train_df = load_and_balance_data(CONFIG["data_paths"]["train_csv"], sample_size=500)
train_df = process_data(train_df, CONFIG["data_paths"]["train_pdf"])



# Encoding labels
label_map = {"Primary": 0, "Secondary": 1}
train_df["label"] = train_df["type"].map(label_map)
train_texts = train_df["context"].tolist()
train_labels = train_df["label"].tolist()




class CitationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def compute_class_weights(labels):
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float).to(device)



def fine_tune_model(train_texts, train_labels):
    # Initializing tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    ).to(device)

    dataset = CitationDataset(train_texts, train_labels, tokenizer, CONFIG["max_length"])
    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"]) # Use torch.optim.AdamW
    total_steps = len(loader) * CONFIG["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    class_weights = compute_class_weights(train_labels)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0
        correct_predictions = 0

        progress_bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(preds == labels).item()

            # Progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{torch.sum(preds == labels).item()/len(labels):.2f}"
            })

        epoch_loss = total_loss / len(loader)
        epoch_acc = correct_predictions / len(dataset)
        print(f"Epoch {epoch + 1} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    return model, tokenizer


# Training the model
model, tokenizer = fine_tune_model(train_texts, train_labels)
print("\nModel training complete.\n")



def calculate_detailed_metrics(true_labels, pred_labels):
    from collections import defaultdict
    
    clf_report = classification_report(
        true_labels,
        pred_labels,
        target_names=CONFIG["label_names"],
        output_dict=True,
        digits=4
    )
    
    cm = confusion_matrix(true_labels, pred_labels)

    
    class_metrics = defaultdict(dict)
    for i, label in enumerate(CONFIG["label_names"]):
        tp = cm[i,i]
        fp = cm[:,i].sum() - tp
        fn = cm[i,:].sum() - tp
        tn = cm.sum() - (tp + fp + fn)

        class_metrics[label] = {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn,
            'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'F1': 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else 0
        }
        
    metrics_df = pd.DataFrame(class_metrics).T

    macro_avg = {
        'F1': f1_score(true_labels, pred_labels, average='macro'),
        'Precision': precision_score(true_labels, pred_labels, average='macro'),
        'Recall': recall_score(true_labels, pred_labels, average='macro')
    }

    return metrics_df, clf_report, macro_avg


def print_metrics(metrics_df, clf_report, macro_avg):
    print("\n" + "="*60)
    print("Detailed Evaluation Metrics".center(60))
    print("="*60)

    print("\nPer-Class Metrics:")
    print(metrics_df)

    print("\nClassification Report:")
    print(pd.DataFrame(clf_report).T)

    print("\nMacro Averages:")
    for metric, value in macro_avg.items():
        print(f"{metric:>10}: {value:.4f}")


def enhanced_evaluate(model, tokenizer, df):
    eval_dataset = CitationDataset(
        df["context"].tolist(),
        df["label"].tolist(),
        tokenizer
    )
    eval_loader = DataLoader(eval_dataset, batch_size=CONFIG["batch_size"])

    true_labels, pred_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            inputs = {k: v.to(device) for k, v in batch.items()
                     if k != 'labels'}
            labels = batch['labels'].to(device)

            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    metrics_df, clf_report, macro_avg = calculate_detailed_metrics(
        true_labels, pred_labels)
    
    print_metrics(metrics_df, clf_report, macro_avg)

    return metrics_df


print("\nRunning enhanced evaluation...")
metrics_df = enhanced_evaluate(model, tokenizer, train_df)


def extract_dataset_ids(text):
    patterns = [
        r"10\.\d{4,9}/[-._;()/:A-Z0-9]+",  # DOI
        r"CHEMBL\d+",                      # ChEMBL
        r"GSE\d+",                         # GEO
        r"GSM\d+",                         # GEO samples
        r"PRJ[DN][A-Z]\d+",               # NCBI projects
        r"SR[PRX]\d+"                      # SRA
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches
    return []


def process_test_data_with_metrics(pdf_dir, model, tokenizer):
    results = []

    for filename in tqdm(os.listdir(pdf_dir), desc="Processing Test PDFs"):
        if filename.endswith(".pdf"):
            article_id = filename[:-4]
            try:
                with fitz.open(os.path.join(pdf_dir, filename)) as doc:
                    text = "\n".join([page.get_text() for page in doc])

                    # Extract dataset IDs
                    dataset_ids = extract_dataset_ids(text)
                    for did in dataset_ids:
                        context = extract_context_enhanced(text, did)
                        if context:
                            preds, confs = predict_with_metrics(
                                model, tokenizer, [context])
                            results.append({
                                "article_id": article_id,
                                "dataset_id": did,
                                "type": CONFIG["label_names"][preds[0]],
                                "confidence": confs[0]
                            })
            except Exception as e:
                continue

    return pd.DataFrame(results)


def predict_with_metrics(model, tokenizer, texts):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=CONFIG["max_length"],
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidences, preds = torch.max(probs, dim=1)

    return preds.cpu().numpy(), confidences.cpu().numpy()


submission_df = process_test_data_with_metrics(
    CONFIG["data_paths"]["test_pdf"],
    model,
    tokenizer
)


submission_df = submission_df[
    submission_df["confidence"] >= CONFIG["confidence_threshold"]
]



submission_df["row_id"] = range(len(submission_df))
submission_df = submission_df[[
    "row_id", "article_id", "dataset_id", "type"
]]
submission_df.to_csv("submission.csv", index=False)

print(f"\nFinal submission saved with {len(submission_df)} predictions")



