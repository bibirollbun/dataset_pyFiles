# =========================
# 1) Imports & Setup
# =========================
import os, re
from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET

# PyPDF2 for PDFs
_HAS_PYPDF2 = False
try:
    import PyPDF2
    _HAS_PYPDF2 = True
except ImportError:
    pass

# Transformers & Torch
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# 2) Paths
# =========================
TRAIN_DIR   = "/kaggle/input/make-data-count-finding-data-references/train"
TEST_DIR    = "/kaggle/input/make-data-count-finding-data-references/test"
LABELS      = "/kaggle/input/make-data-count-finding-data-references/train_labels.csv"
SAMPLE_SUB  = "/kaggle/input/make-data-count-finding-data-references/sample_submission.csv"
OFFLINE_SCIBERT = "/kaggle/input/scibert/scibert"

# =========================
# 3) Helpers: Extract text
# =========================
def extract_pdf_text_pypdf2(path):
    text_parts = []
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for pg in reader.pages:
                text_parts.append(pg.extract_text() or "")
    except Exception as e:
        print("PyPDF2 failed for", path, e)
    return "\n".join(text_parts)

def extract_xml_text(path):
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        texts = [elem.text.strip() for elem in root.iter() if elem.text and elem.text.strip()]
        return "\n".join(texts)
    except Exception as e:
        print("XML parse failed for", path, e)
        return ""

def extract_text_file(path):
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        if _HAS_PYPDF2:
            return extract_pdf_text_pypdf2(path)
        else:
            return ""
    elif p.suffix.lower() in (".xml", ".nxml"):
        return extract_xml_text(path)
    else:
        try:
            return open(path, "r", encoding="utf-8").read()
        except Exception:
            return ""

def load_all_texts(folder):
    all_texts, all_ids = [], []
    folder = Path(folder)
    for file_path in folder.rglob("*"):
        if file_path.suffix.lower() in [".pdf", ".xml", ".nxml"]:
            txt = extract_text_file(str(file_path))
            all_texts.append(txt)
            all_ids.append(file_path.stem)
    return all_texts, all_ids

# =========================
# 4) Load Data
# =========================
print("Extracting train texts...")
train_texts, train_ids = load_all_texts(TRAIN_DIR)
print("Train docs:", len(train_texts))

print("Extracting test texts...")
test_texts, test_ids = load_all_texts(TEST_DIR)
print("Test docs:", len(test_texts))

train_labels = pd.read_csv(LABELS)
sample_sub = pd.read_csv(SAMPLE_SUB)
print("Train labels columns:", train_labels.columns)
print("Sample submission columns:", sample_sub.columns)

# =========================
# 5) Regex Baseline
# =========================
patterns = [
    r"(10\.\d{4,9}/[-._;\(\)/:A-Z0-9]+)",
    r"(arXiv:\d{4}\.\d{4,5}(v\d+)?)",
    r"(figshare\.com/\S+)",
    r"(zenodo\.org/record/\d+)",
    r"(datadryad\.org/stash/\S+)",
    r"(https?://[^\s]+)"
]
REFERENCE_PATTERN = re.compile("|".join(patterns), re.I)

def predict_references(texts, ids):
    preds = []
    row_id = 0
    for doc_id, text in zip(ids, texts):
        matches = REFERENCE_PATTERN.findall(text)
        for m in matches:
            if isinstance(m, tuple):
                m = [x for x in m if x]
                if not m: continue
                m = m[0]
            # classify type
            if m.lower().startswith("10."): ref_type = "DOI"
            elif "arxiv" in m.lower(): ref_type = "arXiv"
            elif "figshare" in m.lower(): ref_type = "Figshare"
            elif "zenodo" in m.lower(): ref_type = "Zenodo"
            elif "dryad" in m.lower(): ref_type = "Dryad"
            elif m.lower().startswith("http"): ref_type = "URL"
            else: ref_type = "Other"

            preds.append({
                "row_id": row_id,
                "article_id": doc_id,
                "dataset_id": m,
                "type": ref_type
            })
            row_id += 1
    return pd.DataFrame(preds)

baseline_submission = predict_references(test_texts, test_ids)
baseline_submission.to_csv("submission_baseline.csv", index=False)
print("✅ Baseline saved.")

# =========================
# 6) SciBERT Fine-tuning
# =========================
tokenizer = AutoTokenizer.from_pretrained(OFFLINE_SCIBERT, local_files_only=True)
model = AutoModelForTokenClassification.from_pretrained(
    OFFLINE_SCIBERT, local_files_only=True, num_labels=2
)
model.to(device)

labels_dict = train_labels.groupby("article_id")["dataset_id"].apply(list).to_dict()
train_records = []

for doc_text, doc_id in zip(train_texts, train_ids):
    if doc_id not in labels_dict: continue
    refs = labels_dict[doc_id]
    tokens = tokenizer(doc_text, truncation=True, max_length=512, return_offsets_mapping=True)
    labels = [0]*len(tokens["input_ids"])
    for i, (start, end) in enumerate(tokens["offset_mapping"]):
        if start==end: continue
        token_str = doc_text[start:end].lower()
        if any(r.lower() in token_str for r in refs):
            labels[i] = 1
    tokens["labels"] = labels
    train_records.append(tokens)

train_dataset = Dataset.from_list(train_records)
data_collator = DataCollatorForTokenClassification(tokenizer)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=3e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

# =========================
# 7) Prediction with SciBERT (GPU-safe)
# =========================
def predict_on_docs(texts, ids):
    preds = []
    row_id = 0
    for doc_id, text in zip(ids, texts):
        inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            pred_labels = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().cpu())
        for tok, label in zip(tokens, pred_labels):
            if label == 1 and tok not in ["[CLS]", "[SEP]"]:
                if tok.lower().startswith("10."): ref_type = "DOI"
                elif "arxiv" in tok.lower(): ref_type = "arXiv"
                elif "figshare" in tok.lower(): ref_type = "Figshare"
                elif "zenodo" in tok.lower(): ref_type = "Zenodo"
                elif "dryad" in tok.lower(): ref_type = "Dryad"
                elif tok.lower().startswith("http"): ref_type = "URL"
                else: ref_type = "Other"
                preds.append({
                    "row_id": row_id,
                    "article_id": doc_id,
                    "dataset_id": tok,
                    "type": ref_type
                })
                row_id += 1
    return pd.DataFrame(preds)

sbert_submission = predict_on_docs(test_texts, test_ids)
sbert_submission.to_csv("submission_sbert.csv", index=False)
print("✅ SciBERT saved.")

# =========================
# 8) Ensemble Baseline + SciBERT
# =========================
ensemble = pd.concat([baseline_submission, sbert_submission], ignore_index=True)
ensemble = ensemble.drop_duplicates(subset=["article_id", "dataset_id", "type"]).reset_index(drop=True)
ensemble["row_id"] = range(len(ensemble))
ensemble.to_csv("submission.csv", index=False)
print("✅ Ensemble submission.csv ready for Kaggle!")





