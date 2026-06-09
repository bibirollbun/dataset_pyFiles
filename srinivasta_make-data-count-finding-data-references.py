import os
import re
import pandas as pd
import numpy as np
from lxml import etree
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
from collections import Counter
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize

# === Paths ===
TRAIN_XML_FOLDER = "/kaggle/input/make-data-count-finding-data-references/train/XML"
TRAIN_PDF_FOLDER = "/kaggle/input/make-data-count-finding-data-references/train/PDF"
TRAIN_LABELS_CSV = "/kaggle/input/make-data-count-finding-data-references/train_labels.csv"
TEST_XML_FOLDER = "/kaggle/input/make-data-count-finding-data-references/test/XML"
TEST_PDF_FOLDER = "/kaggle/input/make-data-count-finding-data-references/test/PDF"

# === Utility Functions ===

def extract_text_from_xml(file_path):
    try:
        tree = etree.parse(file_path)
        return " ".join(tree.xpath('//text()'))
    except Exception as e:
        print(f"â�Œ Failed to parse {file_path}: {e}")
        return ""

def extract_text_by_section_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text("text") + "\n"
    except Exception as e:
        print(f"â�Œ Error reading {pdf_path}: {e}")
        return {}

    sections = {}
    current_section = "Unknown"
    buffer = []

    lines = full_text.split('\n')
    for line in lines:
        line_clean = line.strip().lower()
        if re.match(r'^(abstract|introduction|background|methods?|materials and methods|results?|discussion|conclusion|references?)$', line_clean):
            if buffer:
                sections[current_section] = "\n".join(buffer).strip()
                buffer = []
            current_section = line_clean.title()
        else:
            buffer.append(line.strip())

    if buffer:
        sections[current_section] = "\n".join(buffer).strip()

    return sections

def normalize_doi(doi):
    doi = doi.strip().lower()
    if doi.startswith("doi:"):
        doi = doi[4:]
    doi = re.sub(r'^https?://(dx\.)?doi\.org/', '', doi)
    return "https://doi.org/" + doi

def extract_references(text):
    pattern = r'\b(10\.\d{4,9}/[-._;()/:a-z0-9]+|GSE\d+|E-[A-Z]+-\d+|PRJ[EDNA]\d+|PDB\s*\w+|CHEMBL\d+)\b'
    matches = re.findall(pattern, text, flags=re.I)
    references = set()

    for ref in matches:
        ref_norm = ref.strip()
        is_doi = ref_norm.lower().startswith("10.")
        if is_doi:
            ref_norm = normalize_doi(ref_norm)
            ref_type = "Primary" if re.search(r"(this study|generated|we used|deposited|submitted|figshare|dryad|we generated)", text, flags=re.I) else "Secondary"
        else:
            ref_norm = ref_norm.replace(" ", "").upper()
            ref_type = "Primary" if re.search(r"(deposited|available at|generated|submitted|this study)", text, flags=re.I) else "Secondary"
        references.add((ref_norm, ref_type))

    return list(references)

def extract_snippet_around_match(text, match):
    sentences = sent_tokenize(text)
    for sent in sentences:
        if match and match.group(0).lower() in sent.lower():
            return sent
    return text[:150]  # fallback

keywords_primary = ['this study', 'generated', 'deposited', 'we used', 'available at', 'submitted', 'we generated']
keywords_secondary = ['obtained from', 'reused', 'derived from', 'downloaded from', 'previously published']

def keyword_feature(text):
    text_lower = text.lower()
    primary_flag = any(k in text_lower for k in keywords_primary)
    secondary_flag = any(k in text_lower for k in keywords_secondary)
    return int(primary_flag or not secondary_flag)

# === Load training labels ===
print("ğŸ“¥ Loading training labels...")
train_labels_df = pd.read_csv(TRAIN_LABELS_CSV)
print(f"Total training labels: {len(train_labels_df)}")

# === Extract training data ===
contexts, labels = [], []

for _, row in train_labels_df.iterrows():
    article_id, dataset_id, true_type = row['article_id'], row['dataset_id'], row['type']
    found = False

    xml_path = os.path.join(TRAIN_XML_FOLDER, article_id + ".xml")
    if os.path.exists(xml_path):
        xml_text = extract_text_from_xml(xml_path)
        refs = extract_references(xml_text)
        for ref, _ in refs:
            if ref.lower() == dataset_id.lower():
                match = re.search(re.escape(dataset_id), xml_text, flags=re.I)
                snippet = extract_snippet_around_match(xml_text, match)
                contexts.append(snippet)
                labels.append(1 if true_type.lower() == "primary" else 0)
                found = True
                break

    if not found:
        pdf_path = os.path.join(TRAIN_PDF_FOLDER, article_id + ".pdf")
        if os.path.exists(pdf_path):
            sections = extract_text_by_section_from_pdf(pdf_path)
            for section_text in sections.values():
                refs = extract_references(section_text)
                for ref, _ in refs:
                    if ref.lower() == dataset_id.lower():
                        match = re.search(re.escape(dataset_id), section_text, flags=re.I)
                        snippet = extract_snippet_around_match(section_text, match)
                        contexts.append(snippet)
                        labels.append(1 if true_type.lower() == "primary" else 0)
                        found = True
                        break
                if found:
                    break

print(f"âœ… Extracted {len(contexts)} training snippets.")
print("ğŸ“Š Label distribution:", Counter(labels))

# === Feature Engineering ===
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(contexts)
X_keywords = np.array([keyword_feature(txt) for txt in contexts]).reshape(-1, 1)
X = hstack([X_tfidf, X_keywords])
y = np.array(labels)

# === Train/Test Split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# === Model Training ===
clf = LogisticRegression(max_iter=1000, class_weight='balanced')
clf.fit(X_train, y_train)

# === Threshold Optimization ===
y_proba = clf.predict_proba(X_val)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
best_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_idx]

print(f"ğŸ”§ Best Threshold: {best_thresh:.2f}")
print(f"Precision: {precisions[best_idx]:.4f}")
print(f"Recall:    {recalls[best_idx]:.4f}")
print(f"F1-score:  {f1_scores[best_idx]:.4f}")

# === Plot Threshold Curve ===
plt.figure(figsize=(8, 5))
plt.plot(thresholds, f1_scores[:-1])
plt.axvline(best_thresh, color='red', linestyle='--', label=f'Best: {best_thresh:.2f}')
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.title("F1 Score vs Threshold")
plt.legend()
plt.grid()
plt.show()

# === Inference ===
print("\nğŸ“‚ Running inference on test set...")
submission_rows = []

def normalize_dataset_id(ds_id):
    if ds_id.lower().startswith("10."):
        return normalize_doi(ds_id)
    return ds_id

for file in os.listdir(TEST_XML_FOLDER):
    if not file.endswith(".xml"):
        continue

    article_id = file.replace(".xml", "")
    seen = set()

    # XML
    xml_path = os.path.join(TEST_XML_FOLDER, file)
    xml_text = extract_text_from_xml(xml_path)
    refs = extract_references(xml_text)
    for dataset_id, _ in refs:
        if dataset_id.lower() in seen:
            continue
        seen.add(dataset_id.lower())
        match = re.search(re.escape(dataset_id), xml_text, flags=re.I)
        snippet = extract_snippet_around_match(xml_text, match)
        x_feat = hstack([vectorizer.transform([snippet]), np.array([[keyword_feature(snippet)]])])
        prob = clf.predict_proba(x_feat)[:, 1][0]
        pred_type = "Primary" if prob >= best_thresh else "Secondary"
        submission_rows.append({
            "article_id": article_id,
            "dataset_id": normalize_dataset_id(dataset_id),
            "type": pred_type
        })

    # PDF
    pdf_path = os.path.join(TEST_PDF_FOLDER, article_id + ".pdf")
    if os.path.exists(pdf_path):
        sections = extract_text_by_section_from_pdf(pdf_path)
        for section_text in sections.values():
            refs = extract_references(section_text)
            for dataset_id, _ in refs:
                if dataset_id.lower() in seen:
                    continue
                seen.add(dataset_id.lower())
                match = re.search(re.escape(dataset_id), section_text, flags=re.I)
                snippet = extract_snippet_around_match(section_text, match)
                x_feat = hstack([vectorizer.transform([snippet]), np.array([[keyword_feature(snippet)]])])
                prob = clf.predict_proba(x_feat)[:, 1][0]
                pred_type = "Primary" if prob >= best_thresh else "Secondary"
                submission_rows.append({
                    "article_id": article_id,
                    "dataset_id": normalize_dataset_id(dataset_id),
                    "type": pred_type
                })

# Remove duplicates
submission_df = pd.DataFrame(submission_rows).drop_duplicates(subset=["article_id", "dataset_id", "type"])
submission_df.insert(0, "row_id", range(len(submission_df)))
submission_df.to_csv("submission.csv", index=False)
print(f"âœ… Submission saved: submission.csv with {len(submission_df)} rows")


    

