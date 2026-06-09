# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
# Imports
import os, re, gc
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from pypdf import PdfReader
from xml.etree import ElementTree as ET
import concurrent.futures

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from scipy.sparse import hstack

import lightgbm as lgb

# ---------- 0) Auto-detect base dir ----------
def find_base_dir():
    for p in Path('/kaggle/input').glob('*'):
        if p.is_dir():
            files = {f.name.lower() for f in p.iterdir() if f.is_file()}
            dirs  = {d.name.lower() for d in p.iterdir() if d.is_dir()}
            if 'train_labels.csv' in files and 'train' in dirs and 'test' in dirs:
                return p
    return Path('.')  # fallback

BASE_DIR = find_base_dir()
print("BASE_DIR:", BASE_DIR)

train_csv = BASE_DIR / 'train_labels.csv'
sample_csv = BASE_DIR / 'sample_submission.csv'

train_pdf_dir = BASE_DIR / 'train' / 'PDF'
train_xml_dir = BASE_DIR / 'train' / 'XML'
test_pdf_dir  = BASE_DIR / 'test'  / 'PDF'
test_xml_dir  = BASE_DIR / 'test'  / 'XML'

print("Exists train_pdf:", train_pdf_dir.exists(), "train_xml:", train_xml_dir.exists())
print("Exists test_pdf:", test_pdf_dir.exists(), "test_xml:", test_xml_dir.exists())
print("train_csv exists:", train_csv.exists(), "sample_submission exists:", sample_csv.exists())

# ---------- 1) Quick diagnostics ----------
train_df = pd.read_csv(train_csv)
sample_submission = pd.read_csv(sample_csv)

print("Train rows:", len(train_df))
print("Sample_submission rows:", len(sample_submission))
print("Sample columns:", list(sample_submission.columns))
print("Train label counts:\n", train_df['type'].value_counts())

# ---------- 2) Parsing helpers OPTIMISÉ ----------
def read_pdf_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        reader = PdfReader(str(path))
        parts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t:
                parts.append(t)
        return " ".join(parts)
    except Exception:
        return ""

def read_xml_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        tree = ET.parse(str(path))
        root = tree.getroot()
        parts = []
        for elem in root.iter():
            if elem.text:
                t = str(elem.text).strip()
                if t:
                    parts.append(t)
        return " ".join(parts)
    except Exception:
        return ""

def build_full_text(article_id, xml_dir, pdf_dir):
    pdf_path = pdf_dir / f"{article_id}.pdf"
    xml_path = xml_dir / f"{article_id}.xml"
    return (read_pdf_text(pdf_path) + " " + read_xml_text(xml_path)).strip()

# Fonction pour traitement parallèle
def process_article(args):
    aid, xml_dir, pdf_dir, dataset_id = args
    base_text = build_full_text(aid, xml_dir, pdf_dir)
    return (base_text + " " + str(dataset_id)).strip()

# ---------- 3) Regex pour extraire dataset ids ----------
DOI_CORE = r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)"
re_doi_url  = re.compile(r"https?://(?:dx\.)?doi\.org/" + DOI_CORE, re.I)
re_doi_bare = re.compile(r"\bdoi:\s*" + DOI_CORE, re.I)
re_doi_loose= re.compile(r"\b" + DOI_CORE + r"\b", re.I)

re_gse     = re.compile(r"\bGSE\d+\b", re.I)
re_array1  = re.compile(r"\bE-MTAB-\d+\b", re.I)
re_array2  = re.compile(r"\bE-MEXP-\d+\b", re.I)
re_ena     = re.compile(r"\bPRJ[EDN][A-Z0-9]+\b", re.I)
re_pdb     = re.compile(r"\bPDB\s*[A-Za-z0-9]{4}\b", re.I)

def canonical_doi(s):
    s = s.strip().rstrip(').,;')
    return "https://doi.org/" + s.lower()

def normalize_acc(tok):
    t = tok.strip().rstrip(').,;')
    if t.upper().startswith("PDB"):
        code = t.split()[-1]
        return f"PDB {code.upper()}"
    return t.upper()

def extract_dataset_ids(text):
    found = set()
    if not text:
        return found
    for m in re_doi_url.finditer(text):
        found.add(canonical_doi(m.group(1)))
    for m in re_doi_bare.finditer(text):
        found.add(canonical_doi(m.group(1)))
    for m in re_doi_loose.finditer(text):
        found.add(canonical_doi(m.group(1)))
    for rx in (re_gse, re_array1, re_array2, re_ena, re_pdb):
        for m in rx.finditer(text):
            found.add(normalize_acc(m.group(0)))
    return found

# ---------- 4) Build train texts OPTIMISÉ avec multiprocessing ----------
print("Building train texts with multiprocessing...")
train_args = [(str(r.article_id), train_xml_dir, train_pdf_dir, str(r.dataset_id)) 
              for r in train_df.itertuples()]

train_texts = []
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(tqdm(executor.map(process_article, train_args), total=len(train_args)))
    train_texts = results

# Labels
le = LabelEncoder()
y = le.fit_transform(train_df['type'].astype(str))
print("Label classes:", list(le.classes_))

# ---------- 5) TF-IDF OPTIMISÉ ----------
# Réduction significative des features pour accélérer
word_vec = TfidfVectorizer(max_features=50000, ngram_range=(1,2), 
                          stop_words='english', min_df=2)
char_vec = TfidfVectorizer(analyzer='char', ngram_range=(3,5), 
                          min_df=2, max_features=30000)

print("Fitting TF-IDF ...")
Xw = word_vec.fit_transform(train_texts)
Xc = char_vec.fit_transform(train_texts)
X = hstack([Xw, Xc]).tocsr()
del Xw, Xc; gc.collect()
print("X shape:", X.shape)

# ---------- 6) Train LightGBM SANS K-FOLD (BEAUCOUP PLUS RAPIDE) ----------
n_classes = len(le.classes_)

# Split simple train/validation (80/20)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train: {X_train.shape[0]}, Validation: {X_val.shape[0]}")

# Paramètres optimisés
params = dict(objective='multiclass', num_class=n_classes, metric='multi_logloss',
              boosting_type='gbdt', learning_rate=0.1, num_leaves=63,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=3,
              max_depth=7, min_data_in_leaf=20,
              verbosity=-1, n_jobs=2)

print("Training single LightGBM model...")
dtr = lgb.Dataset(X_train, label=y_train)
dva = lgb.Dataset(X_val, label=y_val, reference=dtr)

# Entraînement avec early stopping
model = lgb.train(params, dtr, num_boost_round=1000,
                  valid_sets=[dva], valid_names=['valid'],
                  callbacks=[
                      lgb.early_stopping(stopping_rounds=50), 
                      lgb.log_evaluation(100)
                  ])

# Validation score
val_proba = model.predict(X_val, num_iteration=model.best_iteration)
val_pred = val_proba.argmax(axis=1)
val_f1 = f1_score(y_val, val_pred, average='macro')
print(f"Validation F1 macro: {val_f1:.4f}")

# ---------- 7) Build test article list ----------
pdf_ids = sorted([p.stem for p in test_pdf_dir.glob("*.pdf")]) if test_pdf_dir.exists() else []
xml_ids = sorted([p.stem for p in test_xml_dir.glob("*.xml")]) if test_xml_dir.exists() else []
test_article_ids = sorted(list(dict.fromkeys(pdf_ids + xml_ids)))
print("Test articles found:", len(test_article_ids))

if len(test_article_ids) == 0:
    print("WARNING: Aucun fichier test trouvé. Vérifiez l'attachement du dataset.")
else:
    print("Exemple test ids (first 5):", test_article_ids[:5])

# ---------- 8) Process test articles OPTIMISÉ ----------
print("Processing test articles with multiprocessing...")

# Fonction simplifiée pour test (sans dataset_id)
def process_test_article(args):
    aid, xml_dir, pdf_dir = args
    return build_full_text(aid, xml_dir, pdf_dir)

test_args = [(aid, test_xml_dir, test_pdf_dir) for aid in test_article_ids]

test_texts_dict = {}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(tqdm(executor.map(process_test_article, test_args), total=len(test_args)))
    for aid, text in zip(test_article_ids, results):
        test_texts_dict[aid] = text

# Extraction des dataset IDs et création des paires
all_pairs = []
mapping = []

print("Extracting dataset IDs from test articles...")
for aid in tqdm(test_article_ids, desc="Extracting dataset IDs"):
    text = test_texts_dict[aid]
    cands = extract_dataset_ids(text)
    if cands:
        for ds in cands:
            pair_text = (text + " " + ds).strip()
            all_pairs.append(pair_text)
            mapping.append((aid, ds))

# Vectorisation batch pour toutes les paires
if all_pairs:
    print(f"Vectorizing {len(all_pairs)} test pairs...")
    Xw_test = word_vec.transform(all_pairs)
    Xc_test = char_vec.transform(all_pairs)
    X_test = hstack([Xw_test, Xc_test]).tocsr()
    
    # Prédiction avec le modèle unique
    print("Predicting on test pairs...")
    probs = model.predict(X_test, num_iteration=model.best_iteration)
    preds = probs.argmax(axis=1)
    labels = le.inverse_transform(preds)
    
    # Création de la soumission
    rows = []
    for i, ((aid, ds), label) in enumerate(zip(mapping, labels)):
        rows.append((i, aid, ds, label))
    
    submission = pd.DataFrame(rows, columns=['row_id','article_id','dataset_id','type'])
    submission = submission.drop_duplicates(subset=['article_id','dataset_id']).reset_index(drop=True)
    submission['row_id'] = np.arange(len(submission))
    
else:
    print("Aucune paire trouvée - création d'une soumission vide")
    submission = pd.DataFrame(columns=['row_id','article_id','dataset_id','type'])

# ---------- 9) Final submission ----------
print("Submission rows:", len(submission))
if len(submission) > 0:
    print("Submission sample:")
    print(submission.head().to_string(index=False))

# Sauvegarde
out = Path("/kaggle/working/submission.csv")
submission.to_csv(out, index=False)
print("Saved submission to:", out)

# Nettoyage mémoire
gc.collect()
print("Processing completed successfully!")
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

