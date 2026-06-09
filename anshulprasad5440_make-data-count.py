# Cell 0 — Environment detection (PDF-only, no pip installs)
# - Detects if PyPDF2 (or pypdf) is available for text extraction from PDFs
# - Does not call pip or rely on network

import importlib

def try_import(name):
    try:
        mod = importlib.import_module(name)
        return mod, True
    except ImportError:
        return None, False

pypdf_mod, HAVE_PYPDF = try_import('pypdf')
pypdf2_mod, HAVE_PYPDF2 = try_import('PyPDF2')  # older naming

print("Environment summary (no network):")
print(f"  pypdf available:  {HAVE_PYPDF}")
print(f"  PyPDF2 available: {HAVE_PYPDF2}")

if not (HAVE_PYPDF or HAVE_PYPDF2):
    raise RuntimeError("No pure-Python PDF parser found (pypdf or PyPDF2). Please run where it's available.")
print("\nThis notebook will process PDFs only. Ensure PDFs are placed in /kaggle/input/.../train/ and test/ directories.")



# Cell 1 — Imports, CSV discovery, and PDF discovery
import os, glob, re
import pandas as pd
from tqdm.auto import tqdm

csv_candidates = glob.glob("/kaggle/input/**/train_labels.csv", recursive=True)
sample_candidates = glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True)
train_csv = csv_candidates[0] if csv_candidates else "/kaggle/working/train_labels.csv"
sample_csv = sample_candidates[0] if sample_candidates else "/kaggle/working/sample_submission.csv"
print("train_labels.csv:", train_csv)
print("sample_submission.csv:", sample_csv)

# Discover PDFs
train_pdfs = glob.glob("/kaggle/input/**/train/PDF/*.pdf", recursive=True)
# train_pdfs += glob.glob("/kaggle/input/**/*.pdf", recursive=True)
train_pdfs = list(dict.fromkeys(train_pdfs))
test_pdfs = glob.glob("/kaggle/input/**/test/PDF/*.pdf", recursive=True)
# test_pdfs += glob.glob("/kaggle/input/**/*.pdf", recursive=True)

print(f"Discovered PDF files -- train: {len(train_pdfs)}, test: {len(test_pdfs)}")



# Cell 2 — Load train_labels.csv and do a quick inspection
# - Loads CSV
# - Prints basic stats: number of rows, unique articles, label counts
# - Categorizes dataset_id into DOI-like vs accession patterns

train = pd.read_csv(train_csv)
print("train shape:", train.shape)
print("unique articles:", train['article_id'].nunique())
print("unique dataset ids:", train['dataset_id'].nunique())
print("\nLabel distribution:")
print(train['type'].value_counts(dropna=False))

# categorize dataset ids function
def is_doi(s):
    s = str(s)
    return bool(re.search(r'(https?://(dx\.)?doi\.org/|doi:|10\.\d{4,9}/)', s, flags=re.I))

def categorize_dataset_id(s):
    s = str(s).strip()
    if is_doi(s):
        return "DOI-like"
    patterns = {
        "GEO": r'GSE\d+',
        "PDB": r'\bPDB\s*\w+|\bpdb\s*\w+',
        "E-MEXP": r'E-MEXP-\d+',
        "E-MTAB": r'E-MTAB-\d+',
        "PRJ": r'PRJ[A-Z0-9_\-]*\d+',
        "CHEMBL": r'CHEMBL\d+',
        "OtherAcc": r'^[A-Za-z0-9\-\._]{4,30}$'
    }
    for k,p in patterns.items():
        if re.search(p, s, flags=re.I):
            return k
    return "Unknown"

train['dataset_cat'] = train['dataset_id'].apply(categorize_dataset_id)
print("\nDataset id category counts:")
print(train['dataset_cat'].value_counts())



# Cell 3 — Regexes and normalization helpers (fixed)
# - Compiled regex patterns for DOIs and common accessions (GEO, PDB, E-MTAB, CHEMBL, PRJ)
# - Normalization helpers to canonicalize DOIs and accessions

import re

# DOI-ish regex (we'll normalize later using normalize_doi)
doi_re = re.compile(
    r'(?:https?://(?:dx\.)?doi\.org/|doi:)?(10\.\d{4,9}/[A-Za-z0-9\-\._;()/:]+)',
    re.IGNORECASE
)

geo_re = re.compile(r'\b(GSE\d{1,9})\b', re.IGNORECASE)
pdb_re = re.compile(r'\b(PDB\s*[0-9A-Za-z]{1,6}|pdb\s*[0-9A-Za-z]{1,6})\b', re.IGNORECASE)
emexp_re = re.compile(r'\b(E-MEXP-\d+)\b', re.IGNORECASE)
emtab_re = re.compile(r'\b(E-MTAB-\d+)\b', re.IGNORECASE)
chembl_re = re.compile(r'\b(CHEMBL\d+)\b', re.IGNORECASE)
prj_re = re.compile(r'\b(PRJ[A-Z0-9\-_]*\d+)\b', re.IGNORECASE)

# fallback (used carefully)
generic_acc_re = re.compile(r'\b([A-Za-z0-9\-\._]{4,30})\b')

def strip_trailing_punct(s: str) -> str:
    """Strip trailing punctuation and closing brackets/parentheses from a string."""
    if s is None:
        return s
    return re.sub(r'[\]\)\.,;:]+$', '', s.strip())

def normalize_doi(raw: str) -> str:
    """
    Normalize DOI representations to the canonical https://doi.org/<prefix>/<suffix>
    If no DOI is found, returns the cleaned input string.
    """
    raw = str(raw).strip()
    raw_clean = strip_trailing_punct(raw)
    # try to find 10.x/... pattern in the cleaned string
    m = re.search(r'(10\.\d{4,9}/[^\s\]\)\.,;]+)', raw_clean)
    if m:
        return "https://doi.org/" + m.group(1)
    # try the full URL pattern in the original raw (case-insensitive)
    m2 = re.search(r'https?://(?:dx\.)?doi\.org/(10\.\d{4,9}/[^\s\]\)\.,;]+)', raw, re.IGNORECASE)
    if m2:
        return "https://doi.org/" + m2.group(1)
    # fallback: return cleaned raw
    return raw_clean

def normalize_accession(raw: str) -> str:
    raw = str(raw).strip()
    if re.search(geo_re, raw):
        return re.search(geo_re, raw).group(1).upper()
    if re.search(emtab_re, raw):
        return re.search(emtab_re, raw).group(1).upper()
    if re.search(emexp_re, raw):
        return re.search(emexp_re, raw).group(1).upper()
    if re.search(chembl_re, raw):
        return re.search(chembl_re, raw).group(1).upper()
    if re.search(prj_re, raw):
        return re.search(prj_re, raw).group(1).upper()
    if re.search(pdb_re, raw):
        # keep a space between PDB and id, canonical uppercase
        return re.search(pdb_re, raw).group(1).upper().replace('  ', ' ')
    return raw



# Cell 4 — PDF text extraction using pure-Python parser (pypdf or PyPDF2)
import os

def extract_text_from_pdf(pdf_path):
    from io import StringIO
    text_parts = []
    try:
        if HAVE_PYPDF:
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        elif HAVE_PYPDF2:
            from PyPDF2 import PdfReader
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        else:
            raise RuntimeError("No PDF parser available.")
    except Exception as e:
        print("PDF extraction failed for", pdf_path, ":", e)
    return "\n".join(text_parts)



# Cell 5 — Candidate extractor
# - extract_candidates(text): returns list of candidate dicts: {kind, raw, norm, context, start, end}
# - keeps a context window of +/- 120 chars (configurable)

def extract_candidates(text, context_window=120):
    text = str(text)
    candidates = []
    # DOIs
    for m in doi_re.finditer(text):
        raw = m.group(0)
        norm = normalize_doi(raw)
        start, end = m.start(), m.end()
        ctx = text[max(0,start-context_window):min(len(text), end+context_window)]
        candidates.append({'kind':'DOI', 'raw':raw, 'norm':norm, 'context':ctx, 'start':start, 'end':end})
    # accessions (multiple patterns)
    for pat, kind in [(geo_re,'GEO'), (pdb_re,'PDB'), (emtab_re,'EMTAB'), (emexp_re,'EMEXP'),
                      (chembl_re,'CHEMBL'), (prj_re,'PRJ')]:
        for m in pat.finditer(text):
            raw = m.group(1) if m.groups() else m.group(0)
            norm = normalize_accession(raw)
            start,end=m.start(),m.end()
            ctx = text[max(0,start-context_window):min(len(text), end+context_window)]
            candidates.append({'kind':kind, 'raw':raw, 'norm':norm, 'context':ctx, 'start':start, 'end':end})
    # deduplicate by (kind,norm)
    seen=set()
    uniq=[]
    for c in candidates:
        key=(c['kind'], c['norm'])
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq

# quick unit test on a sample string
samp = "Data available at doi:10.5061/dryad.6m3n9 and GEO GSE12345. Also see https://doi.org/10.1000/xyz123."
print(extract_candidates(samp))



# Cell 6 — Crawl train PDFs and extract candidate mentions to CSV
import os, glob, re, pandas as pd
from tqdm.auto import tqdm

if 'extract_candidates' not in globals():
    raise RuntimeError("extract_candidates() not defined yet.")

if 'extract_text_from_pdf' not in globals():
    raise RuntimeError("extract_text_from_pdf() not defined yet.")

if not train_pdfs:
    print("No train PDFs found. Ensure they're in expected directories.")
else:
    out_candidates = []
    max_files = 200
    for pdf_path in tqdm(train_pdfs[:max_files]):
        text = extract_text_from_pdf(pdf_path)
        if not text or len(text) < 50:
            continue
        cands = extract_candidates(text)
        if not cands:
            continue
        # Infer article_id via DOI in text
        m = re.search(r'(10\.\d{4,9}/[A-Za-z0-9\-\._;()/:%]+)', text)
        art_id = f"https://doi.org/{m.group(1)}" if m else os.path.basename(pdf_path)
        for c in cands:
            out_candidates.append({
                'file': pdf_path,
                'article_id': art_id,
                'kind': c.get('kind'),
                'raw': c.get('raw'),
                'norm': c.get('norm'),
                'context': c.get('context')
            })
    if out_candidates:
        cand_df = pd.DataFrame(out_candidates)
        cand_df.to_csv("/kaggle/working/candidates.csv", index=False)
        print("Saved candidates.csv with rows:", len(cand_df))
    else:
        print("No candidates extracted.")



# Cell 7 — Build training examples by linking extracted candidates to train labels
# - Loads candidates.csv if exists
# - Joins candidates with train labels by normalized dataset_id equivalence (we normalize both)
# - Creates examples with label: Primary / Secondary / negative (None)
# - Saves a training CSV with columns: article_id, dataset_id, context, label

import os

cand_path = "/kaggle/working/candidates.csv"
if not os.path.exists(cand_path):
    print("No candidates.csv found. Cannot build training examples. Run Cell 6 with actual XML/PDFs present.")
else:
    cand_df = pd.read_csv(cand_path)
    # normalize train label dataset ids for matching
    def norm_label_id(s):
        s = str(s)
        if is_doi(s):
            return normalize_doi(s)
        return normalize_accession(s)
    train['norm_dataset_id'] = train['dataset_id'].apply(norm_label_id)
    cand_df['norm_dataset_id'] = cand_df['norm'].apply(lambda x: norm_label_id(x))
    # left join candidates to train labels on article_id and norm_dataset_id
    # Note: article_id matching can be tricky; here we try best-effort: exact match OR article_id file basename if label appears in file path
    merged = pd.merge(cand_df, train[['article_id','norm_dataset_id','type']], left_on=['article_id','norm_dataset_id'], right_on=['article_id','norm_dataset_id'], how='left')
    # if merged type is NA but candidate norm matches some dataset in same file with same dataset_id, we still keep candidate as unlabeled
    merged['label'] = merged['type'].where(merged['type'].isin(['Primary','Secondary']), None)
    # create simple binary label for classifier: Primary vs Secondary (drop None for supervised)
    train_examples = merged[['article_id','norm_dataset_id','context','label']].rename(columns={'norm_dataset_id':'dataset_id'})
    train_examples.to_csv("/kaggle/working/train_examples.csv", index=False)
    print("Built train_examples.csv with rows:", len(train_examples))
    print("Labeled examples counts:")
    print(train_examples['label'].value_counts(dropna=False))



# Cell 8 — Augment labeled examples by searching train PDFs for dataset IDs (increase labeled contexts)
# - Builds many ID string variants (doi prefix, underscores->slashes)
# - Searches each PDF's full text for dataset IDs belonging to that article
# - Saves /kaggle/working/train_examples_augmented.csv with columns: article_id, dataset_id, context, label
# - Requires: train_labels.csv present and extract_text_from_pdf() defined; if you only have XML, swap to extract_text_from_xml()

import os, re, glob
import pandas as pd
from tqdm.auto import tqdm

# Config
context_window = 120   # chars on each side for context snippet
max_files = None       # set None to process all discovered train_pdfs; else set small integer for debugging

# load train labels
train_labels_path = "/kaggle/input/**/train_labels.csv"
cands = glob.glob(train_labels_path, recursive=True)
if not cands:
    # fallback if earlier cell discovered train_csv variable
    if 'train_csv' in globals() and os.path.exists(train_csv):
        train_labels_file = train_csv
    else:
        raise RuntimeError("train_labels.csv not found. Put it in /kaggle/input/ or set train_csv variable.")
else:
    train_labels_file = cands[0]

train = pd.read_csv(train_labels_file, dtype=str).fillna('')
print("Loaded train labels:", train.shape)

# helper: produce normalized label ID and variants for searching in raw text
def gen_dataset_variants(ds):
    ds = str(ds).strip()
    variants = set()
    variants.add(ds)
    # if DOI-like (detect 10.x pattern), produce 'https://doi.org/..', 'doi:..', plain
    if re.search(r'10\.\d{4,9}/', ds):
        # remove https prefix if present
        m = re.search(r'(10\.\d{4,9}/[^\s\]\)\.,;]+)', ds)
        if m:
            core = m.group(1)
            variants.add(core)
            variants.add("https://doi.org/" + core)
            variants.add("http://dx.doi.org/" + core)
            variants.add("doi:" + core)
    else:
        # accession-like variants (case-insensitive)
        variants.add(ds.upper())
        variants.add(ds.lower())
        variants.add(ds.replace(' ',''))
        variants.add(ds.replace('-',''))
    return sorted(list(variants), key=lambda s:-len(s))

# helper: produce article id variants for matching PDF text
def gen_article_variants(aid):
    aid = str(aid).strip()
    variants = set()
    variants.add(aid)
    if '_' in aid and '/' not in aid:
        # e.g. 10.1002_cssc.202201821 -> 10.1002/cssc.202201821
        variants.add(aid.replace('_','/'))
        variants.add("https://doi.org/" + aid.replace('_','/'))
    if '/' in aid:
        variants.add(aid.replace('/','_'))
        variants.add("https://doi.org/" + aid)
    if not aid.startswith('http'):
        variants.add("https://doi.org/" + aid)
    return sorted(list(variants), key=lambda s:-len(s))

# create a mapping of article_id -> list of (dataset_variant, original_dataset_id, type)
article_to_dsets = {}
for _, r in train.iterrows():
    art = r['article_id']
    ds = r['dataset_id']
    dtype = r['type']
    if not ds:
        continue
    article_to_dsets.setdefault(art, []).append((ds, dtype))

print("Articles in train labels:", len(article_to_dsets))

# get list of train PDF files (use train_pdfs discovered earlier, else search)
if 'train_pdfs' in globals() and train_pdfs:
    pdf_files = train_pdfs
else:
    pdf_files = glob.glob("/kaggle/input/**/train/*.pdf", recursive=True) + glob.glob("/kaggle/input/**/*.pdf", recursive=True)
    pdf_files = sorted(list(dict.fromkeys(pdf_files)))

print("Discovered PDFs to scan:", len(pdf_files))
if max_files:
    pdf_files = pdf_files[:max_files]

# Make an index of article variants -> canonical article_id for quick lookup
variant_to_article = {}
for art in article_to_dsets:
    for v in gen_article_variants(art):
        variant_to_article[v] = art

# Now scan PDFs
aug_examples = []
not_found_counts = 0

for pdf_path in tqdm(pdf_files):
    # extract full text (expects extract_text_from_pdf to be defined)
    if 'extract_text_from_pdf' not in globals():
        raise RuntimeError("extract_text_from_pdf() not found. Define it before running this cell.")
    text = extract_text_from_pdf(pdf_path)
    if not text or len(text) < 50:
        continue
    text_low = text.lower()

    # try to detect which article this PDF corresponds to by trying to match any article variant in text
    matched_article = None
    for var, canonical in variant_to_article.items():
        if var.lower() in text_low:
            matched_article = canonical
            break

    # If matched_article found: only search dataset ids that belong to this article
    candidate_pairs = []
    if matched_article:
        dlist = article_to_dsets.get(matched_article, [])
        for orig_ds, dtype in dlist:
            for ds_var in gen_dataset_variants(orig_ds):
                if ds_var.lower() in text_low:
                    # find all occurrences and extract contexts
                    for m in re.finditer(re.escape(ds_var), text, flags=re.IGNORECASE):
                        start, end = m.start(), m.end()
                        ctx = text[max(0,start-context_window):min(len(text), end+context_window)]
                        aug_examples.append({
                            'article_id': matched_article,
                            'dataset_id': orig_ds,
                            'context': ctx,
                            'label': dtype
                        })
                    break  # if one variant matched, skip other variants for same orig_ds
    else:
        # fallback: search for any dataset ids from train labels across full text (more expensive)
        # We'll only do a lightweight pass: sample small subset of train dsets per PDF to avoid huge loops
        # Build a small candidate list of dataset ids (unique) and search for them
        all_dsets = train['dataset_id'].unique().tolist()
        # Option: use only longer dset strings first to avoid accidental substring hits
        all_dsets_sorted = sorted(all_dsets, key=lambda s: -len(str(s)))
        hits = 0
        for orig_ds in all_dsets_sorted[:2000]:   # limit search to first 2000 to keep runtime reasonable
            if not orig_ds:
                continue
            for ds_var in gen_dataset_variants(orig_ds):
                if ds_var.lower() in text_low:
                    # find occurrences
                    for m in re.finditer(re.escape(ds_var), text, flags=re.IGNORECASE):
                        start, end = m.start(), m.end()
                        ctx = text[max(0,start-context_window):min(len(text), end+context_window)]
                        # we don't reliably know which article row to attach to; attach multiple possible rows
                        # get all train rows with that dataset_id
                        rows = train[train['dataset_id'] == orig_ds]
                        for _, rr in rows.iterrows():
                            aug_examples.append({
                                'article_id': rr['article_id'],
                                'dataset_id': orig_ds,
                                'context': ctx,
                                'label': rr['type']
                            })
                    hits += 1
                    break
            if hits >= 50:  # avoid too many matches per PDF
                break

# Summarize and save
aug_df = pd.DataFrame(aug_examples)
print("Augmented examples found:", len(aug_df))
if len(aug_df) > 0:
    # drop exact duplicates
    aug_df = aug_df.drop_duplicates(subset=['article_id','dataset_id','context','label'])
    out_path = "/kaggle/working/train_examples_augmented.csv"
    aug_df.to_csv(out_path, index=False)
    print("Saved augmented examples to:", out_path)
else:
    print("No augmented examples found. Consider increasing search scope or verifying that PDFs contain the DOIs / ids present in train_labels.")



# Preview augmented training examples
import os, pandas as pd

aug_path = "/kaggle/working/train_examples_augmented.csv"
if os.path.exists(aug_path):
    df = pd.read_csv(aug_path)
    print("Augmented file found:", aug_path)
    print("Rows:", len(df))
    print("Label distribution:")
    print(df['label'].value_counts(dropna=False))
    display(df.sample(min(10, len(df))).reset_index(drop=True))
else:
    print("No augmented file found at", aug_path)
    print("If you didn't run the augmentation cell, run it first (the one that scans PDFs for dataset ids).")



# Cell 9 (replacement) — Feature extraction and baseline training (TF-IDF + logistic regression)
# - Auto-loads augmented examples if present
# - Filters only Primary/Secondary labels
# - Uses GroupKFold by article_id, prints CV metrics, saves model to /kaggle/working/baseline_model.joblib

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from scipy.sparse import hstack
import joblib

# pick dataset
aug_path = "/kaggle/working/train_examples_augmented.csv"
base_path = "/kaggle/working/train_examples.csv"
if os.path.exists(aug_path):
    df = pd.read_csv(aug_path)
    print("Loaded augmented examples:", aug_path)
elif os.path.exists(base_path):
    df = pd.read_csv(base_path)
    print("Loaded base train_examples:", base_path)
else:
    raise RuntimeError("No train_examples file found. Run candidate extraction/augmentation cells first.")

# Make sure expected columns exist
expected_cols = {'article_id','dataset_id','context','label'}
if not expected_cols.issubset(set(df.columns)):
    raise RuntimeError(f"Expected columns {expected_cols} in training CSV. Found: {df.columns.tolist()}")

# Keep only clean labeled examples
df = df[df['label'].isin(['Primary','Secondary'])].copy()
print("Labeled examples count (Primary+Secondary):", len(df))
print(df['label'].value_counts())

if len(df) < 30:
    print("Warning: too few labeled examples (<30). Training will likely be poor. Consider augmenting more PDFs or using heuristic/pseudo labels.")
else:
    # prepare labels
    df['y'] = df['label'].map({'Primary':1,'Secondary':0})

    # cue words (feel free to expand)
    cues = ['available','deposited','supplement','supplementary','we deposited','we uploaded',
            'publicly available','obtained from','downloaded from','previously published','generated','raw data','data are available']

    def cue_features(text):
        t = str(text).lower()
        return [1 if cue in t else 0 for cue in cues]

    cue_matrix = np.vstack(df['context'].apply(cue_features).values)

    # TF-IDF on context
    tf = TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words='english')
    X_tfidf = tf.fit_transform(df['context'].astype(str).values)

    # combine sparse + dense
    X = hstack([X_tfidf, cue_matrix])
    y = df['y'].values
    groups = df['article_id'].values

    # model
    clf = LogisticRegression(max_iter=2000, class_weight='balanced')

    # cross-validated predictions (GroupKFold)
    gkf = GroupKFold(n_splits=min(5, max(2, len(df['article_id'].unique()))))
    print("Running GroupKFold CV with", gkf.get_n_splits(groups=groups), "splits ...")
    oof = cross_val_predict(clf, X, y, cv=gkf.split(X, y, groups), method='predict')
    print("Classification report (OOF):")
    print(classification_report(y, oof, target_names=['Secondary','Primary']))
    print("F1 (Primary class):", f1_score(y, oof, pos_label=1))

    # fit final model on all labeled and save
    clf.fit(X, y)
    model_bundle = {'clf': clf, 'tf': tf, 'cues': cues}
    joblib.dump(model_bundle, "/kaggle/working/baseline_model.joblib")
    print("Saved baseline model to /kaggle/working/baseline_model.joblib")



# Cell 10 — Run extraction on test PDFs and build submission using trained model or fallback
import os, csv, pandas as pd
from scipy.sparse import hstack
import joblib
from tqdm.auto import tqdm

if not test_pdfs:
    print("No test PDFs found.")
else:
    model_path = "/kaggle/working/baseline_model.joblib"
    model_exists = os.path.exists(model_path)
    if model_exists:
        d = joblib.load(model_path)
        clf, tf, cues = d['clf'], d['tf'], d['cues']
    else:
        print("No model found — using heuristic fallback.")
        clf = tf = None
        cues = ['deposited','available','publicly available','we deposited','obtained from','downloaded from','supplementary','raw data']

    preds = []
    for pdf_path in tqdm(test_pdfs):
        text = extract_text_from_pdf(pdf_path)
        if not text or len(text) < 50:
            continue
        cands = extract_candidates(text)
        m = re.search(r'(10\.\d{4,9}/[A-Za-z0-9\-\._;()/:%]+)', text)
        article_id = f"https://doi.org/{m.group(1)}" if m else os.path.basename(pdf_path)
        if not cands:
            continue
        contexts = [c['context'] for c in cands]
        if model_exists:
            X_tfidf = tf.transform(contexts)
            cue_matrix = [[1 if kw in ctx.lower() else 0 for kw in cues] for ctx in contexts]
            from numpy import array
            X = hstack([X_tfidf, array(cue_matrix)])
            probs = clf.predict_proba(X)[:,1]
            for c, p in zip(cands, probs):
                label = 'Primary' if p >= 0.5 else 'Secondary'
                preds.append((article_id, c['norm'], label, p))
        else:
            for c in cands:
                label = 'Primary' if any(kw in c['context'].lower() for kw in cues) else 'Secondary'
                preds.append((article_id, c['norm'], label, 0.0))

    if preds:
        dfp = pd.DataFrame(preds, columns=['article_id','dataset_id','type','prob'])
        dfp = dfp.sort_values('prob', ascending=False).drop_duplicates(['article_id','dataset_id'], keep='first')
        rows = [[i, *r] for i, r in enumerate(dfp[['article_id','dataset_id','type']].values.tolist())]
        with open("/kaggle/working/submission.csv", "w", newline='', encoding='utf8') as f:
            w = csv.writer(f)
            w.writerow(['row_id','article_id','dataset_id','type'])
            w.writerows(rows)
        print("Saved submission.csv:", len(rows), "rows")
    else:
        print("No predictions generated.")



# Cell 11 — Save key helper functions to a small module and print next-steps checklist
# - Writes /kaggle/working/data_citation_utils.py containing normalization and extraction helpers
# - Prints a checklist of recommended improvements and tuning

module_code = r'''
import re
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text

doi_re = re.compile(r'(https?://(?:dx\.)?doi\.org/)?(doi:)?(10\.\d{4,9}/[A-Za-z0-9\-\._;()/:%]+)', re.IGNORECASE)
geo_re = re.compile(r'\b(GSE\d{1,9})\b', re.IGNORECASE)
pdb_re = re.compile(r'\b(PDB\s*[0-9A-Za-z]{1,6}|pdb\s*[0-9A-Za-z]{1,6})\b', re.IGNORECASE)
emexp_re = re.compile(r'\b(E-MEXP-\d+)\b', re.IGNORECASE)
emtab_re = re.compile(r'\b(E-MTAB-\d+)\b', re.IGNORECASE)
chembl_re = re.compile(r'\b(CHEMBL\d+)\b', re.IGNORECASE)
prj_re = re.compile(r'\b(PRJ[A-Z0-9\-_]*\d+)\b', re.IGNORECASE)

def normalize_doi(raw: str) -> str:
    raw = str(raw).strip()
    m = re.search(r'(10\.\d{4,9}/[A-Za-z0-9\-\._;()/:%]+)', raw)
    if m:
        return "https://doi.org/" + m.group(1).rstrip('.,;:)')
    m2 = re.search(r'https?://(?:dx\.)?doi\.org/(10\.\d{4,9}/[A-Za-z0-9\-\._;()/:%]+)', raw, re.I)
    if m2:
        return "https://doi.org/" + m2.group(1).rstrip('.,;:)')
    return raw

def normalize_accession(raw: str) -> str:
    raw = str(raw).strip()
    if re.search(geo_re, raw):
        return re.search(geo_re, raw).group(1).upper()
    if re.search(emtab_re, raw):
        return re.search(emtab_re, raw).group(1).upper()
    if re.search(emexp_re, raw):
        return re.search(emexp_re, raw).group(1).upper()
    if re.search(chembl_re, raw):
        return re.search(chembl_re, raw).group(1).upper()
    if re.search(prj_re, raw):
        return re.search(prj_re, raw).group(1).upper()
    if re.search(pdb_re, raw):
        return re.search(pdb_re, raw).group(1).upper().replace(' ',' ')
    return raw

def extract_text_from_xml(xml_path):
    try:
        with open(xml_path, 'r', encoding='utf8', errors='ignore') as f:
            raw = f.read()
        soup = BeautifulSoup(raw, 'lxml')
        parts = []
        for tag in soup.find_all(['title','abstract','body','sec','p','fig','supplementary-material','sub-section']):
            txt = tag.get_text(separator=' ', strip=True)
            if txt:
                parts.append(txt)
        return '\n'.join(parts)
    except Exception:
        try:
            return open(xml_path, 'r', encoding='utf8', errors='ignore').read()
        except:
            return ""

def extract_text_from_pdf(pdf_path):
    try:
        text = extract_text(pdf_path)
        return text
    except:
        return ""
'''

with open("/kaggle/working/data_citation_utils.py", "w") as f:
    f.write(module_code)
print("Wrote /kaggle/working/data_citation_utils.py")

print("\nNext steps and improvements checklist:")
print("- Increase labeled training examples by extracting from all train XML/PDF files.")
print("- Improve candidate matching between article_id in labels and extracted article identifiers.")
print("- Use SciBERT/BioBERT sentence classifier (fine-tune) for better Primary/Secondary discrimination.")
print("- Add heuristics to combine reference-list DOIs with in-text mentions when the in-text mention lacks a DOI.")
print("- Tune probability threshold for submission to balance precision (avoid false positives) and recall.")


