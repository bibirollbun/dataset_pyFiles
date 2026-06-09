!pip install /kaggle/input/pymudfile/pymupdf-1.26.3-cp39-abi3-manylinux_2_28_x86_64.whl


!pip install --no-index /kaggle/input/pypdf2/pypdf2-3.0.1-py3-none-any.whl


# Basic Libaries
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from io import StringIO
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import openpyxl
import plotly.graph_objects as go
from matplotlib.sankey import Sankey
import re
from collections import Counter

# Dataset Libraries 
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
import json
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchvision
import re, unicodedata, json, sys, csv, pathlib
import fitz
from lxml import etree
from PyPDF2 import PdfReader
import xml.etree.ElementTree as ET
from typing import List, Tuple, Iterable, Dict

# Clustering Algorithms
# from sklearn.cluster import DBSCAN
# import hdbscan
# from sklearn.cluster import KMeans
# from sklearn.datasets import make_blobs
# from collections import Counter
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.cluster import AgglomerativeClustering
# from scipy.cluster.hierarchy import dendrogram, linkage
# from scipy.cluster.hierarchy import fcluster
# from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
# from sklearn.metrics import silhouette_score, davies_bouldin_score


# ML Models- Sklearn/Scikit Libraries
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# from xgboost import XGBRegressor
# import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import Pipeline
# from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingRegressor
# from scipy.stats import norm
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# !pip install hdbscan
from sklearn.ensemble import RandomForestClassifier
# import hdbscan
# from sklearn.linear_model import RidgeCV
# from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_log_error
# from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
# from sklearn.linear_model import Ridge
# from catboost import CatBoostRegressor
# from lightgbm import LGBMRegressor
# import itertools
# from sklearn.linear_model import LassoCV
# from scipy.stats import boxcox
# from scipy.special import inv_boxcox
# from sklearn.preprocessing import QuantileTransformer
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.feature_selection import mutual_info_regression
# from sklearn.ensemble import IsolationForest
# from scipy.spatial.distance import mahalanobis
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import jaccard_score
from difflib import SequenceMatcher
from sklearn.preprocessing import LabelEncoder
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report



INPUT_DIR = "/kaggle/input/make-data-count-finding-data-references/"

test_pdf = glob.glob(os.path.join(INPUT_DIR, "test/PDF", "*.pdf"))
test_xml = glob.glob(os.path.join(INPUT_DIR, "test/XML", "*.xml"))

train_pdf = glob.glob(os.path.join(INPUT_DIR, "train/PDF", "*.pdf"))
train_xml = glob.glob(os.path.join(INPUT_DIR, "train/XML", "*.xml"))

# Optional: sort lists for consistency
test_pdf.sort()
test_xml.sort()

print(f"ðŸ§  Test PDFs: {len(test_pdf)}")
print(f"ðŸ“„ Test XMLs: {len(test_xml)}")

print(f"ðŸ§  Train PDFs: {len(train_pdf)}")
print(f"ðŸ“„ Train XMLs: {len(train_xml)}")


%%writefile mdc_classifier.py

import re, unicodedata, json, sys, csv, pathlib, argparse
from collections import Counter
from typing import List, Tuple, Iterable, Dict, Set

# ------------------------------
# Normalization
# ------------------------------
def norm(txt: str) -> str:
    txt = unicodedata.normalize("NFKC", txt)
    txt = txt.lower()
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()

# ------------------------------
# Section splitting (plain text / PDFs)
# ------------------------------
SECTION_HEADINGS = [
    r"abstract", r"introduction", r"background", r"methods?",
    r"materials and methods", r"results?", r"discussion",
    r"data availability(?: statement)?", r"availability of data(?: and materials)?",
    r"supplementary materials?"
]
SEC_SPLIT = re.compile(
    r"(?:^|\n)\s*(?P<h>" + "|".join(fr"{h}" for h in SECTION_HEADINGS) + r")\s*[:\-]?\s*\n",
    re.I
)

def split_sections_text(txt: str) -> List[Tuple[str, str]]:
    t = norm(txt)
    matches = list(SEC_SPLIT.finditer(t))
    if not matches:
        return [("full", t)]
    sections: List[Tuple[str, str]] = []
    # preamble before first heading
    if matches[0].start() > 0:
        sections.append(("full", t[:matches[0].start()]))
    # each heading slice
    for i, m in enumerate(matches):
        h = m.group("h")
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(t)
        sections.append((h, t[start:end]))
    return sections

# ------------------------------
# Cues (Primary vs Secondary)
# ------------------------------
PRIMARY_VERBS = r"(collect(?:ed|ion|ing)?|measure(?:d|ment|s|ing)?|record(?:ed|ing|s)?|enroll(?:ed|ment|ing)?|recruit(?:ed|ment|ing)?|acquire(?:d|ment|ing)?|conduct(?:ed|ing)?|perform(?:ed|ance|ing)?|experiment(?:ed|al)?)"
PRIMARY_NOUNS = r"(cohort|participants?|subjects?|samples?|field\s+data|trial(?:s)?|observations?)"
PRIMARY_CLAIMS = r"(data\s+(?:generated|collected|produced)\s+(?:in|by)\s+(?:this|our)\s+(?:study|work)|we\s+(?:created|built)\s+(?:a|the)\s+dataset)"

SECONDARY_VERBS = r"(use(?:d|s|ing)?|download(?:ed|ing)?|obtain(?:ed|ing)?|sourc(?:ed|ing)?|derive(?:d|ing)?|reuse(?:d|ing)?|curate(?:d|ing)?|aggregate(?:d|ing)?)"
SECONDARY_PHRASES = r"(public(?:ly)?\s+available\s+dataset|archival\s+data|administrative\s+records?|secondary\s+analysis|retrospective\s+analysis|meta-?analysis|systematic\s+review|survey\s+of|we\s+review)"

REPO_NAMES = r"(zenodo|figshare|dryad|osf|pangaea|icpsr|mimic[-\s]?iv?|physionet|uk\s+biobank|uci\s+machine\s+learning|kaggle|world\s+bank|openalex|crossref|uniprot|genbank|ensembl|gwas\s+catalog|geo|sra|dbgap|gisaid|openneuro|pdb|pangaea)"

DOI_RAW_PAT = r"\b10\.\d{4,9}/\S+\b"
DATASET_NEARBY = r"(dataset|data\s*set|data\s*package|data\s*record|data\s*collection)"

re_primary_verbs     = re.compile(rf"\b{PRIMARY_VERBS}\b")
re_primary_nouns     = re.compile(rf"\b{PRIMARY_NOUNS}\b")
re_primary_claims    = re.compile(rf"{PRIMARY_CLAIMS}")
re_secondary_verbs   = re.compile(rf"\b{SECONDARY_VERBS}\b")
re_secondary_phrases = re.compile(rf"{SECONDARY_PHRASES}")
re_repo              = re.compile(rf"\b{REPO_NAMES}\b")
re_doi               = re.compile(DOI_RAW_PAT)
re_dataset_word      = re.compile(DATASET_NEARBY)

def count(pat, text): return len(pat.findall(text))
def any_(pat, text):   return bool(pat.search(text))

def score_block(text: str, context: str) -> Counter:
    c = Counter()
    # Primary cues
    c["primary"] += 2 * count(re_primary_verbs, text)
    c["primary"] += 2 * count(re_primary_nouns, text)
    c["primary"] += 4 * count(re_primary_claims, text)
    # Secondary cues
    c["secondary"] += 2 * count(re_secondary_verbs, text)
    c["secondary"] += 3 * count(re_secondary_phrases, text)
    c["secondary"] += 2 * count(re_repo, text)
    # DOI near 'dataset'
    if any_(re_doi, text) and any_(re_dataset_word, text):
        c["primary"] += 2
    # Section-aware tweaks
    if context == "Methods":
        c["primary"] += 4
    elif context == "Results":
        c["primary"] += 3
    elif context == "DAS":
        if any_(re_primary_claims, text) or any_(re_primary_verbs, text):
            c["primary"] += 3
        if any_(re_secondary_phrases, text) or any_(re_secondary_verbs, text):
            c["secondary"] += 3
    return c

def classify_from_sections(sections: List[Tuple[str, str]]):
    total = Counter()
    reasons = []
    for h, t in sections:
        ctx = "Other"
        hl = h.lower()
        if "method" in hl: ctx = "Methods"
        elif "result" in hl: ctx = "Results"
        elif "availability" in hl: ctx = "DAS"
        sc = score_block(norm(t), ctx)
        total.update(sc)
        if sc:
            reasons.append((h, dict(sc)))
    # tie-break
    full_text = " ".join(body for _, body in sections)
    if total["primary"] == total["secondary"]:
        if any_(re_secondary_phrases, full_text):
            total["secondary"] += 1
        else:
            total["primary"] += 1
    label = "Primary" if total["primary"] > total["secondary"] else "Secondary"
    return label, dict(total), reasons

# ------------------------------
# PDF / XML parsing
# ------------------------------
def parse_pdf_sections(path: str) -> List[Tuple[str, str]]:
    import fitz  # PyMuPDF
    text_pages = []
    with fitz.open(path) as doc:
        for page in doc:
            text_pages.append(page.get_text("text"))
    raw = "\n".join(text_pages)
    return split_sections_text(raw)

def parse_xml_sections(path: str) -> List[Tuple[str, str]]:
    from lxml import etree
    tree = etree.parse(path)
    root = tree.getroot()
    nsmap = root.nsmap.copy()
    def txt(node): return " ".join(node.itertext())
    sections: List[Tuple[str, str]] = []
    for ab in root.findall(".//abstract", namespaces=nsmap):
        sections.append(("abstract", txt(ab)))
    for sec in root.findall(".//sec", namespaces=nsmap):
        title = sec.find("./title", namespaces=nsmap)
        head = txt(title) if title is not None else "sec"
        body = txt(sec)
        sections.append((head, body))
    if not sections:
        sections = [("full", txt(root))]
    return [(h, norm(t)) for h, t in sections]

# ------------------------------
# Core classifier API
# ------------------------------
def classify_file(path: str) -> Dict[str, object]:
    p = pathlib.Path(path)
    if p.suffix.lower() == ".pdf":
        sections = parse_pdf_sections(str(p))
    elif p.suffix.lower() in {".xml", ".jats"}:
        sections = parse_xml_sections(str(p))
    else:
        sections = split_sections_text(pathlib.Path(path).read_text(encoding="utf-8", errors="ignore"))
    label, score, reasons = classify_from_sections(sections)
    return {
        "file": str(p),
        "label": label,
        "primary_score": score.get("primary", 0),
        "secondary_score": score.get("secondary", 0),
        "reasons": reasons[:12],
        "sections": sections,  # exposed for dataset extraction helpers
    }

# ------------------------------
# Dataset-ID extraction (filters to avoid article DOIs)
# ------------------------------
REFERENCE_HEADINGS = [
    r"references?", r"bibliograph(?:y|ies)", r"works\s+cited", r"citations?", r"literature\s+cited"
]
DATA_SECTIONS = [
    r"data\s+availability(?:\s+statement)?", r"availability\s+of\s+data(?:\s+and\s+materials)?",
    r"supplementary\s+materials?", r"methods?", r"materials\s+and\s+methods", r"results?"
]

DATASET_DOI_PREFIXES = (
    "10.5281",   # Zenodo
    "10.6084",   # Figshare
    "10.5061",   # Dryad
    "10.1594",   # PANGAEA
    "10.7910",   # Harvard Dataverse
    "10.17632",  # Mendeley Data
    "10.3886",   # ICPSR
    "10.18112",  # OpenNeuro
    "10.25504",  # OSF (Datacite)
    "10.6078",   # Dryad (legacy)
    "10.5067",   # NASA/ESDIS
    "10.5524",   # GSA/NGDC
    "10.5255",   # UK Data Service
)

GEO_PAT  = re.compile(r"\b(GSE\d{3,7}|GSM\d{3,7})\b", re.I)
SRA_PAT  = re.compile(r"\b(SRR\d{3,9}|SRP\d{3,9}|SRA\d{3,9}|ERP\d{3,9}|DRR\d{3,9}|PRJ[EDN][A-Z]\d+)\b", re.I)
PDB_PAT  = re.compile(r"\b([0-9A-Z]{4})\b")

DOI_URL_PAT = re.compile(r"https?://(?:dx\.)?doi\.org/(10\.\d{4,9}/\S+)\b", re.I)
DOI_PAT     = re.compile(r"\b(10\.\d{4,9}/\S+)\b", re.I)

DATA_NEAR   = re.compile(r"\b(data\s+(?:set|package|record|collection)|dataset|repository|archive|deposited|available\s+at|hosted\s+at)\b", re.I)
PDB_NEAR    = re.compile(r"\b(pdb|structure|protein)\b", re.I)

def is_reference_heading(h: str) -> bool:
    h = h.lower()
    return any(re.search(p, h, re.I) for p in REFERENCE_HEADINGS)

def is_data_heading(h: str) -> bool:
    h = h.lower()
    return any(re.search(p, h, re.I) for p in DATA_SECTIONS)

def likely_dataset_doi(doi_url: str) -> bool:
    for prefix in DATASET_DOI_PREFIXES:
        if doi_url.startswith(f"https://doi.org/{prefix}"):
            return True
    return False

def canonicalize_doi_strings(text: str) -> List[str]:
    out = []
    for m in DOI_URL_PAT.finditer(text):
        out.append(f"https://doi.org/{m.group(1)}")
    for m in DOI_PAT.finditer(text):
        out.append(f"https://doi.org/{m.group(1)}")
    return out

def extract_dataset_ids_from_sections(sections: List[Tuple[str, str]], article_id_maybe: str = "") -> Set[str]:
    ids: Set[str] = set()
    preferred = [(h, t) for (h, t) in sections if is_data_heading(h) and not is_reference_heading(h)]
    fallback  = [(h, t) for (h, t) in sections if not is_reference_heading(h) and (h, t) not in preferred]

    def scan(blocks: List[Tuple[str,str]]):
        local_ids: Set[str] = set()
        for h, t in blocks:
            # DOIs
            for doi_url in canonicalize_doi_strings(t):
                # skip the article's own DOI if present
                if article_id_maybe and article_id_maybe.replace("_","/",1) in doi_url:
                    continue
                if likely_dataset_doi(doi_url) or DATA_NEAR.search(t):
                    local_ids.add(doi_url)
            # Accessions
            for m in GEO_PAT.finditer(t): local_ids.add(m.group(1).upper())
            for m in SRA_PAT.finditer(t): local_ids.add(m.group(1).upper())
            # PDB with context
            for m in PDB_PAT.finditer(t):
                tok = m.group(1).upper()
                i = m.start()
                win = t[max(0, i-25): i+25]
                if PDB_NEAR.search(win):
                    local_ids.add(tok)
            # Named repos (no accession)
            if re.search(r"\buk\s*biobank\b", t, re.I): local_ids.add("UK Biobank")
            if re.search(r"\bmimic[-\s]?iv\b", t, re.I): local_ids.add("MIMIC-IV")
            if re.search(r"\bphysionet\b", t, re.I):     local_ids.add("PhysioNet")
        return local_ids

    ids = scan(preferred)
    if not ids:
        ids = scan(fallback)
    return ids

def extract_dataset_ids_from_file(path: str, article_id_maybe: str = "") -> Set[str]:
    p = pathlib.Path(path)
    if p.suffix.lower() in {".xml", ".jats"}:
        sections = parse_xml_sections(str(p))
    elif p.suffix.lower() == ".pdf":
        sections = parse_pdf_sections(str(p))
    else:
        sections = [("full", norm(p.read_text(encoding="utf-8", errors="ignore")))]
    return extract_dataset_ids_from_sections(sections, article_id_maybe=article_id_maybe)

def article_id_from_filename(fp: str) -> str:
    base = pathlib.Path(fp).stem
    return base.replace("_", "/", 1) if "_" in base else base

# ------------------------------
# Walk files
# ------------------------------
def walk_files(root: str) -> Iterable[str]:
    for p in pathlib.Path(root).rglob("*"):
        if p.suffix.lower() in {".pdf", ".xml", ".jats", ".txt"}:
            yield str(p)

# ------------------------------
# CLI modes
# ------------------------------
def run_predict(src: str, out_csv: str) -> None:
    paths = [src] if pathlib.Path(src).is_file() else list(walk_files(src))
    rows = []
    for fp in paths:
        try:
            res = classify_file(fp)
            rows.append({
                "file": res["file"],
                "label": res["label"],
                "primary_score": res["primary_score"],
                "secondary_score": res["secondary_score"],
                "reasons_json": json.dumps(res["reasons"], ensure_ascii=False),
            })
        except Exception as e:
            rows.append({"file": fp, "label": "ERROR", "primary_score": 0, "secondary_score": 0, "reasons_json": str(e)})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file","label","primary_score","secondary_score","reasons_json"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows to {out_csv}")

def run_submission(src_dir: str, out_csv: str) -> None:
    # group files by doc_id; prefer XML over PDF for extraction
    from itertools import groupby
    files = sorted(list(walk_files(src_dir)))
    def doc_id(fp: str) -> str:
        return pathlib.Path(fp).stem
    rows = []
    for did, group in groupby(sorted(files, key=doc_id), key=doc_id):
        group = list(group)
        # prefer XML/JATS if present
        preferred = [g for g in group if pathlib.Path(g).suffix.lower() in {".xml",".jats"}] or group
        fp = preferred[0]
        # classify to get article-level label
        cls = classify_file(fp)
        article_label = cls["label"]
        article_id = article_id_from_filename(fp)
        # extract dataset ids (XML preferred)
        ds_ids = extract_dataset_ids_from_file(fp, article_id_maybe=article_id)
        for ds in ds_ids:
            rows.append({"article_id": article_id, "dataset_id": ds, "type": article_label})
    # de-dup tuples
    import pandas as pd
    sub = pd.DataFrame(rows).drop_duplicates(["article_id","dataset_id","type"]).reset_index(drop=True)
    sub.insert(0, "row_id", range(len(sub)))
    sub.to_csv(out_csv, index=False)
    print(f"Submission shape: {sub.shape} -> {out_csv}")

def main():
    # Backward-compatible: if called with 2 positional args â†’ predict mode
    if len(sys.argv) == 3 and not sys.argv[1].startswith("-"):
        src, out_csv = sys.argv[1], sys.argv[2]
        return run_predict(src, out_csv)

    parser = argparse.ArgumentParser(description="MDC classifier + dataset extractor")
    parser.add_argument("--submission", nargs=2, metavar=("SRC_DIR","OUT_CSV"),
                        help="Build submission (row_id, article_id, dataset_id, type) from a directory")
    parser.add_argument("--predict", nargs=2, metavar=("SRC","OUT_CSV"),
                        help="Predict Primary/Secondary for a file or directory (default mode)")
    args, extras = parser.parse_known_args()

    if args.submission:
        src_dir, out_csv = args.submission
        return run_submission(src_dir, out_csv)
    if args.predict:
        src, out_csv = args.predict
        return run_predict(src, out_csv)

    # Fallback help
    print("Usage:")
    print("  python mdc_classifier.py <SRC> <OUT_CSV>              # predict mode (legacy)")
    print("  python mdc_classifier.py --predict <SRC> <OUT_CSV>     # predict mode")
    print("  python mdc_classifier.py --submission <DIR> <OUT_CSV>  # build submission")

if __name__ == "__main__":
    main()



!pip install -q pymupdf lxml


!python mdc_classifier.py /kaggle/input/make-data-count-finding-data-references/train /kaggle/working/train_preds.csv
!python mdc_classifier.py /kaggle/input/make-data-count-finding-data-references/test  /kaggle/working/test_preds.csv


!python mdc_classifier.py --submission /kaggle/input/make-data-count-finding-data-references/test /kaggle/working/submission.csv


test_pred_df  = pd.read_csv("/kaggle/working/test_preds.csv")


test_pred_df.head()


submission_df = pd.read_csv('/kaggle/working/submission.csv')


submission_df


# Show the breakdown to confirm
print(submission_df['type'].value_counts())


submission_df.to_csv("submission.csv", index=False)




