# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


%%time
!uv pip install -q --system --no-index --find-links='/kaggle/input/lastest-mdc-whls/whls' 'pymupdf'


#!pip install --upgrade pymupdf
!pip install pdfplumber


%%time
try:
    import pdfplumber
    print("pdfplumber loaded successfully.")
except ImportError:
    raise ImportError("pdfplumber not installed. Enable internet and install first.")   


%%time
import re

def extract_candidates(text, window_size=100):
    """
    Extract candidate data references from text.
    Returns: List of tuples (matched_text, context, raw_id)
    """
    candidates = []
    
    # 1. Pattern for DOIs (with or without https://doi.org/)
    doi_pattern = r'\b(10\.\d+/[\w.\-_]+)'
    for match in re.finditer(doi_pattern, text):
        span = match.group(0)
        context = text[max(0, match.start() - window_size): match.end() + window_size]
        candidates.append((span, context, span))
    
    # 2. Patterns for common accession IDs
    patterns = {
        'GEO': r'\b(GSE|GSM|GPL|GDS)\d+\b',                    # Gene Expression Omnibus
        'SRA': r'\b(SRX|SRR|SRP|SRX)\d+\b',                    # Sequence Read Archive
        'ENA': r'\b(ERX|ERR|ERP)\d+\b',                        # European Nucleotide Archive
        'PDB': r'\b[PQ][\d\w]{3}[A-Za-z]\b',                   # Protein Data Bank (e.g., 1Y2T, P00750)
        'ArrayExpress': r'\b(E-M\w{4}-\d+)\b',                 # ArrayExpress
        'ChEMBL': r'\bCHEMBL\d+\b',                            # ChEMBL
        'Figshare': r'\b(figshare:\s*\d+|10\.6084/m9\.figshare\.\d+)\b',  # Figshare
    }
    
    for db, pattern in patterns.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            span = match.group(0)
            context = text[max(0, match.start() - window_size): match.end() + window_size]
            candidates.append((span, context, span))
    
    # 3. Optional: Catch informal phrases (e.g., "publicly available data")
    # These won't yield a dataset_id, but you could link them later with coreference
    # For now, we focus on extractable IDs
    
    return candidates   


%%time
def normalize_doi(raw_id):
    """
    Normalize a DOI string to full format: https://doi.org/10.xxxx/...
    Returns None if not a DOI.
    """
    # Extract DOI pattern
    match = re.search(r'10\.\d+/[\w.\-_]+', str(raw_id))
    if match:
        doi = match.group(0)
        return f"https://doi.org/{doi}"
    return None   


%%time
def classify_context(context):
    """
    Simple rule-based classifier for Primary vs Secondary.
    """
    context_lower = context.lower()
    
    # Keywords suggesting PRIMARY (data generated in this study)
    primary_keywords = [
        'generated', 'produced', 'collected', 'this study', 
        'our study', 'measured', 'sequenced', 'determined',
        'created', 'obtained from scratch', 'de novo'
    ]
    
    # Keywords suggesting SECONDARY (data reused)
    secondary_keywords = [
        'downloaded', 'obtained from', 'retrieved', 'publicly available',
        'from the', 'accession', 'database', 'repository', 'archive',
        'taken from', 'sourced from', 'reused', 'existing', 'literature'
    ]
    
    primary_score = sum(1 for word in primary_keywords if word in context_lower)
    secondary_score = sum(1 for word in secondary_keywords if word in context_lower)
    
    if secondary_score > primary_score:
        return "Secondary"
    elif primary_score > 0:
        return "Primary"
    else:
        # Default fallback: if no strong signal, assume Secondary (more common)
        return "Secondary"   


%%time
import pandas as pd

# Load labels
train_labels = pd.read_csv('/kaggle/input/make-data-count-finding-data-references/train_labels.csv')

print("Shape:", train_labels.shape)
print("\nFirst few rows:")
print(train_labels.head())

print("\nColumn info:")
print(train_labels.info())   


%%time
import os
from bs4 import BeautifulSoup

# Pick a sample XML
sample_xml_path = '/kaggle/input/make-data-count-finding-data-references/train/XML/10.1186_s13024-018-0254-8.xml'

with open(sample_xml_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Parse with BeautifulSoup
soup = BeautifulSoup(content, 'xml')  # Use 'xml' parser for clean parsing

# Print root tag to confirm structure
print("Root tag:", soup.find().name)

# Try to find potential data reference sections
data_mentions = soup.find_all(string=re.compile(r'data.*available', re.I))
print("\nPossible data mention snippets:")
for dm in data_mentions[:5]:
    print(f" â†’ {dm.strip()}")


%%time
import pandas as pd

# Load and display train_labels.csv
train_labels = pd.read_csv('/kaggle/input/make-data-count-finding-data-references/train_labels.csv')
print("Columns:", train_labels.columns.tolist())
print("\nFirst 5 rows:")
print(train_labels.head()) 


%%time
import seaborn as sns
import matplotlib.pyplot as plt

type_counts = train_labels['type'].value_counts()

plt.figure(figsize=(10, 6))
sns.barplot(x=type_counts.values, y=type_counts.index, orient='h', palette='coolwarm')
plt.title('Frequency of Data Reference Types')
plt.xlabel('Count')
plt.ylabel('Relationship Type')
for i, v in enumerate(type_counts.values):
    plt.text(v + 1, i, str(v), color='black', va='center')
plt.tight_layout()
plt.show()

print("\nType distribution:")
print(type_counts) 


%%time
def get_repository(doi):
    if pd.isna(doi):
        return 'Unknown'
    doi = str(doi).lower()
    if 'dryad' in doi or '10.5061/dryad' in doi:
        return 'Dryad'
    elif 'figshare' in doi or '10.6084/m9.figshare' in doi:
        return 'Figshare'
    elif 'zenodo' in doi or '10.5281/zenodo' in doi:
        return 'Zenodo'
    elif '10.18112/openneuro' in doi:
        return 'OpenNeuro'
    elif '10.7910/dvn' in doi or 'dataverse' in doi:
        return 'Dataverse'
    elif '10.25346/s6' in doi:  # Often associated with Dryad (legacy)
        return 'Dryad'
    else:
        # Try to classify based on known DOI prefixes
        if doi.startswith('10.'):
            prefix = doi.split('/')[0]
            if prefix == '10.5281':
                return 'Zenodo'
            elif prefix == '10.6084':
                return 'Figshare'
            elif prefix == '10.5061':
                return 'Dryad'
        return 'Other'

# Apply function
train_labels['repository'] = train_labels['dataset_id'].apply(get_repository)

# Plot top repositories
repo_counts = train_labels['repository'].value_counts()

plt.figure(figsize=(10, 6))
sns.barplot(x=repo_counts.values, y=repo_counts.index, orient='h', palette='magma')
plt.title('Top Repositories Referenced in Papers')
plt.xlabel('Number of Mentions')
plt.ylabel('Repository')
for i, v in enumerate(repo_counts.values):
    plt.text(v + 0.5, i, str(v), color='black', va='center')
plt.tight_layout()
plt.show()

print("\nRepository distribution:")
print(repo_counts)   


%%time
import os
import re
import pandas as pd

# Define input directory
input_dir = "/kaggle/input/make-data-count-finding-data-references"
test_xml_dir = os.path.join(input_dir, "test", "XML")
test_pdf_dir = os.path.join(input_dir, "test", "PDF")

# -----------------------------
# ğŸ”§ FUNCTIONS (Defined Once)
# -----------------------------

def extract_candidates(text, window_size=100):
    candidates = []
    # DOI pattern
    doi_pattern = r'\b(10\.\d+/[\w.\-_]+)'
    for match in re.finditer(doi_pattern, text):
        span = match.group(0)
        context = text[max(0, match.start() - window_size): match.end() + window_size]
        candidates.append((span, context, span))
    
    # Accession patterns
    patterns = {
        'GEO': r'\b(GSE|GSM|GPL|GDS)\d+\b',
        'SRA': r'\b(SRX|SRR|SRP|SRX)\d+\b',
        'ENA': r'\b(ERX|ERR|ERP)\d+\b',
        'PDB': r'\b[PQ][\d\w]{3}[A-Za-z]\b',
        'ArrayExpress': r'\b(E-M\w{4}-\d+)\b',
        'ChEMBL': r'\bCHEMBL\d+\b',
    }
    for db, pattern in patterns.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            span = match.group(0)
            context = text[max(0, match.start() - window_size): match.end() + window_size]
            candidates.append((span, context, span))
    return candidates

def normalize_doi(raw_id):
    match = re.search(r'10\.\d+/[\w.\-_]+', str(raw_id))
    if match:
        return f"https://doi.org/{match.group(0)}"
    return None

def classify_context(context):
    context_lower = context.lower()
    primary_keywords = ['generated', 'produced', 'collected', 'this study', 'our study', 'measured']
    secondary_keywords = ['downloaded', 'obtained from', 'retrieved', 'publicly available', 'from the', 'accession']
    
    primary_score = sum(1 for word in primary_keywords if word in context_lower)
    secondary_score = sum(1 for word in secondary_keywords if word in context_lower)
    
    if secondary_score > primary_score:
        return "Secondary"
    elif primary_score > 0:
        return "Primary"
    else:
        return "Secondary"  # default

def parse_xml(xml_path):
    import xml.etree.ElementTree as ET
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        paragraphs = []
        for p in root.findall(".//p"):
            text = ET.tostring(p, encoding='unicode', method='text')
            paragraphs.append(text.strip())
        return {"full_text": " ".join(paragraphs)}
    except Exception as e:
        print(f"XML parse error: {e}")
        return {"full_text": ""}

def extract_pdf_text(pdf_path):
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        print(f"PDF extract error: {e}")
        return ""

def predict_for_paper(doi, xml_path=None, pdf_path=None):
    text = ""
    if xml_path and os.path.exists(xml_path):
        data = parse_xml(xml_path)
        text = data.get("full_text", "")
    elif pdf_path and os.path.exists(pdf_path):
        text = extract_pdf_text(pdf_path)
    else:
        return []
    
    if not text.strip():
        return []
    
    candidates = extract_candidates(text)
    results = []
    for span, context, raw_id in candidates:
        dataset_id = normalize_doi(raw_id) or raw_id  # keep accession as-is
        pred_type = classify_context(context)
        results.append((dataset_id, pred_type))
    
    return list(set(results))  # deduplicate

# -----------------------------
# ğŸš€ RUN PREDICTIONS
# -----------------------------

def get_all_test_dois():
    """Collect all unique test paper DOIs from XML and PDF directories."""
    dois = set()

    # From XML files
    if os.path.exists(test_xml_dir):
        for fname in os.listdir(test_xml_dir):
            if fname.endswith(".xml"):
                doi = fname.replace(".xml", "").replace("_", "/", 1)  # Only first underscore â†’ slash
                dois.add(doi)

    # From PDF files
    if os.path.exists(test_pdf_dir):
        for fname in os.listdir(test_pdf_dir):
            if fname.endswith(".pdf"):
                doi = fname.replace(".pdf", "").replace("_", "/", 1)
                dois.add(doi)

    return sorted(dois)

# Get all test DOIs
test_dois = get_all_test_dois()
print(f"Found {len(test_dois)} test papers.")

# Generate predictions
submission_rows = []
row_id = 0

for doi in test_dois:
    base_name = doi.replace("/", "_")
    xml_path = os.path.join(test_xml_dir, f"{base_name}.xml") if os.path.exists(test_xml_dir) else None
    pdf_path = os.path.join(test_pdf_dir, f"{base_name}.pdf") if os.path.exists(test_pdf_dir) else None

    # Only proceed if file exists
    xml_exists = xml_path and os.path.exists(xml_path)
    pdf_exists = pdf_path and os.path.exists(pdf_path)

    if not (xml_exists or pdf_exists):
        continue  # Skip if no file found

    preds = predict_for_paper(doi, 
                             xml_path if xml_exists else None,
                             pdf_path if pdf_exists else None)
    
    for dataset_id, type_ in preds:
        submission_rows.append({
            "row_id": row_id,
            "article_id": doi,
            "dataset_id": dataset_id,
            "type": type_
        })
        row_id += 1

# Save submission
submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv("/kaggle/working/submission.csv", index=False)
print(f"âœ… Submission saved with {len(submission_df)} entries.")  


! mkdir src


%%writefile src/common.py
import os
from pathlib import Path
from typing import Tuple

import polars as pl

DOI_URL = 'https://doi.org/'

def is_submission() -> bool:
    """Check if running in a Kaggle competition submission context."""
    return bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN'))

def is_kaggle_env() -> bool:
    """Check if running in any Kaggle environment (notebook, script, or submission)."""
    return (len([k for k in os.environ if 'KAGGLE' in k]) > 0) or is_submission()

def get_prefix_path(prefix: str) -> Path:
    """
    Get environment-appropriate path prefix.
    Resolves to `/kaggle/{prefix}` on Kaggle, `./{prefix}` locally.
    """
    path_str = f'/kaggle/{prefix}' if is_kaggle_env() else f'./{prefix}'
    return Path(path_str).expanduser().resolve()

def is_doi(name: str) -> pl.Expr:
    """Return Polars expression checking if column `name` starts with DOI_URL."""
    return pl.col(name).str.starts_with(DOI_URL)

def doi_link_to_id(name: str) -> pl.Expr:
    """
    Convert full DOI URL (e.g. https://doi.org/10.xxxx/...) to just the ID.
    Leaves non-DOI values unchanged.
    """
    return (
        pl.when(is_doi(name))
        .then(pl.col(name).str.replace(DOI_URL, "", literal=True))
        .otherwise(pl.col(name))
        .alias(name)
    )

def doi_id_to_link(name: str, substring: str, url: str = DOI_URL) -> pl.Expr:
    """
    Convert bare DOI ID (e.g. 10.xxxx/...) into full URL.
    Only applies if column value starts with `substring`.
    """
    return (
        pl.when(pl.col(name).str.starts_with(substring))
        .then(url + pl.col(name).str.to_lowercase())
        .otherwise(pl.col(name))
        .alias(name)
    )

def score(
    preds: pl.DataFrame,
    gt: pl.DataFrame,
    on: list = ['article_id', 'dataset_id'],
    verbose: bool = True
) -> Tuple[float, float, float]:
    """
    Compute Precision, Recall, and F1-score based on join between predictions and ground truth.

    Args:
        preds: Predicted matches (must contain columns in `on`)
        gt: Ground truth matches
        on: Columns to join on
        verbose: Whether to print detailed results

    Returns:
        Tuple of (precision, recall, f1)
    """
    # Normalize column name if needed
    if 'id' in preds.columns and 'dataset_id' not in preds.columns:
        preds = preds.rename({'id': 'dataset_id'})

    # Validate join columns
    for col in on:
        if col not in preds.columns:
            raise ValueError(f"Missing column in preds: {col}")
        if col not in gt.columns:
            raise ValueError(f"Missing column in gt: {col}")

    # Perform inner join to find matches (True Positives)
    hits = gt.join(preds, on=on, how="inner")
    tp = hits.height
    fp = preds.height - tp
    fn = gt.height - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    if verbose:
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}")

    return precision, recall, f1   


%%writefile src/parse.py
import argparse
import pymupdf
import pathlib
import tqdm

from common import get_prefix_path, is_submission

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-i', default=f'make-data-count-finding-data-references/{"test" if is_submission() else "train"}/PDF')
    p.add_argument('-o', default='parsed')
    return p.parse_args()

def pdf2text(path: pathlib.Path, out_dir: pathlib.Path) -> None:
    doc = pymupdf.open(str(path))
    out = open(out_dir / f"{path.stem}.txt", "wb")
    for page in doc:
        text = page.get_text().encode("utf8")
        out.write(text)
        out.write(b'\n') # write page delimiter (form feed 0x0C)
    out.close()

def main():
    args = get_args()
    in_dir = get_prefix_path('input') / args.i
    out_dir = get_prefix_path('working') / args.o

    if out_dir.exists() and any(out_dir.iterdir()):
        print(f'{out_dir} already populated, skipping...')
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    if not in_dir.is_dir(): raise ValueError(f'{in_dir} is not a directory...')
    pdf_files = list(in_dir.glob('*.pdf'))
    if not pdf_files: raise ValueError(f'No PDF files found in {in_dir}')

    for pdf in tqdm.tqdm(pdf_files, desc="Processing PDFs"): pdf2text(pdf, out_dir)
    print('ending parsing...')

if __name__ == '__main__': main()


%%writefile src/getacc.py
import polars as pl
import argparse
import pathlib
from common import score, get_prefix_path, is_submission, is_doi, doi_id_to_link

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-i', default='parsed')
    p.add_argument('-o', default='extracted_ids.parquet')
    p.add_argument('--gt', default='make-data-count-finding-data-references/train_labels.csv')
    p.add_argument('--ws', default=100, type=int)
    return p.parse_args()

def get_text_df(parsed_dir: pathlib.Path):
    paths = list(parsed_dir.rglob('*.txt'))
    records = [{'article_id': p.stem, 'text': p.read_text()} for p in paths]
    return (
        pl.DataFrame(records)
        .with_columns(pl.col("text").str.normalize("NFKC").str.replace_all(r"[^\p{Ascii}]", ""))
        .with_columns(pl.col('text').str.split(r'\n{2,}').list.eval(pl.col("").str.replace_all('\n', ' ')).list.join('\n').alias('text'))
        .with_columns([
            pl.col("text").str.slice(pl.col("text").str.len_chars()//4).str.reverse().alias('rtext'),
            pl.col("text").str.slice(0, pl.col("text").str.len_chars()//4).alias('ltext'),
        ])
        .with_columns(pl.col('rtext').str.find(r'(?i)\b(secnerefer|erutaretil detic|stnemegdelwonkca)\b').alias('ref_idx'))
        .with_columns(pl.when(pl.col('ref_idx').is_null()).then(0).otherwise('ref_idx').alias('ref_idx'))
        .with_columns([
            pl.col('rtext').str.slice(0, pl.col('ref_idx')).str.reverse().alias('refs'),
            (pl.col('ltext') + pl.col('rtext').str.slice(pl.col('ref_idx')).str.reverse()).alias('body')
        ])
        .drop('rtext', 'ltext')
    )


def main():
    print('starting extraction of accession ids')
    args = get_args()
    in_path, out_path = map(lambda x: get_prefix_path('working') / x, (args.i, args.o))
    text_df = get_text_df(in_path)

    df = (
        text_df
        .with_columns([
            pl.col("text").str.extract_all(r'(?i)\b(?:CHEMBL\d+|E-GEOD-\d+|E-PROT-\d+|EMPIAR-\d+|ENSBTAG\d+|ENSOARG\d+|EPI_ISL_\d{5,}|EPI\d{6,7}|HPA\d+|CP\d{6}|IPR\d{6}|PF\d{5}|KX\d{6}|K0\d{4}|PRJNA\d+|PXD\d+|SAMN\d+|dryad\.[^\s"<>]+|pasta\/[^\s"<>])').alias('id'),
        ])
        .explode('id')
        .with_columns(pl.col('id').alias('match_id'))
        .with_columns(pl.col('id').str.replace_all(r'\s', ''))
        .with_columns(pl.col('id').str.replace(r'[-.,;:!?\/\)\]\(\[]+$', ''))
        .with_columns(doi_id_to_link(name='id', substring='dryad.', url='https://doi.org/10.5061/'))
        .with_columns(doi_id_to_link(name='id', substring='pasta/', url='https://doi.org/10.6073/'))
        .filter(~pl.col('id').str.to_lowercase().str.contains(pl.col('article_id').str.to_lowercase().str.replace('_', '/')))
        .filter(~pl.col('id').str.contains('figshare', literal=True))
        .filter(pl.when(is_doi('id').and_(pl.col('id').str.split('/').list.last().str.len_chars()<4)).then(pl.lit(False)).otherwise(pl.lit(True)))
        .filter(~pl.col('id').is_in(['https://doi.org/10.5061/dryad', 'https://doi.org/10.6073/pasta', 'https://doi.org/10.5281/zenodo']))
        .filter(pl.col('id').str.count_matches(r'\(') == pl.col('id').str.count_matches(r'\)'))
        .filter(pl.col('id').str.count_matches(r'\[') == pl.col('id').str.count_matches(r'\]'))
        .with_columns(
            pl.col('text').str.slice(pl.col('text').str.find(pl.col('match_id'), literal=True)-args.ws-pl.col('match_id').str.len_chars(), 2*(args.ws+pl.col('match_id').str.len_chars())).alias('window')
        )
        .unique(['article_id', 'id'])
        .rename({'id': 'dataset_id'})
    )
    df.select('article_id', 'dataset_id', 'window').write_parquet(out_path)
    print(f'id extraction written to {out_path}')

    df = df.select('article_id', 'dataset_id').with_columns(pl.lit('Secondary').alias('type'))
    df = df.with_columns(
        pl.when(is_doi('dataset_id').or_(pl.col('dataset_id').str.starts_with('SAMN'))).then(pl.lit('Primary')).otherwise('type').alias('type')
    )

    df.with_row_index(name='row_id').write_csv(get_prefix_path('working')/'submission.csv')

    if not is_submission():
        gt_path = get_prefix_path('input') / args.gt
        gt = pl.read_csv(gt_path).filter(pl.col('type')!='Missing').join(text_df, on='article_id')
        print('### DOI ###')
        score(df.filter(is_doi('dataset_id')), gt.filter(is_doi('dataset_id')))
        print('### ACC ###')
        score(df.filter(~is_doi('dataset_id')), gt.filter(~is_doi('dataset_id')))
        print('### ALL ###')
        score(df, gt)
        print('### TYPE ###')
        score(df, gt, on=['article_id', 'dataset_id', 'type'])

if __name__=='__main__':
    main()


! python src/parse.py


! python src/getacc.py

