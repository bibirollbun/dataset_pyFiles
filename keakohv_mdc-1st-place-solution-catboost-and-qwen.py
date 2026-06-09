! uv pip uninstall -q --system 'tensorflow'
! uv pip install -q --system --no-index --find-links='/kaggle/input/mdc-whls/whls' 'pymupdf' 'vllm' 'triton' 'logits-processor-zoo' 'numpy<2'


import os
import re
import logging
from pathlib import Path
import pickle
import json
import joblib
import shutil
import glob
from tqdm.auto import tqdm
import warnings

import numpy as np
import pandas as pd

# For PDF to TXT conversion
import fitz  # PyMuPDF
import pymupdf
import unicodedata

# For Catboost
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier, Pool

# For Qwen
import torch
import vllm
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor

fitz.TOOLS.mupdf_display_errors(False)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# Set up logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
l = logging.getLogger(__name__)


# Base data locations
BASE_DIR = Path("/kaggle/input/make-data-count-finding-data-references")
WORK_DIR = Path("/kaggle/working")

# Train/Test roots
TRAIN_ROOT = BASE_DIR / "train"
TEST_ROOT = BASE_DIR / "test" if os.getenv("KAGGLE_IS_COMPETITION_RERUN") else BASE_DIR / "train"

# PDF / XML dirs
TRAIN_PDF_DIR = TRAIN_ROOT / "PDF"
TRAIN_XML_PATH = TRAIN_ROOT / "XML"
TEST_PDF_DIR = TEST_ROOT / "PDF"
TEST_XML_PATH = TEST_ROOT / "XML"

# Labels
LABELS_PATH = BASE_DIR / "train_labels.csv"

# Text output dirs (train reused for local experimentation)
TRAIN_TXT_DIR = WORK_DIR / "train_txt"
TEST_TXT_DIR = WORK_DIR / "test_txt" if os.getenv("KAGGLE_IS_COMPETITION_RERUN") else TRAIN_TXT_DIR

# External resources. More info, including recreation, in respective Kaggle dataset description.

# Data Citation Corpus V4.1 where source is eupmc or datacite
CORPUS_PATH = Path(
    "/kaggle/input/data-citation-corpus-v4-1-eupmc-and-datacite/data_citation_corpus_filtered_v4.1.parquet"
)

# EUPMC text-mined accession id - article id mappings
EUPMC_ACC_MAPPING_PATH = Path(
    "/kaggle/input/eupmc-accession-ids-to-article-mapping/eupmc_accession_ids_mapping.csv"
)

# Dataset creators and published year from DataCite
DATACITE_METADATA_PATH = Path(
    "/kaggle/input/datacite-datasets-creators-and-published-year/datacite_dataset_creators_pubyear.parquet"
)

# Articles title, authors, publisher, journal, publisher year from CrossRef, only for articles that include dataset DOIs in Corpus citation pairs
CROSSREF_DOI_METADATA_PATH = Path(
    "/kaggle/input/articles-doi-metadata/doi_metadata.csv"
)

# Article metadata from EUPMC
PMC_ARTICLE_METADATA_PATH = Path(
    "/kaggle/input/pmc-article-metadata/PMC_article_metadata.parquet"
)

# BioSample (SAMN accession ids) submitter and date metadata from NCBI
BIOSAMPLE_SUBMITTER_META_PATH = Path(
    "/kaggle/input/biosample-accession-id-submitter-and-date/biosample_info.parquet"
)

# Catboost config
RANDOM_SEED = 42
N_SPLITS = 6
MODEL_SAVE_DIR = "catboost_doi_type_classifier"


# Load training data filenames
train_filenames = [
    os.path.splitext(f)[0] for f in os.listdir(TRAIN_PDF_DIR) if f.endswith(".pdf")
]

print(f"Found {len(train_filenames)} train files")


# Load test data filenames
test_filenames = [
    os.path.splitext(f)[0] for f in os.listdir(TEST_PDF_DIR) if f.endswith(".pdf")
]

print(f"Found {len(test_filenames)} test files")


test_filenames[:3]


def pdf_to_txt(input_dir: Path, output_dir: Path):
    """Convert all PDF files in input_dir to text files in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.PDF"))
    existing_txt_files = {f.stem for f in output_dir.glob("*.txt")}
    l.info(f"Found {len(pdf_files)} PDF files to process.")

    for pdf_file in pdf_files:
        txt_file = output_dir / f"{pdf_file.stem}.txt"
        if pdf_file.stem in existing_txt_files:
            continue
        try:
            text = ""
            with pymupdf.open(pdf_file) as doc:
                for page in doc:
                    text += page.get_text()
            txt_file.write_text(text, encoding="utf-8")
        except Exception as e:
            l.warning(f"Error processing {pdf_file.name}: {str(e)}")


%%time
# Convert train
l.info('TRAIN PARSE START')
pdf_to_txt(TRAIN_PDF_DIR, TRAIN_TXT_DIR)
l.info('TRAIN PARSE FINISH')


# Convert test when in hidden rerun
if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    l.info('TEST PARSE START')
    pdf_to_txt(TEST_PDF_DIR, TEST_TXT_DIR)
    l.info('TEST PARSE FINISH')


txt_file = next(TRAIN_TXT_DIR.glob("*.txt"), None)

if txt_file:
    print(txt_file)
    print()
    text = txt_file.read_text(encoding="utf-8")
    print(text[:2000])


# Load and filter training labels
df_labels = pd.read_csv(LABELS_PATH)

# Create a mask for rows to remove
missing_mask = df_labels['type'] == 'Missing'

# Articles with Missing rows are sparsely labeled and do not account towards the competition F1 score. Look at our writeup for longer explanation.
df_labels = df_labels[~missing_mask]
print(f"Removed {missing_mask.sum()} rows with type == 'Missing'")

# Add DOI identification column
df_labels['is_doi'] = df_labels['dataset_id'].str.startswith('https://doi.org/')

# Extract DOI components
df_labels['publisher_code'] = None
df_labels['doi_name'] = None
doi_mask = df_labels['is_doi']
df_labels.loc[doi_mask, 'doi_name'] = (
    df_labels.loc[doi_mask, 'dataset_id'].str.replace('https://doi.org/', '')
)

# Get publisher code
doi_split = df_labels.loc[doi_mask, 'doi_name'].str.split('/', n=1)
df_labels.loc[doi_mask, 'publisher_code'] = doi_split.str[0]

# Extract DOI ground truth for evaluation
ground_truth_doi = df_labels[df_labels['is_doi'] == True].copy()[['dataset_id','article_id']]
ground_truth_acc = df_labels[df_labels['is_doi'] == False].copy()[['dataset_id','article_id']]


df_labels.sample(5)


# How many rows are in the training set?
len(df_labels)


# Number of unique articles
df_labels.article_id.nunique()


# Examples of dataset_id values
print("Examples of DOIs (is_doi=True):")
print(df_labels.loc[df_labels.is_doi, "dataset_id"].sample(5).to_list())

print("\nExamples of accession IDs (is_doi=False):")
print(df_labels.loc[~df_labels.is_doi, "dataset_id"].sample(10).to_list())

# Counts of DOIs vs accession IDs
counts = df_labels.is_doi.value_counts()

# Map True → Dataset DOIs, False → Accession IDs
label_map = {True: "Dataset DOIs", False: "Accession IDs"}
counts_named = counts.rename(index=label_map)

print("\nCounts:")
print(counts_named.to_dict())


# For each dataset mention, we need to predict the type: Primary for data created for that study, Secondary if data was reused
def count_types(df):
    vc = df["type"].value_counts()
    return int(vc.get("Primary", 0)), int(vc.get("Secondary", 0))

# Overall (non-missing already filtered)
p_all, s_all = count_types(df_labels)

# DOI-only
p_doi, s_doi = count_types(df_labels[df_labels["is_doi"]])

# Accession ID-only
p_acc, s_acc = count_types(df_labels[~df_labels["is_doi"]])

print(f"In non-missing labels there are {p_all} Primary and {s_all} Secondary type dataset_ids.")
print(f"For DOI-only: {p_doi} Primary, {s_doi} Secondary")
print(f"For accession ID-only: {p_acc} Primary, {s_acc} Secondary")


citation_pairs = pd.read_parquet(CORPUS_PATH)


citation_pairs.source.value_counts()


citation_pairs.sample(5)


df_doi_corpus = citation_pairs[citation_pairs.source == "datacite"].copy() # Keep only datacite for doi
df_acc_corpus = citation_pairs[citation_pairs.source == "eupmc"].copy() # Keep only eupmc for accession_ids


# Remove non-DOI HTTPS links and figshare links. In the training set all figshare DOIs are type Missing.
df_doi_corpus = df_doi_corpus[
    ~df_doi_corpus['dataset'].str.contains("figshare") &
    ~((df_doi_corpus['dataset'].str.contains("https")) &
      ~df_doi_corpus['dataset'].str.contains("doi.org"))
]

# Remove duplicates, there are 39991 such duplicates
df_doi_corpus = df_doi_corpus.drop_duplicates(subset=['dataset', 'publication'])

# Create the doi path / search dataset string that we shall search from the article
df_doi_corpus.loc[:, 'search_dataset'] = df_doi_corpus['dataset'].str.removeprefix('https://doi.org/')


df_doi_corpus.sample(5)


# Remove doi rows from Corpus EUPMC source dataframe, there are 108616 such rows
df_acc_corpus = df_acc_corpus[~df_acc_corpus.dataset.str.contains("doi.org|https?")]


df_acc_corpus.sample(5)


df_acc_corpus.repository.value_counts()


df_eupmc_original = pd.read_csv(EUPMC_ACC_MAPPING_PATH)


df_eupmc_original


df_eupmc_original.database.value_counts()


df_article_metadata_eupmc = pd.read_parquet(PMC_ARTICLE_METADATA_PATH)


df_article_metadata_eupmc


df_biosample = pd.read_parquet(BIOSAMPLE_SUBMITTER_META_PATH)


df_biosample


df_article_metadata_crossref = pd.read_csv(CROSSREF_DOI_METADATA_PATH)
df_article_metadata_crossref['DOI'] = df_article_metadata_crossref['DOI'].str.replace('/', '_')

column_mapping = {
    'Title': 'article_title',
    'Publication Year': 'article_pub_year',
    'Journal': 'article_journal',
    'Publisher': 'article_publisher',
    'Authors': 'article_authors'
}
df_article_metadata_crossref = df_article_metadata_crossref.rename(columns=column_mapping)


df_article_metadata_crossref.sample(5)


df_datacite_metadata = pd.read_parquet(DATACITE_METADATA_PATH)
df_datacite_metadata = df_datacite_metadata.rename(columns={'creators': 'dataset_creators', 'publication_year':'dataset_published_year'})


df_datacite_metadata.sample(5)


df_doi_corpus_enriched = df_doi_corpus.merge(df_datacite_metadata, left_on='dataset', right_on='dataset_id', how='left', suffixes=('_corpus', '_meta'))
df_doi_corpus_enriched = df_doi_corpus_enriched.merge(df_article_metadata_crossref, left_on='publication', right_on='DOI', how='left', suffixes=('_dataset', '_article'))


# If published year is below 1800 set to NaN and if is above 2025 set to NaN
df_doi_corpus_enriched['dataset_published_year'] = df_doi_corpus_enriched['dataset_published_year'].apply(lambda x: x if 1880 <= x <= 2025 else pd.NA)


df_doi_corpus_enriched = df_doi_corpus_enriched.drop(columns=['DOI', 'dataset_id'])


df_doi_corpus_enriched.to_parquet('citation_pairs_v4.1_doi_enriched.parquet', index=False)


df_doi_corpus_enriched.sample(5)


# Filter Datacite citations to training articles only
df_train_doi_citations = df_doi_corpus_enriched[df_doi_corpus_enriched['publication'].isin(train_filenames)].copy()


len(df_train_doi_citations)


def _nfkc_lower(s: str) -> str:
    """Unicode NFKC normalize and lowercase."""
    if s is None:
        return ""
    return unicodedata.normalize("NFKC", s).lower()


def _strip_version(doi: str) -> str:
    """Return DOI without trailing .vN version, e.g., '...19146182.v1' -> '...19146182'."""
    return re.sub(r"\.v\d+\b", "", doi or "", flags=re.IGNORECASE)


def _compress_with_map(text: str, keep_chars=r"a-z0-9/_\.\-:"):
    """
    Compress text by removing everything except characters in keep_chars.
    Returns (compressed_text, index_map) where index_map[i] = original index of compressed_text[i].
    """
    if not text:
        return "", []
    allowed = re.compile(f"[{keep_chars}]")
    compressed_chars = []
    idx_map = []
    for i, ch in enumerate(text):
        if allowed.match(ch):
            compressed_chars.append(ch)
            idx_map.append(i)
    return "".join(compressed_chars), idx_map


def _compress_no_space_with_map(text: str):
    """
    Remove all Unicode whitespace characters from text.
    Returns (compressed_text, index_map) where index_map[i] = original index in text.
    """
    if not text:
        return "", []
    out, idx_map = [], []
    for i, ch in enumerate(text):
        if not ch.isspace():
            out.append(ch)
            idx_map.append(i)
    return "".join(out), idx_map


def _make_flexible_regex(doi: str) -> re.Pattern:
    """
    Build a regex that allows arbitrary whitespace/punct between DOI tokens.
    """
    if not doi:
        return re.compile(r"a^")  # never matches
    noise = r"[\W_]*"
    pieces = [re.escape(ch) for ch in doi]
    pattern = noise.join(pieces)
    return re.compile(pattern, re.IGNORECASE | re.DOTALL)


def _clamp_span_around_match(start: int, end: int, content: str, max_total: int = 900, side: int = 400) -> str:
    """
    Build a snippet centered on [start:end] such that:
      - Desired length = 2*side + (end-start)
      - Snippet length <= max_total (hard cap)
      - Keeps the whole match inside
    """
    match_len = end - start
    desired_total = min(2 * side + match_len, max_total)

    left = max(0, start - side)
    right = min(len(content), end + side)

    current_len = right - left
    if current_len > desired_total:
        excess = current_len - desired_total
        trim_left = excess // 2
        trim_right = excess - trim_left
        left += trim_left
        right -= trim_right

        if left > start:
            shift = left - start
            left -= shift
            right -= shift
        if right < end:
            shift = end - right
            left += shift
            right += shift

        left = max(0, left)
        right = min(len(content), right)

    snippet = content[left:right]
    if len(snippet) > max_total:
        over = len(snippet) - max_total
        trim_left = over // 2
        trim_right = over - trim_left
        snippet = snippet[trim_left: len(snippet) - trim_right]

    snippet_cleaned = ' '.join(snippet.split())
    return f"...{snippet_cleaned}..."


def find_mentions(
    df: pd.DataFrame,
    content_dir: Path,
    extension: str,
    article_col_name: str,
    dataset_col_name: str,
    do_simple: bool = False
) -> pd.DataFrame:
    """
    Checks for file existence and extracts context for each mention of a search term.

    When do_simple=False (default):
        - Multi-strategy search over normalized content:
          exact_norm, compressed, flex_regex
        - Returns ALL unique contexts found.

    When do_simple=True:
        - FIRST match only
        - Strategy order: exact (case-insensitive, no normalization) → whitespace-flexible
        - Snippet: up to 400 + len(match) + 400 (hard-capped at 900)
    """
    file_type = extension.strip('.')
    print(f"Finding mentions in .{file_type} files... (simple={do_simple})")

    # ---------- SIMPLE PATH (replicates the second function) ----------
    def simple_search(search_term: str, content: str):
        """Return (single_snippet_or_None, match_type) for the FIRST match only."""
        if not search_term or not content:
            return None, ""

        # Strategy 1: Exact (first match only), case-insensitive, on raw content
        m = re.search(re.escape(search_term), content, flags=re.IGNORECASE)
        if m:
            start, end = m.span()
            snippet = _clamp_span_around_match(start, end, content, max_total=900, side=400)
            return snippet, "exact"

        # Strategy 2: Whitespace-flexible (first match only)
        st_nowhitespace, st_idx_map = _compress_no_space_with_map(search_term)
        if st_nowhitespace and len(st_nowhitespace) > 3:
            content_nowhitespace, c_idx_map = _compress_no_space_with_map(content)
            m2 = re.search(re.escape(st_nowhitespace), content_nowhitespace, flags=re.IGNORECASE)
            if m2:
                c_start, c_end = m2.span()
                # map back to original content indices
                start_orig = c_idx_map[c_start]
                end_orig = c_idx_map[c_end - 1] + 1
                snippet = _clamp_span_around_match(start_orig, end_orig, content, max_total=900, side=400)
                return snippet, "whitespace_flexible"

        return None, ""

    def enhanced_search(search_term, content):
        """Enhanced search with three strategies (exact, compressed, flexible-regex)."""
        if not search_term or not content:
            return [], ""

        content_norm = _nfkc_lower(content)
        term_norm = _nfkc_lower(search_term)

        term_candidates = []
        if term_norm:
            term_candidates.append(term_norm)

        all_contexts = []
        match_types = []

        def _context_from_span(start_idx, end_idx, source_text, pad_left=335, pad_right=375):
            snippet_start = max(0, start_idx - pad_left)
            snippet_end = min(len(source_text), end_idx + pad_right)
            snippet = source_text[snippet_start:snippet_end]
            snippet_cleaned = ' '.join(snippet.split())
            return f"...{snippet_cleaned}..."

        # Strategy 1: Exact substring on normalized content
        for cand in term_candidates:
            m_iter = list(re.finditer(re.escape(cand), content_norm, flags=re.IGNORECASE))
            if m_iter:
                for m in m_iter:
                    start, end = m.span()
                    context = _context_from_span(start, end, content_norm)
                    if context not in all_contexts:
                        all_contexts.append(context)
                        match_types.append("exact_norm")

        # Strategy 2: Compressed comparison with index map (keep DOI-safe chars)
        keep = r"a-z0-9/_\.\-:"
        content_comp, idx_map = _compress_with_map(content_norm, keep_chars=keep)
        for cand in term_candidates:
            cand_comp, _ = _compress_with_map(cand, keep_chars=keep)
            if cand_comp and len(cand_comp) > 3:
                for m in re.finditer(re.escape(cand_comp), content_comp, flags=re.IGNORECASE):
                    c_start, c_end = m.span()
                    start_orig = idx_map[c_start]
                    end_orig = idx_map[c_end - 1] + 1
                    context = _context_from_span(start_orig, end_orig, content_norm)
                    if context not in all_contexts:
                        all_contexts.append(context)
                        match_types.append("compressed")

        # Strategy 3: Flexible regex
        for cand in term_candidates:
            flex_re = _make_flexible_regex(cand)
            for m in flex_re.finditer(content_norm):
                start, end = m.span()
                context = _context_from_span(start, end, content_norm)
                if context not in all_contexts:
                    all_contexts.append(context)
                    match_types.append("flex_regex")

        final_contexts = []
        if all_contexts:
            for ctx in all_contexts:
                final_contexts.append(ctx)

        return final_contexts or all_contexts, " | ".join(sorted(set(match_types))) if match_types else ""

    def check_row(row):
        # Case-insensitive lookup for "<article>.<ext>"
        file_path = next((p for p in content_dir.glob(f"{row[article_col_name]}.*") if p.suffix.lower() == f".{file_type}"), None)
        exists = file_path is not None
        context_snippets = []
        match_type = ""

        if exists and isinstance(row[dataset_col_name], str):
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                if do_simple:
                    snippet, mtype = simple_search(row[dataset_col_name], content)
                    if snippet:
                        context_snippets = [snippet]
                        match_type = mtype
                else:
                    context_snippets, match_type = enhanced_search(row[dataset_col_name], content)
            except Exception as e:
                print(f"   -> Warning: Could not process {file_path}. Error: {e}")

        is_present = len(context_snippets) > 0
        full_context = " | ".join(context_snippets) if is_present else None
        return pd.Series([exists, is_present, full_context, match_type])

    df[[f'{file_type}_exists', f'{file_type}_present', f'{file_type}_context', f'{file_type}_match_type']] = df.apply(check_row, axis=1)
    print(f"   -> Done checking .{file_type} files (simple={do_simple}).")
    return df



# Apply the mention detection to filtered citations
print("Checking for mentions in XML and TXT files...")
df_train_doi_citations = find_mentions(df_train_doi_citations.copy(), TRAIN_XML_PATH, "xml", "publication", "search_dataset")
df_train_doi_citations = find_mentions(df_train_doi_citations.copy(), TRAIN_TXT_DIR, "txt", "publication", "search_dataset")

# Create any_present column to identify citations found in either XML or TXT
df_train_doi_citations.loc[:, 'any_present'] = df_train_doi_citations[['xml_present', 'txt_present']].any(axis=1)

print("Mention detection results:")
print(df_train_doi_citations.any_present.value_counts())

# Filter to only citations that are present in the text
df_train_doi_present_citations = df_train_doi_citations[df_train_doi_citations['any_present'] == True].copy()

# Create predictions DataFrame for evaluation
predictions = df_train_doi_present_citations[['dataset','publication']].rename(
    columns={'dataset':'dataset_id', 'publication':'article_id'}
)[['dataset_id','article_id']]
predictions = predictions.drop_duplicates()

print(f"\nTotal predictions: {len(predictions)}")
display(predictions.head())


def calculate_mention_metrics(ground_truth_set, predictions_set):
    """Calculate precision, recall, and F1-score for mention-level evaluation."""
    true_positives = len(ground_truth_set.intersection(predictions_set))
    false_positives = len(predictions_set - ground_truth_set)
    false_negatives = len(ground_truth_set - predictions_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'ground_truth_count': len(ground_truth_set),
        'predictions_count': len(predictions_set),
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

# Evaluate DOI-only mention-level performance
ground_truth_doi = df_labels[df_labels['dataset_id'].str.contains('doi.org', na=False)][['article_id', 'dataset_id']]
ground_truth_doi_set = set(zip(ground_truth_doi['article_id'], ground_truth_doi['dataset_id']))
pred_mentions_doi_set = set(zip(predictions['article_id'], predictions['dataset_id']))

doi_metrics = calculate_mention_metrics(ground_truth_doi_set, pred_mentions_doi_set)

print("=== MENTION-LEVEL PERFORMANCE (DOI ONLY) ===")
print(f"Total DOI ground truth mentions: {doi_metrics['ground_truth_count']}")
print(f"Total predicted DOI mentions: {doi_metrics['predictions_count']}")
print(f"True positives (DOI): {doi_metrics['true_positives']}")
print(f"False positives (DOI): {doi_metrics['false_positives']}")
print(f"False negatives (DOI): {doi_metrics['false_negatives']}")
print(f"Precision (DOI): {doi_metrics['precision']:.4f}")
print(f"Recall (DOI): {doi_metrics['recall']:.4f}")
print(f"F1-score (DOI): {doi_metrics['f1_score']:.4f}")


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal features from date and year columns."""
    df = df.copy()

    # Ensure year columns exist with nullable integer dtype
    if 'article_pub_year' in df.columns:
        df['article_pub_year'] = pd.to_numeric(df['article_pub_year'], errors='coerce').astype('Int64')
    if 'dataset_published_year' in df.columns:
        df['dataset_published_year'] = pd.to_numeric(df['dataset_published_year'], errors='coerce').astype('Int64')

    # Article vs dataset year features (only when both columns exist)
    if {'article_pub_year', 'dataset_published_year'}.issubset(df.columns):
        apy = df['article_pub_year']
        dpy = df['dataset_published_year']

        valid = apy.notna() & dpy.notna()

        # Create match as a nullable boolean
        df['article_dataset_year_match'] = False
        df.loc[valid, 'article_dataset_year_match'] = (
            apy[valid].astype('int64').values == dpy[valid].astype('int64').values
        )

        # Nullable int difference (<NA> if either side missing)
        df['article_dataset_year_diff'] = apy - dpy
    else:
        df['article_dataset_year_match'] = False
        df['article_dataset_year_diff'] = pd.NA

    # Extract years from datetime columns
    df['year_updated'] = pd.to_datetime(df.get('updated'), errors='coerce').dt.year.astype('Int64')
    df['year_published'] = pd.to_datetime(df.get('publishedDate'), errors='coerce').dt.year.astype('Int64')

    # Year matching features (nullable-safe)
    df['year_pub_article_match'] = (df['year_published'] == df['article_pub_year'])
    df['year_updated_article_match'] = (df['year_updated'] == df['article_pub_year'])
    df['year_pub_updated_match'] = (df['year_published'] == df['year_updated'])

    # === The Fix ===
    # Convert all relevant nullable integer and boolean columns to float64
    # This is crucial because CatBoost expects numerical features to be standard floats (with NaN for missing)
    numeric_cols_to_convert = [
        'article_pub_year',
        'dataset_published_year',
        'article_dataset_year_diff',
        'year_updated',
        'year_published',
        'article_dataset_year_match',
        'year_pub_article_match',
        'year_updated_article_match',
        'year_pub_updated_match',
    ]

    for col in numeric_cols_to_convert:
        if col in df.columns:
            # .astype('float64') will convert pd.NA to np.nan
            df[col] = df[col].astype('float64')

    return df

def create_publisher_journal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create publisher and journal matching features."""
    df = df.copy()
    
    # Create matching features (case-insensitive comparison)
    df['publisher_match'] = (
        df['article_publisher'].str.lower() == df['publisher'].str.lower()
    )
    df['journal_match'] = (
        df['article_journal'].str.lower() == df['journal'].str.lower()
    )
    
    return df

def create_citation_count_features(df: pd.DataFrame, citation_pairs: pd.DataFrame) -> pd.DataFrame:
    """Create citation count features."""
    df = df.copy()
    
    # Filter to DOI citations only
    citations_doi = citation_pairs[citation_pairs['dataset'].str.contains('doi.org', na=False)].copy()
    
    # Count citations per dataset
    citation_counts = citations_doi.groupby('dataset').size().reset_index(name='citation_count')
    df = df.merge(citation_counts, left_on='dataset', right_on='dataset', how='left')
    
    # Count citations per article
    article_counts = df.groupby('publication').size().reset_index(name='article_citation_count')
    df = df.merge(article_counts, left_on='publication', right_on='publication', how='left')
    
    return df

def create_title_similarity_features_fold(X_train, X_test):
    """Create title similarity features using TF-IDF fitted only on training data."""
    # Fill NaN values with empty strings
    # Check if title column exists, otherwise use empty string
    X_train_titles = X_train.get('title', pd.Series([""] * len(X_train))).fillna("")
    X_train_article_titles = X_train['article_title'].fillna("")
    X_test_titles = X_test.get('title', pd.Series([""] * len(X_test))).fillna("")
    X_test_article_titles = X_test['article_title'].fillna("")
    
    # Initialize TF-IDF vectorizer and fit only on training data
    vectorizer = TfidfVectorizer()
    
    # Combine training titles for vectorization
    train_titles_combined = X_train_titles.tolist() + X_train_article_titles.tolist()
    
    # Fit vectorizer on training data only
    vectorizer.fit(train_titles_combined)
    
    # Transform training data
    train_tfidf_matrix = vectorizer.transform(train_titles_combined)
    train_titles_tfidf = train_tfidf_matrix[:len(X_train)]
    train_article_titles_tfidf = train_tfidf_matrix[len(X_train):]
    train_similarities = cosine_similarity(train_titles_tfidf, train_article_titles_tfidf).diagonal()
    
    # Transform test data with fitted vectorizer
    test_titles_combined = X_test_titles.tolist() + X_test_article_titles.tolist()
    test_tfidf_matrix = vectorizer.transform(test_titles_combined)
    test_titles_tfidf = test_tfidf_matrix[:len(X_test)]
    test_article_titles_tfidf = test_tfidf_matrix[len(X_test):]
    test_similarities = cosine_similarity(test_titles_tfidf, test_article_titles_tfidf).diagonal()
    
    return train_similarities, test_similarities

def add_title_text_features_fold(X_train, X_test):
    """Add enhanced title text features fitted only on training data."""
    # Fill NaN values
    # Check if title column exists, otherwise use empty string
    train_a = X_train.get('title', pd.Series([""] * len(X_train))).fillna('')
    train_b = X_train['article_title'].fillna('')
    test_a = X_test.get('title', pd.Series([""] * len(X_test))).fillna('')
    test_b = X_test['article_title'].fillna('')

    # 1) Char n-gram tf-idf (fit on train only)
    char_vec = TfidfVectorizer(analyzer='char', ngram_range=(3,5), min_df=2)
    train_combined = pd.concat([train_a, train_b], ignore_index=True)
    char_vec.fit(train_combined)
    
    # Transform train
    M_train = char_vec.transform(train_combined)
    Ma_train, Mb_train = M_train[:len(train_a)], M_train[len(train_a):]
    train_char_cosine = cosine_similarity(Ma_train, Mb_train).diagonal()
    
    # Transform test
    test_combined = pd.concat([test_a, test_b], ignore_index=True)
    M_test = char_vec.transform(test_combined)
    Ma_test, Mb_test = M_test[:len(test_a)], M_test[len(test_a):]
    test_char_cosine = cosine_similarity(Ma_test, Mb_test).diagonal()

    # 2) Word n-gram count (fit on train only)
    word_vec = CountVectorizer(ngram_range=(1,2), min_df=2)
    word_vec.fit(train_combined)
    
    # Transform train
    C_train = word_vec.transform(train_combined)
    Ca_train, Cb_train = C_train[:len(train_a)], C_train[len(train_a):]
    Ca_train_bin, Cb_train_bin = (Ca_train>0).astype(int), (Cb_train>0).astype(int)
    inter_train = (Ca_train_bin.multiply(Cb_train_bin)).sum(axis=1).A1
    union_train = (Ca_train_bin + Cb_train_bin).astype(bool).sum(axis=1).A1
    train_token_jaccard = (inter_train / np.maximum(union_train, 1))
    
    # Transform test
    C_test = word_vec.transform(test_combined)
    Ca_test, Cb_test = C_test[:len(test_a)], C_test[len(test_a):]
    Ca_test_bin, Cb_test_bin = (Ca_test>0).astype(int), (Cb_test>0).astype(int)
    inter_test = (Ca_test_bin.multiply(Cb_test_bin)).sum(axis=1).A1
    union_test = (Ca_test_bin + Cb_test_bin).astype(bool).sum(axis=1).A1
    test_token_jaccard = (inter_test / np.maximum(union_test, 1))

    return (train_char_cosine, train_token_jaccard), (test_char_cosine, test_token_jaccard)

def normalize_authors(s):
    """Normalize author strings to extract last names."""
    if not isinstance(s, str) or not s.strip(): 
        return []
    # Split on ; , and 'and'
    parts = re.split(r';|,|\band\b', s, flags=re.I)
    # Keep simple last names: take last token of each non-empty name
    last_names = []
    for p in parts:
        t = p.strip()
        if not t: 
            continue
        toks = t.split()
        if toks:
            last_names.append(toks[-1].lower())
    return [x for x in last_names if x]

def create_author_overlap_features(df: pd.DataFrame, ds_auth_col: str = 'dataset_creators') -> pd.DataFrame:
    """Create author/contributor overlap features."""
    df = df.copy()

    # Normalize author names for both article and dataset
    art_last = df['article_authors'].apply(normalize_authors)
    ds_last = df[ds_auth_col].apply(normalize_authors)

    def jaccard(a, b):
        """Calculate Jaccard similarity between two sets."""
        A, B = set(a), set(b)
        return len(A & B) / len(A | B) if A or B else 0.0

    def check_creator_in_article_text(row):
        """Check if any dataset creator appears in the article text."""
        if not isinstance(row[ds_auth_col], str) or not row[ds_auth_col].strip():
            return False
        
        # Get available article text contexts
        contexts = []
        if pd.notna(row.get('xml_context')):
            contexts.append(row['xml_context'].lower())
        if pd.notna(row.get('txt_context')):
            contexts.append(row['txt_context'].lower())
        
        if not contexts:
            return False
        
        # Combine all contexts
        article_text = ' '.join(contexts)
        
        # Extract creator last names
        creator_names = normalize_authors(row[ds_auth_col])
        
        # Check if any creator last name appears in article text
        for name in creator_names:
            if len(name) > 2 and name in article_text:  # Only check names longer than 2 chars
                return True
        
        return False

    # Calculate author overlap features
    df['author_jaccard'] = [jaccard(a, b) for a, b in zip(art_last, ds_last)]
    df['author_overlap_frac'] = [
        (len(set(b) & set(a)) / max(1, len(set(b)))) for a, b in zip(art_last, ds_last)
    ]
    df['first_author_match'] = [
        (len(a) > 0 and len(b) > 0 and a[0] == b[0]) for a, b in zip(art_last, ds_last)
    ]
    df['last_author_match'] = [
        (len(a) > 0 and len(b) > 0 and a[-1] == b[-1]) for a, b in zip(art_last, ds_last)
    ]
    
    # Check if any creator appears in article text
    df['any_creator_in_article'] = df.apply(check_creator_in_article_text, axis=1)
    
    # Check if GBIF appears in dataset creators
    df['gbif_in_creators'] = df[ds_auth_col].str.contains('GBIF', case=False, na=False)
    
    return df

def create_dataset_count_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features based on dataset count patterns within articles (prediction-safe)."""
    df = df.copy()
    
    # Group by article to count datasets (this info is available at prediction time)
    article_stats = df.groupby('publication').size().reset_index(name='dataset_count_per_article')
    
    # Create dataset count features
    article_stats['is_single_dataset_article'] = article_stats['dataset_count_per_article'] == 1
    
    # Categorize articles by dataset count
    def categorize_dataset_count(count):
        if count == 1:
            return 'single'
        elif count <= 4:
            return 'few'  # 2-4 datasets
        elif count <= 10:
            return 'many'  # 5-10 datasets
        else:
            return 'very_many'  # >10 datasets
    
    article_stats['dataset_count_category'] = article_stats['dataset_count_per_article'].apply(categorize_dataset_count)
    
    # Calculate percentiles for dataset count
    article_stats['dataset_count_percentile'] = article_stats['dataset_count_per_article'].rank(pct=True)
    
    # Merge back with original dataframe
    merge_columns = [
        'dataset_count_per_article', 'is_single_dataset_article',
        'dataset_count_category', 'dataset_count_percentile'
    ]
    
    df = df.merge(article_stats[['publication'] + merge_columns], on='publication', how='left')
    
    return df

# Merge present_citations with article metadata
citations_features = df_train_doi_present_citations.copy()

# Apply feature engineering functions (before removing date columns)
citations_features = create_temporal_features(citations_features)
citations_features = create_publisher_journal_features(citations_features)
citations_features = create_citation_count_features(citations_features, citation_pairs)

# Note: title similarity features will be created inside CV loop to prevent leakage
citations_features = create_author_overlap_features(citations_features)
citations_features = create_dataset_count_features(citations_features)

# Remove unnecessary columns (after feature engineering)
# Keep article_title and title for similarity features
columns_to_drop = ['updated', 'id', 'created', 'search_dataset', 'publishedDate', 'source', 'DOI', 'dataset_id','article_id']
citations_features.drop(columns=[col for col in columns_to_drop if col in citations_features.columns], inplace=True)

# Merge with labels and create training data
df_labels = pd.read_csv(LABELS_PATH)
df_labels = df_labels[df_labels['type'] != "Missing"]

# Merge with labels (left join to include all predictions)
citations_features_labelled = citations_features.merge(
    df_labels,
    left_on=['publication', 'dataset'],
    right_on=['article_id', 'dataset_id'],
    how='left'
)

# Fill missing labels (these are false positives at mention level)
citations_features_labelled['type'] = citations_features_labelled['type'].fillna("Not_present")

# Separate false positives for later analysis
citations_features_labelled_false_pos = citations_features_labelled[
    citations_features_labelled['type'] == 'Not_present'
].copy()


# Remove the false positives, no ground truth for them
# These are predictions that don't match any ground truth labels
citations_features_labelled = citations_features_labelled[citations_features_labelled['type'] != 'Not_present'].copy()


citations_features_labelled.to_csv("citations_features_labelled.csv", index=False)


citations_features_labelled.type.value_counts()


# Define base feature columns for model training (title similarity features added during CV)
BASE_FEATURE_COLUMNS = [
    # Temporal features
    'year_published', 'year_updated', 'article_pub_year','dataset_published_year',
    'year_pub_article_match', 'year_updated_article_match',
    'year_pub_updated_match', 'article_dataset_year_match', 'article_dataset_year_diff',
    # Publisher/journal matching features
    'publisher_match', 'journal_match',
    # Citation count features
    'citation_count', 'article_citation_count',
    # Text presence features
    'xml_present', 'txt_present', 'xml_exists',
    # Author overlap features
    'author_jaccard', 'first_author_match', 'last_author_match', 'author_overlap_frac', 'any_creator_in_article', 'gbif_in_creators',
    # Dataset count features
    'dataset_count_per_article', 'is_single_dataset_article', 'dataset_count_percentile',
    # Categorical features
    'repository', 'publisher', 'journal',
    'article_journal', 'article_publisher', 'dataset_count_category',
    # Title column (needed for similarity features but not used directly as a feature)
    'article_title', 'title'
]

# Title similarity features will be added during cross-validation
TITLE_SIMILARITY_FEATURES = ['titles_similarity', 'title_char_cosine', 'title_token_jaccard']

# Complete feature columns (used inside CV loop)
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + TITLE_SIMILARITY_FEATURES

CATEGORICAL_COLUMNS = [
    'repository', 'publisher', 'journal',
    'article_journal', 'article_publisher', 'dataset_count_category',
]

# Model configuration
TARGET_COLUMN = 'type'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# === Prepare training data ===
# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(citations_features_labelled[TARGET_COLUMN])

# Prepare features and group information
X = citations_features_labelled[BASE_FEATURE_COLUMNS].copy()
groups = citations_features_labelled["publication"]
X['publication'] = citations_features_labelled["publication"]
X['dataset_id'] = citations_features_labelled["dataset_id"]

# Save configurations
joblib.dump(label_encoder, os.path.join(MODEL_SAVE_DIR, 'label_encoder.pkl'))
with open(os.path.join(MODEL_SAVE_DIR, 'columns_config.json'), 'w') as f:
    json.dump({
        'base_features_columns': BASE_FEATURE_COLUMNS,
        'title_similarity_features': TITLE_SIMILARITY_FEATURES,
        'features_columns': [col for col in BASE_FEATURE_COLUMNS if col not in ['article_title', 'title']] + TITLE_SIMILARITY_FEATURES,
        'categorical_columns': CATEGORICAL_COLUMNS
    }, f)

# CatBoost requires categorical column indices (not names) - will be updated in CV loop
cat_features_indices = [i for i, c in enumerate(BASE_FEATURE_COLUMNS + TITLE_SIMILARITY_FEATURES) if c in CATEGORICAL_COLUMNS]

# === Cross-validation setup ===
skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

# === CatBoost hyperparameters ===
catboost_params = {
    'loss_function': 'MultiClass',
    'iterations': 1000,
    'learning_rate': 0.03,
    'depth': 6,
    'random_seed': RANDOM_SEED,
    'verbose': 100,
    'eval_metric': 'MultiClass',
    'task_type': 'CPU',
    'early_stopping_rounds': 20,
    'auto_class_weights': 'Balanced',
    # Feature bagging and regularization
    'rsm': 0.8,  # Random subspace method (feature fraction)
    'bootstrap_type': 'Bayesian',  # Better than default MVS for smaller datasets
    'bagging_temperature': 0.8,  # Controls sampling - lower = less randomness, higher = more diversity
    'l2_leaf_reg': 3.0,  # L2 regularization for leaf values
    'border_count': 128,  # Reduce from default 254 to add more regularization
}

# === Training and evaluation ===
oof_records = []
feature_importance_list = []

# Train model for each fold
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y, groups)):
    print(f"\n=== Fold {fold + 1}/{N_SPLITS} ===")
    
    # Split data
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y[train_idx], y[test_idx]

    # Create title similarity features inside the fold to prevent leakage
    print("Creating title similarity features...")

    train_similarities, test_similarities = create_title_similarity_features_fold(X_train, X_test)
    X_train['titles_similarity'] = train_similarities
    X_test['titles_similarity'] = test_similarities
    
    # Create additional title text features inside the fold to prevent leakage
    print("Creating title text features...")
    (train_char_cosine, train_token_jaccard), (test_char_cosine, test_token_jaccard) = add_title_text_features_fold(X_train, X_test)
    X_train['title_char_cosine'] = train_char_cosine
    X_train['title_token_jaccard'] = train_token_jaccard
    X_test['title_char_cosine'] = test_char_cosine
    X_test['title_token_jaccard'] = test_token_jaccard

    # Handle missing values in categorical columns
    for col in CATEGORICAL_COLUMNS:
        X_train[col] = X_train[col].fillna("missing")
        X_test[col] = X_test[col].fillna("missing")

    # Now update current_feature_columns to include title similarity features (excluding title columns used only for similarity)
    current_feature_columns = [col for col in BASE_FEATURE_COLUMNS if col not in ['article_title', 'title']] + TITLE_SIMILARITY_FEATURES
    cat_features_indices = [i for i, c in enumerate(current_feature_columns) if c in CATEGORICAL_COLUMNS]

    # Create CatBoost pools
    train_pool = Pool(X_train[current_feature_columns], y_train, cat_features=cat_features_indices)
    test_pool = Pool(X_test[current_feature_columns], y_test, cat_features=cat_features_indices)

    # Train model
    model = CatBoostClassifier(**catboost_params)
    model.fit(train_pool, eval_set=test_pool)

    # Store feature importance
    fold_importance_df = model.get_feature_importance(prettified=True)
    fold_importance_df['fold'] = fold + 1
    feature_importance_list.append(fold_importance_df)

    # Save model
    model_path = os.path.join(MODEL_SAVE_DIR, f"catboost_fold{fold + 1}.cbm")
    model.save_model(model_path)

    # Generate predictions
    y_pred = model.predict(test_pool)
    y_pred_proba = model.predict_proba(test_pool)
    y_pred_classes = y_pred.astype(int).flatten()
    
    # Print fold performance
    print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))

    # Store out-of-fold predictions
    fold_df = X_test.copy()
    fold_df["true_label"] = label_encoder.inverse_transform(y_test)
    fold_df["pred_label"] = label_encoder.inverse_transform(y_pred_classes)
    fold_df["fold"] = fold + 1
    
    # Add prediction probabilities for each class
    for i, class_name in enumerate(label_encoder.classes_):
        fold_df[f"pred_proba_{class_name}"] = y_pred_proba[:, i]
    
    # Add confidence score
    fold_df["pred_confidence"] = y_pred_proba.max(axis=1)
    
    oof_records.append(fold_df)

# Combine all out-of-fold predictions
oof_df = pd.concat(oof_records, ignore_index=True)
oof_df.to_csv(os.path.join(MODEL_SAVE_DIR, "oof_predictions.csv"), index=False)

# Calculate and save feature importance
full_importance_df = pd.concat(feature_importance_list, ignore_index=True)
mean_importance_df = full_importance_df.groupby('Feature Id')['Importances'].mean().sort_values(ascending=False).reset_index()

print("\n--- Average Feature Importances across all folds ---")
print(mean_importance_df)

importance_path = os.path.join(MODEL_SAVE_DIR, "feature_importances.csv")
mean_importance_df.to_csv(importance_path, index=False)
print(f"\n✅ Aggregated feature importances saved to '{importance_path}'")

print(f"\n✅ All CatBoost models, configs, and OOF predictions saved to '{MODEL_SAVE_DIR}'")


# Extract cross-validation predictions for evaluation
cv_preds = oof_df[['publication','dataset_id','pred_label']].copy()
cv_preds.rename(columns={'pred_label':'type_pred','publication':'article_id'}, inplace=True)
cv_preds


# Merge cv predictions with ground truth labels
eval_df = cv_preds.merge(df_labels, on=['article_id', 'dataset_id'], how='inner')

print("=== TYPE-LEVEL EVALUATION (article_id, dataset_id, type level) ===")
print("Evaluating if we correctly predict the specific type of each citation")
print("NB! This is CV result, the 18 false positives were removed and false negative mentions not considered")
print()

# Detailed per-class metrics
print("Per-class Type-level Metrics:")
print(classification_report(eval_df['type'], eval_df['type_pred']))


# Add false positive predictions (predictions with no ground truth labels)
citations_features_labelled_false_pos['type_pred'] = "Not_present"
false_positive_preds = citations_features_labelled_false_pos[['dataset','publication', 'type_pred']]

# Combine true predictions with false positive predictions
cv_preds_full = pd.concat([
    cv_preds, 
    false_positive_preds.rename(columns={'dataset':'dataset_id','publication':'article_id'})
], ignore_index=True)


def calculate_triple_metrics(ground_truth_triples, predicted_triples, label=""):
    """Calculate metrics for exact triple matches (article_id, dataset_id, type)."""
    true_positives = len(ground_truth_triples.intersection(predicted_triples))
    false_positives = len(predicted_triples - ground_truth_triples)
    false_negatives = len(ground_truth_triples - predicted_triples)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

# Create full evaluation by merging predictions with ground truth
# Use outer join to capture all predictions and all ground truth
eval_full = cv_preds_full.merge(df_labels[['article_id', 'dataset_id', 'type']], 
                                on=['article_id', 'dataset_id'], 
                                how='outer', 
                                indicator=True)

# Fill missing values
eval_full['type_pred'] = eval_full['type_pred'].fillna('Not_predicted')
eval_full['type'] = eval_full['type'].fillna('Not_present')

print("=== COMPREHENSIVE EVALUATION METRICS ===")
print(f"Total ground truth mentions: {len(df_labels)}")
print(f"Total predicted mentions: {len(cv_preds_full)}")
print(f"Overlap (both predicted and in ground truth): {len(eval_full[eval_full['_merge'] == 'both'])}")
print(f"Only in predictions: {len(eval_full[eval_full['_merge'] == 'left_only'])}")
print(f"Only in ground truth: {len(eval_full[eval_full['_merge'] == 'right_only'])}")
print()

# EVALUATION METRICS - Match only when article_id, dataset_id AND type all match exactly
print("=== EVALUATION METRICS ===")
print("Match definition: article_id, dataset_id AND type must ALL match exactly")
print()

# Define ground truth and predictions as complete tuples (article_id, dataset_id, type)
ground_truth_full = set(zip(df_labels['article_id'], df_labels['dataset_id'], df_labels['type']))
predicted_full = set(zip(cv_preds_full['article_id'], cv_preds_full['dataset_id'], cv_preds_full['type_pred']))

# Calculate metrics based on exact triple matches
triple_metrics = calculate_triple_metrics(ground_truth_full, predicted_full)

print(f"Total ground truth (article_id, dataset_id, type) triples: {len(ground_truth_full)}")
print(f"Total predicted (article_id, dataset_id, type_pred) triples: {len(predicted_full)}")
print(f"True Positives (exact matches): {triple_metrics['true_positives']}")
print(f"False Positives (wrong predictions): {triple_metrics['false_positives']}")
print(f"False Negatives (missed ground truth): {triple_metrics['false_negatives']}")
print(f"Precision: {triple_metrics['precision']:.4f}")
print(f"Recall: {triple_metrics['recall']:.4f}")
print(f"**F1-score: {triple_metrics['f1_score']:.4f}** <--- This is measured as the competition metric**")
print()

print("=== BREAKDOWN ANALYSIS ===")

# Mention-level analysis (ignoring type)
ground_truth_mentions = set(zip(df_labels['article_id'], df_labels['dataset_id']))
predicted_mentions = set(zip(cv_preds_full['article_id'], cv_preds_full['dataset_id']))

mention_tp = len(ground_truth_mentions.intersection(predicted_mentions))
mention_fp = len(predicted_mentions - ground_truth_mentions)
mention_fn = len(ground_truth_mentions - predicted_mentions)

print(f"Mention-level detection (ignoring type):")
print(f"  Mentions correctly detected: {mention_tp}")
print(f"  Mentions wrongly detected: {mention_fp}")
print(f"  Mentions missed: {mention_fn}")
print()

# DOI-only mention-level analysis
ground_truth_doi = df_labels[df_labels['dataset_id'].str.contains('doi.org', na=False)][['article_id', 'dataset_id']]
predicted_doi = cv_preds_full[cv_preds_full['dataset_id'].str.contains('doi.org', na=False)][['article_id', 'dataset_id']]

ground_truth_doi_mentions = set(zip(ground_truth_doi['article_id'], ground_truth_doi['dataset_id']))
predicted_doi_mentions = set(zip(predicted_doi['article_id'], predicted_doi['dataset_id']))

doi_mention_tp = len(ground_truth_doi_mentions.intersection(predicted_doi_mentions))
doi_mention_fp = len(predicted_doi_mentions - ground_truth_doi_mentions)
doi_mention_fn = len(ground_truth_doi_mentions - predicted_doi_mentions)

print(f"DOI-only mention-level detection:")
print(f"  DOI mentions correctly detected: {doi_mention_tp}")
print(f"  DOI mentions wrongly detected: {doi_mention_fp}")
print(f"  DOI mentions missed: {doi_mention_fn}")
if len(ground_truth_doi_mentions) > 0:
    doi_precision = doi_mention_tp / (doi_mention_tp + doi_mention_fp) if (doi_mention_tp + doi_mention_fp) > 0 else 0
    doi_recall = doi_mention_tp / (doi_mention_tp + doi_mention_fn) if (doi_mention_tp + doi_mention_fn) > 0 else 0
    doi_f1 = 2 * (doi_precision * doi_recall) / (doi_precision + doi_recall) if (doi_precision + doi_recall) > 0 else 0
    print(f"  DOI Precision: {doi_precision:.4f}")
    print(f"  DOI Recall: {doi_recall:.4f}")
    print(f"  **DOI F1-score: {doi_f1:.4f}**")
print()

# DOI-only type-level analysis
ground_truth_doi_full = df_labels[df_labels['dataset_id'].str.contains('doi.org', na=False)]
predicted_doi_full = cv_preds_full[cv_preds_full['dataset_id'].str.contains('doi.org', na=False)]

# Create sets for exact triple matches (article_id, dataset_id, type) for DOI-only
ground_truth_doi_triples = set(zip(ground_truth_doi_full['article_id'], ground_truth_doi_full['dataset_id'], ground_truth_doi_full['type']))
predicted_doi_triples = set(zip(predicted_doi_full['article_id'], predicted_doi_full['dataset_id'], predicted_doi_full['type_pred']))

doi_triple_metrics = calculate_triple_metrics(ground_truth_doi_triples, predicted_doi_triples)

print(f"DOI-only type-level (exact triples):")
print(f"  DOI ground truth (article_id, dataset_id, type) triples: {len(ground_truth_doi_triples)}")
print(f"  DOI predicted (article_id, dataset_id, type_pred) triples: {len(predicted_doi_triples)}")
print(f"  DOI exact matches (all three components): {doi_triple_metrics['true_positives']}")
print(f"  DOI wrong predictions: {doi_triple_metrics['false_positives']} <--- false positive mentions + wrong type predictions")
print(f"  DOI missed ground truth: {doi_triple_metrics['false_negatives']}")
print(f"  DOI Triple Precision: {doi_triple_metrics['precision']:.4f}")
print(f"  DOI Triple Recall: {doi_triple_metrics['recall']:.4f}")
print(f"  **DOI Triple F1-score: {doi_triple_metrics['f1_score']:.4f}")
print()

# Type classification analysis (for correctly detected mentions)
correctly_detected = eval_full[eval_full['_merge'] == 'both'].copy()
if len(correctly_detected) > 0:
    correct_types = len(correctly_detected[correctly_detected['type'] == correctly_detected['type_pred']])
    wrong_types = len(correctly_detected[correctly_detected['type'] != correctly_detected['type_pred']])
    
    print(f"Type classification (for correctly detected mentions):")
    print(f"  Correct type predictions: {correct_types}")
    print(f"  Wrong type predictions: {wrong_types}")
    print(f"  Type accuracy: {correct_types / len(correctly_detected):.4f}")
    print()
    
    print("Per-class Type Performance:")
    print(classification_report(correctly_detected['type'], correctly_detected['type_pred'], zero_division=0))
else:
    print("No correctly detected mentions found for type analysis!")
print()


df_test_doi_citations = df_doi_corpus_enriched[df_doi_corpus_enriched['publication'].isin(test_filenames)].copy()

print(f"Test DOI citations to check: {len(df_test_doi_citations)}")


print("Checking for mentions in test XML and TXT files...")
df_test_doi_citations = find_mentions(df_test_doi_citations.copy(), Path(TEST_XML_PATH), "xml", "publication", "search_dataset")
df_test_doi_citations = find_mentions(df_test_doi_citations.copy(), Path(TEST_TXT_DIR), "txt", "publication", "search_dataset")

# Create any_present column to identify citations found in either XML or TXT
df_test_doi_citations.loc[:, 'any_present'] = df_test_doi_citations[['xml_present', 'txt_present']].any(axis=1)

print("Test mention detection results:")
print(df_test_doi_citations.any_present.value_counts())

# Filter to only citations that are present in the text
test_present_citations = df_test_doi_citations[df_test_doi_citations['any_present'] == True].copy()

print(f"\nTest citations with mentions found: {len(test_present_citations)}")


test_citations_features = test_present_citations.copy()

# Remove duplicate (publication, dataset) pairs if any
test_citations_features = test_citations_features.drop_duplicates(subset=['publication', 'dataset'], keep='first')

# Apply feature engineering functions
test_citations_features = create_temporal_features(test_citations_features)
test_citations_features = create_publisher_journal_features(test_citations_features)
test_citations_features = create_citation_count_features(test_citations_features, citation_pairs)
test_citations_features = create_author_overlap_features(test_citations_features)
test_citations_features = create_dataset_count_features(test_citations_features)

# Remove unnecessary columns (keep article_title and title for similarity features)
test_citations_features.drop(columns=[col for col in columns_to_drop if col in test_citations_features.columns], inplace=True)

print(f"Test features prepared for {len(test_citations_features)} citations")


# Load configurations
with open(os.path.join(MODEL_SAVE_DIR, 'columns_config.json'), 'r') as f:
    config = json.load(f)

# Load label encoder
label_encoder = joblib.load(os.path.join(MODEL_SAVE_DIR, 'label_encoder.pkl'))

# Prepare test features
X_test_final = test_citations_features[config['base_features_columns']].copy()

# Add metadata columns needed for prediction
X_test_final['publication'] = test_citations_features['publication']
X_test_final['dataset_id'] = test_citations_features['dataset']

print(f"Making predictions for {len(X_test_final)} test citations...")

# Create title similarity features for entire test set
print("Creating title similarity features for test data...")

# Use training data to fit vectorizers (to prevent data leakage)
train_X = citations_features_labelled[config['base_features_columns']].copy()

# Create title similarity features
train_similarities, test_similarities = create_title_similarity_features_fold(train_X, X_test_final)
X_test_final['titles_similarity'] = test_similarities

# Create additional title text features
print("Creating title text features for test data...")
(train_char_cosine, train_token_jaccard), (test_char_cosine, test_token_jaccard) = add_title_text_features_fold(train_X, X_test_final)
X_test_final['title_char_cosine'] = test_char_cosine
X_test_final['title_token_jaccard'] = test_token_jaccard

# Handle missing values in categorical columns
for col in config['categorical_columns']:
    X_test_final[col] = X_test_final[col].fillna("missing")

# Get feature columns (excluding non-feature metadata)
feature_columns = [col for col in config['base_features_columns'] if col not in ['article_title', 'title']] + config['title_similarity_features']

# Ensemble predictions from all folds
ensemble_predictions = []
ensemble_probabilities = []

for fold in range(1, N_SPLITS + 1):
    print(f"Loading and predicting with fold {fold} model...")
    
    # Load model
    model_path = os.path.join(MODEL_SAVE_DIR, f"catboost_fold{fold}.cbm")
    model = CatBoostClassifier()
    model.load_model(model_path)
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test_final[feature_columns])
    ensemble_probabilities.append(y_pred_proba)

# Average probabilities across all folds
final_probabilities = np.mean(ensemble_probabilities, axis=0)

# Get final predictions
final_predictions = np.argmax(final_probabilities, axis=1)
final_predicted_labels = label_encoder.inverse_transform(final_predictions)

print(f"Ensemble predictions completed!")
print("Prediction distribution:")
print(pd.Series(final_predicted_labels).value_counts())


# Create submission dataframe
submission_df = pd.DataFrame({
    'article_id': X_test_final['publication'],
    'dataset_id': X_test_final['dataset_id'],
    'type': final_predicted_labels
})

# Add confidence scores for analysis
submission_df['confidence'] = np.max(final_probabilities, axis=1)

# Ensure no duplicates in submission (keep highest confidence prediction)
submission_df = submission_df.sort_values('confidence', ascending=False)
submission_df = submission_df.drop_duplicates(subset=['article_id', 'dataset_id'], keep='first')

# Sort by article_id for consistent ordering
submission_df = submission_df.sort_values(['article_id', 'dataset_id'])

# Add row_id column
submission_df['row_id'] = range(1, len(submission_df) + 1)

# Reorder columns for final submission
submission_final = submission_df[['row_id', 'article_id', 'dataset_id', 'type']].copy()

print(f"Created submission with {len(submission_final)} predictions")
print("\nSubmission summary:")
print(submission_final['type'].value_counts())

# Save final submission file
submission_file = "submission_doi.csv"
submission_final.to_csv(submission_file, index=False)
print(f"\n✅ Final submission saved to '{submission_file}'")

# Save detailed submission with confidence scores for analysis
detailed_submission_file = "submission_doi_detailed.csv"
submission_df[['row_id', 'article_id', 'dataset_id', 'type', 'confidence']].to_csv(detailed_submission_file, index=False)
print(f"✅ Detailed submission with confidence scores saved to '{detailed_submission_file}'")

# Display submission statistics
print(f"\nSubmission Statistics:")
print(f"Total predictions: {len(submission_final)}")
print(f"Unique article-dataset pairs: {len(submission_final)}")  # Should be same as total
print(f"Mean confidence: {submission_df['confidence'].mean():.4f}")
print(f"Min confidence: {submission_df['confidence'].min():.4f}")
print(f"Max confidence: {submission_df['confidence'].max():.4f}")

# Display first few rows
print("\nFirst 10 predictions:")
display(submission_final.head(10))


def f1_score(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0.0


if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    pred_df = pd.read_csv("submission_doi.csv")
    label_df = pd.read_csv(LABELS_PATH)

    # Unadjusted F1 score. We remove the Missing type labels, but not articles containing type Missing
    label_df = label_df[label_df['type'] != 'Missing'].reset_index(drop=True)

    hits_df = label_df.merge(pred_df, on=["article_id", "dataset_id", "type"])

    tp = hits_df.shape[0]
    fp = pred_df.shape[0] - tp
    fn = label_df.shape[0] - tp

    print('This is the basic F1 score on the test predictions. However, remember that it is leaky due to models being trained on this data.')
    print('Use instead the OOF CV results calculated after training the Catboost models.')
    print("TP:", tp)
    print("FP:", fp)
    print("FN:", fn)
    print("F1 Score:", round(f1_score(tp, fp, fn), 3))



# vLLM V1 does not currently accept logits processor so we need to disable it
# https://docs.vllm.ai/en/latest/getting_started/v1_user_guide.html#deprecated-features
os.environ["VLLM_USE_V1"] = "0"

model_path = "/kaggle/input/qwen2.5-coder/transformers/32b-instruct-awq/1"
llm = vllm.LLM(
    model_path,
    quantization='awq',
    tensor_parallel_size=torch.cuda.device_count(),
    gpu_memory_utilization=0.91,
    trust_remote_code=True,
    dtype="half",
    enforce_eager=True,
    max_model_len=5120,
    disable_log_stats=True,
    enable_prefix_caching=True
)
tokenizer = llm.get_tokenizer()


# Load DOI pipeline predictions
pred_dois = pd.read_csv("submission_doi.csv")
doi_articles = pred_dois.article_id.unique()


# Include only test filenames. Note that it is case sensitive. Having case insensitive matching decreased the Public LB score.
df_acc_corpus_filtered = df_acc_corpus[df_acc_corpus.publication.isin(test_filenames)].copy()

# Remove DOI articles, this rised the LB score for us
df_acc_corpus_filtered = df_acc_corpus_filtered[~df_acc_corpus_filtered.publication.isin(doi_articles)].copy()


# Selected repos from the Data Citation Corpus eupmc source
repos = [
    "ArrayExpress", 
    "InterPro", 
    "ChEMBL", 
    "CATH", 
    "Electron Microscopy Public Image Archive (EMPIAR)", 
    "PRIDE Proteomics Identification Database", 
    "NCBI Reference Sequence Database",
    "UniProt", 
    "Ensembl", 
    "Pfam Protein Families", 
    "BioProject",
    "The Protein Data Bank",
    "UniParc",
    "IntAct",
    "Orphadata",
    "Reactome",
    "BioStudies",
    "MetaboLights",
    "RNAcentral",
    "The Electron Microscopy Data Bank (EMDB)",
    "GISAID",
    "NCBI dbGaP",
    "BioModels",
    "BioSample",
    "Gene Expression Omnibus",
    "BioImage Archive",
    "The International Genome Sample Resource",
    "TreeFam",
    "Complex Portal (CP)",
    "MGnify",
    "Molecular INTeraction Database",
    
    # European Nucleotide Archive - we add only some records, see conditions below
]

# Filter for repos
acc_test_preds = (
    df_acc_corpus_filtered[
        (df_acc_corpus_filtered['repository'].isin(repos)) |
        (
            (df_acc_corpus_filtered['repository'] == 'European Nucleotide Archive') & (
                df_acc_corpus_filtered['dataset'].str.match(r'^(SR|STH|CP|KX|BX|CAB|EFO|SCV|VCV|ERR)')
                | df_acc_corpus_filtered['dataset'].str.match(r'^K\d{5}$')
            )
        )
    ][['publication', 'dataset']]
    .rename(columns={'publication': 'article_id', 'dataset': 'dataset_id'})
    .copy()
)


acc_test_preds_pairs = set(zip(acc_test_preds['article_id'], acc_test_preds['dataset_id']))

print(f"Articles in test set: {len(test_filenames)}")
print(f"Predicted mentions: {len(acc_test_preds_pairs)}")


# Repository distribution
if len(acc_test_preds_pairs) > 0:
    corpus_repo_mapping = {}
    for _, row in df_acc_corpus_filtered.iterrows():
        key = (row['publication'], row['dataset'])
        corpus_repo_mapping[key] = row['repository']
    
    corpus_repos = [corpus_repo_mapping.get(pair, 'Unknown') for pair in acc_test_preds_pairs]
    print(f"\nRepository distribution of {len(acc_test_preds_pairs)} new pairs:")
    print(pd.Series(corpus_repos).value_counts())
    
    # Create final export dataframe
    df_acc_test_preds_pairs = pd.DataFrame(list(acc_test_preds_pairs), columns=['article_id', 'dataset_id'])
    print(f"\nCorpus accession ids shape: {df_acc_test_preds_pairs.shape}")
    print(f"Sample pairs:")
    print(df_acc_test_preds_pairs.head())


databases_to_include = [
    "gisaid",
    "arrayexpress",
    "interpro",
    "chembl",
    "bioproject",
    "pfam",
    "ensembl",
    "cellosaurus",
    "cath",
    "empiar",
    "hpa",
    "pxd",
    "biomodels",
    "dbgap",
    "biosample",
    "biostudies",
    "emdb",
    "intact",
    "metabolights",
    "metagenomics",
    "rfam",
    "rnacentral",
    "uniparc",
    "reactome",
    "geo",
    "refseq",
    "refsnp",
]


df_eupmc_original = df_eupmc_original[df_eupmc_original.database.isin(databases_to_include)].copy()
df_eupmc_original.sample(5)


df_eupmc_original.database.value_counts()


df_eupmc_original['article_lower'] = df_eupmc_original['article_id'].str.lower()
test_filenames_lower = [fn.lower() for fn in test_filenames]
df_eupmc_test = df_eupmc_original[df_eupmc_original['article_lower'].isin(test_filenames_lower)]


# Create a mapping from lowercase filenames to original filenames
filename_mapping = {fn.lower(): fn for fn in test_filenames}

# Create a proper copy to avoid the warning
df_eupmc_test = df_eupmc_original[df_eupmc_original['article_lower'].isin(test_filenames_lower)].copy()

# Map the article_lower to the corresponding filename from test_filenames
df_eupmc_test['article_id_in_files'] = df_eupmc_test['article_lower'].map(filename_mapping)

# Check for any unmapped entries
unmapped = df_eupmc_test['article_id_in_files'].isna().sum()
if unmapped > 0:
    print(f"Warning: {unmapped} entries could not be mapped to test filenames")
    
print(f"Mapped {len(df_eupmc_test)} entries")


df_eupmc_test.sample(5)


df_eupmc_test.database.value_counts()


eupmc_repo_mapping = {}
for _, row in df_eupmc_test.iterrows():
    key = (row['article_id_in_files'], row['acc_id'])
    eupmc_repo_mapping[key] = row['database']


df_eupmc_test = df_eupmc_test[['article_id_in_files','acc_id']]


df_eupmc_test = df_eupmc_test[~df_eupmc_test['article_id_in_files'].isin(doi_articles)].copy()
len(df_eupmc_test)


new_pairs_eupmc_added = set(zip(df_eupmc_test['article_id_in_files'], df_eupmc_test['acc_id']))


print(f"Number of acc_test_preds_pairs: {len(acc_test_preds_pairs)}")
print(f"Number of new_pairs_eupmc_added: {len(new_pairs_eupmc_added)}")

union_pairs = new_pairs_eupmc_added | acc_test_preds_pairs
print(f"Number of union pairs: {len(union_pairs)}")

df_all_acc_preds = pd.DataFrame(list(union_pairs), columns=['article_id', 'dataset_id'])
print(f"Final DataFrame shape: {df_all_acc_preds.shape}")
df_all_acc_preds.head()


# Apply the mention detection
print("Checking for mentions in XML and TXT files...")
df_all_acc_preds = find_mentions(df_all_acc_preds.copy(), TEST_XML_PATH, "xml", "article_id", "dataset_id", do_simple=True)
df_all_acc_preds = find_mentions(df_all_acc_preds.copy(), TEST_TXT_DIR, "txt", "article_id", "dataset_id", do_simple=True)

# Create any_present column to identify citations found in either XML or TXT
df_all_acc_preds.loc[:, 'any_present'] = df_all_acc_preds[['xml_present', 'txt_present']].any(axis=1)

print("Mention detection results:")
print(df_all_acc_preds.any_present.value_counts())

# Filter to only citations that are present in the text
df_all_acc_preds_present = df_all_acc_preds[df_all_acc_preds['any_present'] == True].copy()


df_all_acc_preds_present["context"] = df_all_acc_preds_present["txt_context"].fillna(
    df_all_acc_preds_present["xml_context"]
)

print(df_all_acc_preds_present['dataset_id'][0])
print(df_all_acc_preds_present['context'][0])


acc_ids = df_all_acc_preds_present.dataset_id.unique()
df_biosample_current = df_biosample[df_biosample['accession'].isin(acc_ids)].copy()
df_biosample_current.rename(columns={'accession':'dataset_id'},inplace=True)
df_biosample_current = df_biosample_current.drop_duplicates(subset='dataset_id')
df_biosample_current.sample(5)


df_article_metadata_eupmc['article_id_lower'] = df_article_metadata_eupmc['article_id'].str.lower()

df_article_metadata_eupmc = df_article_metadata_eupmc[df_article_metadata_eupmc['article_id_lower'].isin(test_filenames_lower)].copy()
df_article_metadata_eupmc.sample(5)


%%time
SYS_PROMPT_ACCESSION = """
You are given a piece of academic text. Your task is to determine whether the provided Accession ID refers to a dataset used in the study.

Classify the data associated with the Accession ID as:
A) Primary — if the data was generated specifically for this study.
B) Secondary — if the data was reused or derived from prior work.

Respond with only one letter: A or B.
"""

def safe_truncate(text, maxlen):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    return text[:maxlen]

prompts = []
meta_cache = {}
for idx, row in df_all_acc_preds_present.iterrows():
    article_id = row['article_id']
    acc_id = row['dataset_id']
    academic_text = row['txt_context'] if pd.notna(row['txt_context']) else row['xml_context']
    article_context_str, acc_context_str = "", ""

    acc_id_is_samn = isinstance(acc_id, str) and ("SAMN" in acc_id)

    if acc_id_is_samn:
        # Load SAMN metadata and the article metadata for that acc_id-article_id pair
        biosample_match = df_biosample_current[df_biosample_current['dataset_id'] == acc_id]
        if not biosample_match.empty:
            biosample_info = biosample_match.iloc[0]
            submitter_and_date = safe_truncate(biosample_info.get('submitter_and_date', ''), 133)
            acc_context_str += f"\nBioSample submitter and date: {submitter_and_date}"

            # Fetch metadata for this article_id (case-insensitive)
            aid_lower = str(article_id).lower()
            if aid_lower not in meta_cache:
                m = df_article_metadata_eupmc.loc[
                    df_article_metadata_eupmc["article_id_lower"] == aid_lower
                ]
                meta_cache[aid_lower] = (m.iloc[0] if not m.empty else None)
            meta = meta_cache[aid_lower]

            if meta is not None:
                article_doi = str(article_id).replace("_", "/")
                title = safe_truncate(meta.get("title", ""), 372)
                authors = safe_truncate(meta.get("authors", ""), 300)
                journal_title = safe_truncate(meta.get("journal_title", ""), 125)
                pub_year = safe_truncate(meta.get("pub_year", ""), 4)
                article_context_str = (
                    f"Article doi: {article_doi}\n"
                    f"Article title: {title}\n"
                    f"Article authors: {authors}\n"
                    f"Article journal: {journal_title}\n"
                    f"Article year: {pub_year}\n"
                )

    messages = [
        {"role": "system", "content": SYS_PROMPT_ACCESSION},
        {"role": "user",
         "content": f"{article_context_str}Accession ID: {acc_id}\n{acc_context_str}\nAcademic text:\n{academic_text}"}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    prompts.append(prompt)

mclp = MultipleChoiceLogitsProcessor(tokenizer, choices=["A", "B"])

outputs = llm.generate(
    prompts,
    vllm.SamplingParams(
        seed=777,
        temperature=0.1,
        skip_special_tokens=True,
        max_tokens=1,
        logits_processors=[mclp],
        logprobs=len(mclp.choices),
    ),
    use_tqdm=True
)

logprobs = []
for output in outputs:
    first_token_logprobs = output.outputs[0].logprobs[0]
    logprobs.append({lp.decoded_token: lp.logprob for lp in first_token_logprobs.values()})

logit_matrix = pd.DataFrame(logprobs)[["A", "B"]].values
choices = ["Primary", "Secondary"]
answers = [choices[pick] for pick in np.argmax(logit_matrix, axis=1)]


# Create DataFrame for new predictions
acc_sub_df = pd.DataFrame()
acc_sub_df["article_id"] = [row['article_id'] for idx, row in df_all_acc_preds_present.iterrows()]
acc_sub_df["dataset_id"] = [row['dataset_id'] for idx, row in df_all_acc_preds_present.iterrows()]
acc_sub_df["type"] = answers

# Add repository information for grouping
acc_sub_df['corpus_repository'] = acc_sub_df.apply(
    lambda row: corpus_repo_mapping.get((row['article_id'], row['dataset_id']), None), 
    axis=1
)

acc_sub_df['eupmc_repository'] = acc_sub_df.apply(
    lambda row: eupmc_repo_mapping.get((row['article_id'], row['dataset_id']), None), 
    axis=1
)

# Doing type postprocessing on also eupmc lowers the LB score a bit
acc_sub_df['repository_both'] = acc_sub_df['corpus_repository'].fillna(
    acc_sub_df['eupmc_repository']
)

# So we'll do type postprocessing only on the corpus repositories, all eupmc will be "non-corpus"
acc_sub_df['repository'] = acc_sub_df['corpus_repository'].fillna(
    "non-corpus"
)


# Group by article_id and repository and make the type within that to be the most frequent
# For each (article_id, repository) combination, find the most frequent type
most_frequent_types = (
    acc_sub_df.groupby(['article_id', 'repository'])['type']
    .agg(lambda x: x.mode()[0])  # mode()[0] gets the most frequent
    .to_dict()
)

# Update type column so all rows of the same (article_id, repository) get the most frequent type
acc_sub_df['type'] = acc_sub_df.apply(
    lambda row: most_frequent_types.get((row['article_id'], row['repository']), row['type']), 
    axis=1
)

# Add new row_id
acc_sub_df['row_id'] = range(len(acc_sub_df))

# Save to new file
acc_sub_df[["row_id", "article_id", "dataset_id", "type"]].to_csv("submission_acc.csv", index=False)
acc_sub_df[["article_id", "dataset_id", "type", "corpus_repository", "eupmc_repository", "repository"]].to_csv("acc_preds_w_repo.csv", index=False)

print(f"Predictions: {len(acc_sub_df)}")
print(f"Type distribution:")
print(acc_sub_df["type"].value_counts())


acc_sub_df[["article_id", "dataset_id", "type", "corpus_repository", "eupmc_repository", "repository_both"]]


acc_sub_df.repository_both.value_counts()


def f1_from_counts(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    pred_df  = pd.read_csv("submission_acc.csv")
    label_df = pd.read_csv(LABELS_PATH)

    # F1 score
    label_no_missing = label_df[label_df['type'] != 'Missing'].reset_index(drop=True)

    hits_df = label_no_missing.merge(pred_df, on=["article_id", "dataset_id", "type"])
    tp = hits_df.shape[0]
    fp = pred_df.shape[0] - tp
    fn = label_no_missing.shape[0] - tp

    p, r, f1 = f1_from_counts(tp, fp, fn)

    print('F1 (labels with "Missing" removed ONLY; FP likely overestimated and FN includes DOIs)')
    print("TP:", tp, "FP:", fp, "FN:", fn)
    print(f"Precision: {p:.3f}  Recall: {r:.3f}  F1: {f1:.3f}")

    # Adjusted F1

    # Find articles that have any Missing labels
    articles_with_missing = label_df.loc[label_df['type'] == 'Missing', 'article_id'].unique()
    
    # Filter ground truth: remove Missing labels AND articles with any Missing labels
    label_adjusted = label_df[
        (label_df['type'] != 'Missing') & 
        (~label_df.article_id.isin(articles_with_missing))
    ].copy()
    
    # Remove DOI rows from ground truth
    doi_mask = label_adjusted.dataset_id.str.contains("doi.org", case=False, na=False)
    label_adjusted = label_adjusted[~doi_mask].copy()
    
    # Filter predictions to match the same articles as ground truth
    # Only keep predictions from articles that don't have any Missing labels
    pred_adjusted = pred_df[~pred_df.article_id.isin(articles_with_missing)].copy()
    
    # Recompute TP/FP/FN using filtered prediction set
    hits_adj = label_adjusted.merge(pred_adjusted, on=["article_id", "dataset_id", "type"])
    tp_adj = hits_adj.shape[0]
    fp_adj = pred_adjusted.shape[0] - tp_adj  # Use filtered predictions, not original
    fn_adj = label_adjusted.shape[0] - tp_adj
    
    # Precision/Recall/F1
    p_adj, r_adj, f1_adj = f1_from_counts(tp_adj, fp_adj, fn_adj)
    
    print('\nAdjusted F1 (removed articles containing any "Missing" labels and DOIs)')
    print("TP:", tp_adj, "FP:", fp_adj, "FN:", fn_adj)
    print(f"Precision: {p_adj:.3f}  Recall: {r_adj:.3f}  F1: {f1_adj:.3f}")


# File paths
file1 = "submission_acc.csv"
file2 = "submission_doi.csv"

# Read the CSV files
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Merge, drop duplicates, reset index
merged_df = pd.concat([df1, df2], ignore_index=True)
merged_df = merged_df.drop_duplicates().reset_index(drop=True)

# Always create a new row_id column (1-based)
merged_df["row_id"] = range(1, len(merged_df) + 1)

# Reorder so row_id is the first column
cols = ["row_id"] + [c for c in merged_df.columns if c != "row_id"]
merged_df = merged_df[cols]

# Save to CSV
output_path = "submission.csv"
merged_df.to_csv(output_path, index=False)

print("File saved to:", output_path)


def score_f1(sub_df, gt_df, with_type=True, mode="dense_adjusted", debug=False):
    """
    General F1 scorer with two modes:
      - mode="dense_adjusted": Use only densely labeled articles for TP/FN;
        exclude predictions from any article that has Missing labels when counting FP.
      - mode="all_articles":   Use all articles; drop only ground-truth rows with type == 'Missing'.
        (No article filtering; FP is computed over all predictions.)
    """

    def _prep(df, with_type):
        cols = ['article_id', 'dataset_id'] + (['type'] if with_type else [])
        use = df[cols].copy()
        if not with_type:
            use = use.drop(columns=['type'], errors='ignore').drop_duplicates()
        return use

    # Ground truth without Missing labels
    gt_non_missing = gt_df[gt_df['type'] != 'Missing'].copy()

    if mode == "dense_adjusted":
        # Densely labeled article IDs (articles where all GT labels are present / non-Missing)
        dense_articles = gt_non_missing['article_id'].unique()

        # Submissions limited to dense articles for TP/FN
        sub_for_tpfn = sub_df[sub_df['article_id'].isin(dense_articles)].copy()

        # For FP, exclude any article that has Missing labels anywhere in GT
        articles_with_missing = gt_df.loc[gt_df['type'] == 'Missing', 'article_id'].unique()
        sub_for_fp = sub_for_tpfn[~sub_for_tpfn['article_id'].isin(articles_with_missing)].copy()

        gt_use = _prep(gt_non_missing, with_type)
        sub_tpfn_use = _prep(sub_for_tpfn, with_type)
        sub_fp_use = _prep(sub_for_fp, with_type)

        on = ['article_id', 'dataset_id'] + (['type'] if with_type else [])

        # TP
        tp_df = sub_tpfn_use.merge(gt_use, on=on, how='inner')
        tp = len(tp_df)

        # FN
        fn_df = gt_use.merge(sub_tpfn_use, on=on, how='left', indicator=True)
        fn = (fn_df['_merge'] == 'left_only').sum()

        # FP
        fp_df = sub_fp_use.merge(gt_use, on=on, how='left', indicator=True)
        fp = (fp_df['_merge'] == 'left_only').sum()

        if debug:
            print(f"=== {mode} (with_type={with_type}) ===")
            print(f"Dense articles: {len(dense_articles)}")
            print(f"Articles with any Missing labels (excluded for FP): {len(articles_with_missing)}")

    else:  # mode == "all_articles"
        # Use all articles; just remove GT rows with 'Missing'
        gt_use = _prep(gt_non_missing, with_type)
        sub_use = _prep(sub_df, with_type)

        on = ['article_id', 'dataset_id'] + (['type'] if with_type else [])

        # TP
        tp_df = sub_use.merge(gt_use, on=on, how='inner')
        tp = len(tp_df)

        # FN
        fn_df = gt_use.merge(sub_use, on=on, how='left', indicator=True)
        fn = (fn_df['_merge'] == 'left_only').sum()

        # FP (no article exclusions here)
        fp_df = sub_use.merge(gt_use, on=on, how='left', indicator=True)
        fp = (fp_df['_merge'] == 'left_only').sum()

        if debug:
            print(f"=== {mode} (with_type={with_type}) ===")
            print("All articles included; GT rows with 'Missing' removed only.")

    denom = 2 * tp + fp + fn
    f1 = 2 * tp / denom if denom else 0.0

    if debug:
        print(f"Join columns: {on}")
        print(f"TP: {tp} | FP: {fp} | FN: {fn}")
        print(f"F1: {f1:.6f}")

    return {"with_type": with_type, "mode": mode, "tp": tp, "fp": fp, "fn": fn, "f1": f1}


if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    pred_df = pd.read_csv("submission.csv")
    label_df = pd.read_csv(LABELS_PATH)

    # All-articles (remove only Missing GT rows)
    res_all_no_type = score_f1(pred_df, label_df, with_type=False, mode="all_articles", debug=False)
    res_all_with_type = score_f1(pred_df, label_df, with_type=True, mode="all_articles", debug=False)

    # Adjusted dense-only
    res_dense_no_type = score_f1(pred_df, label_df, with_type=False, mode="dense_adjusted", debug=False)
    res_dense_with_type = score_f1(pred_df, label_df, with_type=True, mode="dense_adjusted", debug=False)

    print("\nAll Articles (remove only Missing type GT rows):")
    print(f"  Mention-level (no type): F1={res_all_no_type['f1']:.3f} | "
          f"TP={res_all_no_type['tp']} FP={res_all_no_type['fp']} FN={res_all_no_type['fn']}")
    print(f"  With type:               F1={res_all_with_type['f1']:.3f} | "
          f"TP={res_all_with_type['tp']} FP={res_all_with_type['fp']} FN={res_all_with_type['fn']}")

    print("\nDense-Only Articles (remove from GT all articles that have any Missing type rows):")
    print(f"  Mention-level (no type): F1={res_dense_no_type['f1']:.3f} | "
          f"TP={res_dense_no_type['tp']} FP={res_dense_no_type['fp']} FN={res_dense_no_type['fn']}")
    print(f"  With type:               F1={res_dense_with_type['f1']:.3f} | "
          f"TP={res_dense_with_type['tp']} FP={res_dense_with_type['fp']} FN={res_dense_with_type['fn']}")

    print("\nF1 DIFFS")
    print(f"  Mention-level (no type): {(res_dense_no_type['f1'] - res_all_no_type['f1']):+.3f}")
    print(f"  With type:               {(res_dense_with_type['f1'] - res_all_with_type['f1']):+.3f}")

