# Imports and Constants
import os, re, pathlib
import polars as pl
from lxml import etree
import pymupdf
from typing import Tuple

DOI_URL = 'https://doi.org/'

# Polars verbosity for debugging
pl.Config.set_verbose(True)


# Utilities and Helpers

def is_submission():
    return bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN'))

def is_kaggle_env():
    return (len([k for k in os.environ.keys() if 'KAGGLE' in k]) > 0) or is_submission()

def get_prefix_path(prefix: str) -> pathlib.Path:
    # Use correct directory based on environment
    return pathlib.Path(f'/kaggle/{prefix}' if is_kaggle_env() else f'.{prefix}').expanduser().resolve()

def is_doi(name: str) -> pl.Expr:
    return pl.col(name).str.starts_with(DOI_URL)

def doi_link_to_id(name: str) -> pl.Expr:
    return pl.when(is_doi(name)).then(pl.col(name).str.split(DOI_URL).list.last()).otherwise(name).alias(name)

def doi_id_to_link(name: str, substring: str, url: str = DOI_URL) -> pl.Expr:
    return pl.when(pl.col(name).str.starts_with(substring)).then(url + pl.col(name).str.to_lowercase()).otherwise(name).alias(name)

def score(preds: pl.DataFrame, gt: pl.DataFrame, on: list = ['article_id', 'dataset_id'], verbose: bool = True) -> Tuple[float, float, float]:
    if 'id' in preds.columns and 'dataset_id' not in preds.columns:
        preds = preds.rename({'id': 'dataset_id'})
    hits = gt.join(preds, on=on)
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


# XML & PDF Parsing

def xml_kind(path: pathlib.Path) -> str:
    head = path.open('rb').read(2048).decode('utf8', 'ignore')
    if 'www.tei-c.org/ns' in head:
        return 'tei'
    if re.search(r'(NLM|TaxonX)//DTD', head):
        return 'jats'
    if 'www.wiley.com/namespaces' in head:
        return 'wiley'
    if 'BioC.dtd' in head:
        return 'bioc'
    return 'unknown'

def xml2text(path: pathlib.Path) -> str:
    kind = xml_kind(path)
    root = etree.parse(str(path)).getroot()
    if kind in ('tei', 'bioc', 'unknown'):
        txt = ' '.join(root.itertext())
    elif kind == 'jats':
        elems = root.xpath('//body//sec|//ref-list')
        txt = ' '.join(' '.join(e.itertext()) for e in elems)
    elif kind == 'wiley':
        elems = root.xpath('//*[local-name()="body"]|//*[local-name()="refList"]')
        txt = ' '.join(' '.join(e.itertext()) for e in elems)
    else:
        txt = ' '.join(root.itertext())
    txt = re.sub(r'10\.\d{4,9}/\s+', '10.', txt)
    return txt

def pdf2text(path: pathlib.Path, out_dir: pathlib.Path) -> None:
    doc = pymupdf.open(str(path))
    out = out_dir / f"{path.stem}.txt"
    with open(out, "wb") as f:
        for page in doc:
            f.write(page.get_text().encode("utf8"))
            f.write(b"\n")


# Parse All PDFs & XMLs to TXT
from tqdm.auto import tqdm

def parse_all_pdfs_xmls(pdf_dir, xml_dir, parsed_dir):
    pdf_files = list(pdf_dir.glob('*.pdf'))
    if not pdf_files and not xml_dir.exists():
        raise ValueError("No PDF or XML files found.")

    parsed_dir.mkdir(parents=True, exist_ok=True)

    # PDF â†’ TXT
    for pdf in tqdm(pdf_files, desc="PDFâ†’TXT"):
        try:
            pdf2text(pdf, parsed_dir)
        except Exception as e:
            print(f"PDF error {pdf.stem}: {e}")

    # XML â†’ TXT (append mode)
    if xml_dir.exists():
        for xml in tqdm(xml_dir.glob('*.xml'), desc="XMLâ†’TXT"):
            try:
                txt = xml2text(xml).encode("utf8")
                out = parsed_dir / f"{xml.stem}.txt"
                with open(out, "ab") as f:  # 'ab' = append binary
                    f.write(txt)
                    f.write(b"\n")
            except Exception as e:
                print(f"XML error {xml.stem}: {e}")
    print("Done parsing to text.")


# Extraction Helpers
# This cell defines a regex for extracting dataset IDs from text,
# and a helper function to read in all parsed .txt files as a DataFrame.

import matplotlib.pyplot as plt
import polars as pl
from pathlib import Path

REGEX_IDS = (
    r"(?i)\b(?:"
    r"CHEMBL\d+|"
    r"E-GEOD-\d+|E-PROT-\d+|EMPIAR-\d+|"
    r"ENSBTAG\d+|ENSOARG\d+|"
    r"EPI_ISL_\d{5,}|EPI\d{6,7}|"
    r"HPA\d+|CP\d{6}|IPR\d{6}|PF\d{5}|KX\d{6}|K0\d{4}|"
    r"PRJNA\d+|PXD\d+|SAMN\d+|"
    r"dryad\.[^\s\"<>]+|pasta\/[^\s\"<>]+"
    r")"
)

def get_text_df(parsed_dir: Path) -> pl.DataFrame:
    paths = list(parsed_dir.rglob('*.txt'))
    records = [{'article_id': p.stem, 'text': p.read_text(encoding='utf8')} for p in paths]
    return (
        pl.DataFrame(records)
        .with_columns(
            pl.col("text")
              .str.normalize("NFKC")
              .str.replace_all(r"[^\p{Ascii}]", "")
        )
        .with_columns(
            pl.col("text")
              .str.split(r'\n{2,}')
              .list.eval(pl.col("").str.replace_all('\n', ' '))
              .list.join('\n')
              .alias('text')
        )
        .with_columns([
            pl.col("text")
              .str.slice(pl.col("text").str.len_chars() // 4)
              .str.reverse()
              .alias('rtext'),
            pl.col("text")
              .str.slice(0, pl.col("text").str.len_chars() // 4)
              .alias('ltext'),
        ])
        .with_columns(
            pl.col("rtext")
              .str.find(r'(?i)\b(secnerefer|erutaretil detic|stnemegdelwonkca)\b')
              .alias('ref_idx')
        )
        .with_columns(
            pl.when(pl.col("ref_idx").is_null()).then(0).otherwise(pl.col("ref_idx")).alias("ref_idx")
        )
        .with_columns([
            pl.col("rtext")
              .str.slice(0, pl.col("ref_idx"))
              .str.reverse()
              .alias("refs"),
            (pl.col("ltext") + pl.col("rtext").str.slice(pl.col("ref_idx")).str.reverse()).alias("body")
        ])
        .drop("rtext", "ltext")
    )


import pandas as pd
from collections import Counter

def extract_candidates(args):
    parsed_in = get_prefix_path("working") / args['i']
    print(f"ğŸ”µ Stepâ€¯2: Begin ID Extraction Pipeline")
    print(f"   â†’ Will process parsed text files from: {parsed_in}")
    
    # Start from polars then convert to pandas for further steps
    text_df = get_text_df(parsed_in)
    print(f"ğŸŸ¢ Stepâ€¯1: Loaded text DataFrame")
    print(f"   â†’ Rows: {text_df.height}, Columns: {list(text_df.columns)}")
    print(text_df.with_columns(pl.col("text").str.slice(0, 100).alias("text_snippet")).head(2).to_pandas())

    # Step A: Extract candidate IDs (regex)
    df = text_df.with_columns(pl.col("text").str.extract_all(REGEX_IDS).alias("id")).to_pandas()
    print(f"ğŸŸ¦ [A] Extract candidate IDs")
    print(df[["article_id", "id"]].head(2))

    # Step B: Explode for one row per candidate
    df = df.explode("id").rename(columns={"id": "match_id"})
    print(f"ğŸŸ¦ [B] Exploded IDs")
    print(df[["article_id", "match_id"]].head(2))

    # Step C: Clean IDs
    df["id"] = df["match_id"]
    df["id_nospace"] = df["id"].str.replace(r"\s+", "", regex=True)
    df["id_cleaned"] = df["id_nospace"].str.replace(r"[-.,;:!?/)\]\(\[]+$", "", regex=True)
    print(f"ğŸŸ¦ [C] Cleaned IDs")
    print(df[["article_id", "id", "id_cleaned"]].head(2))

    # Step D: Expand DOIs
    def norm_dryad(x):
        return f"https://doi.org/10.5061/{x.lower()}" if isinstance(x, str) and x.startswith("dryad.") else None
    def norm_pasta(x):
        return f"https://doi.org/10.6073/{x.lower()}" if isinstance(x, str) and x.startswith("pasta/") else None

    df["id_final_dryad"] = df["id_cleaned"].map(norm_dryad)
    df["id_final_pasta"] = df["id_cleaned"].map(norm_pasta)
    print(f"ğŸŸ¦ [D] Normalized DOIs (dryad/pasta)")
    print(df[["article_id", "id_final_dryad", "id_final_pasta"]].head(2))

    # Step E: Prioritize full DOI URL, fallback to cleaned
    df["id_use"] = df["id_final_dryad"].combine_first(df["id_final_pasta"]).combine_first(df["id_cleaned"])
    print(f"ğŸŸ¦ [E] Chose ID to use")
    print(df[["article_id", "id_use"]].head(2))

    # Step F: Filter false positives (Enhanced)
    # -- Drop nulls
    df = df[df["id_use"].notnull()]
    # -- Remove IDs that include the article's own ID
    df = df[~df.apply(lambda row: str(row["article_id"]).replace("_", "/").lower() in str(row["id_use"]).lower(), axis=1)]
    # -- Remove 'figshare'
    df = df[~df["id_use"].str.contains("figshare", na=False)]
    # -- Remove DOIs with short suffixes
    def valid_doi(x):
        if isinstance(x, str) and x.startswith(DOI_URL):
            return len(x.rsplit("/", 1)[-1]) >= 4
        return True
    df = df[df["id_use"].apply(valid_doi)]
    # -- Remove stub DOIs
    STUBS = ["https://doi.org/10.5061/dryad", "https://doi.org/10.6073/pasta", "https://doi.org/10.5281/zenodo"]
    df = df[~df["id_use"].isin(STUBS)]
    # -- Paren/bracket matching
    df = df[df["id_use"].str.count(r"\(") == df["id_use"].str.count(r"\)")]
    df = df[df["id_use"].str.count(r"\[") == df["id_use"].str.count(r"\]")]
    print(f"ğŸŸ¦ [F] Filtered false positives (showing a few):")
    print(df[["article_id", "id_use"]].head(5))

    # Step G: Extract window context and rename
    def get_window(row):
        idx = row["text"].find(row["id_use"])
        if idx == -1:
            return ""
        start = max(idx - args['ws'] - len(str(row["id_use"])), 0)
        end = idx + args['ws'] + len(str(row["id_use"]))
        return row["text"][start:end]
    df["window"] = df.apply(get_window, axis=1)
    df = df[["article_id", "id_use", "window"]].drop_duplicates().rename(columns={"id_use": "dataset_id"})
    print(f"\nâœ… Completed extraction: {len(df)} unique (article_id, dataset_id) pairs")
    return df


# Cell 8: Main Pipeline with Validation Scoring
def main_pipeline():
    args = {
        'i': 'parsed',
        'o': 'extracted_ids.parquet',
        'gt': 'make-data-count-finding-data-references/train_labels.csv',
        'ws': 100
    }

    print("ğŸŒŸ STEP 1: Parse all PDFs and XMLs to text files")
    # -- Dynamically choose train or test split --
    # instead of hard-coding test/PDF and test/XML, 
    # switch between the train and test directories based on the value of is_submission().
    base  = pathlib.Path('/kaggle/input/make-data-count-finding-data-references')
    split = 'test' if is_submission() else 'train'
    pdf_dir    = base / split / 'PDF'
    xml_dir    = base / split / 'XML'
    parsed_dir = get_prefix_path('working') / args['i']
    parse_all_pdfs_xmls(pdf_dir, xml_dir, parsed_dir)

    print("\nğŸŒŸ STEP 2: Extract candidate dataset IDs from text")
    df = extract_candidates(args)
    out_parq = get_prefix_path('working') / args['o']
    df.to_parquet(out_parq)
    print(f"âœ” Saved extracted IDs to: {out_parq} â€” {len(df)} rows")

    # Build submission DataFrame with 'type' column
    def assign_type(x):
        if isinstance(x, str) and (x.startswith(DOI_URL) or x.startswith("SAMN")):
            return "Primary"
        else:
            return "Secondary"

    sub = df.copy()
    sub['type'] = sub['dataset_id'].apply(assign_type)
    sub = (
        sub
        .drop_duplicates(subset=['article_id','dataset_id'])
        .reset_index(drop=True)
    )
    sub['row_id'] = range(len(sub))
    sub = sub[['row_id','article_id','dataset_id','type']]
    print("\n[main_pipeline] Submission DataFrame (first rows):")
    print(sub.head())

    # Write submission.csv
    submission_path = get_prefix_path('working') / 'submission.csv'
    sub.to_csv(submission_path, index=False)
    print(f"âœ” Submission saved â€” {len(sub)} rows to {submission_path}")

    # -- Validate on TRAIN split if available --
    gt_path = pathlib.Path('/kaggle/input/make-data-count-finding-data-references/train_labels.csv')
    if gt_path.exists():
        print("\nğŸ“Š Validation on TRAIN SPLIT")
        preds = pl.read_csv(submission_path).select(['article_id','dataset_id','type'])
        gt    = (
            pl.read_csv(gt_path)
              .filter(pl.col('type')!='Missing')
              .select(['article_id','dataset_id','type'])
        )
        score(preds, gt, on=['article_id','dataset_id','type'])

    print("\nâœ… Pipeline finished!")

# Execute the pipeline
main_pipeline()



def show_submission(sub_csv='/kaggle/working/submission.csv'):
    df = pd.read_csv(sub_csv)
    df = df.reset_index(drop=True)
    df['row_id'] = df.index
    print(df[['row_id', 'article_id', 'dataset_id', 'type']].to_string(index=False))

show_submission()


! rm -rf parsed
! rm -rf src
! rm -rf extracted_ids.parquet

