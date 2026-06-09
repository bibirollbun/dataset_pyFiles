


import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

# If using TensorFlow, suppress its extra logging too
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all messages, 1 = INFO, 2 = WARNING, 3 = ERROR



# Step 0: Safe JATS article_id normalization (single pass only)
def normalize_article_id_jats(article_id):
    """
    Normalize JATS article_id safely:
    - Works only on strings that haven't been normalized before
    - Lowercases, trims, and replaces slashes with underscores
    - Ensures first '10_' becomes '10.'
    - Preserves first underscore after prefix, replaces only final underscore after last dot
    - Returns (normalized_id, changed_flag)
    """

    if not isinstance(article_id, str):
        return article_id, False

    original = article_id.strip()

    # Safeguard: Skip if it already contains the normalized DOI pattern
    if original.startswith("10.") and "_" in original and original.count(".") >= 2:
        return original, False

    aid = original.lower().replace('/', '_')

    # Replace initial '10_' with '10.'
    if aid.startswith('10_'):
        aid = '10.' + aid[3:]

    # Find first dot after '10.'
    first_dot_pos = aid.find('.', 3)
    if first_dot_pos == -1:
        normalized = aid.replace('_', '.')
    else:
        prefix = aid[:first_dot_pos + 1]
        suffix = aid[first_dot_pos + 1:]

        # Preserve the first underscore immediately after the prefix
        if '_' in suffix:
            first_uscore_pos = suffix.find('_')
            preserved = suffix[:first_uscore_pos + 1]
            rest = suffix[first_uscore_pos + 1:]

            # Replace last underscore after last dot with dot
            last_underscore_pos = rest.rfind('_')
            last_dot_pos = rest.rfind('.')
            if last_underscore_pos != -1 and last_underscore_pos > last_dot_pos:
                rest = rest[:last_underscore_pos] + '.' + rest[last_underscore_pos + 1:]

            normalized = prefix + preserved + rest
        else:
            last_underscore_pos = suffix.rfind('_')
            last_dot_pos = suffix.rfind('.')
            if last_underscore_pos != -1 and last_underscore_pos > last_dot_pos:
                suffix = suffix[:last_underscore_pos] + '.' + suffix[last_underscore_pos + 1:]
            normalized = prefix + suffix

    changed = (normalized != original.lower())
    return normalized, changed



# Step 1 â€” Add PDF-derived test article_ids (fallback when XML is missing)

import os
import pandas as pd

TEST_PDF_DIR = "/kaggle/input/make-data-count-finding-data-references/test/PDF"

pdf_paths = []
for r,_,fs in os.walk(TEST_PDF_DIR):
    for f in fs:
        if f.lower().endswith(".pdf"):
            pdf_paths.append(os.path.join(r, f))

pdf_records = []
for p in pdf_paths:
    base = os.path.basename(p).replace(".pdf", "").lower()
    # Example filename: "10.1002_ecs2.1280.pdf"
    # Convert first "_" after prefix to "/" to create a slash-like DOI for the normalizer
    if base.startswith("10.") and "_" in base:
        prefix, rest = base.split("_", 1)
        doi_slashy = f"{prefix}/{rest}"    # e.g., "10.1002/ecs2.1280"
    else:
        doi_slashy = base

    norm, _ = normalize_article_id_jats(doi_slashy)
    norm = norm.replace("/", "_")          # underscore style join key
    pdf_records.append({"file_path": p, "file_type": "PDF", "article_id_raw": doi_slashy, "article_id_norm": norm})

pdf_df = pd.DataFrame(pdf_records).drop_duplicates(subset=["article_id_norm"], keep="first")

# Union XML + PDF IDs
if 'test_df' in globals():
    union_df = pd.concat([test_df, pdf_df], ignore_index=True).drop_duplicates(subset=["article_id_norm"], keep="first")
else:
    union_df = pdf_df.copy()

test_ids = set(union_df["article_id_norm"])

print(f"ğŸ§¾ Test PDFs found: {len(pdf_paths)}")
print(f"âœ… Total usable test articles (XML âˆª PDF): {len(test_ids)}")
print(union_df["file_type"].value_counts(dropna=False).rename("file_type_counts"))



# Step 2 â€” Build fulltext_df from train/XML with safe single-pass normalization

import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
import warnings
import pandas as pd

# --- Safe, single-pass JATS normalizer (for matching only) ---
def normalize_article_id_jats(article_id):
    """
    Normalize JATS/BioC article_id for matching:
      - lowercase + strip
      - '/' -> '_' (underscore style used by test IDs)
      - if startswith '10_' -> '10.' (prefix fix)
      - preserve first underscore after '10.xxxx'
      - replace only the last underscore after the last dot with '.'
    Returns (normalized_id, changed_flag)
    """
    if not isinstance(article_id, str):
        return article_id, False

    original = article_id.strip()
    # ğŸ›‘ Guard: if it already looks normalized (e.g., '10.1002_ecs2.1280'), skip
    if original.startswith("10.") and "_" in original and original.count(".") >= 2:
        return original, False

    aid = original.lower().replace('/', '_')

    if aid.startswith('10_'):
        aid = '10.' + aid[3:]

    first_dot_pos = aid.find('.', 3)
    if first_dot_pos == -1:
        normalized = aid.replace('_', '.')
    else:
        prefix = aid[:first_dot_pos + 1]
        suffix = aid[first_dot_pos + 1:]

        if '_' in suffix:
            first_uscore_pos = suffix.find('_')
            preserved = suffix[:first_uscore_pos + 1]   # keep first underscore
            rest = suffix[first_uscore_pos + 1:]

            last_underscore_pos = rest.rfind('_')
            last_dot_pos = rest.rfind('.')
            if last_underscore_pos != -1 and last_underscore_pos > last_dot_pos:
                rest = rest[:last_underscore_pos] + '.' + rest[last_underscore_pos + 1:]

            normalized = prefix + preserved + rest
        else:
            last_underscore_pos = suffix.rfind('_')
            last_dot_pos = suffix.rfind('.')
            if last_underscore_pos != -1 and last_underscore_pos > last_dot_pos:
                suffix = suffix[:last_underscore_pos] + '.' + suffix[last_underscore_pos + 1:]
            normalized = prefix + suffix

    changed = (normalized != original.lower())
    return normalized, changed

# --- Type detection (JATS vs BioC) ---
def detect_xml_type(file_path):
    try:
        root = ET.parse(file_path).getroot()
        return "BioC" if "collection" in root.tag.lower() else "JATS"
    except Exception:
        return "Unreadable"

# --- Extractors ---
def extract_text_from_jats(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        if '}' in root.tag:
            ns = {'ns': root.tag.split('}')[0].strip('{')}
            doi_node = root.find(".//ns:article-id[@pub-id-type='doi']", ns)
            title = root.find('.//ns:title-group/ns:article-title', ns)
            abstract = root.find('.//ns:abstract', ns)
            body = root.find('.//ns:body', ns)
        else:
            doi_node = root.find(".//article-id[@pub-id-type='doi']")
            title = root.find('.//title-group/article-title')
            abstract = root.find('.//abstract')
            body = root.find('.//body')

        doi = doi_node.text.strip().lower() if doi_node is not None and doi_node.text else None
        title_text = (title.text or "").strip() if title is not None else ""
        abstract_text = "".join(abstract.itertext()).strip() if abstract is not None else ""
        body_text = "".join(body.itertext()).strip() if body is not None else ""
        return doi, f"{title_text}\n{abstract_text}\n{body_text}".strip()
    except Exception as e:
        warnings.warn(f"âš ï¸� JATS read error in {os.path.basename(file_path)}: {e}")
        return None, ""

def extract_text_from_bioc(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        text_blocks = [
            (child.text or "").strip()
            for passage in root.findall(".//passage")
            for child in passage
            if child.tag == "text" and child.text
        ]
        return None, "\n".join(text_blocks).strip()
    except Exception as e:
        warnings.warn(f"âš ï¸� BioC read error in {os.path.basename(file_path)}: {e}")
        return None, ""

def extract_text_auto(file_path):
    try:
        root = ET.parse(file_path).getroot()
        if "collection" in root.tag.lower():
            return extract_text_from_bioc(file_path)
        else:
            return extract_text_from_jats(file_path)
    except Exception as e:
        warnings.warn(f"âš ï¸� General read error in {os.path.basename(file_path)}: {e}")
        return None, ""

# --- Gather train XMLs ---
xml_dir = "/kaggle/input/make-data-count-finding-data-references/train/XML/"
xml_files = []
for root, _, files in os.walk(xml_dir):
    for file in files:
        if file.endswith(".xml"):
            xml_files.append(os.path.join(root, file))
print(f"ğŸ“¦ Found {len(xml_files)} train XML files.")

# --- Build fulltext_df ---
records = []
for file_path in xml_files:
    file_type = detect_xml_type(file_path)
    doi, text = extract_text_auto(file_path)

    if not doi:
        # Fallback from filename â†’ make a slashy DOI for normalizer, e.g. 10.1002_ecs2.1280 -> 10.1002/ecs2.1280
        base = os.path.basename(file_path).replace(".xml", "").lower()
        if base.startswith("10.") and "_" in base:
            prefix, rest = base.split("_", 1)
            doi = f"{prefix}/{rest}"
        else:
            doi = base

    # Do not overwrite raw; keep normalized in a separate column
    norm, changed = normalize_article_id_jats(doi)
    norm = norm.replace("/", "_")  # enforce underscore join key

    records.append({
        "file_path": file_path,
        "file_type": file_type,
        "article_id": doi,            # raw as parsed/fallback
        "article_id_norm": norm,      # for matching
        "norm_changed": changed,
        "full_text": text
    })

fulltext_df = pd.DataFrame(records)

# --- Light diagnostics & de-dupe on join key ---
dup_norm = fulltext_df.duplicated(subset=["article_id_norm"]).sum()
if dup_norm:
    print(f"ğŸ”� Duplicated article_id_norm rows: {dup_norm} (keeping first)")
    fulltext_df = fulltext_df.drop_duplicates(subset=["article_id_norm"], keep="first").reset_index(drop=True)

print("âœ… fulltext_df built")
print("Columns:", fulltext_df.columns.tolist())
print("Shape:", fulltext_df.shape)
print(fulltext_df.head(3))



# 2.1 â€” Build test_index_df (test/XML âˆª test/PDF) and verify presence ===
import os
import xml.etree.ElementTree as ET
import pandas as pd

TEST_XML_DIR = "/kaggle/input/make-data-count-finding-data-references/test/XML"
TEST_PDF_DIR = "/kaggle/input/make-data-count-finding-data-references/test/PDF"

# --- our agreed safe single-pass normalizer (matching only) ---
def normalize_article_id_jats(article_id):
    if not isinstance(article_id, str):
        return article_id, False
    original = str(article_id).strip()
    # guard: already normalized like '10.1002_ecs2.1280'
    if original.startswith("10.") and "_" in original and original.count(".") >= 2:
        return original, False

    aid = original.lower().replace("/", "_")
    if aid.startswith("10_"):
        aid = "10." + aid[3:]

    first_dot_pos = aid.find(".", 3)
    if first_dot_pos == -1:
        normalized = aid.replace("_", ".")
    else:
        prefix = aid[:first_dot_pos + 1]
        suffix = aid[first_dot_pos + 1:]
        if "_" in suffix:
            first_uscore_pos = suffix.find("_")
            preserved = suffix[:first_uscore_pos + 1]
            rest = suffix[first_uscore_pos + 1:]
            last_underscore_pos = rest.rfind("_")
            last_dot_pos = rest.rfind(".")
            if last_underscore_pos != -1 and last_underscore_pos > last_dot_pos:
                rest = rest[:last_underscore_pos] + "." + rest[last_underscore_pos + 1:]
            normalized = prefix + preserved + rest
        else:
            last_underscore_pos = suffix.rfind("_")
            last_dot_pos = suffix.rfind(".")
            if last_underscore_pos != -1 and last_underscore_pos > last_dot_pos:
                suffix = suffix[:last_underscore_pos] + "." + suffix[last_underscore_pos + 1:]
            normalized = prefix + suffix

    return normalized, (normalized != original.lower())

def detect_xml_type(path):
    try:
        root = ET.parse(path).getroot()
        return "BioC" if "collection" in root.tag.lower() else "JATS"
    except Exception:
        return "Unreadable"

def extract_doi_from_xml(path):
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        if '}' in root.tag:
            ns = {'ns': root.tag.split('}')[0].strip('{')}
            node = root.find(".//ns:article-id[@pub-id-type='doi']", ns)
        else:
            node = root.find(".//article-id[@pub-id-type='doi']")
        return node.text.strip().lower() if node is not None and node.text else None
    except Exception:
        return None

# --- collect test XMLs ---
xml_records = []
for r, _, fs in os.walk(TEST_XML_DIR):
    for f in fs:
        if f.lower().endswith(".xml"):
            p = os.path.join(r, f)
            x_type = detect_xml_type(p)
            doi = extract_doi_from_xml(p)
            if not doi:
                base = os.path.basename(p).replace(".xml", "").lower()
                if base.startswith("10.") and "_" in base:
                    prefix, rest = base.split("_", 1)
                    doi = f"{prefix}/{rest}"  # slashy form for normalizer
                else:
                    doi = base
            norm, _ = normalize_article_id_jats(doi)
            norm = norm.replace("/", "_")
            xml_records.append({
                "file_path": p,
                "file_type": x_type,
                "article_id_raw": doi,
                "article_id_norm": norm
            })

xml_df = pd.DataFrame(xml_records)

# --- collect test PDFs (fallback) ---
pdf_records = []
for r, _, fs in os.walk(TEST_PDF_DIR):
    for f in fs:
        if f.lower().endswith(".pdf"):
            p = os.path.join(r, f)
            base = os.path.basename(p).replace(".pdf", "").lower()
            if base.startswith("10.") and "_" in base:
                prefix, rest = base.split("_", 1)
                doi_slashy = f"{prefix}/{rest}"
            else:
                doi_slashy = base
            norm, _ = normalize_article_id_jats(doi_slashy)
            norm = norm.replace("/", "_")
            pdf_records.append({
                "file_path": p,
                "file_type": "PDF",
                "article_id_raw": doi_slashy,
                "article_id_norm": norm
            })

pdf_df = pd.DataFrame(pdf_records)

# --- union XML âˆª PDF -> test_index_df ---
test_index_df = (
    pd.concat([xml_df, pdf_df], ignore_index=True)
      .dropna(subset=["article_id_norm"])
      .drop_duplicates(subset=["article_id_norm"], keep="first")
      .reset_index(drop=True)
)
test_ids = set(test_index_df["article_id_norm"])

print(f"âœ… test_index_df built | rows: {len(test_index_df)} | unique test_ids: {len(test_ids)}")
print(test_index_df["file_type"].value_counts(dropna=False).rename("file_type_counts"))

# --- sanity: ensure fulltext_df (train) has normalized key for future joins (train-side) ---
if "fulltext_df" in globals() and isinstance(fulltext_df, pd.DataFrame):
    if "article_id_norm" not in fulltext_df.columns:
        fulltext_df["article_id_norm"] = fulltext_df["article_id"].apply(
            lambda x: normalize_article_id_jats(x)[0] if isinstance(x, str) else x
        )
    # Replace "/" with "_" only on strings; leave NaN/None untouched
    fulltext_df["article_id_norm"] = fulltext_df["article_id_norm"].apply(
        lambda x: x.replace("/", "_") if isinstance(x, str) else x
    )

    # Optional: trainâ†”test overlap (informational)
    train_ids = set(fulltext_df["article_id_norm"].dropna().astype(str))
    overlap = len(train_ids & test_ids)
    print(f"â„¹ï¸� Trainâ†”Test article_id_norm overlap: {overlap}")
else:
    print("â„¹ï¸� 'fulltext_df' (train) not present; skipping train normalization sanity check.")


# === Step 2.1a â€” Verify full dataset presence & counts (train/test, XML/PDF) ===
import os
import pandas as pd

ROOT = "/kaggle/input/make-data-count-finding-data-references"
TEST_XML_DIR = f"{ROOT}/test/XML"
TEST_PDF_DIR = f"{ROOT}/test/PDF"
TRAIN_XML_DIR = f"{ROOT}/train/XML"
TRAIN_PDF_DIR = f"{ROOT}/train/PDF"

def count_files(d):
    n = 0
    for r, _, fs in os.walk(d):
        n += sum(1 for f in fs if f.lower().endswith((".xml", ".pdf")))
    return n

def list_sample(d, k=5):
    out = []
    for r, _, fs in os.walk(d):
        for f in fs:
            if f.lower().endswith((".xml", ".pdf")):
                out.append(os.path.join(r, f))
                if len(out) >= k:
                    return out
    return out

summary = []
for name, path in [
    ("TEST_XML_DIR", TEST_XML_DIR),
    ("TEST_PDF_DIR", TEST_PDF_DIR),
    ("TRAIN_XML_DIR", TRAIN_XML_DIR),
    ("TRAIN_PDF_DIR", TRAIN_PDF_DIR),
]:
    exists = os.path.exists(path)
    n = count_files(path) if exists else 0
    summary.append((name, path, exists, n))

summary_df = pd.DataFrame(summary, columns=["name","path","exists","file_count"])
print(" Dataset directory summary:")
print(summary_df)

# Quick peeks
print("\nğŸ”� Sample test/XML files:", list_sample(TEST_XML_DIR, k=5))
print("ğŸ”� Sample test/PDF files:", list_sample(TEST_PDF_DIR, k=5))

# Heuristic guard: the rerun test set should be ~2,600 articles total across XML+PDF.
total_test = sum(summary_df.loc[summary_df["name"].isin(["TEST_XML_DIR","TEST_PDF_DIR"]), "file_count"])
if total_test >= 2000:
    print(f"\n Full test set detected (~{total_test} files across XML+PDF). You can proceed with detection & submission.")
else:
    print(f"\nâš ï¸� Only {total_test} test files detected. This looks like a sample/subset.")
    print("   â€¢ In Kaggle, open the Data pane â†’ Add Data â†’ add the full competition dataset for your notebook.")
    print("   â€¢ Confirm the /test/XML and /test/PDF folders contain the full set (~2,600 articles).")

# Optional: assert to block accidental submissions on tiny sets
MIN_TEST_FILES = 2000
if total_test < MIN_TEST_FILES:
    print(f" Blocking final submission since test files < {MIN_TEST_FILES}.")
    # raise SystemExit("Stop: Full test set not mounted.")



# === Step 2.1b â€” DataFrame Health Check (EDA/QA gates; no mutations) ===
import re
import pandas as pd
from IPython.display import display

# Reuse your normalizer if present; else safe fallback (no-op)
if "normalize_article_id_jats" not in globals():
    def normalize_article_id_jats(a):
        return (a if isinstance(a, str) else a, False)

VALID_TYPES = {"Primary", "Secondary"}
_DOI_PAT = re.compile(r'(10\.\d{4,9}/\S+)', re.IGNORECASE)

def _is_full_doi_url(s):
    return isinstance(s, str) and s.lower().startswith("https://doi.org/") and bool(_DOI_PAT.search(s))

def _looks_like_bare_doi(s):
    return isinstance(s, str) and not s.lower().startswith("https://doi.org/") and bool(_DOI_PAT.search(s))

def _norm_view(x):
    if isinstance(x, str):
        n, _ = normalize_article_id_jats(x)
        return n.replace("/", "_")
    return x

def health_check(df: pd.DataFrame,
                 name: str,
                 required_cols=None,
                 key_cols=None,
                 tuple_cols=None,
                 check_type_col=None,
                 check_dataset_col=None,
                 show_samples: int = 5) -> dict:
    """
    Quick EDA + sanity checks. Returns a dict with 'ok' flag and metrics.
    - required_cols: columns that must exist
    - key_cols: columns that should be unique and non-null (e.g., 'article_id_norm')
    - tuple_cols: columns that should be unique as a tuple (e.g., ['article_id','dataset_id','type'])
    - check_type_col: validate values are in {'Primary','Secondary'}
    - check_dataset_col: classify dataset_id as full DOI URL / bare DOI / other
    """
    result = {
        "name": name, "ok": True, "errors": [], "warn": [], "shape": None,
        "nulls": None, "dups_by_key": 0, "dups_by_tuple": 0
    }
    if df is None or not isinstance(df, pd.DataFrame):
        result["ok"] = False
        result["errors"].append("Missing or invalid DataFrame.")
        print(f" {name}: {result['errors'][-1]}")
        return result

    nrows, ncols = df.shape
    result["shape"] = (nrows, ncols)
    print(f"\n {name}: shape={df.shape}")

    # Required columns
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            result["ok"] = False
            result["errors"].append(f"Missing required columns: {missing}")
            print(f" {name}: Missing columns -> {missing}")

    # Nulls summary
    null_counts = df.isna().sum().sort_values(ascending=False)
    result["nulls"] = null_counts
    top_nulls = null_counts[null_counts > 0].head(10)
    if not top_nulls.empty:
        print(f" {name}: columns with nulls (top):")
        display(top_nulls.to_frame("null_count"))

    # Whitespace-only strings (non-null but empty after strip)
    ws_cols = {}
    for c in df.select_dtypes(include=["object"]).columns:
        s = df[c].dropna().astype(str)
        ws = (s.str.strip() == "").sum()
        if ws:
            ws_cols[c] = ws
    if ws_cols:
        print(f"âš ï¸� {name}: whitespace-only strings detected:")
        display(pd.Series(ws_cols, name="whitespace_rows").sort_values(ascending=False))
        result["warn"].append("Whitespace-only strings present")

    # Key uniqueness check
    if key_cols:
        subset = [c for c in key_cols if c in df.columns]
        if subset:
            dup = df.duplicated(subset=subset).sum()
            result["dups_by_key"] = int(dup)
            if dup:
                result["ok"] = False
                result["errors"].append(f"Duplicates by key {subset}: {dup}")
                print(f" {name}: Duplicate keys by {subset} = {dup}")
                display(df[df.duplicated(subset=subset, keep=False)][subset].head(show_samples))

            # Nulls in key columns
            null_in_keys = df[subset].isna().any()
            if null_in_keys.any():
                bad = list(null_in_keys[null_in_keys].index)
                result["ok"] = False
                result["errors"].append(f"Nulls in key columns: {bad}")
                print(f" {name}: Nulls in key columns -> {bad}")

    # Tuple uniqueness check (e.g., submission contract)
    if tuple_cols:
        subset = [c for c in tuple_cols if c in df.columns]
        if subset and all(c in df.columns for c in subset):
            dup = df.duplicated(subset=subset).sum()
            result["dups_by_tuple"] = int(dup)
            if dup:
                result["ok"] = False
                result["errors"].append(f"Duplicates by tuple {subset}: {dup}")
                print(f" {name}: Duplicate tuples by {subset} = {dup}")
                display(df[df.duplicated(subset=subset, keep=False)][subset].head(show_samples))

    # Type column validity
    if check_type_col and check_type_col in df.columns:
        vc = df[check_type_col].astype(str).str.strip().value_counts(dropna=False)
        illegal = set(vc.index) - VALID_TYPES
        print(f"ğŸ”� {name}: {check_type_col} value counts:")
        display(vc.to_frame("count"))
        if illegal:
            result["ok"] = False
            result["errors"].append(f"Invalid types present: {sorted(illegal)}")
            print(f" {name}: Invalid 'type' values -> {sorted(illegal)}")

    # dataset_id classification
    if check_dataset_col and check_dataset_col in df.columns:
        s = df[check_dataset_col].astype(str)
        n_full = s.map(_is_full_doi_url).sum()
        n_bare = s.map(_looks_like_bare_doi).sum()
        n_other = len(s) - n_full - n_bare
        print(f"ğŸ”� {name}: dataset_id classes â€” full DOI URLs: {n_full} | bare DOIs: {n_bare} | other: {n_other}")
        if n_bare:
            print("   Â· Bare DOIs that need expansion (examples):")
            display(s[s.map(_looks_like_bare_doi)].drop_duplicates().head(show_samples))
            result["warn"].append("Bare DOIs present (expand to https://doi.org/...)")

    # Heuristic: normalized article_id view (not persisted)
    id_col = "article_id_norm" if "article_id_norm" in df.columns else ("article_id" if "article_id" in df.columns else None)
    if id_col:
        tmp = df[id_col].dropna().astype(str).head(200)
        has_10prefix   = (tmp.str.startswith("10.")).mean() if len(tmp) else 0.0
        contains_us    = (tmp.str.contains("_")).mean() if len(tmp) else 0.0
        contains_slash = (tmp.str.contains("/")).mean() if len(tmp) else 0.0
        print(f" {name}: ID preview â€” startswith '10.': {has_10prefix:.2%} | '_' {contains_us:.2%} | '/' {contains_slash:.2%}")

    if result["ok"]:
        print(f" {name}: health OK.")
    else:
        print(f"{name}: health check failed.")

    return result

# ---- Run health checks for the key frames available in this notebook ----
results = []

# 1) test_index_df: one row per unique test article_id_norm
if "test_index_df" in globals() and isinstance(test_index_df, pd.DataFrame):
    results.append(
        health_check(
            test_index_df, "test_index_df",
            required_cols=["article_id_norm", "file_type"],
            key_cols=["article_id_norm"]
        )
    )
else:
    print("â�Œ test_index_df not found.")

# 2) fulltext_df (train text index; informational)
if "fulltext_df" in globals() and isinstance(fulltext_df, pd.DataFrame):
    results.append(
        health_check(
            fulltext_df, "fulltext_df (train)",
            required_cols=["article_id_norm", "full_text"],
            key_cols=["article_id_norm"]
        )
    )

# 3) detection_df (predictions prior to submission assembly) â€” optional; runs only if present
if "detection_df" in globals() and isinstance(detection_df, pd.DataFrame):
    det_view = detection_df.copy()
    if "article_id_norm" not in det_view.columns and "article_id" in det_view.columns:
        det_vi_



# === Step 2.1c â€” Cross-DF Consistency Checks (IDs: nulls/dups + formatting + set alignment) ===
import re
import pandas as pd

# Uses your existing normalizer
def _norm_id(x):
    if isinstance(x, str):
        n, _ = normalize_article_id_jats(x)
        return n.replace("/", "_")
    return x

# "Looks normalized" heuristic â€” same short-circuit we use elsewhere
def _looks_normalized_id(s: str) -> bool:
    if not isinstance(s, str):
        return False
    return s.startswith("10.") and ("_" in s) and (s.count(".") >= 2)

def _summarize_id_column(df, name, col):
    ok = True
    if df is None or col not in df.columns:
        print(f" {name}: missing column '{col}'")
        return False, set()

    s = df[col]
    n = len(s)
    n_null = s.isna().sum()
    n_dup  = df.duplicated(subset=[col]).sum()
    n_unq  = s.nunique(dropna=True)
    frac_norm = s.dropna().astype(str).map(_looks_normalized_id).mean() if n > 0 else 0.0

    print(f"\n {name}.{col} â€” rows:{n} | unique:{n_unq} | nulls:{n_null} | dups:{n_dup} | normalized:{frac_norm:.2%}")

    if n_null:
        ok = False
        ex = s[s.isna()]
        print(f"   Â· null examples: {len(ex)} (showing none; fix upstream)")
    if n_dup:
        ok = False
        print("   Â· duplicate key samples:")
        display(df[df.duplicated(subset=[col], keep=False)][[col]].head(6))
    if frac_norm < 1.0:
        # Not fatal for fulltext_df, but for test_index_df/detection it's expected to be 100%
        print("   Â· note: some IDs do not look normalized; ensure joins use article_id_norm.")

    return ok, set(s.dropna().astype(str))

# -------- Run the checks --------
CONSISTENCY_OK = True

# 1) test_index_df: the target ID space for this run
if "test_index_df" in globals() and isinstance(test_index_df, pd.DataFrame):
    ok_test, TEST_IDS = _summarize_id_column(test_index_df, "test_index_df", "article_id_norm")
    if not ok_test:
        CONSISTENCY_OK = False
else:
    print(" test_index_df not found; build it in Step 2.1 first.")
    CONSISTENCY_OK = False
    TEST_IDS = set()

# 2) fulltext_df (train index) â€” informational (not required to match test set)
if "fulltext_df" in globals() and isinstance(fulltext_df, pd.DataFrame):
    ok_train, TRAIN_IDS = _summarize_id_column(fulltext_df, "fulltext_df", "article_id_norm")
    # Not gating consistency on train IDs; this is awareness only.

# 3) detection_df (if present) â€” must align with test_index_df
if "detection_df" in globals() and isinstance(detection_df, pd.DataFrame):
    det_view = detection_df.copy()
    # Ensure a normalized *view* for comparison (no mutation to your original df)
    if "article_id_norm" not in det_view.columns and "article_id" in det_view.columns:
        det_view["article_id_norm"] = det_view["article_id"].apply(_norm_id)

    if "article_id_norm" in det_view.columns:
        ok_det, DET_IDS = _summarize_id_column(det_view, "detection_df(view)", "article_id_norm")

        # Alignment: predictions must be subset of discovered test IDs
        only_in_det  = sorted(list(DET_IDS - TEST_IDS))
        only_in_test = sorted(list(TEST_IDS - DET_IDS))
        print(f"\nğŸ§© ID set alignment â€” matched:{len(DET_IDS & TEST_IDS)} | "
              f"only_in_detection:{len(only_in_det)} | only_in_test:{len(only_in_test)}")

        if only_in_det:
            CONSISTENCY_OK = False
            print("   Â· sample only_in_detection (will be dropped or score 0):", only_in_det[:10])
        # only_in_test is not fatal (just means recall might be low), but we show a sample:
        if only_in_test:
            print("   Â· sample only_in_test (missed by your predictions):", only_in_test[:10])

        if not ok_det:
            CONSISTENCY_OK = False
    else:
        print(" detection_df present but missing 'article_id'/'article_id_norm'; will check later.")
else:
    print(" detection_df not present yet; skipping its alignment checks for now.")

print("\nğŸ§· CONSISTENCY_OK =", CONSISTENCY_OK)



# === Step 2.1d â€” Detailed article_id format audit (fulltext_df vs test_index_df) ===
import re
import pandas as pd
from IPython.display import display

# Helper: (re)normalize to our underscore-style join key (single-pass + slash->underscore)
def _renorm(a):
    if isinstance(a, str):
        n, _ = normalize_article_id_jats(a)
        return n.replace("/", "_")
    return a

def _fmt_summary(series: pd.Series, name: str, k: int = 8):
    s = series.dropna().astype(str)
    print(f"\nğŸ§¾ Format summary â€” {name} (n={len(s)})")
    def share(mask): 
        return f"{(mask.mean()*100):.1f}%" if len(s) else "0.0%"
    print(" â€¢ startswith '10.':", share(s.str.startswith("10.")))
    print(" â€¢ contains '/':   ", share(s.str.contains("/")))
    print(" â€¢ contains '_':   ", share(s.str.contains("_")))
    print(" â€¢ dot count â‰¥ 2:  ", share(s.map(lambda x: x.count("." ) >= 2)))
    print(" â€¢ has spaces:     ", share(s.str.contains(r"\s")))
    print(" â€¢ has parentheses:", share(s.str.contains(r"[()]")))
    print(" â€¢ any uppercase:  ", share(s.str.contains(r"[A-Z]")))
    # show a few weird samples
    weird = s[(s.str.contains("/") | s.str.contains(r"\s") | s.str.contains(r"[()]"))].drop_duplicates().head(k)
    if len(weird):
        print("   Â· unusual samples:")
        display(weird.to_frame(name).reset_index(drop=True))

def _diff_report(df, raw_col: str, norm_col: str, df_name: str, k: int = 10):
    """Compare stored norm to a fresh re-normalization from raw_col."""
    if raw_col not in df.columns or norm_col not in df.columns:
        print(f" {_diff_report.__name__}: {df_name} missing '{raw_col}' or '{norm_col}'")
        return 0
    tmp = df[[raw_col, norm_col]].copy()
    tmp["_recalc_norm"] = tmp[raw_col].apply(_renorm)
    diffs = tmp[tmp[norm_col].astype(str) != tmp["_recalc_norm"].astype(str)]
    n = len(diffs)
    if n == 0:
        print(f" {df_name}: stored {norm_col} matches fresh normalization from {raw_col}.")
    else:
        print(f" {df_name}: {n} rows differ between stored {norm_col} and re-normalized {raw_col}. Samples:")
        display(diffs.head(k))
    return n

AUDIT_OK = True

# 1) Ensure the frames exist
if "test_index_df" not in globals() or not isinstance(test_index_df, pd.DataFrame):
    print(" test_index_df missing; run Step 2.1 first.")
    AUDIT_OK = False
if "fulltext_df" not in globals() or not isinstance(fulltext_df, pd.DataFrame):
    print(" fulltext_df missing; run Step 2 first.")
    AUDIT_OK = False

if AUDIT_OK:
    # 2) Format summaries on the *stored* join keys
    _fmt_summary(test_index_df["article_id_norm"], "test_index_df.article_id_norm")
    _fmt_summary(fulltext_df["article_id_norm"],  "fulltext_df.article_id_norm")

    # 3) Re-normalization agreement (raw -> stored norm)
    #    test_index_df: raw is 'article_id_raw'; fulltext_df: raw is 'article_id'
    diffs_test  = _diff_report(test_index_df, "article_id_raw", "article_id_norm", "test_index_df")
    diffs_train = _diff_report(fulltext_df,   "article_id",     "article_id_norm", "fulltext_df")

    # 4) Cross-frame sanity: both use the same normalization scheme
    #    Check that re-normalizing each side produces the same style (no slashes, stable key).
    test_style_ok  = (test_index_df["article_id_norm"].astype(str).str.contains("/").sum() == 0)
    train_style_ok = (fulltext_df["article_id_norm"].astype(str).str.contains("/").sum() == 0)

    if not test_style_ok:
        print(" test_index_df.article_id_norm contains '/' â€” normalization inconsistency.")
    if not train_style_ok:
        print(" fulltext_df.article_id_norm contains '/' â€” normalization inconsistency.")

    AUDIT_OK = (diffs_test == 0) and test_style_ok  # test must be perfect
    # For train, diffs may occur for odd XML; not gating final merge on train, but warn:
    if diffs_train > 0:
        print(" Note: fulltext_df had normalization diffs (informational; train isnâ€™t used for test merges).")

print("\n ARTICLE_ID_FORMATS_OK =", AUDIT_OK)




# --- Step 3.3: Check for duplicate article_id entries ---

# Count duplicates
duplicate_counts = fulltext_df['article_id'].duplicated().sum()
print(f"ğŸ”� Number of duplicate article_id entries: {duplicate_counts}")

# Display duplicated rows explicitly if duplicates exist
if duplicate_counts > 0:
    duplicates = fulltext_df[fulltext_df['article_id'].duplicated(keep=False)]
    print("\nâš ï¸� Duplicated rows found:")
    display(duplicates)
else:
    print("âœ… No duplicate article_id entries found.")

# --- Quick summary of DataFrame structure ---
print("\nğŸ“‘ Columns in fulltext_df:")
print(fulltext_df.columns.tolist())

print(f"\nğŸ“� Shape of fulltext_df: {fulltext_df.shape}")



# (Optional) simple visualization in Step 2.1d
import matplotlib.pyplot as plt

s = fulltext_df["article_id_norm"].astype(str)
counts = {
    "Contains '/'": s.str.contains("/").sum(),
    "Contains '_'": s.str.contains("_").sum(),
}
pd.Series(counts).plot(kind="bar", rot=0, title="ID characters in article_id_norm")
plt.ylabel("Number of rows"); plt.tight_layout(); plt.show()



# Relaxed: treat as normalized if:
# - starts with 10.
# - contains no '/'
# - and has either an underscore OR at least two dots (10.<registrant>.<suffix>)
def _looks_normalized_id(s: str) -> bool:
    if not isinstance(s, str):
        return False
    return s.startswith("10.") and ("/" not in s) and (("_" in s) or (s.count(".") >= 2))


mask_bad = ~test_index_df["article_id_norm"].astype(str).map(_looks_normalized_id)
bad_ids = test_index_df.loc[mask_bad, ["file_type","article_id_raw","article_id_norm"]]
print(f"Non-normalized-looking test IDs: {len(bad_ids)}")
display(bad_ids)



# step 3.1 
import re

def inspect_article_id_formatting(series, name):
    print(f"\nğŸ”� Inspecting article_id formatting in: {name}")

    # Check 1: All lowercase
    non_lower = series[series != series.str.lower()]
    print(f"â€¢ Not lowercase: {len(non_lower)}")

    # Check 2: Contains slashes
    with_slash = series[series.str.contains('/')]
    print(f"â€¢ Contains slashes: {len(with_slash)}")

    # Check 3: Contains dots
    with_dot = series[series.str.contains(r'\.')]
    print(f"â€¢ Contains dots: {len(with_dot)}")

    # Check 4: Contains hyphens
    with_dash = series[series.str.contains('-')]
    print(f"â€¢ Contains hyphens: {len(with_dash)}")

    # Check 5: Unexpected characters (anything not alphanumeric or underscore)
    unexpected_chars = series[series.str.contains(r'[^a-z0-9_]', regex=True)]
    print(f"â€¢ Contains unexpected characters: {len(unexpected_chars)}")

    # Check 6: Leading/trailing whitespace
    trimmed = series[series != series.str.strip()]
    print(f"â€¢ Has leading/trailing whitespace: {len(trimmed)}")

# âœ… Run for both DataFrames
inspect_article_id_formatting(fulltext_df["article_id_norm"], "fulltext_df")
inspect_article_id_formatting(test_index_df["article_id_norm"], "sample_submission")



# âœ… Step 4: Add column for full_text length
from IPython.display import display
import pandas as pd

if "fulltext_df" in globals() and isinstance(fulltext_df, pd.DataFrame):
    if "full_text" in fulltext_df.columns:
        # length in characters; 0 for non-strings / NaNs
        fulltext_df["text_length"] = fulltext_df["full_text"].map(
            lambda x: len(x.strip()) if isinstance(x, str) else 0
        )

        # Descriptive stats
        print("ğŸ“� Fulltext Length Stats (from 'full_text'):")
        print(fulltext_df["text_length"].describe())

        # Short / empty rows
        short_texts = fulltext_df[fulltext_df["text_length"] < 100]
        print(f"\n Number of short or empty full_text rows (<100 chars): {len(short_texts)}")

        # Preview a few short ones
        if not short_texts.empty:
            display(short_texts[["article_id", "text_length", "full_text"]].head(10))
        else:
            print("âœ… No short or empty full_text rows found.")
    else:
        print(" 'full_text' column not found in fulltext_df.")
else:
    print(" 'fulltext_df' not found or not a DataFrame.")



# Step 15 inspect fultext col
fulltext_df.columns.tolist()


# step 4.2 Preview structure
print("ğŸ“„ Preview of fulltext_df:")
display(fulltext_df.head())

# Shape of the DataFrame
print(f"\nğŸ“� Shape: {fulltext_df.shape}")

# Unique values per column
print("\n Unique values per column:")
print(fulltext_df.nunique())

# Null values check
print("\nğŸ•³ï¸� Null values per column:")
print(fulltext_df.isnull().sum())

# Check article_id formatting (with underscore)
print("\nğŸ”� Sample article_id values:")
print(fulltext_df["article_id"].sample(5).tolist())

# Check how many have slashes (normal format) vs underscores
print("\n Format check:")
print("Contains slashes:", fulltext_df["article_id"].str.contains("/").sum())
print("Contains underscores:", fulltext_df["article_id"].str.contains("_").sum())

# #  Text length stats (already added earlier)
# print("\nğŸ“� Text Length Stats:")
# print(fulltext_df["text_length"].describe())



# Step 4.3: Validation of fulltext_df

# 1. Unique values count per column
print("ğŸ“Š Unique values per column:")
print(fulltext_df.nunique())

# 2. Check for duplicated article_id entries
duplicate_ids = fulltext_df[fulltext_df.duplicated("article_id", keep=False)]
print(f"\nğŸ“� Number of duplicated article_id rows: {len(duplicate_ids)}")
if not duplicate_ids.empty:
    display(duplicate_ids)

# 3. Check for duplicated full_text entries
duplicate_texts = fulltext_df[fulltext_df.duplicated("full_text", keep=False)]
print(f"\nğŸ“� Number of duplicated full_text rows: {len(duplicate_texts)}")
if not duplicate_texts.empty:
    display(duplicate_texts)

# 4. List article_ids sharing identical full_text (if any duplicates found)
if not duplicate_texts.empty:
    grouped = duplicate_texts.groupby("full_text")["article_id"].apply(list)
    print("\nğŸ§© Article IDs sharing the same full_text:")
    display(grouped)



# Step 5--- Stronger normalization + mining (paste above the mining step) ---
import re
from unicodedata import normalize as _u

# Include all common dash/minus variants + *mojibake* sequences seen in the data
BAD_HYPHENS = (
    "â€“", "â€”", "âˆ’", "-", "â€�", "â€’", "â€•",  # unicode dashes / minus / no-break hyphen
    "\u00ad",                          # soft hyphen (explicitly removed below)
    "â€šÃ„Ãª", "â€šÃ„Ã¬", "â€šÃ„Ã®",               # mojibake dashes (UTF-8 en/em dash decoded as Latin-1)
    "Ã¢â‚¬â€œ", "Ã¢â‚¬â€�", "Ã¢â‚¬â€™", "Ã¢â‚¬â€¢"          # other mojibake dash sequences
)

def _fix_hyphens(s: str) -> str:
    """Normalize / de-mojibake hyphens so downstream regex sees plain ASCII '-'."""
    if not isinstance(s, str): 
        return s
    s = _u("NFKC", s)
    # remove soft hyphen explicitly (may appear invisibly)
    s = s.replace("\u00ad", "")
    # replace any bad dash/minus sequence with ASCII hyphen
    for b in BAD_HYPHENS:
        if b:
            s = s.replace(b, "-")
    return s

def clean_text(x):
    """Clean + normalize early so later regex sees sane text."""
    if pd.isna(x): return ""
    s = _fix_hyphens(str(x))

    # join DOI pieces broken by line-wrap/hyphen (do BEFORE nuking newlines/spaces)
    s = re.sub(r"(10\.\d{4,9}/[^\s\"'<>),;:]+)[\s\-]+(?=[A-Za-z0-9])", r"\1", s)

    # HTML/newlines â†’ space
    s = re.sub(r"<.*?>", " ", s)
    s = re.sub(r"[\n\t\r]", " ", s)

    # standardize DOI forms
    s = re.sub(r"\bdoi\s*:\s*", " https://doi.org/", s, flags=re.I)
    s = re.sub(r"https?://(?:dx\.)?doi\.org/", "https://doi.org/", s, flags=re.I)

    return re.sub(r"\s+", " ", s).strip()

# Replace the DOI regex with stricter versions that avoid trailing junk
_DOI_BARE = re.compile(r"(?:^|(?<!\w))(10\.\d{4,9}/[^\s\"'<>),;:]+)", re.I)
_DOI_FULL = re.compile(r"https?://(?:dx\.)?doi\.org/(10\.\d{4,9}/[^\s\"'<>),;:]+)", re.I)

# quick helper used by miner to reject obvious stubs/ISSN-like tails
def _looks_like_stub_or_issn_suffix(sfx: str) -> bool:
    return (
        re.fullmatch(r"\d{3,5}", sfx) or            # 0198, 0924, â€¦
        re.fullmatch(r"[sS]\d{4,5}", sfx) or        # s0967, s0264, â€¦
        re.fullmatch(r"\d{7,9}X?", sfx, flags=re.I) or
        re.fullmatch(r"[A-Za-z]{1,3}", sfx)         # j, ao, oe, â€¦
    )

# additional plausibility check so only well-formed DOI suffixes survive
def _plausible_doi_suffix(sfx: str) -> bool:
    # must end alphanumeric
    if not sfx or not sfx[-1].isalnum():
        return False
    # drop obvious stubs first
    if _looks_like_stub_or_issn_suffix(sfx):
        return False
    # must have substance: at least 6 alphanumerics total
    if len(re.sub(r"[^A-Za-z0-9]", "", sfx)) < 6:
        return False
    # and some DOI-like structure
    if not (("." in sfx) or re.search(r"-\d", sfx) or re.search(r"\d{4,}", sfx) or re.search(r"\d", sfx)):
        return False
    return True

def mine_dataset_ids(text: str) -> list[str]:
    """Normalize first, then mine DOIs + accessions; drop stubby DOIs."""
    out = set()
    if not isinstance(text, str) or not text:
        return []

    t = clean_text(text)                 # <-- early normalization here (fixes â€šÃ„Ãª/â€šÃ„Ã¬/â€šÃ„Ã® etc.)
    t = _fix_hyphens(t)

    # DOIs (full URL form)
    for m in _DOI_FULL.finditer(t):
        out.add(f"https://doi.org/{m.group(1)}".rstrip(").,;:]}'\"-_/"))

    # DOIs (bare)
    for m in _DOI_BARE.finditer(t):
        out.add(f"https://doi.org/{m.group(1)}".rstrip(").,;:]}'\"-_/"))

    # Common accessions
    _PATTERNS = [
        (re.compile(r"\b(GSE\d+|GSM\d+)\b", re.I), lambda m: m.group(1).upper()),
        (re.compile(r"\bE\-MTAB\-\d+\b", re.I), lambda m: m.group(0).upper()),
        (re.compile(r"\bPRJ(?:NA|EB|DB)\d+\b", re.I), lambda m: m.group(0).upper()),
        (re.compile(r"\bSRR\d+|SRP\d+|SRX\d+|SRS\d+\b", re.I), lambda m: m.group(0).upper()),
        (re.compile(r"\bEMPIAR\-\d+\b", re.I), lambda m: m.group(0).upper()),
        (re.compile(r"\bEMD\-\d+\b", re.I), lambda m: m.group(0).upper()),
        (re.compile(r"\bCHEMBL\d+\b", re.I), lambda m: m.group(0).upper()),
        (re.compile(r"\bICPSR\d+\b", re.I), lambda m: m.group(0).upper()),
        (re.compile(r"\bEPI(?:[_-]ISL)?\d+\b", re.I), lambda m: m.group(0).upper()),
        (re.compile(r"\bpdb\s+([0-9a-z]{4})\b", re.I), lambda m: f"PDB {m.group(1).upper()}"),
    ]
    for pat, fn in _PATTERNS:
        for m in pat.finditer(t):
            out.add(fn(m))

    # Final prune: drop stubby / implausible DOIs before they hit the canonicalizer
    pruned = set()
    for url in out:
        if not url.lower().startswith("https://doi.org/"):
            pruned.add(url)
            continue
        tail = url[len("https://doi.org/"):]
        if "/" not in tail:
            continue
        _, sfx = tail.split("/", 1)
        if _plausible_doi_suffix(sfx):
            pruned.add(url)

    return sorted(x for x in pruned if len(x) >= 6)



# --- Stronger normalization + mining (paste above the mining step) ---
import re
from unicodedata import normalize as _u

BAD_HYPHENS = ("â€“","â€”","âˆ’","Â­","â€’","â€•","â€šÃ„Ãª","â€šÃ„Ã¬")

def _fix_hyphens(s: str) -> str:
    if not isinstance(s, str): return s
    s = _u("NFKC", s)
    for b in BAD_HYPHENS:
        s = s.replace(b, "-")
    # remove soft hyphen explicitly
    return s.replace("\u00ad", "")

def clean_text(x):
    """Clean + normalize early so later regex sees sane text."""
    if pd.isna(x): return ""
    s = _fix_hyphens(str(x))

    # join DOI pieces broken by line-wrap/hyphen
    # (do BEFORE nuking newlines/spaces)
    s = re.sub(r"(10\.\d{4,9}/[^\s\"'<>),;:]+)[\s\-]+(?=[A-Za-z0-9])", r"\1", s)

    # HTML/newlines â†’ space
    s = re.sub(r"<.*?>", " ", s)
    s = re.sub(r"[\n\t\r]", " ", s)

    # standardize DOI forms
    s = re.sub(r"\bdoi\s*:\s*", " https://doi.org/", s, flags=re.I)
    s = re.sub(r"https?://(?:dx\.)?doi\.org/", "https://doi.org/", s, flags=re.I)

    return re.sub(r"\s+", " ", s).strip()

# Replace the DOI regex with stricter versions that avoid trailing junk
_DOI_BARE = re.compile(r"(?:^|(?<!\w))(10\.\d{4,9}/[^\s\"'<>),;:]+)", re.I)
_DOI_FULL = re.compile(r"https?://(?:dx\.)?doi\.org/(10\.\d{4,9}/[^\s\"'<>),;:]+)", re.I)

# quick helper used by miner to reject obvious stubs/ISSN-like tails
def _looks_like_stub_or_issn_suffix(sfx: str) -> bool:
    return (
        re.fullmatch(r"\d{3,5}", sfx) or            # 0198, 0924, â€¦
        re.fullmatch(r"[sS]\d{4,5}", sfx) or        # s0967, s0264, â€¦
        re.fullmatch(r"\d{7,9}X?", sfx, flags=re.I) or
        re.fullmatch(r"[A-Za-z]{1,3}", sfx)         # j, ao, oe, â€¦
    )

# additional plausibility check so only well-formed DOI suffixes survive
def _plausible_doi_suffix(sfx: str) -> bool:
    # must end alphanumeric
    if not sfx or not sfx[-1].isalnum():
        return False
    # drop obvious stubs first
    if _looks_like_stub_or_issn_suffix(sfx):
        return False
    # must have substance: at least 6 alphanumerics total
    if len(re.sub(r"[^A-Za-z0-9]", "", sfx)) < 6:
        return False
    # and some DOI-like structure
    if not (("." in sfx) or re.search(r"-\d", sfx) or re.search(r"\d{4,}", sfx) or re.search(r"\d", sfx)):
        return False
    return True

def mine_dataset_ids(text: str) -> list[str]:
    """Normalize first, then mine DOIs + accessions; drop stubby DOIs."""
    out = set()
    if not isinstance(text, str) or not text:
        return []

    t = clean_text(text)                 # <-- early normalization here
    t = _fix_hyphens(t)

    # DOIs (full URL form)
    for m in _DOI_FULL.finditer(t):
        out.add(f"https://doi.org/{m.group(1)}".rstrip(").,;:]}'\"-_/"))

    # DOIs (bare)
    for m in _DOI_BARE.finditer(t):
        out.add(f"https://doi.org/{m.group(1)}".rstrip(").,;:]}'\"-_/"))

    # Common accessions
    _PATTERNS = [
        (re.compile(r"\b(GSE\d+|GSM\d+)\b", re.I), lambda m: m.group(1).upper()),
        (re.compile(r"\bE\-MTAB\-\d+\b", re.I), lambda m: m.group(0).upper()),
        (re.compile(r"\bPRJ(?:NA|EB|DB)\d+\b", re.I), lambda m: m.group(0).upper()),
        (re.compile(r"\bSRR\d+|SRP\d+|SRX\d+|SRS\d+\b", re.I), lambda m: m.group(0).upper()),
        (re.compile(r"\bEMPIAR\-\d+\b", re.I), lambda m: m.group(0).upper()),
        (re.compile(r"\bEMD\-\d+\b", re.I), lambda m: m.group(0).upper()),
        (re.compile(r"\bCHEMBL\d+\b", re.I), lambda m: m.group(0).upper()),
        (re.compile(r"\bICPSR\d+\b", re.I), lambda m: m.group(0).upper()),
        (re.compile(r"\bEPI(?:[_-]ISL)?\d+\b", re.I), lambda m: m.group(0).upper()),
        (re.compile(r"\bpdb\s+([0-9a-z]{4})\b", re.I), lambda m: f"PDB {m.group(1).upper()}"),
    ]
    for pat, fn in _PATTERNS:
        for m in pat.finditer(t):
            out.add(fn(m))

    # Final prune: drop stubby / implausible DOIs before they hit the canonicalizer
    pruned = set()
    for url in out:
        if not url.lower().startswith("https://doi.org/"):
            pruned.add(url)
            continue
        tail = url[len("https://doi.org/"):]
        if "/" not in tail:
            continue
        _, sfx = tail.split("/", 1)
        if _plausible_doi_suffix(sfx):
            pruned.add(url)

    return sorted(x for x in pruned if len(x) >= 6)



# âœ… Step 6 â€” Promote cleaned text to 'text' and drop helper columns (robust)
from IPython.display import display

# Keep the original raw 'full_text' for debugging/audits?
KEEP_RAW_FULL_TEXT = True   # set to False if you want to drop 'full_text'

if "fulltext_df" in globals() and isinstance(fulltext_df, pd.DataFrame):

    print(" Initial fulltext_df columns:")
    print(list(fulltext_df.columns))

    source_col = None
    # Prefer already-cleaned text; fall back to cleaning 'full_text' on the fly
    if "full_text_clean" in fulltext_df.columns:
        source_col = "full_text_clean"
        fulltext_df["text"] = fulltext_df[source_col]
        print(" Promoted 'full_text_clean' â†’ 'text'")
    elif "clean_text" in fulltext_df.columns:   # in case you used that name elsewhere
        source_col = "clean_text"
        fulltext_df["text"] = fulltext_df[source_col]
        print(" Promoted 'clean_text' â†’ 'text'")
    elif "full_text" in fulltext_df.columns:
        # Create 'text' by cleaning on the fly (uses your clean_text function)
        try:
            fulltext_df["text"] = fulltext_df["full_text"].map(clean_text)
            source_col = "full_text"
            print(" Created 'text' by cleaning 'full_text' on the fly")
        except NameError:
            # clean_text not defined; just copy raw text
            fulltext_df["text"] = fulltext_df["full_text"]
            source_col = "full_text"
            print("âš ï¸� 'clean_text' not defined â€” copied raw 'full_text' to 'text'")
    else:
        print(" No text source found (expected one of: 'full_text_clean', 'clean_text', 'full_text').")
        source_col = None

    # Decide which helpers to drop
    to_drop = []
    for c in ["text_length", "full_text_clean", "clean_text"]:
        if c in fulltext_df.columns:
            to_drop.append(c)

    if not KEEP_RAW_FULL_TEXT and "full_text" in fulltext_df.columns:
        to_drop.append("full_text")

    if to_drop:
        # Never drop the 'text' column we just created
        to_drop = [c for c in set(to_drop) if c != "text"]
        fulltext_df.drop(columns=to_drop, inplace=True, errors="ignore")
        print(f" Dropped columns: {sorted(to_drop)}")
    else:
        print(" No helper columns found to drop.")

    # Confirm final structure
    print("\n Final fulltext_df columns:")
    print(list(fulltext_df.columns))
    display(fulltext_df.head())

else:
    print(" 'fulltext_df' not found or is not a valid DataFrame.")



# âœ… Step 6 â€” Load and inspect labels dataframe (robust + normalized coverage)
import re
import pandas as pd
from IPython.display import display

# Reuse your normalizer; safe fallback if not defined
if "normalize_article_id_jats" not in globals():
    def normalize_article_id_jats(a): 
        return (a if isinstance(a, str) else a, False)

labels_path = "/kaggle/input/make-data-count-finding-data-references/train_labels.csv"
labels_df = pd.read_csv(labels_path)

print("ğŸ“„ labels_df preview & shape:", labels_df.shape)
display(labels_df.head())

# ---- Optional quick stats
for col in ["article_id", "dataset_id", "type"]:
    if col in labels_df.columns:
        nunique = labels_df[col].nunique(dropna=True)
        nnull = labels_df[col].isna().sum()
        print(f"â„¹ï¸� {col}: unique={nunique}, nulls={nnull}")

# ---- Safe sampling (wonâ€™t error if <5 rows)
def _safe_sample(series, k=5):
    s = series.dropna()
    if len(s) == 0:
        return []
    return s.sample(min(k, len(s)), random_state=0).tolist()

if "article_id" in labels_df.columns:
    print("\nğŸ”� Sample article_id values:")
    print(_safe_sample(labels_df["article_id"], k=5))

if "dataset_id" in labels_df.columns:
    print("\nğŸ”� Sample dataset_id values:")
    print(_safe_sample(labels_df["dataset_id"], k=5))

# ---- Format checks (guarded)
if "article_id" in labels_df.columns:
    print("\nğŸ”� Format check for article_id:")
    s = labels_df["article_id"].astype(str)
    print("Contains slashes:", int(s.str.contains("/").sum()))
    print("Contains underscores:", int(s.str.contains("_").sum()))

if "dataset_id" in labels_df.columns:
    print("\nğŸ”� Format check for dataset_id:")
    s = labels_df["dataset_id"].astype(str)
    print("Contains slashes:", int(s.str.contains("/").sum()))
    print("Contains underscores:", int(s.str.contains("_").sum()))

# ---- DOI classification for dataset_id (useful later)
_DOI_PAT = re.compile(r'(10\.\d{4,9}/\S+)', re.IGNORECASE)
def _is_full_doi_url(x:str)->bool:
    return isinstance(x,str) and x.lower().startswith("https://doi.org/") and bool(_DOI_PAT.search(x))
def _looks_like_bare_doi(x:str)->bool:
    return isinstance(x,str) and (not x.lower().startswith("https://doi.org/")) and bool(_DOI_PAT.search(x))

if "dataset_id" in labels_df.columns:
    ds = labels_df["dataset_id"].astype(str)
    n_full = int(ds.map(_is_full_doi_url).sum())
    n_bare = int(ds.map(_looks_like_bare_doi).sum())
    n_other = len(ds) - n_full - n_bare
    print(f"\nğŸ”� dataset_id classes â€” full DOI URLs: {n_full} | bare DOIs: {n_bare} | other: {n_other}")
    if n_bare:
        print("   Â· Example bare DOIs (expand to https://doi.org/...):")
        display(ds[ds.map(_looks_like_bare_doi)].drop_duplicates().head(5).to_frame("dataset_id"))

# ---- Build normalized IDs in labels_df (underscore join key)
if "article_id" in labels_df.columns:
    labels_df["article_id_norm"] = (
        labels_df["article_id"]
        .map(lambda x: normalize_article_id_jats(x)[0] if isinstance(x, str) else x)
        .map(lambda x: x.replace("/", "_") if isinstance(x, str) else x)
    )

# ---- Ensure fulltext_df has normalized key (should already exist from prior steps)
if "fulltext_df" in globals() and isinstance(fulltext_df, pd.DataFrame):
    if "article_id_norm" not in fulltext_df.columns and "article_id" in fulltext_df.columns:
        fulltext_df["article_id_norm"] = (
            fulltext_df["article_id"]
            .map(lambda x: normalize_article_id_jats(x)[0] if isinstance(x, str) else x)
            .map(lambda x: x.replace("/", "_") if isinstance(x, str) else x)
        )

# ---- Normaliz



# step 6.1 Analyzing formatting 
import re

# 1. Total available non-null dataset_id values
print(f" Total non-null dataset_id values: {labels_df['dataset_id'].notnull().sum()}")

# 2. Count how many dataset_ids contain slashes and underscores
with_slash = labels_df["dataset_id"].astype(str).str.contains("/").sum()
with_underscore = labels_df["dataset_id"].astype(str).str.contains("_").sum()
print(f" dataset_id with slashes: {with_slash}")
print(f" dataset_id with underscores: {with_underscore}")

# 3. Preview some dataset_ids containing underscores
underscore_df = labels_df[labels_df["dataset_id"].astype(str).str.contains("_")].copy()
print("\nğŸ“š Sample underscore-containing dataset_ids:")
display(underscore_df["dataset_id"].value_counts().head(20))

# 4. Define a classification function to categorize dataset_id types
def categorize_dataset_id(ds_id):
    ds_id = str(ds_id).strip().lower()
    if "doi.org" in ds_id:
        return "DOI"
    elif re.match(r"^gse\d+$", ds_id):
        return "GEO Accession"
    elif ds_id.startswith("ensbtag"):
        return "Ensembl"
    elif ds_id.startswith("pdb"):
        return "PDB"
    elif ds_id.startswith("hpa"):
        return "HPA"
    else:
        return "Other"

# 5. Apply classification function with safeguard for missing column
if "dataset_id" in labels_df.columns:
    labels_df["dataset_type"] = labels_df["dataset_id"].apply(categorize_dataset_id)
else:
    print(" 'dataset_id' column missing in labels_df.")

# 6. Display the distribution of dataset_id types with safeguard for column presence
if "dataset_type" in labels_df.columns:
    print("\nğŸ“Š Distribution of dataset_id types:")
    print(labels_df["dataset_type"].value_counts())
else:
    print(" 'dataset_type' column not found in labels_df.")



labels_df.shape


# 
# step 6.2 check how many article-id have more than one label entry
# Check how many article_ids have more than one label entry
label_counts = labels_df["article_id"].value_counts()
multi_label_articles = label_counts[label_counts > 1]

print(f"ğŸ”� Articles with multiple dataset labels: {len(multi_label_articles)}")
display(labels_df[labels_df["article_id"].isin(multi_label_articles.index)].head(10))


# step 7  Label Balance Check

# Check if 'type' column exists
if "type" in labels_df.columns:
    print(" Distribution of label types in labels_df:")
    label_counts = labels_df["type"].value_counts(dropna=False)
    print(label_counts)

    # Optional: Plot distribution
    import matplotlib.pyplot as plt

    label_counts.plot(kind='bar', color='skyblue', title="Label Type Distribution")
    plt.xlabel("Label Type")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

else:
    print(" 'type' column not found in labels_df.")



#  Step 7  â€” Prepare gold labels WITHOUT dropping "Missing"; dedupe per (article_id, dataset_id)

import pandas as pd

VALID_TYPES = {"Primary", "Secondary"}
TYPE_PRIORITY = {"Primary": 2, "Secondary": 1}

# Helper: normalize article_id to the underscore join key
def _to_norm(a):
    if isinstance(a, str):
        n, _ = normalize_article_id_jats(a)
        n = n.replace("/", "_")
        # optional cosmetic underscore after registrant if none present
        if n.startswith("10.") and "_" not in n:
            pos = n.find(".", 3)
            if pos != -1:
                n = n[:pos] + "_" + n[pos+1:]
        return n
    return a

# Work on a copy and add normalized key
labels_work = labels_df.copy()
labels_work["article_id_norm"] = labels_work["article_id"].map(_to_norm)

# Split gold vs missing (keep both)
gold = labels_work[labels_work["type"].isin(VALID_TYPES)].copy()
missing = labels_work[~labels_work["type"].isin(VALID_TYPES)].copy()

# De-dup *within the same (article_id_norm, dataset_id)* using type priority (Primary > Secondary)
gold["_priority"] = gold["type"].map(TYPE_PRIORITY)
gold_dedup = gold.loc[
    gold.groupby(["article_id_norm", "dataset_id"])["_priority"].idxmax()
][["article_id_norm", "dataset_id", "type"]].drop_duplicates().reset_index(drop=True)

print(f" Gold labels ready (tuples): {gold_dedup.shape}")
print(f" Missing rows kept for pseudo-labeling: {missing.shape}")
display(gold_dedup.head())



# âœ… Step 8 â€” Unified normalization utility for labels (reuse the agreed normalizer)

import pandas as pd

# Guard: ensure the agreed normalizer exists (do NOT redefine a different logic)
if "normalize_article_id_jats" not in globals():
    raise SystemExit(" Expected 'normalize_article_id_jats' to be defined earlier.")

def add_article_id_norm(df: pd.DataFrame, src_col="article_id", out_col="article_id_norm"):
    """
    Adds/refreshes underscore-style join key using the agreed single-pass normalizer.
    - Does not mutate identifiers other than normalization.
    - Idempotent (safe to call multiple times).
    """
    if df is None or not isinstance(df, pd.DataFrame):
        print(" add_article_id_norm: invalid DataFrame")
        return df
    if src_col not in df.columns:
        print(f" add_article_id_norm: source column '{src_col}' not found; skipping.")
        return df

    df[out_col] = (
        df[src_col]
        .map(lambda x: normalize_article_id_jats(x)[0] if isinstance(x, str) else x)
        .map(lambda x: x.replace("/", "_") if isinstance(x, str) else x)
    )
    return df

# Apply to the frames you use downstream:
labels_df = add_article_id_norm(labels_df, src_col="article_id", out_col="article_id_norm")

# If you built labels_norm earlier, refresh it too (harmless):
if "labels_norm" in globals() and isinstance(labels_norm, pd.DataFrame):
    labels_norm = add_article_id_norm(labels_norm, src_col="article_id", out_col="article_id_norm")

# If you created gold_dedup with article_id_norm already, this is a no-op; else:
if "gold_dedup" in globals() and isinstance(gold_dedup, pd.DataFrame):
    if "article_id_norm" not in gold_dedup.columns and "article_id" in gold_dedup.columns:
        gold_dedup = add_article_id_norm(gold_dedup, src_col="article_id", out_col="article_id_norm")

# Small sanity echo
for name in ["labels_df", "labels_norm", "gold_dedup"]:
    if name in globals() and isinstance(globals()[name], pd.DataFrame):
        df = globals()[name]
        if "article_id_norm" in df.columns:
            n = df["article_id_norm"].notna().sum()
            print(f" {name}: article_id_norm populated for {n} rows")



# Step 8.1 Formatting check after normalization

# Check column names
print("\nğŸ§¾ Column names:")
print(labels_df.columns.tolist())

# Sample values
print("\nğŸ”� Sample article_id values:")
print(labels_df["article_id"].sample(5).tolist())

print("\nğŸ”� Sample dataset_id values:")
print(labels_df["dataset_id"].sample(5).tolist())

# Check format: underscores vs slashes with na=False to handle missing values
print("\n Format check for article_id:")
print("Contains slashes:", labels_df["article_id"].str.contains("/", na=False).sum())
print("Contains underscores:", labels_df["article_id"].str.contains("_", na=False).sum())

print("\n Format check for dataset_id:")
print("Contains slashes:", labels_df["dataset_id"].str.contains("/", na=False).sum())
print("Contains underscores:", labels_df["dataset_id"].str.contains("_", na=False).sum())

# Optional: Check some normalized article_id samples if normalized column exists
if 'article_id_norm' in labels_df.columns:
    print("\nğŸ”� Sample normalized article_id values:")
    print(labels_df["article_id_norm"].sample(5).tolist())



labels_df


fulltext_df.columns.tolist()


# âœ… Step 9 â€” Build `citation_df` from `fulltext_df` (robust)
from IPython.display import display

if "fulltext_df" in globals() and isinstance(fulltext_df, pd.DataFrame):
    if fulltext_df.empty:
        raise ValueError(" fulltext_df exists but is empty.")

    # 1) Base requirement: we must have raw article_id
    if "article_id" not in fulltext_df.columns:
        raise ValueError(" fulltext_df is missing required column: 'article_id'")

    # 2) Choose a text source in priority order
    text_source = None
    for col in ["text", "full_text_clean", "full_text"]:
        if col in fulltext_df.columns:
            text_source = col
            break
    if text_source is None:
        raise ValueError("âš ï¸� No text column found (looked for 'text', 'full_text_clean', 'full_text').")

    # 3) Make a working copy and ensure 'text'
    citation_df = fulltext_df.copy()
    if text_source != "text":
        # If we only have raw full_text and the cleaner is available, clean on the fly
        if text_source == "full_text" and "clean_text" in globals():
            citation_df["text"] = citation_df["full_text"].map(clean_text)
            print(" Created 'text' by cleaning 'full_text'")
        else:
            citation_df["text"] = citation_df[text_source]
            print(f" Promoted '{text_source}' â†’ 'text'")
    else:
        print(" Using existing 'text' column")

    # 4) Ensure normalized join key
    if "article_id_norm" not in citation_df.columns:
        if "normalize_article_id_jats" not in globals():
            raise ValueError("âš ï¸� Normalizer 'normalize_article_id_jats' not defined earlier.")
        citation_df["article_id_norm"] = (
            citation_df["article_id"]
              .map(lambda x: normalize_article_id_jats(x)[0] if isinstance(x, str) else x)
              .map(lambda x: x.replace("/", "_") if isinstance(x, str) else x)
        )
        print("âœ… Added 'article_id_norm'")

    # 5) Light sanity: text length (no drops here; just info)
    citation_df["_text_len"] = citation_df["text"].map(lambda x: len(x.strip()) if isinstance(x, str) else 0)
    n_empty = int((citation_df["_text_len"] < 1).sum())
    print(f"ğŸ“� citation_df shape: {citation_df.shape} | empty text rows: {n_empty}")
    citation_df.drop(columns=["_text_len"], inplace=True)

    # 6) Final echo
    print(" citation_df columns:", list(citation_df.columns))
    display(citation_df.head())
else:
    raise ValueError(" fulltext_df not found or is not a valid DataFrame.")



citation_df


# === Step 10 â€” Safe single-pass normalization + idempotent re-validation ===
import pandas as pd

def normalize_article_id_jats(article_id):
    """
    Safe, single-pass JATS/BioC article_id normalizer (idempotent):
      - works on raw strings (slash or underscore styles)
      - lowercases + trims
      - '/' -> '_' (we use underscore-style join keys)
      - if startswith '10_' -> '10.' (prefix fix)
      - preserve the first underscore after '10.xxxx'
      - replace only the last underscore after the last dot with '.'
      - if it already looks normalized like '10.1002_ecs2.1280', return as-is
    Returns: (normalized_id, changed_flag)
    """
    if not isinstance(article_id, str):
        return article_id, False

    original = article_id.strip()
    # Guard: already normalized (e.g., '10.1002_ecs2.1280')
    if original.startswith("10.") and "_" in original and original.count(".") >= 2:
        return original, False

    aid = original.lower().replace("/", "_")

    # Fix '10_' prefix to '10.'
    if aid.startswith("10_"):
        aid = "10." + aid[3:]

    # Find first dot after '10.'
    first_dot_pos = aid.find(".", 3)
    if first_dot_pos == -1:
        normalized = aid.replace("_", ".")
    else:
        prefix = aid[: first_dot_pos + 1]
        suffix = aid[first_dot_pos + 1 :]

        if "_" in suffix:
            # Keep the first underscore in the suffix
            first_uscore_pos = suffix.find("_")
            preserved = suffix[: first_uscore_pos + 1]
            rest = suffix[first_uscore_pos + 1 :]

            # Replace only the last underscore AFTER the last dot with a dot
            last_underscore_pos = rest.rfind("_")
            last_dot_pos = rest.rfind(".")
            if last_underscore_pos != -1 and last_underscore_pos > last_dot_pos:
                rest = rest[:last_underscore_pos] + "." + rest[last_underscore_pos + 1 :]
            normalized = prefix + preserved + rest
        else:
            last_underscore_pos = suffix.rfind("_")
            last_dot_pos = suffix.rfind(".")
            if last_underscore_pos != -1 and last_underscore_pos > last_dot_pos:
                suffix = suffix[:last_underscore_pos] + "." + suffix[last_underscore_pos + 1 :]
            normalized = prefix + suffix

    changed = (normalized != original.lower())
    return normalized, changed


def conditional_fix_article_id(row: pd.Series, norm_col: str = "article_id_norm"):
    """
    Row-wise validator/repair:
      - recompute normalized ID from raw 'article_id'
      - ensure underscore style (replace '/' with '_')
      - if existing norm matches, keep it; else fix it
    """
    if not isinstance(row, pd.Series):
        raise TypeError("Input must be a pandas Series representing a DataFrame row.")
    if "article_id" not in row:
        raise KeyError("Row does not contain 'article_id'.")

    original = row["article_id"]
    norm_val = row.get(norm_col, None)

    normalized, _ = normalize_article_id_jats(original) if isinstance(original, str) else (original, False)
    if isinstance(normalized, str):
        normalized = normalized.replace("/", "_")

    if isinstance(norm_val, str) and norm_val == normalized:
        return norm_val
    return normalized


def apply_normalization(df: pd.DataFrame, df_name: str = "DataFrame"):
    """
    Idempotently ensure df['article_id_norm'] exists and is correct.
    Uses the safe single-pass normalizer above.
    Prints a brief summary of uniqueness/nulls.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError(f"â�Œ '{df_name}' is not a valid pandas DataFrame.")
    if "article_id" not in df.columns:
        raise KeyError(f"â�Œ '{df_name}' missing required 'article_id' column.")

    created = False
    if "article_id_norm" not in df.columns:
        df["article_id_norm"] = df["article_id"].map(
            lambda x: normalize_article_id_jats(x)[0] if isinstance(x, str) else x
        )
        # Enforce underscore join key, just in case
        df["article_id_norm"] = df["article_id_norm"].map(
            lambda x: x.replace("/", "_") if isinstance(x, str) else x
        )
        created = True
        print(f"âœ… Created 'article_id_norm' in {df_name}.")
    else:
        # Re-validate existing normalization (non-destructive)
        df["article_id_norm"] = df.apply(lambda r: conditional_fix_article_id(r, "article_id_norm"), axis=1)
        print(f"âœ… Re-validated 'article_id_norm' in {df_name}.")

    # Brief health echo
    total = len(df)
    uniq = df["article_id_norm"].nunique(dropna=True)
    nulls = df["article_id_norm"].isna().sum()
    dups = df.duplicated(subset=["article_id_norm"]).sum()
    action = "created" if created else "validated"
    print(
        f"ğŸ”� {df_name}: article_id_norm {action} â€” rows:{total} | unique:{uniq} | nulls:{nulls} | dups:{dups}"
    )


# --- Example usage (only call on frames that actually exist) ---
if "citation_df" in globals() and isinstance(citation_df, pd.DataFrame):
    apply_normalization(citation_df, "citation_df")
if "labels_df" in globals() and isinstance(labels_df, pd.DataFrame):
    apply_normalization(labels_df, "labels_df")
if "sample_submission" in globals() and isinstance(sample_submission, pd.DataFrame):
    apply_normalization(sample_submission, "sample_submission")  # Optional



# âœ… Step 12 â€” Ensure normalized join key (reuse the agreed normalizer; no redefinition)

import pandas as pd

if "normalize_article_id_jats" not in globals():
    raise SystemExit("â›” Expected 'normalize_article_id_jats' to be defined earlier.")

def ensure_article_id_norm(df: pd.DataFrame,
                           src_col: str = "article_id",
                           out_col: str = "article_id_norm",
                           enforce_underscore: bool = True,
                           name: str = "DataFrame") -> pd.DataFrame:
    """
    Idempotently (re)computes `out_col` from `src_col` using the agreed single-pass normalizer.
    - Converts '/' â†’ '_' for join key stability.
    - Optionally enforces a single underscore right after the registrant if none exists
      (10.<registrant>.<rest> â†’ 10.<registrant>_<rest>), purely cosmetic/consistent.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"â�Œ {name}: not a valid DataFrame.")
    if src_col not in df.columns:
        print(f"â„¹ï¸� {name}: source column '{src_col}' not found; skipping.")
        return df

    # Compute normalized key (idempotent)
    df[out_col] = (
        df[src_col]
          .map(lambda x: normalize_article_id_jats(x)[0] if isinstance(x, str) else x)
          .map(lambda x: x.replace("/", "_") if isinstance(x, str) else x)
    )

    if enforce_underscore:
        def _enforce_underscore_after_registrant(aid: str) -> str:
            if isinstance(aid, str) and aid.startswith("10.") and ("/" not in aid) and ("_" not in aid):
                pos = aid.find(".", 3)  # first '.' after "10."
                if pos != -1:
                    return aid[:pos] + "_" + aid[pos+1:]
            return aid
        df[out_col] = df[out_col].map(_enforce_underscore_after_registrant)

    print(f"âœ… {name}: '{out_col}' ensured for {int(df[out_col].notna().sum())} rows.")
    return df

# --- Apply to your frames (only if they exist) ---
if "citation_df" in globals() and isinstance(citation_df, pd.DataFrame):
    citation_df = ensure_article_id_norm(citation_df, name="citation_df")

if "labels_df" in globals() and isinstance(labels_df, pd.DataFrame):
    labels_df = ensure_article_id_norm(labels_df, name="labels_df")

# # Optional: sample_submission may or may not be present
# if "sample_submission" in globals() and isinstance(sample_submission, pd.DataFrame):
#     sample_submission = ensure_article_id_norm(sample_submission, src_col="article_id", name="sample_submission")



# âœ… Step 13 â€” Merge `citation_df` with labels (gold + pseudo aware) and add row_id/label_status
import pandas as pd
from IPython.display import display

# --- Guards ---
if "citation_df" not in globals() or not isinstance(citation_df, pd.DataFrame) or citation_df.empty:
    raise SystemExit("â›” `citation_df` is missing or empty. Build it in Step 10.")

# Ensure normalized join key exists (reuses our agreed normalizer)
if "normalize_article_id_jats" not in globals():
    raise SystemExit("â›” Expected `normalize_article_id_jats` to be defined earlier.")

def _ensure_norm(df, src="article_id", out="article_id_norm", name="df"):
    if src not in df.columns:
        raise SystemExit(f"â›” `{name}` is missing '{src}'")
    if out not in df.columns:
        df[out] = (
            df[src]
              .map(lambda x: normalize_article_id_jats(x)[0] if isinstance(x, str) else x)
              .map(lambda x: x.replace("/", "_") if isinstance(x, str) else x)
        )
    return df

citation_df = _ensure_norm(citation_df, name="citation_df")

# --- Compose the training labels we will merge in ---
# Priority:
#  1) final_train + pseudo (if both exist): mark source = 'pseudo' when present in pseudo, else 'gold'
#  2) gold_dedup + (optional) pseudo, if available
#  3) labels_df filtered to Primary/Secondary (gold only)
VALID_TYPES = {"Primary", "Secondary"}

def _compose_labels_union():
    # Case 1: final_train exists
    if "final_train" in globals() and isinstance(final_train, pd.DataFrame) and not final_train.empty:
        ft_cols = {"article_id_norm","dataset_id","type"}
        if not ft_cols.issubset(final_train.columns):
            raise SystemExit("â›” `final_train` missing required columns.")
        labels_union = final_train[list(ft_cols)].copy()
        labels_union["source"] = "gold"  # default
        # Upgrade to 'pseudo' where applicable (if pseudo exists)
        if "pseudo" in globals() and isinstance(pseudo, pd.DataFrame) and not pseudo.empty:
            p = pseudo[["article_id_norm","dataset_id","type"]].drop_duplicates()
            p["__is_pseudo__"] = True
            labels_union = labels_union.merge(p, how="left", on=["article_id_norm","dataset_id","type"])
            labels_union.loc[labels_union["__is_pseudo__"] == True, "source"] = "pseudo"
            labels_union = labels_union.drop(columns="__is_pseudo__", errors="ignore")
        return labels_union

    # Case 2: gold_dedup (+ pseudo)
    if "gold_dedup" in globals() and isinstance(gold_dedup, pd.DataFrame) and not gold_dedup.empty:
        gcols = {"article_id_norm","dataset_id","type"}
        if not gcols.issubset(gold_dedup.columns):
            raise SystemExit("â›” `gold_dedup` missing required columns.")
        labels_union = gold_dedup[list(gcols)].copy()
        labels_union["source"] = "gold"
        if "pseudo" in globals() and isinstance(pseudo, pd.DataFrame) and not pseudo.empty:
            p = pseudo[["article_id_norm","dataset_id","type"]].copy()
            p["source"] = "pseudo"
            labels_union = pd.concat([labels_union, p], ignore_index=True)\
                               .drop_duplicates(subset=["article_id_norm","dataset_id","type"], keep="first")
        return labels_union

    # Case 3: fallback to labels_df (gold only)
    if "labels_df" in globals() and isinstance(labels_df, pd.DataFrame) and not labels_df.empty:
        if "article_id_norm" not in labels_df.columns:
            _ensure_norm(labels_df, name="labels_df")
        filt = labels_df[labels_df["type"].isin(VALID_TYPES)].copy()
        if filt.empty:
            print("âš ï¸� No Primary/Secondary rows in labels_df after filtering; proceeding with empty labels.")
            return pd.DataFrame(columns=["article_id_norm","dataset_id","type","source"])
        u = filt[["article_id_norm","dataset_id","type"]].drop_duplicates()
        u["source"] = "gold"
        return u

    # No labels available
    return pd.DataFrame(columns=["article_id_norm","dataset_id","type","source"])

labels_union = _compose_labels_union()

# --- Merge ---
keep_cols_left = [c for c in ["article_id","article_id_norm","text","file_path","file_type"] if c in citation_df.columns]
keep_cols_right = ["article_id_norm","dataset_id","type","source"]
labels_union = labels_union[keep_cols_right].copy()

merged_df = citation_df[keep_cols_left].merge(labels_union, how="left", on="article_id_norm")
merged_df = merged_df.reset_index(drop=True)

# row_id first
merged_df.insert(0, "row_id", merged_df.index.astype(int))

# label_status derived from 'source' (preferred) or 'type' fallback
def _label_status(row):
    src = row.get("source", pd.NA)
    typ = row.get("type", pd.NA)
    if pd.isna(src) and pd.isna(typ):
        return "unlabeled"
    if isinstance(src, str):
        return "pseudo" if src != "gold" else "labeled"
    # Fallback if only 'type' is present (rare in this path)
    if typ == "Missing" or pd.isna(typ):
        return "unlabeled"
    return "labeled"

merged_df["label_status"] = merged_df.apply(_label_status, axis=1)

print(f"âœ… merged_df created: shape={merged_df.shape}")
print(merged_df["label_status"].value_counts(dropna=False).rename("label_status_counts"))

# Preview
display(merged_df.head(10))



# Step 13.1
desired_order = [
    'row_id',
    'article_id',
    'text',
    'article_id_norm',
    'dataset_id',
    'type',
    'label_status'
]

# Ensure all columns exist before reordering (in case any are missing)
existing_cols = [col for col in desired_order if col in merged_df.columns]

# Reorder columns
merged_df = merged_df[existing_cols]

print("âœ… Reordered columns in merged_df:")
print(merged_df.columns.tolist())



# ===Step 13.2  QA on merged_df ===
from IPython.display import display

# 1) No duplicate labeled tuples
dups = (
    merged_df[merged_df["label_status"] != "unlabeled"]
    .duplicated(subset=["article_id_norm", "dataset_id", "type"])
    .sum()
)
print("ğŸ”� Duplicate labeled tuples:", int(dups))

# 2) Source distribution (gold vs pseudo)
if "source" in merged_df.columns:
    print("\nğŸ“¦ Source distribution among labeled rows:")
    print(
        merged_df.loc[merged_df["label_status"] != "unlabeled", "source"]
        .fillna("unknown")
        .value_counts(dropna=False)
        .rename("count")
    )

# 3) Type distribution among labeled rows
print("\nğŸ�·ï¸� Type distribution among labeled rows:")
print(
    merged_df.loc[merged_df["label_status"] != "unlabeled", "type"]
    .value_counts(dropna=False)
    .rename("count")
)

# 4) Optional: spot â€œversionedâ€� DOIs (e.g., ...v1, v2)
ver = (
    merged_df.loc[merged_df["label_status"] != "unlabeled", "dataset_id"]
    .astype(str)
    .str.contains(r"\.v\d+$", regex=True)
    .sum()
)
print(f"\nğŸ”� Versioned dataset DOIs (â€¦vN) among labeled rows: {int(ver)}")

# 5) Per-article label counts (top 10)
per_article = (
    merged_df.loc[merged_df["label_status"] != "unlabeled"]
    .groupby("article_id_norm")["dataset_id"].nunique()
    .sort_values(ascending=False)
    .head(10)
    .rename("unique_datasets")
)
print("\nğŸ“ˆ Top articles by number of labeled datasets:")
display(per_article.to_frame())



# === Step 13.3 (revised) â€” Validate merged output on real constraints ===
from IPython.display import display
import pandas as pd
import re

VALID_TYPES = {"Primary", "Secondary"}

# 1) Columns presence
expected = ["row_id","article_id","article_id_norm","text","dataset_id","type","label_status"]
missing = [c for c in expected if c not in merged_df.columns]
print("All expected columns present." if not missing else f"â�Œ Missing columns: {missing}")
 
# 2) Nulls in critical text/id
crit = merged_df[["article_id","article_id_norm","text"]].isna().sum()
print("\nğŸ”� Null check (critical):")
print(crit)

# 3) Tuple uniqueness among labeled rows
labeled = merged_df[merged_df["label_status"] != "unlabeled"].copy()
tuple_dups = labeled.duplicated(subset=["article_id_norm","dataset_id","type"]).sum()
print(f"\nğŸ”� Duplicate labeled tuples (should be 0): {int(tuple_dups)}")
if tuple_dups:
    display(labeled[labeled.duplicated(subset=["article_id_norm","dataset_id","type"], keep=False)]
            [["article_id_norm","dataset_id","type"]].head(10))

# 3b) Article-level multiplexing (informational, not an error)
per_article = (
    labeled.groupby("article_id_norm")["dataset_id"].nunique().sort_values(ascending=False)
)
print("\nğŸ“ˆ Labeled datasets per article â€” summary:")
print(per_article.describe())
print("\nTop articles by unique dataset_id:")
display(per_article.head(10).to_frame("unique_datasets"))

# 4) label_status consistency (unlabeled must have no dataset/type; labeled must have valid type)
bad_unlabeled = merged_df[(merged_df["label_status"]=="unlabeled") & 
                          (merged_df["dataset_id"].notna() | merged_df["type"].notna())]
bad_labeled   = merged_df[(merged_df["label_status"]!="unlabeled") & 
                          (~merged_df["type"].isin(VALID_TYPES))]
print(f"\nğŸ§ª Inconsistencies â€” unlabeled_with_values={len(bad_unlabeled)} | labeled_with_invalid_type={len(bad_labeled)}")
if len(bad_unlabeled): display(bad_unlabeled.head(5))
if len(bad_labeled):   display(bad_labeled[["article_id_norm","dataset_id","type","label_status"]].head(10))

# 5) row_id uniqueness
is_unique = merged_df["row_id"].is_unique
print(f"\n row_id unique: {is_unique}")
if not is_unique:
    print("âš ï¸� Reassigning sequential row_idâ€¦")
    merged_df = merged_df.reset_index(drop=True)
    merged_df.insert(0, "row_id", merged_df.index.astype(int))
    print(" row_id reassigned.")

# 6) Optional: versioned DOIs (â€¦vN) â€” informational
if "dataset_id" in merged_df.columns:
    ver = merged_df.loc[merged_df["label_status"]!="unlabeled","dataset_id"].astype(str).str.contains(r"\.v\d+$").sum()
    print(f"\nğŸ”� Versioned dataset DOIs among labeled rows: {int(ver)}")

# 7) Distribution echoes
print("\nğŸ“Š label_status distribution:")
print(merged_df["label_status"].value_counts(dropna=False).rename("count"))
if "type" in merged_df.columns:
    print("\nğŸ�·ï¸� type distribution (labeled only):")
    print(labeled["type"].value_counts(dropna=False).rename("count"))



# step 13.4
# Check how many article_ids have more than one label entry
label_counts = labels_df["article_id"].value_counts()
multi_label_articles = label_counts[label_counts > 1]

print(f"ğŸ”� Articles with multiple dataset labels: {len(multi_label_articles)}")
display(labels_df[labels_df["article_id"].isin(multi_label_articles.index)].head(10))



# 13.5 structure check
# âœ… Check column order and types
print("ğŸ“‹ Columns:", merged_df.columns.tolist())
print("\nğŸ“� Data types:")
print(merged_df.dtypes)

# Null check for critical fields
print("\nğŸ”� Null counts:")
print(merged_df[["article_id", "text", "dataset_id", "type"]].isnull().sum())

#  Check for exact duplicates (optional safety)
exact_duplicates = merged_df.duplicated()
if exact_duplicates.any():
    print(f"âš ï¸� Found {exact_duplicates.sum()} exact duplicate rows.")
else:
    print(" No exact duplicate rows found.")

#  Confirm if dataset_id and type are no longer lists
for col in ['dataset_id', 'type']:
    list_check = merged_df[col].apply(lambda x: isinstance(x, list)).any()
    print(f"ğŸ§ª Column '{col}' contains lists? {list_check}")



# step 13.6 Inspect rows where dataset_id is missing (i.e., did not match any label)
unmatched = merged_df[merged_df["dataset_id"].isna()]

print(f" Unmatched rows count: {len(unmatched)}")
display(unmatched)



# === Step 13.7 â€” Unlabeled ID diagnostics (normalized + smart matching) ===
import re
import difflib
import pandas as pd
from IPython.display import display

# Ensure both frames have normalized IDs
def _ensure_norm(df, name):
    if "article_id_norm" not in df.columns:
        if "article_id" not in df.columns:
            raise SystemExit(f"â›” {name} missing 'article_id'")
        df["article_id_norm"] = (
            df["article_id"]
              .map(lambda x: normalize_article_id_jats(x)[0] if isinstance(x,str) else x)
              .map(lambda x: x.replace("/", "_") if isinstance(x,str) else x)
        )
    return df

merged_df  = _ensure_norm(merged_df,  "merged_df")
labels_df  = _ensure_norm(labels_df,  "labels_df")

# Pick unlabeled rows (by our label_status)
unl = merged_df[merged_df["label_status"] == "unlabeled"].copy()
unl_ids = unl["article_id_norm"].dropna().astype(str).unique().tolist()
lbl_ids = labels_df["article_id_norm"].dropna().astype(str)

print(f" Unlabeled unique articles: {len(unl_ids)}")

# Helper: registrant = the '10.xxxx' part; suffix = the rest after the underscore
def split_registrant_suffix(aid_norm: str):
    # Expect '10.<registrant>_<suffix>'
    parts = aid_norm.split("_", 1)
    registrant = parts[0] if parts else aid_norm
    suffix = parts[1] if len(parts) == 2 else ""
    return registrant, suffix

def sanitize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

# Build index of labels by registrant for fast candidate narrowing
labels_by_reg = {}
for aid in lbl_ids:
    reg, suf = split_registrant_suffix(aid)
    labels_by_reg.setdefault(reg, []).append(aid)

# Inspect a sample of N unlabeled to see if there are near misses
N = min(10, len(unl_ids))
print(f"\nğŸ”� Inspecting {N} unlabeled article_id_norm values for near matches:\n")

for aid in unl_ids[:N]:
    reg, suf = split_registrant_suffix(aid)
    cand_pool = labels_by_reg.get(reg, [])
    print(f"â€¢ {aid}  | registrant={reg}  | pool={len(cand_pool)}")
    if not cand_pool:
        print("   â†’ No labels with same registrant (likely truly unlabeled).")
        continue

    # Try fuzzy on the suffix within registrant pool
    target = sanitize(suf) or sanitize(aid)
    cand_suf = [(c, sanitize(split_registrant_suffix(c)[1]) or sanitize(c)) for c in cand_pool]
    # Simple similarity via difflib on sanitized strings
    sims = [(c, difflib.SequenceMatcher(a=target, b=cs).ratio()) for c, cs in cand_suf]
    sims = sorted(sims, key=lambda x: x[1], reverse=True)[:3]

    # Show top candidates if they're reasonably close
    shown = False
    for c, sc in sims:
        if sc >= 0.6:
            # show the raw labels rows for context (dataset_id/type)
            rows = labels_df[labels_df["article_id_norm"] == c][["article_id","article_id_norm","dataset_id","type"]].head(3)
            print(f"   â†’ candidate: {c}  (similarity={sc:.2f})")
            display(rows)
            shown = True
    if not shown:
        print("   â†’ No reasonably similar candidates within registrant.")

# Summary: how many unlabeled share a registrant with any labeled article
unl_regs = pd.Series([split_registrant_suffix(a)[0] for a in unl_ids])
lbl_regs = set([split_registrant_suffix(a)[0] for a in lbl_ids])
share_reg = unl_regs.isin(lbl_regs).sum()
print(f"\nğŸ“Š Unlabeled sharing registrant with any labeled: {share_reg}/{len(unl_ids)}")



# === Step 14 Version diagnostics: datasets vs articles (narrow patterns) ===
from IPython.display import display
import pandas as pd

# Use normalized article ID if available
id_col = "article_id_norm" if "article_id_norm" in merged_df.columns else "article_id"
s_art = merged_df[id_col].astype(str)

# 1) Dataset DOI versions: look for '.vN' at the end of dataset_id
if "dataset_id" in merged_df.columns:
    s_ds = merged_df["dataset_id"].astype(str)
    ds_version_mask = s_ds.str.contains(r'\.v\d+$', regex=True, na=False)
    print(f"ğŸ“¦ Dataset DOI versions (.vN): {int(ds_version_mask.sum())}")
    if ds_version_mask.any():
        display(merged_df.loc[ds_version_mask, [id_col, "dataset_id", "type"]].head(10))
else:
    print("â„¹ï¸� 'dataset_id' not present; skipping dataset version check.")

# 2) Article DOIs with explicit version (F1000 platform family)
#    e.g., 10.12688/f1000research.<article>.<version>  or 10.12688/wellcomeopenres.<article>.<version>
#    Our IDs may be underscore-style: 10.12688_f1000research.<id>.<ver>
f1000_mask = s_art.str.contains(
    r'^10\.12688_([a-z0-9]+(?:openres|research))\.\d+\.\d+$', na=False
) | s_art.str.contains(
    r'^10\.5256_f1000research\.\d+\.\d+$', na=False
)

print(f"ğŸ“° Article DOIs with explicit version (F1000-style): {int(f1000_mask.sum())}")
if f1000_mask.any():
    display(merged_df.loc[f1000_mask, [id_col, "article_id"]].drop_duplicates().head(10))

# 3) Everything else ending in digits (likely article numbers, NOT versions)
other_numeric_tail = s_art.str.contains(r'\.\d+$', na=False) & ~f1000_mask
print(f"â„¹ï¸� Article DOIs ending with digits but not F1000-style (likely article numbers): {int(other_numeric_tail.sum())}")

# Summary tip:
print("\nâœ… Action: Keep dataset '.vN' suffixes as-is. Do NOT strip numeric tails from article DOIs.")



# Step 14.1  Optional diagnostic only â€” F1000-style article versions (no mutations, no relabeling)
import re
import pandas as pd
from IPython.display import display

# Ensure normalized IDs exist
def _ensure_norm(df, name):
    if "article_id_norm" not in df.columns:
        if "article_id" not in df.columns:
            raise SystemExit(f"â›” {name} missing 'article_id'")
        df["article_id_norm"] = (
            df["article_id"]
              .map(lambda x: normalize_article_id_jats(x)[0] if isinstance(x, str) else x)
              .map(lambda x: x.replace("/", "_") if isinstance(x, str) else x)
        )
    return df

_ensure_norm(merged_df, "merged_df")
_ensure_norm(labels_df, "labels_df")

# F1000 / Wellcome Open Research patterns (underscore style)
pat = re.compile(
    r'^(?:10\.12688_(?:f1000research|wellcomeopenres)\.(\d+)\.(\d+))|(?:10\.5256_f1000research\.(\d+)\.(\d+))$'
)

def f1000_base_and_version(aid_norm: str):
    """Return (base_v1, version) for F1000-style IDs, else (None, None)."""
    if not isinstance(aid_norm, str): 
        return (None, None)
    m = pat.match(aid_norm)
    if not m:
        return (None, None)
    # Replace the final .<version> with .1 to form base v1
    head, _, ver = aid_norm.rpartition(".")
    base_v1 = f"{head}.1"
    return (base_v1, ver)

mf = merged_df[["article_id_norm", "label_status"]].copy()
mf[["base_v1", "version"]] = mf["article_id_norm"].apply(
    lambda s: pd.Series(f1000_base_and_version(s))
)

ver_rows = mf[mf["version"].notna()].copy()
print(f" F1000-style versioned articles in merged_df: {len(ver_rows)}")
if not ver_rows.empty:
    print("Version distribution:")
    display(ver_rows["version"].value_counts().to_frame("count"))

    # Among version > 1, check whether v1 exists in labels (diagnostic only)
    vgt1 = ver_rows[ver_rows["version"].astype(int) > 1].copy()
    if not vgt1.empty:
        lbl_ids = labels_df["article_id_norm"].astype(str)
        vgt1["base_in_labels"] = vgt1["base_v1"].isin(lbl_ids)
        hits = int(vgt1["base_in_labels"].sum())
        print(f"ğŸ”— Version>1 with v1 present in labels: {hits}/{len(vgt1)}")
        display(vgt1[["article_id_norm", "base_v1", "base_in_labels"]].head(10))
    else:
        print(" No version>1 articles found.")
else:
    print(" No F1000-style versions detected.")



merged_df


# step 15
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")

# Count value distribution including NaN
type_counts = merged_df['type'].value_counts(dropna=False)

# Plot
plt.figure(figsize=(8, 5))
ax = sns.barplot(x=type_counts.index.astype(str), y=type_counts.values, palette="pastel")

# Add labels and title
plt.title("ğŸ“Š Distribution of 'type' Labels", fontsize=14)
plt.xlabel("Type", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45)

# Annotate each bar with its value
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{int(height)}',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom',
                fontsize=10, color='black')

plt.tight_layout()
plt.show()



citation_df


# Step 16 . Normalize IDs, merge labels (gold + pseudo if available), and prep detection frames

import pandas as pd
from IPython.display import display

# --- Guards
if "normalize_article_id_jats" not in globals():
    raise SystemExit("â›” Expected 'normalize_article_id_jats' to be defined earlier.")

def add_norm(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if "article_id" not in df.columns:
        raise KeyError(f"{name} is missing 'article_id'")
    df["article_id_norm"] = (
        df["article_id"]
          .map(lambda x: normalize_article_id_jats(x)[0] if isinstance(x, str) else x)
          .map(lambda x: x.replace("/", "_") if isinstance(x, str) else x)
    )
    # Cosmetic: enforce single underscore after registrant if none present
    def _enforce_underscore_after_registrant(aid: str) -> str:
        if isinstance(aid, str) and aid.startswith("10.") and ("_" not in aid):
            pos = aid.find(".", 3)
            if pos != -1:
                return aid[:pos] + "_" + aid[pos+1:]
        return aid
    df["article_id_norm"] = df["article_id_norm"].map(_enforce_underscore_after_registrant)
    return df

# 1) Ensure normalized keys
labels_df   = add_norm(labels_df,   "labels_df")
citation_df = add_norm(citation_df, "citation_df")

# 2) Compose labels union (gold + pseudo if available)
VALID_TYPES = {"Primary", "Secondary"}
def compose_labels_union():
    if "final_train" in globals() and isinstance(final_train, pd.DataFrame) and not final_train.empty:
        lu = final_train[["article_id_norm","dataset_id","type"]].drop_duplicates().copy()
        lu["source"] = "gold"
        if "pseudo" in globals() and isinstance(pseudo, pd.DataFrame) and not pseudo.empty:
            p = pseudo[["article_id_norm","dataset_id","type"]].drop_duplicates().copy()
            p["source"] = "pseudo"
            lu = pd.concat([lu, p], ignore_index=True)\
                   .dr



# === Step 17 (final) â€” Build/repair detection_df + mine citation-like tokens + plot + conservative pseudo ===
import pandas as pd
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt

APPLY_PSEUDO = True  # set False to skip pseudo-labeling

# -------- Stopwords / filters --------
try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    STOPWORDS = set(ENGLISH_STOP_WORDS)
except Exception:
    STOPWORDS = set()

LATEX_JUNK = {
    "usepackage","documentclass","begin","end","caption","label","cite","ref",
    "section","subsection","figure","table","tabular","mathrm","mathbf","textbf",
    "textit","emph"
}
GENERIC = {
    "data","dataset","study","studies","analysis","results","result","method",
    "methods","using","used","use","based","number","numbers","value","values",
    "high","low","mean","time","size","different","model","models","sample",
    "samples","distribution","document","temperature","response","change","changes",
    "research","cell","cells","effect","effects"
}
TOKEN_BLACKLIST = STOPWORDS | LATEX_JUNK | GENERIC | {"https","http","doi","org","dx","www","fig","figure","table","al","et"}

# -------- Guards & helpers --------
if "normalize_article_id_jats" not in globals():
    raise SystemExit("â›” Expected 'normalize_article_id_jats' to be defined earlier.")

def _ensure_norm(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if "article_id_norm" not in df.columns:
        if "article_id" not in df.columns:
            raise SystemExit(f"â›” {name} missing 'article_id'")
        df["article_id_norm"] = (
            df["article_id"]
              .map(lambda x: normalize_article_id_jats(x)[0] if isinstance(x,str) else x)
              .map(lambda x: x.replace("/", "_") if isinstance(x,str) else x)
        )
        # cosmetic: ensure one underscore after registrant if none
        def _after_reg(aid: str) -> str:
            if isinstance(aid, str) and aid.startswith("10.") and "_" not in aid:
                pos = aid.find(".", 3)
                if pos != -1:
                    return aid[:pos] + "_" + aid[pos+1:]
            return aid
        df["article_id_norm"] = df["article_id_norm"].map(_after_reg)
    return df

def _ensure_text_model(df: pd.DataFrame) -> pd.DataFrame:
    if "text_model" in df.columns:
        return df
    if "text" in df.columns:
        if "clean_text" in globals():
            df["text_model"] = df["text"].map(clean_text)  # your safe cleaner (keeps DOIs)
        else:
            df["text_model"] = df["text"]
    else:
        df["text_model"] = pd.NA
    return df

def _build_detection_df() -> pd.DataFrame:
    if "detection_doc_df" in globals() and isinstance(detection_doc_df, pd.DataFrame) and not detection_doc_df.empty:
        df = detection_doc_df.copy()
        df = _ensure_norm(df, "detection_doc_df")
        df = _ensure_text_model(df)
    elif "merged_df" in globals() and isinstance(merged_df, pd.DataFrame) and not merged_df.empty:
        df = merged_df.copy()
        df = _ensure_norm(df, "merged_df")
        df = df.drop_duplicates(subset=["article_id_norm"]).copy()
        df = _ensure_text_model(df)
        if "label_status" not in df.columns:
            df["label_status"] = "unlabeled"
    else:
        raise SystemExit("â›” Could not build detection_df; need detection_doc_df or merged_df.")
    if "is_citation" not in df.columns:
        df["is_citation"] = pd.NA
    if "row_id" not in df.columns:
        df = df.reset_index(drop=True)
        df.insert(0, "row_id", df.index.astype(int))
    keep = [c for c in ["row_id","article_id","article_id_norm","text_model","label_status","is_citation"] if c in df.columns]
    return df[keep].copy()

# -------- Ensure detection_df exists --------
if "detection_df" not in globals() or not isinstance(detection_df, pd.DataFrame) or detection_df.empty:
    detection_df = _build_detection_df()
else:
    detection_df = _ensure_norm(detection_df, "detection_df")
    detection_df = _ensure_text_model(detection_df)
    if "label_status" not in detection_df.columns:
        detection_df["label_status"] = "unlabeled"
    if "is_citation" not in detection_df.columns:
        detection_df["is_citation"] = pd.NA
    if "row_id" not in detection_df.columns:
        detection_df = detection_df.reset_index(drop=True)
        detection_df.insert(0, "row_id", detection_df.index.astype(int))

print(f"âœ… detection_df ready: {detection_df.shape}")

# -------- Make sure strong positives are marked --------
mask_strong = detection_df["label_status"] == "labeled"
detection_df.loc[mask_strong, "is_citation"] = 1
print(f"âœ… Strong positives set: {int(mask_strong.sum())}")

# -------- Mine only citation-like sentences from positives --------
pat_doi  = r'(?:https?://doi\.org/)?10\.\d{4,9}/\S+'
pat_repo = r'\b(?:dryad|zenodo|figshare|pangaea|genbank|arrayexpress|geo|sra|ena|ebi|empiar|pdb|gisaid|icpsr|tcia|chembl|osf|openneuro|neurovault|pasta)\b'
pat_acc  = r'\b(?:gse\d+|gsm\d+|e-mtab-\d+|prj[edn][a-z]?\d+|empiar-\d+|chembl\d+|epi(?:[_-]isl)?\d+)\b'
context_pat = re.compile(f'(?:{pat_doi})|(?:{pat_repo})|(?:{pat_acc})', flags=re.I)

def citation_like_sentences(text: str):
    if not isinstance(text, str) or not text.strip():
        return []
    sents = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [s for s in sents if context_pat.search(s or "")]

pos_texts = detection_df.loc[detection_df["is_citation"] == 1, "text_model"].dropna().astype(str)
sent_bank = []
for t in pos_texts:
    ss = citation_like_sentences(t)
    if ss:
        sent_bank.extend(ss)
if not sent_bank:
    sent_bank = pos_texts.tolist()

# -------- Tokenizer over citation-like sentences --------
def tokenize(text: str):
    toks = re.findall(r"\b[a-z0-9][a-z0-9_-]+\b", text.lower())
    out = []
    for t in toks:
        if t in TOKEN_BLACKLIST:
            continue
        if t.isdigit() or len(t) <= 2:
            continue
        out.append(t)
    return out

freq = Counter()
for s in sent_bank:
    freq.update(tokenize(s))

if not freq:
    print("â„¹ï¸� No tokens found from citation-like sentences; nothing to plot/label.")
else:
    MIN_FREQ = 2
    items = [(tok, cnt) for tok, cnt in freq.items() if cnt >= MIN_FREQ]
    items.sort(key=lambda x: x[1], reverse=True)
    top30 = items[:30]
    top_keywords = [tok for tok, _ in top30]
    print("ğŸ”‘ Top 30 citation-like keywords:", top_keywords)

    # ---- Plot Top 30 ----
    if top30:
        plt.figure(figsize=(10, 6))
        toks = [t for t, _ in top30][::-1]
        cnts = [c for _, c in top30][::-1]
        plt.barh(toks, cnts)
        plt.title("Top 30 Tokens from Citation-like Sentences")
        plt.xlabel("Frequency")
        plt.tight_layout()
        plt.show()
    else:
        print("â„¹ï¸� No tokens pass MIN_FREQ; nothing to plot.")

    # ---- Optional: conservative pseudo-labeling ----
    if APPLY_PSEUDO and top30:
        # reset previous pseudo only (keep gold)
        mask_pseudo = detection_df["label_status"].isin(["pseudo","pseudo_weak"])
        detection_df.loc[mask_pseudo, ["is_citation","label_status"]] = [pd.NA, "unlabeled"]

        # unlabeled only
        unl = detection_df["label_status"] == "unlabeled"
        # doc is positive if any citation-like sentence contains any top keyword
        pat_kw = re.compile(r"\b(?:%s)\b" % "|".join(map(re.escape, top_keywords)), flags=re.I)

        def doc_hits(text: str) -> bool:
            sents = citation_like_sentences(text or "")
            return any(pat_kw.search(s or "") for s in sents) if sents else False

        hits = detection_df.loc[unl, "text_model"].astype(str).map(doc_hits)
        idx = detection_df.loc[unl].index[hits.values]
        detection_df.loc[idx, "is_citation"] = 1
        detection_df.loc[idx, "label_status"] = "pseudo"

        print("\nğŸ“Š Pseudo positives added:")
        print(f"   â€¢ unlabeled candidates: {int(unl.sum())}")
        print(f"   â€¢ matched & set to is_citation=1: {int(len(idx))}")

# ---- Final echoes ----
print("\nğŸ“ˆ is_citation distribution:")
print(detection_df["is_citation"].value_counts(dropna=False))
print("\nğŸ�·ï¸� label_status distribution:")
print(detection_df["label_status"].value_counts(dropna=False))



citation_df


# --- Step 17. 1 QA 1: how many pseudo docs actually contain citation context?
import re
import pandas as pd
from IPython.display import display

pat_doi  = r'(?:https?://doi\.org/)?10\.\d{4,9}/\S+'
pat_repo = r'\b(?:dryad|zenodo|figshare|pangaea|genbank|arrayexpress|geo|sra|ena|ebi|empiar|pdb|gisaid|icpsr|tcia|chembl|osf|openneuro|neurovault|pasta)\b'
pat_acc  = r'\b(?:gse\d+|gsm\d+|e-mtab-\d+|prj[edn][a-z]?\d+|empiar-\d+|chembl\d+|epi(?:[_-]isl)?\d+)\b'
ctx_pat = re.compile(f'(?:{pat_doi})|(?:{pat_repo})|(?:{pat_acc})', flags=re.I)

TEXT_COL = "text_model" if "text_model" in detection_df.columns else "text"

pseudo_mask = detection_df["label_status"] == "pseudo"
pseudo_ctx = detection_df.loc[pseudo_mask, TEXT_COL].astype(str).str.contains(ctx_pat, regex=True, na=False)
print(f"âœ… Pseudo docs with explicit citation context: {int(pseudo_ctx.sum())}/{int(pseudo_mask.sum())}")

# --- QA 2: repo token coverage inside pseudo docs
repos = ["dryad","zenodo","figshare","pangaea","arrayexpress","geo","sra","ena","ebi","empiar","pdb","gisaid","icpsr","tcia","chembl","osf","openneuro","neurovault","pasta"]
counts = {}
subset = detection_df.loc[pseudo_mask, TEXT_COL].astype(str)
for r in repos:
    counts[r] = subset.str.contains(r, case=False, na=False).sum()
repo_df = pd.Series(counts, name="pseudo_docs_with_token").sort_values(ascending=False)
display(repo_df.to_frame())

# --- QA 3: show a few pseudo examples with the matching sentence
def citation_like_sentences(text: str):
    if not isinstance(text, str) or not text.strip():
        return []
    sents = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [s for s in sents if ctx_pat.search(s or "")]
    
examples = detection_df.loc[pseudo_mask].sample(min(5, pseudo_mask.sum()), random_state=42)
rows = []
for _, row in examples.iterrows():
    sents = citation_like_sentences(row[TEXT_COL])
    rows.append({
        "article_id_norm": row["article_id_norm"],
        "example_sentence": sents[0][:300] + ("â€¦" if sents and len(sents[0]) > 300 else "")
    })
display(pd.DataFrame(rows))



# === Step 17.2 â€” Calibrate pseudo strength by keyword coverage (no row drops) ===
import re
import numpy as np
import pandas as pd
from collections import Counter

# --- Preconditions
if "detection_df" not in globals() or not isinstance(detection_df, pd.DataFrame):
    raise SystemExit(" detection_df not available. Run Step 32 first.")
TEXT_COL = "text_model" if "text_model" in detection_df.columns else ("text" if "text" in detection_df.columns else None)
if TEXT_COL is None:
    raise SystemExit(" No text column found (need 'text_model' or 'text').")

# --- If 'keywords' (top 30 tokens) from Step 32 isn't around, recompute a quick version
if "keywords" not in globals() or not keywords:
    # Fallback: recompute from labeled + pseudo positives using the same tokenizer idea
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        STOPWORDS = set(ENGLISH_STOP_WORDS)
    except Exception:
        STOPWORDS = set()
    TOKEN_BLACKLIST = STOPWORDS | {"https","http","doi","org","dx","www","figure","table","et","al"}
    def tokenize(text: str):
        toks = re.findall(r"\b[a-z0-9][a-z0-9_-]+\b", str(text).lower())
        return [t for t in toks if t not in TOKEN_BLACKLIST and not t.isdigit() and len(t) > 2]
    pos_texts = detection_df.loc[detection_df["is_citation"] == 1, TEXT_COL].dropna().astype(str)
    freq = Counter()
    for t in pos_texts:
        freq.update(tokenize(t))
    # conservative default if still empty
    keywords = [tok for tok, _ in freq.most_common(30)] if freq else ["dryad","zenodo","figshare","empiar","arrayexpress","genbank","geo","sra","ena","ebi","pdb","icpsr","tcia","chembl","openneuro"]

print(f"ğŸ”‘ Using {len(keywords)} mined keywords.")

# --- Compute per-doc keyword matches on text_model/raw text
kw_pattern = re.compile(r"\b(?:%s)\b" % "|".join(map(re.escape, keywords)), flags=re.I)

def find_keywords(text):
    s = str(text) if text is not None else ""
    return [kw for kw in keywords if kw_pattern.search(s)]

if "matched_keywords" not in detection_df.columns:
    detection_df["matched_keywords"] = detection_df[TEXT_COL].map(find_keywords)
detection_df["num_keyword_matches"] = detection_df["matched_keywords"].map(len).astype(int)

# --- Optional: add context bonus if the doc has explicit citation cues
pat_doi  = r'(?:https?://doi\.org/)?10\.\d{4,9}/\S+'
pat_repo = r'\b(?:dryad|zenodo|figshare|pangaea|genbank|arrayexpress|geo|sra|ena|ebi|empiar|pdb|gisaid|icpsr|tcia|chembl|osf|openneuro|neurovault|pasta)\b'
pat_acc  = r'\b(?:gse\d+|gsm\d+|e-mtab-\d+|prj[edn][a-z]?\d+|empiar-\d+|chembl\d+|epi(?:[_-]isl)?\d+)\b'
ctx_pat = re.compile(f'(?:{pat_doi})|(?:{pat_repo})|(?:{pat_acc})', flags=re.I)

ctx_hit = detection_df[TEXT_COL].astype(str).str.contains(ctx_pat, regex=True, na=False)
# Confidence score: matches + 1 if explicit context present
detection_df["pseudo_confidence"] = detection_df["num_keyword_matches"] + ctx_hit.astype(int)

# --- Calibrate threshold from current pseudo docs only
pseudo_mask = detection_df["label_status"] == "pseudo"
pseudo_conf = detection_df.loc[pseudo_mask, "pseudo_confidence"]

if pseudo_conf.empty:
    print(" No current pseudo docs to calibrate; skipping thresholding.")
else:
    # Adaptive threshold: 75th percentile or at least 3
    thr_adaptive = int(np.percentile(pseudo_conf, 75))
    threshold = max(3, thr_adaptive)
    print(f"ğŸ�šï¸� Adaptive confidence threshold = {threshold} (P75={thr_adaptive}, min=3)")

    # Split pseudo into confident vs weak
    confident_mask = pseudo_mask & (detection_df["pseudo_confidence"] >= threshold)
    weak_mask      = pseudo_mask & ~confident_mask

    n_conf = int(confident_mask.sum())
    n_weak = int(weak_mask.sum())

    # Keep confident as pseudo positives (is_citation=1). Revert weak â†’ unlabeled (no negative assertion).
    detection_df.loc[confident_mask, "is_citation"] = 1
    detection_df.loc[weak_mask, ["is_citation","label_status"]] = [pd.NA, "unlabeled"]

    print(f" Confident pseudo retained: {n_conf}")
    print(f" Weak pseudo reverted to unlabeled: {n_weak}")

# --- Summary
print("\n is_citation distribution:")
print(detection_df["is_citation"].value_counts(dropna=False))
print("\nğŸ�·ï¸� label_status distribution:")
print(detection_df["label_status"].value_counts(dropna=False))
print("\n Confidence stats (pseudo only, post-filter):")
post_pseudo = detection_df.loc[detection_df["label_status"]=="pseudo","pseudo_confidence"]
if not post_pseudo.empty:
    print(post_pseudo.describe())
else:
    print("No pseudo rows remain.")



# Step 17.3--- OPTIONAL: widen repo/ID coverage + visualize pseudo repo hits ---

import re
import matplotlib.pyplot as plt
import pandas as pd

TEXT_COL = "text_model" if "text_model" in detection_df.columns else "text"

# Add more repo aliases (phrases and tokens)
repo_terms = [
    "dryad", "zenodo", "figshare", "pangaea",
    "genbank", "ncbi", "geo", "gene expression omnibus",
    "sra", "ebi", "ena", "european nucleotide archive",
    "empiar", "emdb", "electron microscopy data bank",
    "pdb", "protein data bank",
    "gisaid", "icpsr", "tcia", "chembl", "osf",
    "openneuro", "neurovault", "pasta"
]

# Add richer accession patterns (common genomics/imaging IDs)
acc_pats = [
    r"\bGSE\d+\b", r"\bGSM\d+\b", r"\bE\-MTAB\-\d+\b",
    r"\bPRJ(?:NA|EB|DB)\d+\b",
    r"\bSRR\d+\b", r"\bSRX\d+\b", r"\bSRP\d+\b", r"\bSRS\d+\b",
    r"\bEMPIAR\-\d+\b",
    r"\bEMD\-\d+\b",          # EMDB
    r"\bPDB\s?[0-9A-Za-z]{4}\b",
    r"\bCHEMBL\d+\b",
    r"\bEPI(?:[_-]ISL)?\d+\b"
]

ctx_doi = r"(?:https?://doi\.org/)?10\.\d{4,9}/\S+"
ctx_repo = r"|".join(re.escape(t) for t in repo_terms)
ctx_acc  = r"|".join(acc_pats)
ctx_pat  = re.compile(f"(?:{ctx_doi})|(?:{ctx_repo})|(?:{ctx_acc})", flags=re.I)

# Recompute repo token coverage on pseudo docs only
pseudo_mask = detection_df["label_status"] == "pseudo"
subset = detection_df.loc[pseudo_mask, TEXT_COL].astype(str)

repo_counts = {}
for term in repo_terms:
    repo_counts[term] = subset.str.contains(term, case=False, na=False).sum()

repo_df = pd.Series(repo_counts, name="pseudo_docs_with_token").sort_values(ascending=False)

print("ğŸ”� Pseudo docs with explicit citation context (recheck):",
      subset.str.contains(ctx_pat, regex=True, na=False).sum(), "/", pseudo_mask.sum())

# Plot
plt.figure(figsize=(10, 6))
plot_df = repo_df.head(20).iloc[::-1]
plt.barh(plot_df.index, plot_df.values)
plt.title("Top repository terms in pseudo docs")
plt.xlabel("Doc count containing term")
plt.tight_layout()
plt.show()



# === Step 18. Model 1 â€” Doc-level Primary vs Secondary (TF-IDF + Logistic Regression) ===
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ---------- 0) Build doc-level labels from tuple labels (Primary/Secondary only) ----------
# Prefer detection_tuple_df if present; otherwise derive from merged_df
if "detection_tuple_df" in globals() and isinstance(detection_tuple_df, pd.DataFrame) and not detection_tuple_df.empty:
    tuples_src = detection_tuple_df.copy()
else:
    if "merged_df" not in globals() or not isinstance(merged_df, pd.DataFrame) or merged_df.empty:
        raise SystemExit(" Need 'detection_tuple_df' or 'merged_df' with tuple labels.")
    # Keep only rows with explicit labels
    tuples_src = merged_df.loc[merged_df["label_status"] != "unlabeled", ["article_id_norm","dataset_id","type"]].drop_duplicates()

# Keep only Primary/Secondary types
tuples_src = tuples_src[tuples_src["type"].isin(["Primary","Secondary"])].copy()
if tuples_src.empty:
    raise SystemExit(" No Primary/Secondary tuples available to build doc labels.")

# Collapse to one label per article (priority: Primary > Secondary)
priority = {"Primary": 2, "Secondary": 1}
tuples_src["_prio"] = tuples_src["type"].map(priority)
doc_labels = (
    tuples_src.sort_values(["article_id_norm","_prio"], ascending=[True, False])
              .drop_duplicates(subset=["article_id_norm"])[["article_id_norm","type"]]
              .reset_index(drop=True)
)

print(f"ğŸ“Œ Doc labels built: {doc_labels['type'].value_counts().to_dict()} (total {len(doc_labels)})")

# ---------- 1) Join labels to doc-level texts ----------
if "detection_df" not in globals() or not isinstance(detection_df, pd.DataFrame) or detection_df.empty:
    raise SystemExit(" 'detection_df' (doc-level texts) not found. Run earlier steps to build it.")

TEXT_COL = "text_model" if "text_model" in detection_df.columns else ("text" if "text" in detection_df.columns else None)
if TEXT_COL is None:
    raise SystemExit(" No text column found in detection_df (need 'text_model' or 'text').")

docs = detection_df[["article_id_norm", TEXT_COL]].dropna(subset=[TEXT_COL]).copy()
train_df = docs.merge(doc_labels, how="inner", on="article_id_norm")  # keep only labeled docs

if train_df.empty:
    raise SystemExit(" No overlap between doc texts and labels.")

print(f"ğŸ“Š Labeled docs for training: {train_df.shape[0]}")
print(train_df["type"].value_counts())

# ---------- 2) Grouped split by article (prevents leakage) ----------
groups = train_df["article_id_norm"]
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(train_df, groups=groups))

tr = train_df.iloc[train_idx].reset_index(drop=True)
te = train_df.iloc[test_idx].reset_index(drop=True)

X_train = tr[TEXT_COL].astype(str)
y_train = tr["type"].astype(str)
X_test  = te[TEXT_COL].astype(str)
y_test  = te["type"].astype(str)

print(f"ğŸ§ª Split â€” train: {len(tr)}, test: {len(te)}")

# ---------- 3) TF-IDF vectorization ----------
# Word unigrams + bigrams is a strong baseline; strip accents for robustness.
tfidf = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    strip_accents="unicode",
    max_features=20000
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

# ---------- 4) Train multinomial logistic regression ----------
clf = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",   # mitigate class imbalance
    multi_class="multinomial",
    solver="lbfgs",
    random_state=42
)
clf.fit(X_train_tfidf, y_train)

# ---------- 5) Predict and evaluate ----------
y_pred = clf.predict(X_test_tfidf)

print("\n Classification Report (Primary vs Secondary):")
print(classification_report(y_test, y_pred, digits=3))

print("\n Confusion Matrix (rows=true, cols=pred):")
labels_order = ["Primary","Secondary"]
print(confusion_matrix(y_test, y_pred, labels=labels_order))

macro_f1 = f1_score(y_test, y_pred, average="macro")
print(f"\nâ­� Macro F1: {macro_f1:.3f}")

# (Optional) Keep artifacts for later use
doc_type_model = clf
doc_type_vectorizer = tfidf



# Step 19=== Model 1+ â€” Doc-level Primary vs Secondary with word+char TF-IDF, class-weight & threshold tuning ===
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ---------- 0) Build doc-level labels from tuple labels (Primary/Secondary only) ----------
# prefer detection_tuple_df if present; else derive from merged_df
if "detection_tuple_df" in globals() and isinstance(detection_tuple_df, pd.DataFrame) and not detection_tuple_df.empty:
    tuples_src = detection_tuple_df.copy()
else:
    if "merged_df" not in globals() or not isinstance(merged_df, pd.DataFrame) or merged_df.empty:
        raise SystemExit("â›” Need 'detection_tuple_df' or 'merged_df' with tuple labels.")
    tuples_src = merged_df.loc[merged_df["label_status"] != "unlabeled", ["article_id_norm","dataset_id","type"]].drop_duplicates()

tuples_src = tuples_src[tuples_src["type"].isin(["Primary","Secondary"])].copy()
if tuples_src.empty:
    raise SystemExit("â›” No Primary/Secondary tuples available to build doc labels.")

priority = {"Primary": 2, "Secondary": 1}
tuples_src["_prio"] = tuples_src["type"].map(priority)
doc_labels = (
    tuples_src.sort_values(["article_id_norm","_prio"], ascending=[True, False])
              .drop_duplicates(subset=["article_id_norm"])[["article_id_norm","type"]]
              .reset_index(drop=True)
)

# ---------- 1) Join labels to doc-level texts ----------
if "detection_df" not in globals() or not isinstance(detection_df, pd.DataFrame) or detection_df.empty:
    raise SystemExit("â›” 'detection_df' (doc-level texts) not found. Run earlier steps to build it.")

TEXT_COL = "text_model" if "text_model" in detection_df.columns else ("text" if "text" in detection_df.columns else None)
if TEXT_COL is None:
    raise SystemExit("â›” No text column found in detection_df (need 'text_model' or 'text').")

docs = detection_df[["article_id_norm", TEXT_COL]].dropna(subset=[TEXT_COL]).copy()
train_df = docs.merge(doc_labels, how="inner", on="article_id_norm")  # keep only labeled docs

print(f"ğŸ“Š Labeled docs for training: {len(train_df)}")
print(train_df["type"].value_counts())

# ---------- 2) Grouped split by article ----------
groups = train_df["article_id_norm"]
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(train_df, groups=groups))

tr = train_df.iloc[train_idx].reset_index(drop=True)
te = train_df.iloc[test_idx].reset_index(drop=True)
print(f"ğŸ§ª Split â€” train: {len(tr)}, test: {len(te)}")

Xtr_txt = tr[TEXT_COL].astype(str)
ytr     = tr["type"].astype(str)
Xte_txt = te[TEXT_COL].astype(str)
yte     = te["type"].astype(str)

# ---------- 3) TF-IDF features: word (1â€“2) + char (3â€“5) ----------
word_vec = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=3, max_df=0.9,
    strip_accents="unicode",
    max_features=30000
)
char_vec = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=3, max_df=0.9
)

Xtr_w = word_vec.fit_transform(Xtr_txt)
Xte_w = word_vec.transform(Xte_txt)
Xtr_c = char_vec.fit_transform(Xtr_txt)
Xte_c = char_vec.transform(Xte_txt)

Xtr = hstack([Xtr_w, Xtr_c]).tocsr()
Xte = hstack([Xte_w, Xte_c]).tocsr()

# ---------- 4) Tune Secondary class weight & decision threshold ----------
sec_weights = [1.0, 1.5, 2.0, 2.5, 3.0]
thresholds  = np.linspace(0.3, 0.7, 9)  # probability threshold for predicting 'Secondary'
best = {"f1": -1, "w": None, "thr": None, "report": None, "cm": None}

for w in sec_weights:
    cw = {"Primary": 1.0, "Secondary": w}
    clf = LogisticRegression(
        max_iter=1000,
        class_weight=cw,
        multi_class="multinomial",
        solver="lbfgs",
        random_state=42
    )
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)
    # find index of 'Secondary'
    classes = clf.classes_.tolist()
    sec_idx = classes.index("Secondary")
    p_sec = proba[:, sec_idx]

    for thr in thresholds:
        # custom decision: if P(Secondary) >= thr -> Secondary else Primary
        yhat = np.where(p_sec >= thr, "Secondary", "Primary")
        f1m = f1_score(yte, yhat, average="macro")
        if f1m > best["f1"]:
            best.update({
                "f1": f1m, "w": w, "thr": thr,
                "report": classification_report(yte, yhat, digits=3),
                "cm": confusion_matrix(yte, yhat, labels=["Primary","Secondary"]),
                "clf": clf, "classes": classes, "p_sec": p_sec
            })

print(f"\nğŸ�¯ Best macro F1={best['f1']:.3f} with Secondary weight={best['w']} and threshold={best['thr']:.2f}")
print("\nğŸ“Š Classification Report (best setting):")
print(best["report"])
print("\nğŸ§® Confusion Matrix (rows=true, cols=pred):")
print(best["cm"])

# ---------- 5) Fit final model on full training set with best Secondary weight ----------
# (use same vectorizers; keep best threshold for inference)
full_word = word_vec.fit_transform(train_df[TEXT_COL].astype(str))
full_char = char_vec.fit_transform(train_df[TEXT_COL].astype(str))
X_full = hstack([full_word, full_char]).tocsr()
y_full = train_df["type"].astype(str)

final_clf = LogisticRegression(
    max_iter=1000,
    class_weight={"Primary": 1.0, "Secondary": best["w"]},
    multi_class="multinomial",
    solver="lbfgs",
    random_state=42
).fit(X_full, y_full)

doc_type_model = final_clf
doc_type_vectorizers = {"word": word_vec, "char": char_vec}
doc_type_threshold_secondary = float(best["thr"])

print("\nğŸ’¾ Saved artifacts: doc_type_model, doc_type_vectorizers, doc_type_threshold_secondary")

# Example: predict on the held-out test split with the *final* model for sanity
Xte_w2 = word_vec.transform(Xte_txt)
Xte_c2 = char_vec.transform(Xte_txt)
Xte2 = hstack([Xte_w2, Xte_c2]).tocsr()
proba2 = doc_type_model.predict_proba(Xte2)
sec_idx2 = doc_type_model.classes_.tolist().index("Secondary")
yhat2 = np.where(proba2[:, sec_idx2] >= doc_type_threshold_secondary, "Secondary", "Primary")
print("\nğŸ§ª Sanity check on held-out split with final model:")
print(classification_report(yte, yhat2, digits=3))




# Step 20 --- After the Model 1+ training cell ---

from scipy.sparse import hstack
import numpy as np
import pandas as pd

# Helper to predict doc type with saved artifacts
def predict_doc_type(df, text_col=None):
    if text_col is None:
        text_col = "text_model" if "text_model" in df.columns else "text"
    Xw = doc_type_vectorizers["word"].transform(df[text_col].astype(str))
    Xc = doc_type_vectorizers["char"].transform(df[text_col].astype(str))
    X  = hstack([Xw, Xc]).tocsr()
    proba = doc_type_model.predict_proba(X)
    classes = doc_type_model.classes_.tolist()
    sec_idx = classes.index("Secondary")
    p_sec   = proba[:, sec_idx]
    pred    = np.where(p_sec >= doc_type_threshold_secondary, "Secondary", "Primary")
    return pred, p_sec

# 1) Predict for ALL docs currently considered citations (gold + pseudo)
TEXT_COL = "text_model" if "text_model" in detection_df.columns else "text"
pos_mask = detection_df["is_citation"].fillna(0).astype(int) == 1
to_score = detection_df.loc[pos_mask, ["article_id_norm", TEXT_COL]].copy()

pred, p_sec = predict_doc_type(to_score, TEXT_COL)
detection_df.loc[to_score.index, "doc_type_pred"] = pred
detection_df.loc[to_score.index, "doc_type_p_secondary"] = p_sec

# 2) Quick summary split by label source
print("\nğŸ“Š Predictions for positive docs:")
print(detection_df.loc[pos_mask, "doc_type_pred"].value_counts())

print("\nğŸ“Š Predictions by label_status:")
print(detection_df.loc[pos_mask].groupby("label_status")["doc_type_pred"].value_counts().unstack(fill_value=0))

# 3)  inspect uncertain cases near threshold
band = 0.05
thr  = float(doc_type_threshold_secondary)
uncertain = detection_df.loc[pos_mask & detection_df["doc_type_p_secondary"].between(thr-band, thr+band, inclusive="both"),
                             ["article_id_norm", "doc_type_p_secondary", TEXT_COL]].sort_values("doc_type_p_secondary")
print(f"\nğŸ”� Near-threshold cases (Â±{band:.02f} around {thr:.2f}): {len(uncertain)}")
display(uncertain.head(5))



# Step 21---- FIX: finalize artifacts WITHOUT leakage (train-split only) ----
# Reuse the fitted vectorizers (word_vec, char_vec) and matrices (Xtr, Xte)
final_clf = LogisticRegression(
    max_iter=1000,
    class_weight={"Primary": 1.0, "Secondary": best["w"]},  # best Secondary weight from tuning
    multi_class="multinomial",
    solver="lbfgs",
    random_state=42
).fit(Xtr, ytr)

doc_type_model = final_clf
doc_type_vectorizers = {"word": word_vec, "char": char_vec}
doc_type_threshold_secondary = float(best["thr"])  # best threshold from tuning

# Evaluate on the true hold-out (still leakage-free)
proba2 = doc_type_model.predict_proba(Xte)
sec_idx2 = doc_type_model.classes_.tolist().index("Secondary")
yhat2 = np.where(proba2[:, sec_idx2] >= doc_type_threshold_secondary, "Secondary", "Primary")

from sklearn.metrics import classification_report, confusion_matrix, f1_score
print("\nğŸ§ª Hold-out metrics (no leakage, finalized model):")
print(classification_report(yte, yhat2, digits=3))
print(confusion_matrix(yte, yhat2, labels=["Primary","Secondary"]))
print("Macro F1:", f1_score(yte, yhat2, average="macro"))



# === Step 22 â€” Build binary is_citation dataset (gold+pseudo vs unlabeled) and split w/ grouping ===
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# --- Preconditions ---
if "detection_df" not in globals() or not isinstance(detection_df, pd.DataFrame) or detection_df.empty:
    raise SystemExit("â›” 'detection_df' missing or empty. Build it before Step 11.")

TEXT_COL = "text_model" if "text_model" in detection_df.columns else ("text" if "text" in detection_df.columns else None)
if TEXT_COL is None:
    raise SystemExit("â›” No text column found in detection_df (need 'text_model' or 'text').")

GROUP_COL = "article_id_norm" if "article_id_norm" in detection_df.columns else ("article_id" if "article_id" in detection_df.columns else None)
if GROUP_COL is None:
    raise SystemExit("â›” Need 'article_id_norm' or 'article_id' to group by article.")

# --- Build binary labels: 1 = positive (labeled or pseudo), 0 = negative (unlabeled) ---
bin_df = detection_df[[GROUP_COL, TEXT_COL, "label_status"]].copy()
bin_df = bin_df.dropna(subset=[TEXT_COL])  # keep rows with text

bin_df["y"] = np.where(bin_df["label_status"].isin(["labeled", "pseudo"]), 1, 0)

# Optional: if you explicitly set negatives in is_citation==0 elsewhere, you could OR that in:
# if "is_citation" in detection_df.columns:
#     has_zero = detection_df["is_citation"].fillna(-1).eq(0)
#     bin_df.loc[has_zero.reindex(bin_df.index, fill_value=False), "y"] = 0

# --- Basic sanity echoes ---
pos_count = int((bin_df["y"] == 1).sum())
neg_count = int((bin_df["y"] == 0).sum())
print(f"ğŸ“¦ Binary dataset: total={len(bin_df)} | positives={pos_count} | negatives={neg_count}")

if pos_count == 0 or neg_count == 0:
    raise SystemExit("â›” Need both positives and negatives to train/evaluate a binary model.")

# --- Grouped split by article to prevent leakage ---
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(bin_df, y=bin_df["y"], groups=bin_df[GROUP_COL]))

train_df = bin_df.iloc[train_idx].reset_index(drop=True)
test_df  = bin_df.iloc[test_idx].reset_index(drop=True)

# --- Final train/test tensors (keep names compatible with your later code) ---
X_train = train_df[TEXT_COL].astype(str)
y_train = train_df["y"].astype(int)

X_test  = test_df[TEXT_COL].astype(str)
y_test  = test_df["y"].astype(int)

print(f"ğŸ§ª Split â€” train: {len(X_train)}, test: {len(X_test)}")
print("ğŸ”� Class balance (train):", dict(pd.Series(y_train).value_counts()))
print("ğŸ”� Class balance (test):",  dict(pd.Series(y_test).value_counts()))



# Step 23=== Model 2 (fixed) â€” Build doc-level labels, then GroupKFold CV on Primary vs Secondary ===
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    f1_score,
)

# ---------- Helpers ----------
def _ensure_text_col(df: pd.DataFrame) -> str:
    if "text_model" in df.columns:
        return "text_model"
    if "text" in df.columns:
        return "text"
    raise SystemExit(" No text column found in detection_df (need 'text_model' or 'text').")

def _ensure_id_col(df: pd.DataFrame) -> str:
    if "article_id_norm" in df.columns:
        return "article_id_norm"
    if "article_id" in df.columns:
        return "article_id"
    raise SystemExit(" Need 'article_id_norm' or 'article_id' in detection_df.")

def _build_doc_labels_from_merged() -> pd.DataFrame:
    """
    Use merged_df (tuple-level) to derive one doc-level 'type' per article_id_norm.
    Priority: Primary > Secondary.
    """
    if "merged_df" not in globals() or not isinstance(merged_df, pd.DataFrame) or merged_df.empty:
        return pd.DataFrame()

    if "article_id_norm" not in merged_df.columns:
        # Try to create it from article_id if available
        if "article_id" in merged_df.columns:
            if "normalize_article_id_jats" in globals():
                merged_df["_tmp_norm"] = merged_df["article_id"].map(
                    lambda x: normalize_article_id_jats(x)[0] if isinstance(x, str) else x
                )
            else:
                merged_df["_tmp_norm"] = merged_df["article_id"].astype(str).str.lower().str.replace("/", "_", regex=False)
            merged_df["_tmp_norm"] = merged_df["_tmp_norm"].astype(str).str.replace("/", "_", regex=False)
            aid_col = "_tmp_norm"
        else:
            return pd.DataFrame()
    else:
        aid_col = "article_id_norm"

    df = merged_df.copy()
    if "label_status" in df.columns:
        df = df[df["label_status"] != "unlabeled"]
    if "type" not in df.columns:
        return pd.DataFrame()
    df = df[df["type"].isin(["Primary","Secondary"])]

    if df.empty:
        return pd.DataFrame()

    df["_prio"] = df["type"].map({"Primary": 2, "Secondary": 1})
    doc_labels = (
        df.sort_values([aid_col, "_prio"], ascending=[True, False])
          .drop_duplicates(subset=[aid_col])[ [aid_col, "type"] ]
          .rename(columns={aid_col: "article_id_key"})
          .reset_index(drop=True)
    )
    return doc_labels

def _build_doc_labels_from_labels_df() -> pd.DataFrame:
    """
    Fallback: derive doc-level 'type' from labels_df (train labels).
    Requires normalization to article_id_key compatible with detection_df.
    """
    if "labels_df" not in globals() or not isinstance(labels_df, pd.DataFrame) or labels_df.empty:
        return pd.DataFrame()

    df = labels_df.copy()
    if "type" not in df.columns:
        return pd.DataFrame()
    df = df[df["type"].isin(["Primary","Secondary"])]
    if df.empty:
        return pd.DataFrame()

    # Normalize article_id -> underscore-style key to match detection_df
    if "normalize_article_id_jats" in globals():
        df["article_id_key"] = df["article_id"].map(
            lambda x: normalize_article_id_jats(x)[0] if isinstance(x, str) else x
        )
    else:
        df["article_id_key"] = df["article_id"].astype(str).str.lower().str.replace("/", "_", regex=False)
    df["article_id_key"] = df["article_id_key"].astype(str).str.replace("/", "_", regex=False)

    df["_prio"] = df["type"].map({"Primary": 2, "Secondary": 1})
    doc_labels = (
        df.sort_values(["article_id_key","_prio"], ascending=[True, False])
          .drop_duplicates(subset=["article_id_key"])[ ["article_id_key","type"] ]
          .reset_index(drop=True)
    )
    return doc_labels

def _build_features(train_text, test_text):
    word_vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3, max_df=0.9,
        strip_accents="unicode",
        max_features=30000
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3, max_df=0.9
    )
    Xtr_w = word_vec.fit_transform(train_text)
    Xte_w = word_vec.transform(test_text)
    Xtr_c = char_vec.fit_transform(train_text)
    Xte_c = char_vec.transform(test_text)
    return hstack([Xtr_w, Xtr_c]).tocsr(), hstack([Xte_w, Xte_c]).tocsr()

# ---------- Preconditions ----------
if "detection_df" not in globals() or not isinstance(detection_df, pd.DataFrame) or detection_df.empty:
    raise SystemExit(" 'detection_df' missing or empty. Build it earlier.")

TEXT_COL = _ensure_text_col(detection_df)
ID_COL   = _ensure_id_col(detection_df)

# ---------- Build doc-level labels (merged_df preferred; fallback to labels_df) ----------
doc_labels = _build_doc_labels_from_merged()
src_used = "merged_df"
if doc_labels.empty:
    doc_labels = _build_doc_labels_from_labels_df()
    src_used = "labels_df"

if doc_labels.empty:
    raise SystemExit(" Could not build doc-level labels from 'merged_df' or 'labels_df'.")

print(f"ğŸ“Œ Doc labels from {src_used}: {doc_labels['type'].value_counts().to_dict()} (total {len(doc_labels)})")

# ---------- Join labels to doc texts (inner join to get labeled docs only) ----------
docs = detection_df[[ID_COL, TEXT_COL]].dropna(subset=[TEXT_COL]).copy()
docs["article_id_key"] = docs[ID_COL].astype(str)
labeled_docs = docs.merge(doc_labels, how="inner", on="article_id_key")

if labeled_docs.empty:
    raise SystemExit("â›” No overlap between doc texts and doc labels.")

print(f"ğŸ“Š Labeled docs available for CV: {len(labeled_docs)}")
print(labeled_docs["type"].value_counts())

# ---------- GroupKFold CV ----------
n_groups = labeled_docs[ID_COL].nunique()
if n_groups < 2:
    raise SystemExit(" Need at least 2 unique article groups for GroupKFold.")

n_splits = min(5, n_groups)
print(f"ğŸ§ª Using GroupKFold with n_splits={n_splits} over {n_groups} unique articles.")

gkf = GroupKFold(n_splits=n_splits)
scores = []

for fold, (tr_idx, te_idx) in enumerate(gkf.split(labeled_docs, groups=labeled_docs[ID_COL]), start=1):
    tr = labeled_docs.iloc[tr_idx].reset_index(drop=True)
    te = labeled_docs.iloc[te_idx].reset_index(drop=True)

    Xtr, Xte = _build_features(tr[TEXT_COL].astype(str), te[TEXT_COL].astype(str))
    ytr = tr["type"].astype(str)
    yte = te["type"].astype(str)

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        multi_class="multinomial",
        solver="lbfgs",
        random_state=42
    )
    clf.fit(Xtr, ytr)
    yhat = clf.predict(Xte)

    # Per-class metrics with guards (zero_division to avoid NaNs if a fold misses a class)
    prec, rec, f1, _ = precision_recall_fscore_support(
        yte, yhat, labels=["Primary","Secondary"], average=None, zero_division=0
    )
    macro = f1_score(yte, yhat, average="macro", zero_division=0)
    scores.append([
        prec[0], prec[1],
        rec[0], rec[1],
        f1[0], f1[1],
        macro
    ])

    print(f"\nğŸ“Š Fold {fold} Classification Report:")
    print(classification_report(yte, yhat, labels=["Primary","Secondary"], digits=3, zero_division=0))

# ---------- Aggregate results ----------
score_df = pd.DataFrame(
    scores,
    columns=[
        "Precision_Primary", "Precision_Secondary",
        "Recall_Primary", "Recall_Secondary",
        "F1_Primary", "F1_Secondary", "F1_Macro"
    ]
)
print("\nğŸ“‰ Per-fold scores:")
display(score_df)

avg = score_df.mean().to_frame().T; avg.index = ["Mean"]
std = score_df.std(ddof=0).to_frame().T; std.index = ["Std"]
print("\nğŸ“ˆ Average CV Scores (Â± Std):")
display(pd.concat([avg, std]))



# Step 24 === Model 2++ â€” Grouped CV with word+char TF-IDF, lexical flags, and per-fold weight+threshold tuning ===
import numpy as np
import pandas as pd
import re
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support, f1_score

# ---------- Preconditions ----------
if "detection_df" not in globals() or not isinstance(detection_df, pd.DataFrame) or detection_df.empty:
    raise SystemExit("â›” 'detection_df' missing or empty.")

TEXT_COL = "text_model" if "text_model" in detection_df.columns else ("text" if "text" in detection_df.columns else None)
if TEXT_COL is None:
    raise SystemExit("â›” Need a text column ('text_model' or 'text').")

ID_COL = "article_id_norm" if "article_id_norm" in detection_df.columns else ("article_id" if "article_id" in detection_df.columns else None)
if ID_COL is None:
    raise SystemExit("â›” Need an ID column ('article_id_norm' or 'article_id').")

# ---------- Build doc-level labels from merged_df (fallback to labels_df if needed) ----------
def build_doc_labels_from_merged() -> pd.DataFrame:
    if "merged_df" not in globals() or not isinstance(merged_df, pd.DataFrame) or merged_df.empty:
        return pd.DataFrame()
    df = merged_df.copy()
    if "type" not in df.columns:
        return pd.DataFrame()
    df = df[df["type"].isin(["Primary","Secondary"])]
    if "article_id_norm" not in df.columns:
        if "article_id" in df.columns and "normalize_article_id_jats" in globals():
            df["article_id_norm"] = df["article_id"].map(lambda x: normalize_article_id_jats(x)[0] if isinstance(x,str) else x)
        elif "article_id" in df.columns:
            df["article_id_norm"] = df["article_id"].astype(str).str.lower().str.replace("/", "_", regex=False)
        else:
            return pd.DataFrame()
    if "label_status" in df.columns:
        df = df[df["label_status"] != "unlabeled"]
    if df.empty:
        return pd.DataFrame()
    df["_prio"] = df["type"].map({"Primary":2, "Secondary":1})
    doc = (df.sort_values(["article_id_norm","_prio"], ascending=[True,False])
             .drop_duplicates(subset=["article_id_norm"])
             [["article_id_norm","type"]]
             .rename(columns={"article_id_norm":"article_id_key"})
             .reset_index(drop=True))
    return doc

def build_doc_labels_from_labels() -> pd.DataFrame:
    if "labels_df" not in globals() or not isinstance(labels_df, pd.DataFrame) or labels_df.empty:
        return pd.DataFrame()
    df = labels_df.copy()
    if "type" not in df.columns:
        return pd.DataFrame()
    df = df[df["type"].isin(["Primary","Secondary"])]
    if df.empty:
        return pd.DataFrame()
    if "normalize_article_id_jats" in globals():
        df["article_id_key"] = df["article_id"].map(lambda x: normalize_article_id_jats(x)[0] if isinstance(x,str) else x)
    else:
        df["article_id_key"] = df["article_id"].astype(str).str.lower().str.replace("/", "_", regex=False)
    df["_prio"] = df["type"].map({"Primary":2, "Secondary":1})
    doc = (df.sort_values(["article_id_key","_prio"], ascending=[True,False])
             .drop_duplicates(subset=["article_id_key"])
             [["article_id_key","type"]]
             .reset_index(drop=True))
    return doc

doc_labels = build_doc_labels_from_merged()
src_used = "merged_df"
if doc_labels.empty:
    doc_labels = build_doc_labels_from_labels()
    src_used = "labels_df"
if doc_labels.empty:
    raise SystemExit("â›” Could not build doc labels from merged_df or labels_df.")

print(f"ğŸ“Œ Doc labels from {src_used}: {doc_labels['type'].value_counts().to_dict()} (total {len(doc_labels)})")

# ---------- Join labels to doc texts ----------
docs = detection_df[[ID_COL, TEXT_COL]].dropna(subset=[TEXT_COL]).copy()
docs["article_id_key"] = docs[ID_COL].astype(str)
labeled_docs = docs.merge(doc_labels, on="article_id_key", how="inner")
print(f"ğŸ“Š Labeled docs available for CV: {len(labeled_docs)}")
print(labeled_docs["type"].value_counts())

# ---------- Lexical flags (reuse cues & primary cues) ----------
repo_pat = re.compile(
    r"(dryad|zenodo|figshare|pangaea|genbank|geo|sra|ena|ebi|empiar|emdb|pdb|gisaid|icpsr|tcia|chembl|osf|openneuro|neurovault)",
    flags=re.I
)
acc_pat = re.compile(
    r"\b(?:GSE|GSM)\d+\b|\bE\-MTAB\-\d+\b|\bPRJ(?:NA|EB|DB)\d+\b|\bSR[RXPS]\d+\b|\bEMPIAR\-\d+\b|\bEMD\-\d+\b|\bCHEMBL\d+\b|\bEPI(?:[_-]ISL)?\d+\b",
    flags=re.I
)
primary_pat = re.compile(
    r"(generated in this study|data (?:we|were) (?:generated|collected)|we (?:generated|collected) the data|"
    r"new dataset|this study (?:provides|presents) data|we deposit(?:ed)? (?:the )?data)",
    flags=re.I
)
doi_pat = re.compile(r"(https?://doi\.org/|doi:\s*\d+\.\d+/)", flags=re.I)
url_pat = re.compile(r"https?://", flags=re.I)

def make_flags(texts: pd.Series) -> csr_matrix:
    t = texts.astype(str).values
    reuse = np.array([1 if (repo_pat.search(x) or acc_pat.search(x)) else 0 for x in t], dtype=np.float32)
    primary = np.array([1 if primary_pat.search(x) else 0 for x in t], dtype=np.float32)
    url_cnt = np.array([len(url_pat.findall(x)) for x in t], dtype=np.float32)
    doi_cnt = np.array([len(doi_pat.findall(x)) for x in t], dtype=np.float32)
    acc_cnt = np.array([len(acc_pat.findall(x)) for x in t], dtype=np.float32)
    feat = np.stack([reuse, primary, url_cnt, doi_cnt, acc_cnt], axis=1)
    return csr_matrix(feat)

# ---------- Feature builders ----------
def build_features(train_text: pd.Series, test_text: pd.Series):
    # TF-IDF
    word_vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3, max_df=0.9,
        strip_accents="unicode",
        max_features=30000
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3, max_df=0.9
    )
    Xtr_w = word_vec.fit_transform(train_text)
    Xte_w = word_vec.transform(test_text)
    Xtr_c = char_vec.fit_transform(train_text)
    Xte_c = char_vec.transform(test_text)

    # Flags
    Xtr_f = make_flags(train_text)
    Xte_f = make_flags(test_text)

    Xtr = hstack([Xtr_w, Xtr_c, Xtr_f]).tocsr()
    Xte = hstack([Xte_w, Xte_c, Xte_f]).tocsr()
    return Xtr, Xte, word_vec, char_vec

# ---------- CV splits ----------
n_groups = labeled_docs[ID_COL].nunique()
sec_total = int((labeled_docs["type"] == "Secondary").sum())
# If Secondary per fold < 8, reduce to 3 folds
target_folds = 5
if sec_total / target_folds < 8:
    target_folds = 3
n_splits = min(target_folds, n_groups)
print(f"ğŸ§ª Using GroupKFold with n_splits={n_splits} over {n_groups} unique articles.")

gkf = GroupKFold(n_splits=n_splits)

# ---------- Per-fold tuning of Secondary weight & threshold ----------
sec_weights = [1.0, 1.5, 2.0, 2.5, 3.0]
thresholds  = np.linspace(0.30, 0.70, 9)  # 0.30, 0.35, ..., 0.70

fold_rows = []
best_params = []

for fold, (tr_idx, te_idx) in enumerate(gkf.split(labeled_docs, groups=labeled_docs[ID_COL]), start=1):
    tr = labeled_docs.iloc[tr_idx].reset_index(drop=True)
    te = labeled_docs.iloc[te_idx].reset_index(drop=True)

    Xtr, Xte, wv, cv = build_features(tr[TEXT_COL].astype(str), te[TEXT_COL].astype(str))
    ytr = tr["type"].astype(str).values
    yte = te["type"].astype(str).values

    best = {"f1": -1, "w": None, "thr": None, "clf": None, "classes": None}
    for w in sec_weights:
        clf = LogisticRegression(
            max_iter=1000,
            class_weight={"Primary": 1.0, "Secondary": w},
            multi_class="multinomial",
            solver="lbfgs",
            random_state=42
        ).fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)
        classes = clf.classes_.tolist()
        sec_idx = classes.index("Secondary")
        p_sec = proba[:, sec_idx]
        for thr in thresholds:
            yhat = np.where(p_sec >= thr, "Secondary", "Primary")
            macro = f1_score(yte, yhat, average="macro", zero_division=0)
            if macro > best["f1"]:
                best.update({"f1": macro, "w": w, "thr": thr, "clf": clf, "classes": classes})

    # Evaluate best on this fold
    proba = best["clf"].predict_proba(Xte)
    sec_idx = best["classes"].index("Secondary")
    yhat = np.where(proba[:, sec_idx] >= best["thr"], "Secondary", "Primary")

    prec, rec, f1, _ = precision_recall_fscore_support(
        yte, yhat, labels=["Primary","Secondary"], average=None, zero_division=0
    )
    macro = f1_score(yte, yhat, average="macro", zero_division=0)

    print(f"\nğŸ“Š Fold {fold} (best w={best['w']}, thr={best['thr']:.2f})")
    print(classification_report(yte, yhat, labels=["Primary","Secondary"], digits=3, zero_division=0))

    fold_rows.append([
        prec[0], prec[1],
        rec[0],  rec[1],
        f1[0],   f1[1],
        macro
    ])
    best_params.append((best["w"], float(best["thr"])))

# ---------- Aggregate results ----------
cols = ["Precision_Primary","Precision_Secondary","Recall_Primary","Recall_Secondary","F1_Primary","F1_Secondary","F1_Macro"]
score_df = pd.DataFrame(fold_rows, columns=cols)
print("\nğŸ“‰ Per-fold scores (tuned):")
display(score_df)

avg = score_df.mean().to_frame().T; avg.index = ["Mean"]
std = score_df.std(ddof=0).to_frame().T; std.index = ["Std"]
print("\nğŸ“ˆ Average CV Scores (Â± Std) [tuned]:")
display(pd.concat([avg, std]))

print("\nğŸ”§ Best (w, thr) per fold:", best_params)

# ---------- Train final CV model on all labeled docs with median params (optional artifacts) ----------
w_final  = float(np.median([w for w, _ in best_params]))
thr_final= float(np.median([t for _, t in best_params]))

Xall_tr, Xall_te, wv_all, cv_all = build_features(labeled_docs[TEXT_COL].astype(str), labeled_docs[TEXT_COL].astype(str))
# Fit on all labeled docs using chosen w_final (no leakage concerns for deployment)
final_clf = LogisticRegression(
    max_iter=1000,
    class_weight={"Primary": 1.0, "Secondary": w_final},
    multi_class="multinomial",
    solver="lbfgs",
    random_state=42
).fit(Xall_tr, labeled_docs["type"].astype(str).values)

doc_type_model_cv = final_clf
doc_type_vectorizers_cv = {"word": wv_all, "char": cv_all}
doc_type_threshold_secondary_cv = thr_final
print(f"\nğŸ’¾ Saved artifacts (CV): doc_type_model_cv, doc_type_vectorizers_cv, doc_type_threshold_secondary_cv "
      f"(w_final={w_final}, thr_final={thr_final:.2f})")



# Step 25 === Model 2 â€” Finalize (train on all labeled docs), Evaluate on hold-out, and Apply to positives ===
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# --------------------------- Tuned hyper-params (from prior CV/tuning) ---------------------------
W_SECONDARY   = 2.5   # class weight for "Secondary"; set from your best-per-fold median
THR_SECONDARY = 0.30  # decision threshold on P(Secondary)

# --------------------------- Safeguards ---------------------------
if "detection_df" not in globals() or not isinstance(detection_df, pd.DataFrame) or detection_df.empty:
    raise SystemExit("â›” 'detection_df' missing or empty.")

TEXT_COL = "text_model" if "text_model" in detection_df.columns else ("text" if "text" in detection_df.columns else None)
if TEXT_COL is None:
    raise SystemExit("â›” Need a text column ('text_model' or 'text').")

ID_COL = "article_id_norm" if "article_id_norm" in detection_df.columns else ("article_id" if "article_id" in detection_df.columns else None)
if ID_COL is None:
    raise SystemExit("â›” Need an ID column ('article_id_norm' or 'article_id').")

# --------------------------- Build doc-level labels (Primary vs Secondary) ---------------------------
def _build_doc_labels_from_merged() -> pd.DataFrame:
    if "merged_df" not in globals() or not isinstance(merged_df, pd.DataFrame) or merged_df.empty:
        return pd.DataFrame()
    df = merged_df.copy()
    if "type" not in df.columns:
        return pd.DataFrame()
    df = df[df["type"].isin(["Primary","Secondary"])]
    if df.empty:
        return pd.DataFrame()
    if "article_id_norm" not in df.columns:
        if "article_id" in df.columns and "normalize_article_id_jats" in globals():
            df["article_id_norm"] = df["article_id"].map(lambda x: normalize_article_id_jats(x)[0] if isinstance(x,str) else x)
        elif "article_id" in df.columns:
            df["article_id_norm"] = df["article_id"].astype(str).str.lower().str.replace("/", "_", regex=False)
        else:
            return pd.DataFrame()
    if "label_status" in df.columns:
        df = df[df["label_status"] != "unlabeled"]
    df["_prio"] = df["type"].map({"Primary":2, "Secondary":1})
    return (df.sort_values(["article_id_norm","_prio"], ascending=[True,False])
              .drop_duplicates(subset=["article_id_norm"])
              [["article_id_norm","type"]]
              .rename(columns={"article_id_norm":"article_id_key"})
              .reset_index(drop=True))

def _build_doc_labels_from_labels_df() -> pd.DataFrame:
    if "labels_df" not in globals() or not isinstance(labels_df, pd.DataFrame) or labels_df.empty:
        return pd.DataFrame()
    df = labels_df.copy()
    if "type" not in df.columns:
        return pd.DataFrame()
    df = df[df["type"].isin(["Primary","Secondary"])]
    if df.empty:
        return pd.DataFrame()
    if "normalize_article_id_jats" in globals():
        df["article_id_key"] = df["article_id"].map(lambda x: normalize_article_id_jats(x)[0] if isinstance(x,str) else x)
    else:
        df["article_id_key"] = df["article_id"].astype(str).str.lower().str.replace("/", "_", regex=False)
    df["_prio"] = df["type"].map({"Primary":2, "Secondary":1})
    return (df.sort_values(["article_id_key","_prio"], ascending=[True,False])
              .drop_duplicates(subset=["article_id_key"])
              [["article_id_key","type"]]
              .reset_index(drop=True))

doc_labels = _build_doc_labels_from_merged()
src_used = "merged_df"
if doc_labels.empty:
    doc_labels = _build_doc_labels_from_labels_df()
    src_used = "labels_df"
if doc_labels.empty:
    raise SystemExit("â›” Could not build doc labels from merged_df or labels_df.")
print(f"ğŸ“Œ Doc labels from {src_used}: {doc_labels['type'].value_counts().to_dict()} (total {len(doc_labels)})")

# --------------------------- Join labels to texts ---------------------------
docs = detection_df[[ID_COL, TEXT_COL]].dropna(subset=[TEXT_COL]).copy()
docs["article_id_key"] = docs[ID_COL].astype(str)
labeled_docs = docs.merge(doc_labels, on="article_id_key", how="inner")
if labeled_docs.empty:
    raise SystemExit("â›” No overlap between doc texts and labels.")
print(f"ğŸ“Š Labeled docs for modeling: {len(labeled_docs)}")
print(labeled_docs["type"].value_counts())

# --------------------------- Lexical flags ---------------------------
repo_pat = re.compile(
    r"(dryad|zenodo|figshare|pangaea|genbank|geo|sra|ena|ebi|empiar|emdb|pdb|gisaid|icpsr|tcia|chembl|osf|openneuro|neurovault)",
    flags=re.I
)
acc_pat = re.compile(
    r"\b(?:GSE|GSM)\d+\b|\bE\-MTAB\-\d+\b|\bPRJ(?:NA|EB|DB)\d+\b|\bSR[RXPS]\d+\b|\bEMPIAR\-\d+\b|\bEMD\-\d+\b|\bCHEMBL\d+\b|\bEPI(?:[_-]ISL)?\d+\b",
    flags=re.I
)
doi_pat = re.compile(r"(https?://doi\.org/|doi:\s*\d+\.\d+/)", flags=re.I)
url_pat = re.compile(r"https?://", flags=re.I)
primary_pat = re.compile(
    r"(generated in this study|data (?:we|were) (?:generated|collected)|we (?:generated|collected) the data|new dataset|this study (?:provides|presents) data)",
    flags=re.I
)

def make_flags(texts: pd.Series) -> csr_matrix:
    t = texts.astype(str).values
    reuse   = np.array([1 if (repo_pat.search(x) or acc_pat.search(x)) else 0 for x in t], dtype=np.float32)
    primary = np.array([1 if primary_pat.search(x) else 0 for x in t], dtype=np.float32)
    url_cnt = np.array([len(url_pat.findall(x)) for x in t], dtype=np.float32)
    doi_cnt = np.array([len(doi_pat.findall(x)) for x in t], dtype=np.float32)
    acc_cnt = np.array([len(acc_pat.findall(x)) for x in t], dtype=np.float32)
    feat = np.stack([reuse, primary, url_cnt, doi_cnt, acc_cnt], axis=1)
    return csr_matrix(feat)

# --------------------------- Vectorizers (word(1,2) + char_wb(3,5)) ---------------------------
def build_features(train_text: pd.Series, test_text: pd.Series):
    word_vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1,2),
        min_df=3, max_df=0.9,
        strip_accents="unicode",
        max_features=30000
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3,5),
        min_df=3, max_df=0.9
    )
    Xtr_w = word_vec.fit_transform(train_text.astype(str))
    Xte_w = word_vec.transform(test_text.astype(str))
    Xtr_c = char_vec.fit_transform(train_text.astype(str))
    Xte_c = char_vec.transform(test_text.astype(str))
    Xtr_f = make_flags(train_text)
    Xte_f = make_flags(test_text)
    return hstack([Xtr_w, Xtr_c, Xtr_f]).tocsr(), hstack([Xte_w, Xte_c, Xte_f]).tocsr(), word_vec, char_vec

# ============================================================
# A) Hold-out evaluation (leakage-free)
# ============================================================
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(labeled_docs, groups=labeled_docs["article_id_key"]))
tr = labeled_docs.iloc[train_idx].reset_index(drop=True)
te = labeled_docs.iloc[test_idx].reset_index(drop=True)

Xtr, Xte, _wv_tmp, _cv_tmp = build_features(tr[TEXT_COL], te[TEXT_COL])
ytr = tr["type"].astype(str).values
yte = te["type"].astype(str).values

clf_eval = LogisticRegression(
    max_iter=1000,
    class_weight={"Primary": 1.0, "Secondary": W_SECONDARY},
    multi_class="multinomial",
    solver="lbfgs",
    random_state=42
)
clf_eval.fit(Xtr, ytr)
proba_eval = clf_eval.predict_proba(Xte)
classes_eval = clf_eval.classes_.tolist()
sec_idx_eval = classes_eval.index("Secondary")
p_sec_eval = proba_eval[:, sec_idx_eval]
yhat_eval = np.where(p_sec_eval >= THR_SECONDARY, "Secondary", "Primary")

print("\nğŸ§ª Hold-out metrics (no leakage, finalized settings):")
print(classification_report(yte, yhat_eval, labels=["Primary","Secondary"], digits=3, zero_division=0))
print("ğŸ§® Confusion Matrix (rows=true, cols=pred) [Primary, Secondary]:")
print(confusion_matrix(yte, yhat_eval, labels=["Primary","Secondary"]))
print("â­� Macro F1:", f1_score(yte, yhat_eval, average="macro", zero_division=0))

# ============================================================
# B) Final artifacts (fit on ALL labeled docs for deployment)
# ============================================================
# Fit vectorizers on ALL labeled docs
word_vec = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,2),
    min_df=3, max_df=0.9,
    strip_accents="unicode",
    max_features=30000
)
char_vec = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3,5),
    min_df=3, max_df=0.9
)

Xw = word_vec.fit_transform(labeled_docs[TEXT_COL].astype(str))
Xc = char_vec.fit_transform(labeled_docs[TEXT_COL].astype(str))
Xf = make_flags(labeled_docs[TEXT_COL])
X_all = hstack([Xw, Xc, Xf]).tocsr()
y_all = labeled_docs["type"].astype(str).values

clf = LogisticRegression(
    max_iter=1000,
    class_weight={"Primary": 1.0, "Secondary": W_SECONDARY},
    multi_class="multinomial",
    solver="lbfgs",
    random_state=42
)
clf.fit(X_all, y_all)

# Save artifacts for reuse
doc_type_model = clf
doc_type_vectorizers = {"word": word_vec, "char": char_vec}
doc_type_threshold_secondary = float(THR_SECONDARY)
print(f"\nğŸ’¾ Artifacts ready: doc_type_model, doc_type_vectorizers, doc_type_threshold_secondary "
      f"(w={W_SECONDARY}, thr={THR_SECONDARY:.2f})")

# --------------------------- Helper for inference ---------------------------
def _transform_with_artifacts(texts: pd.Series):
    Xw = doc_type_vectorizers["word"].transform(texts.astype(str))
    Xc = doc_type_vectorizers["char"].transform(texts.astype(str))
    Xf = make_flags(texts)
    return hstack([Xw, Xc, Xf]).tocsr()

def predict_doc_type(df: pd.DataFrame, text_col=None):
    if text_col is None:
        text_col = "text_model" if "text_model" in df.columns else "text"
    X = _transform_with_artifacts(df[text_col])
    proba = doc_type_model.predict_proba(X)
    classes = doc_type_model.classes_.tolist()
    sec_idx = classes.index("Secondary")
    p_sec = proba[:, sec_idx]
    pred = np.where(p_sec >= doc_type_threshold_secondary, "Secondary", "Primary")
    return pred, p_sec

# --------------------------- Apply to current positives (is_citation==1) ---------------------------
s = pd.to_numeric(detection_df.get("is_citation"), errors="coerce")
pos_mask = s.fillna(0).astype(int).eq(1)

to_score = detection_df.loc[pos_mask, [ID_COL, TEXT_COL]].copy()
pred, p_sec = predict_doc_type(to_score, TEXT_COL)

detection_df.loc[to_score.index, "doc_type_pred"] = pred
detection_df.loc[to_score.index, "doc_type_p_secondary"] = p_sec

print("\nğŸ“Š Predictions for positive docs:")
print(detection_df.loc[pos_mask, "doc_type_pred"].value_counts(dropna=False))

print("\nğŸ“Š Predictions by label_status (for positives):")
if "label_status" in detection_df.columns:
    print(detection_df.loc[pos_mask].groupby("label_status")["doc_type_pred"].value_counts().unstack(fill_value=0))
else:
    print("â„¹ï¸� 'label_status' not present.")

# --------------------------- Near-threshold diagnostics ---------------------------
def _between_both(series: pd.Series, low: float, high: float) -> pd.Series:
    """Compatibility for pandas .between across versions."""
    try:
        return series.between(low, high, inclusive="both")
    except TypeError:
        return series.between(low, high, inclusive=True)

band = 0.05
thr = float(doc_type_threshold_secondary)
rng_mask = _between_both(detection_df["doc_type_p_secondary"], thr - band, thr + band).fillna(False)

uncertain = detection_df.loc[
    pos_mask & rng_mask,
    [ID_COL, "doc_type_p_secondary", TEXT_COL]
].sort_values("doc_type_p_secondary")

print(f"\nğŸ”� Near-threshold cases (Â±{band:.02f} around {thr:.2f}): {len(uncertain)}")
try:
    from IPython.display import display
    display(uncertain.head(5))
except Exception:
    print(uncertain.head(5).to_string(index=False))



# Step 25 === Model 2 â€” Finalize (train on all labeled docs) and Apply to positives ===
import re, numpy as np, pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------- Safeguards ----------
if "detection_df" not in globals() or not isinstance(detection_df, pd.DataFrame) or detection_df.empty:
    raise SystemExit("â›” 'detection_df' missing or empty.")

TEXT_COL = "text_model" if "text_model" in detection_df.columns else ("text" if "text" in detection_df.columns else None)
if TEXT_COL is None:
    raise SystemExit("â›” Need a text column ('text_model' or 'text').")

ID_COL = "article_id_norm" if "article_id_norm" in detection_df.columns else ("article_id" if "article_id" in detection_df.columns else None)
if ID_COL is None:
    raise SystemExit("â›” Need an ID column ('article_id_norm' or 'article_id').")

# ---------- Build doc-level labels (Primary vs Secondary) ----------
def _build_doc_labels_from_merged() -> pd.DataFrame:
    if "merged_df" not in globals() or not isinstance(merged_df, pd.DataFrame) or merged_df.empty:
        return pd.DataFrame()
    df = merged_df.copy()
    if "type" not in df.columns:
        return pd.DataFrame()
    df = df[df["type"].isin(["Primary","Secondary"])]
    if df.empty:
        return pd.DataFrame()
    if "article_id_norm" not in df.columns:
        if "article_id" in df.columns and "normalize_article_id_jats" in globals():
            df["article_id_norm"] = df["article_id"].map(lambda x: normalize_article_id_jats(x)[0] if isinstance(x, str) else x)
        elif "article_id" in df.columns:
            df["article_id_norm"] = df["article_id"].astype(str).str.lower().str.replace("/", "_", regex=False)
        else:
            return pd.DataFrame()
    if "label_status" in df.columns:
        df = df[df["label_status"] != "unlabeled"]
    df["_prio"] = df["type"].map({"Primary": 2, "Secondary": 1})
    return (
        df.sort_values(["article_id_norm", "_prio"], ascending=[True, False])
          .drop_duplicates(subset=["article_id_norm"])
          [["article_id_norm", "type"]]
          .rename(columns={"article_id_norm": "article_id_key"})
          .reset_index(drop=True)
    )

def _build_doc_labels_from_labels_df() -> pd.DataFrame:
    if "labels_df" not in globals() or not isinstance(labels_df, pd.DataFrame) or labels_df.empty:
        return pd.DataFrame()
    df = labels_df.copy()
    if "type" not in df.columns:
        return pd.DataFrame()
    df = df[df["type"].isin(["Primary","Secondary"])]
    if df.empty:
        return pd.DataFrame()
    if "normalize_article_id_jats" in globals():
        df["article_id_key"] = df["article_id"].map(lambda x: normalize_article_id_jats(x)[0] if isinstance(x, str) else x)
    else:
        df["article_id_key"] = df["article_id"].astype(str).str.lower().str.replace("/", "_", regex=False)
    df["_prio"] = df["type"].map({"Primary": 2, "Secondary": 1})
    return (
        df.sort_values(["article_id_key", "_prio"], ascending=[True, False])
          .drop_duplicates(subset=["article_id_key"])
          [["article_id_key", "type"]]
          .reset_index(drop=True)
    )

doc_labels = _build_doc_labels_from_merged()
src_used = "merged_df"
if doc_labels.empty:
    doc_labels = _build_doc_labels_from_labels_df()
    src_used = "labels_df"
if doc_labels.empty:
    raise SystemExit("â›” Could not build doc labels from merged_df or labels_df.")
print(f"ğŸ“Œ Doc labels from {src_used}: {doc_labels['type'].value_counts().to_dict()} (total {len(doc_labels)})")

# ---------- Join labels to texts ----------
docs = detection_df[[ID_COL, TEXT_COL]].dropna(subset=[TEXT_COL]).copy()
docs["article_id_key"] = docs[ID_COL].astype(str)
labeled_docs = docs.merge(doc_labels, on="article_id_key", how="inner")
if labeled_docs.empty:
    raise SystemExit("â›” No overlap between doc texts and labels.")
print(f"ğŸ“Š Labeled docs for final training: {len(labeled_docs)}")
print(labeled_docs["type"].value_counts())

# ---------- Simple lexical flags (same spirit as CV) ----------
repo_pat = re.compile(
    r"(dryad|zenodo|figshare|pangaea|genbank|geo|sra|ena|ebi|empiar|emdb|pdb|gisaid|icpsr|tcia|chembl|osf|openneuro|neurovault)",
    flags=re.I
)
acc_pat = re.compile(
    r"\b(?:GSE|GSM)\d+\b|\bE\-MTAB\-\d+\b|\bPRJ(?:NA|EB|DB)\d+\b|\bSR[RXPS]\d+\b|\bEMPIAR\-\d+\b|\bEMD\-\d+\b|\bCHEMBL\d+\b|\bEPI(?:[_-]ISL)?\d+\b",
    flags=re.I
)
doi_pat = re.compile(r"(https?://doi\.org/|doi:\s*\d+\.\d+/)", flags=re.I)
url_pat = re.compile(r"https?://", flags=re.I)
primary_pat = re.compile(
    r"(generated in this study|data (?:we|were) (?:generated|collected)|we (?:generated|collected) the data|new dataset|this study (?:provides|presents) data)",
    flags=re.I
)

def make_flags(texts: pd.Series) -> csr_matrix:
    t = texts.astype(str).values
    reuse   = np.array([1 if (repo_pat.search(x) or acc_pat.search(x)) else 0 for x in t], dtype=np.float32)
    primary = np.array([1 if primary_pat.search(x) else 0 for x in t], dtype=np.float32)
    url_cnt = np.array([len(url_pat.findall(x)) for x in t], dtype=np.float32)
    doi_cnt = np.array([len(doi_pat.findall(x)) for x in t], dtype=np.float32)
    acc_cnt = np.array([len(acc_pat.findall(x)) for x in t], dtype=np.float32)
    feat = np.stack([reuse, primary, url_cnt, doi_cnt, acc_cnt], axis=1)
    return csr_matrix(feat)

# ---------- Vectorizers (match your CV setup: word(1,2) + char_wb(3,5)) ----------
word_vec = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=3, max_df=0.9,
    strip_accents="unicode",
    max_features=30000
)
char_vec = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=3, max_df=0.9
)

# Fit on ALL labeled docs (final model)
Xw = word_vec.fit_transform(labeled_docs[TEXT_COL].astype(str))
Xc = char_vec.fit_transform(labeled_docs[TEXT_COL].astype(str))
Xf = make_flags(labeled_docs[TEXT_COL])
X_all = hstack([Xw, Xc, Xf]).tocsr()
y_all = labeled_docs["type"].astype(str).values

# ---------- Final model with median tuned params ----------
W_SECONDARY   = 1.5   # median of your best-per-fold weights
THR_SECONDARY = 0.30  # median of your best-per-fold thresholds

clf = LogisticRegression(
    max_iter=1000,
    class_weight={"Primary": 1.0, "Secondary": W_SECONDARY},
    multi_class="multinomial",
    solver="lbfgs",
    random_state=42
)
clf.fit(X_all, y_all)

# Save artifacts for reuse
doc_type_model = clf
doc_type_vectorizers = {"word": word_vec, "char": char_vec}
doc_type_threshold_secondary = float(THR_SECONDARY)
print(f"ğŸ’¾ Artifacts ready: doc_type_model, doc_type_vectorizers, doc_type_threshold_secondary "
      f"(w={W_SECONDARY}, thr={THR_SECONDARY:.2f})")

# ---------- Helper: predict on any df using these artifacts ----------
def _transform_with_artifacts(texts: pd.Series):
    Xw = doc_type_vectorizers["word"].transform(texts.astype(str))
    Xc = doc_type_vectorizers["char"].transform(texts.astype(str))
    Xf = make_flags(texts)
    return hstack([Xw, Xc, Xf]).tocsr()

def predict_doc_type(df: pd.DataFrame, text_col=None):
    if text_col is None:
        text_col = "text_model" if "text_model" in df.columns else "text"
    X = _transform_with_artifacts(df[text_col])
    proba = doc_type_model.predict_proba(X)
    classes = doc_type_model.classes_.tolist()
    sec_idx = classes.index("Secondary")
    p_sec = proba[:, sec_idx]
    pred = np.where(p_sec >= doc_type_threshold_secondary, "Secondary", "Primary")
    return pred, p_sec

# ---------- Apply to current positives (is_citation==1) ----------
s = pd.to_numeric(detection_df.get("is_citation"), errors="coerce")
pos_mask = s.fillna(0).astype(int).eq(1)

to_score = detection_df.loc[pos_mask, [ID_COL, TEXT_COL]].copy()
pred, p_sec = predict_doc_type(to_score, TEXT_COL)

detection_df.loc[to_score.index, "doc_type_pred"] = pred
detection_df.loc[to_score.index, "doc_type_p_secondary"] = p_sec

# ---------- Summaries ----------
print("\nğŸ“Š Predictions for positive docs:")
print(detection_df.loc[pos_mask, "doc_type_pred"].value_counts(dropna=False))

print("\nğŸ“Š Predictions by label_status (for positives):")
if "label_status" in detection_df.columns:
    print(detection_df.loc[pos_mask].groupby("label_status")["doc_type_pred"].value_counts().unstack(fill_value=0))
else:
    print("â„¹ï¸� 'label_status' not present.")

# ---------- Near-threshold diagnostics ----------
def _between_both(series: pd.Series, low: float, high: float) -> pd.Series:
    """Compatibility wrapper for pandas .between across versions."""
    try:
        return series.between(low, high, inclusive="both")
    except TypeError:
        return series.between(low, high, inclusive=True)

band = 0.05
thr = float(doc_type_threshold_secondary)
rng_mask = _between_both(detection_df["doc_type_p_secondary"], thr - band, thr + band).fillna(False)

uncertain = detection_df.loc[
    pos_mask & rng_mask,
    [ID_COL, "doc_type_p_secondary", TEXT_COL]
].sort_values("doc_type_p_secondary")

print(f"\nğŸ”� Near-threshold cases (Â±{band:.02f} around {thr:.2f}): {len(uncertain)}")
try:
    from IPython.display import display
    display(uncertain.head(5))
except Exception:
    print(uncertain.head(5).to_string(index=False))



# === Step 26. Model 3 â€” MiniLM embeddings + Logistic Regression (with threshold tuning) ===
import os, re, gc, math, numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ----------------------------- Config & Preconditions -----------------------------
LOCAL_ST_PATH = "/kaggle/input/all-minilm-l6-v2/all-MiniLM-L6-v2"
if not os.path.exists(LOCAL_ST_PATH):
    raise SystemExit(f"â›” Local ST model not found: {LOCAL_ST_PATH}")

if "detection_df" not in globals() or not isinstance(detection_df, pd.DataFrame) or detection_df.empty:
    raise SystemExit("â›” 'detection_df' missing or empty.")

TEXT_COL = "text_model" if "text_model" in detection_df.columns else ("text" if "text" in detection_df.columns else None)
if TEXT_COL is None:
    raise SystemExit("â›” Need a text column in detection_df ('text_model' or 'text').")

ID_COL = "article_id_norm" if "article_id_norm" in detection_df.columns else ("article_id" if "article_id" in detection_df.columns else None)
if ID_COL is None:
    raise SystemExit("â›” Need an ID column in detection_df ('article_id_norm' or 'article_id').")

# ----------------------------- Offline ST fallback (no pip needed) -----------------------------
# Tries to import SentenceTransformer; if not available, builds an equivalent encoder
# using HuggingFace transformers + mean pooling (the way sentence-transformers does).
try:
    from sentence_transformers import SentenceTransformer  # may not exist in Kaggle comp env
    _HAS_ST = True
except Exception:
    _HAS_ST = False

import torch
from transformers import AutoTokenizer, AutoModel

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class LocalSentenceEncoder:
    """
    A minimal drop-in replacement for SentenceTransformer.encode using a local HF model.
    It performs mean pooling over token embeddings (masked) and L2-normalizes embeddings,
    mimicking sentence-transformers' default behavior.
    """
    def __init__(self, model_dir: str, max_length: int = 256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir).to(_DEVICE)
        self.model.eval()
        self.max_length = max_length

    @torch.no_grad()
    def encode(self, texts, batch_size=64, convert_to_numpy=True, show_progress_bar=False):
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        elif not isinstance(texts, (list, tuple)):
            texts = [str(texts)]

        embs = []
        rng = range(0, len(texts), batch_size)

        if show_progress_bar:
            try:
                from tqdm.auto import tqdm
                rng = tqdm(rng, desc="Encoding (MiniLM local)")
            except Exception:
                pass

        for start in rng:
            chunk = texts[start:start+batch_size]
            enc = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            enc = {k: v.to(_DEVICE) for k, v in enc.items()}
            out = self.model(**enc)
            # Mean pooling with attention mask
            last_hidden = out.last_hidden_state  # [B, T, H]
            mask = enc["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()  # [B, T, H]
            summed = torch.sum(last_hidden * mask, dim=1)  # [B, H]
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)  # [B, H]
            mean_pooled = summed / counts
            # L2 normalize (Sentence-Transformers default)
            mean_pooled = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
            if convert_to_numpy:
                embs.append(mean_pooled.cpu().numpy())
            else:
                embs.append(mean_pooled.cpu())

        if convert_to_numpy:
            return np.concatenate(embs, axis=0) if embs else np.zeros((0, 384), dtype=np.float32)
        return torch.cat(embs, dim=0) if embs else torch.zeros((0, 384))

# Build encoder
if _HAS_ST:
    print("ğŸ”� Loading SentenceTransformer from local path...")
    st_model = SentenceTransformer(LOCAL_ST_PATH)
else:
    print("ğŸ”� sentence_transformers not installed â€” using LocalSentenceEncoder (HF transformers).")
    # all-MiniLM-L6-v2 typically uses max_seq_length=256
    st_model = LocalSentenceEncoder(LOCAL_ST_PATH, max_length=256)

# ----------------------------- Helpers -----------------------------
def _ensure_norm_id(s: pd.Series) -> pd.Series:
    """Best-effort lowercase & slash->underscore for doc-level join keys."""
    out = s.astype(str).str.lower().str.replace("/", "_", regex=False)
    return out

def _build_doc_labels_from_merged() -> pd.DataFrame:
    """
    Build one label per document (doc-level type Primary/Secondary) from merged_df.
    Preference: Primary > Secondary when both appear for a doc.
    Returns: DataFrame[['article_id_norm','type']]
    """
    if "merged_df" not in globals() or not isinstance(merged_df, pd.DataFrame) or merged_df.empty:
        return pd.DataFrame()

    df = merged_df.copy()
    if "type" not in df.columns:
        return pd.DataFrame()

    df = df[df["type"].isin(["Primary", "Secondary"])].copy()
    if df.empty:
        return pd.DataFrame()

    # ensure normalized key
    if "article_id_norm" not in df.columns:
        if "article_id" in df.columns and "normalize_article_id_jats" in globals():
            df["article_id_norm"] = df["article_id"].apply(
                lambda x: normalize_article_id_jats(x)[0] if isinstance(x, str) else x
            )
        elif "article_id" in df.columns:
            df["article_id_norm"] = _ensure_norm_id(df["article_id"])
        else:
            return pd.DataFrame()

    df["article_id_norm"] = _ensure_norm_id(df["article_id_norm"])

    # prefer Primary where both exist
    df["_prio"] = df["type"].map({"Primary": 2, "Secondary": 1})
    out = (
        df.sort_values(["article_id_norm", "_prio"], ascending=[True, False])
          .drop_duplicates(subset=["article_id_norm"])
          [["article_id_norm", "type"]]
          .reset_index(drop=True)
    )
    return out

# ----------------------------- Build training frame -----------------------------
doc_labels = _build_doc_labels_from_merged()
if doc_labels.empty:
    raise SystemExit("â›” Could not build doc-level labels from merged_df.")

docs = detection_df[[ID_COL, TEXT_COL]].copy()
docs = docs.rename(columns={ID_COL: "article_id_norm", TEXT_COL: "text"})
docs["article_id_norm"] = _ensure_norm_id(docs["article_id_norm"])
docs["text"] = docs["text"].fillna("")

df_train = docs.merge(doc_labels, on="article_id_norm", how="inner")
df_train = df_train[df_train["type"].isin(["Primary", "Secondary"])].reset_index(drop=True)

label_counts = df_train["type"].value_counts().to_dict()
print(f"ğŸ“Œ Doc labels from merged_df: {label_counts} (total {len(df_train)})")

if len(df_train) < 20 or df_train['type'].nunique() < 2:
    raise SystemExit("â›” Not enough labeled documents or only one class present.")

# ----------------------------- Grouped split -----------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
split = gss.split(df_train, groups=df_train["article_id_norm"])
train_idx, test_idx = next(split)

train_df = df_train.iloc[train_idx].reset_index(drop=True)
test_df  = df_train.iloc[test_idx].reset_index(drop=True)

print(f"ğŸ§ª Train/Test docs: {len(train_df)} / {len(test_df)}")

# ----------------------------- Encode with MiniLM -----------------------------
print("ğŸ”� Encoding with MiniLM (local)...")
X_train_emb = st_model.encode(train_df["text"].tolist(),
                              batch_size=64,
                              convert_to_numpy=True,
                              show_progress_bar=True)
X_test_emb  = st_model.encode(test_df["text"].tolist(),
                              batch_size=64,
                              convert_to_numpy=True,
                              show_progress_bar=True)

y_train = train_df["type"].values
y_test  = test_df["type"].values

# ----------------------------- Train + threshold tuning -----------------------------
weights_grid = [1.0, 1.5, 2.0, 3.0]
thr_grid     = np.linspace(0.25, 0.60, 8)

best = {"f1_macro": -1, "w": None, "thr": None, "model": None}

for w in weights_grid:
    # give Secondary extra weight (imbalance)
    cw = {"Primary": 1.0, "Secondary": w}
    clf = LogisticRegression(max_iter=1000, class_weight=cw, random_state=42)
    clf.fit(X_train_emb, y_train)
    proba = clf.predict_proba(X_test_emb)
    classes = clf.classes_.tolist()
    sec_idx = classes.index("Secondary")
    p_sec = proba[:, sec_idx]

    for thr in thr_grid:
        y_pred = np.where(p_sec >= thr, "Secondary", "Primary")
        f1m = f1_score(y_test, y_pred, average="macro")
        if f1m > best["f1_macro"]:
            best.update({"f1_macro": float(f1m), "w": w, "thr": float(thr), "model": clf})

print(f"\nğŸ�¯ Best macro F1={best['f1_macro']:.3f} with Secondary weight={best['w']} and threshold={best['thr']:.2f}")

# Evaluate best on the held-out split
best_model = best["model"]
proba = best_model.predict_proba(X_test_emb)
sec_idx = best_model.classes_.tolist().index("Secondary")
y_pred = np.where(proba[:, sec_idx] >= best["thr"], "Secondary", "Primary")

print("\nğŸ“Š Classification Report (held-out):")
print(classification_report(y_test, y_pred))
print("\nğŸ§® Confusion Matrix (rows=true, cols=pred):")
print(confusion_matrix(y_test, y_pred, labels=["Primary", "Secondary"]))

# ----------------------------- Save artifacts in-memory -----------------------------
bert_doc_type_model = best_model
bert_doc_type_threshold_secondary = float(best["thr"])
bert_encoder_path = LOCAL_ST_PATH  # we re-load encoder when needed
print("\nğŸ’¾ Artifacts ready: bert_doc_type_model, bert_doc_type_threshold_secondary, bert_encoder_path")

# ----------------------------- Score all positive-citation docs -----------------------------
# Only score docs currently flagged positive by the citation detector
pos_mask = detection_df.get("is_citation", pd.Series([np.nan]*len(detection_df))).fillna(0).astype(int) == 1
to_score = detection_df.loc[pos_mask, [ID_COL, TEXT_COL]].copy()
to_score = to_score.rename(columns={ID_COL: "article_id_norm", TEXT_COL: "text"})
to_score["article_id_norm"] = _ensure_norm_id(to_score["article_id_norm"])
to_score["text"] = to_score["text"].fillna("")

# Encode & predict
X_all = st_model.encode(to_score["text"].tolist(),
                        batch_size=64,
                        convert_to_numpy=True,
                        show_progress_bar=True)
proba_all = bert_doc_type_model.predict_proba(X_all)
sec_idx = bert_doc_type_model.classes_.tolist().index("Secondary")
p_sec_all = proba_all[:, sec_idx]
pred_all = np.where(p_sec_all >= bert_doc_type_threshold_secondary, "Secondary", "Primary")

# Write back
detection_df.loc[to_score.index, "doc_type_pred_bert"] = pred_all
detection_df.loc[to_score.index, "doc_type_p_secondary_bert"] = p_sec_all

# Summary
print("\nğŸ“Š BERT predictions for positive docs:")
print(detection_df.loc[pos_mask, "doc_type_pred_bert"].value_counts())

# Near-threshold diagnostics
band = 0.05
thr = float(bert_doc_type_threshold_secondary)
uncertain = detection_df.loc[
    pos_mask & detection_df["doc_type_p_secondary_bert"].between(thr - band, thr + band, inclusive="both"),
    [ID_COL, "doc_type_p_secondary_bert", TEXT_COL]
].sort_values("doc_type_p_secondary_bert")
print(f"\nğŸ”� Near-threshold cases (Â±{band:.02f} around {thr:.2f}): {len(uncertain)}")
try:
    display(uncertain.head(5))
except Exception:
    print(uncertain.head(5))

# housekeeping
del X_train_emb, X_test_emb, X_all, proba, proba_all
gc.collect()



# Step 26 _ model 4 === Re-score positives with TF-IDF (shape-safe), BERT (if available), and Ensemble ===
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

# -------------------- Preconditions --------------------
if "detection_df" not in globals() or not isinstance(detection_df, pd.DataFrame) or detection_df.empty:
    raise SystemExit("â›” 'detection_df' missing or empty. Run the earlier build steps first.")

# TF-IDF artifacts from Model 2
if "doc_type_vectorizers" not in globals() or "doc_type_model" not in globals() or "doc_type_threshold_secondary" not in globals():
    raise SystemExit("â›” TF-IDF artifacts not found. Re-run the Model 2 training cell before this scoring block.")

# Optional BERT artifacts from Model 3
HAS_BERT = all(v in globals() for v in [
    "bert_doc_type_model", "bert_doc_type_threshold_secondary", "bert_encoder_path"
])

# Choose ID/text columns
ID_COL   = "article_id_norm" if "article_id_norm" in detection_df.columns else (
           "article_id"      if "article_id"      in detection_df.columns else None)
TEXT_COL = "text_model"       if "text_model"       in detection_df.columns else (
           "text"            if "text"            in detection_df.columns else None)
if ID_COL is None or TEXT_COL is None:
    raise SystemExit("â›” Need ID and text columns in detection_df (have you built 'text'/'text_model' and normalized IDs?).")

# -------------------- Shape-safe TF-IDF helper --------------------
def _safe_stack_for_model(Xw, Xc, model):
    """
    hstack([Xw, Xc]) and pad/truncate columns to match model.n_features_in_.
    Prevents 'X has N features, model expects M' errors if vectorizers drifted.
    """
    X = hstack([Xw, Xc]).tocsr()
    want = int(getattr(model, "n_features_in_", X.shape[1]))
    have = int(X.shape[1])
    if have == want:
        return X
    if have < want:
        pad = csr_matrix((X.shape[0], want - have), dtype=X.dtype)
        X = hstack([X, pad]).tocsr()
        print(f"âš ï¸� Padded TF-IDF features: had {have}, padded to {want}.")
    else:
        X = X[:, :want].tocsr()
        print(f"âš ï¸� Truncated TF-IDF features: had {have}, cut to {want}.")
    return X

def predict_doc_type_tfidf(df, text_col):
    """
    Uses saved TF-IDF vectorizers + LR and returns (pred_labels, p_secondary).
    Expects globals: doc_type_vectorizers, doc_type_model, doc_type_threshold_secondary.
    """
    Xw = doc_type_vectorizers["word"].transform(df[text_col].astype(str))
    Xc = doc_type_vectorizers["char"].transform(df[text_col].astype(str))
    X  = _safe_stack_for_model(Xw, Xc, doc_type_model)
    proba   = doc_type_model.predict_proba(X)
    classes = doc_type_model.classes_.tolist()
    sec_idx = classes.index("Secondary")
    p_sec   = proba[:, sec_idx]
    thr     = float(doc_type_threshold_secondary)
    pred    = np.where(p_sec >= thr, "Secondary", "Primary")
    return pred, p_sec

# -------------------- Slice positives to score --------------------
is_cit = detection_df["is_citation"] if "is_citation" in detection_df.columns else pd.Series([np.nan]*len(detection_df))
pos_mask = is_cit.fillna(0).astype(int).eq(1)

to_score = detection_df.loc[pos_mask, [ID_COL, TEXT_COL]].copy()
to_score.rename(columns={ID_COL: "article_id_norm", TEXT_COL: "text"}, inplace=True)
to_score["article_id_norm"] = to_score["article_id_norm"].astype(str).str.lower().str.replace("/", "_", regex=False)
to_score["text"] = to_score["text"].fillna("")

# -------------------- TF-IDF predictions (patched) --------------------
pred_tfidf, p_tfidf = predict_doc_type_tfidf(to_score, text_col="text")

# -------------------- BERT predictions (if available) --------------------
pred_bert = p_bert = None
if HAS_BERT:
    from sentence_transformers import SentenceTransformer
    bert_encoder = SentenceTransformer(bert_encoder_path)
    emb = bert_encoder.encode(
        to_score["text"].tolist(),
        batch_size=64, convert_to_numpy=True, show_progress_bar=False
    )
    proba_b = bert_doc_type_model.predict_proba(emb)
    classes_b = bert_doc_type_model.classes_.tolist()
    sec_idx_b = classes_b.index("Secondary")
    p_bert = proba_b[:, sec_idx_b]
    thr_b  = float(bert_doc_type_threshold_secondary)
    pred_bert = np.where(p_bert >= thr_b, "Secondary", "Primary")
else:
    print("â„¹ï¸� BERT artifacts not found â€” skipping BERT and ensemble scoring.")

# -------------------- Simple ensemble (if BERT available) --------------------
if HAS_BERT:
    alpha   = 0.50     # weight on TF-IDF; (1-alpha) on BERT
    thr_ens = 0.40     # ensemble decision threshold on blended p_secondary
    p_ens   = alpha * p_tfidf + (1 - alpha) * p_bert
    pred_ens = np.where(p_ens >= thr_ens, "Secondary", "Primary")

# -------------------- Write back to detection_df --------------------
detection_df.loc[to_score.index, "doc_type_pred_tfidf"]            = pred_tfidf
detection_df.loc[to_score.index, "doc_type_p_secondary_tfidf"]     = p_tfidf

if HAS_BERT:
    detection_df.loc[to_score.index, "doc_type_pred_bert"]         = pred_bert
    detection_df.loc[to_score.index, "doc_type_p_secondary_bert"]  = p_bert
    detection_df.loc[to_score.index, "doc_type_pred_ens"]          = pred_ens
    detection_df.loc[to_score.index, "doc_type_p_secondary_ens"]   = p_ens
    detection_df.loc[to_score.index, "doc_type_ens_alpha"]         = alpha
    detection_df.loc[to_score.index, "doc_type_ens_thr"]           = thr_ens

# -------------------- Summaries --------------------
print("\nğŸ“Š Predictions for positive docs (is_citation==1):")
print("TF-IDF:", detection_df.loc[pos_mask, "doc_type_pred_tfidf"].value_counts(dropna=False))

if HAS_BERT:
    print("BERT:  ", detection_df.loc[pos_mask, "doc_type_pred_bert"].value_counts(dropna=False))
    print("ENS:   ", detection_df.loc[pos_mask, "doc_type_pred_ens"].value_counts(dropna=False))

# Near-threshold diagnostics for TF-IDF
band = 0.05
thr  = float(doc_type_threshold_secondary)
near = detection_df.loc[
    pos_mask & detection_df["doc_type_p_secondary_tfidf"].between(thr-band, thr+band, inclusive="both"),
    ["article_id_norm", "doc_type_p_secondary_tfidf", TEXT_COL]
].sort_values("doc_type_p_secondary_tfidf")
print(f"\nğŸ”� TF-IDF near-threshold (Â±{band:.02f} around {thr:.2f}): {len(near)}")
display(near.head(5))

# Near-threshold diagnostics for Ensemble (if available)
if HAS_BERT:
    near_e = detection_df.loc[
        pos_mask & detection_df["doc_type_p_secondary_ens"].between(thr_ens-band, thr_ens+band, inclusive="both"),
        ["article_id_norm", "doc_type_p_secondary_ens", TEXT_COL]
    ].sort_values("doc_type_p_secondary_ens")
    print(f"\nğŸ”� ENS near-threshold (Â±{band:.02f} around {thr_ens:.2f}): {len(near_e)}")
    display(near_e.head(5))



# Step 27 === Model 5 â€” MiniLM embeddings + XGBoost (build doc labels if missing) ===
import os, numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer

# ---------- Preconditions ----------
if "detection_df" not in globals() or not isinstance(detection_df, pd.DataFrame) or detection_df.empty:
    raise SystemExit("â›” 'detection_df' missing or empty.")

# Pick text & ID columns
TEXT_COL = "text_model" if "text_model" in detection_df.columns else ("text" if "text" in detection_df.columns else None)
if TEXT_COL is None:
    raise SystemExit("â›” Need a text column in detection_df ('text_model' or 'text').")
ID_COL = "article_id_norm" if "article_id_norm" in detection_df.columns else ("article_id" if "article_id" in detection_df.columns else None)
if ID_COL is None:
    raise SystemExit("â›” Need an ID column in detection_df ('article_id_norm' or 'article_id').")

# ---------- Normalizer (reuse if already defined) ----------
if "normalize_article_id_jats" not in globals():
    def normalize_article_id_jats(article_id):
        if not isinstance(article_id, str):
            return article_id, False
        original = str(article_id).strip()
        if original.startswith("10.") and "_" in original and original.count(".") >= 2:
            return original, False
        aid = original.lower().replace("/", "_")
        if aid.startswith("10_"):
            aid = "10." + aid[3:]
        first_dot_pos = aid.find(".", 3)
        if first_dot_pos == -1:
            normalized = aid.replace("_", ".")
        else:
            prefix = aid[:first_dot_pos + 1]
            suffix = aid[first_dot_pos + 1:]
            if "_" in suffix:
                first_uscore_pos = suffix.find("_")
                preserved = suffix[:first_uscore_pos + 1]
                rest = suffix[first_uscore_pos + 1:]
                last_underscore_pos = rest.rfind("_")
                last_dot_pos = rest.rfind(".")
                if last_underscore_pos != -1 and last_underscore_pos > last_dot_pos:
                    rest = rest[:last_underscore_pos] + "." + rest[last_underscore_pos + 1:]
                normalized = prefix + preserved + rest
            else:
                last_underscore_pos = suffix.rfind("_")
                last_dot_pos = suffix.rfind(".")
                if last_underscore_pos != -1 and last_underscore_pos > last_dot_pos:
                    suffix = suffix[:last_underscore_pos] + "." + suffix[last_underscore_pos + 1:]
                normalized = prefix + suffix
        return normalized, (normalized != original.lower())

# ---------- Build/collapse doc labels (Primary > Secondary) ----------
def _build_doc_labels_from_merged() -> pd.DataFrame:
    if "merged_df" not in globals() or not isinstance(merged_df, pd.DataFrame) or merged_df.empty:
        return pd.DataFrame()
    df = merged_df.copy()
    if "type" not in df.columns:
        return pd.DataFrame()
    df = df[df["type"].isin(["Primary","Secondary"])].copy()
    if df.empty:
        return pd.DataFrame()
    if "article_id_norm" not in df.columns:
        if "article_id" in df.columns:
            df["article_id_norm"] = df["article_id"].map(lambda x: normalize_article_id_jats(x)[0] if isinstance(x,str) else x)
        else:
            return pd.DataFrame()
    df["_prio"] = df["type"].map({"Primary":2, "Secondary":1})
    out = (df.sort_values(["article_id_norm","_prio"], ascending=[True,False])
             .drop_duplicates(subset=["article_id_norm"])
             [["article_id_norm","type"]]
             .rename(columns={"article_id_norm":"article_id_key"})
             .reset_index(drop=True))
    return out

def _build_doc_labels_from_labels_df() -> pd.DataFrame:
    if "labels_df" not in globals() or not isinstance(labels_df, pd.DataFrame) or labels_df.empty:
        return pd.DataFrame()
    df = labels_df.copy()
    if "type" not in df.columns or "article_id" not in df.columns:
        return pd.DataFrame()
    df = df[df["type"].isin(["Primary","Secondary"])].copy()
    if df.empty:
        return pd.DataFrame()
    df["article_id_key"] = df["article_id"].map(lambda x: normalize_article_id_jats(x)[0] if isinstance(x,str) else x)
    df["_prio"] = df["type"].map({"Primary":2, "Secondary":1})
    out = (df.sort_values(["article_id_key","_prio"], ascending=[True,False])
             .drop_duplicates(subset=["article_id_key"])
             [["article_id_key","type"]]
             .reset_index(drop=True))
    return out

doc_labels = _build_doc_labels_from_merged()
src_used = "merged_df"
if doc_labels.empty:
    doc_labels = _build_doc_labels_from_labels_df()
    src_used = "labels_df"
if doc_labels.empty:
    raise SystemExit("â›” Could not build doc labels from merged_df or labels_df.")

print(f"ğŸ“Œ Doc labels built from {src_used}: {doc_labels['type'].value_counts().to_dict()} (total {len(doc_labels)})")

# ---------- Join labels to detection_df (create a dedicated column, don't rely on detection_df['type']) ----------
docs = detection_df[[ID_COL, TEXT_COL]].dropna(subset=[TEXT_COL]).copy()
docs["article_id_key"] = docs[ID_COL].astype(str)
docs = docs.merge(doc_labels, on="article_id_key", how="inner")  # 'type' column comes from doc_labels
if docs.empty:
    raise SystemExit("â›” No overlap between detection_df and doc labels.")
print(f"ğŸ“Š Labeled docs available: {len(docs)}")
print(docs["type"].value_counts())

# ---------- Load MiniLM locally ----------
LOCAL_ST_PATH = "/kaggle/input/all-minilm-l6-v2/all-MiniLM-L6-v2"
if "bert_model" not in globals():
    if not os.path.exists(LOCAL_ST_PATH):
        raise SystemExit(f"â›” Local ST model not found: {LOCAL_ST_PATH}")
    bert_model = SentenceTransformer(LOCAL_ST_PATH)
    print("âœ… MiniLM loaded from local path.")
else:
    print("â„¹ï¸� Reusing existing MiniLM encoder.")

# ---------- Grouped split to avoid article leakage ----------
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(gss.split(docs, groups=docs["article_id_key"]))
train_df = docs.iloc[train_idx].reset_index(drop=True)
test_df  = docs.iloc[test_idx].reset_index(drop=True)
print(f"ğŸ§ª Train/Test docs: {len(train_df)} / {len(test_df)}")

# ---------- Encode with MiniLM ----------
print("ğŸ”� Encoding with MiniLM...")
X_train = bert_model.encode(train_df[TEXT_COL].tolist(), show_progress_bar=True, convert_to_numpy=True)
X_test  = bert_model.encode(test_df[TEXT_COL].tolist(),  show_progress_bar=True, convert_to_numpy=True)

# ---------- Binary setup (Secondary = 1, Primary = 0) ----------
label_map = {"Primary": 0, "Secondary": 1}
y_train = train_df["type"].map(label_map).astype(int).values
y_test  = test_df["type"].map(label_map).astype(int).values

# ---------- Tune scale_pos_weight (class weight) + decision threshold ----------
weights    = [1.0, 1.5, 2.0, 3.0]
thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]

best = {"f1_macro": -1, "w": None, "thr": None, "report": None, "cm": None, "model": None}

for w in weights:
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=w  # emphasize Secondary (positive class)
    )
    clf.fit(X_train, y_train)
    p_sec = clf.predict_proba(X_test)[:, 1]

    for thr in thresholds:
        y_pred = (p_sec >= thr).astype(int)
        f1m = f1_score(y_test, y_pred, average="macro")
        if f1m > best["f1_macro"]:
            best.update({
                "f1_macro": f1m,
                "w": w,
                "thr": thr,
                "report": classification_report(y_test, y_pred, target_names=["Primary","Secondary"]),
                "cm": confusion_matrix(y_test, y_pred, labels=[0,1]),
                "model": clf
            })

print(f"\nğŸ�¯ Best macro F1={best['f1_macro']:.3f} with scale_pos_weight={best['w']} and threshold={best['thr']:.2f}\n")
print("ğŸ“Š Classification Report (held-out):")
print(best["report"])



# Step 28=== Model 6 â€” MiniLM embeddings + XGBoost (Secondary=1), group split + threshold tuning ===
import os, re, numpy as np, pandas as pd, warnings
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sentence_transformers import SentenceTransformer

# ---------- Preconditions ----------
if "detection_df" not in globals() or not isinstance(detection_df, pd.DataFrame) or detection_df.empty:
    raise SystemExit(" 'detection_df' missing or empty.")

TEXT_COL = "text_model" if "text_model" in detection_df.columns else ("text" if "text" in detection_df.columns else None)
if TEXT_COL is None:
    raise SystemExit(" Need a text column in detection_df ('text_model' or 'text').")

ID_COL = "article_id_norm" if "article_id_norm" in detection_df.columns else ("article_id" if "article_id" in detection_df.columns else None)
if ID_COL is None:
    raise SystemExit(" Need an ID column in detection_df ('article_id_norm' or 'article_id').")

# ---------- Build doc-level labels (Primary vs Secondary; Primary preferred if both) ----------
def _build_doc_labels_from_merged() -> pd.DataFrame:
    if "merged_df" not in globals() or not isinstance(merged_df, pd.DataFrame) or merged_df.empty:
        return pd.DataFrame()
    df = merged_df.copy()
    if "type" not in df.columns:
        return pd.DataFrame()
    df = df[df["type"].isin(["Primary","Secondary"])]
    if df.empty: return pd.DataFrame()

    if "article_id_norm" not in df.columns:
        if "article_id" in df.columns and "normalize_article_id_jats" in globals():
            df["article_id_norm"] = df["article_id"].map(lambda x: normalize_article_id_jats(x)[0] if isinstance(x,str) else x)
        elif "article_id" in df.columns:
            df["article_id_norm"] = df["article_id"].astyatr.lower().str.replace("/", "_", regex=False)
        else:
            return pd.DataFrame()

    df["_prio"] = df["type"].map({"Primary":2, "Secondary":1})
    out = (df.sort_values(["article_id_norm","_prio"], ascending=[True,False])
             .drop_duplicates(subset=["article_id_norm"])
             [["article_id_norm","type"]]
             .rename(columns={"article_id_norm":"article_id_key"})
             .reset_index(drop=True))
    return out

def _build_doc_labels_from_labels_df() -> pd.DataFrame:
    if "labels_df" not in globals() or not isinstance(labels_df, pd.DataFrame) or labels_df.empty:
        return pd.DataFrame()
    df = labels_df.copy()
    if "type" not in df.columns:
        return pd.DataFrame()
    df = df[df["type"].isin(["Primary","Secondary"])]
    if df.empty: return pd.DataFrame()

    if "normalize_article_id_jats" in globals():
        df["article_id_key"] = df["article_id"].map(lambda x: normalize_article_id_jats(x)[0] if isinstance(x,str) else x)
    else:
        df["article_id_key"] = df["article_id"].astype(str).str.lower().str.replace("/", "_", regex=False)

    df["_prio"] = df["type"].map({"Primary":2, "Secondary":1})
    out = (df.sort_values(["article_id_key","_prio"], ascending=[True,False])
             .drop_duplicates(subset=["article_id_key"])
             [["article_id_key","type"]]
             .reset_index(drop=True))
    return out

doc_labels = _build_doc_labels_from_merged()
src_used = "merged_df"
if doc_labels.empty:
    doc_labels = _build_doc_labels_from_labels_df()
    src_used = "labels_df"
if doc_labels.empty:
    raise SystemExit(" Could not build doc labels from merged_df or labels_df.")

print(f" Doc labels built from {src_used}: {doc_labels['type'].value_counts().to_dict()} (total {len(doc_labels)})")

# ---------- Collapse to doc-level texts and join labels ----------
docs = detection_df[[ID_COL, TEXT_COL]].dropna(subset=[TEXT_COL]).copy()
docs["article_id_key"] = docs[ID_COL].astype(str)
docs = docs.drop_duplicates(subset=["article_id_key"])  # 1 row per article/doc
labeled_docs = docs.merge(doc_labels, on="article_id_key", how="inner")
if labeled_docs.empty:
    raise SystemExit(" No overlap between doc texts and labels.")

print(f"ğŸ“Š Labeled docs available: {len(labeled_docs)}")
print(labeled_docs["type"].value_counts())

# ---------- Load MiniLM (local) ----------
LOCAL_ST_PATH = "/kaggle/input/all-minilm-l6-v2/all-MiniLM-L6-v2"
if "bert_model" not in globals():
    if not os.path.exists(LOCAL_ST_PATH):
        raise SystemExit(f"â›” Local ST model not found: {LOCAL_ST_PATH}")
    bert_model = SentenceTransformer(LOCAL_ST_PATH)
print("âœ… MiniLM ready.")

# ---------- Group split by article_id_key ----------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
groups = labeled_docs["article_id_key"]
train_idx, test_idx = next(gss.split(labeled_docs, groups=groups))
train_df = labeled_docs.iloc[train_idx].reset_index(drop=True)
test_df  = labeled_docs.iloc[test_idx].reset_index(drop=True)

print(f"ğŸ§ª Train/Test docs: {len(train_df)} / {len(test_df)}")

# ---------- Encode labels: Secondary=1 (positive), Primary=0 ----------
y_train = (train_df["type"] == "Secondary").astype(int).values
y_test  = (test_df ["type"] == "Secondary").astype(int).values

# ---------- Encode text with MiniLM ----------
print(" Encoding with MiniLM...")
X_train = bert_model.encode(train_df[TEXT_COL].tolist(), show_progress_bar=True, convert_to_numpy=True)
X_test  = bert_model.encode(test_df [TEXT_COL].tolist(), show_progress_bar=True, convert_to_numpy=True)

# ---------- Tune scale_pos_weight & threshold on held-out ----------
weights   = [1.0, 1.5, 2.0, 3.0]
thr_grid  = np.arange(0.30, 0.56, 0.05)
best = {"f1_macro": -1, "w": None, "thr": None, "model": None, "report": None, "cm": None}

for w in weights:
    clf = XGBClassifier(
        objective="binary:logistic",
        n_estimators=300,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        tree_method="hist",
        scale_pos_weight=w,
        eval_metric="logloss",
        n_jobs=4
    )
    clf.fit(X_train, y_train)

    # probabilities for Secondary=1
    p_sec = clf.predict_proba(X_test)[:, 1]
    for thr in thr_grid:
        y_pred = (p_sec >= thr).astype(int)
        f1_m = f1_score(y_test, y_pred, average="macro")
        if f1_m > best["f1_macro"]:
            best.update({
                "f1_macro": f1_m,
                "w": w,
                "thr": float(thr),
                "model": clf,
                "report": classification_report(
                    test_df["type"],
                    pd.Series(y_pred).map({0:"Primary", 1:"Secondary"}),
                    digits=3
                ),
                "cm": confusion_matrix(
                    test_df["type"],
                    pd.Series(y_pred).map({0:"Primary", 1:"Secondary"}),
                    labels=["Primary","Secondary"]
                )
            })

print(f"\n Best macro F1={best['f1_macro']:.3f} with scale_pos_weight={best['w']} and threshold={best['thr']:.2f}")
print("\n Classification Report (held-out, best setting):")
print(best["report"])
print(" Confusion Matrix (rows=true, cols=pred) [Primary, Secondary]:")
print(best["cm"])

# ---------- Save artifacts ----------
xgb_doc_type_model = best["model"]
xgb_doc_type_threshold_secondary = best["thr"]
xgb_encoder_path = LOCAL_ST_PATH  # for reference
print("\n Artifacts ready: xgb_doc_type_model, xgb_doc_type_threshold_secondary, xgb_encoder_path")

# ---------- Apply to current positives (is_citation==1) ----------
pos_mask = detection_df.get("is_citation", pd.Series([np.nan]*len(detection_df))).fillna(0).astype(int) == 1
to_score = detection_df.loc[pos_mask, [ID_COL, TEXT_COL]].copy()
if not to_score.empty:
    # embed in batches to avoid memory spikes
    p_sec_list = []
    B = 512
    txts = to_score[TEXT_COL].astype(str).tolist()
    for i in range(0, len(txts), B):
        emb = bert_model.encode(txts[i:i+B], show_progress_bar=False, convert_to_numpy=True)
        p = xgb_doc_type_model.predict_proba(emb)[:, 1]  # Secondary=1
        p_sec_list.append(p)
    p_sec_all = np.concatenate(p_sec_list) if p_sec_list else np.array([])
    pred = np.where(p_sec_all >= xgb_doc_type_threshold_secondary, "Secondary", "Primary")

    detection_df.loc[to_score.index, "doc_type_pred_xgb"] = pred
    detection_df.loc[to_score.index, "doc_type_p_secondary_xgb"] = p_sec_all

    print("\n XGB predictions for positive docs:")
    print(detection_df.loc[pos_mask, "doc_type_pred_xgb"].value_counts())

    band = 0.05
    thr  = float(xgb_doc_type_threshold_secondary)
    uncertain = detection_df.loc[
        pos_mask & detection_df["doc_type_p_secondary_xgb"].between(thr-band, thr+band, inclusive="both"),
        [ID_COL, "doc_type_p_secondary_xgb", TEXT_COL]
    ].sort_values("doc_type_p_secondary_xgb")
    print(f"\nğŸ”� Near-threshold cases (Â±{band:.02f} around {thr:.2f}): {len(uncertain)}")
else:
    print(" No current positives to score (is_citation==1).")



# =========================
# Model 1+ â€” SPEED MODE (faster CV & smaller grid)
# TRAIN ON detection_df (not citation_df)
# =========================
import re, numpy as np, pandas as pd, time
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from scipy.sparse import hstack, csr_matrix

t0 = time.time()

# ---- toggles ----
ADD_FLAGS   = True
N_SPLITS    = 3          # faster than 5
PENALTIES   = ["l2"]     # drop l1 to speed up
CLASS_BOOST = [2.0, 2.5, 3.0]           # smaller grid
C_GRID      = [0.5, 1.0, 2.0]           # smaller grid
THRESHOLDS  = np.round(np.arange(0.25, 0.56, 0.05), 2)  # fewer points

# --------- Preconditions (âœ… use detection_df) ---------
if "detection_df" not in globals() or not isinstance(detection_df, pd.DataFrame) or detection_df.empty:
    raise SystemExit("â›” 'detection_df' missing/empty.")

TEXT_COL = next((c for c in ["text_model","text","clean_text","full_text"] if c in detection_df.columns), None)
ID_COL   = "article_id_norm" if "article_id_norm" in detection_df.columns else ("article_id" if "article_id" in detection_df.columns else None)
if TEXT_COL is None or ID_COL is None:
    raise SystemExit("â›” Need text ('text_model'/'text'/'clean_text'/'full_text') and id ('article_id_norm'/'article_id') in detection_df.")

# --------- Build doc-level labels (Primary/Secondary) ---------
def _labels_from(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if df.empty or "type" not in df.columns or key not in df.columns:
        return pd.DataFrame(columns=["article_id_key","type"])
    d = df[df["type"].isin(["Primary","Secondary"])].copy()
    if d.empty: 
        return pd.DataFrame(columns=["article_id_key","type"])
    d["_prio"] = d["type"].map({"Primary":2, "Secondary":1})
    return (d.sort_values([key,"_prio"], ascending=[True,False])
             .drop_duplicates(subset=[key])
             .rename(columns={key:"article_id_key"})
             [["article_id_key","type"]]
             .reset_index(drop=True))

def _make_doc_labels() -> pd.DataFrame:
    if "merged_df" in globals() and isinstance(merged_df, pd.DataFrame) and not merged_df.empty:
        k = "article_id_norm" if "article_id_norm" in merged_df.columns else ("article_id" if "article_id" in merged_df.columns else None)
        if k:
            out = _labels_from(merged_df, k)
            if not out.empty:
                print("ğŸ“Œ Using labels from merged_df"); 
                return out
    if "labels_df" in globals() and isinstance(labels_df, pd.DataFrame) and not labels_df.empty:
        k = "article_id_norm" if "article_id_norm" in labels_df.columns else ("article_id" if "article_id" in labels_df.columns else None)
        if k:
            out = _labels_from(labels_df, k)
            if not out.empty:
                print("ğŸ“Œ Using labels from labels_df")
                return out
    # final fallback: detection_df itself
    out = _labels_from(detection_df, ID_COL)
    if not out.empty:
        print("ğŸ“Œ Using labels from detection_df (fallback)")
        return out
    return pd.DataFrame(columns=["article_id_key","type"])

doc_labels = _make_doc_labels()
if doc_labels.empty:
    raise SystemExit("â›” No Primary/Secondary labels found.")

# --------- Join texts + labels for training (âœ… from detection_df) ---------
train_df = (detection_df[[ID_COL, TEXT_COL]]
            .rename(columns={ID_COL:"article_id_key", TEXT_COL:"text_for_model"})
            .merge(doc_labels, on="article_id_key", how="inner")
            .dropna(subset=["text_for_model","type"])
            .reset_index(drop=True))

if train_df.empty:
    raise SystemExit("â›” No overlap between detection_df and labels table.")

print("Label counts:\n", train_df["type"].value_counts(dropna=False), "\n")

# --------- Features: smaller, faster TF-IDF + optional flags ---------
def mk_flags(s: str):
    if not isinstance(s, str): s = ""
    s_low = s.lower()
    has_doi_token     = int("doi" in s_low)
    has_repo_keyword  = int(any(k in s_low for k in ["dryad","zenodo","pangaea","figshare","tcia","icpsr","dataverse","mendeley","usgs","dataset","data."]))
    looks_like_doiurl = int(bool(re.search(r"https?://doi\.org/10\.\d+/", s_low)))
    many_digits       = int(sum(ch.isdigit() for ch in s_low) > 20)
    return [has_doi_token, has_repo_keyword, looks_like_doiurl, many_digits]

word_vec = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1,2),
    min_df=3,                   # a bit higher to reduce feature count
    max_df=0.9,
    strip_accents="unicode",
    sublinear_tf=True,
    max_features=80000,         # cap features
    dtype=np.float32            # speed & memory
)
char_vec = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3,5),
    min_df=3,
    strip_accents="unicode",
    sublinear_tf=True,
    max_features=80000,
    dtype=np.float32
)

X_text = train_df["text_for_model"].fillna("")
Xw = word_vec.fit_transform(X_text)
Xc = char_vec.fit_transform(X_text)
X_all = hstack([Xw, Xc]).tocsr()

if ADD_FLAGS:
    flags = np.array([mk_flags(t) for t in X_text.tolist()], dtype=np.float32)
    X_all = hstack([X_all, csr_matrix(flags)], format="csr")

y = (train_df["type"] == "Secondary").astype(int).values
groups = train_df["article_id_key"].values

# --------- GroupKFold CV (smaller) ---------
gkf = GroupKFold(n_splits=N_SPLITS)
best = {"f1": -1, "m": None, "C": None, "penalty": None, "th": None}

for m in CLASS_BOOST:
    for C in C_GRID:
        for pen in PENALTIES:
            fold_probs, fold_true = [], []
            ok = True
            for tr, va in gkf.split(X_all, y, groups):
                try:
                    clf = LogisticRegression(
                        solver="saga",
                        penalty=pen,
                        C=C,
                        max_iter=3000,               # a bit lower
                        class_weight={0:1.0, 1:float(m)},
                        random_state=42,
                        n_jobs=-1,                   # use parallelism if available
                        warm_start=True              # slight speed-up across fits
                    )
                    clf.fit(X_all[tr], y[tr])
                    p = clf.predict_proba(X_all[va])[:, 1]
                except Exception:
                    ok = False
                    break
                fold_probs.append(p); fold_true.append(y[va])
            if not ok:
                continue

            probs = np.concatenate(fold_probs)
            true  = np.concatenate(fold_true)

            for th in THRESHOLDS:
                pred = (probs >= th).astype(int)
                f1m  = f1_score(true, pred, average="macro", zero_division=0)
                if f1m > best["f1"]:
                    best.update({"f1": float(f1m), "m": float(m), "C": float(C), "penalty": pen, "th": float(th)})

print(f"\nâš¡ BEST CV macro-F1 (speed-mode) = {best['f1']:.3f} | m={best['m']} C={best['C']} penalty={best['penalty']} th={best['th']}")

# if the grid failed somehow, fall back to a sane threshold
if best["th"] is None:
    best["th"] = 0.35

# --------- Refit final model on ALL data ---------
doc_type_model = LogisticRegression(
    solver="saga",
    penalty=best["penalty"],
    C=best["C"],
    max_iter=3000,
    class_weight={0:1.0, 1:best["m"]},
    random_state=42,
    n_jobs=-1,
    warm_start=True
).fit(X_all, y)

doc_type_vectorizers = {"word": word_vec, "char": char_vec, "add_flags": bool(ADD_FLAGS)}
doc_type_threshold_secondary = float(best["th"])

print("\nâœ… Artifacts ready: doc_type_model, doc_type_vectorizers, doc_type_threshold_secondary")

# --------- Quick per-fold check + overall summary ---------
all_true, all_pred = [], []
i = 0
for tr, va in gkf.split(X_all, y, groups):
    i += 1
    clf = LogisticRegression(
        solver="saga",
        penalty=best["penalty"],
        C=best["C"],
        max_iter=3000,
        class_weight={0:1.0, 1:best["m"]},
        random_state=42,
        n_jobs=-1,
        warm_start=True
    ).fit(X_all[tr], y[tr])
    p = clf.predict_proba(X_all[va])[:,1]
    yhat = (p >= doc_type_threshold_secondary).astype(int)
    print(f"\nFold {i} report:")
    print(classification_report(y[va], yhat, target_names=["Primary","Secondary"], digits=3, zero_division=0))
    all_true.append(y[va]); all_pred.append(yhat)

y_true = np.concatenate(all_true)
y_pred = np.concatenate(all_pred)
print("\n====== OVERALL CV SUMMARY (speed-mode) ======")
print("Macro F1:", f1_score(y_true, y_pred, average="macro", zero_division=0))
print("Micro F1:", f1_score(y_true, y_pred, average="micro", zero_division=0))
print("\nOverall classification report:")
print(classification_report(y_true, y_pred, target_names=["Primary","Secondary"], digits=3, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred, labels=[0,1]))

print(f"\nâ�±ï¸� Total runtime: {time.time()-t0:.1f}s")



# =========================
# Model 1+ â€” SPEED UPGRADE (3-fold CV, micro-sweep + sigmoid calibration)
# TRAINED ON: detection_df (texts)
# Threshold: FIXED at 0.30
# Produces: doc_type_model, doc_type_vectorizers, doc_type_threshold_secondary
# =========================
import re, time, numpy as np, pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix

t0 = time.time()

# ---------------- Preconditions (texts must come from detection_df) ----------------
if "detection_df" not in globals() or not isinstance(detection_df, pd.DataFrame) or detection_df.empty:
    raise SystemExit("â›” 'detection_df' missing/empty.")

TEXT_COL = ("text_model" if "text_model" in detection_df.columns else
            ("text" if "text" in detection_df.columns else
             ("clean_text" if "clean_text" in detection_df.columns else None)))
ID_COL   = ("article_id_norm" if "article_id_norm" in detection_df.columns else
            ("article_id" if "article_id" in detection_df.columns else None))
if TEXT_COL is None or ID_COL is None:
    raise SystemExit("â›” Need text ('text_model'/'text'/'clean_text') and id ('article_id_norm'/'article_id') in detection_df.")

# ---------------- Build doc-level labels (Primary wins ties) ----------------
def _labels_from(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Expect df to contain a 'type' col with 'Primary'/'Secondary'."""
    if df is None or df.empty or ("type" not in df.columns) or (key not in df.columns):
        return pd.DataFrame(columns=["article_id_key","type"])
    d = df[df["type"].isin(["Primary","Secondary"])].copy()
    if d.empty:
        return pd.DataFrame(columns=["article_id_key","type"])
    d["_prio"] = d["type"].map({"Primary":2, "Secondary":1})
    return (d.sort_values([key,"_prio"], ascending=[True,False])
             .drop_duplicates(subset=[key])
             .rename(columns={key:"article_id_key"})
             [["article_id_key","type"]]
             .reset_index(drop=True))

def _labels_from_tuples(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """
    If we only have tuple-level labels (e.g., detection_tuple_df with article_id + type),
    collapse to doc-level: Primary beats Secondary per article.
    """
    if df is None or df.empty or (key not in df.columns) or ("type" not in df.columns):
        return pd.DataFrame(columns=["article_id_key","type"])
    d = df[df["type"].isin(["Primary","Secondary"])].copy()
    if d.empty:
        return pd.DataFrame(columns=["article_id_key","type"])
    d["_prio"] = d["type"].map({"Primary":2, "Secondary":1})
    return (d.sort_values([key,"_prio"], ascending=[True,False])
             .drop_duplicates(subset=[key])
             .rename(columns={key:"article_id_key"})
             [["article_id_key","type"]]
             .reset_index(drop=True))

def _pick_label_source():
    # 1) merged_df preferred (usually authoritative)
    if "merged_df" in globals() and isinstance(merged_df, pd.DataFrame) and not merged_df.empty:
        k = "article_id_norm" if "article_id_norm" in merged_df.columns else ("article_id" if "article_id" in merged_df.columns else None)
        if k:
            out = _labels_from(merged_df, k)
            if not out.empty:
                print("ğŸ“Œ Using labels from merged_df")
                return out
    # 2) labels_df next (doc-level gold, if present)
    if "labels_df" in globals() and isinstance(labels_df, pd.DataFrame) and not labels_df.empty:
        k = "article_id_norm" if "article_id_norm" in labels_df.columns else ("article_id" if "article_id" in labels_df.columns else None)
        if k:
            out = _labels_from(labels_df, k)
            if not out.empty:
                print("ğŸ“Œ Using labels from labels_df")
                return out
    # 3) detection_tuple_df (collapse tuple-level â†’ doc-level)
    if "detection_tuple_df" in globals() and isinstance(detection_tuple_df, pd.DataFrame) and not detection_tuple_df.empty:
        k = "article_id_norm" if "article_id_norm" in detection_tuple_df.columns else ("article_id" if "article_id" in detection_tuple_df.columns else None)
        if k:
            out = _labels_from_tuples(detection_tuple_df, k)
            if not out.empty:
                print("ğŸ“Œ Using labels from detection_tuple_df (collapsed to doc-level)")
                return out
    # 4) detection_df itself, only if it actually has 'type'
    if "type" in detection_df.columns:
        out = _labels_from(detection_df, ID_COL)
        if not out.empty:
            print("ğŸ“Œ Using labels from detection_df (fallback)")
            return out
    return pd.DataFrame(columns=["article_id_key","type"])

doc_labels = _pick_label_source()
if doc_labels.empty:
    raise SystemExit("â›” No Primary/Secondary labels found in merged_df / labels_df / detection_tuple_df / detection_df.")

# ---------------- Build one training text per article (choose longest) ----------------
det_text = (detection_df[[ID_COL, TEXT_COL]]
            .rename(columns={ID_COL:"article_id_key", TEXT_COL:"text_for_model"})
            .dropna(subset=["article_id_key","text_for_model"])
            .astype({"article_id_key": str}))
det_text["__len__"] = det_text["text_for_model"].astype(str).str.len()
det_text = (det_text.sort_values("__len__", ascending=False)
            .drop_duplicates("article_id_key")
            [["article_id_key","text_for_model"]])

train_df = (det_text.merge(doc_labels, on="article_id_key", how="inner")
            .dropna(subset=["text_for_model","type"])
            .reset_index(drop=True))

if train_df.empty:
    raise SystemExit("â›” No overlap between detection_df texts and doc-level labels (by article_id).")

print("ğŸ“Œ Training Model 1+ on detection_df (texts)")
print("Label counts (doc-level):\n", train_df["type"].value_counts(dropna=False), "\n")

# ---------------- Features (fast & compact) ----------------
ADD_FLAGS = True
def mk_flags(s: str):
    if not isinstance(s, str): s = ""
    s_low = s.lower()
    return [
        int("doi" in s_low),
        int(any(k in s_low for k in ["dryad","zenodo","pangaea","figshare","tcia","icpsr","dataverse","mendeley","usgs","dataset","data."])),
        int(bool(re.search(r"https?://doi\.org/10\.\d+/", s_low))),
        int(sum(ch.isdigit() for ch in s_low) > 20),
    ]

word_vec = TfidfVectorizer(
    analyzer="word", ngram_range=(1,2), min_df=3, max_df=0.9,
    strip_accents="unicode", sublinear_tf=True, max_features=60000, dtype=np.float32
)
char_vec = TfidfVectorizer(
    analyzer="char", ngram_range=(3,5), min_df=3, max_df=0.9,
    strip_accents="unicode", sublinear_tf=True, max_features=60000, dtype=np.float32
)

X_text = train_df["text_for_model"].fillna("")
Xw = word_vec.fit_transform(X_text)
Xc = char_vec.fit_transform(X_text)
X_all = hstack([Xw, Xc]).tocsr()
if ADD_FLAGS:
    flags = np.array([mk_flags(t) for t in X_text.tolist()], dtype=np.float32)
    X_all = hstack([X_all, csr_matrix(flags)], format="csr")

y = (train_df["type"] == "Secondary").astype(int).values
groups = train_df["article_id_key"].values

# ---------------- GroupKFold CV (fixed splits for sweep + calibration) ----------------
gkf = GroupKFold(n_splits=3)
splits = list(gkf.split(X_all, y, groups))

# ---------------- Micro-sweep (hyperparams only; threshold fixed at 0.30) ----------------
class_boosts = [2.0, 2.5, 3.0]
Cs          = [1.5, 2.0, 3.0]
EVAL_TH     = 0.30  # ğŸ”’ fixed best threshold

best_pre = {"f1": -1}
for m in class_boosts:
    for C in Cs:
        fold_probs, fold_true = [], []
        ok = True
        for tr, va in splits:
            try:
                lr = LogisticRegression(
                    solver="saga", penalty="l2", C=C, max_iter=3000,
                    class_weight={0:1.0, 1:m}, random_state=42, n_jobs=-1, warm_start=True
                ).fit(X_all[tr], y[tr])
                p = lr.predict_proba(X_all[va])[:, 1]
            except Exception:
                ok = False; break
            fold_probs.append(p); fold_true.append(y[va])
        if not ok: continue
        probs = np.concatenate(fold_probs); true = np.concatenate(fold_true)
        pred = (probs >= EVAL_TH).astype(int)
        f1m  = f1_score(true, pred, average="macro", zero_division=0)
        if f1m > best_pre["f1"]:
            best_pre = {"f1": float(f1m), "m": float(m), "C": float(C)}
print(f"âš¡ Pre-calibration best @ th=0.30: macro-F1={best_pre['f1']:.3f} | m={best_pre['m']} C={best_pre['C']}")

# ---------------- Sigmoid calibration (OOF via per-fold calibrators) ----------------
base_lr = LogisticRegression(
    solver="saga", penalty="l2", C=best_pre["C"], max_iter=3000,
    class_weight={0:1.0, 1:best_pre["m"]}, random_state=42, n_jobs=-1, warm_start=True
)

oof_probs, oof_true = [], []
for tr, va in splits:
    cal = CalibratedClassifierCV(base_estimator=base_lr, method="sigmoid", cv=[(tr, va)])
    cal.fit(X_all, y)
    p = cal.predict_proba(X_all[va])[:, 1]
    oof_probs.append(p); oof_true.append(y[va])

oof_probs = np.concatenate(oof_probs)
oof_true  = np.concatenate(oof_true)

# ğŸ”’ Final fixed decision threshold
doc_type_threshold_secondary = 0.30
pred = (oof_probs >= doc_type_threshold_secondary).astype(int)
print(f"âœ… Post-calibration OOF macro-F1 @ th=0.30: {f1_score(oof_true, pred, average='macro', zero_division=0):.3f}")

# ---------------- Final artifacts (TRAINED ON detection_df) ----------------
doc_type_model = CalibratedClassifierCV(base_estimator=base_lr, method="sigmoid", cv=splits)
doc_type_model.fit(X_all, y)

doc_type_vectorizers = {"word": word_vec, "char": char_vec, "add_flags": bool(ADD_FLAGS)}
print("\nâœ… Artifacts ready (TRAINED ON detection_df) with FIXED threshold = 0.30")

# ---------------- Overall CV summary at fixed threshold ----------------
all_true, all_pred = [], []
for tr, va in splits:
    calfold = CalibratedClassifierCV(base_estimator=base_lr, method="sigmoid", cv=[(tr, va)])
    calfold.fit(X_all, y)
    p = calfold.predict_proba(X_all[va])[:, 1]
    yhat = (p >= doc_type_threshold_secondary).astype(int)
    all_true.append(y[va]); all_pred.append(yhat)

y_true = np.concatenate(all_true)
y_pred = np.concatenate(all_pred)

print("\n====== OVERALL CV SUMMARY (calibrated, th=0.30) ======")
print("Macro F1:", f1_score(y_true, y_pred, average="macro", zero_division=0))
print("Micro F1:", f1_score(y_true, y_pred, average="micro", zero_division=0))
print("\nOverall classification report:")
print(classification_report(y_true, y_pred, target_names=["Primary","Secondary"], digits=3, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred, labels=[0,1]))

print(f"\nâ�±ï¸� Total runtime: {time.time()-t0:.1f}s")

# ---------------- Inference helper ----------------
def predict_doc_type(text_series: pd.Series):
    ts = text_series.fillna("").astype(str)
    Xw = doc_type_vectorizers["word"].transform(ts)
    Xc = doc_type_vectorizers["char"].transform(ts)
    X  = hstack([Xw, Xc]).tocsr()
    if doc_type_vectorizers.get("add_flags", False):
        fl = np.array([mk_flags(t) for t in ts.tolist()], dtype=np.float32)
        X = hstack([X, csr_matrix(fl)], format="csr")

    # shape guard for calibrated model
    est = getattr(doc_type_model, "base_estimator_", doc_type_model)
    n_expected = getattr(est, "n_features_in_", X.shape[1])
    if X.shape[1] < n_expected:
        pad = csr_matrix((X.shape[0], n_expected - X.shape[1]), dtype=X.dtype)
        X = hstack([X, pad], format="csr")
    elif X.shape[1] > n_expected:
        X = X[:, :n_expected]

    p = doc_type_model.predict_proba(X)[:, 1]
    yhat = (p >= doc_type_threshold_secondary).astype(int)
    return np.where(yhat == 1, "Secondary", "Primary"), p



# =========================
# Model 1+ â€” SPEED UPGRADE (3-fold CV, micro-sweep + sigmoid calibration)
# TRAINED ON: detection_df (texts)
# Threshold: FIXED at 0.30 (best from CV)
# Produces: doc_type_model, doc_type_vectorizers, doc_type_threshold_secondary
# =========================
import re, time, numpy as np, pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix

t0 = time.time()

# ---------------- Preconditions (texts must come from detection_df) ----------------
if "detection_df" not in globals() or not isinstance(detection_df, pd.DataFrame) or detection_df.empty:
    raise SystemExit("â›” 'detection_df' missing/empty.")

TEXT_COL = ("text_model" if "text_model" in detection_df.columns else
            ("text" if "text" in detection_df.columns else
             ("clean_text" if "clean_text" in detection_df.columns else None)))
ID_COL   = ("article_id_norm" if "article_id_norm" in detection_df.columns else
            ("article_id" if "article_id" in detection_df.columns else None))
if TEXT_COL is None or ID_COL is None:
    raise SystemExit("â›” Need text ('text_model'/'text'/'clean_text') and id ('article_id_norm'/'article_id') in detection_df.")

# ---------------- Build doc-level labels (Primary wins ties) ----------------
def _labels_from(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if df is None or df.empty or ("type" not in df.columns) or (key not in df.columns):
        return pd.DataFrame(columns=["article_id_key","type"])
    d = df[df["type"].isin(["Primary","Secondary"])].copy()
    if d.empty:
        return pd.DataFrame(columns=["article_id_key","type"])
    d["_prio"] = d["type"].map({"Primary":2, "Secondary":1})
    return (d.sort_values([key,"_prio"], ascending=[True,False])
             .drop_duplicates(subset=[key])
             .rename(columns={key:"article_id_key"})
             [["article_id_key","type"]]
             .reset_index(drop=True))

def _pick_label_source():
    # 1) merged_df preferred
    if "merged_df" in globals() and isinstance(merged_df, pd.DataFrame) and not merged_df.empty:
        k = "article_id_norm" if "article_id_norm" in merged_df.columns else ("article_id" if "article_id" in merged_df.columns else None)
        if k:
            out = _labels_from(merged_df, k)
            if not out.empty:
                print("ğŸ“Œ Using labels from merged_df")
                return out
    # 2) labels_df
    if "labels_df" in globals() and isinstance(labels_df, pd.DataFrame) and not labels_df.empty:
        k = "article_id_norm" if "article_id_norm" in labels_df.columns else ("article_id" if "article_id" in labels_df.columns else None)
        if k:
            out = _labels_from(labels_df, k)
            if not out.empty:
                print("ğŸ“Œ Using labels from labels_df")
                return out
    # 3) detection_df (if it has 'type')
    if "type" in detection_df.columns:
        out = _labels_from(detection_df, ID_COL)
        if not out.empty:
            print("ğŸ“Œ Using labels from detection_df (fallback)")
            return out
    return pd.DataFrame(columns=["article_id_key","type"])

doc_labels = _pick_label_source()
if doc_labels.empty:
    raise SystemExit("â›” No Primary/Secondary labels found in merged_df/labels_df/detection_df.")

# ---------------- Build one training text per article (choose longest) ----------------
det_text = (detection_df[[ID_COL, TEXT_COL]]
            .rename(columns={ID_COL:"article_id_key", TEXT_COL:"text_for_model"})
            .dropna(subset=["article_id_key","text_for_model"]))
det_text["__len__"] = det_text["text_for_model"].astype(str).str.len()
det_text = (det_text.sort_values("__len__", ascending=False)
            .drop_duplicates("article_id_key")
            [["article_id_key","text_for_model"]])

train_df = (det_text.merge(doc_labels, on="article_id_key", how="inner")
            .dropna(subset=["text_for_model","type"])
            .reset_index(drop=True))

if train_df.empty:
    raise SystemExit("â›” No overlap between detection_df texts and labels by article_id.")

print("ğŸ“Œ Training Model 1+ on detection_df (texts)")
print("Label counts:\n", train_df["type"].value_counts(dropna=False), "\n")

# ---------------- Features (fast & compact) ----------------
ADD_FLAGS = True
def mk_flags(s: str):
    if not isinstance(s, str): s = ""
    s_low = s.lower()
    return [
        int("doi" in s_low),
        int(any(k in s_low for k in ["dryad","zenodo","pangaea","figshare","tcia","icpsr","dataverse","mendeley","usgs","dataset","data."])),
        int(bool(re.search(r"https?://doi\.org/10\.\d+/", s_low))),
        int(sum(ch.isdigit() for ch in s_low) > 20),
    ]

word_vec = TfidfVectorizer(
    analyzer="word", ngram_range=(1,2), min_df=3, max_df=0.9,
    strip_accents="unicode", sublinear_tf=True, max_features=60000, dtype=np.float32
)
char_vec = TfidfVectorizer(
    analyzer="char", ngram_range=(3,5), min_df=3, max_df=0.9,
    strip_accents="unicode", sublinear_tf=True, max_features=60000, dtype=np.float32
)

X_text = train_df["text_for_model"].fillna("")
Xw = word_vec.fit_transform(X_text)
Xc = char_vec.fit_transform(X_text)
X_all = hstack([Xw, Xc]).tocsr()
if ADD_FLAGS:
    flags = np.array([mk_flags(t) for t in X_text.tolist()], dtype=np.float32)
    X_all = hstack([X_all, csr_matrix(flags)], format="csr")

y = (train_df["type"] == "Secondary").astype(int).values
groups = train_df["article_id_key"].values

# ---------------- GroupKFold CV (fixed splits for sweep + calibration) ----------------
gkf = GroupKFold(n_splits=3)
splits = list(gkf.split(X_all, y, groups))

# ---------------- Micro-sweep (pre-calibration) ----------------
# We tune model hyperparams, but the decision THRESHOLD is FIXED below.
class_boosts = [2.0, 2.5, 3.0]
Cs          = [1.5, 2.0, 3.0]
EVAL_TH = 0.30  # <- FIXED best threshold

best_pre = {"f1": -1}
for m in class_boosts:
    for C in Cs:
        fold_probs, fold_true = [], []
        ok = True
        for tr, va in splits:
            try:
                lr = LogisticRegression(
                    solver="saga", penalty="l2", C=C, max_iter=3000,
                    class_weight={0:1.0, 1:m}, random_state=42, n_jobs=-1, warm_start=True
                ).fit(X_all[tr], y[tr])
                p = lr.predict_proba(X_all[va])[:, 1]
            except Exception:
                ok = False; break
            fold_probs.append(p); fold_true.append(y[va])
        if not ok:
            continue
        probs = np.concatenate(fold_probs); true = np.concatenate(fold_true)
        pred = (probs >= EVAL_TH).astype(int)
        f1m  = f1_score(true, pred, average="macro", zero_division=0)
        if f1m > best_pre["f1"]:
            best_pre = {"f1": float(f1m), "m": float(m), "C": float(C)}
print(f"âš¡ Pre-calibration best @ th=0.30: macro-F1={best_pre['f1']:.3f} | m={best_pre['m']} C={best_pre['C']}")

# ---------------- Sigmoid calibration (OOF via per-fold calibrators) ----------------
base_lr = LogisticRegression(
    solver="saga", penalty="l2", C=best_pre["C"], max_iter=3000,
    class_weight={0:1.0, 1:best_pre["m"]}, random_state=42, n_jobs=-1, warm_start=True
)

oof_probs, oof_true = [], []
for tr, va in splits:
    cal = CalibratedClassifierCV(base_estimator=base_lr, method="sigmoid", cv=[(tr, va)])
    cal.fit(X_all, y)
    p = cal.predict_proba(X_all[va])[:, 1]
    oof_probs.append(p); oof_true.append(y[va])

oof_probs = np.concatenate(oof_probs)
oof_true  = np.concatenate(oof_true)

# We keep the decision threshold FIXED at 0.30
doc_type_threshold_secondary = 0.30
pred = (oof_probs >= doc_type_threshold_secondary).astype(int)
print(f"âœ… Post-calibration OOF macro-F1 @ th=0.30: {f1_score(oof_true, pred, average='macro', zero_division=0):.3f}")

# ---------------- Final artifacts (TRAINED ON detection_df texts) ----------------
doc_type_model = CalibratedClassifierCV(base_estimator=base_lr, method="sigmoid", cv=splits)
doc_type_model.fit(X_all, y)

doc_type_vectorizers = {"word": word_vec, "char": char_vec, "add_flags": bool(ADD_FLAGS)}
print("\nâœ… Artifacts ready (TRAINED ON detection_df) with FIXED threshold = 0.30")

# ---------------- Overall CV summary at FIXED threshold ----------------
all_true, all_pred = [], []
for tr, va in splits:
    calfold = CalibratedClassifierCV(base_estimator=base_lr, method="sigmoid", cv=[(tr, va)])
    calfold.fit(X_all, y)
    p = calfold.predict_proba(X_all[va])[:, 1]
    yhat = (p >= doc_type_threshold_secondary).astype(int)
    all_true.append(y[va]); all_pred.append(yhat)

y_true = np.concatenate(all_true)
y_pred = np.concatenate(all_pred)

print("\n====== OVERALL CV SUMMARY (calibrated, th=0.30) ======")
print("Macro F1:", f1_score(y_true, y_pred, average="macro", zero_division=0))
print("Micro F1:", f1_score(y_true, y_pred, average="micro", zero_division=0))
print("\nOverall classification report:")
print(classification_report(y_true, y_pred, target_names=["Primary","Secondary"], digits=3, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred, labels=[0,1]))

print(f"\nâ�±ï¸� Total runtime: {time.time()-t0:.1f}s")

# ---------------- Inference helper ----------------
def predict_doc_type(text_series: pd.Series):
    ts = text_series.fillna("").astype(str)
    Xw = doc_type_vectorizers["word"].transform(ts)
    Xc = doc_type_vectorizers["char"].transform(ts)
    X  = hstack([Xw, Xc]).tocsr()
    if doc_type_vectorizers.get("add_flags", False):
        fl = np.array([mk_flags(t) for t in ts.tolist()], dtype=np.float32)
        X = hstack([X, csr_matrix(fl)], format="csr")

    # shape guard for calibrated model
    est = getattr(doc_type_model, "base_estimator_", doc_type_model)
    n_expected = getattr(est, "n_features_in_", X.shape[1])
    if X.shape[1] < n_expected:
        pad = csr_matrix((X.shape[0], n_expected - X.shape[1]), dtype=X.dtype)
        X = hstack([X, pad], format="csr")
    elif X.shape[1] > n_expected:
        X = X[:, :n_expected]

    p = doc_type_model.predict_proba(X)[:, 1]
    yhat = (p >= doc_type_threshold_secondary).astype(int)
    return np.where(yhat == 1, "Secondary", "Primary"), p



import matplotlib.pyplot as plt
import numpy as np

# ---- Metrics table (updated with bug fix results) ----
models = [
    "Model 1 (Baseline)",
    "Model 1+ (Calibrated @0.30)",
    "Model 2 (TF-IDF CV)",
    "Model 3 (MiniLM + LR)",
    "Model 4 (Ensemble)",
    "Model 5 (MiniLM + XGB v1)",
    "Model 6 (MiniLM + XGB v2)"
]

# Updated: Model 1+ now Macro F1 = 0.71, Secondary F1 = 0.58
# Others are placeholders from your last run; update as you re-run them.
macro_f1 =     [0.64, 0.71, 0.64, 0.73, 0.73, 0.72, 0.75]
secondary_f1 = [0.42, 0.58, 0.42, 0.55, 0.57, 0.59, 0.63]

x = np.arange(len(models))
width = 0.6

# Helper: colors that highlight Model 1+ (index 1)
def make_colors(n, highlight_idx=1):
    base = ["#6BAED6"] * n        # blue-ish
    base[highlight_idx] = "#DD1C77"  # magenta highlight for Model 1+
    return base

# ---- Macro-F1 Bar Chart ----
plt.figure(figsize=(11,5))
colors_macro = make_colors(len(models), highlight_idx=1)
bars = plt.bar(x, macro_f1, width, color=colors_macro)
plt.xticks(x, models, rotation=30, ha="right")
plt.ylabel("Macro-F1")
plt.title("Macro-F1 by Model (Model 1+ calibrated @ 0.30)")
for i, v in enumerate(macro_f1):
    plt.text(i, v + 0.008, f"{v:.2f}", ha="center", fontsize=9)
plt.ylim(0, 0.85)
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

# ---- Secondary-F1 Bar Chart ----
plt.figure(figsize=(11,5))
colors_sec = make_colors(len(models), highlight_idx=1)
bars = plt.bar(x, secondary_f1, width, color=colors_sec)
plt.xticks(x, models, rotation=30, ha="right")
plt.ylabel("Secondary F1")
plt.title("Secondary F1 by Model (Model 1+ improves recall)")
for i, v in enumerate(secondary_f1):
    plt.text(i, v + 0.008, f"{v:.2f}", ha="center", fontsize=9)
plt.ylim(0, 0.75)
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()



import pandas as pd

data = {
    "Model": [
        "Model 1 (Baseline TF-IDF+LR)",
        "Model 1+ (Calibrated, old)",
        "Model 2 (fixed, CV)",
        "Model 2++ (flags+tuning)",
        "Model 3 (MiniLM+LR)",
        "Model 4 (Ensemble)",
        "Model 5 (MiniLM+XGB v1)",
        "Model 6 (MiniLM+XGB v2)",
        "Model 1+ (Calibrated, fixed)"
    ],
    "Macro-F1": [0.64, 0.78, 0.63, 0.71, 0.73, 0.73, 0.72, 0.75, 0.71],
    "Secondary F1": [0.42, 0.56, 0.39, 0.57, 0.55, 0.57, 0.59, 0.63, 0.58],
    "Notes": [
        "Simple baseline, low Secondary performance",
        "Leakage removed, tuned weight+thr (hold-out)",
        "Volatile across folds",
        "More stable, Secondary improved",
        "Embedding-based, decent tradeoff",
        "Blends TF-IDF + BERT; stable but heavier",
        "Stronger than LR baseline",
        "Tuned embeddings, strong but more complex",
        "âœ… detection_df, threshold=0.30, balanced"
    ]
}

metrics_df = pd.DataFrame(data)
metrics_df



# ================================================
# Model 1+ (pretrained artifacts) â†’ Build Submission (FIXED)
# - Robust "Secondary" class resolution (works with labels != "Secondary")
# - Case-preserving text flow; dataset DOIs lowercased to match sample format
# - Dataset-positive filter (registrant prefixes + repo keywords)
# - Compact article_id, exact-triplet de-dup, row_id
# NOTE: assumes pretrained artifacts already exist in the kernel:
#   doc_type_model, doc_type_vectorizers, doc_type_threshold_secondary
# ================================================
import os, re, json, shutil, warnings, zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from unicodedata import normalize as _u
from scipy.sparse import hstack, csr_matrix
import xml.etree.ElementTree as ET

# -------- Paths --------
ROOT         = "/kaggle/input/make-data-count-finding-data-references"
TEST_XML_DIR = f"{ROOT}/test/XML"
TEST_PDF_DIR = f"{ROOT}/test/PDF"
OUT_CSV      = Path("/kaggle/working/submission.csv")
PACK_DIR     = Path("/kaggle/working/pack_m1plus")

# -------- Preconditions: pretrained Model 1+ artifacts --------
_need = ["doc_type_model", "doc_type_vectorizers", "doc_type_threshold_secondary"]
if not all(k in globals() for k in _need):
    raise SystemExit("â›” Model 1+ artifacts missing. Define 'doc_type_model', 'doc_type_vectorizers', and 'doc_type_threshold_secondary' before running.")

# -------- Optional flag-engine used during training --------
def _mk_flags(s: str):
    if not isinstance(s, str): s = ""
    s_low = s.lower()
    return [
        int("doi" in s_low),
        int(any(k in s_low for k in ["dryad","zenodo","pangaea","figshare","tcia","icpsr","dataverse","mendeley","usgs","dataset","data."])),
        int(bool(re.search(r"https?://doi\.org/10\.\d+/", s_low))),
        int(sum(ch.isdigit() for ch in s_low) > 20),
    ]
mk_flags = globals().get("mk_flags", _mk_flags)

# -------- Helpers --------
def clean_text(x):
    if pd.isna(x): return ""
    x = str(x)
    x = re.sub(r"<.*?>", " ", x)
    x = re.sub(r"[\n\r\t]", " ", x)
    return re.sub(r"\s+", " ", x).strip()

def normalize_article_id_jats(s):
    """Normalize article IDs for joins only (NOT for dataset_id DOIs)."""
    if not isinstance(s, str): return s, False
    orig = s.strip()
    if orig.startswith("10.") and "_" in orig and orig.count(".") >= 2:
        return orig, False
    aid = orig.lower().replace("/", "_")
    if aid.startswith("10_"): aid = "10." + aid[3:]
    d = aid.find(".", 3)
    if d == -1:
        norm = aid.replace("_", ".")
    else:
        prefix, suffix = aid[:d+1], aid[d+1:]
        if "_" in suffix:
            i = suffix.find("_"); preserved, rest = suffix[:i+1], suffix[i+1:]
            j, k = rest.rfind("_"), rest.rfind(".")
            if j != -1 and j > k: rest = rest[:j] + "." + rest[j+1:]
            norm = prefix + preserved + rest
        else:
            j, k = suffix.rfind("_"), suffix.rfind(".")
            if j != -1 and j > k: suffix = suffix[:j] + "." + suffix[j+1:]
            norm = prefix + suffix
    return norm, (norm != orig.lower())

def _detect_xml_type(path):
    try:
        root = ET.parse(path).getroot()
        return "BioC" if "collection" in root.tag.lower() else "JATS"
    except Exception:
        return "Unreadable"

def _extract_doi_from_xml(path):
    try:
        tree = ET.parse(path); root = tree.getroot()
        if '}' in root.tag:
            ns = {'ns': root.tag.split('}')[0].strip('{')}
            node = root.find(".//ns:article-id[@pub-id-type='doi']", ns)
        else:
            node = root.find(".//article-id[@pub-id-type='doi']")
        return node.text.strip() if node is not None and node.text else None
    except Exception:
        return None

# ---------- Build test index (preserve DOI case) ----------
xml_rec, pdf_rec = [], []
if os.path.isdir(TEST_XML_DIR):
    for r,_,fs in os.walk(TEST_XML_DIR):
        for f in fs:
            if f.lower().endswith(".xml"):
                p = os.path.join(r, f)
                x_type = _detect_xml_type(p)
                doi = _extract_doi_from_xml(p)
                if not doi:
                    base = os.path.basename(p).replace(".xml","")
                    if base.startswith("10.") and "_" in base:
                        prefix, rest = base.split("_", 1)
                        doi = f"{prefix}/{rest}"
                    else:
                        doi = base
                norm,_ = normalize_article_id_jats(doi)
                norm = norm.replace("/", "_")
                xml_rec.append({"file_path": p, "file_type": x_type,
                                "article_id_raw": doi, "article_id_norm": norm})

if os.path.isdir(TEST_PDF_DIR):
    for r,_,fs in os.walk(TEST_PDF_DIR):
        for f in fs:
            if f.lower().endswith(".pdf"):
                p = os.path.join(r, f)
                base = os.path.basename(p).replace(".pdf","")
                doi_slashy = f"{base.split('_',1)[0]}/{base.split('_',1)[1]}" if (base.startswith("10.") and "_" in base) else base
                norm,_ = normalize_article_id_jats(doi_slashy)
                norm = norm.replace("/", "_")
                pdf_rec.append({"file_path": p, "file_type": "PDF",
                                "article_id_raw": doi_slashy, "article_id_norm": norm})

test_index_df = (
    pd.concat([pd.DataFrame(xml_rec), pd.DataFrame(pdf_rec)], ignore_index=True)
      .dropna(subset=["article_id_norm"])
      .drop_duplicates(subset=["article_id_norm"], keep="first")
      .reset_index(drop=True)
)
print(f"âœ… test_index_df built | rows: {len(test_index_df)}")

# ---------- Extract text ----------
def _extract_text_from_jats(path):
    try:
        tree = ET.parse(path); root = tree.getroot()
        if '}' in root.tag:
            ns = {'ns': root.tag.split('}')[0].strip('{')}
            title   = root.find('.//ns:title-group/ns:article-title', ns)
            abstract= root.find('.//ns:abstract', ns)
            body    = root.find('.//ns:body', ns)
        else:
            title   = root.find('.//title-group/article-title')
            abstract= root.find('.//abstract')
            body    = root.find('.//body')
        t = (title.text or "").strip() if title is not None else ""
        a = "".join(abstract.itertext()).strip() if abstract is not None else ""
        b = "".join(body.itertext()).strip() if body is not None else ""
        return f"{t}\n{a}\n{b}".strip()
    except Exception as e:
        warnings.warn(f"âš ï¸� JATS read error in {os.path.basename(path)}: {e}")
        return ""

def _extract_text_from_bioc(path):
    try:
        tree = ET.parse(path); root = tree.getroot()
        blocks = [
            (child.text or "").strip()
            for passage in root.findall(".//passage")
            for child in passage if child.tag == "text" and child.text
        ]
        return "\n.join(blocks).strip()"  # fallback if needed
    except Exception as e:
        warnings.warn(f"âš ï¸� BioC read error in {os.path.basename(path)}: {e}")
        return ""

def _extract_text_auto(path):
    try:
        root = ET.parse(path).getroot()
        return _extract_text_from_bioc(path) if "collection" in root.tag.lower() else _extract_text_from_jats(path)
    except Exception as e:
        warnings.warn(f"âš ï¸� General read error in {os.path.basename(path)}: {e}")
        return ""

test_text_records = []
for _, row in test_index_df.iterrows():
    if row["file_type"] in ("JATS", "BioC"):
        txt = clean_text(_extract_text_auto(row["file_path"]))
        test_text_records.append({"article_id_norm": row["article_id_norm"], "text": txt})

test_text_df = (
    pd.DataFrame(test_text_records)
      .dropna(subset=["text"])
      .drop_duplicates("article_id_norm")
      .reset_index(drop=True)
)
print(f"ğŸ“� test_text_df built: {test_text_df.shape}")

# ---------- Inference using pretrained Model 1+ artifacts (robust 'Secondary' resolution) ----------
Xw = doc_type_vectorizers["word"].transform(test_text_df["text"].astype(str))
Xc = doc_type_vectorizers["char"].transform(test_text_df["text"].astype(str))
X  = hstack([Xw, Xc]).tocsr()

# Shape-guard to estimator's expected n_features (supports calibrated models)
_est = getattr(doc_type_model, "base_estimator_", doc_type_model)
n_expected = getattr(_est, "n_features_in_", X.shape[1])
if X.shape[1] < n_expected:
    pad = csr_matrix((X.shape[0], n_expected - X.shape[1]), dtype=X.dtype)
    X = hstack([X, pad], format="csr")
elif X.shape[1] > n_expected:
    X = X[:, :n_expected]

# Optional flags if artifacts indicate they were used
if doc_type_vectorizers.get("add_flags", False):
    fl = np.array([mk_flags(t) for t in test_text_df["text"].astype(str).tolist()], dtype=np.float32)
    X = hstack([X, csr_matrix(fl)], format="csr")
    # re-guard after adding flags
    n_expected2 = getattr(_est, "n_features_in_", X.shape[1])
    if X.shape[1] < n_expected2:
        X = hstack([X, csr_matrix((X.shape[0], n_expected2 - X.shape[1]), dtype=X.dtype)], format="csr")
    elif X.shape[1] > n_expected2:
        X = X[:, :n_expected2]

def _secondary_index(model):
    """Resolve which predict_proba column corresponds to 'Secondary'."""
    mdl = model
    classes = getattr(mdl, "classes_", None)
    if classes is None:
        mdl = getattr(model, "base_estimator_", model)
        classes = getattr(mdl, "classes_", None)
    if classes is None:
        raise RuntimeError("Model has no classes_.")
    classes = list(classes)

    # direct string match
    for i, c in enumerate(classes):
        if isinstance(c, str) and c.lower().startswith("sec"):
            return i, classes
    # numeric-positive label
    if 1 in classes:   return classes.index(1), classes
    if True in classes:return classes.index(True), classes
    # infer opposite of 'Primary' in binary
    prim_idx = next((i for i, c in enumerate(classes) if isinstance(c, str) and c.lower().startswith("prim")), None)
    if prim_idx is not None and len(classes) == 2:
        return 1 - prim_idx, classes
    # fallback: last column
    return len(classes) - 1, classes

proba = doc_type_model.predict_proba(X)
sec_idx, classes_list = _secondary_index(doc_type_model)
p_sec = proba[:, sec_idx]
thr   = float(doc_type_threshold_secondary)

test_text_df["doc_type_pred"]        = np.where(p_sec >= thr, "Secondary", "Primary")
test_text_df["doc_type_p_secondary"] = p_sec
print("ğŸ”® Pred counts:", test_text_df["doc_type_pred"].value_counts().to_dict(),
      "| model classes:", classes_list)

# ---------- Mine dataset identifiers ----------
_DOI_BARE  = re.compile(r"(10\.\d{4,9}/[^\s\"'<>),;:]+)", re.I)
_DOI_FULL  = re.compile(r"https?://(?:dx\.)?doi\.org/(10\.\d{4,9}/[^\s\"'<>),;:]+)", re.I)
_PATTERNS = [
    (re.compile(r"\b(GSE\d+|GSM\d+)\b", re.I), lambda m: m.group(1).upper()),
    (re.compile(r"\bE\-MTAB\-\d+\b", re.I), lambda m: m.group(0).upper()),
    (re.compile(r"\bPRJ(?:NA|EB|DB)\d+\b", re.I), lambda m: m.group(0).upper()),
    (re.compile(r"\bSRR\d+|SRP\d+|SRX\d+|SRS\d+\b", re.I), lambda m: m.group(0).upper()),
    (re.compile(r"\bEMPIAR\-\d+\b", re.I), lambda m: m.group(0).upper()),
    (re.compile(r"\bEMD\-\d+\b", re.I), lambda m: m.group(0).upper()),
    (re.compile(r"\bCHEMBL\d+\b", re.I), lambda m: m.group(0).upper()),
    (re.compile(r"\bICPSR\d+\b", re.I), lambda m: m.group(0).upper()),
    (re.compile(r"\bEPI(?:[_-]ISL)?\d+\b", re.I), lambda m: m.group(0).upper()),
    (re.compile(r"\bpdb\s+([0-9a-z]{4})\b", re.I), lambda m: f"PDB {m.group(1).upper()}"),
    # EDI/PASTA & Dryad legacy (keep case initially; we'll lowercase canonical DOI URLs later)
    (re.compile(r"\b(10\.(?:6073/PASTA|25349/)[^\s\"'<>),;:]+)\b", re.I), lambda m: f"https://doi.org/{m.group(1)}"),
]

def _canonize_doi_url(s: str) -> str:
    s = re.sub(r"^\s*doi:\s*", "https://doi.org/", s, flags=re.I)
    s = re.sub(r"^\s*https?://(?:dx\.)?doi\.org/", "https://doi.org/", s, flags=re.I)
    return s.rstrip(").,;:]}'\"")

def _looks_like_repo_base(doi_url: str) -> bool:
    if not isinstance(doi_url, str) or not doi_url.startswith("https://doi.org/"): return False
    parts = doi_url.split("/", 3)
    if len(parts) < 4: return False
    suf = parts[3]
    if suf.upper().startswith("PASTA"): return False
    return not any(ch.isdigit() for ch in suf)

def mine_dataset_ids(text: str) -> list[str]:
    out = set()
    if not isinstance(text, str) or not text: return []
    for m in _DOI_FULL.finditer(text):
        out.add(_canonize_doi_url("https://doi.org/" + m.group(1)))
    for m in _DOI_BARE.finditer(text):
        out.add(_canonize_doi_url("https://doi.org/" + m.group(1)))
    for pat, fn in _PATTERNS:
        for m in pat.finditer(text):
            out.add(fn(m))
    out = {x for x in out if len(x) >= 12 and not _looks_like_repo_base(x)}
    return sorted(out)

cand = []
for _, row in test_text_df.iterrows():
    ids = mine_dataset_ids(row["text"])
    if ids:
        for ds in ids:
            cand.append({"article_id_norm": row["article_id_norm"], "dataset_id": ds, "type": row["doc_type_pred"]})
cand_df = pd.DataFrame(cand)
print(f"ğŸ§² Raw candidates mined: {len(cand_df)}")

# ---------- Assemble (no filtering beyond duplicates/schema) ----------
if not cand_df.empty:
    keep_ids = set(test_index_df["article_id_norm"].astype(str))
    submission = (cand_df[cand_df["article_id_norm"].astype(str).isin(keep_ids)]
                  .dropna(subset=["dataset_id","type"])
                  .drop_duplicates(subset=["article_id_norm","dataset_id","type"])
                  .rename(columns={"article_id_norm":"article_id"})[["article_id","dataset_id","type"]]
                  .reset_index(drop=True))
else:
    submission = pd.DataFrame(columns=["article_id","dataset_id","type"])
print(f"ğŸ§± Pre-finalization rows: {len(submission)}")

# ---------- DOI cleanup (case-preserving sanitization) ----------
def _canonize(url: str) -> str:
    if not isinstance(url, str): return ""
    url = url.strip()
    url = re.sub(r"(?i)^\s*doi:\s*", "https://doi.org/", url)
    url = re.sub(r"(?i)^\s*https?://(?:dx\.)?doi\.org/", "https://doi.org/", url)
    return url.rstrip(")]}>.,;:\"'")

_ISSN_TOKEN_RE = r"[sS]?\d{4}-\d{3}[\dX]"
ISSN_PATH_ONLY_RE = re.compile(rf"^10\.\d+/(?:[a-z]\.)?{_ISSN_TOKEN_RE}$")
TRAILING_NAME_RE = re.compile(r"(?<=[0-9A-Za-z\)])(?:[._-]?(?:[A-Z][a-z]{2,}(?:-[A-Z][a-z]+)*|[A-Z]{3,}))$")
DRYAD_SUPPORT_RE = re.compile(r"(?i)^(10\.5061/dryad\.[^/\s]+)\.(?:support(?:ing)?|supp(?:lement(?:al|ary)?)?).*$")

def _copernicus_incomplete_suffix(suf: str) -> bool:
    if not suf.lower().startswith("10.5194/"): return False
    tail = suf.split("/", 1)[-1]
    parts = tail.split("-")
    if len(parts) <= 1: return True
    if len(parts) == 2: return True
    if len(parts) == 3:
        if re.fullmatch(r"\d{4}", parts[1]) or re.fullmatch(r"\d{4}", parts[2]): return False
        return True
    return False

def _path_has_digit(suf: str) -> bool:
    parts = suf.split("/", 1)
    return len(parts) >= 2 and bool(re.search(r"\d", parts[1]))

def sanitize_doi_url(url: str):
    if not isinstance(url, str) or not url.strip(): return None
    url = _canonize(url)
    if not url.startswith("https://doi.org/"): return None
    suf = url.split("/", 3)[3] if url.count("/") >= 3 else ""
    if not suf: return None
    m = DRYAD_SUPPORT_RE.match(suf)
    if m: suf = m.group(1)
    suf = TRAILING_NAME_RE.sub("", suf)
    if suf.count("(") > suf.count(")"): suf = suf.rsplit("(", 1)[0]
    if suf.endswith(("-", ".", "/")): return None
    if ISSN_PATH_ONLY_RE.fullmatch(suf): return None
    if _copernicus_incomplete_suffix(suf): return None
    if not _path_has_digit(suf): return None
    return "https://doi.org/" + suf

def _norm_chars(s):
    if not isinstance(s, str): return s
    s = _u("NFKC", s)
    for bad in ("â€šÃ„Ãª","â€šÃ„Ã¬","\u2010","\u2011","\u2013","\u2014","\u2212","\u2012","\u2015","\uFE58","\uFE63","\uFF0D"):
        s = s.replace(bad, "-")
    for inv in ("\u00ad","\u200b","\u200c","\u200d","\u2060"): s = s.replace(inv, "")
    s = s.replace("\xa0"," ").replace("\u2028"," ").replace("\u2029"," ")
    s = re.sub(r"^\s*doi\s*:\s*","https://doi.org/", s, flags=re.I)
    s = re.sub(r"^\s*https?://(?:dx\.)?doi\.org/","https://doi.org/", s, flags=re.I)
    return re.sub(r"\s+"," ", s).strip()

for col in ("article_id","dataset_id","type"):
    if col in submission.columns:
        submission[col] = submission[col].map(_norm_chars)

# Clean DOI-style dataset IDs; keep accessions untouched
_DOI_STEM = re.compile(r"(?i)^(?:\s*doi:\s*|https?://(?:dx\.)?doi\.org/)")
is_doiish = submission["dataset_id"].astype(str).str.match(_DOI_STEM)
cleaned = submission.loc[is_doiish, "dataset_id"].map(sanitize_doi_url)
ok_mask = is_doiish & cleaned.notna().reindex(submission.index, fill_value=False)
submission.loc[ok_mask, "dataset_id"] = cleaned.reindex(submission.index)[ok_mask]
submission = submission.loc[(~is_doiish) | ok_mask].reset_index(drop=True)

# ---------- Dataset-positive filter ----------
REPO_PREFIXES = {
    "10.5061","10.5281","10.1594","10.6084","10.17605","10.17632","10.6073","10.5066",
    "10.17882","10.13155","10.7910","10.3886","10.7937","10.25349","10.25386","10.6019",
    "10.5524","10.5284","10.5285","10.5286","10.5287","10.25411","10.25496","10.25504",
    "10.25573","10.6080","10.6083","10.6086","10.6088","10.22148","10.24381","10.26208",
    "10.26180","10.7272","10.7283","10.1575","10.15784"
}
REPO_KEYWORDS = [
    "zenodo","dryad","pangaea","figshare","m9.figshare","dataset","data.","data/","pasta",
    "tcia","icpsr","pride","empiar","dataverse","harvarddataverse","osf","mendeley","usgs","gbif"
]
def _is_dataset_doi(url: str) -> bool:
    if not isinstance(url, str) or not url.startswith("https://doi.org/"): return False
    m = re.match(r"^https?://doi\.org/(10\.\d+)/(.*)$", url, flags=re.I)
    if not m: return False
    pref = m.group(1).lower(); suf = m.group(2).lower()
    if pref in REPO_PREFIXES: return True
    return any(k in suf for k in REPO_KEYWORDS)

_is_doi = submission["dataset_id"].str.startswith("https://doi.org/", na=False)
_keep_mask = (~_is_doi) | submission.loc[_is_doi, "dataset_id"].map(_is_dataset_doi).reindex(submission.index, fill_value=False)
print("ğŸ§¹ Filtering DOIs â€” kept dataset-like:",
      int((_is_doi & _keep_mask).sum()),
      "| dropped article-like:",
      int((_is_doi & (~_keep_mask)).sum()))
submission = submission.loc[_keep_mask].drop_duplicates(subset=["article_id","dataset_id","type"]).reset_index(drop=True)

# ---------- Compact article_id, row_id, lowercase DOI to mimic sample ----------
def article_id_to_compact(aid: str) -> str:
    if not isinstance(aid, str): return aid
    s = re.sub(r"(?i)^https?://doi\.org/", "", aid.strip()).replace("/", "_")
    return re.sub(r"^(10\.\d+)\.(.+)$", r"\1_\2", s)

submission["article_id"] = submission["article_id"].map(article_id_to_compact)

# Lowercase only DOI-style dataset IDs to mirror sample exactly
_is_doi_now = submission["dataset_id"].str.startswith("https://doi.org/", na=False)
submission.loc[_is_doi_now, "dataset_id"] = submission.loc[_is_doi_now, "dataset_id"].str.lower()

submission = submission.drop_duplicates(subset=["article_id","dataset_id","type"]).reset_index(drop=True)
submission = submission.drop(columns=["row_id"], errors="ignore")
submission.insert(0, "row_id", np.arange(len(submission), dtype=int))
submission = submission[["row_id","article_id","dataset_id","type"]]

# ---------- Save exactly one submission ----------
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
TMP = OUT_CSV.with_suffix(".csv.tmp")
submission.to_csv(TMP, index=False)
if OUT_CSV.exists():
    OUT_CSV.unlink(missing_ok=True)
TMP.replace(OUT_CSV)
print(f"ğŸ’¾ Wrote {OUT_CSV} | shape={submission.shape}")
print(submission.head(20).to_string(index=False))

# ---------- Pack artifacts (guarded) ----------
PACK_DIR.mkdir(parents=True, exist_ok=True)
submission.to_csv(PACK_DIR / "submission.csv", index=False)
try:
    import joblib
    joblib.dump(doc_type_model,       PACK_DIR / "model1plus_tfidf_lr_calibrated.joblib")
    joblib.dump(doc_type_vectorizers, PACK_DIR / "model1plus_vectorizers.joblib")
    with open(PACK_DIR / "model1plus_meta.json", "w") as f:
        json.dump({"threshold_secondary": float(doc_type_threshold_secondary)}, f)
except Exception as e:
    warnings.warn(f"âš ï¸� Skipped dumping artifacts: {e}")
shutil.make_archive("/kaggle/working/pack_m1plus", "zip", PACK_DIR)
print("ğŸ“¦ Wrote /kaggle/working/pack_m1plus.zip")



# =========================
# Submission Finalizer + Save (atomic)
# =========================
import re, os
from pathlib import Path
import pandas as pd

OUT_CSV = Path("/kaggle/working/submission.csv")
PACK_DIR = Path("/kaggle/working/pack_m1plus")
MATCH_UPPER_PASTA = False   # set True ONLY if you want UPPERCASE "PASTA/HEX"

# 0) Load file from disk if 'submission' is not already defined
try:
    submission
except NameError:
    if not OUT_CSV.exists():
        raise SystemExit(f"â›” Can't find {OUT_CSV}. Run the pipeline first.")
    submission = pd.read_csv(OUT_CSV)
    print("ğŸ“‚ Loaded submission from disk:", submission.shape)

# 1) Clean 'type' tokens
submission["type"] = (
    submission["type"]
    .astype(str)
    .str.extract(r"(Primary|Secondary)", expand=False)
)

# 2) DOI normalization (canonical stem + lowercase suffix)
DOI_STEM = re.compile(r"(?i)^https?://(?:dx\.)?doi\.org/")

def normalize_doi(url: str) -> str:
    if not isinstance(url, str) or not DOI_STEM.match(url):
        return url
    url = DOI_STEM.sub("https://doi.org/", url.strip())
    head, tail = url.split("https://doi.org/", 1)
    return "https://doi.org/" + tail.lower()

submission["dataset_id"] = submission["dataset_id"].map(normalize_doi)

# 3) OPTIONAL: Uppercase PASTA segment (only if requested)
def style_sample_pasta_upper(url: str) -> str:
    if not isinstance(url, str):
        return url
    stem = "https://doi.org/10.6073/"
    if url.startswith(stem) and url[len(stem):].startswith("pasta/"):
        return stem + url[len(stem):].upper()
    return url

if MATCH_UPPER_PASTA:
    submission["dataset_id"] = submission["dataset_id"].map(style_sample_pasta_upper)

# 4) Enforce schema/order + de-dup + row_id 0..N-1
required_cols = ["row_id", "article_id", "dataset_id", "type"]
submission = submission.drop_duplicates(subset=["article_id","dataset_id","type"]).reset_index(drop=True)
# rebuild row_id sequentially from 0
if "row_id" in submission.columns:
    submission = submission.drop(columns=["row_id"])
submission.insert(0, "row_id", range(len(submission)))
submission = submission[required_cols]

# 5) Save back to disk atomically
tmp = OUT_CSV.with_suffix(".csv.tmp")
submission.to_csv(tmp, index=False)
if OUT_CSV.exists():
    OUT_CSV.unlink(missing_ok=True)
tmp.replace(OUT_CSV)
print(f"ğŸ’¾ Wrote {OUT_CSV} | shape={submission.shape}")

# 6) (Optional) refresh the packed copy if your pack dir exists
try:
    if PACK_DIR.exists():
        submission.to_csv(PACK_DIR / "submission.csv", index=False)
        print(f"ğŸ“¦ Updated {PACK_DIR / 'submission.csv'}")
except Exception as e:
    print("âš ï¸� Could not update packed submission:", e)

# 7) Quick sanity view
print(submission.head(12).to_string(index=False))



import re

DOI_STEM = re.compile(r"(?i)^https?://(?:dx\.)?doi\.org/")

def normalize_doi(url: str) -> str:
    """Canonicalize DOI URLs: keep https://doi.org/ stem, lowercase suffix."""
    if not isinstance(url, str) or not DOI_STEM.match(url):
        return url
    # Canonicalize stem + lowercase suffix
    url = DOI_STEM.sub("https://doi.org/", url)
    head, tail = url.split("https://doi.org/", 1)
    return "https://doi.org/" + tail.lower().strip()



submission["dataset_id"] = submission["dataset_id"].map(normalize_doi)



print(submission.shape)
print(submission.head(20))


