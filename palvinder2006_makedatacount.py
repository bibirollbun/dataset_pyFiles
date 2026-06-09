import os, re, sys, gc, math, json
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
import numpy as np


KAGGLE_INPUT = Path("/kaggle/input")
cand_roots = []
if KAGGLE_INPUT.exists():
    for p in KAGGLE_INPUT.iterdir():
        name = p.name.lower()
        if p.is_dir() and ("make" in name and "data" in name and "count" in name):
            cand_roots.append(p)

INPUT_ROOT = cand_roots[0] if cand_roots else (KAGGLE_INPUT if KAGGLE_INPUT.exists() else Path.cwd())

TRAIN_DIR = INPUT_ROOT / "train"
TEST_DIR  = INPUT_ROOT / "test"
TRAIN_XML = TRAIN_DIR / "XML"
TRAIN_PDF = TRAIN_DIR / "PDF"
TEST_XML  = TEST_DIR  / "XML"
TEST_PDF  = TEST_DIR  / "PDF"
TRAIN_LABELS = INPUT_ROOT / "train_labels.csv"
SAMPLE_SUB   = INPUT_ROOT / "sample_submission.csv"

OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
SUB_CSV = OUT_DIR / "submission.csv"

print("INPUT_ROOT:", INPUT_ROOT)
print("Exists:", {
    "train_xml": TRAIN_XML.exists(),
    "train_pdf": TRAIN_PDF.exists(),
    "test_xml":  TEST_XML.exists(),
    "test_pdf":  TEST_PDF.exists(),
    "labels":    TRAIN_LABELS.exists(),
    "sample":    SAMPLE_SUB.exists(),
})
print()


DOI_CORE_RE = re.compile(r'\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b', re.I)

ACC_PATTERNS = {
    "GEO": re.compile(r'\bGSE\d+\b', re.I),
    "ArrayExpress": re.compile(r'\bE\-(?:MTAB|MEXP)-\d+\b', re.I),
    "ENA/EBI": re.compile(r'\bPRJ[EDNA]\d+\b', re.I),   # PRJEB, PRJNA, etc.
    "SRA_RUN": re.compile(r'\b(?:SRR|ERR|DRR)\d+\b', re.I),
    "SRA_PROJ": re.compile(r'\b(?:SRP|ERP|DRP)\d+\b', re.I),
    "EGA": re.compile(r'\bEGA[DS]\w*\d+\b', re.I),      # EGAS, EGAD, EGAF...
    "PDB": re.compile(r'\bpdb\s+[0-9A-Za-z]{4}\b', re.I),  # keep "pdb 5yfp" style
    "ChEMBL": re.compile(r'\bCHEMBL\d+\b', re.I),
    "UniProt": re.compile(r'\b[A-NR-Z][0-9][A-Z0-9]{3}[0-9]\b', re.I),  # rough
}

PRIMARY_HINTS = [
    "generated in this study", "we generated", "we collected", "our data",
    "in this study we", "deposited to", "submitted to", "made available at",
    "data available at", "supplemental material available at", "uploaded to"
]
SECONDARY_HINTS = [
    "obtained from", "downloaded from", "sourced from", "previously published",
    "retrieved from", "reused", "derived from existing", "publicly available",
    "from the repository", "available from"
]

def normalize_doi_to_https(text: str) -> str|None:
    m = DOI_CORE_RE.search(text)
    return f"https://doi.org/{m.group(1)}" if m else None

def canonicalize_accession(raw: str) -> str:
    # try to match label styles (tune on your CV)
    s = re.sub(r'\s+', ' ', raw.strip())
    if ACC_PATTERNS["PDB"].search(s):
        # normalize to "pdb xxxx" (lowercase prefix, lowercase code)
        parts = s.split()
        if len(parts) == 2 and parts[0].lower() == "pdb":
            return f"pdb {parts[1].lower()}"
        return s.lower()
    # Most others upper-case is normal
    for name, pat in ACC_PATTERNS.items():
        if name == "PDB": 
            continue
        if pat.search(s):
            return s.upper()
    return s

def is_doi_like(s: str) -> bool:
    return DOI_CORE_RE.search(s) is not None or "doi.org/" in s.lower()


from lxml import etree

def clean_text(x: str) -> str:
    return re.sub(r'\s+', ' ', (x or "").strip())

def parse_xml_sections(xml_path: Path):
    """
    Returns (sections, extras, raw_xml):
      sections: list[(section_name, text)]
      extras:   list[str] containing ext-link/pub-id/href attrs for DOI crawling
      raw_xml:  raw XML string (used as last resort mining surface)
    """
    sections, extras, raw_xml = [], [], ""
    try:
        raw_xml = xml_path.read_text(errors="ignore")
    except Exception:
        raw_xml = ""
    try:
        root = etree.parse(str(xml_path)).getroot()
        ns = root.nsmap
        def txt(node): 
            return clean_text(" ".join(node.itertext()))
        # Titles
        for t in root.findall(".//article-title", ns) + root.findall(".//title", ns):
            s = txt(t)
            if s: sections.append(("Title", s))
        # Abstract
        for a in root.findall(".//abstract", ns):
            s = txt(a)
            if s: sections.append(("Abstract", s))
        # Body sections
        body = root.find(".//body", ns)
        if body is not None:
            for sec in body.findall(".//sec", ns):
                name_node = sec.find(".//title", ns)
                name = txt(name_node) if name_node is not None else "Section"
                s = txt(sec)
                if s: sections.append((name, s))
        # ext-links & pub-ids
        for el in root.findall(".//ext-link", ns):
            s = txt(el)
            if s: extras.append(s)
        for pid in root.findall(".//pub-id", ns):
            s = txt(pid)
            if s: extras.append(s)
        # attributes (href/xlink:href/doi)
        for el in root.xpath('//*[@href or @xlink:href or @doi]'):
            for k in ["href","{http://www.w3.org/1999/xlink}href","doi"]:
                v = el.attrib.get(k)
                if v: extras.append(v)
    except Exception:
        pass
    return sections, extras, raw_xml


# PDF extraction (offline, optional): try pdfminer → try PyPDF2 → else skip gracefully
from pathlib import Path
import re

# Optional imports
try:
    from pdfminer.high_level import extract_text as pdfminer_extract
    _HAS_PDFMINER = True
except Exception:
    _HAS_PDFMINER = False

try:
    import PyPDF2  # only import the module; reader used inside the function
    _HAS_PYPDF2 = True
except Exception:
    _HAS_PYPDF2 = False

def pdf_text_pdfminer(p: Path) -> str:
    if not _HAS_PDFMINER:
        return ""
    try:
        return clean_text(pdfminer_extract(str(p)))
    except Exception:
        return ""

def pdf_text_pypdf(p: Path) -> str:
    if not _HAS_PYPDF2:
        return ""
    try:
        text = []
        with open(p, "rb") as f:
            r = PyPDF2.PdfReader(f)
            for page in r.pages:
                t = page.extract_text() or ""
                text.append(t)
        return clean_text(" ".join(text))
    except Exception:
        return ""

def read_pdf_text(p: Path) -> str:
    # Try pdfminer first if present
    t = pdf_text_pdfminer(p)
    # If too short or unavailable, try PyPDF2
    if len(t) < 50:
        t2 = pdf_text_pypdf(p)
        if len(t2) > len(t):
            return t2
    return t

print(f"PDF extractors available → pdfminer: {_HAS_PDFMINER}, PyPDF2: {_HAS_PYPDF2}")


def sentence_split(text: str):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def mine_candidates_in_text(section_name: str, text: str):
    """
    Yield tuples: (dataset_id, is_doi, section_name, context_sentence)
    """
    out = []
    sents = sentence_split(text)
    # DOIs (bare core handled)
    for m in DOI_CORE_RE.finditer(text):
        doi_full = f"https://doi.org/{m.group(1)}"
        ctx = next((s for s in sents if m.group(1) in s), text[:300])
        out.append((doi_full, True, section_name, ctx))
    # Accession IDs
    for name, pat in ACC_PATTERNS.items():
        for m in pat.finditer(text):
            token = clean_text(m.group(0))
            ctx = next((s for s in sents if token in s), text[:300])
            out.append((token, False, section_name, ctx))
    return out

def rule_based_type(section: str|None, ctx: str) -> str:
    t = f"{section or ''}. {ctx}".lower()
    pri = any(h in t for h in PRIMARY_HINTS)
    sec = any(h in t for h in SECONDARY_HINTS)
    if pri and not sec: return "Primary"
    if sec and not pri: return "Secondary"
    if section and any(k in (section or "").lower() for k in ["introduction","background","related"]):
        return "Secondary"
    return "Primary"

def extract_article_candidates(article_stem: str, xml_path: Path|None, pdf_path: Path|None, use_rule_types=False):
    """
    Return: list of dicts:
      {
        "article_id": article_stem,
        "dataset_id": normalized id (DOI→https form, accessions canonicalized),
        "is_doi": bool,
        "section": str,
        "context": str,
        "rule_type": "Primary"/"Secondary" (if use_rule_types)
      }
    """
    cands = {}
    # 1) XML first
    if xml_path and xml_path.exists():
        sections, extras, raw_xml = parse_xml_sections(xml_path)
        mined = []
        for sec, txt_ in sections:
            mined += mine_candidates_in_text(sec, txt_)
        if extras:
            mined += mine_candidates_in_text("References", ". ".join(extras))
        if not mined and raw_xml:
            mined += mine_candidates_in_text("XMLRaw", raw_xml)
        for ds_raw, is_doi, sec, ctx in mined:
            ds = normalize_doi_to_https(ds_raw) if is_doi else canonicalize_accession(ds_raw)
            if not ds: 
                continue
            k = (ds, sec, ctx)
            if k not in cands:
                cands[k] = {
                    "article_id": article_stem,
                    "dataset_id": ds,
                    "is_doi": is_doi,
                    "section": sec,
                    "context": ctx,
                }
                if use_rule_types:
                    cands[k]["rule_type"] = rule_based_type(sec, ctx)

    # 2) PDF fallback if no XML hits
    if not cands and pdf_path and pdf_path.exists():
        text = read_pdf_text(pdf_path)
        mined = []
        # try some coarse sections
        for sec in ["Data Availability","Availability of Data","Methods","Materials and Methods",
                    "Results","Introduction","References"]:
            pat = re.compile(rf'(?is){sec}.*?(?=(?:\n[A-Z][^\n]{{,80}}\n)|\Z)')
            for m in pat.finditer(text):
                mined += mine_candidates_in_text(sec, m.group(0))
        if not mined:
            mined = mine_candidates_in_text("FullText", text)
        for ds_raw, is_doi, sec, ctx in mined:
            ds = normalize_doi_to_https(ds_raw) if is_doi else canonicalize_accession(ds_raw)
            if not ds: 
                continue
            k = (ds, sec, ctx)
            if k not in cands:
                cands[k] = {
                    "article_id": article_stem,
                    "dataset_id": ds,
                    "is_doi": is_doi,
                    "section": sec,
                    "context": ctx,
                }
                if use_rule_types:
                    cands[k]["rule_type"] = rule_based_type(sec, ctx)

    return list(cands.values())

def iter_articles(root_dir: Path):
    # Map by stem for XML/PDF
    xml = {p.stem: p for p in root_dir.rglob("*.xml")}
    pdf = {p.stem: p for p in root_dir.rglob("*.pdf")}
    stems = sorted(set(xml) | set(pdf))
    for s in stems:
        yield s, xml.get(s), pdf.get(s)


if TRAIN_LABELS.exists():
    train_labels = pd.read_csv(TRAIN_LABELS)
    # Normalize label dataset_id DOIs to https form; keep accessions as-is
    def norm_label_dataset_id(x):
        x = str(x)
        if is_doi_like(x):
            n = normalize_doi_to_https(x)
            return n if n else x
        # canonicalize accessions (helps matching minor variations)
        return canonicalize_accession(x)
    train_labels["dataset_id_norm"] = train_labels["dataset_id"].map(norm_label_dataset_id)

    # Many competitions use article filename stem as the article_id (DOI with '/' -> '_').
    # We'll create both direct and underscore variants for matching.
    def id_variants(aid: str):
        s = str(aid).strip()
        return {s, s.replace("/", "_"), s.replace("_", "/")}

    label_key = set((a, d, t) for a, d, t in zip(
        train_labels["article_id"].astype(str),
        train_labels["dataset_id_norm"].astype(str),
        train_labels["type"].astype(str)
    ))

    print("Loaded train_labels:", train_labels.shape)

    # Mine candidates from train articles
    train_rows = []
    for stem, xmlp, pdfp in iter_articles(TRAIN_DIR):
        cands = extract_article_candidates(stem, xmlp, pdfp, use_rule_types=True)
        if not cands:
            continue
        # mark positives if (article_id, dataset_id) is in labels (regardless of type first),
        # then pick the correct type label for supervised training.
        # We'll match article_id by variants (stem vs labels)
        possible_article_ids = id_variants(stem)
        # get all labeled rows for those article ids
        sublab = train_labels[train_labels["article_id"].astype(str).isin(possible_article_ids)]
        # build lookup by dataset_id_norm
        ds_to_types = defaultdict(set)
        for _, r in sublab.iterrows():
            ds_to_types[r["dataset_id_norm"]].add(r["type"])
        for c in cands:
            ds = c["dataset_id"]
            label_types = ds_to_types.get(ds, set())
            if label_types:
                # In training, if both P/S exist (rare), duplicate two entries
                for typ in sorted(label_types):
                    train_rows.append({
                        "article_id": list(possible_article_ids)[0],  # store one form
                        "dataset_id": ds,
                        "context": c["context"],
                        "section": c["section"],
                        "label_type": typ,
                    })
            else:
                # hard negative (looks like an id but not in labels)
                # We mark as "NEG" and will drop or use for mention-filtering if needed
                train_rows.append({
                    "article_id": list(possible_article_ids)[0],
                    "dataset_id": ds,
                    "context": c["context"],
                    "section": c["section"],
                    "label_type": "NEG",
                })

    train_df = pd.DataFrame(train_rows)
    print("Train mined rows:", train_df.shape)
else:
    print("train_labels.csv not found — running inference-only.")
    train_df = pd.DataFrame(columns=["article_id","dataset_id","context","section","label_type"])


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import classification_report, f1_score

def build_text(c: str, s: str) -> str:
    return f"[{s}] {c}"

have_train = len(train_df) and any(t in ("Primary","Secondary") for t in train_df["label_type"])
if have_train:
    pos_df = train_df[train_df["label_type"].isin(["Primary","Secondary"])].copy()
    pos_df["text"] = pos_df.apply(lambda r: build_text(r["context"], r["section"]), axis=1)

    X_train, X_val, y_train, y_val = train_test_split(
        pos_df["text"], pos_df["label_type"], test_size=0.2, random_state=42, stratify=pos_df["label_type"]
    )

    # TF-IDF: word + char ngrams
    word_vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=150000)
    char_vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2, max_features=200000)
    # Combine by simple hstack via FeatureUnion-like manual fit/transform
    # We'll use a simple approach: concatenate features after fit.
    from scipy.sparse import hstack

    Xw = word_vec.fit_transform(X_train)
    Xc = char_vec.fit_transform(X_train)
    X_tr = hstack([Xw, Xc]).tocsr()

    Xw_val = word_vec.transform(X_val)
    Xc_val = char_vec.transform(X_val)
    X_vl = hstack([Xw_val, Xc_val]).tocsr()

    clf = LogisticRegression(max_iter=200, n_jobs=1) if "n_jobs" in LogisticRegression().get_params() else LogisticRegression(max_iter=200)
    clf.fit(X_tr, y_train)

    y_pred = clf.predict(X_vl)
    print("\nValidation classification report:\n", classification_report(y_val, y_pred))
    print("Macro F1:", f1_score(y_val, y_pred, average="macro"))
else:
    print("No labeled Primary/Secondary rows found — will use rule-based typing at inference.")
    word_vec = None
    char_vec = None
    clf = None


def predict_type(section: str, context: str) -> str:
    if clf is None:
        return rule_based_type(section, context)
    txt = build_text(context, section)
    Xw = word_vec.transform([txt])
    Xc = char_vec.transform([txt])
    from scipy.sparse import hstack
    X = hstack([Xw, Xc])
    return clf.predict(X)[0]

def extract_article_predictions(article_stem: str, xml_path: Path|None, pdf_path: Path|None):
    """
    Returns unique (article_id, dataset_id, type)
    """
    cands = extract_article_candidates(article_stem, xml_path, pdf_path, use_rule_types=False)
    if not cands:
        return []
    # Predict type per candidate and then dedup per (dataset_id, type)
    pred_map = {}  # ds -> chosen type (prefer Primary if any conflict)
    for c in cands:
        typ = predict_type(c["section"], c["context"])
        ds = c["dataset_id"]
        prev = pred_map.get(ds)
        if (prev is None) or (prev == "Secondary" and typ == "Primary"):
            pred_map[ds] = typ
    return [(article_stem, ds, t) for ds, t in pred_map.items()]

rows = []
pred_article_count = 0
for stem, xmlp, pdfp in iter_articles(TEST_DIR):
    preds = extract_article_predictions(stem, xmlp, pdfp)
    if preds:
        pred_article_count += 1
        rows.extend(preds)

sub = pd.DataFrame(rows, columns=["article_id","dataset_id","type"]).drop_duplicates()
# Enforce DOI normalization in final output (safety pass)
def enforce_doi(ds: str) -> str:
    if ds.lower().startswith("http"):
        return ds
    n = normalize_doi_to_https(ds)
    return n if n else ds

if len(sub):
    sub["dataset_id"] = sub["dataset_id"].map(enforce_doi)
    sub = sub.reset_index(drop=True).reset_index().rename(columns={"index":"row_id"})
    sub = sub[["row_id","article_id","dataset_id","type"]]
else:
    sub = pd.DataFrame(columns=["row_id","article_id","dataset_id","type"])

sub.to_csv(SUB_CSV, index=False)
print(sub.head(15))
print(f"\nSaved submission → {SUB_CSV}")
print(f"Total rows: {len(sub)}, Articles with ≥1 prediction: {pred_article_count}")




