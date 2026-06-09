! uv pip uninstall --system 'tensorflow'
! uv pip install --system --no-index --find-links='/kaggle/input/latest-mdc-whls/whls' 'pymupdf' 'vllm' 'triton' 'logits-processor-zoo' 'numpy<2'
! mkdir -p /tmp/src


%%writefile /tmp/src/helpers.py
import os, re, logging, sys
from pathlib import Path
from typing import Optional, Iterable
import polars as pl

# ---- Env & paths ----
IS_KAGGLE = "KAGGLE_URL_BASE" in os.environ or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
IS_SUBMIT = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
COMP_DIR = Path("/kaggle/input/make-data-count-finding-data-references") if IS_KAGGLE else Path("./input")
PDF_DIR = COMP_DIR / ("test" if IS_SUBMIT else "train") / "PDF"
LOG_PATH = Path("/tmp/logs/project.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---- Regex (보강) ----
DOI_RE = re.compile(r"(?i)\b10\.\d{4,9}/[-._;()/:A-Z0-9]+")
ACCESSION_RE = re.compile(
    r"(?i)\b(?:GSE|GSM|GDS|SRR|SRS|SRX|SRA|PRJ[EDN][A-Z]|ENA|EGAS|EGAD|PXD|E-MTAB|E-GEOD|PDB|EMD|ERP\d+)\d+\b"
)
CONTEXT_RE = re.compile(
    r"(?i)\b(data(?: ?set)?|repository|database|archive|supplement(?:ary|al)?|supp\.?|accession|doi|"
    r"available at|deposited|submitted|uploaded|hosted|stored|raw\s*data)\b"
)

# ---- Logger ----
def get_logger():
    log = logging.getLogger("mdc")
    if not log.handlers:
        log.setLevel(logging.INFO)
        fh = logging.FileHandler(LOG_PATH)
        sh = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        for h in (fh, sh):
            h.setFormatter(fmt); h.setLevel(logging.INFO); log.addHandler(h)
        log.propagate = False
    return log

L = get_logger()

# ---- Text utils ----
def norm_text(s: str) -> str:
    s = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", s)
    s = re.sub(r"[\s\t\n]+", " ", s).strip()
    return s

def normalize_doi_from_text(s: str) -> Optional[str]:
    m = DOI_RE.search(s)
    if not m: 
        return None
    return m.group(0).strip("[](){}.,;> ").lower()

def windows(text: str, size: int = 700, step: int = 450) -> Iterable[str]:
    words = text.split()
    if not words:
        return
    for i in range(0, max(1, len(words)-1), step):
        chunk = " ".join(words[i:i+size])
        if chunk:
            yield chunk

# ---- Eval ----
def load_gt() -> Optional[pl.DataFrame]:
    p = COMP_DIR / "train.csv"
    if p.exists():
        return pl.read_csv(p).select(["article_id", "dataset_id"])
    return None

def f1_report(pred: pl.DataFrame, tag="all") -> str:
    gt = load_gt()
    if gt is None: 
        return ""
    on = ["article_id","dataset_id"]
    p = pred.unique(subset=on)
    g = gt.unique(subset=on)
    hits = g.join(p, on=on, how="inner")
    tp = hits.height; fp = p.height - tp; fn = g.height - tp
    f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.0
    return f"{tag} - f1: {f1:.4f} [{tp}/{fp}/{fn}]"



%%writefile /tmp/src/parse.py
import sys, fitz
from pathlib import Path
from helpers import PDF_DIR, L, norm_text

def parse_pdf(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    texts = []
    for page in doc:
        try:
            t = page.get_text("text")
        except Exception:
            t = page.get_text()
        texts.append(t)
    return norm_text("\n".join(texts))

def main(out_dir: str):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        aid = pdf.stem
        try:
            text = parse_pdf(pdf)
            (out/f"{aid}.txt").write_text(text, encoding="utf-8")
        except Exception as e:
            L.error(f"parse error {pdf}: {e}")
    L.info(f"parsed to {out}")

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv)>1 else "/tmp/train_parse"
    main(out)



%%writefile /tmp/src/getid.py
import polars as pl, re
from pathlib import Path
from helpers import L, normalize_doi_from_text, ACCESSION_RE, CONTEXT_RE

PARSE_DIR = Path("/tmp/train_parse")

def extract_candidates(txt: str):
    cands = set()
    # DOI
    d = normalize_doi_from_text(txt)
    if d: 
        cands.add(("doi", d))
    # Accession
    for m in ACCESSION_RE.finditer(txt):
        cands.add(("acc", m.group(0).upper()))
    return list(cands)

def main():
    records = []
    for fp in sorted(PARSE_DIR.glob("*.txt")):
        aid = fp.stem
        text = fp.read_text(encoding="utf-8", errors="ignore")
        # 문장 경계 기준으로 쪼개 precision을 높임 (참조/서론 bleed 방지)
        for sent in re.split(r"(?<=\\.)\\s+", text):
            if not sent: 
                continue
            has_ctx = bool(CONTEXT_RE.search(sent))
            for typ, did in extract_candidates(sent):
                records.append({
                    "article_id": aid, 
                    "dataset_id": did, 
                    "type": "Primary" if has_ctx else "Candidate", 
                    "window": sent[:800]
                })
    df = pl.DataFrame(records) if records else pl.DataFrame({"article_id":[],"dataset_id":[],"type":[],"window":[]})
    df.write_parquet("/tmp/candidates.parquet")
    L.info(f"candidates: {df.shape}")

if __name__ == "__main__":
    main()



%%writefile /tmp/src/getid.py
import re
import polars as pl
from typing import Optional, Tuple

from helpers import *

COMPILED_PATTERNS = {
    'ref_header_patterns': [re.compile(r'\b(R\s*E\s*F\s*E\s*R\s*E\s*N\s*C\s*E\s*S|BIBLIOGRAPHY|LITERATURE CITED|WORKS CITED|CITED WORKS|ACKNOWLEDGEMENTS)\b[:\s]*', re.IGNORECASE)],    
    'citation_pattern': re.compile(r'^\s*(\[\d+\]|\(\d+\)|\d+\.|\d+\)|\d+(?=\s|$))\s*'),
    'first_citation_patterns': [
        re.compile(r'^\s*\[1\]\s*'),
        re.compile(r'^\s*\(1\)\s*'),
        re.compile(r'^\s*1\.\s*'),
        re.compile(r'^\s*1\)\s*'),
        re.compile(r'^\s*1(?=\s|$)'),
    ],
}

l = get_logger()

def find_last_reference_header(text: str, header_patterns: list[re.Pattern]) -> Optional[int]:
    last_match_idx = None
    for pattern in header_patterns:
        matches = list(pattern.finditer(text))
        if matches:
            last_match_idx = matches[-1].start()
    return last_match_idx

def find_last_first_citation(text: str) -> Optional[int]:
    lines = text.splitlines()
    last_match_line = None
    for line_num, line in enumerate(lines):
        line = line.strip()
        for pattern in COMPILED_PATTERNS['first_citation_patterns']:
            if pattern.match(line):
                next_lines = lines[line_num:line_num+3]
                if any(COMPILED_PATTERNS['citation_pattern'].match(l.strip()) for l in next_lines[1:]):
                    last_match_line = line_num
                break
    return last_match_line

def find_reference_start(text: str) -> Optional[int]:
    lines = text.splitlines()
    last_first_citation = find_last_first_citation(text)
    if last_first_citation is not None:
        return last_first_citation
    start_search_idx = int(len(lines) * 0.5)
    for i in range(start_search_idx, len(lines)):
        line = lines[i].strip()
        if COMPILED_PATTERNS['citation_pattern'].match(line):
            next_lines = lines[i:i+3]
            if sum(1 for l in next_lines if COMPILED_PATTERNS['citation_pattern'].match(l.strip())) >= 2:
                for j in range(i, max(-1, i-10), -1):
                    if not COMPILED_PATTERNS['citation_pattern'].match(lines[j].strip()):
                        return j + 1
                return max(0, i-10)
    return None

def split_text_and_references(text: str) -> Tuple[str, str]:
    header_idx = find_last_reference_header(text, COMPILED_PATTERNS['ref_header_patterns'])
    if header_idx is not None:
        header_idx2 = find_last_reference_header(text[:header_idx].strip(), COMPILED_PATTERNS['ref_header_patterns'])
        if header_idx2 is not None:
            header_idx3 = find_last_reference_header(text[:header_idx2].strip(), COMPILED_PATTERNS['ref_header_patterns'])
            if header_idx3 is not None:
                return text[:header_idx3].strip(), text[header_idx3:].strip()
            return text[:header_idx2].strip(), text[header_idx2:].strip()
        return text[:header_idx].strip(), text[header_idx:].strip()
    ref_start_line = find_reference_start(text)
    if ref_start_line is not None:
        lines = text.splitlines()
        body = '\n'.join(lines[:ref_start_line])
        refs = '\n'.join(lines[ref_start_line:])
        return body.strip(), refs.strip()
    return text.strip(), ''

def get_splits(df: pl.DataFrame) -> pl.DataFrame:
    bodies, refs = [], []
    for raw_text in df['text']:
        main, ref = split_text_and_references(raw_text)
        bodies.append(main)
        refs.append(ref)
    return df.with_columns(pl.Series('body', bodies), pl.Series('ref', refs))

def tidy_extraction(df) -> pl.DataFrame:
    bad_ids = [f'{DOI_LINK}{e}' for e in ['10.5061/dryad', '10.5281/zenodo', '10.6073/pasta']]

    doi_df = (
        df.with_columns(pl.col('body').str.extract_all(r'10\s*\.\s*\d{4,9}\s*/\s*\S+').alias('match'))
          .explode('match')
          .drop_nulls('match')
          .with_columns(
              pl.col('match').str.replace_all(r'\s+', '')
                             .str.replace(r'[^A-Za-z0-9]+$', '')
                             .str.to_lowercase()
                             .alias('dataset_id')
          )
          .group_by('article_id', 'dataset_id')
          .agg('match')
          .with_columns((DOI_LINK + pl.col('dataset_id')).alias('dataset_id'))
    )

    # REGEX_IDS = (
    #     r"(?i)\b(?:"
    #     r"CHEMBL\d+|"
    #     r"E-GEOD-\d+|E-PROT-\d+|E-MTAB-\d+|E-MEXP-\d+|EMPIAR-\d+|"
    #     r"ENSBTAG\d+|ENSOARG\d+|"
    #     r"EPI_ISL_\d{5,}|EPI\d{6,7}|"
    #     r"HPA\d+|CP\d{6}|IPR\d{6}|PF\d{5}|BX\d{6}|KX\d{6}|K0\d{4}|CAB\d{6}|"
    #     r"NC_\d{6}\.\d{1}|NM_\d{9}|"
    #     r"PRJNA\d+|PRJEB\d+|PRJDB\d+|PXD\d+|SAMN\d+|"
    #     r"GSE\d+|GSM\d+|GPL\d+|"
    #     r"PDB\s?[1-9][A-Z0-9]{3}|HMDB\d+|"
    #     r"dryad\.[^\s\"<>]+|pasta\/[^\s\"<>]+|"
    #     r"(?:SR[PRX]|STH|ERR|DRR|DRX|DRP|ERP|ERX)\d+"
    #     r")"
    # )  

    REGEX_IDS = (
        r"(?i)\b(?:"
        r"CHEMBL\d+|"
        r"E-GEOD-\d+|E-PROT-\d+|E-MTAB-\d+|E-MEXP-\d+|EMPIAR-\d+|"
        r"ENSBTAG\d+|ENSOARG\d+|"
        r"EPI_ISL_\d{5,}|EPI\d{6,7}|"
        r"HPA\d+|CP\d{6}|IPR\d{6}|PF\d{5}|BX\d{6}|KX\d{6}|K0\d{4}|CAB\d{6}|"
        r"NC_\d{6}\.\d{1}|NM_\d{9}|"
        # 修正点1: PRJEB を追加
        r"PRJNA\d+|PRJEB\d+|PRJDB\d+|PXD\d+|SAMN\d+|"
        r"GSE\d+|GSM\d+|GPL\d+|"
        r"PDB\s?[1-9][A-Z0-9]{3}|HMDB\d+|"
        r"dryad\.[^\s\"<>]+|pasta\/[^\s\"<>]+|"
        # 修正点2: SRR と SRA にもマッチするように SR[PX] を SR[RPAX] へ変更
        r"(?:SR[RPAX]|STH|ERR|DRR|DRX|DRP|ERP|ERX)\d+|"
        r"CVCL_[A-Z0-9]{4}"
        r")"
    )
    
    acc_df = (
        df.with_columns(
            pl.col('text').str.extract_all(REGEX_IDS).alias('match')
        )
        .explode('match')
        .drop_nulls('match')
        .with_columns(
            pl.col('match').str.replace_all(r'\s+', '')
                           .str.replace(r'[^A-Za-z0-9]+$', '')
                           .str.replace(r'(?i)^PDB', '')
                           .alias('dataset_id')
        )
        .group_by('article_id', 'dataset_id')
        .agg('match')
        .with_columns(
            pl.when(pl.col('dataset_id').str.starts_with('dryad.'))
              .then(f'{DOI_LINK}10.5061/' + pl.col('dataset_id'))
              .otherwise('dataset_id')
              .alias('dataset_id')
        )
        .with_columns(
            pl.when(pl.col('dataset_id').str.starts_with('pasta/'))
              .then(f'{DOI_LINK}10.6073/' + pl.col('dataset_id'))
              .otherwise('dataset_id')
              .alias('dataset_id')
        )
    )

    df = pl.concat([doi_df, acc_df])

    df = (
        df.unique(['article_id', 'dataset_id'])  # CHANGED
          .filter(~pl.col('article_id').str.replace('_','/').str.contains(pl.col('dataset_id').str.split(DOI_LINK).list.last().str.escape_regex()))
          .filter(~pl.col('dataset_id').str.contains(pl.col('article_id').str.replace('_','/').str.escape_regex()))
          .filter(~pl.col('dataset_id').str.contains('figshare', literal=True))
          .filter(~pl.col('dataset_id').is_in(bad_ids))
          .filter(
              pl.when(is_doi_link('dataset_id') &
                      (pl.col('dataset_id').str.split('/').list.last().str.len_chars() < 5))
               .then(False)
               .otherwise(True)
          )
          .with_columns(pl.col('match').list.unique())
    )
    return df

def get_context_window(text: str, substring: str, window: int = 100) -> str:
    idx = text.find(substring)
    if idx == -1:
        raise ValueError
    start = max(idx - window, 0)
    end = min(idx + len(substring) + window, len(text))
    return text[start:end]

def get_window_df(text_df, ids_df):
    df = ids_df.join(text_df, on='article_id')
    windows = []
    for text, match_ids in df.select('text', 'match').rows():
        windows.append(get_context_window(text, match_ids[0]))
    return df.with_columns(pl.Series('window', windows)).select('article_id', 'dataset_id', 'window')

def main():
    text_df = get_df('/tmp/train_parse')
    df = get_splits(text_df)
    df = tidy_extraction(df)
    df = get_window_df(text_df, df)
    df.write_parquet('/tmp/extracted.parquet')
    df = assume_type(df)
    df.select(['article_id', 'dataset_id', 'type']).with_row_index(name='row_id').write_csv('/kaggle/working/submission.csv')
    if not IS_KAGGLE_SUBMISSION:
        results = evaluate(df)
        for r in results: l.info(r)
        results = evaluate(df, on=['article_id', 'dataset_id', 'type'])
        for r in results: l.info(r)

if __name__=='__main__': main()


%%writefile /tmp/src/llm_validate.py
import polars as pl, re
from pathlib import Path
from helpers import L

P = Path("/tmp/candidates.parquet")
DF = pl.read_parquet(P) if P.exists() else pl.DataFrame({"article_id":[],"dataset_id":[],"type":[],"window":[]})

def valid_doi(s: str) -> bool:
    return s.startswith("10.") and ("/" in s) and len(s) >= 12

def valid_acc(s: str) -> bool:
    return bool(re.match(r"(?i)^(?:GSE|GSM|GDS|SR[RSX]|SRA|PRJ[EDN][A-Z]|ENA|EGAS|EGAD|PXD|E\\-MTAB|E\\-GEOD|PDB|EMD|ERP\\d+)\\d+$", s))

def main():
    if DF.is_empty():
        pl.DataFrame({"article_id":[],"dataset_id":[],"type":[]}).write_parquet("/tmp/validated.parquet"); return
    keep = []
    for r in DF.iter_rows(named=True):
        s = r["dataset_id"]
        if valid_doi(s) or valid_acc(s):
            keep.append({"article_id": r["article_id"], "dataset_id": s, "type": r["type"], "window": r["window"]})
    out = pl.DataFrame(keep).unique(subset=["article_id","dataset_id"])
    out.write_parquet("/tmp/validated.parquet")
    L.info(f"validated: {out.shape}")

if __name__ == "__main__":
    main()



%%writefile /tmp/src/post_filter.py
import polars as pl, re
from helpers import L, CONTEXT_RE, f1_report

VAL = pl.read_parquet("/tmp/validated.parquet")

# 너무 짧은 DOI 꼬리 문자열 차단(자주 나오는 FP 패턴)
BLACKLIST = [re.compile(r"10\\.\\d{4,9}/\\w{1,6}$", re.I)]

def strong_context(s: str) -> bool:
    # 컨텍스트 키워드가 최소 1개라도 나오면 Primary 승격
    return bool(CONTEXT_RE.search(s))

def main():
    if VAL.is_empty():
        pl.DataFrame({"row_id":[], "article_id":[], "dataset_id":[], "type":[]}).write_csv("/kaggle/working/submission.csv"); 
        return

    df = VAL

    # 블랙리스트 필터
    ok_mask = pl.Series([not any(p.search(d) for p in BLACKLIST) for d in df["dataset_id"]])
    df = df.with_columns(ok_mask.alias("ok")).filter(pl.col("ok")).drop("ok")

    # 컨텍스트 기반 승격
    df = df.with_columns(
        pl.when(pl.col("type")=="Primary").then(pl.lit("Primary"))
         .otherwise(pl.when(pl.col("window").map_elements(strong_context))
         .then(pl.lit("Primary")).otherwise(pl.lit("Secondary"))).alias("type")
    )

    final = df.select(["article_id","dataset_id","type"]).unique()
    rep = f1_report(final)
    if rep: L.info(rep)
    final.with_row_index("row_id").write_csv("/kaggle/working/submission.csv")

if __name__ == "__main__":
    main()



%cd /tmp
! python src/parse.py /tmp/train_parse
! python src/getid.py
! python src/llm_validate.py
! python src/post_filter.py
! grep "f1:" /tmp/logs/project.log || true
! head -n 20 /kaggle/working/submission.csv





