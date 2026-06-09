! uv pip uninstall --system 'tensorflow'\n
! uv pip install --system --no-index --find-links='/kaggle/input/latest-mdc-whls/whls' 'pymupdf' 'vllm' 'triton' 'logits-processor-zoo' 'numpy<2'\n
! mkdir -p /tmp/src


%%writefile /tmp/src/helpers.py
import logging, os, kagglehub, inspect
from pathlib import Path
import polars as pl

IS_KAGGLE_ENV = sum(['KAGGLE' in k for k in os.environ]) > 0
IS_KAGGLE_SUBMISSION = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
COMP_DIR = Path(('/kaggle/input/make-data-count-finding-data-references' if IS_KAGGLE_SUBMISSION else kagglehub.competition_download('make-data-count-finding-data-references')))
PDF_DIR = COMP_DIR / ('test' if IS_KAGGLE_SUBMISSION else 'train') / 'PDF'
WORKING_DIR = Path(('/kaggle/working/' if IS_KAGGLE_ENV else '.working/'))

DOI_LINK = 'https://doi.org/'

DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper() if not IS_KAGGLE_SUBMISSION else "WARNING"
LOG_FILE_PATH = os.getenv("LOG_FILE", "logs/project.log")
LOG_DIR = Path(LOG_FILE_PATH).parent

LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FORMAT = "%(levelname)s %(asctime)s  [%(filename)s:%(lineno)d - %(funcName)s()] %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

def get_logger(name=None):
    if name is None:
        frame = inspect.currentframe()
        if frame is None or frame.f_back is None:
            name = "__main__"
        else:
            name = frame.f_back.f_globals.get("__name__", "__main__")

    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(DEFAULT_LOG_LEVEL)
        formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATEFMT)
        ch = logging.StreamHandler()
        ch.setLevel(DEFAULT_LOG_LEVEL)
        ch.setFormatter(formatter)
        fh = logging.FileHandler(LOG_FILE_PATH)
        fh.setLevel(DEFAULT_LOG_LEVEL)
        fh.setFormatter(formatter)
        logger.addHandler(ch)
        logger.addHandler(fh)
        logger.propagate = False
    return logger

def is_doi_link(name: str) -> pl.Expr:
    return pl.col(name).str.starts_with(DOI_LINK)

def string_normalization(name: str) -> pl.Expr:
    return pl.col(name).str.normalize("NFKC").str.replace_all(r"[^\p{Ascii}]", '').str.replace_all(r"https?://zenodo\.org/record/(\d+)", r" 10.5281/zenodo.$1 ")

def get_df(parse_dir: str):
    records = []
    txt_files = list(Path(parse_dir).glob('*.txt'))
    for txt_file in txt_files:
        id_ = txt_file.stem
        with open(txt_file, 'r') as f:
            text = f.read()
        records.append({'article_id': id_, 'text': text})
    return pl.DataFrame(records).with_columns(string_normalization('text').alias('text'))

def assume_type(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(pl.when(is_doi_link('dataset_id').or_(pl.col('dataset_id').str.starts_with('SAMN'))).then(pl.lit('Primary')).otherwise(pl.lit('Secondary')).alias('type'))
    )

def score(df, gt, on, tag='all'):
    hits = gt.join(df, on=on)
    tp = hits.height
    fp = df.height - tp
    fn = gt.height - tp
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0.0
    return f"{tag} - f1: {f1:.4f} [{tp}/{fp}/{fn}]"

def evaluate(df, on=['article_id', 'dataset_id']):
    gt = pl.read_csv(COMP_DIR/'train_labels.csv').filter(pl.col('type')!='Missing')
    return (
        score(df, gt, on),
        score(df.filter(is_doi_link('dataset_id')), gt.filter(is_doi_link('dataset_id')), on, 'doi'),
        score(df.filter(~is_doi_link('dataset_id')), gt.filter(~is_doi_link('dataset_id')), on, 'acc'),
    )


%%writefile /tmp/src/parse.py
import argparse
from pathlib import Path
import pymupdf
from helpers import get_logger, PDF_DIR

l = get_logger()

def pdf_to_txt(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = list(PDF_DIR.glob("*.pdf")) + list(PDF_DIR.glob("*.PDF"))
    existing_txt_files = {f.stem for f in output_dir.glob("*.txt")}
    for pdf_file in pdf_files:
        txt_file = output_dir / f"{pdf_file.stem}.txt"
        if pdf_file.stem in existing_txt_files:
            continue
        try:
            text = ""
            with pymupdf.open(pdf_file) as doc:
                for page in doc:
                    text += page.get_text()
            txt_file.write_text(text, encoding='utf-8')
        except Exception:
            pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=Path, help='Directory to save text files')
    args = parser.parse_args()
    pdf_to_txt(args.output_dir)

if __name__ == "__main__":
    main()


%%writefile /tmp/src/check_parse.py
import polars as pl
from pathlib import Path
from helpers import *

l=get_logger()

def gt_dataset_id_normalization(name:str) -> pl.Expr:
    return (
        pl.when(is_doi_link(name))
        .then(pl.col(name).str.split(DOI_LINK).list.last())
        .otherwise(name)
        .str.to_lowercase()
    )

def main():
    if IS_KAGGLE_SUBMISSION:
        l.debug('skipping check_parse for submission')
        return
    df = (
        get_df('/tmp/train_parse')
        .with_columns(pl.col('text').str.replace_all('\s+', '').str.to_lowercase().alias('text'))
    )

    gt = (
        pl.read_csv(COMP_DIR/'train_labels.csv')
        .filter(pl.col('article_id').is_in(df['article_id']))
        .filter(pl.col('type')!='Missing')
        .with_columns(gt_dataset_id_normalization('dataset_id').alias('norm_id'))
    )

    l.info(f"pymupdf misses: {gt.join(df, on='article_id').with_columns(hit=pl.col('text').str.contains(pl.col('norm_id'), literal=True)).filter(~pl.col('hit')).height} dataset_ids")

if __name__=='__main__': main()


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
                             .str.replace(r'[^A-Za-z0-9.\-_;()/]+$', '') # Allow more valid suffix characters
                             .str.to_lowercase()
                             .alias('dataset_id')
          )
          .group_by('article_id', 'dataset_id')
          .agg('match')
          .with_columns((DOI_LINK + pl.col('dataset_id')).alias('dataset_id'))
    )

    # UPDATED: Added more accession number patterns
    REGEX_IDS = (
        r"(?i)\b(?:"
        r"CHEMBL\d+|"
        r"E-GEOD-\d+|E-PROT-\d+|E-MTAB-\d+|E-MEXP-\d+|EMPIAR-\d+|"
        r"ENSBTAG\d+|ENSOARG\d+|"
        r"EPI_ISL_\d{5,}|EPI\d{6,7}|"
        r"HPA\d+|CP\d{6}|IPR\d{6}|PF\d{5}|BX\d{6}|KX\d{6}|K0\d{4}|CAB\d{6}|"
        r"NC_\d{6}\.\d{1}|NM_\d{9}|"
        r"PRJNA\d+|PRJDB\d+|PXD\d+|SAMN\d+|MSV\d+|"
        r"GSE\d+|GSM\d+|GPL\d+|MTBLS\d+|"
        r"CVCL_[A-Z0-9]{4}|"
        r"PDB\s?[1-9][A-Z0-9]{3}|HMDB\d+|"
        r"dryad\.[^\s\"<>]+|pasta\/[^\s\"<>]+|osf\.io/[a-z0-9]+|"
        r"(?:SR[APX]|STH|ER[PRX]|DR[PRX]|(E|D|S)RZ)\d{6,}|"
        r"[1-5]\.(?:10|20|30|40|50|60|70|80|90)\.\d{2,4}\.\d{2,4}|"
        r"10\.17632/[a-z0-9]+\.[0-9]"
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
                           .str.replace(r'[^A-Za-z0-9./_]+$', '') # More permissive cleaning
                           .str.replace(r'(?i)^PDB', '')
                           .alias('dataset_id')
        )
        .group_by('article_id', 'dataset_id')
        .agg('match')
        .with_columns(
            pl.when(pl.col('dataset_id').str.starts_with('dryad.'))
              .then(f'{DOI_LINK}10.5061/' + pl.col('dataset_id'))
              .when(pl.col('dataset_id').str.starts_with('pasta/'))
              .then(f'{DOI_LINK}10.6073/' + pl.col('dataset_id'))
              .when(pl.col('dataset_id').str.starts_with('osf.io/'))
              .then(f'{DOI_LINK}10.17605/' + pl.col('dataset_id'))
               .when(pl.col('dataset_id').str.contains(r'10\.17632/'))
              .then(f'{DOI_LINK}' + pl.col('dataset_id'))
              .otherwise('dataset_id')
              .alias('dataset_id')
        )
    )

    df = pl.concat([doi_df, acc_df])

    df = (
        df.unique(['article_id', 'dataset_id'])
          .filter(~pl.col('article_id').str.replace('_','/').str.contains(pl.col('dataset_id').str.split(DOI_LINK).list.last().str.escape_regex()))
          .filter(~pl.col('dataset_id').str.contains(pl.col('article_id').str.replace('_','/').str.escape_regex()))
          # UPDATED: Removed the aggressive figshare filter
          .filter(~pl.col('dataset_id').is_in(bad_ids))
          .filter(
              pl.when(is_doi_link('dataset_id') & (pl.col('dataset_id').str.split('/').list.last().str.len_chars() < 5))
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
import polars as pl
import os

from helpers import *

l = get_logger()

# UPDATED: Significantly improved and expanded system prompt
SYS_PROMPT_CLASSIFY_DOI = """
You are an expert in scientific literature and data citation. Your task is to classify a given text snippet containing a DOI or other identifier as either a reference to a dataset or a reference to literature (e.g., journal article, book, protocol).

**Output Format:**
- Respond with only the letter 'A' if it is a dataset.
- Respond with only the letter 'B' if it is literature.

---

### Classification Rules

#### 1. Classify as 'A' (Dataset)

**1.1. Identifier Prefix:** The identifier comes from a known data repository.
* **DOI Prefixes:**
    * **Dryad:** 10.5061
    * **Zenodo:** 10.5281
    * **Figshare:** 10.6084
    * **Mendeley Data:** 10.17632
    * **Dataverse:** 10.7910, 10.5683
    * **OSF:** 10.17605
    * **PANGAEA:** 10.1594
    * **EMPIAR:** 10.6019
    * **Neotoma Paleoecology:** 10.21233
    * **ICPSR:** 10.3886
* **Accession Number Prefixes (Non-DOI):**
    * **NCBI (SRA, BioProject, etc.):** SRP, ERP, DRP, PRJNA, PRJEB, SAMN
    * **EBI (ArrayExpress, ENA, etc.):** E-MTAB, E-GEOD, ERX, ERR
    * **Proteomics:** PXD, MSV
    * **Genomics/Sequence:** GSE, GSM, GEO, GenBank (NC_, NM_, etc.)
    * **Metabolomics:** MTBLS
    
**1.2. Contextual Keywords:** The surrounding text explicitly mentions data availability, deposition, or archiving.
* **Keywords:** `dataset`, `data set`, `data repository`, `data archive`, `data portal`, `deposited in`, `uploaded to`, `archived at`, `available at`, `stored on`, `hosted by`, `accessible via`, `retrieved from`, `supporting data`, `raw data`, `data availability`, `accession code`, `accession number`.

---

#### 2. Classify as 'B' (Literature)

**2.1. Identifier Prefix:** The DOI prefix belongs to a major academic publisher. This is a strong, but not absolute, signal.
* **Publisher DOI Prefixes:** `10.1016` (Elsevier), `10.1038` (Nature), `10.1126` (Science), `10.1007` (Springer), `10.1002` (Wiley), `10.1021` (ACS), `10.1101` (bioRxiv/medRxiv), `10.1093` (Oxford), `10.1111` (Wiley), `10.1080` (Taylor & Francis), `10.1177` (SAGE), `10.1242` (Company of Biologists), `10.1152` (American Physiological Society), `10.1371` (PLOS).

**2.2. Contextual Keywords:** The text refers to a journal, article, book, chapter, conference paper, preprint, or method without any of the data-related keywords from section 1.2.
* **Keywords:** `supplementary material`, `supplementary information` (when used alone without a repository name), `article`, `journal`, `published in`, `citation`, `reference`.

---

### Decision Process

1.  **Check for Dataset Prefixes (Rule 1.1):** If a match is found, classify as **'A'**.
2.  **Check for Publisher Prefixes (Rule 2.1):** If a publisher prefix is found, check the context.
3.  **Analyze Context (Rules 1.2 & 2.2):**
    * If data keywords are present, classify as **'A'**.
    * If only literature keywords are present, classify as **'B'**.
4.  **Default:** If the prefix is unrecognized and the context is ambiguous, default to **'B'**.

---

### Examples

* `Raw images are stored on Figshare (DOI 10.6084/m9.figshare.1234567).` -> **A** (Rule 1.1)
* `Sequence reads available under BioProject accession PRJNA765432.` -> **A** (Rule 1.1)
* `As described in Nature Methods (DOI 10.1038/s41592-020-0793-2).` -> **B** (Rule 2.1)
* `See Supplementary Data at Zenodo (10.5281/zenodo.987654).` -> **A** (Rule 1.1 & 1.2)
* `Method details published in J. Proteome Res. DOI: 10.1021/acs.jproteome.0c00845.` -> **B** (Rule 2.1)
* `Data uploaded to Dryad (10.5061/dryad.x1y2z3).` -> **A** (Rule 1.1)
* `Referenced paper: DOI 10.1101/2020.01.01.123456 (bioRxiv preprint).` -> **B** (Rule 2.1)
""".strip()

def build_df():
    df = pl.read_parquet('/tmp/extracted.parquet')
    df.filter(~is_doi_link('dataset_id')).select('article_id', 'dataset_id').write_csv('/tmp/accid_sub.csv')
    return df.filter(is_doi_link('dataset_id'))

def build_prompt(tokenizer, df):
    prompts = []
    for doi, text in df.select('dataset_id', 'window').rows():
        messages = [{'role':'system','content': SYS_PROMPT_CLASSIFY_DOI}, {'role':'user', 'content': text}]
        prompts.append(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))
    return df.with_columns(pl.Series('prompt', prompts))

if __name__=='__main__':
    os.environ["VLLM_USE_V1"] = "0"
    import vllm
    from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
    model_path = "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"
    llm = vllm.LLM(model_path, quantization='awq', tensor_parallel_size=2, gpu_memory_utilization=0.9, trust_remote_code=True, dtype="half", enforce_eager=True, max_model_len=2048, disable_log_stats=True, disable_custom_all_reduce=True, enable_prefix_caching=True, task='generate')
    tokenizer = llm.get_tokenizer()
    df = build_df()
    df = build_prompt(tokenizer, df)
    prompts = df['prompt'].to_list()
    mclp = MultipleChoiceLogitsProcessor(tokenizer, choices=["A", "B"])
    outputs = llm.generate(prompts, vllm.SamplingParams(seed=777, temperature=0, skip_special_tokens=True, max_tokens=1, logits_processors=[mclp], logprobs=len(mclp.choices)), use_tqdm=True)
    logprobs = [{lp.decoded_token: lp.logprob for lp in list(lps)} for lps in [output.outputs[0].logprobs[0].values() for output in outputs]]
    choices = [max(d, key=d.get) for d in logprobs]
    types = {'A': True, 'B': False}
    choices = [types[c] for c in choices]
    df = df.with_columns(pl.Series('type', choices))
    df.filter(pl.col('type')).select('article_id', 'dataset_id').write_csv('/tmp/doi_sub.csv')
    df = pl.concat([pl.read_csv('/tmp/doi_sub.csv'), pl.read_csv('/tmp/accid_sub.csv')])
    df = assume_type(df)
    df.select(['article_id', 'dataset_id', 'type']).with_row_index(name='row_id').write_csv('/kaggle/working/submission.csv')
    if not IS_KAGGLE_SUBMISSION:
        results = evaluate(df)
        for r in results: l.info(r) 
        results = evaluate(df, on=['article_id', 'dataset_id', 'type'])
        for r in results: l.info(r)


%%writefile /tmp/src/post_filter.py
import polars as pl
from helpers import *

"""
Fourth essence: Post-filter to cut FP DOIs that look like literature.
- Read /kaggle/working/submission.csv (output of llm_validate.py)
- Join with /tmp/extracted.parquet to get context window
- NEW LOGIC: Drop a DOI row ONLY IF it (1) starts with a typical publisher prefix AND (2) has no data-related keywords nearby.
- Keep accessions untouched.
"""

l = get_logger()

# UPDATED: List of common academic publisher DOI prefixes
PUBLISHER_PREFIXES = [
    "10.1016", # Elsevier
    "10.1038", # Nature
    "10.1126", # Science
    "10.1007", # Springer
    "10.1002", # Wiley
    "10.1021", # ACS
    "10.1101", # bioRxiv/medRxiv
    "10.1093", # Oxford University Press
    "10.1111", # Wiley-Blackwell
    "10.1080", # Taylor & Francis
    "10.1177", # SAGE
    "10.1242", # The Company of Biologists
    "10.1152", # American Physiological Society
    "10.1371", # PLOS
    "10.1186", # BMC
    "10.3389", # Frontiers
    "10.1098", # Royal Society
]

# UPDATED: List of known data repository prefixes
DATA_REPO_PREFIXES = [
    "10.5061", # Dryad
    "10.5281", # Zenodo
    "10.6084", # Figshare
    "10.17632",# Mendeley Data
    "10.7910", # Dataverse
    "10.17605",# OSF
    "10.1594", # PANGAEA
    "10.6019", # EMPIAR
    "10.21233",# Neotoma Paleoecology
    "10.3886", # ICPSR
    "10.4225", # CSIRO
    "10.15468" # GBIF
]

# UPDATED: Expanded context keywords to match the LLM prompt
CONTEXT_RE = r"(?i)\\b(data(?:set)?|repositor|archive|accession|deposit|available|upload|host|stor|retriev|raw data|supporting data)\\b"

def is_publisher_prefix(col: str = "dataset_id") -> pl.Expr:
    expr = pl.lit(False)
    for p in PUBLISHER_PREFIXES:
        expr = expr | pl.col(col).str.starts_with(f"{DOI_LINK}{p}")
    return expr

def is_data_repo_prefix(col: str = "dataset_id") -> pl.Expr:
    expr = pl.lit(False)
    for p in DATA_REPO_PREFIXES:
        expr = expr | pl.col(col).str.starts_with(f"{DOI_LINK}{p}")
    return expr

def main():
    sub = pl.read_csv("/kaggle/working/submission.csv")

    if "row_id" in sub.columns:
        sub = sub.drop("row_id")

    win = pl.read_parquet("/tmp/extracted.parquet").select("article_id", "dataset_id", "window")

    doi_rows = sub.filter(is_doi_link("dataset_id")).join(win, on=["article_id", "dataset_id"], how="left")
    acc_rows = sub.filter(~is_doi_link("dataset_id"))

    # UPDATED: Corrected filtering logic
    # Keep a DOI if it's:
    # 1. From a known data repository, OR
    # 2. Has data-related keywords in its context, OR
    # 3. Is NOT from a known publisher.
    # This correctly targets and removes only high-confidence literature DOIs.
    keep_mask = (
        is_data_repo_prefix("dataset_id")
        | doi_rows["window"].fill_null("").str.contains(CONTEXT_RE)
        | ~is_publisher_prefix("dataset_id")
    )

    kept_doi = doi_rows.filter(keep_mask).select("article_id", "dataset_id", "type")
    final = pl.concat([kept_doi, acc_rows.select("article_id", "dataset_id", "type")])

    if not IS_KAGGLE_SUBMISSION:
        for r in evaluate(final): l.info(r)
        for r in evaluate(final, on=["article_id", "dataset_id", "type"]): l.info(r)

    final.with_row_index("row_id").write_csv("/kaggle/working/submission.csv")

if __name__ == "__main__":
    main()


%cd /tmp
!LOG_LEVEL=INFO python src/parse.py /tmp/train_parse
! python src/check_parse.py
! python src/getid.py
! python src/llm_validate.py
! python src/post_filter.py
! grep "f1:" /tmp/logs/project.log

