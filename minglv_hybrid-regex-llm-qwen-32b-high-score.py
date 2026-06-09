# åˆ é™¤/kaggle/workingç›®å½•ä¸‹æ‰€æœ‰æ–‡ä»¶
!rm -f /kaggle/working/*

# ç¡®è®¤åˆ é™¤ç»“æ�œ
!ls -la /kaggle/working/


! uv pip uninstall --system 'tensorflow'
! uv pip install --system --no-index --find-links='/kaggle/input/latest-mdc-whls/whls' 'pymupdf' 'vllm' 'triton' 'logits-processor-zoo' 'numpy<2'
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
LOG_FORMAT = "%(levelname)s %(asctime)s [%(filename)s:%(lineno)d - %(funcName)s()] %(message)s"
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
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0.0
    return f"{tag} - f1: {f1:.4f} precision: {precision:.4f} recall: {recall:.4f} [{tp}/{fp}/{fn}]"

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

# åŸºäº�åˆ†æ��ä¼˜åŒ–çš„æ•°æ�®ä»“åº“å‰�ç¼€
DATA_REPOSITORY_PREFIXES = [
    "10.5061", "10.5281", "10.5066", "10.15468", "10.1594", "10.5256", 
    "10.3886", "10.6073", "10.17882", "10.18150", "10.7937", "10.6075",   
    "10.17632", "10.6096", "10.4121", "10.25377", "10.25387", "10.22033", 
    "10.13020", "10.11583", "10.17862", "10.23642", "10.4231", "10.5518", 
    "10.17863", "10.17638", "10.5441", "10.34973", "10.3334", "10.5067", 
    "10.25326", "10.13012", "10.5285", "10.21942", "10.25349", "10.25386", 
    "10.15482", "10.24381", "10.18434", "10.15131", "10.11588", "10.7291",
    "10.6078", "10.15125", "10.25422", "10.5291", "10.7910", "10.15485",
    "10.25921"
]

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
    """ç²¾ç¡®åº¦ä¼˜åŒ–çš„æ��å�–é€»è¾‘"""
    
    # 1. DOIæ��å�– - å�ªæ��å�–æ•°æ�®ä»“åº“å‰�ç¼€
    data_prefixes_pattern = '|'.join([p.replace('.', r'\.') for p in DATA_REPOSITORY_PREFIXES])
    doi_patterns = [
        r'(?:https?://)?(?:www\.)?(?:doi\.org/)?(' + data_prefixes_pattern + r')/[^\s\]\)\}\"\'<>\(\)]+',
        r'(?:doi:\s*)?(' + data_prefixes_pattern + r')/[^\s\]\)\}\"\'<>\(\)]+',
        r'\b(' + data_prefixes_pattern + r')/[^\s\]\)\}\"\'<>\(\)]+'
    ]
    
    doi_dfs = []
    for pattern in doi_patterns:
        doi_df = (
            df.with_columns(pl.col('text').str.extract_all(pattern).alias('match'))
            .explode('match')
            .drop_nulls('match')
        )
        if doi_df.height > 0:
            doi_dfs.append(doi_df)
    
    if doi_dfs:
        doi_df = pl.concat(doi_dfs, how='vertical')
        doi_df = (
            doi_df.with_columns(
                pl.col('match')
                .str.replace_all(r'^(?:https?://)?(?:www\.)?(?:doi\.org/)?', '')
                .str.replace_all(r'^(?:doi:\s*)?', '')
                .str.replace_all(r'\s+', '')
                .str.replace(r'[^\w\.\-/]+$', '')
                .str.to_lowercase()
                .alias('dataset_id')
            )
            # æ”¹è¿›ï¼šDOIå��ç¼€é•¿åº¦è¿‡æ»¤
            .filter(pl.col('dataset_id').str.split('/').list.last().str.len_chars() >= 4)
            .filter(pl.col('dataset_id').str.split('/').list.last().str.len_chars() <= 50)
            .group_by('article_id', 'dataset_id')
            .agg('match')
            .with_columns((DOI_LINK + pl.col('dataset_id')).alias('dataset_id'))
        )
    else:
        doi_df = pl.DataFrame({'article_id': [], 'dataset_id': [], 'match': []})
    
    # 2. ç²¾ç¡®åº¦ä¼˜åŒ–çš„Accessionæ��å�–
    ENHANCED_REGEX = (
        r"\b(?:"
        r"SAMN\d{8,}|"           # æ��é«˜è¦�æ±‚åˆ°8ä½�ï¼Œå‡�å°‘è¯¯æŠ¥
        r"EPI_ISL_\d{6,}|EPI\d{7,}|"  # æ��é«˜EPIè¦�æ±‚     
        r"CHEMBL\d{2,}|"         # è¦�æ±‚è‡³å°‘2ä½�æ•°å­—
        r"IPR\d{6,}|"            
        r"PRJNA\d{6,}|PRJEB\d{6,}|PRJDB\d{6,}|"  
        r"CVCL_[A-Z0-9]{4,}|"    
        r"ENS[A-Z]{3}[GT]\d{11}|"  
        r"GSE\d{4,}|GSM\d{7,}|GPL\d{3,}|"  # æ��é«˜GSMè¦�æ±‚
        r"(?:SRR|ERR|DRR)\d{7,}|"    # æ��é«˜è¦�æ±‚åˆ°7ä½�
        r"(?:SRX|ERX|DRX)\d{6,}|"      
        r"(?:SRP|ERP|DRP)\d{6,}|"    
        r"EMPIAR-\d{5,}|"        
        r"PDB[1-9][A-Z0-9]{3}|"  
        r"[A-Z]{2}\d{6,}\.\d+|"  
        r"Q[0-9][A-Z0-9]{4}[0-9]|"
        r"PF\d{5,}|"             
        r"STH\d{5,}|"
        r"5VA1|3\.\d{2,}\.\d{2,}\.\d{2,}"  # è¦�æ±‚è‡³å°‘2ä½�æ•°å­—
        r")"
    )
    
    acc_df = (
        df.with_columns(
            pl.col('text').str.extract_all(ENHANCED_REGEX).alias('match')
        )
        .explode('match')
        .drop_nulls('match')
        .with_columns(
            pl.col('match')
            .str.replace_all(r'\s+', '')
            .str.replace(r'[^\w\.\-]+$', '')
            .str.replace(r'(?i)^PDB', '')
            .alias('dataset_id')
        )
        # æ”¹è¿›ï¼šæ›´ä¸¥æ ¼çš„è¿‡æ»¤
        .filter(pl.col('dataset_id').str.len_chars() >= 4)  
        .filter(pl.col('dataset_id').str.len_chars() <= 30)  # å‡�å°‘ä¸Šé™�
        # æ–°å¢�ï¼šç§»é™¤æ˜�æ˜¾çš„æ–‡æœ¬ç‰‡æ®µ
        .filter(~pl.col('dataset_id').str.contains(r'(?i)(abstract|introduction|conclusion|method|result|discussion|figure|table|supplementary|reference)'))
        .filter(~pl.col('dataset_id').str.contains(r'(?i)(university|department|center|college|institute)'))
        # ç§»é™¤æ˜�æ˜¾çš„é‡�å¤�æ¨¡å¼�ï¼ˆç”¨ç®€å�•æ–¹å¼�ï¼‰
        .filter(~pl.col('dataset_id').str.contains(r'aaaa|bbbb|cccc|dddd|1111|2222|3333|4444|5555|6666|7777|8888|9999|0000'))
        .group_by('article_id', 'dataset_id')
        .agg('match')
    )
    
    # å�ˆå¹¶ç»“æ�œ
    all_dfs = [doi_df, acc_df]
    df = pl.concat([d for d in all_dfs if d.height > 0], how='vertical')
    
    # æ”¹è¿›çš„æ¸…ç�†
    bad_ids = [DOI_LINK + e for e in ['10.5061/dryad', '10.5281/zenodo', '10.6073/pasta']]
    
    df = (
        df.unique(['article_id', 'dataset_id'])
        # ç§»é™¤ä¸�article_idé‡�å� çš„dataset_id
        .filter(~pl.col('article_id').str.replace('_','/').str.contains(pl.col('dataset_id').str.split(DOI_LINK).list.last().str.escape_regex()))
        .filter(~pl.col('dataset_id').str.contains(pl.col('article_id').str.replace('_','/').str.escape_regex()))
        .filter(~pl.col('dataset_id').str.contains('figshare', literal=True))
        .filter(~pl.col('dataset_id').is_in(bad_ids))
        .with_columns(pl.col('match').list.unique())
    )
    
    return df

def get_context_window(text: str, substring: str, window: int = 200) -> str:
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
        windows.append(get_context_window(text, match_ids[0], window=200))
    return df.with_columns(pl.Series('window', windows)).select('article_id', 'dataset_id', 'window')

def main():
    parse_dir = '/tmp/test_parse' if IS_KAGGLE_SUBMISSION else '/tmp/train_parse'
    
    text_df = get_df(parse_dir)
    df = get_splits(text_df)
    df = tidy_extraction(df)
    df = get_window_df(text_df, df)
    df.write_parquet('/tmp/extracted.parquet')
    
    df = assume_type(df)
    df.select(['article_id', 'dataset_id', 'type']).with_row_index(name='row_id').write_csv('/kaggle/working/submission.csv')
    
    if not IS_KAGGLE_SUBMISSION:
        print("=== ç²¾ç¡®åº¦ä¼˜åŒ–ç»“æ�œ ===")
        results = evaluate(df)
        for r in results:
            l.info(r)
            print(r)
        print("\n=== æŒ‰ç±»å�‹è¯„ä¼° ===")  
        results = evaluate(df, on=['article_id', 'dataset_id', 'type'])
        for r in results:
            l.info(r)
            print(r)

if __name__=='__main__':
    main()


%%writefile /tmp/src/llm_validate.py
import polars as pl
import os
from helpers import *

l = get_logger()

# ä¼˜åŒ–çš„æ��ç¤ºè¯�
SYS_PROMPT_CLASSIFY_DOI = """You are a DOI/accession classifier. Given text with an identifier, output ONLY:
A = Data (repository-stored research data)
B = Literature (journal article, book, protocol, preprint)

PRIORITY RULES:
1. ALWAYS classify as A (Data) if DOI starts with:
10.5061 (Dryad) 10.5281 (Zenodo) 10.6084 (Figshare) 10.17632 (Mendeley Data)
10.7910/DVN (Dataverse) 10.1594/PANGAEA (PANGAEA) 10.15468 (GBIF)

2. ALWAYS classify as A if accession starts with:
SRA/SRP/SRR/ERR/DRR (sequencing data) PRJNA/PRJEB/PRJDB (BioProject)
PXD (ProteomeXchange) E-MTAB/E-GEOD (ArrayExpress) GSE/GSM (GEO)
SAMN/SAMEA (BioSample) EMPIAR- (electron microscopy)

3. Context keywords indicating A (Data):
- "deposited in/at", "uploaded to", "archived at"
- "data repository", "data archive" 
- "available at [repository name]"
- "raw data", "dataset", "data set"

4. ALWAYS classify as B if DOI starts with publisher prefixes:
10.1038, 10.1007, 10.1126, 10.1016, 10.1101, 10.1021, 10.1093, 10.1080, 10.1111, 10.1371

EXAMPLES:
"Raw sequencing data deposited in SRA under PRJNA765432" â†’ A
"As described in Nature (10.1038/s41586-021-03819-2)" â†’ B
"Data available at Zenodo (10.5281/zenodo.987654)" â†’ A
"Following protocol from bioRxiv (10.1101/2021.05.01.123456)" â†’ B

Output: A or B only""".strip()

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
    
    llm = vllm.LLM(
        model_path,
        quantization='awq',
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        dtype="half",
        enforce_eager=True,
        max_model_len=2048,
        disable_log_stats=True,
        disable_custom_all_reduce=True,
        enable_prefix_caching=True,
        task='generate'
    )
    
    tokenizer = llm.get_tokenizer()
    df = build_df()
    df = build_prompt(tokenizer, df)
    prompts = df['prompt'].to_list()
    
    mclp = MultipleChoiceLogitsProcessor(tokenizer, choices=["A", "B"])
    
    outputs = llm.generate(
        prompts,
        vllm.SamplingParams(
            seed=42,
            temperature=0,
            skip_special_tokens=True,
            max_tokens=1,
            logits_processors=[mclp],
            logprobs=len(mclp.choices)
        ),
        use_tqdm=True
    )
    
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
        for r in results:
            l.info(r)
        results = evaluate(df, on=['article_id', 'dataset_id', 'type'])
        for r in results:
            l.info(r)


%%writefile /tmp/src/post_filter.py
import polars as pl
from helpers import *

l = get_logger()

# æ•°æ�®ä»“åº“å‰�ç¼€ï¼ˆæ­£å�‘è¿‡æ»¤ï¼‰
DATA_PREFIXES = [
    "10.5061", "10.5281", "10.17632", "10.1594", "10.15468", "10.17882", 
    "10.7937", "10.7910", "10.6073", "10.6084", "10.3886", "10.3334", 
    "10.4121", "10.5066", "10.5067", "10.18150", "10.25377", "10.25387", 
    "10.23642", "10.24381", "10.22033", "10.21233", "10.7289", "10.5255", 
    "10.6019"
]

# æ–‡çŒ®å‡ºç‰ˆå•†å‰�ç¼€ï¼ˆè´Ÿå�‘è¿‡æ»¤ï¼‰
PAPER_PREFIXES = [
    "10.1007", "10.1002", "10.1016", "10.1021", "10.1038", "10.1056", 
    "10.1073", "10.1080", "10.1093", "10.1101", "10.1186", "10.1371", 
    "10.1111", "10.5194", "10.3390", "10.1126", "10.1145", "10.1177"
]

# ä¸Šä¸‹æ–‡å…³é”®è¯�
CONTEXT_RE = r"(?i)\b(data(?:set|base)?|repository|archive|deposited|available|supplementary\s+data|raw(?:\s+data)?|uploaded|hosted|stored|accession|NCBI|SRA|GEO|PRIDE|ArrayExpress|BioProject|Zenodo|Dryad|Figshare|Mendeley)\b"

def is_data_prefix(col: str = "dataset_id") -> pl.Expr:
    expr = pl.lit(False)
    for p in DATA_PREFIXES:
        expr = expr | pl.col(col).str.starts_with(f"{DOI_LINK}{p}")
    return expr

def is_paper_prefix(col: str = "dataset_id") -> pl.Expr:
    expr = pl.lit(False)
    for p in PAPER_PREFIXES:
        expr = expr | pl.col(col).str.starts_with(f"{DOI_LINK}{p}")
    return expr

def main():
    sub = pl.read_csv("/kaggle/working/submission.csv")
    if "row_id" in sub.columns:
        sub = sub.drop("row_id")
    win = pl.read_parquet("/tmp/extracted.parquet").select("article_id", "dataset_id", "window")
    
    # åˆ†ç¦»DOIå’ŒAccession
    doi_rows = sub.filter(is_doi_link("dataset_id")).join(win, on=["article_id", "dataset_id"], how="left")
    acc_rows = sub.filter(~is_doi_link("dataset_id"))
    
    # è¿‡æ»¤ç­–ç•¥
    keep_mask = (
        is_data_prefix("dataset_id") | 
        (~is_paper_prefix("dataset_id") & doi_rows["window"].fill_null("").str.contains(CONTEXT_RE))
    )
    
    kept_doi = doi_rows.filter(keep_mask).select("article_id", "dataset_id", "type")
    final = pl.concat([kept_doi, acc_rows.select("article_id", "dataset_id", "type")])
    
    # å�»é‡�
    final = final.unique(["article_id", "dataset_id"])
    
    if not IS_KAGGLE_SUBMISSION:
        for r in evaluate(final):
            l.info(r)
        for r in evaluate(final, on=["article_id", "dataset_id", "type"]):
            l.info(r)
    
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


%%writefile /tmp/src/ensemble_llm_validate.py
import polars as pl
from helpers import *
import os
import re

l = get_logger()

def ensemble_llm_validation():
    """é›†æˆ�LLMç­–ç•¥ - å¤šç­–ç•¥+å¤šæ¸©åº¦+æŠ•ç¥¨æœºåˆ¶"""
    
    os.environ["VLLM_USE_V1"] = "0"
    import vllm
    from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
    
    # 1. åŠ è½½åŸºç¡€æ•°æ�®
    df = pl.read_parquet('/tmp/extracted.parquet')
    print(f"ğŸ�¯ å¾…åˆ†ç±»æ•°æ�®: {df.height}")
    
    # 2. ä¸‰ç§�ä¸�å�Œçš„æ��ç¤ºç­–ç•¥
    PROMPTS = {
        "conservative": """You are a strict academic data citation validator. Classify as:
A = Primary research data (original datasets in repositories)
B = Secondary literature reference (papers, methods, protocols)

ULTRA-STRICT RULES for A:
- Must have explicit data language: "data deposited", "accession number", "database ID"
- Clear data repository context: Dryad, Zenodo, NCBI, SRA
- Direct data access information

Always B if:
- Any author names or "et al."
- "described in", "according to", "method from", "following protocol"
- Journal citation format
- Method/software references

When ANY doubt exists, choose B.
Output: A or B only""",
        
        "balanced": """You are an expert academic data citation classifier. Classify as:
A = Primary research data (original datasets)
B = Secondary reference (literature, methods)

Key indicators for A:
- Database accession numbers (SRR*, GSE*, PRJNA*, etc.)
- Data repository DOIs (10.5061, 10.5281, 10.17632)
- "data available at", "deposited in", "supplementary data"
- Clear dataset descriptions

Key indicators for B:
- Author citations with "et al."
- Method references: "described by", "following"
- Journal publisher DOIs (10.1038, 10.1016, 10.1007)
- Protocol or software citations

Analyze context carefully for data vs literature signals.
Output: A or B only""",
        
        "data_focused": """You are a research data discovery specialist. Classify as:
A = Primary research data (datasets, databases)
B = Secondary reference (papers, citations)

Prioritize identifying research data:
A indicators:
- Any database identifiers or accession numbers
- Data repository mentions
- Supplementary data files
- Raw data, sequence data, experimental data
- Data availability statements

B indicators:
- Clear literature citations
- Author name patterns
- Method/protocol references
- Journal article formats

When data-related context exists, favor A unless clearly literature.
Output: A or B only"""
    }
    
    # 3. åˆ�å§‹åŒ–LLM
    model_path = "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"
    
    llm = vllm.LLM(
        model_path,
        quantization='awq',
        tensor_parallel_size=2,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        dtype="half",
        enforce_eager=True,
        max_model_len=1536,
        disable_log_stats=True,
        disable_custom_all_reduce=True,
        enable_prefix_caching=True
    )
    
    tokenizer = llm.get_tokenizer()
    mclp = MultipleChoiceLogitsProcessor(tokenizer, choices=["A", "B"])
    
    # 4. å¤šç­–ç•¥æ�¨ç�†
    all_predictions = {}
    
    for strategy_name, sys_prompt in PROMPTS.items():
        print(f"\n=== ğŸ¤– {strategy_name.upper()} ç­–ç•¥æ�¨ç�† ===")
        
        # æ�„å»ºè¯¥ç­–ç•¥çš„æ‰€æœ‰æ��ç¤º
        prompts = []
        for row in df.select('dataset_id', 'window').rows():
            dataset_id, window = row
            
            # å¢�å¼ºä¸Šä¸‹æ–‡ä¿¡æ�¯
            enhanced_context = f"Dataset ID: {dataset_id}\n"
            enhanced_context += f"Context: {window}\n"
            
            # æ·»åŠ IDç±»å�‹æ��ç¤º
            if dataset_id.startswith('https://doi.org/'):
                enhanced_context += f"Type: DOI\n"
            else:
                enhanced_context += f"Type: Accession\n"
            
            enhanced_context += "Classification needed:"
            
            messages = [
                {'role': 'system', 'content': sys_prompt},
                {'role': 'user', 'content': enhanced_context}
            ]
            
            prompts.append(tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            ))
        
        # å¤šæ¸©åº¦æ�¨ç�†ï¼ˆä¸ºbalancedç­–ç•¥å¢�åŠ éš�æœºæ€§ï¼‰
        if strategy_name == "balanced":
            temperatures = [0.0, 0.1, 0.2]
            seeds = [42, 123, 456]
        else:
            temperatures = [0.0]
            seeds = [42]
        
        strategy_results = []
        
        for temp, seed in zip(temperatures, seeds):
            print(f"  æ¸©åº¦: {temp}, ç§�å­�: {seed}")
            
            outputs = llm.generate(
                prompts,
                vllm.SamplingParams(
                    temperature=temp,
                    seed=seed,
                    skip_special_tokens=True,
                    max_tokens=1,
                    logits_processors=[mclp],
                    logprobs=2
                ),
                use_tqdm=True
            )
            
            # æ��å�–é¢„æµ‹å’Œç½®ä¿¡åº¦
            predictions = []
            confidences = []
            
            for output in outputs:
                logprobs = {lp.decoded_token: lp.logprob 
                           for lp in output.outputs[0].logprobs[0].values()}
                
                pred = max(logprobs, key=logprobs.get)
                conf = max(logprobs.values())
                
                predictions.append(pred)
                confidences.append(conf)
            
            strategy_results.append({
                'predictions': predictions,
                'confidences': confidences,
                'temperature': temp
            })
        
        all_predictions[strategy_name] = strategy_results
        
        # ç»Ÿè®¡è¯¥ç­–ç•¥ç»“æ�œ
        total_a = sum(sum(1 for p in result['predictions'] if p == 'A') for result in strategy_results)
        total_predictions = sum(len(result['predictions']) for result in strategy_results)
        print(f"  è¯¥ç­–ç•¥Aç±»é¢„æµ‹: {total_a}/{total_predictions} ({total_a/total_predictions*100:.1f}%)")
    
    # 5. æ™ºèƒ½é›†æˆ�æŠ•ç¥¨ç®—æ³•
    print(f"\n=== ğŸ—³ï¸� æ™ºèƒ½é›†æˆ�æŠ•ç¥¨ ===")
    
    final_predictions = []
    high_confidence_count = 0
    
    for i in range(df.height):
        # æ”¶é›†æ‰€æœ‰æŠ•ç¥¨
        votes = {'A': [], 'B': []}
        
        for strategy_name, strategy_results in all_predictions.items():
            # ç­–ç•¥æ�ƒé‡�
            strategy_weight = {
                'conservative': 1.2,  # ä¿�å®ˆç­–ç•¥æ�ƒé‡�é«˜ï¼ˆç²¾ç¡®åº¦ä¼˜å…ˆï¼‰
                'balanced': 1.0,      # å¹³è¡¡ç­–ç•¥æ ‡å‡†æ�ƒé‡�
                'data_focused': 0.8   # æ•°æ�®ä¼˜å…ˆç­–ç•¥æ�ƒé‡�ç¨�ä½�
            }[strategy_name]
            
            for result in strategy_results:
                pred = result['predictions'][i]
                conf = result['confidences'][i]
                temp = result['temperature']
                
                # è®¡ç®—åŠ æ�ƒåˆ†æ•°
                base_weight = strategy_weight
                confidence_bonus = abs(conf) * 0.5  # é«˜ç½®ä¿¡åº¦å¥–åŠ±
                temperature_penalty = temp * 0.2    # é«˜æ¸©åº¦æƒ©ç½š
                
                final_weight = base_weight + confidence_bonus - temperature_penalty
                
                votes[pred].append(final_weight)
        
        # è®¡ç®—åŠ æ�ƒå¾—åˆ†
        score_a = sum(votes['A'])
        score_b = sum(votes['B'])
        
        # å†³ç­–é€»è¾‘
        if score_a > score_b * 1.3:  # Aéœ€è¦�æ›´å¼ºçš„è¯�æ�®
            final_pred = 'A'
            if score_a > score_b * 2.0:  # é«˜ç½®ä¿¡åº¦
                high_confidence_count += 1
        elif score_b > score_a:
            final_pred = 'B'
        else:
            # å¹³ç¥¨æ—¶ä¿�å®ˆé€‰æ‹©B
            final_pred = 'B'
        
        final_predictions.append(final_pred)
    
    # 6. ç»Ÿè®¡å’Œåˆ†æ��
    a_count = sum(1 for p in final_predictions if p == 'A')
    print(f"æœ€ç»ˆAç±»é¢„æµ‹: {a_count}/{len(final_predictions)} ({a_count/len(final_predictions)*100:.1f}%)")
    print(f"é«˜ç½®ä¿¡åº¦Aç±»: {high_confidence_count}")
    
    # 7. ç”Ÿæˆ�æœ€ç»ˆç»“æ�œ
    result_df = df.with_columns(
        pl.Series('final_prediction', final_predictions)
    ).filter(
        pl.col('final_prediction') == 'A'
    ).select(['article_id', 'dataset_id']).with_columns(
        pl.lit('Primary').alias('type')
    )
    
    print(f"ğŸ�¯ é›†æˆ�LLMæœ€ç»ˆé¢„æµ‹Primary: {result_df.height}")
    
    # ä¿�å­˜ç»“æ�œ
    result_df.with_row_index('row_id').write_csv('/kaggle/working/submission.csv')
    
    # 8. å¦‚æ�œåœ¨è®­ç»ƒç�¯å¢ƒï¼Œè¯„ä¼°æ€§èƒ½
    if not IS_KAGGLE_SUBMISSION:
        print(f"\n=== ğŸ“Š é›†æˆ�LLMæ€§èƒ½è¯„ä¼° ===")
        results = evaluate(result_df)
        for r in results:
            l.info(r)
            print(r)
        
        # è¯¦ç»†åˆ†æ��
        f1 = float(results[0].split('f1: ')[1].split(' ')[0])
        precision = float(results[0].split('precision: ')[1].split(' ')[0])
        recall = float(results[0].split('recall: ')[1].split(' ')[0])
        
        print(f"\nğŸ�¯ é›†æˆ�LLMç»“æ�œ:")
        print(f"F1: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        
        if f1 > 0.55:
            print("ğŸ�‰ é›†æˆ�ç­–ç•¥æˆ�åŠŸè¶…è¶Šbaseline!")
        elif f1 > 0.52:
            print("âœ… é›†æˆ�ç­–ç•¥è¡¨ç�°è‰¯å¥½")
        else:
            print("âš ï¸� é›†æˆ�ç­–ç•¥éœ€è¦�è¿›ä¸€æ­¥è°ƒä¼˜")

if __name__ == '__main__':
    ensemble_llm_validation()


%cd /tmp
!LOG_LEVEL=INFO python src/parse.py /tmp/train_parse
! python src/check_parse.py
! python src/getid.py
! python src/ensemble_llm_validate.py
! python src/post_filter.py
! grep "f1:" /tmp/logs/project.log


# æ£€æŸ¥submission.csvæ˜¯å�¦æ­£ç¡®ç”Ÿæˆ�
import polars as pl
import os

print("=== æ£€æŸ¥submission.csv ===")

# 1. æ–‡ä»¶æ˜¯å�¦å­˜åœ¨
submission_path = '/kaggle/working/submission.csv'
if os.path.exists(submission_path):
    print("âœ… submission.csvå­˜åœ¨")
    
    # 2. è¯»å�–å’Œæ£€æŸ¥æ ¼å¼�
    df = pl.read_csv(submission_path)
    print(f"æ–‡ä»¶å¤§å°�: {df.height} è¡Œ")
    print(f"åˆ—å��: {df.columns}")
    print(f"å‰�5è¡Œé¢„è§ˆ:")
    print(df.head())
    
    # 3. æ£€æŸ¥å¿…éœ€åˆ—
    required_cols = ['row_id', 'article_id', 'dataset_id', 'type']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"â�Œ ç¼ºå°‘å¿…éœ€åˆ—: {missing_cols}")
    else:
        print("âœ… æ‰€æœ‰å¿…éœ€åˆ—éƒ½å­˜åœ¨")
    
    # 4. æ£€æŸ¥æ•°æ�®ç±»å�‹
    print(f"\n=== æ•°æ�®ç±»å�‹æ£€æŸ¥ ===")
    print(f"typeåˆ—å”¯ä¸€å€¼: {df['type'].unique()}")
    print(f"typeåˆ—åˆ†å¸ƒ: {df['type'].value_counts()}")
    
    # 5. æ£€æŸ¥æ˜¯å�¦æœ‰ç©ºå€¼
    null_counts = df.null_count()
    print(f"ç©ºå€¼ç»Ÿè®¡: {null_counts}")
    
    # 6. æ£€æŸ¥row_idæ˜¯å�¦è¿�ç»­
    if 'row_id' in df.columns:
        print(f"row_idèŒƒå›´: {df['row_id'].min()} åˆ° {df['row_id'].max()}")
        expected_count = df['row_id'].max() + 1  # å�‡è®¾ä»�0å¼€å§‹
        if df.height == expected_count:
            print("âœ… row_idè¿�ç»­")
        else:
            print(f"â�Œ row_idä¸�è¿�ç»­: æœŸæœ›{expected_count}è¡Œï¼Œå®�é™…{df.height}è¡Œ")
    
else:
    print("â�Œ submission.csvä¸�å­˜åœ¨ï¼�")
    print("æ£€æŸ¥ /kaggle/working/ ç›®å½•:")
    if os.path.exists('/kaggle/working/'):
        files = os.listdir('/kaggle/working/')
        print(f"ç�°æœ‰æ–‡ä»¶: {files}")
    else:
        print("â�Œ /kaggle/working/ ç›®å½•ä¸�å­˜åœ¨ï¼�")




