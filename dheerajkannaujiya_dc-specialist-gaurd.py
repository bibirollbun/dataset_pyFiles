! uv pip install --system --no-index --find-links='/kaggle/input/all-whl-lib/whls' 'vllm' 'triton' 'logits-processor-zoo' 'numpy<2'
import logging, os, inspect, re, gc
from pathlib import Path
import polars as pl
import torch
from bs4 import BeautifulSoup
from typing import List
try:
    import vllm
    from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
    VLLM_AVAILABLE = True
except ImportError: VLLM_AVAILABLE = False

# ==============================================================================
# 1. CONFIG & LOGGING
# ==============================================================================
IS_KAGGLE_SUBMISSION = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
COMP_DIR = Path('/kaggle/input/make-data-count-finding-data-references')
XML_DIR = COMP_DIR / ('test' if IS_KAGGLE_SUBMISSION else 'train') / 'XML'
WORKING_DIR = Path('/kaggle/working/')
DOI_LINK = 'https://doi.org/'
# ... (poora config aur logging wala code)

LOG_FORMAT = "%(levelname)s %(asctime)s  [%(filename)s:%(lineno)d - %(funcName)s()] %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper() if not IS_KAGGLE_SUBMISSION else "INFO"
LOG_FILE_PATH = WORKING_DIR / "project.log"
LOG_DIR = LOG_FILE_PATH.parent
LOG_DIR.mkdir(parents=True, exist_ok=True)

def get_logger(name=None):
    if name is None:
        frame = inspect.currentframe()
        name = frame.f_back.f_globals.get("__name__", "__main__") if frame and frame.f_back else "__main__"
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

l = get_logger(__name__)

# ==============================================================================
# 2. XML-ONLY PARSING
# ==============================================================================
def xml_to_text_df(xml_dir: Path) -> pl.DataFrame:
    l.info(f"Parsing all XML files from {xml_dir}...")
    records = []
    for xml_file in xml_dir.glob("*.xml"):
        try:
            with open(xml_file, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'xml')
            text = soup.get_text(separator=' ', strip=True)
            if text: # Sirf un files ko rakho jinmein text hai
                records.append({'article_id': xml_file.stem, 'text': text})
        except Exception as e:
            l.warning(f"Could not parse {xml_file.name}: {e}")
            continue
    l.info(f"Successfully parsed {len(records)} XML files.")
    return pl.DataFrame(records) if records else pl.DataFrame()

# ==============================================================================
# 3. HIGH-RECALL EXTRACTION ("Loose Detective")
# ==============================================================================
def extract_from_xml(df: pl.DataFrame) -> pl.DataFrame:
    l.info("Extracting all potential ID candidates from XML text...")
    # Yeh humara v5.0 wala, sabse powerful "loose" regex hai
    # doi_patterns = [ r'10\.\d{4,9}/[^\s"<>()]+' ]    
    # master_regex = '|'.join(doi_patterns)
    
    return (
        df.select(["article_id", "text"])
        .with_columns(pl.col('text').str.extract_all(r'(?i)\b(?:CHEMBL\d+|E-GEOD-\d+|E-PROT-\d+|EMPIAR-\d+|ENSBTAG\d+|ENSOARG\d+|EPI_ISL_\d{5,}|EPI\d{6,7}|HPA\d+|CP\d{6}|IPR\d{6}|PF\d{5}|KX\d{6}|K0\d{4}|PRJNA\d+|PXD\d+|SAMN\d+|dryad\s*\.\s*[^\s"<>]+|pasta\s*/\s*[^\s"<>])').alias('match_list'))
        .explode('match_list').drop_nulls('match_list')
        .rename({'match_list': 'candidate_id'})
        .group_by("article_id", "candidate_id").agg(pl.first("text"))
    )

# ==============================================================================
# 4. THE "LLM JUDGE"
# ==============================================================================
def get_context_paragraph(text: str, substring: str) -> str:
    """
    Finds the substring and returns the entire paragraph it appears in.
    A paragraph is assumed to be separated by double newlines ('\n\n') or be a long single line.
    """
    try:
        if not isinstance(substring, str) or not substring:
            return ""
        paragraphs = text.split('\n\n')
        for p in paragraphs:
            if substring.lower() in p.lower():
                clean_paragraph = " ".join(p.replace('\n', ' ').split())
                return clean_paragraph[:1000] # Max 1000 characters
        
        lines = text.split('\n')
        for l in lines:
            if substring.lower() in l.lower():
                return l.strip()

        return "" # Agar kahin na mile
    except Exception:
        return ""


# ==============================================================================
# ---> 'run_llm_classification' FUNCTION (COMPLETE MISTRAL 7B VERSION) <---
# ==============================================================================
SYS_PROMPT_CLASSIFY_TYPE = """
You are an expert assistant analyzing scientific texts. You are given a text snippet containing a proven DATASET identifier.
Your task is to classify it as Primary or Secondary based ONLY on the text provided.
- 'Primary': Respond with this if the authors CREATED or GENERATED this data FOR THIS STUDY. Keywords: "we collected", "our data", "data are available at".
- 'Secondary': Respond with this if the authors REUSED, REFERENCED, or ANALYZED data that ALREADY EXISTED. Keywords: "obtained from", "retrieved from", "from a previous study".
Respond with a single word: Primary or Secondary.
""".strip()

def run_llm_classification(df_to_classify: pl.DataFrame):
    if not VLLM_AVAILABLE or df_to_classify.is_empty():
        l.info("LLM not available or no data to classify. Returning empty.")
        return pl.DataFrame()

    os.environ["VLLM_USE_V1"] = "0"
    model_path = "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"
    l.info('Loading vLLM model...')
    llm = vllm.LLM(
        model_path, quantization='awq', tensor_parallel_size=2,
        gpu_memory_utilization=0.9, trust_remote_code=True, dtype="half",
        enforce_eager=True, max_model_len=2048, disable_log_stats=True,
        disable_custom_all_reduce=True, enable_prefix_caching=True, task='generate',
    )
    tokenizer = llm.get_tokenizer()

    l.info(f"Starting LLM classification for {len(df_to_classify)} IDs...")
    # 'df_to_classify' ke andar 'paragraph_context' column pehle se hai
    prompts = [
        tokenizer.apply_chat_template(
            [{'role': 'system', 'content': SYS_PROMPT_CLASSIFY_TYPE}, {'role': 'user', 'content': w}],
            tokenize=False, add_generation_prompt=True
        ) for w in df_to_classify['paragraph_context']
    ]
    
    mclp = MultipleChoiceLogitsProcessor(tokenizer, choices=["Primary", "Secondary"])
    
    outputs = llm.generate(
        prompts, 
        vllm.SamplingParams(seed=777,
                            temperature=0.1,
                            skip_special_tokens=True, 
                            max_tokens=1,
                            logits_processors=[mclp], 
                            logprobs=len(mclp.choices)),
                 use_tqdm=True
                )

    
    final_types = [o.outputs[0].text.strip().capitalize() for o in outputs]
    
    # Final classified DataFrame return karo
    classified_df = df_to_classify.with_columns(pl.Series("type", final_types))
    return classified_df.filter(pl.col("type").is_in(["Primary", "Secondary"]))
# ==============================================================================
# 5. MAIN WORKFLOW ("Mr. Aggressive")
# ==============================================================================
if __name__ == "__main__":
    l.info("Starting the 'Mr. Aggressive' (XML-only) workflow...")
    # --- Step 1: Parse ALL XMLs ---
    xml_text_df = xml_to_text_df(XML_DIR)
    
    if not xml_text_df.is_empty():
        
        all_candidates = extract_from_xml(xml_text_df)
        del xml_text_df
        gc.collect()

        # --- Step 3: Get Paragraph Context ---
        context_df = all_candidates.with_columns(
            pl.struct(["text", "candidate_id"]).map_elements(
                lambda x: get_context_paragraph(x["text"], x["candidate_id"]), return_dtype=pl.Utf8
            ).alias("paragraph_context")
        )

        classified_df = run_llm_classification(context_df)

        if not classified_df.is_empty():
            # --- Step 5: Final Formatting & Sanitizing ---
            final_df = classified_df.with_columns(
                pl.when(pl.col("candidate_id").str.contains(r"^10\."))
                .then(DOI_LINK + pl.col("candidate_id").str.to_lowercase())
                .otherwise(pl.col("candidate_id"))
                .alias("dataset_id")
            )
            # Sanitizer (aakhir ke punctuation ko saaf karna)
            sanitized_df = final_df.with_columns(
                pl.col("dataset_id").str.replace_all(r"[;\.]$", "").alias("dataset_id")
            )
            final_submission = sanitized_df.unique(subset=['article_id', 'dataset_id', 'type'], keep='first')
            
            # Submission file ka naam alag rakhein
            final_submission.select(['article_id', 'dataset_id', 'type']).with_row_index(name='row_id').write_csv("submission.csv")
            l.info("submission.csv created successfully.")
            
    else:
        # Agar koi XML file hi nahi hai, to khaali submission banayein
        pl.DataFrame({'row_id': [], 'article_id': [], 'dataset_id': [], 'type': []}).write_csv("submission.csv")
        l.info("No XML files found. Empty submission.csv created.")

