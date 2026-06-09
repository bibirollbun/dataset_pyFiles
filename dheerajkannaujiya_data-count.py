# # =================================================================================
# # FINAL SUBMISSION PIPELINE (DAY 2 MISSION)
# # =================================================================================
# ! uv pip install --no-index --find-links='/kaggle/input/all-whl-lib/whls' 'pymupdf' 'vllm' 'triton' 'logits-processor-zoo' 'numpy<2'
# import os
# from pathlib import Path
# import polars as pl
# import re
# import pymupdf
# import lxml.etree as ET
# import gc

# LIMIT_ARTICLES = None

# # --- Environment and Path Configuration ---
# IS_KAGGLE_ENV = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
# IS_KAGGLE_SUBMISSION = IS_KAGGLE_ENV and os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == 'Batch'

# if IS_KAGGLE_ENV:
#     COMP_DIR = Path('/kaggle/input/make-data-count-finding-data-references')
#     VLLM_MODEL_PATH = "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"
#     WORKING_DIR = Path('/kaggle/working/')
#     BLACKLIST_PATH = Path('/kaggle/input/blocklist-fp/false_positive_blacklist.csv')
#     TRUTH_FINDER_REPORT_PATH = WORKING_DIR / 'true_positive_analysis_v3_xml_first.csv'
#     if IS_KAGGLE_SUBMISSION:
#         LIMIT_ARTICLES = None
# else:
#     COMP_DIR = Path('./make-data-count-finding-data-references')
#     VLLM_MODEL_PATH = "Qwen/Qwen2-32B-Instruct-AWQ"
#     WORKING_DIR = Path('./.working/')
#     BLACKLIST_PATH = WORKING_DIR / 'false_positive_blacklist.csv'

# DATA_DIR = COMP_DIR / ('test' if IS_KAGGLE_SUBMISSION else 'train')
# DOI_LINK = 'https://doi.org/'
# WORKING_DIR.mkdir(parents=True, exist_ok=True)

# try:
#     import vllm
#     from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
#     VLLM_AVAILABLE = True
# except ImportError:
#     VLLM_AVAILABLE = False
#     print("WARNING: vLLM is not installed.")


# # --- LLM PROMPTS (UPGRADED) ---
# SYS_PROMPT_SHERLOCK_FP_HUNTER = """
# You are Sherlock Holmes, a meticulous detective for scientific papers. Determine if an identifier is a true DATASET or something else. Be SKEPTICAL but FAIR â€“ prefer VALID if context suggests data.
# - Reference sections often have citations (B), but if "data" or "dataset" mentioned, it's A.
# - Software/tools are C unless explicitly data.
# - If context mentions "data", "dataset", "repository", "deposited", "accession", or known banks (GEO, SRA, PDB, Dryad, Zenodo), prefer A.
# Examples:
# - Context: "Data deposited in GEO under accession GSE12345." -> A (VALID DATASET)
# - Context: "Our dataset available at Dryad 10.5061/dryad.abc." -> A
# - Context: "Accession SAMN123 in BioSample." -> A
# - Context: "PDB ID 1ABC used for analysis." -> A (if data context)
# - Context: "Figure 1 shows PDB ID 1ABC." -> C (OTHER/JUNK)
# - Context: "Cited in vol. 10, pp. 123, DOI 10.1234/paper." -> B (PAPER CITATION)
# - Context: "Software MATLAB v9.0 used." -> C
# - Context: "The accession for our data is PXD00123." -> A
# - Context: "EPI_ISL_12345 from GISAID database." -> A
# - Context: "IPR006 in InterPro entry." -> A
# Classify into ONE category:
# A) **VALID DATASET**: Likely research data, context supports.
# B) **PAPER CITATION**: Citation to paper/article/book.
# C) **OTHER/JUNK**: Anything else. Use C only if NO doubt it's not data.
# Respond with ONLY ONE letter: A, B, or C.
# """

# SYS_PROMPT_CLASSIFY_TYPE = """
# You are an expert research assistant. Read the following text. It mentions a dataset. Your task is to classify if the dataset was CREATED for this study (Primary) or REUSED from another source (Secondary).
# Context: "{context}"
# Classify the dataset as: A) Primary or B) Secondary. Respond with only one letter.
# """.strip()

# # =================================================================================
# # HELPER FUNCTIONS
# # =================================================================================
# def get_text_pdf_only(article_id: str) -> str:
#     print(f"  - Extracting text from PDF: {article_id}")
#     pdf_path = DATA_DIR / 'PDF' / f"{article_id}.pdf"
#     if not pdf_path.exists(): return ""
#     try:
#         with pymupdf.open(pdf_path) as doc: text = "".join(page.get_text("text", sort=True) for page in doc)
#     except Exception: return ""
#     text = text.replace('-\n', '')
#     text = re.sub(r'\s+', ' ', text.replace('\n', ' '))
#     return text.strip()

# def prepare_text_dataframe() -> pl.DataFrame:
#     print(f"Preparing text DataFrame from '{DATA_DIR.name}' (PDF-Only)...")
#     article_files = sorted([p.stem for p in (DATA_DIR / 'PDF').glob("*.pdf")])
#     if LIMIT_ARTICLES and not IS_KAGGLE_SUBMISSION:
#         print(f"WARNING: LIMIT is active. Processing only {LIMIT_ARTICLES} articles.")
#         article_files = article_files[:LIMIT_ARTICLES]
#     records = [{'article_id': article_id, 'text': get_text_pdf_only(article_id)} for article_id in article_files]
#     df = pl.DataFrame(records).filter(pl.col('text').str.len_chars() > 0)
#     print(f"Successfully prepared text for {df.height} articles.")
#     return df

# def supercharged_extraction_90_percent(df: pl.DataFrame) -> pl.DataFrame:
#     print("Running extraction with 90.90% Recall RegEx...")
    
#     generous_reg_ids =  r"""(?ix)
#     \b(?:
#         10\.\d{4,9}/[^\s"<>]+ |
#         CHEMBL\d+\b | E-GEOD-\d+\b | E-PROT-\d+\b | EMPIAR-\d+\b | E-MTAB-\d+\b |
#         ENSBTAG\d+\b | ENSOARG\d+\b | EPI_ISL_\d{5,}\b | EPI\d{6,7}\b | EPI\d{7,}\b |
#         HPA\d+\b | CP\d{6}\b | IPR\d{6}\b | PF\d{5}\b | KX\d{6}\b | K0\d{4}\b | PXD\d+\b |
#         (?:SRP|ERP|DRP|GSE|SRX|DRA)\d+\b | (?:PRJNA|PRJEB|PRJDB|PRJCA)\d+\b |
#         (?:SAMN|SAMEA|SAMD|SAMC)\d+\b | GCA_[0-9.]+\b |
#         ERS\d+\b | EGAS\d+\b | EGAD\d+\b | CRD\d+\b | syn\d+\b | idr\d+\b |
#         [A-Z]{4}\d{10,}\b | arXiv:\d{4}\.\d{4,5}\b | mcc\d{5}\b |

#         # <<< YEH HAI NAYA, SUPPORTED PATTERN 5VA1 JAISE IDs KE LIYE >>>
#         \b[A-Z0-9]{4}\b |

#         (?:PDB\sID:?)\s*(?:[A-Z0-9]{4})\b |
#         (?:CCDC|Deposition\sNumber.?:?|Deposition\sNumbers.?:?)\s*(?:\d{6,8})\b |
#         (?:GenBank|accession(?:s|\snumber)?s?:?)\s*(?:[A-Z]{2,4}\d{6,8}(?:\.\d+)?)\b |
#         (?:IUCr\sElectronic\sArchives\s\(Reference:\s*[A-Z]{2}\d{4}\)) |
        
#         https?://github\.com/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/? |
#         https?://gitlab\.[^/]+/[^\s"<>]+/?
#     )"""
#     generous_reg_ids_1 = r"""(?i)\b(?:CHEMBL\d+|E-GEOD-\d+|E-PROT-\d+|EMPIAR-\d+
#     |ENSBTAG\d+|ENSOARG\d+|EPI_ISL_\d{5,}|EPI\d{6,7}|HPA\d+|CP\d{6}|IPR\d{6}|
#     PF\d{5}|KX\d{6}|K0\d{4}|PRJNA\d+|PXD\d+|SAMN\d+|dryad\s*\.\s*[^\s"<>]+|pasta\s*/\s*[^\s"<>]|
#     E-MTAB-\d+|SRP\d+|ERP\d+|DRP\d+|SRX\d+|DRA\d+|PRJEB\d+|PRJDB\d+|PRJCA\d+|
#     SAMEA\d+|SAMD\d+|SAMC\d+|GCA_[0-9.]+|ERS\d+|EGAS\d+|EGAD\d+|CRD\d+|syn\d+|idr\d+|
#     [A-Z]{4}\d{8,10}(?:\.\d+)?|mcc\d{5}|[A-Z]{1,2}\d{5,8}(?:\.\d+)?|[A-Z]{2}_\d{6,9}(?:\.\d+)?|
#     GSE\d+|E-GEOD-\d+|PXD\d+|EPI_ISL_\d{5,}|EPI\d{6,7}|PF\d{5}|IPR\d{6}|HPA\d{5,6}|CAB\d{6}|
#     HGNC:\d{1,5}|rs\d+)"""
#     all_matches = df['text'].str.extract_all(f'{generous_reg_ids}|{generous_reg_ids_1}')
#     df = df.with_columns(all_matches.alias("matches"))
#     candidates_df = df.explode("matches").select([pl.col("article_id"), pl.col("matches").alias("dataset_id")]).unique()
#     candidates_df = candidates_df.with_columns(pl.col("dataset_id").str.strip_chars(" .,;()[]'\"").str.replace_all(r'\s+', ' ')).filter(pl.col('dataset_id').str.len_chars() > 3)
#     print(f"Master extraction found {candidates_df.height} total unique candidate IDs.")
#     return candidates_df

# def get_context_window_df(text_df: pl.DataFrame, ids_df: pl.DataFrame, window_size: int = 200) -> pl.DataFrame:
#     df = ids_df.join(text_df, on='article_id')
#     contexts = []
#     for row in df.iter_rows(named=True):
#         text, id_to_find_raw = row['text'], row['dataset_id']
#         id_to_find_norm = id_to_find_raw.replace(DOI_LINK, "").replace(" ", "")
#         index = text.lower().find(id_to_find_norm.lower())
#         if index != -1: contexts.append(text[max(0, index - window_size):min(len(text), index + len(id_to_find_norm) + window_size)])
#         else: contexts.append("")
#     df_with_context = df.with_columns(pl.Series("context", contexts, dtype=pl.Utf8))
#     return df_with_context.filter(pl.col("context").str.len_chars() > 0)

# def evaluate_and_log(predictions_df: pl.DataFrame, gt_df: pl.DataFrame, stage_name: str):
#     if gt_df.is_empty(): return
#     def normalize_for_join(col_name: str) -> pl.Expr:
#         return pl.col(col_name).str.replace("https://doi.org/", "").str.replace_all(r"[\s\-_:.]+", "").str.to_lowercase()
#     relevant_articles = predictions_df['article_id'].unique().to_list()
#     gt_subset = gt_df.filter(pl.col('article_id').is_in(relevant_articles))
#     hits = gt_subset.with_columns(normalize_for_join("dataset_id").alias("norm_id")).join(
#         predictions_df.with_columns(normalize_for_join("dataset_id").alias("norm_id")),
#         on=['article_id', 'norm_id'], how='inner'
#     )
#     tp = hits.select(['article_id', 'dataset_id']).n_unique()
#     fp = predictions_df.height - tp
#     fn = gt_subset.height - tp
#     f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
#     print("="*50 + f"\nSTATS FOR STAGE: '{stage_name}'\n" + f"  - True Positives (TP) : {tp}\n" + f"  - False Positives (FP): {fp}\n" + f"  - False Negatives (FN): {fn}\n" + f"  - F1 Score (ID only)  : {f1:.4f}\n" + "="*50)

# # <<< YEH HAI HUMARA NAYA, SUPER-SMART HEURISTIC FILTER V12 >>>

# def heuristic_filter_v4(context_df: pl.DataFrame) -> pl.DataFrame:
#     print("Running 'Ultimate FP Slayer v4' â€“ Destroy 66,000 FPs, Save TPs...")
    
#     # Positive keywords (at least 2 required in context for TP preservation)
#     positive_keywords = r'(?:data|dataset|database|accession|available|repository|deposited|code|supplementary|archive|supplemental|uploaded|archived|stored|hosted|shared|access code|data bank|data base|geo|sra|pdb|dryad|zenodo|figshare|pangaea|pride|bioproject|biosample)'
    
#     # Negative keywords (expanded from search: citations, figs, params, software, dates)
#     negative_keywords = r'(?:figure|fig\.|table|equation|supplementary\sfigure|supplementary\stable|et al\.|vol\.|pp\.|author|journal|references|section|chapter|citation|cite|ref\.|equation|eq\.|page|pg\.|volume|vol\.\d+|pp\.\d+|et al|doi without data|software|tool|method|algorithm|program|code but not data|model|simulation|parameter|value|result|calculation|formula|equation|fig\s\d+|table\s\d+|supp\. fig|supp\. table|reference \d+|chapter \d+|section \d+|appendix|annex|supplement but not data|footnote|endnote|bibliography|literature|cited in|as in ref|see ref|from ref|\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|v\d+\.\d+|version\s?\d+|build\s?\d+|model\s?\d+|parameter\s?\d+|calculation|formula|equation|[\[\(]\d+[\]\)]|et\s?al\.)'
    
#     # FP patterns for ID direct match (random strings, numbers)
#     fp_patterns = r'(?:vol\.\s?\d+|pp\.\s?\d+|fig\.\s?\d+|table\s?\d+|eq\.\s?\d+|et\s?al\.|^\d+$|^\d+\.\d+$|^\s*[A-Z]\s*$|software|tool|matlab|python|excel|word|powerpoint|chapter\s?\d+|section\s?\d+|appendix|annex|reference\s?\d+|footnote|endnote|bibliography|literature|cited\s?in|as\s?in\s?ref|see\s?ref|from\s?ref|\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|v\d+\.\d+|version\s?\d+|build\s?\d+|model\s?\d+|parameter\s?\d+|calculation|formula|equation|[\[\(]\d+[\]\)]|et\s?al\.)'
    
#     # Known dataset format validation (only keep if matches)
#     dataset_formats = r'(?:10\.\d{4,9}/[^\s"<>]+|CHEMBL\d+|E-GEOD-\d+|E-PROT-\d+|EMPIAR-\d+|ENSBTAG\d+|ENSOARG\d+|EPI_ISL_\d{5,}|EPI\d{6,7}|HPA\d+|CP\d{6}|IPR\d{6}|PF\d{5}|KX\d{6}|K0\d{4}|PRJNA\d+|PXD\d+|SAMN\d+|dryad\s*\.\s*[^\s"<>]+|pasta\s*/\s*[^\s"<>]|E-MTAB-\d+|SRP\d+|ERP\d+|DRP\d+|SRX\d+|DRA\d+|PRJEB\d+|PRJDB\d+|PRJCA\d+|SAMEA\d+|SAMD\d+|SAMC\d+|GCA_[0-9.]+|ERS\d+|EGAS\d+|EGAD\d+|CRD\d+|syn\d+|idr\d+|[A-Z]{4}\d{8,10}(?:\.\d+)?|mcc\d{5}|[A-Z]{1,2}\d{5,8}(?:\.\d+)?|[A-Z]{2}_\d{6,9}(?:\.\d+)?|GSE\d+|E-GEOD-\d+|PXD\d+|EPI_ISL_\d{5,}|EPI\d{6,7}|PF\d{5}|IPR\d{6}|HPA\d{5,6}|CAB\d{6}|HGNC:\d{1,5}|rs\d+| [A-Z0-9]{4})'
    
#     # Frequency threshold (if ID in >10 articles, FP)
#     id_freq = context_df.group_by('dataset_id').agg(pl.count().alias('freq'))
#     high_freq_ids = id_freq.filter(pl.col('freq') > 10)['dataset_id'].to_list()
    
#     initial_count = context_df.height
    
#     # Clean context and ID
#     context_df = context_df.with_columns(
#         pl.col("context").str.to_lowercase().alias("clean_context"),
#         pl.col("dataset_id").str.to_lowercase().alias("clean_id")
#     )
    
#     filtered_df = context_df.filter(
#         # Rule 1: Length checks (5-100 chars)
#         (pl.col("dataset_id").str.len_chars() >= 5) & (pl.col("dataset_id").str.len_chars() <= 100) &
        
#         # Rule 2: No FP patterns in ID
#         (~pl.col("clean_id").str.contains(fp_patterns)) &
        
#         # Rule 3: At least 2 positive keywords in context (for TP preserve)
#         (pl.col("clean_context").str.count_matches(positive_keywords) >= 2) &
        
#         # Rule 4: No negative keywords in context
#         (~pl.col("clean_context").str.contains(negative_keywords)) &
        
#         # Rule 5: Not high frequency ID
#         (~pl.col("dataset_id").is_in(high_freq_ids)) &
        
#         # Rule 6: ID matches known dataset format
#         pl.col("clean_id").str.contains(dataset_formats) &
        
#         # Rule 7: No citation markers in context
#         (~pl.col("clean_context").str.contains(r'\[\d+\]|\(\d{4}\)|\d{4},\s?vol|\d{4},\s?pp|reference\s?\d+')) &
        
#         # Rule 8: Context length >50 (for meaningful context)
#         (pl.col("context").str.len_chars() > 50)
#     )
    
#     removed_count = initial_count - filtered_df.height
#     print(f"'Ultimate FP Slayer v4' complete. Removed {removed_count} junk (aim: all 66,000 FPs gone, TPs safe).")
#     print(f"Remaining candidates: {filtered_df.height}")
    
#     return filtered_df.drop(["clean_context", "clean_id"])
    
# # =================================================================================
# # FINAL SUBMISSION PIPELINE
# # =================================================================================
# def create_submission_final_strategy():
#     print("Starting FINAL Submission Pipeline (Super Cop V2, NO BATCHING)...")
#     gt_df, findable_gt_df = pl.DataFrame(), pl.DataFrame()
#     if not IS_KAGGLE_SUBMISSION:
#         if (COMP_DIR / 'train_labels.csv').exists():
#             gt_df = pl.read_csv(COMP_DIR / 'train_labels.csv')
#         if TRUTH_FINDER_REPORT_PATH.exists():
#             findable_gt_df = pl.read_csv(TRUTH_FINDER_REPORT_PATH).filter(pl.col('status') == 'FOUND').select(['article_id', 'dataset_id'])
#     if not BLACKLIST_PATH.exists():
#         print(f"WARNING: BLACKLIST NOT FOUND.")
#         blacklist_df = pl.DataFrame({'article_id': [], 'dataset_id': []})
#     else:
#         blacklist_df = pl.read_csv(BLACKLIST_PATH).select(['article_id', 'dataset_id'])

#     text_df = prepare_text_dataframe()
#     all_candidates_df = supercharged_extraction_90_percent(text_df)

#     if not IS_KAGGLE_SUBMISSION:
#         evaluate_and_log(all_candidates_df, gt_df, "Before Any Filtering (Raw RegEx Output)")
    
#     # DOI ko pehle hi normalize karlo
#     all_candidates_df = all_candidates_df.with_columns(
#         pl.when(pl.col("dataset_id").str.starts_with("10."))
#           .then(DOI_LINK + pl.col("dataset_id"))
#           .otherwise(pl.col("dataset_id"))
#     )
    
#     # Context poore set par nikalo
#     context_df = get_context_window_df(text_df, all_candidates_df)
    
#     # <<< YAHAN ULTIMATE FP SLAYER KO KAAM PAR LAGAO >>>
#     filtered_context_df = heuristic_filter_v4(context_df)
    
#     print(f"Found {filtered_context_df.height} high-quality candidates to investigate with LLM.")


#     if not VLLM_AVAILABLE: print("ERROR: vLLM not available. Exiting."); return
#     if filtered_context_df.is_empty():
#         print("No candidates remaining after filtering. Creating empty submission.")
#         submission_df = pl.DataFrame({'row_id': [], 'article_id': [], 'dataset_id': [], 'type': []})
#     else:
#         print(f"Loading vLLM model from: {VLLM_MODEL_PATH}")
#         llm = vllm.LLM(VLLM_MODEL_PATH, quantization='awq', tensor_parallel_size=2, gpu_memory_utilization=0.9, trust_remote_code=True, dtype="half", max_model_len=4096)
#         tokenizer = llm.get_tokenizer()

#         if not IS_KAGGLE_SUBMISSION:
#           evaluate_and_log(filtered_context_df.select(['article_id', 'dataset_id']), gt_df, "After Heuristic Filter (Pre-LLM)")

#         print("\nRunning Sherlock Holmes FP Detection...")
#         fp_prompts = [tokenizer.apply_chat_template([{'role':'system', 'content': SYS_PROMPT_SHERLOCK_FP_HUNTER}, {'role':'user', 'content': c}], add_generation_prompt=True, tokenize=False) for c in filtered_context_df['context']]
#         print(f"Total {len(fp_prompts)} prompts FP Hunter ke liye taiyaar hain...")
#         mclp_fp = MultipleChoiceLogitsProcessor(tokenizer, choices=["A", "B", "C"])
#         fp_outputs = llm.generate(fp_prompts, vllm.SamplingParams(temperature=0.0, max_tokens=1, logits_processors=[mclp_fp]), use_tqdm=True)
#         fp_choices = [output.outputs[0].text for output in fp_outputs]
#         del fp_outputs; gc.collect()

#         filtered_context_df = filtered_context_df.with_columns(pl.Series('fp_classification', fp_choices))
#         clean_candidates_df = filtered_context_df.filter(pl.col('fp_classification') == 'A')
#         print(f"{clean_candidates_df.height} candidates passed the Sherlock Holmes check.")
    
#         print("\nRunning Double Check with pre-made blacklist...")
#         final_clean_df = clean_candidates_df.join(blacklist_df, on=['article_id', 'dataset_id'], how='anti')
#         print(f"Removed {clean_candidates_df.height - final_clean_df.height} additional entries using the blacklist.")

#         if not IS_KAGGLE_SUBMISSION:
#             evaluate_and_log(final_clean_df.select(['article_id', 'dataset_id']), gt_df, "After ALL FP Filtering")

#         if final_clean_df.is_empty():
#             print("No valid candidates remaining. Creating empty submission.")
#             submission_df = pl.DataFrame({'row_id': [], 'article_id': [], 'dataset_id': [], 'type': []})
#         else:
#             print(f"\nClassifying the remaining {final_clean_df.height} clean candidates...")
#             type_prompts = [tokenizer.apply_chat_template([{'role':'system', 'content': SYS_PROMPT_CLASSIFY_TYPE.format(context=c)}], add_generation_prompt=True, tokenize=False) for c in final_clean_df['context']]
#             print(f"Total {len(type_prompts)} prompts Primary/Secondary classification ke liye taiyaar hain...")
#             mclp_type = MultipleChoiceLogitsProcessor(tokenizer, choices=["A", "B"])
#             type_outputs = llm.generate(type_prompts, vllm.SamplingParams(temperature=0.0, max_tokens=1, logits_processors=[mclp_type]), use_tqdm=True)
#             type_choices = [output.outputs[0].text for output in type_outputs]
#             del type_outputs; gc.collect()
            
#             types = ['Primary' if c == 'A' else 'Secondary' for c in type_choices]
#             final_submission_df = final_clean_df.with_columns(pl.Series('type', types))
#             submission_df = final_submission_df.select(['article_id', 'dataset_id', 'type']).with_row_index(name='row_id')

#     submission_df.write_csv('submission.csv')
#     print("="*50 + f"\nSUCCESS! Final strategy complete. Submission file created with {submission_df.height} rows.\n" + "="*50)
# if __name__ == "__main__":
#     create_submission_final_strategy()


# =================================================================================
! uv pip install --no-index --find-links='/kaggle/input/all-whl-lib/whls' 'pymupdf' 'vllm' 'triton' 'logits-processor-zoo' 'numpy<2'
import os
from pathlib import Path
import polars as pl
import re
import pymupdf
import gc
from typing import Optional, Tuple, List

LIMIT_ARTICLES = None
try:
    import vllm
    from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("WARNING: vLLM is not installed.")

IS_KAGGLE_ENV = sum(['KAGGLE' in k for k in os.environ]) > 0
IS_KAGGLE_SUBMISSION = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))

if IS_KAGGLE_ENV:
    COMP_DIR = Path('/kaggle/input/make-data-count-finding-data-references')
    VLLM_MODEL_PATH = "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"
    if IS_KAGGLE_SUBMISSION:
        LIMIT_ARTICLES = None
else:
    try:
        COMP_DIR = Path(kagglehub.competition_download('make-data-count-finding-data-references'))
    except Exception as e:
        print(f"Could not download competition data: {e}")
        COMP_DIR = Path('make-data-count-finding-data-references')

# DATA_DIR = COMP_DIR / ('test' if IS_KAGGLE_SUBMISSION else 'train') / 'PDF'
DATA_DIR = COMP_DIR / ('test' if IS_KAGGLE_SUBMISSION else 'train')
WORKING_DIR = Path(('/kaggle/working/' if IS_KAGGLE_ENV else '.working/'))

DOI_LINK = 'https://doi.org/'

WORKING_DIR.mkdir(parents=True, exist_ok=True)



# --- LLM PROMPTS ---
SYS_PROMPT_SHERLOCK_FP_HUNTER = """
You are Sherlock Holmes, a meticulous detective for scientific papers. Determine if an identifier is a true DATASET or something else. Be SKEPTICAL but FAIR â€“ prefer VALID if context suggests data to maximize true positives and remove false positives.
- Reference sections often have citations (B), but if "data" or "dataset" mentioned, it's A.
- Software/tools are C unless explicitly data.
- If context mentions "data", "dataset", "repository", "deposited", "accession", or known banks (GEO, SRA, PDB, Dryad, Zenodo), prefer A.
- If ambiguous but looks like a dataset ID (e.g., alphanumeric codes like GSE123, PXD001, SAMN123), lean towards A to avoid losing true positives.
- For DOIs, if context implies data (e.g., 'deposited at DOI', 'data available at DOI'), A; else B or C.
Examples:
- Context: "Data deposited in GEO under accession GSE12345." -> A (VALID DATASET)
- Context: "Our dataset available at Dryad 10.5061/dryad.abc." -> A
- Context: "Accession SAMN123 in BioSample." -> A
- Context: "PDB ID 1ABC used for analysis." -> A (if data context)
- Context: "EPI_ISL_12345 from GISAID database." -> A
- Context: "IPR006 in InterPro entry." -> A
- Context: "The accession for our data is PXD00123." -> A
- Context: "Figure 1 shows PDB ID 1ABC." -> C (OTHER/JUNK)
- Context: "Cited in vol. 10, pp. 123, DOI 10.1234/paper." -> B (PAPER CITATION)
- Context: "Software MATLAB v9.0 used." -> C
- Context: "This is a citation to a paper: 10.1234/abcd." -> B
- Context: "Term used in the method: vivo." -> C
- Context: "From the result: 2012." -> C
- Context: "Main parameter: 1977." -> C
- Context: "Note in the text: note." -> C
Classify into ONE category:
A) **VALID DATASET**: Likely research data, context supports. Prefer this if any data hint.
B) **PAPER CITATION**: Citation to paper/article/book.
C) **OTHER/JUNK**: Anything else. Use C only if NO doubt it's not data.
Respond with ONLY ONE letter: A, B, or C.
""".strip()

SYS_PROMPT_CLASSIFY_TYPE = """
You are an expert research assistant. Read the following text. It mentions a dataset. Your task is to classify if the dataset was CREATED for this study (Primary) or REUSED from another source (Secondary).
Context: "{context}"
Classify the dataset as: A) Primary or B) Secondary. Respond with only one letter.
""".strip()

# =================================================================================
# HELPER FUNCTIONS
# =================================================================================

def get_text_pdf_only(article_id: str) -> str:
    pdf_path = DATA_DIR / 'PDF' / f"{article_id}.pdf"
    if not pdf_path.exists(): return ""
    try:
        with pymupdf.open(pdf_path) as doc: 
            text = "".join(page.get_text("text", sort=True) for page in doc)
    except Exception: return ""
    text = text.replace('-\n', '')
    text = re.sub(r'\s+', ' ', text.replace('\n', ' '))
    return text.strip()

def prepare_text_dataframe() -> pl.DataFrame:
    print("Preparing text dataframe from PDFs...")
    article_files = sorted([p.stem for p in (DATA_DIR / 'PDF').glob("*.pdf")])
    if LIMIT_ARTICLES and not IS_KAGGLE_SUBMISSION:
        print(f"LIMITING to {LIMIT_ARTICLES} articles for testing.")
        article_files = article_files[:LIMIT_ARTICLES]
    records = [{'article_id': article_id, 'text': get_text_pdf_only(article_id)} for article_id in article_files]
    df = pl.DataFrame(records).filter(pl.col('text').str.len_chars() > 0)
    print(f"Loaded text for {df.height} articles.")
    return df

# --- Text Splitting Functions (to create 'body') ---
COMPILED_PATTERNS = {
    'ref_header_patterns': [re.compile(r'\b(R\s*E\s*F\s*E\s*R\s*E\s*N\s*C\s*E\s*S|BIBLIOGRAPHY|LITERATURE CITED|WORKS CITED)\b[:\s]*', re.IGNORECASE)],
    'citation_pattern': re.compile(r'^\s*(\[\d+\]|\(\d+\)|\d+\.|\d+\))\s*'),
}

def find_last_reference_header(text: str) -> Optional[int]:
    last_match_idx = None
    for pattern in COMPILED_PATTERNS['ref_header_patterns']:
        matches = list(pattern.finditer(text))
        if matches:
            current_last = matches[-1].start()
            if last_match_idx is None or current_last > last_match_idx:
                last_match_idx = current_last
    return last_match_idx

def find_reference_start_by_citation(text: str) -> Optional[int]:
    lines = text.splitlines()
    start_search_line = int(len(lines) * 0.5)
    for i in range(start_search_line, len(lines)):
        line = lines[i].strip()
        if COMPILED_PATTERNS['citation_pattern'].match(line):
            next_lines = lines[i+1:i+4]
            if any(COMPILED_PATTERNS['citation_pattern'].match(l.strip()) for l in next_lines):
                return sum(len(l) + 1 for l in lines[:i])
    return None

def split_text_and_references(text: str) -> Tuple[str, str]:
    header_idx = find_last_reference_header(text)
    if header_idx is not None:
        return text[:header_idx].strip(), text[header_idx:].strip()
    ref_start_idx = find_reference_start_by_citation(text)
    if ref_start_idx is not None:
        return text[:ref_start_idx].strip(), text[ref_start_idx:].strip()
    return text.strip(), ''

def get_splits(df: pl.DataFrame) -> pl.DataFrame:
    main_texts, ref_texts = [], []
    for raw_text in df['text']:
        main, refs = split_text_and_references(raw_text)
        main_texts.append(main)
        ref_texts.append(refs)
    return df.with_columns(pl.Series('body', main_texts), pl.Series('ref', ref_texts))

# --- Extraction Helper Function ---
def clean_and_extract_id(match_str: str) -> str:
    if not match_str: return ""
    context_patterns = [
        r"(?:PDB\sID:?)\s*([A-Z0-9]{4})\b",
        r"(?:CCDC|Deposition\sNumber.?:?|Deposition\sNumbers.?:?)\s*(\d{6,8})\b",
        r"(?:GenBank|accession(?:s|\snumber)?s?:?)\s*([A-Z]{2,4}\d{6,8}(?:\.\d+)?)\b",
        r"(?:IUCr\sElectronic\sArchives\s\(Reference:\s*([A-Z]{2}\d{4})\))"
    ]
    for pattern in context_patterns:
        m = re.search(pattern, match_str, re.IGNORECASE)
        if m and m.group(1): return m.group(1).strip()
    return match_str.strip()

# --- >>> AAPKA FINAL, UPGRADED tidy_extraction FUNCTION <<< ---
def tidy_extraction(df: pl.DataFrame) -> pl.DataFrame:
    
    # Pehli Chhalni: High-Precision Regex (poore text par chalega)
    high_precision_reg_ids = r"""(?ix)
    \b(?:
        # SECTION 1: REPOSITORY-SPECIFIC DOIs & URLs
        10\.5061/dryad\.[a-zA-Z0-9]+\b | https?://datadryad\.org/stash/share/[^\s"<>]+ |
        10\.6075/J0[A-Z0-9]+\b | 10\.5281/zenodo\.\d+\b | 10\.6084/m9\.figshare\.[a-zA-Z0-9.]+\b |
        10\.25387/g3\.[a-zA-Z0-9.]+\b | 10\.25386/genetics\.[a-zA-Z0-9.]+\b |
        10\.1594/PANGAEA\.\d+\b | 10\.5441/[^\s"<>]+ | 10\.11583/DTU\.[^\s"<>]+ |
        10\.17862/cranfield\.rd\.[^\s"<>]+ | 10\.7937/tcia\.[^\s"<>]+ |
        10\.17863/CAM\.\d+\b | 10\.7910/DVN/[A-Z0-9]+\b | 10\.21942/uva\.\d+\b |
        10\.1101/202\d\.\d{2}\.\d{2}\.[^ \s<>"()]+ | 10\.25377/sussex\.\d+\b |
        10\.5066/[A-Z0-9]+\b | https?://www\.ncdc\.noaa\.gov/paleo/study/\d+ |
        https?://www\.icare\.univ-lille\.fr/data-access/[^\s"<>]+ | 10\.13020/[a-zA-Z0-9-]+\b |
        10\.25921/[a-z0-9-]+\b | 10\.21203/rs\.3\.rs-\d+/v\d+\b | 10\.15482/USDA\.ADC/\d+\b |
        10\.15468/dl\.[a-z0-9]+\b | 10\.3334/CDIAC/[^\s"<>]+ | 10\.5291/ILL-DATA\.[^\s"<>]+ |
        10\.15131/shef\.data\.\d+\b | 10\.4121/\d+\b | https?://edms\.cern\.ch/document/\d+/\d+/? |
        10\.3886/[^\s"<>]+ | 10\.6070/[A-Z0-9]+\b | https?://data\.sbgrid\.org/dataset/\d+/? |
        10\.15454/[A-Z0-9]+\b | https?://datacat\.liverpool\.ac\.uk/id/eprint/\d+/? |
        https?://bioportal\.bioontology\.org/ontologies/[A-Z]+\b/? | 10\.13012/[^\s"<>]+ |
        10\.15125/BATH-\d+\b | 10\.5518/\d+\b | 10\.17882/[^\s"<>]+ |
        10\.18434/[A-Z0-9]+\b | 10\.5067/[^\s"<>]+ | 10\.24381/cds\.[a-z0-9]+\b |
        10\.6096/AEROCLO\.\d+\b | 10\.3897/[^\s"<>]+?\.suppl\d+\b |

        # SECTION 2: HIGH-CONFIDENCE PREFIX-BASED IDS
        CHEMBL\d+\b | E-GEOD-\d+\b | E-PROT-\d+\b | EMPIAR-\d+\b | E-MTAB-\d+\b |
        ENSBTAG\d+\b | ENSOARG\d+\b | EPI_ISL_\d{5,}\b | EPI\d{6,7}\b | EPI\d{7,}\b |
        HPA\d+\b | CP\d{6}\b | IPR\d{6}\b | PF\d{5}\b | KX\d{6}\b | K0\d{4}\b | PXD\d+\b |
        (?:SRP|ERP|DRP|GSE|SRX|DRA)\d+\b | (?:PRJNA|PRJEB|PRJDB|PRJCA)\d+\b |
        (?:SAMN|SAMEA|SAMD|SAMC)\d+\b | GCA_[0-9.]+\b | [A-Z]{4}\d{8,10}(?:\.\d+)?\b |
        ERS\d+\b | EGAS\d+\b | EGAD\d+\b | CRD\d+\b | syn\d+\b | idr\d+\b | [A-Z]{4}\d{10,}\b |
        arXiv:\d{4}\.\d{4,5}\b | mcc\d{5}\b |

        # SECTION 3: CONTEXT-AWARE PATTERNS
        (?:PDB\sID:?)\s*([A-Z0-9]{4})\b | (?:CCDC|Deposition\sNumber.?:?|Deposition\sNumbers.?:?)\s*(\d{6,8})\b |
        (?:GenBank|accession(?:s|\snumber)?s?:?)\s*([A-Z]{2,4}\d{6,8}(?:\.\d+)?)\b |
        (?:IUCr\sElectronic\sArchives\s\(Reference:\s*([A-Z]{2}\d{4})\)) |
        
        # SECTION 4: GITHUB/GITLAB REPOSITORY LINK
        https?://github\.com/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/? | https?://gitlab\.[^/]+/[^\s"<>]+/?
    )"""

    # Doosri Chhalni: Generous DOI Regex (sirf 'body' par chalega)
    generous_doi_re = r'10\.\d{4,9}/[^\s"<>()]+'

    # Step 1: High-precision regex ko poore 'text' par chalao
    df_high_precision = (
        df.with_columns(pl.col('text').str.extract_all(high_precision_reg_ids).alias('match'))
        .explode('match').drop_nulls('match')
    )

    # Step 2: Generous DOI regex ko sirf 'body' par chalao
    df_body_doi = (
        df.with_columns(pl.col('body').str.extract_all(generous_doi_re).alias('match'))
        .explode('match').drop_nulls('match')
    )
    
    # Step 3: Dono results ko jod do aur duplicates hatao
    df_combined = pl.concat([df_high_precision, df_body_doi]).unique(subset=['article_id', 'match'], keep='first')
    
    # Step 4: Ab is combined list ko saaf karo
    df_cleaned = df_combined.with_columns(
        pl.col('match').map_elements(clean_and_extract_id, return_dtype=pl.String).alias('dataset_id')
    ).filter(pl.col('dataset_id') != "")

    # Step 5: Group karo, format karo, aur final filtering karo
    df_processed = (
        df_cleaned
        .group_by('article_id', 'dataset_id')
        .agg(pl.col('match'))
        .with_columns(
            pl.when(pl.col('dataset_id').str.starts_with('10.'))
            .then(DOI_LINK + pl.col('dataset_id'))
            .otherwise(pl.col('dataset_id'))
            .alias('dataset_id')
        )
    )
    
    bad_ids = [f'{DOI_LINK}{e}' for e in ['10.5061/dryad', '10.5281/zenodo', '10.6073/pasta']]
    
    final_df = (
        df_processed
        .unique(subset=['article_id', 'dataset_id'], keep='first')
        .filter(~pl.col('article_id').str.replace('_','/').str.contains(pl.col('dataset_id').str.split(DOI_LINK).list.last().str.escape_regex()))
        .filter(~pl.col('dataset_id').str.contains(pl.col('article_id').str.replace('_','/').str.escape_regex()))
        .filter(~pl.col('dataset_id').is_in(bad_ids))
        .filter(pl.when(pl.col('dataset_id').str.starts_with(DOI_LINK).and_(pl.col('dataset_id').str.split('/').list.last().str.len_chars()<5)).then(False).otherwise(True))
    )
    
    return final_df

def get_context_window_df(text_df: pl.DataFrame, ids_df: pl.DataFrame, window_size: int = 250) -> pl.DataFrame:
    df = ids_df.join(text_df, on='article_id')
    contexts = []
    for row in df.iter_rows(named=True):
        text = row['text']
        # 'match' column mein list hai, hum pehla element istemal karenge.
        match_list = row['match']
        if not match_list:
            contexts.append("")
            continue
            
        id_to_find_raw = match_list[0]
        id_to_find_norm = id_to_find_raw.replace(" ", "")
        
        index = text.lower().find(id_to_find_norm.lower())
        if index != -1:
            start = max(0, index - window_size)
            end = min(len(text), index + len(id_to_find_norm) + window_size)
            contexts.append(text[start:end])
        else:
            contexts.append(text[:window_size*2]) # Agar match na mile, to shuru ka text le lo
            
    df_with_context = df.with_columns(pl.Series("context", contexts, dtype=pl.Utf8))
    return df_with_context.filter(pl.col("context").str.len_chars() > 0)
    
def evaluate_and_log(predictions_df: pl.DataFrame, gt_df: pl.DataFrame, stage_name: str):
    if gt_df.is_empty(): 
        return
    def normalize_for_join(col_name: str) -> pl.Expr:
        return pl.col(col_name).str.replace("https://doi.org/", "").str.replace_all(r"[\s\-_:.]+", "").str.to_lowercase()
    relevant_articles = predictions_df['article_id'].unique().to_list()
    gt_subset = gt_df.filter(pl.col('article_id').is_in(relevant_articles))
    hits = gt_subset.with_columns(normalize_for_join("dataset_id").alias("norm_id")).join(
        predictions_df.with_columns(normalize_for_join("dataset_id").alias("norm_id")),
        on=['article_id', 'norm_id'], how='inner'
    )
    tp = hits.select(['article_id', 'dataset_id']).n_unique()
    fp = predictions_df.height - tp
    fn = gt_subset.height - tp
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    print("="*50 + f"\nSTATS FOR STAGE: '{stage_name}'\n" + f"  - True Positives (TP) : {tp}\n" + f"  - False Positives (FP): {fp}\n" + f"  - False Negatives (FN): {fn}\n" + f"  - F1 Score (ID only)  : {f1:.4f}\n" + "="*50)

import polars as pl
import re

def ultra_safe_fp_filter_v2(context_df: pl.DataFrame) -> pl.DataFrame:
    print("Running 'Ultra-Safe FP Filter v2' â€“ Removing more 100% certain junk...")
    
    initial_count = context_df.height

    # === "GOLDEN KEYWORDS" (Yeh waise hi rahenge) ===
    golden_keywords_re = r"""(?ix)
    data\savailability | data\saccessibility | data\sdeposition |
    data\srecords | code\savailability | data\sarchiving |
    available\sat | available\sin | available\sfrom |
    deposited\sin | deposited\swith | submitted\sto |

    can\sbe\sfound\sin | can\sbe\saccessed\sthrough | can\sbe\sdownloaded\sfrom |
    accession\s?(?:number|no|id)s?:? | deposition\s?(?:number|no|id)s?:? |
    dryad | zenodo | figshare | pangaea | github | gitlab |
    ccdc | genbank | geo\sdatabase | sequence\sread\sarchive |
    bioproject | biosample | proteomexchange | gisaid
    """

    # === "RED FLAG PATTERNS" (Yahan humne naye, safe niyam jode hain) ===
    red_flag_re_v2 = r"""(?ix)
    # Pattern 1: Common reference phrases
    et\sal\.?,?\s\(\d{4}\) | # et al. (YYYY)
    journal\sof | proc\.\sof | ann\.\sof | rev\.\sof | # Journal of...
    vol\.\s\d+ | pp\.\s\d+ | # vol. 123, pp. 456
    
    # Pattern 2: Citation markers (very high confidence)
    [\[\(]\d+(?:,\s\d+)?(?:â€“\d+)?[\]\)]\s?$ | # [1], [1,2], [1-3] ID se theek pehle
    ^,?\s?[\[\(]\d+(?:,\s\d+)?(?:â€“\d+)?[\]\)] | # ID ke theek baad

    # Pattern 3: Major academic publishers
    springer | elsevier | wiley | taylor\s&\sfrancis |
    nature\spublishing\sgroup | oxford\suniversity\spress |
    
    # Pattern 4: Obvious non-data context
    fig(?:ure)?\s\d+ | table\s\d+
    """

    # DataFrame ko aage ke kaam ke liye taiyaar karein
    df = context_df.with_columns(
        pl.col("context").str.to_lowercase().alias("clean_context")
    )

    # Naya, "Ultra-Safe v2" filter lagayein
    filtered_df = df.filter(
        ~(
            (pl.col("clean_context").str.contains(red_flag_re_v2)) &
            (~pl.col("clean_context").str.contains(golden_keywords_re))
        )
    )
    
    removed_count = initial_count - filtered_df.height
    print(f"'Ultra-Safe FP Filter v2' complete. Safely removed {removed_count} certain references.")
    print(f"Candidates remaining for LLM: {filtered_df.height}")
    
    return filtered_df.drop("clean_context")

def golden_rule_filter(context_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
    print("Running 'Golden Rule' Filter â€“ Separating VIPs from the crowd...")
    
    # === "GOLDEN KEYWORDS" - Yeh 100% data hone ka saboot hain ===
    # Humne is list ko aur behtar banaya hai.
    golden_keywords_re = r"""(?ix)
    # Phrases that GUARANTEE it's a data statement
    data\savailability\sstatement | data\saccessibility | data\sdeposition |
    data\srecords | code\savailability | data\sarchiving |
    availability\sof\sdata\sand\smaterials |
    
    # Phrases that are extremely strong indicators
    deposited\sin | deposited\swith | available\sin | available\sat | available\sfrom |
    submitted\sto | can\sbe\sfound\sin | can\sbe\saccessed\sthrough |
    can\sbe\sdownloaded\sfrom |
    
    # Specific repository mentions
    dryad\sdigital\srepository | zenodo\sdata\srepository | figshare |
    pangaea | github | gitlab | ccdc | genbank |
    gene\sexpression\somnibus | geo\sdatabase | sequence\sread\sarchive |
    bioproject | biosample | proteomexchange | gisaid |
    
    # Accession/Deposition phrases
    accession\s?(?:number|no|id)s?:? | deposition\s?(?:number|no|id)s?:?
    """

    # DataFrame ko aage ke kaam ke liye taiyaar karein
    df = context_df.with_columns(
        pl.col("context").str.to_lowercase().alias("clean_context")
    )

    # Niyam 1: "Golden" candidates ko alag karo. Yeh hamare 100% TPs hain.
    golden_candidates = df.filter(
        pl.col("clean_context").str.contains(golden_keywords_re)
    )
    
    # Niyam 2: Baaki sabhi candidates ko LLM verification ke liye alag karo.
    # Hum unhe 'golden_candidates' se hata denge taaki koi duplicate na ho.
    candidates_for_llm = df.join(
        golden_candidates, on=['article_id', 'dataset_id'], how='anti'
    )
    
    print(f"'Golden Rule' Filter complete.")
    print(f"  - Found {golden_candidates.height} 'sure-shot' True Positives (Auto-Accepted).")
    print(f"  - Sending {candidates_for_llm.height} remaining candidates to LLM for verification.")
    
    return golden_candidates.drop("clean_context"), candidates_for_llm.drop("clean_context")


# =================================================================================
# <<< FINAL UPDATED SUBMISSION PIPELINE >>>
# =================================================================================
def create_submission_final_strategy():
    print("Starting FINAL Submission Pipeline (Super Cop V2, NO BATCHING)...")
    gt_df = pl.DataFrame()
    if not IS_KAGGLE_SUBMISSION:
        if (COMP_DIR / 'train_labels.csv').exists():
            gt_df = pl.read_csv(COMP_DIR / 'train_labels.csv')

    
    text_df = prepare_text_dataframe()

    
    print("Splitting text into body and references...")
    split_df = get_splits(text_df)

    print("Extracting potential candidates using Hybrid Funnel strategy...")
    all_candidates_df = tidy_extraction(split_df)

    if not IS_KAGGLE_SUBMISSION:
        evaluate_and_log(all_candidates_df, gt_df, "Before Any Filtering (Raw RegEx Output)")
    
    # --- FILTER STAGE 3: LLM VERIFICATION ---
    print(f"\n--- FILTER STAGE 3: LLM Verification ---")
    if not VLLM_AVAILABLE: 
        print("ERROR: vLLM not available. Exiting.")
        return
        
    if all_candidates_df.is_empty():
        print("No candidates remaining after filtering. Creating empty submission.")
        submission_df = pl.DataFrame({'row_id': [], 'article_id': [], 'dataset_id': [], 'type': []})
    else:
          cotext_df_for_widow = get_context_window_df(text_df, all_candidates_df)
          context_df_for_llm  = ultra_safe_fp_filter_v2(cotext_df_for_widow)
          # auto_accepted_tps, context_df_for_llm = golden_rule_filter(context_medium_filter)
    
          # --- TIER 2: EXPERT "DETECTIVE" LLM VERIFICATION ---
          # print(f"\n--- TIER 2: Applying Sherlock Holmes LLM to {candidates_for_sherlock_df.height} candidates ---")    
          print(f"Loading vLLM model from: {VLLM_MODEL_PATH}")
          llm = vllm.LLM(VLLM_MODEL_PATH, quantization='awq', tensor_parallel_size=2, gpu_memory_utilization=0.9, trust_remote_code=True, dtype="half", max_model_len=4096)
          tokenizer = llm.get_tokenizer()
          if not IS_KAGGLE_SUBMISSION:
            evaluate_and_log(context_df_for_llm.select(['article_id', 'dataset_id']), gt_df, "After Heuristic Filter (Pre-LLM)")

          clean_llm_verified_df = pl.DataFrame(schema=context_df_for_llm.schema)


          fp_prompts = [tokenizer.apply_chat_template([
            {'role':'system', 'content': SYS_PROMPT_SHERLOCK_FP_HUNTER}, {'role':'user', 'content': c}], add_generation_prompt=True, tokenize=False) for c in context_df_for_llm['context']]
          print(f"Total {len(fp_prompts)} prompts FP Hunter ke liye taiyaar hain...")
          mclp_fp = MultipleChoiceLogitsProcessor(tokenizer, choices=["A", "B", "C"])
          fp_outputs = llm.generate(fp_prompts, vllm.SamplingParams(temperature=0.0, max_tokens=1, logits_processors=[mclp_fp]), use_tqdm=True)
          fp_choices = [output.outputs[0].text for output in fp_outputs]
          del fp_outputs; gc.collect()

          context_df_for_llm = context_df_for_llm.with_columns(pl.Series('fp_classification', fp_choices))
          clean_candidates_df = context_df_for_llm.filter(pl.col('fp_classification') == 'A')
          print(f"{clean_candidates_df.height} candidates passed the Sherlock Holmes check.")

          # required_cols  = ['article_id', 'dataset_id', 'match','context' ]
          # final_clean_df = pl.concat([
          #             auto_accepted_tps.select(required_cols),
          #             clean_candidates_df.select(required_cols)
          #            ]).unique(subset=['article_id', 'dataset_id'], keep='first')

          final_clean_df = clean_candidates_df

          if not IS_KAGGLE_SUBMISSION:
            evaluate_and_log(final_clean_df.select(['article_id', 'dataset_id']), gt_df, "After Heuristic Filter (Pre-LLM)")
              
          if final_clean_df.is_empty():
             print("No valid candidates remaining. Creating empty submission.")
             submission_df = pl.DataFrame({'row_id': [], 'article_id': [], 'dataset_id': [], 'type': []})
          else:
             print(f"\nClassifying the remaining {final_clean_df.height} clean candidates...")
             type_prompts = [tokenizer.apply_chat_template([{'role':'system', 'content': SYS_PROMPT_CLASSIFY_TYPE.format(context=c)}], add_generation_prompt=True, tokenize=False) for c in final_clean_df['context']]
             print(f"Total {len(type_prompts)} prompts Primary/Secondary classification ke liye taiyaar hain...")
             mclp_type = MultipleChoiceLogitsProcessor(tokenizer, choices=["A", "B"])
             type_outputs = llm.generate(type_prompts, vllm.SamplingParams(temperature=0.0, max_tokens=1, logits_processors=[mclp_type]), use_tqdm=True)
             type_choices = [output.outputs[0].text for output in type_outputs]
             del type_outputs; gc.collect()
            
             types = ['Primary' if c == 'A' else 'Secondary' for c in type_choices]
             final_submission_df = final_clean_df.with_columns(pl.Series('type', types))
             submission_df = final_submission_df.select(['article_id', 'dataset_id', 'type']).with_row_index(name='row_id')


    submission_df.write_csv('submission.csv')
    print("="*50 + f"\nSUCCESS! Final strategy complete. Submission file created with {submission_df.height} rows.\n" + "="*50)



if __name__ == "__main__":
    create_submission_final_strategy()

