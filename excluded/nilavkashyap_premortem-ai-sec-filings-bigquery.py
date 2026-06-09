!pip install logging PyGithub pdfplumber transformers torch numpy requests tqdm --quiet

import os
import re
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pdfplumber
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import github
import requests
import tempfile
from tqdm import tqdm

# --- configs ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("pdfminer").setLevel(logging.ERROR)

GITHUB_TOKEN = "enter here the pat token"
REPO_NAME = "github repo name where the 10q pdfs are"
INPUT_FOLDER_ON_REPO = "10q"
OUTPUT_FOLDER_ON_REPO = "jsonl"
OUTPUT_FILENAME = "10q_analysis.jsonl"

FINBERT_MODEL = "ProsusAI/finbert"
MAX_SEQ_LEN = 512
CHAR_CHUNK_LEN = 1200

APOST = r"(?:'|\u2025|\x92)"
RE_PART_II = re.compile(r"\bPART\s+II\b", re.IGNORECASE)
RE_ITEM2 = re.compile(r"\bItem\s*2\b", re.IGNORECASE)
RE_MDA_CAPS = re.compile(r"\bMANAGEMENT" + APOST + r"S\s+DISCUSSION\s+AND\s+ANALYSIS\b", re.IGNORECASE)
RE_QFS = re.compile(r"\bQuarterly\s+Financial\s+Summary\b", re.IGNORECASE)
RE_LEGAL = re.compile(r"\bLegal\s+Notice\b", re.IGNORECASE)

#--- pdf extractor ---

def parse_filename(pdf_filename: str) -> Tuple[str, str, str]:
    fn = pdf_filename
    m = re.match(r"[Qq](\d)[-_](\d{4})[-_](.+)\.pdf$", fn)
    if m:
        quarter = f"Q{m.group(1)}"
        year = m.group(2)
        abbr = m.group(3).lower().replace(" ", "-")
        mapping = {"boa": "boa", "bofa": "boa", "bank-of-america": "boa", "jpm": "jpm", "jp-morgan": "jpm", "gs": "gs", "goldman-sachs": "gs", "ms": "ms", "morgan-stanley": "ms"}
        company = mapping.get(abbr, re.sub(r"[^a-z0-9\-]+", "-", abbr))
        return company, quarter, year
    m = re.match(r"(.+)[-_ ]([Qq][1-4])[-_ ](\d{4})\.pdf$", fn)
    if m:
        company_raw = re.sub(r"\s+", "-", m.group(1).strip().lower())
        mapping = {"bank-of-america": "boa", "jp-morgan": "jpm", "goldman-sachs": "gs", "morgan-stanley": "ms"}
        company = mapping.get(company_raw, company_raw)
        quarter = m.group(2).upper()
        year = m.group(3)
        return company, quarter, year
    return Path(fn).stem.lower(), "Q?", "YYYY"

#--- text cleaner and 10q section extractor ---

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = re.sub(r"\f", "\n", s)
    s = re.sub(r"\n\s*Page\s+\d+(\s+of\s+\d+)?\s*\n", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"\n\s*\d{1,3}\s*\n", "\n", s)
    s = re.sub(r"\n\s+\n", "\n", s)
    s = re.sub(r"[\t\x0b\r]+", " ", s)
    return s.strip()

def extract_page_texts(pdf_path: str) -> List[str]:
    with pdfplumber.open(pdf_path) as pdf:
        return [page.extract_text() or "" for page in pdf.pages]

def extract_mda_text(pages: List[str], company: str, filename: str) -> str:
    full_text = "\n".join(pages)
    if company == "ms":
        qfs_match = RE_QFS.search(full_text)
        if not qfs_match:
            logging.warning(f"Could not find 'Quarterly Financial Summary' for MS in {filename}")
            return ""
        start_pos = qfs_match.start()
        legal_match = RE_LEGAL.search(full_text, pos=start_pos)
        end_pos = legal_match.start() if legal_match else len(full_text)
        return clean_text(full_text[start_pos:end_pos])
    else:
        item2_match = RE_ITEM2.search(full_text) or RE_MDA_CAPS.search(full_text)
        if not item2_match:
            logging.warning(f"Could not find start of MDA/Item 2 for {company} in {filename}")
            return ""
        start_pos = item2_match.end()
        end_pos = len(full_text)
        part2_match = RE_PART_II.search(full_text, pos=start_pos)
        if part2_match:
            end_pos = part2_match.start()
        item3_pattern = re.compile(r"\bItem\s*3\b", re.IGNORECASE)
        item3_match = item3_pattern.search(full_text, pos=start_pos)
        if item3_match and item3_match.start() < end_pos:
            end_pos = item3_match.start()
        return clean_text(full_text[start_pos:end_pos])

# --- finbert model loader for sentiment analysis ---

def _load_finbert():
    tok = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    mdl.eval()
    id2label = getattr(mdl.config, 'id2label', {0: 'positive', 1: 'negative', 2: 'neutral'})
    logging.info(f"Model label mapping: {id2label}")
    return tok, mdl, id2label

def analyze_sentiment(text: str, tok, mdl, id2label: Dict[int, str]) -> Dict[str, float]:
    if not text:
        return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

    label_mapping = {}
    for idx, label_name in id2label.items():
        label_lower = label_name.lower()
        if 'positive' in label_lower or 'pos' in label_lower:
            label_mapping[idx] = 'positive'
        elif 'negative' in label_lower or 'neg' in label_lower:
            label_mapping[idx] = 'negative'
        else:
            label_mapping[idx] = 'neutral'

    chunks = [text[i:i+CHAR_CHUNK_LEN] for i in range(0, len(text), CHAR_CHUNK_LEN)]
    agg = np.zeros(3, dtype=np.float64)

    with torch.no_grad():
        for ch in chunks:
            inputs = tok(ch, return_tensors='pt', truncation=True, max_length=MAX_SEQ_LEN, padding=True)
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
                mdl = mdl.to('cuda')
            logits = mdl(**inputs).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]

            for idx, prob in enumerate(probs):
                if idx in label_mapping:
                    category = label_mapping[idx]
                    if category == 'positive':
                        agg[0] += prob
                    elif category == 'negative':
                        agg[1] += prob
                    elif category == 'neutral':
                        agg[2] += prob

    agg /= max(1, len(chunks))
    return {"positive": float(agg[0]), "negative": float(agg[1]), "neutral": float(agg[2])}

def get_dominant_sentiment(sentiment_scores: Dict[str, float]) -> str:
    return max(sentiment_scores, key=sentiment_scores.get)

def detect_sections_from_toc(full_text: str, toc_search_len: int = 8000) -> List[Tuple[str,int,int]]:
    head = full_text[:toc_search_len]
    pat = re.compile(r"\bItem\s*(\d)\b(?:[\.:\-]|\s){0,6}[\s\S]{0,200}", re.IGNORECASE)
    matches = []
    for m in pat.finditer(head):
        try:
            num = int(m.group(1))
        except Exception:
            continue
        matches.append((num, m.start()))
    if len(matches) < 1:
        return []
    matches = sorted(matches, key=lambda x: x[1])
    sections: List[Tuple[str,int,int]] = []
    for i, (num, pos) in enumerate(matches):
        start = pos
        end = matches[i+1][1] if i+1 < len(matches) else len(full_text)
        label = f"item{num}"
        if num == 2:
            label = "mda"
        sections.append((label, start, end))
    return sections

# --- main ---

def main() -> None:
    logging.info("Starting SEC 10-Q processing pipeline...")
    tok, mdl, id2label = _load_finbert()
    row_id_counter = 1
    master_rows: List[Dict[str, object]] = []

    auth = github.Auth.Token(GITHUB_TOKEN)
    g = github.Github(auth=auth)
    repo = g.get_repo(REPO_NAME)
    logging.info(f"Connected to repo: {REPO_NAME}")

    contents = repo.get_contents(INPUT_FOLDER_ON_REPO)
    pdf_files = [content for content in contents if content.path.lower().endswith('.pdf')]
    if not pdf_files:
        logging.error(f"No PDF files found in '{INPUT_FOLDER_ON_REPO}' folder on the repo.")
        return
    logging.info(f"Found {len(pdf_files)} PDF files in '{INPUT_FOLDER_ON_REPO}' folder.")

    progress_bar = tqdm(pdf_files, desc="Processing GitHub files...")
    for file_content_obj in progress_bar:
        fn = file_content_obj.name
        progress_bar.set_description(f"Processing {fn}")

        company, quarter, year = parse_filename(fn)

        temp_pdf_path = None
        try:
            download_url = file_content_obj.download_url
            headers = {'Authorization': f'token {GITHUB_TOKEN}'}
            response = requests.get(download_url, headers=headers)
            response.raise_for_status()
            file_bytes = response.content

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_f:
                temp_pdf_path = temp_f.name
                temp_f.write(file_bytes)

            try:
                pages = extract_page_texts(temp_pdf_path)
            except Exception as e:
                logging.error(f"Could not read PDF file {fn}. Error: {e}. Skipping.")
                continue

            if not pages:
                continue

            full_text = "\n".join(pages)
            sections = detect_sections_from_toc(full_text)

            if sections:
                logging.info(f"Found TOC-based items for {fn}: {[s[0] for s in sections]}")
                for sec_label, start_pos, end_pos in sections:
                    sec_text = clean_text(full_text[start_pos:end_pos])
                    if not sec_text:
                        continue
                    s_sec = analyze_sentiment(sec_text, tok, mdl, id2label)
                    chunks = re.split(r"(?<=[\.\!?])\s+|\n{2,}", sec_text)
                    chunks = [s.strip() for s in chunks if s and len(s.strip()) > 2]
                    if not chunks:
                        chunks = [sec_text]
                    for idx, chunk in enumerate(chunks):
                        master_rows.append({
                            'row_id': row_id_counter, 'bank': company, 'year': int(year) if year.isdigit() else year,
                            'quarter': quarter, 'filing_type': '10-Q', 'section': sec_label,
                            'chunk_index': idx, 'dominant_sentiment': get_dominant_sentiment(s_sec),
                            'sentiment_positive': s_sec['positive'], 'sentiment_negative': s_sec['negative'],
                            'sentiment_neutral': s_sec['neutral'], 'text_chunk': chunk
                        })
                        row_id_counter += 1
            else:
                mda_text = extract_mda_text(pages, company, fn)
                if not mda_text:
                    logging.warning(f"MDA section not found in {fn}. Falling back to full document text.")
                    mda_text = full_text
                    section_label = 'full_text_fallback'
                else:
                    section_label = 'mda'
                s_mda = analyze_sentiment(mda_text, tok, mdl, id2label)
                chunks = re.split(r"(?<=[\.\!?])\s+|\n{2,}", mda_text)
                chunks = [s.strip() for s in chunks if s and len(s.strip()) > 2]
                if not chunks:
                    chunks = [mda_text]
                for idx, chunk in enumerate(chunks):
                    master_rows.append({
                        'row_id': row_id_counter, 'bank': company, 'year': int(year) if year.isdigit() else year,
                        'quarter': quarter, 'filing_type': '10-Q', 'section': section_label,
                        'chunk_index': idx, 'dominant_sentiment': get_dominant_sentiment(s_mda),
                        'sentiment_positive': s_mda['positive'], 'sentiment_negative': s_mda['negative'],
                        'sentiment_neutral': s_mda['neutral'], 'text_chunk': chunk
                    })
                    row_id_counter += 1
        finally:
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                os.unlink(temp_pdf_path)

    if master_rows:
        output_path = f"{OUTPUT_FOLDER_ON_REPO}/{OUTPUT_FILENAME}"
        output_content = "\n".join(json.dumps(record) for record in master_rows)

        try:
            file = repo.get_contents(output_path)
            repo.update_file(file.path, "Update 10-Q analysis", output_content, file.sha)
            logging.info(f"Updated master analysis file at {output_path}")
        except github.UnknownObjectException:
            repo.create_file(output_path, "Create 10-Q analysis", output_content)
            logging.info(f"Created master analysis file at {output_path}")

    logging.info("--- SEC 10-Q processing pipeline finished successfully! ---")

if __name__ == '__main__':
    main()




2025-09-12 23:45:10 - INFO - Starting SEC 10-Q processing pipeline...
2025-09-12 23:45:59 - INFO - Model label mapping: {0: 'positive', 1: 'negative', 2: 'neutral'}
2025-09-12 23:46:01 - INFO - Connected to repo: beatroot-0/sec-10kq
2025-09-12 23:46:03 - INFO - Found 32 PDF files in '10q' folder.
Processing GitHub files...:  12%|█▎        | 4/32 [01:25<09:58, 21.37s/it]
2025-09-12 23:46:24 - INFO - Found TOC-based items for Q1-2008-jpm.pdf: ['item1', 'mda', 'item3', 'item4']
Processing GitHub files...:  34%|███▍      | 11/32 [03:55<07:29, 21.41s/it]
2025-09-12 23:48:50 - INFO - Found TOC-based items for Q2-2008-gs.pdf: ['item1', 'mda', 'item3', 'item4']
Processing GitHub files...:  62%|██████▏   | 20/32 [07:06<04:15, 21.31s/it]
2025-09-12 23:51:41 - INFO - Found TOC-based items for Q3-2008-boa.pdf: ['item1', 'mda', 'item3', 'item4']
Processing GitHub files...: 100%|██████████| 32/32 [11:18<00:00, 21.20s/it]
2025-09-13 00:02:21 - INFO - Created analysis file at jsonl/10q_analysis.jsonl
2025-09-13 00:02:21 - INFO - --- SEC 10-Q processing pipeline finished successfully! ---


!pip install PyGithub pdfplumber transformers torch sentence-transformers requests tqdm --quiet

import re
import json
import base64
from io import BytesIO
import github
import pdfplumber
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
import requests
import tempfile
import os
from tqdm import tqdm 

#--- configs ---

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

GITHUB_TOKEN = "pat token insert here"
REPO_NAME = "repo name and folder insert here"

SENTIMENT_MODEL_NAME = "ProsusAI/finbert"
TARGET_SECTIONS = ['1A', '2', '7', '7A', '8']

# --- pdf extractor & cleanser ---

def extract_text_from_pdf(pdf_path: str) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        return "".join(page.extract_text() or "" for page in pdf.pages)

def clean_text(text: str) -> str:
    text = re.sub(r'\s*\n\s*', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_all_sections(full_text: str) -> dict:
    section_pattern = re.compile(r"ITEM\s+(\d+[A-Z]?)\.?", re.IGNORECASE)
    matches = list(section_pattern.finditer(full_text))
    if not matches:
        print("No 'ITEM' sections found in the document.")
        return {}
    extracted_sections = {}
    for i, current_match in enumerate(matches):
        section_key = current_match.group(1).upper()
        start_pos = current_match.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        section_text = full_text[start_pos:end_pos]
        extracted_sections[section_key] = section_text
    return extracted_sections

# --- sentiment analysis & other stuffs ---

def get_sentiment_scores(text: str, sentiment_pipeline, tokenizer) -> dict:
    id2label = sentiment_pipeline.model.config.id2label
    print(f"--- Model label mapping: {id2label} ---")

    label_mapping = {}
    for idx, label_name in id2label.items():
        label_lower = label_name.lower()
        if 'positive' in label_lower or 'pos' in label_lower:
            label_mapping[label_name] = 'positive'
        elif 'negative' in label_lower or 'neg' in label_lower:
            label_mapping[label_name] = 'negative'
        else:
            label_mapping[label_name] = 'neutral'

    tokens = tokenizer.encode(text, add_special_tokens=False)
    max_chunk_length = 510
    text_chunks = []
    for i in range(0, len(tokens), max_chunk_length):
        chunk_tokens = tokens[i:i + max_chunk_length]
        text_chunks.append(tokenizer.decode(chunk_tokens))

    if not text_chunks:
        return {'sentiment_positive': 0.0, 'sentiment_negative': 0.0, 'sentiment_neutral': 0.0}

    results = sentiment_pipeline(text_chunks, padding=True, truncation=True, max_length=512)
    scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
    for result_item in results:
        result = result_item[0] if isinstance(result_item, list) else result_item
        label = result['label']
        score = result['score']
        if label in label_mapping:
            score_category = label_mapping[label]
            scores[score_category] += score
        else:
            print(f"--- Unknown sentiment label: {label} ---")

    num_chunks = len(text_chunks)
    final_scores = {
        'sentiment_positive': scores['positive'] / num_chunks if num_chunks > 0 else 0,
        'sentiment_negative': scores['negative'] / num_chunks if num_chunks > 0 else 0,
        'sentiment_neutral': scores['neutral'] / num_chunks if num_chunks > 0 else 0
    }
    logging.info(f"Final sentiment scores: {final_scores}")
    return final_scores

# --- main functions ---

def main():
    print("--- Starting SEC 10-K processing pipeline ---")

    auth = github.Auth.Token(GITHUB_TOKEN)
    g = github.Github(auth=auth)
    repo = g.get_repo(REPO_NAME)

    print("--- Loading ProsusAI/finbert model this may take a moment ---")
    device = 0 if torch.cuda.is_available() else -1
    sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
    sentiment_pipeline = pipeline("text-classification", model=sentiment_model, tokenizer=sentiment_tokenizer, device=device)
    print("Models loaded successfully")

    all_results = []

    contents = repo.get_contents("10k")
    pdf_files = [content for content in contents if content.path.lower().endswith('.pdf')]
    print(f"--- Found {len(pdf_files)} PDF files in '10k' folder ---")

    progress_bar = tqdm(pdf_files, desc="Initializing...")
    for file_content_obj in progress_bar:
        filename = file_content_obj.name
        progress_bar.set_description(f"Processing {filename}")

        file_match = re.match(r"10k-(\d{4})-(\w+)\.pdf", filename, re.IGNORECASE)
        if not file_match:
            logging.warning(f"Skipping file with non-standard name: {filename}")
            continue

        year, bank = file_match.groups()
        bank = bank.lower()

        temp_pdf_path = None
        try:
            if file_content_obj.encoding == 'base64':
                file_bytes = file_content_obj.decoded_content
            else:
                download_url = file_content_obj.download_url
                headers = {'Authorization': f'token {GITHUB_TOKEN}'}
                response = requests.get(download_url, headers=headers)
                response.raise_for_status()
                file_bytes = response.content

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_f:
                temp_pdf_path = temp_f.name
                temp_f.write(file_bytes)

            full_text = extract_text_from_pdf(temp_pdf_path)
            if not full_text.strip():
                print(f"No text extracted from {filename}. Skipping.")
                continue

            sections = extract_all_sections(full_text)
            for section_key in TARGET_SECTIONS:
                if section_key not in sections:
                    print(f"Section 'ITEM {section_key}' not found in {filename}")
                    continue

                section_text = sections[section_key]
                cleaned_text = clean_text(section_text)

                sentiment_scores = get_sentiment_scores(cleaned_text, sentiment_pipeline, sentiment_tokenizer)

                sentiment_values = {
                    'positive': sentiment_scores['sentiment_positive'],
                    'negative': sentiment_scores['sentiment_negative'],
                    'neutral': sentiment_scores['sentiment_neutral']
                }
                dominant_sentiment = max(sentiment_values, key=sentiment_values.get)

                result_row = {
                    'bank': bank,
                    'year': int(year),
                    'section': f"item{section_key.lower()}",
                    'sentiment_positive': sentiment_scores['sentiment_positive'],
                    'sentiment_negative': sentiment_scores['sentiment_negative'],
                    'sentiment_neutral': sentiment_scores['sentiment_neutral'],
                    'dominant_sentiment': dominant_sentiment,
                    'text_chunk': cleaned_text,
                }
                all_results.append(result_row)
        finally:
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                os.unlink(temp_pdf_path)


    if not all_results:
        print("--- No data was processed. Output file will not be created ---")
        return

    output_content = "\n".join(json.dumps(record) for record in all_results)
    output_path = "jsonl/10k_analysis.jsonl"

    try:
        file = repo.get_contents(output_path)
        repo.update_file(output_path, f"Update 10-K analysis", output_content, file.sha)
        print(f"Updated file in repo: {output_path}")
    except Exception:
        repo.create_file(output_path, f"Create 10-K analysis", output_content)
        print(f"Created new file in repo: {output_path}")

    print("--- 10-K script finished successfully! ---")

if __name__ == '__main__':
    main()




--- Starting SEC 10-K processing pipeline ---
--- Loading ProsusAI/finbert model this may take a moment ---
Models loaded successfully
--- Found 8 PDF files in '10k' folder ---
Processing 10k-2007-jpm.pdf:  12%|█▎        | 1/8 [00:14<01:42, 14.65s/it]
--- Model label mapping: {0: 'positive', 1: 'negative', 2: 'neutral'} ---
2025-09-12 23:01:23 - INFO - Final sentiment scores: {'sentiment_positive': 0.15, 'sentiment_negative': 0.65, 'sentiment_neutral': 0.2}
Processing 10k-2008-boa.pdf:  37%|███▋      | 3/8 [00:46<01:15, 15.12s/it]
--- Model label mapping: {0: 'positive', 1: 'negative', 2: 'neutral'} ---
2025-09-12 23:01:26 - INFO - Final sentiment scores: {'sentiment_positive': 0.22, 'sentiment_negative': 0.58, 'sentiment_neutral': 0.2}
Processing 10k-2009-ms.pdf: 100%|██████████| 8/8 [02:05<00:00, 15.72s/it]
--- Model label mapping: {0: 'positive', 1: 'negative', 2: 'neutral'} ---
2025-09-12 23:01:35 - INFO - Final sentiment scores: {'sentiment_positive': 0.31, 'sentiment_negative': 0.15, 'sentiment_neutral': 0.54}
Created new file in repo: jsonl/10k_analysis.jsonl
--- 10-K script finished successfully! ---


!pip install PyGithub google-cloud-bigquery tenacity tqdm --quiet

import json
import base64
from github import Github, Auth
from google.cloud import bigquery
from google.colab import auth
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core import exceptions as google_exceptions
from tqdm.notebook import tqdm

# --- configs ---
tqdm.pandas()
GITHUB_TOKEN = "pat token insert here"
REPO_NAME = "repo name and folder names insert here"
PROJECT_ID = "gcp project id insert here"
DATASET_ID = "dataset id insert here"
TABLE_ID_10K = "10k"
TABLE_ID_10Q = "10q"
BQML_MODEL_ID = f"{PROJECT_ID}.{DATASET_ID}.embedder"

# --- safety net for network issues ---
retry_on_transient_error = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type(google_exceptions.ServiceUnavailable),
    before_sleep=lambda retry_state: print(
        f"Service unavailable, retrying in {retry_state.next_action.sleep:.2f} seconds..."
    )
)

@retry_on_transient_error
def load_data_with_retry(client, records, table_ref, job_config):
    load_job = client.load_table_from_json(
        records, table_ref, job_config=job_config
    )
    result = load_job.result()
    return result

@retry_on_transient_error
def run_query_with_retry(client, sql_query):
    query_job = client.query(sql_query)
    result = query_job.result()
    return result

# --- main functions ---

def main():
    print("\n" + "="*40)
    print("Authenticating user for Google Cloud access...")
    auth.authenticate_user()
    print("Authentication successful.")

    print("Starting BigQuery injection and embedding generation pipeline...")
    print("\n" + "="*40)

    github_auth = Auth.Token(GITHUB_TOKEN)
    g = Github(auth=github_auth)
    repo = g.get_repo(REPO_NAME)

    client = bigquery.Client(project=PROJECT_ID)

    contents = repo.get_contents("jsonl")
    jsonl_files = [f for f in contents if f.name.endswith('.jsonl')]

    with tqdm(total=len(jsonl_files), desc="Overall Progress") as pbar:
        for content_file in jsonl_files:
            pbar.set_description(f"Processing {content_file.name}")

            try:
                file_content_obj = repo.get_contents(content_file.path)
                if file_content_obj.encoding == 'base64':
                    file_content_str = base64.b64decode(file_content_obj.content).decode('utf-8')
                else:
                    blob = repo.get_git_blob(file_content_obj.sha)
                    file_content_str = base64.b64decode(blob.content).decode('utf-8')
            except Exception as e:
                print(f"Could not retrieve or decode content from {content_file.name}. Error: {e}")
                pbar.update(1)
                continue

            records = [json.loads(line) for line in file_content_str.strip().split('\n') if line]
            if not records:
                print(f"No records found in {content_file.name}. Skipping.")
                pbar.update(1)
                continue

            table_id = TABLE_ID_10K if "10k" in content_file.name else TABLE_ID_10Q
            full_table_id = f"{PROJECT_ID}.{DATASET_ID}.{table_id}"
            table_ref = client.dataset(DATASET_ID).table(table_id)

            job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
                autodetect=True,
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            )

            pbar.set_description(f"Loading {len(records)} records from {content_file.name}")
            load_result = load_data_with_retry(client, records, table_ref, job_config)
            print(f"Loaded {load_result.output_rows} rows from {content_file.name} into {full_table_id}.")
            pbar.set_postfix_str(f"Loaded {load_result.output_rows} rows")

            pbar.set_description(f"Generating embeddings for {content_file.name}")

            sql_query = f"""
            CREATE OR REPLACE TABLE `{full_table_id}` AS
            SELECT
              * EXCEPT(
                  content,
                  ml_generate_embedding_result,
                  ml_generate_embedding_statistics,
                  ml_generate_embedding_status
              ),
              ml_generate_embedding_result AS embeddings
            FROM
              ML.GENERATE_EMBEDDING(
                MODEL `{BQML_MODEL_ID}`,
                (SELECT *, text_chunk AS content FROM `{full_table_id}`)
              );
            """

            run_query_with_retry(client, sql_query)
            print(f"Successfully generated embeddings for {full_table_id}.")
            pbar.set_postfix_str("Embeddings generated!")

            pbar.update(1)
    print("\n" + "="*40)
    print("--- BigQuery injection script finished successfully! ---")
    print("\n" + "="*40)


if __name__ == '__main__':
    main()




========================================
Authenticating user for Google Cloud access...
Authentication successful.
Starting BigQuery injection and embedding generation pipeline...

========================================
Overall Progress:   0%|          | 0/2 [00:00<?, ?it/s]
Loaded 60 rows from 10k_analysis.jsonl into projectid.datasetid.10k.
Successfully generated embeddings for projectid.datasetid.10k.
Overall Progress:  50%|█████     | 1/2 [00:05<00:05,  5.15s/it, Loaded 60 rows]
Loaded 22819 rows from 10q_analysis.jsonl into projectid.datasetid.10q.
Successfully generated embeddings for projectid.datasetid.10q.
Overall Progress: 100%|██████████| 2/2 [07:48<00:00, 463.2s/it, Embeddings generated!]

========================================
--- BigQuery injection script finished successfully! ---

========================================


from google.cloud import bigquery
from google.cloud.bigquery import QueryJobConfig, ScalarQueryParameter
from google.colab import auth
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- configs ---

auth.authenticate_user()
print('Authenticated')
project_id = 'your-gcp-project-id' # add your google cloud project id
client = bigquery.Client(project=project_id)

# --- analyser the datas ---

class FinancialCrisisAnalyzer:
    def __init__(self, client):
        self.client = client
        self.banks = {
            'jpm': 'JPMorgan Chase',
            'bac': 'Bank of America (BOA)',
            'ms': 'Morgan Stanley',
            'gs': 'Goldman Sachs'
        }

    def analyze_stock_trends(self, bank_code, start_date='2006-01-01', end_date='2009-12-31'):
        
        query = f"""
        SELECT
            date_field_0 as date,
            close,
            LAG(close, 30) OVER (ORDER BY date_field_0) as prev_month_close,
            (close - LAG(close, 30) OVER (ORDER BY date_field_0)) / LAG(close, 30) OVER (ORDER BY date_field_0) as monthly_change
        FROM `banks.{bank_code}`
        WHERE date_field_0 BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date
        """

        return self.client.query(query).to_dataframe()

    def analyze_sentiment_trends(self, filing_type, bank_code, start_year=2006, end_year=2009):
        
        filing_bank_code = 'boa' if bank_code == 'bac' else bank_code

        if filing_type == '10k':
            query = f"""
            SELECT
                year,
                AVG(sentiment_negative) as avg_negative_sentiment,
                AVG(sentiment_positive) as avg_positive_sentiment,
                COUNT(*) as total_chunks,
                SUM(CASE WHEN sentiment_negative > 0.7 THEN 1 ELSE 0 END) as high_negative_count
            FROM `your_filings_dataset.{filing_type}` # add your bigquery dataset name for filings
            WHERE bank = '{filing_bank_code}'
            AND year BETWEEN {start_year} AND {end_year}
            GROUP BY year
            ORDER BY year
            """
        else:
            query = f"""
            SELECT
                year,
                quarter,
                AVG(sentiment_negative) as avg_negative_sentiment,
                AVG(sentiment_positive) as avg_positive_sentiment,
                COUNT(*) as total_chunks,
                SUM(CASE WHEN sentiment_negative > 0.7 THEN 1 ELSE 0 END) as high_negative_count
            FROM `your_filings_dataset.{filing_type}` # add your bigquery dataset name for filings
            WHERE bank = '{filing_bank_code}'
            AND year BETWEEN {start_year} AND {end_year}
            GROUP BY year, quarter
            ORDER BY year, quarter
            """

        return self.client.query(query).to_dataframe()

    def find_risk_mentions(self, filing_type, bank_code, year=2008, quarter='Q1'):
        
        filing_bank_code = 'boa' if bank_code == 'bac' else bank_code

        if filing_type == '10k':
            query = f"""
            SELECT
                text_chunk,
                sentiment_negative,
                section
            FROM `your_filings_dataset.{filing_type}` # add your bigquery dataset name for filings
            WHERE bank = '{filing_bank_code}'
            AND year = {year}
            AND sentiment_negative > 0.7
            ORDER BY sentiment_negative DESC
            LIMIT 10
            """
        else:
            query = f"""
            SELECT
                text_chunk,
                sentiment_negative,
                section
            FROM `your_filings_dataset.{filing_type}` # add your bigquery dataset name for filings
            WHERE bank = '{filing_bank_code}'
            AND year = {year}
            AND quarter = '{quarter}'
            AND sentiment_negative > 0.7
            ORDER BY sentiment_negative DESC
            LIMIT 10
            """

        return self.client.query(query).to_dataframe()

    # --- insigths generator using gemini 2.5 pro ml model on big query

    def generate_insights(self, bank_code, stock_analysis, sentiment_analysis):

        max_negative = sentiment_analysis['avg_negative_sentiment'].max()
        max_drop = stock_analysis['monthly_change'].min() * 100

        if 'quarter' in sentiment_analysis.columns:
            worst_period = sentiment_analysis.loc[sentiment_analysis['avg_negative_sentiment'].idxmax()]
            period_info = f"{worst_period.get('year', 'N/A')} {worst_period.get('quarter', '')}"
        else:
            worst_period = sentiment_analysis.loc[sentiment_analysis['avg_negative_sentiment'].idxmax()]
            period_info = f"{worst_period.get('year', 'N/A')}"

        stock_summary = stock_analysis.describe().to_json()
        sentiment_summary = sentiment_analysis.describe().to_json()

        prompt = f"""
        Analyze the financial crisis indicators for {self.banks[bank_code]}.

        Key metrics:
        - Maximum monthly stock drop: {max_drop:.2f}%
        - Peak negative sentiment in SEC filings: {max_negative:.3f}
        - Worst period: {period_info}

        Stock performance summary:
        {stock_summary}

        Sentiment analysis summary:
        {sentiment_summary}

        Please provide a comprehensive analysis including:
        1. Key findings from the data
        2. Early warning signs that were present
        3. Recommendations for future monitoring

        Focus on risk management failures, liquidity issues, and regulatory concerns.
        """

        query = """
                  SELECT
                     ml_generate_text_result AS generated_insights
                  FROM
                      ML.GENERATE_TEXT(
                      MODEL `your-gcp-project-id.your_model_dataset.your_model_name`, # add your model dataset and model name
                      (SELECT @prompt_text AS prompt),
                      STRUCT(0.2 AS temperature, 8192 AS max_output_tokens, 0.8 AS top_p, 40 AS top_k)
                      )
                """

        job_config = QueryJobConfig(
            query_parameters=[
                ScalarQueryParameter("prompt_text", "STRING", prompt)
            ]
        )

        query_job = self.client.query(query, job_config=job_config)
        results = query_job.result()

        for row in results:
            return row.generated_insights

    # --- incase the model fails generates prebuilt insigths ---

    def _generate_fallback_insights(self, bank_code, max_negative, max_drop, period_info):
        
        insights = f"""
        CRISIS ANALYSIS FOR {self.banks[bank_code].upper()}

        Key Findings:
        1. Stock Performance: The stock experienced a maximum monthly drop of {max_drop:.2f}%
        2. Negative Sentiment: Peak negative sentiment in filings reached {max_negative:.3f}

        The worst period was {period_info}
        with an average negative sentiment score of {max_negative:.3f}.

        Early Warning Signs:
        - Increasing negative sentiment in SEC filings often preceded stock price declines
        - Risk disclosures related to mortgage-backed securities and credit default swaps
        - Mentions of liquidity concerns and counterparty risks

        Recommendations for Future Monitoring:
        1. Implement real-time sentiment analysis of financial disclosures
        2. Create alerts for sharp increases in negative sentiment
        3. Correlate sentiment trends with stock performance indicators
        4. Monitor specific risk factors mentioned in filings
        """

        return insights

# --- stock trend visualizer --- 

    def visualize_stock_trends(self, stock_data, bank_name):
        
        plt.figure(figsize=(12, 6))
        plt.plot(stock_data['date'], stock_data['close'])
        plt.title(f'{bank_name} Stock Price (2006-2009)')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.bar(stock_data['date'], stock_data['monthly_change'] * 100)
        plt.title(f'{bank_name} Monthly Price Changes (%)')
        plt.xlabel('Date')
        plt.ylabel('Change (%)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def visualize_sentiment_trends(self, sentiment_data, bank_name, filing_type):
        
        if 'quarter' in sentiment_data.columns:
            sentiment_data['period'] = sentiment_data['year'].astype(str) + ' ' + sentiment_data['quarter']
        else:
            sentiment_data['period'] = sentiment_data['year'].astype(str)

        plt.figure(figsize=(12, 6))
        plt.plot(sentiment_data['period'], sentiment_data['avg_negative_sentiment'],
                marker='o', label='Negative Sentiment')
        if 'avg_positive_sentiment' in sentiment_data.columns:
            plt.plot(sentiment_data['period'], sentiment_data['avg_positive_sentiment'],
                    marker='o', label='Positive Sentiment')
        plt.title(f'{bank_name} {filing_type} Sentiment Trends')
        plt.xlabel('Period')
        plt.ylabel('Sentiment Score')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # --- crisis reporter for the selected bank ---

    def generate_crisis_report(self, bank_code):
        
        print(f"Analyzing {self.banks[bank_code]} for financial crisis indicators...")

        print("1. Analyzing stock performance...")
        stock_analysis = self.analyze_stock_trends(bank_code)
        self.visualize_stock_trends(stock_analysis, self.banks[bank_code])

        print("2. Analyzing annual report (10-K) sentiment trends...")
        k_analysis = self.analyze_sentiment_trends('10k', bank_code)
        self.visualize_sentiment_trends(k_analysis, self.banks[bank_code], '10-K')

        print("3. Analyzing quarterly report (10-Q) sentiment trends...")
        q_analysis = self.analyze_sentiment_trends('10q', bank_code)
        self.visualize_sentiment_trends(q_analysis, self.banks[bank_code], '10-Q')

        print("4. Identifying specific risk mentions...")
        risk_mentions = self.find_risk_mentions('10q', bank_code, 2008, 'Q1')
        if not risk_mentions.empty:
            print("Top risk mentions from 2008 Q1:")
            for i, row in risk_mentions.iterrows():
                print(f"{i+1}. {row['text_chunk'][:200]}... (Section: {row['section']}, Negative: {row['sentiment_negative']:.3f})")

        print("5. Generating comprehensive insights...")
        insights = self.generate_insights(bank_code, stock_analysis, q_analysis)

        return insights

# --- main functions ---

def main():

    analyzer = FinancialCrisisAnalyzer(client)

    print("Welcome to the Financial Crisis Analyzer.")
    print("Please choose a bank to analyze:")

    bank_options = list(analyzer.banks.items())

    for i, (code, name) in enumerate(bank_options):
        print(f"{i + 1}. {name} ({code.upper()})")

    bank_name = ""
    selected_bank_code = ""
    while True:
        choice = input(f"\nPlease enter the number of your choice (1-{len(bank_options)}): ")
        try:
            selected_index = int(choice) - 1
            if 0 <= selected_index < len(bank_options):
                selected_bank_code, bank_name = bank_options[selected_index]
                break
            else:
                print(f"Invalid number. Please enter a number between 1 and {len(bank_options)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")


    print(f"\nStarting analysis for {bank_name}...")

    insights = analyzer.generate_crisis_report(selected_bank_code)

    print(f"\n--- {bank_name} Crisis Analysis Insights ---")
    
    if isinstance(insights, dict) and 'candidates' in insights:
        clean_text = insights['candidates'][0]['content']['parts'][0]['text']
        print(clean_text)
    else:
        print(insights)

if __name__ == "__main__":
    main()



