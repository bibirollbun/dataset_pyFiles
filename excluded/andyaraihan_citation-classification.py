pip install pymupdf


import os
import re
from pathlib import Path
from typing import List, Dict
import fitz  # PyMuPDF
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from lxml import etree
import warnings
import logging

# Setup logging dasar untuk debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mengatur konfigurasi alokasi memori CUDA untuk mencegah fragmentasi
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Mengabaikan peringatan yang tidak relevan
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. KONFIGURASI DAN KONSTANTA
# ==============================================================================
# Jalur direktori utama untuk lingkungan Kaggle
INPUT_DIR = Path("/kaggle/input/make-data-count-finding-data-references")
PDF_TRAIN_DIR = INPUT_DIR / "train/PDF"
XML_TRAIN_DIR = INPUT_DIR / "train/XML"
TRAIN_LABELS_PATH = INPUT_DIR / "train_labels.csv"
PDF_TEST_DIR = INPUT_DIR / "test/PDF"
XML_TEST_DIR = INPUT_DIR / "test/XML"

# --- Konfigurasi Model dan Efisiensi ---
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4  # Ditingkatkan untuk efisiensi, turunkan jika terjadi Out-of-Memory
RANDOM_STATE = 42

# Prefix DOI yang dianggap sebagai dataset valid
VALID_DATASET_PREFIXES = ["10.5061/dryad", "10.5281/zenodo", "10.25386/genetics", "10.7937"]

# ==============================================================================
# 2. REGULAR EXPRESSIONS (REGEX)
# ==============================================================================
RE_DOI = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.IGNORECASE)
ACCESSION_PATTERNS = [
    "GSE\d+", "SR[APRX]\d+", "PRJ[NAED][A-Z]?\d+", "EPI(?:_ISL_)?\d+",
    "PXD\d{6}", "SAM[ND]\d+", "ERR\d+", "PDB\s+[A-Z0-9]+", "E-MTAB-\d+",
    "IPR\d{6}", "PF\d{5}", "EMPIAR-\d{5}", "CHEMBL\d+", "CVCL_[A-Z0-9]{4}",
    "ENS[A-Z]{0,6}[GT]\d{11}", "N[MC]_\d+(?:\.\d+)?", "rs\d+",
    "uniprot:(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9])",
]
RE_ACCESSION = re.compile(r"\b(" + "|".join(ACCESSION_PATTERNS) + r")\b", re.IGNORECASE)
RE_REFERENCES_SECTION = re.compile(
    r"^(REFERENCES?|BIBLIOGRAPHY|Literature\s+Cited|Works\s+Cited|\d+\.?\s+(REFERENCES?|Bibliography))(:)?$",
    re.IGNORECASE | re.MULTILINE
)
RE_CITATION_PATTERNS = [
    r'\(\d{4}\)',      # (2020)
    r'\d{4}\.',        # 2020.
    r'doi:',           # doi:
    r'\bet al\b',      # et al
]
# Regex yang lebih fleksibel, mencari huruf A, B, atau C di mana saja, case-insensitive
RE_CLEAN_LLM_OUTPUT = re.compile(r"([ABC])", re.IGNORECASE)


# ==============================================================================
# 3. FUNGSI EKSTRAKSI DATA
# ==============================================================================
def normalize_doi(doi: str) -> str:
    doi = doi.strip().lower()
    if not doi.startswith("https://doi.org/"):
        doi = doi.lstrip("doi:").strip()
        if doi.startswith("10."):
            return f"https://doi.org/{doi}"
    return doi

def is_valid_dataset_doi(doi: str) -> bool:
    return any(doi.lower().startswith(prefix) for prefix in VALID_DATASET_PREFIXES)

def remove_references_section(text: str) -> str:
    lines = text.split('\n')
    cut_index = -1
    for i in range(len(lines) - 1, int(len(lines) * 0.7), -1):
        line = lines[i].strip()
        if RE_REFERENCES_SECTION.match(line):
            following_lines = lines[i+1:i+4]
            has_citations = False
            for follow_line in following_lines:
                if follow_line.strip() and any(re.search(pat, follow_line, re.IGNORECASE) for pat in RE_CITATION_PATTERNS):
                    has_citations = True
                    break
            if has_citations or i >= len(lines) - 3:
                cut_index = i
                break
    if cut_index != -1:
        ref_section = '\n'.join(lines[cut_index:])
        if RE_DOI.search(ref_section) or RE_ACCESSION.search(ref_section):
            return text
        return '\n'.join(lines[:cut_index]).strip()
    return text.strip()

def extract_text_from_xml(xml_path: Path) -> str:
    try:
        tree = etree.parse(str(xml_path))
        sections = tree.xpath("//sec[contains(@sec-type, 'method') or contains(@sec-type, 'data') or contains(@sec-type, 'supplementary')]//p//text()")
        text = "\n".join(sections) if sections else ""
        return text
    except Exception as e:
        logger.error(f"Gagal memproses XML {xml_path}: {e}")
        return ""

def find_potential_citations(text: str, article_id: str) -> List[Dict[str, str]]:
    citations = []
    paragraphs = text.split("\n\n")
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        patterns = {"doi": RE_DOI, "accession": RE_ACCESSION}
        for source, pattern in patterns.items():
            for match in pattern.finditer(para):
                dataset_id = match.group(0)
                if source == "doi":
                    dataset_id = normalize_doi(dataset_id)
                    if not is_valid_dataset_doi(dataset_id) or dataset_id == f"https://doi.org/{article_id}":
                        continue
                citations.append({
                    "article_id": article_id,
                    "text": para,
                    "dataset_id": dataset_id,
                    "source": source
                })
    return citations

def extract_chunks_from_paths(pdf_paths: List[Path], xml_dir: Path) -> pd.DataFrame:
    all_chunks = []
    print(f"Memulai ekstraksi dari {len(pdf_paths)} file...")
    for pdf_path in tqdm(pdf_paths, desc="ğŸ“„ Mengekstrak Teks & ID"):
        article_id = pdf_path.stem
        try:
            xml_path = xml_dir / f"{article_id}.xml"
            if xml_path.exists():
                full_text = extract_text_from_xml(xml_path)
            else:
                with fitz.open(pdf_path) as doc:
                    full_text = "\n".join([page.get_text("text") for page in doc])
            
            cleaned_text = remove_references_section(full_text)
            chunks = find_potential_citations(cleaned_text, article_id)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Gagal memproses file {article_id}: {e}")
            
    df_chunks = pd.DataFrame(all_chunks)
    logger.info(f"Total chunk kandidat yang diekstrak: {len(df_chunks)}")
    return df_chunks

# ==============================================================================
# 4. FUNGSI KLASIFIKASI LLM
# ==============================================================================
def build_prompt_messages(batch_df: pd.DataFrame) -> List[Dict[str, str]]:
    system_message = """You are an expert research analyst specializing in scientific data citation. Your task is to classify the relationship between a scientific paper and a dataset ID mentioned in a text snippet. The classification must be based on the context provided, focusing on specific linguistic cues.

Classify as follows:
- **Primary (A)**: The dataset was generated by the authors for this study. Look for explicit phrases indicating the authors created or deposited the data, such as:
  - "data are available at", "we generated", "our data have been deposited in", "data for this study", "deposited at", "available at", "supplemental material", "our dataset", "we collected".
- **Secondary (B)**: The dataset was reused from an external source or previous study. Look for phrases indicating the data was sourced externally, such as:
  - "data were obtained from", "retrieved from", "we used the dataset from", "downloaded from", "sourced from", "accessed from", "obtained from repository".
- **None (C)**: The ID refers to something else (e.g., another publication, software, or unclear context), is mentioned in passing, or lacks sufficient context to determine A or B.

Rules:
1. Focus on the exact phrasing in the context to determine the relationship.
2. If the context is ambiguous or lacks explicit cues, default to C.
3. Return only a single letter (A, B, or C) for each snippet, one per line.

Your final output must ONLY be the letters, one per line. Do not add numbers, explanations, or any other text.
"""
    user_prompts = []
    for i, row in enumerate(batch_df.itertuples(), 1):
        snippet = ' '.join(str(row.text).split())[:700]
        user_prompts.append(f"{i}. ID: {row.dataset_id}\n   Context: \"...{snippet}...\"")
        
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": "\n\n".join(user_prompts)}
    ]
    return messages

def classify_batch_with_llm(batch_df: pd.DataFrame, model, tokenizer) -> List[str]:
    messages = build_prompt_messages(batch_df)
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=20 * len(batch_df),
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        decoded_output = tokenizer.decode(outputs[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Baris debugging untuk melihat output mentah dari LLM
        print(f"\n--- Raw LLM Output ---\n{decoded_output}\n----------------------")
        
        labels = RE_CLEAN_LLM_OUTPUT.findall(decoded_output)
        
        return labels + ["C"] * (len(batch_df) - len(labels))
    except Exception as e:
        logger.error(f"Error dalam klasifikasi LLM: {e}")
        return ["C"] * len(batch_df)

def run_llm_classification(df_chunks: pd.DataFrame, model, tokenizer) -> pd.DataFrame:
    results = []
    print(f"\nMemulai klasifikasi dengan LLM untuk {len(df_chunks)} chunk...")
    for i in tqdm(range(0, len(df_chunks), BATCH_SIZE), desc="ğŸ¤– Mengklasifikasi"):
        batch_df = df_chunks.iloc[i:i+BATCH_SIZE].reset_index(drop=True)
        labels = classify_batch_with_llm(batch_df, model, tokenizer)
        
        for j, label_code in enumerate(labels):
            if j < len(batch_df) and label_code.upper() in ["A", "B"]:
                row = batch_df.iloc[j]
                results.append({
                    "article_id": row.article_id,
                    "dataset_id": row.dataset_id,
                    "type": "Primary" if label_code.upper() == "A" else "Secondary"
                })
    
    if not results:
        return pd.DataFrame(columns=['article_id', 'dataset_id', 'type'])

    df_results = pd.DataFrame(results)
    logger.info(f"Menghasilkan {len(df_results)} prediksi (Primary/Secondary)")
    return df_results.drop_duplicates(subset=['article_id', 'dataset_id']).reset_index(drop=True)

# ==============================================================================
# 5. FUNGSI EVALUASI
# ==============================================================================
def calculate_f1_score(true_labels: pd.DataFrame, pred_labels: pd.DataFrame):
    print("\n" + "="*25)
    print("--- HASIL EVALUASI ---")
    print("="*25)
    
    if pred_labels.empty:
        print("âš ï¸� Tidak ada prediksi valid (A/B) yang dihasilkan. F1-Score adalah 0.")
        logger.warning("No valid predictions generated, F1 is 0.")
        return 0
        
    true_labels_norm = true_labels[['article_id', 'dataset_id', 'type']].astype(str).apply(lambda x: x.str.lower())
    pred_labels_norm = pred_labels[['article_id', 'dataset_id', 'type']].astype(str).apply(lambda x: x.str.lower())

    true_set = set(map(tuple, true_labels_norm.values))
    pred_set = set(map(tuple, pred_labels_norm.values))
    
    tp = len(true_set.intersection(pred_set))
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"ğŸ“Š True Positives (TP) : {tp}")
    print(f"ğŸ“Š False Positives (FP): {fp}")
    print(f"ğŸ“Š False Negatives (FN): {fn}")
    print("-" * 25)
    print(f"ğŸ�¯ Precision : {precision:.4f}")
    print(f"ğŸ”� Recall    : {recall:.4f}")
    print(f"â­� F1-Score  : {f1:.4f}")
    print("=" * 25)
    logger.info(f"Evaluation metrics: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    return f1

# ==============================================================================
# 6. BLOK EKSEKUSI UTAMA (MAIN)
# ==============================================================================
if __name__ == "__main__":
    # --- Langkah 1: Muat Model & Tokenizer (Hanya sekali) ---
    print(f"Memuat model {MODEL_NAME} ke {DEVICE}...")
    logger.info(f"Loading model {MODEL_NAME}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        )
        model.eval()
        print("âœ… Model berhasil dimuat.")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.critical(f"Gagal memuat model. Pastikan Anda memiliki koneksi internet dan akses ke Hugging Face. Error: {e}")
        exit()

    # --- Langkah 2: Evaluasi Kinerja pada Seluruh Data Latih ---
    print("\n--- Memulai Evaluasi pada Data Latih ---")
    df_train_labels = pd.read_csv(TRAIN_LABELS_PATH)
    train_pdf_paths = list(PDF_TRAIN_DIR.glob("*.pdf"))
    
    if train_pdf_paths:
        df_train_chunks = extract_chunks_from_paths(train_pdf_paths, XML_TRAIN_DIR)
        
        # ==> ANALISIS CHUNK DATA LATIH
        if not df_train_chunks.empty:
            print("\n--- 10 Contoh Chunk Hasil Ekstraksi (Data Latih) ---")
            print(df_train_chunks.head(10).to_string())
            print("----------------------------------------------------\n")
            
            pred_train_labels = run_llm_classification(df_train_chunks, model, tokenizer)
            f1_score = calculate_f1_score(df_train_labels, pred_train_labels)
            if f1_score < 0.60:
                print(f"âš ï¸� PERINGATAN: F1-Score ({f1_score:.4f}) di bawah target 0.60.")
            else:
                print(f"ğŸ�‰ SELAMAT: F1-Score ({f1_score:.4f}) mencapai target 0.60!")
        else:
            print("âš ï¸� Tidak ada chunk yang diekstrak dari data latih. Evaluasi dilewati.")
            logger.warning("No chunks extracted from training data. Skipping evaluation.")
    else:
        print("âš ï¸� Direktori PDF latih tidak ditemukan atau kosong.")

    # --- Langkah 3: Proses Test Set untuk Submission ---
    print("\n--- Memulai Prediksi pada Data Tes ---")
    test_pdf_paths = list(PDF_TEST_DIR.glob("*.pdf"))
    
    if not test_pdf_paths:
        print("âš ï¸� Tidak ada file PDF di folder test. Tidak dapat membuat file submission.")
    else:
        df_test_chunks = extract_chunks_from_paths(test_pdf_paths, XML_TEST_DIR)
        
        # ==> ANALISIS CHUNK DATA TES
        if not df_test_chunks.empty:
            print("\n--- 10 Contoh Chunk Hasil Ekstraksi (Data Tes) ---")
            print(df_test_chunks.head(10).to_string())
            print("--------------------------------------------------\n")
            
            pred_test_labels = run_llm_classification(df_test_chunks, model, tokenizer)
            
            if not pred_test_labels.empty:
                submission = pred_test_labels[['article_id', 'dataset_id', 'type']].copy()
                submission = submission.sort_values(by=["article_id", "dataset_id"]).drop_duplicates(subset=['article_id', 'dataset_id'], keep="first")
                submission.insert(0, 'row_id', range(len(submission)))
                
                submission.to_csv('submission.csv', index=False)
                print("\nâœ… File submission.csv telah berhasil dibuat.")
                print("Distribusi tipe dalam submission:")
                print(submission['type'].value_counts())
                logger.info("Submission file created successfully.")
            else:
                print("âš ï¸� Tidak ada prediksi (Primary/Secondary) yang dihasilkan untuk data tes.")
                pd.DataFrame(columns=['row_id', 'article_id', 'dataset_id', 'type']).to_csv('submission.csv', index=False)
                print("âœ… File submission.csv kosong telah dibuat.")
                logger.warning("No valid predictions for test set, creating empty submission file.")
        else:
            print("âš ï¸� Tidak ada chunk yang diekstrak dari data tes.")
            pd.DataFrame(columns=['row_id', 'article_id', 'dataset_id', 'type']).to_csv('submission.csv', index=False)
            print("âœ… File submission.csv kosong telah dibuat.")
            logger.warning("No chunks extracted from test set, creating empty submission file.")

    # Membersihkan memori GPU
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\nProses selesai. Sumber daya telah dilepaskan.")

