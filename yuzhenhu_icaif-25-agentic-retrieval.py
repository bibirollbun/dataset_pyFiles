import os

# List all  model folders in the input directory
input_path = '/kaggle/input/finetuned-retrieval-pipeline-3-models'

for folder in os.listdir(input_path):
    folder_path = os.path.join(input_path, folder)
    if os.path.isdir(folder_path):
        print(folder)



# --- Config ---
DOC_EVAL_PATH   = "document_ranking_kaggle_eval.jsonl"
CHUNK_EVAL_PATH = "chunk_ranking_kaggle_eval.jsonl"
#SUBMISSION_PATH = "submission.csv"

COMPETITION_NAME = "agentic_retrieval_grand_challenge"



# Updated to use Kaggle dataset paths
DOC_MODEL_DIR = "/kaggle/input/finetuned-retrieval-pipeline-3-models/finetuned-document-ranking"
PRETRAINED_BIENC = "/kaggle/input/finetuned-retrieval-pipeline-3-models/finetuned-chunk-retriever"
XENC_DIR = "/kaggle/input/finetuned-retrieval-pipeline-3-models/finetuned-chunk-reranker"


MAX_SEQ_LEN = 256 # for document ranking biencoder
TOP_K       = 5

# Stage 1 retrieval (mirror evaluate_hybrid_retriever_rrf)
CANDIDATE_CHUNK_DEPTH   = 140
MAX_SUBCHUNKS_PER_CHUNK = 5
RRF_K                   = 60
CAND_PER_SYSTEM         = None
MAX_LEN                 = 400
OVERLAP                 = 50

BATCH_SIZE_RETR         = 128   
BATCH_SIZE_RERANK       = 32
#BATCH_SIZE_RETR         = 4   # Reduced from 128 to run on cpu
#BATCH_SIZE_RERANK       = 16   # Reduced from 32 to run on cpu



# --- Config ---
INPUT_PATH = "/kaggle/input/acm-icaif-25-ai-agentic-retrieval-grand-challenge"

DOC_EVAL_PATH = f"{INPUT_PATH}/document_ranking_kaggle_eval.jsonl"
CHUNK_EVAL_PATH = f"{INPUT_PATH}/chunk_ranking_kaggle_eval.jsonl"



# Updated to use Kaggle dataset paths
DOC_MODEL_DIR = "/kaggle/input/finetuned-retrieval-pipeline-3-models/finetuned-document-ranking"
PRETRAINED_BIENC = "/kaggle/input/finetuned-retrieval-pipeline-3-models/finetuned-chunk-retriever"
XENC_DIR = "/kaggle/input/finetuned-retrieval-pipeline-3-models/finetuned-chunk-reranker"


MAX_SEQ_LEN = 256 # for document ranking biencoder
TOP_K       = 5

# Stage 1 retrieval (mirror evaluate_hybrid_retriever_rrf)
CANDIDATE_CHUNK_DEPTH   = 140
MAX_SUBCHUNKS_PER_CHUNK = 5
RRF_K                   = 60
CAND_PER_SYSTEM         = None
MAX_LEN                 = 400
OVERLAP                 = 50
BATCH_SIZE_RETR         = 1
BATCH_SIZE_RERANK       = 32



!pip install rank-bm25


import re
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any
#from tqdm import tqdm
from tqdm.auto import tqdm

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from transformers import AutoTokenizer
from rank_bm25 import BM25L





def load_jsonl(path: str):
    # Load a UTF-8 JSONL file into a list of dictionaries.
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

## extract questions from message content
def extract_question(content: str) -> str:
    # Extract question text from message, preferring the section between "Question:" and "Text chunks:"
    text = content.strip()
    match = re.search(r"Question:\s*(.*?)\s*Text chunks:", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    all_matches = re.findall(r"Question:\s*(.*)", text)
    if all_matches:
        return all_matches[-1].strip()
    

# extract chunks from message as list
def extract_chunks_as_strings(content: str):
    # Extract all text chunks as list of strings, ignoring chunk indices
    parts = content.split("Text chunks:", 1)
    if len(parts) < 2:
        return []
    chunks_text = parts[1].strip()
    pattern = re.compile(r"\[Chunk Index (\d+)\]\s*(.*?)(?=\[Chunk Index \d+\]|\Z)", re.DOTALL)
    chunks = []
    for match in pattern.finditer(chunks_text):
        chunks.append(match.group(2).strip())
    return chunks


def finance_tokenizer(text: str):
    raw = _FINANCE_PATTERN.findall(text)
    tokens = []
    for tok in raw:
        # handle optional capture group (tickers)
        if isinstance(tok, tuple):
            tok = next(t for t in tok if t)
        # preserve uppercase tickers; lowercase everything else
        tokens.append(tok if _TICKER_FULL.fullmatch(tok) else tok.lower())
    return tokens

# split text into <= max_len tokens with overlap
def chunk_text(text, tokenizer, max_len=400, overlap=80):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_len, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens,skip_special_tokens=True)
        chunks.append(chunk_text)
        if end == len(tokens):
            break
        start = end - overlap
    return chunks


# Specific patterns first, generic last
_FINANCE_PATTERN = re.compile(r"""
    \$\d{1,3}(?:,\d{3})*(?:\.\d+)?         # $1,000.50
  | \d+(?:\.\d+)?%                         # 12.4%
  | (?<![A-Za-z0-9])([A-Z]{1,5})(?![A-Za-z0-9])  # Tickers: AAPL, TSLA (whole token)
  | \b\d{1,2}-[A-Za-z]+\b                  # SEC forms: 10-K, 8-K, 20-F, S-1, etc.
  | [A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)*       # Words (keep hyphenated compounds)
""", re.VERBOSE)

_TICKER_FULL = re.compile(r"^[A-Z]{1,5}$")

def scores_to_ranks_desc(scores: np.ndarray) -> np.ndarray:
    """
    Convert scores to 1-based ranks in descending order (higher score = better rank).
    Handles NaNs by treating them as -inf (worst).
    Ties get the same rank as their first occurrence (stable ranking by order).
    """
    if scores.ndim != 1:
        scores = scores.ravel()

    # Treat NaNs as -inf so they go to the bottom
    scores = np.where(np.isnan(scores), -np.inf, scores)

    order = np.argsort(-scores, kind="mergesort")  # stable sort
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, scores.shape[0] + 1, dtype=np.int32)
    return ranks



# Must match training order
DOC_TYPE_TEXTS = [
    "DEF14A Proxy Statement: executive compensation, board governance, and shareholder proposals.",
    "10-K Annual Report: audited yearly SEC filing with comprehensive financials and risk factors.",
    "10-Q Quarterly Report: unaudited quarterly results, MD&A, and updated risk disclosures.",
    "8-K Current Report: ad-hoc disclosure of material events and earnings releases.",
    "Earnings Report: company-issued press release & earnings call highlights with KPIs and outlook."
]

def generate_document_submission(doc_eval_path: str,
                                 doc_model_dir: str,
                                 max_seq_len: int = 256,
                                 top_k: int = 5) -> pd.DataFrame:
    model = SentenceTransformer(doc_model_dir)
    model.max_seq_length = max_seq_len
    device = model.device

    type_embs = model.encode(
        DOC_TYPE_TEXTS,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        device=device
    )
    
    records = load_jsonl(doc_eval_path)
    rows = []
    for rec in tqdm(records, desc="Document ranking"):
        sample_id = rec["_id"]
        q = extract_question(rec["messages"][0]["content"])
        q_emb = model.encode(q, convert_to_tensor=True, normalize_embeddings=True,
                             show_progress_bar=False, device=device)
        sims = util.cos_sim(q_emb, type_embs).flatten()
        ranked = torch.argsort(sims, descending=True).tolist()
        for idx in ranked[:top_k]:
            rows.append({"sample_id": sample_id, "target_index": idx})

    return pd.DataFrame(rows)



    doc_df = generate_document_submission(
        doc_eval_path=DOC_EVAL_PATH,
        doc_model_dir=DOC_MODEL_DIR,
        max_seq_len=MAX_SEQ_LEN,
        top_k=TOP_K
    )



doc_df


def stage1_candidates_from_rrf(chunk_eval_path: str,
                               pretrained_biencoder_name: str,
                               rrf_k: int = 60,
                               cand_per_system: int | None = None,
                               candidate_chunk_depth: int = 80,
                               max_subchunks_per_chunk: int = 5,
                               max_len: int = 400,
                               overlap: int = 80,
                               batch_size: int = 128,
                               max_seq_len: int = 256) -> List[Dict[str, Any]]:
    dense_model = SentenceTransformer(pretrained_biencoder_name)
    dense_model.max_seq_length = max_seq_len
    device = dense_model.device
    hf_model_name = dense_model[0].auto_model.config._name_or_path
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

    data = load_jsonl(chunk_eval_path)
    pool: List[Dict[str, Any]] = []

    for ex in tqdm(data, desc="Stage 1: Hybrid RRF"):
        sample_id = ex["_id"]
        content   = ex["messages"][0]["content"]
        query     = extract_question(content)
        chunks    = extract_chunks_as_strings(content)
        if not query or not chunks:
            continue

        # Sub-chunking
        subchunks, sub2parent = [], []
        for cid, ch in enumerate(chunks):
            pieces = chunk_text(ch, tokenizer, max_len=max_len, overlap=overlap)
            subchunks.extend(pieces)
            sub2parent.extend([cid] * len(pieces))
        if not subchunks:
            continue

        # Dense scores
        embs = dense_model.encode([query] + subchunks, convert_to_tensor=True,
                                  show_progress_bar=False, device=device, batch_size=batch_size)
        q_emb, sub_embs = embs[0], embs[1:]
        dense_scores_all = util.cos_sim(q_emb, sub_embs)[0].cpu().numpy()

        # BM25L scores (finance tokenizer)
        tokenized_sub = [finance_tokenizer(sc) for sc in subchunks]
        tokenized_q   = finance_tokenizer(query)
        bm25 = BM25L(tokenized_sub, k1=1.2, b=0.9, delta=0.5)
        bm25_scores_all = bm25.get_scores(tokenized_q)

        # Candidate prefilter per system (optional)
        if cand_per_system is not None:
            dense_top = np.argsort(-dense_scores_all)[:cand_per_system]
            bm25_top  = np.argsort(-bm25_scores_all)[:cand_per_system]
            keep = np.unique(np.concatenate([dense_top, bm25_top]))
        else:
            keep = np.arange(len(subchunks))

        # RRF fusion (mirror of evaluate_hybrid_retriever_rrf)
        dense_rank = scores_to_ranks_desc(dense_scores_all[keep])
        bm25_rank  = scores_to_ranks_desc(bm25_scores_all[keep])
        rrf_scores_sub = 1.0/(rrf_k + dense_rank) + 1.0/(rrf_k + bm25_rank)
        ranked_sub = sorted(zip(keep, rrf_scores_sub), key=lambda x: x[1], reverse=True)

        # Collapse to parent chunk via max fused score
        chunk_scores = {}
        for sidx, s in ranked_sub:
            pid = sub2parent[sidx]
            if (pid not in chunk_scores) or (s > chunk_scores[pid]):
                chunk_scores[pid] = s

        # Keep top-N parent chunks
        top_parents = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:candidate_chunk_depth]
        keep_parent_ids = {pid for pid, _ in top_parents}

        # Up to M subchunks per kept parent
        selected_sub, per_parent = [], defaultdict(int)
        for sidx, s in ranked_sub:
            pid = sub2parent[sidx]
            if pid in keep_parent_ids and per_parent[pid] < max_subchunks_per_chunk:
                selected_sub.append((sidx, pid))
                per_parent[pid] += 1

        pool.append({
            "sample_id": sample_id,
            "query": query,
            "candidates": [subchunks[sid] for sid, _ in selected_sub],
            "parent_chunk_ids": [pid for _, pid in selected_sub]  # aligned by position with candidates
        })

    return pool



    # Chunk Stage 1 (pretrained bi-encoder + BM25L + RRF; mirrors evaluate_hybrid_retriever_rrf)
    pool = stage1_candidates_from_rrf(
        chunk_eval_path=CHUNK_EVAL_PATH,
        pretrained_biencoder_name=PRETRAINED_BIENC,
        rrf_k=RRF_K,
        cand_per_system=CAND_PER_SYSTEM,
        candidate_chunk_depth=CANDIDATE_CHUNK_DEPTH,
        max_subchunks_per_chunk=MAX_SUBCHUNKS_PER_CHUNK,
        max_len=MAX_LEN,
        overlap=OVERLAP,
        batch_size=BATCH_SIZE_RETR,
        max_seq_len=MAX_SEQ_LEN
    )




def stage2_rerank_and_submission(candidate_pool: List[Dict[str, Any]],
                                 cross_encoder_dir: str,
                                 top_k: int = 5,
                                 batch_size: int = 32) -> pd.DataFrame:
    reranker = CrossEncoder(cross_encoder_dir)
    rows = []
    
    for item in tqdm(candidate_pool, desc="Stage 2: Reranking", position=0, leave=True, disable=False):
        q           = item["query"]
        candidates  = item["candidates"]
        parent_cids = item["parent_chunk_ids"]
        sample_id   = item["sample_id"]
        
        pairs  = [(q, c) for c in candidates]
        scores = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        
        ranked = sorted(zip(parent_cids, scores), key=lambda x: -float(x[1]))
        
        seen, final_parent = set(), []
        for cid, sc in ranked:
            if cid not in seen:
                seen.add(cid)
                final_parent.append(cid)
            if len(final_parent) >= top_k:
                break
        
        for cid in final_parent:
            rows.append({"sample_id": sample_id, "target_index": cid})
    
    return pd.DataFrame(rows)


    # Chunk Stage 2 (fine-tuned cross-encoder)
    chunk_df = stage2_rerank_and_submission(
        candidate_pool=pool,
        cross_encoder_dir=XENC_DIR,
        top_k=TOP_K,
        batch_size=BATCH_SIZE_RERANK
    )



chunk_df


import os

# Save to current directory
SUBMISSION_PATH = "kaggle_submission.csv"

# Merge and write
final_df = pd.concat([chunk_df, doc_df], ignore_index=True)
final_df.to_csv(SUBMISSION_PATH, index=False)
print(f"Saved submission: {SUBMISSION_PATH}")

