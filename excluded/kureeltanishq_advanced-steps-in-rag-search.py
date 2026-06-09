# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install -q sentence-transformers faiss-cpu transformers accelerate pypdf rank_bm25 rouge-score


import os
import json
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity



DATA_DIR = Path("/kaggle/input/casml-generative-ai-hackathon/Dataset_RAG (1)")  
pdf_path = DATA_DIR / "book.pdf"
queries_path = DATA_DIR / "queries.json"

OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(exist_ok=True)

EMB_MODEL = "sentence-transformers/all-mpnet-base-v2"   
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2" 
GEN_MODEL = "google/flan-t5-small"                      



reader = PdfReader(str(pdf_path))
pages = []
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if not text:
        text = ""
    text = re.sub(r'\n+', ' ', text).strip()
    pages.append({"page": i+1, "text": text})
df_pages = pd.DataFrame(pages)
print("Pages parsed:", len(df_pages))
df_pages.head(3)


def chunk_text(text, chunk_size=200, overlap=50):
    tokens = text.split()
    if len(tokens) == 0:
        return []
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = " ".join(tokens[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

corpus_rows = []
for _, r in df_pages.iterrows():
    chs = chunk_text(r["text"], chunk_size=250, overlap=60)
    if len(chs) == 0:
        corpus_rows.append({"page": r["page"], "chunk_id": 0, "text": r["text"]})
    else:
        for ci, ch in enumerate(chs):
            corpus_rows.append({"page": r["page"], "chunk_id": ci, "text": ch})
corpus = pd.DataFrame(corpus_rows).reset_index(drop=True)
print("Corpus chunks:", len(corpus))
corpus.head()


tokenized_corpus = [re.findall(r"\w+|\S", t.lower()) for t in corpus['text'].tolist()]
bm25 = BM25Okapi(tokenized_corpus)


emb_cache = OUT_DIR / "embeddings.npy"
index_file = OUT_DIR / "faiss.index"

if emb_cache.exists() and Path(index_file).exists():
    print("Loading cached embeddings & FAISS index...")
    embeddings = np.load(str(emb_cache))
    index = faiss.read_index(str(index_file))
else:
    print("Computing embeddings (this can take time)...")
    embedder = SentenceTransformer(EMB_MODEL)
    texts = corpus['text'].tolist()
    embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    np.save(str(emb_cache), embeddings)
    faiss.write_index(index, str(index_file))
    print("Saved embeddings and index.")
if 'embedder' not in globals():
    embedder = SentenceTransformer(EMB_MODEL)


reranker = CrossEncoder(RERANKER_MODEL)
tokenizer_gen = AutoTokenizer.from_pretrained(GEN_MODEL)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)


def hybrid_retrieve(query, top_k_dense=50, top_k_bm25=50, alpha=0.5, candidate_pool=100):
    q_tokens = re.findall(r"\w+|\S", query.lower())
    bm25_scores = bm25.get_scores(q_tokens)  
    bm25_top = np.argsort(bm25_scores)[-top_k_bm25:][::-1]
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k_dense)
    dense_idx = I[0].tolist()
    dense_scores = {idx: float(D[0][i]) for i, idx in enumerate(dense_idx)}
    bm25_top_pool = np.argsort(bm25_scores)[-candidate_pool:]
    candidates = set(dense_idx) | set(bm25_top_pool.tolist())
    bm_vals = np.array([bm25_scores[i] for i in candidates], dtype=float)
    if bm_vals.max() - bm_vals.min() > 0:
        bm_norm_map = {idx: (bm25_scores[idx] - bm_vals.min()) / (bm_vals.max() - bm_vals.min()) for idx in candidates}
    else:
        bm_norm_map = {idx: 0.0 for idx in candidates}
    combined = []
    for idx in candidates:
        dense_s = dense_scores.get(idx, 0.0)
        bm_s = bm_norm_map.get(idx, 0.0)
        combined_score = alpha * dense_s + (1 - alpha) * bm_s
        combined.append((idx, combined_score))
    combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)
    hits = []
    for idx, sc in combined_sorted[:candidate_pool]:
        row = corpus.iloc[idx]
        hits.append({"idx": int(idx), "score": float(sc), "text": row['text'], "page": int(row['page'])})
    return hits



def rerank_with_crossencoder(query, hits, top_n=5):
    if len(hits) == 0:
        return []
    pairs = [(query, h['text']) for h in hits]
    scores = reranker.predict(pairs)  
    for i, s in enumerate(scores):
        hits[i]['rerank_score'] = float(s)
    hits_sorted = sorted(hits, key=lambda x: x['rerank_score'], reverse=True)
    return hits_sorted[:top_n]


vectorizer = CountVectorizer().fit([" ".join(tokenized_corpus[0])] + [" ".join(x) for x in tokenized_corpus])  

def extractive_answer(query, top_hits, overlap_threshold=0.18):
    sentences = []
    page_meta = []
    for h in top_hits:
        # basic sentence split
        for sent in re.split(r'(?<=[.!?]) +', h['text']):
            s = sent.strip()
            if len(s) < 10:
                continue
            sentences.append(s)
            page_meta.append(h['page'])
    if len(sentences) == 0:
        return None, []
    docs = [query] + sentences
    vecs = vectorizer.transform(docs).toarray()
    sim = cosine_similarity(vecs[0:1], vecs[1:]).flatten()
    best_idx = int(sim.argmax())
    if sim[best_idx] >= overlap_threshold:
        return sentences[best_idx].strip(), page_meta[best_idx]
    return None, []


def generate_faithful(query, top_hits, max_len=120):
    context = "\n\n".join([h['text'] for h in top_hits])
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer is not present in the context, respond with EXACTLY: NOT_IN_TEXT\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    inputs = tokenizer_gen(prompt, return_tensors="pt", truncation=True, max_length=512)
    out = gen_model.generate(**inputs, max_length=max_len, num_beams=3)
    ans = tokenizer_gen.decode(out[0], skip_special_tokens=True)
    return ans


def answer_query(query, params):
    hits = hybrid_retrieve(query,
                           top_k_dense=params.get("top_k_dense", 50),
                           top_k_bm25=params.get("top_k_bm25", 50),
                           alpha=params.get("alpha", 0.5),
                           candidate_pool=params.get("candidate_pool", 100))
    reranked = rerank_with_crossencoder(query, hits, top_n=params.get("rerank_top_n", 5))
    ext_ans, ext_page = extractive_answer(query, reranked, overlap_threshold=params.get("overlap_threshold", 0.18))
    if ext_ans:
        final_answer = ext_ans
        used_hits = reranked[:3]  
    else:
        final_answer = generate_faithful(query, reranked[:params.get("gen_context_k", 3)], max_len=params.get("gen_max_len", 120))
        used_hits = reranked[:3]
    context_text = " ".join([h['text'] for h in used_hits])
    pages = list(dict.fromkeys([str(h['page']) for h in used_hits]))
    references = {"sections": ["unknown"], "pages": pages}
    return final_answer, context_text, references, used_hits


with open(queries_path, "r", encoding="utf-8") as f:
    queries = json.load(f)

print("Total queries loaded:", len(queries))
queries[:2]


params = default_params
try:
    if 'best_alpha' in locals() and best_alpha.get("alpha"):
        params["alpha"] = best_alpha["alpha"]
except Exception:
    pass

rows = []
for q in tqdm(queries):
    qid = q["query_id"]
    question = q["question"]
    ans, ctx, refs, used_hits = answer_query(question, params)
    # format references JSON string
    refs_json = json.dumps(refs, ensure_ascii=False)
    rows.append({"ID": qid, "context": ctx, "answer": ans, "references": refs_json})

submission = pd.DataFrame(rows)
# reorder to exact header: ID,context,answer,references
submission = submission[["ID", "context", "answer", "references"]]
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv with", len(submission), "rows.")
submission.head(3)


