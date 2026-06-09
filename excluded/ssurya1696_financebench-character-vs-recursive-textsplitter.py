import pandas as pd
import os
import gzip
import json
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    Language,
)
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm
from typing import List, Dict, Optional
from sentence_transformers import CrossEncoder
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

import logging
logging.disable(logging.CRITICAL)


data_dir = "data"
records = []

file_path = os.path.join(data_dir, "financebench_corpus.jsonl.gz")
with gzip.open(file_path, "rt", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        data["source_file"] = "financebench_corpus.jsonl.gz"
        records.append(data)

# Convert all records to a pandas DataFrame
df = pd.DataFrame(records)


# unique records
print(len(df['_id'].unique()))

# null records
print(len(df.isna().sum()))

df.dropna(inplace=True)
df.drop_duplicates('_id',inplace=True)
print(len(df['_id'].unique()))


df.head()


CHAR_SIZES = [64, 128, 256, 368, 512]
RECURSIVE_SIZES = CHAR_SIZES
RECURSIVE_OVERLAP = 20


def rowdict_iter(df: pd.DataFrame):
    cols = list(df.columns)
    for vals in df.itertuples(index=False, name=None):
        yield dict(zip(cols, vals))

def _coerce_text(x: Optional[str]) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x)

def _emit_rows(base_row: Dict, splitter_name: str, chunk_size: int, chunk_overlap: int, chunks: List[str]) -> List[Dict]:
    return [
        {
            **base_row,
            "splitter": splitter_name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunk_index": i,
            "chunk_text": ch,
        }
        for i, ch in enumerate(chunks)
    ]

def _chunk_all_rows_with_splitter(df: pd.DataFrame, splitter, splitter_name: str, size: int, overlap: int) -> List[Dict]:
    rows: List[Dict] = []
    for rd in rowdict_iter(df):
        base = {
            "_id": rd.get("_id"),
            "title": rd.get("title"),
            "source_file": rd.get("source_file"),
        }
        text = _coerce_text(rd.get("text", ""))
        if not text:
            continue
        chunks = splitter.split_text(text)
        rows.extend(_emit_rows(base, f"{splitter_name}_{size}", size, overlap, chunks))
    return rows

def make_all_chunks_with_docs(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.name and df.index.name not in df.columns:
        df = df.reset_index()

    all_rows: List[Dict] = []
    steps = ["character", "recursive"]

    with tqdm(total=len(steps), desc="Chunking pipeline", ncols=100) as pbar:
        for size in CHAR_SIZES:
            s = CharacterTextSplitter(chunk_size=size, chunk_overlap=0)
            all_rows.extend(_chunk_all_rows_with_splitter(df, s, "character", size, 0))
        pbar.update(1)

        for size in RECURSIVE_SIZES:
            s = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=RECURSIVE_OVERLAP)
            all_rows.extend(_chunk_all_rows_with_splitter(df, s, "recursive", size, RECURSIVE_OVERLAP))
        pbar.update(1)

    chunks_df = pd.DataFrame(all_rows)
    
    cols = ["_id", "title", "source_file", "splitter", "chunk_size", "chunk_overlap", "chunk_index", "chunk_text"]
    chunks_df = chunks_df[[c for c in cols if c in chunks_df.columns] + [c for c in chunks_df.columns if c not in cols]]
    return chunks_df


df.head()


chunks_df = make_all_chunks_with_docs(df)


chunks_df.groupby("splitter").size().sort_values(ascending=True)


chunks_df.head()


PARENT_DIR = "./vectordbs"

EMBED_MODEL_NAME = "intfloat/e5-small-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"batch_size": 64}
)


def build_chroma_for_splitter(
    chunks_df: pd.DataFrame,
    splitter_name: str,
    parent_dir: str = PARENT_DIR,
) -> Chroma:
    sub = chunks_df[chunks_df["splitter"] == splitter_name].copy()
    sub = sub[sub["chunk_text"].notna() & (sub["chunk_text"].str.len() > 0)]
    texts = sub["chunk_text"].astype(str).tolist()
    metadatas: List[Dict] = sub[["_id", "source_file", "splitter", "chunk_index"]].to_dict("records")

    persist_dir = os.path.join(parent_dir, f"chroma_{splitter_name}")
    os.makedirs(persist_dir, exist_ok=True)

    db = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=persist_dir,
        collection_name=f"col_{splitter_name}"
    )
    db.persist()
    return db

def load_chroma(splitter_name: str, parent_dir: str = PARENT_DIR) -> Chroma:
    persist_dir = os.path.join(parent_dir, f"chroma_{splitter_name}")
    return Chroma(
        persist_directory=persist_dir,
        collection_name=f"col_{splitter_name}",
        embedding_function=embeddings
    )

splitters_to_build = sorted(chunks_df["splitter"].unique().tolist())

built = {}
for sp in splitters_to_build:
    print(f"ðŸ”§ Building Chroma for splitter: {sp}")
    built[sp] = build_chroma_for_splitter(chunks_df, sp, parent_dir=PARENT_DIR)


search_text = "How does Boeing's effective tax rate in FY2022 compare to FY2021?"
splitters = ["character_512", "recursive_512"]

def search_chroma(splitter):
    db = Chroma(
        persist_directory=f"{PARENT_DIR}/chroma_{splitter}",
        collection_name=f"col_{splitter}",
        embedding_function=embeddings,
    )
    retriever = db.as_retriever(search_kwargs={"k": 5})
    return retriever.get_relevant_documents(search_text)

for sp in splitters:
    print(f"\n=== {sp} ===")
    for i, d in enumerate(search_chroma(sp), 1):
        meta = d.metadata or {}
        print(f"[{i}] {meta.get('_id')} | {meta.get('source_file')} | {meta.get('chunk_index')}")
        print(d.page_content[:250].replace("\n", " ") + "...\n")


reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device="cpu")

def rerank_top_k(query, docs, top_n=5):
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:top_n]
    out = []
    for d, s in ranked:
        meta = d.metadata or {}
        out.append({
            "score": float(s),
            "_id": meta.get("_id"),
            "source_file": meta.get("source_file"),
            "chunk_index": meta.get("chunk_index"),
            "text": d.page_content
        })
    return out

for sp in splitters: 
    print(f"\n===== {sp} | Reranked Top 5 =====")
    retrieved = search_chroma(sp)
    top5 = rerank_top_k(search_text, retrieved, top_n=5)
    for i, r in enumerate(top5, 1):
        print(f"[{i}] score={r['score']:.3f}  id={r['_id']}  chunk={r['chunk_index']}  file={r['source_file']}")
        print(r["text"][:300].replace("\n", " ") + ("..." if len(r["text"]) > 300 else ""))
        print()


data_dir = "data"
records = []

file_path = os.path.join(data_dir, "financebench_queries.jsonl.gz")
with gzip.open(file_path, "rt", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        records.append(data)

# Convert all records to a pandas DataFrame
df_test = pd.DataFrame(records)
df_test = df_test[['_id','text']]
df_test.columns = ['query_id','text']


df_eval = pd.read_csv('FinanceBench_qrels.tsv', sep='\t')


len(df_eval['query_id'].unique())


df_eval = df_eval.merge(df_test,on='query_id',how='left')


df_eval.head()


SPLITTERS = splitters_to_build
TOP_K_RETRIEVE = 10
TOP_K_RERANK = 5

def load_chroma(splitter):
    return Chroma(
        persist_directory=f"{PARENT_DIR}/chroma_{splitter}",
        collection_name=f"col_{splitter}",
        embedding_function=embeddings,
    )

def retrieve_docs(db, query_text, k=TOP_K_RETRIEVE):
    retriever = db.as_retriever(search_kwargs={"k": k})
    return retriever.get_relevant_documents(query_text)

def rerank_docs(query_text, docs, top_n=TOP_K_RERANK):
    pairs = [(query_text, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]

results = []

for splitter in SPLITTERS:
    print(f"\nðŸ”Ž Evaluating splitter: {splitter}")
    db = load_chroma(splitter)
    labels = []

    for _, row in tqdm(df_eval.iterrows(), total=len(df_eval)):
        corpus_id = row["corpus_id"]
        query_text = row["text"]

        retrieved_docs = retrieve_docs(db, query_text, k=TOP_K_RETRIEVE)

        reranked = rerank_docs(query_text, retrieved_docs, top_n=TOP_K_RERANK)

        top_ids = [d.metadata.get("_id") for d, _ in reranked if d.metadata and "_id" in d.metadata]

        label = 1 if corpus_id in top_ids else 0
        labels.append(label)

    df_eval[f"label_{splitter}_rerank"] = labels
    results.append((splitter, sum(labels), len(labels), sum(labels) / len(labels)))


df_summary = pd.DataFrame(results, columns=["splitter", "correct", "total", "accuracy"])


df_summary


chunks_df = chunks_df.groupby("splitter").size().sort_values(ascending=True).reset_index()


chunks_df.columns = ['splitter','chunkSize']


df_summary = chunks_df.merge(df_summary[['splitter','accuracy']],on='splitter',how='left')


df_summary['accuracy'] = round(df_summary['accuracy'] * 100,1)


df_summary


df_summary['splitter_type'] = df_summary['splitter'].apply(lambda x: 'character' if 'character' in x else 'recursive')
df_summary['char_size'] = df_summary['splitter'].str.extract(r'(\d+)').astype(int)

sns.relplot(data=df_summary, x='chunkSize', y='accuracy', hue='char_size',
            col='splitter_type', kind='scatter', palette='viridis', height=4, aspect=1.2)

plt.gca().invert_xaxis()
plt.show()

