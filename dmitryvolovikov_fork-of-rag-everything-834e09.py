!pip install -q langchain_community langchain-experimental faiss-cpu rank-bm25 pypdf langchain-text-splitters
!pip install -q sentence-transformers



import os, random, re, json, html, unicodedata, time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from dataclasses import dataclass
from collections import Counter

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder, InputExample
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as LCFAISS





seed = 824
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
import random
random.seed(seed)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


_WS_IN_LINE = re.compile(r'[ \t]+')
_URL_RE = re.compile(r'https?://\S+|www\.\S+', flags=re.IGNORECASE)
_EMAIL_RE = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')

def cleaning(
    text: str,
    *,
    lower: bool = False,
    keep_newlines: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
) -> str:
    """Аккуратная нормализация текста (без убийства структуры)."""
    if text is None:
        return ""
    s = html.unescape(str(text))
    s = unicodedata.normalize("NFKC", s)

    # убираем zero-width и control chars (но не \n, \t)
    s = "".join(
        ch for ch in s
        if not (
            unicodedata.category(ch) in {"Cf", "Cc"} and ch not in ("\n", "\t")
        )
    )

    # типографика: кавычки/тире/многоточия
    s = s.translate({
        ord("“"): '"', ord("”"): '"', ord("„"): '"', ord("‟"): '"',
        ord("’"): "'", ord("‘"): "'", ord("‚"): "'", ord("′"): "'",
        ord("–"): "-", ord("—"): "-", ord("−"): "-", ord("‐"): "-",
        ord("…"): "...",
    })

    if remove_urls:
        s = _URL_RE.sub(" ", s)
    if remove_emails:
        s = _EMAIL_RE.sub(" ", s)

    s = s.replace("\r\n", "\n").replace("\r", "\n")
    if keep_newlines:
        s = "\n".join(_WS_IN_LINE.sub(" ", line).strip() for line in s.split("\n"))
        s = re.sub(r"\n{3,}", "\n\n", s)
    else:
        s = re.sub(r"\s+", " ", s).strip()

    s = re.sub(r"\s+([,.;:!?])", r"\1", s)

    if lower:
        s = s.lower()
    return s


def load_any(path: str) -> List[Document]:
    """Загрузка PDF / txt / md в список Document с базовыми метаданными."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".pdf":
        docs = PyPDFLoader(str(p)).load()
    elif suf in {".txt", ".md"}:
        docs = TextLoader(str(p), encoding="utf-8").load()
    else:
        raise ValueError(f"Unsupported format: {suf}")

    out = []
    for d in docs:
        cleaned = cleaning(d.page_content)
        meta = {**(d.metadata or {}), "source": str(p), "doc_id": p.stem}
        out.append(Document(page_content=cleaned, metadata=meta))
    return out


# 2. STRUCTURE HEURISTICS
# =====================
_HEADING_MD  = re.compile(r"^(#{1,6})\s+(.+)")
_HEADING_NUM = re.compile(r"^(\d+(?:\.\d+){0,5})\s+(.+)")

def _is_mostly_upper(s: str, min_len=3, ratio=0.75) -> bool:
    letters = [c for c in s if c.isalpha()]
    return len(letters) >= min_len and sum(c.isupper() for c in letters)/len(letters) >= ratio

def detect_heading_line(line: str) -> Optional[dict]:
    line = line.strip()
    if not line:
        return None
    m = _HEADING_MD.match(line)
    if m:
        level = len(m.group(1))
        text = m.group(2).strip()
        return {"heading": text, "level": level}
    m = _HEADING_NUM.match(line)
    if m:
        depth = m.group(1).count(".") + 1
        text = m.group(2).strip()
        return {"heading": f"{m.group(1)} {text}", "level": min(6, depth)}
    if len(line) < 80 and _is_mostly_upper(line):
        return {"heading": line.title(), "level": 1}
    if line.endswith(":") and len(line) < 120:
        return {"heading": line[:-1].strip(), "level": 2}
    return None

@dataclass
class OutlineNode:
    level: int
    title: str
    page: Optional[int]
    path: List[str]
    node_id: str

def build_outline_from_corpus(corpus: List[Document]) -> Dict[str, List[OutlineNode]]:
    outlines: Dict[str, List[OutlineNode]] = {}
    stacks: Dict[str, List[OutlineNode]] = {}
    counters: Dict[str, int] = {}

    for d in corpus:
        doc_id = d.metadata.get("doc_id", "_")
        page = d.metadata.get("page") or d.metadata.get("page_number")
        text = d.page_content or ""
        lines = [ln for ln in text.splitlines() if ln.strip()]

        if doc_id not in stacks:
            stacks[doc_id] = []
            outlines[doc_id] = []
            counters[doc_id] = 0

        for ln in lines:
            h = detect_heading_line(ln)
            if not h:
                continue
            level = h["level"]
            title = h["heading"]

            while stacks[doc_id] and stacks[doc_id][-1].level >= level:
                stacks[doc_id].pop()

            path = [n.title for n in stacks[doc_id]] + [title]
            counters[doc_id] += 1
            nid = f"{doc_id}::h{counters[doc_id]}"
            node = OutlineNode(level=level, title=title, page=page, path=path, node_id=nid)
            outlines[doc_id].append(node)
            stacks[doc_id].append(node)
    return outlines

def attach_structure_to_chunks(chunks: List[Document], outlines: Dict[str, List[OutlineNode]]) -> List[Document]:
    by_doc = outlines
    for d in chunks:
        doc_id = d.metadata.get("doc_id", "_")
        page = d.metadata.get("page") or d.metadata.get("page_number") or -1
        cand = None
        if doc_id in by_doc:
            cands = [n for n in by_doc[doc_id] if (n.page is None or page is None or int(n.page) <= int(page))]
            if cands:
                cand = cands[-1]
        if cand:
            d.metadata["heading"] = cand.title
            d.metadata["heading_level"] = cand.level
            d.metadata["section_path"] = cand.path
            d.metadata["parent_id"] = cand.node_id
        else:
            d.metadata.setdefault("heading", None)
            d.metadata.setdefault("heading_level", None)
            d.metadata.setdefault("section_path", [])
            d.metadata.setdefault("parent_id", None)
    return chunks

# тип блока: текст/таблица/рисунок (очень грубо)
_TABLE_HINTS = re.compile(r"\b(table|таблица|column|row|строк|столб|csv|tsv)\b", re.I)

def is_tabular_like(txt: str) -> bool:
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    if not lines:
        return False
    bars = sum(ln.count("|") for ln in lines)
    tabs = sum(ln.count("\t") for ln in lines)
    hint = bool(_TABLE_HINTS.search(txt[:1000]))
    return (bars >= 3) or (tabs >= 3) or hint

def label_block_type(d: Document) -> Document:
    txt = d.page_content[:2000]
    if is_tabular_like(txt):
        d.metadata["type"] = "table"
    else:
        d.metadata["type"] = "text"
    return d

def structure_report(chunks: List[Document]):
    lvls = Counter(d.metadata.get("heading_level") for d in chunks)
    types = Counter(d.metadata.get("type") for d in chunks)
    paths_nonempty = sum(1 for d in chunks if d.metadata.get("section_path"))
    print("Heading levels:", dict(lvls))
    print("Types:", dict(types))
    print("Sections with path:", paths_nonempty, "/", len(chunks))


files = [
    "/kaggle/input/casml-generative-ai-hackathon/Dataset_RAG (1)/book.pdf",
    # можно добавить ещё txt/md
]



corpus: List[Document] = []
for p in files:
    print("Loading:", p)
    corpus += load_any(p)
print(f"Loaded {len(corpus)} raw docs/pages")


# 2) Структура (outline) по страницам
outlines = build_outline_from_corpus(corpus)


import statistics
from collections import defaultdict

def corpus_eda(corpus: List[Document], outlines: Dict[str, List[OutlineNode]]):
    print("\n=== CORPUS EDA ===")
    # 1) какие doc_id и сколько страниц у каждого
    pages_per_doc = defaultdict(int)
    lens_chars = []
    for d in corpus:
        doc_id = d.metadata.get("doc_id", "_")
        pages_per_doc[doc_id] += 1
        lens_chars.append(len(d.page_content))

    print("Документы и кол-во страниц:")
    for doc_id, n in pages_per_doc.items():
        print(f"  - {doc_id}: {n} pages")

    print(f"\nВсего страниц: {len(corpus)}")
    print(f"Длина страницы (символы): min={min(lens_chars)}, "
          f"median={int(statistics.median(lens_chars))}, "
          f"mean={int(statistics.mean(lens_chars))}, "
          f"max={max(lens_chars)}")

    # 2) EDA по outline
    print("\nOutline по документам:")
    for doc_id, nodes in outlines.items():
        levels = [n.level for n in nodes]
        titles = [n.title for n in nodes]
        print(f"  - {doc_id}: {len(nodes)} заголовков,"
              f" уровни={sorted(set(levels))}")
        print("    Примеры заголовков:")
        for t in titles[:5]:
            print("      •", t)
        print()
        # один-два документа достаточно
        # break  # можно раскомментировать, если outline очень большой

# Вызов:
corpus_eda(corpus, outlines)



# 3. CHUNKING
# =====================

def make_recursive_chunks(corpus: List[Document],
                          chunk_size: int = 800,
                          chunk_overlap: int = 120) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(corpus)
    for i, d in enumerate(chunks):
        d.metadata["_chunk_id"] = i
    return chunks


def make_semantic_chunks(corpus: List[Document],
                         emb_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                         breakpoint_percentile: int = 90) -> List[Document]:
    emb = HuggingFaceEmbeddings(model_name=emb_model_name)
    splitter = SemanticChunker(
        emb,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=breakpoint_percentile,
    )
    chunks = splitter.split_documents(corpus)
    for i, d in enumerate(chunks):
        d.metadata["_chunk_id"] = i
    return chunks


chunks = make_recursive_chunks(corpus, chunk_size=800, chunk_overlap=120)
for i, d in enumerate(chunks):
    d.metadata["_chunk_id"] = i


# 4) Привязка структуры и тип блока
chunks = attach_structure_to_chunks(chunks, outlines)
chunks = [label_block_type(d) for d in chunks]
structure_report(chunks)


# 5) Dedup по тексту
seen, uniq = set(), []
for d in chunks:
    key = re.sub(r"\s+", " ", d.page_content.strip().lower())[:2000]
    if key in seen:
        continue
    seen.add(key)
    uniq.append(d)
chunks = uniq
print("Chunks after dedup:", len(chunks))


import statistics

def chunks_eda_basic(chunks: List[Document]):
    print("\n=== CHUNKS EDA (BASIC) ===")
    n = len(chunks)
    print(f"Всего чанков: {n}")

    lens_chars = [len(d.page_content) for d in chunks]
    print(f"Длина чанка (символы): min={min(lens_chars)}, "
          f"median={int(statistics.median(lens_chars))}, "
          f"mean={int(statistics.mean(lens_chars))}, "
          f"max={max(lens_chars)}")

    # распределение по doc_id
    from collections import Counter
    doc_counts = Counter(d.metadata.get("doc_id", "_") for d in chunks)
    print("\nЧанков по doc_id (топ 10):")
    for doc_id, c in doc_counts.most_common(10):
        print(f"  - {doc_id}: {c}")

    # распределение по heading_level
    lvl_counts = Counter(d.metadata.get("heading_level") for d in chunks)
    print("\nРаспределение по heading_level:")
    print(dict(lvl_counts))

    # распределение по type
    type_counts = Counter(d.metadata.get("type") for d in chunks)
    print("\nРаспределение по type:")
    print(dict(type_counts))

def chunks_eda_tokens(chunks: List[Document], sample_size: int = 200):
    """
    Грубая оценка длины чанков в словах (без LLM-токенизатора,
    чтобы не зависеть от модели).
    """
    print("\n=== CHUNKS EDA (TOKENS ROUGH) ===")
    import re, random
    sample = chunks if len(chunks) <= sample_size else random.sample(chunks, sample_size)
    lens_words = [len(re.findall(r"\w+", d.page_content)) for d in sample]
    print(f"По сэмплу из {len(sample)} чанков:")
    print(f"Слова: min={min(lens_words)}, "
          f"median={int(statistics.median(lens_words))}, "
          f"mean={int(statistics.mean(lens_words))}, "
          f"max={max(lens_words)}")

# Вызов:
chunks_eda_basic(chunks)
chunks_eda_tokens(chunks, sample_size=300)



import matplotlib.pyplot as plt

def plot_chunk_length_hist(chunks: List[Document], max_chars: int = 3000):
    lens = [len(d.page_content) for d in chunks]
    lens = [min(x, max_chars) for x in lens]  # обрезаем хвост
    plt.figure(figsize=(6,4))
    plt.hist(lens, bins=40)
    plt.xlabel("Chunk length (chars, clipped)")
    plt.ylabel("Count")
    plt.title("Распределение длин чанков")
    plt.show()

# Вызов:
plot_chunk_length_hist(chunks)



def filter_too_short_chunks(chunks, min_chars=80):
    out = []
    for d in chunks:
        if len(d.page_content.strip()) >= min_chars:
            out.append(d)
    return out


chunks = filter_too_short_chunks(chunks, min_chars=80)


try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")   # для RU можно ru_core_news_sm
except Exception:
    _NLP = None


# 4. TOKENIZATION / COMPOSITE TEXT
# =====================

def _tok(s: str) -> List[str]:
    if _NLP is not None:
        doc = _NLP(s)
        return [t.lemma_.lower() for t in doc if not t.is_space and not t.is_punct]
    return re.findall(r"\b\w+\b", s.lower())

def _tok_with_heading(d: Document) -> List[str]:
    base = _tok(d.page_content)
    head = _tok(d.metadata.get("heading") or "")
    path = _tok(" / ".join(d.metadata.get("section_path", [])))
    return base + head*2 + path


def composite_text(d: Document) -> str:
    path = " / ".join(d.metadata.get("section_path", [])) if d.metadata.get("section_path") else ""
    head = d.metadata.get("heading") or ""
    typ  = d.metadata.get("type") or "text"
    head_block = [f"[TYPE] {typ}"]
    if path:
        head_block.append(f"[PATH] {path}")
    if head:
        head_block.append(f"[HEAD] {head}")
    return "\n".join(head_block + ["[TEXT] " + d.page_content])


def dedup_by_chunk_id(docs: List[Document]) -> List[Document]:
    seen, out = set(), []
    for d in docs:
        cid = d.metadata.get("_chunk_id")
        if cid in seen:
            continue
        seen.add(cid)
        out.append(d)
    return out

def build_bm25(chunks: List[Document]) -> BM25Okapi:
    tokenized = []
    for d in tqdm(chunks, desc="BM25 tokenization"):
        tokenized.append(_tok_with_heading(d))
    bm25 = BM25Okapi(tokenized)
    return bm25


def build_faiss_index(chunks: List[Document],
                      emb_model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> LCFAISS:
    emb = HuggingFaceEmbeddings(
        model_name=emb_model_name,
        encode_kwargs={"normalize_embeddings": True}
    )
    aug_docs = []
    for d in chunks:
        aug = composite_text(d)
        md = {**d.metadata, "_raw": d.page_content}
        aug_docs.append(Document(page_content=aug, metadata=md))
    vs = LCFAISS.from_documents(aug_docs, emb)
    return vs


def to_raw(doc: Document) -> Document:
    raw = doc.metadata.get("_raw")
    if raw:
        return Document(page_content=raw, metadata={k:v for k,v in doc.metadata.items() if k != "_raw"})
    return doc

def map_to_base(docs_vs: List[Document], base_docs: List[Document]) -> List[Document]:
    pool = {d.metadata.get("_chunk_id"): d for d in base_docs}
    out, seen = [], set()
    for d in docs_vs:
        cid = d.metadata.get("_chunk_id")
        if cid in pool and cid not in seen:
            seen.add(cid)
            out.append(pool[cid])
    return out


bm25 = build_bm25(chunks)
vs   = build_faiss_index(chunks, emb_model_name="sentence-transformers/all-MiniLM-L6-v2")


def rrf_fusion(query: str,
               vs: LCFAISS,
               bm25: BM25Okapi,
               base_docs: List[Document],
               bm25_k: int = 200,
               vec_k: int = 200,
               final_k: int = 60,
               C: int = 60) -> List[Document]:
    q_tokens = _tok(query)
    scores = bm25.get_scores(q_tokens)
    order = np.argsort(scores)[::-1][:bm25_k]

    vec = vs.similarity_search_with_score(query, k=vec_k)

    rrf = {}
    for r, idx in enumerate(order):
        cid = base_docs[idx].metadata["_chunk_id"]
        rrf[cid] = rrf.get(cid, 0.0) + 1.0/(C + r + 1)
    for r, (doc, _) in enumerate(vec):
        cid = doc.metadata.get("_chunk_id")
        if cid is not None:
            rrf[cid] = rrf.get(cid, 0.0) + 1.0/(C + r + 1)

    top_ids = sorted(rrf, key=rrf.get, reverse=True)[:final_k]
    id2doc = {d.metadata["_chunk_id"]: d for d in base_docs}
    return [id2doc[i] for i in top_ids]





def retrieve_mmr(vs: LCFAISS, query: str,
                 k: int = 10,
                 fetch_k: int = 60,
                 lambda_mult: float = 0.5) -> List[Document]:
    return vs.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)





def as_ce_text(d: Document) -> str:
    head = d.metadata.get("heading") or ""
    path = " / ".join(d.metadata.get("section_path", [])) or ""
    prefix = ""
    if path:
        prefix += f"SECTION: {path}\n"
    if head:
        prefix += f"HEADING: {head}\n"
    return (prefix + d.page_content).strip()



def retrieve_with_rerank(
    query: str,
    chunks: List[Document],
    bm25: Optional[BM25Okapi],
    vs: Optional[LCFAISS],
    ce: Optional[CrossEncoder],
    pre_k: int = 60,
    final_k: int = 10,
    use_rrf: bool = True,
    use_mmr: bool = True,
    adaptive_tau: Optional[float] = None,  # порог CE для adaptive k, если None → обычный top-k   0.35 good start  но сначала руками глянуть 
    min_k: int = 3,
    max_k: int = 15,
) -> List[Document]:
    """
    1) BM25 + FAISS (RRF) -> кандидаты.
    2) (опц.) MMR для разнообразия.
    3) CrossEncoder:
        - если adaptive_tau is None -> берём top-final_k;
        - если adaptive_tau задан:
            * берём все кандидаты со score >= tau,
            * если их < min_k -> добираем до min_k top'ами,
            * если их > max_k -> режем до max_k.
    """
    # 1) Кандидаты
    if bm25 is not None and vs is not None and use_rrf:
        cands = rrf_fusion(query, vs, bm25, base_docs=chunks, final_k=pre_k)
        if use_mmr:
            try:
                mmr_raw = retrieve_mmr(vs, query, k=min(pre_k, 50), fetch_k=min(3*pre_k, 180), lambda_mult=0.5)
                mmr = map_to_base(mmr_raw, chunks)
                cands = dedup_by_chunk_id(cands + mmr)[:pre_k]
            except Exception:
                pass
    elif bm25 is not None:
        q_tokens = _tok(query)
        scores = bm25.get_scores(q_tokens)
        order = np.argsort(scores)[::-1][:pre_k]
        cands = [chunks[i] for i in order]
    elif vs is not None:
        raw = vs.similarity_search(query, k=pre_k)
        cands = map_to_base(raw, chunks)
    else:
        raise RuntimeError("No indices available (BM25/FAISS).")

    # 2) CrossEncoder-rerank + adaptive k
    if ce is not None and cands:
        pairs = [[query, d.page_content] for d in cands]
        #pairs = [[query, as_ce_text(d)] for d in cands] ________ Тут если мета дата 
        scores = ce.predict(pairs, batch_size=64)
        # сортируем по score по убыванию
        ranked = list(sorted(zip(cands, map(float, scores)), key=lambda x: x[1], reverse=True))

        if adaptive_tau is not None:
            # все кандидаты, у которых score >= tau
            selected = [d for d, s in ranked if s >= adaptive_tau]

            # если слишком мало — доберём top'ами, чтобы контекст не был пустой
            if len(selected) < min_k:
                selected = [d for d, _ in ranked[:min_k]]

            # если слишком много — ограничим max_k, чтобы не взорвать окно LLM
            if len(selected) > max_k:
                selected = selected[:max_k]

            return selected

        # classic fixed top-k
        return [d for d, _ in ranked[:final_k]]

    # 3) Без CE — обычный top-k
    return cands[:final_k]





# 7. CROSS-ENCODER TRAINING (SELF-SUPERVISED)
# =====================

def make_training_pairs_sentence_query(chunks: List[Document],
                                       negatives_per_pos: int = 1,
                                       max_q_len: int = 200) -> List[InputExample]:
    """
    Вариант 1: "одна фраза из чанка" → чанк (позитив), любые чанки из других doc_id (негативы).
    """
    examples: List[InputExample] = []
    all_docs = chunks[:]
    for d in tqdm(chunks, desc="CE training pairs"):
        sentences = re.split(r'(?<=[.!?])\s+', d.page_content)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            continue
        q = sentences[0][:max_q_len]
        pos = d.page_content

        other = [x for x in all_docs if x.metadata.get("doc_id") != d.metadata.get("doc_id")]
        if not other:
            continue
        negs = random.sample(other, k=min(negatives_per_pos, len(other)))

        examples.append(InputExample(texts=[q, pos], label=1.0))
        for n in negs:
            examples.append(InputExample(texts=[q, n.page_content], label=0.0))
    return examples


def train_cross_encoder_from_chunks(chunks: List[Document],
                                    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                                    epochs: int = 1,
                                    batch_size: int = 16,
                                    negatives_per_pos: int = 2,
                                    output_path: Optional[str] = None) -> CrossEncoder:
    train_ex = make_training_pairs_sentence_query(chunks, negatives_per_pos=negatives_per_pos)
    dl = DataLoader(train_ex, shuffle=True, batch_size=batch_size)
    ce = CrossEncoder(model_name, num_labels=1, max_length=512)
    warmup = max(10, int(0.1 * len(dl)))
    ce.fit(train_dataloader=dl,
           epochs=epochs,
           warmup_steps=warmup,
           output_path=output_path)
    if output_path:
        ce = CrossEncoder(output_path, device=device, max_length=512)
    else:
        ce = CrossEncoder(model_name, device=device, max_length=512)
    return ce


# 7) CrossEncoder (можно просто загрузить готовый, а можно дообучить)
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2",
                  device=device, max_length=512)
# или так, если хочешь дообучить:
# ce = train_cross_encoder_from_chunks(chunks,
#                                      model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
#                                      epochs=1,
#                                      batch_size=16,
#                                      negatives_per_pos=2,
#                                      output_path="ce_finetuned")


GEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

tok = AutoTokenizer.from_pretrained(GEN_MODEL, use_fast=True)
mdl = AutoModelForCausalLM.from_pretrained(
    GEN_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
).eval()


def token_len(text: str) -> int:
    return len(tok(text, add_special_tokens=False).input_ids)

def truncate_by_tokens(text: str, max_tokens: int) -> str:
    ids = tok(text, add_special_tokens=False, truncation=True, max_length=max_tokens).input_ids
    return tok.decode(ids, skip_special_tokens=True)


def build_prompt(question: str,
                 ctx_docs: List[Document],
                 max_ctx_tokens: int = 2500,
                 max_per_snippet: int = 400) -> str:
    """
    Формируем prompt:
    - каждый чанк как [S#] ...;
    - просим отвечать кратко, с цитатами вида [S1], [S2].
    """
    chunks_text = []
    used = 0
    for i, d in enumerate(ctx_docs, 1):
        body = d.page_content.strip()
        body = truncate_by_tokens(body, max_per_snippet)
        tag  = f"[S{i}]"
        block = f"{tag} {body}"
        t = token_len(block)
        if used + t > max_ctx_tokens:
            break
        chunks_text.append(block)
        used += t

    ctx = "\n\n".join(chunks_text)
    prompt = f"""You are a careful QA assistant.

Use ONLY the context snippets [S1..S{len(chunks_text)}] below.
Every factual sentence in your answer MUST end with one or more citations like [S1], [S2].
If the answer is not supported by the context, say "I don't know".

CONTEXT:
{ctx}

QUESTION:
{question}

Answer in the same language as the question. Be concise (3–6 sentences)."""
    return prompt


def llm_generate(prompt: str,
                 max_new_tokens: int = 300,
                 temperature: float = 0.2,
                 top_p: float = 0.9) -> str:
    if hasattr(tok, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "You are a precise assistant. Follow instructions exactly."},
            {"role": "user", "content": prompt},
        ]
        input_ids = tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(mdl.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            out_ids = mdl.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature if temperature > 0.0 else None,
                top_p=top_p if temperature > 0.0 else None,
            )
        text = tok.decode(out_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
        return text
    else:
        pipe = hf_pipeline("text-generation", model=mdl, tokenizer=tok)
        out = pipe(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
                   return_full_text=False)[0]["generated_text"]
        return out


def generate_answer(question: str,
                    ctx_docs: List[Document],
                    temperature: float = 0.2) -> str:
    prompt = build_prompt(question, ctx_docs)
    answer = llm_generate(prompt, max_new_tokens=300, temperature=temperature)
    return answer.strip()


def build_context_for_submission(ctx_docs: List[Document], max_chars: int = 8000) -> str:
    parts, used = [], 0
    for i, d in enumerate(ctx_docs, 1):
        block = f"[S{i}] {d.page_content.strip()}"
        #block = f"[S{i}] (section: {' / '.join(path)})\n{body}" ___________ Если хочется мета данные в контекст

        if used + len(block) + 2 > max_chars:
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)

def build_references_simple(ctx_docs: List[Document]) -> Dict[str, List[str]]:
    """Простая ссылка: секции + страницы из метаданных."""
    sections, pages = [], []
    for d in ctx_docs:
        path = d.metadata.get("section_path") or []
        if path:
            sections.append("/".join(path))
        pg = d.metadata.get("page") or d.metadata.get("page_number") or d.metadata.get("_page")
        if pg is not None:
            pages.append(str(pg))
    # dedup с сохранением порядка
    def uniq(seq):
        seen, out = set(), []
        for x in seq:
            if x not in seen:
                seen.add(x); out.append(x)
        return out
    return {
        "sections": uniq(sections)[:6],
        "pages": uniq(pages)[:10],
    }





def answer_one_question(
    question: str,
    chunks: List[Document],
    bm25: BM25Okapi,
    vs: LCFAISS,
    ce: CrossEncoder,
    adaptive_tau: float = 0.35,
) -> Dict[str, Any]:
    # 1) Retrieval
    ctx_docs = retrieve_with_rerank(
        query=question,
        chunks=chunks,
        bm25=bm25,
        vs=vs,
        ce=ce,
        pre_k=60,
        final_k=10,
        use_rrf=True,
        use_mmr=True,
        adaptive_tau=adaptive_tau,
        min_k=3,
        max_k=15,
    )

    # 2) Answer
    answer_text = generate_answer(question, ctx_docs)

    # 3) Context + references
    context_text = build_context_for_submission(ctx_docs)
    refs = build_references_simple(ctx_docs)

    return {
        "answer": answer_text,
        "context": context_text,
        "references": json.dumps(refs, ensure_ascii=False),
        "ctx_docs": ctx_docs,
    }



q = "What is the scientific method in psychology?"
res = answer_one_question(q, chunks, bm25, vs, ce, adaptive_tau=0.35)

print("ANSWER:\n", res["answer"])
print("\nCONTEXT SNIPPETS:\n", res["context"][:2000])
print("\nREFERENCES:", res["references"])






def make_submission(
    queries_json_path: str,
    out_csv: str = "submission.csv",
    adaptive_tau: float = 0.35,
) -> pd.DataFrame:
    """
    Делает submission:
    - читает queries.json,
    - для каждого вопроса вызывает answer_one_question,
    - собирает DataFrame с колонками [ID, context, answer, references],
    - сохраняет в CSV.
    ОПИРАЕТСЯ на уже подготовленные: chunks, bm25, vs, ce.
    """
    # 1) читаем queries.json
    data = json.loads(Path(queries_json_path).read_text(encoding="utf-8"))

    # допускаем, что формат может быть либо {"queries": [...]}, либо просто [...]
    if isinstance(data, dict):
        items = data.get("queries") or data.get("data") or data.get("items") or data.get("questions") or []
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("Unsupported queries.json format")

    assert items, "Не нашли список запросов в queries.json"

    rows = []
    for it in tqdm(items, desc="Answering"):
        # вытаскиваем ID и сам текст запроса
        qid = it.get("ID") or it.get("id") or it.get("query_id")
        question = it.get("question") or it.get("query") or it.get("text")

        if qid is None or not question:
            continue

        # аккуратно приводим ID к int
        try:
            qid_int = int(qid)
        except Exception:
            qid_int = int("".join(ch for ch in str(qid) if ch.isdigit()) or 0)

        # получаем ответ с помощью нашего пайплайна
        out = answer_one_question(
            question=question,
            chunks=chunks,
            bm25=bm25,
            vs=vs,
            ce=ce,
            adaptive_tau=adaptive_tau,
        )

        ans = out["answer"].replace("\r", " ").replace("\n", " ").strip()

        rows.append({
            "ID": qid_int,
            "context": out["context"],
            "answer": ans,
            "references": out["references"],
        })

    df = pd.DataFrame(rows).sort_values("ID")
    df.to_csv(out_csv, index=False)
    _log(f"Saved submission to: {out_csv}")
    return df



queries_path = "/kaggle/input/casml-generative-ai-hackathon/Dataset_RAG (1)/queries.json"


def queries_eda(queries_json_path: str):
    print("\n=== QUERIES EDA ===")
    data = json.loads(Path(queries_json_path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        items = data.get("queries") or data.get("data") or data.get("items") or data.get("questions") or []
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("Unsupported queries.json format")

    texts = []
    for it in items:
        q = it.get("question") or it.get("query") or it.get("text")
        if q:
            texts.append(q.strip())

    print(f"Всего запросов: {len(texts)}")
    lens_chars = [len(t) for t in texts]
    lens_words = [len(t.split()) for t in texts]

    import statistics
    print(f"Длина запроса (символы): min={min(lens_chars)}, "
          f"median={int(statistics.median(lens_chars))}, "
          f"mean={int(statistics.mean(lens_chars))}, "
          f"max={max(lens_chars)}")
    print(f"Длина запроса (слова):   min={min(lens_words)}, "
          f"median={int(statistics.median(lens_words))}, "
          f"mean={int(statistics.mean(lens_words))}, "
          f"max={max(lens_words)}")

    print("\nПримеры коротких запросов:")
    for t in sorted(texts, key=len)[:5]:
        print("  -", t)

    print("\nПримеры длинных запросов:")
    for t in sorted(texts, key=len, reverse=True)[:5]:
        print("  -", t)

# Вызов:
queries_eda("/kaggle/input/casml-generative-ai-hackathon/Dataset_RAG (1)/queries.json")



queries_path = "/kaggle/input/casml-generative-ai-hackathon/Dataset_RAG (1)/queries.json"

df_sub = make_submission(
    queries_json_path=queries_path,
    out_csv="submission.csv",
    adaptive_tau=0.35,
)

df_sub.head()

