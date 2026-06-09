!nvidia-smi
!nvcc --version
!pip install openai rank_bm25 sentence-transformers transformers[torch] faiss-cpu yt_dlp tomli pdfplumber


#set dify api

notebook_config = {
    "Summary": {
        "Dify": {
            "enable": True,
            "api-key": "app-7Y3IBLr24bWPfqZdHXbZydST", #dify api
            "base-url": "https://api.dify.ai/v1",
            "http-proxy": ""
        },
        "Settings": {
            "max_text_length": 1000,
        }
    }
}


import os

base = "/kaggle/input"
dirs = os.listdir(base)
dirs

import pandas as pd

base = "/kaggle/input/WattBot2025"

def smart_read_csv(path):
    encodings = ["utf-8", "latin1", "ISO-8859-1", "cp1252"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_error = e
    raise last_error


train_df = pd.read_csv(f"{base}/train_QA.csv")
test_df = pd.read_csv(f"{base}/test_Q.csv")
meta_df = smart_read_csv(f"{base}/metadata.csv")

train_df.head(), test_df.head(), meta_df.head()



import logging
import aiohttp
import asyncio
import re
import os
import ssl
import certifi
import tomli
import time
import pdfplumber
import json
import html
import xml.etree.ElementTree as ET
import hashlib
import faiss
import torch
import numpy as np
import pickle
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional

print("FAISS version:", faiss.__version__)
print("Has GPU module:", hasattr(faiss, "GpuIndexFlatL2"))
print("Available attributes:", [a for a in dir(faiss) if "Gpu" in a])


# --- DifyArxivSummarizer Class (from previous refactor) ---
# Please ensure this class definition is available or pasted above this new RAG system.
# For simplicity, I'm including a placeholder for it, assuming it's available.

# Re-pasting DifyArxivSummarizer here for self-contained example
# (Ensure you use the version with `config_data` in __init__ for Kaggle compatibility)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DifyArxivSummarizer:
    URL_PATTERN = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[-\w./?=&]*'

    def __init__(self, config_data: Optional[Dict] = None):
        if config_data:
            self.config = config_data.get("Summary", {})
        else:
            logger.warning("No config_data provided, attempting to load from config.toml (may fail in Kaggle).")
            self.config = self._load_config_from_file("config.toml").get("Summary", {})

        dify_config = self.config.get("Dify", {})
        self.dify_enable = dify_config.get("enable", False)
        self.dify_key = dify_config.get("api-key", "")
        self.dify_base_url = dify_config.get("base-url", "")
        self.http_proxy = dify_config.get("http-proxy", "")

        settings = self.config.get("Settings", {})
        self.max_text_length = settings.get("max_text_length", 10000)
        self.black_list = settings.get("black_list", [])
        self.white_list = settings.get("white_list", [])

        self.http_session: Optional[aiohttp.ClientSession] = None
        if not self.dify_enable or not self.dify_key or not self.dify_base_url:
            logger.warning("Dify configuration incomplete, summary functionality may not work.")
            self.dify_enable = False
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

    def _load_config_from_file(self, config_path: str) -> Dict:
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            return {"Dify": {"enable": False}, "Settings": {}}
        try:
            with open(config_path, "rb") as f:
                full_config = tomli.load(f)
            return full_config
        except Exception as e:
            logger.error(f"Error loading config file {config_path}: {e}")
            return {"Dify": {"enable": False}, "Settings": {}}

    async def initialize(self):
        if not self.http_session or self.http_session.closed:
            self.http_session = aiohttp.ClientSession()
            logger.info("HTTP session initialized.")

    async def close(self):
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()
            self.http_session = None
            logger.info("HTTP session closed.")

    async def get_arxiv_paper_text(self, arxiv_url: str) -> Optional[str]:
        if not self.http_session:
            await self.initialize()

        paper_id_match = re.search(r'arxiv\.org/(?:pdf|abs)/([\w.-]+)', arxiv_url)
        if not paper_id_match:
            logger.error(f"Could not extract arXiv paper ID from URL: {arxiv_url}")
            return None
        paper_id = paper_id_match.group(1)

        pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
        logger.info(f"Attempting to download PDF file: {pdf_url}")

        pdf_content = None
        try:
            timeout = aiohttp.ClientTimeout(total=60)
            async with self.http_session.get(pdf_url, timeout=timeout, ssl=self.ssl_context) as response:
                if response.status == 200:
                    pdf_content = await response.read()
                    logger.info(f"Successfully downloaded PDF: {pdf_url}, size: {len(pdf_content)} bytes")
                else:
                    logger.error(f"Failed to download PDF, status code: {response.status}, URL: {pdf_url}")
                    return None
        except asyncio.TimeoutError:
            logger.error(f"PDF download timed out: {pdf_url}")
            return None
        except Exception as e:
            logger.error(f"Error while downloading PDF: {e}, URL: {pdf_url}")
            return None

        temp_pdf_path = f"temp_arxiv_{paper_id}.pdf"
        try:
            with open(temp_pdf_path, 'wb') as f:
                f.write(pdf_content)
            logger.info(f"PDF content saved to temporary file: {temp_pdf_path}")
        except Exception as e:
            logger.error(f"Error saving PDF to temporary file: {e}")
            return None

        text_content = ""
        try:
            with pdfplumber.open(temp_pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n\n"
            logger.info("PDF text extraction completed.")
        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")
            text_content = "Could not extract text from PDF."
        finally:
            if os.path.exists(temp_pdf_path):
                try:
                    os.remove(temp_pdf_path)
                    logger.info(f"Temporary PDF file deleted: {temp_pdf_path}")
                except Exception as e:
                    logger.warning(f"Error deleting temporary PDF file: {e}")
        return text_content


    async def _send_to_dify(self, content_to_summarize: str, prompt_template: str) -> Optional[str]:
        if not self.dify_enable:
            logger.warning("Dify function disabled, skipping request.")
            return None
        if not self.http_session:
            await self.initialize()

        headers = {
            "Authorization": f"Bearer {self.dify_key}",
            "Content-Type": "application/json"
        }
        url = f"{self.dify_base_url}/chat-messages"

        # 使用传入的 prompt_template
        query_prompt = prompt_template.format(content=content_to_summarize)

        payload = {
            "inputs": {},
            "query": query_prompt,
            "response_mode": "blocking",
            "conversation_id": None,
            "user": "arxiv_rag_system"
        }

        try:
            logger.info("Sending content to Dify API for response generation...")
            async with self.http_session.post(
                    url=url,
                    headers=headers,
                    json=payload,
                    proxy=self.http_proxy if self.http_proxy else None
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    answer = result.get("answer", "")
                    logger.info(f"Successfully received response from Dify, length: {len(answer)}.")
                    return answer
                else:
                    error_text = await response.text()
                    logger.error(f"Dify API call failed: {response.status} - {error_text}")
                    return None
        except Exception as e:
            logger.error(f"Error calling Dify API: {e}")
            return None

    async def summarize_arxiv_url(self, arxiv_url: str) -> Optional[str]:
        # This method is specific for summarization, not for RAG answer generation.
        # RAG will use _send_to_dify directly with its own prompt_template
        pass



# --- RAG Components (from your provided code) ---
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

class EmbeddingModel:
    def __init__(self, model_name="jinaai/jina-embeddings-v3"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        if torch.cuda.is_available():
            self.model.to("cuda")
        self.cache_db = Path("embed_cache.db")
        self._init_cache()
    def _init_cache(self):
        conn = sqlite3.connect(self.cache_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                hash TEXT PRIMARY KEY,
                embedding BLOB
            )
        """)
        conn.commit()
        conn.close()
    def _hash(self, text: str):
        return hashlib.md5(text.encode()).hexdigest()

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        out = []
        conn = sqlite3.connect(self.cache_db)
        cur = conn.cursor()

        for t in texts:
            key = self._hash(t)
            row = cur.execute("SELECT embedding FROM cache WHERE hash=?", (key,)).fetchone()

            if row:
                emb = pickle.loads(row[0])
                out.append(emb)
                continue

            tok = self.tokenizer(t, return_tensors="pt", truncation=True)
            if torch.cuda.is_available():
                tok = {k: v.to("cuda") for k, v in tok.items()}

            with torch.no_grad():
                h = self.model(**tok).last_hidden_state.mean(dim=1).squeeze()
                h = h.cpu().to(torch.float32).numpy()

            cur.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (key, pickle.dumps(h)))
            out.append(h)

        conn.commit()
        conn.close()
        return out

class FaissCPUIndex:
    def __init__(self, dim=1024, nlist=1, m=8):
        self.dim = dim
        self.nlist = max(1, nlist)
        self.m = m

        quantizer = faiss.IndexFlatL2(dim)
        self.index = faiss.IndexIVFPQ(quantizer, dim, self.nlist, m, 8)

        self.is_trained = False
        self.doc_texts = []
        self.doc_ids = []

    def train(self, embeddings: np.ndarray):
        if not self.is_trained:
            nlist = min(self.nlist, embeddings.shape[0])
            if nlist < self.nlist:
                self.index.nlist = nlist
            self.index.train(embeddings)
            self.is_trained = True

    def add(self, embeddings: np.ndarray, doc_ids: List[str], raw_texts: List[str]):
        self.train(embeddings)
        self.index.add(embeddings)
        self.doc_ids.extend(doc_ids)
        self.doc_texts.extend(raw_texts)

    def search(self, query_emb: np.ndarray, topk: int) -> List[tuple]:
        scores, I = self.index.search(query_emb, topk)
        return [
            (self.doc_ids[i], self.doc_texts[i], float(scores[0][j]))
            for j, i in enumerate(I[0]) if i >= 0
        ]

class BM25Retriever:
    def __init__(self, docs: List[str]):
        self.docs = docs
        self.tok = [d.split() for d in docs]
        self.bm25 = BM25Okapi(self.tok)

    def search(self, query: str, topk: int) -> List[tuple]:
        scores = self.bm25.get_scores(query.split())
        idx = scores.argsort()[-topk:][::-1]
        return [(i, self.docs[i], float(scores[i])) for i in idx]
class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], topk: int) -> List[Dict[str, Any]]:
        pairs = [(query, c["text"]) for c in candidates]
        tok = self.tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            return_tensors='pt',
            truncation=True,
            padding=True
        )
        with torch.no_grad():
            scores = self.model(**tok).logits.squeeze().numpy()

        ranked = sorted(
            [{"id": c["id"], "text": c["text"], "score": float(s)}
             for c, s in zip(candidates, scores)],
            key=lambda x: -x["score"]
        )
        return ranked[:topk]



class DifyChatModel:
    """
    Adapts DifyArxivSummarizer's _send_to_dify method to the ChatModel interface.
    """
    def __init__(self, dify_summarizer: DifyArxivSummarizer):
        self.dify_summarizer = dify_summarizer
        # Define a default system prompt for Dify here,
        # or it can be passed dynamically.
        self.system_prompt = "You are a helpful assistant. Based on the provided context, answer the user's question concisely and accurately."
    async def complete(self, prompt: str) -> str:
        # _send_to_dify expects a content_to_summarize and a prompt_template
        # Here, the 'prompt' contains the user question and context already formatted by RAGPipeline.
        # So, we pass the entire 'prompt' as content and use a simple template for Dify.
        dify_prompt_template = "{content}" # Dify will get the full prompt from RAGPipeline
        response = await self.dify_summarizer._send_to_dify(prompt, dify_prompt_template)
        return response if response else "Error: Could not get response from Dify."


class RAGPipeline:
    def __init__(self, store: FaissCPUIndex, embedder: EmbeddingModel, chat_model: DifyChatModel,
                 bm25: Optional[BM25Retriever] = None, reranker: Optional[Reranker] = None):
        self.store = store
        self.embedder = embedder
        self.chat = chat_model
        self.bm25 = bm25
        self.reranker = reranker

    def retrieve(self, question: str, topk: int = 10) -> List[Dict[str, Any]]:
        # embed is synchronous, consider running in executor if it blocks
        emb = self.embedder.embed([question])[0]
        emb = emb.reshape(1, -1).astype("float32")
        ann_hits = self.store.search(emb, topk)
        bm25_hits = []
        if self.bm25:
            bm25_hits = self.bm25.search(question, topk)
        merged = {}
        for did, text, score in ann_hits + bm25_hits:
            merged[did] = {"id": did, "text": text, "score": score}
        merged_list = list(merged.values())
        if self.reranker:
            merged_list = self.reranker.rerank(question, merged_list, topk)
        return merged_list

    async def run_qa(self, question: str, system_prompt: str, user_template: str, additional_info: Dict, top_k: int = 5):
        # Retrieve context asynchronously
        ctx_docs = await asyncio.to_thread(self.retrieve, question, top_k) # Run synchronous retrieve in a thread

        context = "\n\n".join([f"[{c['id']}]\n{c['text']}" for c in ctx_docs])

        # Prepare user prompt with context for Dify
        # Note: DifyChatModel's complete method takes this entire formatted prompt
        # and passes it as 'content' to _send_to_dify.
        user_prompt = user_template.format(
            question=question,
            context=context,
            additional_info_json=json.dumps(additional_info)
        )
        # Call DifyChatModel's complete method asynchronously
        raw_response_text = await self.chat.complete(user_prompt)
        # Attempt to parse structured response (assuming Dify might return JSON in its answer)
        structured = {"answer": raw_response_text, "explanation": "N/A", "ref_id": [c["id"] for c in ctx_docs]}
        try:
            # If Dify's answer *itself* is a JSON string, try to parse it
            parsed_dify_answer = json.loads(raw_response_text[raw_response_text.index('{'):raw_response_text.rindex('}')+1])
            structured.update(parsed_dify_answer)
        except (ValueError, IndexError):
            pass # Dify's answer was not structured JSON, use raw_response_text as answer
        # Mimic KohakuRAG's standard output format
        class QAResult:
            def __init__(self, answer, raw_response, prompt):
                self.answer = answer
                self.raw_response = raw_response
                self.prompt = prompt
        return QAResult(
            answer=structured,
            raw_response=raw_response_text,
            prompt=user_prompt
        )


# --- RAG Components (from your provided code) ---
async def build_arxiv_rag_pipeline(
    arxiv_url: str,
    dify_config_data: Optional[Dict] = None
) -> RAGPipeline:
    """
    构建一个针对特定 ArXiv URL 的 RAG Pipeline。
    下载论文，提取文本，并使用其内容构建检索索引。
    Args:
        arxiv_url (str): 要作为知识库的 ArXiv 论文 URL。
        dify_config_data (Optional[Dict]): 用于初始化 DifyArxivSummarizer 的配置字典。
    Returns:
        RAGPipeline: 初始化好的 RAG Pipeline 实例。
    """
    logger.info(f"Initializing DifyArxivSummarizer for ArXiv URL: {arxiv_url}")
    dify_summarizer = DifyArxivSummarizer(config_data=dify_config_data)
    await dify_summarizer.initialize()

    logger.info("Downloading and extracting text from ArXiv paper...")
    arxiv_text = await dify_summarizer.get_arxiv_paper_text(arxiv_url)
    if not arxiv_text:
        await dify_summarizer.close()
        raise ValueError(f"Failed to get text from ArXiv URL: {arxiv_url}")

    # 将整个论文文本作为一个文档处理，或者根据需要进行分块
    # 这里为了简化，假设整个论文作为一个上下文。实际RAG需要更精细的文本分块。
    docs = [arxiv_text]
    ids = ["arxiv_doc_1"] # 或者更复杂的ID生成方式
    logger.info("Initializing Embedding Model and generating embeddings...")
    embedder = EmbeddingModel()
    embs = await asyncio.to_thread(embedder.embed, docs)
    embs_np = np.array(embs).astype("float32") # Convert list of np.ndarray to single np.ndarray

    
    logger.info("Building FAISS index...")
    faiss_index = FaissCPUIndex(dim=embs_np.shape[1])
    faiss_index.add(embs_np, ids, docs)

    logger.info("Initializing BM25 Retriever...")
    bm25 = BM25Retriever(docs)
    
    logger.info("Initializing Reranker...")
    reranker = Reranker()
    
    logger.info("Initializing Dify Chat Model Adapter...")
    dify_chat_model = DifyChatModel(dify_summarizer) # Pass the initialized dify_summarizer
    
    logger.info("Assembling RAG Pipeline...")
    pipeline = RAGPipeline(
        store=faiss_index,
        embedder=embedder,
        chat_model=dify_chat_model,
        bm25=bm25,
        reranker=reranker
    )
    # Store the dify_summarizer to ensure its close() method can be called later
    pipeline.dify_summarizer = dify_summarizer
    return pipeline




async def test_arxiv_rag_system():
    # Dify Configuration for Kaggle Notebook
    notebook_dify_config = notebook_config
    arxiv_url_to_process = "https://arxiv.org/abs/2407.08630" # Example ArXiv URL
    user_question = "What is the main contribution of this paper?"
    # Define a user template for Dify to integrate context and question
    # This template will be filled by RAGPipeline.run_qa
    user_template = """Based on the following context, please answer the question:
    Question: {question}
    Context:
    {context}
    Additional Info: {additional_info_json}
    """
    #  prompt
    dify_system_prompt = "You are a helpful assistant who provides concise and accurate answers based on the provided context."
    rag_pipeline = None
    try:
        logger.info("Building ArXiv RAG pipeline...")
        rag_pipeline = await build_arxiv_rag_pipeline(
            arxiv_url=arxiv_url_to_process,
            dify_config_data=notebook_dify_config
        )
        rag_pipeline.chat.system_prompt = dify_system_prompt # Set system prompt for Dify

        logger.info(f"Running QA for question: {user_question}")
        qa_result = await rag_pipeline.run_qa(
            question=user_question,
            system_prompt=dify_system_prompt, # This is passed to DifyChatModel but its effect depends on Dify's internal handling
            user_template=user_template,
            additional_info={"source_arxiv_url": arxiv_url_to_process},
            top_k=3 # Retrieve top 3 relevant sections
        )

        print("\n--- RAG System Output ---")
        print(f"Question: {user_question}")
        print(f"Answer: {qa_result.answer['answer']}")
        print(f"Raw Dify Response: {qa_result.raw_response}")
        print(f"Prompt sent to Dify: {qa_result.prompt}")
        print(f"References: {qa_result.answer.get('ref_id', 'N/A')}")

    except Exception as e:
        logger.error(f"An error occurred in the RAG system: {e}")
    finally:
        if rag_pipeline and hasattr(rag_pipeline, 'dify_summarizer'):
            await rag_pipeline.dify_summarizer.close()
            logger.info("DifyArxivSummarizer HTTP session closed.")
        logger.info("ArXiv RAG system finished.")



#test our pdf url
await test_arxiv_rag_system()

