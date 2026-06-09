import asyncio
import csv
import json
import os
import re
import traceback
from typing import Dict, List, Tuple

import boto3
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from rank_bm25 import BM25Okapi
from collections import defaultdict
import httpx

load_dotenv()

print("✓ Imports loaded")


# AWS credentials check
session = boto3.Session()
credentials = session.get_credentials()
print("AWS credentials:", "Found" if credentials else "Not found")

# Initialize clients
BEDROCK_TOKEN = os.environ.get('BEDROCK_TOKEN')
if not BEDROCK_TOKEN:
    raise ValueError("BEDROCK_TOKEN not found in environment variables")

client = AsyncOpenAI(
    api_key=BEDROCK_TOKEN,
    base_url="https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1"
)

print("✓ Clients initialized")


class VLLMEmbeddingModel:
    """Async wrapper for vLLM embedding API with batch processing"""
    
    def __init__(self, api_url: str, model_name: str):
        self.api_url = api_url
        self.model_name = model_name
    
    async def encode(self, texts: List[str], client: httpx.AsyncClient) -> np.ndarray:
        """Get embeddings for a batch of texts"""
        payload = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float"
        }
        
        response = await client.post(self.api_url, json=payload, timeout=30.0)
        response.raise_for_status()
        result = response.json()
        
        embeddings = [item['embedding'] for item in result['data']]
        return np.array(embeddings, dtype=np.float32)

# Configure embedding model
EMBEDDING_MODEL_NAME = "Linq-AI-Research/Linq-Embed-Mistral"
EMBEDDING_MODEL_URL = 'http://10.50.1.6:8080/v1/embeddings'
embedding_model = VLLMEmbeddingModel(EMBEDDING_MODEL_URL, EMBEDDING_MODEL_NAME)

print(f"✓ Embedding model: {EMBEDDING_MODEL_NAME}")


def load_data(filepath: str) -> List[Dict]:
    """Load JSONL data"""
    print(f"Loading data from: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]
        print(f"Loaded {len(data)} items")
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return []


def extract_question_and_chunks(content: str) -> Tuple[str, List[str], List[int]]:
    """Extract question and chunks from message content"""
    question_start = content.find('Question:')
    text_chunks_start = content.find('Text chunks:')
    
    if question_start == -1 or text_chunks_start == -1:
        return None, [], []
    
    question = content[question_start + len('Question:'):text_chunks_start].strip()
    
    chunks = []
    chunk_indices = []
    chunk_pattern = r'\[Chunk Index (\d+)\]\s*(.*?)(?=\[Chunk Index|\nTask:|$)'
    matches = re.findall(chunk_pattern, content, re.DOTALL)
    
    for match in matches:
        orig_idx = int(match[0])
        chunk_content = match[1].strip()
        if '\nTask:' in chunk_content:
            chunk_content = chunk_content.split('\nTask:')[0].strip()
        if chunk_content:
            chunks.append(chunk_content)
            chunk_indices.append(orig_idx)
    
    return question, chunks, chunk_indices


def append_to_csv(entry: Dict, filename: str):
    """Append a single entry to CSV file"""
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['sample_id', 'target_index'])
        writer.writerow([entry['sample_id'], entry['target_index']])


def get_processed_ids(filename: str) -> set:
    """Get set of sample_ids that have already been processed"""
    if not os.path.isfile(filename):
        return set()
    
    processed = set()
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed.add(row['sample_id'])
    return processed

print("✓ Utility functions loaded")


class HybridRetriever:
    """Combines sparse (BM25) and dense (embeddings) retrieval with RRF fusion"""
    
    def __init__(
        self,
        embedding_model,
        embedding_semaphore: asyncio.Semaphore,
        rrf_k: int = 60,
        vllm_batch_size: int = 32
    ):
        self.embedding_model = embedding_model
        self.embedding_semaphore = embedding_semaphore
        self.rrf_k = rrf_k
        self.vllm_batch_size = vllm_batch_size
        
        # Domain-specific financial synonyms for query expansion
        self.domain_synonyms = {
            # Compensation & Equity
            'equity award burn rate': ['dilution rate', 'share pool availability'],
            'restricted stock units': ['rsu', 'equity grants', 'stock grants'],
            'executive compensation': ['executive pay', 'management compensation'],
            'share repurchase': ['stock buyback', 'share buyback'],
            'dividend policy': ['dividend strategy', 'payout policy'],
            
            # Financial Metrics
            'profitability': ['operating margin', 'ebitda margin'],
            'revenue': ['sales', 'top-line', 'turnover'],
            'recurring revenue': ['subscription revenue', 'contracted revenue'],
            'guidance': ['outlook', 'forecast', 'projection'],
            'margin': ['operating margin', 'profit margin'],
            'fcf': ['free cash flow', 'operating cash flow'],
            'capex': ['capital expenditures', 'capital spending'],
            'inventory': ['working capital', 'stock levels'],
            'combined ratio': ['underwriting ratio', 'loss ratio'],
            
            # Strategy & Operations
            'innovation cycles': ['technology advancement', 'product development cycles'],
            'market competitiveness': ['competitive positioning', 'market position'],
            'geographic expansion': ['international expansion', 'market expansion'],
            'customer engagement': ['user engagement', 'customer retention'],
            'supply chain': ['logistics', 'distribution', 'procurement'],
            
            # Risk & Governance
            'dependency risks': ['concentration risk', 'customer concentration'],
            'foreign exchange': ['fx', 'currency', 'forex'],
            'scenario analyses': ['stress testing', 'risk assessment'],
            'climate risks': ['environmental risks', 'esg risks'],
            'geopolitical': ['political risk', 'country risk'],
            'governance': ['corporate governance', 'board oversight'],
            
            # ESG
            'esg': ['environmental social governance', 'sustainability'],
            
            # Customer/Financing
            'customer financing': ['vendor financing', 'lease program'],
            'penetration rate': ['adoption rate', 'market penetration'],
            'retention': ['retention rate', 'churn'],
            'booking': ['reservations', 'orders'],
            'occupancy': ['occupancy rate', 'utilization'],
        }
    
    def _enrich_query(self, question: str) -> str:
        """Expand query with domain synonyms for better keyword recall"""
        query_lower = question.lower()
        enriched_terms = [question]
        
        # Match multi-word phrases (longer matches first)
        matched_terms = set()
        sorted_terms = sorted(self.domain_synonyms.keys(), key=len, reverse=True)
        
        for term in sorted_terms:
            if term in query_lower and term not in matched_terms:
                # Add top 2 most relevant synonyms
                synonyms = self.domain_synonyms[term][:2]
                enriched_terms.extend(synonyms)
                matched_terms.add(term)
                
                # Mark overlapping terms to avoid duplication
                for other_term in sorted_terms:
                    if other_term != term and (other_term in term or term in other_term):
                        matched_terms.add(other_term)
        
        result = ' '.join(enriched_terms)
        
        # Safety: limit expansion to 2x original length
        if len(result) > len(question) * 2:
            enriched_terms = enriched_terms[:5]
            result = ' '.join(enriched_terms)
        
        return result
    
    def _sparse_retrieval(
        self,
        query: str,
        chunks: List[str],
        chunk_indices: List[int],
        top_k: int = 100
    ) -> List[Tuple[int, float]]:
        """BM25 keyword-based retrieval"""
        if not chunks:
            return []
        
        top_k = min(top_k, len(chunks))
        
        # Tokenize and build BM25 index
        tokenized_chunks = [chunk.lower().split() for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        
        # Score query
        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)
        
        # Return top-k with scores
        ranked = sorted(
            zip(chunk_indices, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return ranked[:top_k]
    
    async def _get_embeddings_concurrently(self, texts: List[str]) -> np.ndarray:
        """Batch texts and get embeddings concurrently"""
        if not texts:
            return np.array([], dtype=np.float32)

        all_embeddings = []
        
        async with httpx.AsyncClient() as http_client:
            tasks = []
            batch_size = self.vllm_batch_size
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                
                async def get_batch_embeddings(b):
                    async with self.embedding_semaphore:
                        return await self.embedding_model.encode(b, http_client)
                
                tasks.append(get_batch_embeddings(batch))

            results = await asyncio.gather(*tasks)

            for batch_embeddings in results:
                all_embeddings.append(batch_embeddings)
                    
        if all_embeddings:
            return np.vstack(all_embeddings)
        return np.array([], dtype=np.float32)
    
    async def _dense_retrieval(
        self,
        query: str,
        chunks: List[str],
        chunk_indices: List[int],
        top_k: int = 100
    ) -> List[Tuple[int, float]]:
        """Embedding-based semantic retrieval"""
        if not chunks:
            return []
        
        top_k = min(top_k, len(chunks))
        
        try:
            # Get embeddings for query + all chunks
            embeddings = await self._get_embeddings_concurrently([query] + chunks)
        except Exception as e:
            print(f"Embedding API failed: {e}")
            return []

        if embeddings.shape[0] < 1:
            return []
            
        # Split: first is query, rest are chunks
        query_emb = embeddings[0]
        chunk_embs = embeddings[1:]
        
        # Compute cosine similarities
        similarities = np.dot(chunk_embs, query_emb)
        
        # Get top-k
        ranked_idx = np.argsort(similarities)[::-1][:top_k]
        ranked = [(chunk_indices[i], float(similarities[i])) for i in ranked_idx]
        
        return ranked
    
    def _reciprocal_rank_fusion(
        self,
        sparse_results: List[Tuple[int, float]],
        dense_results: List[Tuple[int, float]],
        k: int = 60
    ) -> List[int]:
        """Fuse sparse and dense results using RRF"""
        rrf_scores = defaultdict(float)
        
        # Add sparse scores
        for rank, (idx, score) in enumerate(sparse_results, start=1):
            rrf_scores[idx] += 1.0 / (k + rank)
        
        # Add dense scores
        for rank, (idx, score) in enumerate(dense_results, start=1):
            rrf_scores[idx] += 1.0 / (k + rank)
        
        # Sort by RRF score
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, score in ranked]
    
    async def retrieve(
        self,
        question: str,
        chunks: List[str],
        chunk_indices: List[int],
        k: int = 30
    ) -> Tuple[List[str], List[int]]:
        """Full hybrid retrieval pipeline"""
        if len(chunks) <= k:
            print(f"Only {len(chunks)} chunks available, returning all")
            return chunks, chunk_indices
        
        print(f"Hybrid retrieval: {len(chunks)} chunks → {k}")
        
        # Step 1: Enrich query with domain synonyms
        enriched_query = self._enrich_query(question)
        print(f"Enriched query: {enriched_query[:100]}...")
        
        # Step 2: Sparse retrieval (BM25)
        sparse_top_k = min(100, len(chunks))
        sparse_results = self._sparse_retrieval(
            enriched_query, chunks, chunk_indices, top_k=sparse_top_k
        )
        print(f"Sparse (BM25): {len(sparse_results)} results")
        
        # Step 3: Dense retrieval (embeddings)
        dense_top_k = min(100, len(chunks))
        dense_results = await self._dense_retrieval(
            question, chunks, chunk_indices, top_k=dense_top_k
        )
        print(f"Dense (embeddings): {len(dense_results)} results")
        
        # Step 4: Fusion with RRF
        fused_indices = self._reciprocal_rank_fusion(
            sparse_results, dense_results, k=self.rrf_k
        )
        print(f"Fused: {len(fused_indices)} unique candidates")
        
        # Take top-k from fused results
        final_indices = fused_indices[:k]
        
        # Get corresponding chunks
        idx_to_chunk = {idx: chunk for idx, chunk in zip(chunk_indices, chunks)}
        final_chunks = [idx_to_chunk[idx] for idx in final_indices if idx in idx_to_chunk]
        
        return final_chunks, final_indices

print("✓ HybridRetriever loaded")


class SetwiseReranker:
    """Setwise reranking using heapsort with c-way comparisons"""
    
    def __init__(
        self,
        client: AsyncOpenAI,
        semaphore: asyncio.Semaphore,
        compare_size: int = 5,
        model: str = "deepseek.v3-v1:0"
    ):
        self.client = client
        self.semaphore = semaphore
        self.compare_size = compare_size
        self.model = model
        
        if compare_size < 2:
            raise ValueError("compare_size must be at least 2")
    
    async def _setwise_compare(
        self,
        question: str,
        chunks: List[str],
        chunk_indices: List[int]
    ) -> int:
        """
        Compare multiple chunks at once and return the index of the most relevant one.
        Uses LLM to determine which chunk is most relevant to the question.
        """
        if len(chunks) != len(chunk_indices):
            raise ValueError("chunks and chunk_indices must have same length")
        
        # Create prompt with labeled chunks (A, B, C, D, E)
        prompt = f"""Given a query, which of the following chunk is more relevant to the query?

Query: {question}

"""
        for i, (chunk, idx) in enumerate(zip(chunks, chunk_indices)):
            label = chr(65 + i)  # A, B, C, D, E...
            prompt += f"Chunk {label}: {chunk}\n\n"
        
        prompt += "Output only the chunk label of the most relevant chunk, only a single letter (A, B, C, etc.):"
        
        system_message = {
            "role": "system",
            "content": """You are a helpful financial analyst.
Given a question about a company and multiple chunks from the financial documents about the company, your task is to find the most relevant chunk to the question.

Use this relevance hierarchy when ranking chunks:

Tier 1 - Direct Answers (Highest Priority):
- Chunks that explicitly address the question with specific data, policies, or direct statements
- Exact keyword matches for the topic (e.g., "dividend policy," "ESG," "free cash flow")
- Direct quotes from management discussing the specific topic
- Specific financial figures or metrics requested in the question

Tier 2 - Strong Contextual Relevance:
- Related financial metrics that inform or support the answer
- Management discussion of broader themes that encompass the question topic
- Risk disclosures or business factors that directly relate to the question
- Forward-looking statements and guidance relevant to the topic

Tier 3 - Supporting Information:
- Background information that provides important context for understanding the answer
- Historical trends or comparative data that illuminate the topic
- Industry or market factors that affect the specific area being questioned
- Explanatory text that clarifies related financial data

Tier 4 - Weak/Tangential (Lowest Priority):
- General company information with minimal connection to the question
- Unrelated financial data that doesn't inform the specific topic
- Boilerplate text without substantive relevance

Prioritize chunk from higher tiers, but consider that a comprehensive answer may require information from multiple tiers. Focus on chunk that would be most valuable for providing a thorough, accurate response to the specific question asked."""
        }
        
        try:
            async with self.semaphore:
                response = await self.client.chat.completions.create(
                    messages=[system_message, {"role": "user", "content": prompt}],
                    model=self.model,
                    temperature=0.0,
                    reasoning_effort='high'
                )
                
                result = response.choices[0].message.content.strip().upper()
                
                # Parse the letter response (A, B, C, etc.)
                for i, label in enumerate([chr(65 + j) for j in range(len(chunks))]):
                    if result and result[-1] == label:
                        return chunk_indices[i]
                
                # Fallback: return first if parsing fails
                return chunk_indices[0]
                
        except Exception as e:
            print(f"Error in setwise comparison: {e}")
            return chunk_indices[0]
    
    async def _setwise_heapsort(
        self,
        question: str,
        chunks: List[str],
        chunk_indices: List[int],
        k: int
    ) -> List[int]:
        """
        Heap sort with setwise comparisons and parallel batch processing.
        Finds top-k items efficiently using c-way comparisons.
        """
        print(f"Setwise heap sort: top-{k}, compare_size={self.compare_size}")
        
        current_ranking = list(zip(chunks, chunk_indices))
        result = []
        
        for iteration in range(min(k, len(current_ranking))):
            remaining = current_ranking[iteration:]
            
            if len(remaining) <= 1:
                result.extend([idx for _, idx in remaining])
                break
            
            # Process in batches of compare_size in parallel
            batch_tasks = []
            batches = []
            for i in range(0, len(remaining), self.compare_size):
                batch = remaining[i:i + self.compare_size]
                batch_chunks = [item[0] for item in batch]
                batch_indices = [item[1] for item in batch]
                
                batches.append(batch)
                batch_tasks.append(
                    self._setwise_compare(question, batch_chunks, batch_indices)
                )
            
            # Execute all batch comparisons in parallel
            best_indices = await asyncio.gather(*batch_tasks)
            
            # Build candidates from batch winners
            candidates = []
            for batch, best_idx in zip(batches, best_indices):
                best_item = next(item for item in batch if item[1] == best_idx)
                candidates.append(best_item)
            
            # If multiple batch winners, compare them to find overall best
            if len(candidates) > 1:
                cand_chunks = [item[0] for item in candidates]
                cand_indices = [item[1] for item in candidates]
                best_idx = await self._setwise_compare(
                    question, cand_chunks, cand_indices
                )
                best_item = next(item for item in candidates if item[1] == best_idx)
            else:
                best_item = candidates[0]
            
            result.append(best_item[1])
            
            # Remove best_item from remaining candidates
            current_ranking = [item for item in current_ranking if item[1] != best_item[1]]
            
            print(f"Iteration {iteration + 1}: found rank {iteration + 1}")
        
        return result
    
    async def rerank(
        self,
        question: str,
        chunks: List[str],
        chunk_indices: List[int],
        k: int = 5
    ) -> List[int]:
        """Rerank using setwise heapsort approach"""
        return await self._setwise_heapsort(question, chunks, chunk_indices, k)

print("✓ SetwiseReranker loaded")


async def process_item(
    messages: List[Dict],
    query_id: str,
    retriever: HybridRetriever,
    reranker: SetwiseReranker
) -> List[int]:
    """Process single item through the full pipeline"""
    try:
        content = messages[0].get('content', '')
        question, chunks, chunk_indices = extract_question_and_chunks(content)
        
        if not question or not chunks:
            print(f"Could not extract data for {query_id}")
            return list(range(5))
        
        print(f"\n{'='*60}")
        print(f"Processing {query_id}: {len(chunks)} chunks")
        print(f"Question: {question[:100]}...")
        print(f"{'='*60}")
 
        # Stage 1: Hybrid Retrieval (60 candidates)
        retrieved_chunks, retrieved_indices = await retriever.retrieve(
            question, chunks, chunk_indices, k=60
        )
        
        # Stage 2: Setwise Reranking (top 5)
        final_ranking = await reranker.rerank(
            question,
            retrieved_chunks, 
            retrieved_indices, 
            k=5
        )
        
        print(f"Final ranking: {final_ranking}")
        
        return final_ranking
        
    except Exception as e:
        traceback.print_exc()
        print(f"Error processing {query_id}: {e}")
        return list(range(5))

print("✓ Processing pipeline loaded")


async def evaluate(
    data_path: str,
    retriever: HybridRetriever,
    reranker: SetwiseReranker,
    output_file: str
):
    """Run evaluation with incremental saving and resume capability"""
    data = load_data(data_path)
    if not data:
        return
    

    # Create new file with header
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id', 'target_index'])
    
    # Process each item sequentially
    for item in tqdm(data, desc="Processing samples"):
        result = await process_item(
            item['messages'],
            item['_id'],
            retriever,
            reranker
        )
        
        if result:
            # Save top 5 results
            for rank, doc_idx in enumerate(result[:5]):
                entry = {
                    'sample_id': item['_id'], 
                    'target_index': doc_idx
                }
                append_to_csv(entry, output_file)
    
    print(f"\n✓ Evaluation complete! Results saved to: {output_file}")

print("✓ Evaluation function loaded")


# Configure concurrency
llm_semaphore = asyncio.Semaphore(20)  # Max 20 concurrent LLM calls
embedding_semaphore = asyncio.Semaphore(4)  # Max 4 concurrent embedding calls

# Initialize retriever
retriever = HybridRetriever(
    embedding_model=embedding_model,
    embedding_semaphore=embedding_semaphore,
    rrf_k=60,
    vllm_batch_size=32
)

# Initialize reranker
reranker = SetwiseReranker(
    client=client,
    semaphore=llm_semaphore,
    compare_size=5,
    model="deepseek.v3-v1:0"
)

print(f"\n{'='*60}")
print(f"Configuration:")
print(f"  Retriever: HybridRetriever (BM25 + Embeddings + RRF)")
print(f"  Reranker: SetwiseReranker (heapsort, compare_size=5)")
print(f"  Model: {reranker.model}")
print(f"  Embedding: {EMBEDDING_MODEL_NAME}")
print(f"{'='*60}\n")


# Configure paths
DATA_PATH = "./output/chunk_ranking_kaggle_eval.jsonl"
OUTPUT_FILE = f"submission_chunk_ranking.csv"

# Run evaluation
await evaluate(DATA_PATH, retriever, reranker, OUTPUT_FILE)

print(f"\n{'='*60}")
print(f"✓ Complete! Output: {OUTPUT_FILE}")
print(f"{'='*60}")

