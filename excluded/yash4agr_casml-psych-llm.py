%%capture
!pip install torch pypdf diskcache sentence_transformers transformers "vllm<=0.7" pdfplumber chromadb


from typing import Dict, Optional, List, Tuple, Any
import json
import os
import numpy as np
import re
import hashlib
import pdfplumber
from functools import lru_cache
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import vllm
print('vllm version=', vllm.__version__)
os.environ["CUDA_VISIBLE_DEVICES"]= "1"


@dataclass
class ModelConfig:
    model_name: str = "/kaggle/input/qwen2.5/transformers/3b-instruct/1" 
    embedding_model: str = "/kaggle/input/baai/transformers/bge-large-en-v1.5/1"
    max_tokens: int = 512
    temperature: float = 0.7
    cache_size: int = 100

@dataclass
class VectorDBConfig:
    collection_name: str = "psychology_book"
    chunk_size: int = 500
    chunk_overlap: int = 50


class PDFParser:
    def __init__(self):
        self.current_page = 0
    
    def parse_pdf(self, pdf_path: str) -> str:
        """
        Parse PDF and return text with page markers.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            String containing all text with page markers
        """
        extracted_text = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text from page
                text = page.extract_text()
                if text:
                    # Add page marker
                    extracted_text.append(f"[PAGE {page_num}]")
                    extracted_text.append(text.strip())
        
        return '\n'.join(extracted_text)


class TextChunker:
    def __init__(self, config: VectorDBConfig):
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        
    def create_chunks_with_metadata(self, text: str, sections_metadata: Dict) -> List[Tuple[str, Dict]]:
        """
        Split text into overlapping chunks while preserving section context and page numbers.
        
        Args:
            text: String containing the full text with page markers
            sections_metadata: Dictionary containing section and page information
        
        Returns:
            List of tuples containing (chunk_text, metadata)
        """
        # Split text into pages first
        pages = self._split_into_pages(text)
        chunks = []
        
        current_chunk = []
        current_chunk_size = 0
        current_page = 1
        
        for page_num, page_text in pages.items():
            # Clean the text
            cleaned_text = self._clean_text(page_text)
            words = cleaned_text.split()
            
            i = 0
            while i < len(words):
                # If current chunk is empty, start new chunk
                if not current_chunk:
                    metadata = self.get_section_for_page(page_num, sections_metadata)
                    current_page = page_num
                
                # Add words to current chunk until reaching chunk_size
                while i < len(words) and current_chunk_size < self.chunk_size:
                    current_chunk.append(words[i])
                    current_chunk_size += 1
                    i += 1
                
                # If chunk is full or we're at end of page, save it
                if current_chunk_size >= self.chunk_size or i >= len(words):
                    chunk_text = ' '.join(current_chunk)
                    chunks.append((
                        chunk_text,
                        {
                            'section': metadata['section'],
                            'subsection': metadata['subsection'],
                            'page': current_page
                        }
                    ))
                    
                    # Start new chunk with overlap
                    if i < len(words):
                        overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                        current_chunk = current_chunk[overlap_start:]
                        current_chunk_size = len(current_chunk)
                    else:
                        current_chunk = []
                        current_chunk_size = 0
            
        return chunks

    def _split_into_pages(self, text: str) -> Dict[int, str]:
        """Split text into pages based on page markers."""
        pages = {}
        current_page = None
        current_text = []
        
        for line in text.split('\n'):
            # Check for page marker (assuming format like [PAGE 1] or similar)
            page_match = re.match(r'\[PAGE (\d+)\]', line)
            if page_match:
                if current_page is not None and current_page <= 645:
                    pages[current_page] = '\n'.join(current_text)
                current_page = int(page_match.group(1))
                current_text = []
                
                if current_page > 645:
                    break
            else:
                if current_page is not None and current_page <= 645:
                    current_text.append(line)
        
        # Don't forget to add the last page
        if current_page is not None:
            pages[current_page] = '\n'.join(current_text)
        
        return pages
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace and special characters."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:-]', '', text)
        return text

    def get_section_for_page(self, page_num: int, sections_metadata: Dict) -> Dict:
        for section, data in sections_metadata.items():
            if data['page_start'] <= page_num <= data['page_end']:
                for subsection, sub_data in data.get('subsections', {}).items():
                    if sub_data['page_start'] <= page_num <= sub_data['page_end']:
                        return {
                            'section': section,
                            'subsection': subsection,
                            'page': page_num
                        }
        return {'section': 'Unknown', 'subsection': 'Unknown', 'page': page_num}


class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, device="cpu")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
    
    def embed_query(self, query: str) -> List[float]:
        return self.model.encode(query).tolist()


class VectorStore:
    def __init__(self, config: VectorDBConfig):
        self.client = chromadb.Client(Settings(allow_reset=True))
        self.collection = self.client.get_or_create_collection(name=config.collection_name, )
    
    def add_documents(self, chunks: List[Tuple[str, Dict]], embeddings: List[List[float]]):
        texts = [chunk[0] for chunk in chunks]
        metadatas = [chunk[1] for chunk in chunks]
        ids = [f"doc_{i}" for i in range(len(chunks))]
        
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    
    def query(self, query_embedding: List[float], n_results: int = 3) -> List[Dict]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results


class ResponseGenerator:
    def __init__(self, config: ModelConfig):
        self.llm = vllm.LLM(model=config.model_name, dtype = 'half', device="auto")
        self.sampling_params = vllm.SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
    
    def generate_response(self, query: str, context: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate response based on query and retrieved context
        
        Args:
            query: User's query string
            context: Dictionary of query results from vector store
        
        Returns:
            Generated response as a string
        """
        # Extract documents and metadatas from context
        documents = context.get('documents', [[]])[0]
        metadatas = context.get('metadatas', [[]])[0]
        
        # Combine documents with their metadata
        context_items = [
            {
                'text': doc, 
                'metadata': metadata
            } 
            for doc, metadata in zip(documents, metadatas)
        ]
        
        prompt = self._create_prompt(query, context_items)
        
        response = self.llm.generate([prompt], self.sampling_params)
        return response[0].outputs[0].text, context_items

    def _create_prompt(self, query: str, context: List[Dict]) -> str:
        """
        Create a prompt with context and query
        
        Args:
            query: User's query string
            context: List of context dictionaries with 'text' and 'metadata'
        
        Returns:
            Formatted prompt string
        """
        context_str = "\n".join([
            f"[{c['metadata']['section']} - {c['metadata']['subsection']}, Page {c['metadata']['page']}]: {c['text']}" 
            for c in context
        ])
        
        prompt = f"""You are a knowledgeable psychology assistant. Use the provided sources to answer the question.
Context:
{context_str}

Instructions:

Use the provided context to ensure relevance in your response.
Maintain clarity, accuracy, and conciseness while avoiding unnecessary repetition.
Limit the response to 500 words.
Do not include closing phrases like "Best regards" or "Let me know if I can help you further."
Keep the response strictly relevant to the context.
Task:
Provide a clear, detailed, and well-structured answer to the following question, ensuring accuracy and relevance based on the context.

Question: {query}

Answer:"""
        return prompt


class RAGPipeline:
    def __init__(
        self,
        model_config: ModelConfig,
        vector_config: VectorDBConfig,
        sections_metadata: Dict
    ):
        self.embedder = Embedder(model_config.embedding_model)
        self.vector_store = VectorStore(vector_config)
        self.generator = ResponseGenerator(model_config)
        self.chunker = TextChunker(vector_config)
        self.pdf_parser = PDFParser()
        self.sections_metadata = sections_metadata
    
    def index_document(self, pdf_path: str):
        text = self.pdf_parser.parse_pdf(pdf_path)
        chunks = self.chunker.create_chunks_with_metadata(text, self.sections_metadata)
        embeddings = self.embedder.embed_documents([chunk[0] for chunk in chunks])
        self.vector_store.add_documents(chunks, embeddings)
    
    def query(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        query_embedding = self.embedder.embed_query(query)
        relevant_chunks = self.vector_store.query(query_embedding)
        response, context = self.generator.generate_response(query, relevant_chunks)
        return response, context


if __name__ == "__main__":
    # Load configurations
    model_config = ModelConfig()
    vector_config = VectorDBConfig()
    
    # # Load sections metadata
    with open('/kaggle/input/cadml-dataset/Data/sections_metadata.json', 'r') as f:
        sections_metadata = json.load(f)
    
    # Initialize pipeline
    pipeline = RAGPipeline(model_config, vector_config, sections_metadata)
    
    # Index document
    pipeline.index_document("/kaggle/input/cadml-dataset/Data/book.pdf")
    
    # Query example
    query = "What are the contributions made by Freud in psycgology?"
    response, context = pipeline.query(query)
    print(response)


import csv

def save_row_in_csv(query_id, context_text, response_text, references, csv_file):    
    """
    Save a row in the CSV file with the provided data.

    Args:
        query_id (str): Unique identifier for the query.
        context_text (str): The context text used to generate the response.
        response_text (str): The generated response text.
        references (dict): References containing sections and pages.
        csv_file (str): Path to the CSV file.

    """
    # Ensure references are converted to a string format for CSV storage
    references_str = str(references)
    assert isinstance(references, dict)

    write_header = not os.path.exists(csv_file) or os.stat(csv_file).st_size == 0
    # Open the CSV file in append mode and write the row
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write headers if needed
        if write_header:
            writer.writerow(["ID", "context", "answer", "references"])
        # Write the row data
        writer.writerow([query_id, context_text, response_text, references_str])

    print(f"Row saved successfully to {csv_file}")


!rm -rf /kaggle/working/submission.csv


with open("/kaggle/input/casml-generative-ai-hackathon/Dataset_RAG (1)/queries.json") as f:
    queries = json.load(f)
    
for idx, query in enumerate(queries):
    print(f"Answering question number : {idx+1} \n\n ")
    query_id = query["query_id"]
    question_text = query["question"]
    response, context = pipeline.query(question_text)
    context_text = "/n".join([c["text"] for c in context])
    reference = {
        "sections": [c["metadata"]["section"] for c in context], 
        "pages": [c["metadata"]["page"] for c in context]
    }
    save_row_in_csv(query_id, context_text, response, reference, "submission.csv")


