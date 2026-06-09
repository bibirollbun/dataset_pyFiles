# Install dependencies

!pip install requests PyPDF2 openai faiss-cpu numpy nltk


# Import statement
import os
from kaggle_secrets import UserSecretsClient
import json
import tempfile
import re
from io import BytesIO
from typing import List, Dict, Optional

import requests
import PyPDF2
import openai
import faiss
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt', quiet=True)


# Use Kaggle's user secrets tool to get the user's OpenAI Key
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("OPENAI_API_KEY")

openai.api_key = api_key

if not openai.api_key:
    raise ValueError("Please set the OPENAI_API_KEY in your .env file")


# Functions to download, clean, and extract text from pdfs given the url

def download_pdf(url: str) -> Optional[str]:
    """Download a PDF file from a URL and return a local file path"""
    try:
        response = requests.get(url)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            return tmp_file.name
    except Exception as e:
        print(f"Error downloading PDF from {url}: {str(e)}")
        return None

def extract_text_from_pdf(pdf_file: str) -> List[str]:
    """Extract text from PDF file and return cleaned list of sentences"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
                
        text = re.sub(r'\\(?![ntrbfv"\'\\ux])', '', text)

        # Normalize unicode escapes (e.g., \u2019 → ')
        text = text.encode('utf-8').decode('unicode_escape')

        # Remove newline characters
        text = re.sub(r"(?<!\n)\n(?!\n)", "", text)  # avoid double \n (paragraph breaks)

        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)

        # Sentence tokenize
        sentences = sent_tokenize(text)

        # Keep only meaningful sentences
        cleaned_sentences = [s.strip() for s in sentences if len(s.strip().split()) > 3]

        return cleaned_sentences
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return []


# Functions to create embeddings from sentences

def get_embedding(text: str) -> Optional[List[float]]:
    """Get embedding for a text using OpenAI's embedding model"""
    try:
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {str(e)}")
        return None

def get_batch_embeddings(texts: List[str], batch_size: int = 100) -> List[Optional[List[float]]]:
    """Get embeddings for a batch of texts using OpenAI's embedding model"""
    all_embeddings = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            response = openai.embeddings.create(
                input=batch,
                model="text-embedding-3-small"
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            print(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            # Add None for failed embeddings
            all_embeddings.extend([None] * len(batch))
    
    return all_embeddings


# Putting it all together--given a file listing urls, output a list of dictionaries saving 
# Sentences, where they are from, and their embedding

def process_pdf_urls(urls_file: str, batch_size: int = 100) -> List[Dict]:
    """Process PDFs from a file containing URLs with batch processing"""
    results = []
    
    with open(urls_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    for url in urls:
        print(f"\nProcessing {url}...")
        pdf_path = download_pdf(url)
        if not pdf_path:
            continue
        
        # Extract and clean sentences
        sentences = extract_text_from_pdf(pdf_path)
        print(f"Found {len(sentences)} sentences")
        
        # Filter out very long sentences
        valid_sentences = [
            (i, s) for i, s in enumerate(sentences)
            if len(s.split()) <= 100  # Skip sentences with more than 100 words
        ]
        
        if not valid_sentences:
            continue
            
        # Prepare batch for embedding
        indices, texts = zip(*valid_sentences)
        
        # Get embeddings for all sentences in batch
        embeddings = get_batch_embeddings(texts, batch_size)
        
        # Create results for successful embeddings
        for idx, text, embedding in zip(indices, texts, embeddings):
            if embedding is not None:
                results.append({
                    'url': url,
                    'sentence_index': idx,
                    'text': text,
                    'embedding': embedding
                })
        
        # Clean up temporary PDF file
        try:
            os.remove(pdf_path)
        except:
            pass
    
    return results


# Create a file named 'pdf_urls.txt' with one URL per line and place it in the imports tab
FILE_PATH = 'Paste your file-path here'

# Call function to create a list of embeddings
results = process_pdf_urls(FILE_PATH)

# Save results to JSON file
with open('embeddings_results.json', 'w') as f:
    json.dump(results, f, indent=2)


# Small VectorDatabase class
class VectorDatabase:
    def __init__(self, dimension: int = 1536):
        """Initialize FAISS index and document store"""
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []

    def add_documents(self, documents: List[Dict]):
        """Add documents and their embeddings to the FAISS index"""
        embeddings = np.array([doc['embedding'] for doc in documents]).astype('float32')
        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for top-k similar documents given a query string"""
        query_embedding = get_embedding(query)
        if not query_embedding:
            return []

        # Debug information
        print(f"Query embedding type: {type(query_embedding)}")
        print(f"Query embedding length: {len(query_embedding)}")
        
        # Convert query embedding to numpy array and reshape for FAISS
        query_vector = np.array([query_embedding]).astype('float32')
        print(f"Query vector type: {type(query_vector)}")
        print(f"Query vector shape: {query_vector.shape}")
        print(f"Query vector dtype: {query_vector.dtype}")
        
        # Ensure the vector has the correct shape (1, dimension)
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
            print(f"Reshaped query vector shape: {query_vector.shape}")

        try:
            distances, indices = self.index.search(query_vector, k)
            print(f"Search successful. Distances shape: {distances.shape}, Indices shape: {indices.shape}")
        except Exception as e:
            print(f"Error during search: {str(e)}")
            print(f"Index dimension: {self.index.d}")
            raise

        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                result = self.documents[idx].copy()
                result['similarity_score'] = float(1 / (1 + distance))  # Convert L2 distance to similarity
                results.append(result)

        return results


# Initialize vector database
db = VectorDatabase()

# Load existing embeddings from JSON
with open('embeddings_results.json', 'r') as f:
    results = json.load(f)

# Add documents to FAISS
db.add_documents(results)
print(f"Added {len(results)} documents to the vector database.")


# Demo query

query = "I canoed in the river"
results = db.search(query, k=5)

for i, result in enumerate(results, 1):
    print(f"\nResult {i} (Similarity: {result['similarity_score']:.3f}):")
    print(f"URL: {result['url']}")
    print(f"Text: {result['text'][:200]}...")


def save_vector_db(db: VectorDatabase, filename: str):
    """Save FAISS index and documents to disk"""
    faiss.write_index(db.index, f"{filename}.index")

    print(f'FAISS index saved to {filename}.index')

    with open(f"{filename}.json", 'w') as f:
        json.dump(db.documents, f)

    print(f'Text, metadata saved to {filename}.json')

def load_vector_db(filename: str) -> VectorDatabase:
    """Load FAISS index and documents from disk"""
    db = VectorDatabase()
    db.index = faiss.read_index(f"{filename}.index")
    with open(f"{filename}.json", 'r') as f:
        db.documents = json.load(f)

    print(f'Loaded Vector Database from {filename}.index and {filename}.index')
    return db


# Save the database
save_vector_db(db, 'pdf_vector_db')

# Load it later with:
# db = load_vector_db('pdf_vector_db')

