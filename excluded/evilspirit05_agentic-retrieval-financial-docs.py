# Import required libraries
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import re
from collections import defaultdict, Counter
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings("ignore")


# Attempt to load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None


# Define paths and configuration
INPUT_DIR = Path('/kaggle/input/acm-icaif-25-ai-agentic-retrieval-grand-challenge')
OUTPUT_DIR = Path('/kaggle/working')

class Config:
    TOP_K = 5
    CHUNK_LIMIT = 40
    DOC_MAX_FEATURES = 200
    CHUNK_MAX_FEATURES = 25000
    NGRAM_RANGE = (1, 4)

config = Config()


# Define financial keywords and stop words
def financial_keywords():
    return [
        'revenue', 'earnings', 'profit', 'loss', 'income', 'cash', 'flow', 'assets', 
        'liabilities', 'equity', 'debt', 'shares', 'stock', 'dividend', 'eps',
        'quarter', 'year', 'fiscal', 'guidance', 'outlook', 'forecast', 'target',
        'growth', 'margin', 'cost', 'expense', 'acquisition', 'merger', 'spin',
        'risk', 'litigation', 'regulatory', 'compliance', 'audit', 'board',
        'executive', 'compensation', 'proxy', 'filing', 'sec', 'form', '10k',
        '10q', '8k', 'def14a', 'transcript', 'call', 'conference'
    ]

FINANCIAL_TERMS = set(financial_keywords())
STOP_WORDS = set(stopwords.words('english')) - FINANCIAL_TERMS


# Function to load JSONL files
def load_jsonl(filepath):
    print(f"Loading {filepath.name}...")
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in tqdm(lines, desc="Parsing"):
        if line.strip():
            try:
                data.append(json.loads(line.strip()))
            except:
                continue
    print(f"Loaded {len(data)} samples")
    return data


# Advanced text cleaning function
def advanced_clean(text):
    if not text:
        return ""
    
    text = re.sub(r'[^a-zA-Z0-9\s\-]', ' ', text.lower())
    text = re.sub(r'\d+[kmb]?(?:\.\d+)?%', ' NUM ', text)
    text = re.sub(r'\$\d+(?:,\d+)*(?:\.\d+)?', ' $AMOUNT ', text)
    text = re.sub(r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s?\d{0,2}', ' DATE ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Entity extraction and question extraction functions
def extract_entities(text):
    if nlp is None or len(text) < 50:
        return text
    
    doc = nlp(text)
    entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ['ORG', 'MONEY', 'PERCENT', 'DATE']]
    return ' '.join(entities) if entities else text

def extract_question(messages):
    for msg in messages:
        if msg.get('role') == 'user':
            content = msg.get('content', '')
            cleaned = advanced_clean(content)
            if nlp:
                cleaned = extract_entities(cleaned)
            return cleaned
    return advanced_clean(messages[0].get('content', '') if messages else '')


# Smart chunking function
def smart_chunking(content):
    if not content:
        return []
    
    chunks = []
    
    paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 50]
    if len(paragraphs) >= 8:
        chunks = paragraphs
    else:
        sentences = []
        for para in paragraphs:
            sents = re.split(r'(?<=[.!?])\s+', para)
            sentences.extend([s.strip() for s in sents if len(s.strip()) > 20])
        
        if len(sentences) >= 12:
            chunks = sentences
        else:
            for i in range(0, len(content), 60):
                chunk = content[i:i+60].strip()
                if len(chunk) > 30:
                    chunks.append(chunk)
    
    cleaned_chunks = []
    for chunk in chunks:
        cleaned = advanced_clean(chunk)
        if nlp and len(cleaned) > 30:
            cleaned = extract_entities(cleaned)
        if len(cleaned) > 20:
            cleaned_chunks.append(cleaned)
    
    return cleaned_chunks[:config.CHUNK_LIMIT]


# Preprocess documents and chunks
def preprocess_docs(data):
    print("Preprocessing documents...")
    processed = []
    doc_types = ["10-K", "10-Q", "8-K", "DEF-14A", "Earnings Transcript"]
    
    for item in tqdm(data, desc="Docs"):
        sample_id = item.get('_id')
        question = extract_question(item.get('messages', []))
        processed.append({
            'sample_id': sample_id,
            'question': question,
            'docs': doc_types
        })
    return processed

def preprocess_chunks(data):
    print("Preprocessing chunks...")
    processed = []
    
    for item in tqdm(data, desc="Chunks"):
        sample_id = item.get('_id')
        question = extract_question(item.get('messages', []))
        chunks = []
        
        for msg in item.get('messages', []):
            if msg.get('role') == 'assistant':
                chunks = smart_chunking(msg.get('content', ''))
                break
        
        processed.append({
            'sample_id': sample_id,
            'question': question,
            'chunks': chunks
        })
    return processed


# Ensemble TF-IDF model definition
class EnsembleTfidfModel:
    def __init__(self):
        self.models = {
            'dense': TfidfVectorizer(
                max_features=config.CHUNK_MAX_FEATURES,
                ngram_range=config.NGRAM_RANGE,
                stop_words=list(STOP_WORDS),
                lowercase=True,
                token_pattern=r"(?u)\b[a-z]{2,}\b",
                sublinear_tf=True,
                min_df=1,
                max_df=0.9,
                smooth_idf=True
            ),
            'sparse': TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words=list(STOP_WORDS),
                lowercase=True,
                binary=True,
                min_df=2
            ),
            'doc': TfidfVectorizer(
                max_features=config.DOC_MAX_FEATURES,
                ngram_range=(1, 2),
                lowercase=True,
                stop_words=None,
                binary=False
            )
        }
    
    def fit_all(self, doc_types, all_chunks, all_questions):
        print("Fitting ensemble TF-IDF models...")
        
        all_texts = all_chunks + all_questions + doc_types
        self.models['dense'].fit(all_texts)
        self.models['sparse'].fit(all_texts)
        self.models['doc'].fit(doc_types + all_questions)
        
        print("âœ… Ensemble models fitted!")
    
    def encode_ensemble(self, texts, model_type='dense'):
        dense_vec = self.models['dense'].transform(texts).toarray()
        sparse_vec = self.models['sparse'].transform(texts).toarray()
        combined = np.hstack([dense_vec, sparse_vec * 0.3])
        return normalize(combined, norm='l2', axis=1)
    
    def encode_doc_ensemble(self, texts):
        return normalize(self.models['doc'].transform(texts).toarray(), norm='l2', axis=1)
    
    def ensemble_similarity(self, q_vec, doc_vecs, alpha=0.7):
        sim1 = cosine_similarity(q_vec, doc_vecs)
        return alpha * sim1 + (1-alpha) * (sim1 ** 2)


# Document and chunk ranking functions
def rank_documents(model, question, docs):
    if not question.strip():
        return [0, 1, 2, 3, 4]
    
    q_vec = model.encode_doc_ensemble([question])
    doc_vecs = model.encode_doc_ensemble(docs)
    scores = model.ensemble_similarity(q_vec, doc_vecs)
    
    return np.argsort(scores.flatten())[-5:][::-1].tolist()

def rank_chunks(model, question, chunks):
    if not chunks or not question.strip():
        return list(range(min(5, len(chunks))))
    
    q_vec = model.encode_ensemble([question])
    chunk_vecs = model.encode_ensemble(chunks)
    scores = model.ensemble_similarity(q_vec, chunk_vecs)
    
    return np.argsort(scores.flatten())[-5:][::-1].tolist()

def predict_documents(model, processed_docs):
    print("ğŸ”� Advanced document ranking...")
    predictions = []
    for item in tqdm(processed_docs, desc="Doc Ranking"):
        ranking = rank_documents(model, item['question'], item['docs'])
        for idx in ranking:
            predictions.append((item['sample_id'], idx))
    return predictions

def predict_chunks(model, processed_chunks):
    print("ğŸ”� Advanced chunk ranking...")
    predictions = []
    for item in tqdm(processed_chunks, desc="Chunk Ranking"):
        if not item['chunks']:
            for j in range(5):
                predictions.append((item['sample_id'], j))
        else:
            ranking = rank_chunks(model, item['question'], item['chunks'])
            for idx in ranking:
                predictions.append((item['sample_id'], idx))
    return predictions


# Main execution and submission
def main():
    print("=== ENSEMBLE TF-IDF + NLP RETRIEVAL ===")
    
    doc_eval = load_jsonl(INPUT_DIR / 'document_ranking_kaggle_eval.jsonl')
    chunk_eval = load_jsonl(INPUT_DIR / 'chunk_ranking_kaggle_eval.jsonl')
    
    doc_processed = preprocess_docs(doc_eval)
    chunk_processed = preprocess_chunks(chunk_eval)
    
    all_questions = []
    all_chunks = []
    
    for item in doc_processed + chunk_processed:
        if item['question']:
            all_questions.append(item['question'])
    
    for item in chunk_processed:
        all_chunks.extend(item['chunks'])
    
    print(f"Training corpus: {len(all_questions)} questions, {len(all_chunks)} chunks")
    
    model = EnsembleTfidfModel()
    doc_types = ["10-K", "10-Q", "8-K", "DEF-14A", "Earnings Transcript"]
    model.fit_all(doc_types, all_chunks, all_questions)
    
    doc_preds = predict_documents(model, doc_processed)
    chunk_preds = predict_chunks(model, chunk_processed)
    
    all_preds = doc_preds + chunk_preds
    df = pd.DataFrame(all_preds, columns=['sample_id', 'target_index'])
    
    df = df.sort_values('sample_id').groupby('sample_id').head(5).reset_index(drop=True)
    
    output_path = OUTPUT_DIR / 'submission.csv'
    df.to_csv(output_path, index=False)
    
    final_counts = df['sample_id'].value_counts()
    print(f"\nâœ… SUBMISSION SAVED: {output_path}")
    print(f"Total predictions: {len(df)}")
    print(f"Perfect samples: {(final_counts == 5).sum()}/{len(final_counts)}")
    
    return df

if __name__ == "__main__":
    submission = main()




