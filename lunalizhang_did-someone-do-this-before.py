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


import subprocess
import sys
import os
import re
import json
import numpy as np
import pandas as pd
import requests
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("SEC FILING CROSS-COMPANY SIMILARITY & DIVERGENCE ANALYZER")
print("="*100)

def install_packages():
    packages = [
        'yfinance', 'scikit-learn', 'scipy', 'seaborn',
        'plotly', 'python-dateutil', 'beautifulsoup4', 'lxml',
        'sentence-transformers', 'transformers', 'torch',
        'faiss-cpu', 'matplotlib', 'tqdm', 'colorama', 'statsmodels', 
        'kaleido', 'wordcloud', 'Pillow', 'networkx'
    ]
    
    print("\n[SETUP] Installing required packages...")
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✓ {package} already installed")
        except ImportError:
            try:
                print(f"  → Installing {package}...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])
                print(f"  ✓ {package} installed successfully")
            except:
                print(f"  ⚠ Warning: Could not install {package}")
    print("[SETUP] Package installation complete.\n")

install_packages()

import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from tqdm import tqdm
import time
from bs4 import BeautifulSoup
from colorama import init, Fore, Back, Style
import faiss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import display, HTML, Image as IPImage, Markdown
import base64
from io import BytesIO
from PIL import Image
from wordcloud import WordCloud
import networkx as nx
from collections import defaultdict

init(autoreset=True)
pyo.init_notebook_mode(connected=True)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def show_plot_kaggle(fig=None, title="Chart"):
    if fig is None:
        fig = plt.gcf()
    
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white', edgecolor='none')
    buf.seek(0)
    
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    html = f"""
    <div style='text-align: center; margin: 20px 0;'>
        <h3 style='color: #333; margin-bottom: 10px;'>{title}</h3>
        <img src='data:image/png;base64,{img_base64}' style='max-width: 100%; border: 1px solid #ddd; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
    </div>
    """
    display(HTML(html))
    
    try:
        filename = f"{title.replace(' ', '_').replace(':', '').replace('/', '_')}.png"
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  → Chart saved as: {filename}")
    except:
        pass
    
    plt.close(fig)

@dataclass
class AnalyzerConfig:
    llama_model_path: str = "/kaggle/input/llama-3.2/transformers/3b-instruct/1"
    qwen_model_path: str = "/kaggle/input/qwen2.5/transformers/7b-instruct/1"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    sentence_chunk_size: int = 2
    paragraph_chunk_size: int = 3
    section_chunk_size: int = 5
    overlap_size: int = 1
    max_chunks_per_doc: int = 300
    min_chunk_length: int = 100
    
    max_filings_per_company: int = 3
    analysis_years: int = 1
    
    similarity_threshold: float = 0.80
    divergence_threshold: float = 0.25
    min_similarity_score: float = 0.75
    max_divergence_score: float = 0.30
    
    create_visualizations: bool = True
    verbose_output: bool = True
    show_chunk_details: bool = True
    save_charts: bool = True
    display_inline: bool = True
    
    companies: Dict[str, Dict] = field(default_factory=lambda: {
        'AAPL': {'cik': '0000320193', 'name': 'Apple Inc', 'sector': 'Technology'},
        'MSFT': {'cik': '0000789019', 'name': 'Microsoft Corporation', 'sector': 'Technology'},
        'GOOGL': {'cik': '0001652044', 'name': 'Alphabet Inc', 'sector': 'Technology'},
        'META': {'cik': '0001326801', 'name': 'Meta Platforms', 'sector': 'Technology'},
        'AMZN': {'cik': '0001018724', 'name': 'Amazon.com Inc', 'sector': 'Technology'},
        'NVDA': {'cik': '0001045810', 'name': 'NVIDIA Corporation', 'sector': 'Technology'},
        'TSLA': {'cik': '0001318605', 'name': 'Tesla Inc', 'sector': 'Automotive'},
        'NFLX': {'cik': '0001065280', 'name': 'Netflix Inc', 'sector': 'Entertainment'}
    })
    
    activity_patterns: Dict[str, List[str]] = field(default_factory=lambda: {
        'ai_investment': [
            "artificial intelligence research and development",
            "machine learning infrastructure investments",
            "AI model training and deployment",
            "generative AI capabilities",
            "neural network development"
        ],
        'cloud_expansion': [
            "cloud infrastructure expansion",
            "data center construction and capacity",
            "cloud services revenue growth",
            "enterprise cloud solutions",
            "cloud computing investments"
        ],
        'cost_reduction': [
            "workforce reduction initiatives",
            "operational efficiency programs",
            "cost cutting measures",
            "restructuring activities",
            "expense optimization"
        ],
        'market_expansion': [
            "international market entry",
            "geographic expansion strategy",
            "new market penetration",
            "regional growth initiatives",
            "market share expansion"
        ],
        'product_innovation': [
            "new product development",
            "innovation pipeline",
            "R&D investments",
            "technology breakthroughs",
            "product portfolio expansion"
        ],
        'acquisitions': [
            "strategic acquisitions",
            "merger and acquisition activity",
            "business combinations",
            "acquisition integration",
            "M&A strategy"
        ],
        'partnerships': [
            "strategic partnerships",
            "joint ventures",
            "collaboration agreements",
            "partnership ecosystem",
            "alliance formation"
        ],
        'regulatory_compliance': [
            "regulatory compliance initiatives",
            "data privacy regulations",
            "compliance infrastructure",
            "regulatory requirements",
            "government regulations"
        ],
        'sustainability': [
            "sustainability initiatives",
            "carbon neutrality goals",
            "renewable energy adoption",
            "environmental commitments",
            "ESG initiatives"
        ],
        'cybersecurity': [
            "cybersecurity investments",
            "security infrastructure",
            "data protection measures",
            "security breach prevention",
            "cyber threat mitigation"
        ]
    })

class SECDataFetcher:
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.base_url = "https://www.sec.gov/Archives/edgar/data"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 AnalyzerBot/1.0',
            'Accept': 'application/json, text/html'
        }
        
    def fetch_filings(self, ticker: str, company_info: Dict) -> List[Dict]:
        print(f"\n{Fore.CYAN}[SEC FETCHER] Retrieving filings for {ticker}...{Style.RESET_ALL}")
        
        filings = []
        cik = company_info['cik'].lstrip('0')
        
        try:
            index_url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
            
            print(f"  → Connecting to SEC EDGAR...")
            response = requests.get(index_url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                print(f"  {Fore.GREEN}✓ Connected successfully{Style.RESET_ALL}")
                data = response.json()
                recent = data.get('filings', {}).get('recent', {})
                
                forms = recent.get('form', [])
                dates = recent.get('filingDate', [])
                accession_numbers = recent.get('accessionNumber', [])
                
                print(f"  → Found {len(forms)} total filings")
                
                cutoff_date = datetime.now() - timedelta(days=self.config.analysis_years * 365)
                target_forms = ['10-K', '10-Q', '8-K']
                filing_count = 0
                
                for i in range(min(len(forms), 50)):
                    if forms[i] in target_forms and filing_count < self.config.max_filings_per_company:
                        filing_date = pd.to_datetime(dates[i])
                        if filing_date >= cutoff_date:
                            print(f"\n  → Fetching {forms[i]} from {dates[i]}...")
                            content = self._fetch_filing_content(cik, accession_numbers[i])
                            
                            if content and len(content) > 1000:
                                filings.append({
                                    'ticker': ticker,
                                    'company_name': company_info['name'],
                                    'sector': company_info.get('sector', 'Unknown'),
                                    'form_type': forms[i],
                                    'filing_date': dates[i],
                                    'accession_number': accession_numbers[i],
                                    'content': content
                                })
                                print(f"  {Fore.GREEN}✓ Fetched {forms[i]} ({len(content):,} characters){Style.RESET_ALL}")
                                filing_count += 1
                                
                                if filing_count >= self.config.max_filings_per_company:
                                    break
                
                print(f"\n  {Fore.GREEN}Successfully retrieved {len(filings)} filings{Style.RESET_ALL}")
                
            else:
                print(f"  {Fore.YELLOW}⚠ Could not connect (Status: {response.status_code}){Style.RESET_ALL}")
                
        except Exception as e:
            print(f"  {Fore.RED}✗ Error: {e}{Style.RESET_ALL}")
        
        return filings
    
    def _fetch_filing_content(self, cik: str, accession_number: str) -> str:
        try:
            acc_no_dash = accession_number.replace('-', '')
            url = f"{self.base_url}/{cik}/{acc_no_dash}/{accession_number}.txt"
            
            response = requests.get(url, headers=self.headers, timeout=15)
            
            if response.status_code == 200:
                text = response.text
                
                if '<html' in text.lower():
                    soup = BeautifulSoup(text, 'html.parser')
                    text = soup.get_text()
                
                return self._clean_text(text)
                
        except Exception as e:
            print(f"    ⚠ Error: {e}")
            
        return ""
    
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = text.strip()
        return text[:500000]

class ChunkProcessor:
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        
    def create_chunks(self, text: str, chunk_type: str = 'mixed') -> List[Dict]:
        chunks = []
        
        if chunk_type == 'sentence':
            chunks.extend(self._create_sentence_chunks(text))
        elif chunk_type == 'paragraph':
            chunks.extend(self._create_paragraph_chunks(text))
        elif chunk_type == 'section':
            chunks.extend(self._create_section_chunks(text))
        else:
            text_len = len(text)
            chunks.extend(self._create_sentence_chunks(text[:text_len//3]))
            chunks.extend(self._create_paragraph_chunks(text[text_len//3:2*text_len//3]))
            chunks.extend(self._create_section_chunks(text[2*text_len//3:]))
        
        print(f"  → Created {len(chunks)} chunks ({chunk_type} mode)")
        return chunks[:self.config.max_chunks_per_doc]
    
    def _create_sentence_chunks(self, text: str) -> List[Dict]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > self.config.min_chunk_length]
        
        chunks = []
        for i in range(0, len(sentences), self.config.sentence_chunk_size):
            chunk_sentences = sentences[i:i + self.config.sentence_chunk_size]
            if chunk_sentences:
                chunk_text = ' '.join(chunk_sentences)
                if len(chunk_text) > self.config.min_chunk_length:
                    chunks.append({
                        'id': len(chunks),
                        'type': 'sentence',
                        'text': chunk_text,
                        'start_idx': i,
                        'end_idx': i + len(chunk_sentences)
                    })
        
        return chunks
    
    def _create_paragraph_chunks(self, text: str) -> List[Dict]:
        paragraphs = re.split(r'\n\n+', text)
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > self.config.min_chunk_length]
        
        chunks = []
        for i in range(0, len(paragraphs), self.config.paragraph_chunk_size):
            chunk_paragraphs = paragraphs[i:i + self.config.paragraph_chunk_size]
            if chunk_paragraphs:
                chunk_text = '\n\n'.join(chunk_paragraphs)
                if len(chunk_text) > self.config.min_chunk_length:
                    chunks.append({
                        'id': len(chunks),
                        'type': 'paragraph',
                        'text': chunk_text,
                        'start_idx': i,
                        'end_idx': i + len(chunk_paragraphs)
                    })
        
        return chunks
    
    def _create_section_chunks(self, text: str) -> List[Dict]:
        section_markers = [
            r'item\s+\d+[^\d]',
            r'section\s+\d+',
            r'part\s+[IVX]+',
            r'\d+\.\s+[A-Z]',
        ]
        
        sections = []
        current_section = ""
        
        for line in text.split('\n'):
            if any(re.search(marker, line, re.IGNORECASE) for marker in section_markers):
                if current_section and len(current_section) > self.config.min_chunk_length:
                    sections.append(current_section)
                current_section = line
            else:
                current_section += " " + line
        
        if current_section:
            sections.append(current_section)
        
        chunks = []
        for i, section in enumerate(sections[:self.config.max_chunks_per_doc // 3]):
            if len(section.strip()) > self.config.min_chunk_length:
                chunks.append({
                    'id': len(chunks),
                    'type': 'section',
                    'text': section.strip()[:5000],
                    'start_idx': i,
                    'end_idx': i + 1
                })
        
        return chunks

class SimilarityAnalyzer:
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.llama_model = None
        self.tokenizer = None
        self.qwen_model = None
        self.qwen_tokenizer = None
        self.embedding_model = None
        self.activity_embeddings = {}
        
        self._initialize_models()
        
    def _initialize_models(self):
        print(f"\n{Fore.YELLOW}[MODEL INIT] Loading AI models...{Style.RESET_ALL}")
        
        print(f"  → Loading embedding model...")
        self.embedding_model = SentenceTransformer(self.config.embedding_model)
        print(f"  {Fore.GREEN}✓ Embedding model ready{Style.RESET_ALL}")
        
        try:
            print(f"  → Loading Llama 3.2...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.llama_model_path,
                trust_remote_code=True
            )
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                self.config.llama_model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"  {Fore.GREEN}✓ Llama 3.2 ready{Style.RESET_ALL}")
        except:
            print(f"  {Fore.YELLOW}⚠ Llama unavailable, using fallback{Style.RESET_ALL}")
        
        try:
            print(f"  → Loading Qwen 2.5...")
            self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                self.config.qwen_model_path,
                trust_remote_code=True
            )
            self.qwen_model = AutoModelForCausalLM.from_pretrained(
                self.config.qwen_model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            print(f"  {Fore.GREEN}✓ Qwen 2.5 ready{Style.RESET_ALL}")
        except:
            print(f"  {Fore.YELLOW}⚠ Qwen unavailable{Style.RESET_ALL}")
        
        print(f"\n  → Computing activity pattern embeddings...")
        for activity_name, examples in self.config.activity_patterns.items():
            self.activity_embeddings[activity_name] = self.embedding_model.encode(examples)
            print(f"    • {activity_name}: {len(examples)} examples encoded")
        
        print(f"\n{Fore.GREEN}[MODEL INIT] All models initialized{Style.RESET_ALL}")
    
    def analyze_chunk(self, chunk: Dict, company: str) -> Dict:
        text = chunk['text']
        
        chunk_embedding = self.embedding_model.encode(text)
        
        activity_scores = {}
        best_activity = None
        best_score = 0
        
        for activity_name, activity_embeds in self.activity_embeddings.items():
            similarities = util.cos_sim(chunk_embedding, activity_embeds)
            max_similarity = float(torch.max(similarities))
            activity_scores[activity_name] = max_similarity
            
            if max_similarity > best_score:
                best_score = max_similarity
                best_activity = activity_name
        
        llm_analysis = self._analyze_with_llm(text, company)
        
        result = {
            'chunk_id': chunk['id'],
            'chunk_type': chunk['type'],
            'text': text,
            'text_preview': text[:200],
            'company': company,
            'activity_scores': activity_scores,
            'best_activity': best_activity,
            'best_activity_score': float(best_score),
            'llm_analysis': llm_analysis,
            'embedding': chunk_embedding.tolist()
        }
        
        return result
    
    def _analyze_with_llm(self, text: str, company: str) -> Dict:
        if self.llama_model:
            return self._llama_analysis(text, company)
        elif self.qwen_model:
            return self._qwen_analysis(text, company)
        else:
            return self._keyword_analysis(text, company)
    
    def _llama_analysis(self, text: str, company: str) -> Dict:
        try:
            prompt = f"""Analyze this SEC filing excerpt from {company} and identify key business activities.

Text: {text[:400]}

Identify:
1. Main business activity or strategy described
2. Specific action being taken (expanding/reducing/investing/divesting)
3. Target area or market

Format: activity|action|target"""

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=600)
            
            with torch.no_grad():
                outputs = self.llama_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], 
                                            skip_special_tokens=True)
            
            parts = response.split('|')
            if len(parts) >= 3:
                activity = parts[0].strip()
                action = parts[1].strip()
                target = parts[2].strip()
            else:
                activity = "general business"
                action = "operating"
                target = "various markets"
            
            return {
                'activity': activity[:100],
                'action': action[:50],
                'target': target[:100],
                'method': 'llama'
            }
            
        except Exception as e:
            return self._keyword_analysis(text, company)
    
    def _qwen_analysis(self, text: str, company: str) -> Dict:
        try:
            prompt = f"""SEC Filing Analysis for {company}:
{text[:400]}

Extract:
1. Business activity
2. Action type
3. Focus area

Response: activity|action|area"""

            inputs = self.qwen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=600)
            
            with torch.no_grad():
                outputs = self.qwen_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.qwen_tokenizer.eos_token_id
                )
            
            response = self.qwen_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], 
                                                  skip_special_tokens=True)
            
            parts = response.split('|')
            if len(parts) >= 3:
                activity = parts[0].strip()
                action = parts[1].strip()
                target = parts[2].strip()
            else:
                activity = "general business"
                action = "operating"
                target = "various markets"
            
            return {
                'activity': activity[:100],
                'action': action[:50],
                'target': target[:100],
                'method': 'qwen'
            }
            
        except Exception as e:
            return self._keyword_analysis(text, company)
    
    def _keyword_analysis(self, text: str, company: str) -> Dict:
        text_lower = text.lower()
        
        activities = {
            'expansion': ['expand', 'growth', 'increase', 'scale', 'enter'],
            'reduction': ['reduce', 'cut', 'decrease', 'eliminate', 'streamline'],
            'investment': ['invest', 'develop', 'build', 'acquire', 'purchase'],
            'innovation': ['innovate', 'research', 'develop', 'create', 'design'],
            'partnership': ['partner', 'collaborate', 'joint', 'alliance', 'agreement']
        }
        
        detected_activity = "general operations"
        detected_action = "maintaining"
        
        for activity, keywords in activities.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_activity = activity
                detected_action = keywords[0]
                break
        
        market_keywords = ['market', 'region', 'segment', 'industry', 'sector']
        target = "core business"
        for keyword in market_keywords:
            if keyword in text_lower:
                idx = text_lower.index(keyword)
                target = text[max(0, idx-20):min(len(text), idx+50)]
                target = ' '.join(target.split()[:5])
                break
        
        return {
            'activity': detected_activity,
            'action': detected_action,
            'target': target,
            'method': 'keyword'
        }

class CrossCompanyComparator:
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.similarity_analyzer = None
        
    def find_similarities_and_differences(self, all_chunks: Dict[str, List[Dict]]) -> Dict:
        print(f"\n{Fore.YELLOW}[CROSS-COMPANY] Analyzing similarities and differences...{Style.RESET_ALL}")
        
        similarities = []
        differences = []
        company_pairs = []
        
        companies = list(all_chunks.keys())
        
        for i, company1 in enumerate(companies):
            for j, company2 in enumerate(companies):
                if i < j:
                    print(f"\n  Comparing {company1} vs {company2}...")
                    
                    for chunk1 in all_chunks[company1][:50]:
                        embedding1 = np.array(chunk1['embedding'])
                        
                        for chunk2 in all_chunks[company2][:50]:
                            embedding2 = np.array(chunk2['embedding'])
                            
                            similarity = 1 - cosine(embedding1, embedding2)
                            
                            if similarity > self.config.similarity_threshold:
                                similarity_entry = {
                                    'company1': company1,
                                    'company2': company2,
                                    'chunk1': chunk1,
                                    'chunk2': chunk2,
                                    'similarity_score': float(similarity),
                                    'activity1': chunk1['best_activity'],
                                    'activity2': chunk2['best_activity'],
                                    'is_similar': True
                                }
                                similarities.append(similarity_entry)
                                
                                if similarity > 0.9:
                                    print(f"    {Fore.GREEN}★ HIGH SIMILARITY ({similarity:.3f}): "
                                          f"{chunk1['best_activity']} ↔ {chunk2['best_activity']}{Style.RESET_ALL}")
                            
                            elif chunk1['best_activity'] == chunk2['best_activity'] and similarity < self.config.divergence_threshold:
                                difference_entry = {
                                    'company1': company1,
                                    'company2': company2,
                                    'chunk1': chunk1,
                                    'chunk2': chunk2,
                                    'similarity_score': float(similarity),
                                    'activity': chunk1['best_activity'],
                                    'is_opposite': True
                                }
                                differences.append(difference_entry)
                                
                                print(f"    {Fore.RED}◆ OPPOSITE APPROACH ({similarity:.3f}): "
                                      f"{chunk1['best_activity']}{Style.RESET_ALL}")
                    
                    company_pairs.append({
                        'company1': company1,
                        'company2': company2,
                        'num_similarities': len([s for s in similarities 
                                               if s['company1'] == company1 and s['company2'] == company2]),
                        'num_differences': len([d for d in differences 
                                              if d['company1'] == company1 and d['company2'] == company2])
                    })
        
        activity_clusters = self._cluster_by_activity(all_chunks)
        
        return {
            'similarities': similarities,
            'differences': differences,
            'company_pairs': company_pairs,
            'activity_clusters': activity_clusters,
            'summary_stats': {
                'total_similarities': len(similarities),
                'total_differences': len(differences),
                'avg_similarity_score': np.mean([s['similarity_score'] for s in similarities]) if similarities else 0,
                'companies_analyzed': len(companies)
            }
        }
    
    def _cluster_by_activity(self, all_chunks: Dict[str, List[Dict]]) -> Dict:
        activity_clusters = defaultdict(list)
        
        for company, chunks in all_chunks.items():
            for chunk in chunks:
                best_activity = chunk['best_activity']
                if chunk['best_activity_score'] > 0.6:
                    activity_clusters[best_activity].append({
                        'company': company,
                        'chunk_preview': chunk['text_preview'],
                        'score': chunk['best_activity_score'],
                        'llm_activity': chunk['llm_analysis']['activity']
                    })
        
        for activity in activity_clusters:
            activity_clusters[activity] = sorted(
                activity_clusters[activity], 
                key=lambda x: x['score'], 
                reverse=True
            )[:20]
        
        return dict(activity_clusters)

class VisualizationEngine:
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.charts_created = 0
        
    def create_similarity_network(self, comparison_results: Dict):
        if not self.config.create_visualizations:
            return
        
        G = nx.Graph()
        
        company_pairs = comparison_results['company_pairs']
        
        for pair in company_pairs:
            if pair['num_similarities'] > 0:
                G.add_edge(
                    pair['company1'], 
                    pair['company2'],
                    weight=pair['num_similarities'],
                    differences=pair['num_differences']
                )
        
        if len(G.nodes()) == 0:
            return
        
        plt.figure(figsize=(15, 10))
        
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', 
                              edgecolors='darkblue', linewidths=2)
        
        nx.draw_networkx_edges(G, pos, width=[w/5 for w in weights], alpha=0.6)
        
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        edge_labels = nx.get_edge_attributes(G, 'weight')
        edge_labels = {k: f"Sim:{v}\nDiff:{G[k[0]][k[1]]['differences']}" 
                      for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
        
        plt.title("Company Similarity Network\n(Edge width = number of similar passages)", 
                 fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if self.config.save_charts:
            filename = f"company_similarity_network.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"  → Saved: {filename}")
        
        if self.config.display_inline:
            show_plot_kaggle(plt.gcf(), "Company Similarity Network")
        
        plt.close()
    
    def create_activity_heatmap(self, activity_clusters: Dict, companies: List[str]):
        if not self.config.create_visualizations:
            return
        
        activities = list(activity_clusters.keys())
        
        matrix = np.zeros((len(companies), len(activities)))
        
        for j, activity in enumerate(activities):
            for entry in activity_clusters[activity]:
                if entry['company'] in companies:
                    i = companies.index(entry['company'])
                    matrix[i, j] += 1
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        
        ax.set_xticks(np.arange(len(activities)))
        ax.set_yticks(np.arange(len(companies)))
        ax.set_xticklabels(activities, rotation=45, ha='right')
        ax.set_yticklabels(companies)
        
        for i in range(len(companies)):
            for j in range(len(activities)):
                text = ax.text(j, i, int(matrix[i, j]),
                             ha="center", va="center", color="black")
        
        ax.set_title("Company Activity Heatmap\n(Number of mentions per activity type)", 
                    fontsize=14, fontweight='bold')
        fig.tight_layout()
        
        plt.colorbar(im, ax=ax)
        
        if self.config.save_charts:
            filename = f"activity_heatmap.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"  → Saved: {filename}")
        
        if self.config.display_inline:
            show_plot_kaggle(fig, "Company Activity Heatmap")
        
        plt.close()
    
    def create_similarity_matrix(self, comparison_results: Dict, companies: List[str]):
        if not self.config.create_visualizations:
            return
        
        n = len(companies)
        matrix = np.zeros((n, n))
        
        for sim in comparison_results['similarities']:
            i = companies.index(sim['company1'])
            j = companies.index(sim['company2'])
            matrix[i, j] += 1
            matrix[j, i] += 1
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        mask = np.triu(np.ones_like(matrix, dtype=bool))
        
        sns.heatmap(matrix, mask=mask, annot=True, fmt='.0f', cmap='Blues',
                   xticklabels=companies, yticklabels=companies,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title("Company Similarity Matrix\n(Count of similar passages)", 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if self.config.save_charts:
            filename = f"similarity_matrix.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"  → Saved: {filename}")
        
        if self.config.display_inline:
            show_plot_kaggle(fig, "Company Similarity Matrix")
        
        plt.close()
    
    def create_divergence_chart(self, comparison_results: Dict):
        if not self.config.create_visualizations or not comparison_results['differences']:
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        differences = comparison_results['differences'][:20]
        
        ax1 = axes[0]
        activities = [d['activity'] for d in differences]
        scores = [d['similarity_score'] for d in differences]
        labels = [f"{d['company1']} vs {d['company2']}" for d in differences]
        
        bars = ax1.barh(range(len(differences)), scores, color='red', alpha=0.7)
        ax1.set_yticks(range(len(differences)))
        ax1.set_yticklabels([f"{l}\n{a}" for l, a in zip(labels, activities)], fontsize=8)
        ax1.set_xlabel('Divergence Score (lower = more opposite)')
        ax1.set_title('Top Divergent Approaches Between Companies', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        activity_counts = defaultdict(int)
        for d in comparison_results['differences']:
            activity_counts[d['activity']] += 1
        
        activities = list(activity_counts.keys())
        counts = list(activity_counts.values())
        
        ax2.bar(activities, counts, color='coral', alpha=0.7)
        ax2.set_xlabel('Activity Type')
        ax2.set_ylabel('Number of Divergent Approaches')
        ax2.set_title('Activities with Most Divergent Approaches', fontweight='bold')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.config.save_charts:
            filename = f"divergence_analysis.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"  → Saved: {filename}")
        
        if self.config.display_inline:
            show_plot_kaggle(fig, "Divergence Analysis")
        
        plt.close()

class CrossCompanyAnalyzer:
    def __init__(self):
        self.config = AnalyzerConfig()
        
        print(f"\n{Fore.CYAN}{'='*100}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}INITIALIZING CROSS-COMPANY SEC ANALYZER{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*100}{Style.RESET_ALL}")
        
        self.data_fetcher = SECDataFetcher(self.config)
        self.chunk_processor = ChunkProcessor(self.config)
        self.similarity_analyzer = SimilarityAnalyzer(self.config)
        self.comparator = CrossCompanyComparator(self.config)
        self.visualization_engine = VisualizationEngine(self.config)
        
        print(f"\n{Fore.GREEN}[SYSTEM] All components initialized successfully{Style.RESET_ALL}")
    
    def analyze_all_companies(self) -> Dict:
        all_filings = {}
        all_chunks = {}
        
        for ticker, company_info in self.config.companies.items():
            print(f"\n{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}FETCHING FILINGS FOR {ticker} - {company_info['name']}{Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}")
            
            filings = self.data_fetcher.fetch_filings(ticker, company_info)
            
            if filings:
                all_filings[ticker] = filings
                
                company_chunks = []
                
                for filing in filings:
                    print(f"\n  Processing {filing['form_type']} from {filing['filing_date']}...")
                    
                    chunks = self.chunk_processor.create_chunks(filing['content'], chunk_type='mixed')
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        if chunk_idx % 20 == 0 and chunk_idx > 0:
                            print(f"    Progress: {chunk_idx}/{len(chunks)} chunks")
                        
                        chunk_analysis = self.similarity_analyzer.analyze_chunk(chunk, ticker)
                        chunk_analysis['filing_date'] = filing['filing_date']
                        chunk_analysis['form_type'] = filing['form_type']
                        company_chunks.append(chunk_analysis)
                
                all_chunks[ticker] = company_chunks
                print(f"\n  {Fore.GREEN}✓ Analyzed {len(company_chunks)} chunks for {ticker}{Style.RESET_ALL}")
            
            time.sleep(1)
        
        print(f"\n{Fore.CYAN}{'='*100}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}PERFORMING CROSS-COMPANY COMPARISON{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*100}{Style.RESET_ALL}")
        
        comparison_results = self.comparator.find_similarities_and_differences(all_chunks)
        
        self._generate_reports(comparison_results, all_chunks)
        
        if self.config.create_visualizations:
            print(f"\n{Fore.YELLOW}[VISUALIZATION] Creating charts...{Style.RESET_ALL}")
            companies = list(all_chunks.keys())
            self.visualization_engine.create_similarity_network(comparison_results)
            self.visualization_engine.create_activity_heatmap(
                comparison_results['activity_clusters'], 
                companies
            )
            self.visualization_engine.create_similarity_matrix(comparison_results, companies)
            self.visualization_engine.create_divergence_chart(comparison_results)
        
        return {
            'all_filings': all_filings,
            'all_chunks': all_chunks,
            'comparison_results': comparison_results
        }
    
    def _generate_reports(self, comparison_results: Dict, all_chunks: Dict):
        print(f"\n{Fore.CYAN}{'='*100}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}SIMILARITY REPORT{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*100}{Style.RESET_ALL}")
        
        if comparison_results['similarities']:
            top_similarities = sorted(
                comparison_results['similarities'], 
                key=lambda x: x['similarity_score'], 
                reverse=True
            )[:10]
            
            for idx, sim in enumerate(top_similarities, 1):
                print(f"\n{Fore.GREEN}[SIMILARITY #{idx}] Score: {sim['similarity_score']:.3f}{Style.RESET_ALL}")
                print(f"  {sim['company1']} ↔ {sim['company2']}")
                print(f"  Activity: {sim['activity1']} ↔ {sim['activity2']}")
                print(f"  {sim['company1']}: {sim['chunk1']['text_preview'][:100]}...")
                print(f"  {sim['company2']}: {sim['chunk2']['text_preview'][:100]}...")
        
        print(f"\n{Fore.CYAN}{'='*100}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}DIVERGENCE REPORT{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*100}{Style.RESET_ALL}")
        
        if comparison_results['differences']:
            top_differences = sorted(
                comparison_results['differences'], 
                key=lambda x: x['similarity_score']
            )[:10]
            
            for idx, diff in enumerate(top_differences, 1):
                print(f"\n{Fore.RED}[OPPOSITE APPROACH #{idx}] Divergence: {diff['similarity_score']:.3f}{Style.RESET_ALL}")
                print(f"  {diff['company1']} vs {diff['company2']}")
                print(f"  Activity: {diff['activity']}")
                print(f"  {diff['company1']}: {diff['chunk1']['text_preview'][:100]}...")
                print(f"  {diff['company2']}: {diff['chunk2']['text_preview'][:100]}...")
        
        print(f"\n{Fore.CYAN}{'='*100}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}ACTIVITY CLUSTERS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*100}{Style.RESET_ALL}")
        
        for activity, entries in comparison_results['activity_clusters'].items():
            if entries:
                companies_in_cluster = list(set([e['company'] for e in entries]))
                print(f"\n{Fore.YELLOW}{activity.upper()}:{Style.RESET_ALL}")
                print(f"  Companies involved: {', '.join(companies_in_cluster)}")
                print(f"  Number of mentions: {len(entries)}")

def main():
    print(f"\n{Fore.MAGENTA}{'='*100}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}SEC FILING CROSS-COMPANY SIMILARITY & DIVERGENCE ANALYZER{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{'='*100}{Style.RESET_ALL}")
    
    analyzer = CrossCompanyAnalyzer()
    
    print(f"\n{Fore.GREEN}[CONFIG]{Style.RESET_ALL}")
    print(f"  Companies: {', '.join(analyzer.config.companies.keys())}")
    print(f"  Max filings per company: {analyzer.config.max_filings_per_company}")
    print(f"  Max chunks per document: {analyzer.config.max_chunks_per_doc}")
    print(f"  Similarity threshold: {analyzer.config.similarity_threshold}")
    print(f"  Divergence threshold: {analyzer.config.divergence_threshold}")
    
    results = analyzer.analyze_all_companies()
    
    print(f"\n{Fore.CYAN}{'='*100}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}FINAL SUMMARY{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*100}{Style.RESET_ALL}")
    
    summary = results['comparison_results']['summary_stats']
    print(f"\n  Companies analyzed: {summary['companies_analyzed']}")
    print(f"  Total similarities found: {summary['total_similarities']}")
    print(f"  Total divergences found: {summary['total_differences']}")
    print(f"  Average similarity score: {summary['avg_similarity_score']:.3f}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"cross_company_analysis_{timestamp}.json"
    
    output_data = {
        'config': {
            'companies': list(analyzer.config.companies.keys()),
            'analysis_date': timestamp
        },
        'summary': summary,
        'top_similarities': results['comparison_results']['similarities'][:50] if results['comparison_results']['similarities'] else [],
        'top_differences': results['comparison_results']['differences'][:50] if results['comparison_results']['differences'] else [],
        'activity_clusters': {k: v[:10] for k, v in results['comparison_results']['activity_clusters'].items()}
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n{Fore.GREEN}[SUCCESS] Results saved to: {output_file}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}[COMPLETE] Cross-company analysis finished successfully!{Style.RESET_ALL}")
    
    return results

if __name__ == "__main__":
    results = main()
    print(f"\n{Fore.CYAN}[END] Cross-Company SEC Analysis Complete{Style.RESET_ALL}")

