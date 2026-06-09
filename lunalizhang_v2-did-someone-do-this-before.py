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


#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ADVANCED SEC FILING SPECIFIC ACTION ANALYZER
Complete self-contained script for extracting and comparing specific business actions from SEC filings
"""

# Install required packages
import subprocess
import sys

print("Installing required packages...")
packages = [
    'yfinance',
    'scikit-learn',
    'scipy',
    'seaborn',
    'plotly',
    'python-dateutil',
    'beautifulsoup4',
    'lxml',
    'sentence-transformers',
    'transformers',
    'torch',
    'faiss-cpu',
    'matplotlib',
    'tqdm',
    'colorama',
    'statsmodels',
    'networkx',
    'wordcloud',
    'Pillow'
]

for package in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

print("Package installation complete.\n")

import os
import re
import json
import numpy as np
import pandas as pd
import requests
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
from tqdm import tqdm
import time
from bs4 import BeautifulSoup
from colorama import init, Fore, Back, Style
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import display, HTML
import base64
from io import BytesIO
import networkx as nx
from collections import defaultdict

init(autoreset=True)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

print("="*100)
print("ADVANCED SEC FILING SPECIFIC ACTION ANALYZER")
print("="*100)

@dataclass
class SpecificAction:
    action_type: str
    action_verb: str
    target: str
    location: Optional[str] = None
    timeframe: Optional[str] = None
    magnitude: Optional[str] = None
    reason: Optional[str] = None
    confidence: float = 0.0

@dataclass
class AnalyzerConfig:
    embedding_model: str = "all-MiniLM-L6-v2"
    
    sentence_chunk_size: int = 3
    paragraph_chunk_size: int = 2
    max_chunks_per_doc: int = 200
    min_chunk_length: int = 150
    
    max_filings_per_company: int = 2
    analysis_years: int = 1
    
    similarity_threshold: float = 0.85
    
    companies: Dict[str, Dict] = field(default_factory=lambda: {
        'AAPL': {'cik': '0000320193', 'name': 'Apple Inc', 'sector': 'Technology'},
        'MSFT': {'cik': '0000789019', 'name': 'Microsoft Corporation', 'sector': 'Technology'},
        'GOOGL': {'cik': '0001652044', 'name': 'Alphabet Inc', 'sector': 'Technology'},
        'META': {'cik': '0001326801', 'name': 'Meta Platforms', 'sector': 'Technology'},
        'AMZN': {'cik': '0001018724', 'name': 'Amazon.com Inc', 'sector': 'Technology'},
        'NVDA': {'cik': '0001045810', 'name': 'NVIDIA Corporation', 'sector': 'Technology'}
    })
    
    specific_action_patterns: Dict[str, List[Dict]] = field(default_factory=lambda: {
        'market_exit': [
            {'pattern': r'(exit|withdraw|leave|discontinue|shut down|close|ceased|terminated).{0,50}(market|operations|business|stores?|facilities?|countries?|regions?)', 'priority': 1},
            {'pattern': r'(divest|sell off|dispose).{0,30}(operations|business|division)', 'priority': 2}
        ],
        'market_entry': [
            {'pattern': r'(enter|expand into|launch in|penetrate|establish|opened|launched).{0,50}(market|region|country|territory|stores?|offices?)', 'priority': 1},
            {'pattern': r'(beginning|starting|commencing).{0,30}operations.{0,30}in', 'priority': 2}
        ],
        'product_launch': [
            {'pattern': r'(launched|introduced|unveiled|released|announced|rolling out).{0,50}(product|service|platform|feature|solution|application)', 'priority': 1}
        ],
        'product_discontinuation': [
            {'pattern': r'(discontinue|sunset|retire|end.of.life|deprecate|phasing out).{0,50}(product|service|feature|offering)', 'priority': 1}
        ],
        'workforce_changes': [
            {'pattern': r'(layoffs?|reduction in force|workforce reduction|eliminate|cut).{0,30}(\d+|thousand|hundred|percent)', 'priority': 1},
            {'pattern': r'(hiring|recruiting|adding).{0,30}(\d+|thousand|hundred).{0,30}(employees|workers|staff)', 'priority': 2}
        ],
        'acquisition': [
            {'pattern': r'(acquired|purchased|bought|merger|acquisition).{0,50}(company|firm|business).{0,30}(\$?\d+|\w+)', 'priority': 1}
        ],
        'investment': [
            {'pattern': r'(invest|allocate|commit).{0,30}(\$?\d+\.?\d*\s*(billion|million|B|M)).{0,30}(in|to|for)', 'priority': 1}
        ]
    })

class ChunkProcessor:
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        
    def create_chunks(self, text: str) -> List[Dict]:
        chunks = []
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 50]
        
        for i in range(0, len(sentences), self.config.sentence_chunk_size):
            chunk_sentences = sentences[i:i + self.config.sentence_chunk_size]
            if chunk_sentences:
                chunk_text = ' '.join(chunk_sentences)
                if len(chunk_text) > self.config.min_chunk_length:
                    chunks.append({
                        'id': len(chunks),
                        'type': 'sentence',
                        'text': chunk_text
                    })
        
        paragraphs = re.split(r'\n\n+', text)
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > self.config.min_chunk_length]
        
        for i in range(0, len(paragraphs), self.config.paragraph_chunk_size):
            chunk_paragraphs = paragraphs[i:i + self.config.paragraph_chunk_size]
            if chunk_paragraphs:
                chunk_text = '\n\n'.join(chunk_paragraphs)
                if len(chunk_text) > self.config.min_chunk_length:
                    chunks.append({
                        'id': len(chunks),
                        'type': 'paragraph',
                        'text': chunk_text[:2000]
                    })
        
        return chunks[:self.config.max_chunks_per_doc]

class AdvancedActionExtractor:
    def __init__(self):
        self.locations = {
            'countries': ['United States', 'China', 'India', 'Germany', 'Japan', 'UK', 'France', 
                         'Brazil', 'Canada', 'Australia', 'Mexico', 'Russia', 'Italy', 'Spain'],
            'regions': ['Asia', 'Europe', 'Americas', 'Africa', 'Middle East', 'Latin America', 
                       'North America', 'Southeast Asia', 'Eastern Europe', 'Western Europe'],
            'states': ['California', 'Texas', 'New York', 'Florida', 'Illinois', 'Pennsylvania', 
                      'Ohio', 'Georgia', 'Washington', 'Massachusetts', 'Virginia', 'Arizona']
        }
        
        self.market_segments = ['enterprise', 'consumer', 'SMB', 'government', 'education', 
                               'healthcare', 'retail', 'financial', 'automotive', 'industrial']
    
    def extract_specific_actions(self, text: str, patterns: Dict) -> List[SpecificAction]:
        actions = []
        text_lower = text.lower()
        
        for action_type, pattern_list in patterns.items():
            for pattern_dict in pattern_list:
                pattern = pattern_dict['pattern']
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                
                for match in matches:
                    context_start = max(0, match.start() - 150)
                    context_end = min(len(text), match.end() + 150)
                    context = text[context_start:context_end]
                    
                    action = self._extract_action_details(context, action_type, match.group())
                    if action and action.confidence > 0.5:
                        actions.append(action)
        
        return actions
    
    def _extract_action_details(self, context: str, action_type: str, matched_text: str) -> Optional[SpecificAction]:
        action = SpecificAction(action_type=action_type, action_verb="", target="")
        
        verbs = re.findall(r'\b(exit|enter|launch|discontinue|acquire|invest|expand|reduce|close|open|divest|hire|layoff)\w*\b', 
                          context.lower())
        if verbs:
            action.action_verb = verbs[0]
        
        for location_type, location_list in self.locations.items():
            for location in location_list:
                if location.lower() in context.lower():
                    action.location = location
                    break
        
        amounts = re.findall(r'\$?([\d,]+(?:\.\d+)?)\s*(?:billion|million|thousand|B|M|K)', context, re.IGNORECASE)
        if amounts:
            action.magnitude = amounts[0]
        
        percentages = re.findall(r'(\d+(?:\.\d+)?)\s*(?:percent|%)', context, re.IGNORECASE)
        if percentages and not action.magnitude:
            action.magnitude = f"{percentages[0]}%"
        
        timeframes = re.findall(r'(?:Q[1-4]\s*)?20\d{2}|next\s+(?:quarter|year)|by\s+(?:end\s+of\s+)?20\d{2}', context, re.IGNORECASE)
        if timeframes:
            action.timeframe = timeframes[0]
        
        for segment in self.market_segments:
            if segment in context.lower():
                action.target = segment
                break
        
        if not action.target:
            products = re.findall(r'(?:our|the|new)\s+(\w+(?:\s+\w+)?)\s+(?:product|service|platform|business)', context.lower())
            if products:
                action.target = products[0]
        
        named_entities = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', context)
        for entity in named_entities:
            if len(entity) > 3 and entity not in ['The', 'We', 'Our', 'This']:
                if not action.target:
                    action.target = entity
        
        confidence = 0.3
        if action.action_verb:
            confidence += 0.2
        if action.location:
            confidence += 0.25
        if action.magnitude:
            confidence += 0.15
        if action.timeframe:
            confidence += 0.1
        if action.target:
            confidence += 0.15
        
        action.confidence = min(1.0, confidence)
        
        return action if action.confidence > 0.4 else None

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
            
            response = requests.get(index_url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                recent = data.get('filings', {}).get('recent', {})
                
                forms = recent.get('form', [])
                dates = recent.get('filingDate', [])
                accession_numbers = recent.get('accessionNumber', [])
                
                print(f"  → Found {len(forms)} total filings")
                
                cutoff_date = datetime.now() - timedelta(days=self.config.analysis_years * 365)
                target_forms = ['10-K', '10-Q', '8-K']
                filing_count = 0
                
                for i in range(min(len(forms), 20)):
                    if forms[i] in target_forms and filing_count < self.config.max_filings_per_company:
                        filing_date = pd.to_datetime(dates[i])
                        if filing_date >= cutoff_date:
                            content = self._fetch_filing_content(cik, accession_numbers[i])
                            
                            if content and len(content) > 1000:
                                filings.append({
                                    'ticker': ticker,
                                    'company_name': company_info['name'],
                                    'form_type': forms[i],
                                    'filing_date': dates[i],
                                    'content': content
                                })
                                print(f"  {Fore.GREEN}✓ Fetched {forms[i]} from {dates[i]}{Style.RESET_ALL}")
                                filing_count += 1
                                
                                if filing_count >= self.config.max_filings_per_company:
                                    break
                
                print(f"  {Fore.GREEN}Retrieved {len(filings)} filings{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"  {Fore.RED}Error: {e}{Style.RESET_ALL}")
        
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
                
        except:
            pass
            
        return ""
    
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        return text.strip()[:300000]

class EnhancedSimilarityAnalyzer:
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.embedding_model = None
        self.action_extractor = AdvancedActionExtractor()
        self._initialize_models()
    
    def _initialize_models(self):
        print(f"\n{Fore.YELLOW}[MODEL INIT] Loading embedding model...{Style.RESET_ALL}")
        self.embedding_model = SentenceTransformer(self.config.embedding_model)
        print(f"  {Fore.GREEN}✓ Embedding model ready{Style.RESET_ALL}")
    
    def analyze_chunk(self, chunk: Dict, company: str) -> Dict:
        text = chunk['text']
        
        chunk_embedding = self.embedding_model.encode(text)
        
        specific_actions = self.action_extractor.extract_specific_actions(
            text, 
            self.config.specific_action_patterns
        )
        
        best_action = None
        if specific_actions:
            best_action = max(specific_actions, key=lambda x: x.confidence)
        
        result = {
            'chunk_id': chunk['id'],
            'text': text[:500],
            'company': company,
            'specific_actions': [self._action_to_dict(a) for a in specific_actions],
            'best_action': self._action_to_dict(best_action) if best_action else None,
            'embedding': chunk_embedding.tolist()
        }
        
        return result
    
    def _action_to_dict(self, action: Optional[SpecificAction]) -> Optional[Dict]:
        if not action:
            return None
        
        return {
            'type': action.action_type,
            'verb': action.action_verb,
            'target': action.target,
            'location': action.location,
            'timeframe': action.timeframe,
            'magnitude': action.magnitude,
            'reason': action.reason,
            'confidence': action.confidence
        }

class ActionComparisonEngine:
    def __init__(self, config: AnalyzerConfig):
        self.config = config
    
    def compare_company_actions(self, all_chunks: Dict[str, List[Dict]]) -> Dict:
        print(f"\n{Fore.YELLOW}[COMPARISON] Analyzing actions across companies...{Style.RESET_ALL}")
        
        similar_actions = []
        opposite_actions = []
        unique_actions = defaultdict(list)
        action_timeline = []
        
        companies = list(all_chunks.keys())
        
        for company1 in companies:
            for chunk1 in all_chunks[company1]:
                if not chunk1.get('best_action'):
                    continue
                
                action1 = chunk1['best_action']
                action_timeline.append({
                    'company': company1,
                    'date': chunk1.get('filing_date', '2024-01-01'),
                    'action': action1
                })
                
                is_unique = True
                
                for company2 in companies:
                    if company1 == company2:
                        continue
                    
                    for chunk2 in all_chunks[company2]:
                        if not chunk2.get('best_action'):
                            continue
                        
                        action2 = chunk2['best_action']
                        
                        if self._are_similar_actions(action1, action2):
                            similar_actions.append({
                                'company1': company1,
                                'company2': company2,
                                'action1': action1,
                                'action2': action2
                            })
                            is_unique = False
                            
                            if action1.get('location') == action2.get('location'):
                                print(f"  {Fore.GREEN}★ SIMILAR: {company1} and {company2} both {action1['verb']} "
                                      f"in {action1.get('location', 'unspecified')}{Style.RESET_ALL}")
                        
                        elif self._are_opposite_actions(action1, action2):
                            opposite_actions.append({
                                'company1': company1,
                                'company2': company2,
                                'action1': action1,
                                'action2': action2
                            })
                            print(f"  {Fore.RED}◆ OPPOSITE: {company1} {action1['verb']} vs "
                                  f"{company2} {action2['verb']}{Style.RESET_ALL}")
                
                if is_unique and action1['confidence'] > 0.6:
                    unique_actions[company1].append(action1)
        
        return {
            'similar_actions': similar_actions[:20],
            'opposite_actions': opposite_actions[:20],
            'unique_actions': dict(unique_actions),
            'action_timeline': sorted(action_timeline, key=lambda x: x.get('date', ''))[:50],
            'summary': {
                'total_actions': len(action_timeline),
                'similar_pairs': len(similar_actions),
                'opposite_pairs': len(opposite_actions),
                'companies_with_unique': len(unique_actions)
            }
        }
    
    def _are_similar_actions(self, action1: Dict, action2: Dict) -> bool:
        if action1['type'] != action2['type']:
            return False
        
        verb_match = action1.get('verb') == action2.get('verb')
        location_match = action1.get('location') == action2.get('location')
        target_match = action1.get('target') == action2.get('target')
        
        matches = sum([verb_match, location_match, target_match])
        return matches >= 2
    
    def _are_opposite_actions(self, action1: Dict, action2: Dict) -> bool:
        opposite_pairs = [
            ('expand', 'exit'), ('enter', 'exit'), ('launch', 'discontinue'),
            ('acquire', 'divest'), ('hire', 'layoff'), ('open', 'close'),
            ('increase', 'decrease'), ('grow', 'reduce')
        ]
        
        verb1 = action1.get('verb', '').lower()
        verb2 = action2.get('verb', '').lower()
        
        for pair in opposite_pairs:
            if (pair[0] in verb1 and pair[1] in verb2) or (pair[1] in verb1 and pair[0] in verb2):
                return True
        
        return False

def visualize_results(comparison_results: Dict, companies: List[str]):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    summary = comparison_results['summary']
    categories = ['Total Actions', 'Similar Pairs', 'Opposite Pairs']
    values = [summary['total_actions'], summary['similar_pairs'], summary['opposite_pairs']]
    
    ax1.bar(categories, values, color=['blue', 'green', 'red'], alpha=0.7)
    ax1.set_title('Action Analysis Summary', fontweight='bold')
    ax1.set_ylabel('Count')
    
    n = len(companies)
    matrix = np.zeros((n, n))
    
    for sim in comparison_results['similar_actions']:
        i = companies.index(sim['company1'])
        j = companies.index(sim['company2'])
        matrix[i, j] += 1
        matrix[j, i] += 1
    
    im = ax2.imshow(matrix, cmap='Greens', aspect='auto')
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(companies, rotation=45)
    ax2.set_yticklabels(companies)
    ax2.set_title('Similar Actions Heatmap', fontweight='bold')
    
    for i in range(n):
        for j in range(n):
            if matrix[i, j] > 0:
                ax2.text(j, i, int(matrix[i, j]), ha="center", va="center")
    
    plt.colorbar(im, ax=ax2)
    plt.tight_layout()
    plt.savefig('action_analysis.png', dpi=100, bbox_inches='tight')
    plt.show()
    print(f"\n{Fore.GREEN}Visualization saved as 'action_analysis.png'{Style.RESET_ALL}")

def main():
    print(f"\n{Fore.MAGENTA}{'='*100}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}STARTING SPECIFIC ACTION ANALYSIS{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{'='*100}{Style.RESET_ALL}")
    
    config = AnalyzerConfig()
    fetcher = SECDataFetcher(config)
    chunk_processor = ChunkProcessor(config)
    analyzer = EnhancedSimilarityAnalyzer(config)
    comparator = ActionComparisonEngine(config)
    
    all_chunks = {}
    
    for ticker, company_info in config.companies.items():
        print(f"\n{Fore.CYAN}Analyzing {ticker} - {company_info['name']}{Style.RESET_ALL}")
        
        filings = fetcher.fetch_filings(ticker, company_info)
        
        if filings:
            company_chunks = []
            
            for filing in filings:
                chunks = chunk_processor.create_chunks(filing['content'])
                
                for chunk in chunks[:50]:
                    chunk_analysis = analyzer.analyze_chunk(chunk, ticker)
                    chunk_analysis['filing_date'] = filing['filing_date']
                    company_chunks.append(chunk_analysis)
                    
                    if chunk_analysis.get('best_action'):
                        action = chunk_analysis['best_action']
                        if action['confidence'] > 0.6:
                            print(f"  → Found: {action['verb']} {action.get('target', 'N/A')} "
                                  f"in {action.get('location', 'unspecified')} "
                                  f"({action.get('timeframe', 'no timeframe')})")
            
            all_chunks[ticker] = company_chunks
    
    comparison_results = comparator.compare_company_actions(all_chunks)
    
    print(f"\n{Fore.CYAN}{'='*100}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}ANALYSIS SUMMARY{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*100}{Style.RESET_ALL}")
    
    summary = comparison_results['summary']
    print(f"\nTotal specific actions found: {summary['total_actions']}")
    print(f"Similar action pairs: {summary['similar_pairs']}")
    print(f"Opposite action pairs: {summary['opposite_pairs']}")
    print(f"Companies with unique actions: {summary['companies_with_unique']}")
    
    if comparison_results['similar_actions']:
        print(f"\n{Fore.GREEN}TOP SIMILAR ACTIONS:{Style.RESET_ALL}")
        for sim in comparison_results['similar_actions'][:5]:
            print(f"  • {sim['company1']} and {sim['company2']}: "
                  f"{sim['action1']['verb']} in {sim['action1'].get('location', 'unspecified')}")
    
    if comparison_results['opposite_actions']:
        print(f"\n{Fore.RED}TOP OPPOSITE ACTIONS:{Style.RESET_ALL}")
        for opp in comparison_results['opposite_actions'][:5]:
            print(f"  • {opp['company1']} ({opp['action1']['verb']}) vs "
                  f"{opp['company2']} ({opp['action2']['verb']})")
    
    visualize_results(comparison_results, list(all_chunks.keys()))
    
    output_file = f"specific_actions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    
    print(f"\n{Fore.GREEN}Results saved to: {output_file}{Style.RESET_ALL}")
    
    return comparison_results

if __name__ == "__main__":
    results = main()
    print(f"\n{Fore.CYAN}[END] Analysis Complete{Style.RESET_ALL}")

