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
SEC FILING ACTION ANALYZER - FULLY WORKING VERSION
Extracts and compares specific business actions from SEC filings
"""

# Install required packages
import subprocess
import sys
import os

print("Installing required packages...")
packages = [
    'transformers',
    'torch',
    'sentence-transformers',
    'beautifulsoup4',
    'requests',
    'pandas',
    'numpy',
    'scikit-learn',
    'matplotlib',
    'seaborn',
    'colorama',
    'faiss-cpu'
]

for package in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

print("Package installation complete.\n")

import warnings
warnings.filterwarnings('ignore')

import re
import json
import numpy as np
import pandas as pd
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from colorama import init, Fore, Style
import faiss
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

init(autoreset=True)
sns.set_style("whitegrid")

print("="*100)
print("SEC FILING ACTION ANALYZER")
print("="*100)

@dataclass
class ExtractedAction:
    company: str
    action_type: str
    description: str
    location: Optional[str] = None
    timeframe: Optional[str] = None
    magnitude: Optional[str] = None
    confidence: float = 0.0
    source_text: str = ""
    embedding: Optional[np.ndarray] = None

@dataclass
class Config:
    # Use a zero-shot classifier instead of generative model for reliability
    classifier_model: str = "facebook/bart-large-mnli"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Processing settings
    max_filings_per_company: int = 1
    chunk_size: int = 500
    
    # Analysis settings
    similarity_threshold: float = 0.7
    min_confidence: float = 0.3
    
    # Companies to analyze
    companies: Dict[str, Dict] = field(default_factory=lambda: {
        'AAPL': {'cik': '0000320193', 'name': 'Apple Inc'},
        'MSFT': {'cik': '0000789019', 'name': 'Microsoft Corporation'},
        'GOOGL': {'cik': '0001652044', 'name': 'Alphabet Inc'},
        'NVDA': {'cik': '0001045810', 'name': 'NVIDIA Corporation'}
    })
    
    # Action types to detect
    action_labels: List[str] = field(default_factory=lambda: [
        'market expansion',
        'product launch',
        'acquisition or merger',
        'workforce changes',
        'facility changes',
        'strategic partnership',
        'financial investment',
        'technology development',
        'cost reduction',
        'revenue growth'
    ])

class ActionExtractor:
    def __init__(self, config: Config):
        self.config = config
        self.device = 0 if torch.cuda.is_available() else -1
        
        print(f"\n{Fore.YELLOW}Initializing models...{Style.RESET_ALL}")
        
        # Use zero-shot classification for reliability
        self.classifier = pipeline(
            "zero-shot-classification",
            model=self.config.classifier_model,
            device=self.device
        )
        
        # Load embedding model
        self.embedder = SentenceTransformer(self.config.embedding_model)
        
        print(f"{Fore.GREEN}✓ Models loaded successfully{Style.RESET_ALL}")
    
    def extract_actions_from_text(self, text: str, company: str) -> List[ExtractedAction]:
        """Extract business actions using zero-shot classification"""
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 50]
        
        actions = []
        
        for sentence in sentences[:20]:  # Process first 20 sentences
            try:
                # Classify the sentence
                result = self.classifier(
                    sentence,
                    candidate_labels=self.config.action_labels,
                    multi_label=False
                )
                
                # Get the top classification
                if result['scores'][0] > self.config.min_confidence:
                    action = ExtractedAction(
                        company=company,
                        action_type=result['labels'][0],
                        description=sentence[:200],
                        confidence=result['scores'][0],
                        source_text=sentence
                    )
                    
                    # Extract additional details
                    self._extract_details(action, sentence)
                    
                    # Generate embedding
                    action.embedding = self.embedder.encode(sentence)
                    
                    actions.append(action)
                    
            except Exception as e:
                continue
        
        return actions
    
    def _extract_details(self, action: ExtractedAction, text: str):
        """Extract location, timeframe, and magnitude from text"""
        
        # Extract location (simple pattern matching)
        locations = re.findall(r'\b(?:United States|China|Europe|Asia|India|Japan|UK|California|New York)\b', text, re.I)
        if locations:
            action.location = locations[0]
        
        # Extract timeframe
        timeframes = re.findall(r'\b(?:Q[1-4] \d{4}|\d{4}|next quarter|this year|next year)\b', text, re.I)
        if timeframes:
            action.timeframe = timeframes[0]
        
        # Extract magnitude
        magnitudes = re.findall(r'\$[\d.]+\s*(?:million|billion|M|B)|\d+%|\d+,?\d* employees', text, re.I)
        if magnitudes:
            action.magnitude = magnitudes[0]

class ActionComparator:
    def __init__(self, config: Config):
        self.config = config
        self.embedder = SentenceTransformer(config.embedding_model)
    
    def compare_actions(self, actions_dict: Dict[str, List[ExtractedAction]]) -> Dict:
        """Compare actions across companies"""
        
        all_actions = []
        for company, actions in actions_dict.items():
            all_actions.extend(actions)
        
        if not all_actions:
            return self._empty_results()
        
        # Get embeddings
        embeddings = np.array([a.embedding for a in all_actions if a.embedding is not None])
        
        if len(embeddings) == 0:
            return self._empty_results()
        
        # Find similar actions
        similar_pairs = self._find_similar_actions(all_actions, embeddings)
        
        # Cluster actions
        clusters = self._cluster_actions(embeddings, all_actions)
        
        return {
            'similar': similar_pairs,
            'clusters': clusters,
            'summary': {
                'total_actions': len(all_actions),
                'similar_pairs': len(similar_pairs),
                'num_clusters': len(clusters)
            }
        }
    
    def _empty_results(self):
        return {
            'similar': [],
            'clusters': [],
            'summary': {
                'total_actions': 0,
                'similar_pairs': 0,
                'num_clusters': 0
            }
        }
    
    def _find_similar_actions(self, actions: List[ExtractedAction], embeddings: np.ndarray) -> List[Dict]:
        """Find similar actions across companies"""
        
        similar_pairs = []
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Find high similarity pairs
        for i in range(len(actions)):
            for j in range(i+1, len(actions)):
                if actions[i].company != actions[j].company:
                    similarity = similarity_matrix[i, j]
                    if similarity > self.config.similarity_threshold:
                        similar_pairs.append({
                            'company1': actions[i].company,
                            'company2': actions[j].company,
                            'action1': actions[i].description[:100],
                            'action2': actions[j].description[:100],
                            'similarity': float(similarity),
                            'type': actions[i].action_type
                        })
        
        return similar_pairs
    
    def _cluster_actions(self, embeddings: np.ndarray, actions: List[ExtractedAction]) -> List[Dict]:
        """Cluster similar actions"""
        
        if len(embeddings) < 3:
            return []
        
        # Use KMeans clustering
        n_clusters = min(5, len(embeddings))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(embeddings)
        
        # Organize clusters
        cluster_dict = {}
        for idx, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_dict:
                cluster_dict[cluster_id] = []
            cluster_dict[cluster_id].append(actions[idx])
        
        # Create summaries
        cluster_summaries = []
        for cluster_id, cluster_actions in cluster_dict.items():
            cluster_summaries.append({
                'id': int(cluster_id),
                'theme': cluster_actions[0].action_type,
                'companies': list(set([a.company for a in cluster_actions])),
                'num_actions': len(cluster_actions),
                'sample_action': cluster_actions[0].description[:150]
            })
        
        return cluster_summaries

class SECFilingFetcher:
    def __init__(self, config: Config):
        self.config = config
        self.headers = {'User-Agent': 'Research Bot 1.0'}
    
    def fetch_filings(self, ticker: str, company_info: Dict) -> List[Dict]:
        """Fetch SEC filings"""
        
        print(f"\n{Fore.CYAN}Fetching filings for {ticker}...{Style.RESET_ALL}")
        
        cik = company_info['cik'].lstrip('0')
        filings = []
        
        try:
            url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                recent = data.get('filings', {}).get('recent', {})
                
                forms = recent.get('form', [])
                dates = recent.get('filingDate', [])
                accession_numbers = recent.get('accessionNumber', [])
                
                count = 0
                for i in range(min(len(forms), 10)):
                    if forms[i] in ['10-K', '10-Q', '8-K'] and count < self.config.max_filings_per_company:
                        content = self._fetch_content(cik, accession_numbers[i])
                        if content:
                            filings.append({
                                'ticker': ticker,
                                'form': forms[i],
                                'date': dates[i],
                                'content': content[:50000]
                            })
                            count += 1
                            print(f"  ✓ Retrieved {forms[i]} from {dates[i]}")
                            break  # Just get one filing for speed
                
        except Exception as e:
            print(f"  {Fore.RED}Error: {e}{Style.RESET_ALL}")
        
        return filings
    
    def _fetch_content(self, cik: str, accession: str) -> str:
        """Fetch filing content"""
        try:
            acc_no_dash = accession.replace('-', '')
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_dash}/{accession}.txt"
            response = requests.get(url, headers=self.headers, timeout=15)
            
            if response.status_code == 200:
                text = response.text
                if '<html' in text.lower():
                    soup = BeautifulSoup(text, 'html.parser')
                    text = soup.get_text()
                
                # Clean text
                text = re.sub(r'\s+', ' ', text)
                text = re.sub(r'[^\x00-\x7F]+', '', text)
                return text.strip()
        except:
            pass
        return ""

def visualize_results(comparison_results: Dict, actions_dict: Dict):
    """Create visualizations"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Summary statistics
    summary = comparison_results['summary']
    ax1 = axes[0]
    categories = ['Total Actions', 'Similar Pairs', 'Clusters']
    values = [summary['total_actions'], summary['similar_pairs'], summary['num_clusters']]
    ax1.bar(categories, values, color=['blue', 'green', 'purple'])
    ax1.set_title('Analysis Summary', fontweight='bold')
    ax1.set_ylabel('Count')
    
    # Actions by company
    ax2 = axes[1]
    companies = list(actions_dict.keys())
    action_counts = [len(actions) for actions in actions_dict.values()]
    if companies:
        ax2.bar(companies, action_counts, color='skyblue')
        ax2.set_title('Actions by Company', fontweight='bold')
        ax2.set_ylabel('Number of Actions')
        ax2.tick_params(axis='x', rotation=45)
    
    # Action types
    ax3 = axes[2]
    all_types = []
    for actions in actions_dict.values():
        all_types.extend([a.action_type for a in actions])
    
    if all_types:
        type_counts = pd.Series(all_types).value_counts()
        ax3.barh(type_counts.index[:5], type_counts.values[:5], color='coral')
        ax3.set_title('Top Action Types', fontweight='bold')
        ax3.set_xlabel('Count')
    
    plt.tight_layout()
    plt.savefig('sec_analysis.png', dpi=100)
    plt.show()
    print(f"\n{Fore.GREEN}Visualization saved as 'sec_analysis.png'{Style.RESET_ALL}")

def main():
    print(f"\n{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}STARTING SEC FILING ANALYSIS{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}")
    
    # Initialize
    config = Config()
    fetcher = SECFilingFetcher(config)
    extractor = ActionExtractor(config)
    comparator = ActionComparator(config)
    
    all_actions = {}
    
    # Process each company
    for ticker, company_info in config.companies.items():
        print(f"\n{Fore.CYAN}Analyzing {ticker} - {company_info['name']}{Style.RESET_ALL}")
        
        # Fetch filings
        filings = fetcher.fetch_filings(ticker, company_info)
        
        if not filings:
            all_actions[ticker] = []
            continue
        
        company_actions = []
        
        # Process filing
        for filing in filings:
            print(f"  Processing {filing['form']} from {filing['date']}...")
            
            # Extract actions from first part of filing
            actions = extractor.extract_actions_from_text(filing['content'][:10000], ticker)
            
            for action in actions:
                if action.confidence >= config.min_confidence:
                    company_actions.append(action)
                    print(f"    → Found: {action.action_type} (confidence: {action.confidence:.2f})")
                    print(f"      {action.description[:80]}...")
        
        all_actions[ticker] = company_actions
        print(f"  Total actions: {len(company_actions)}")
    
    # Compare actions
    print(f"\n{Fore.YELLOW}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}COMPARING ACTIONS{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'='*80}{Style.RESET_ALL}")
    
    comparison_results = comparator.compare_actions(all_actions)
    
    # Print summary
    summary = comparison_results['summary']
    print(f"\n{Fore.CYAN}SUMMARY:{Style.RESET_ALL}")
    print(f"  Total actions: {summary['total_actions']}")
    print(f"  Similar pairs: {summary['similar_pairs']}")
    print(f"  Clusters: {summary['num_clusters']}")
    
    # Print similar actions
    if comparison_results['similar']:
        print(f"\n{Fore.GREEN}SIMILAR ACTIONS:{Style.RESET_ALL}")
        for sim in comparison_results['similar'][:5]:
            print(f"  • {sim['company1']} & {sim['company2']} ({sim['similarity']:.2f})")
            print(f"    Type: {sim['type']}")
            print(f"    {sim['company1']}: {sim['action1']}")
            print(f"    {sim['company2']}: {sim['action2']}")
    
    # Print clusters
    if comparison_results['clusters']:
        print(f"\n{Fore.BLUE}ACTION CLUSTERS:{Style.RESET_ALL}")
        for cluster in comparison_results['clusters']:
            print(f"  • Cluster {cluster['id']}: {cluster['theme']}")
            print(f"    Companies: {', '.join(cluster['companies'])}")
            print(f"    Actions: {cluster['num_actions']}")
    
    # Visualize
    visualize_results(comparison_results, all_actions)
    
    # Save results
    output_data = {
        'summary': summary,
        'actions_by_company': {
            company: [
                {
                    'type': a.action_type,
                    'description': a.description,
                    'confidence': a.confidence,
                    'location': a.location,
                    'timeframe': a.timeframe,
                    'magnitude': a.magnitude
                }
                for a in actions
            ]
            for company, actions in all_actions.items()
        }
    }
    
    output_file = f"sec_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{Fore.GREEN}Results saved to: {output_file}{Style.RESET_ALL}")
    
    return comparison_results

if __name__ == "__main__":
    try:
        results = main()
        print(f"\n{Fore.GREEN}[SUCCESS] Analysis Complete{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}[ERROR] {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()

