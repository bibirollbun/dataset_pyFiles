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
ADVANCED MULTI-MODEL SEC FILING ACTION ANALYZER
Combines embeddings, zero-shot classification, and LLMs for comprehensive analysis
"""

import subprocess
import sys
import os

print("Installing required packages...")
packages = [
    'transformers>=4.35.0',
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
    'faiss-cpu',
    'accelerate',
    'bitsandbytes'
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
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import torch
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig
)
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from colorama import init, Fore, Style
import faiss
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

init(autoreset=True)
sns.set_style("whitegrid")

print("="*100)
print("ADVANCED MULTI-MODEL SEC FILING ANALYZER")
print("="*100)

@dataclass
class DetailedAction:
    company: str
    action_category: str
    specific_action: str
    description: str
    entities_mentioned: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=list)
    financial_impacts: List[str] = field(default_factory=list)
    strategic_rationale: str = ""
    confidence: float = 0.0
    extraction_method: str = ""
    source_text: str = ""
    embedding: Optional[np.ndarray] = None

@dataclass
class AdvancedConfig:
    # Multiple models for different tasks
    llm_model: str = "microsoft/Phi-2"  # 2.7B params, good for Kaggle
    classifier_model: str = "facebook/bart-large-mnli"
    ner_model: str = "dslim/bert-base-NER"
    finance_bert: str = "ProsusAI/finbert"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    
    # Processing
    max_filings_per_company: int = 2
    chunk_size: int = 800
    
    # Thresholds
    similarity_threshold: float = 0.75
    min_confidence: float = 0.35
    
    # Companies
    companies: Dict[str, Dict] = field(default_factory=lambda: {
        'AAPL': {'cik': '0000320193', 'name': 'Apple Inc'},
        'MSFT': {'cik': '0000789019', 'name': 'Microsoft Corporation'},
        'GOOGL': {'cik': '0001652044', 'name': 'Alphabet Inc'},
        'NVDA': {'cik': '0001045810', 'name': 'NVIDIA Corporation'},
        'META': {'cik': '0001326801', 'name': 'Meta Platforms'},
        'AMZN': {'cik': '0001018724', 'name': 'Amazon'}
    })
    
    # Detailed action categories
    action_categories: Dict[str, List[str]] = field(default_factory=lambda: {
        'strategic_moves': [
            'market entry or exit',
            'geographic expansion',
            'business pivot',
            'vertical integration',
            'diversification'
        ],
        'operational_changes': [
            'facility opening or closure',
            'supply chain modification',
            'manufacturing changes',
            'workforce restructuring',
            'operational efficiency'
        ],
        'financial_actions': [
            'capital raising',
            'debt restructuring',
            'dividend changes',
            'share buyback',
            'investment or divestiture'
        ],
        'innovation_tech': [
            'product launch',
            'R&D investment',
            'technology acquisition',
            'patent filing',
            'platform development'
        ],
        'partnerships': [
            'strategic alliance',
            'joint venture',
            'licensing deal',
            'distribution agreement',
            'collaboration'
        ],
        'regulatory': [
            'compliance initiative',
            'regulatory approval',
            'legal settlement',
            'policy change response',
            'governance change'
        ]
    })

class MultiModelExtractor:
    def __init__(self, config: AdvancedConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {self.device}")
        
        self._initialize_models()
    
    def _initialize_models(self):
        print(f"\n{Fore.YELLOW}Loading multiple AI models...{Style.RESET_ALL}")
        
        # 1. Embeddings for semantic search
        print("Loading embedding model...")
        self.embedder = SentenceTransformer(self.config.embedding_model)
        
        # 2. Zero-shot classifier
        print("Loading zero-shot classifier...")
        self.classifier = pipeline(
            "zero-shot-classification",
            model=self.config.classifier_model,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # 3. NER for entity extraction
        print("Loading NER model...")
        self.ner = pipeline(
            "ner",
            model=self.config.ner_model,
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # 4. Financial sentiment analyzer
        print("Loading FinBERT...")
        self.finbert = pipeline(
            "sentiment-analysis",
            model=self.config.finance_bert,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # 5. Small LLM for detailed extraction
        print("Loading Phi-2 for detailed analysis...")
        try:
            # Use 8-bit quantization for memory efficiency
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model)
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.config.llm_model,
                quantization_config=bnb_config if torch.cuda.is_available() else None,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print(f"{Fore.GREEN}✓ All models loaded successfully{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"Warning: Could not load LLM: {e}")
            self.llm = None
            self.tokenizer = None
    
    def extract_comprehensive_actions(self, text: str, company: str) -> List[DetailedAction]:
        """Extract actions using multiple techniques"""
        
        actions = []
        
        # Split into meaningful chunks
        chunks = self._smart_chunk_text(text)
        
        for chunk in chunks[:10]:  # Process first 10 chunks
            # 1. Zero-shot classification for action categories
            category_actions = self._classify_actions(chunk, company)
            
            # 2. NER for entities and locations
            entities = self._extract_entities(chunk)
            
            # 3. Financial sentiment and implications
            sentiment = self._analyze_financial_sentiment(chunk)
            
            # 4. LLM for detailed extraction (if available)
            if self.llm:
                llm_actions = self._extract_with_llm(chunk, company)
                actions.extend(llm_actions)
            
            # 5. Pattern-based extraction for specific items
            pattern_actions = self._extract_patterns(chunk, company)
            
            # Combine all extractions
            for action in category_actions + pattern_actions:
                # Enrich with entities
                action.entities_mentioned = entities.get('organizations', [])
                action.locations = entities.get('locations', [])
                
                # Add financial context
                if sentiment and sentiment[0]['score'] > 0.7:
                    action.financial_impacts.append(f"{sentiment[0]['label']}: {sentiment[0]['score']:.2f}")
                
                # Generate embedding
                action.embedding = self.embedder.encode(action.description)
                
                actions.append(action)
        
        return self._deduplicate_actions(actions)
    
    def _smart_chunk_text(self, text: str) -> List[str]:
        """Create intelligent chunks based on document structure"""
        
        chunks = []
        
        # Try to split by sections
        section_patterns = [
            r'Item \d+[A-Z]?\.',
            r'Part [IVX]+',
            r'Note \d+',
            r'\n[A-Z][A-Z\s]+\n',  # All caps headers
        ]
        
        sections = [text]
        for pattern in section_patterns:
            new_sections = []
            for section in sections:
                splits = re.split(f'({pattern})', section)
                new_sections.extend(splits)
            sections = new_sections
        
        # Clean and filter sections
        for section in sections:
            section = section.strip()
            if 100 < len(section) < 5000:  # Reasonable size
                chunks.append(section)
        
        # If no good sections, fall back to sentence chunks
        if len(chunks) < 3:
            sentences = re.split(r'[.!?]+', text)
            for i in range(0, len(sentences), 3):
                chunk = ' '.join(sentences[i:i+3])
                if len(chunk) > 100:
                    chunks.append(chunk)
        
        return chunks[:50]  # Limit chunks
    
    def _classify_actions(self, text: str, company: str) -> List[DetailedAction]:
        """Use zero-shot classification for action categories"""
        
        actions = []
        
        # Flatten categories for classification
        all_labels = []
        for category, subcategories in self.config.action_categories.items():
            all_labels.extend(subcategories)
        
        try:
            result = self.classifier(
                text[:500],
                candidate_labels=all_labels,
                multi_label=True
            )
            
            # Take top classifications
            for label, score in zip(result['labels'][:3], result['scores'][:3]):
                if score > self.config.min_confidence:
                    # Find parent category
                    category = "general"
                    for cat, subs in self.config.action_categories.items():
                        if label in subs:
                            category = cat
                            break
                    
                    action = DetailedAction(
                        company=company,
                        action_category=category,
                        specific_action=label,
                        description=text[:300],
                        confidence=score,
                        extraction_method="zero-shot",
                        source_text=text[:500]
                    )
                    actions.append(action)
                    
        except Exception as e:
            pass
        
        return actions
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities"""
        
        entities = {
            'organizations': [],
            'locations': [],
            'persons': [],
            'money': []
        }
        
        try:
            ner_results = self.ner(text[:500])
            
            for entity in ner_results:
                if entity['entity_group'] == 'ORG':
                    entities['organizations'].append(entity['word'])
                elif entity['entity_group'] == 'LOC':
                    entities['locations'].append(entity['word'])
                elif entity['entity_group'] == 'PER':
                    entities['persons'].append(entity['word'])
        except:
            pass
        
        # Also extract money amounts
        money_patterns = re.findall(r'\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|M|B))?', text)
        entities['money'] = money_patterns
        
        return entities
    
    def _analyze_financial_sentiment(self, text: str) -> List[Dict]:
        """Analyze financial sentiment"""
        
        try:
            return self.finbert(text[:500])
        except:
            return []
    
    def _extract_with_llm(self, text: str, company: str) -> List[DetailedAction]:
        """Use LLM for detailed extraction"""
        
        if not self.llm:
            return []
        
        prompt = f"""Analyze this SEC filing excerpt from {company} and extract specific business actions.

Text: {text[:500]}

Extract specific actions with details. For each action provide:
- Type of action (e.g., acquisition, product launch, facility closure)
- Specific details (names, locations, amounts)
- Strategic reason

Be very specific. Format: ACTION: [type] | DETAILS: [specifics] | REASON: [why]"""

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=700)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Parse response
            actions = []
            lines = response.split('\n')
            for line in lines:
                if 'ACTION:' in line:
                    parts = line.split('|')
                    if len(parts) >= 2:
                        action = DetailedAction(
                            company=company,
                            action_category="llm_extracted",
                            specific_action=parts[0].replace('ACTION:', '').strip(),
                            description=parts[1].replace('DETAILS:', '').strip() if len(parts) > 1 else "",
                            strategic_rationale=parts[2].replace('REASON:', '').strip() if len(parts) > 2 else "",
                            confidence=0.7,
                            extraction_method="llm",
                            source_text=text[:300]
                        )
                        actions.append(action)
            
            return actions
            
        except Exception as e:
            return []
    
    def _extract_patterns(self, text: str, company: str) -> List[DetailedAction]:
        """Extract specific patterns"""
        
        actions = []
        
        # Acquisition patterns
        acq_pattern = r'(?:acquired|purchased|bought|merger with|acquisition of)\s+([A-Z][A-Za-z\s&]+?)(?:\s+for\s+\$?([\d,]+(?:\.\d+)?(?:\s*(?:million|billion|M|B))?))?'
        acq_matches = re.findall(acq_pattern, text, re.I)
        for match in acq_matches:
            action = DetailedAction(
                company=company,
                action_category="partnerships",
                specific_action="acquisition",
                description=f"Acquisition of {match[0]}",
                financial_impacts=[match[1]] if len(match) > 1 and match[1] else [],
                confidence=0.8,
                extraction_method="pattern",
                source_text=text[:300]
            )
            actions.append(action)
        
        # Product launch patterns
        product_pattern = r'(?:launched|introduced|unveiled|announced)\s+(?:a\s+)?(?:new\s+)?([A-Za-z0-9\s]+?)(?:\s+(?:product|service|platform|feature))'
        product_matches = re.findall(product_pattern, text, re.I)
        for match in product_matches:
            action = DetailedAction(
                company=company,
                action_category="innovation_tech",
                specific_action="product launch",
                description=f"Launch of {match}",
                confidence=0.75,
                extraction_method="pattern",
                source_text=text[:300]
            )
            actions.append(action)
        
        # Geographic expansion patterns
        geo_pattern = r'(?:expand|enter|launch)\s+(?:into|in|operations\s+in)\s+([A-Z][A-Za-z\s]+?)(?:\s+market)?'
        geo_matches = re.findall(geo_pattern, text, re.I)
        for match in geo_matches:
            action = DetailedAction(
                company=company,
                action_category="strategic_moves",
                specific_action="geographic expansion",
                description=f"Expansion into {match}",
                locations=[match],
                confidence=0.7,
                extraction_method="pattern",
                source_text=text[:300]
            )
            actions.append(action)
        
        return actions
    
    def _deduplicate_actions(self, actions: List[DetailedAction]) -> List[DetailedAction]:
        """Remove duplicate actions using embeddings"""
        
        if len(actions) <= 1:
            return actions
        
        # Create embedding matrix
        embeddings = np.array([a.embedding for a in actions if a.embedding is not None])
        
        if len(embeddings) < 2:
            return actions
        
        # Calculate similarity matrix
        similarity = cosine_similarity(embeddings)
        
        # Keep track of duplicates
        keep_indices = []
        for i in range(len(actions)):
            is_duplicate = False
            for j in keep_indices:
                if similarity[i, j] > 0.9:  # Very similar
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep_indices.append(i)
        
        return [actions[i] for i in keep_indices]

class AdvancedComparator:
    def __init__(self, config: AdvancedConfig):
        self.config = config
        self.embedder = SentenceTransformer(config.embedding_model)
    
    def analyze_cross_company(self, actions_dict: Dict[str, List[DetailedAction]]) -> Dict:
        """Advanced cross-company analysis"""
        
        all_actions = []
        for company, actions in actions_dict.items():
            all_actions.extend(actions)
        
        if not all_actions:
            return self._empty_results()
        
        # Create embeddings
        embeddings = np.array([a.embedding for a in all_actions if a.embedding is not None])
        
        if len(embeddings) == 0:
            return self._empty_results()
        
        # 1. Find similar strategies
        similar_strategies = self._find_similar_strategies(all_actions, embeddings)
        
        # 2. Find opposite strategies
        opposite_strategies = self._find_opposite_strategies(all_actions)
        
        # 3. Hierarchical clustering for themes
        themes = self._hierarchical_clustering(all_actions, embeddings)
        
        # 4. Trend analysis
        trends = self._analyze_trends(actions_dict)
        
        # 5. Company similarity matrix
        company_similarity = self._compute_company_similarity(actions_dict)
        
        return {
            'similar_strategies': similar_strategies,
            'opposite_strategies': opposite_strategies,
            'themes': themes,
            'trends': trends,
            'company_similarity': company_similarity,
            'summary': {
                'total_actions': len(all_actions),
                'companies_analyzed': len(actions_dict),
                'similar_pairs': len(similar_strategies),
                'themes_identified': len(themes)
            }
        }
    
    def _empty_results(self):
        return {
            'similar_strategies': [],
            'opposite_strategies': [],
            'themes': [],
            'trends': {},
            'company_similarity': {},
            'summary': {
                'total_actions': 0,
                'companies_analyzed': 0,
                'similar_pairs': 0,
                'themes_identified': 0
            }
        }
    
    def _find_similar_strategies(self, actions: List[DetailedAction], embeddings: np.ndarray) -> List[Dict]:
        """Find similar strategies across companies"""
        
        similar = []
        
        # Build FAISS index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings.astype('float32'))
        
        # Search for similar actions
        k = min(5, len(embeddings))
        distances, indices = index.search(embeddings.astype('float32'), k)
        
        processed = set()
        
        for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
            for j, (dist, idx) in enumerate(zip(dist_row, idx_row)):
                if i != idx:
                    pair = tuple(sorted([i, idx]))
                    if pair not in processed:
                        processed.add(pair)
                        
                        if actions[i].company != actions[idx].company:
                            similarity = 1 / (1 + dist)  # Convert distance to similarity
                            if similarity > self.config.similarity_threshold:
                                similar.append({
                                    'company1': actions[i].company,
                                    'company2': actions[idx].company,
                                    'action1': {
                                        'category': actions[i].action_category,
                                        'specific': actions[i].specific_action,
                                        'description': actions[i].description[:150]
                                    },
                                    'action2': {
                                        'category': actions[idx].action_category,
                                        'specific': actions[idx].specific_action,
                                        'description': actions[idx].description[:150]
                                    },
                                    'similarity': float(similarity)
                                })
        
        return sorted(similar, key=lambda x: x['similarity'], reverse=True)[:20]
    
    def _find_opposite_strategies(self, actions: List[DetailedAction]) -> List[Dict]:
        """Find companies taking opposite approaches"""
        
        opposite = []
        
        # Define opposite action pairs
        opposite_pairs = {
            'market entry or exit': 'market entry or exit',
            'facility opening or closure': 'facility opening or closure',
            'workforce restructuring': 'workforce restructuring',
            'investment or divestiture': 'investment or divestiture'
        }
        
        for i, action1 in enumerate(actions):
            for j, action2 in enumerate(actions[i+1:], i+1):
                if action1.company != action2.company:
                    # Check if same category but opposite sentiment
                    if action1.specific_action in opposite_pairs:
                        if action1.specific_action == action2.specific_action:
                            # Use financial sentiment to determine if opposite
                            if action1.financial_impacts and action2.financial_impacts:
                                if ('positive' in str(action1.financial_impacts).lower() and 
                                    'negative' in str(action2.financial_impacts).lower()) or \
                                   ('negative' in str(action1.financial_impacts).lower() and 
                                    'positive' in str(action2.financial_impacts).lower()):
                                    
                                    opposite.append({
                                        'company1': action1.company,
                                        'company2': action2.company,
                                        'action_type': action1.specific_action,
                                        'company1_approach': action1.description[:150],
                                        'company2_approach': action2.description[:150]
                                    })
        
        return opposite[:10]
    
    def _hierarchical_clustering(self, actions: List[DetailedAction], embeddings: np.ndarray) -> List[Dict]:
        """Create hierarchical clusters of actions"""
        
        if len(embeddings) < 3:
            return []
        
        # Use hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=min(8, len(embeddings)//2),
            linkage='ward'
        )
        
        # Reduce dimensions for clustering
        pca = PCA(n_components=min(10, len(embeddings)-1))
        embeddings_reduced = pca.fit_transform(embeddings)
        
        clusters = clustering.fit_predict(embeddings_reduced)
        
        # Organize clusters
        cluster_dict = {}
        for idx, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_dict:
                cluster_dict[cluster_id] = []
            cluster_dict[cluster_id].append(actions[idx])
        
        # Create cluster summaries
        themes = []
        for cluster_id, cluster_actions in cluster_dict.items():
            # Find most common category
            categories = [a.action_category for a in cluster_actions]
            most_common = max(set(categories), key=categories.count)
            
            # Get unique companies
            companies = list(set([a.company for a in cluster_actions]))
            
            # Find representative action
            rep_action = cluster_actions[0]
            
            themes.append({
                'theme_id': int(cluster_id),
                'primary_category': most_common,
                'companies_involved': companies,
                'num_actions': len(cluster_actions),
                'representative_action': rep_action.description[:200],
                'common_entities': list(set([e for a in cluster_actions for e in a.entities_mentioned]))[:5],
                'common_locations': list(set([l for a in cluster_actions for l in a.locations]))[:5]
            })
        
        return themes
    
    def _analyze_trends(self, actions_dict: Dict[str, List[DetailedAction]]) -> Dict:
        """Analyze trends across companies"""
        
        trends = {
            'most_common_actions': {},
            'company_focus': {},
            'emerging_themes': []
        }
        
        # Count action types
        action_counts = {}
        for company, actions in actions_dict.items():
            for action in actions:
                key = action.specific_action
                if key not in action_counts:
                    action_counts[key] = 0
                action_counts[key] += 1
        
        # Most common actions
        trends['most_common_actions'] = dict(sorted(
            action_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5])
        
        # Company focus areas
        for company, actions in actions_dict.items():
            if actions:
                categories = [a.action_category for a in actions]
                if categories:
                    trends['company_focus'][company] = max(set(categories), key=categories.count)
        
        # Emerging themes (actions mentioned by multiple companies)
        action_companies = {}
        for company, actions in actions_dict.items():
            for action in actions:
                key = action.specific_action
                if key not in action_companies:
                    action_companies[key] = set()
                action_companies[key].add(company)
        
        for action, companies in action_companies.items():
            if len(companies) >= 2:
                trends['emerging_themes'].append({
                    'theme': action,
                    'companies': list(companies)
                })
        
        return trends
    
    def _compute_company_similarity(self, actions_dict: Dict[str, List[DetailedAction]]) -> Dict:
        """Compute similarity between companies based on actions"""
        
        companies = list(actions_dict.keys())
        similarity_matrix = {}
        
        for comp1 in companies:
            similarity_matrix[comp1] = {}
            for comp2 in companies:
                if comp1 == comp2:
                    similarity_matrix[comp1][comp2] = 1.0
                else:
                    # Compare action portfolios
                    actions1 = set([a.specific_action for a in actions_dict.get(comp1, [])])
                    actions2 = set([a.specific_action for a in actions_dict.get(comp2, [])])
                    
                    if actions1 and actions2:
                        intersection = len(actions1.intersection(actions2))
                        union = len(actions1.union(actions2))
                        similarity = intersection / union if union > 0 else 0
                        similarity_matrix[comp1][comp2] = similarity
                    else:
                        similarity_matrix[comp1][comp2] = 0.0
        
        return similarity_matrix

class SECFetcher:
    def __init__(self, config: AdvancedConfig):
        self.config = config
        self.headers = {'User-Agent': 'Research Bot 1.0'}
    
    def fetch_filings(self, ticker: str, company_info: Dict) -> List[Dict]:
        """Fetch SEC filings"""
        
        print(f"\n{Fore.CYAN}Fetching {ticker}...{Style.RESET_ALL}")
        
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
                
                for i in range(min(len(forms), 20)):
                    if forms[i] in ['10-K', '10-Q', '8-K']:
                        if len(filings) >= self.config.max_filings_per_company:
                            break
                        
                        content = self._fetch_content(cik, accession_numbers[i])
                        if content:
                            filings.append({
                                'ticker': ticker,
                                'form': forms[i],
                                'date': dates[i],
                                'content': content[:100000]
                            })
                            print(f"  ✓ {forms[i]} - {dates[i]}")
                
        except Exception as e:
            print(f"  Error: {e}")
        
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
                
                text = re.sub(r'\s+', ' ', text)
                text = re.sub(r'[^\x00-\x7F]+', '', text)
                return text.strip()
        except:
            pass
        return ""

def create_advanced_visualizations(results: Dict, actions_dict: Dict):
    """Create comprehensive visualizations"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Summary metrics
    ax1 = axes[0, 0]
    summary = results['summary']
    metrics = list(summary.keys())
    values = list(summary.values())
    ax1.barh(metrics, values, color='steelblue')
    ax1.set_title('Analysis Metrics', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Count')
    
    # 2. Actions by company
    ax2 = axes[0, 1]
    companies = list(actions_dict.keys())
    action_counts = [len(actions) for actions in actions_dict.values()]
    if companies:
        ax2.bar(companies, action_counts, color='coral')
        ax2.set_title('Actions per Company', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Number of Actions')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Action categories distribution
    ax3 = axes[0, 2]
    all_categories = []
    for actions in actions_dict.values():
        all_categories.extend([a.action_category for a in actions])
    
    if all_categories:
        cat_counts = pd.Series(all_categories).value_counts()
        ax3.pie(cat_counts.values[:6], labels=cat_counts.index[:6], autopct='%1.1f%%')
        ax3.set_title('Action Categories', fontweight='bold', fontsize=12)
    
    # 4. Company similarity heatmap
    ax4 = axes[1, 0]
    if 'company_similarity' in results and results['company_similarity']:
        sim_matrix = results['company_similarity']
        companies = list(sim_matrix.keys())
        matrix = [[sim_matrix[c1][c2] for c2 in companies] for c1 in companies]
        
        im = ax4.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax4.set_xticks(range(len(companies)))
        ax4.set_yticks(range(len(companies)))
        ax4.set_xticklabels(companies, rotation=45)
        ax4.set_yticklabels(companies)
        ax4.set_title('Company Strategy Similarity', fontweight='bold', fontsize=12)
        plt.colorbar(im, ax=ax4)
    
    # 5. Trends
    ax5 = axes[1, 1]
    if 'trends' in results and results['trends']['most_common_actions']:
        trends = results['trends']['most_common_actions']
        ax5.barh(list(trends.keys())[:5], list(trends.values())[:5], color='purple')
        ax5.set_title('Most Common Actions', fontweight='bold', fontsize=12)
        ax5.set_xlabel('Frequency')
    
    # 6. Themes
    ax6 = axes[1, 2]
    if 'themes' in results and results['themes']:
        theme_sizes = [t['num_actions'] for t in results['themes']]
        theme_labels = [f"Theme {t['theme_id']}" for t in results['themes']]
        ax6.scatter(range(len(theme_sizes)), theme_sizes, s=100, alpha=0.6)
        ax6.set_title('Action Clusters', fontweight='bold', fontsize=12)
        ax6.set_xlabel('Cluster ID')
        ax6.set_ylabel('Number of Actions')
    
    plt.suptitle('Advanced SEC Filing Analysis Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('advanced_sec_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n{Fore.GREEN}Visualization saved as 'advanced_sec_analysis.png'{Style.RESET_ALL}")

def main():
    print(f"\n{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}STARTING ADVANCED MULTI-MODEL ANALYSIS{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}")
    
    # Initialize
    config = AdvancedConfig()
    fetcher = SECFetcher(config)
    extractor = MultiModelExtractor(config)
    comparator = AdvancedComparator(config)
    
    all_actions = {}
    
    # Process companies
    for ticker, company_info in config.companies.items():
        print(f"\n{Fore.CYAN}Analyzing {ticker} - {company_info['name']}{Style.RESET_ALL}")
        
        filings = fetcher.fetch_filings(ticker, company_info)
        
        if not filings:
            all_actions[ticker] = []
            continue
        
        company_actions = []
        
        for filing in filings:
            print(f"  Processing {filing['form']}...")
            
            # Extract comprehensive actions
            actions = extractor.extract_comprehensive_actions(
                filing['content'][:20000],  # Process more content
                ticker
            )
            
            for action in actions:
                if action.confidence >= config.min_confidence:
                    company_actions.append(action)
                    
                    # Print high-quality extractions
                    if action.confidence > 0.6:
                        print(f"    {Fore.GREEN}→ {action.specific_action}{Style.RESET_ALL}")
                        print(f"      Category: {action.action_category}")
                        print(f"      Details: {action.description[:100]}...")
                        if action.entities_mentioned:
                            print(f"      Entities: {', '.join(action.entities_mentioned[:3])}")
                        if action.locations:
                            print(f"      Locations: {', '.join(action.locations)}")
                        if action.financial_impacts:
                            print(f"      Financial: {action.financial_impacts[0]}")
        
        all_actions[ticker] = company_actions
        print(f"  {Fore.YELLOW}Total: {len(company_actions)} actions extracted{Style.RESET_ALL}")
    
    # Advanced comparison
    print(f"\n{Fore.YELLOW}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}CROSS-COMPANY ANALYSIS{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'='*80}{Style.RESET_ALL}")
    
    results = comparator.analyze_cross_company(all_actions)
    
    # Print results
    summary = results['summary']
    print(f"\n{Fore.CYAN}SUMMARY:{Style.RESET_ALL}")
    print(f"  Total actions: {summary['total_actions']}")
    print(f"  Companies: {summary['companies_analyzed']}")
    print(f"  Similar strategies: {summary['similar_pairs']}")
    print(f"  Themes identified: {summary['themes_identified']}")
    
    # Similar strategies
    if results['similar_strategies']:
        print(f"\n{Fore.GREEN}SIMILAR STRATEGIES:{Style.RESET_ALL}")
        for sim in results['similar_strategies'][:5]:
            print(f"  • {sim['company1']} ↔ {sim['company2']} (similarity: {sim['similarity']:.2f})")
            print(f"    {sim['company1']}: {sim['action1']['specific']}")
            print(f"    {sim['company2']}: {sim['action2']['specific']}")
    
    # Themes
    if results['themes']:
        print(f"\n{Fore.BLUE}KEY THEMES:{Style.RESET_ALL}")
        for theme in results['themes'][:5]:
            print(f"  • Theme {theme['theme_id']}: {theme['primary_category']}")
            print(f"    Companies: {', '.join(theme['companies_involved'])}")
            print(f"    Actions: {theme['num_actions']}")
            if theme['common_entities']:
                print(f"    Key entities: {', '.join(theme['common_entities'][:3])}")
    
    # Trends
    if results['trends']['emerging_themes']:
        print(f"\n{Fore.YELLOW}EMERGING TRENDS:{Style.RESET_ALL}")
        for trend in results['trends']['emerging_themes'][:5]:
            print(f"  • {trend['theme']}: {', '.join(trend['companies'])}")
    
    # Visualizations
    create_advanced_visualizations(results, all_actions)
    
    # Save comprehensive results
    output_file = f"advanced_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    output_data = {
        'summary': summary,
        'similar_strategies': results['similar_strategies'][:10],
        'themes': results['themes'],
        'trends': results['trends'],
        'actions_by_company': {
            company: [{
                'category': a.action_category,
                'specific': a.specific_action,
                'description': a.description,
                'entities': a.entities_mentioned,
                'locations': a.locations,
                'confidence': a.confidence
            } for a in actions[:10]]
            for company, actions in all_actions.items()
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n{Fore.GREEN}Results saved to: {output_file}{Style.RESET_ALL}")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
        print(f"\n{Fore.GREEN}[SUCCESS] Advanced Analysis Complete{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}[ERROR] {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()

