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


!pip uninstall -y transformers accelerate -q
!pip install transformers==4.44.0 accelerate==0.33.0 -q
!pip install torch sentencepiece protobuf safetensors -q
!pip install pytrends feedparser gnews beautifulsoup4 newspaper3k lxml_html_clean -q
!pip install textblob wordcloud vaderSentiment spacy -q
!python -m spacy download en_core_web_sm -q

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import requests
import json
import time
import warnings
import re
from collections import Counter, defaultdict
import gc
from io import BytesIO
import sys
warnings.filterwarnings('ignore')

from newspaper import Article
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from pytrends.request import TrendReq
import spacy

import torch
import torch.nn as nn
from transformers import AutoTokenizer, Qwen2ForCausalLM, Qwen2Config

try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm", "-q"])
    nlp = spacy.load("en_core_web_sm")

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (28, 20)

print("ğŸš€ Starting Ultra-Enhanced Market Sentiment Analysis with Qwen-3 8B LLM...")
print("="*80)

if torch.cuda.is_available():
    print(f"ğŸ�® GPUs Available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"   Memory: {props.total_memory / 1e9:.1f} GB")
    torch.cuda.empty_cache()
    gc.collect()
else:
    print("ğŸ’» Running on CPU")

print("="*80)

class UltraQwen3MarketAnalyzer:
    def __init__(self):
        print("\nğŸ¤– Loading Ultra-Enhanced Qwen-3 8B Model for Market Analysis...")
        self.model = None
        self.tokenizer = None
        self.vader = SentimentIntensityAnalyzer()
        self.model_loaded = False
        
        model_path = '/kaggle/input/qwen-3/transformers/8b-base/1'
        
        if os.path.exists(model_path):
            print(f"  Found Qwen-3 8B at: {model_path}")
            try:
                print("  Loading tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    padding_side='left'
                )
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
                print("  Loading configuration...")
                with open(os.path.join(model_path, 'config.json'), 'r') as f:
                    config_dict = json.load(f)
                
                config_dict['model_type'] = 'qwen2'
                config = Qwen2Config(**config_dict)
                
                print("  Loading Qwen-3 8B (using Qwen2ForCausalLM architecture)...")
                self.model = Qwen2ForCausalLM.from_pretrained(
                    model_path,
                    config=config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    ignore_mismatched_sizes=True
                )
                
                self.model.eval()
                self.model_loaded = True
                print("  âœ… Qwen-3 8B loaded successfully!")
                
                if hasattr(self.model, 'hf_device_map'):
                    print("\n  Model distribution across devices:")
                    device_counts = {}
                    for name, device in list(self.model.hf_device_map.items())[:10]:
                        device_str = f"GPU {device}" if isinstance(device, int) else str(device)
                        device_counts[device_str] = device_counts.get(device_str, 0) + 1
                    for device, count in device_counts.items():
                        print(f"    {device}: {count} layers")
                        
            except Exception as e:
                print(f"  âš ï¸� Could not load Qwen-3 model: {str(e)[:200]}")
                print("  Using fallback analysis methods")
                self.model_loaded = False
        else:
            print(f"  âš ï¸� Model path not found: {model_path}")
            print("  Using fallback analysis methods")
            self.model_loaded = False
    
    def generate_text(self, prompt, max_new_tokens=100, temperature=0.7):
        if not self.model_loaded or not self.model or not self.tokenizer:
            return None
            
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    top_k=50,
                    do_sample=True,
                    repetition_penalty=1.3,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if prompt in response:
                response = response[len(prompt):].strip()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return response
            
        except Exception as e:
            return None
    
    def analyze_market_implications(self, text):
        prompts = {
            'bullish_signals': f"Identify bullish signals for stocks/futures in: '{text[:300]}'\nBullish factors:",
            'bearish_signals': f"Identify bearish signals for stocks/futures in: '{text[:300]}'\nBearish factors:",
            'affected_sectors': f"Which market sectors are affected by: '{text[:300]}'\nSectors:",
            'trading_opportunity': f"What trading opportunities exist from: '{text[:300]}'\nOpportunities:",
            'risk_assessment': f"Market risks from: '{text[:300]}'\nRisks:",
            'price_impact': f"Potential price impact on S&P500/NASDAQ from: '{text[:300]}'\nImpact:"
        }
        
        analysis = {}
        for key, prompt in prompts.items():
            response = self.generate_text(prompt, max_new_tokens=40)
            if response:
                analysis[key] = response[:150]
        
        return analysis
    
    def market_sentiment_analysis(self, text):
        sentiment_dimensions = {
            'market_bullishness': f"Rate market bullishness (0-10): '{text[:200]}'\nScore:",
            'market_bearishness': f"Rate market bearishness (0-10): '{text[:200]}'\nScore:",
            'volatility_expectation': f"Expected volatility (0-10): '{text[:200]}'\nScore:",
            'investor_confidence': f"Investor confidence level (0-10): '{text[:200]}'\nScore:",
            'fear_greed_index': f"Fear vs Greed sentiment (0-10): '{text[:200]}'\nScore:",
            'institutional_sentiment': f"Institutional investor sentiment: '{text[:200]}'\nSentiment:",
            'retail_sentiment': f"Retail investor sentiment: '{text[:200]}'\nSentiment:",
            'momentum_strength': f"Market momentum strength (0-10): '{text[:200]}'\nScore:",
            'trend_reversal_risk': f"Risk of trend reversal (0-10): '{text[:200]}'\nRisk:",
            'earnings_impact': f"Impact on earnings expectations: '{text[:200]}'\nImpact:",
            'fed_policy_impact': f"Federal Reserve policy impact: '{text[:200]}'\nImpact:",
            'global_market_impact': f"Global market correlation: '{text[:200]}'\nCorrelation:",
            'sector_rotation': f"Sector rotation signals: '{text[:200]}'\nSignals:",
            'options_sentiment': f"Options market sentiment: '{text[:200]}'\nSentiment:",
            'futures_positioning': f"Futures positioning bias: '{text[:200]}'\nBias:"
        }
        
        sentiments = {}
        for key, prompt in sentiment_dimensions.items():
            response = self.generate_text(prompt, max_new_tokens=25, temperature=0.5)
            if response:
                sentiments[key] = response[:100]
        
        return sentiments
    
    def analyze_trading_signals(self, paragraph):
        analysis_prompts = {
            'entry_points': f"Trading entry points from: '{paragraph[:400]}'\nEntries:",
            'exit_strategy': f"Exit strategy based on: '{paragraph[:400]}'\nExits:",
            'stop_loss_levels': f"Stop loss considerations: '{paragraph[:400]}'\nLevels:",
            'position_sizing': f"Position sizing recommendation: '{paragraph[:400]}'\nSizing:",
            'timeframe': f"Best trading timeframe: '{paragraph[:400]}'\nTimeframe:",
            'technical_levels': f"Key technical levels: '{paragraph[:400]}'\nLevels:",
            'market_catalyst': f"Market catalysts identified: '{paragraph[:400]}'\nCatalysts:",
            'correlation_trades': f"Correlated trades: '{paragraph[:400]}'\nTrades:",
            'hedge_strategies': f"Hedging strategies: '{paragraph[:400]}'\nHedges:",
            'risk_reward': f"Risk/reward assessment: '{paragraph[:400]}'\nRatio:"
        }
        
        analysis = {}
        for key, prompt in analysis_prompts.items():
            response = self.generate_text(prompt, max_new_tokens=60)
            if response:
                analysis[key] = response[:200]
        
        return analysis
    
    def comprehensive_market_analysis(self, text):
        doc = nlp(text[:1000])
        
        sentences = [sent.text for sent in doc.sents]
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        nouns = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
        verbs = [token.text for token in doc if token.pos_ == "VERB"]
        adjectives = [token.text for token in doc if token.pos_ == "ADJ"]
        
        sentiment_scores = self.vader.polarity_scores(text[:5000])
        
        bullish_keywords = ['rally', 'surge', 'gain', 'rise', 'bullish', 'uptrend', 'breakout', 
                           'momentum', 'buy', 'long', 'growth', 'expand', 'recovery', 'strength',
                           'outperform', 'upgrade', 'positive', 'optimistic', 'boom', 'soar']
        bullish_score = sum(1 for word in bullish_keywords if word in text.lower()) / len(bullish_keywords)
        
        bearish_keywords = ['crash', 'plunge', 'fall', 'decline', 'bearish', 'downtrend', 'breakdown',
                           'sell', 'short', 'recession', 'weakness', 'underperform', 'downgrade',
                           'negative', 'pessimistic', 'correction', 'volatility', 'fear', 'panic']
        bearish_score = sum(1 for word in bearish_keywords if word in text.lower()) / len(bearish_keywords)
        
        result = {
            'sentiment': sentiment_scores['compound'],
            'positive': sentiment_scores['pos'],
            'negative': sentiment_scores['neg'],
            'neutral': sentiment_scores['neu'],
            'entities': entities[:30],
            'entity_types': Counter([ent[1] for ent in entities]),
            'nouns': Counter(nouns).most_common(20),
            'verbs': Counter(verbs).most_common(15),
            'adjectives': Counter(adjectives).most_common(10),
            'word_count': len(doc),
            'sentence_count': len(sentences),
            'bullish_level': bullish_score,
            'bearish_level': bearish_score,
            'summary': text[:400] + '...' if len(text) > 400 else text
        }
        
        if self.model_loaded and self.model and self.tokenizer:
            try:
                market_implications = self.analyze_market_implications(text)
                result['market_implications'] = market_implications
                
                market_sentiment = self.market_sentiment_analysis(text)
                result['market_sentiment'] = market_sentiment
                
                if len(sentences) > 0:
                    first_sentence_analysis = {
                        'sentence': sentences[0][:200],
                        'analysis': self.analyze_market_implications(sentences[0])
                    }
                    result['first_sentence_analysis'] = first_sentence_analysis
                
                trading_signals = self.analyze_trading_signals(text[:500])
                result['trading_signals'] = trading_signals
                
                strategic_prompts = [
                    f"Portfolio implications: '{text[:300]}'\nImplications:",
                    f"Market risk assessment: '{text[:300]}'\nRisks:",
                    f"Trading opportunities: '{text[:300]}'\nOpportunities:",
                    f"Recommended market actions: '{text[:300]}'\nActions:",
                    f"Affected stocks/ETFs: '{text[:300]}'\nTickers:",
                    f"Market timing signals: '{text[:300]}'\nTiming:",
                    f"Intermarket relationships: '{text[:300]}'\nRelationships:",
                    f"Market predictions: '{text[:300]}'\nPredictions:"
                ]
                
                strategic_analysis = {}
                for i, prompt in enumerate(strategic_prompts):
                    response = self.generate_text(prompt, max_new_tokens=50)
                    if response:
                        key = prompt.split(':')[0].lower().replace(' ', '_')
                        strategic_analysis[key] = response[:150]
                
                result['strategic_analysis'] = strategic_analysis
                
                questions_prompt = f"Generate critical market questions about: '{text[:300]}'\nQuestions:\n1."
                questions_response = self.generate_text(questions_prompt, max_new_tokens=80)
                if questions_response:
                    result['critical_questions'] = questions_response[:250]
                
            except Exception as e:
                result['llm_error'] = str(e)[:100]
        
        return result

def fetch_rss_content(url, timeout=10):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
        return feedparser.parse(BytesIO(response.content))
    except:
        return None

class UltraMarketDataCollector:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.all_articles = []
        
    def collect_market_news(self):
        print("\nğŸ“¥ Collecting and ultra-analyzing market news and data...")
        
        feeds = {
            'MarketWatch': 'https://feeds.content.dowjones.io/public/rss/mw_topstories',
            'Bloomberg Markets': 'https://feeds.bloomberg.com/markets/news.rss',
            'WSJ Markets': 'https://feeds.a.dj.com/rss/RSSMarketsMain.xml',
            'CNBC Markets': 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258',
            'Reuters Markets': 'https://feeds.reuters.com/reuters/businessNews',
            'Yahoo Finance': 'https://finance.yahoo.com/rss/topstories',
            'Seeking Alpha': 'https://seekingalpha.com/feed.xml',
            'Financial Times': 'https://www.ft.com/markets?format=rss',
            'Investing.com': 'https://www.investing.com/rss/news.rss',
            'TheStreet': 'https://www.thestreet.com/feeds/rss/markets',
            'Benzinga': 'https://feeds.benzinga.com/benzinga',
            'Forbes Markets': 'https://www.forbes.com/markets/feed/',
            'Barrons': 'https://feeds.barrons.com/rss/RSSMarketsMain.xml',
            'FXStreet': 'https://www.fxstreet.com/rss/news',
            'Zacks': 'https://www.zacks.com/feed/rss/news/all'
        }
        
        search_terms = ['stock', 'market', 'S&P', 'nasdaq', 'dow', 'futures', 
                       'trading', 'earnings', 'fed', 'rate', 'inflation', 'economy',
                       'bull', 'bear', 'rally', 'selloff', 'volatility', 'options',
                       'tesla', 'apple', 'microsoft', 'nvidia', 'amazon', 'meta',
                       'oil', 'gold', 'dollar', 'yield', 'bond', 'crypto', 'bitcoin',
                       'forecast', 'analyst', 'upgrade', 'downgrade', 'ipo', 'merger',
                       'guidance', 'revenue', 'profit', 'recession', 'growth', 'gdp']
        
        print(f"  Checking {len(feeds)} market RSS feeds with ultra-deep analysis...")
        
        for source, url in feeds.items():
            feed = fetch_rss_content(url, timeout=10)
            
            if feed and hasattr(feed, 'entries'):
                count = 0
                for entry in feed.entries[:10]:
                    text = f"{entry.get('title', '')} {entry.get('summary', '')}".lower()
                    
                    if any(term in text for term in search_terms):
                        article_data = {
                            'source': source,
                            'title': entry.get('title', ''),
                            'summary': entry.get('summary', '')[:3000],
                            'link': entry.get('link', ''),
                            'published': entry.get('published', ''),
                            'collected_at': datetime.now()
                        }
                        
                        analysis = self.analyzer.comprehensive_market_analysis(
                            f"{article_data['title']} {article_data['summary']}"
                        )
                        article_data.update(analysis)
                        
                        self.all_articles.append(article_data)
                        count += 1
                        
                        if count >= 5:
                            break
                
                if count > 0:
                    print(f"    âœ“ {source}: {count} articles ultra-analyzed")
        
        try:
            from gnews import GNews
            print("  Fetching Google News market coverage...")
            
            google_news = GNews(language='en', country='US', period='7d', max_results=20)
            
            searches = ['US stock market today', 'S&P 500 futures', 'NASDAQ trading news', 
                       'Dow Jones latest', 'Federal Reserve policy', 'Market volatility today',
                       'Tech stocks news', 'Earnings reports today', 'Stock market forecast',
                       'Trading opportunities', 'Market sentiment analysis', 'Wall Street news',
                       'Options market activity', 'Futures trading signals', 'Market crash risk']
            
            for search_term in searches:
                try:
                    news = google_news.get_news(search_term)
                    
                    for item in news[:3]:
                        article_data = {
                            'source': 'Google News Markets',
                            'title': item.get('title', ''),
                            'summary': item.get('description', ''),
                            'link': item.get('url', ''),
                            'published': item.get('published date', ''),
                            'collected_at': datetime.now(),
                            'search_term': search_term
                        }
                        
                        analysis = self.analyzer.comprehensive_market_analysis(
                            f"{article_data['title']} {article_data['summary']}"
                        )
                        article_data.update(analysis)
                        
                        self.all_articles.append(article_data)
                    
                    time.sleep(0.5)
                    
                except:
                    pass
            
            gnews_count = len([a for a in self.all_articles if a['source'] == 'Google News Markets'])
            if gnews_count > 0:
                print(f"    âœ“ Google News Markets: {gnews_count} articles with ultra-deep analysis")
                
        except:
            print("    âš  Google News unavailable")
        
        if len(self.all_articles) < 10:
            print("  Generating ultra-enhanced sample market data...")
            sample_sources = ['MarketWatch', 'CNBC', 'Bloomberg', 'WSJ', 'Reuters', 'Yahoo Finance', 'Barrons', 'TheStreet']
            sample_titles = [
                "Breaking: S&P 500 futures surge on strong earnings beats from tech giants",
                "Federal Reserve signals potential rate pause as inflation data moderates",
                "NASDAQ enters correction territory as growth stocks face selling pressure",
                "Options market shows extreme bullish positioning in semiconductor stocks",
                "Dollar strengthens as Treasury yields hit multi-year highs on hawkish Fed",
                "Volatility index spikes as geopolitical tensions impact global markets",
                "Tech mega-caps lead market rally as AI optimism drives investor sentiment",
                "Futures point to lower open as recession fears resurface on weak data"
            ]
            
            for i in range(8):
                article_data = {
                    'source': sample_sources[i % len(sample_sources)],
                    'title': sample_titles[i % len(sample_titles)],
                    'summary': f"Market Article {i+1}: Critical developments in US equity and futures markets continue to drive volatility and trading opportunities.",
                    'link': f"https://example.com/market-article{i+1}",
                    'published': datetime.now() - timedelta(hours=np.random.randint(1, 72)),
                    'collected_at': datetime.now()
                }
                
                analysis = self.analyzer.comprehensive_market_analysis(
                    f"{article_data['title']} {article_data['summary']}"
                )
                article_data.update(analysis)
                
                self.all_articles.append(article_data)
        
        df = pd.DataFrame(self.all_articles) if self.all_articles else pd.DataFrame()
        print(f"\n  ğŸ“Š TOTAL MARKET ARTICLES ULTRA-ANALYZED: {len(df)}")
        
        return df

class UltraMarketAnalysisEngine:
    def __init__(self, analyzer):
        self.analyzer = analyzer
    
    def process_market_articles(self, articles_df):
        results = {
            'processed_articles': articles_df,
            'sentiment_stats': {},
            'market_sentiment_breakdown': {},
            'keywords': [],
            'entities': [],
            'entity_breakdown': {},
            'nouns_analysis': [],
            'verbs_analysis': [],
            'adjectives_analysis': [],
            'bullish_analysis': [],
            'bearish_analysis': [],
            'market_implications_summary': [],
            'trading_signals_insights': [],
            'strategic_insights': [],
            'critical_questions': []
        }
        
        if articles_df.empty:
            return results
        
        print("  Processing with ultra-enhanced market sentiment analysis...")
        
        if 'sentiment' in articles_df.columns:
            sentiments = articles_df['sentiment'].tolist()
            results['sentiment_stats'] = {
                'mean': np.mean(sentiments),
                'std': np.std(sentiments),
                'median': np.median(sentiments),
                'min': np.min(sentiments),
                'max': np.max(sentiments),
                'bullish_pct': (np.array(sentiments) > 0.1).mean() * 100,
                'bearish_pct': (np.array(sentiments) < -0.1).mean() * 100,
                'neutral_pct': ((np.array(sentiments) >= -0.1) & (np.array(sentiments) <= 0.1)).mean() * 100,
                'extreme_bullish': (np.array(sentiments) > 0.5).mean() * 100,
                'extreme_bearish': (np.array(sentiments) < -0.5).mean() * 100,
                'volatility': np.std(sentiments) / abs(np.mean(sentiments)) if np.mean(sentiments) != 0 else 0
            }
        
        if 'market_sentiment' in articles_df.columns:
            market_sentiments = articles_df['market_sentiment'].dropna()
            if len(market_sentiments) > 0:
                sentiment_dimensions = set()
                for ms in market_sentiments:
                    if isinstance(ms, dict):
                        sentiment_dimensions.update(ms.keys())
                results['market_sentiment_breakdown'] = {
                    'dimensions_analyzed': list(sentiment_dimensions),
                    'samples': len(market_sentiments)
                }
        
        texts = articles_df['title'].tolist() + articles_df['summary'].tolist()
        texts = [str(t) for t in texts if pd.notna(t)]
        
        if texts:
            try:
                vectorizer = TfidfVectorizer(max_features=200, stop_words='english', ngram_range=(1, 3))
                tfidf_matrix = vectorizer.fit_transform(texts[:500])
                feature_names = vectorizer.get_feature_names_out()
                scores = tfidf_matrix.sum(axis=0).A1
                top_indices = scores.argsort()[-100:][::-1]
                keywords = [(feature_names[i], scores[i]) for i in top_indices]
                results['keywords'] = keywords
                
                lda = LatentDirichletAllocation(n_components=5, random_state=42)
                lda.fit(tfidf_matrix)
                topics = []
                for topic_idx, topic in enumerate(lda.components_):
                    top_words_idx = topic.argsort()[-10:][::-1]
                    top_words = [feature_names[i] for i in top_words_idx]
                    topics.append(top_words)
                results['topics'] = topics
            except:
                results['keywords'] = []
                results['topics'] = []
        
        if 'entities' in articles_df.columns:
            all_entities = []
            entity_types = defaultdict(list)
            for entity_list in articles_df['entities'].dropna():
                if isinstance(entity_list, list):
                    for ent in entity_list:
                        if isinstance(ent, tuple) and len(ent) == 2:
                            all_entities.append(ent[0])
                            entity_types[ent[1]].append(ent[0])
            
            entity_counts = Counter(all_entities)
            results['entities'] = entity_counts.most_common(40)
            results['entity_breakdown'] = {
                ent_type: Counter(ents).most_common(10) 
                for ent_type, ents in entity_types.items()
            }
        
        if 'nouns' in articles_df.columns:
            all_nouns = []
            for noun_list in articles_df['nouns'].dropna():
                if isinstance(noun_list, list):
                    all_nouns.extend([n[0] for n in noun_list if isinstance(n, tuple)])
            noun_counts = Counter(all_nouns)
            results['nouns_analysis'] = noun_counts.most_common(30)
        
        if 'verbs' in articles_df.columns:
            all_verbs = []
            for verb_list in articles_df['verbs'].dropna():
                if isinstance(verb_list, list):
                    all_verbs.extend([v[0] for v in verb_list if isinstance(v, tuple)])
            verb_counts = Counter(all_verbs)
            results['verbs_analysis'] = verb_counts.most_common(20)
        
        if 'adjectives' in articles_df.columns:
            all_adjectives = []
            for adj_list in articles_df['adjectives'].dropna():
                if isinstance(adj_list, list):
                    all_adjectives.extend([a[0] for a in adj_list if isinstance(a, tuple)])
            adj_counts = Counter(all_adjectives)
            results['adjectives_analysis'] = adj_counts.most_common(15)
        
        if 'bullish_level' in articles_df.columns:
            high_bullish = articles_df[articles_df['bullish_level'] > 0.1].nlargest(10, 'bullish_level')
            for idx, row in high_bullish.iterrows():
                results['bullish_analysis'].append({
                    'title': row['title'],
                    'bullish_level': row['bullish_level'],
                    'source': row['source']
                })
        
        if 'bearish_level' in articles_df.columns:
            high_bearish = articles_df[articles_df['bearish_level'] > 0.1].nlargest(10, 'bearish_level')
            for idx, row in high_bearish.iterrows():
                results['bearish_analysis'].append({
                    'title': row['title'],
                    'bearish_level': row['bearish_level'],
                    'source': row['source']
                })
        
        if 'market_implications' in articles_df.columns:
            for idx, row in articles_df[articles_df['market_implications'].notna()].head(5).iterrows():
                results['market_implications_summary'].append({
                    'title': row['title'],
                    'analysis': row['market_implications']
                })
        
        if 'trading_signals' in articles_df.columns:
            for idx, row in articles_df[articles_df['trading_signals'].notna()].head(5).iterrows():
                results['trading_signals_insights'].append({
                    'title': row['title'],
                    'insights': row['trading_signals']
                })
        
        if 'strategic_analysis' in articles_df.columns:
            for idx, row in articles_df[articles_df['strategic_analysis'].notna()].head(5).iterrows():
                results['strategic_insights'].append({
                    'title': row['title'],
                    'strategy': row['strategic_analysis']
                })
        
        if 'critical_questions' in articles_df.columns:
            for idx, row in articles_df[articles_df['critical_questions'].notna()].head(5).iterrows():
                results['critical_questions'].append({
                    'title': row['title'],
                    'questions': row['critical_questions']
                })
        
        return results

class UltraMarketTrendsCollector:
    def collect_market_trends(self):
        print("\nğŸ“Š Collecting comprehensive market trends and indicators...")
        all_trends = {}
        
        try:
            pytrends = TrendReq(hl='en-US', tz=360)
            
            keyword_sets = [
                ['SPY', 'QQQ', 'DIA', 'IWM', 'VIX'],
                ['Stock market', 'Bull market', 'Bear market', 'Recession', 'Rally'],
                ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
                ['Tesla stock', 'Meta stock', 'Netflix stock', 'AMD stock', 'Intel stock'],
                ['Fed rate', 'Interest rates', 'Inflation', 'CPI', 'PCE'],
                ['Earnings season', 'Stock earnings', 'Revenue growth', 'EPS', 'Guidance'],
                ['Options trading', 'Call options', 'Put options', 'Options volume', 'Greeks'],
                ['Futures trading', 'ES futures', 'NQ futures', 'Gold futures', 'Oil futures'],
                ['Crypto', 'Bitcoin', 'Ethereum', 'DeFi', 'NFT'],
                ['Day trading', 'Swing trading', 'Scalping', 'Algorithm trading', 'HFT']
            ]
            
            for i, keywords in enumerate(keyword_sets):
                try:
                    pytrends.build_payload(keywords[:5], timeframe='today 3-m', geo='US')
                    trends = pytrends.interest_over_time()
                    
                    if not trends.empty:
                        trends = trends.drop('isPartial', axis=1, errors='ignore')
                        all_trends[f'set_{i}'] = trends
                        print(f"    âœ“ Market trend set {i+1}: {', '.join(keywords[:3])}")
                    
                    time.sleep(1)
                    
                except:
                    print(f"    âš  Market trend set {i+1} error")
            
            try:
                for term in ['SPY', 'Stock market', 'Trading']:
                    pytrends.build_payload([term], timeframe='today 3-m', geo='US')
                    related = pytrends.related_queries()
                    if related and term in related:
                        all_trends[f'{term.lower()}_related'] = related[term]
                        print(f"    âœ“ {term} related queries collected")
                    time.sleep(1)
            except:
                pass
            
            print(f"  âœ“ Collected {len(all_trends)} market trend datasets")
            
        except:
            print(f"  âš  Market trends collection failed")
        
        return all_trends

def create_ultra_market_dashboard(data_dict, analyzer):
    print("\nğŸ“Š Creating ultra-enhanced market sentiment dashboard...")
    
    fig = plt.figure(figsize=(32, 24))
    gs = fig.add_gridspec(7, 6, hspace=0.35, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :2])
    if 'processed_articles' in data_dict and not data_dict['processed_articles'].empty:
        source_counts = data_dict['processed_articles']['source'].value_counts().head(15)
        colors = plt.cm.RdYlGn(np.linspace(0, 1, len(source_counts)))
        bars = ax1.bar(source_counts.index, source_counts.values, color=colors)
        ax1.set_title('Market News Sources Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Source')
        ax1.set_ylabel('Article Count')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        for i, (bar, v) in enumerate(zip(bars, source_counts.values)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(v), ha='center', va='bottom', fontsize=8)
    
    ax2 = fig.add_subplot(gs[0, 2:4])
    if 'sentiment_stats' in data_dict and data_dict['sentiment_stats']:
        stats = data_dict['sentiment_stats']
        labels = ['Bullish', 'Bearish', 'Neutral', 'Extreme Bull', 'Extreme Bear']
        sizes = [stats.get('bullish_pct', 0), 
                stats.get('bearish_pct', 0),
                stats.get('neutral_pct', 0),
                stats.get('extreme_bullish', 0),
                stats.get('extreme_bearish', 0)]
        colors = ['#00ff00', '#ff0000', '#808080', '#006400', '#8b0000']
        explode = (0.05, 0.05, 0, 0.1, 0.1)
        if sum(sizes) > 0:
            wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, 
                                                autopct='%1.1f%%', explode=explode, 
                                                shadow=True, startangle=90)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(9)
        ax2.set_title('Market Sentiment Distribution', fontweight='bold')
    
    ax3 = fig.add_subplot(gs[0, 4:])
    if 'sentiment_stats' in data_dict and data_dict['sentiment_stats']:
        stats = data_dict['sentiment_stats']
        metrics = ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Volatility']
        values = [stats.get('mean', 0), stats.get('median', 0), stats.get('std', 0),
                 stats.get('min', 0), stats.get('max', 0), stats.get('volatility', 0)]
        colors_bar = ['blue', 'cyan', 'orange', 'red', 'green', 'purple']
        bars = ax3.bar(metrics, values, color=colors_bar, alpha=0.7)
        ax3.set_title('Market Sentiment Metrics Analysis', fontweight='bold')
        ax3.set_ylabel('Score')
        ax3.grid(True, alpha=0.3)
        for bar, val in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax4 = fig.add_subplot(gs[1, :3])
    if 'keywords' in data_dict and data_dict['keywords']:
        keywords = data_dict['keywords'][:40]
        words = [k[0] for k in keywords]
        scores = [k[1] for k in keywords]
        colors = plt.cm.coolwarm(np.linspace(0.3, 0.9, len(words)))
        ax4.barh(words, scores, color=colors)
        ax4.set_title('Top 40 Market Keywords/Phrases (TF-IDF)', fontweight='bold')
        ax4.set_xlabel('TF-IDF Score')
        ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 3:])
    if 'topics' in data_dict and data_dict['topics']:
        topics_text = "MARKET TOPIC MODELING (LDA):\n" + "="*40 + "\n"
        for i, topic in enumerate(data_dict['topics'][:5]):
            topics_text += f"\nMarket Topic {i+1}:\n"
            topics_text += f"  {', '.join(topic[:8])}\n"
        ax5.text(0.02, 0.98, topics_text, transform=ax5.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.3))
        ax5.axis('off')
        ax5.set_title('Market Topic Discovery', fontweight='bold', y=0.98)
    
    ax6 = fig.add_subplot(gs[2, :2])
    if 'entities' in data_dict and data_dict['entities']:
        entities = data_dict['entities'][:25]
        entity_names = [e[0] for e in entities]
        entity_counts = [e[1] for e in entities]
        ax6.bar(entity_names, entity_counts, color='darkblue', alpha=0.7)
        ax6.set_title('Top Market Entities (Companies/Tickers)', fontweight='bold')
        ax6.set_xlabel('Entity')
        ax6.set_ylabel('Frequency')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3)
    
    ax7 = fig.add_subplot(gs[2, 2:4])
    if 'entity_breakdown' in data_dict and data_dict['entity_breakdown']:
        entity_types = list(data_dict['entity_breakdown'].keys())[:5]
        entity_counts = [len(data_dict['entity_breakdown'][et]) for et in entity_types]
        ax7.bar(entity_types, entity_counts, color='darkgreen', alpha=0.6)
        ax7.set_title('Market Entity Types Distribution', fontweight='bold')
        ax7.set_xlabel('Entity Type')
        ax7.set_ylabel('Unique Count')
        ax7.grid(True, alpha=0.3)
    
    ax8 = fig.add_subplot(gs[2, 4:])
    if 'trends' in data_dict and data_dict['trends']:
        for i, (key, trends) in enumerate(list(data_dict['trends'].items())[:2]):
            if isinstance(trends, pd.DataFrame) and not trends.empty:
                for col in trends.columns[:3]:
                    ax8.plot(trends.index, trends[col], marker='.', alpha=0.7, 
                            label=f"{key[:10]}-{col[:12]}", linewidth=2)
        ax8.set_title('Market Search Trends (Google Trends)', fontweight='bold')
        ax8.set_xlabel('Date')
        ax8.set_ylabel('Search Interest')
        ax8.legend(fontsize=7, loc='best', ncol=2)
        ax8.grid(True, alpha=0.3)
    
    ax9 = fig.add_subplot(gs[3, :3])
    if 'nouns_analysis' in data_dict and data_dict['nouns_analysis']:
        nouns = data_dict['nouns_analysis'][:20]
        noun_words = [n[0] for n in nouns]
        noun_counts = [n[1] for n in nouns]
        ax9.bar(noun_words, noun_counts, color='navy', alpha=0.6)
        ax9.set_title('Most Common Market Terms', fontweight='bold')
        ax9.set_xlabel('Term')
        ax9.set_ylabel('Frequency')
        ax9.tick_params(axis='x', rotation=45)
        ax9.grid(True, alpha=0.3)
    
    ax10 = fig.add_subplot(gs[3, 3:])
    verbs_adjs_text = ""
    if 'verbs_analysis' in data_dict and data_dict['verbs_analysis']:
        verbs_adjs_text += "MARKET ACTION VERBS:\n"
        for verb, count in data_dict['verbs_analysis'][:8]:
            verbs_adjs_text += f"  {verb}: {count}\n"
    if 'adjectives_analysis' in data_dict and data_dict['adjectives_analysis']:
        verbs_adjs_text += "\nMARKET DESCRIPTORS:\n"
        for adj, count in data_dict['adjectives_analysis'][:8]:
            verbs_adjs_text += f"  {adj}: {count}\n"
    ax10.text(0.02, 0.98, verbs_adjs_text, transform=ax10.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.3))
    ax10.axis('off')
    ax10.set_title('Market Language Analysis', fontweight='bold', y=0.98)
    
    ax11 = fig.add_subplot(gs[4, :])
    analysis_text = "QWEN-3 8B MARKET IMPLICATIONS ANALYSIS:\n" + "="*100 + "\n"
    
    if 'market_implications_summary' in data_dict and data_dict['market_implications_summary']:
        analysis_text += "\nğŸ“ˆ MARKET IMPLICATIONS:\n"
        for item in data_dict['market_implications_summary'][:2]:
            analysis_text += f"\n{item['title'][:70]}...\n"
            if isinstance(item['analysis'], dict):
                for key, value in list(item['analysis'].items())[:4]:
                    analysis_text += f"  {key.upper()}: {str(value)[:60]}\n"
    
    if 'strategic_insights' in data_dict and data_dict['strategic_insights']:
        analysis_text += "\nğŸ’¹ TRADING STRATEGY INSIGHTS:\n"
        for item in data_dict['strategic_insights'][:2]:
            analysis_text += f"\n{item['title'][:60]}...\n"
            if isinstance(item['strategy'], dict):
                for key, value in list(item['strategy'].items())[:3]:
                    analysis_text += f"  â€¢ {key.replace('_', ' ').title()}: {str(value)[:50]}\n"
    
    ax11.text(0.02, 0.98, analysis_text, transform=ax11.transAxes,
             fontsize=8, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.2))
    ax11.axis('off')
    
    ax12 = fig.add_subplot(gs[5, :])
    insights_text = "TRADING SIGNALS & CRITICAL MARKET QUESTIONS:\n" + "-"*100 + "\n"
    
    if 'trading_signals_insights' in data_dict and data_dict['trading_signals_insights']:
        insights_text += "\nğŸ“Š TRADING SIGNALS:\n"
        for item in data_dict['trading_signals_insights'][:2]:
            insights_text += f"\n{item['title'][:50]}...\n"
            if isinstance(item['insights'], dict):
                for key, value in list(item['insights'].items())[:3]:
                    insights_text += f"  {key}: {str(value)[:60]}\n"
    
    if 'critical_questions' in data_dict and data_dict['critical_questions']:
        insights_text += "\nâ�“ CRITICAL MARKET QUESTIONS:\n"
        for item in data_dict['critical_questions'][:3]:
            insights_text += f"â€¢ From: {item['title'][:50]}...\n"
            insights_text += f"  {str(item['questions'])[:100]}\n"
    
    ax12.text(0.02, 0.98, insights_text, transform=ax12.transAxes,
             fontsize=8, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.2))
    ax12.axis('off')
    
    ax13 = fig.add_subplot(gs[6, :])
    stats_text = "ULTRA-ENHANCED MARKET STATISTICS & MODEL STATUS:\n" + "-"*100 + "\n"
    
    if 'processed_articles' in data_dict and not data_dict['processed_articles'].empty:
        df = data_dict['processed_articles']
        stats_text += f"ğŸ“° Market articles analyzed: {len(df)}\n"
        stats_text += f"ğŸ“¡ Unique market sources: {df['source'].nunique()}\n"
        stats_text += f"ğŸ“� Total sentences: {df['sentence_count'].sum() if 'sentence_count' in df.columns else 'N/A'}\n"
        stats_text += f"ğŸ“Š Total words: {df['word_count'].sum() if 'word_count' in df.columns else 'N/A'}\n"
        
        if 'bullish_level' in df.columns:
            stats_text += f"ğŸŸ¢ Avg bullish level: {df['bullish_level'].mean():.3f} (Max: {df['bullish_level'].max():.3f})\n"
        
        if 'bearish_level' in df.columns:
            stats_text += f"ğŸ”´ Avg bearish level: {df['bearish_level'].mean():.3f} (Max: {df['bearish_level'].max():.3f})\n"
        
        if 'market_sentiment_breakdown' in data_dict and data_dict['market_sentiment_breakdown']:
            stats_text += f"ğŸ“ˆ Market dimensions analyzed: {len(data_dict['market_sentiment_breakdown'].get('dimensions_analyzed', []))}\n"
    
    model_status = "ğŸŸ¢ Qwen-3 8B Active (Market Analysis Mode)" if analyzer.model_loaded else "ğŸ”´ Fallback Mode"
    stats_text += f"\nMODEL: {model_status}\n"
    
    if torch.cuda.is_available():
        stats_text += "\nGPU UTILIZATION:\n"
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            stats_text += f"  GPU {i}: {allocated:.1f}GB active / {reserved:.1f}GB reserved\n"
    
    ax13.text(0.02, 0.98, stats_text, transform=ax13.transAxes,
             fontsize=8, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.2))
    ax13.axis('off')
    
    model_status_short = "Qwen-3 8B Market" if analyzer.model_loaded else "Fallback"
    plt.suptitle(f'Ultra-Enhanced Market Sentiment Analysis Dashboard - {model_status_short}', 
                fontsize=22, fontweight='bold', y=0.995)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.figtext(0.99, 0.003, f'Generated: {timestamp}', ha='right', fontsize=9)
    
    return fig

def main():
    print("\n" + "="*100)
    print("ğŸš€ ULTRA-ENHANCED MARKET SENTIMENT ANALYSIS SYSTEM WITH QWEN-3 8B")
    print("="*100)
    
    analyzer = UltraQwen3MarketAnalyzer()
    
    collector = UltraMarketDataCollector(analyzer)
    engine = UltraMarketAnalysisEngine(analyzer)
    trends_collector = UltraMarketTrendsCollector()
    
    data_dict = {}
    
    print("\nğŸ“¥ PHASE 1: ULTRA-ENHANCED MARKET DATA COLLECTION")
    print("-"*60)
    
    articles = collector.collect_market_news()
    if not articles.empty:
        print(f"  âœ… Successfully collected and ultra-analyzed {len(articles)} market articles")
    
    print("\nğŸ”� PHASE 2: ULTRA MARKET SENTIMENT PROCESSING")
    print("-"*60)
    
    if not articles.empty:
        processing_results = engine.process_market_articles(articles)
        data_dict.update(processing_results)
        print(f"  âœ“ Extracted {len(processing_results.get('keywords', []))} market keywords/phrases")
        print(f"  âœ“ Identified {len(processing_results.get('entities', []))} market entities")
        print(f"  âœ“ Analyzed {len(processing_results.get('nouns_analysis', []))} key market terms")
        print(f"  âœ“ Analyzed {len(processing_results.get('verbs_analysis', []))} market action verbs")
        print(f"  âœ“ Analyzed {len(processing_results.get('adjectives_analysis', []))} market descriptors")
        print(f"  âœ“ Found {len(processing_results.get('bullish_analysis', []))} bullish indicators")
        print(f"  âœ“ Found {len(processing_results.get('bearish_analysis', []))} bearish indicators")
        print(f"  âœ“ Generated {len(processing_results.get('market_implications_summary', []))} market implications")
        print(f"  âœ“ Created {len(processing_results.get('strategic_insights', []))} trading strategies")
        print(f"  âœ“ Identified {len(processing_results.get('trading_signals_insights', []))} trading signals")
    
    print("\nğŸ“ˆ PHASE 3: COMPREHENSIVE MARKET TRENDS ANALYSIS")
    print("-"*60)
    
    trends = trends_collector.collect_market_trends()
    data_dict['trends'] = trends
    
    print("\nğŸ“Š PHASE 4: ULTRA-ENHANCED MARKET VISUALIZATION")
    print("-"*60)
    
    fig = create_ultra_market_dashboard(data_dict, analyzer)
    plt.savefig('ultra_market_sentiment_dashboard_qwen3.png', dpi=150, bbox_inches='tight')
    print("  âœ“ Market dashboard saved: ultra_market_sentiment_dashboard_qwen3.png")
    plt.show()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not articles.empty:
        articles.to_csv(f'ultra_market_articles_qwen3_{timestamp}.csv', index=False)
        print(f"  âœ“ Market articles saved: ultra_market_articles_qwen3_{timestamp}.csv")
    
    with open(f'ultra_market_report_qwen3_{timestamp}.txt', 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("ULTRA-ENHANCED MARKET SENTIMENT ANALYSIS REPORT WITH QWEN-3 8B\n")
        f.write("="*100 + "\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Model: {'Qwen-3 8B Active (Market Analysis)' if analyzer.model_loaded else 'Fallback Mode'}\n\n")
        
        if not articles.empty:
            f.write("MARKET SUMMARY:\n")
            f.write("-"*60 + "\n")
            f.write(f"Total market articles: {len(articles)}\n")
            f.write(f"Market sources: {articles['source'].nunique()}\n")
            if 'collected_at' in articles.columns:
                f.write(f"Collection period: {articles['collected_at'].min()} to {articles['collected_at'].max()}\n")
            f.write("\n")
        
        if 'sentiment_stats' in data_dict and data_dict['sentiment_stats']:
            f.write("MARKET SENTIMENT ANALYSIS:\n")
            f.write("-"*60 + "\n")
            for key, value in data_dict['sentiment_stats'].items():
                f.write(f"  {key}: {value:.3f}\n")
            f.write("\n")
        
        if 'market_implications_summary' in data_dict and data_dict['market_implications_summary']:
            f.write("MARKET IMPLICATIONS:\n")
            f.write("-"*60 + "\n")
            for item in data_dict['market_implications_summary']:
                f.write(f"\nArticle: {item['title']}\n")
                if isinstance(item['analysis'], dict):
                    for k, v in item['analysis'].items():
                        f.write(f"  {k.upper()}: {v}\n")
        
        if 'strategic_insights' in data_dict and data_dict['strategic_insights']:
            f.write("\nTRADING STRATEGIES:\n")
            f.write("-"*60 + "\n")
            for item in data_dict['strategic_insights']:
                f.write(f"\nArticle: {item['title']}\n")
                if isinstance(item['strategy'], dict):
                    for k, v in item['strategy'].items():
                        f.write(f"  {k}: {v}\n")
        
        if 'trading_signals_insights' in data_dict and data_dict['trading_signals_insights']:
            f.write("\nTRADING SIGNALS:\n")
            f.write("-"*60 + "\n")
            for item in data_dict['trading_signals_insights']:
                f.write(f"\nFrom: {item['title']}\n")
                if isinstance(item['insights'], dict):
                    for k, v in item['insights'].items():
                        f.write(f"  {k}: {v}\n")
        
        if 'keywords' in data_dict and data_dict['keywords']:
            f.write("\nTOP MARKET KEYWORDS:\n")
            f.write("-"*60 + "\n")
            for keyword, score in data_dict['keywords'][:50]:
                f.write(f"  {keyword}: {score:.3f}\n")
    
    print(f"  âœ“ Market report saved: ultra_market_report_qwen3_{timestamp}.txt")
    
    print("\n" + "="*100)
    print("âœ¨ ULTRA-ENHANCED MARKET ANALYSIS COMPLETE!")
    print("="*100)
    
    print("\nFinal Market Summary:")
    print(f"  ğŸ¤– Model: {'Qwen-3 8B Active (Market)' if analyzer.model_loaded else 'Fallback'}")
    print(f"  ğŸ“° Market Articles: {len(articles) if not articles.empty else 0}")
    print(f"  ğŸ“Š Keywords: {len(data_dict.get('keywords', []))}")
    print(f"  ğŸ�·ï¸� Entities: {len(data_dict.get('entities', []))}")
    print(f"  ğŸ“ˆ Market Trends: {len(data_dict.get('trends', {}))}")
    print(f"  ğŸ’¹ Market Implications: {len(data_dict.get('market_implications_summary', []))}")
    print(f"  ğŸ�¯ Trading Strategies: {len(data_dict.get('strategic_insights', []))}")
    print(f"  ğŸ“Š Trading Signals: {len(data_dict.get('trading_signals_insights', []))}")
    print(f"  ğŸ’¾ Timestamp: {timestamp}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            print(f"  GPU {i}: {allocated:.1f} GB")
        torch.cuda.empty_cache()
    
    gc.collect()
    
    return data_dict, analyzer

if __name__ == "__main__":
    print("Initializing Ultra-Enhanced Market Sentiment System with Qwen-3 8B...")
    print("Loading comprehensive market analysis capabilities")
    print("-"*80)
    
    try:
        results, analyzer = main()
        print("\nğŸ�¯ SUCCESS!")
        print("Dashboard: ultra_market_sentiment_dashboard_qwen3.png")
        print("="*100)
    except Exception as e:
        print(f"\nâš ï¸� Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nAttempting recovery...")
        
        analyzer = UltraQwen3MarketAnalyzer()
        if not analyzer.model_loaded:
            print("Continuing with traditional methods")
        
        try:
            results, analyzer = main()
            print("\nâœ… Completed with available methods")
        except Exception as e2:
            print(f"\nâ�Œ Fatal: {str(e2)}")
            raise

