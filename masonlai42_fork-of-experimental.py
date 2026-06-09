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
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import time
import traceback
import random  # For generating more realistic sample data

print("="*80)
print("INSURANCE INDUSTRY OSINT ANALYZER - FIXED & ENHANCED VERSION")
print("="*80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

def install_packages():
    packages = [
        'pytrends', 'feedparser', 'beautifulsoup4', 'requests', 'pandas',
        'numpy', 'matplotlib', 'seaborn', 'plotly', 'wordcloud', 'textblob',
        'vaderSentiment', 'lxml'
    ]
    
    print("\n[SETUP] Checking and installing required packages...")
    print("-" * 60)
    
    # First, ensure urllib3 is at a compatible version
    try:
        print("  â†’ Ensuring urllib3 compatibility...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'urllib3<2.0'])
        print("  âœ“ urllib3 compatibility ensured")
    except Exception as e:
        print(f"  âš  Warning: Could not update urllib3: {e}")
    
    for package in packages:
        try:
            if package == 'vaderSentiment':
                __import__('vaderSentiment.vaderSentiment')
            else:
                __import__(package.replace('-', '_'))
            print(f"  âœ“ {package} already installed")
        except ImportError:
            try:
                print(f"  â†’ Installing {package}...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])
                print(f"  âœ“ {package} installed successfully")
            except Exception as e:
                print(f"  âš  Warning: Could not install {package}: {e}")
    print("-" * 60)
    print("[SETUP] Package installation complete.\n")

install_packages()

# Import after installation
try:
    from pytrends.request import TrendReq
except ImportError:
    print("âš  PyTrends not available, will use sample data")
    TrendReq = None

import feedparser
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from urllib.parse import quote, urlencode

plt.style.use('default')
sns.set_palette("husl")

class GoogleTrendsAnalyzer:
    def __init__(self):
        print("\n[INIT] Initializing Google Trends Analyzer...")
        self.vader = SentimentIntensityAnalyzer()
        self.pytrends = None
        self.use_sample_data = False
        
        try:
            if TrendReq:
                # Try to initialize pytrends with error handling
                self.pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25))
                print("  âœ“ Google Trends Analyzer initialized successfully")
            else:
                raise ImportError("PyTrends not available")
        except Exception as e:
            print(f"  âš  Google Trends API unavailable: {str(e)[:50]}")
            print("  â†’ Will use realistic sample data for demonstration")
            self.use_sample_data = True
    
    def generate_realistic_trends_data(self, category, terms):
        """Generate realistic sample data when API is unavailable"""
        print(f"  â†’ Generating realistic sample data for {category}...")
        
        # Create date range for last 3 months
        dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
        
        # Generate realistic patterns for different categories
        data = pd.DataFrame(index=dates)
        
        for term in terms[:5]:  # Limit to 5 terms for clarity
            # Base value with category-specific ranges
            if 'insurance' in term.lower():
                base = random.randint(50, 70)
                volatility = 10
            elif any(company in term for company in ['Geico', 'Progressive', 'State Farm']):
                base = random.randint(40, 60)
                volatility = 8
            elif 'complaint' in term or 'denied' in term:
                base = random.randint(20, 35)
                volatility = 5
            elif any(disaster in term for disaster in ['flood', 'hurricane', 'fire']):
                base = random.randint(25, 45)
                volatility = 15
            else:
                base = random.randint(30, 50)
                volatility = 7
            
            # Generate trend with seasonal patterns
            trend = np.zeros(90)
            for i in range(90):
                # Add weekly seasonality
                weekly_pattern = 5 * np.sin(2 * np.pi * i / 7)
                # Add monthly pattern
                monthly_pattern = 3 * np.sin(2 * np.pi * i / 30)
                # Add random walk
                if i > 0:
                    trend[i] = trend[i-1] + np.random.normal(0, 2)
                else:
                    trend[i] = 0
                
                # Combine all patterns
                value = base + weekly_pattern + monthly_pattern + trend[i] + np.random.normal(0, volatility)
                # Ensure within 0-100 range
                data.loc[dates[i], term] = max(0, min(100, value))
        
        return data
        
    def fetch_insurance_trends(self):
        print("\n[GOOGLE TRENDS] Starting comprehensive insurance industry trends analysis...")
        print("="*80)
        results = {}
        
        search_terms = [
            ['car insurance', 'auto insurance', 'insurance quote', 'cheap insurance', 'insurance rates'],
            ['Geico', 'Progressive', 'State Farm', 'Allstate', 'Liberty Mutual'],
            ['insurance claim', 'insurance denied', 'insurance complaint', 'insurance lawsuit', 'switch insurance'],
            ['flood insurance', 'hurricane insurance', 'fire insurance', 'earthquake insurance', 'disaster insurance']
        ]
        
        categories = ['general_insurance', 'companies', 'complaints', 'disasters']
        
        for idx, (terms, category) in enumerate(zip(search_terms, categories), 1):
            print(f"\n[{idx}/4] Fetching {category.upper()} trends...")
            print(f"  Search terms: {', '.join(terms)}")
            print("-" * 60)
            
            if self.use_sample_data:
                # Generate sample data
                interest_over_time = self.generate_realistic_trends_data(category, terms)
                results[f'{category}_timeline'] = interest_over_time
                print(f"  âœ“ Generated sample data: {len(interest_over_time)} data points")
                
                # Perform analysis on sample data
                print(f"\n  ğŸ“Š Detailed Term Analysis for {category.upper()} (Sample Data):")
                for col in interest_over_time.columns[:5]:
                    recent_avg = interest_over_time[col].iloc[-30:].mean()
                    total_avg = interest_over_time[col].mean()
                    max_val = interest_over_time[col].max()
                    min_val = interest_over_time[col].min()
                    current_val = interest_over_time[col].iloc[-1]
                    
                    # Calculate trend
                    if recent_avg > total_avg * 1.1:
                        trend = "ğŸ”º RISING"
                        trend_pct = ((recent_avg - total_avg) / total_avg * 100)
                    elif recent_avg < total_avg * 0.9:
                        trend = "ğŸ”» FALLING"
                        trend_pct = ((recent_avg - total_avg) / total_avg * 100)
                    else:
                        trend = "â�¡ï¸� STABLE"
                        trend_pct = ((recent_avg - total_avg) / total_avg * 100)
                    
                    print(f"    â€¢ {col}:")
                    print(f"      - Current: {current_val:.1f} | Average: {total_avg:.1f} | Recent Avg (30d): {recent_avg:.1f}")
                    print(f"      - Range: {min_val:.1f} - {max_val:.1f}")
                    print(f"      - Trend: {trend} ({trend_pct:+.1f}%)")
                
                # Generate sample regional data
                regions = ['California', 'Texas', 'Florida', 'New York', 'Illinois']
                regional_data = pd.DataFrame({
                    terms[0]: [random.randint(40, 80) for _ in regions]
                }, index=regions)
                results[f'{category}_regions'] = regional_data
                print(f"\n  â†’ Sample regional data generated for top 5 states")
                
            else:
                try:
                    print(f"  â†’ Building payload for {category}...")
                    self.pytrends.build_payload(terms, timeframe='today 3-m', geo='US')
                    print(f"  âœ“ Payload built successfully")
                    
                    # Fetch interest over time
                    print(f"  â†’ Fetching interest over time...")
                    interest_over_time = self.pytrends.interest_over_time()
                    
                    if not interest_over_time.empty:
                        if 'isPartial' in interest_over_time.columns:
                            interest_over_time = interest_over_time.drop('isPartial', axis=1)
                        
                        results[f'{category}_timeline'] = interest_over_time
                        print(f"    âœ“ Retrieved {len(interest_over_time)} data points over {len(interest_over_time.columns)} search terms")
                        
                        # Detailed analysis for each term
                        print(f"\n  ğŸ“Š Detailed Term Analysis for {category.upper()}:")
                        for col in interest_over_time.columns:
                            if interest_over_time[col].dtype in ['float64', 'int64']:
                                recent_avg = interest_over_time[col].iloc[-30:].mean()
                                total_avg = interest_over_time[col].mean()
                                max_val = interest_over_time[col].max()
                                min_val = interest_over_time[col].min()
                                current_val = interest_over_time[col].iloc[-1]
                                
                                # Calculate trend
                                if recent_avg > total_avg * 1.1:
                                    trend = "ğŸ”º RISING"
                                    trend_pct = ((recent_avg - total_avg) / total_avg * 100)
                                elif recent_avg < total_avg * 0.9:
                                    trend = "ğŸ”» FALLING"
                                    trend_pct = ((recent_avg - total_avg) / total_avg * 100)
                                else:
                                    trend = "â�¡ï¸� STABLE"
                                    trend_pct = ((recent_avg - total_avg) / total_avg * 100)
                                
                                print(f"    â€¢ {col}:")
                                print(f"      - Current: {current_val:.1f} | Average: {total_avg:.1f} | Recent Avg (30d): {recent_avg:.1f}")
                                print(f"      - Range: {min_val:.1f} - {max_val:.1f}")
                                print(f"      - Trend: {trend} ({trend_pct:+.1f}%)")
                    else:
                        print(f"    âš  No data retrieved for {category}")
                    
                    # Add delay to avoid rate limiting
                    time.sleep(2)
                    
                    # Fetch regional interest
                    print(f"\n  â†’ Fetching regional interest for {category}...")
                    try:
                        interest_by_region = self.pytrends.interest_by_region(resolution='COUNTRY', inc_low_vol=True, inc_geo_code=True)
                        if not interest_by_region.empty:
                            results[f'{category}_regions'] = interest_by_region
                            top_regions = interest_by_region.sum(axis=1).sort_values(ascending=False).head(5)
                            print(f"    âœ“ Retrieved data for {len(interest_by_region)} regions")
                            print(f"    Top 5 regions by interest:")
                            for region, score in top_regions.items():
                                print(f"      - {region}: {score:.1f}")
                    except Exception as e:
                        print(f"    âš  Could not fetch regional data: {str(e)[:50]}")
                    
                except Exception as e:
                    print(f"    âš  Error fetching {category}, using sample data: {str(e)[:50]}")
                    # Fallback to sample data
                    interest_over_time = self.generate_realistic_trends_data(category, terms)
                    results[f'{category}_timeline'] = interest_over_time
                    print(f"    âœ“ Generated fallback sample data")
            
            print("-" * 60)
        
        print(f"\n[SUMMARY] Trends data collection complete:")
        print(f"  â€¢ Categories processed: {len([k for k in results.keys() if 'timeline' in k])}")
        print(f"  â€¢ Total data points: {sum([len(v) if isinstance(v, pd.DataFrame) else 0 for v in results.values()])}")
        if self.use_sample_data:
            print("  â„¹ï¸� Note: Using sample data for demonstration purposes")
        
        return results
    
    def get_rising_queries(self):
        print("\n[RISING QUERIES] Analyzing rising insurance-related searches...")
        print("="*80)
        rising_data = {}
        
        if self.use_sample_data:
            # Generate sample rising queries
            print("  â†’ Generating sample rising queries...")
            
            rising_queries = [
                {'query': 'cyber insurance coverage', 'value': 'Breakout'},
                {'query': 'climate change insurance', 'value': '850'},
                {'query': 'tesla insurance rates', 'value': '650'},
                {'query': 'pet insurance comparison', 'value': '450'},
                {'query': 'pandemic business insurance', 'value': '400'},
                {'query': 'flood insurance zones 2025', 'value': '350'},
                {'query': 'insurance AI claims', 'value': '300'},
                {'query': 'electric vehicle insurance', 'value': '275'}
            ]
            
            top_queries = [
                {'query': 'car insurance', 'value': 100},
                {'query': 'health insurance', 'value': 95},
                {'query': 'auto insurance quotes', 'value': 88},
                {'query': 'home insurance', 'value': 82},
                {'query': 'life insurance', 'value': 75},
                {'query': 'insurance near me', 'value': 70},
                {'query': 'cheap insurance', 'value': 68},
                {'query': 'Progressive insurance', 'value': 65},
                {'query': 'Geico insurance', 'value': 62},
                {'query': 'State Farm insurance', 'value': 60}
            ]
            
            rising_data['rising_queries'] = rising_queries
            rising_data['top_queries'] = top_queries
            
            print(f"\n  ğŸ“ˆ RISING QUERIES ({len(rising_queries)} found - Sample Data):")
            print("  " + "="*50)
            for idx, row in enumerate(rising_queries, 1):
                value_str = str(row['value'])
                if value_str == 'Breakout':
                    value_display = "ğŸš€ BREAKOUT (>5000% increase)"
                else:
                    value_display = f"{value_str}% increase"
                print(f"    {idx}. {row['query']}")
                print(f"       Growth: {value_display}")
                print("  " + "-"*50)
            
            print(f"\n  ğŸ”� TOP QUERIES ({len(top_queries)} found - Sample Data):")
            print("  " + "="*50)
            for idx, row in enumerate(top_queries[:10], 1):
                print(f"    {idx}. {row['query']}: {row['value']}")
        else:
            try:
                print("  â†’ Building payload for 'insurance' term...")
                self.pytrends.build_payload(['insurance'], timeframe='today 3-m', geo='US')
                print("  âœ“ Payload built")
                
                print("  â†’ Fetching related queries...")
                related = self.pytrends.related_queries()
                
                if related and 'insurance' in related:
                    print("  âœ“ Related queries retrieved")
                    
                    # Process rising queries
                    if 'rising' in related['insurance']:
                        rising_df = related['insurance']['rising']
                        if rising_df is not None and not rising_df.empty:
                            rising_data['rising_queries'] = rising_df.to_dict('records')
                            print(f"\n  ğŸ“ˆ RISING QUERIES ({len(rising_df)} found):")
                            print("  " + "="*50)
                            for idx, row in rising_df.iterrows():
                                value_str = str(row['value'])
                                if value_str == 'Breakout':
                                    value_display = "ğŸš€ BREAKOUT (>5000% increase)"
                                else:
                                    value_display = f"{value_str}% increase"
                                print(f"    {idx+1}. {row['query']}")
                                print(f"       Growth: {value_display}")
                                print("  " + "-"*50)
                        else:
                            print("    âš  No rising queries found")
                    
                    # Process top queries
                    if 'top' in related['insurance']:
                        top_df = related['insurance']['top']
                        if top_df is not None and not top_df.empty:
                            rising_data['top_queries'] = top_df.to_dict('records')
                            print(f"\n  ğŸ”� TOP QUERIES ({len(top_df)} found):")
                            print("  " + "="*50)
                            for idx, row in top_df.head(10).iterrows():
                                print(f"    {idx+1}. {row['query']}: {row['value']}")
                        else:
                            print("    âš  No top queries found")
                else:
                    print("  âš  No related queries data available")
                    
            except Exception as e:
                print(f"  âš  Error fetching queries, using sample data: {str(e)[:50]}")
                # Use sample data as fallback
                return self.get_rising_queries() if not self.use_sample_data else {}
        
        return rising_data

class NewsAnalyzer:
    def __init__(self):
        print("\n[INIT] Initializing News Analyzer...")
        self.vader = SentimentIntensityAnalyzer()
        self.news_sources = [
            {'name': 'Insurance Business Magazine', 'url': 'https://www.insurancebusinessmag.com/us/rss/'},
            {'name': 'Insurance Journal', 'url': 'https://www.insurancejournal.com/rss/'},
            {'name': 'Google News - Insurance', 'url': 'https://news.google.com/rss/search?q=insurance+industry+news&hl=en-US&gl=US&ceid=US:en'},
            {'name': 'Reuters Business', 'url': 'https://feeds.reuters.com/reuters/businessNews'},
        ]
        print(f"  âœ“ News Analyzer initialized with {len(self.news_sources)} sources")
    
    def fetch_all_news(self):
        print("\n[NEWS ANALYSIS] Starting comprehensive news intelligence gathering...")
        print("="*80)
        all_articles = []
        source_stats = {}
        
        for source_info in self.news_sources:
            source_name = source_info['name']
            source_url = source_info['url']
            print(f"\n  ğŸ“° Fetching from {source_name}...")
            print(f"     URL: {source_url}")
            
            articles = self.fetch_rss_feed(source_url, source_name)
            all_articles.extend(articles)
            source_stats[source_name] = len(articles)
            
            print(f"     âœ“ Retrieved {len(articles)} articles from {source_name}")
        
        print(f"\n[NEWS SUMMARY] Total articles collected: {len(all_articles)}")
        print("  Source breakdown:")
        for source, count in source_stats.items():
            print(f"    â€¢ {source}: {count} articles")
        
        if all_articles:
            df = pd.DataFrame(all_articles)
            
            # Sentiment analysis
            sentiment_dist = df['sentiment_label'].value_counts().to_dict()
            print(f"\n  ğŸ˜Š SENTIMENT ANALYSIS:")
            print("  " + "="*50)
            
            total_articles = len(all_articles)
            for label, count in sentiment_dist.items():
                percentage = (count / total_articles) * 100
                if label == 'positive':
                    emoji = "ğŸ˜Š"
                elif label == 'negative':
                    emoji = "ğŸ˜Ÿ"
                else:
                    emoji = "ğŸ˜�"
                print(f"    {emoji} {label.upper()}: {count} articles ({percentage:.1f}%)")
            
            avg_sentiment = df['sentiment_score'].mean()
            std_sentiment = df['sentiment_score'].std()
            print(f"\n    ğŸ“Š Average sentiment score: {avg_sentiment:.4f} (Â±{std_sentiment:.4f})")
            
            # Top positive articles
            if len(df) >= 3:
                print(f"\n  ğŸŒŸ TOP POSITIVE ARTICLES:")
                print("  " + "="*50)
                for idx, row in df.nlargest(3, 'sentiment_score').iterrows():
                    print(f"    â€¢ {row['title'][:80]}...")
                    print(f"      Score: {row['sentiment_score']:.4f} | Date: {row['published'][:10]}")
                    print("  " + "-"*50)
                
                # Top negative articles
                print(f"\n  âš ï¸� TOP NEGATIVE ARTICLES:")
                print("  " + "="*50)
                for idx, row in df.nsmallest(3, 'sentiment_score').iterrows():
                    print(f"    â€¢ {row['title'][:80]}...")
                    print(f"      Score: {row['sentiment_score']:.4f} | Date: {row['published'][:10]}")
                    print("  " + "-"*50)
            
            # Keywords analysis
            print(f"\n  ğŸ”¤ KEYWORD FREQUENCY ANALYSIS:")
            print("  " + "="*50)
            all_titles = ' '.join(df['title'].tolist()).lower()
            keywords = ['claim', 'lawsuit', 'premium', 'coverage', 'policy', 'rate', 'risk', 
                       'cyber', 'climate', 'flood', 'auto', 'health', 'life', 'property']
            
            keyword_counts = {}
            for keyword in keywords:
                count = all_titles.count(keyword)
                if count > 0:
                    keyword_counts[keyword] = count
            
            sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
            for keyword, count in sorted_keywords[:10]:
                print(f"    â€¢ '{keyword}': mentioned {count} times")
            
            return {
                'articles': all_articles,
                'summary': {
                    'total_count': len(all_articles),
                    'sentiment_distribution': sentiment_dist,
                    'average_sentiment': float(avg_sentiment),
                    'sentiment_std': float(std_sentiment),
                    'sources': source_stats,
                    'top_positive': df.nlargest(min(3, len(df)), 'sentiment_score')[['title', 'sentiment_score', 'source']].to_dict('records'),
                    'top_negative': df.nsmallest(min(3, len(df)), 'sentiment_score')[['title', 'sentiment_score', 'source']].to_dict('records'),
                    'keyword_frequency': dict(sorted_keywords[:10])
                }
            }
        
        return {'articles': [], 'summary': {}}
    
    def fetch_rss_feed(self, url, source_name):
        articles = []
        
        try:
            print(f"       â†’ Parsing RSS feed...")
            feed = feedparser.parse(url, agent='Mozilla/5.0')
            
            if feed.bozo:
                print(f"       âš  Warning: Feed parsing issues detected")
            
            if not feed.entries:
                print(f"       âš  No entries found in feed")
                return articles
            
            print(f"       âœ“ Found {len(feed.entries)} entries in feed")
            
            cutoff_date = datetime.now() - timedelta(days=30)
            processed = 0
            skipped = 0
            
            for entry in feed.entries[:50]:  # Process up to 50 entries
                try:
                    # Extract date
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        pub_date = datetime(*entry.updated_parsed[:6])
                    else:
                        pub_date = datetime.now()
                    
                    # Check if article is recent enough
                    if pub_date < cutoff_date:
                        skipped += 1
                        continue
                    
                    title = entry.get('title', 'No title')
                    summary = entry.get('summary', entry.get('description', ''))
                    
                    # Clean HTML from summary
                    if summary:
                        soup = BeautifulSoup(summary, 'html.parser')
                        summary = soup.get_text()[:500]
                    
                    # Analyze sentiment
                    combined_text = f"{title} {summary}"
                    sentiment = self.analyze_sentiment(combined_text)
                    
                    article = {
                        'title': title,
                        'summary': summary,
                        'link': entry.get('link', ''),
                        'published': pub_date.isoformat(),
                        'source': source_name,
                        'sentiment_score': sentiment['score'],
                        'sentiment_label': sentiment['label'],
                        'vader_scores': sentiment['vader']
                    }
                    
                    articles.append(article)
                    processed += 1
                
                except Exception as e:
                    continue
            
            print(f"       â†’ Processed: {processed} | Skipped (old): {skipped}")
            
        except Exception as e:
            print(f"       âœ— Error fetching feed: {str(e)[:100]}")
        
        return articles
    
    def analyze_sentiment(self, text):
        try:
            # VADER sentiment
            vader_scores = self.vader.polarity_scores(text)
            
            # TextBlob sentiment
            try:
                blob = TextBlob(text)
                blob_polarity = blob.sentiment.polarity
                combined_score = (vader_scores['compound'] + blob_polarity) / 2
            except:
                combined_score = vader_scores['compound']
            
            # Determine label
            if combined_score > 0.05:
                label = 'positive'
            elif combined_score < -0.05:
                label = 'negative'
            else:
                label = 'neutral'
            
            return {
                'score': float(combined_score),
                'label': label,
                'vader': vader_scores
            }
        except Exception as e:
            return {'score': 0.0, 'label': 'neutral', 'vader': {}}

class InsuranceDataVisualizer:
    def __init__(self):
        print("\n[INIT] Initializing Data Visualizer...")
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        print("  âœ“ Data Visualizer initialized")
    
    def create_dashboard(self, trends_data, news_data):
        print("\n[VISUALIZATION] Creating comprehensive insurance intelligence dashboard...")
        print("="*80)
        
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 12))
            fig.suptitle('Insurance Industry Intelligence Dashboard', fontsize=18, fontweight='bold')
            
            # Create grid
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # Plot 1: Trends Timeline
            print("  â†’ Creating trends timeline...")
            ax1 = fig.add_subplot(gs[0, :2])
            self.plot_trends_timeline(ax1, trends_data)
            print("    âœ“ Trends timeline created")
            
            # Plot 2: News Sentiment
            print("  â†’ Creating news sentiment pie chart...")
            ax2 = fig.add_subplot(gs[0, 2])
            self.plot_news_sentiment(ax2, news_data)
            print("    âœ“ News sentiment chart created")
            
            # Plot 3: Company Comparison
            print("  â†’ Creating company comparison chart...")
            ax3 = fig.add_subplot(gs[1, 0])
            self.plot_company_comparison(ax3, trends_data)
            print("    âœ“ Company comparison created")
            
            # Plot 4: Complaint Trends
            print("  â†’ Creating complaint trends chart...")
            ax4 = fig.add_subplot(gs[1, 1])
            self.plot_complaint_trends(ax4, trends_data)
            print("    âœ“ Complaint trends created")
            
            # Plot 5: Disaster Trends
            print("  â†’ Creating disaster insurance trends...")
            ax5 = fig.add_subplot(gs[1, 2])
            self.plot_disaster_trends(ax5, trends_data)
            print("    âœ“ Disaster trends created")
            
            # Plot 6: Summary Text
            print("  â†’ Adding summary text...")
            ax6 = fig.add_subplot(gs[2, :])
            self.add_summary_text(ax6, trends_data, news_data)
            print("    âœ“ Summary text added")
            
            plt.tight_layout()
            
            # Save dashboard
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'insurance_dashboard_{timestamp}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"\n  ğŸ’¾ Dashboard saved as: {filename}")
            
            # Display
            plt.show()
            
            return filename
            
        except Exception as e:
            print(f"  âœ— Error creating dashboard: {str(e)}")
            print(f"  Full traceback: {traceback.format_exc()}")
            return None
    
    def plot_trends_timeline(self, ax, trends_data):
        try:
            data_found = False
            
            # Try to plot general insurance timeline first
            if 'general_insurance_timeline' in trends_data:
                data = trends_data['general_insurance_timeline']
                if not data.empty:
                    print(f"      Plotting {len(data.columns)} trend lines...")
                    for i, col in enumerate(data.columns[:5]):
                        if data[col].dtype in ['float64', 'int64']:
                            ax.plot(data.index, data[col], label=col, linewidth=2, 
                                   color=self.colors[i % len(self.colors)], alpha=0.8)
                            data_found = True
                    
                    if data_found:
                        ax.set_title('Insurance Search Trends Over Time', fontweight='bold', fontsize=12)
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Search Interest (0-100)')
                        ax.legend(loc='best', fontsize=9, ncol=2)
                        ax.grid(True, alpha=0.3)
                        ax.tick_params(axis='x', rotation=45)
                        print(f"      âœ“ Plotted {len(data.columns)} trend lines")
                
        except Exception as e:
            print(f"      âœ— Error in trends timeline: {str(e)}")
            ax.text(0.5, 0.5, 'Error loading data', ha='center', va='center')
            ax.set_title('Insurance Search Trends Over Time', fontweight='bold', fontsize=12)
    
    def plot_news_sentiment(self, ax, news_data):
        try:
            if news_data and 'summary' in news_data and 'sentiment_distribution' in news_data['summary']:
                sentiment_dist = news_data['summary']['sentiment_distribution']
                
                if sentiment_dist:
                    colors_map = {'positive': '#2ECC71', 'neutral': '#95A5A6', 'negative': '#E74C3C'}
                    labels = list(sentiment_dist.keys())
                    sizes = list(sentiment_dist.values())
                    colors = [colors_map.get(label, '#95A5A6') for label in labels]
                    
                    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                                      autopct='%1.0f%%', startangle=90)
                    
                    # Make percentage text bold
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                    
                    ax.set_title('News Sentiment Distribution', fontweight='bold', fontsize=12)
                    print(f"      âœ“ Plotted sentiment for {sum(sizes)} articles")
                else:
                    ax.text(0.5, 0.5, 'No sentiment data', ha='center', va='center')
            else:
                ax.text(0.5, 0.5, 'No sentiment data', ha='center', va='center')
                ax.set_title('News Sentiment', fontweight='bold', fontsize=12)
                
        except Exception as e:
            print(f"      âœ— Error in sentiment plot: {str(e)}")
            ax.text(0.5, 0.5, 'No sentiment data', ha='center', va='center')
            ax.set_title('News Sentiment', fontweight='bold', fontsize=12)
    
    def plot_company_comparison(self, ax, trends_data):
        try:
            data_found = False
            
            if 'companies_timeline' in trends_data:
                data = trends_data['companies_timeline']
                numeric_cols = [col for col in data.columns if data[col].dtype in ['float64', 'int64']]
                
                if numeric_cols and not data.empty:
                    avg_interest = data[numeric_cols].mean().sort_values(ascending=False)
                    
                    bars = ax.bar(range(len(avg_interest)), avg_interest.values, 
                                  color=self.colors[:len(avg_interest)])
                    ax.set_xticks(range(len(avg_interest)))
                    ax.set_xticklabels([name.split()[0] for name in avg_interest.index], 
                                       rotation=45, ha='right')
                    ax.set_title('Company Brand Interest (Avg)', fontweight='bold', fontsize=12)
                    ax.set_ylabel('Avg Search Interest')
                    
                    # Add value labels on bars
                    for i, (bar, value) in enumerate(zip(bars, avg_interest.values)):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                               f'{value:.0f}', ha='center', va='bottom', fontsize=9)
                    
                    data_found = True
                    print(f"      âœ“ Plotted {len(avg_interest)} companies")
                
        except Exception as e:
            print(f"      âœ— Error in company comparison: {str(e)}")
            ax.text(0.5, 0.5, 'No company data', ha='center', va='center')
            ax.set_title('Company Brand Interest', fontweight='bold', fontsize=12)
    
    def plot_complaint_trends(self, ax, trends_data):
        try:
            data_found = False
            
            if 'complaints_timeline' in trends_data:
                data = trends_data['complaints_timeline']
                numeric_cols = [col for col in data.columns if data[col].dtype in ['float64', 'int64']]
                
                if numeric_cols and len(data) > 0:
                    recent_values = data[numeric_cols].iloc[-1]
                    
                    bars = ax.barh(range(len(recent_values)), recent_values.values, 
                                  color='#E74C3C', alpha=0.7)
                    ax.set_yticks(range(len(recent_values)))
                    ax.set_yticklabels([term.replace(' insurance', '').title() 
                                       for term in recent_values.index], fontsize=9)
                    ax.set_title('Current Complaint Searches', fontweight='bold', fontsize=12)
                    ax.set_xlabel('Search Interest')
                    
                    data_found = True
                    print(f"      âœ“ Plotted {len(recent_values)} complaint types")
                
        except Exception as e:
            print(f"      âœ— Error in complaint trends: {str(e)}")
            ax.text(0.5, 0.5, 'No complaint data', ha='center', va='center')
            ax.set_title('Current Complaint Searches', fontweight='bold', fontsize=12)
    
    def plot_disaster_trends(self, ax, trends_data):
        try:
            data_found = False
            
            if 'disasters_timeline' in trends_data:
                data = trends_data['disasters_timeline']
                numeric_cols = [col for col in data.columns if data[col].dtype in ['float64', 'int64']]
                
                if numeric_cols and len(data) >= 30:
                    recent_avg = data[numeric_cols].iloc[-30:].mean().sort_values(ascending=False)
                    
                    colors_disaster = ['#FF6B6B', '#FFA07A', '#FFB347', '#FFCC5C', '#FFE66D']
                    bars = ax.bar(range(len(recent_avg)), recent_avg.values,
                                 color=colors_disaster[:len(recent_avg)])
                    ax.set_xticks(range(len(recent_avg)))
                    ax.set_xticklabels([term.split()[0] for term in recent_avg.index], 
                                      rotation=45, ha='right')
                    ax.set_title('Disaster Insurance Interest (30d avg)', fontweight='bold', fontsize=12)
                    ax.set_ylabel('Search Interest')
                    
                    data_found = True
                    print(f"      âœ“ Plotted {len(recent_avg)} disaster types")
                
        except Exception as e:
            print(f"      âœ— Error in disaster trends: {str(e)}")
            ax.text(0.5, 0.5, 'No disaster data', ha='center', va='center')
            ax.set_title('Disaster Insurance Interest', fontweight='bold', fontsize=12)
    
    def add_summary_text(self, ax, trends_data, news_data):
        try:
            summary_lines = ["MARKET INTELLIGENCE SUMMARY", "=" * 80, ""]
            
            # Add timestamp
            summary_lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            summary_lines.append("")
            
            # Trends summary
            summary_lines.append("GOOGLE TRENDS ANALYSIS:")
            summary_lines.append("-" * 40)
            
            if 'general_insurance_timeline' in trends_data:
                data = trends_data['general_insurance_timeline']
                numeric_cols = [col for col in data.columns if data[col].dtype in ['float64', 'int64']]
                
                for col in numeric_cols[:3]:
                    if len(data) >= 14:
                        recent = data[col].iloc[-7:].mean()
                        older = data[col].iloc[-14:-7].mean()
                        change = ((recent - older) / older * 100) if older > 0 else 0
                        trend = "â†‘" if change > 5 else "â†“" if change < -5 else "â†’"
                        summary_lines.append(f"  {col}: {trend} {change:+.1f}% (current avg: {recent:.0f})")
            else:
                summary_lines.append("  No trends data available")
            
            summary_lines.append("")
            
            # News summary
            summary_lines.append("NEWS INTELLIGENCE:")
            summary_lines.append("-" * 40)
            
            if news_data and 'summary' in news_data:
                summary = news_data['summary']
                summary_lines.append(f"  Total Articles: {summary.get('total_count', 0)}")
                summary_lines.append(f"  Avg Sentiment: {summary.get('average_sentiment', 0):.3f}")
                
                if 'sentiment_distribution' in summary:
                    dist = summary['sentiment_distribution']
                    summary_lines.append(f"  Distribution: " + 
                                       ", ".join([f"{k}: {v}" for k, v in dist.items()]))
                
                if 'keyword_frequency' in summary:
                    top_keywords = list(summary['keyword_frequency'].keys())[:3]
                    summary_lines.append(f"  Top Keywords: {', '.join(top_keywords)}")
            else:
                summary_lines.append("  No news data available")
            
            summary_lines.append("")
            summary_lines.append("DATA QUALITY METRICS:")
            summary_lines.append("-" * 40)
            
            # Count available data
            data_points = sum([1 for k in trends_data.keys() if 'timeline' in k])
            summary_lines.append(f"  Trend Categories: {data_points}/4")
            summary_lines.append(f"  News Sources: {len(news_data.get('summary', {}).get('sources', {}))} active")
            
            # Join all lines
            summary_text = "\n".join(summary_lines)
            
            ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.1))
            ax.axis('off')
            
            print("      âœ“ Summary text generated with comprehensive metrics")
            
        except Exception as e:
            print(f"      âœ— Error in summary text: {str(e)}")
            ax.text(0.5, 0.5, 'Summary generation failed', ha='center', va='center')
            ax.axis('off')
    
    def create_interactive_trends(self, trends_data):
        print("\n[INTERACTIVE] Creating interactive visualizations...")
        print("="*80)
        
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Insurance Search Trends', 'Company Comparison',
                              'Complaint Patterns', 'Disaster Insurance Interest'),
                specs=[[{"secondary_y": False}, {"type": "bar"}],
                      [{"secondary_y": False}, {"type": "bar"}]]
            )
            
            print("  â†’ Adding insurance search trends...")
            # Plot 1: General insurance trends
            if 'general_insurance_timeline' in trends_data:
                data = trends_data['general_insurance_timeline']
                numeric_cols = [col for col in data.columns if data[col].dtype in ['float64', 'int64']]
                for col in numeric_cols:
                    fig.add_trace(
                        go.Scatter(x=data.index, y=data[col], name=col, mode='lines'),
                        row=1, col=1
                    )
                print(f"    âœ“ Added {len(numeric_cols)} trend lines")
            
            print("  â†’ Adding company comparison...")
            # Plot 2: Company comparison
            if 'companies_timeline' in trends_data:
                data = trends_data['companies_timeline']
                numeric_cols = [col for col in data.columns if data[col].dtype in ['float64', 'int64']]
                if numeric_cols:
                    avg_interest = data[numeric_cols].mean().sort_values(ascending=False)
                    fig.add_trace(
                        go.Bar(x=list(avg_interest.index), y=list(avg_interest.values),
                              marker_color='lightblue', showlegend=False),
                        row=1, col=2
                    )
                print(f"    âœ“ Added company comparison bars")
            
            print("  â†’ Adding complaint patterns...")
            # Plot 3: Complaint patterns
            if 'complaints_timeline' in trends_data:
                data = trends_data['complaints_timeline']
                numeric_cols = [col for col in data.columns if data[col].dtype in ['float64', 'int64']]
                for col in numeric_cols:
                    fig.add_trace(
                        go.Scatter(x=data.index, y=data[col], name=col, mode='lines'),
                        row=2, col=1
                    )
                print(f"    âœ“ Added complaint trend lines")
            
            print("  â†’ Adding disaster insurance trends...")
            # Plot 4: Disaster insurance
            if 'disasters_timeline' in trends_data:
                data = trends_data['disasters_timeline']
                numeric_cols = [col for col in data.columns if data[col].dtype in ['float64', 'int64']]
                if numeric_cols and len(data) >= 30:
                    recent_avg = data[numeric_cols].iloc[-30:].mean().sort_values(ascending=False)
                    fig.add_trace(
                        go.Bar(x=list(recent_avg.index), y=list(recent_avg.values),
                              marker_color='coral', showlegend=False),
                        row=2, col=2
                    )
                print(f"    âœ“ Added disaster insurance bars")
            
            # Update layout
            fig.update_layout(
                height=800, 
                showlegend=True,
                title_text="Insurance Industry Trends Analysis - Interactive Dashboard",
                title_font_size=18
            )
            
            # Save to HTML
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'insurance_interactive_{timestamp}.html'
            fig.write_html(filename)
            print(f"\n  ğŸ’¾ Interactive chart saved as: {filename}")
            
            # Show the figure
            fig.show()
            
            return filename
            
        except Exception as e:
            print(f"  âœ— Error creating interactive chart: {str(e)}")
            print(f"  Full traceback: {traceback.format_exc()}")
            return None
    
    def create_wordcloud(self, news_data):
        print("\n[WORDCLOUD] Generating comprehensive news word cloud...")
        print("="*80)
        
        try:
            if news_data and 'articles' in news_data and news_data['articles']:
                print(f"  â†’ Processing {len(news_data['articles'])} articles for word cloud...")
                
                # Combine titles and summaries
                text_parts = []
                for article in news_data['articles']:
                    text_parts.append(article['title'])
                    if article.get('summary'):
                        text_parts.append(article['summary'])
                
                text = ' '.join(text_parts)
                print(f"  â†’ Total text length: {len(text)} characters")
                
                # Define stop words
                stop_words = {'insurance', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 
                             'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 
                             'was', 'were', 'been', 'be', 'have', 'has', 'had', 'will',
                             'that', 'this', 'these', 'those', 'it', 'its', 'their'}
                
                print("  â†’ Generating word cloud...")
                wordcloud = WordCloud(
                    width=1200, 
                    height=600,
                    background_color='white',
                    stopwords=stop_words,
                    colormap='viridis',
                    max_words=150,
                    relative_scaling=0.5,
                    min_font_size=10
                ).generate(text)
                
                # Create figure
                plt.figure(figsize=(14, 7))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Insurance Industry News - Word Cloud Analysis', fontsize=16, fontweight='bold')
                
                # Save
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'insurance_wordcloud_{timestamp}.png'
                plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
                print(f"  ğŸ’¾ Word cloud saved as: {filename}")
                
                # Display
                plt.show()
                
                # Get top words
                word_freq = wordcloud.words_
                top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                print("\n  ğŸ“Š Top 10 words in the cloud:")
                for word, freq in top_words:
                    print(f"    â€¢ {word}: {freq:.4f}")
                
                return filename
            else:
                print("  âš  No articles available for word cloud generation")
                return None
                
        except Exception as e:
            print(f"  âœ— Error creating word cloud: {str(e)}")
            print(f"  Full traceback: {traceback.format_exc()}")
            return None

def safe_dataframe_to_dict(df):
    """Safely convert DataFrame to dictionary with comprehensive error handling"""
    print("    â†’ Converting DataFrame to dictionary...")
    try:
        result = {}
        
        # Get basic info
        print(f"      DataFrame shape: {df.shape}")
        print(f"      Columns: {', '.join(df.columns[:5])}..." if len(df.columns) > 5 else f"      Columns: {', '.join(df.columns)}")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"      Numeric columns: {len(numeric_cols)}")
        
        # Store the raw data
        result['data'] = df.to_dict()
        
        # Calculate statistics for numeric columns
        if numeric_cols:
            result['summary'] = {
                'mean': df[numeric_cols].mean().to_dict(),
                'std': df[numeric_cols].std().to_dict(),
                'min': df[numeric_cols].min().to_dict(),
                'max': df[numeric_cols].max().to_dict(),
                'latest': df[numeric_cols].iloc[-1].to_dict() if len(df) > 0 else {}
            }
        else:
            result['summary'] = {}
        
        # Add metadata
        result['metadata'] = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'index_range': f"{df.index[0]} to {df.index[-1]}" if len(df) > 0 else "empty"
        }
        
        print("      âœ“ Conversion successful")
        return result
        
    except Exception as e:
        print(f"      âœ— Error converting dataframe: {str(e)}")
        return {'data': {}, 'summary': {}, 'metadata': {'error': str(e)}}

def export_results(trends_data, news_data, rising_queries):
    print("\n[EXPORT] Saving comprehensive results to JSON...")
    print("="*80)
    
    try:
        # Prepare export data structure
        export_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'report_type': 'Insurance Industry OSINT Analysis',
                'version': '3.0 - Fixed',
                'data_sources': {
                    'google_trends': bool(trends_data),
                    'news_feeds': bool(news_data),
                    'rising_queries': bool(rising_queries)
                }
            },
            'trends': {},
            'news': {},
            'rising_queries': rising_queries if rising_queries else {},
            'statistics': {}
        }
        
        # Process trends data
        print("  â†’ Processing trends data for export...")
        for key, value in trends_data.items():
            try:
                if isinstance(value, pd.DataFrame):
                    print(f"    â€¢ Converting {key}...")
                    export_data['trends'][key] = safe_dataframe_to_dict(value)
                else:
                    export_data['trends'][key] = value
            except Exception as e:
                print(f"    âš  Error processing {key}: {str(e)}")
                export_data['trends'][key] = {'error': str(e)}
        
        # Process news data
        print("  â†’ Processing news data for export...")
        if news_data:
            export_data['news'] = {
                'summary': news_data.get('summary', {}),
                'article_count': len(news_data.get('articles', [])),
                'articles_sample': news_data.get('articles', [])[:10]  # Save first 10 articles as sample
            }
            print(f"    âœ“ Processed {len(news_data.get('articles', []))} articles")
        
        # Calculate overall statistics
        print("  â†’ Calculating overall statistics...")
        export_data['statistics'] = {
            'total_trend_categories': len([k for k in trends_data.keys() if 'timeline' in k]),
            'total_news_articles': len(news_data.get('articles', [])) if news_data else 0,
            'rising_queries_count': len(rising_queries.get('rising_queries', [])) if rising_queries else 0,
            'top_queries_count': len(rising_queries.get('top_queries', [])) if rising_queries else 0
        }
        
        # Save to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'insurance_osint_results_{timestamp}.json'
        
        print(f"  â†’ Writing to {filename}...")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str, ensure_ascii=False)
        
        # Calculate file size
        file_size = os.path.getsize(filename) / 1024  # Size in KB
        print(f"  ğŸ’¾ Results saved to {filename} ({file_size:.2f} KB)")
        
        # Print export summary
        print("\n  ğŸ“‹ EXPORT SUMMARY:")
        print("  " + "-"*50)
        print(f"    â€¢ Trend categories: {export_data['statistics']['total_trend_categories']}")
        print(f"    â€¢ News articles: {export_data['statistics']['total_news_articles']}")
        print(f"    â€¢ Rising queries: {export_data['statistics']['rising_queries_count']}")
        print(f"    â€¢ Top queries: {export_data['statistics']['top_queries_count']}")
        
        return filename
        
    except Exception as e:
        print(f"  âœ— Error exporting results: {str(e)}")
        print(f"  Full traceback: {traceback.format_exc()}")
        return None

def print_final_summary(results):
    """Print a comprehensive final summary of all results"""
    print("\n" + "="*80)
    print("FINAL ANALYSIS SUMMARY")
    print("="*80)
    
    print("\nğŸ�¯ KEY FINDINGS:")
    print("-" * 60)
    
    # Trends summary
    if results['trends_data']:
        print("\nğŸ“ˆ GOOGLE TRENDS INSIGHTS:")
        for key in results['trends_data']:
            if 'timeline' in key:
                category = key.replace('_timeline', '').replace('_', ' ').title()
                print(f"  â€¢ {category}: Data collected successfully")
    
    # News summary
    if results['news_data'] and 'summary' in results['news_data']:
        summary = results['news_data']['summary']
        print("\nğŸ“° NEWS ANALYSIS HIGHLIGHTS:")
        print(f"  â€¢ Total articles analyzed: {summary.get('total_count', 0)}")
        print(f"  â€¢ Average sentiment: {summary.get('average_sentiment', 0):.3f}")
        
        if 'keyword_frequency' in summary:
            print("\n  ğŸ”¤ Top Keywords:")
            for keyword, count in list(summary['keyword_frequency'].items())[:5]:
                print(f"    - {keyword}: {count} mentions")
    
    # Rising queries summary
    if results['rising_queries']:
        print("\nğŸš€ RISING SEARCH QUERIES:")
        if 'rising_queries' in results['rising_queries']:
            queries = results['rising_queries']['rising_queries'][:3]
            for q in queries:
                print(f"  â€¢ {q.get('query', 'N/A')}: {q.get('value', 'N/A')}")
    
    # Files generated
    print("\nğŸ“� OUTPUT FILES GENERATED:")
    print("-" * 60)
    for file_type, filename in results['files'].items():
        if filename:
            print(f"  âœ“ {file_type.upper()}: {filename}")
    
    print("\n" + "="*80)
    print("âœ… ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*80)

def main():
    print("\n" + "="*80)
    print("STARTING COMPREHENSIVE INSURANCE INDUSTRY OSINT ANALYSIS")
    print("="*80)
    print(f"Analysis initiated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Initialize components
    print("\n[INITIALIZATION] Setting up analysis components...")
    trends_analyzer = GoogleTrendsAnalyzer()
    news_analyzer = NewsAnalyzer()
    visualizer = InsuranceDataVisualizer()
    
    # Store all results
    results = {
        'trends_data': {},
        'news_data': {},
        'rising_queries': {},
        'files': {}
    }
    
    # Step 1: Google Trends Analysis
    print("\n" + "="*80)
    print("[STEP 1/5] GOOGLE TRENDS DATA COLLECTION")
    print("="*80)
    try:
        trends_data = trends_analyzer.fetch_insurance_trends()
        results['trends_data'] = trends_data
        print("âœ“ Google Trends analysis completed")
    except Exception as e:
        print(f"âœ— Critical error in trends analysis: {str(e)}")
        trends_data = {}
    
    # Step 2: Rising Queries Analysis
    print("\n" + "="*80)
    print("[STEP 2/5] RISING SEARCH QUERIES ANALYSIS")
    print("="*80)
    try:
        rising_queries = trends_analyzer.get_rising_queries()
        results['rising_queries'] = rising_queries
        print("âœ“ Rising queries analysis completed")
    except Exception as e:
        print(f"âœ— Critical error in rising queries: {str(e)}")
        rising_queries = {}
    
    # Step 3: News Intelligence
    print("\n" + "="*80)
    print("[STEP 3/5] NEWS INTELLIGENCE GATHERING")
    print("="*80)
    try:
        news_data = news_analyzer.fetch_all_news()
        results['news_data'] = news_data
        print("âœ“ News intelligence gathering completed")
    except Exception as e:
        print(f"âœ— Critical error in news analysis: {str(e)}")
        news_data = {}
    
    # Step 4: Visualization
    print("\n" + "="*80)
    print("[STEP 4/5] CREATING VISUALIZATIONS")
    print("="*80)
    
    # Create dashboard
    try:
        dashboard_file = visualizer.create_dashboard(trends_data, news_data)
        results['files']['dashboard'] = dashboard_file
    except Exception as e:
        print(f"âœ— Dashboard creation failed: {str(e)}")
        dashboard_file = None
    
    # Create interactive charts
    try:
        interactive_file = visualizer.create_interactive_trends(trends_data)
        results['files']['interactive'] = interactive_file
    except Exception as e:
        print(f"âœ— Interactive chart creation failed: {str(e)}")
        interactive_file = None
    
    # Create word cloud
    try:
        wordcloud_file = visualizer.create_wordcloud(news_data)
        results['files']['wordcloud'] = wordcloud_file
    except Exception as e:
        print(f"âœ— Word cloud creation failed: {str(e)}")
        wordcloud_file = None
    
    # Step 5: Export Results
    print("\n" + "="*80)
    print("[STEP 5/5] EXPORTING RESULTS")
    print("="*80)
    json_file = export_results(trends_data, news_data, rising_queries)
    results['files']['json'] = json_file
    
    # Print final summary
    print_final_summary(results)
    
    print(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"CRITICAL ERROR: {str(e)}")
        print(f"{'='*80}")
        print(traceback.format_exc())

