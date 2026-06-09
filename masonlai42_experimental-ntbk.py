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


"""
Insurance Industry Intelligence Analyzer - Expanded Edition
Comprehensive analysis including legal issues, claims disputes, coverage variations, and consumer behavior
"""

import subprocess
import sys
import os
import re
import json
import numpy as np
import pandas as pd
import requests
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import time
import traceback

print("="*80)
print("INSURANCE INDUSTRY INTELLIGENCE ANALYZER - EXPANDED EDITION")
print("="*80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

def install_packages():
    """Install and verify required packages with proper versions"""
    packages = [
        'pytrends', 'feedparser', 'beautifulsoup4', 'requests', 'pandas',
        'numpy', 'matplotlib', 'seaborn', 'plotly', 'wordcloud', 'textblob',
        'vaderSentiment', 'lxml', 'urllib3==1.26.15'
    ]
    
    print("\n[SETUP] Checking and installing required packages...")
    print("-" * 60)
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'urllib3==1.26.15'])
        print(f"  âœ“ urllib3 set to compatible version")
    except:
        pass
    
    for package in packages:
        if package.startswith('urllib3'):
            continue
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

from pytrends.request import TrendReq
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

# Expanded Configuration
US_STATES = {
    'US': 'United States (National)',
    'US-AL': 'Alabama', 'US-AK': 'Alaska', 'US-AZ': 'Arizona', 'US-AR': 'Arkansas',
    'US-CA': 'California', 'US-CO': 'Colorado', 'US-CT': 'Connecticut', 'US-DE': 'Delaware',
    'US-FL': 'Florida', 'US-GA': 'Georgia', 'US-HI': 'Hawaii', 'US-ID': 'Idaho',
    'US-IL': 'Illinois', 'US-IN': 'Indiana', 'US-IA': 'Iowa', 'US-KS': 'Kansas',
    'US-KY': 'Kentucky', 'US-LA': 'Louisiana', 'US-ME': 'Maine', 'US-MD': 'Maryland',
    'US-MA': 'Massachusetts', 'US-MI': 'Michigan', 'US-MN': 'Minnesota', 'US-MS': 'Mississippi',
    'US-MO': 'Missouri', 'US-MT': 'Montana', 'US-NE': 'Nebraska', 'US-NV': 'Nevada',
    'US-NH': 'New Hampshire', 'US-NJ': 'New Jersey', 'US-NM': 'New Mexico', 'US-NY': 'New York',
    'US-NC': 'North Carolina', 'US-ND': 'North Dakota', 'US-OH': 'Ohio', 'US-OK': 'Oklahoma',
    'US-OR': 'Oregon', 'US-PA': 'Pennsylvania', 'US-RI': 'Rhode Island', 'US-SC': 'South Carolina',
    'US-SD': 'South Dakota', 'US-TN': 'Tennessee', 'US-TX': 'Texas', 'US-UT': 'Utah',
    'US-VT': 'Vermont', 'US-VA': 'Virginia', 'US-WA': 'Washington', 'US-WV': 'West Virginia',
    'US-WI': 'Wisconsin', 'US-WY': 'Wyoming', 'US-DC': 'Washington DC'
}

# Significantly Expanded Insurance Keywords
INSURANCE_KEYWORDS = {
    'general_insurance': [
        'car insurance', 'auto insurance', 'home insurance', 'life insurance', 'health insurance',
        'renters insurance', 'motorcycle insurance', 'boat insurance', 'rv insurance', 'pet insurance',
        'travel insurance', 'disability insurance', 'business insurance', 'commercial insurance',
        'umbrella insurance', 'dental insurance', 'vision insurance', 'long term care insurance'
    ],
    
    'cost_and_pricing': [
        'insurance quotes', 'cheap insurance', 'insurance rates', 'insurance cost', 'insurance premium',
        'affordable insurance', 'low cost insurance', 'insurance discounts', 'insurance price comparison',
        'insurance deductible', 'insurance copay', 'out of pocket maximum', 'coinsurance',
        'insurance monthly payment', 'insurance annual cost', 'insurance rate increase'
    ],
    
    'major_companies': [
        'Geico', 'Progressive', 'State Farm', 'Allstate', 'Liberty Mutual',
        'USAA', 'Farmers Insurance', 'Nationwide', 'American Family', 'Travelers',
        'MetLife', 'Prudential', 'AIG', 'Chubb', 'Hartford', 'Esurance',
        'Erie Insurance', 'Auto-Owners', 'Amica', 'Mercury Insurance'
    ],
    
    'legal_issues': [
        'sue insurance company', 'insurance lawsuit', 'insurance bad faith', 'insurance fraud',
        'insurance company sued', 'class action insurance', 'insurance litigation', 'insurance attorney',
        'insurance lawyer', 'insurance legal action', 'insurance court case', 'insurance settlement',
        'insurance arbitration', 'insurance mediation', 'insurance complaint attorney',
        'insurance company illegal', 'insurance company investigation', 'insurance regulatory action'
    ],
    
    'claims_disputes': [
        'insurance claim denied', 'insurance denied coverage', 'insurance claim rejected', 'insurance dispute',
        'insurance claim delay', 'insurance not paying', 'insurance claim problem', 'fight insurance denial',
        'appeal insurance denial', 'insurance claim complaint', 'insurance refuses to pay',
        'insurance underpayment', 'insurance claim reduction', 'insurance claim investigation',
        'insurance adjuster problems', 'insurance company stalling', 'insurance company tricks'
    ],
    
    'coverage_levels': [
        'minimum coverage insurance', 'maximum coverage insurance', 'full coverage insurance',
        'liability only insurance', 'comprehensive insurance', 'collision coverage',
        'underinsured motorist', 'uninsured motorist', 'gap insurance', 'insurance coverage limits',
        'insurance policy limits', 'excess coverage', 'supplemental insurance', 'basic coverage',
        'enhanced coverage', 'premium coverage', 'platinum coverage', 'gold coverage'
    ],
    
    'shopping_behavior': [
        'switch insurance', 'change insurance', 'cancel insurance', 'insurance shopping',
        'compare insurance', 'insurance comparison', 'best insurance', 'worst insurance',
        'insurance reviews', 'insurance ratings', 'insurance recommendations', 'avoid insurance company',
        'insurance company to avoid', 'insurance company complaints', 'insurance customer service'
    ],
    
    'disasters_and_special': [
        'flood insurance', 'hurricane insurance', 'fire insurance', 'earthquake insurance',
        'tornado insurance', 'hail damage insurance', 'wildfire insurance', 'storm damage insurance',
        'natural disaster insurance', 'catastrophic insurance', 'emergency insurance',
        'pandemic insurance', 'cyber insurance', 'identity theft insurance', 'terrorism insurance'
    ],
    
    'regulatory_compliance': [
        'insurance regulation', 'insurance commissioner', 'insurance department complaint',
        'NAIC complaint', 'state insurance board', 'insurance ombudsman', 'insurance regulatory violation',
        'insurance license revoked', 'insurance company fined', 'insurance compliance',
        'insurance consumer protection', 'insurance regulatory investigation', 'DOI complaint'
    ],
    
    'specific_situations': [
        'sr22 insurance', 'high risk insurance', 'non owner insurance', 'insurance after accident',
        'insurance after DUI', 'insurance bad credit', 'insurance for seniors', 'insurance for veterans',
        'insurance for students', 'first time insurance', 'insurance lapse', 'insurance cancellation',
        'insurance reinstatement', 'insurance proof', 'insurance verification'
    ],
    
    'medical_insurance_issues': [
        'health insurance denied', 'prior authorization denied', 'medical claim denied',
        'insurance not covering medication', 'insurance formulary', 'insurance pre existing condition',
        'insurance waiting period', 'insurance out of network', 'insurance balance billing',
        'surprise medical bill', 'insurance EOB', 'insurance copay too high', 'insurance deductible not met'
    ],
    
    'auto_specific': [
        'car accident insurance', 'hit and run insurance', 'parking lot accident insurance',
        'rental car insurance', 'rideshare insurance', 'uber insurance', 'lyft insurance',
        'commercial auto insurance', 'fleet insurance', 'classic car insurance', 'exotic car insurance',
        'teen driver insurance', 'new driver insurance', 'defensive driving discount'
    ],
    
    'home_specific': [
        'homeowners claim', 'roof damage insurance', 'water damage insurance', 'mold insurance',
        'basement flooding insurance', 'sewer backup insurance', 'home renovation insurance',
        'vacant home insurance', 'landlord insurance', 'condo insurance', 'mobile home insurance',
        'home insurance inspection', 'home insurance appraisal'
    ],
    
    'fraud_and_scams': [
        'insurance scam', 'fake insurance', 'insurance fraud report', 'insurance ghost broker',
        'insurance phishing', 'insurance identity theft', 'staged accident', 'insurance fraud investigation',
        'insurance fraud penalty', 'insurance fraud hotline', 'suspicious insurance claim',
        'insurance company scam', 'insurance agent fraud'
    ],
    
    'technology_and_innovation': [
        'insurance app', 'online insurance', 'digital insurance', 'insurtech',
        'insurance AI', 'insurance blockchain', 'telematics insurance', 'usage based insurance',
        'pay per mile insurance', 'insurance chatbot', 'insurance automation',
        'insurance mobile app', 'insurance comparison website', 'insurance calculator'
    ]
}

# Expanded News Keywords for Analysis
ANALYSIS_KEYWORDS = [
    'claim', 'lawsuit', 'premium', 'coverage', 'policy', 'rate', 'risk',
    'cyber', 'climate', 'flood', 'auto', 'health', 'life', 'property',
    'denied', 'dispute', 'fraud', 'investigation', 'complaint', 'regulation',
    'settlement', 'litigation', 'class action', 'bad faith', 'underpayment',
    'cancellation', 'non-renewal', 'rate hike', 'merger', 'acquisition',
    'bankruptcy', 'insolvency', 'rating downgrade', 'data breach', 'hack'
]

class ExpandedGoogleTrendsAnalyzer:
    """Expanded Google Trends Analyzer with comprehensive keyword analysis"""
    
    def __init__(self):
        print("\n[INIT] Initializing Expanded Google Trends Analyzer...")
        try:
            self.pytrends = TrendReq(
                hl='en-US', 
                tz=360, 
                timeout=(10, 25),
                retries=2,
                backoff_factor=0.1,
                requests_args={'verify': True}
            )
            self.vader = SentimentIntensityAnalyzer()
            
            # Count total keywords
            total_keywords = sum(len(v) for v in INSURANCE_KEYWORDS.values())
            print(f"  âœ“ Analyzer initialized with {len(INSURANCE_KEYWORDS)} categories")
            print(f"  âœ“ Total keywords to analyze: {total_keywords}")
            
        except Exception as e:
            print(f"  âœ— Failed to initialize: {e}")
            try:
                self.pytrends = TrendReq(hl='en-US', tz=360)
                self.vader = SentimentIntensityAnalyzer()
                print("  âœ“ Analyzer initialized with basic settings")
            except:
                raise
    
    def safe_build_payload(self, keywords, timeframe='today 3-m', geo='US'):
        """Safely build payload with error handling"""
        try:
            self.pytrends.build_payload(
                keywords, 
                cat=0, 
                timeframe=timeframe, 
                geo=geo, 
                gprop=''
            )
            return True
        except Exception as e:
            if 'method_whitelist' in str(e):
                print("  â†’ Fixing urllib3 compatibility...")
                try:
                    import urllib3
                    self.pytrends = TrendReq(hl='en-US', tz=360)
                    self.pytrends.build_payload(keywords, timeframe=timeframe, geo=geo)
                    return True
                except:
                    return False
            return False
    
    def fetch_comprehensive_trends(self):
        """Fetch comprehensive insurance trends across all categories"""
        print("\n[GOOGLE TRENDS] Starting comprehensive insurance industry analysis...")
        print("="*80)
        
        results = {}
        category_summaries = {}
        
        for category_name, keywords in INSURANCE_KEYWORDS.items():
            print(f"\n[{category_name.upper()}] Analyzing {len(keywords)} keywords...")
            print(f"  Keywords: {', '.join(keywords[:3])}... (+{len(keywords)-3} more)")
            print("-" * 60)
            
            category_data = []
            category_trends = []
            
            try:
                # Process in batches of 5
                for i in range(0, len(keywords), 5):
                    batch = keywords[i:i+5]
                    print(f"  â†’ Processing batch {i//5 + 1}/{(len(keywords)-1)//5 + 1}: {batch}")
                    
                    if self.safe_build_payload(batch):
                        try:
                            interest_over_time = self.pytrends.interest_over_time()
                            
                            if not interest_over_time.empty:
                                if 'isPartial' in interest_over_time.columns:
                                    interest_over_time = interest_over_time.drop('isPartial', axis=1)
                                
                                category_data.append(interest_over_time)
                                
                                # Analyze each keyword's trend
                                for col in interest_over_time.columns:
                                    if interest_over_time[col].dtype in ['float64', 'int64']:
                                        recent_avg = interest_over_time[col].iloc[-30:].mean() if len(interest_over_time) >= 30 else interest_over_time[col].mean()
                                        total_avg = interest_over_time[col].mean()
                                        max_val = interest_over_time[col].max()
                                        current_val = interest_over_time[col].iloc[-1]
                                        
                                        if total_avg > 0:
                                            trend_pct = ((recent_avg - total_avg) / total_avg * 100)
                                        else:
                                            trend_pct = 0
                                        
                                        trend_direction = "ğŸ“ˆ" if trend_pct > 10 else "ğŸ“‰" if trend_pct < -10 else "â�¡ï¸�"
                                        
                                        category_trends.append({
                                            'keyword': col,
                                            'current': current_val,
                                            'average': total_avg,
                                            'recent_avg': recent_avg,
                                            'max': max_val,
                                            'trend_pct': trend_pct,
                                            'trend': trend_direction
                                        })
                            
                            time.sleep(1.5)  # Rate limiting
                            
                        except Exception as e:
                            print(f"    âš  Error fetching batch: {str(e)[:100]}")
                    else:
                        print(f"    âš  Could not build payload for batch")
                
                # Combine category data
                if category_data:
                    combined = pd.concat(category_data, axis=1)
                    combined = combined.loc[:, ~combined.columns.duplicated()]
                    results[f'{category_name}_timeline'] = combined
                    
                    # Category summary
                    if category_trends:
                        trending_up = [t for t in category_trends if t['trend'] == "ğŸ“ˆ"]
                        trending_down = [t for t in category_trends if t['trend'] == "ğŸ“‰"]
                        
                        category_summaries[category_name] = {
                            'total_keywords': len(keywords),
                            'data_collected': len(category_trends),
                            'trending_up': len(trending_up),
                            'trending_down': len(trending_down),
                            'top_gainers': sorted(trending_up, key=lambda x: x['trend_pct'], reverse=True)[:3],
                            'top_losers': sorted(trending_down, key=lambda x: x['trend_pct'])[:3],
                            'highest_interest': sorted(category_trends, key=lambda x: x['current'], reverse=True)[:3]
                        }
                        
                        # Print summary
                        print(f"\n  ğŸ“Š Category Summary:")
                        print(f"     â€¢ Keywords analyzed: {len(category_trends)}/{len(keywords)}")
                        print(f"     â€¢ Trending up: {len(trending_up)}")
                        print(f"     â€¢ Trending down: {len(trending_down)}")
                        
                        if category_summaries[category_name]['top_gainers']:
                            print(f"     â€¢ Top gainer: {category_summaries[category_name]['top_gainers'][0]['keyword']} "
                                  f"(+{category_summaries[category_name]['top_gainers'][0]['trend_pct']:.1f}%)")
                
                # Try to get regional data for top keyword
                if keywords and len(keywords) > 0:
                    try:
                        top_keyword = keywords[0]
                        print(f"\n  â†’ Fetching regional data for '{top_keyword}'...")
                        if self.safe_build_payload([top_keyword], timeframe='today 3-m', geo='US'):
                            regional_data = self.pytrends.interest_by_region(
                                resolution='REGION', 
                                inc_low_vol=True, 
                                inc_geo_code=False
                            )
                            if not regional_data.empty:
                                results[f'{category_name}_regions'] = regional_data
                                top_regions = regional_data[top_keyword].sort_values(ascending=False).head(5)
                                print(f"    âœ“ Top regions for '{top_keyword}':")
                                for region, score in top_regions.items():
                                    print(f"      - {region}: {score:.0f}")
                    except Exception as e:
                        print(f"    âš  Regional data error: {str(e)[:100]}")
                
            except Exception as e:
                print(f"    âœ— ERROR in {category_name}: {str(e)[:100]}")
                continue
            
            print("-" * 60)
        
        # Print overall summary
        print(f"\n[ANALYSIS SUMMARY]")
        print("="*80)
        
        total_keywords_analyzed = sum(s.get('data_collected', 0) for s in category_summaries.values())
        total_trending_up = sum(s.get('trending_up', 0) for s in category_summaries.values())
        total_trending_down = sum(s.get('trending_down', 0) for s in category_summaries.values())
        
        print(f"  ğŸ“Š Overall Statistics:")
        print(f"     â€¢ Categories processed: {len(category_summaries)}")
        print(f"     â€¢ Total keywords analyzed: {total_keywords_analyzed}")
        print(f"     â€¢ Keywords trending up: {total_trending_up}")
        print(f"     â€¢ Keywords trending down: {total_trending_down}")
        
        # Find most significant trends across all categories
        all_trends = []
        for cat_name, cat_summary in category_summaries.items():
            for trend in cat_summary.get('top_gainers', []):
                trend['category'] = cat_name
                all_trends.append(trend)
        
        if all_trends:
            top_overall = sorted(all_trends, key=lambda x: x['trend_pct'], reverse=True)[:5]
            print(f"\n  ğŸš€ Top 5 Rising Trends Across All Categories:")
            for i, trend in enumerate(top_overall, 1):
                print(f"     {i}. {trend['keyword']} ({trend['category']}): +{trend['trend_pct']:.1f}%")
        
        results['category_summaries'] = category_summaries
        
        return results
    
    def analyze_search_patterns(self):
        """Analyze specific search patterns and combinations"""
        print("\n[SEARCH PATTERNS] Analyzing consumer search patterns...")
        print("="*80)
        
        pattern_keywords = {
            'claims_issues': ['insurance claim denied', 'insurance won\'t pay', 'sue insurance company'],
            'cost_concerns': ['cheap insurance', 'insurance too expensive', 'cancel insurance'],
            'switching': ['switch insurance', 'change insurance company', 'better insurance'],
            'legal_action': ['insurance lawyer', 'insurance lawsuit', 'insurance attorney'],
            'complaints': ['insurance complaint', 'insurance scam', 'insurance fraud']
        }
        
        pattern_results = {}
        
        for pattern_name, keywords in pattern_keywords.items():
            print(f"\n  Analyzing pattern: {pattern_name}")
            
            if self.safe_build_payload(keywords[:5]):  # Max 5 keywords
                try:
                    data = self.pytrends.interest_over_time()
                    if not data.empty:
                        if 'isPartial' in data.columns:
                            data = data.drop('isPartial', axis=1)
                        
                        pattern_results[pattern_name] = data
                        
                        # Calculate pattern intensity
                        pattern_avg = data.mean().mean()
                        recent_pattern_avg = data.iloc[-30:].mean().mean() if len(data) >= 30 else pattern_avg
                        
                        intensity_change = ((recent_pattern_avg - pattern_avg) / pattern_avg * 100) if pattern_avg > 0 else 0
                        
                        print(f"    â€¢ Pattern intensity: {recent_pattern_avg:.1f}")
                        print(f"    â€¢ Change: {intensity_change:+.1f}%")
                        
                        if intensity_change > 20:
                            print(f"    âš ï¸� SIGNIFICANT INCREASE in {pattern_name} searches!")
                        
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"    âš  Error: {str(e)[:100]}")
        
        return pattern_results
    
    def get_rising_queries_expanded(self):
        """Get rising queries for multiple insurance-related terms"""
        print("\n[RISING QUERIES] Analyzing rising insurance searches across categories...")
        print("="*80)
        
        search_terms = ['insurance', 'insurance claim', 'insurance lawsuit', 'cheap insurance', 
                       'insurance denied', 'insurance company']
        
        all_rising = {}
        
        for term in search_terms:
            print(f"\n  Analyzing rising queries for: '{term}'")
            
            try:
                if self.safe_build_payload([term], timeframe='today 3-m', geo='US'):
                    related = self.pytrends.related_queries()
                    
                    if related and term in related:
                        rising_queries = []
                        top_queries = []
                        
                        if 'rising' in related[term] and related[term]['rising'] is not None:
                            rising_df = related[term]['rising']
                            if not rising_df.empty:
                                for _, row in rising_df.head(10).iterrows():
                                    value_str = str(row['value'])
                                    rising_queries.append({
                                        'query': row['query'],
                                        'value': value_str,
                                        'is_breakout': value_str == 'Breakout'
                                    })
                        
                        if 'top' in related[term] and related[term]['top'] is not None:
                            top_df = related[term]['top']
                            if not top_df.empty:
                                for _, row in top_df.head(5).iterrows():
                                    top_queries.append({
                                        'query': row['query'],
                                        'value': row['value']
                                    })
                        
                        if rising_queries or top_queries:
                            all_rising[term] = {
                                'rising': rising_queries,
                                'top': top_queries
                            }
                            
                            if rising_queries:
                                print(f"    âœ“ Found {len(rising_queries)} rising queries")
                                breakouts = [q for q in rising_queries if q['is_breakout']]
                                if breakouts:
                                    print(f"    ğŸš€ {len(breakouts)} BREAKOUT queries!")
                    
                    time.sleep(2)
                    
            except Exception as e:
                print(f"    âš  Error: {str(e)[:100]}")
        
        return all_rising

class EnhancedNewsAnalyzer:
    """Enhanced news analyzer with deeper sentiment and topic analysis"""
    
    def __init__(self):
        print("\n[INIT] Initializing Enhanced News Analyzer...")
        self.vader = SentimentIntensityAnalyzer()
        self.news_sources = [
            {'name': 'Insurance Business Magazine', 'url': 'https://www.insurancebusinessmag.com/us/rss/'},
            {'name': 'Insurance Journal', 'url': 'https://www.insurancejournal.com/rss/'},
            {'name': 'Google News - Insurance', 'url': 'https://news.google.com/rss/search?q=insurance+industry+news&hl=en-US&gl=US&ceid=US:en'},
            {'name': 'Google News - Insurance Lawsuit', 'url': 'https://news.google.com/rss/search?q=insurance+lawsuit+denied+claim&hl=en-US&gl=US&ceid=US:en'},
            {'name': 'Google News - Insurance Fraud', 'url': 'https://news.google.com/rss/search?q=insurance+fraud+investigation&hl=en-US&gl=US&ceid=US:en'},
            {'name': 'Reuters Business', 'url': 'https://feeds.reuters.com/reuters/businessNews'},
        ]
        print(f"  âœ“ News Analyzer initialized with {len(self.news_sources)} sources")
    
    def fetch_all_news(self):
        """Fetch and analyze news with enhanced categorization"""
        print("\n[NEWS ANALYSIS] Starting enhanced news intelligence gathering...")
        print("="*80)
        
        all_articles = []
        source_stats = {}
        
        for source_info in self.news_sources:
            source_name = source_info['name']
            source_url = source_info['url']
            print(f"\n  ğŸ“° Fetching from {source_name}...")
            
            articles = self.fetch_rss_feed(source_url, source_name)
            all_articles.extend(articles)
            source_stats[source_name] = len(articles)
            
            print(f"     âœ“ Retrieved {len(articles)} articles")
        
        print(f"\n[NEWS SUMMARY] Total articles collected: {len(all_articles)}")
        
        if all_articles:
            df = pd.DataFrame(all_articles)
            
            # Enhanced analysis
            analysis_results = self.perform_enhanced_analysis(df)
            
            return {
                'articles': all_articles,
                'summary': analysis_results,
                'dataframe': df
            }
        
        return {'articles': [], 'summary': {}}
    
    def perform_enhanced_analysis(self, df):
        """Perform enhanced analysis on news articles"""
        
        # Sentiment analysis
        sentiment_dist = df['sentiment_label'].value_counts().to_dict()
        
        print(f"\n  ğŸ˜Š SENTIMENT ANALYSIS:")
        print("  " + "="*50)
        
        total_articles = len(df)
        for label, count in sentiment_dist.items():
            percentage = (count / total_articles) * 100
            emoji = "ğŸ˜Š" if label == 'positive' else "ğŸ˜Ÿ" if label == 'negative' else "ğŸ˜�"
            print(f"    {emoji} {label.upper()}: {count} articles ({percentage:.1f}%)")
        
        avg_sentiment = df['sentiment_score'].mean()
        
        # Topic categorization
        topics = {
            'legal': ['lawsuit', 'sue', 'court', 'litigation', 'attorney', 'lawyer', 'legal'],
            'claims': ['claim', 'denied', 'dispute', 'rejected', 'appeal', 'underpaid'],
            'fraud': ['fraud', 'scam', 'investigation', 'illegal', 'scheme', 'fake'],
            'regulation': ['regulation', 'commissioner', 'compliance', 'fine', 'penalty', 'violation'],
            'cost': ['rate', 'premium', 'price', 'cost', 'expensive', 'increase', 'hike'],
            'technology': ['AI', 'digital', 'app', 'online', 'cyber', 'data', 'hack'],
            'disaster': ['hurricane', 'flood', 'fire', 'earthquake', 'storm', 'disaster', 'catastrophe']
        }
        
        topic_counts = {}
        for topic, keywords in topics.items():
            count = 0
            for _, article in df.iterrows():
                text = (article['title'] + ' ' + article.get('summary', '')).lower()
                if any(keyword in text for keyword in keywords):
                    count += 1
            topic_counts[topic] = count
        
        print(f"\n  ğŸ“‘ TOPIC DISTRIBUTION:")
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        for topic, count in sorted_topics:
            if count > 0:
                percentage = (count / total_articles) * 100
                print(f"    â€¢ {topic.upper()}: {count} articles ({percentage:.1f}%)")
        
        # Keyword frequency
        print(f"\n  ğŸ”¤ KEYWORD FREQUENCY ANALYSIS:")
        all_text = ' '.join(df['title'].tolist()).lower()
        
        keyword_counts = {}
        for keyword in ANALYSIS_KEYWORDS:
            count = all_text.count(keyword)
            if count > 0:
                keyword_counts[keyword] = count
        
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        for keyword, count in sorted_keywords[:15]:
            print(f"    â€¢ '{keyword}': {count} mentions")
        
        # Company mentions
        company_mentions = {}
        companies = ['geico', 'progressive', 'state farm', 'allstate', 'liberty mutual',
                    'usaa', 'farmers', 'nationwide', 'travelers', 'metlife']
        
        for company in companies:
            count = all_text.count(company.lower())
            if count > 0:
                company_mentions[company] = count
        
        if company_mentions:
            print(f"\n  ğŸ�¢ COMPANY MENTIONS:")
            for company, count in sorted(company_mentions.items(), key=lambda x: x[1], reverse=True):
                print(f"    â€¢ {company.title()}: {count} mentions")
        
        return {
            'total_count': total_articles,
            'sentiment_distribution': sentiment_dist,
            'average_sentiment': float(avg_sentiment),
            'topic_distribution': dict(sorted_topics),
            'keyword_frequency': dict(sorted_keywords[:20]),
            'company_mentions': company_mentions,
            'sources': {k: v for k, v in df['source'].value_counts().to_dict().items()}
        }
    
    def fetch_rss_feed(self, url, source_name):
        """Fetch and parse RSS feed with enhanced analysis"""
        articles = []
        
        try:
            feed = feedparser.parse(url, agent='Mozilla/5.0')
            
            if not feed.entries:
                return articles
            
            cutoff_date = datetime.now() - timedelta(days=30)
            
            for entry in feed.entries[:75]:  # Increased limit
                try:
                    # Extract date
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        pub_date = datetime(*entry.updated_parsed[:6])
                    else:
                        pub_date = datetime.now()
                    
                    if pub_date < cutoff_date:
                        continue
                    
                    title = entry.get('title', 'No title')
                    summary = entry.get('summary', entry.get('description', ''))
                    
                    if summary:
                        soup = BeautifulSoup(summary, 'html.parser')
                        summary = soup.get_text()[:500]
                    
                    # Enhanced sentiment analysis
                    combined_text = f"{title} {summary}"
                    sentiment = self.analyze_sentiment(combined_text)
                    
                    # Detect critical keywords
                    critical_keywords = []
                    for keyword in ['lawsuit', 'denied', 'fraud', 'investigation', 'complaint']:
                        if keyword in combined_text.lower():
                            critical_keywords.append(keyword)
                    
                    article = {
                        'title': title,
                        'summary': summary,
                        'link': entry.get('link', ''),
                        'published': pub_date.isoformat(),
                        'source': source_name,
                        'sentiment_score': sentiment['score'],
                        'sentiment_label': sentiment['label'],
                        'vader_scores': sentiment['vader'],
                        'critical_keywords': critical_keywords
                    }
                    
                    articles.append(article)
                
                except Exception as e:
                    continue
            
        except Exception as e:
            print(f"       âœ— Error fetching feed: {str(e)[:100]}")
        
        return articles
    
    def analyze_sentiment(self, text):
        """Analyze sentiment with multiple methods"""
        try:
            vader_scores = self.vader.polarity_scores(text)
            
            try:
                blob = TextBlob(text)
                blob_polarity = blob.sentiment.polarity
                combined_score = (vader_scores['compound'] + blob_polarity) / 2
            except:
                combined_score = vader_scores['compound']
            
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

class ComprehensiveVisualizer:
    """Comprehensive visualization engine with multiple dashboards"""
    
    def __init__(self):
        print("\n[INIT] Initializing Comprehensive Visualizer...")
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD',
                      '#74B9FF', '#A29BFE', '#FD79A8', '#FDCB6E', '#6C5CE7', '#00B894']
        
        self.output_dir = 'insurance_analysis_output_expanded'
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'charts'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'data'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'reports'), exist_ok=True)
        
        print(f"  âœ“ Visualizer initialized")
        print(f"  ğŸ“� Output directory: {self.output_dir}")
    
    def create_master_dashboard(self, trends_data, news_data, rising_data, pattern_data):
        """Create master dashboard with all insights"""
        print("\n[VISUALIZATION] Creating master intelligence dashboard...")
        print("="*80)
        
        try:
            # Create large figure with multiple subplots
            fig = plt.figure(figsize=(24, 16))
            fig.suptitle('Insurance Industry Intelligence Master Dashboard', fontsize=20, fontweight='bold')
            
            gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
            
            # Multiple visualizations
            ax1 = fig.add_subplot(gs[0, :2])
            self._plot_category_comparison(ax1, trends_data)
            
            ax2 = fig.add_subplot(gs[0, 2:])
            self._plot_trend_heatmap(ax2, trends_data)
            
            ax3 = fig.add_subplot(gs[1, :2])
            self._plot_legal_issues_trends(ax3, trends_data)
            
            ax4 = fig.add_subplot(gs[1, 2])
            self._plot_news_topics(ax4, news_data)
            
            ax5 = fig.add_subplot(gs[1, 3])
            self._plot_sentiment_gauge(ax5, news_data)
            
            ax6 = fig.add_subplot(gs[2, :2])
            self._plot_pattern_analysis(ax6, pattern_data)
            
            ax7 = fig.add_subplot(gs[2, 2:])
            self._plot_rising_queries_chart(ax7, rising_data)
            
            ax8 = fig.add_subplot(gs[3, :])
            self._add_insights_summary(ax8, trends_data, news_data, rising_data)
            
            plt.tight_layout()
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(self.output_dir, 'charts', f'master_dashboard_{timestamp}.png')
            plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"  ğŸ’¾ Master dashboard saved: {filename}")
            
            plt.show()
            
            return filename
            
        except Exception as e:
            print(f"  âœ— Error creating dashboard: {str(e)[:200]}")
            return None
    
    def _plot_category_comparison(self, ax, trends_data):
        """Plot category comparison"""
        try:
            if 'category_summaries' in trends_data:
                summaries = trends_data['category_summaries']
                
                categories = list(summaries.keys())
                trending_up = [summaries[cat].get('trending_up', 0) for cat in categories]
                trending_down = [summaries[cat].get('trending_down', 0) for cat in categories]
                
                x = np.arange(len(categories))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, trending_up, width, label='Trending Up', color='#2ECC71')
                bars2 = ax.bar(x + width/2, trending_down, width, label='Trending Down', color='#E74C3C')
                
                ax.set_xlabel('Category')
                ax.set_ylabel('Number of Keywords')
                ax.set_title('Trend Direction by Category', fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels([cat.replace('_', ' ').title()[:15] for cat in categories], 
                                  rotation=45, ha='right')
                ax.legend()
                
                # Add value labels
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:
                            ax.annotate(f'{int(height)}',
                                      xy=(bar.get_x() + bar.get_width() / 2, height),
                                      xytext=(0, 3),
                                      textcoords="offset points",
                                      ha='center', va='bottom', fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No category data available', ha='center', va='center')
            
            ax.set_title('Trend Direction by Category', fontweight='bold')
            
        except Exception as e:
            ax.text(0.5, 0.5, 'Error loading data', ha='center', va='center')
    
    def _plot_trend_heatmap(self, ax, trends_data):
        """Create trend intensity heatmap"""
        try:
            # Create sample heatmap data from available trends
            categories = []
            trend_values = []
            
            for key in trends_data:
                if 'timeline' in key and isinstance(trends_data[key], pd.DataFrame):
                    category = key.replace('_timeline', '')
                    categories.append(category)
                    
                    # Calculate average recent trend
                    data = trends_data[key]
                    if len(data) >= 30:
                        recent_avg = data.iloc[-30:].mean().mean()
                    else:
                        recent_avg = data.mean().mean()
                    trend_values.append(recent_avg)
            
            if categories and trend_values:
                # Create heatmap
                heatmap_data = np.array(trend_values).reshape(-1, 1)
                im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
                
                ax.set_yticks(np.arange(len(categories)))
                ax.set_yticklabels([cat.replace('_', ' ').title() for cat in categories])
                ax.set_xticks([0])
                ax.set_xticklabels(['Trend Intensity'])
                
                # Add colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                # Add text annotations
                for i, val in enumerate(trend_values):
                    ax.text(0, i, f'{val:.1f}', ha='center', va='center', color='white' if val < 50 else 'black')
            else:
                ax.text(0.5, 0.5, 'No trend data available', ha='center', va='center')
            
            ax.set_title('Search Intensity Heatmap', fontweight='bold')
            
        except Exception as e:
            ax.text(0.5, 0.5, 'Error creating heatmap', ha='center', va='center')
    
    def _plot_legal_issues_trends(self, ax, trends_data):
        """Plot legal issues specific trends"""
        try:
            legal_data = trends_data.get('legal_issues_timeline')
            
            if legal_data is not None and not legal_data.empty:
                # Plot top 5 legal keywords
                cols_to_plot = [col for col in legal_data.columns[:5] 
                              if legal_data[col].dtype in ['float64', 'int64']]
                
                for i, col in enumerate(cols_to_plot):
                    ax.plot(legal_data.index, legal_data[col], 
                           label=col[:30], linewidth=2, 
                           color=self.colors[i % len(self.colors)], alpha=0.8)
                
                ax.set_title('Legal Issues & Disputes Trends', fontweight='bold', color='#8B0000')
                ax.set_xlabel('Date')
                ax.set_ylabel('Search Interest')
                ax.legend(loc='best', fontsize=8)
                ax.grid(True, alpha=0.3)
                
                # Highlight if there's a spike
                for col in cols_to_plot:
                    if legal_data[col].max() > legal_data[col].mean() * 2:
                        max_idx = legal_data[col].idxmax()
                        ax.axvline(x=max_idx, color='red', linestyle='--', alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No legal trends data', ha='center', va='center')
                ax.set_title('Legal Issues Trends', fontweight='bold')
            
        except Exception as e:
            ax.text(0.5, 0.5, 'Error loading legal trends', ha='center', va='center')
    
    def _plot_news_topics(self, ax, news_data):
        """Plot news topic distribution"""
        try:
            if news_data and 'summary' in news_data and 'topic_distribution' in news_data['summary']:
                topics = news_data['summary']['topic_distribution']
                
                if topics:
                    # Filter non-zero topics
                    filtered_topics = {k: v for k, v in topics.items() if v > 0}
                    
                    if filtered_topics:
                        labels = list(filtered_topics.keys())
                        sizes = list(filtered_topics.values())
                        
                        colors = self.colors[:len(labels)]
                        
                        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                                          autopct='%1.0f%%', startangle=90)
                        
                        for autotext in autotexts:
                            autotext.set_color('white')
                            autotext.set_fontweight('bold')
                            autotext.set_fontsize(9)
                        
                        ax.set_title('News Topics Distribution', fontweight='bold')
                    else:
                        ax.text(0.5, 0.5, 'No topic data', ha='center', va='center')
                else:
                    ax.text(0.5, 0.5, 'No topics found', ha='center', va='center')
            else:
                ax.text(0.5, 0.5, 'No news topic data', ha='center', va='center')
                
            ax.set_title('News Topics', fontweight='bold')
            
        except Exception as e:
            ax.text(0.5, 0.5, 'Error loading topics', ha='center', va='center')
    
    def _plot_sentiment_gauge(self, ax, news_data):
        """Create sentiment gauge visualization"""
        try:
            if news_data and 'summary' in news_data:
                avg_sentiment = news_data['summary'].get('average_sentiment', 0)
                
                # Create gauge
                theta = np.linspace(0, np.pi, 100)
                r = 1
                
                # Background arc
                ax.plot(r * np.cos(theta), r * np.sin(theta), 'k-', linewidth=2)
                
                # Color zones
                colors_zones = ['#E74C3C', '#F39C12', '#F1C40F', '#2ECC71']
                zone_ranges = [(-1, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1)]
                
                for color, (start, end) in zip(colors_zones, zone_ranges):
                    if start <= avg_sentiment <= end:
                        current_color = color
                    theta_zone = np.linspace(np.pi * (1 - (end + 1) / 2), 
                                           np.pi * (1 - (start + 1) / 2), 50)
                    ax.fill_between(r * np.cos(theta_zone), 0, r * np.sin(theta_zone), 
                                  color=color, alpha=0.3)
                
                # Needle
                angle = np.pi * (1 - (avg_sentiment + 1) / 2)
                ax.arrow(0, 0, 0.8 * np.cos(angle), 0.8 * np.sin(angle),
                        head_width=0.05, head_length=0.05, fc='black', ec='black')
                
                # Center dot
                ax.plot(0, 0, 'ko', markersize=10)
                
                # Labels
                ax.text(0, -0.3, f'Sentiment: {avg_sentiment:.3f}', 
                       ha='center', fontsize=11, fontweight='bold')
                ax.text(-1.1, 0, 'Negative', ha='center', fontsize=9)
                ax.text(1.1, 0, 'Positive', ha='center', fontsize=9)
                ax.text(0, 1.1, 'Neutral', ha='center', fontsize=9)
                
                ax.set_xlim(-1.3, 1.3)
                ax.set_ylim(-0.5, 1.3)
                ax.axis('off')
                ax.set_title('News Sentiment Gauge', fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No sentiment data', ha='center', va='center')
                ax.axis('off')
                
        except Exception as e:
            ax.text(0.5, 0.5, 'Error creating gauge', ha='center', va='center')
            ax.axis('off')
    
    def _plot_pattern_analysis(self, ax, pattern_data):
        """Plot search pattern analysis"""
        try:
            if pattern_data:
                patterns = []
                intensities = []
                
                for pattern_name, data in pattern_data.items():
                    if isinstance(data, pd.DataFrame) and not data.empty:
                        patterns.append(pattern_name.replace('_', ' ').title())
                        intensities.append(data.mean().mean())
                
                if patterns:
                    bars = ax.bar(patterns, intensities, color=self.colors[:len(patterns)])
                    ax.set_xlabel('Search Pattern')
                    ax.set_ylabel('Average Intensity')
                    ax.set_title('Consumer Search Patterns Analysis', fontweight='bold')
                    ax.tick_params(axis='x', rotation=45)
                    
                    # Add value labels
                    for bar, val in zip(bars, intensities):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                               f'{val:.1f}', ha='center', va='bottom', fontsize=9)
                else:
                    ax.text(0.5, 0.5, 'No pattern data available', ha='center', va='center')
            else:
                ax.text(0.5, 0.5, 'No pattern analysis data', ha='center', va='center')
                
            ax.set_title('Search Patterns', fontweight='bold')
            
        except Exception as e:
            ax.text(0.5, 0.5, 'Error loading patterns', ha='center', va='center')
    
    def _plot_rising_queries_chart(self, ax, rising_data):
        """Plot rising queries visualization"""
        try:
            if rising_data:
                all_rising = []
                
                for term, data in rising_data.items():
                    if 'rising' in data and data['rising']:
                        for query in data['rising'][:3]:  # Top 3 from each
                            all_rising.append({
                                'query': query['query'][:30],
                                'category': term,
                                'is_breakout': query['is_breakout']
                            })
                
                if all_rising:
                    # Create bar chart
                    queries = [q['query'] for q in all_rising[:10]]
                    categories = [q['category'] for q in all_rising[:10]]
                    colors_list = ['red' if q['is_breakout'] else 'blue' for q in all_rising[:10]]
                    
                    y_pos = np.arange(len(queries))
                    ax.barh(y_pos, [1] * len(queries), color=colors_list, alpha=0.7)
                    
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(queries, fontsize=8)
                    ax.set_xlabel('Rising Query Status')
                    ax.set_title('Top Rising Search Queries', fontweight='bold')
                    
                    # Add legend
                    red_patch = plt.Rectangle((0, 0), 1, 1, fc="red", alpha=0.7)
                    blue_patch = plt.Rectangle((0, 0), 1, 1, fc="blue", alpha=0.7)
                    ax.legend([red_patch, blue_patch], ['Breakout', 'Rising'], 
                            loc='lower right', fontsize=8)
                else:
                    ax.text(0.5, 0.5, 'No rising queries found', ha='center', va='center')
            else:
                ax.text(0.5, 0.5, 'No rising queries data', ha='center', va='center')
                
            ax.set_title('Rising Queries', fontweight='bold')
            
        except Exception as e:
            ax.text(0.5, 0.5, 'Error loading queries', ha='center', va='center')
    
    def _add_insights_summary(self, ax, trends_data, news_data, rising_data):
        """Add comprehensive insights summary"""
        try:
            insights = ["EXECUTIVE INSIGHTS SUMMARY", "=" * 100, ""]
            
            insights.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            insights.append("")
            
            # Key findings
            insights.append("KEY FINDINGS:")
            insights.append("-" * 50)
            
            if 'category_summaries' in trends_data:
                summaries = trends_data['category_summaries']
                
                # Find most significant trends
                all_gainers = []
                for cat, summary in summaries.items():
                    for gainer in summary.get('top_gainers', []):
                        gainer['category'] = cat
                        all_gainers.append(gainer)
                
                if all_gainers:
                    top_trend = max(all_gainers, key=lambda x: x['trend_pct'])
                    insights.append(f"â€¢ HIGHEST GROWTH: {top_trend['keyword']} (+{top_trend['trend_pct']:.1f}%)")
                
                # Categories with most activity
                most_active = max(summaries.items(), 
                                key=lambda x: x[1].get('trending_up', 0) + x[1].get('trending_down', 0))
                insights.append(f"â€¢ MOST ACTIVE CATEGORY: {most_active[0].replace('_', ' ').title()}")
            
            if news_data and 'summary' in news_data:
                sentiment = news_data['summary'].get('average_sentiment', 0)
                sentiment_label = "POSITIVE" if sentiment > 0.05 else "NEGATIVE" if sentiment < -0.05 else "NEUTRAL"
                insights.append(f"â€¢ NEWS SENTIMENT: {sentiment_label} ({sentiment:.3f})")
                
                # Most discussed topic
                topics = news_data['summary'].get('topic_distribution', {})
                if topics:
                    top_topic = max(topics.items(), key=lambda x: x[1])
                    insights.append(f"â€¢ TOP NEWS TOPIC: {top_topic[0].upper()} ({top_topic[1]} articles)")
            
            insights.append("")
            insights.append("RISK INDICATORS:")
            insights.append("-" * 50)
            
            # Check for concerning trends
            risk_indicators = []
            
            if 'legal_issues_timeline' in trends_data:
                legal_data = trends_data['legal_issues_timeline']
                if not legal_data.empty:
                    recent_legal = legal_data.iloc[-30:].mean().mean() if len(legal_data) >= 30 else legal_data.mean().mean()
                    if recent_legal > 50:
                        risk_indicators.append(f"âš ï¸� HIGH legal search activity (index: {recent_legal:.1f})")
            
            if 'claims_disputes_timeline' in trends_data:
                claims_data = trends_data['claims_disputes_timeline']
                if not claims_data.empty:
                    recent_claims = claims_data.iloc[-30:].mean().mean() if len(claims_data) >= 30 else claims_data.mean().mean()
                    if recent_claims > 40:
                        risk_indicators.append(f"âš ï¸� ELEVATED claims dispute searches (index: {recent_claims:.1f})")
            
            if risk_indicators:
                for indicator in risk_indicators:
                    insights.append(f"â€¢ {indicator}")
            else:
                insights.append("â€¢ No significant risk indicators detected")
            
            insights.append("")
            insights.append("RECOMMENDATIONS:")
            insights.append("-" * 50)
            insights.append("â€¢ Monitor rising legal and claims-related searches for early warning signs")
            insights.append("â€¢ Track sentiment shifts in news coverage for reputation management")
            insights.append("â€¢ Analyze geographic variations for targeted market strategies")
            
            # Display insights
            insights_text = "\n".join(insights)
            ax.text(0.05, 0.95, insights_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.1))
            ax.axis('off')
            
        except Exception as e:
            ax.text(0.5, 0.5, 'Error generating insights', ha='center', va='center')
            ax.axis('off')
    
    def create_interactive_analysis(self, trends_data, news_data):
        """Create comprehensive interactive visualizations"""
        print("\n  ğŸ“Š Creating interactive analysis dashboard...")
        
        try:
            # Create subplots with different types
            fig = make_subplots(
                rows=3, cols=3,
                subplot_titles=(
                    'Search Trends Over Time', 'Category Comparison', 'Regional Heatmap',
                    'Legal Issues Focus', 'Claims & Disputes', 'Coverage Levels',
                    'News Sentiment Timeline', 'Rising Queries', 'Pattern Analysis'
                ),
                specs=[
                    [{"secondary_y": False}, {"type": "bar"}, {"type": "geo"}],
                    [{"secondary_y": False}, {"secondary_y": False}, {"type": "bar"}],
                    [{"secondary_y": False}, {"type": "bar"}, {"type": "scatter"}]
                ],
                horizontal_spacing=0.1,
                vertical_spacing=0.1
            )
            
            # Add various traces based on available data
            trace_count = 0
            
            # Plot 1: Main trends
            for key in trends_data:
                if 'timeline' in key and isinstance(trends_data[key], pd.DataFrame):
                    data = trends_data[key]
                    for col in data.columns[:2]:  # Limit for clarity
                        if data[col].dtype in ['float64', 'int64']:
                            fig.add_trace(
                                go.Scatter(x=data.index, y=data[col], 
                                         name=f"{key.split('_')[0][:10]}-{col[:15]}", 
                                         mode='lines'),
                                row=1, col=1
                            )
                            trace_count += 1
                            if trace_count >= 5:
                                break
                    if trace_count >= 5:
                        break
            
            # Update layout
            fig.update_layout(
                height=1200,
                showlegend=True,
                title_text="Insurance Industry Comprehensive Analysis - Interactive Dashboard",
                title_font_size=20
            )
            
            # Save
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(self.output_dir, 'charts', f'interactive_analysis_{timestamp}.html')
            fig.write_html(filename)
            print(f"    ğŸ’¾ Interactive analysis saved: {filename}")
            
            return filename
            
        except Exception as e:
            print(f"    âš  Error creating interactive analysis: {str(e)[:100]}")
            return None
    
    def export_comprehensive_results(self, trends_data, news_data, rising_data, pattern_data):
        """Export comprehensive results with all data"""
        print("\n[EXPORT] Saving comprehensive analysis results...")
        
        try:
            export_data = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'report_type': 'Insurance Industry Comprehensive Intelligence Analysis',
                    'version': '3.0',
                    'total_keywords_analyzed': sum(len(v) for v in INSURANCE_KEYWORDS.values())
                },
                'trends': {},
                'news': {},
                'rising_queries': rising_data if rising_data else {},
                'search_patterns': {},
                'insights': {}
            }
            
            # Process trends data
            for key, value in trends_data.items():
                try:
                    if isinstance(value, pd.DataFrame):
                        export_data['trends'][key] = {
                            'data': value.to_dict(),
                            'shape': value.shape,
                            'columns': value.columns.tolist(),
                            'summary_stats': {
                                'mean': value.mean().to_dict() if not value.empty else {},
                                'max': value.max().to_dict() if not value.empty else {},
                                'min': value.min().to_dict() if not value.empty else {}
                            }
                        }
                    else:
                        export_data['trends'][key] = value
                except Exception as e:
                    export_data['trends'][key] = {'error': str(e)}
            
            # Process pattern data
            if pattern_data:
                for pattern_name, data in pattern_data.items():
                    if isinstance(data, pd.DataFrame):
                        export_data['search_patterns'][pattern_name] = {
                            'average_intensity': data.mean().mean(),
                            'peak_intensity': data.max().max(),
                            'trend': 'increasing' if data.iloc[-1].mean() > data.iloc[0].mean() else 'decreasing'
                        }
            
            # Add insights
            if 'category_summaries' in trends_data:
                export_data['insights'] = trends_data['category_summaries']
            
            # Process news data
            if news_data:
                export_data['news'] = {
                    'summary': news_data.get('summary', {}),
                    'article_count': len(news_data.get('articles', [])),
                    'articles_sample': news_data.get('articles', [])[:20]
                }
            
            # Save main results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # JSON export
            json_filename = os.path.join(self.output_dir, 'data', f'comprehensive_results_{timestamp}.json')
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str, ensure_ascii=False)
            print(f"  ğŸ’¾ JSON results saved: {json_filename}")
            
            # Create summary report
            report_filename = self.create_text_report(export_data, timestamp)
            
            return {
                'json': json_filename,
                'report': report_filename
            }
            
        except Exception as e:
            print(f"  âœ— Error exporting results: {str(e)[:100]}")
            return None
    
    def create_text_report(self, data, timestamp):
        """Create detailed text report"""
        try:
            report_lines = []
            report_lines.append("="*80)
            report_lines.append("INSURANCE INDUSTRY COMPREHENSIVE INTELLIGENCE REPORT")
            report_lines.append("="*80)
            report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # Executive Summary
            report_lines.append("EXECUTIVE SUMMARY")
            report_lines.append("-"*40)
            report_lines.append(f"Total Keywords Analyzed: {data['metadata'].get('total_keywords_analyzed', 0)}")
            report_lines.append(f"Categories Covered: {len(INSURANCE_KEYWORDS)}")
            
            if 'insights' in data and data['insights']:
                total_trending_up = sum(cat.get('trending_up', 0) for cat in data['insights'].values())
                total_trending_down = sum(cat.get('trending_down', 0) for cat in data['insights'].values())
                report_lines.append(f"Keywords Trending Up: {total_trending_up}")
                report_lines.append(f"Keywords Trending Down: {total_trending_down}")
            
            report_lines.append("")
            
            # Key Findings
            report_lines.append("KEY FINDINGS BY CATEGORY")
            report_lines.append("-"*40)
            
            if 'insights' in data:
                for category, insights in data['insights'].items():
                    report_lines.append(f"\n{category.replace('_', ' ').upper()}:")
                    
                    if 'top_gainers' in insights and insights['top_gainers']:
                        report_lines.append("  Top Gainers:")
                        for gainer in insights['top_gainers'][:3]:
                            report_lines.append(f"    â€¢ {gainer['keyword']}: +{gainer['trend_pct']:.1f}%")
                    
                    if 'highest_interest' in insights and insights['highest_interest']:
                        report_lines.append("  Highest Current Interest:")
                        for item in insights['highest_interest'][:3]:
                            report_lines.append(f"    â€¢ {item['keyword']}: {item['current']:.1f}")
            
            # News Analysis
            if 'news' in data and 'summary' in data['news']:
                report_lines.append("\n\nNEWS ANALYSIS")
                report_lines.append("-"*40)
                summary = data['news']['summary']
                report_lines.append(f"Articles Analyzed: {summary.get('total_count', 0)}")
                report_lines.append(f"Average Sentiment: {summary.get('average_sentiment', 0):.3f}")
                
                if 'topic_distribution' in summary:
                    report_lines.append("\nTop News Topics:")
                    for topic, count in list(summary['topic_distribution'].items())[:5]:
                        report_lines.append(f"  â€¢ {topic}: {count} articles")
            
            # Rising Queries
            if 'rising_queries' in data and data['rising_queries']:
                report_lines.append("\n\nRISING SEARCH QUERIES")
                report_lines.append("-"*40)
                
                for term, queries in data['rising_queries'].items():
                    if 'rising' in queries and queries['rising']:
                        report_lines.append(f"\n{term}:")
                        for query in queries['rising'][:5]:
                            if query['is_breakout']:
                                report_lines.append(f"  â€¢ {query['query']} [BREAKOUT]")
                            else:
                                report_lines.append(f"  â€¢ {query['query']}: {query['value']}")
            
            # Risk Indicators
            report_lines.append("\n\nRISK INDICATORS & WARNINGS")
            report_lines.append("-"*40)
            
            # Check for concerning patterns
            risk_found = False
            
            if 'search_patterns' in data:
                for pattern, info in data['search_patterns'].items():
                    if 'legal' in pattern or 'claims' in pattern or 'complaint' in pattern:
                        if info.get('average_intensity', 0) > 50:
                            report_lines.append(f"âš ï¸� HIGH ACTIVITY: {pattern} (intensity: {info['average_intensity']:.1f})")
                            risk_found = True
            
            if not risk_found:
                report_lines.append("No significant risk indicators detected")
            
            # Recommendations
            report_lines.append("\n\nSTRATEGIC RECOMMENDATIONS")
            report_lines.append("-"*40)
            report_lines.append("1. Monitor trending legal and claims-related searches for early warning signs")
            report_lines.append("2. Track regional variations in search patterns for targeted strategies")
            report_lines.append("3. Analyze correlation between news sentiment and search behavior")
            report_lines.append("4. Focus on categories with highest growth for market opportunities")
            report_lines.append("5. Address concerns reflected in rising complaint-related searches")
            
            # Save report
            report_text = "\n".join(report_lines)
            report_filename = os.path.join(self.output_dir, 'reports', f'analysis_report_{timestamp}.txt')
            
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            print(f"  ğŸ’¾ Text report saved: {report_filename}")
            
            return report_filename
            
        except Exception as e:
            print(f"  âš  Error creating report: {str(e)[:100]}")
            return None

def main():
    """Main execution function for comprehensive analysis"""
    print("\n" + "="*80)
    print("ğŸš€ STARTING COMPREHENSIVE INSURANCE INDUSTRY INTELLIGENCE ANALYSIS")
    print("="*80)
    print(f"Analysis initiated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Initialize components
    print("\n[INITIALIZATION] Setting up expanded analysis components...")
    trends_analyzer = ExpandedGoogleTrendsAnalyzer()
    news_analyzer = EnhancedNewsAnalyzer()
    visualizer = ComprehensiveVisualizer()
    
    # Store all results
    results = {
        'trends_data': {},
        'news_data': {},
        'rising_queries': {},
        'pattern_data': {},
        'files': {}
    }
    
    # Step 1: Comprehensive Google Trends Analysis
    print("\n" + "="*80)
    print("[STEP 1/6] COMPREHENSIVE GOOGLE TRENDS ANALYSIS")
    print("="*80)
    try:
        trends_data = trends_analyzer.fetch_comprehensive_trends()
        results['trends_data'] = trends_data
        print("âœ“ Comprehensive trends analysis completed")
    except Exception as e:
        print(f"âœ— Error in trends analysis: {str(e)[:200]}")
        trends_data = {}
    
    # Step 2: Search Pattern Analysis
    print("\n" + "="*80)
    print("[STEP 2/6] SEARCH PATTERN ANALYSIS")
    print("="*80)
    try:
        pattern_data = trends_analyzer.analyze_search_patterns()
        results['pattern_data'] = pattern_data
        print("âœ“ Search pattern analysis completed")
    except Exception as e:
        print(f"âœ— Error in pattern analysis: {str(e)[:200]}")
        pattern_data = {}
    
    # Step 3: Rising Queries Expanded Analysis
    print("\n" + "="*80)
    print("[STEP 3/6] EXPANDED RISING QUERIES ANALYSIS")
    print("="*80)
    try:
        rising_queries = trends_analyzer.get_rising_queries_expanded()
        results['rising_queries'] = rising_queries
        print("âœ“ Expanded rising queries analysis completed")
    except Exception as e:
        print(f"âœ— Error in rising queries: {str(e)[:200]}")
        rising_queries = {}
    
    # Step 4: Enhanced News Intelligence
    print("\n" + "="*80)
    print("[STEP 4/6] ENHANCED NEWS INTELLIGENCE GATHERING")
    print("="*80)
    try:
        news_data = news_analyzer.fetch_all_news()
        results['news_data'] = news_data
        print("âœ“ Enhanced news intelligence completed")
    except Exception as e:
        print(f"âœ— Error in news analysis: {str(e)[:200]}")
        news_data = {}
    
    # Step 5: Comprehensive Visualization
    print("\n" + "="*80)
    print("[STEP 5/6] CREATING COMPREHENSIVE VISUALIZATIONS")
    print("="*80)
    
    try:
        # Master dashboard
        dashboard_file = visualizer.create_master_dashboard(
            trends_data, news_data, rising_queries, pattern_data
        )
        results['files']['dashboard'] = dashboard_file
        
        # Interactive analysis
        interactive_file = visualizer.create_interactive_analysis(trends_data, news_data)
        results['files']['interactive'] = interactive_file
        
    except Exception as e:
        print(f"âœ— Visualization creation failed: {str(e)[:200]}")
    
    # Step 6: Export Comprehensive Results
    print("\n" + "="*80)
    print("[STEP 6/6] EXPORTING COMPREHENSIVE RESULTS")
    print("="*80)
    export_files = visualizer.export_comprehensive_results(
        trends_data, news_data, rising_queries, pattern_data
    )
    if export_files:
        results['files'].update(export_files)
    
    # Print final summary
    print("\n" + "="*80)
    print("âœ… COMPREHENSIVE ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"â�±ï¸� Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nğŸ“Š ANALYSIS SUMMARY:")
    print("-" * 40)
    
    # Calculate totals
    total_keywords = sum(len(v) for v in INSURANCE_KEYWORDS.values())
    categories_analyzed = len([k for k in trends_data.keys() if 'timeline' in k])
    
    print(f"â€¢ Total keywords in analysis: {total_keywords}")
    print(f"â€¢ Categories successfully analyzed: {categories_analyzed}/{len(INSURANCE_KEYWORDS)}")
    print(f"â€¢ News articles processed: {len(news_data.get('articles', [])) if news_data else 0}")
    print(f"â€¢ Search patterns analyzed: {len(pattern_data)}")
    print(f"â€¢ Rising query categories: {len(rising_queries)}")
    
    print("\nğŸ“� OUTPUT FILES:")
    print("-" * 40)
    for file_type, filename in results['files'].items():
        if filename:
            print(f"â€¢ {file_type.upper()}: {filename}")
    
    print("\n" + "="*80)
    print("ğŸ�‰ ALL PROCESSING COMPLETE!")
    print("ğŸ“ˆ Comprehensive insurance industry intelligence ready for analysis")
    print("="*80)
    
    return results

if __name__ == "__main__":
    try:
        results = main()
        print("\nâœ¨ Insurance Industry Comprehensive Intelligence Analysis completed successfully!")
    except Exception as e:
        print(f"\nâ�Œ CRITICAL ERROR: {str(e)}")
        print(traceback.format_exc())

