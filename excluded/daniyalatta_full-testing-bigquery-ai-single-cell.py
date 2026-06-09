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


pip install pytrends feedparser beautifulsoup4 requests pandas numpy matplotlib seaborn plotly wordcloud textblob vaderSentiment lxml


pip install --upgrade --force-reinstall urllib3==1.26.15



pip install --upgrade pytrends


pip install --upgrade requests


pip show urllib3


"""
Insurance Industry Intelligence Analyzer - Enhanced Version (2025 Update)
Combines Google Trends analysis, news intelligence, social media insights, and advanced visualizations
Updated for 2025 compatibility, added wordcloud, more news sources, and improved error handling
"""

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

# Import libraries
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

print("="*80)
print("INSURANCE INDUSTRY INTELLIGENCE ANALYZER - ENHANCED 2025 VERSION")
print("="*80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# Configuration
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

INSURANCE_KEYWORDS = {
    'general': ['car insurance', 'auto insurance', 'home insurance', 'life insurance', 'health insurance'],
    'cost': ['insurance quotes', 'cheap insurance', 'insurance rates', 'insurance cost', 'insurance premium'],
    'companies': ['Geico', 'Progressive', 'State Farm', 'Allstate', 'Liberty Mutual'],
    'claims': ['insurance claim', 'insurance denied', 'insurance complaint', 'insurance lawsuit', 'file claim'],
    'disasters': ['flood insurance', 'hurricane insurance', 'fire insurance', 'earthquake insurance', 'disaster insurance'],
    'emerging': ['cyber insurance', 'climate insurance', 'AI in insurance']
}

class EnhancedGoogleTrendsAnalyzer:
    """Enhanced Google Trends Analyzer with improved error handling and 2025 compatibility"""
    
    def __init__(self):
        print("\n[INIT] Initializing Enhanced Google Trends Analyzer...")
        try:
            # Initialize with modern urllib3 compatibility
            self.pytrends = TrendReq(
                hl='en-US', 
                tz=360, 
                timeout=(10, 25),
                retries=3,
                backoff_factor=0.2,
                requests_args={'verify': True}
            )
            self.vader = SentimentIntensityAnalyzer()
            print("  âœ“ Google Trends Analyzer initialized successfully")
        except Exception as e:
            print(f"  âœ— Failed to initialize: {e}")
            raise
    
    def safe_build_payload(self, keywords, timeframe='today 3-m', geo='US'):
        """Safely build payload with enhanced error handling"""
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
            print(f"    âš  Error building payload: {str(e)[:100]}")
            if 'method_whitelist' in str(e).lower():
                print("    â†’ Detected urllib3 compatibility issue. Ensure urllib3==1.26.15 is installed.")
            time.sleep(5)
            return False
    
    def fetch_insurance_trends(self):
        """Fetch insurance trends with enhanced processing"""
        print("\n[GOOGLE TRENDS] Starting comprehensive insurance industry trends analysis...")
        print("="*80)
        results = {}
        
        for category_name, keywords in INSURANCE_KEYWORDS.items():
            print(f"\n[{category_name.upper()}] Fetching trends...")
            print(f"  Search terms: {', '.join(keywords)}")
            print("-" * 60)
            
            try:
                # Process in batches of 5 (Google Trends limit)
                all_data = []
                for i in range(0, len(keywords), 5):
                    batch = keywords[i:i+5]
                    print(f"  â†’ Building payload for batch: {batch}")
                    
                    if self.safe_build_payload(batch):
                        print(f"  âœ“ Payload built successfully")
                        
                        # Fetch interest over time
                        print(f"  â†’ Fetching interest over time...")
                        try:
                            interest_over_time = self.pytrends.interest_over_time()
                            
                            if not interest_over_time.empty:
                                if 'isPartial' in interest_over_time.columns:
                                    interest_over_time = interest_over_time.drop('isPartial', axis=1)
                                
                                all_data.append(interest_over_time)
                                print(f"    âœ“ Retrieved {len(interest_over_time)} data points")
                                
                                # Analyze trends with enhanced metrics
                                for col in interest_over_time.columns:
                                    if interest_over_time[col].dtype in ['float64', 'int64']:
                                        recent_avg = interest_over_time[col].iloc[-30:].mean()
                                        total_avg = interest_over_time[col].mean()
                                        current_val = interest_over_time[col].iloc[-1]
                                        volatility = interest_over_time[col].std()
                                        
                                        trend_pct = ((recent_avg - total_avg) / total_avg * 100) if total_avg > 0 else 0
                                        trend = "ğŸ”º RISING" if trend_pct > 5 else "ğŸ”» FALLING" if trend_pct < -5 else "â�¡ï¸� STABLE"
                                        
                                        print(f"    â€¢ {col}: {trend} ({trend_pct:+.1f}%) | Current: {current_val:.1f} | Volatility: {volatility:.1f}")
                            else:
                                print(f"    âš  No data retrieved for batch")
                        except Exception as e:
                            print(f"    âš  Error fetching data: {str(e)[:100]}")
                    else:
                        print(f"    âš  Could not build payload for batch")
                    
                    time.sleep(3)
                
                # Combine all data for this category
                if all_data:
                    combined = pd.concat(all_data, axis=1)
                    combined = combined.loc[:, ~combined.columns.duplicated()]
                    results[f'{category_name}_timeline'] = combined
                
                # Fetch regional data for first keyword
                try:
                    print(f"\n  â†’ Fetching regional interest for {keywords[0]}...")
                    if self.safe_build_payload([keywords[0]], timeframe='today 3-m', geo='US'):
                        interest_by_region = self.pytrends.interest_by_region(
                            resolution='REGION', 
                            inc_low_vol=True, 
                            inc_geo_code=False
                        )
                        if not interest_by_region.empty:
                            results[f'{category_name}_regions'] = interest_by_region
                            top_regions = interest_by_region[keywords[0]].sort_values(ascending=False).head(5)
                            print(f"    âœ“ Top 5 regions by interest:")
                            for region, score in top_regions.items():
                                print(f"      - {region}: {score:.1f}")
                except Exception as e:
                    print(f"    âš  Could not fetch regional data: {str(e)[:100]}")
                
            except Exception as e:
                print(f"    âœ— ERROR fetching {category_name}: {str(e)[:100]}")
                continue
            
            print("-" * 60)
        
        print(f"\n[SUMMARY] Trends data collection complete:")
        print(f"  â€¢ Categories processed: {len([k for k in results.keys() if 'timeline' in k])}")
        print(f"  â€¢ Total data points: {sum([len(v) if isinstance(v, pd.DataFrame) else 0 for v in results.values()])}")
        
        return results
    
    def get_rising_queries(self):
        """Get rising queries with enhanced implementation"""
        print("\n[RISING QUERIES] Analyzing rising insurance-related searches...")
        print("="*80)
        rising_data = {}
        
        try:
            print("  â†’ Building payload for 'insurance' term...")
            if self.safe_build_payload(['insurance'], timeframe='today 3-m', geo='US'):
                print("  âœ“ Payload built")
                
                print("  â†’ Fetching related queries...")
                try:
                    related = self.pytrends.related_queries()
                    
                    if related and 'insurance' in related:
                        print("  âœ“ Related queries retrieved")
                        
                        # Process rising queries
                        if 'rising' in related['insurance'] and related['insurance']['rising'] is not None:
                            rising_df = related['insurance']['rising']
                            if not rising_df.empty:
                                rising_data['rising_queries'] = rising_df.to_dict('records')
                                print(f"\n  ğŸ“ˆ RISING QUERIES ({len(rising_df)} found):")
                                for idx, row in rising_df.head(10).iterrows():
                                    value_str = str(row['value'])
                                    if value_str == 'Breakout':
                                        value_display = "ğŸš€ BREAKOUT (>5000% increase)"
                                    else:
                                        value_display = f"{value_str}% increase"
                                    print(f"    {idx+1}. {row['query']}: {value_display}")
                        
                        # Process top queries
                        if 'top' in related['insurance'] and related['insurance']['top'] is not None:
                            top_df = related['insurance']['top']
                            if not top_df.empty:
                                rising_data['top_queries'] = top_df.to_dict('records')
                                print(f"\n  ğŸ”� TOP QUERIES ({len(top_df)} found):")
                                for idx, row in top_df.head(5).iterrows():
                                    print(f"    {idx+1}. {row['query']}: {row['value']}")
                    else:
                        print("  âš  No related queries data available")
                except Exception as e:
                    print(f"  âš  Error fetching related queries: {str(e)[:100]}")
            else:
                print("  âš  Could not build payload")
                
        except Exception as e:
            print(f"  âœ— ERROR: {str(e)[:100]}")
        
        return rising_data

class EnhancedNewsAnalyzer:
    """Enhanced news analyzer with more sources and wordcloud support"""
    
    def __init__(self):
        print("\n[INIT] Initializing Enhanced News Analyzer...")
        self.vader = SentimentIntensityAnalyzer()
        self.news_sources = [
            {'name': 'Insurance Business Magazine', 'url': 'https://www.insurancebusinessmag.com/us/rss/'},
            {'name': 'Insurance Journal', 'url': 'https://www.insurancejournal.com/rss/'},
            {'name': 'Google News - Insurance', 'url': 'https://news.google.com/rss/search?q=insurance+industry+news&hl=en-US&gl=US&ceid=US:en'},
            {'name': 'Reuters Business', 'url': 'https://feeds.reuters.com/reuters/businessNews'},
            {'name': 'Best\'s News', 'url': 'https://news.ambest.com/rss/'},
            {'name': 'InsuranceNewsNet', 'url': 'https://insurancenewsnet.com/rss/'},
            {'name': 'Insurance Day', 'url': 'https://www.insuranceday.com/rss'}
        ]
        print(f"  âœ“ News Analyzer initialized with {len(self.news_sources)} sources")
    
    def fetch_all_news(self):
        """Fetch and analyze news from all sources"""
        print("\n[NEWS ANALYSIS] Starting comprehensive news intelligence gathering...")
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
            
            print(f"     âœ“ Retrieved {len(articles)} articles from {source_name}")
        
        print(f"\n[NEWS SUMMARY] Total articles collected: {len(all_articles)}")
        
        if all_articles:
            df = pd.DataFrame(all_articles)
            
            # Sentiment analysis
            sentiment_dist = df['sentiment_label'].value_counts().to_dict()
            print(f"\n  ğŸ˜Š SENTIMENT ANALYSIS:")
            print("  " + "="*50)
            
            total_articles = len(all_articles)
            for label, count in sentiment_dist.items():
                percentage = (count / total_articles) * 100
                emoji = "ğŸ˜Š" if label == 'positive' else "ğŸ˜Ÿ" if label == 'negative' else "ğŸ˜�"
                print(f"    {emoji} {label.upper()}: {count} articles ({percentage:.1f}%)")
            
            avg_sentiment = df['sentiment_score'].mean()
            print(f"\n    ğŸ“Š Average sentiment score: {avg_sentiment:.4f}")
            
            # Keywords analysis
            print(f"\n  ğŸ”¤ KEYWORD FREQUENCY ANALYSIS:")
            all_text = ' '.join(df['title'].tolist() + df['summary'].tolist()).lower()
            keywords = ['claim', 'lawsuit', 'premium', 'coverage', 'policy', 'rate', 'risk', 
                       'cyber', 'climate', 'flood', 'auto', 'health', 'life', 'property', 'ai', 'tech']
            
            keyword_counts = {}
            for keyword in keywords:
                count = all_text.count(keyword)
                if count > 0:
                    keyword_counts[keyword] = count
            
            sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
            for keyword, count in sorted_keywords[:10]:
                print(f"    â€¢ '{keyword}': mentioned {count} times")
            
            return {
                'articles': all_articles,
                'all_text': all_text,
                'summary': {
                    'total_count': len(all_articles),
                    'sentiment_distribution': sentiment_dist,
                    'average_sentiment': float(avg_sentiment),
                    'sources': source_stats,
                    'keyword_frequency': dict(sorted_keywords[:10])
                }
            }
        
        return {'articles': [], 'all_text': '', 'summary': {}}
    
    def fetch_rss_feed(self, url, source_name):
        """Fetch and parse RSS feed with improved timeout"""
        articles = []
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            feed = feedparser.parse(response.content)
            
            if not feed.entries:
                return articles
            
            cutoff_date = datetime.now() - timedelta(days=30)
            
            for entry in feed.entries[:50]:
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
                
                except Exception as e:
                    continue
            
        except Exception as e:
            print(f"       âœ— Error fetching feed: {str(e)[:100]}")
        
        return articles
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text with combined VADER and TextBlob"""
        try:
            # VADER sentiment
            vader_scores = self.vader.polarity_scores(text)
            
            # TextBlob sentiment
            blob = TextBlob(text)
            blob_polarity = blob.sentiment.polarity
            combined_score = (vader_scores['compound'] + blob_polarity) / 2
            
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

class EnhancedDataVisualizer:
    """Enhanced visualization engine with wordcloud and improved dashboards"""
    
    def __init__(self):
        print("\n[INIT] Initializing Enhanced Data Visualizer...")
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        # Create output directory
        self.output_dir = 'insurance_analysis_output'
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'charts'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'data'), exist_ok=True)
        
        print(f"  âœ“ Data Visualizer initialized")
        print(f"  ğŸ“� Output directory: {self.output_dir}")
    
    def create_comprehensive_dashboard(self, trends_data, news_data, rising_data):
        """Create comprehensive dashboard with added wordcloud"""
        print("\n[VISUALIZATION] Creating comprehensive insurance intelligence dashboard...")
        print("="*80)
        
        try:
            # Create matplotlib dashboard with extra subplot for wordcloud
            fig = plt.figure(figsize=(24, 16))
            fig.suptitle('Insurance Industry Intelligence Dashboard - 2025 Edition', fontsize=20, fontweight='bold')
            
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # Plot 1: Trends Timeline
            ax1 = fig.add_subplot(gs[0, :2])
            self._plot_trends_timeline(ax1, trends_data)
            
            # Plot 2: News Sentiment
            ax2 = fig.add_subplot(gs[0, 2])
            self._plot_news_sentiment(ax2, news_data)
            
            # Plot 3: Regional Heatmap Data
            ax3 = fig.add_subplot(gs[1, :2])
            self._plot_regional_comparison(ax3, trends_data)
            
            # Plot 4: Wordcloud
            ax4 = fig.add_subplot(gs[1, 2])
            self._plot_wordcloud(ax4, news_data)
            
            # Plot 5: Summary Text
            ax5 = fig.add_subplot(gs[2, :])
            self._add_comprehensive_summary(ax5, trends_data, news_data, rising_data)
            
            plt.tight_layout()
            
            # Save dashboard
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(self.output_dir, 'charts', f'dashboard_{timestamp}.png')
            plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"  ğŸ’¾ Dashboard saved as: {filename}")
            
            plt.show()
            
            # Create interactive Plotly charts
            self._create_interactive_charts(trends_data, news_data)
            
            return filename
            
        except Exception as e:
            print(f"  âœ— Error creating dashboard: {str(e)[:200]}")
            return None
    
    def _plot_trends_timeline(self, ax, trends_data):
        """Plot trends timeline"""
        try:
            data_found = False
            
            for key in trends_data:
                if 'timeline' in key and isinstance(trends_data[key], pd.DataFrame):
                    data = trends_data[key]
                    if not data.empty:
                        # Plot up to 5 columns
                        cols_to_plot = [col for col in data.columns[:5] if data[col].dtype in ['float64', 'int64']]
                        for i, col in enumerate(cols_to_plot):
                            ax.plot(data.index, data[col], label=col, linewidth=2, 
                                   color=self.colors[i % len(self.colors)], alpha=0.8)
                            data_found = True
                        
                        if data_found:
                            break
            
            if data_found:
                ax.set_title('Insurance Search Trends Over Time', fontweight='bold', fontsize=12)
                ax.set_xlabel('Date')
                ax.set_ylabel('Search Interest (0-100)')
                ax.legend(loc='best', fontsize=9, ncol=2)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.text(0.5, 0.5, 'No trends data available', ha='center', va='center')
                ax.set_title('Insurance Search Trends', fontweight='bold', fontsize=12)
                
        except Exception as e:
            ax.text(0.5, 0.5, 'Error loading trends', ha='center', va='center')
    
    def _plot_news_sentiment(self, ax, news_data):
        """Plot news sentiment distribution"""
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
                    
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                    
                    ax.set_title('News Sentiment Distribution', fontweight='bold', fontsize=12)
                else:
                    ax.text(0.5, 0.5, 'No sentiment data', ha='center', va='center')
            else:
                ax.text(0.5, 0.5, 'No news data available', ha='center', va='center')
                ax.set_title('News Sentiment', fontweight='bold', fontsize=12)
                
        except Exception as e:
            ax.text(0.5, 0.5, 'Error loading sentiment', ha='center', va='center')
    
    def _plot_regional_comparison(self, ax, trends_data):
        """Plot regional comparison"""
        try:
            regional_data = None
            
            # Find regional data
            for key in trends_data:
                if 'regions' in key and isinstance(trends_data[key], pd.DataFrame):
                    regional_data = trends_data[key]
                    break
            
            if regional_data is not None and not regional_data.empty:
                # Get top 10 regions
                if len(regional_data.columns) > 0:
                    col = regional_data.columns[0]
                    top_regions = regional_data[col].sort_values(ascending=False).head(10)
                    
                    bars = ax.barh(range(len(top_regions)), top_regions.values, color=self.colors[0])
                    ax.set_yticks(range(len(top_regions)))
                    ax.set_yticklabels(top_regions.index, fontsize=9)
                    ax.set_xlabel('Search Interest')
                    ax.set_title('Top Regions by Insurance Search Interest', fontweight='bold', fontsize=12)
                    
                    # Add value labels
                    for i, (bar, value) in enumerate(zip(bars, top_regions.values)):
                        ax.text(value + 1, bar.get_y() + bar.get_height()/2, f'{value:.0f}', 
                               va='center', fontsize=8)
                else:
                    ax.text(0.5, 0.5, 'No regional data columns', ha='center', va='center')
            else:
                ax.text(0.5, 0.5, 'No regional data available', ha='center', va='center')
                ax.set_title('Regional Analysis', fontweight='bold', fontsize=12)
                
        except Exception as e:
            ax.text(0.5, 0.5, 'Error loading regional data', ha='center', va='center')
    
    def _plot_wordcloud(self, ax, news_data):
        """Plot wordcloud from news text"""
        try:
            if news_data and 'all_text' in news_data and news_data['all_text']:
                wordcloud = WordCloud(width=400, height=300, background_color='white',
                                      max_words=100, colormap='viridis').generate(news_data['all_text'])
                
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.set_title('News Keyword Wordcloud', fontweight='bold', fontsize=12)
                ax.axis('off')
            else:
                ax.text(0.5, 0.5, 'No text for wordcloud', ha='center', va='center')
                ax.set_title('News Wordcloud', fontweight='bold', fontsize=12)
                ax.axis('off')
        except Exception as e:
            ax.text(0.5, 0.5, 'Error generating wordcloud', ha='center', va='center')
            ax.axis('off')
    
    def _add_comprehensive_summary(self, ax, trends_data, news_data, rising_data):
        """Add comprehensive summary text with more details"""
        try:
            summary_lines = ["MARKET INTELLIGENCE SUMMARY - 2025 EDITION", "=" * 80, ""]
            
            # Add timestamp
            summary_lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            summary_lines.append("")
            
            # Trends summary
            summary_lines.append("GOOGLE TRENDS ANALYSIS:")
            summary_lines.append("-" * 40)
            
            trends_count = len([k for k in trends_data.keys() if 'timeline' in k])
            summary_lines.append(f"  Categories analyzed: {trends_count}")
            
            # Find trending keywords
            trending_up = []
            trending_down = []
            
            for key in trends_data:
                if 'timeline' in key and isinstance(trends_data[key], pd.DataFrame):
                    data = trends_data[key]
                    for col in data.columns:
                        if data[col].dtype in ['float64', 'int64'] and len(data) >= 30:
                            recent = data[col].iloc[-15:].mean()
                            older = data[col].iloc[-30:-15].mean()
                            if older > 0:
                                change = ((recent - older) / older * 100)
                                if change > 10:
                                    trending_up.append((col, change))
                                elif change < -10:
                                    trending_down.append((col, change))
            
            if trending_up:
                summary_lines.append(f"  Trending UP: {', '.join([f'{t[0]} (+{t[1]:.1f}%)' for t in trending_up[:3]])}")
            if trending_down:
                summary_lines.append(f"  Trending DOWN: {', '.join([f'{t[0]} ({t[1]:.1f}%)' for t in trending_down[:3]])}")
            
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
                    top_keywords = list(summary['keyword_frequency'].keys())[:5]
                    summary_lines.append(f"  Top Keywords: {', '.join(top_keywords)}")
            
            summary_lines.append("")
            
            # Rising queries summary
            if rising_data:
                summary_lines.append("RISING SEARCHES:")
                summary_lines.append("-" * 40)
                
                if 'rising_queries' in rising_data and rising_data['rising_queries']:
                    for query in rising_data['rising_queries'][:3]:
                        summary_lines.append(f"  â€¢ {query.get('query', 'N/A')}: {query.get('value', 'N/A')}")
            
            # Join all lines
            summary_text = "\n".join(summary_lines)
            
            ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.1))
            ax.axis('off')
            
        except Exception as e:
            ax.text(0.5, 0.5, 'Summary generation failed', ha='center', va='center')
            ax.axis('off')
    
    def _create_interactive_charts(self, trends_data, news_data):
        """Create interactive Plotly charts with added wordcloud if possible"""
        try:
            print("\n  ğŸ“Š Creating interactive visualizations...")
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Insurance Search Trends', 'Category Comparison',
                              'Regional Interest', 'Sentiment Analysis'),
                specs=[[{"secondary_y": False}, {"type": "bar"}],
                      [{"type": "geo"}, {"type": "bar"}]]
            )
            
            # Add traces based on available data
            trace_added = False
            
            # Plot 1: Time series trends
            for key in trends_data:
                if 'timeline' in key and isinstance(trends_data[key], pd.DataFrame):
                    data = trends_data[key]
                    for col in data.columns[:3]:
                        if data[col].dtype in ['float64', 'int64']:
                            fig.add_trace(
                                go.Scatter(x=data.index, y=data[col], name=col, mode='lines'),
                                row=1, col=1
                            )
                            trace_added = True
                    if trace_added:
                        break
            
            # Update layout
            fig.update_layout(
                height=800,
                showlegend=True,
                title_text="Insurance Industry Analysis - Interactive Dashboard",
                title_font_size=18
            )
            
            # Save to HTML
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(self.output_dir, 'charts', f'interactive_{timestamp}.html')
            fig.write_html(filename)
            print(f"    ğŸ’¾ Interactive chart saved as: {filename}")
            
        except Exception as e:
            print(f"    âš  Error creating interactive charts: {str(e)[:100]}")
    
    def export_results(self, trends_data, news_data, rising_data):
        """Export all results to JSON with added metadata"""
        print("\n[EXPORT] Saving comprehensive results...")
        
        try:
            export_data = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'report_type': 'Insurance Industry Intelligence Analysis',
                    'version': '3.0 - 2025 Enhanced'
                },
                'trends': {},
                'news': {},
                'rising_queries': rising_data if rising_data else {}
            }
            
            # Process trends data
            for key, value in trends_data.items():
                try:
                    if isinstance(value, pd.DataFrame):
                        export_data['trends'][key] = {
                            'data': value.to_dict(),
                            'shape': value.shape,
                            'columns': value.columns.tolist()
                        }
                    else:
                        export_data['trends'][key] = value
                except Exception as e:
                    export_data['trends'][key] = {'error': str(e)}
            
            # Process news data
            if news_data:
                export_data['news'] = {
                    'summary': news_data.get('summary', {}),
                    'article_count': len(news_data.get('articles', [])),
                    'articles_sample': news_data.get('articles', [])[:10]
                }
            
            # Save to file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(self.output_dir, 'data', f'results_{timestamp}.json')
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str, ensure_ascii=False)
            
            print(f"  ğŸ’¾ Results saved to {filename}")
            
            return filename
            
        except Exception as e:
            print(f"  âœ— Error exporting results: {str(e)[:100]}")
            return None

def main():
    """Main execution function with enhanced steps"""
    print("\n" + "="*80)
    print("ğŸš€ STARTING ENHANCED INSURANCE INDUSTRY ANALYSIS - 2025")
    print("="*80)
    print(f"Analysis initiated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Initialize components
    print("\n[INITIALIZATION] Setting up analysis components...")
    trends_analyzer = EnhancedGoogleTrendsAnalyzer()
    news_analyzer = EnhancedNewsAnalyzer()
    visualizer = EnhancedDataVisualizer()
    
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
        print(f"âœ— Error in trends analysis: {str(e)[:200]}")
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
        print(f"âœ— Error in rising queries: {str(e)[:200]}")
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
        print(f"âœ— Error in news analysis: {str(e)[:200]}")
        news_data = {}
    
    # Step 4: Visualization
    print("\n" + "="*80)
    print("[STEP 4/5] CREATING VISUALIZATIONS")
    print("="*80)
    
    try:
        dashboard_file = visualizer.create_comprehensive_dashboard(trends_data, news_data, rising_queries)
        results['files']['dashboard'] = dashboard_file
    except Exception as e:
        print(f"âœ— Dashboard creation failed: {str(e)[:200]}")
    
    # Step 5: Export Results
    print("\n" + "="*80)
    print("[STEP 5/5] EXPORTING RESULTS")
    print("="*80)
    json_file = visualizer.export_results(trends_data, news_data, rising_queries)
    results['files']['json'] = json_file
    
    # Print final summary
    print("\n" + "="*80)
    print("âœ… ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"â�±ï¸� Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nğŸ“Š RESULTS SUMMARY:")
    print("-" * 40)
    print(f"â€¢ Trend categories analyzed: {len([k for k in trends_data.keys() if 'timeline' in k])}")
    print(f"â€¢ News articles processed: {len(news_data.get('articles', [])) if news_data else 0}")
    print(f"â€¢ Rising queries found: {len(rising_queries.get('rising_queries', [])) if rising_queries else 0}")
    
    print("\nğŸ“� OUTPUT FILES:")
    print("-" * 40)
    for file_type, filename in results['files'].items():
        if filename:
            print(f"â€¢ {file_type.upper()}: {filename}")
    
    print("\n" + "="*80)
    print("ğŸ�‰ ALL PROCESSING COMPLETE!")
    print("="*80)
    
    return results

if __name__ == "__main__":
    try:
        results = main()
        print("\nâœ¨ Insurance Industry Intelligence Analysis completed successfully!")
    except Exception as e:
        print(f"\nâ�Œ CRITICAL ERROR: {str(e)}")
        print(traceback.format_exc())







