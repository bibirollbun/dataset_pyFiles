import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import warnings
from collections import Counter, defaultdict
from bs4 import BeautifulSoup
from pathlib import Path
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available - using matplotlib only")

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    print("WordCloud not available")

warnings.filterwarnings('ignore')

# ğŸ�¨ Set up plotting style
plt.style.use('default')
sns.set_palette('husl')
sns.set_context('talk')

# ğŸ“� Define paths (corrected for eda/ subdirectory)
TRAIN_LABELS = '/kaggle/input/make-data-count-finding-data-references/train_labels.csv'
SAMPLE_SUBMISSION = '/kaggle/input/make-data-count-finding-data-references/sample_submission.csv'
TRAIN_XML_DIR = '/kaggle/input/make-data-count-finding-data-references/train/XML/'
TRAIN_PDF_DIR = '/kaggle/input/make-data-count-finding-data-references/train/PDF/'
TEST_XML_DIR = '/kaggle/input/make-data-count-finding-data-references/test/XML/'
TEST_PDF_DIR = '/kaggle/input/make-data-count-finding-data-references/test/PDF/'

print("ğŸš€ Setup complete! Ready for comprehensive analysis...")
print(f"âœ… Working directory: {os.getcwd()}")
print(f"âœ… Checking file existence:")
print(f"   - Train labels: {os.path.exists(TRAIN_LABELS)}")
print(f"   - Sample submission: {os.path.exists(SAMPLE_SUBMISSION)}")
print(f"   - Train XML dir: {os.path.exists(TRAIN_XML_DIR)}")
print(f"   - Test XML dir: {os.path.exists(TEST_XML_DIR)}")

if all([os.path.exists(TRAIN_LABELS), os.path.exists(SAMPLE_SUBMISSION), 
        os.path.exists(TRAIN_XML_DIR), os.path.exists(TEST_XML_DIR)]):
    print("\nğŸ�‰ All paths are correct and files exist!")
else:
    print("\nâ�Œ Some paths are incorrect - please check the file structure")


# ğŸ“Š Load and examine the training data
train_df = pd.read_csv(TRAIN_LABELS)
sample_sub = pd.read_csv(SAMPLE_SUBMISSION)

print("ğŸ“‹ DATASET OVERVIEW")
print("=" * 50)
print(f"Training records: {len(train_df):,}")
print(f"Unique articles: {train_df['article_id'].nunique():,}")
print(f"Unique datasets: {train_df['dataset_id'].nunique():,}")
print(f"Sample submission records: {len(sample_sub):,}")

print("\nğŸ“� FILE STRUCTURE")
print("=" * 30)
if os.path.exists(TRAIN_XML_DIR):
    train_xml_files = len([f for f in os.listdir(TRAIN_XML_DIR) if f.endswith('.xml')])
    print(f"Training XML files: {train_xml_files}")
else:
    print("Training XML directory not found")

if os.path.exists(TRAIN_PDF_DIR):
    train_pdf_files = len([f for f in os.listdir(TRAIN_PDF_DIR) if f.endswith('.pdf')])
    print(f"Training PDF files: {train_pdf_files}")
else:
    print("Training PDF directory not found")

if os.path.exists(TEST_XML_DIR):
    test_xml_files = len([f for f in os.listdir(TEST_XML_DIR) if f.endswith('.xml')])
    print(f"Test XML files: {test_xml_files}")
else:
    print("Test XML directory not found")

if os.path.exists(TEST_PDF_DIR):
    test_pdf_files = len([f for f in os.listdir(TEST_PDF_DIR) if f.endswith('.pdf')])
    print(f"Test PDF files: {test_pdf_files}")
else:
    print("Test PDF directory not found")

print("\nğŸ”� FIRST 5 TRAINING RECORDS")
print("=" * 35)
display(train_df.head())

print("\nğŸ“Š DATA INFO")
print("=" * 15)
print(train_df.info())


# ğŸ“ˆ Target Distribution Analysis
target_counts = train_df['type'].value_counts()
target_pct = train_df['type'].value_counts(normalize=True) * 100

print("ğŸ�¯ TARGET DISTRIBUTION")
print("=" * 30)
for label, count in target_counts.items():
    pct = target_pct[label]
    print(f"{label:>10}: {count:>5,} ({pct:>5.1f}%)")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Pie chart
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
wedges, texts, autotexts = ax1.pie(target_counts.values, labels=target_counts.index, 
                                  autopct='%1.1f%%', colors=colors, startangle=90)
ax1.set_title('Target Distribution', fontsize=16, fontweight='bold')

# Bar chart
bars = ax2.bar(target_counts.index, target_counts.values, color=colors)
ax2.set_title('Citation Type Counts', fontsize=16, fontweight='bold')
ax2.set_xlabel('Citation Type')
ax2.set_ylabel('Count')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 10,
             f'{int(height):,}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# ğŸš¨ Class Imbalance Check
print("\nâš ï¸�  CLASS IMBALANCE ANALYSIS")
print("=" * 35)
majority_class = target_counts.max()
minority_class = target_counts.min()
imbalance_ratio = majority_class / minority_class
print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
if imbalance_ratio > 3:
    print("âš ï¸�  Significant class imbalance detected! Consider sampling strategies.")
else:
    print("âœ… Class distribution is reasonably balanced.")


# ğŸ�›ï¸� Publication Analysis
def extract_publisher_info(article_id):
    """Extract publisher and journal information from DOI"""
    parts = article_id.split('_')
    doi_parts = parts[0].split('.')
    
    publisher_code = doi_parts[1] if len(doi_parts) > 1 else 'unknown'
    journal_info = '_'.join(parts[1:]) if len(parts) > 1 else 'unknown'
    
    return publisher_code, journal_info

# Extract publisher and journal information
train_df[['publisher', 'journal']] = train_df['article_id'].apply(
    lambda x: pd.Series(extract_publisher_info(x))
)

print("ğŸ�›ï¸� PUBLISHER ANALYSIS")
print("=" * 25)
publisher_stats = train_df.groupby('publisher').agg({
    'article_id': 'nunique',
    'dataset_id': 'count',
    'type': lambda x: x.value_counts().to_dict()
}).round(2)

publisher_stats.columns = ['Unique_Articles', 'Total_Citations', 'Type_Distribution']
publisher_stats = publisher_stats.sort_values('Total_Citations', ascending=False)

print("Top 10 Publishers by Citation Volume:")
display(publisher_stats.head(10))

# Visualize publisher distribution
top_publishers = publisher_stats.head(15)

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(top_publishers.index, top_publishers['Total_Citations'], 
               color='skyblue', alpha=0.8)
ax.set_xlabel('Total Citations')
ax.set_title('Top 15 Publishers by Citation Volume', fontsize=16, fontweight='bold')

# Add value labels
for bar in bars:
    width = bar.get_width()
    ax.text(width + 5, bar.get_y() + bar.get_height()/2, 
            f'{int(width)}', ha='left', va='center')

plt.tight_layout()
plt.show()


# ğŸ”— Dataset Repository Analysis
def extract_repository_info(dataset_id):
    """Extract repository information from dataset URLs"""
    if pd.isna(dataset_id) or dataset_id == 'Missing':
        return 'Missing', 'Missing'
    
    if isinstance(dataset_id, str) and dataset_id.startswith('http'):
        parsed = urlparse(dataset_id)
        domain = parsed.netloc.lower()
        
        # Map domains to repository names
        repo_mapping = {
            'doi.org': 'DOI',
            'dryad.org': 'Dryad',
            'zenodo.org': 'Zenodo',
            'figshare.com': 'Figshare',
            'pangaea.de': 'PANGAEA',
            'ncbi.nlm.nih.gov': 'NCBI',
            'gbif.org': 'GBIF',
            'tcia.at': 'TCIA'
        }
        
        repo_name = 'Other'
        for key, value in repo_mapping.items():
            if key in domain:
                repo_name = value
                break
        
        return repo_name, domain
    else:
        # Handle non-URL identifiers (like gene IDs, etc.)
        return 'Direct_ID', 'direct_identifier'

# Extract repository information
train_df[['repository', 'domain']] = train_df['dataset_id'].apply(
    lambda x: pd.Series(extract_repository_info(x))
)

print("ğŸ”— REPOSITORY ANALYSIS")
print("=" * 25)
repo_stats = train_df.groupby(['repository', 'type']).size().unstack(fill_value=0)
repo_stats['Total'] = repo_stats.sum(axis=1)
repo_stats = repo_stats.sort_values('Total', ascending=False)

print("Dataset Repository Distribution:")
display(repo_stats.head(15))

# Calculate repository preferences by citation type
repo_type_pct = train_df.groupby(['repository', 'type']).size().unstack(fill_value=0)
repo_type_pct = repo_type_pct.div(repo_type_pct.sum(axis=1), axis=0) * 100

print("\nğŸ“Š Repository Preferences by Citation Type (%)")
display(repo_type_pct.head(10).round(1))


# ğŸ“Š Articles with Multiple Citations Analysis
article_citation_counts = train_df.groupby('article_id').agg({
    'dataset_id': 'count',
    'type': lambda x: x.value_counts().to_dict()
}).rename(columns={'dataset_id': 'citation_count', 'type': 'type_distribution'})

print("ğŸ“Š MULTI-CITATION ANALYSIS")
print("=" * 30)
citation_count_dist = article_citation_counts['citation_count'].value_counts().sort_index()
print("Distribution of citations per article:")
for count, articles in citation_count_dist.head(10).items():
    print(f"{count:>2} citations: {articles:>4} articles")

# Analyze patterns in multi-citation articles
multi_citation_articles = article_citation_counts[article_citation_counts['citation_count'] > 1]
print(f"\nArticles with multiple citations: {len(multi_citation_articles)} ({len(multi_citation_articles)/len(article_citation_counts)*100:.1f}%)")

# Visualize citation count distribution
plt.figure(figsize=(12, 6))
citation_counts_to_plot = citation_count_dist.head(20)  # Top 20 for readability
bars = plt.bar(citation_counts_to_plot.index.astype(str), citation_counts_to_plot.values, 
               color='lightcoral', alpha=0.8)
plt.xlabel('Number of Citations per Article')
plt.ylabel('Number of Articles')
plt.title('Distribution of Citations per Article', fontsize=16, fontweight='bold')
plt.xticks(rotation=45)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{int(height)}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()


# ğŸ“„ Sample XML Document Analysis
def parse_xml_document(xml_path):
    """Parse XML document and extract structured information"""
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'xml')
        
        # Extract different sections
        sections = {
            'title': '',
            'abstract': '',
            'methods': '',
            'results': '',
            'discussion': '',
            'references': '',
            'data_availability': '',
            'full_text': ''
        }
        
        # Extract title
        title_elem = soup.find('title')
        if title_elem:
            sections['title'] = title_elem.get_text(strip=True)
        
        # Extract abstract
        abstract_elem = soup.find('abstract')
        if abstract_elem:
            sections['abstract'] = abstract_elem.get_text(strip=True)
        
        # Extract full text
        sections['full_text'] = soup.get_text(strip=True)
        
        # Extract sections by looking for div or p tags with relevant classes/text
        for div in soup.find_all(['div', 'p', 'section']):
            text = div.get_text(strip=True).lower()
            if any(keyword in text[:100] for keyword in ['method', 'material', 'procedure']):
                sections['methods'] += ' ' + div.get_text(strip=True)
            elif any(keyword in text[:100] for keyword in ['result', 'finding', 'outcome']):
                sections['results'] += ' ' + div.get_text(strip=True)
            elif any(keyword in text[:100] for keyword in ['discussion', 'conclusion']):
                sections['discussion'] += ' ' + div.get_text(strip=True)
            elif any(keyword in text[:100] for keyword in ['data availability', 'data access']):
                sections['data_availability'] += ' ' + div.get_text(strip=True)
        
        return sections
    
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return None

# Analyze a sample of documents
sample_articles = train_df['article_id'].unique()[:5]
document_analysis = []

print("ğŸ“„ DOCUMENT STRUCTURE ANALYSIS")
print("=" * 35)

for article_id in sample_articles:
    xml_path = os.path.join(TRAIN_XML_DIR, f"{article_id}.xml")
    if os.path.exists(xml_path):
        sections = parse_xml_document(xml_path)
        if sections:
            doc_info = {
                'article_id': article_id,
                'title_length': len(sections['title']),
                'abstract_length': len(sections['abstract']),
                'full_text_length': len(sections['full_text']),
                'has_methods': len(sections['methods']) > 100,
                'has_results': len(sections['results']) > 100,
                'has_data_availability': len(sections['data_availability']) > 50
            }
            document_analysis.append(doc_info)
            
            print(f"\nğŸ“‹ {article_id}:")
            print(f"  Title: {doc_info['title_length']} chars")
            print(f"  Abstract: {doc_info['abstract_length']} chars")
            print(f"  Full text: {doc_info['full_text_length']:,} chars")
            print(f"  Has structured sections: Methods={doc_info['has_methods']}, Results={doc_info['has_results']}")
    else:
        print(f"XML file not found for {article_id}")

if document_analysis:
    doc_df = pd.DataFrame(document_analysis)
    print("\nğŸ“Š DOCUMENT STATISTICS")
    print("=" * 25)
    print(f"Average full text length: {doc_df['full_text_length'].mean():,.0f} characters")
    print(f"Documents with structured methods: {doc_df['has_methods'].sum()}/{len(doc_df)}")
    print(f"Documents with structured results: {doc_df['has_results'].sum()}/{len(doc_df)}")
    print(f"Documents with data availability: {doc_df['has_data_availability'].sum()}/{len(doc_df)}")
else:
    print("No documents could be analyzed - check XML file paths")



from urllib.parse import urlparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def extract_repository_info(dataset_id):
    """Extract repository and domain information from dataset_id"""
    if pd.isna(dataset_id) or dataset_id == 'Missing':
        return {'repository': 'Missing', 'domain': 'Missing'}
    
    try:
        url = str(dataset_id).lower()
        
        # Extract domain
        if url.startswith('http'):
            parsed = urlparse(url)
            domain = parsed.netloc
        else:
            domain = 'Unknown'
        
        # Map domains to repository names
        if 'zenodo.org' in url:
            repository = 'Zenodo'
        elif 'dryad' in url:
            repository = 'Dryad'
        elif 'figshare' in url:
            repository = 'Figshare'
        elif 'dataverse' in url:
            repository = 'Dataverse'
        elif 'ccdc.csd' in url:
            repository = 'CCDC'
        elif 'github' in url:
            repository = 'GitHub'
        elif 'osf.io' in url:
            repository = 'OSF'
        elif 'pangaea' in url:
            repository = 'PANGAEA'
        elif 'ncbi.nlm.nih.gov' in url:
            repository = 'NCBI'
        elif 'ebi.ac.uk' in url:
            repository = 'EBI'
        else:
            repository = 'Other'
            
        return {'repository': repository, 'domain': domain}
    except:
        return {'repository': 'Unknown', 'domain': 'Unknown'}

def extract_publisher_from_article_id(article_id):
    """Extract publisher from article DOI"""
    if pd.isna(article_id):
        return 'Unknown'
    
    article_str = str(article_id).lower()
    
    if article_str.startswith('10.1002'):
        return 'Wiley'
    elif article_str.startswith('10.1038'):
        return 'Nature'
    elif article_str.startswith('10.1016'):
        return 'Elsevier'  
    elif article_str.startswith('10.1021'):
        return 'ACS'
    elif article_str.startswith('10.1007'):
        return 'Springer'
    elif article_str.startswith('10.1073'):
        return 'PNAS'
    elif article_str.startswith('10.1039'):
        return 'RSC'
    elif article_str.startswith('10.1098'):
        return 'Royal Society'
    elif article_str.startswith('10.1111'):
        return 'Wiley'
    elif article_str.startswith('10.1080'):
        return 'Taylor & Francis'
    elif article_str.startswith('10.1093'):
        return 'Oxford'
    elif article_str.startswith('10.1103'):
        return 'APS'
    elif article_str.startswith('10.1107'):
        return 'IUCr'
    elif article_str.startswith('10.1128'):
        return 'ASM'
    elif article_str.startswith('10.1140'):
        return 'Springer'
    elif article_str.startswith('10.1155'):
        return 'Hindawi'
    elif article_str.startswith('10.1186'):
        return 'BMC'
    else:
        return 'Other'

# Create repository and publisher columns
print("ğŸ”§ Creating repository and publisher columns...")

# Apply functions to create the missing columns
repo_info = train_df['dataset_id'].apply(extract_repository_info)
train_df[['repository', 'domain']] = pd.DataFrame(repo_info.tolist(), index=train_df.index)
train_df['publisher'] = train_df['article_id'].apply(extract_publisher_from_article_id)

print("âœ… Columns created successfully!")
print(f"Repository distribution:\n{train_df['repository'].value_counts()}")
print(f"\nPublisher distribution:\n{train_df['publisher'].value_counts()}")
print(f"\nDataFrame shape: {train_df.shape}")

# Advanced Visualizations
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Default colors for citation types

# 1. Repository vs Citation Type Heatmap
plt.figure(figsize=(12, 8))
repo_type_matrix = train_df.groupby(['repository', 'type']).size().unstack(fill_value=0)
repo_type_matrix_pct = repo_type_matrix.div(repo_type_matrix.sum(axis=1), axis=0) * 100

sns.heatmap(repo_type_matrix_pct.head(10), annot=True, fmt='.1f', cmap='YlOrRd', 
            cbar_kws={'label': 'Percentage'})
plt.title('Repository vs Citation Type Distribution (%)', fontsize=16, fontweight='bold')
plt.xlabel('Citation Type')
plt.ylabel('Repository')
plt.tight_layout()
plt.show()

# 2. Publisher vs Citation Type Analysis
plt.figure(figsize=(14, 8))
publisher_type_matrix = train_df.groupby(['publisher', 'type']).size().unstack(fill_value=0)
publisher_type_matrix_pct = publisher_type_matrix.div(publisher_type_matrix.sum(axis=1), axis=0) * 100

# Get top 15 publishers by count
publisher_stats = train_df['publisher'].value_counts()
top_publishers_matrix = publisher_type_matrix_pct.loc[publisher_stats.head(15).index]

sns.heatmap(top_publishers_matrix, annot=True, fmt='.1f', cmap='viridis', 
            cbar_kws={'label': 'Percentage'})
plt.title('Top 15 Publishers vs Citation Type Distribution (%)', fontsize=16, fontweight='bold')
plt.xlabel('Citation Type')
plt.ylabel('Publisher')
plt.tight_layout()
plt.show()

# 3. Citation Count Distribution by Type
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, citation_type in enumerate(['Primary', 'Secondary', 'Missing']):
    type_data = train_df[train_df['type'] == citation_type]
    article_counts = type_data.groupby('article_id').size()
    
    if len(article_counts) > 0:
        axes[i].hist(article_counts, bins=range(1, min(21, article_counts.max() + 2)), 
                     alpha=0.7, color=colors[i], edgecolor='black')
        axes[i].set_title(f'{citation_type} Citations per Article', fontweight='bold')
        axes[i].set_xlabel('Number of Citations')
        axes[i].set_ylabel('Number of Articles')
        axes[i].grid(True, alpha=0.3)
    else:
        axes[i].text(0.5, 0.5, f'No {citation_type} citations', 
                     ha='center', va='center', transform=axes[i].transAxes)

plt.tight_layout()
plt.show()


# ğŸ”§ Feature Engineering Pipeline

class DataCitationFeatureEngineer:
    """Comprehensive feature engineering for data citation classification"""
    
    def __init__(self):
        self.repository_encoder = {}
        self.publisher_encoder = {}
    
    def extract_url_features(self, dataset_id):
        """Extract features from dataset URL structure"""
        features = {}
        
        if pd.isna(dataset_id) or dataset_id == 'Missing':
            features.update({
                'url_length': 0,
                'has_doi': False,
                'has_version': False,
                'path_depth': 0,
                'is_https': False
            })
        else:
            features.update({
                'url_length': len(str(dataset_id)),
                'has_doi': 'doi.org' in str(dataset_id).lower(),
                'has_version': any(v in str(dataset_id).lower() for v in ['v1', 'v2', 'version']),
                'path_depth': str(dataset_id).count('/'),
                'is_https': str(dataset_id).startswith('https')
            })
        
        return features
    
    def extract_article_features(self, article_id, article_citations):
        """Extract article-level features"""
        features = {
            'total_citations': len(article_citations),
            'unique_repositories': article_citations['repository'].nunique(),
            'has_missing_citations': (article_citations['type'] == 'Missing').any(),
            'primary_ratio': (article_citations['type'] == 'Primary').mean(),
            'secondary_ratio': (article_citations['type'] == 'Secondary').mean(),
        }
        
        return features
    
    def create_features(self, df):
        """Create comprehensive feature set"""
        features_list = []
        
        for idx, row in df.iterrows():
            # Basic features
            features = {
                'publisher': row['publisher'],
                'repository': row['repository'],
                'type': row['type']  # target
            }
            
            # URL features
            url_features = self.extract_url_features(row['dataset_id'])
            features.update(url_features)
            
            # Article-level features
            article_citations = df[df['article_id'] == row['article_id']]
            article_features = self.extract_article_features(row['article_id'], article_citations)
            features.update(article_features)
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)

# Create feature set
print("ğŸ”§ FEATURE ENGINEERING")
print("=" * 25)

feature_engineer = DataCitationFeatureEngineer()
features_df = feature_engineer.create_features(train_df)

print(f"Feature set created with {len(features_df)} samples and {len(features_df.columns)} features")
print("\nFeature columns:")
for col in features_df.columns:
    print(f"  - {col}")

print("\nğŸ“Š Feature Statistics:")
display(features_df.describe())

