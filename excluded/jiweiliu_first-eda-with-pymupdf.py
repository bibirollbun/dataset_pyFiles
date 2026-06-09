!pip install pymupdf


# Configuration
RUN_PHASE = False  # Set to True for full execution with plot saving

import os
import pymupdf  # PyMuPDF
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set up plotting
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


# Define paths
DATA_DIR = Path('/kaggle/input/make-data-count-finding-data-references')
TRAIN_PDF_DIR = DATA_DIR / 'train' / 'PDF'
TEST_PDF_DIR = DATA_DIR / 'test' / 'PDF'
TRAIN_LABELS_PATH = DATA_DIR / 'train_labels.csv'
PLOTS_DIR = Path('notebooks/plots')

# Ensure plots directory exists
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Data directory: {DATA_DIR}")
print(f"Train PDF directory: {TRAIN_PDF_DIR}")
print(f"Test PDF directory: {TEST_PDF_DIR}")
print(f"Plots directory: {PLOTS_DIR}")


# Load training labels
train_labels = pd.read_csv(TRAIN_LABELS_PATH)
print(f"Training labels shape: {train_labels.shape}")
print(f"\nColumns: {train_labels.columns.tolist()}")
print(f"\nFirst 5 rows:")
train_labels.head()


# Analyze label distribution
print("Citation type distribution:")
type_counts = train_labels['type'].value_counts()
print(type_counts)
print(f"\nPrimary citations: {type_counts.get('Primary', 0):,} ({type_counts.get('Primary', 0)/len(train_labels)*100:.1f}%)")
print(f"Secondary citations: {type_counts.get('Secondary', 0):,} ({type_counts.get('Secondary', 0)/len(train_labels)*100:.1f}%)")


# Get list of PDF files
train_pdf_files = list(TRAIN_PDF_DIR.glob('*.pdf'))
test_pdf_files = list(TEST_PDF_DIR.glob('*.pdf'))

print(f"Number of training PDFs: {len(train_pdf_files)}")
print(f"Number of test PDFs: {len(test_pdf_files)}")

# Sample PDFs for exploration
if RUN_PHASE:
    sample_size = 10
else:
    sample_size = 3  # Smaller sample for debug

sample_pdfs = train_pdf_files[:sample_size]
print(f"\nAnalyzing {sample_size} sample PDFs...")


def extract_pdf_info(pdf_path):
    """Extract basic information from a PDF file."""
    try:
        doc = pymupdf.open(pdf_path)
        info = {
            'filename': pdf_path.name,
            'article_id': pdf_path.stem,
            'num_pages': len(doc),
            'metadata': doc.metadata,
            'total_chars': 0,
            'total_words': 0,
            'has_images': False,
            'has_tables': False
        }
        
        # Extract text and count words/chars
        full_text = ""
        for page in doc:
            text = page.get_text()
            full_text += text
            
            # Check for images
            if page.get_images():
                info['has_images'] = True
            
            # Simple heuristic for tables (looking for grid-like patterns)
            if '|' in text and text.count('|') > 10:
                info['has_tables'] = True
        
        info['total_chars'] = len(full_text)
        info['total_words'] = len(full_text.split())
        info['text_sample'] = full_text[:500]  # First 500 chars
        
        doc.close()
        return info
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None


# Analyze sample PDFs
pdf_info_list = []
for pdf_path in sample_pdfs:
    info = extract_pdf_info(pdf_path)
    if info:
        pdf_info_list.append(info)

# Create DataFrame
pdf_df = pd.DataFrame(pdf_info_list)
print(f"Successfully processed {len(pdf_df)} PDFs")
print(f"\nBasic statistics:")
print(pdf_df[['num_pages', 'total_chars', 'total_words']].describe())


# Display sample PDF information
print("Sample PDF details:")
for i, row in pdf_df.head(3).iterrows():
    print(f"\n{'='*60}")
    print(f"File: {row['filename']}")
    print(f"Pages: {row['num_pages']}")
    print(f"Words: {row['total_words']:,}")
    print(f"Has images: {row['has_images']}")
    print(f"Has tables: {row['has_tables']}")
    print(f"\nText sample (first 200 chars):")
    print(row['text_sample'][:200])


def find_data_citations(pdf_path, patterns=None):
    """Find potential data citations in a PDF."""
    if patterns is None:
        patterns = [
            r'doi[:\s]+10\.\d{4,}/[^\s]+',  # DOI pattern
            r'https?://doi\.org/10\.\d{4,}/[^\s]+',  # Full DOI URLs
            r'GSE\d+',  # Gene Expression Omnibus
            r'PDB[\s:]?\w+',  # Protein Data Bank
            r'E-[A-Z]{4}-\d+',  # ArrayExpress
            r'PRJE\d+',  # European Nucleotide Archive
            r'dryad\.[a-z0-9]+',  # Dryad
            r'zenodo\.\d+',  # Zenodo
            r'figshare\.\d+',  # Figshare
            r'CHEMBL\d+',  # ChEMBL
        ]
    
    try:
        doc = pymupdf.open(pdf_path)
        citations = []
        
        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    citations.append({
                        'page': page_num,
                        'pattern': pattern,
                        'citation': match.group(),
                        'context': text[max(0, match.start()-50):min(len(text), match.end()+50)]
                    })
        
        doc.close()
        return citations
    except Exception as e:
        print(f"Error finding citations in {pdf_path}: {e}")
        return []


# Find citations in sample PDFs
all_citations = []
for pdf_path in sample_pdfs[:3]:  # Analyze first 3 for detail
    citations = find_data_citations(pdf_path)
    if citations:
        print(f"\n{pdf_path.name}: Found {len(citations)} potential citations")
        for cite in citations[:5]:  # Show first 5
            print(f"  Page {cite['page']}: {cite['citation']}")
        all_citations.extend(citations)


# Analyze citation patterns
if all_citations:
    citation_types = Counter([c['pattern'] for c in all_citations])
    print("\nCitation pattern frequency:")
    for pattern, count in citation_types.most_common():
        print(f"  {pattern}: {count}")


# Visualization 1: Page count distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pages distribution
axes[0].hist(pdf_df['num_pages'], bins=20, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Number of Pages')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of PDF Page Counts')
axes[0].grid(True, alpha=0.3)

# Word count distribution
axes[1].hist(pdf_df['total_words'], bins=20, edgecolor='black', alpha=0.7, color='green')
axes[1].set_xlabel('Number of Words')
axes[1].set_ylabel('Count')
axes[1].set_title('Distribution of Word Counts')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

if RUN_PHASE:
    plt.savefig(PLOTS_DIR / 'pdf_exploration_20250820_120000_plot_001.png', dpi=100, bbox_inches='tight')
    print(f"Saved plot to {PLOTS_DIR / 'pdf_exploration_20250820_120000_plot_001.png'}")

plt.show()


# Visualization 2: Citation type distribution
fig, ax = plt.subplots(figsize=(10, 6))

type_counts.plot(kind='bar', ax=ax, color=['#2E86AB', '#A23B72'])
ax.set_xlabel('Citation Type')
ax.set_ylabel('Count')
ax.set_title('Distribution of Citation Types in Training Data')
ax.grid(True, alpha=0.3)

# Add value labels on bars
for i, v in enumerate(type_counts.values):
    ax.text(i, v + 50, str(v), ha='center', va='bottom')

plt.xticks(rotation=0)
plt.tight_layout()

if RUN_PHASE:
    plt.savefig(PLOTS_DIR / 'pdf_exploration_20250820_120000_plot_002.png', dpi=100, bbox_inches='tight')
    print(f"Saved plot to {PLOTS_DIR / 'pdf_exploration_20250820_120000_plot_002.png'}")

plt.show()


# Analyze a specific PDF in detail
sample_pdf = sample_pdfs[0]
print(f"Detailed analysis of: {sample_pdf.name}")

doc = pymupdf.open(sample_pdf)

# Extract sections (simplified - looking for common section headers)
section_headers = ['abstract', 'introduction', 'methods', 'results', 'discussion', 
                  'conclusion', 'references', 'data availability', 'acknowledgments']

full_text = ""
for page in doc:
    full_text += page.get_text()

full_text_lower = full_text.lower()

print("\nSections found:")
for header in section_headers:
    if header in full_text_lower:
        # Find position
        pos = full_text_lower.find(header)
        context = full_text[max(0, pos-20):min(len(full_text), pos+100)]
        print(f"  ✓ {header.title()}: ...{context.replace(chr(10), ' ')}...")

doc.close()


# Look for specific data-related keywords
data_keywords = [
    'publicly available', 'obtained from', 'downloaded from', 'accessed from',
    'data availability', 'supplementary data', 'raw data', 'processed data',
    'dataset', 'repository', 'accession', 'doi:', 'http://doi.org', 'https://doi.org'
]

keyword_counts = {}
for keyword in data_keywords:
    count = full_text_lower.count(keyword)
    if count > 0:
        keyword_counts[keyword] = count

if keyword_counts:
    print("\nData-related keyword frequencies:")
    for keyword, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  '{keyword}': {count}")


# Create summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Total PDFs analyzed: {len(pdf_df)}")
print(f"Average pages per PDF: {pdf_df['num_pages'].mean():.1f}")
print(f"Average words per PDF: {pdf_df['total_words'].mean():.0f}")
print(f"PDFs with images: {pdf_df['has_images'].sum()} ({pdf_df['has_images'].mean()*100:.1f}%)")
print(f"PDFs with tables: {pdf_df['has_tables'].sum()} ({pdf_df['has_tables'].mean()*100:.1f}%)")
print(f"\nTotal potential citations found: {len(all_citations)}")
print(f"Average citations per PDF: {len(all_citations)/len(sample_pdfs[:3]):.1f}")

