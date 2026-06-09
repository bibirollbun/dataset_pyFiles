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


!pip install pymupdf lxml tqdm


import re
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET
import fitz  # PyMuPDF
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Constants
DATA_PATH = Path('/kaggle/input/make-data-count-finding-data-references')
TRAIN_PDF = DATA_PATH / 'train' / 'PDF'
TRAIN_XML = DATA_PATH / 'train' / 'XML'
TEST_PDF = DATA_PATH / 'test' / 'PDF'
TEST_XML = DATA_PATH / 'test' / 'XML'

# Regex patterns for dataset identification
PATTERNS = {
    # DOI patterns (with/without URL prefix)
    'doi_plain': re.compile(r'\b(10\.\d{4,}(?:\.\d+)*/[^\s\\()"\']+)\b'),
    'doi_url': re.compile(r'https?://(?:dx\.)?doi\.org/(10\.\d{4,}(?:\.\d+)*/[^\s\\()"\']+)'),
    
    # Repository-specific patterns
    'geo': re.compile(r'\b(GSE\d+)\b', re.I),
    'arrayexpress': re.compile(r'\b(E-[A-Z]+-\d+)\b', re.I),
    'pdb': re.compile(r'\b(PDB\s?[\d\w]+)\b', re.I),
    'embl': re.compile(r'\b(PRJ[EDN]\d+)\b', re.I),
    'dryad': re.compile(r'\b(10\.5061/dryad\.[\w\d]+)\b', re.I),
    'figshare': re.compile(r'\b(10\.\d+/figshare\.[\w\d]+)\b', re.I),
    'zenodo': re.compile(r'\b(10\.\d+/zenodo\.[\w\d]+)\b', re.I),
    'chembl': re.compile(r'\b(CHEMBL\d+)\b', re.I),
    'uniprot': re.compile(r'\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})\b')
}

def extract_text(file_path, file_type):
    """Extract text from PDF or XML file"""
    try:
        if file_type == 'pdf':
            with fitz.open(file_path) as doc:
                return ' '.join(page.get_text() for page in doc)
        elif file_type == 'xml':
            tree = ET.parse(file_path)
            return ' '.join(elem.text.strip() for elem in tree.iter() if elem.text and elem.text.strip())
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return ''

def find_datasets(text):
    """Identify all dataset references in text"""
    datasets = set()
    
    # Check all patterns
    for pattern in PATTERNS.values():
        for match in pattern.finditer(text):
            id_str = match.group(1) if len(match.groups()) > 0 else match.group(0)
            
            # Standardize formats
            if 'pdb' in pattern.pattern:
                id_str = id_str.replace(' ', '').lower()
            elif 'doi' in pattern.pattern and not id_str.startswith('http'):
                id_str = f'https://doi.org/{id_str}'
            
            datasets.add(id_str)
    
    return datasets

def classify_citation(text, dataset_id):
    """Determine if dataset is Primary or Secondary based on context"""
    # Find the dataset mention in text (case insensitive)
    idx = text.lower().find(dataset_id.lower())
    if idx == -1:
        return 'Secondary'  # Default if context not found
    
    # Extract context window around the mention
    window = 250
    start = max(0, idx - window)
    end = min(len(text), idx + len(dataset_id) + window)
    context = text[start:end].lower()
    
    # Keywords indicating data generation (Primary)
    primary_indicators = [
        'generated in this study',
        'collected for this research',
        'produced by the authors',
        'original data',
        'newly sequenced',
        'experimental data',
        'data generated',
        'this study produced',
        'we collected',
        'we generated',
        'our data',
        'data available at',
        'in this paper we',
        'this work presents'
    ]
    
    # Keywords indicating data reuse (Secondary)
    secondary_indicators = [
        'obtained from',
        'retrieved from',
        'publicly available',
        'downloaded from',
        'previously published',
        'existing dataset',
        'data repository',
        'reused data',
        'public dataset',
        'from the database',
        'accession number',
        'previously described',
        'published by'
    ]
    
    # Score the context
    primary_score = sum(1 for kw in primary_indicators if kw in context)
    secondary_score = sum(1 for kw in secondary_indicators if kw in context)
    
    return 'Primary' if primary_score > secondary_score else 'Secondary'

def process_article(article_id, pdf_path, xml_path=None):
    """Process a single article to find dataset citations"""
    # Prefer XML if available
    if xml_path and xml_path.exists():
        text = extract_text(xml_path, 'xml')
    else:
        text = extract_text(pdf_path, 'pdf')
    
    if not text:
        return []
    
    datasets = find_datasets(text)
    results = []
    
    for dataset_id in datasets:
        citation_type = classify_citation(text, dataset_id)
        results.append((article_id, dataset_id, citation_type))
    
    return results

def generate_submission():
    """Generate competition submission file"""
    submission = []
    row_id = 0
    
    # Process all test PDFs
    test_files = list(TEST_PDF.glob('*.pdf'))
    for pdf_path in tqdm(test_files, desc='Processing test articles'):
        article_id = pdf_path.stem.replace('_', '/')
        xml_path = TEST_XML / f'{pdf_path.stem}.xml'
        
        citations = process_article(article_id, pdf_path, xml_path)
        
        for article_id, dataset_id, citation_type in citations:
            submission.append({
                'row_id': row_id,
                'article_id': article_id.replace('/', '_'),
                'dataset_id': dataset_id,
                'type': citation_type
            })
            row_id += 1
    
    return pd.DataFrame(submission)

# Generate and save submission
submission_df = generate_submission()
submission_df.to_csv('submission.csv', index=False)
print(f"Generated submission with {len(submission_df)} predictions")

