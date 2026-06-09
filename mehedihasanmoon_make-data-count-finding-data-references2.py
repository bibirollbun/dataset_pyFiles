


# Import Required Libraries
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Detect Kaggle environment
IS_KAGGLE = bool(os.getenv('KAGGLE_KERNEL_RUN_TYPE'))
print(f"Running in Kaggle environment: {IS_KAGGLE}")

import fitz  # PyMuPDF for PDF processing
print("âœ… All libraries imported successfully!")


# Load and Explore Dataset
# Unified Kaggle competition data access (works on Kaggle & remote server with API token)
from pathlib import Path
import os, pandas as pd, zipfile, subprocess, json, sys

COMP_NAME = "make-data-count-finding-data-references"
IS_KAGGLE = bool(os.getenv('KAGGLE_KERNEL_RUN_TYPE')) or os.path.exists('/kaggle/input')

# Preferred canonical input dir on Kaggle
DEFAULT_INPUT_DIR = Path(f"/kaggle/input/{COMP_NAME}")
WORK_DIR = Path('/kaggle/working') if IS_KAGGLE else Path('.')
OUTPUT_PATH = str(WORK_DIR / 'submission.csv')

# Local cache directory (for non-Kaggle environment)
LOCAL_DATA_ROOT = Path('data_cache') / COMP_NAME

if IS_KAGGLE and DEFAULT_INPUT_DIR.exists():
    # On official Kaggle runtime: test data only
    DATA_DIR = DEFAULT_INPUT_DIR
    PDF_DIR = DATA_DIR / 'test' / 'PDF'
    XML_DIR = DATA_DIR / 'test' / 'XML'
    MODE = 'KAGGLE_TEST'
    print(f"ğŸ“� Using mounted Kaggle input: {DEFAULT_INPUT_DIR}")
else:
    # Off-Kaggle server: attempt download via kaggle API if not cached
    DATA_DIR = LOCAL_DATA_ROOT
    if not DATA_DIR.exists():
        print(f"â¬‡ï¸� Downloading competition data to {DATA_DIR} ...")
        DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
        # Ensure kaggle.json credentials present
        kaggle_config_dir = Path.home() / '.kaggle'
        if not (kaggle_config_dir / 'kaggle.json').exists():
            raise FileNotFoundError("kaggle.json credentials not found in ~/.kaggle. Please provide them to enable download.")
        # Download zip via kaggle CLI
        subprocess.run(['kaggle', 'competitions', 'download', '-c', COMP_NAME, '-p', str(DATA_DIR)], check=True)
        # Unzip all archives
        for zf in DATA_DIR.glob('*.zip'):
            print(f"Unzipping {zf.name} ...")
            with zipfile.ZipFile(zf, 'r') as z:
                z.extractall(DATA_DIR)
        print("âœ… Download & extraction complete")
    # Use train for local validation if present else test
    train_dir = DATA_DIR / 'train'
    if train_dir.exists():
        PDF_DIR = train_dir / 'PDF'
        XML_DIR = train_dir / 'XML'
        MODE = 'LOCAL_TRAIN'
    else:
        PDF_DIR = DATA_DIR / 'test' / 'PDF'
        XML_DIR = DATA_DIR / 'test' / 'XML'
        MODE = 'LOCAL_TEST'
    print(f"ğŸ“� Using local cached data: {DATA_DIR} (mode={MODE})")

print(f"MODE: {MODE}")
print(f"PDF dir: {PDF_DIR}")
print(f"XML dir: {XML_DIR}")

# Verify existence
if not PDF_DIR.exists():
    raise FileNotFoundError(f"PDF directory missing: {PDF_DIR}")
if not XML_DIR.exists():
    print(f"âš ï¸� XML directory missing (will rely on PDFs): {XML_DIR}")

# List files
pdf_files = list(PDF_DIR.glob('*.pdf'))
xml_files = list(XML_DIR.glob('*.xml')) if XML_DIR.exists() else []
print(f"- PDF files: {len(pdf_files)}")
print(f"- XML files: {len(xml_files)}")
if pdf_files:
    print("  â€¢ Sample PDF:", pdf_files[0].name)
if xml_files:
    print("  â€¢ Sample XML:", xml_files[0].name)

# Sample submission (only if present)
sample_path_candidates = [DATA_DIR / 'sample_submission.csv', Path(f'/kaggle/input/{COMP_NAME}/sample_submission.csv')]
for sp in sample_path_candidates:
    if sp.exists():
        sample_sub = pd.read_csv(sp)
        print("\nğŸ“� Sample submission columns:", list(sample_sub.columns))
        break


# Data Preprocessing - Text Extraction Functions

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file using PyMuPDF."""
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""

def extract_text_from_xml(xml_path: str) -> str:
    """Extract text from XML file."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Extract all text content
        text_content = []
        for elem in root.iter():
            if elem.text and elem.text.strip():
                text_content.append(elem.text.strip())
            if elem.tail and elem.tail.strip():
                text_content.append(elem.tail.strip())
        
        return ' '.join(text_content)
    except Exception as e:
        print(f"Error parsing XML {xml_path}: {e}")
        return ""

def remove_references_section(text: str) -> str:
    """Remove references section from text to avoid false positives."""
    lines = text.split('\n')
    cut_index = -1
    
    # Look backwards from end of document for references section
    for i in range(len(lines) - 1, max(0, int(len(lines) * 0.6)), -1):
        line = lines[i].strip()
        
        # Reference section patterns
        reference_patterns = [
            r'^REFERENCES?$', r'^\d+\.?\s+REFERENCES?$', r'^References?:?$',
            r'^BIBLIOGRAPHY$', r'^Bibliography:?$', r'^Literature\s+Cited$',
            r'^Works\s+Cited$', r'^ACKNOWLEDGMENTS?$', r'^Acknowledgments?$'
        ]
        
        if any(re.match(pattern, line, re.IGNORECASE) for pattern in reference_patterns):
            # Check following lines for citation patterns
            following_lines = lines[i+1:i+5]
            has_citations = any(
                re.search(r'\(\d{4}\)|^\[\d+\]|^\d+\.|doi:|et al', follow_line, re.IGNORECASE)
                for follow_line in following_lines if follow_line.strip()
            )
            
            if has_citations or i >= len(lines) - 5:
                cut_index = i
                break
    
    if cut_index != -1:
        return '\n'.join(lines[:cut_index]).strip()
    return text.strip()

print("âœ… Text extraction functions defined!")


# Feature Engineering - Pattern Definitions

class DataReferencePatterns:
    """Comprehensive patterns for identifying data references."""
    
    def __init__(self):
        # DOI pattern - matches standard DOI format
        self.doi_pattern = re.compile(
            r"(?:(?:https?://)?(?:dx\.)?doi\.org/|doi:\s*|DOI:\s*)?(?=10\.)\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)",
            re.IGNORECASE
        )
        
        # Database-specific patterns
        self.database_patterns = {
            'dryad': re.compile(r"(?:dryad\.org/|dryad:)\s*(?:doi:)?\s*(10\.5061/dryad\.[^\s\)\]\>,;]+)", re.IGNORECASE),
            'zenodo': re.compile(r"(?:zenodo\.org/record/|zenodo:)\s*(\d+)|(?=10\.5281/zenodo\.)\b(10\.5281/zenodo\.\d+)", re.IGNORECASE),
            'figshare': re.compile(r"(?:figshare\.com/|figshare:)\s*(\d+)|(?=10\.6084/m9\.figshare\.)\b(10\.6084/m9\.figshare\.\d+)", re.IGNORECASE),
            'ncbi': re.compile(r"\b(?:GSE\d+|GSM\d+|GPL\d+|GDS\d+|SR[APRX]\d+|ERR\d+|DRR\d+|PRJ[NAED][A-Z]?\d+|SAM[ND]\d+)\b", re.IGNORECASE),
            'arrayexpress': re.compile(r"\bE-[A-Z]+-\d+\b", re.IGNORECASE),
            'pride': re.compile(r"\bPXD\d{6}\b", re.IGNORECASE),
            'empiar': re.compile(r"\bEMPIAR-\d{4,5}\b", re.IGNORECASE),
            'pdb': re.compile(r"\b(?:PDB\s*[:\s]*)?([1-9][A-Z0-9]{3})\b", re.IGNORECASE),
            'chembl': re.compile(r"\bCHEMBL\d+\b", re.IGNORECASE),
            'uniprot': re.compile(r"\b(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9])\b", re.IGNORECASE),
            'ensembl': re.compile(r"\bENS[A-Z]{0,6}[GT]\d{11}\b", re.IGNORECASE),
            'refseq': re.compile(r"\b(?:N[MC]_\d+(?:\.\d+)?|XM_\d+|XP_\d+)\b", re.IGNORECASE),
            'gisaid': re.compile(r"\bEPI(?:_ISL_)?\d+\b", re.IGNORECASE),
            'ipr': re.compile(r"\bIPR\d{6}\b", re.IGNORECASE),
            'pfam': re.compile(r"\bPF\d{5}\b", re.IGNORECASE)
        }
        
        # Classification keywords
        self.primary_keywords = [
            'we deposited', 'our data', 'this study', 'newly generated', 'experimental data',
            'raw data from this', 'data presented here', 'data from this study', 'original data',
            'we submitted', 'submitted to', 'deposited in', 'archived in', 'uploaded to',
            'data are available', 'datasets are available', 'accession number', 'accession code'
        ]
        
        self.secondary_keywords = [
            'publicly available', 'previously published', 'downloaded from', 'obtained from',
            'retrieved from', 'reference data', 'comparative data', 'external data',
            'published data', 'existing data', 'database', 'repository'
        ]
        
        # Repository-specific DOI prefixes (typically Primary)
        self.data_repository_prefixes = [
            '10.5061',   # Dryad
            '10.5281',   # Zenodo
            '10.6084',   # Figshare
            '10.24433',  # Mendeley Data
            '10.17632',  # Mendeley Data
            '10.6073',   # PASTA
            '10.5066',   # USGS
            '10.7937'    # TCIA
        ]
        
        # Publisher prefixes (typically literature, not data)
        self.publisher_prefixes = [
            '10.1007', '10.1002', '10.1016', '10.1021', '10.1038', '10.1056',
            '10.1073', '10.1080', '10.1093', '10.1101', '10.1186', '10.1371',
            '10.1111', '10.5194', '10.3390', '10.1126'
        ]

patterns = DataReferencePatterns()
print("âœ… Data reference patterns initialized!")
print(f"- DOI pattern ready")
print(f"- {len(patterns.database_patterns)} database patterns")
print(f"- {len(patterns.primary_keywords)} primary keywords")
print(f"- {len(patterns.secondary_keywords)} secondary keywords")


# Model Training - Reference Extraction and Classification Logic

def extract_context_window(text: str, match_start: int, match_end: int, window_size: int = 200) -> str:
    """Extract context window around a match."""
    start = max(0, match_start - window_size)
    end = min(len(text), match_end + window_size)
    return text[start:end].strip()

def normalize_doi(doi: str) -> str:
    """Normalize DOI to full URL format."""
    doi = doi.strip()
    if doi.startswith('https://doi.org/'):
        return doi
    elif doi.startswith('doi.org/'):
        return f'https://{doi}'
    elif doi.startswith('10.'):
        return f'https://doi.org/{doi}'
    return doi

def classify_reference_type(context: str, dataset_id: str, patterns: DataReferencePatterns) -> str:
    """
    Classify whether a data reference is Primary or Secondary.
    
    Args:
        context: Text context around the data reference
        dataset_id: The dataset identifier
        patterns: Pattern definitions
        
    Returns:
        'Primary' or 'Secondary'
    """
    context_lower = context.lower()
    
    # Check for data repository DOI prefixes (usually Primary)
    if dataset_id.startswith('https://doi.org/'):
        doi_part = dataset_id.replace('https://doi.org/', '')
        for prefix in patterns.data_repository_prefixes:
            if doi_part.startswith(prefix):
                return 'Primary'
    
    # Score based on keywords
    primary_score = sum(1 for keyword in patterns.primary_keywords if keyword in context_lower)
    secondary_score = sum(1 for keyword in patterns.secondary_keywords if keyword in context_lower)
    
    # Additional heuristics
    if any(phrase in context_lower for phrase in ['we deposited', 'our data', 'this study']):
        primary_score += 2
    
    if any(phrase in context_lower for phrase in ['downloaded from', 'obtained from', 'publicly available']):
        secondary_score += 2
    
    # Default classification based on dataset type
    if dataset_id.startswith('https://doi.org/'):
        # DOI - check if it's a known data repository
        doi_part = dataset_id.replace('https://doi.org/', '')
        if any(doi_part.startswith(prefix) for prefix in patterns.data_repository_prefixes):
            return 'Primary' if primary_score >= secondary_score else 'Primary'
        else:
            return 'Secondary'  # Unknown DOI, likely external data
    else:
        # Accession ID - likely secondary unless context indicates otherwise
        return 'Primary' if primary_score > secondary_score else 'Secondary'

def extract_references_from_text(text: str, article_id: str, patterns: DataReferencePatterns) -> List[Dict]:
    """Extract all data references from text."""
    references = []
    
    # Remove references section to reduce false positives
    text = remove_references_section(text)
    
    # Extract DOI references
    for match in patterns.doi_pattern.finditer(text):
        doi = match.group(1)
        
        # Skip if this is the article's own DOI
        if doi.replace('/', '_') in article_id:
            continue
        
        # Check if it's a publisher DOI (likely literature, not data)
        is_publisher_doi = any(doi.startswith(prefix) for prefix in patterns.publisher_prefixes)
        
        # Get context window
        context = extract_context_window(text, match.start(), match.end())
        
        # Check if context suggests this is data-related
        has_data_context = any(keyword in context.lower() for keyword in 
                             ['data', 'dataset', 'repository', 'deposited', 'archived', 'supplementary'])
        
        # Skip publisher DOIs unless they have strong data context
        if is_publisher_doi and not has_data_context:
            continue
        
        normalized_doi = normalize_doi(doi)
        ref_type = classify_reference_type(context, normalized_doi, patterns)
        
        references.append({
            'article_id': article_id,
            'dataset_id': normalized_doi,
            'type': ref_type,
            'context': context
        })
    
    # Extract database-specific accession IDs
    for db_name, pattern in patterns.database_patterns.items():
        for match in pattern.finditer(text):
            accession = match.group().strip()
            
            # Clean up accession ID
            if db_name == 'pdb':
                accession = re.sub(r'^PDB\s*[:\s]*', '', accession, flags=re.IGNORECASE)
            
            context = extract_context_window(text, match.start(), match.end())
            ref_type = classify_reference_type(context, accession, patterns)
            
            references.append({
                'article_id': article_id,
                'dataset_id': accession,
                'type': ref_type,
                'context': context
            })
    
    return references

print("âœ… Reference extraction and classification functions ready!")


# Model Evaluation - Process All Documents

def process_all_documents(pdf_dir: Path, xml_dir: Path, patterns: DataReferencePatterns) -> pd.DataFrame:
    """Process all documents and extract data references."""
    all_references = []

    # Get list of PDF files
    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"ğŸ”„ Processing {len(pdf_files)} PDF files...")

    processed_count = 0
    error_count = 0

    for pdf_file in pdf_files:
        try:
            article_id = pdf_file.stem

            # Try to extract text from XML first (usually cleaner), then PDF
            xml_file = xml_dir / f"{article_id}.xml"
            text = ""

            if xml_file.exists():
                text = extract_text_from_xml(str(xml_file))

            if not text.strip():  # If XML extraction failed or no XML file
                text = extract_text_from_pdf(str(pdf_file))

            if text.strip():
                references = extract_references_from_text(text, article_id, patterns)
                all_references.extend(references)
                processed_count += 1
            else:
                error_count += 1

        except Exception as e:
            error_count += 1
            if error_count <= 5:  # Only print first few errors
                print(f"âš ï¸� Error processing {pdf_file.name}: {e}")

        # Progress update
        if (processed_count + error_count) % 100 == 0 and (processed_count + error_count) > 0:
            print(f"ğŸ“Š Processed: {processed_count}, Errors: {error_count}")

    print(f"âœ… Completed processing!")
    print(f"ğŸ“Š Successfully processed: {processed_count}")
    print(f"âš ï¸� Errors: {error_count}")

    # Convert to DataFrame and deduplicate
    df = pd.DataFrame(all_references)

    if not df.empty:
        # Remove duplicates (same article_id, dataset_id combination)
        initial_count = len(df)
        df = df.drop_duplicates(subset=['article_id', 'dataset_id'])
        final_count = len(df)

        print(f"ğŸ”� Removed {initial_count - final_count} duplicate references")
        print(f"ğŸ“ˆ Final unique references: {final_count}")

        # Sort by article_id for consistent output
        df = df.sort_values(['article_id', 'dataset_id']).reset_index(drop=True)

        # Display summary statistics
        print(f"\nğŸ“Š Reference Summary:")
        print(f"- Total references: {len(df)}")
        print(f"- Primary references: {len(df[df['type'] == 'Primary'])}")
        print(f"- Secondary references: {len(df[df['type'] == 'Secondary'])}")
        print(f"- Unique articles with references: {df['article_id'].nunique()}")
        print(f"- DOI references: {len(df[df['dataset_id'].str.startswith('https://doi.org/')])}")

    else:
        print("âš ï¸� No data references extracted! Possible reasons:")
        print("  â€¢ Regex patterns too strict")
        print("  â€¢ Text extraction failed (empty text)")
        print("  â€¢ Dataset paths incorrect (no PDFs loaded)")
        print("  â€¢ References only in removed sections")
        # Quick check: show one small text sample if any PDF had text
        sample_pdf = next(iter(pdf_files), None)
        if sample_pdf:
            try:
                raw_sample = extract_text_from_pdf(str(sample_pdf))[:500].replace('\n',' ') if sample_pdf else ''
                print("  â€¢ Sample first 500 chars of first PDF text:")
                print(raw_sample)
            except Exception as e:
                print(f"  â€¢ Could not read sample PDF: {e}")
        df = pd.DataFrame(columns=['article_id', 'dataset_id', 'type'])

    return df

# Process all documents
results_df = process_all_documents(PDF_DIR, XML_DIR, patterns)


# Local Validation (only when NOT Kaggle rerun and train labels exist)
if not IS_KAGGLE:
    labels_path = DATA_DIR / 'train_labels.csv'
    if labels_path.exists() and not results_df.empty:
        gt = pd.read_csv(labels_path)
        gt = gt[gt['type'] != 'Missing'].copy()
        # Normalize prediction casing
        preds = results_df[['article_id','dataset_id','type']].copy()
        # Merge for TP calculation
        merged = gt.merge(preds, on=['article_id','dataset_id','type'])
        tp = len(merged)
        fp = len(preds) - tp
        fn = len(gt) - tp
        precision = tp/(tp+fp) if (tp+fp)>0 else 0
        recall = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
        print(f"Validation â€” TP:{tp} FP:{fp} FN:{fn}")
        print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
    else:
        print('No validation performed (either Kaggle test environment or no labels/results).')


# Generate Predictions - Additional Analysis and Validation

if not results_df.empty:
    print("ğŸ”� Additional Analysis:")
    
    # Analyze reference types
    type_counts = results_df['type'].value_counts()
    print(f"\nğŸ“Š Reference Type Distribution:")
    for ref_type, count in type_counts.items():
        percentage = (count / len(results_df)) * 100
        print(f"- {ref_type}: {count} ({percentage:.1f}%)")
    
    # Analyze dataset ID patterns
    doi_refs = results_df[results_df['dataset_id'].str.startswith('https://doi.org/')]
    accession_refs = results_df[~results_df['dataset_id'].str.startswith('https://doi.org/')]
    
    print(f"\nğŸ”— Dataset ID Types:")
    print(f"- DOI references: {len(doi_refs)} ({len(doi_refs)/len(results_df)*100:.1f}%)")
    print(f"- Accession IDs: {len(accession_refs)} ({len(accession_refs)/len(results_df)*100:.1f}%)")
    
    # Show sample references
    print(f"\nğŸ“� Sample References:")
    sample_refs = results_df.head(10)[['article_id', 'dataset_id', 'type']]
    for idx, row in sample_refs.iterrows():
        print(f"- {row['article_id']} â†’ {row['dataset_id']} ({row['type']})")
    
    # Validate DOI format
    invalid_dois = doi_refs[~doi_refs['dataset_id'].str.match(r'https://doi\.org/10\.\d+/.+')]
    if len(invalid_dois) > 0:
        print(f"âš ï¸� Found {len(invalid_dois)} potentially invalid DOI formats")
    else:
        print("âœ… All DOI formats appear valid")
        
else:
    print("âš ï¸� No references found - will create empty submission")


# Create Submission File

def create_submission_file(df: pd.DataFrame, output_path: str):
    """Create submission file in the required format."""
    
    if not df.empty:
        # Create submission format
        submission_df = df[['article_id', 'dataset_id', 'type']].copy()
        submission_df.insert(0, 'row_id', range(len(submission_df)))
        
        # Final validation
        required_cols = ['row_id', 'article_id', 'dataset_id', 'type']
        assert all(col in submission_df.columns for col in required_cols), "Missing required columns"
        
        # Check for valid types
        valid_types = {'Primary', 'Secondary'}
        invalid_types = set(submission_df['type'].unique()) - valid_types
        if invalid_types:
            print(f"âš ï¸� Found invalid reference types: {invalid_types}")
            submission_df = submission_df[submission_df['type'].isin(valid_types)]
        
        # Save submission
        submission_df.to_csv(output_path, index=False)
        
        print(f"âœ… Submission saved to {output_path}")
        print(f"ğŸ“Š Final submission statistics:")
        print(f"- Total predictions: {len(submission_df)}")
        print(f"- Primary references: {len(submission_df[submission_df['type'] == 'Primary'])}")
        print(f"- Secondary references: {len(submission_df[submission_df['type'] == 'Secondary'])}")
        print(f"- Unique articles: {submission_df['article_id'].nunique()}")
        
        # Show submission format
        print(f"\nğŸ“� Submission preview:")
        print(submission_df.head())
        
        return submission_df
        
    else:
        # Create empty submission
        empty_submission = pd.DataFrame(columns=['row_id', 'article_id', 'dataset_id', 'type'])
        empty_submission.to_csv(output_path, index=False)
        print(f"âš ï¸� Created empty submission file at {output_path}")
        return empty_submission

# Create the final submission
final_submission = create_submission_file(results_df, OUTPUT_PATH)

print(f"\nğŸ�¯ Submission ready for upload!")
print(f"ğŸ“� File location: {OUTPUT_PATH}")

