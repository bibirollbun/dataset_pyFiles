# Suppress all warnings and set up environment
# Environment Setup
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

# Install required packages
!pip install pdfminer.six xmltodict transformers torch -q
!pip install pymupdf -q



# Core Imports
import re
import pandas as pd
import fitz  # for using PyMuPDF
import xmltodict
import torch
from tqdm.auto import tqdm
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer


# GPU Configuration
if torch.cuda.is_available():
    if torch.cuda.device_count() >= 2:
        print(f"âš¡ Dual GPU Detected (Devices: {torch.cuda.device_count()})")
    else:
        print(f"âš¡ Single GPU Detected ({torch.cuda.get_device_name(0)})")
else:
    print("âš ï¸� No GPU detected - Falling back to CPU")


# Initialize Models with GPU Optimization
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Multi-GPU Setup
if torch.cuda.device_count() >= 2:
    model = torch.nn.DataParallel(base_model)
model = model.to('cuda').half()  # Use FP16 for memory efficiency

classifier = pipeline(
    "text-classification",
    model=base_model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    framework="pt",
    truncation=True,
    max_length=512
)


# File Path Configuration
INPUT_DIR = '/kaggle/input/make-data-count-finding-data-references/'
TEST_PDF_DIR = os.path.join(INPUT_DIR, 'test/PDF')
TEST_XML_DIR = os.path.join(INPUT_DIR, 'test/XML')


# performing text extraction

def extract_text_from_pdf(pdf_path):
    """Optimized PDF text extraction with PyMuPDF"""
    try:
        text = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text.append(page.get_text("text"))
        return "\n".join(text)
    except Exception as e:
        print(f"âš ï¸� PDF Error ({os.path.basename(pdf_path)}): {str(e)}")
        return ""

def extract_text_from_xml(xml_path):
    """XML text extraction with error handling"""
    try:
        with open(xml_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()  # Fallback to raw text if parsing fails
    except Exception as e:
        print(f"âš ï¸� XML Error ({os.path.basename(xml_path)}): {str(e)}")
        return ""


# DATASET DETECTION PATTERNS

DOI_PATTERNS = [
    r'\b(?:doi|DOI|Doi)[\s:]*([\S]+)',  # More lenient DOI matching
    r'10\.\d{4,}\/[\S]+'  # Raw DOI pattern
]

ACCESSION_PATTERNS = {
    'GEO': r'\b(?:GSE|GSM|GDS)\d+\b',
    'ArrayExpress': r'\bE-[A-Z]{3,}-\d+\b',
    'ENA': r'\b(?:PRJ[EDN]\d+|ERX\d+)\b',
    'PDB': r'\b(?:PDB|pdb)[\s\-_]?([1-9][0-9A-Z]{3})\b',
    'SRA': r'\b(?:SR[APRX]\d+|SAM[END]\d+)\b'
}

def clean_doi(doi):
    """Standardizes DOI formatting"""
    if not doi:
        return ""
    doi = doi.strip(' .)')  # Remove trailing punctuation
    if doi.startswith('https://doi.org/'):
        return doi.lower()
    if doi.startswith('10.'):
        return f"https://doi.org/{doi.lower()}"
    return doi

def find_dataset_mentions(text):
    """Comprehensive dataset ID detection with standardization"""
    found = set()
    
    # DOI detection
    for pattern in DOI_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            doi = match.group(1) if match.groups() else match.group(0)
            doi = clean_doi(doi)
            found.add(doi)
    
    # Accession ID detection
    for repo, pattern in ACCESSION_PATTERNS.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            id = match.group(1) if match.groups() else match.group(0)
            if repo == 'PDB':
                id = f"PDB{id.upper()}"
            found.add(id.upper())
    
    return list(found)


# CLASSIFICATION SYSTEM

PRIMARY_KEYWORDS = ['generated', 'collected', 'produced', 'created', 'measured', 'our data']
SECONDARY_KEYWORDS = ['obtained', 'downloaded', 'retrieved', 'available', 'previous study']

# Enhance the classify_context function
def classify_context(text, dataset_id):
    """Improved classification with better context analysis"""
    # Find all mention positions
    mention_positions = [m.start() for m in re.finditer(re.escape(dataset_id), text, re.IGNORECASE)]
    
    if not mention_positions:
        return "Secondary"
    
    # Analyze each mention
    classifications = []
    for pos in mention_positions:
        context = text[max(0, pos-250):min(len(text), pos+250)].lower()
        
        # Strong primary indicators
        primary_terms = ['generated', 'collected', 'produced', 'our data', 
                        'experimental data', 'this study', 'we measured']
        if any(term in context for term in primary_terms):
            classifications.append("Primary")
            continue
            
        # Strong secondary indicators
        secondary_terms = ['obtained from', 'downloaded from', 'retrieved from',
                          'previous study', 'available at', 'public dataset']
        if any(term in context for term in secondary_terms):
            classifications.append("Secondary")
            continue
            
        # Model classification for ambiguous cases
        prompt = f"""Is this dataset primary (generated by authors) or secondary (obtained externally)?
Dataset: {dataset_id}
Context: "{context}"
Answer (Primary/Secondary):"""
        
        try:
            result = classifier(prompt, truncation=True, max_length=512)
            label = "Primary" if result[0]['label'] in ["LABEL_0", "Primary"] else "Secondary"
            classifications.append(label)
        except:
            classifications.append("Secondary")
    
    # Return most common classification
    return max(set(classifications), key=classifications.count)



# FILE HANDLING FUNCTIONS

def get_paper_ids():
    """Extracts unique paper IDs from filenames"""
    pdf_ids = [f.split('_')[0] for f in os.listdir(TEST_PDF_DIR)]
    xml_ids = [f.split('_')[0] for f in os.listdir(TEST_XML_DIR)]
    return list(set(pdf_ids + xml_ids))

def find_actual_file(paper_id):
    """Finds files starting with paper_id and correct extension"""
    for f in os.listdir(TEST_PDF_DIR):
        if f.startswith(paper_id) and f.lower().endswith('.pdf'):
            return os.path.join(TEST_PDF_DIR, f)
    for f in os.listdir(TEST_XML_DIR):
        if f.startswith(paper_id) and f.lower().endswith('.xml'):
            return os.path.join(TEST_XML_DIR, f)
    return None



# PAPER PROCESSING FUNCTIONS

def process_paper(paper_id):
    """Processes a single paper with comprehensive error handling"""
    file_path = find_actual_file(paper_id)
    
    if not file_path:
        return [{
            'article_id': paper_id,
            'dataset_id': 'FILE_NOT_FOUND',
            'type': 'ERROR'
        }]
    
    try:
        # Extract text based on file type
        if file_path.endswith('.xml'):
            text = extract_text_from_xml(file_path)
        else:
            text = extract_text_from_pdf(file_path)
            
        if not text.strip():
            return [{
                'article_id': paper_id,
                'dataset_id': 'EMPTY_CONTENT',
                'type': 'ERROR'
            }]
            
        # Find and process dataset mentions
        mentions = find_dataset_mentions(text)
        unique_mentions = list(set(mentions))  # Deduplicate
        
        results = []
        for mention in unique_mentions:
            results.append({
                'article_id': paper_id,
                'dataset_id': mention,
                'type': classify_context(text, mention)
            })
        
        return results if results else [{
            'article_id': paper_id,
            'dataset_id': 'NO_MENTIONS_FOUND',
            'type': 'NONE'
        }]

        filtered_mentions = [
        m for m in mentions 
        if not (m.startswith('PDB1') and m[4:].isdigit() and len(m) == 8)
        ]
    
        if not filtered_mentions:
            return [{
            'article_id': paper_id,
            'dataset_id': 'NO_VALID_MENTIONS',
            'type': 'NONE'
        }]
            
    except Exception as e:
            return [{
            'article_id': paper_id,
            'dataset_id': f'ERROR_{type(e).__name__}',
            'type': 'ERROR'
        }]

def clean_submission(df):
    """Final cleanup before saving"""
    # Clean DOI formatting
    df['dataset_id'] = df['dataset_id'].apply(clean_doi)
    
    # Remove error rows from final submission

    df = df[~(
        df['dataset_id'].str.startswith('PDB') & 
        df['dataset_id'].str[3:].str.match(r'^\d{4}$')
    )]
    df = df[~df['dataset_id'].str.startswith('ERROR_')]
    df = df[df['dataset_id'] != 'FILE_NOT_FOUND']
    df = df[df['dataset_id'] != 'EMPTY_CONTENT']
    df = df[df['dataset_id'] != 'NO_MENTIONS_FOUND']
    
    # Deduplicate identical article_id + dataset_id pairs
    df = df.drop_duplicates(['article_id', 'dataset_id'])
    
    # Reset row IDs
    df['row_id'] = range(len(df))
    return df[['row_id', 'article_id', 'dataset_id', 'type']]



# MAIN EXECUTION

if __name__ == "__main__":
    # Get all test paper IDs
    test_ids = get_paper_ids()
    print(f"ğŸ“„ Found {len(test_ids)} papers to process")
    
    # Process papers
    all_results = []
    for paper_id in tqdm(test_ids, desc="Processing papers"):
        all_results.extend(process_paper(paper_id))
    
    # Create and clean submission
    submission_df = pd.DataFrame(all_results)
    clean_df = clean_submission(submission_df)
    
    # Save results
    clean_df.to_csv('/kaggle/working/submission.csv', index=False)
    
    print("\nâœ… Submission created with the following predictions:")
    print(clean_df.head())
    print(f"\nTotal valid dataset mentions found: {len(clean_df)}")

