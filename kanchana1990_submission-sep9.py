# ==================================================================================
# Cell 1 (Final Pivot): Rules-Based Engine Setup
# We are pivoting away from the broken ML libraries.
# This cell installs only the necessary components for a pure regex/rules-based approach.
# ==================================================================================
import os
import sys
import subprocess
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

print("--- Pivoting to Rules-Based Approach ---")
print("--- Installing Core Dependencies ---")

# --- 1. Install Essential Libraries from Wheels ---
wheels_dir = '/kaggle/input/wheels-cpu/wheels_cpu'

if not os.path.isdir(wheels_dir):
    print(f"ERROR: The directory {wheels_dir} was not found.")
else:
    # We only need pymupdf, lxml, and rapidfuzz for our new approach.
    # We will install them forcefully to ensure they are available.
    essential_libs = ['pymupdf', 'lxml', 'rapidfuzz', 'numpy']
    all_wheels = {os.path.basename(f): os.path.join(wheels_dir, f) for f in os.listdir(wheels_dir) if f.endswith('.whl')}

    for lib_prefix in essential_libs:
        for wheel_name, wheel_path in all_wheels.items():
            if wheel_name.startswith(lib_prefix):
                print(f"  -> Installing {wheel_name}")
                # Force reinstall without dependencies to ensure a clean state
                subprocess.run([sys.executable, '-m', 'pip', 'install', '--force-reinstall', '--no-deps', wheel_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print("\n--- Essential libraries have been installed. ---")


# --- 2. Import Libraries for Rules-Based Solution ---
print("\n--- Importing libraries for the new plan... ---")
try:
    # File Parsing and Data Handling
    import pandas as pd
    import numpy as np
    import pymupdf
    from lxml import etree
    
    # Utilities
    import re
    from tqdm.notebook import tqdm
    import json

    # Print versions to confirm success
    print("\n--- Library Version Verification ---")
    print(f"pandas: {pd.__version__}")
    print(f"numpy: {np.__version__}")
    print(f"pymupdf: {pymupdf.version[0]}")
    print(f"lxml: {etree.__version__}")
    print("------------------------------------")
    print("\n✅ Environment is ready for a rules-based solution.")
    print("We will proceed without scikit-learn and lightgbm.")

except ImportError as e:
    print(f"\n❌ A critical error occurred during import: {e}")
    print("The environment is unstable. Cannot proceed.")
except Exception as e:
    print(f"\n❌ An unexpected error occurred: {e}")



#cell2
import os
import re
import polars as pl
from lxml import etree
from tqdm.notebook import tqdm
import warnings

# --- 1. SETUP AND CONFIGURATION ---
# This cell assumes 'polars', 'lxml', and 'pymupdf' are installed from Cell 1.
try:
    import fitz  # PyMuPDF
except ImportError:
    print("FATAL ERROR: PyMuPDF (fitz) is not installed. Please run your installation cell first.")
    exit()

warnings.filterwarnings('ignore')
print("--- Polars-Powered Engine Configuration ---")

# Define file paths
BASE_PATH = '/kaggle/input/make-data-count-finding-data-references/'
PDF_DIR = os.path.join(BASE_PATH, 'test/PDF')
XML_DIR = os.path.join(BASE_PATH, 'test/XML')
OUTPUT_CSV = 'submission.csv'
TARGET_ROW_COUNT = 500

# Define regex and keywords for the high-precision scoring model
IDENTIFIER_RE = re.compile(r'(10\.\d{4,9}/[-._;()/:A-Z0-9]+|https?://[^\s)\]<>\"]+)', re.IGNORECASE)
TIER_1_REPOS = ['zenodo', 'figshare', 'dryad', 'osf.io', 'github.com']
TIER_2_REPOS = ['ncbi.nlm.nih.gov', 'ebi.ac.uk', 'pangea.de']
TIER_1_PHRASES = ['data for this study', 'raw data have been deposited', 'data are available at', 'can be found at', 'data availability statement']
NEGATIVE_KEYWORDS = ['software', 'pipeline', 'et al', 'retrieved from', 'documentation', 'version', 'toolkit']
EXCLUDED_DOMAINS = ['creativecommons.org', 'wiley.com', 'springer.com', 'elsevier.com', 'oup.com']

print(f"Targeting {TARGET_ROW_COUNT} high-quality rows for {OUTPUT_CSV}")

# --- 2. CORE FUNCTIONS: TEXT EXTRACTION AND CANDIDATE SCORING ---

def get_text_from_file(pub_id):
    """Extracts text from PDF, falling back to XML if PDF fails or is empty."""
    text = ""
    try:
        pdf_path = os.path.join(PDF_DIR, f"{pub_id}.pdf")
        if os.path.exists(pdf_path):
            with fitz.open(pdf_path) as doc:
                text = "".join(page.get_text() for page in doc).lower()
    except Exception:
        text = ""

    if len(text) < 200:
        try:
            xml_path = os.path.join(XML_DIR, f"{pub_id}.xml")
            if os.path.exists(xml_path):
                tree = etree.parse(xml_path)
                text = " ".join(tree.xpath('//text()')).lower()
        except Exception:
            pass
            
    return re.sub(r'\s+', ' ', text)

def mine_and_score_candidates(pub_id, text):
    """Finds all potential dataset identifiers and assigns a quality score."""
    candidates = []
    for match in IDENTIFIER_RE.finditer(text):
        identifier = match.group(0).rstrip(').,;\'"')
        
        if len(identifier) < 15 or pub_id.lower() in identifier.lower() or any(domain in identifier for domain in EXCLUDED_DOMAINS):
            continue

        score = 0
        context = text[max(0, match.start() - 200):min(len(text), match.end() + 200)]
        
        if any(repo in identifier.lower() for repo in TIER_1_REPOS): score += 1000
        elif any(repo in identifier.lower() for repo in TIER_2_REPOS): score += 500
        if any(phrase in context for phrase in TIER_1_PHRASES): score += 200
        if any(keyword in context for keyword in NEGATIVE_KEYWORDS): score -= 100
        score += len(identifier)
        
        citation_type = 'Primary' if score > 500 else 'Secondary'
        
        candidates.append({
            'article_id': pub_id,
            'dataset_id': identifier,
            'type': citation_type,
            'quality_score': score
        })
    return candidates

# --- 3. MAIN PROCESSING LOOP ---
print("\n--- Starting Data Mining with Polars ---")
all_candidates = []
test_files = [f.replace('.pdf', '') for f in os.listdir(PDF_DIR)]

for pub_id in tqdm(test_files, desc="Processing Articles"):
    document_text = get_text_from_file(pub_id)
    if document_text:
        all_candidates.extend(mine_and_score_candidates(pub_id, document_text))

# --- 4. FINAL SELECTION AND SUBMISSION FILE CREATION ---
print(f"\n--- Filtering and Ranking {len(all_candidates)} Found Candidates ---")

# Create a Polars DataFrame directly from the list of dictionaries
candidates_df = pl.DataFrame(all_candidates)

# Sort by score and remove duplicates, keeping the highest-scored entry
candidates_df = candidates_df.sort('quality_score', descending=True).unique(subset=['article_id', 'dataset_id'], keep='first')

# Select the top N candidates
final_df = candidates_df.head(TARGET_ROW_COUNT)

# Format for submission using Polars expressions
final_df = final_df.with_columns(
    (pl.col("article_id") + "_" + pl.col("dataset_id")).alias("row_id")
).select(
    ["row_id", "article_id", "dataset_id", "type"]
)

# Rename columns to match the sample submission file
final_df = final_df.rename({"article_id": "article_id", "dataset_id": "dataset_id", "type": "type"})

# Save the final file
final_df.write_csv(OUTPUT_CSV)

print(f"\n--- ✅ High-Quality Polars Submission Complete ---")
print(f"Generated '{OUTPUT_CSV}' with the top {len(final_df)} candidates.")
print("\n--- Final Submission Preview ---")
print(final_df.head())


