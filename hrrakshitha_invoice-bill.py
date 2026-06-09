# ==============================================================================
#  INSTALL REQUIRED PACKAGES (FOR KAGGLE/COLAB ENVIRONMENTS)
# ==============================================================================
import subprocess
import sys

# List of packages to install
required_packages = ['pandas', 'pypdfium2', 'pytesseract']

print("Installing required packages...")

for package in required_packages:
    try:
        __import__(package)
        print(f"Package '{package}' is already installed.")
    except ImportError:
        print(f"Installing '{package}'...")
        try:
            # Use the environment's pip
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}.")
        except Exception as e:
            print(f"Failed to install {package}. Error: {e}")
            sys.exit(1) # Exit if critical package installation fails

# Additionally, tesseract needs to be installed as an external system tool
# On a full environment like a VM, you'd use apt-get/choco. 
# In a standard Kaggle environment, you'd typically need a custom environment
# or check if it's pre-installed. For simplicity, we'll assume the 
# Tesseract path config later is sufficient if tesseract is pre-installed 
# or if a user provides an environment with it. 
# **NOTE: For this code to fully execute in Kaggle, Tesseract-OCR MUST be 
# available as a system dependency, which is often not the case without extra setup.**
# We will comment out the Windows-specific path setting.
print("Package installation complete.")

# ==============================================================================
#  AUTOMATED INVOICE PROCESSOR (Enhanced Version)
# ==============================================================================

import os
import re
import shutil
import sys
import pandas as pd
from pathlib import Path
import pypdfium2 as pdfium
import pytesseract
import logging  # NEW: For structured error tracking
from datetime import datetime # Already imported by pandas, but kept for clarity

# ==============================================================================
#  CONFIG
# ==============================================================================
# 1. TESSERACT PATH (Required for OCR)
# The default path is usually sufficient in Unix/Kaggle environments if Tesseract is installed.
# We comment out the Windows path setting.
# TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CMD = "tesseract" # Assuming Tesseract is in PATH or its default location

try:
    # Set the command. If Tesseract isn't in PATH, this will fail later.
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD 
    # Check if Tesseract is actually accessible (optional but good practice)
    # pytesseract.get_tesseract_version() 
except pytesseract.TesseractNotFoundError:
    print("\n--- WARNING: Tesseract-OCR not found. OCR functions will fail. ---")
    print("Ensure Tesseract is installed and available in the system PATH.")
    # We will let the code continue to show other functionalities, but OCR will break.

# 2. FOLDERS
IN_DIR   = Path("invoices/in")
OUT_DIR  = Path("invoices/out")
PROC_DIR = Path("invoices/processed")
CSV_FILE = OUT_DIR / "invoices.csv"

# 3. REGEX
INV_NUM_RE = re.compile(r"(invoice\s*(?:#|no|number)?[:\s]*)([a-z0-9\-/]+)", re.I)
DATE_RE    = re.compile(r"(invoice\s*date[:\s]*|date[:\s]*)([\d]{1,2}[/-][\d]{1,2}[/-][\d]{2,4})", re.I)
TOTAL_RE = re.compile(r"(?:total|balance|amount due)[:\s]*(?:[€\\$£]\s*)?(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)\s*(?:[€\\$£])?", re.I)

# ==============================================================================
#  HELPER FUNCTIONS
# ==============================================================================
def configure_logging():    
    """Sets up file and console logging."""
    logging.basicConfig(
        filename='invoice_processor.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    # Also log to console for immediate feedback
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console.setFormatter(formatter)
    # Ensure handler isn't added multiple times if main is called repeatedly
    if not logging.getLogger('').handlers:
        logging.getLogger('').addHandler(console)
        
    logging.info("--- Invoice Processor Started ---")

def ocr_pdf_no_poppler(pdf_path: Path) -> str:    
    """Return raw text from scanned PDF using pypdfium2."""
    text = ""
    pdf = pdfium.PdfDocument(pdf_path)
    
    try:
        for i in range(len(pdf)):
            page = pdf.get_page(i)
            # Render page to a high-resolution bitmap
            bitmap = page.render(scale=3) 
            pil_image = bitmap.to_pil()
            # Perform OCR on the PIL image
            text += pytesseract.image_to_string(pil_image) 
    finally:
        pdf.close()
        
    return text

def extract_fields(text: str) -> dict:    
    """Return dict with best-guess fields, including debug logs for failures."""
    
    # 1. Get Match Objects
    inv_num_match = INV_NUM_RE.search(text)
    date_match = DATE_RE.search(text)
    total_match = TOTAL_RE.search(text)
    
    inv_num, inv_date, total = None, None, None
    
    # 2. SAFE EXTRACTION AND DEBUGGING
    
    if inv_num_match and len(inv_num_match.groups()) >= 2:
        inv_num = inv_num_match.group(2)
    else:
        logging.warning("Invoice Number Regex failed to find a value.")

    if date_match and len(date_match.groups()) >= 2:
        inv_date = date_match.group(2)
    else:
        logging.warning("Date Regex failed to find a value.")

    if total_match and len(total_match.groups()) >= 2:
        total = total_match.group(1) # Group 1 is the actual number part, Group 2 would be the currency if we captured it.
    else:
        logging.warning("Total Regex failed to find a value.")
        
    # 3. Return results
    return {
        "invoice_number": inv_num.strip() if inv_num else None,
        "invoice_date":   inv_date.strip() if inv_date else None,
        "total":          total.strip() if total else None,
        "raw_text":       text[:500]
    }

def clean_extracted_data(df: pd.DataFrame) -> pd.DataFrame:    
    """Converts total and date fields to their proper data types."""
    
    # 1. Clean Total (Convert string to float)
    def clean_total(total_str):
        if pd.isna(total_str) or total_str is None:
            return None
        
        # Remove thousands separators (commas or dots depending on the use)
        # Assuming the last comma/dot is the decimal point.
        # This is a robust way to handle both 1,234.56 and 1.234,56 
        if total_str.count(',') > 1 and total_str.count('.') == 1 and total_str.index(',') < total_str.index('.'):
            # Case: 1,234.56 (US/UK) - remove commas
            cleaned = total_str.replace(',', '')
        elif total_str.count('.') > 1 and total_str.count(',') == 1 and total_str.index('.') < total_str.index(','):
            # Case: 1.234,56 (European) - swap comma and dot, then remove dots
            cleaned = total_str.replace('.', '').replace(',', '.')
        else:
            # Default to basic cleaning: remove all non-decimal-or-digit characters except the last dot
            cleaned = re.sub(r'[^\d.]', '', total_str.replace(',', '.')) 
        
        try:
            return float(cleaned)
        except ValueError:
            logging.warning(f"Failed to convert total '{total_str}' to float.")
            return None

    df['total'] = df['total'].apply(clean_total)

    # 2. Clean Date (Convert string to datetime object)
    df['invoice_date'] = pd.to_datetime(
        df['invoice_date'], 
        errors='coerce', # Coerce invalid dates to NaT (Not a Time)
        dayfirst=False   # Assumes US standard date format (MM/DD/YYYY)
    )
    
    return df

def process_file(file_path: Path) -> dict:    
    """Processes a single file and returns the extracted row data."""
    logging.info(f"Processing {file_path.name} …")
    suffix = file_path.suffix.lower()
    
    try:
        if suffix == ".pdf":
            text = ocr_pdf_no_poppler(file_path)
        elif suffix in (".png", ".jpg", ".jpeg", ".tiff"):
            text = pytesseract.image_to_string(file_path)
        else:
            logging.warning(f"Skipping unsupported file: {suffix}")
            return None
            
        row = extract_fields(text)
        row["file_name"] = file_path.name
        return row
        
    except Exception as e:
        # Catch specific Tesseract error if it's the issue
        if "TesseractNotFoundError" in str(e):
             logging.error(f"Tesseract is not installed or not in PATH. Cannot process {file_path.name}.")
        else:
            logging.error(f"Fatal error during processing of {file_path.name}: {e}")
        return None

# ==============================================================================
#  MAIN EXECUTION
# ==============================================================================
def main():
    # 1. Configure Logging
    configure_logging()
    
    # 2. Setup folders
    for d in (IN_DIR, OUT_DIR, PROC_DIR):
        d.mkdir(parents=True, exist_ok=True)
        
    # --- IMPORTANT: Create a dummy file for demonstration since 
    # --- we cannot upload files to the 'invoices/in' folder in the environment.
    dummy_invoice_text = """
    Invoice # ABC-12345
    Date: 04/21/2023
    Subtotal: 100.00
    Tax: 5.00
    Amount Due: $105.00
    """
    dummy_file_path = IN_DIR / "dummy_invoice.txt"
    if not dummy_file_path.exists():
        with open(dummy_file_path, "w") as f:
            f.write(dummy_invoice_text)
        logging.info(f"Created dummy file for testing: {dummy_file_path}")
        
    # 3. Find files
    files = list(IN_DIR.iterdir())
    if not files:
        logging.info("No files in 'invoices/in'. Please add a PDF.")
        return

    rows = []
    
    # 4. Process files
    for file_path in files:
        if file_path.is_file():
            # A simple text file must be handled differently as it doesn't need OCR
            if file_path.suffix.lower() == ".txt":
                with open(file_path, 'r') as f:
                    text = f.read()
                row = extract_fields(text)
                row["file_name"] = file_path.name
            else:
                row = process_file(file_path)
                
            if row:
                rows.append(row)
                logging.info(f"   -> Success. File '{file_path.name}' remains in input folder.")

    # 5. Save and Clean results
    if rows:
        df = pd.DataFrame(rows)
        
        # NEW: Clean and validate data types
        df = clean_extracted_data(df)
        
        df.to_csv(CSV_FILE, index=False)
        logging.info(f"✅ Done! CSV saved to {CSV_FILE.resolve()}")
        
        # Print summary to console for immediate feedback
        print(f"\n✅ Done! CSV saved to {CSV_FILE.resolve()}")
        print("\n--- Extracted Data Sample ---")
        print(df[["file_name", "invoice_number", "invoice_date", "total"]].head())
    else:
        logging.info("No new data extracted.")
        print("\nNo new data extracted.")

if __name__ == "__main__":
    main()

