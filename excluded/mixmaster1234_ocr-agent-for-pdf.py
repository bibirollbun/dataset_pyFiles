!pip install /kaggle/input/dependencies/pymupdf-1.26.6-cp310-abi3-manylinux_2_28_x86_64.whl


!pip install /kaggle/input/dependencies/pypdf2-3.0.1-py3-none-any.whl


# Install dependencies if not already installed
#!pip install google-generativeai PyPDF2 pymupdf pytesseract python-dotenv pillow


import os
import time
import datetime as dt
import google.generativeai as genai
from PyPDF2 import PdfReader
import fitz # PyMuPDF
from PIL import Image
import io
import pytesseract
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("GEMINI_API_KEY")
print(secret_value_0)


def extract_text_from_pdf(pdf_path, use_ocr=False):
    """
    Extracts text from a PDF file. Returns the extracted text string.
    If use_ocr=True or PDF has no extractable text, OCR is applied.
    """
    extracted_text = ""

    if not use_ocr:
        try:
            reader = PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text:
                    extracted_text += f"\n--- Page {page_num} ---\n{text}\n"
            
            # If no text found, switch to OCR automatically
            if not extracted_text.strip():
                print("No selectable text found â€” switching to OCR mode.")
                return extract_text_from_pdf(pdf_path, use_ocr=True)
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""
    else:
        print("Performing OCR extraction (this may take a while)...")
        try:
            # Open the PDF with PyMuPDF
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc, start=1):
                # Render page to an image (pixmap)
                pix = page.get_pixmap()
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # Run OCR on the image
                text = pytesseract.image_to_string(img)
                extracted_text += f"\n--- Page {page_num} ---\n{text}\n"
        except Exception as e:
            print(f"Error during OCR: {e}")
            return ""

    return extracted_text


def correct_text_chunk(text_chunk, api_key, chunk_index, total_chunks):
    """
    Corrects a single chunk of text using Google Gemini.
    """
    genai.configure(api_key=api_key)

    generation_config = {
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    # Using the requested model
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-lite",
        generation_config=generation_config,
    )

    prompt = f"""
    You are an expert editor. Your task is to correct the following text chunk (Part {chunk_index}/{total_chunks}) which was extracted from a PDF using OCR.
    
    Please follow these rules:
    1. Fix grammar, spelling, and punctuation errors.
    2. Remove OCR artifacts (random characters, noise).
    3. Remove page headers, footers, and page numbers that interrupt the flow of text.
    4. Merge broken paragraphs where appropriate.
    5. Maintain the original meaning and general structure (headings, lists).
    6. Do NOT add any introductory or concluding remarks. Just provide the corrected text.
    
    Here is the text chunk:
    
    """ + text_chunk

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"â�Œ Error correcting chunk {chunk_index}: {e}")
        return text_chunk # Return original text on failure

def split_text_into_chunks(text):
    """
    Splits text into chunks based on page delimiters or size.
    """
    pages = text.split("--- Page")
    
    chunks = []
    current_chunk = ""
    MAX_CHUNK_SIZE = 15000
    
    if len(pages) > 1:
        for i, page in enumerate(pages):
            if i == 0 and not page.strip(): continue
            page_content = "--- Page" + page if i > 0 else page
            if len(current_chunk) + len(page_content) > MAX_CHUNK_SIZE:
                chunks.append(current_chunk)
                current_chunk = page_content
            else:
                current_chunk += page_content
        if current_chunk:
            chunks.append(current_chunk)
    else:
        lines = text.split('\n')
        for line in lines:
             if len(current_chunk) + len(line) > MAX_CHUNK_SIZE:
                chunks.append(current_chunk)
                current_chunk = line + "\n"
             else:
                current_chunk += line + "\n"
        if current_chunk:
            chunks.append(current_chunk)
            
    return chunks


def run_ocr_pipeline(pdf_path):
    if not os.path.exists(pdf_path):
        print("â�Œ File not found!")
        return

    # 1. Extract Text
    print(f"ğŸ“„ Extracting text from '{pdf_path}'...")
    extracted_text = extract_text_from_pdf(pdf_path)
    
    if not extracted_text.strip():
        print("âš ï¸� No text extracted.")
        return

    print(f"âœ… Extraction complete ({len(extracted_text)} chars).")

    # 2. Correct Text
    if not api_key:
        print("â�Œ Error: GEMINI_API_KEY not found in environment variables.")
        return

    chunks = split_text_into_chunks(extracted_text)
    total_chunks = len(chunks)
    print(f"ğŸ§© Split text into {total_chunks} chunks for correction.")

    full_corrected_text = ""
    print("âœ¨ Starting correction with Gemini (Lite)...")

    for i, chunk in enumerate(chunks, 1):
        print(f"   ... Processing chunk {i}/{total_chunks}...")
        corrected_chunk = correct_text_chunk(chunk, api_key, i, total_chunks)
        full_corrected_text += corrected_chunk + "\n\n"
        time.sleep(1)

    # 3. Save Output
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_name = os.path.splitext(pdf_path)[0]
    output_file = f"/kaggle/working/{base_name}_corrected_{timestamp}.txt"

    print(f"ğŸ’¾ Saving final corrected text to '{output_file}'...")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_corrected_text)

    print("âœ… Pipeline complete!")


# Run the pipeline
pdf_file = "/kaggle/input/ocr-data/kaggle.pdf"
if pdf_file:
    run_ocr_pipeline(pdf_file)

