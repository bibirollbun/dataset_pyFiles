!pip install python-docx PyPDF2 pdf2image nltk



import os
import re
import PyPDF2
from docx import Document
import ipywidgets as widgets
from IPython.display import display

uploaded_text = ""

# -----------------------------
# SAFETY CHECK (Reject Virus / Suspicious files)
# -----------------------------
def is_suspicious(filename):
    virus_patterns = [
        r"\.exe$", r"\.bat$", r"\.cmd$", r"\.scr$", r"\.js$",
        r"\.vbs$", r"\.msi$", r"\.dll$", r"\.bin$",
        r"virus", r"malware", r"trojan"
    ]
    filename = filename.lower()
    return any(re.search(p, filename) for p in virus_patterns)


# -----------------------------
# TEXT EXTRACTION
# -----------------------------
def extract_text_from_pdf(path):
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += (page.extract_text() or "")
    return text

def extract_text_from_docx(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# -----------------------------
# UPLOAD UI
# -----------------------------
upload_btn = widgets.FileUpload(accept='', multiple=False)
output_area = widgets.Output()

def on_upload_change(change):
    global uploaded_text

    with output_area:
        output_area.clear_output()

        if not upload_btn.value:
            print("No file uploaded.")
            return

        # FIXED: Kaggle gives tuple of dicts
        file_info = upload_btn.value[0]

        filename = file_info['name']

        # Virus check
        if is_suspicious(filename):
            print("â�Œ Suspicious file detected! Upload blocked.")
            uploaded_text = ""
            return

        # Save file
        file_path = "/kaggle/working/" + filename
        with open(file_path, "wb") as f:
            f.write(file_info['content'])

        print("âœ… File uploaded:", filename)

        # Extract text based on file type
        if filename.endswith(".pdf"):
            uploaded_text = extract_text_from_pdf(file_path)
        elif filename.endswith(".docx"):
            uploaded_text = extract_text_from_docx(file_path)
        elif filename.endswith(".txt"):
            uploaded_text = extract_text_from_txt(file_path)
        else:
            print("âš  Unsupported file type.")
            uploaded_text = ""
            return

        print("\nğŸ“„ Extracted Text Preview:")
        print(uploaded_text[:500], "..." if len(uploaded_text) > 500 else "")

upload_btn.observe(on_upload_change, names="value")
display(upload_btn, output_area)



import nltk
import re

nltk.download('punkt')

def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()



import re
import nltk
nltk.download('punkt')

def clean_text(text):
    # Basic cleaning
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def summarize_text(text, max_sentences=3):
    sentences = nltk.sent_tokenize(text)
    if len(sentences) <= max_sentences:
        return text
    return " ".join(sentences[:max_sentences])

def extract_keywords(text, top_n=10):
    words = re.findall(r'\b[A-Za-z]{4,}\b', text.lower())
    freq = nltk.FreqDist(words)
    return [word for word, count in freq.most_common(top_n)]

def answer_question(text, question):
    question = question.lower()
    text = text.lower()
    sentences = nltk.sent_tokenize(text)

    for s in sentences:
        if any(q in s for q in question.split()):
            return s
    return "No direct answer found. Try a different question!"



from nltk.tokenize import sent_tokenize
from nltk import FreqDist

def summarize_text(text, max_sentences=3):
    sentences = sent_tokenize(text)
    if len(sentences) <= max_sentences:
        return text
    return " ".join(sentences[:max_sentences])

def extract_keywords(text, top_n=10):
    words = re.findall(r"\b[A-Za-z]{4,}\b", text.lower())
    freq = FreqDist(words)
    return [w for w, c in freq.most_common(top_n)]

def answer_question(text, question):
    question = question.lower()
    text = text.lower()
    sentences = sent_tokenize(text)

    for s in sentences:
        if any(q in s for q in question.split()):
            return s
    
    return "No direct answer found! Try a different question."



import ipywidgets as widgets
from IPython.display import display, Markdown

summary_btn = widgets.Button(description="Generate Summary", button_style="info")
keywords_btn = widgets.Button(description="Extract Keywords", button_style="warning")
qa_btn = widgets.Button(description="Ask Question", button_style="success")
question_box = widgets.Text(placeholder="Enter your question about the document")

output_area2 = widgets.Output()

def on_summary_clicked(b):
    with output_area2:
        output_area2.clear_output()
        if not uploaded_text:
            print("âš  Upload a file first!")
            return
        cleaned = clean_text(uploaded_text)
        summary = summarize_text(cleaned)
        display(Markdown(f"### ğŸ“˜ Summary\n{summary}"))

def on_keywords_clicked(b):
    with output_area2:
        output_area2.clear_output()
        if not uploaded_text:
            print("âš  Upload a file first!")
            return
        cleaned = clean_text(uploaded_text)
        keys = extract_keywords(cleaned)
        display(Markdown(f"### ğŸ”‘ Keywords\n{', '.join(keys)}"))

def on_qa_clicked(b):
    with output_area2:
        output_area2.clear_output()
        if not uploaded_text:
            print("âš  Upload a file first!")
            return
        cleaned = clean_text(uploaded_text)
        ans = answer_question(cleaned, question_box.value)
        display(Markdown(f"### â�“ Answer\n{ans}"))

summary_btn.on_click(on_summary_clicked)
keywords_btn.on_click(on_keywords_clicked)
qa_btn.on_click(on_qa_clicked)

display(summary_btn, keywords_btn, question_box, qa_btn, output_area2)


