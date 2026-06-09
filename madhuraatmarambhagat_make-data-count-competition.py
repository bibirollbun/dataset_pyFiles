!pip install xmltodict
!pip install fitz
!pip install tools
!pip install PyMuPDF
!pip install PyPDF2


import os
import pandas as pd
import xmltodict
import fitz
import re
import concurrent.futures
from sklearn.model_selection import train_test_split

# 1. Load metadata & filepaths

def load_files(pdf_dir, xml_dir):
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    
    pdf_df = pd.DataFrame({
        'article_id': [f[:-4] for f in pdf_files],
        'path': [os.path.join(pdf_dir, f) for f in pdf_files],
        'format': 'pdf'
    })
    
    xml_df = pd.DataFrame({
        'article_id': [f[:-4] for f in xml_files],
        'path': [os.path.join(xml_dir, f) for f in xml_files],
        'format': 'xml'
    })
    
    return pd.concat([pdf_df, xml_df], ignore_index=True)

# 2. Parse XML articles for text fields

def parse_xml(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            xml_dict = xmltodict.parse(f.read())
        article = xml_dict.get('article') or xml_dict.get('ns:article')
        front = article.get('front', {})
        meta = front.get('article-meta', {})
        title = meta.get('title-group', {}).get('article-title', '')
        abstract = meta.get('abstract', '')
        body = article.get('body', '')
        # Extract textual content from body (can be recursive if needed)
        return title + ' ' + str(abstract) + ' ' + str(body)
    except Exception:
        return ''

# 3. Extract text from PDFs using PyMuPDF (fitz)

def extract_pdf_text(path):
    doc = fitz.open(path)
    text = []
    for page in doc:
        text.append(page.get_text())
    return ' '.join(text)

# 4. Extract full article text depending on format

def extract_text(row):
    if row['format'] == 'pdf':
        return extract_pdf_text(row['path'])
    else:
        return parse_xml(row['path'])

# 5. Define dataset mention extraction rules (simple heuristic + regex)

DATASET_KEYWORDS = ['dataset', 'data set', 'data-set', 'corpus', 'benchmark', 'repository']

def find_dataset_mentions(text):
    # Lowercase for consistency
    text = text.lower()
    # Simple heuristic: find sentences or phrases mentioning dataset keywords
    pattern = r'([^.]*?\b(?:' + '|'.join(DATASET_KEYWORDS) + r')\b[^.]*)'
    matches = re.findall(pattern, text)
    # Return unique mentions
    return list(set([m.strip() for m in matches if len(m.strip()) > 10]))

# 6. Load and prepare dataset (train + test)

train_files = load_files('/kaggle/input/make-data-count-finding-data-references/train/PDF',
                         '/kaggle/input/make-data-count-finding-data-references/train/XML')
test_files = load_files('/kaggle/input/make-data-count-finding-data-references/test/PDF',
                        '/kaggle/input/make-data-count-finding-data-references/test/XML')

train_files['dataset'] = 'train'
test_files['dataset'] = 'test'

full_data = pd.concat([train_files, test_files], ignore_index=True)

# 7. Extract full text in parallel (speed up)

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    texts = list(executor.map(extract_text, [row for _, row in full_data.iterrows()]))

full_data['full_text'] = texts

# 8. Extract dataset mentions from the full text

full_data['dataset_mentions'] = full_data['full_text'].apply(find_dataset_mentions)

# 9. (Train only) Merge with labels for supervised learning or evaluation

train_labels = pd.read_csv('/kaggle/input/make-data-count-finding-data-references/train_labels.csv')

train_full = full_data[full_data['dataset'] == 'train']
train_full = train_full.merge(train_labels, on='article_id', how='left')

# 10. Now you have:
# - article_id
# - format (pdf or xml)
# - full_text
# - dataset_mentions (list of strings)
# - ground truth dataset_id, type, etc. for train

# You can now:
# - Use extracted dataset_mentions as features
# - Engineer features from text + metadata
# - Train ML or rule-based models to predict dataset references
# - Evaluate predictions against ground truth labels
# - Predict on test set using the same pipeline

# -----------------------------------------
# Example usage: print some extracted mentions

for idx, row in train_full.head(5).iterrows():
    print(f"Article: {row['article_id']}")
    print("Extracted mentions:", row['dataset_mentions'])
    print("Ground truth dataset id:", row['dataset_id'])
    print('-----')




import os
import pandas as pd
import xmltodict
import fitz
from PyPDF2 import PdfReader

test_pdf_path = '/kaggle/input/make-data-count-finding-data-references/test/PDF'
test_xml_path = '/kaggle/input/make-data-count-finding-data-references/test/XML'
sample_submission_path = '/kaggle/input/make-data-count-finding-data-references/sample_submission.csv'

# === Load Sample Submission ===
submission_df = pd.read_csv(sample_submission_path)

# === Collect all test files ===
def get_test_files(pdf_path, xml_path):
    pdfs = [f.replace('.pdf', '') for f in os.listdir(pdf_path) if f.endswith('.pdf')]
    xmls = [f.replace('.xml', '') for f in os.listdir(xml_path) if f.endswith('.xml')]
    article_ids = sorted(set(pdfs + xmls))
    return article_ids

test_article_ids = get_test_files(test_pdf_path, test_xml_path)

# === Extract text from XML ===
def extract_xml_text(xml_file_path):
    try:
        with open(xml_file_path, 'r', encoding='utf-8') as f:
            data = xmltodict.parse(f.read())
        article = data.get('article') or data.get('ns:article')
        body = article.get('body', '')
        return str(body).lower()
    except:
        return ""

# === Extract text from PDF ===
def extract_pdf_text(pdf_file_path):
    try:
        doc = fitz.open(pdf_file_path)
        return "\n".join([page.get_text().lower() for page in doc])
    except:
        return ""

# === Simple keyword-based matcher ===
def predict_dataset_info(text):
    text = text.lower()

    # --- Try to extract DOI ---
    doi_match = re.search(r'https://doi\.org/\S+', text)
    doi = doi_match.group(0) if doi_match else "Missing"

    # --- Type Detection Rules ---
    if re.search(r'\bwe (collected|generated|gathered|recorded|conducted)\b', text):
        type_detected = "primary"
    elif re.search(r'\b(used|retrieved|downloaded|sourced|obtained|reused|analyzed) (the )?(data|dataset|corpus)\b', text):
        type_detected = "secondary"
    elif "dryad" in text or "figshare" in text or "zenodo" in text:
        type_detected = "secondary"
    else:
        type_detected = "secondary" if doi != "Missing" else "Missing"

    return doi, type_detected


predictions = []

for article_id in test_article_ids:
    xml_file = os.path.join(test_xml_path, f"{article_id}.xml")
    pdf_file = os.path.join(test_pdf_path, f"{article_id}.pdf")
    
    xml_text = extract_xml_text(xml_file) if os.path.exists(xml_file) else ""
    pdf_text = extract_pdf_text(pdf_file) if os.path.exists(pdf_file) else ""
    
    combined_text = xml_text + "\n" + pdf_text
    dataset_id, type_pred = predict_dataset_info(combined_text)
    
    predictions.append((article_id, dataset_id, type_pred))

predictions_df = pd.DataFrame(predictions, columns=["article_id", "dataset_id", "type"])
predictions_df.to_csv('submission.csv', index=False)
print(f"Saved {len(predictions_df)} predictions to submission.csv")


