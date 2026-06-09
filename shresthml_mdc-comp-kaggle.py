# !pip install -U scikit-learn==1.3.2 imbalanced-learn==0.11.0
# !pip install --upgrade --force-reinstall scikit-learn==1.3.2 imbalanced-learn==0.11.0
# # âœ… Install compatible versions for imbalanced-learn + scikit-learn
# !pip install --upgrade --force-reinstall scikit-learn==1.3.2 imbalanced-learn==0.11.0
# !pip install -U imbalanced-learn



!pip install --no-index --find-links=/kaggle/input/sklearnzip/offline_pkgs_sklearn scikit-learn==1.4.2 imbalanced-learn==0.11.0



!pip install --no-index --find-links=/kaggle/input/pkgs-senttfmr-xgb-lgbm-nltk-spacy-bs4/offline_pkgs \
    xgboost lightgbm tqdm PyPDF2 pymupdf xmltodict beautifulsoup4 spacy nltk



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')


# Load data
train = pd.read_csv('/kaggle/input/make-data-count-finding-data-references/train_labels.csv')
print("Train shape:", train.shape)
print(train.dtypes)
train.head()


# !pip install xmltodict
# !pip install pymupdf
# !pip install PyPDF2


!pip install --no-index --find-links=/kaggle/input/pkgs-senttfmr-xgb-lgbm-nltk-spacy-bs4/offline_pkgs xmltodict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import seaborn as sns

# reading pdf/xml files
import xmltodict
import fitz
import contextlib
from PyPDF2 import PdfReader


import warnings
warnings.filterwarnings("ignore")


# Training Data Paths
train_path = '/kaggle/input/make-data-count-finding-data-references/train'
train_pdf_path = f'{train_path}/PDF'
train_xml_path = f'{train_path}/XML'
train_labels = f'{train_path}_labels.csv'

# Test Data Paths
test_path = '/kaggle/input/make-data-count-finding-data-references/test'
test_pdf_path = f'{test_path}/PDF'
test_xml_path = f'{test_path}/XML'

# Sample Submission Path
sample_submission = '/kaggle/input/make-data-count-finding-data-references/sample_submission.csv'



# !pip install pymupdf


# Load the training labels CSV
train_labels_df = pd.read_csv(train_labels)
train_labels_df.head()



import os

def get_files(pdf_path: str, xml_path: str) -> pd.DataFrame:
    """Returns a DataFrame containing metadata for PDF and XML files."""
    
    # List PDF and XML files
    pdf_files = [f for f in os.listdir(pdf_path) if f.endswith('.pdf')]
    xml_files = [f for f in os.listdir(xml_path) if f.endswith('.xml')]
    
    # Create DataFrames
    df_pdf = pd.DataFrame({
        'article_id': [f.replace('.pdf', '') for f in pdf_files],
        'file': pdf_files,
        'path': [os.path.join(pdf_path, f) for f in pdf_files],
        'format': 'pdf',
    })
    
    df_xml = pd.DataFrame({
        'article_id': [f.replace('.xml', '') for f in xml_files],
        'file': xml_files,
        'path': [os.path.join(xml_path, f) for f in xml_files],
        'format': 'xml'
    })
    
    return pd.concat([df_pdf, df_xml], ignore_index=True)



# Load files and label their dataset origin
train_files = get_files(train_pdf_path, train_xml_path)
test_files = get_files(test_pdf_path, test_xml_path)

train_files['dataset'] = 'train'
test_files['dataset'] = 'test'

# Combine train and test files
full_data = pd.concat([train_files, test_files], ignore_index=False)



# Merge with training labels
df_final = pd.merge(full_data, train_labels_df, on='article_id', how='left')

# Filter train dataset
train_df = df_final[df_final.dataset == 'train'].copy()

# Flag for presence of dataset
train_df['has_dataset'] = train_df['dataset_id'] != 'Missing'



# Citation type distribution (excluding 'Missing')
citation_counts = train_df[train_df['type'] != 'Missing']['type'].value_counts().reset_index()
citation_counts.columns = ['type', 'count']

# Dataset availability by format
dataset_by_format = train_df.groupby(['format', 'has_dataset']).size().reset_index(name='count')

# Articles with multiple datasets
datasets_per_article = (
    train_df[train_df['dataset_id'] != 'Missing']
    .groupby('article_id')['dataset_id']
    .count()
    .reset_index(name='dataset_count')
)

multi_dataset_articles = (
    datasets_per_article['dataset_count']
    .value_counts()
    .reset_index()
    .rename(columns={'index': 'number_of_datasets', 'dataset_count': 'number_of_articles'})
)

# Number of unique articles per format
articles_per_format = train_df.groupby('format')['article_id'].nunique().reset_index(name='unique_articles')



# !pip install pandas beautifulsoup4 tqdm spacy pymupdf nltk



# âœ… Extra Imports for Parsing PDFs and XML (Offline-compatible)
import nltk  
import re
import csv
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import spacy
from PyPDF2 import PdfReader  # âœ… Replace fitz with PyPDF2 (Kaggle-safe)
from nltk.tokenize import sent_tokenize

# âœ… Set custom NLTK data path (pre-downloaded)
nltk.data.path.append("/kaggle/input/pkgs-senttfmr-xgb-lgbm-nltk-spacy-bs4/nltk_data")

# â�Œ Don't call nltk.download('punkt') â€” it's already offline

# âœ… Load spaCy model from offline path
import spacy

# âœ… Correct path with versioned folder
nlp = spacy.load("/kaggle/input/pkgs-senttfmr-xgb-lgbm-nltk-spacy-bs4/spacy_model/en_core_web_sm/en_core_web_sm-3.8.0")
# /kaggle/input/pkgs-senttfmr-xgb-lgbm-nltk-spacy-bs4/spacy_model/en_core_web_sm/en_core_web_sm-3.8.0/
# â”œâ”€â”€ offline_pkgs/
# â”œâ”€â”€ nltk_data/
# â”‚   â””â”€â”€ tokenizers/
# â”‚       â””â”€â”€ punkt/
# â””â”€â”€ spacy_model/
#     â””â”€â”€ en_core_web_sm/



# ğŸ“„ Extract text from XML files using BeautifulSoup (Kaggle-compatible)

from bs4 import BeautifulSoup

def extract_text_from_xml(xml_path):
    with open(xml_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'xml')

    texts = []
    for tag in ['abstract', 'body', 'ref-list', 'sec']:
        for el in soup.find_all(tag):
            if el.text:
                texts.append(el.text.strip())

    return " ".join(texts).replace('\n', ' ').strip()



# ğŸ“ƒ Extract text from PDF files using PyMuPDF
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text.strip()
    except Exception as e:
        print(f"Error in PDF: {pdf_path} â€” {e}")
        return ""



# âœ… Extract features like DOIs, accession IDs, repo names, etc.
import re

def extract_all_features(text):
    features = {}

    # Patterns
    doi_pattern = r'10\.\d{4,9}/[-._;()/:A-Z0-9]+'
    accession_pattern = r'\b(GSE|PXD|EGA|EGAD|SRR|ERP|DRR|PRJ|EMPIAR)[-_]?\d+\b'
    dataset_repos = r'\b(figshare|zenodo|dataverse|dryad|pangaea|openei|datadryad|genbank|GEO|ArrayExpress|NCBI|ICPSR|osf|openicpsr|proteinatlasebi|pdb|pride|openneuro|ega|metabolights|MG-RAST|bioRxiv|SRA)\b'
    contextual_pattern = r'\b(dataset|data)\b.{0,100}?\b(available|deposited|shared|provided|stored|hosted|submitted|archived|published|released|registered)\b'
    url_pattern = r'(https?://[^\s\)\]\}\>\"\']+)'

    # Matching
    dois = re.findall(doi_pattern, text, re.IGNORECASE)
    accessions = re.findall(accession_pattern, text, re.IGNORECASE)
    repo_names = re.findall(dataset_repos, text, re.IGNORECASE)
    context_matches = [m.group(0) for m in re.finditer(contextual_pattern, text, re.IGNORECASE)]
    urls = re.findall(url_pattern, text)

    # NER using spaCy
    doc = nlp(text)
    ner_entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART"]]

    # Features dictionary
    features.update({
        "doi_count": len(dois),
        "accession_id_count": len(accessions),
        "repo_name_count": len(repo_names),
        "context_mentions_count": len(context_matches),
        "url_count": len(urls),
        "ner_entities_count": len(set(ner_entities)),
        "dois": "; ".join(set(dois)),
        "accession_ids": "; ".join(set(accessions)),
        "repo_names": "; ".join(set(repo_names)),
        "context_mentions": "; ".join(set(context_matches)),
        "urls": "; ".join(set(urls)),
        "ner_entities": "; ".join(set(ner_entities))
    })

    return features, dois, accessions, repo_names, urls, context_matches



# âœ‚ï¸� Extract contextual sentences around specific keywords
from nltk.tokenize import sent_tokenize

def extract_contextual_sentences(text, keywords, window=2):
    sentences = sent_tokenize(text)
    contexts = []
    for i, sent in enumerate(sentences):
        for keyword in keywords:
            if keyword.lower() in sent.lower():
                start = max(0, i - window)
                end = min(len(sentences), i + window + 1)
                context = " ".join(sentences[start:end])
                contexts.append(context)
                break
    return list(set(contexts))



# !pip install -q sentence-transformers 


# âœ… Function to process XML and PDF documents and extract features + context sentences
import csv
from tqdm import tqdm

def process_all_documents(xml_folder, pdf_folder):
    # Get all unique paper IDs from both XML and PDF
    all_files = set([f.replace('.xml', '') for f in os.listdir(xml_folder) if f.endswith('.xml')])
    all_files |= set([f.replace('.pdf', '') for f in os.listdir(pdf_folder) if f.endswith('.pdf')])
    
    results = []

    # Open a CSV file to save contextual sentences
    with open("context_sentences_combined.csv", "w", encoding='utf-8', newline='') as ctx_file:
        writer = csv.writer(ctx_file)
        writer.writerow(["paper_id", "context_sentence"])

        # Process each paper
        for paper_id in tqdm(sorted(all_files), desc="Processing XML + PDF"):
        #for paper_id in tqdm(sorted(list(all_files))[:20], desc="Processing first 50 XML + PDF"):

            xml_path = os.path.join(xml_folder, f"{paper_id}.xml")
            pdf_path = os.path.join(pdf_folder, f"{paper_id}.pdf")

            text_parts = []

            # Extract text from XML and/or PDF
            if os.path.exists(xml_path):
                text_parts.append(extract_text_from_xml(xml_path))
            if os.path.exists(pdf_path):
                text_parts.append(extract_text_from_pdf(pdf_path))

            full_text = "\n".join(text_parts).strip()
            if not full_text:
                continue

            # Extract features
            features, dois, accessions, repo_names, urls, context_matches = extract_all_features(full_text)
            features['paper_id'] = paper_id
            features['text'] = full_text[:5000]  # Store a sample of the text
            results.append(features)

            # Extract context sentences and save
            keywords = dois + accessions + repo_names + urls + context_matches
            context_sentences = extract_contextual_sentences(full_text, keywords, window=2)
            for ctx in context_sentences:
                writer.writerow([paper_id, ctx])

    return results



# âœ… Set XML and PDF folder paths
xml_folder = '/kaggle/input/make-data-count-finding-data-references/train/XML'
pdf_folder = '/kaggle/input/make-data-count-finding-data-references/train/PDF'

# âœ… Run the processing pipeline
output = process_all_documents(xml_folder, pdf_folder)



# âœ… Save output to CSV
df = pd.DataFrame(output)
df.to_csv("/kaggle/working/combined_xml_pdf_features.csv", index=False)

# âœ… Status messages
print("\nâœ… Combined XML + PDF extraction complete.")
print("âœ… Saved features â†’ combined_xml_pdf_features.csv")
print("âœ… Saved context sentences â†’ context_sentences_combined.csv")



# âœ… ML & NLP Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import StackingClassifier
from imblearn.over_sampling import SMOTE
from sentence_transformers import SentenceTransformer
import xgboost as xgb
import lightgbm as lgb

# âœ… Load extracted features
features_df = pd.read_csv("/kaggle/working/combined_xml_pdf_features.csv")

# âœ… Load original labels
labels_df = pd.read_csv('/kaggle/input/make-data-count-finding-data-references/train_labels.csv')
labels_df.rename(columns={'article_id': 'paper_id'}, inplace=True)

# Use only one label per paper (if multiple, use the 'max' as final type)
labels_df = labels_df.groupby('paper_id')['type'].max().reset_index()


# âœ… Merge features with labels
merged_df = features_df.merge(labels_df, on='paper_id')

# Encode labels: Primary = 1, Secondary = 0
label_map = {'Primary': 1, 'Secondary': 0}
merged_df = merged_df[merged_df['type'].isin(label_map)]
merged_df['label'] = merged_df['type'].map(label_map)

# Drop original 'type' column
merged_df.drop(columns=['type'], inplace=True)



# âœ… Load context sentences
ctx_df = pd.read_csv("/kaggle/working/context_sentences_combined.csv")

# Group sentences by paper_id
ctx_grouped = ctx_df.groupby('paper_id')['context_sentence'].apply(lambda x: " ".join(x)).reset_index()

# Merge with main dataframe
merged_df = merged_df.merge(ctx_grouped, on='paper_id', how='left')

# Fill missing context sentences with empty string
merged_df['context_sentence'] = merged_df['context_sentence'].fillna("")

# âœ… Final dataset ready for embedding and modeling
print("\nâœ… Final merged_df shape:", merged_df.shape)
print(merged_df[['paper_id', 'label', 'context_sentence']].sample(5))




from sentence_transformers import SentenceTransformer

# âœ… Load SciBERT model from the corrected offline directory
embedder = SentenceTransformer("/kaggle/input/mdc-scibert/kaggle/working/scibert_model")

# âœ… Encode context sentences
X = embedder.encode(merged_df['context_sentence'].tolist(), show_progress_bar=True)

# âœ… Labels
y = merged_df['label'].values



from imblearn.over_sampling import SMOTE

# âœ… Apply SMOTE to balance class distribution
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)



from sklearn.model_selection import train_test_split

# âœ… Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_resampled, y_resampled,
    test_size=0.2,
    stratify=y_resampled,
    random_state=42
)



from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import StackingClassifier
import xgboost as xgb
import lightgbm as lgb

# âœ… Define individual models
xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1, verbosity=0)
lgb_model = lgb.LGBMClassifier(random_state=42)
ridge_model = RidgeClassifier()

# âœ… Stacking ensemble with XGBoost as the final estimator
stack = StackingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('ridge', ridge_model)
    ],
    final_estimator=xgb.XGBClassifier(random_state=42, n_jobs=-1, verbosity=0)
)



from sklearn.model_selection import GridSearchCV, StratifiedKFold

# âœ… Define grid search parameters for XGB
param_grid = {
    'xgb__n_estimators': [200, 300],
    'xgb__max_depth': [4, 6],
    'xgb__learning_rate': [0.03, 0.05]
}

# âœ… Grid search setup
grid_search = GridSearchCV(
    estimator=stack,
    param_grid=param_grid,
    scoring='f1_macro',
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    verbose=1,
    n_jobs=-1
)

# âœ… Fit grid search
grid_search.fit(X_train, y_train)



from sklearn.metrics import classification_report

# âœ… Best model from grid search
best_model = grid_search.best_estimator_

# âœ… Make predictions and evaluate
y_pred = best_model.predict(X_val)

print("ğŸ“Š Classification Report:")
print(classification_report(y_val, y_pred))



# âœ… Function to extract features and context sentences from test documents
from tqdm import tqdm
import os
import pandas as pd

def process_test_documents(xml_folder, pdf_folder):
    all_files = set([f.replace('.xml','') for f in os.listdir(xml_folder) if f.endswith('.xml')])
    all_files |= set([f.replace('.pdf','') for f in os.listdir(pdf_folder) if f.endswith('.pdf')])
    
    results = []
    context_data = []

    for paper_id in tqdm(sorted(all_files), desc="Processing TEST XML + PDF"):
        xml_path = os.path.join(xml_folder, f"{paper_id}.xml")
        pdf_path = os.path.join(pdf_folder, f"{paper_id}.pdf")

        text_parts = []
        if os.path.exists(xml_path):
            text_parts.append(extract_text_from_xml(xml_path))
        if os.path.exists(pdf_path):
            text_parts.append(extract_text_from_pdf(pdf_path))

        full_text = "\n".join(text_parts).strip()
        if not full_text:
            continue

        features, dois, accessions, repo_names, urls, context_matches = extract_all_features(full_text)
        features['paper_id'] = paper_id
        features['text'] = full_text[:5000]
        results.append(features)

        keywords = dois + accessions + repo_names + urls + context_matches
        context_sentences = extract_contextual_sentences(full_text, keywords, window=2)
        for ctx in context_sentences:
            context_data.append([paper_id, ctx])

    pd.DataFrame(results).to_csv("/kaggle/working/test_features.csv", index=False)
    pd.DataFrame(context_data, columns=['paper_id', 'context_sentence']).to_csv("/kaggle/working/test_context_sentences.csv", index=False)

    print("\nâœ… Test feature extraction done.")
    return results



# âœ… Set paths
test_xml_folder = "/kaggle/input/make-data-count-finding-data-references/test/XML"
test_pdf_folder = "/kaggle/input/make-data-count-finding-data-references/test/PDF"

# âœ… Run test document processor
process_test_documents(test_xml_folder, test_pdf_folder)



from sentence_transformers import SentenceTransformer

# âœ… Load context sentences
test_ctx_df = pd.read_csv("/kaggle/working/test_context_sentences.csv")

# âœ… Combine context per paper
test_ctx_grouped = test_ctx_df.groupby('paper_id')['context_sentence'].apply(lambda x: " ".join(x)).reset_index()
test_ctx_grouped['context_sentence'] = test_ctx_grouped['context_sentence'].fillna("")

# # âœ… Load SciBERT model
# embedder = SentenceTransformer('allenai/scibert_scivocab_uncased')

# âœ… Generate embeddings
X_test = embedder.encode(test_ctx_grouped['context_sentence'].tolist(), show_progress_bar=True)

# âœ… Predict using trained model
test_ctx_grouped['predicted_label'] = best_model.predict(X_test)

# âœ… Map numeric labels back to type
test_ctx_grouped['type'] = test_ctx_grouped['predicted_label'].map({1: 'Primary', 0: 'Secondary'})



# âœ… Load sample submission
sample_sub = pd.read_csv('/kaggle/input/make-data-count-finding-data-references/sample_submission.csv')

# âœ… Merge with predictions
submission = sample_sub.drop(columns=['type']).merge(
    test_ctx_grouped[['paper_id', 'type']],
    left_on='article_id', right_on='paper_id', how='left'
)

# âœ… Fill any missing values with 'Secondary'
submission['type'] = submission['type'].fillna("Secondary")

# âœ… Save final submission file
submission[['row_id', 'article_id', 'dataset_id', 'type']].to_csv("submission.csv", index=False)
print("âœ… submission.csv generated!")



# âœ… Evaluation using true labels (if accessible)
from sklearn.metrics import f1_score

# Load prediction and true labels
pred_df = pd.read_csv("/kaggle/working/submission.csv")
true_df = pd.read_csv("/kaggle/input/make-data-count-finding-data-references/sample_submission.csv")

# Ensure column integrity
assert all(col in pred_df.columns for col in ['row_id', 'article_id', 'dataset_id', 'type'])
assert all(col in true_df.columns for col in ['row_id', 'article_id', 'dataset_id', 'type'])

# Merge on row_id
merged_df = true_df.merge(pred_df, on="row_id", suffixes=('_true', '_pred'))

# Encode labels
label_map = {'Primary': 1, 'Secondary': 0}
y_true = merged_df['type_true'].map(label_map)
y_pred = merged_df['type_pred'].map(label_map)

# âœ… Calculate F1 Score
f1 = f1_score(y_true, y_pred)
print(f"âœ… F1 Score: {f1:.4f}")



pred_df.head()




