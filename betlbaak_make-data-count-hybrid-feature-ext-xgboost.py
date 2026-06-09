!pip install pymupdf --no-deps --find-links=/kaggle/input/pymupdf-linux/pymupdf-1.26.4-cp39-abi3-manylinux_2_28_x86_64.whl
!pip install vaderSentiment --no-deps --find-links=/kaggle/input/vadersentiment/vaderSentiment-3.3.2-py2.py3-none-any.whl
!pip install sentence-transformers --no-deps --find-links=/kaggle/input/sentence-transformers/sentence_transformers-5.1.0-py3-none-any.whl
!pip install requests --no-deps --find-links=/kaggle/input/request/requests-2.32.5-py3-none-any.whl
!pip install tqdm --no-deps --find-links=/kaggle/input/tqdm-data/tqdm-4.67.1-py3-none-any.whl


!pip install transformers --no-deps --find-links=/kaggle/input/transformers/transformers-4.56.1-py3-none-any.whl


import pandas as pd
import numpy as np
import seaborn as sns
import fitz
import re
import os
from tqdm import tqdm
import torch
import nltk
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from scipy.sparse import csr_matrix, hstack
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.2f" % x)
pd.set_option("display.max_colwidth", None)


train_labels = pd.read_csv("/kaggle/input/make-data-count-finding-data-references/train_labels.csv")
train_labels.shape


train_labels["type"].value_counts()


def clean_doi(doi):
    if pd.isna(doi):
        return None
    doi = doi.strip().lower()
    doi = re.sub(r'^https?://doi\.org/', '', doi) 
    doi = re.sub(r'^doi:', '', doi)                
    return doi


train_labels["clean_title"] = train_labels["dataset_id"].apply(clean_doi)


train_labels.head()


def clean_text(text):
    text = text.replace('\n', 
                        ' ')
    text = re.sub(r'\s+', 
                  ' ', 
                  text)
    text = text.strip()
    return text

def extract_and_clean_pdfs(input_dir, 
                           output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, 
                                pdf_file)
        try:
            doc = fitz.open(pdf_path)
            all_text = ""
            for i, page in enumerate(doc):
                try:
                    page_text = page.get_text()
                    cleaned = clean_text(page_text)
                    all_text += cleaned + "\n\n"
                except Exception as e:
                    print(f"âš ï¸� Page skipped ({pdf_file}, page {i}): {e}")
            # Save as .txt
            output_filename = os.path.splitext(pdf_file)[0] + ".txt"
            output_path = os.path.join(output_dir, 
                                       output_filename)
            with open(output_path, 
                      "w", 
                      encoding="utf-8") as f:
                f.write(all_text)
            print(f"âœ“ Processed: {pdf_file}")
        except Exception as e:
            print(f"â�Œ File could not be processed ({pdf_file}): {e}")


# ğŸ”§ Usage
train_input_folder = "/kaggle/input/make-data-count-finding-data-references/train/PDF" # Folder with PDF files
train_output_folder = "train/TXT" # Folder where cleaned texts will added

extract_and_clean_pdfs(train_input_folder, 
                       train_output_folder)


test_input_folder = "/kaggle/input/make-data-count-finding-data-references/test/PDF"
test_output_folder = "test/TXT"

extract_and_clean_pdfs(test_input_folder, 
                       test_output_folder)


train_sentences = {}  # {article_id: [sentence1, sentence2, ...]}

for file in os.listdir("train/TXT"):
    article_id = file.replace(".txt", 
                              "")
    with open(f"train/TXT/{file}",
              "r", 
              encoding="utf-8") as f:

        text = f.read()
        sentences = sent_tokenize(text)
        train_sentences[article_id] = sentences

train_contexts = []

for _, row in train_labels.iterrows():
    article_id = row["article_id"]
    clean_title = row["clean_title"]

    found = False
    for sent in train_sentences.get(article_id, []):
        if clean_title.lower() in sent.lower():
            train_contexts.append(sent)
            found = True
            break

    if not found:
        train_contexts.append("")  # If not found, leave space


train_labels["context"] = train_contexts
train = train_labels
train.head()


train.shape


train["type"].value_counts()


train.isnull().sum()


missing_context = train.loc[(train["context"] == "") & (train["type"] != "Missing")]
missing_context.shape


miss_index = missing_context.index.tolist()


train.drop(miss_index,
          axis = 0,
          inplace = True)


train = train.reset_index(drop=True)
train.head()



TXT_DIRECTORY_PATH = '/kaggle/working/test/TXT' 
OUTPUT_CSV_PATH = '/kaggle/working/test.csv'

patterns_to_define = [
    (r'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+', "DOI"),
    (r'CHEMBL\d+', "Accession"),
    (r'CVCL_[A-Z0-9]{4}', "Accession"),
    (r'E-GEOD-\d+|E-PROT-\d+|E-MTAB-\d+|E-MEXP-\d+|EMPIAR-\d+', "Accession"),
    (r'PMC\d+', "Accession"),
    (r'S-BSST\d+', "Accession"),
    (r'ATCC\d+', "Accession"),
    (r'ENSBTAG\d+|ENSOARG\d+|ENSMMUT\d{11}', "Accession"),
    (r'ENST\d+\.\d+|ENST\d+', "Accession"),
    (r'R-HSA-\d+', "Accession"),
    (r'EPI_ISL_\d{5,}|EPI\d{6,7}', "Accession"),
    (r'AF-[A-Z0-9]+-F1', "Accession"),
    (r'HPA\d+|CP\d{6}|IPR\d{6}|PF\d{5}|BX\d{6}|KX\d{6}|K0\d{4}|CAB\d{6}', "Accession"),
    (r'PRJNA\d+|PRJEB\d+|PRJDB\d+|PXD\d+|SAMN\d+|SAMEA\d+', "Accession"),
    (r'phs\d{6}(?:\.v\d{1,2}\.p\d{1,2})?', "Accession"),
    (r'NC_\d{6}\.\d{1}|NM_\d{9}|MODEL\d{10}', "Accession"),
    (r'XM_\d{9}\.\d{1}|XM_\d{9}', "Accession"),
    (r'GSE\d+|GSM\d+|GPL\d+', "Accession"),
    (r'PDB\s?[1-9][A-Z0-9]{3}|HMDB\d+', "Accession"),
    (r'PDB:[1-9][A-Z0-9]{3}|PDB ID:[1-9][A-Z0-9]{3}|PDB ID code [1-9][A-Z0-9]{3}|PDB entry [1-9][A-Z0-9]{3}', "Accession"),
    (r'JNS\d+', "Accession"),
    (r'(?:SR[PRX]|STH|ERR|DRR|DRX|DRP|ERP|ERX)\d+', "Accession"),
    (r'GM\d+', "Accession"),
    (r'AF\d+\.\d+|AF\d+', "Accession"),
    (r'AB\d+\.\d+|AB\d+', "Accession"),
    (r'AI\d+', "Accession"),
    (r'[1-5]\.(?:10|20|30|40|50|60|70|80|90)\.\d{2,4}\.\d{2,4}', "Accession")
]

compiled_patterns = [(re.compile(pattern), label) for pattern, label in patterns_to_define]
print(f"{len(compiled_patterns)} Regex patterns compiled and ready.")

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("NLTK 'punkt' model downloading...")
    nltk.download('punkt')
    print("download completed.")


print(f"'{TXT_DIRECTORY_PATH}' Scanning .txt files in the folder...")

results = []
try:
    txt_files = [f for f in os.listdir(TXT_DIRECTORY_PATH) if f.endswith('.txt')]
except FileNotFoundError:
    print(f"ERROR: '{TXT_DIRECTORY_PATH}' folder not found. Please check the folder path.")
    exit()

for filename in tqdm(txt_files, desc="Scanning files"):
    file_path = os.path.join(TXT_DIRECTORY_PATH, filename)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            sentences = sent_tokenize(text)
            
            for sentence in sentences:
                for pattern, label in compiled_patterns:
                    matches = pattern.findall(sentence)
                    
                    if matches:
                        for match in matches:
                            results.append({
                                'article_id': filename,
                                'dataset_id': match,
                                'label': label,
                                'context': sentence.strip()
                            })
    except Exception as e:
        print(f"'{filename}' an error occurred while reading the file: {e}")


if not results:
    print("\nScan completed. No sentences matching Regex patterns were found in any files.")
else:
    print(f"\nScan completed. Total {len(results)} matches found.")
    test = pd.DataFrame(results)
    test.drop_duplicates(inplace=True)
    
    test.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')


test[test["label"] == "Accession"]


test.head(10)


def extract_regex_features(sentences):
    """
    Extracts specific dataset ID patterns as binary features.

    Parameter:
        sentences (list of str): Sentences
    
    Return:
        pd.DataFrame: Binary feature (0/1) for each regex
    """
    patterns = {
        "has_doi": r"(10\.\d{4,}/[^\s]+|doi\.org/10\.\d{4,}/[^\s]+)",
        "has_gse": r"GSE\d+",
        "has_pdb": r"PDB[\s:]?\w+|\b[0-9][A-Za-z0-9]{3}\b",  # 4-char PDB IDs
        "has_arrayexpress": r"E-[A-Z]{4}-\d+",
        "has_prj": r"(PRJ[EDN][A-Z]*\d+|SRP\d+|ERP\d+|DRP\d+)",
        "has_dryad": r"dryad\.[a-z0-9]+",
        "has_zenodo": r"zenodo\.\d+",
        "has_figshare": r"figshare\.\d+",
        "has_chembl": r"CHEMBL\d+",
        "has_protein_family": r"IPR\d+|PF\d+",   # InterPro / Pfam
        "has_accession": r"(ENS[A-Z0-9]+|NM_\d+|CP\d+|BX\d+|K\d+)",
        "has_epi": r"EPI[_-]?[A-Z0-9]+",
        "has_sra": r"(SRR\d+|SAMN\d+|PXD\d+)",
        "has_model": r"MODEL\d+"
    }

    features = {}
    for name, pat in patterns.items():
        features[name] = [1 if re.search(pat, s) else 0 for s in sentences]

    return pd.DataFrame(features)


regex_feats_train = extract_regex_features(train["context"])
regex_feats_test = extract_regex_features(test["context"])


X_regex_train = csr_matrix(regex_feats_train)
X_regex_test = csr_matrix(regex_feats_test)


print(X_regex_train.shape)
print(X_regex_test.shape)


model = SentenceTransformer("/kaggle/input/sentence-transformersall-minilm-l6-v2/other/default/1/all-MiniLM-L6-v2")


train_embeddings = model.encode(train["context"].tolist(), 
                                convert_to_tensor=True, 
                                show_progress_bar=True)

test_embeddings = model.encode(test["context"].tolist(), 
                               convert_to_tensor=True, 
                               show_progress_bar=True)


X_sbert_train = csr_matrix(train_embeddings)
X_sbert_test = csr_matrix(test_embeddings) 


print(X_sbert_train.shape)
print(X_sbert_test.shape)


sent = SentimentIntensityAnalyzer()
train_sentiment = np.array([sent.polarity_scores(t)["compound"] for t in train["context"]]).reshape(-1,1)
X_sent_train = csr_matrix(train_sentiment)

test_sentiment = np.array([sent.polarity_scores(t)["compound"] for t in test["context"]]).reshape(-1,1)
X_sent_test = csr_matrix(test_sentiment)


print(X_sent_train.shape)
print(X_sent_test.shape)


tfidf = TfidfVectorizer(max_features=5000, 
                        ngram_range=(1,2))

X_tfidf_train = tfidf.fit_transform(train["context"])
X_tfidf_test = tfidf.transform(test["context"])


print(X_tfidf_train.shape)
print(X_tfidf_test.shape)


X = hstack([X_tfidf_train, 
            X_sbert_train, 
            X_sent_train, 
            X_regex_train], 
           format="csr")

y = train["type"]


le = LabelEncoder()
y_encode = le.fit_transform(y)  # XGB requires multiclass integer label
print(le.inverse_transform([0,1,2]))


kf = StratifiedKFold(n_splits=5, 
                     shuffle=True, 
                     random_state=42)
xgb_scores = []


# Example for class imbalance: inverse frequency sample_weight
class_counts = np.bincount(y_encode)
class_weights = {i: (len(y_encode) / (len(class_counts) * class_counts[i])) for i in range(len(class_counts))}
sample_weight = np.array([class_weights[c] for c in y_encode])

for fold, (tr, va) in enumerate(kf.split(X, y_encode), 1):
    X_tr, X_va = X[tr], X[va]
    y_tr, y_va = y_encode[tr], y_encode[va]
    w_tr, w_va = sample_weight[tr], sample_weight[va]

    xgboost = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",  
        random_state=42
    )

    xgboost.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_va, y_va)],
        early_stopping_rounds=100,
        verbose=False
    )

    y_pred = xgboost.predict(X_va)
    f1 = f1_score(y_va, y_pred, average='weighted')
    print(f"[XGB] Fold {fold} F1: {f1:.4f}")
    xgb_scores.append(f1)

print(f"[XGB] CV F1: {np.mean(xgb_scores):.4f} (+/- {np.std(xgb_scores):.4f})")


print(classification_report(y_va, y_pred))


X_test = hstack([X_tfidf_test, 
                 X_sbert_test, 
                 X_sent_test, 
                 X_regex_test], 
                format="csr")


y_test_pred = xgboost.predict(X_test)
y_test_labels = le.inverse_transform(y_test_pred)  # convert back to string labels


test["type"] = y_test_labels
test.head()


test.shape


test.drop(["label",
          "context"],
         axis = 1,
         inplace = True)


test["type"].value_counts()


test.to_csv("submission.csv", index=False, encoding="utf-8")

