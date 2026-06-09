#!pip install PyMuPDF
#!pip install PyPDF2

print("ok")


import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier


import fitz  # PyMuPDF
#import PyPDF2

from pathlib import Path
from tqdm.auto import tqdm


import warnings
warnings.filterwarnings("ignore")
print("ok")


def standardize_doi(doi):
    """Standardize DOI format"""
    if pd.isna(doi):
        return doi
    doi = str(doi).strip()
    if doi.startswith('http'):
        return doi.lower()
    if doi.startswith('doi:'):
        return 'https://doi.org/' + doi[4:].lower()
    if doi.startswith('10.'):
        return 'https://doi.org/' + doi.lower()
    return doi.lower()
print("ok")


def find_accession_ids_in_text(text):
    """Find common accession ID patterns in text"""
    patterns = [
        r'\b(GSE\d+)\b',  # GEO
        r'\b(PRJ[ENAD]\d+)\b',  # ENA/NCBI projects
        r'\b(SRP\d+)\b',  # SRA projects
        r'\b(E-[A-Z]+\-\d+)\b',  # ArrayExpress
        r'\b(pdb\s[\d\w]+)\b',  # Protein Data Bank
        r'\b(CHEMBL\d+)\b',  # ChEMBL
        r'\b(PDB\s[\d\w]+)\b',  # Protein Data Bank alternate format
    ]
    
    matches = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text, re.IGNORECASE))
    
    # Standardize format
    standardized = []
    for match in matches:
        if isinstance(match, tuple):
            match = match[0]
        standardized.append(match.lower().replace(' ', ''))
    
    return standardized

print("ok")


def find_doi_in_text(text):
    """Find all DOIs in text"""
    doi_pattern = r'\b(10[.][0-9]{4,}(?:[.][0-9]+)*/(?:(?!["&\'<>])\S)+)\b'
    dois = re.findall(doi_pattern, text.lower())
    return [standardize_doi(doi) for doi in dois]
print("ok")


def find_all_dataset_ids(text):
    """Find all potential dataset IDs in text"""
    dois = find_doi_in_text(text)
    accession_ids = find_accession_ids_in_text(text)
    return dois + accession_ids
print("ok")


pdf_directory = "/kaggle/input/make-data-count-finding-data-references/test/PDF"

all_data = []
texts = []

for filename in tqdm(os.listdir(pdf_directory), total=len(os.listdir(pdf_directory))):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, filename)
        
        # Extract article_id from filename
        article_id = filename.split(".pdf")[0]
        text = ""
        doc = fitz.open(pdf_path)
        for page in doc:
            page_text = page.get_text().lower()
            if 'references' in page_text:
                page_text = page_text.split("references")[0]
                text += page_text
                break
            else:
                text += page_text

        doc.close()
        texts.append((article_id, text))



chunks = []
sum_=0
for article_id, text in texts:
    dataset_id_list=find_all_dataset_ids(text)
    chunks.append((article_id, dataset_id_list))
    #print(len(dataset_id_list))
    sum_+=len(dataset_id_list)
print(sum_)
print("ok")


data=[]
id_=0
temp=[]
print(chunks[0][1])
for k in chunks:
    for j in k[1]:
        if j not in temp:
            data.append((id_,k[0],j,'Secondary'))
            id_+=1
            temp.append(j)

print("ok")



# 定义列名
column_names = ["row_id", "article_id", "dataset_id", "type"]
# 创建 DataFrame
df = pd.DataFrame(data, columns=column_names)
# 写入 CSV 文件
df.to_csv('submission.csv', index=False)
print("CSV 文件已创建成功。")





