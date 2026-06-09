# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os, re, sys, json, math, gc, random, string, itertools, unicodedata
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable, Optional
import pandas as pd
import numpy as np
from lxml import etree
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.tokenize import sent_tokenize

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DATA_DIR = '/kaggle/input/make-data-count-finding-data-references'
TRAIN_PDF_DIR = os.path.join(DATA_DIR, 'train', 'PDF')
TRAIN_XML_DIR = os.path.join(DATA_DIR, 'train', 'XML')
TEST_PDF_DIR = os.path.join(DATA_DIR, 'test', 'PDF')
TEST_XML_DIR = os.path.join(DATA_DIR, 'test', 'XML')

DOI_REGEX_CORE = r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+"
DOI_REGEX = re.compile(rf"(?:https?://(?:dx\.)?doi\.org/)?({DOI_REGEX_CORE})", re.I)
ACC_PATTERNS = [
    ("GEO", re.compile(r"\bGSE\d+\b", re.I)),
    ("ArrayExpress", re.compile(r"\bE-[A-Z]{3,5}-\d+\b", re.I)),
    ("ENA/PRJ", re.compile(r"\bPRJ[EDN][A-Z]?\d+\b", re.I)),
    ("PDB", re.compile(r"\b(?:PDB\s*)?([0-9][A-Za-z0-9]{3})\b", re.I)),
    ("Chembl", re.compile(r"\bCHEMBL\d+\b", re.I)),
]
SECTION_HINTS_PRIMARY = re.compile(r"\b(data (?:availability|accessibility)|we (?:generated|collected|produced)|this study (?:generated|produced)|deposited in|uploaded to)\b", re.I)
SECTION_HINTS_SECONDARY = re.compile(r"\b(obtained from|downloaded from|sourced from|previously published|existing (?:records|data))\b", re.I)

def normalize_doi(text: str) -> Optional[str]:
    m = DOI_REGEX.search(text)
    if not m: return None
    core = m.group(1)
    core = core.rstrip(').,;')
    return f"https://doi.org/{core}"

def extract_dois(text: str) -> List[str]:
    out = []
    for m in DOI_REGEX.finditer(text):
        core = m.group(1).rstrip(').,;')
        out.append(f"https://doi.org/{core}")
    return out

def extract_accessions(text: str) -> List[str]:
    ids = []
    for name, rx in ACC_PATTERNS:
        for m in rx.finditer(text):
            if name == 'PDB':
                gid = m.group(1).upper()
                if len(gid)==4: ids.append(f"PDB {gid}")
            else:
                ids.append(m.group(0).upper())
    return ids

def uniq(seq):
    seen = set(); out = []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def read_xml_sections(xml_path: str) -> List[Tuple[str,str]]:
    try:
        tree = etree.parse(xml_path)
        root = tree.getroot()
        ns = {k:v for k,v in root.nsmap.items() if k}
        if None in root.nsmap:
            ns['ns'] = root.nsmap[None]
        texts = []
        for sec in root.findall('.//sec'):
            title = ''.join(sec.xpath('string(child::title)'))
            text = ''.join(sec.xpath('string(.)'))
            if text:
                texts.append((title or '', text))
        if not texts:
            body = ''.join(root.xpath('string(.)'))
            return [("", body)]
        return texts
    except Exception as e:
        return []

def generate_candidates_from_text(text: str) -> List[str]:
    cands = []
    cands.extend(extract_dois(text))
    cands.extend(extract_accessions(text))
    return uniq(cands)

@dataclass
class Mention:
    dataset_id: str
    sentence: str
    section_title: str
    article_id: str

def mentions_from_sections(article_id: str, sections: List[Tuple[str,str]]) -> List[Mention]:
    out = []
    for title, text in sections:
        for sent in sent_tokenize(text):
            ids = generate_candidates_from_text(sent)
            for gid in ids:
                out.append(Mention(gid, sent, title, article_id))
    return out

def weak_label_type(m: Mention) -> Optional[str]:
    s = m.sentence
    if re.search(r"\b(deposited|archived|available at|accessible at)\b", s, re.I):
        return 'Primary'
    if re.search(r"\b(obtained from|sourced from|downloaded from|previously published)\b", s, re.I):
        return 'Secondary'
    if re.search(r"\bmethods|data (availability|accessibility)\b", m.section_title, re.I):
        if SECTION_HINTS_PRIMARY.search(s):
            return 'Primary'
        if SECTION_HINTS_SECONDARY.search(s):
            return 'Secondary'
    return None

class SimpleSpanClassifier(nn.Module):
    def __init__(self, vocab_size=30000, emb_dim=128, hidden=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden//2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(hidden, 2)
    def forward(self, x, mask=None):
        e = self.emb(x)
        o,_ = self.lstm(e)
        logits = self.out(o)
        return logits

CUE_WORDS_PRIMARY = [
    'generated','collected','produced','acquired','deposited','submitted','made available','this study','present study','we provide','we collected'
]
CUE_WORDS_SECONDARY = [
    'obtained from','downloaded from','sourced from','previously published','publicly available from','reused','derived from','existing data'
]

class PrimarySecondaryClassifier:
    def __init__(self):
        self.vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=200000)
        self.clf = LogisticRegression(max_iter=200)
    def fit(self, texts: List[str], labels: List[str]):
        X = self.vec.fit_transform(texts)
        self.clf.fit(X, labels)
    def predict(self, texts: List[str]):
        X = self.vec.transform(texts)
        return self.clf.predict(texts=X)
    def predict_proba(self, texts: List[str]):
        X = self.vec.transform(texts)
        return self.clf.predict_proba(X)

def run_article(article_id: str, xml_path: Optional[str], pdf_path: Optional[str]) -> List[Tuple[str,str,str]]:
    sections = []
    if xml_path and os.path.exists(xml_path):
        sections = read_xml_sections(xml_path)
    if not sections and pdf_path and os.path.exists(pdf_path):
        text = ''
        sections = [("", text)]
    mentions = mentions_from_sections(article_id, sections)
    preds = []
    for m in mentions:
        t = weak_label_type(m)
        if t is None:
            if re.search(r"available|deposited|submitted", m.sentence, re.I): t = 'Primary'
            elif re.search(r"from\s+(?:GEO|ArrayExpress|ENA|SRA|PDB|Dryad|Figshare|Zenodo|Dataverse|\w+ repository)", m.sentence, re.I): t = 'Secondary'
        if t:
            preds.append((article_id, m.dataset_id, t))
    uniq_preds = {}
    for a, d, t in preds:
        uniq_preds[(d,t)] = (a,d,t)
    filtered = []
    for a,d,t in uniq_preds.values():
        if d.startswith('PDB '):
            ok = any(('PDB' in m.sentence) and (m.dataset_id==d) for m in mentions)
            if not ok: continue
        filtered.append((a,d,t))
    return filtered

def write_submission(pred_rows: List[Tuple[str,str,str]], path='submission.csv'):
    rows = []
    for i,(a,d,t) in enumerate(pred_rows):
        if re.match(DOI_REGEX_CORE, d):
            d = f"https://doi.org/{d}"
        rows.append((i, a, d, t))
    df = pd.DataFrame(rows, columns=['row_id','article_id','dataset_id','type'])
    df.to_csv(path, index=False)
    print('Wrote', path, 'with', len(df), 'rows')

def evaluate_offline():
    labels = pd.read_csv(os.path.join(DATA_DIR,'train_labels.csv'))
    gold = {}
    for _,r in labels.iterrows():
        gold.setdefault(r.article_id, set()).add((r.dataset_id if r.dataset_id.startswith('http') else r.dataset_id.upper(), r.type))
    preds_all = []
    for a in os.listdir(TRAIN_PDF_DIR):
        if not a.endswith('.pdf'): continue
        article_doi = a[:-4]
        xml = os.path.join(TRAIN_XML_DIR, f"{article_doi}.xml")
        pdf = os.path.join(TRAIN_PDF_DIR, a)
        preds = run_article(article_doi, xml if os.path.exists(xml) else None, pdf)
        for p in preds:
            preds_all.append(p)
    pred_map = {}
    for a,d,t in preds_all:
        pred_map.setdefault(a, set()).add((d, t))
    tps=fps=fns=0
    for a in set(list(gold.keys()) + list(pred_map.keys())):
        g = gold.get(a, set())
        p = pred_map.get(a, set())
        tps += len(g & p)
        fps += len(p - g)
        fns += len(g - p)
    prec = tps/(tps+fps+1e-9)
    rec = tps/(tps+fns+1e-9)
    f1 = 2*prec*rec/(prec+rec+1e-9)
    print({'tp':tps,'fp':fps,'fn':fns,'precision':prec,'recall':rec,'f1':f1})

def predict_test():
    all_preds = []
    for a in os.listdir(TEST_PDF_DIR):
        if not a.endswith('.pdf'): continue
        article_doi = a[:-4]
        xml = os.path.join(TEST_XML_DIR, f"{article_doi}.xml")
        pdf = os.path.join(TEST_PDF_DIR, a)
        preds = run_article(article_doi, xml if os.path.exists(xml) else None, pdf)
        all_preds.extend(preds)
    write_submission(all_preds, path='submission.csv')

if __name__ == '__main__':
    predict_test()


