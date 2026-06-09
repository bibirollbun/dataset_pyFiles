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


# CÃ©lula 1: Carregamento de SpaCy e Logger
import spacy
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    nlp = spacy.load('en_core_web_sm')
    logger.info("SpaCy carregado.")
except OSError:
    raise RuntimeError("Instale SpaCy: python -m spacy download en_core_web_sm")

BASE = '/kaggle/input/make-data-count-finding-data-references/'


# CÃ©lula 2: FunÃ§Ãµes de ExtraÃ§Ã£o, ValidaÃ§Ã£o e TF-IDF Training
import os
import re
import unicodedata
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score

# PadrÃµes restritos para accessions e DOIs completos
EXTRA_PATTERNS = [
    r"\bCHEMBL\d+\b",
    r"\bGSE\d+\b",
    r"\bzenodo\.\d+\b",
    r"\bfigshare\.\d+\b",
    r"\bdryad\.[\w-]+\b"
]
# RepositÃ³rios confiÃ¡veis
DATA_REPOS = [
    "chembl","gse","zenodo","figshare","dryad",
    "ebi","ncbi","dataverse","osf","datadryad"
]

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize('NFKD', text)
    return t.replace('â€�','-').replace('â€“','-').replace('â€”','-')


def extract_text_from_xml(path, remove_refs=False):
    try:
        raw = open(path, 'r', encoding='utf-8', errors='ignore').read()
        tree = ET.fromstring(raw)
        refs = ''
        if remove_refs:
            for node in tree.findall('.//ref-list'):
                for elem in node.iter():
                    if elem.text and elem.text.strip():
                        refs += elem.text.strip() + ' '
                node.clear()
        main = ' '.join(
            elem.text.strip() for elem in tree.iter() if elem.text and elem.text.strip()
        )
        return normalize_text(main), normalize_text(refs)
    except:
        return "", ""


def extract_sections(path):
    try:
        raw = open(path, 'r', encoding='utf-8', errors='ignore').read()
        tree = ET.fromstring(raw)
    except:
        return ""
    parts = []
    for sec in tree.findall(".//sec[@sec-type='methods']") + tree.findall(".//sec[@sec-type='data-availability']"):
        parts.append(' '.join(
            e.text.strip() for e in sec.iter() if e.text and e.text.strip()
        ))
    for sec in tree.findall('.//sec'):
        title = sec.find('title')
        if title is not None and re.search(
            r'(?i)data availability|materials', normalize_text(title.text or '')
        ):
            parts.append(' '.join(
                e.text.strip() for e in sec.iter() if e.text and e.text.strip()
            ))
    return normalize_text(' '.join(parts))


def validate_id(id_):
    return bool(re.match(
        r"^(?:10\.\d{4,}/[\w\.-]+|CHEMBL\d+|GSE\d+|zenodo\.\d+|figshare\.\d+|dryad\.[\w-]+)$",
        id_, re.IGNORECASE
    ))


def extract_dois(text):
    txt = normalize_text(text)
    found = set()
    patterns = EXTRA_PATTERNS + [r"\b10\.\d{4,}/[\w\.-]+\b"]
    for pat in patterns:
        for m in re.finditer(pat, txt, re.IGNORECASE):
            d = m.group(0)
            if validate_id(d):
                found.add(d)
    return list(found)


def get_training_data(df):
    contexts, labels = [], []
    for aid, grp in df.groupby('article_id'):
        path = os.path.join(BASE, 'train/XML', f"{aid}.xml")
        main_text, _ = extract_text_from_xml(path, remove_refs=True)
        text = extract_sections(path) or main_text
        true_ids = {d.split('/')[-1] for d in grp['dataset_id']}
        for doi in extract_dois(text):
            contexts.append(text)
            labels.append(int(doi in true_ids))
    return contexts, np.array(labels)


def train_and_tune():
    df = pd.read_csv(os.path.join(BASE, 'train_labels.csv'))
    contexts, y = get_training_data(df)
    vect = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
    X = vect.fit_transform(contexts)
    if len(set(y)) < 2:
        logger.info("Ãšnica classe detectada; pulando tuning.")
        return vect, None, 0.5
    Xtr, Xdv, ytr, ydv = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, class_weight='balanced'),
        {'C': [0.1, 1, 10]}, scoring='f1', cv=3
    )
    grid.fit(Xtr, ytr)
    clf = grid.best_estimator_
    probs = clf.predict_proba(Xdv)[:, 1]
    best_thr, best_f1 = 0.5, 0
    for t in np.linspace(0.4, 0.9, 10):
        f = f1_score(ydv, (probs >= t).astype(int))
        if f > best_f1:
            best_f1, best_thr = f, t
    logger.info(f"Dev F1={best_f1:.3f}@{best_thr}")
    return vect, clf, best_thr


# CÃ©lula 3: PrediÃ§Ã£o por Artigo + Fallback Aprimorado
vect, clf, thr = train_and_tune()
results = []
for fname in os.listdir(os.path.join(BASE,'test/XML')):
    if not fname.endswith('.xml'): continue
    aid = fname[:-4]
    path = os.path.join(BASE,'test/XML',fname)
    main_txt, _ = extract_text_from_xml(path, remove_refs=True)
    text = extract_sections(path) or main_txt

    # 1) ClassificaÃ§Ã£o TF-IDF
    preds = []
    dois = extract_dois(text)
    if clf and dois:
        X = vect.transform([text]*len(dois))
        probs = clf.predict_proba(X)[:,1]
        for doi,p in zip(dois,probs):
            if p>=thr and any(r in doi.lower() for r in DATA_REPOS):
                ctype = 'Secondary' if 'obtained from' in text.lower() else 'Primary'
                preds.append({'doi':doi,'type':ctype})

    # 2) Regex fallback apenas para outros DOIs
    raw = open(path,'r',encoding='utf-8',errors='ignore').read()
    regex_ids = {
        m.group(0)
        for pat in EXTRA_PATTERNS + [r"\b10\.\d{4,}/[\w\.-]+\b"]
        for m in re.finditer(pat, raw, re.IGNORECASE)
        if any(r in m.group(0).lower() for r in DATA_REPOS)
    }
    pred_set = {item['doi'] for item in preds}
    fallback = [{'doi':d,'type':'Primary'} for d in sorted(regex_ids) if d not in pred_set]

    # Combina e adiciona ao resultado
    for entry in preds + fallback:
        results.append({'article_id':aid,'dataset_id':entry['doi'],'type':entry['type']})


# CÃ©lula 4: FormataÃ§Ã£o e Salvamento do submission.csv
import pandas as pd

def format_doi(x):
    if x.startswith('zenodo.'):
        return f"https://doi.org/10.5281/{x}"
    elif x.startswith('dryad.'):
        return f"https://doi.org/10.5061/{x}"
    elif x.startswith('figshare.'):
        return f"https://doi.org/10.6084/m9.{x}"
    else:
        return f"https://doi.org/{x}"

sub = pd.DataFrame(results)
sub['dataset_id'] = sub['dataset_id'].apply(format_doi)
sub = sub.drop_duplicates(['article_id','dataset_id','type']).reset_index(drop=True)
sub['row_id'] = sub.index
sub = sub[['row_id','article_id','dataset_id','type']]
sub.to_csv('submission.csv', index=False)
import IPython; IPython.display.display(sub.head(10))
logger.info(f"submission.csv gerado com {len(sub)} linhas")


plt.figure(figsize=(7,4))
sns.countplot(x='type', data=labels, palette='Set2', order=['Primary', 'Secondary', 'Missing'])
plt.title('DistribuiÃ§Ã£o dos Tipos de CitaÃ§Ã£o')
plt.xlabel('Tipo')
plt.ylabel('FrequÃªncia')
plt.show()

# ComentÃ¡rios:
print("""
ğŸ”� ComentÃ¡rios:
- O grÃ¡fico mostra que a maior parte das citaÃ§Ãµes Ã© do tipo 'Secondary', ou seja, sÃ£o dados reutilizados de outros estudos.
- A quantidade considerÃ¡vel de 'Missing' indica tanto limitaÃ§Ãµes de extraÃ§Ã£o automÃ¡tica quanto desafios prÃ³prios do texto cientÃ­fico.
""")


plt.figure(figsize=(7,4))
sns.countplot(y='id_type', data=labels, palette='pastel', order=labels['id_type'].value_counts().index)
plt.title('Tipos de Identificadores de Dataset')
plt.xlabel('FrequÃªncia')
plt.ylabel('Tipo de ID')
plt.show()

# ComentÃ¡rios:
print("""
ğŸ”� ComentÃ¡rios:
- Os identificadores DOI predominam, mas hÃ¡ muitos outros padrÃµes relevantes (ex: CHEMBL, GSE), mostrando a diversidade de repositÃ³rios e bancos citados.
- Esta diversidade demanda regexs e lÃ³gica de normalizaÃ§Ã£o robustas no pipeline.
""")


datasets_por_artigo = labels.groupby('article_id')['dataset_id'].nunique()
plt.figure(figsize=(10,4))
sns.histplot(datasets_por_artigo, bins=20, log=True, color='skyblue')
plt.title('DistribuiÃ§Ã£o: Datasets por Artigo')
plt.xlabel('NÂº de datasets')
plt.ylabel('NÂº de artigos')
plt.show()

# ComentÃ¡rios:
print("""
ğŸ”� ComentÃ¡rios:
- A maioria dos artigos cita poucos datasets, mas existem outliers que mencionam dezenas.
- Isso destaca a necessidade de deduplicaÃ§Ã£o eficiente e alta cobertura (recall) para modelos competitivos.
""")



mais_citados = labels['dataset_id'].value_counts().head(10)
plt.figure(figsize=(8,4))
sns.barplot(x=mais_citados.values, y=mais_citados.index, palette='flare')
plt.title('Top 10 Datasets Mais Citados')
plt.xlabel('NÃºmero de CitaÃ§Ãµes')
plt.ylabel('dataset_id')
plt.show()

# ComentÃ¡rios:
print("""
ğŸ”� ComentÃ¡rios:
- Alguns datasets sÃ£o citados mÃºltiplas vezes, indicando conjuntos de dados centrais para a Ã¡rea.
- Esses casos podem ser benchmarks para avaliar recall e precisÃ£o do pipeline.
""")



type_counts = labels['type'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=150, explode=[0.05]*len(type_counts), colors=sns.color_palette('pastel'))
plt.title('ProporÃ§Ã£o de Tipos de CitaÃ§Ã£o')
plt.show()

# ComentÃ¡rios:
print("""
ğŸ”� ComentÃ¡rios:
- VisualizaÃ§Ã£o rÃ¡pida da proporÃ§Ã£o: citaÃ§Ãµes secundÃ¡rias dominam, mas citaÃ§Ãµes primÃ¡rias tÃªm papel crucial na avaliaÃ§Ã£o de impacto de dados abertos.
- Balancear recall e precisÃ£o entre os dois tipos Ã© fundamental para maximizar o F1-score.
""")



cross = pd.crosstab(labels['id_type'], labels['type'])
cross.plot(kind='bar', stacked=True, colormap='Set2', figsize=(8,5))
plt.title('CitaÃ§Ãµes por Tipo de Identificador e ClassificaÃ§Ã£o')
plt.ylabel('FrequÃªncia')
plt.show()

# ComentÃ¡rios:
print("""
ğŸ”� ComentÃ¡rios:
- Certos tipos de identificador (ex: DOIs) tÃªm mais chances de aparecer como 'Secondary'.
- O comportamento por tipo pode indicar oportunidades para heurÃ­sticas especÃ­ficas no pipeline.
""")


