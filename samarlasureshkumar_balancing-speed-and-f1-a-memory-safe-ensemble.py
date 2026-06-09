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


import pandas as pd
from pathlib import Path

df = pd.read_csv("/kaggle/input/make-data-count-finding-data-references/train_labels.csv")
print("Columns:", df.columns.tolist())
df.head()



import xml.etree.ElementTree as ET
from pathlib import Path

xml_file = Path("/kaggle/input/make-data-count-finding-data-references/test/XML/10.1002_ece3.5260.xml")
tree = ET.parse(xml_file)
root = tree.getroot()

# Get all paragraphs or sections (varies by journal)
texts = []
for elem in root.iter():
    if elem.tag.lower().endswith("p"):  # Paragraph
        texts.append(elem.text)

doc_text = "\n".join([t for t in texts if t])
print(doc_text[:1500])  # Preview



import os
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd

# Set paths
train_xml_dir = Path("/kaggle/input/make-data-count-finding-data-references/train/XML")
label_csv = Path("/kaggle/input/make-data-count-finding-data-references/train_labels.csv")

# Load labels
labels_df = pd.read_csv(label_csv)

# XML parser function
def extract_text_from_xml(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        texts = []
        for elem in root.iter():
            if elem.tag.lower().endswith("p") and elem.text:
                texts.append(elem.text.strip())
        return " ".join(texts)
    except Exception as e:
        return ""

# Iterate through XML files and build training dataframe
parsed_records = []
for xml_file in train_xml_dir.glob("*.xml"):
    article_id = xml_file.stem
    text = extract_text_from_xml(xml_file)
    parsed_records.append({"article_id": article_id, "text": text})

# Merge with labels
parsed_df = pd.DataFrame(parsed_records)
train_df = pd.merge(parsed_df, labels_df, on="article_id", how="inner")

# Display sample without ace_tools
print("Train Articles with Labels (Sample):")
display(train_df[['article_id', 'text', 'dataset_id', 'type']].head(10))



import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Define paths
base_dir = Path("/kaggle/input/make-data-count-finding-data-references")
train_xml_dir = base_dir / "train/XML"
train_labels_path = base_dir / "train_labels.csv"

# Load labels
labels_df = pd.read_csv(train_labels_path)

# Extract full text from XML files
def extract_text_from_xml(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        text = " ".join([elem.text.strip() for elem in root.iter() if elem.tag.lower().endswith("p") and elem.text])
        return text
    except Exception:
        return ""

# Process XML files and join with labels
xml_texts = []
for xml_file in train_xml_dir.glob("*.xml"):
    article_id = xml_file.stem
    text = extract_text_from_xml(xml_file)
    xml_texts.append({"article_id": article_id, "text": text})

xml_df = pd.DataFrame(xml_texts)
train_df = pd.merge(labels_df, xml_df, on="article_id", how="inner")

# Prepare final training data by combining dataset_id and article_id to predict 'type'
train_df = train_df.dropna(subset=["text", "type"]).reset_index(drop=True)
train_df["label"] = train_df["type"].map({"Primary": 0, "Secondary": 1, "Missing": 2})

# Display sample without ace_tools
print("Train Articles with Labels (Sample):")
display(train_df[['article_id', 'text', 'dataset_id', 'type']].head(10))



############################################################
# 1 ── Imports
############################################################
import pandas as pd, numpy as np, torch
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, DataCollatorWithPadding, TrainingArguments)
import datasets, inspect




############################################################
# 2 ── Helper: TrainingArguments filter (works on any HF version)
############################################################
def make_args(**kw):
    valid = inspect.signature(TrainingArguments).parameters
    return TrainingArguments(**{k:v for k,v in kw.items() if k in valid})

############################################################
# 3 ── Load XML ↦ text  + label CSVs
############################################################
BASE = Path("/kaggle/input/make-data-count-finding-data-references")
TRAIN_XML, TEST_XML = BASE/"train/XML", BASE/"test/XML"
train_lbl = pd.read_csv(BASE/"train_labels.csv")
sample_sub = pd.read_csv(BASE/"sample_submission.csv")

def xml2txt(p: Path) -> str:
    try:
        root = ET.parse(p).getroot()
        return re.sub(r"\s+"," "," ".join(t.strip() for t in root.itertext()
                                          if t and t.strip()))
    except: return ""

print("Parsing XML …")
train_txt = {f.stem: xml2txt(f) for f in TRAIN_XML.glob("*.xml")}
test_txt  = {f.stem: xml2txt(f) for f in TEST_XML .glob("*.xml")}

train = train_lbl.assign(text=lambda d: d.article_id.map(train_txt)).dropna(subset=["text"])
test  = sample_sub.assign(text=lambda d: d.article_id.map(test_txt)).fillna({"text":""})

lab2id = {"Primary":0,"Secondary":1,"Missing":2}; id2lab={v:k for k,v in lab2id.items()}
train["y"] = train.type.map(lab2id)





############################################################
# 3 ── TF-IDF + Linear SVM + Probability Calibration
############################################################
word_tf = TfidfVectorizer(analyzer="word", ngram_range=(1,2),
                          min_df=2, max_features=100_000,
                          stop_words="english", sublinear_tf=True)
char_tf = TfidfVectorizer(analyzer="char", ngram_range=(3,5),
                          min_df=2, max_features=150_000,
                          sublinear_tf=True)

vect = ColumnTransformer([("w", word_tf, "text"),
                          ("c", char_tf, "text")])
svm = LinearSVC(C=1.0, class_weight="balanced", dual=False)

tfidf_clf = Pipeline([("vect", vect), ("svm", svm)]).fit(tr, tr.y)
tfidf_cal = CalibratedClassifierCV(tfidf_clf, method="sigmoid", cv="prefit").fit(va, va.y)

proba_va = tfidf_cal.predict_proba(va)
f1 = f1_score(va.y, proba_va.argmax(1), average="macro")
print(f"✅ TF-IDF macro-F1 on validation: {f1:.4f}")

############################################################
# 4 ── Predict on Test Set and Save Submission
############################################################
proba_test = tfidf_cal.predict_proba(test)
test["type"] = [id2lab[i] for i in proba_test.argmax(1)]

test[["article_id", "dataset_id", "type"]].to_csv("submission.csv", index=False)
print("✅ submission.csv saved with", len(test), "rows using TF-IDF only")



print("submission.csv exists:", os.path.exists("submission.csv"))
print("submission.csv preview:")
print(test[["article_id", "dataset_id", "type"]].head())
print("Rows in submission:", len(test))



def xml2txt(p: Path) -> str:
    try:
        root = ET.parse(p).getroot()
        text = " ".join(t.strip() for t in root.itertext() if t.strip())
        return re.sub(r"\s+", " ", text)
    except Exception as e:
        print(f"❌ Failed to parse {p.name}: {e}")
        return ""



print("Total test XML files:", len(list(TEST_XML.glob('*.xml'))))
print("Parsed texts:", len(test_txt))
print("Non-empty parsed test rows:", sum(bool(v.strip()) for v in test_txt.values()))



print("Example article_id in sample_sub:", sample_sub['article_id'].iloc[0])
print("Example XML filename stem:", next(TEST_XML.glob("*.xml")).stem)



def normalize_id(s): return s.replace("/", "_")
train["article_id"] = train["article_id"].apply(normalize_id)
test["article_id"] = test["article_id"].apply(normalize_id)
test_txt = {normalize_id(f.stem): xml2txt(f) for f in TEST_XML.glob("*.xml")}



print("submission.csv exists:", os.path.exists("submission.csv"))
print("Rows in submission:", len(test))



print("Total test XML files:", len(list(TEST_XML.glob("*.xml"))))  # Should print 875


