import importlib

# List of packages to check
packages = ["pandas", "numpy", "spacy", "scikit-learn", "joblib", "PyPDF2"]

# Check each package
for package in packages:
    try:
        importlib.import_module(package)
        print(f"✅ {package} is installed.")
    except ImportError:
        print(f"❌ {package} is NOT installed.")


import os
import re
import joblib
import numpy as np
import pandas as pd
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer

warnings.filterwarnings('ignore')


try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None
    print("Run `python -m spacy download en_core_web_sm` to enable NLP features.")

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x  # fallback


class DataCitationDetector:
    def __init__(self):
        self.vectorizer = DictVectorizer(sparse=False)
        self.label_encoder = LabelEncoder()
        self.nlp = nlp

        self.classifier = VotingClassifier(estimators=[
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)),
            ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
        ], voting='soft')

        self.primary_patterns = [
            r'\b(?:we\s+)?(?:generated|collected|produced|created|measured|recorded|obtained|acquired)\b',
            r'\bthis\s+study\b', r'\bour\s+(?:data|dataset|measurements|results)\b',
            r'\bnew\s+data\b', r'\boriginal\s+data\b', r'\bnewly\s+(?:generated|collected)\b',
            r'\bspecifically\s+(?:for\s+)?this\s+(?:study|work)\b', r'\bin-house\s+(?:generated|created)\b'
        ]

        self.secondary_patterns = [
            r'\bobtained\s+from\b', r'\breused?\b', r'\bexisting\s+data\b',
            r'\bpreviously\s+(?:published|reported)\b', r'\bderived\s+from\b',
            r'\bopen\s+data\b', r'\bthird-party\s+data\b', r'\bbenchmark\s+data\b'
        ]

        self.section_patterns = {
            'methods': r'\bmethods?\b', 'results': r'\bresults?\b', 'introduction': r'\bintroduction\b',
            'discussion': r'\bdiscussion\b', 'references': r'\breferences?\b',
            'data_availability': r'\bdata\s+availability\b'
        }

        self.citation_patterns = {
            'doi': r'(?:https?://)?(?:dx\.)?doi\.org/(10\.\d{4,9}/[\w./-]+)',
            'zenodo': r'(?:https?://)?zenodo\.org/record/(\d+)',
            'github': r'(?:https?://)?github\.com/[\w\-_]+/[\w\-_]+',
            'gse': r'GSE\d+', 'sra': r'SRA\d+', 'prjna': r'PRJNA\d+',
            'chembl': r'CHEMBL\d+', 'pdb': r'PDB:\w+', 'uniprot': r'UniProt:\w+'
        }

    def extract_text_from_xml(self, xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            return ' '.join(elem.text.strip() for elem in root.iter() if elem.text)
        except:
            return ""

    def extract_text_from_pdf(self, pdf_path):
        try:
            if fitz:
                doc = fitz.open(pdf_path)
                text = "".join([page.get_text() for page in doc])
                doc.close()
                return text
        except:
            return ""
        return ""

    def load_documents(self, base_path, split='train'):
        documents = {}
        for xml_file in Path(base_path, split, 'XML').glob('*.xml'):
            documents[xml_file.stem] = self.extract_text_from_xml(xml_file)
        for pdf_file in Path(base_path, split, 'PDF').glob('*.pdf'):
            if pdf_file.stem not in documents:
                documents[pdf_file.stem] = self.extract_text_from_pdf(pdf_file)
        return documents

    def extract_data_citations(self, text):
        citations = set()
        for name, pattern in self.citation_patterns.items():
            for match in re.findall(pattern, text, re.IGNORECASE):
                if name == 'doi':
                    citations.add(f"https://doi.org/{match}")
                elif name == 'zenodo':
                    citations.add(f"https://zenodo.org/record/{match}")
                else:
                    citations.add(match.strip().rstrip('.,;)'))
        return list(citations)

    def create_features(self, text, citation):
        features = {}
        text_lower = text.lower()
        citation_lower = citation.lower()
        context_match = re.search(f'.{{0,500}}{re.escape(citation_lower)}.{{0,500}}', text_lower)
        context = context_match.group() if context_match else text_lower

        features['text_length'] = len(text)
        features['citation_count'] = text_lower.count(citation_lower)

        for i, pattern in enumerate(self.primary_patterns):
            features[f'primary_{i}'] = len(re.findall(pattern, context, re.IGNORECASE))
        for i, pattern in enumerate(self.secondary_patterns):
            features[f'secondary_{i}'] = len(re.findall(pattern, context, re.IGNORECASE))

        for sec, sec_pattern in self.section_patterns.items():
            features[f'in_{sec}'] = int(bool(re.search(sec_pattern, context)))

        features['is_doi'] = int('doi.org' in citation_lower)
        features['is_github'] = int('github.com' in citation_lower)
        features['is_database_id'] = int(any(x in citation_lower for x in ['chembl', 'gse', 'sra', 'prjna']))

        features['near_figure'] = len(re.findall(r'fig(?:ure)?\s*\d+', context, re.IGNORECASE))
        features['near_table'] = len(re.findall(r'table\s*\d+', context, re.IGNORECASE))
        features['near_supplement'] = len(re.findall(r'supplement', context, re.IGNORECASE))

        if self.nlp:
            doc = self.nlp(context[:1000])
            features['num_entities'] = len(doc.ents)
        else:
            features['num_entities'] = 0

        return features

    def train_model(self, base_path):
        print("Training model...")
        documents = self.load_documents(base_path, 'train')
        labels_df = pd.read_csv(Path(base_path) / 'train_labels.csv')

        X_dicts, y = [], []
        for _, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
            article_id, dataset_id, citation_type = row['article_id'], row['dataset_id'], row['type']
            if article_id in documents:
                features = self.create_features(documents[article_id], dataset_id)
                X_dicts.append(features)
                y.append(citation_type)

        if not X_dicts:
            print("No training data found.")
            return

        X = self.vectorizer.fit_transform(X_dicts)
        y_enc = self.label_encoder.fit_transform(y)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(X, y_enc):
            self.classifier.fit(X[train_idx], y_enc[train_idx])
            preds = self.classifier.predict(X[val_idx])
            scores.append(f1_score(y_enc[val_idx], preds, average='weighted'))

        print(f"Cross-val F1: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        self.classifier.fit(X, y_enc)

    def predict(self, base_path):
        print("Generating predictions...")
        documents = self.load_documents(base_path, 'test')
        predictions = []
        row_id = 0

        for article_id, text in tqdm(documents.items()):
            citations = self.extract_data_citations(text)
            for citation in citations:
                features = self.create_features(text, citation)
                X = self.vectorizer.transform([features])
                pred = self.classifier.predict(X)[0]
                citation_type = self.label_encoder.inverse_transform([pred])[0]
                predictions.append({
                    'row_id': row_id,
                    'article_id': article_id,
                    'dataset_id': citation,
                    'type': citation_type
                })
                row_id += 1

        return pd.DataFrame(predictions)

    def save_model(self, path):
        joblib.dump({
            'classifier': self.classifier,
            'label_encoder': self.label_encoder,
            'vectorizer': self.vectorizer
        }, path)

    def load_model(self, path):
        data = joblib.load(path)
        self.classifier = data['classifier']
        self.label_encoder = data['label_encoder']
        self.vectorizer = data['vectorizer']


if __name__ == "__main__":
    base_path = "/kaggle/input/make-data-count-finding-data-references"
    detector = DataCitationDetector()
    detector.train_model(base_path)
    df = detector.predict(base_path)
    df.to_csv("submission.csv", index=False)
    print(f"Saved {len(df)} predictions to submission.csv")

