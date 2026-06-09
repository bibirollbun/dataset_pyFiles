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



import pandas as pd
import numpy as np
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# NLP and ML libraries
import spacy
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
import joblib


def create_features(self, text, citation):
        """Create comprehensive features for ML classification"""
        features = {}
        text_lower = text.lower()
        citation_lower = citation.lower()
        
        # Basic text features
        features['text_length'] = len(text)
        features['citation_count'] = text_lower.count(citation_lower)
        
        # Find context around citation (wider window)
        citation_escaped = re.escape(citation_lower)
        context_match = re.search(f'.{{0,500}}{citation_escaped}.{{0,500}}', text_lower)
        context = context_match.group() if context_match else text_lower
        
        # Pattern-based features with better scoring
        primary_score = 0
        secondary_score = 0
        
        for pattern in self.primary_patterns:
            matches = len(re.findall(pattern, context, re.IGNORECASE))
            primary_score += matches
            features[f'primary_{pattern[:20]}'] = matches
        
        for pattern in self.secondary_patterns:
            matches = len(re.findall(pattern, context, re.IGNORECASE))
            secondary_score += matches
            features[f'secondary_{pattern[:20]}'] = matches
        
        features['primary_total'] = primary_score
        features['secondary_total'] = secondary_score
        features['primary_secondary_ratio'] = primary_score / (secondary_score + 1)
        
        # Section-based features
        for section_name, section_pattern in self.section_patterns.items():
            # Check if citation appears in this section
            section_matches = list(re.finditer(section_pattern, text_lower, re.IGNORECASE))
            in_section = 0
            for match in section_matches:
                section_start = max(0, match.start() - 1000)
                section_end = min(len(text_lower), match.end() + 2000)
                section_text = text_lower[section_start:section_end]
                if citation_lower in section_text:
                    in_section = 1
                    break
            features[f'in_{section_name}'] = in_section
        
        # Citation type features based on format
        features['is_doi'] = 1 if 'doi.org' in citation_lower else 0
        features['is_zenodo'] = 1 if 'zenodo' in citation_lower else 0
        features['is_github'] = 1 if 'github' in citation_lower else 0
        features['is_database_id'] = 1 if any(pattern in citation_lower for pattern in ['chembl', 'gse', 'sra', 'prjna']) else 0
        
        # Proximity features
        features['near_figure'] = len(re.findall(r'fig(?:ure)?\s*\d+', context, re.IGNORECASE))
        features['near_table'] = len(re.findall(r'table\s*\d+', context, re.IGNORECASE))
        features['near_supplement'] = len(re.findall(r'supplement', context, re.IGNORECASE))
        
        # Linguistic features
        if self.nlp:
            doc = self.nlp(context[:1000])  # Limit for performance
            features['num_entities']


# PDF processing
try:
    import PyPDF2
    import fitz  # PyMuPDF
except ImportError:
    print("PDF processing libraries not available. Install PyPDF2 and PyMuPDF if needed.")

# Additional NLP libraries
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Transformers not available. Using traditional NLP methods.")
    TRANSFORMERS_AVAILABLE = False


class DataCitationDetector:
    def __init__(self):
        # Load spacy model for NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Please install spacy English model: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Initialize components
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
        # Ensemble of classifiers
        self.rf_classifier = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight='balanced',
            max_depth=10
        )
        
        self.gb_classifier = GradientBoostingClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=6
        )
        
        self.lr_classifier = LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        )
        
        # Voting classifier
        self.classifier = VotingClassifier(
            estimators=[
                ('rf', self.rf_classifier),
                ('gb', self.gb_classifier),
                ('lr', self.lr_classifier)
            ],
            voting='soft'
        )
        
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Enhanced data citation patterns
        self.data_patterns = [
            r'(?:https?://)?doi\.org/[\w\./\-]+',
            r'(?:https?://)?zenodo\.\w+/[\w\./\-]+',
            r'(?:https?://)?figshare\.\w+/[\w\./\-]+',
            r'(?:https?://)?github\.com/[\w\./\-]+',
            r'(?:https?://)?dryad\.\w+/[\w\./\-]+',
            r'(?:https?://)?datadryad\.org/[\w\./\-]+',
            r'(?:https?://)?ncbi\.nlm\.nih\.gov/[\w\./\-]+',
            r'(?:https?://)?ebi\.ac\.uk/[\w\./\-]+',
            r'(?:https?://)?arrayexpress\.[\w\./\-]+',
            r'(?:https?://)?geo\.ncbi\.[\w\./\-]+',
            r'CHEMBL\d+',
            r'GSE\d+',
            r'SRA\d+',
            r'PRJNA\d+',
            r'PDB:\w+',
            r'UniProt:\w+',
            r'data(?:set|base)?[\s\w]*(?:available|accessible|deposited)',
            r'supplementary (?:data|material)',
            r'supporting information',
            r'raw data',
            r'processed data',
            r'publicly available',
            r'obtained from',
            r'downloaded from',
            r'retrieved from',
            r'deposited (?:in|at)',
            r'archived (?:in|at)'
        ]
        
        # Enhanced Primary vs Secondary indicators
        self.primary_patterns = [
            r'\b(?:we\s+)?(?:generated|collected|produced|created|measured|recorded|obtained|acquired)\b',
            r'\bthis\s+study\b',
            r'\bour\s+(?:data|dataset|measurements|results)\b',
            r'\bnew\s+data\b',
            r'\boriginal\s+data\b',
            r'\b(?:newly|fresh(?:ly)?)\s+(?:generated|collected|obtained)\b',
            r'\bspecific(?:ally)?\s+(?:for\s+)?this\s+(?:study|work|research)\b',
            r'\b(?:custom|purpose-built|specifically designed)\b',
            r'\b(?:in-house|internal(?:ly)?)\s+(?:generated|developed|created)\b'
        ]
        
        self.secondary_patterns = [
            r'\b(?:obtained|downloaded|retrieved|sourced|derived|extracted)\s+from\b',
            r'\breused?\b',
            r'\bexisting\s+(?:data|dataset)\b',
            r'\bprevious(?:ly)?\s+(?:published|reported|described)\b',
            r'\bpublished\s+data\b',
            r'\bderived\s+from\b',
            r'\bbased\s+on\b',
            r'\b(?:publicly\s+)?available\s+(?:data|dataset)\b',
            r'\bopen\s+(?:data|source)\b',
            r'\b(?:third-party|external)\s+(?:data|source)\b',
            r'\breference\s+(?:data|dataset)\b',
            r'\bbenchmark\s+(?:data|dataset)\b'
        ]
        
        # Section identifiers for better context
        self.section_patterns = {
            'methods': r'\b(?:methods?|methodology|experimental\s+(?:design|procedure)|materials?\s+and\s+methods?)\b',
            'results': r'\b(?:results?|findings|outcomes?)\b',
            'introduction': r'\b(?:introduction|background)\b',
            'discussion': r'\b(?:discussion|conclusion)\b',
            'references': r'\b(?:references?|bibliography|citations?)\b',
            'data_availability': r'\b(?:data\s+availability|code\s+availability|software\s+availability)\b'
        }

    def extract_text_from_xml(self, xml_path):
        """Extract text content from XML files"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Extract text from all elements
            text_parts = []
            for elem in root.iter():
                if elem.text:
                    text_parts.append(elem.text.strip())
                if elem.tail:
                    text_parts.append(elem.tail.strip())
            
            return ' '.join(filter(None, text_parts))
        except Exception as e:
            print(f"Error processing XML {xml_path}: {e}")
            return ""

    def extract_text_from_pdf(self, pdf_path):
        """Extract text content from PDF files"""
        try:
            # Try PyMuPDF first
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except:
            try:
                # Fallback to PyPDF2
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                return text
            except Exception as e:
                print(f"Error processing PDF {pdf_path}: {e}")
                return ""

    def load_documents(self, base_path, split='train'):
        """Load and process documents from XML and PDF directories"""
        documents = {}
        
        # Process XML files
        xml_dir = Path(base_path) / split / 'XML'
        if xml_dir.exists():
            for xml_file in xml_dir.glob('*.xml'):
                article_id = xml_file.stem
                text = self.extract_text_from_xml(xml_file)
                documents[article_id] = text
        
        # Process PDF files
        pdf_dir = Path(base_path) / split / 'PDF'
        if pdf_dir.exists():
            for pdf_file in pdf_dir.glob('*.pdf'):
                article_id = pdf_file.stem
                if article_id not in documents:  # Only add if not already processed from XML
                    text = self.extract_text_from_pdf(pdf_file)
                    documents[article_id] = text
        
        return documents

    def extract_data_citations(self, text):
        """Extract potential data citations from text with improved pattern matching"""
        citations = []
        text_lower = text.lower()
        
        # Find DOI patterns with better cleaning
        doi_matches = re.findall(r'(?:https?://)?(?:dx\.)?doi\.org/(10\.\d+/[^\s\)\]\},;]+)', text, re.IGNORECASE)
        for doi in doi_matches:
            full_doi = f"https://doi.org/{doi}"
            citations.append(full_doi)
        
        # Find database-specific patterns
        database_patterns = {
            'zenodo': r'(?:https?://)?zenodo\.org/record/(\d+)',
            'figshare': r'(?:https?://)?figshare\.com/articles/[^\s\)\]\},;]+',
            'github': r'(?:https?://)?github\.com/[^\s\)\]\},;]+',
            'dryad': r'(?:https?://)?(?:datadryad\.org|dryad\.org)/[^\s\)\]\},;]+',
            'chembl': r'CHEMBL\d+',
            'geo': r'GSE\d+',
            'sra': r'SRA\d+',
            'bioproject': r'PRJNA\d+',
            'pdb': r'PDB:\w+',
            'uniprot': r'UniProt:\w+'
        }
        
        for db_name, pattern in database_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if db_name == 'zenodo':
                    citations.append(f"https://zenodo.org/record/{match}")
                elif isinstance(match, str):
                    if not match.startswith('http') and db_name in ['figshare', 'github', 'dryad']:
                        citations.append(f"https://{match}")
                    else:
                        citations.append(match)
        
        # Remove duplicates and clean
        cleaned_citations = []
        seen = set()
        for citation in citations:
            # Normalize citation
            citation = citation.strip().rstrip('.,;)')
            if citation and citation not in seen:
                seen.add(citation)
                cleaned_citations.append(citation)
        
        return cleaned_citations

    def classify_citation_type(self, text, citation):
        """Classify citation as Primary or Secondary based on context"""
        # Find the context around the citation
        citation_pattern = re.escape(citation)
        context_match = re.search(f'.{{0,200}}{citation_pattern}.{{0,200}}', text, re.IGNORECASE)
        
        if context_match:
            context = context_match.group().lower()
        else:
            context = text.lower()
        
        # Count primary and secondary indicators
        primary_score = sum(1 for pattern in self.primary_patterns 
                          if re.search(pattern, context, re.IGNORECASE))
        secondary_score = sum(1 for pattern in self.secondary_patterns 
                            if re.search(pattern, context, re.IGNORECASE))
        
        # Default classification logic
        if primary_score > secondary_score:
            return 'Primary'
        elif secondary_score > primary_score:
            return 'Secondary'
        else:
            # Use additional heuristics
            if any(word in context for word in ['our', 'this study', 'generated', 'collected']):
                return 'Primary'
            else:
                return 'Secondary'

    def create_features(self, text, citation):
        """Create features for ML classification"""
        features = {}
        text_lower = text.lower()
        
        # Basic text features
        features['text_length'] = len(text)
        features['citation_count'] = text_lower.count(citation.lower())
        
        # Context features
        citation_pattern = re.escape(citation.lower())
        context_match = re.search(f'.{{0,200}}{citation_pattern}.{{0,200}}', text_lower)
        context = context_match.group() if context_match else text_lower
        
        # Pattern-based features
        for i, pattern in enumerate(self.primary_patterns):
            features[f'primary_pattern_{i}'] = len(re.findall(pattern, context, re.IGNORECASE))
        
        for i, pattern in enumerate(self.secondary_patterns):
            features[f'secondary_pattern_{i}'] = len(re.findall(pattern, context, re.IGNORECASE))
        
        # Section-based features (heuristic)
        features['in_methods'] = 1 if 'method' in context else 0
        features['in_results'] = 1 if 'result' in context else 0
        features['in_references'] = 1 if 'reference' in context else 0
        
        return features

    def train_model(self, base_path):
        """Train the classification model"""
        print("Loading training data...")
        
        # Load documents
        documents = self.load_documents(base_path, 'train')
        
        # Load labels
        labels_df = pd.read_csv(Path(base_path) / 'train_labels.csv')
        
        # Prepare training data
        X_features = []
        X_text = []
        y = []
        
        for _, row in labels_df.iterrows():
            article_id = row['article_id']
            dataset_id = row['dataset_id']
            citation_type = row['type']
            
            if article_id in documents:
                text = documents[article_id]
                
                # Extract features
                features = self.create_features(text, dataset_id)
                X_features.append(list(features.values()))
                X_text.append(text)
                y.append(citation_type)
        
        if not X_features:
            print("No training data found!")
            return
        
        print(f"Training on {len(X_features)} samples...")
        
        # Convert to arrays
        X_features = np.array(X_features)
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Train classifier
        self.classifier.fit(X_features, y_encoded)
        
        # Evaluate with cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in skf.split(X_features, y_encoded):
            X_train, X_val = X_features[train_idx], X_features[val_idx]
            y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
            
            self.classifier.fit(X_train, y_train)
            y_pred = self.classifier.predict(X_val)
            score = f1_score(y_val, y_pred, average='weighted')
            cv_scores.append(score)
        
        print(f"Cross-validation F1 Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")

    def predict(self, base_path):
        """Make predictions on test data"""
        print("Loading test data...")
        
        # Load test documents
        documents = self.load_documents(base_path, 'test')
        
        predictions = []
        row_id = 0
        
        for article_id, text in documents.items():
            # Extract potential data citations
            citations = self.extract_data_citations(text)
            
            for citation in citations:
                # Create features
                features = self.create_features(text, citation)
                X_features = np.array([list(features.values())])
                
                # Predict type
                type_encoded = self.classifier.predict(X_features)[0]
                citation_type = self.label_encoder.inverse_transform([type_encoded])[0]
                
                predictions.append({
                    'row_id': row_id,
                    'article_id': article_id,
                    'dataset_id': citation,
                    'type': citation_type
                })
                row_id += 1
        
        return pd.DataFrame(predictions)

    def save_model(self, path):
        """Save the trained model"""
        model_data = {
            'classifier': self.classifier,
            'label_encoder': self.label_encoder,
            'vectorizer': self.vectorizer
        }
        joblib.dump(model_data, path)

    def load_model(self, path):
        """Load a trained model"""
        model_data = joblib.load(path)
        self.classifier = model_data['classifier']
        self.label_encoder = model_data['label_encoder']
        self.vectorizer = model_data['vectorizer']


# Main execution
if __name__ == "__main__":
    # Initialize detector
    detector = DataCitationDetector()
    
    # Set paths
    base_path = "/kaggle/input/make-data-count-finding-data-references"
    
    # Train model
    print("Training model...")
    detector.train_model(base_path)
    
    # Make predictions
    print("Making predictions...")
    predictions_df = detector.predict(base_path)
    
    # Save predictions
    predictions_df.to_csv('submission.csv', index=False)
    print(f"Saved {len(predictions_df)} predictions to submission.csv")
    
    # Save model
    detector.save_model('data_citation_model.joblib')
    print("Model saved successfully!")
    
    # Display sample predictions
    print("\nSample predictions:")
    print(predictions_df.head(10))

