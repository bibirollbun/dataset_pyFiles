import pandas as pd
import numpy as np
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import joblib

# Optional libraries with fallbacks
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
    print("PyMuPDF not available. PDF processing will be limited.")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x  # fallback

# Transformer support (optional)
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
    print("Transformers available for enhanced features.")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Using traditional features only.")

class EnhancedDataCitationDetector:
    def __init__(self, use_transformers=False):
        self.use_transformers = use_transformers and TRANSFORMERS_AVAILABLE
        self.vectorizer = DictVectorizer(sparse=False)
        self.label_encoder = LabelEncoder()
        self.nlp = nlp
        
        # Initialize transformer model if available
        if self.use_transformers:
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            self.transformer_model = AutoModel.from_pretrained('distilbert-base-uncased')
            self.transformer_model.eval()
        
        # Enhanced ensemble classifier
        self.classifier = VotingClassifier(estimators=[
            ('rf', RandomForestClassifier(
                n_estimators=300, 
                max_depth=12, 
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced', 
                random_state=42,
                n_jobs=-1
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=200, 
                max_depth=8, 
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )),
            ('lr', LogisticRegression(
                max_iter=2000, 
                class_weight='balanced', 
                random_state=42,
                C=0.1
            ))
        ], voting='soft')

        # Enhanced pattern matching
        self.primary_patterns = [
            r'\b(?:we\s+)?(?:generated|collected|produced|created|measured|recorded|obtained|acquired|gathered)\b',
            r'\bthis\s+(?:study|work|research|paper|investigation)\b',
            r'\bour\s+(?:data|dataset|measurements|results|experiments|analysis)\b',
            r'\b(?:new|novel|original|fresh)\s+(?:data|dataset|measurements)\b',
            r'\b(?:newly|recently|specifically)\s+(?:generated|collected|obtained|created)\b',
            r'\b(?:specifically|exclusively|solely)\s+(?:for\s+)?this\s+(?:study|work|research|purpose)\b',
            r'\b(?:custom|purpose-built|tailor-made|specifically\s+designed)\b',
            r'\bin-house\s+(?:generated|developed|created|produced)\b',
            r'\b(?:experimental|empirical)\s+(?:data|measurements|observations)\b',
            r'\b(?:direct|first-hand)\s+(?:measurements|observations|data)\b'
        ]

        self.secondary_patterns = [
            r'\b(?:obtained|downloaded|retrieved|sourced|derived|extracted|accessed)\s+from\b',
            r'\b(?:reused?|repurposed|recycled|borrowed)\b',
            r'\b(?:existing|available|published|public)\s+(?:data|dataset|database)\b',
            r'\b(?:previously|earlier|formerly)\s+(?:published|reported|described|collected)\b',
            r'\b(?:published|public|open|shared)\s+(?:data|dataset|database)\b',
            r'\b(?:derived|adapted|modified|transformed)\s+from\b',
            r'\b(?:based|built|founded)\s+(?:on|upon)\b',
            r'\b(?:third-party|external|outside|independent)\s+(?:data|source|database)\b',
            r'\b(?:reference|benchmark|standard|control)\s+(?:data|dataset|database)\b',
            r'\b(?:publicly|freely|openly)\s+(?:available|accessible|distributed)\b',
            r'\b(?:repository|archive|collection|library)\s+(?:data|dataset)\b',
            r'\b(?:curated|maintained|hosted)\s+(?:by|at)\b'
        ]

        # Enhanced section patterns
        self.section_patterns = {
            'methods': r'\b(?:methods?|methodology|experimental\s+(?:design|procedure|setup)|materials?\s+and\s+methods?|approach)\b',
            'results': r'\b(?:results?|findings|outcomes?|observations?)\b',
            'introduction': r'\b(?:introduction|background|overview)\b',
            'discussion': r'\b(?:discussion|conclusion|implications?)\b',
            'references': r'\b(?:references?|bibliography|citations?|literature\s+cited)\b',
            'data_availability': r'\b(?:data\s+availability|code\s+availability|software\s+availability|resource\s+availability)\b',
            'supplementary': r'\b(?:supplementary|supporting|additional)\s+(?:information|material|data)\b'
        }

        # Enhanced citation patterns with better regex
        self.citation_patterns = {
            'doi': r'(?:https?://)?(?:dx\.)?doi\.org/(10\.\d{4,9}/[^\s\)\]\},;\"\'<>\n]+)',
            'zenodo': r'(?:https?://)?zenodo\.org/record/(\d+)',
            'figshare': r'(?:https?://)?figshare\.com/(?:articles|s)/[^\s\)\]\},;\"\'<>\n]+',
            'github': r'(?:https?://)?github\.com/[\w\-_]+/[\w\-_]+(?:/[^\s\)\]\},;\"\'<>\n]*)?',
            'dryad': r'(?:https?://)?(?:datadryad\.org|dryad\.org)/[^\s\)\]\},;\"\'<>\n]+',
            'gse': r'GSE\d+',
            'sra': r'SRA\d+',
            'prjna': r'PRJNA\d+',
            'chembl': r'CHEMBL\d+',
            'pdb': r'PDB:\w+',
            'uniprot': r'UniProt:\w+',
            'arrayexpress': r'E-\w+-\d+',
            'ena': r'(?:ERR|SRR|DRR)\d+',
            'biosample': r'SAMN\d+',
            'bioproject': r'PRJNA\d+'
        }

    def extract_text_from_xml(self, xml_path):
        """Enhanced XML text extraction with better structure preservation"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Try to preserve document structure
            sections = {}
            text_parts = []
            
            def extract_with_structure(elem, section_name=None):
                if elem.text:
                    text = elem.text.strip()
                    if text:
                        text_parts.append(text)
                        if section_name:
                            sections.setdefault(section_name, []).append(text)
                
                # Check if this element represents a section
                tag_lower = elem.tag.lower()
                if any(sec in tag_lower for sec in ['method', 'result', 'intro', 'discuss', 'ref']):
                    section_name = tag_lower
                
                for child in elem:
                    extract_with_structure(child, section_name)
                
                if elem.tail:
                    tail = elem.tail.strip()
                    if tail:
                        text_parts.append(tail)
            
            extract_with_structure(root)
            return ' '.join(text_parts)
            
        except Exception as e:
            print(f"Error processing XML {xml_path}: {e}")
            return ""

    def extract_text_from_pdf(self, pdf_path):
        """Enhanced PDF text extraction"""
        try:
            if fitz:
                doc = fitz.open(pdf_path)
                text_parts = []
                for page_num, page in enumerate(doc):
                    page_text = page.get_text()
                    if page_text.strip():
                        text_parts.append(page_text)
                doc.close()
                return '\n'.join(text_parts)
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
        return ""

    def load_documents(self, base_path, split='train'):
        """Load documents with progress tracking"""
        documents = {}
        
        xml_path = Path(base_path) / split / 'XML'
        pdf_path = Path(base_path) / split / 'PDF'
        
        # Load XML files
        if xml_path.exists():
            xml_files = list(xml_path.glob('*.xml'))
            for xml_file in tqdm(xml_files, desc=f"Loading {split} XML files"):
                text = self.extract_text_from_xml(xml_file)
                if text.strip():
                    documents[xml_file.stem] = text
        
        # Load PDF files (only if not already loaded from XML)
        if pdf_path.exists():
            pdf_files = list(pdf_path.glob('*.pdf'))
            for pdf_file in tqdm(pdf_files, desc=f"Loading {split} PDF files"):
                if pdf_file.stem not in documents:
                    text = self.extract_text_from_pdf(pdf_file)
                    if text.strip():
                        documents[pdf_file.stem] = text
        
        print(f"Loaded {len(documents)} documents from {split} set")
        return documents

    def extract_data_citations(self, text):
        """Enhanced citation extraction with better cleaning"""
        citations = set()
        
        for name, pattern in self.citation_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if name == 'doi':
                    # Clean DOI and ensure proper format
                    doi = match.strip().rstrip('.,;)\]}')
                    citations.add(f"https://doi.org/{doi}")
                elif name == 'zenodo':
                    citations.add(f"https://zenodo.org/record/{match}")
                elif name == 'figshare':
                    if not match.startswith('http'):
                        citations.add(f"https://figshare.com/{match}")
                    else:
                        citations.add(match)
                elif name == 'github':
                    if not match.startswith('http'):
                        citations.add(f"https://github.com/{match}")
                    else:
                        citations.add(match)
                else:
                    # Clean generic matches
                    clean_match = match.strip().rstrip('.,;)\]}')
                    if clean_match:
                        citations.add(clean_match)
        
        # Additional heuristic patterns for data mentions
        data_mention_patterns = [
            r'(?:data|dataset)\s+(?:is\s+)?(?:available|accessible|deposited)\s+(?:at|from|in)\s+([^\s\)\]\},;\"\'<>\n]+)',
            r'(?:available|accessible|deposited)\s+(?:at|from|in)\s+([^\s\)\]\},;\"\'<>\n]*(?:zenodo|figshare|github|dryad)[^\s\)\]\},;\"\'<>\n]*)'
        ]
        
        for pattern in data_mention_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match and len(match) > 5:  # Basic validation
                    citations.add(match.strip().rstrip('.,;)\]}'))
        
        return list(citations)

    def get_transformer_embeddings(self, text, max_length=512):
        """Get transformer embeddings for text"""
        if not self.use_transformers:
            return np.zeros(768)  # Default BERT embedding size
        
        try:
            # Tokenize and encode
            inputs = self.tokenizer(
                text[:max_length], 
                return_tensors='pt', 
                truncation=True, 
                padding=True, 
                max_length=max_length
            )
            
            with torch.no_grad():
                outputs = self.transformer_model(**inputs)
                # Use mean pooling of last hidden states
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            
            return embeddings
        except Exception as e:
            print(f"Error getting transformer embeddings: {e}")
            return np.zeros(768)

    def create_features(self, text, citation):
        """Create comprehensive features with optional transformer embeddings"""
        features = {}
        text_lower = text.lower()
        citation_lower = citation.lower()
        
        # Find extended context around citation
        citation_escaped = re.escape(citation_lower)
        context_matches = list(re.finditer(citation_escaped, text_lower))
        
        if context_matches:
            # Use the first occurrence for context
            match = context_matches[0]
            start = max(0, match.start() - 1000)
            end = min(len(text_lower), match.end() + 1000)
            context = text_lower[start:end]
        else:
            context = text_lower[:2000]  # Fallback to beginning of text
        
        # Basic features
        features['text_length'] = min(len(text), 100000)  # Cap for normalization
        features['citation_count'] = text_lower.count(citation_lower)
        features['context_length'] = len(context)
        
        # Pattern-based features
        primary_total = 0
        secondary_total = 0
        
        for i, pattern in enumerate(self.primary_patterns):
            count = len(re.findall(pattern, context, re.IGNORECASE))
            features[f'primary_{i}'] = count
            primary_total += count
        
        for i, pattern in enumerate(self.secondary_patterns):
            count = len(re.findall(pattern, context, re.IGNORECASE))
            features[f'secondary_{i}'] = count
            secondary_total += count
        
        features['primary_total'] = primary_total
        features['secondary_total'] = secondary_total
        features['primary_secondary_ratio'] = primary_total / (secondary_total + 1)
        features['pattern_confidence'] = abs(primary_total - secondary_total) / (primary_total + secondary_total + 1)
        
        # Section-based features
        for sec_name, sec_pattern in self.section_patterns.items():
            # Check if citation appears near section headers
            section_matches = list(re.finditer(sec_pattern, text_lower, re.IGNORECASE))
            in_section = 0
            for sec_match in section_matches:
                sec_start = max(0, sec_match.start() - 500)
                sec_end = min(len(text_lower), sec_match.end() + 2000)
                section_text = text_lower[sec_start:sec_end]
                if citation_lower in section_text:
                    in_section = 1
                    break
            features[f'in_{sec_name}'] = in_section
        
        # Citation type features
        features['is_doi'] = int('doi.org' in citation_lower)
        features['is_zenodo'] = int('zenodo' in citation_lower)
        features['is_github'] = int('github.com' in citation_lower)
        features['is_figshare'] = int('figshare' in citation_lower)
        features['is_database_id'] = int(any(x in citation_lower for x in ['chembl', 'gse', 'sra', 'prjna', 'err', 'srr']))
        
        # Proximity features
        features['near_figure'] = len(re.findall(r'fig(?:ure)?\s*\d+', context, re.IGNORECASE))
        features['near_table'] = len(re.findall(r'table\s*\d+', context, re.IGNORECASE))
        features['near_supplement'] = len(re.findall(r'supplement', context, re.IGNORECASE))
        features['near_method'] = len(re.findall(r'method', context, re.IGNORECASE))
        features['near_data'] = len(re.findall(r'\bdata\b', context, re.IGNORECASE))
        
        # Linguistic features
        if self.nlp:
            try:
                doc = self.nlp(context[:1500])  # Limit for performance
                features['num_entities'] = len(doc.ents)
                features['num_sentences'] = len(list(doc.sents))
                features['num_tokens'] = len(doc)
                
                # Count specific entity types
                entity_counts = {}
                for ent in doc.ents:
                    entity_counts[ent.label_] = entity_counts.get(ent.label_, 0) + 1
                
                features['num_org_entities'] = entity_counts.get('ORG', 0)
                features['num_person_entities'] = entity_counts.get('PERSON', 0)
                features['num_gpe_entities'] = entity_counts.get('GPE', 0)
                
            except Exception as e:
                print(f"Error in NLP processing: {e}")
                features['num_entities'] = 0
                features['num_sentences'] = 0
                features['num_tokens'] = 0
        else:
            features['num_entities'] = 0
            features['num_sentences'] = context.count('.') + context.count('!') + context.count('?')
            features['num_tokens'] = len(context.split())
        
        # Add transformer features if available
        if self.use_transformers:
            try:
                embeddings = self.get_transformer_embeddings(context)
                for i, emb_val in enumerate(embeddings[:50]):  # Use first 50 dimensions
                    features[f'transformer_{i}'] = float(emb_val)
            except Exception as e:
                print(f"Error adding transformer features: {e}")
        
        return features

    def train_model(self, base_path):
        """Enhanced training with better validation"""
        print("Loading training data...")
        documents = self.load_documents(base_path, 'train')
        labels_df = pd.read_csv(Path(base_path) / 'train_labels.csv')
        
        print("Extracting features...")
        X_dicts, y = [], []
        
        for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Processing training samples"):
            article_id = row['article_id']
            dataset_id = row['dataset_id']
            citation_type = row['type']
            
            if article_id in documents:
                try:
                    features = self.create_features(documents[article_id], dataset_id)
                    X_dicts.append(features)
                    y.append(citation_type)
                except Exception as e:
                    print(f"Error processing {article_id}: {e}")
                    continue
        
        if not X_dicts:
            print("No training data found.")
            return
        
        print(f"Training on {len(X_dicts)} samples...")
        
        # Transform features
        X = self.vectorizer.fit_transform(X_dicts)
        y_enc = self.label_encoder.fit_transform(y)
        
        # Cross-validation with stratification
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_enc)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_enc[train_idx], y_enc[val_idx]
            
            # Train classifier
            self.classifier.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = self.classifier.predict(X_val)
            f1 = f1_score(y_val, y_pred, average='weighted')
            cv_scores.append(f1)
            
            print(f"Fold {fold + 1}: F1 = {f1:.4f}")
        
        print(f"Cross-validation F1: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        # Train final model on all data
        print("Training final model...")
        self.classifier.fit(X, y_enc)
        
        # Feature importance analysis
        if hasattr(self.classifier.estimators_[0][1], 'feature_importances_'):
            feature_names = self.vectorizer.get_feature_names_out()
            rf_importance = self.classifier.estimators_[0][1].feature_importances_
            
            # Get top 20 most important features
            top_indices = np.argsort(rf_importance)[-20:]
            print("\nTop 20 most important features:")
            for i in reversed(top_indices):
                print(f"{feature_names[i]}: {rf_importance[i]:.4f}")

    def predict(self, base_path):
        """Enhanced prediction with better error handling"""
        print("Loading test data...")
        documents = self.load_documents(base_path, 'test')
        
        print("Generating predictions...")
        predictions = []
        row_id = 0
        
        for article_id, text in tqdm(documents.items(), desc="Processing test documents"):
            try:
                citations = self.extract_data_citations(text)
                
                for citation in citations:
                    try:
                        features = self.create_features(text, citation)
                        X = self.vectorizer.transform([features])
                        
                        # Get prediction probabilities for confidence
                        pred_proba = self.classifier.predict_proba(X)[0]
                        pred = self.classifier.predict(X)[0]
                        confidence = np.max(pred_proba)
                        
                        citation_type = self.label_encoder.inverse_transform([pred])[0]
                        
                        predictions.append({
                            'row_id': row_id,
                            'article_id': article_id,
                            'dataset_id': citation,
                            'type': citation_type,
                            'confidence': confidence
                        })
                        row_id += 1
                        
                    except Exception as e:
                        print(f"Error processing citation {citation} in {article_id}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error processing article {article_id}: {e}")
                continue
        
        df = pd.DataFrame(predictions)
        
        # Remove low-confidence predictions (optional filtering)
        if len(df) > 0:
            print(f"Generated {len(df)} predictions")
            print(f"Confidence distribution: {df['confidence'].describe()}")
            
            # Optional: filter out very low confidence predictions
            # df = df[df['confidence'] > 0.6]  # Uncomment to enable filtering
            # print(f"After confidence filtering: {len(df)} predictions")
        
        return df[['row_id', 'article_id', 'dataset_id', 'type']]  # Return required columns only

    def save_model(self, path):
        """Save the complete model pipeline"""
        model_data = {
            'classifier': self.classifier,
            'label_encoder': self.label_encoder,
            'vectorizer': self.vectorizer,
            'use_transformers': self.use_transformers
        }
        
        if self.use_transformers:
            # Save transformer model path instead of the model itself
            model_data['transformer_model_name'] = 'distilbert-base-uncased'
        
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load the complete model pipeline"""
        model_data = joblib.load(path)
        self.classifier = model_data['classifier']
        self.label_encoder = model_data['label_encoder']
        self.vectorizer = model_data['vectorizer']
        self.use_transformers = model_data.get('use_transformers', False)
        
        if self.use_transformers and TRANSFORMERS_AVAILABLE:
            # Reload transformer model
            model_name = model_data.get('transformer_model_name', 'distilbert-base-uncased')
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.transformer_model = AutoModel.from_pretrained(model_name)
            self.transformer_model.eval()
        
        print(f"Model loaded from {path}")


def main():
    """Main execution function"""
    base_path = "/kaggle/input/make-data-count-finding-data-references"
    
    # Initialize detector (set use_transformers=True if you want to use transformers)
    detector = EnhancedDataCitationDetector(use_transformers=False)
    
    # Train model
    detector.train_model(base_path)
    
    # Generate predictions
    predictions_df = detector.predict(base_path)
    
    # Save predictions
    predictions_df.to_csv("submission.csv", index=False)
    print(f"Saved {len(predictions_df)} predictions to submission.csv")
    
    # Save model for future use
    detector.save_model("enhanced_data_citation_model.joblib")
    
    # Display some statistics
    if len(predictions_df) > 0:
        print("\nPrediction statistics:")
        print(predictions_df['type'].value_counts())
        print(f"\nSample predictions:")
        print(predictions_df.head(10))
    else:
        print("No predictions generated!")

if __name__ == "__main__":
    main()

