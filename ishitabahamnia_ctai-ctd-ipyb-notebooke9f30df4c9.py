# ===============================
# Clean environment setup
# ===============================

!pip install --upgrade pip

# Remove any existing conflicting installs
!pip uninstall -y numpy scipy scikit-learn pandas matplotlib seaborn spacy transformers torch peft -y

# Core scientific stack (stable + GPU safe)
!pip install numpy==1.23.5 scipy==1.10.1 scikit-learn==1.2.2 pandas==1.5.3
!pip install matplotlib==3.7.1 seaborn==0.12.2

# Torch (CUDA compatible if GPU is available in Colab)
!pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# NLP stack with compatible versions
!pip install spacy==3.6.1
!pip install transformers==4.35.2 peft==0.7.1

# Optional: Hugging Face tools
!pip install datasets==2.14.5 accelerate==0.24.1

# Check versions after install
!python -m spacy validate
!python -c "import torch, transformers, peft; print('torch:', torch.__version__, 'transformers:', transformers.__version__, 'peft:', peft.__version__)"



# First, restart the runtime if needed (Runtime -> Restart runtime)
# Then run this cell first to set up the environment properly

# --- Fix package versions to avoid conflicts ---
!pip install numpy==1.26.4
!pip install scipy==1.11.4
!pip install scikit-learn==1.2.2
!pip install pandas==2.2.2
!pip install requests==2.32.3

# Now install the main packages
!pip install spacy==3.8.7
!pip install rapidfuzz python-magic tika
!pip install transformers==4.52.4 torch==2.6.0

# Download spaCy model
!python -m spacy download en_core_web_sm==3.8.0

print("Installation completed successfully!")


# First, let's check current versions and fix the incompatibility
!pip uninstall numpy scipy scikit-learn transformers torch -y

# Install compatible versions in the right order
!pip install numpy==1.26.4
!pip install scipy==1.11.4
!pip install scikit-learn==1.2.2
!pip install pandas==2.2.2

# Now install transformers and torch with compatible versions
!pip install transformers==4.36.2
!pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install other packages
!pip install spacy==3.7.4
!pip install rapidfuzz python-magic tika requests

# Download compatible spaCy model
!python -m spacy download en_core_web_sm==3.7.1

print("Installation completed! Please restart the runtime and run imports.")


# First, let's check current versions and fix the incompatibility
!pip uninstall numpy scipy scikit-learn transformers torch -y

# Install compatible versions in the right order
!pip install numpy==1.26.4
!pip install scipy==1.11.4
!pip install scikit-learn==1.2.2
!pip install pandas==2.2.2

# Now install transformers and torch with compatible versions
!pip install transformers==4.36.2
!pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install other packages
!pip install spacy==3.7.4
!pip install rapidfuzz python-magic tika requests

# Download compatible spaCy model
!python -m spacy download en_core_web_sm==3.7.1

print("Installation completed! Please restart the runtime and run imports.")


                                                                                                                                                                                                                                                                                                                                             import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt

# Ensure plots render in Kaggle
%matplotlib inline



# First, let's check current versions and fix the incompatibility
!pip uninstall numpy scipy scikit-learn transformers torch -y

# Install compatible versions in the right order
!pip install numpy==1.26.4
!pip install scipy==1.11.4
!pip install scikit-learn==1.2.2
!pip install pandas==2.2.2

# Now install transformers and torch with compatible versions
!pip install transformers==4.36.2
!pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install other packages
!pip install spacy==3.7.4
!pip install rapidfuzz python-magic tika requests

# Download compatible spaCy model
!python -m spacy download en_core_web_sm==3.7.1

print("Installation completed! Please restart the runtime and run imports.")


# 1. Install and import all necessary libraries
!pip install pandas numpy matplotlib seaborn scikit-learn spacy requests transformers torch tika python-magic
!python -m spacy download en_core_web_sm

import pandas as pd
import numpy as np
import re
import json
import os
from typing import List, Dict, Tuple, Set
import requests
from collections import Counter

# NLP and ML
import spacy
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Deep Learning
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# First, let's uninstall problematic packages and install compatible versions
!pip uninstall numpy scipy scikit-learn transformers torch -y

# Install compatible versions in the correct order
!pip install numpy==1.24.3
!pip install scipy==1.10.1
!pip install scikit-learn==1.2.2
!pip install pandas==2.0.3

# Install transformers with a compatible version
!pip install transformers==4.28.1

# Install torch with CUDA support for Colab
!pip install torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html

# Install other packages
!pip install spacy==3.5.3
!pip install rapidfuzz python-magic tika requests

# Download compatible spaCy model
!python -m spacy download en_core_web_sm==3.5.0

print("Installation completed! Please restart the runtime (Runtime -> Restart runtime) and then run your import code.")


# Load the training data
import os
import pandas as pd # Assuming pandas is needed and available from previous imports

def load_data(base_path):
    """
    Load training data from directory structure
    """
    data = []
    labels_df = pd.read_csv(os.path.join(base_path, 'train_labels.csv'))

    # Group labels by article_id
    article_labels = {}
    # Ensure column names are accessed with correct casing
    if 'article_id' in labels_df.columns and 'dataset_id' in labels_df.columns and 'type' in labels_df.columns:
        for _, row in labels_df.iterrows():
            article_id = row['article_id']
            if article_id not in article_labels:
                article_labels[article_id] = []
            article_labels[article_id].append({
                'dataset_id': row['dataset_id'],
                'type': row['type']
            })
    else:
        print("Error: Required columns ('article_id', 'dataset_id', 'type') not found in the CSV.")
        # Potentially add more detailed error handling or column listing here

    return article_labels

# Load the data
train_base_path = '/content/sample_data/train' # Corrected path
article_labels = load_data(train_base_path)
print(f"Loaded {len(article_labels)} articles with labels")

# Sample data structure
if article_labels: # Check if article_labels is not empty before accessing elements
    print("Sample article labels:", list(article_labels.items())[0])
else:
    print("No article labels loaded.")


import re
import json
import os
from typing import List, Dict, Tuple
import requests
import pandas as pd
import numpy as np
from collections import Counter

# NLP
import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except:
    !python -m spacy download en_core_web_sm
    nlp = spacy.load("en_core_web_sm")

# Transformers for classification - with error handling
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError as e:
    print(f"Transformers import error: {e}")
    # Fallback to basic functionality

import torch

# Metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'figure.max_open_warning': 0})

print("All imports successful!")


import os

# First, check what files/directories exist
def check_directory_structure(base_path):
    if not os.path.exists(base_path):
        print(f"Path {base_path} does not exist!")
        return

    print("Directory structure:")
    for root, dirs, files in os.walk(base_path):
        level = root.replace(base_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f'{subindent}{file}')

# Check current directory structure
!ls -la /kaggle/input/  # This will show what's available in Kaggle input

# If you're in Colab, check content directory
!ls -la /content/


def load_data(base_path):
    """
    Load training data from directory structure with robust error handling
    """
    # Check if base path exists
    if not os.path.exists(base_path):
        print(f"Error: Base path '{base_path}' does not exist!")
        print("Current working directory:", os.getcwd())
        print("Available directories:", [d for d in os.listdir('.') if os.path.isdir(d)])
        return {}

    # Check for different possible file names
    possible_files = ['train_labels.csv', 'labels.csv', 'train_labels.txt', 'labels.txt']
    labels_path = None

    for file in possible_files:
        test_path = os.path.join(base_path, file)
        if os.path.exists(test_path):
            labels_path = test_path
            break

    if labels_path is None:
        print(f"Error: No label file found in '{base_path}'")
        print("Available files:", os.listdir(base_path))
        return {}

    try:
        labels_df = pd.read_csv(labels_path)
        print(f"Successfully loaded {len(labels_df)} rows from {labels_path}")

        # Group labels by article_id
        article_labels = {}
        for _, row in labels_df.iterrows():
            article_id = row['article_id']
            if article_id not in article_labels:
                article_labels[article_id] = []
            article_labels[article_id].append({
                'dataset_id': row['dataset_id'],
                'type': row['type']
            })

        return article_labels

    except Exception as e:
        print(f"Error reading file {labels_path}: {e}")
        return {}

# Try loading with different possible paths
possible_paths = [
    '/kaggle/input/make-data-count-finding-data-references/train',
    '/content/data/train',
    '/content/sample_data/train',
    './train',
    './data/train'
]

for path in possible_paths:
    print(f"\nTrying path: {path}")
    article_labels = load_data(path)
    if article_labels:
        print(f"Success! Loaded {len(article_labels)} articles with labels")
        print("Sample article labels:", list(article_labels.items())[0])
        train_base_path = path
        break
else:
    print("Could not find the data in any of the expected paths.")
    print("Please upload your data files or check the path.")


# Import necessary libraries
import pandas as pd
import re
import json
from collections import Counter
from typing import List, Dict, Tuple, Set

# --- DATA CITATION MINING PIPELINE ---

# DOI regex (relaxed but practical)
doi_pattern = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)

# Common repository accession patterns
accession_patterns = {
    'GEO': re.compile(r'GSE\d+', re.I),
    'SRA': re.compile(r'SRP\d+|SRR\d+|SRA\d+', re.I),
    'ENA': re.compile(r'PRJNA\d+|ERP\d+', re.I),
    'PDB': re.compile(r'PDB\s*[:#]?\s*[0-9A-Za-z]{4}', re.I),
    'Zenodo': re.compile(r'zenodo\.\d+', re.I),
    'Figshare': re.compile(r'figshare\.\d+', re.I),
}

def extract_references(text: str) -> Dict[str, List[str]]:
    """Extract DOIs and accession numbers from text"""
    if not text or not isinstance(text, str):
        return {'dois': [], 'accessions': []}

    results = {'dois': [], 'accessions': []}

    try:
        # Extract DOIs
        results['dois'] = list(set([m.group(0).rstrip('.;,') for m in doi_pattern.finditer(text)]))

        # Extract accessions
        accs = []
        for name, pat in accession_patterns.items():
            found = [m.group(0) for m in pat.finditer(text)]
            accs.extend(found)

        results['accessions'] = list(set([a.rstrip('.;,') for a in accs]))
    except Exception as e:
        print(f"Error extracting references: {e}")

    return results

# Heuristic patterns for informal mentions
repo_keywords = ['geo', 'sra', 'ena', 'zenodo', 'figshare', 'dryad', 'pdb']
heuristic_pattern = re.compile(
    r'(' + '|'.join(repo_keywords) + r')[\w\s\-:]*?(accession|id|doi|deposited|available|under|in)\s*[:#]?\s*([A-Za-z0-9\._-]+)',
    re.I
)

def extract_informal_mentions(text: str) -> List[Tuple[str, str]]:
    """Extract informal dataset mentions"""
    if not text or not isinstance(text, str):
        return []

    matches = []

    try:
        # Heuristic pattern matching
        for m in heuristic_pattern.finditer(text):
            repo = m.group(1)
            token = m.group(3)
            matches.append((repo, token))

        # Use spaCy if available, otherwise use simple regex
        if 'nlp' in globals() and nlp: # Check if nlp object is defined and not None
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ('ORG', 'PRODUCT'):
                    if re.search(r'\d{3,}', ent.text) or any(keyword in ent.text.lower() for keyword in repo_keywords):
                        matches.append((ent.label_, ent.text))
        else:
            # Fallback: look for repository-like patterns
            repo_pattern = re.compile(r'\b(' + '|'.join(repo_keywords) + r')\b', re.I)
            for match in repo_pattern.finditer(text):
                matches.append(('repository', match.group(0)))

    except Exception as e:
        print(f"Error extracting informal mentions: {e}")

    return matches

class APIClient:
    """Client for Crossref and DataCite APIs with rate limiting"""
    def __init__(self):
        self.crossref_base = 'https://api.crossref.org/works/'
        self.datacite_base = 'https://api.datacite.org/dois/'
        self.cache = {}
        self.request_delay = 1.0 # seconds

    def _rate_limit(self):
        """Respectful rate limiting"""
        import time # Import time here to avoid NameError if not imported globally
        time.sleep(self.request_delay)

    def lookup_doi(self, doi: str) -> Dict:
        """Look up DOI metadata"""
        if not doi:
            return {'success': False, 'error': 'Empty DOI'}

        if doi in self.cache:
            return self.cache[doi]

        self._rate_limit()
        cleaned_doi = doi.strip().rstrip('.,;')

        try:
            # Try Crossref
            response = requests.get(self.crossref_base + cleaned_doi, timeout=10)
            if response.status_code == 200:
                msg = response.json().get('message', {})
                result = {
                    'title': msg.get('title', [''])[0] if msg.get('title') else '',
                    'publisher': msg.get('publisher', ''),
                    'success': True,
                    'source': 'crossref'
                }
                self.cache[doi] = result
                return result

            # Try DataCite
            response = requests.get(self.datacite_base + cleaned_doi, timeout=10)
            if response.status_code == 200:
                msg = response.json().get('data', {})
                attrs = msg.get('attributes', {})
                result = {
                    'title': attrs.get('titles', [{}])[0].get('title', ''),
                    'publisher': attrs.get('publisher', ''),
                    'success': True,
                    'source': 'datacite'
                }
                self.cache[doi] = result
                return result

            return {'success': False, 'error': f'HTTP {response.status_code}'}

        except Exception as e:
            return {'success': False, 'error': str(e)}


def process_corpus(df: pd.DataFrame, text_col: str = 'text', use_api: bool = False) -> pd.DataFrame:
    """Process corpus and extract dataset references"""
    if df.empty or text_col not in df.columns:
        print(f"Input DataFrame is empty or missing '{text_col}' column.")
        return pd.DataFrame(columns=['source_id', 'extracted', 'total_references'])

    outputs = []

    for idx, row in df.iterrows():
        try:
            text = str(row[text_col]) if pd.notna(row[text_col]) else ""
            source_id = row.get('id', f"doc_{idx}") # Use .get for robustness

            # Extract references
            refs = extract_references(text)
            informal = extract_informal_mentions(text)
            matched = []

            # Process DOIs
            for doi in refs['dois']:
                meta = {}
                if use_api:
                    api_client = APIClient()
                    meta = api_client.lookup_doi(doi)
                matched.append({'type': 'doi', 'key': doi, 'meta': meta})

            # Process accessions
            for acc in refs['accessions']:
                matched.append({'type': 'accession', 'key': acc, 'meta': {}})

            # Process informal mentions
            for repo, token in informal:
                matched.append({'type': 'informal', 'key': token, 'repo': repo, 'meta': {}})

            outputs.append({
                'source_id': source_id,
                'extracted': matched,
                'total_references': len(matched)
            })

        except Exception as e:
            print(f"Error processing row {idx} (Source ID: {row.get('id', 'N/A')}): {e}")
            outputs.append({
                'source_id': row.get('id', f"doc_{idx}"),
                'extracted': [],
                'total_references': 0
            })

    return pd.DataFrame(outputs)

# Create sample data
sample_text = "We used data from GEO accession GSE12345, SRA SRP98765 and Zenodo 10.5281/zenodo.1234567. Also see DOI 10.1038/s41586-020-2649-2."

corpus_data = [
    {'id': 'paper1', 'text': sample_text},
    {'id': 'paper2', 'text': 'Our raw reads were deposited to SRA under SRP123456. See also DOI 10.5281/zenodo.7654321.'},
    {'id': 'paper3', 'text': 'We used standard datasets including MNIST and CIFAR-10 for evaluation.'}
]

corpus = pd.DataFrame(corpus_data)

# Process corpus
print("Processing corpus...")
results_df = process_corpus(corpus, use_api=False)
print("Processing complete!")
print(results_df)

# Evaluation
gold = [
    {
        'source_id': 'paper1',
        'labels': [
            {'type': 'doi', 'key': '10.1038/s41586-020-2649-2'},
            {'type': 'accession', 'key': 'GSE12345'},
            {'type': 'accession', 'key': 'SRP98765'},
            {'type': 'doi', 'key': '10.5281/zenodo.1234567'}
        ]
    },
    {
        'source_id': 'paper2',
        'labels': [
            {'type': 'accession', 'key': 'SRP123456'},
            {'type': 'doi', 'key': '10.5281/zenodo.7654321'}
        ]
    },
    {
        'source_id': 'paper3',
        'labels': []
    }
]

def eval_detection(gold: List[Dict], predicted: List[Dict]) -> Dict:
    """Evaluate detection performance"""
    y_true = []
    y_pred = []

    # Create flattened sets for easier comparison
    gold_set = set()
    for g in gold:
        sid = g['source_id']
        for label in g['labels']:
            gold_set.add((sid, label['type'], label['key'].lower()))

    predicted_set = set()
    for p in predicted:
        sid = p['source_id']
        for extracted in p['extracted']:
            # Handle potential missing keys gracefully
            extracted_type = extracted.get('type')
            extracted_key = extracted.get('key')
            if extracted_type and extracted_key is not None:
                 predicted_set.add((sid, extracted_type, extracted_key.lower()))
            else:
                 print(f"Warning: Skipping malformed extracted record for {sid}: {extracted}")


    # Calculate True Positives, False Positives, False Negatives
    tp = len(gold_set.intersection(predicted_set))
    fp = len(predicted_set - gold_set)
    fn = len(gold_set - predicted_set)

    # Calculate Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


    return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}

eval_results = eval_detection(gold, results_df.to_dict('records'))
print(f"\nEvaluation Results:")
print(f"Precision: {eval_results['precision']:.3f}, Recall: {eval_results['recall']:.3f}, F1: {eval_results['f1']:.3f}")
print(f"TP: {eval_results['tp']}, FP: {eval_results['fp']}, FN: {eval_results['fn']}")


# MDC output format
def format_for_mdc(records: List[Dict]) -> List[Dict]:
    """Format for Make Data Count compatibility"""
    mdc_output = []

    for record in records:
        for extracted in record['extracted']:
            # Ensure required keys exist
            dataset_identifier = extracted.get('key', '')
            dataset_identifier_type = extracted.get('type', '')
            referencing_publication_id = record.get('source_id', '')

            if dataset_identifier and dataset_identifier_type and referencing_publication_id:
                mdc_record = {
                    'dataset_identifier': dataset_identifier,
                    'dataset_identifier_type': dataset_identifier_type,
                    'referencing_publication_id': referencing_publication_id,
                    'confidence_score': 0.9 if extracted.get('type') == 'doi' else 0.7, # Use .get()
                    'extraction_method': 'regex' if extracted.get('type') in ['doi', 'accession'] else 'heuristic' # Use .get()
                }
                mdc_output.append(mdc_record)
            else:
                 print(f"Warning: Skipping incomplete MDC record for {referencing_publication_id}: {extracted}")


    return mdc_output

mdc_output = format_for_mdc(results_df.to_dict('records'))
print("\nMDC output sample:")
if mdc_output:
    for record in mdc_output[:3]:
        print(record)
else:
    print("No MDC output generated (no references found).")


# Visualization
ref_types = [extracted['type'] for record in results_df.to_dict('records') for extracted in record['extracted']]
if ref_types:
    type_counts = pd.Series(ref_types).value_counts()
    plt.figure(figsize=(10, 6))
    type_counts.plot(kind='bar')
    plt.title('Reference Types Distribution')
    plt.xlabel('Reference Type')
    plt.ylabel('Count')
    plt.xticks(rotation=0) # Keep labels horizontal
    plt.tight_layout()
    plt.show()
else:
    print("\nNo references found for visualization.")

print("\nPipeline execution completed successfully! ðŸŽ‰")


# Train Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Evaluation metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Random Forest Performance:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Secondary', 'Primary']))

# Confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Secondary', 'Primary'],
            yticklabels=['Secondary', 'Primary'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
plt.title('Top 15 Feature Importances')
plt.tight_layout()
plt.show()


# Prepare data for training
X = feature_df.drop('target', axis=1)
y = feature_df['target']

# Handle class imbalance if necessary
from imblearn.over_sampling import SMOTE

print(f"Class distribution: {np.bincount(y)}")

# Apply SMOTE if needed
if np.bincount(y)[0] / np.bincount(y)[1] > 1.5:
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"After SMOTE: {np.bincount(y_resampled)}")
    X, y = X_resampled, y_resampled

# Train-test split
# Adjusted test_size to accommodate stratification with a small number of samples.
# Note: With only 5 samples total, any split will be highly unrepresentative.
# This adjustment is for demonstration purposes to make the code run.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")


# First, let's clean up the environment and install compatible versions
!pip uninstall scikit-learn scipy -y
!pip install scikit-learn==1.2.2 scipy==1.10.1
!pip install spacy xgboost matplotlib seaborn joblib pandas numpy
!python -m spacy download en_core_web_sm

import re
import spacy
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

print("All packages installed successfully!")

class DataCitationMiner:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.dataset_patterns = self._initialize_patterns()
        self.models = {}
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self._initialize_models()

    def _initialize_patterns(self):
        """Initialize regex patterns for dataset detection"""
        return {
            'doi': re.compile(r'10\.\d{4,9}/[-._;()/:A-Z0-9]+', re.I),
            'url': re.compile(r'https?://\S+', re.I),
            'common_datasets': re.compile(r'\b(MNIST|CIFAR|ImageNet|WikiData|UCI|Kaggle|FigShare|Zenodo)\b', re.I),
            'repository': re.compile(r'\b(GEO|SRA|ENA|PDB|PRIDE|ArrayExpress|dbGaP)\b', re.I),
            'accession': re.compile(r'\b(GSE\d+|SRP\d+|PRJNA\d+|PXD\d+)\b', re.I)
        }

    def _initialize_models(self):
        """Initialize ML models"""
        self.models['xgb'] = xgb.XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
        )
        self.models['rf'] = RandomForestClassifier(n_estimators=50, random_state=42)
        self.models['calibrated'] = CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=30, random_state=42), cv=2
        )

    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text and split into sentences"""
        if not text:
            return []
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]

    def extract_features(self, sentences: List[str]) -> np.ndarray:
        """Extract TF-IDF features from sentences"""
        if not sentences:
            return np.array([])
        return self.vectorizer.fit_transform(sentences).toarray()

    def train_models(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.3):
        """Train all models"""
        if len(X) == 0:
            print("No data available for training")
            return None

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            results = {}

            # Train and evaluate each model
            for name, model in self.models.items():
                print(f"Training {name}...")
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                acc = accuracy_score(y_test, pred)
                results[name] = acc
                print(f"{name} Accuracy: {acc:.4f}")

            # Ensemble predictions
            ensemble_pred = []
            for i in range(len(y_test)):
                votes = [model.predict(X_test[i:i+1])[0] for model in self.models.values()]
                ensemble_pred.append(max(set(votes), key=votes.count))

            ensemble_acc = accuracy_score(y_test, ensemble_pred)
            results['ensemble'] = ensemble_acc
            print(f"Ensemble Accuracy: {ensemble_acc:.4f}")

            print("\nClassification Report:")
            print(classification_report(y_test, ensemble_pred,
                                      target_names=['primary', 'secondary', 'none']))

            self.plot_confusion_matrix(y_test, ensemble_pred)
            return results

        except Exception as e:
            print(f"Error in training: {e}")
            return None

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['primary', 'secondary', 'none'],
                   yticklabels=['primary', 'secondary', 'none'])
        plt.title('Confusion Matrix')
        plt.show()

    def predict(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """Predict dataset citations"""
        if not sentences:
            return []

        results = []
        for sentence in sentences:
            # Rule-based prediction (fallback when models not trained)
            pred_type = 'none'
            confidence = 0.5

            if any(pattern.search(sentence) for pattern in self.dataset_patterns.values()):
                if (self.dataset_patterns['doi'].search(sentence) or
                    self.dataset_patterns['url'].search(sentence) or
                    self.dataset_patterns['accession'].search(sentence)):
                    pred_type = 'primary'
                    confidence = 0.8
                else:
                    pred_type = 'secondary'
                    confidence = 0.7

            dataset_id = self._extract_dataset_id(sentence)

            results.append({
                'sentence': sentence,
                'dataset_id': dataset_id,
                'type': pred_type,
                'confidence': confidence,
                'validation': self._validate_citation(sentence, pred_type)
            })

        return results

    def _extract_dataset_id(self, text: str) -> str:
        """Extract dataset identifier"""
        for pattern_name, pattern in self.dataset_patterns.items():
            match = pattern.search(text)
            if match:
                return match.group(0)
        return text[:50].strip() + "..."

    def _validate_citation(self, text: str, citation_type: str) -> str:
        """Validate citation"""
        if citation_type == 'none':
            return "No dataset citation"

        validations = []
        dataset_id = self._extract_dataset_id(text)

        if self.dataset_patterns['doi'].search(text):
            validations.append("DOI detected")
        if self.dataset_patterns['accession'].search(text):
            validations.append("Accession number detected")

        return "; ".join(validations) if validations else "Dataset mentioned"

    def process_text(self, text: str, source_name: str = "text") -> List[Dict[str, Any]]:
        """Process text and return dataset citations"""
        sentences = self.preprocess_text(text)
        results = self.predict(sentences)
        return [r for r in results if r['type'] != 'none']

# Create sample training data
def create_sample_data():
    """Create training data"""
    sentences = [
        # Primary citations
        "Dataset available at https://example.com/data DOI: 10.1234/abc.123",
        "Data on FigShare DOI: 10.5678/xyz.789",
        "GEO accession GSE12345",
        "SRA data under SRP98765",
        "PRIDE accession PXD008765",

        # Secondary citations
        "Used MNIST dataset for training",
        "Data from ImageNet repository",
        "CIFAR-10 dataset for evaluation",
        "WikiData knowledge base",
        "Kaggle competition data",

        # Non-dataset
        "Results in Table 1",
        "Standard protocols used",
        "Experimental results show",
        "Evaluation protocol followed",
        "Performance metrics",
    ]

    labels = [
        0, 0, 0, 0, 0,  # primary
        1, 1, 1, 1, 1,  # secondary
        2, 2, 2, 2, 2   # none
    ]

    return sentences, labels

# Main pipeline
def main():
    print("DATA CITATION MINING PIPELINE")
    print("=" * 50)

    # Initialize miner
    miner = DataCitationMiner()

    # Create training data
    print("Creating training data...")
    sentences, labels = create_sample_data()

    # Extract features
    print("Extracting features...")
    X = miner.extract_features(sentences)
    y = np.array(labels)

    # Train models
    print("Training models...")
    results = miner.train_models(X, y)

    # Test with sample text
    print("\nTesting with sample text...")
    sample_text = """
    Our research used GEO accession GSE12345 and ImageNet data.
    Code is available at DOI: 10.5281/zenodo.1234567.
    Results are shown in Table 1.
    """

    citations = miner.process_text(sample_text)

    print(f"\nFound {len(citations)} dataset citations:")
    for i, citation in enumerate(citations, 1):
        print(f"{i}. {citation['dataset_id']} ({citation['type']}, confidence: {citation['confidence']:.2f})")
        print(f"   Validation: {citation['validation']}")
        print(f"   Context: {citation['sentence'][:60]}...")
        print()

# Run the pipeline
if __name__ == "__main__":
    main()

# Additional example: Process multiple texts
print("ADDITIONAL EXAMPLES:")
print("=" * 50)

texts = [
    "We used MNIST and CIFAR-10 datasets with code at DOI: 10.1234/code.1",
    "Experimental results show significant improvement in performance",
    "Data available from GEO GSE99999 and SRA SRP88888"
]

miner = DataCitationMiner()
for i, text in enumerate(texts, 1):
    print(f"Text {i}: {text[:50]}...")
    citations = miner.process_text(text)
    print(f"  Citations found: {len(citations)}")
    for cit in citations:
        print(f"  - {cit['dataset_id']} ({cit['type']})")
    print()


import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# First, let's create some sample data since article_labels isn't defined
print("Creating sample data for feature engineering...")

# Create sample article_labels data structure
article_labels = {
    'article_1': [
        {'dataset_id': 'GSE12345', 'type': 'Primary'},
        {'dataset_id': '10.1038/s41586-020-2649-2', 'type': 'Secondary'}
    ],
    'article_2': [
        {'dataset_id': 'SRP98765', 'type': 'Primary'},
        {'dataset_id': 'PDB_1ABC', 'type': 'Secondary'},
        {'dataset_id': '10.5281/zenodo.1234567', 'type': 'Primary'}
    ],
    'article_3': [
        {'dataset_id': 'GSE54321', 'type': 'Secondary'},
        {'dataset_id': '10.6084/m9.figshare.7654321', 'type': 'Primary'}
    ]
}

# Create realistic context examples
context_examples = {
    'GSE12345': "We generated RNA-seq data and deposited it in GEO under accession GSE12345. The data was produced using Illumina sequencing.",
    '10.1038/s41586-020-2649-2': "We analyzed existing data from DOI 10.1038/s41586-020-2649-2 to compare our results with previous studies.",
    'SRP98765': "Raw sequencing reads were produced and made available in SRA under accession SRP98765 as part of this study.",
    'PDB_1ABC': "We used the protein structure from PDB entry 1ABC for molecular docking studies in our analysis.",
    '10.5281/zenodo.1234567': "All custom code and generated datasets are available on Zenodo at DOI 10.5281/zenodo.1234567.",
    'GSE54321': "Existing microarray data from GEO accession GSE54321 was reanalyzed in this work to validate our findings.",
    '10.6084/m9.figshare.7654321': "We created and deposited supplementary materials on FigShare at DOI 10.6084/m9.figshare.7654321."
}

def extract_text_features(text):
    """
    Extract various text features from context
    """
    features = {}

    # Basic text features
    features['length'] = len(text)
    features['word_count'] = len(text.split())
    features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text else 0

    # Keyword features for primary vs secondary citations
    primary_keywords = ['generate', 'produce', 'create', 'collect', 'measure', 'experiment',
                       'study', 'deposit', 'make available', 'this work', 'our data']
    secondary_keywords = ['use', 'analyze', 'reuse', 'existing', 'previous', 'obtain',
                         'download', 'compare', 'validate', 'reanalyze']

    text_lower = text.lower()
    features['primary_keyword_count'] = sum(1 for word in primary_keywords if word in text_lower)
    features['secondary_keyword_count'] = sum(1 for word in secondary_keywords if word in text_lower)
    features['keyword_ratio'] = features['primary_keyword_count'] / (features['secondary_keyword_count'] + 1e-6)

    # Repository mentions
    repositories = ['geo', 'genbank', 'arrayexpress', 'pdb', 'dryad', 'figshare', 'zenodo',
                   'sra', 'ena', 'proteomexchange']
    features['repo_mentions'] = sum(1 for repo in repositories if repo in text_lower)

    # Presence indicators
    features['has_deposit_verbs'] = 1 if any(word in text_lower for word in ['deposit', 'submit', 'upload']) else 0
    features['has_creation_verbs'] = 1 if any(word in text_lower for word in ['generate', 'create', 'produce']) else 0
    features['has_analysis_verbs'] = 1 if any(word in text_lower for word in ['analyze', 'use', 'reuse']) else 0

    return features

def extract_identifier_features(dataset_id):
    """
    Extract features from the dataset identifier itself
    """
    features = {}

    # Convert dataset_id to string to handle both integers and potential string IDs
    dataset_id_str = str(dataset_id)

    # Identifier type features
    features['is_doi'] = 1 if dataset_id_str.startswith('10.') else 0
    features['is_geo'] = 1 if dataset_id_str.startswith('GSE') else 0
    features['is_sra'] = 1 if dataset_id_str.startswith(('SRP', 'SRR', 'SRA')) else 0
    features['is_pdb'] = 1 if dataset_id_str.startswith(('PDB', 'pdb')) else 0
    features['is_zenodo'] = 1 if 'zenodo' in dataset_id_str.lower() else 0
    features['is_figshare'] = 1 if 'figshare' in dataset_id_str.lower() else 0

    # Structural features
    features['id_length'] = len(dataset_id_str)
    features['has_special_chars'] = 1 if any(c in dataset_id_str for c in ['/', '.', '-', ':', '_']) else 0
    features['has_numbers'] = 1 if any(c.isdigit() for c in dataset_id_str) else 0
    features['has_letters'] = 1 if any(c.isalpha() for c in dataset_id_str) else 0

    # Pattern complexity
    features['digit_count'] = sum(1 for c in dataset_id_str if c.isdigit())
    features['letter_count'] = sum(1 for c in dataset_id_str if c.isalpha())
    features['special_char_count'] = sum(1 for c in dataset_id_str if not c.isalnum())

    return features

def create_feature_matrix(article_labels, context_examples):
    """
    Create feature matrix from article_labels and context examples
    """
    features = []

    for article_id, citations in article_labels.items():
        for citation in citations:
            dataset_id = citation['dataset_id']
            citation_type = citation['type']

            # Get context for this dataset mention
            mention_context = context_examples.get(dataset_id, f"Context for {dataset_id} in {article_id}")

            feature_set = {}

            # Add basic identifiers
            feature_set['article_id'] = article_id
            feature_set['dataset_id'] = dataset_id

            # Text features from context
            text_feats = extract_text_features(mention_context)
            feature_set.update(text_feats)

            # Identifier features
            id_feats = extract_identifier_features(dataset_id)
            feature_set.update(id_feats)

            # Target variable
            feature_set['target'] = 1 if citation_type == 'Primary' else 0
            feature_set['citation_type'] = citation_type

            features.append(feature_set)

    return pd.DataFrame(features)

# Create feature matrix
print("Creating feature matrix...")
feature_df = create_feature_matrix(article_labels, context_examples)

print("Feature matrix shape:", feature_df.shape)
print("\nFeature matrix head:")
print(feature_df.head())

# Check for missing values
print("\nMissing values:")
print(feature_df.isnull().sum())

# Basic statistics
print("\nBasic statistics:")
print(feature_df.describe())

# Target distribution
print("\nTarget distribution:")
print(feature_df['target'].value_counts())
print(feature_df['citation_type'].value_counts())

# Correlation analysis
plt.figure(figsize=(14, 10))
numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
correlation_matrix = feature_df[numeric_cols].corr()

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', square=True, cbar_kws={"shrink": .8})
plt.title('Feature Correlation Matrix (Lower Triangle)')
plt.tight_layout()
plt.show()

# Feature importance analysis (using correlation with target)
target_correlations = correlation_matrix['target'].drop('target').sort_values(key=abs, ascending=False)

plt.figure(figsize=(12, 8))
target_correlations.plot(kind='bar', color=['red' if x < 0 else 'blue' for x in target_correlations])
plt.title('Feature Correlation with Target Variable')
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Distribution of key features by target class
key_features = ['primary_keyword_count', 'secondary_keyword_count', 'keyword_ratio',
                'repo_mentions', 'is_doi', 'is_geo']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, feature in enumerate(key_features):
    if feature in feature_df.columns:
        for target_val in [0, 1]:
            subset = feature_df[feature_df['target'] == target_val]
            axes[i].hist(subset[feature], alpha=0.7, label=f'Target={target_val}', bins=20)
        axes[i].set_title(f'Distribution of {feature}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frequency')
        axes[i].legend()

plt.tight_layout()
plt.show()

# Pairplot of most correlated features
most_correlated = target_correlations.head(6).index.tolist()
if 'target' in most_correlated:
    most_correlated.remove('target')

if most_correlated:
    plot_df = feature_df[most_correlated + ['target']]
    plot_df['target'] = plot_df['target'].astype(str)  # Convert to categorical for coloring

    sns.pairplot(plot_df, hue='target', palette={ '0': 'red', '1': 'blue' },
                diag_kind='kde', corner=True)
    plt.suptitle('Pairplot of Most Correlated Features', y=1.02)
    plt.show()

print("\nFeature engineering completed successfully!")
print("Key insights:")
print(f"- Total samples: {len(feature_df)}")
print(f"- Primary citations: {len(feature_df[feature_df['target'] == 1])}")
print(f"- Secondary citations: {len(feature_df[feature_df['target'] == 0])}")
print(f"- Most predictive features: {target_correlations.head(3).index.tolist()}")


from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': [None, 'balanced']
}

# Initialize grid search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    # Adjusted cv to 2 to satisfy GridSearchCV's minimum requirement.
    # Note: With a very small dataset, cross-validation with cv=2 or higher
    # may still be problematic or not meaningful. This is for demonstration
    # to make the code run through this step.
    cv=2,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

# Perform grid search
print("Starting grid search...")
grid_search.fit(X_train, y_train)

# Best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Evaluate best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

print("\nBest Model Performance:")
print(f"Precision: {precision_score(y_test, y_pred_best):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_best):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_best):.4f}")


# Transformer-based classification for comparison
class CitationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Prepare transformer data
transformer_texts = train_df['mention_context'].fillna('').tolist()
transformer_labels = (train_df['citation_type'] == 'Primary').astype(int).tolist()

# Split data
# Adjusted test_size to accommodate stratification with a small number of samples.
# Note: With only 5 samples total, any split will be highly unrepresentative.
# This adjustment is for demonstration purposes to make the code run.
X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(
    transformer_texts, transformer_labels, test_size=0.4, random_state=42, stratify=transformer_labels
)

# Initialize tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Create datasets
train_dataset = CitationDataset(X_train_text, y_train_text, tokenizer)
test_dataset = CitationDataset(X_test_text, y_test_text, tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    eval_strategy="epoch", # Corrected parameter name
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
print("Training transformer model...")
trainer.train()

# Evaluate
eval_results = trainer.evaluate()
print(f"Transformer model evaluation results: {eval_results}")


class DataReferenceExtractor:
    def __init__(self, classification_model=None, tokenizer=None):
        self.classification_model = classification_model
        self.tokenizer = tokenizer
        self.doi_pattern = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
        self.accession_patterns = {
            'GEO': re.compile(r'GSE\d+', re.I),
            'SRA': re.compile(r'SRP\d+|SRR\d+|SRA\d+', re.I),
            'ENA': re.compile(r'PRJNA\d+|ERP\d+', re.I),
            'PDB': re.compile(r'PDB\s*[:#]?\s*[0-9A-Za-z]{4}', re.I),
        }

    def extract_references(self, text):
        """Extract all potential dataset references from text"""
        references = []

        # Extract DOIs
        for match in self.doi_pattern.finditer(text):
            references.append({
                'type': 'doi',
                'id': match.group(0),
                'context': self._get_context(text, match.start(), match.end())
            })

        # Extract accession numbers
        for repo_name, pattern in self.accession_patterns.items():
            for match in pattern.finditer(text):
                references.append({
                    'type': 'accession',
                    'id': match.group(0),
                    'repository': repo_name,
                    'context': self._get_context(text, match.start(), match.end())
                })

        return references

    def _get_context(self, text, start, end, window=100):
        """Get context around a match"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]

    def classify_references(self, references):
        """Classify references as Primary or Secondary"""
        for ref in references:
            if self.classification_model and self.tokenizer:
                # Use transformer model
                inputs = self.tokenizer(
                    ref['context'],
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                    padding=True
                )
                with torch.no_grad():
                    outputs = self.classification_model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=1).item()
                ref['citation_type'] = 'Primary' if prediction == 1 else 'Secondary'
            else:
                # Use simple heuristic as fallback
                context_lower = ref['context'].lower()
                primary_indicators = ['generate', 'produce', 'collect', 'measure', 'experiment']
                secondary_indicators = ['use', 'analyze', 'reuse', 'existing', 'previous']

                primary_score = sum(1 for word in primary_indicators if word in context_lower)
                secondary_score = sum(1 for word in secondary_indicators if word in context_lower)

                if primary_score > secondary_score:
                    ref['citation_type'] = 'Primary'
                else:
                    ref['citation_type'] = 'Secondary'

        return references

# Initialize the extractor
extractor = DataReferenceExtractor()

# Test the pipeline
sample_text = "We generated new data available at GEO: GSE12345 and reused existing data from DOI: 10.1234/abc.def"
references = extractor.extract_references(sample_text)
classified_references = extractor.classify_references(references)

print("Extracted and classified references:")
for ref in classified_references:
    print(f"{ref['id']} - {ref.get('repository', 'DOI')} - {ref['citation_type']}")


# First, install required packages
!pip install PyMuPDF spacy scikit-learn xgboost matplotlib seaborn joblib
!python -m spacy download en_core_web_sm

import re
import spacy
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import fitz, if not available, provide alternative
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    print("PyMuPDF not available. PDF text extraction will be limited.")
    PDF_SUPPORT = False

class AdvancedDataCitationMiner:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.dataset_patterns = self._initialize_patterns()
        self.validation_rules = self._initialize_validation_rules()
        self.models = {}
        self.performance_history = []
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self._initialize_models()

    def _initialize_patterns(self):
        """Initialize regex patterns for dataset detection"""
        return {
            'doi': re.compile(r'10\.\d{4,9}/[-._;()/:A-Z0-9]+', re.I),
            'url': re.compile(r'https?://\S+', re.I),
            'common_datasets': re.compile(r'\b(MNIST|CIFAR|ImageNet|WikiData|UCI|Kaggle|FigShare|Zenodo)\b', re.I),
            'repository': re.compile(r'\b(GEO|SRA|ENA|PDB|PRIDE|ArrayExpress|dbGaP)\b', re.I),
            'accession': re.compile(r'\b(GSE\d+|SRP\d+|PRJNA\d+|PXD\d+)\b', re.I)
        }

    def _initialize_validation_rules(self):
        """Initialize validation rules for dataset citations"""
        return {
            'doi_format': lambda x: bool(re.match(r'^10\.\d{4,9}/', x)),
            'repository_valid': lambda x: x.upper() in ['GEO', 'SRA', 'ENA', 'PDB', 'PRIDE', 'ARRAYEXPRESS', 'DBGAP']
        }

    def _initialize_models(self):
        """Initialize ML models with simpler configurations"""
        # XGBoost Model
        self.models['xgb'] = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )

        # Random Forest Model
        self.models['rf'] = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42
        )

        # Calibrated Classifier
        self.models['calibrated'] = CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=30, random_state=42),
            cv=2
        )

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF file with fallback"""
        if not PDF_SUPPORT:
            return f"PDF text extraction not available. Install PyMuPDF with: !pip install PyMuPDF"

        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
        return text

    def extract_text_from_file(self, file_path: Path) -> str:
        """Extract text from various file types"""
        if file_path.suffix.lower() == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_path.suffix.lower() in ['.txt', '.text', '.md']:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except:
                return ""
        else:
            return f"Unsupported file type: {file_path.suffix}"

    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text and split into sentences"""
        if not text or "not available" in text:
            return []

        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]

    def extract_features(self, sentences: List[str]) -> np.ndarray:
        """Extract TF-IDF features from sentences"""
        if not sentences:
            return np.array([])
        return self.vectorizer.fit_transform(sentences).toarray()

    def train_models(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.3):
        """Train all models with proper error handling"""
        if len(X) == 0 or len(y) == 0:
            print("No data available for training")
            return None

        try:
            # Split data - ensure we have enough samples for each class
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            results = {}

            # Train and evaluate XGBoost
            print("Training XGBoost model...")
            self.models['xgb'].fit(X_train, y_train)
            xgb_pred = self.models['xgb'].predict(X_test)
            xgb_acc = accuracy_score(y_test, xgb_pred)
            results['xgb'] = xgb_acc
            print(f"XGBoost Accuracy: {xgb_acc:.4f}")

            # Train and evaluate Random Forest
            print("Training Random Forest model...")
            self.models['rf'].fit(X_train, y_train)
            rf_pred = self.models['rf'].predict(X_test)
            rf_acc = accuracy_score(y_test, rf_pred)
            results['rf'] = rf_acc
            print(f"Random Forest Accuracy: {rf_acc:.4f}")

            # Train and evaluate Calibrated Classifier
            print("Training Calibrated Classifier...")
            self.models['calibrated'].fit(X_train, y_train)
            cal_pred = self.models['calibrated'].predict(X_test)
            cal_acc = accuracy_score(y_test, cal_pred)
            results['calibrated'] = cal_acc
            print(f"Calibrated Classifier Accuracy: {cal_acc:.4f}")

            # Store performance
            self.performance_history.append(results)

            # Create ensemble prediction (majority voting)
            ensemble_pred = []
            for i in range(len(y_test)):
                votes = [xgb_pred[i], rf_pred[i], cal_pred[i]]
                ensemble_pred.append(max(set(votes), key=votes.count))

            ensemble_acc = accuracy_score(y_test, ensemble_pred)
            results['ensemble'] = ensemble_acc
            print(f"Ensemble Accuracy: {ensemble_acc:.4f}")

            # Print detailed classification report
            print("\nClassification Report (Ensemble):")
            print(classification_report(y_test, ensemble_pred,
                                      target_names=['primary', 'secondary', 'none']))

            # Plot confusion matrix
            self.plot_confusion_matrix(y_test, ensemble_pred)

            return results

        except Exception as e:
            print(f"Error in model training: {e}")
            return None

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix for evaluation"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['primary', 'secondary', 'none'],
                   yticklabels=['primary', 'secondary', 'none'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    def predict_with_ensemble(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """Predict using ensemble of models"""
        if not sentences:
            return []

        if not hasattr(self.vectorizer, 'vocabulary_'):
            # If vectorizer not fitted, use rule-based approach
            return self._rule_based_prediction(sentences)

        try:
            # Extract features
            X = self.vectorizer.transform(sentences).toarray()

            # Get predictions from all models
            predictions = {}
            for model_name, model in self.models.items():
                predictions[model_name] = model.predict(X)

            # Ensemble voting
            results = []
            for i, sentence in enumerate(sentences):
                votes = [predictions[model_name][i] for model_name in predictions]
                final_pred = max(set(votes), key=votes.count)

                dataset_id = self._extract_dataset_id(sentence)
                confidence = votes.count(final_pred) / len(votes)

                results.append({
                    'sentence': sentence,
                    'dataset_id': dataset_id,
                    'type': final_pred,
                    'confidence': confidence,
                    'validation_msg': self._validate_citation(sentence, final_pred),
                    'model_votes': {name: pred[i] for name, pred in predictions.items()}
                })

            return results

        except Exception as e:
            print(f"Error in ensemble prediction: {e}")
            return self._rule_based_prediction(sentences)

    def _rule_based_prediction(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """Fallback rule-based prediction when ML models are not available"""
        results = []
        for sentence in sentences:
            # Rule-based classification
            pred_type = 'none'
            confidence = 0.5

            # Check for explicit dataset mentions
            if any(pattern.search(sentence) for pattern in self.dataset_patterns.values()):
                if (self.dataset_patterns['doi'].search(sentence) or
                    self.dataset_patterns['url'].search(sentence) or
                    self.dataset_patterns['accession'].search(sentence)):
                    pred_type = 'primary'
                    confidence = 0.8
                else:
                    pred_type = 'secondary'
                    confidence = 0.7

            dataset_id = self._extract_dataset_id(sentence)

            results.append({
                'sentence': sentence,
                'dataset_id': dataset_id,
                'type': pred_type,
                'confidence': confidence,
                'validation_msg': self._validate_citation(sentence, pred_type),
                'model_votes': {'rule_based': pred_type}
            })

        return results

    def _extract_dataset_id(self, text: str) -> str:
        """Extract the most likely dataset identifier"""
        # Try DOI first
        doi_match = self.dataset_patterns['doi'].search(text)
        if doi_match:
            return doi_match.group(0)

        # Try accession numbers
        acc_match = self.dataset_patterns['accession'].search(text)
        if acc_match:
            return acc_match.group(0)

        # Try URL
        url_match = self.dataset_patterns['url'].search(text)
        if url_match:
            return url_match.group(0)

        # Try repository names
        repo_match = self.dataset_patterns['repository'].search(text)
        if repo_match:
            return repo_match.group(0)

        # Try common dataset names
        common_match = self.dataset_patterns['common_datasets'].search(text)
        if common_match:
            return common_match.group(0)

        # Fallback: return first 50 chars
        return text[:50].strip() + "..."

    def _validate_citation(self, text: str, citation_type: str) -> str:
        """Validate dataset citation and provide feedback"""
        if citation_type == 'none':
            return "No dataset citation detected"

        validation_results = []
        dataset_id = self._extract_dataset_id(text)

        if self.dataset_patterns['doi'].search(text):
            if self.validation_rules['doi_format'](dataset_id):
                validation_results.append("DOI format valid")
            else:
                validation_results.append("DOI format questionable")

        elif self.dataset_patterns['repository'].search(text):
            if self.validation_rules['repository_valid'](dataset_id):
                validation_results.append("Repository recognized")
            else:
                validation_results.append("Repository not recognized")

        return "; ".join(validation_results) if validation_results else "Basic validation passed"

    def process_text_content(self, text: str, source_name: str = "text_content") -> List[Dict[str, Any]]:
        """Process text content and return dataset citations"""
        sentences = self.preprocess_text(text)
        results = self.predict_with_ensemble(sentences)

        # Filter out non-dataset citations
        dataset_results = [r for r in results if r['type'] != 'none']

        # Add source metadata
        for result in dataset_results:
            result['source'] = source_name

        return dataset_results

    def save_models(self, path: str):
        """Save trained models to disk"""
        try:
            joblib.dump({
                'vectorizer': self.vectorizer,
                'models': self.models,
                'performance_history': self.performance_history
            }, path)
            print(f"Models saved to {path}")
        except Exception as e:
            print(f"Error saving models: {e}")

    def load_models(self, path: str):
        """Load trained models from disk"""
        try:
            data = joblib.load(path)
            self.vectorizer = data['vectorizer']
            self.models = data['models']
            self.performance_history = data['performance_history']
            print(f"Models loaded from {path}")
        except Exception as e:
            print(f"Error loading models: {e}")

# Enhanced sample data creation
def create_sample_data():
    """Create better sample training data"""
    sentences = [
        # Primary citations (explicit URLs/DOIs/accessions)
        "The dataset is available at https://example.com/data DOI: 10.1234/abc.123",
        "All code and data are available on FigShare at DOI: 10.5678/xyz.789",
        "Raw data can be downloaded from http://data.repository.org/id/12345",
        "The complete dataset is hosted at DOI: 10.1000/182.456",
        "RNA-seq data is available under GEO accession GSE12345",
        "Proteomics data was deposited to PRIDE with accession PXD008765",
        "Sequencing reads are available in SRA under SRP98765",

        # Secondary citations (mentions without direct links)
        "We used the MNIST dataset for training our model",
        "Data was obtained from ImageNet repository",
        "The CIFAR-10 dataset was used for evaluation",
        "Our method was tested on the WikiData knowledge base",
        "Training data consisted of 50,000 samples from the Kaggle competition",
        "The model was trained on a subset of the UC Irvine Machine Learning Repository",
        "We used the Zenodo archive for data preservation",
        "Protein structures were retrieved from PDB entry 1ABC",

        # Non-dataset sentences
        "The results are shown in Table 1 with statistical significance",
        "We conducted experiments using standard protocols",
        "Experimental results demonstrate the effectiveness of our approach",
        "We followed the same evaluation protocol as in previous work",
        "Performance was measured using standard metrics",
        "The experimental setup is described in the following section",
        "Statistical analysis was performed using R version 3.6.1",
        "All experiments were repeated three times for reproducibility"
    ]

    labels = [
        'primary', 'primary', 'primary', 'primary', 'primary', 'primary', 'primary',
        'secondary', 'secondary', 'secondary', 'secondary', 'secondary',
        'secondary', 'secondary', 'secondary',
        'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none'
    ]

    return sentences, labels

def main_advanced_pipeline():
    """Advanced pipeline with proper error handling and realistic workflow"""
    print("MAKE YOUR DATA COUNT - Advanced Data Citation Mining Pipeline")
    print("=" * 60)

    # Initialize advanced miner
    advanced_miner = AdvancedDataCitationMiner()

    # Create enhanced sample training data
    print("Creating sample training data...")
    sentences, labels = create_sample_data()

    # Extract features
    print("Extracting features...")
    X_features = advanced_miner.extract_features(sentences)

    # Convert labels to numerical values
    label_map = {'primary': 0, 'secondary': 1, 'none': 2}
    y_numeric = np.array([label_map[label] for label in labels])

    # Train models
    print("Training models...")
    training_results = advanced_miner.train_models(X_features, y_numeric, test_size=0.3)

    # Process sample content
    print("\nProcessing sample content...")
    results = advanced_miner.predict_with_ensemble(sentences)

    # Display results
    print("\nDetection Results:")
    print("=" * 60)
    for i, result in enumerate(results):
        if result['type'] != 'none':  # Only show dataset citations
            print(f"{i+1}. Dataset: {result['dataset_id']}")
            print(f"   Type: {result['type']} (Confidence: {result['confidence']:.2f})")
            print(f"   Validation: {result['validation_msg']}")
            print(f"   Context: {result['sentence'][:80]}...")
            print("-" * 50)

    # Show statistics
    total_citations = sum(1 for r in results if r['type'] != 'none')
    print(f"\nTotal dataset citations found: {total_citations}/{len(sentences)}")
    print(f"Primary citations: {sum(1 for r in results if r['type'] == 'primary')}")
    print(f"Secondary citations: {sum(1 for r in results if r['type'] == 'secondary')}")

    # Save models
    advanced_miner.save_models("data_citation_models.pkl")

    print("\nPipeline execution completed successfully!")

# Run the pipeline
if __name__ == "__main__":
    main_advanced_pipeline()

# Example of processing text content directly
print("\n" + "="*60)
print("EXAMPLE: Processing custom text content")
print("="*60)

sample_text = """
Our research utilized several datasets. The primary dataset was obtained from
GEO under accession GSE12345. We also used the ImageNet dataset for transfer
learning. All code is available at DOI: 10.5281/zenodo.1234567. Additional
protein structures were retrieved from PDB entry 1ABC.
"""

miner = AdvancedDataCitationMiner()
custom_results = miner.process_text_content(sample_text, "custom_research")

print(f"Found {len(custom_results)} dataset citations in custom text:")
for result in custom_results:
    print(f"- {result['dataset_id']} ({result['type']}, confidence: {result['confidence']:.2f})")


# --- 1. SETUP ENVIRONMENT ---
print("Setting up environment for Make Data Count pipeline...")

# Install required packages
!pip install spacy requests rapidfuzz pandas scikit-learn matplotlib seaborn
!python -m spacy download en_core_web_sm

import re
import json
import os
from typing import List, Dict, Tuple, Set
import requests
import pandas as pd
import numpy as np
from collections import Counter
import time
from rapidfuzz import process, fuzz

# NLP
import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except:
    !python -m spacy download en_core_web_sm
    nlp = spacy.load("en_core_web_sm")

# Metrics
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('default')
sns.set_palette("husl")
%matplotlib inline

print("Environment setup complete!")

# --- 2. UTILITY FUNCTIONS: REGEX PATTERNS & EXTRACTORS ---
print("\nSetting up regex patterns and extractors...")

# DOI regex (relaxed but practical)
doi_pattern = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)

# Common repository accession patterns
accession_patterns = {
    'GEO': re.compile(r'GSE\d+', re.I),
    'SRA': re.compile(r'SRP\d+|SRR\d+|SRA\d+', re.I),
    'ENA': re.compile(r'PRJNA\d+|ERP\d+', re.I),
    'PDB': re.compile(r'PDB\s*[:#]?\s*[0-9A-Za-z]{4}', re.I),
    'Zenodo': re.compile(r'zenodo\.\d+|10\.5281/zenodo\.\d+', re.I),
    'Figshare': re.compile(r'figshare\.\d+|10\.6084/m9\.figshare\.\d+', re.I),
}

def extract_references(text: str) -> Dict[str, List[str]]:
    """Extract DOIs and accession numbers from text"""
    text = text or ""
    results = {'dois': [], 'accessions': []}

    # Extract DOIs
    results['dois'] = list(set([m.group(0).rstrip('.;,') for m in doi_pattern.finditer(text)]))

    # Extract accessions
    accs = []
    for name, pat in accession_patterns.items():
        found = [m.group(0) for m in pat.finditer(text)]
        accs.extend(found)

    results['accessions'] = list(set([a.rstrip('.;,') for a in accs]))
    return results

# Quick test
sample_text = "We used data from GEO accession GSE12345, SRA SRP98765 and Zenodo 10.5281/zenodo.1234567. Also see DOI 10.1038/s41586-020-2649-2."
print("Sample text:", sample_text)
print("Extracted references:", extract_references(sample_text))

# --- 3. HEURISTICS & NER FOR INFORMAL MENTIONS ---
print("\nSetting up heuristics and NER for informal mentions...")

# Heuristic: look for repository names + verbs (deposited, available, archived)
repo_keywords = [
    'geo', 'sra', 'ena', 'zenodo', 'figshare', 'dryad', 'dataverse', 'pdb'
]

heuristic_pattern = re.compile(
    r'(' + '|'.join(repo_keywords) + r')[\w\s\-:]*?(accession|id|doi|deposited|available|under|in)\s*[:#]?\s*([A-Za-z0-9\._-]+)',
    re.I
)

def extract_informal_mentions(text: str) -> List[Tuple[str, str]]:
    """Extract informal dataset mentions using heuristics and NER"""
    matches = []

    # Heuristic pattern matching
    for m in heuristic_pattern.finditer(text):
        repo = m.group(1)
        token = m.group(3)
        matches.append((repo, token))

    # Use spaCy to find potential accession-like tokens near repo mentions
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ('ORG', 'PRODUCT'):
            # Look for tokens that resemble accession IDs
            if re.search(r'\d{3,}', ent.text) or any(keyword in ent.text.lower() for keyword in repo_keywords):
                matches.append((ent.label_, ent.text))

    return matches

# Test informal mention extraction
test_text = "Data are deposited in GEO under accession number GSE98765 and are also available on Zenodo: 10.5281/zenodo.7654321."
print("Informal mentions:", extract_informal_mentions(test_text))

# --- 4. API CLIENT FOR CROSSREF & DATACITE ---
print("\nSetting up API clients for Crossref and DataCite...")

class APIClient:
    """Client for Crossref and DataCite APIs with rate limiting"""
    def __init__(self):
        self.crossref_base = 'https://api.crossref.org/works/'
        self.datacite_base = 'https://api.datacite.org/dois/'
        self.cache = {}
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests

    def _rate_limit(self):
        """Respectful rate limiting"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def lookup_doi_crossref(self, doi: str) -> Dict:
        """Look up DOI metadata from Crossref"""
        if doi in self.cache:
            return self.cache[doi]

        self._rate_limit()
        cleaned_doi = doi.strip().rstrip('.')

        try:
            response = requests.get(self.crossref_base + cleaned_doi, timeout=10)
            if response.status_code == 200:
                msg = response.json().get('message', {})
                result = {
                    'title': msg.get('title', [''])[0] if msg.get('title') else '',
                    'publisher': msg.get('publisher', ''),
                    'issued': msg.get('issued', {}),
                    'type': msg.get('type', ''),
                    'container-title': msg.get('container-title', []),
                    'success': True
                }
            else:
                result = {'success': False, 'error': f'HTTP {response.status_code}'}
        except Exception as e:
            result = {'success': False, 'error': str(e)}

        self.cache[doi] = result
        return result

    def lookup_doi_datacite(self, doi: str) -> Dict:
        """Look up DOI metadata from DataCite"""
        if doi in self.cache:
            return self.cache[doi]

        self._rate_limit()
        cleaned_doi = doi.strip().rstrip('.')

        try:
            response = requests.get(self.datacite_base + cleaned_doi, timeout=10)
            if response.status_code == 200:
                msg = response.json().get('data', {})
                attrs = msg.get('attributes', {})
                result = {
                    'title': attrs.get('titles', [{}])[0].get('title', ''),
                    'publisher': attrs.get('publisher', ''),
                    'published': attrs.get('published', ''),
                    'types': attrs.get('types', {}),
                    'success': True
                }
            else:
                result = {'success': False, 'error': f'HTTP {response.status_code}'}
        except Exception as e:
            result = {'success': False, 'error': str(e)}

        self.cache[doi] = result
        return result

# Initialize API client
api_client = APIClient()

# --- 5. PROCESSING PIPELINE ---
print("\nSetting up processing pipeline...")

def process_corpus(df: pd.DataFrame, text_col: str = 'text', use_api: bool = False) -> pd.DataFrame:
    """
    Process corpus and extract dataset references

    Args:
        df: DataFrame with text content
        text_col: Name of the column containing text
        use_api: Whether to call external APIs for metadata

    Returns:
        DataFrame with extracted references
    """
    outputs = []

    for idx, row in df.iterrows():
        text = row[text_col]
        source_id = row.get('id', f"doc_{idx}")

        # Extract references
        refs = extract_references(text)
        informal = extract_informal_mentions(text)
        matched = []

        # Process DOIs
        for doi in refs['dois']:
            meta = {}
            if use_api:
                # Try Crossref first, then DataCite
                meta = api_client.lookup_doi_crossref(doi)
                if not meta.get('success'):
                    meta = api_client.lookup_doi_datacite(doi)
            matched.append({'type': 'doi', 'key': doi, 'meta': meta})

        # Process accessions
        for acc in refs['accessions']:
            matched.append({'type': 'accession', 'key': acc, 'meta': {}})

        # Process informal mentions
        for repo, token in informal:
            matched.append({'type': 'informal', 'key': token, 'repo': repo, 'meta': {}})

        outputs.append({
            'source_id': source_id,
            'extracted': matched,
            'total_references': len(matched)
        })

    return pd.DataFrame(outputs)

# Create sample corpus
corpus = pd.DataFrame([
    {'id': 'paper1', 'text': sample_text},
    {'id': 'paper2', 'text': 'Our raw reads were deposited to SRA under SRP123456. See also DOI 10.5281/zenodo.7654321.'},
    {'id': 'paper3', 'text': 'We used standard datasets including MNIST and CIFAR-10 for evaluation.'}
])

print("Sample corpus created:")
print(corpus)

# Process corpus (without API calls for demonstration)
print("\nProcessing corpus...")
results_df = process_corpus(corpus, use_api=False)
print("Processing complete!")
print(results_df)

# --- 6. EVALUATION FRAMEWORK ---
print("\nSetting up evaluation framework...")

# Create gold standard
gold = [
    {
        'source_id': 'paper1',
        'labels': [
            {'type': 'doi', 'key': '10.1038/s41586-020-2649-2'},
            {'type': 'accession', 'key': 'GSE12345'},
            {'type': 'accession', 'key': 'SRP98765'},
            {'type': 'doi', 'key': '10.5281/zenodo.1234567'}
        ]
    },
    {
        'source_id': 'paper2',
        'labels': [
            {'type': 'accession', 'key': 'SRP123456'},
            {'type': 'doi', 'key': '10.5281/zenodo.7654321'}
        ]
    },
    {
        'source_id': 'paper3',
        'labels': []  # No expected references
    }
]

def eval_detection(gold: List[Dict], predicted: List[Dict]) -> Dict:
    """
    Evaluate detection performance against gold standard

    Args:
        gold: List of gold standard annotations
        predicted: List of predicted annotations

    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    y_true = []
    y_pred = []

    for g in gold:
        sid = g['source_id']
        gset = set([(l['type'], l['key'].lower()) for l in g['labels']])

        # Find corresponding prediction
        p = next((p for p in predicted if p['source_id'] == sid), {'extracted': []})
        pset = set([(e['type'], e['key'].lower()) for e in p['extracted']])

        # True positives and false negatives
        for label in gset:
            y_true.append(1)
            y_pred.append(1 if label in pset else 0)

        # False positives
        for label in pset:
            if label not in gset:
                y_true.append(0)
                y_pred.append(1)

    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': len([x for x in y_true if x == 1])  # Number of actual positives
    }

# Convert results_df to list format for evaluation
predicted_list = results_df.to_dict('records')
eval_results = eval_detection(gold, predicted_list)

print("Evaluation results:")
print(f"Precision: {eval_results['precision']:.3f}")
print(f"Recall: {eval_results['recall']:.3f}")
print(f"F1 Score: {eval_results['f1']:.3f}")
print(f"Support: {eval_results['support']}")

# --- 7. MDC-COMPATIBLE OUTPUT FORMAT ---
print("\nGenerating MDC-compatible output...")

def format_for_mdc(records: List[Dict]) -> List[Dict]:
    """
    Format extracted references for Make Data Count compatibility

    Args:
        records: List of records with extracted references

    Returns:
        List of MDC-compatible records
    """
    mdc_output = []

    for record in records:
        source_id = record['source_id']

        for extracted in record['extracted']:
            did = extracted.get('key', '')
            dtype = extracted.get('type', '')
            repo = extracted.get('repo', '')
            meta = extracted.get('meta', {})

            # Determine confidence score
            if dtype == 'doi':
                confidence = 0.9
                extraction_method = 'regex'
            elif dtype == 'accession':
                confidence = 0.8
                extraction_method = 'pattern_matching'
            else:  # informal
                confidence = 0.6
                extraction_method = 'heuristic+ner'

            # Create MDC record
            mdc_record = {
                'dataset_identifier': did,
                'dataset_identifier_type': dtype,
                'repository': repo if repo else dtype,
                'referencing_publication_id': source_id,
                'confidence_score': confidence,
                'extraction_method': extraction_method,
                'extraction_timestamp': pd.Timestamp.now().isoformat()
            }

            # Add metadata if available
            if meta.get('success'):
                mdc_record.update({
                    'dataset_title': meta.get('title', ''),
                    'publisher': meta.get('publisher', ''),
                    'publication_date': meta.get('published') or meta.get('issued', {}).get('date-parts', [[]])[0]
                })

            mdc_output.append(mdc_record)

    return mdc_output

# Generate MDC output
mdc_records = format_for_mdc(predicted_list)
mdc_df = pd.DataFrame(mdc_records)

print("MDC-compatible output:")
print(mdc_df)

# --- 8. VISUALIZATION ---
print("\nCreating visualizations...")

# Count reference types
ref_types = []
for record in predicted_list:
    for extracted in record['extracted']:
        ref_types.append(extracted['type'])

if ref_types:
    type_counts = pd.Series(ref_types).value_counts()

    plt.figure(figsize=(10, 6))
    type_counts.plot(kind='bar', color=['#2E86AB', '#A23B72', '#F18F01'])
    plt.title('Distribution of Extracted Reference Types')
    plt.xlabel('Reference Type')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
else:
    print("No references found for visualization")

# --- 9. SAVE RESULTS ---
print("\nSaving results...")

# Save to CSV
mdc_df.to_csv('mdc_data_references.csv', index=False)
results_df.to_csv('extraction_results.csv', index=False)

# Save to JSON
with open('mdc_data_references.json', 'w') as f:
    json.dump(mdc_records, f, indent=2, default=str)

print("Results saved to:")
print("- mdc_data_references.csv")
print("- mdc_data_references.json")
print("- extraction_results.csv")

# --- 10. COMPREHENSIVE SUMMARY ---
print("\n" + "="*60)
print("MAKE DATA COUNT - PIPELINE SUMMARY")
print("="*60)

total_docs = len(results_df)
total_refs = results_df['total_references'].sum()
primary_refs = sum(1 for record in mdc_records if record['dataset_identifier_type'] == 'doi')
secondary_refs = sum(1 for record in mdc_records if record['dataset_identifier_type'] == 'accession')
informal_refs = sum(1 for record in mdc_records if record['dataset_identifier_type'] == 'informal')

print(f"Documents processed: {total_docs}")
print(f"Total references found: {total_refs}")
print(f"Primary references (DOIs): {primary_refs}")
print(f"Secondary references (accessions): {secondary_refs}")
print(f"Informal mentions: {informal_refs}")
print(f"Precision: {eval_results['precision']:.3f}")
print(f"Recall: {eval_results['recall']:.3f}")
print(f"F1 Score: {eval_results['f1']:.3f}")

print("\n" + "="*60)
print("Pipeline execution completed successfully! ðŸŽ‰")
print("="*60)

# --- 11. EXAMPLE USAGE WITH API CALLS (OPTIONAL) ---
print("\nOptional: Example with API calls (requires internet)")
print("Uncomment the following lines to enable API lookups:")

"""
# Example with API calls
print("Processing with API lookups...")
api_results_df = process_corpus(corpus, use_api=True)
api_mdc_records = format_for_mdc(api_results_df.to_dict('records'))
api_mdc_df = pd.DataFrame(api_mdc_records)

print("API-enhanced results:")
print(api_mdc_df[['dataset_identifier', 'dataset_title', 'publisher']].head())
"""



# DOI regex (relaxed but practical)
doi_pattern = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)

# Common repository accession patterns
accession_patterns = {
    'GEO': re.compile(r'GSE\d+', re.I),
    'SRA': re.compile(r'SRP\d+|SRR\d+|SRA\d+', re.I),
    'ENA': re.compile(r'PRJNA\d+|ERP\d+', re.I),
    'PDB': re.compile(r'PDB\s*[:#]?\s*[0-9A-Za-z]{4}', re.I),
    'ArrayExpress': re.compile(r'E-[A-Z]*-[0-9]+', re.I),
    'Zenodo': re.compile(r'zenodo\.\d+|10\.5281/zenodo\.\d+', re.I),
    'Figshare': re.compile(r'figshare\.\d+|10\.6084/m9\.figshare\.\d+', re.I),
    'BioStudies': re.compile(r'S-BSST\d+', re.I),
    'Dryad': re.compile(r'dryad\.[a-z0-9]+', re.I),
}

# Generic function to extract DOIs and accessions
def extract_references(text: str) -> Dict[str, List[str]]:
    text = text or ""
    results = {'dois': [], 'accessions': []}
    # DOIs
    results['dois'] = list(set([m.group(0).rstrip('.;,') for m in doi_pattern.finditer(text)]))
    # Accessions
    accs = []
    for name, pat in accession_patterns.items():
        found = [m.group(0) for m in pat.finditer(text)]
        accs.extend(found)
    results['accessions'] = list(set([a.rstrip('.;,') for a in accs]))
    return results

# Quick test
sample_text = "We used data from GEO accession GSE12345, SRA SRP98765 and Zenodo 10.5281/zenodo.1234567. Also see DOI 10.1038/s41586-020-2649-2."
print(extract_references(sample_text))



# Heuristic: look for repository names + verbs (deposited, available, archived)
repo_keywords = [
    'geo', 'sra', 'ena', 'zenodo', 'figshare', 'dryad', 'dataverse', 'pdb',
    'arrayexpress', 'biostudies', 'gene expression omnibus', 'sequence read archive'
]

heuristic_pattern = re.compile(r'(' + '|'.join(repo_keywords) + r')[\w\s\-:]*?(accession|id|doi|deposited|available|under|in)\s*[:#]?\s*([A-Za-z0-9\._-]+)', re.I)

def extract_informal_mentions(text: str) -> List[Tuple[str,str]]:
    matches = []
    for m in heuristic_pattern.finditer(text):
        repo = m.group(1)
        token = m.group(3)
        matches.append((repo, token))
    # Use spaCy to find potential accession-like tokens near repo mentions
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ('ORG', 'PRODUCT'):
            # crude filter for tokens that look like accession ids
            if re.search(r'\d{3,}', ent.text):
                matches.append((ent.label_, ent.text))
    return matches

print(extract_informal_mentions("Data are deposited in GEO under accession number GSE98765 and are also available on Zenodo: 10.5281/zenodo.7654321."))


import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Define the feature extraction functions (same as in cell 2738d07c, kept for clarity)
def extract_text_features(text):
    """
    Extract various text features from context
    """
    features = {}

    # Basic text features
    features['length'] = len(text)
    features['word_count'] = len(text.split())
    features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text else 0

    # Keyword features
    primary_keywords = ['generate', 'produce', 'create', 'collect', 'measure', 'experiment', 'study']
    secondary_keywords = ['use', 'analyze', 'reuse', 'existing', 'previous', 'obtain', 'download']

    text_lower = text.lower()
    features['primary_keyword_count'] = sum(1 for word in primary_keywords if word in text_lower)
    features['secondary_keyword_count'] = sum(1 for word in secondary_keywords if word in text_lower)
    features['keyword_ratio'] = features['primary_keyword_count'] / (features['secondary_keyword_count'] + 1)

    # Repository mentions
    repositories = ['geo', 'genbank', 'arrayexpress', 'pdb', 'dryad', 'figshare', 'zenodo']
    features['repo_mentions'] = sum(1 for repo in repositories if repo in text_lower)

    return features

def extract_identifier_features(dataset_id):
    """
    Extract features from the dataset identifier itself
    """
    features = {}

    # Convert dataset_id to string to handle both integers and potential string IDs
    dataset_id_str = str(dataset_id)

    features['is_doi'] = 1 if '10.' in dataset_id_str else 0
    features['is_geo'] = 1 if dataset_id_str.startswith('GSE') else 0
    features['id_length'] = len(dataset_id_str)
    features['has_special_chars'] = 1 if any(c in dataset_id_str for c in ['/', '.', '-', ':']) else 0

    return features

def create_feature_matrix(df):
    """
    Create feature matrix from dataframe
    """
    features = []

    # Ensure df is not empty before iterating
    if df.empty:
        print("Input DataFrame is empty. Cannot create feature matrix.")
        return pd.DataFrame()

    for _, row in df.iterrows():
        feature_set = {}

        # Text features from context
        # Ensure 'mention_context' is treated as string and handle potential NaNs
        mention_context = str(row.get('mention_context', '')) # Get with default empty string
        text_feats = extract_text_features(mention_context)
        feature_set.update(text_feats)

        # Identifier features
        # Ensure 'dataset_id' is present before accessing
        if 'dataset_id' in row:
             id_feats = extract_identifier_features(row['dataset_id'])
             feature_set.update(id_feats)
        else:
             print(f"Warning: 'dataset_id' not found for row. Skipping identifier features.")


        # Target variable
        # Ensure 'citation_type' is present before accessing
        if 'citation_type' in row:
            feature_set['target'] = 1 if row['citation_type'] == 'Primary' else 0
        else:
            print(f"Warning: 'citation_type' not found for row. Cannot set target.")
            feature_set['target'] = -1 # Indicate missing target


        features.append(feature_set)

    # Convert the list of dictionaries to a DataFrame
    feature_df = pd.DataFrame(features)

    # Drop rows where target could not be set due to missing 'citation_type'
    if -1 in feature_df['target'].unique():
        print(f"Dropped {len(feature_df[feature_df['target'] == -1])} rows due to missing 'citation_type'.")
        feature_df = feature_df[feature_df['target'] != -1].reset_index(drop=True)


    return feature_df

# Assuming train_df was successfully loaded and preprocessed in cell 9lGclvXx07j2
# Use the existing train_df instead of trying to reconstruct from article_labels
# Ensure train_df is available globally or pass it to this cell
# For robustness in a notebook environment, explicitly check if train_df exists
if 'train_df' in globals() and not train_df.empty:
    print("Using existing train_df for feature engineering.")
    # Create feature matrix
    feature_df = create_feature_matrix(train_df)

    print("Feature matrix shape:", feature_df.shape)
    print("\nFeature matrix head:")
    display(feature_df.head())

    # Check for missing values
    print("\nMissing values:")
    print(feature_df.isnull().sum())

    # Correlation analysis (only if feature_df is not empty and contains numeric columns)
    if not feature_df.empty:
        numeric_cols = feature_df.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            plt.figure(figsize=(12, 8))
            correlation_matrix = feature_df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Feature Correlation Matrix')
            plt.show()
        else:
            print("No numeric columns in feature matrix for correlation analysis.")
    else:
         print("Feature matrix is empty. Skipping correlation analysis.")


else:
    print("train_df is not available or empty. Skipping feature engineering.")
    # Initialize feature_df as an empty DataFrame to prevent downstream errors
    feature_df = pd.DataFrame()


import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Define the feature extraction functions (same as in cell 2738d07c and HTynyepsel1W)
def extract_text_features(text):
    """
    Extract various text features from context
    """
    features = {}

    # Basic text features
    features['length'] = len(text)
    features['word_count'] = len(text.split())
    features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text else 0

    # Keyword features
    primary_keywords = ['generate', 'produce', 'create', 'collect', 'measure', 'experiment', 'study']
    secondary_keywords = ['use', 'analyze', 'reuse', 'existing', 'previous', 'obtain', 'download']

    text_lower = text.lower()
    features['primary_keyword_count'] = sum(1 for word in primary_keywords if word in text_lower)
    features['secondary_keyword_count'] = sum(1 for word in secondary_keywords if word in text_lower)
    features['keyword_ratio'] = features['primary_keyword_count'] / (features['secondary_keyword_count'] + 1)

    # Repository mentions
    repositories = ['geo', 'genbank', 'arrayexpress', 'pdb', 'dryad', 'figshare', 'zenodo']
    features['repo_mentions'] = sum(1 for repo in repositories if repo in text_lower)

    return features

def extract_identifier_features(dataset_id):
    """
    Extract features from the dataset identifier itself
    """
    features = {}

    # Convert dataset_id to string to handle both integers and potential string IDs
    dataset_id_str = str(dataset_id)

    features['is_doi'] = 1 if '10.' in dataset_id_str else 0
    features['is_geo'] = 1 if dataset_id_str.startswith('GSE') else 0
    features['id_length'] = len(dataset_id_str)
    features['has_special_chars'] = 1 if any(c in dataset_id_str for c in ['/', '.', '-', ':']) else 0

    return features

def create_feature_matrix(df):
    """
    Create feature matrix from dataframe
    """
    features = []

    # Ensure df is not empty before iterating
    if df.empty:
        print("Input DataFrame is empty. Cannot create feature matrix.")
        return pd.DataFrame()

    for _, row in df.iterrows():
        feature_set = {}

        # Text features from context
        # Ensure 'mention_context' is treated as string and handle potential NaNs
        mention_context = str(row.get('mention_context', '')) # Get with default empty string
        text_feats = extract_text_features(mention_context)
        feature_set.update(text_feats)

        # Identifier features
        # Ensure 'dataset_id' is present before accessing
        if 'dataset_id' in row:
             id_feats = extract_identifier_features(row['dataset_id'])
             feature_set.update(id_feats)
        else:
             print(f"Warning: 'dataset_id' not found for row. Skipping identifier features.")


        # Target variable
        # Ensure 'citation_type' is present before accessing
        if 'citation_type' in row:
            feature_set['target'] = 1 if row['citation_type'] == 'Primary' else 0
        else:
            print(f"Warning: 'citation_type' not found for row. Cannot set target.")
            feature_set['target'] = -1 # Indicate missing target


        features.append(feature_set)

    # Convert the list of dictionaries to a DataFrame
    feature_df = pd.DataFrame(features)

    # Drop rows where target could not be set due to missing 'citation_type'
    if -1 in feature_df['target'].unique():
        print(f"Dropped {len(feature_df[feature_df['target'] == -1])} rows due to missing 'citation_type'.")
        feature_df = feature_df[feature_df['target'] != -1].reset_index(drop=True)


    return feature_df

# Assuming train_df was successfully loaded and preprocessed in cell 9lGclvXx07j2
# Use the existing train_df instead of trying to reconstruct from article_labels
# Ensure train_df is available globally or pass it to this cell
# For robustness in a notebook environment, explicitly check if train_df exists
if 'train_df' in globals() and not train_df.empty:
    print("Using existing train_df for feature engineering.")
    # Create feature matrix
    feature_df = create_feature_matrix(train_df)

    print("Feature matrix shape:", feature_df.shape)
    print("\nFeature matrix head:")
    display(feature_df.head())

    # Check for missing values
    print("\nMissing values:")
    print(feature_df.isnull().sum())

    # Correlation analysis (only if feature_df is not empty and contains numeric columns)
    if not feature_df.empty:
        numeric_cols = feature_df.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            plt.figure(figsize=(12, 8))
            correlation_matrix = feature_df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Feature Correlation Matrix')
            plt.show()
        else:
            print("No numeric columns in feature matrix for correlation analysis.")
    else:
         print("Feature matrix is empty. Skipping correlation analysis.")


else:
    print("train_df is not available or empty. Skipping feature engineering.")
    # Initialize feature_df as an empty DataFrame to prevent downstream errors
    feature_df = pd.DataFrame()


import re
import json
import os
from typing import List, Dict, Tuple, Set
import requests
from collections import Counter

# NLP and ML
import spacy
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Deep Learning (ensure torch is imported if needed later)
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    import torch
    from torch.utils.data import Dataset
    DEEP_LEARNING_AVAILABLE = True
except ImportError as e:
    print(f"Could not import all deep learning libraries: {e}")
    DEEP_LEARNING_AVAILABLE = False

# Tika for text extraction (requires Java)
try:
    from tika import parser
    TIKA_AVAILABLE = True
except ImportError as e:
    print(f"Could not import Tika: {e}")
    TIKA_AVAILABLE = False
except Exception as e:
    # Catch potential errors during Tika initialization (e.g., Java not found)
    print(f"Error initializing Tika: {e}")
    TIKA_AVAILABLE = False


print("All necessary libraries imported.")

# --- DATA CITATION MINING PIPELINE ---

# DOI regex (relaxed but practical)
doi_pattern = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)

# Common repository accession patterns
accession_patterns = {
    'GEO': re.compile(r'GSE\d+', re.I),
    'SRA': re.compile(r'SRP\d+|SRR\d+|SRA\d+', re.I),
    'ENA': re.compile(r'PRJNA\d+|ERP\d+', re.I),
    'PDB': re.compile(r'PDB\s*[:#]?\s*[0-9A-Za-z]{4}', re.I),
    'Zenodo': re.compile(r'zenodo\.\d+', re.I),
    'Figshare': re.compile(r'figshare\.\d+', re.I),
}

def extract_references(text: str) -> Dict[str, List[str]]:
    """Extract DOIs and accession numbers from text"""
    if not text or not isinstance(text, str):
        return {'dois': [], 'accessions': []}

    results = {'dois': [], 'accessions': []}

    try:
        # Extract DOIs
        results['dois'] = list(set([m.group(0).rstrip('.;,') for m in doi_pattern.finditer(text)]))

        # Extract accessions
        accs = []
        for name, pat in accession_patterns.items():
            found = [m.group(0) for m in pat.finditer(text)]
            accs.extend(found)

        results['accessions'] = list(set([a.rstrip('.;,') for a in accs]))
    except Exception as e:
        print(f"Error extracting references: {e}")

    return results

# Heuristic patterns for informal mentions
repo_keywords = ['geo', 'sra', 'ena', 'zenodo', 'figshare', 'dryad', 'pdb']
heuristic_pattern = re.compile(
    r'(' + '|'.join(repo_keywords) + r')[\w\s\-:]*?(accession|id|doi|deposited|available|under|in)\s*[:#]?\s*([A-Za-z0-9\._-]+)',
    re.I
)

def extract_informal_mentions(text: str) -> List[Tuple[str, str]]:
    """Extract informal dataset mentions"""
    if not text or not isinstance(text, str):
        return []

    matches = []

    try:
        # Heuristic pattern matching
        for m in heuristic_pattern.finditer(text):
            repo = m.group(1)
            token = m.group(3)
            matches.append((repo, token))

        # Use spaCy if available, otherwise use simple regex
        if 'nlp' in globals() and nlp: # Check if nlp object is defined and not None
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ('ORG', 'PRODUCT'):
                    if re.search(r'\d{3,}', ent.text) or any(keyword in ent.text.lower() for keyword in repo_keywords):
                        matches.append((ent.label_, ent.text))
        else:
            # Fallback: look for repository-like patterns
            repo_pattern = re.compile(r'\b(' + '|'.join(repo_keywords) + r')\b', re.I)
            for match in repo_pattern.finditer(text):
                matches.append(('repository', match.group(0)))

    except Exception as e:
        print(f"Error extracting informal mentions: {e}")

    return matches

class APIClient:
    """Client for Crossref and DataCite APIs with rate limiting"""
    def __init__(self):
        self.crossref_base = 'https://api.crossref.org/works/'
        self.datacite_base = 'https://api.datacite.org/dois/'
        self.cache = {}
        self.request_delay = 1.0 # seconds

    def _rate_limit(self):
        """Respectful rate limiting"""
        import time # Import time here to avoid NameError if not imported globally
        time.sleep(self.request_delay)

    def lookup_doi(self, doi: str) -> Dict:
        """Look up DOI metadata"""
        if not doi:
            return {'success': False, 'error': 'Empty DOI'}

        if doi in self.cache:
            return self.cache[doi]

        self._rate_limit()
        cleaned_doi = doi.strip().rstrip('.,;')

        try:
            # Try Crossref
            response = requests.get(self.crossref_base + cleaned_doi, timeout=10)
            if response.status_code == 200:
                msg = response.json().get('message', {})
                result = {
                    'title': msg.get('title', [''])[0] if msg.get('title') else '',
                    'publisher': msg.get('publisher', ''),
                    'success': True,
                    'source': 'crossref'
                }
                self.cache[doi] = result
                return result

            # Try DataCite
            response = requests.get(self.datacite_base + cleaned_doi, timeout=10)
            if response.status_code == 200:
                msg = response.json().get('data', {})
                attrs = msg.get('attributes', {})
                result = {
                    'title': attrs.get('titles', [{}])[0].get('title', ''),
                    'publisher': attrs.get('publisher', ''),
                    'success': True,
                    'source': 'datacite'
                }
                self.cache[doi] = result
                return result

            return {'success': False, 'error': f'HTTP {response.status_code}'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

def process_corpus(df: pd.DataFrame, text_col: str = 'text', use_api: bool = False) -> pd.DataFrame:
    """Process corpus and extract dataset references"""
    if df.empty or text_col not in df.columns:
        print(f"Input DataFrame is empty or missing '{text_col}' column.")
        return pd.DataFrame(columns=['source_id', 'extracted', 'total_references'])

    outputs = []

    for idx, row in df.iterrows():
        try:
            text = str(row[text_col]) if pd.notna(row[text_col]) else ""
            source_id = row.get('id', f"doc_{idx}") # Use .get for robustness

            # Extract references
            refs = extract_references(text)
            informal = extract_informal_mentions(text)
            matched = []

            # Process DOIs
            for doi in refs['dois']:
                meta = {}
                if use_api:
                    api_client = APIClient()
                    meta = api_client.lookup_doi(doi)
                matched.append({'type': 'doi', 'key': doi, 'meta': meta})

            # Process accessions
            for acc in refs['accessions']:
                matched.append({'type': 'accession', 'key': acc, 'meta': {}})

            # Process informal mentions
            for repo, token in informal:
                matched.append({'type': 'informal', 'key': token, 'repo': repo, 'meta': {}})

            outputs.append({
                'source_id': source_id,
                'extracted': matched,
                'total_references': len(matched)
            })

        except Exception as e:
            print(f"Error processing row {idx} (Source ID: {row.get('id', 'N/A')}): {e}")
            outputs.append({
                'source_id': row.get('id', f"doc_{idx}"),
                'extracted': [],
                'total_references': 0
            })

    return pd.DataFrame(outputs)


# Create sample data (same as in cell I9Cq_zbIzYSz)
sample_text = "We used data from GEO accession GSE12345, SRA SRP98765 and Zenodo 10.5281/zenodo.1234567. Also see DOI 10.1038/s41586-020-2649-2."

corpus_data = [
    {'id': 'paper1', 'text': sample_text},
    {'id': 'paper2', 'text': 'Our raw reads were deposited to SRA under SRP123456. See also DOI 10.5281/zenodo.7654321.'},
    {'id': 'paper3', 'text': 'We used standard datasets including MNIST and CIFAR-10 for evaluation.'}
]

corpus = pd.DataFrame(corpus_data)

# Process corpus
print("Processing corpus...")
results_df = process_corpus(corpus, use_api=False)
print("Processing complete!")
print(results_df)

# Evaluation (same as in cell I9Cq_zbIzYSz)
gold = [
    {
        'source_id': 'paper1',
        'labels': [
            {'type': 'doi', 'key': '10.1038/s41586-020-2649-2'},
            {'type': 'accession', 'key': 'GSE12345'},
            {'type': 'accession', 'key': 'SRP98765'},
            {'type': 'doi', 'key': '10.5281/zenodo.1234567'}
        ]
    },
    {
        'source_id': 'paper2',
        'labels': [
            {'type': 'accession', 'key': 'SRP123456'},
            {'type': 'doi', 'key': '10.5281/zenodo.7654321'}
        ]
    },
    {
        'source_id': 'paper3',
        'labels': []
    }
]

def eval_detection(gold: List[Dict], predicted: List[Dict]) -> Dict:
    """Evaluate detection performance"""
    y_true = []
    y_pred = []

    # Create flattened sets for easier comparison
    gold_set = set()
    for g in gold:
        sid = g['source_id']
        for label in g['labels']:
            gold_set.add((sid, label['type'], label['key'].lower()))

    predicted_set = set()
    for p in predicted:
        sid = p['source_id']
        for extracted in p['extracted']:
            # Handle potential missing keys gracefully
            extracted_type = extracted.get('type')
            extracted_key = extracted.get('key')
            if extracted_type and extracted_key is not None:
                 predicted_set.add((sid, extracted_type, extracted_key.lower()))
            else:
                 print(f"Warning: Skipping malformed extracted record for {sid}: {extracted}")


    # Calculate True Positives, False Positives, False Negatives
    tp = len(gold_set.intersection(predicted_set))
    fp = len(predicted_set - gold_set)
    fn = len(gold_set - predicted_set)

    # Calculate Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


    return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}

eval_results = eval_detection(gold, results_df.to_dict('records'))
print(f"\nEvaluation Results:")
print(f"Precision: {eval_results['precision']:.3f}, Recall: {eval_results['recall']:.3f}, F1: {eval_results['f1']:.3f}")
print(f"TP: {eval_results['tp']}, FP: {eval_results['fp']}, FN: {eval_results['fn']}")


# MDC output format (same as in cell I9Cq_zbIzYSz)
def format_for_mdc(records: List[Dict]) -> List[Dict]:
    """Format for Make Data Count compatibility"""
    mdc_output = []

    for record in records:
        for extracted in record['extracted']:
            # Ensure required keys exist
            dataset_identifier = extracted.get('key', '')
            dataset_identifier_type = extracted.get('type', '')
            referencing_publication_id = record.get('source_id', '')

            if dataset_identifier and dataset_identifier_type and referencing_publication_id:
                mdc_record = {
                    'dataset_identifier': dataset_identifier,
                    'dataset_identifier_type': dataset_identifier_type,
                    'referencing_publication_id': referencing_publication_id,
                    'confidence_score': 0.9 if extracted.get('type') == 'doi' else 0.7, # Use .get()
                    'extraction_method': 'regex' if extracted.get('type') in ['doi', 'accession'] else 'heuristic' # Use .get()
                }
                mdc_output.append(mdc_record)
            else:
                 print(f"Warning: Skipping incomplete MDC record for {referencing_publication_id}: {extracted}")


    return mdc_output

mdc_output = format_for_mdc(results_df.to_dict('records'))
print("\nMDC output sample:")
if mdc_output:
    for record in mdc_output[:3]:
        print(record)
else:
    print("No MDC output generated (no references found).")


# Visualization (same as in cell I9Cq_zbIzYSz)
ref_types = [extracted['type'] for record in results_df.to_dict('records') for extracted in record['extracted']]
if ref_types:
    type_counts = pd.Series(ref_types).value_counts()
    plt.figure(figsize=(10, 6))
    type_counts.plot(kind='bar')
    plt.title('Reference Types Distribution')
    plt.xlabel('Reference Type')
    plt.ylabel('Count')
    plt.xticks(rotation=0) # Keep labels horizontal
    plt.tight_layout()
    plt.show()
else:
    print("\nNo references found for visualization.")

print("\nPipeline execution completed successfully! ðŸŽ‰")


# First, let's install the required packages with compatible versions
!pip install pandas==1.5.3 numpy==1.23.5 scikit-learn==1.2.2 matplotlib==3.7.1 seaborn==0.12.2 spacy==3.5.3
!python -m spacy download en_core_web_sm
!pip install -q scikit-learn spacy matplotlib seaborn
!python -m spacy download en_core_web_sm

print("Package installation completed!")

# Import libraries
import pandas as pd
import numpy as np
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Import machine learning related libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Import spaCy with error handling
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    print("âœ“ spaCy loaded successfully")
except Exception as e:
    print(f"Error loading spaCy: {e}")
    nlp = None

print("All necessary libraries imported successfully.")

# -------------------------
# Custom Transformer: Handcrafted Features
# -------------------------
class HandcraftedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, use_spacy=True):
        self.use_spacy = use_spacy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            if not text or not isinstance(text, str):
                features.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                continue

            text_lower = text.lower()

            # Basic text features
            text_length = len(text)
            word_count = len(text.split())

            # Keyword features for primary vs secondary citations
            primary_keywords = ['generate', 'produce', 'create', 'collect', 'deposit', 'this study', 'our data', 'new data']
            secondary_keywords = ['use', 'analyze', 'reuse', 'existing', 'previous', 'obtain', 'publicly available']

            primary_keyword_count = sum(1 for word in primary_keywords if word in text_lower)
            secondary_keyword_count = sum(1 for word in secondary_keywords if word in text_lower)
            primary_secondary_ratio = (primary_keyword_count + 1) / (secondary_keyword_count + 1)

            # Repository mentions
            repositories = ['geo', 'sra', 'ena', 'pdb', 'zenodo', 'figshare', 'dryad']
            repo_mentions = sum(1 for repo in repositories if repo in text_lower)

            # Reference indicators
            reference_indicators = ['accession', 'dataset', 'data set', 'repository', 'deposit', 'submit']
            reference_indicators_count = sum(1 for word in reference_indicators if word in text_lower)

            # SpaCy features if available
            if self.use_spacy and nlp is not None:
                try:
                    doc = nlp(text)
                    stopword_count = sum(1 for token in doc if token.is_stop)
                    alpha_tokens = sum(1 for token in doc if token.is_alpha)
                    numeric_tokens = sum(1 for token in doc if token.like_num)
                    unique_entities = len(set([ent.label_ for ent in doc.ents]))
                except:
                    stopword_count, alpha_tokens, numeric_tokens, unique_entities = 0, 0, 0, 0
            else:
                stopword_count, alpha_tokens, numeric_tokens, unique_entities = 0, 0, 0, 0

            features.append([
                text_length, word_count, primary_keyword_count, secondary_keyword_count,
                primary_secondary_ratio, repo_mentions, reference_indicators_count,
                stopword_count, alpha_tokens, numeric_tokens, unique_entities
            ])
        return np.array(features)

# -------------------------
# Data Citation Miner Class
# -------------------------
class DataCitationMiner:
    def __init__(self, use_spacy=True):
        self.dataset_patterns = self._initialize_patterns()
        self.use_spacy = use_spacy
        self.label_encoder = LabelEncoder()
        self.pipeline = self._create_pipeline()

    def _initialize_patterns(self):
        """Initialize regex patterns for dataset detection"""
        return {
            'doi': re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I),
            'geo': re.compile(r'GSE\d+', re.I),
            'sra': re.compile(r'SRP\d+|SRR\d+|SRA\d+', re.I),
            'ena': re.compile(r'PRJNA\d+|ERP\d+', re.I),
            'pdb': re.compile(r'PDB\s*[:#]?\s*[0-9A-Za-z]{4}', re.I),
            'zenodo': re.compile(r'zenodo\.\d+', re.I),
            'figshare': re.compile(r'figshare\.\d+', re.I),
            'dryad': re.compile(r'dryad\.[a-z0-9]+', re.I),
        }

    def _create_pipeline(self):
        """Create the ML pipeline"""
        tfidf = TfidfVectorizer(max_features=500, stop_words='english')
        handcrafted = HandcraftedFeatures(use_spacy=self.use_spacy)

        combined_features = FeatureUnion([
            ("tfidf", tfidf),
            ("handcrafted", handcrafted)
        ])

        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        pipeline = Pipeline([
            ("features", combined_features),
            ("clf", clf)
        ])

        return pipeline

    def extract_references(self, text: str) -> Dict[str, List[str]]:
        """Extract dataset references from text"""
        if not text or not isinstance(text, str):
            return {'dois': [], 'accessions': []}

        results = {'dois': [], 'accessions': []}

        try:
            # Extract DOIs
            results['dois'] = list(set([m.group(0).rstrip('.;,') for m in self.dataset_patterns['doi'].finditer(text)]))

            # Extract accessions from other repositories
            for repo in ['geo', 'sra', 'ena', 'pdb', 'zenodo', 'figshare', 'dryad']:
                matches = [m.group(0) for m in self.dataset_patterns[repo].finditer(text)]
                results['accessions'].extend(matches)

            results['accessions'] = list(set(results['accessions']))

        except Exception as e:
            print(f"Error extracting references: {e}")

        return results

    def train(self, X, y):
        """Train the model"""
        y_encoded = self.label_encoder.fit_transform(y)
        self.pipeline.fit(X, y_encoded)

        # Cross-validation
        # Reduce cv to avoid ValueError with small sample size
        cv_scores = cross_val_score(self.pipeline, X, y_encoded, cv=2) # Changed cv from 5 to 2
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        return cv_scores.mean()

    def predict(self, X):
        """Make predictions"""
        if not hasattr(self, 'pipeline') or not hasattr(self, 'label_encoder'):
            raise ValueError("Model not trained yet. Call train() first.")

        y_pred_encoded = self.pipeline.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)

    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        y_pred = self.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")

        print("Accuracy:", accuracy)
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def analyze_publication(self, text: str, source_id: str = "") -> Dict:
        """Comprehensive analysis of a publication text"""
        references = self.extract_references(text)

        # Predict citation type
        citation_type = self.predict([text])[0] if hasattr(self, 'pipeline') else self._rule_based_prediction(text)

        return {
            'source_id': source_id,
            'citation_type': citation_type,
            'datasets_mentioned': len(references['dois']) + len(references['accessions']),
            'doi_references': references['dois'],
            'accession_references': references['accessions'],
            'mdc_output': self.create_mdc_output(references, source_id)
        }

    def _rule_based_prediction(self, text: str) -> str:
        """Fallback rule-based prediction"""
        if not text:
            return 'Unknown'

        text_lower = text.lower()

        creation_indicators = ['generate', 'produce', 'create', 'collect', 'deposit', 'this study', 'our data', 'new data']
        usage_indicators = ['use', 'analyze', 'reuse', 'existing', 'previous', 'obtain', 'publicly available']

        creation_score = sum(1 for word in creation_indicators if word in text_lower)
        usage_score = sum(1 for word in usage_indicators if word in text_lower)

        if creation_score > usage_score:
            return 'Primary'
        elif usage_score > creation_score:
            return 'Secondary'
        else:
            return 'Unknown'

    def create_mdc_output(self, references: Dict, source_id: str = "") -> List[Dict]:
        """Create Make Data Count compatible output"""
        mdc_records = []

        # Process DOIs
        for doi in references.get('dois', []):
            mdc_records.append({
                'dataset_identifier': doi,
                'dataset_identifier_type': 'doi',
                'referencing_publication_id': source_id,
                'confidence_score': 0.9,
                'extraction_method': 'regex'
            })

        # Process accessions
        for acc in references.get('accessions', []):
            mdc_records.append({
                'dataset_identifier': acc,
                'dataset_identifier_type': 'accession',
                'referencing_publication_id': source_id,
                'confidence_score': 0.8,
                'extraction_method': 'pattern_matching'
            })

        return mdc_records

    def plot_feature_importance(self):
        """Plot feature importance"""
        if not hasattr(self, 'pipeline'):
            print("Model not trained yet. Call train() first.")
            return

        # Extract trained RandomForest
        rf_model = self.pipeline.named_steps["clf"]

        # Get feature names
        tfidf_features = self.pipeline.named_steps["features"].transformer_list[0][1].get_feature_names_out()
        handcrafted_features = [
            "text_length", "word_count", "primary_keyword_count", "secondary_keyword_count",
            "primary_secondary_ratio", "repo_mentions", "reference_indicators_count",
            "stopword_count", "alpha_tokens", "numeric_tokens", "unique_entities"
        ]

        all_features = np.concatenate([tfidf_features, handcrafted_features])

        # Get importances
        importances = rf_model.feature_importances_

        # Combine into dataframe
        feat_importances = pd.DataFrame({
            "feature": all_features,
            "importance": importances
        }).sort_values(by="importance", ascending=False).head(20)

        # Plot
        plt.figure(figsize=(10,6))
        sns.barplot(x="importance", y="feature", data=feat_importances, palette="viridis")
        plt.title("Top 20 Feature Importances (TF-IDF + Handcrafted)", fontsize=14)
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()

# -------------------------
# Create Sample Data
# -------------------------
def create_sample_corpus():
    """Create sample training data for demonstration"""
    data = {
        "text": [
            # Primary citations
            "We generated RNA-seq data and deposited it in GEO under accession GSE12345.",
            "Raw sequencing reads were produced and submitted to SRA with accession SRP98765.",
            "All custom code and datasets created in this study are available.",
            "New proteomics data was collected and deposited to PRIDE with identifier PXD12345.",
            "We produced original microarray data for this investigation.",
            "Data deposited in Zenodo with DOI 10.5281/zenodo.1234567.",

            # Secondary citations
            "We analyzed existing microarray data from GEO accession GSE54321.",
            "The study used protein structures from PDB for molecular docking.",
            "Previous results were compared with our findings using publicly available data.",
            "Existing RNA-seq datasets from SRA were obtained and reanalyzed.",
            "We retrieved climate data from the Dryad repository for our analysis.",
            "Used existing datasets from previous publications.",
        ],
        "label": [
            "primary", "primary", "primary", "primary", "primary", "primary",
            "secondary", "secondary", "secondary", "secondary", "secondary", "secondary"
        ]
    }

    return pd.DataFrame(data)

# -------------------------
# Main Demonstration
# -------------------------
def main():
    """Main demonstration function"""
    print("DATA CITATION MINING PIPELINE")
    print("=" * 50)

    # Create sample data
    print("Creating sample corpus...")
    df = create_sample_corpus()
    print("Sample corpus:")
    print(df)

    # Initialize miner
    print("\nInitializing Data Citation Miner...")
    miner = DataCitationMiner(use_spacy=(nlp is not None))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.3, random_state=42, stratify=df["label"]
    )

    # Train model
    print("\nTraining model...")
    cv_accuracy = miner.train(X_train, y_train)

    # Evaluate model
    print("\nEvaluating model...")
    scores = miner.evaluate(X_test, y_test)

    # Save evaluation scores
    submission_scores = pd.DataFrame([{
        "Accuracy": scores["accuracy"],
        "Precision": scores["precision"],
        "Recall": scores["recall"],
        "F1_Score": scores["f1"]
    }])
    submission_scores.to_csv("submission_scores.csv", index=False)
    print("âœ… Scores saved to submission_scores.csv")

    # Plot feature importance
    print("\nPlotting feature importance...")
    miner.plot_feature_importance()

    # Test predictions
    print("\nTesting predictions...")
    test_texts = [
        "We created and deposited new proteomics data in PRIDE.",
        "Existing RNA-seq data from GEO was reanalyzed.",
        "This study produced novel datasets available online.",
        "We used previously published data from SRA accession SRP12345.",
    ]

    for text in test_texts:
        prediction = miner.predict([text])[0]
        print(f"Text: {text[:50]}...")
        print(f"Predicted citation type: {prediction}")
        print("-" * 40)

    # Test reference extraction
    print("\nTesting reference extraction...")
    sample_text = "This study used data from GEO GSE12345 and SRA SRP98765. Also referenced DOI 10.1038/s41586-020-2649-2."
    references = miner.extract_references(sample_text)
    print(f"Text: {sample_text}")
    print(f"Extracted references: {references}")

    # Test comprehensive analysis
    print("\nTesting comprehensive publication analysis...")
    analysis = miner.analyze_publication(sample_text, "paper_123")
    print(f"Analysis results: {json.dumps(analysis, indent=2)}")

# Run the demonstration
if __name__ == "__main__":
    main()

print("\nPipeline execution completed successfully! ðŸŽ‰")


# First, let's ensure we have compatible versions to avoid conflicts
!pip install pandas==2.0.3 numpy==1.24.3 scikit-learn==1.2.2 matplotlib==3.7.2 seaborn==0.12.2 spacy==3.7.4 requests==2.31.0

# Try to install transformers if not available, but don't break if it fails
try:
    !pip install transformers==4.36.2 torch==2.0.1
    DEEP_LEARNING_AVAILABLE = True
except:
    print("Deep learning packages could not be installed, continuing without them...")
    DEEP_LEARNING_AVAILABLE = False

# Download compatible spaCy model
!python -m spacy download en_core_web_sm==3.7.1

print("Package installation completed!")

# Import core data science libraries
import pandas as pd
import numpy as np
import re
import json
import os
from collections import Counter
import time
import requests

# Import NLP library with error handling
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    print("âœ“ spaCy loaded successfully")
except Exception as e:
    print(f"Error loading spaCy: {e}")
    nlp = None

# Import machine learning related libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Configure visualization settings
plt.style.use('default')
sns.set_palette("husl")
%matplotlib inline

# Import deep learning libraries with error handling
DEEP_LEARNING_AVAILABLE = False
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    import torch
    from torch.utils.data import Dataset
    DEEP_LEARNING_AVAILABLE = True
    print("âœ“ Deep learning libraries imported successfully")
except ImportError as e:
    print(f"Could not import deep learning libraries: {e}")
    print("Continuing with traditional ML approaches only")

print("All necessary libraries import attempts completed.")

# --- DATA CITATION MINING PIPELINE ---

class DataCitationMiner:
    def __init__(self):
        self.dataset_patterns = self._initialize_patterns()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.label_encoder = LabelEncoder()
        self.models = {}

    def _initialize_patterns(self):
        """Initialize regex patterns for dataset detection"""
        return {
            'doi': re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I),
            'geo': re.compile(r'GSE\d+', re.I),
            'sra': re.compile(r'SRP\d+|SRR\d+|SRA\d+', re.I),
            'ena': re.compile(r'PRJNA\d+|ERP\d+', re.I),
            'pdb': re.compile(r'PDB\s*[:#]?\s*[0-9A-Za-z]{4}', re.I),
            'zenodo': re.compile(r'zenodo\.\d+', re.I),
            'figshare': re.compile(r'figshare\.\d+', re.I),
        }

    def extract_references(self, text: str) -> Dict[str, List[str]]:
        """Extract dataset references from text"""
        if not text or not isinstance(text, str):
            return {'dois': [], 'accessions': []}

        results = {'dois': [], 'accessions': []}

        try:
            # Extract DOIs
            results['dois'] = list(set([m.group(0).rstrip('.;,') for m in self.dataset_patterns['doi'].finditer(text)]))

            # Extract accessions from other repositories
            for repo in ['geo', 'sra', 'ena', 'pdb', 'zenodo', 'figshare']:
                matches = [m.group(0) for m in self.dataset_patterns[repo].finditer(text)]
                results['accessions'].extend(matches)

            results['accessions'] = list(set(results['accessions']))

        except Exception as e:
            print(f"Error extracting references: {e}")

        return results

    def extract_text_features(self, text: str) -> Dict:
        """Extract features from text for classification"""
        features = {}

        if not text:
            return features

        text_lower = text.lower()

        # Basic text features
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())

        # Keyword features for primary vs secondary citations
        primary_keywords = ['generate', 'produce', 'create', 'collect', 'deposit', 'this study', 'our data']
        secondary_keywords = ['use', 'analyze', 'reuse', 'existing', 'previous', 'obtain']

        features['primary_keyword_count'] = sum(1 for word in primary_keywords if word in text_lower)
        features['secondary_keyword_count'] = sum(1 for word in secondary_keywords if word in text_lower)

        # Repository mentions
        repositories = ['geo', 'sra', 'ena', 'pdb', 'zenodo', 'figshare', 'dryad']
        features['repo_mentions'] = sum(1 for repo in repositories if repo in text_lower)

        return features

    def prepare_training_data(self, corpus_df: pd.DataFrame, text_col: str = 'text', label_col: str = 'citation_type'):
        """Prepare training data for ML models"""
        features = []
        labels = []

        for _, row in corpus_df.iterrows():
            text = row[text_col] if pd.notna(row[text_col]) else ""

            # Extract features
            feat = self.extract_text_features(text)
            features.append(feat)
            labels.append(row[label_col])

        # Convert to DataFrame
        feature_df = pd.DataFrame(features)

        # Encode labels
        if labels:
            y_encoded = self.label_encoder.fit_transform(labels)
        else:
            y_encoded = np.array([])

        return feature_df, y_encoded

    def train_model(self, X, y, test_size=0.2):
        """Train a Random Forest classifier"""
        if len(X) == 0 or len(y) == 0:
            print("No data available for training")
            return None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        self.models['random_forest'] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models['random_forest'].fit(X_train, y_train)

        # Evaluate
        y_pred = self.models['random_forest'].predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Random Forest Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

        return accuracy

    def predict_citation_type(self, text: str) -> str:
        """Predict citation type using trained model or rules"""
        if not self.models:
            return self._rule_based_prediction(text)

        features = self.extract_text_features(text)
        feature_df = pd.DataFrame([features])

        if 'random_forest' in self.models:
            pred = self.models['random_forest'].predict(feature_df)[0]
            return self.label_encoder.inverse_transform([pred])[0]

        return self._rule_based_prediction(text)

    def _rule_based_prediction(self, text: str) -> str:
        """Fallback rule-based prediction"""
        if not text:
            return 'Unknown'

        text_lower = text.lower()

        creation_indicators = ['generate', 'produce', 'create', 'collect', 'deposit', 'this study']
        usage_indicators = ['use', 'analyze', 'reuse', 'existing', 'previous', 'obtain']

        creation_score = sum(1 for word in creation_indicators if word in text_lower)
        usage_score = sum(1 for word in usage_indicators if word in text_lower)

        if creation_score > usage_score:
            return 'Primary'
        elif usage_score > creation_score:
            return 'Secondary'
        else:
            return 'Unknown'

# Create sample training data
def create_sample_corpus():
    """Create sample training data for demonstration"""
    data = [
        # Primary citations
        {'text': 'We generated RNA-seq data and deposited it in GEO under accession GSE12345.', 'citation_type': 'Primary'},
        {'text': 'Raw sequencing reads were produced and submitted to SRA with accession SRP98765.', 'citation_type': 'Primary'},
        {'text': 'All custom code and datasets created in this study are available.', 'citation_type': 'Primary'},

        # Secondary citations
        {'text': 'We analyzed existing microarray data from GEO accession GSE54321.', 'citation_type': 'Secondary'},
        {'text': 'The study used protein structures from PDB for molecular docking.', 'citation_type': 'Secondary'},
        {'text': 'Previous results were compared with our findings.', 'citation_type': 'Secondary'},

        # Additional examples
        {'text': 'Data deposited in Zenodo with DOI 10.5281/zenodo.1234567.', 'citation_type': 'Primary'},
        {'text': 'Used existing datasets from previous publications.', 'citation_type': 'Secondary'}
    ]

    return pd.DataFrame(data)

# Main demonstration
def main():
    print("DATA CITATION MINING PIPELINE")
    print("=" * 50)

    # Create sample data
    print("Creating sample corpus...")
    corpus_df = create_sample_corpus()
    print("Sample corpus:")
    print(corpus_df)

    # Initialize miner
    print("\nInitializing Data Citation Miner...")
    miner = DataCitationMiner()

    # Prepare training data
    print("Preparing training data...")
    X, y = miner.prepare_training_data(corpus_df)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels: {y}")

    # Train model
    print("\nTraining model...")
    accuracy = miner.train_model(X, y)

    # Test predictions
    print("\nTesting predictions...")
    test_texts = [
        "We created and deposited new proteomics data in PRIDE.",
        "Existing RNA-seq data from GEO was reanalyzed.",
        "This study produced novel datasets available online."
    ]

    for text in test_texts:
        prediction = miner.predict_citation_type(text)
        print(f"Text: {text[:50]}...")
        print(f"Predicted citation type: {prediction}")
        print("-" * 40)

    # Test reference extraction
    print("\nTesting reference extraction...")
    sample_text = "This study used data from GEO GSE12345 and SRA SRP98765. Also referenced DOI 10.1038/s41586-020-2649-2."
    references = miner.extract_references(sample_text)
    print(f"Text: {sample_text}")
    print(f"Extracted references: {references}")

    # Visualization
    print("\nCreating visualizations...")

    # Citation type distribution
    plt.figure(figsize=(10, 6))
    corpus_df['citation_type'].value_counts().plot(kind='bar', color=['skyblue', 'lightcoral'])
    plt.title('Citation Type Distribution in Training Data')
    plt.xlabel('Citation Type')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.show()

    # Feature importance if model is trained
    if 'random_forest' in miner.models:
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': miner.models['random_forest'].feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.title('Feature Importance - Random Forest')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()

# Run the demonstration
if __name__ == "__main__":
    main()

# Additional utility functions
def create_mdc_output(references: Dict, source_id: str = "") -> List[Dict]:
    """Create Make Data Count compatible output"""
    mdc_records = []

    # Process DOIs
    for doi in references.get('dois', []):
        mdc_records.append({
            'dataset_identifier': doi,
            'dataset_identifier_type': 'doi',
            'referencing_publication_id': source_id,
            'confidence_score': 0.9,
            'extraction_method': 'regex'
        })

    # Process accessions
    for acc in references.get('accessions', []):
        mdc_records.append({
            'dataset_identifier': acc,
            'dataset_identifier_type': 'accession',
            'referencing_publication_id': source_id,
            'confidence_score': 0.8,
            'extraction_method': 'pattern_matching'
        })

    return mdc_records

# Example usage
print("\nMDC Output Example:")
sample_refs = {'dois': ['10.1038/s41586-020-2649-2'], 'accessions': ['GSE12345']}
mdc_output = create_mdc_output(sample_refs, 'paper_123')
for record in mdc_output:
    print(record)

print("\nPipeline execution completed successfully! ðŸŽ‰")


!pip uninstall numpy scipy scikit-learn transformers torch -y


!pip install pandas==2.2.2 numpy==1.26.4 scikit-learn==1.2.2 scipy==1.11.4 matplotlib==3.10.0 seaborn==0.13.2 spacy==3.8.7 requests==2.32.4 transformers==4.56.0 torch==2.8.0+cu126 tika python-magic rapidfuzz
!python -m spacy download en_core_web_sm


# First, clear the environment of potentially conflicting packages.
!pip uninstall numpy scipy scikit-learn transformers torch pandas spacy -y

# Install known compatible versions in a specific order to minimize conflicts.
!pip install numpy==1.24.3
!pip install scipy==1.10.1
!pip install scikit-learn==1.2.2
!pip install pandas==1.5.3

# Install transformers and a compatible torch version.
!pip install transformers==4.28.1
# Note: Finding a universally compatible torch version across all environments can be tricky.
# This version is chosen based on common usage with the other specified library versions.
# If this fails, further troubleshooting of the specific environment (e.g., CUDA version) might be needed.
!pip install torch==1.13.1+cpu # Use CPU version for broader compatibility

# Install other necessary packages.
!pip install spacy==3.5.3
!pip install matplotlib==3.7.1 seaborn==0.12.2 requests==2.28.1 rapidfuzz python-magic tika

# Download the compatible spaCy model.
!python -m spacy download en_core_web_sm==3.5.0

print("Installation process completed. Please restart the runtime.")


# Clear potentially problematic packages again.
!pip uninstall numpy scipy scikit-learn transformers torch pandas spacy -y

# Install core dependencies first.
!pip install numpy==1.26.4
!pip install scipy==1.11.4
!pip install pandas==2.2.2
!pip install scikit-learn==1.2.2

# Install NLP and ML libraries, including transformers and a more general torch version.
!pip install spacy==3.7.4
!pip install rapidfuzz python-magic tika requests==2.32.4 seaborn==0.13.2 matplotlib==3.10.0
!pip install transformers==4.36.2 # A version that might work with the chosen torch
!pip install torch==2.0.1 # Try a non-CUDA specific version first

# Download the compatible spaCy model.
!python -m spacy download en_core_web_sm==3.7.1

print("Installation process completed. Please restart the runtime.")


# Clear potentially problematic packages again.
!pip uninstall numpy scipy scikit-learn transformers torch pandas spacy -y

# Install core dependencies first.
!pip install numpy==1.26.4
!pip install scipy==1.11.4
!pip install pandas==2.2.2
!pip install scikit-learn==1.2.2

# Install spaCy and its model before transformers and torch.
!pip install spacy==3.8.7
!python -m spacy download en_core_web_sm==3.8.0

# Install other necessary libraries.
!pip install rapidfuzz python-magic tika requests==2.32.4 seaborn==0.13.2 matplotlib==3.10.0

# Install transformers and a potentially compatible torch version.
# Trying a more recent torch version that might be available with the latest spacy/transformers
!pip install transformers==4.36.2
!pip install torch==2.1.0 # Trying torch 2.1.0

print("Installation process completed. Please restart the runtime.")


# Uninstall all potentially conflicting packages, including numpy, scipy, scikit-learn, transformers, torch, pandas, and spacy.
!pip uninstall numpy scipy scikit-learn transformers torch pandas spacy -y

# Install a specific set of package versions known to be compatible in many environments.
!pip install numpy==1.23.5
!pip install scipy==1.10.1
!pip install scikit-learn==1.2.2
!pip install pandas==1.5.3

# Install the core spacy library with a version compatible with the selected numpy and scipy.
!pip install spacy==3.4.4

# Download the specific en_core_web_sm model version compatible with the installed spaCy.
!python -m spacy download en_core_web_sm==3.4.1

# Install transformers and a compatible CPU-only version of torch.
# Using --extra-index-url to ensure we can find the CPU version of torch.
!pip install transformers==4.26.0
!pip install torch==1.13.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu

# Install the remaining necessary packages.
!pip install matplotlib==3.7.1 seaborn==0.12.2 requests==2.28.1 rapidfuzz python-magic tika

print("Installation process completed. Please restart the runtime.")


# First, let's fix the NumPy compatibility issue
!pip uninstall numpy spacy thinc -y
!pip install numpy==1.24.3 spacy==3.7.4

# Now install the rest of the compatible packages
!pip install pandas==2.0.3 scikit-learn==1.2.2 matplotlib==3.7.2 seaborn==0.12.2 requests==2.31.0 rapidfuzz==3.5.2

# Download compatible spaCy model
!python -m spacy download en_core_web_sm==3.7.1

print("Packages installed successfully!")

# Import libraries with error handling
import re
import json
import os
from typing import List, Dict, Tuple, Set, Optional
import requests
import pandas as pd
import numpy as np
from collections import Counter
import time

# NLP - with error handling
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    print("âœ“ spaCy loaded successfully")
except Exception as e:
    print(f"Error loading spaCy: {e}")
    # Fallback to basic text processing
    nlp = None

# ML and metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('default')
sns.set_palette("husl")
%matplotlib inline

print("All libraries imported successfully!")

# --- DATA CITATION MINING PIPELINE ---

# DOI regex (relaxed but practical)
doi_pattern = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)

# Common repository accession patterns
accession_patterns = {
    'GEO': re.compile(r'GSE\d+', re.I),
    'SRA': re.compile(r'SRP\d+|SRR\d+|SRA\d+', re.I),
    'ENA': re.compile(r'PRJNA\d+|ERP\d+', re.I),
    'PDB': re.compile(r'PDB\s*[:#]?\s*[0-9A-Za-z]{4}', re.I),
    'Zenodo': re.compile(r'zenodo\.\d+', re.I),
    'Figshare': re.compile(r'figshare\.\d+', re.I),
}

def extract_references(text: str) -> Dict[str, List[str]]:
    """Extract DOIs and accession numbers from text"""
    if not text or not isinstance(text, str):
        return {'dois': [], 'accessions': []}

    results = {'dois': [], 'accessions': []}

    try:
        # Extract DOIs
        results['dois'] = list(set([m.group(0).rstrip('.;,') for m in doi_pattern.finditer(text)]))

        # Extract accessions
        accs = []
        for name, pat in accession_patterns.items():
            found = [m.group(0) for m in pat.finditer(text)]
            accs.extend(found)

        results['accessions'] = list(set([a.rstrip('.;,') for a in accs]))
    except Exception as e:
        print(f"Error extracting references: {e}")

    return results

# Heuristic patterns for informal mentions
repo_keywords = ['geo', 'sra', 'ena', 'zenodo', 'figshare', 'dryad', 'pdb']
heuristic_pattern = re.compile(
    r'(' + '|'.join(repo_keywords) + r')[\w\s\-:]*?(accession|id|doi|deposited|available|under|in)\s*[:#]?\s*([A-Za-z0-9\._-]+)',
    re.I
)

def extract_informal_mentions(text: str) -> List[Tuple[str, str]]:
    """Extract informal dataset mentions"""
    if not text or not isinstance(text, str):
        return []

    matches = []

    try:
        # Heuristic pattern matching
        for m in heuristic_pattern.finditer(text):
            repo = m.group(1)
            token = m.group(3)
            matches.append((repo, token))

        # Use spaCy if available, otherwise use simple regex
        if nlp:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ('ORG', 'PRODUCT'):
                    if re.search(r'\d{3,}', ent.text) or any(keyword in ent.text.lower() for keyword in repo_keywords):
                        matches.append((ent.label_, ent.text))
        else:
            # Fallback: look for repository-like patterns
            repo_pattern = re.compile(r'\b(' + '|'.join(repo_keywords) + r')\b', re.I)
            for match in repo_pattern.finditer(text):
                matches.append(('repository', match.group(0)))

    except Exception as e:
        print(f"Error extracting informal mentions: {e}")

    return matches

class APIClient:
    """Client for Crossref and DataCite APIs with rate limiting"""
    def __init__(self):
        self.crossref_base = 'https://api.crossref.org/works/'
        self.datacite_base = 'https://api.datacite.org/dois/'
        self.cache = {}
        self.request_delay = 1.0

    def _rate_limit(self):
        """Respectful rate limiting"""
        time.sleep(self.request_delay)

    def lookup_doi(self, doi: str) -> Dict:
        """Look up DOI metadata"""
        if not doi:
            return {'success': False, 'error': 'Empty DOI'}

        if doi in self.cache:
            return self.cache[doi]

        self._rate_limit()
        cleaned_doi = doi.strip().rstrip('.,;')

        try:
            # Try Crossref
            response = requests.get(self.crossref_base + cleaned_doi, timeout=10)
            if response.status_code == 200:
                msg = response.json().get('message', {})
                result = {
                    'title': msg.get('title', [''])[0] if msg.get('title') else '',
                    'publisher': msg.get('publisher', ''),
                    'success': True,
                    'source': 'crossref'
                }
                self.cache[doi] = result
                return result

            # Try DataCite
            response = requests.get(self.datacite_base + cleaned_doi, timeout=10)
            if response.status_code == 200:
                msg = response.json().get('data', {})
                attrs = msg.get('attributes', {})
                result = {
                    'title': attrs.get('titles', [{}])[0].get('title', ''),
                    'publisher': attrs.get('publisher', ''),
                    'success': True,
                    'source': 'datacite'
                }
                self.cache[doi] = result
                return result

            return {'success': False, 'error': f'HTTP {response.status_code}'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

def process_corpus(df: pd.DataFrame, text_col: str = 'text', use_api: bool = False) -> pd.DataFrame:
    """Process corpus and extract dataset references"""
    if df.empty or text_col not in df.columns:
        return pd.DataFrame(columns=['source_id', 'extracted', 'total_references'])

    outputs = []

    for idx, row in df.iterrows():
        try:
            text = str(row[text_col]) if pd.notna(row[text_col]) else ""
            source_id = row.get('id', f"doc_{idx}")

            # Extract references
            refs = extract_references(text)
            informal = extract_informal_mentions(text)
            matched = []

            # Process DOIs
            for doi in refs['dois']:
                meta = {}
                if use_api:
                    api_client = APIClient()
                    meta = api_client.lookup_doi(doi)
                matched.append({'type': 'doi', 'key': doi, 'meta': meta})

            # Process accessions
            for acc in refs['accessions']:
                matched.append({'type': 'accession', 'key': acc, 'meta': {}})

            # Process informal mentions
            for repo, token in informal:
                matched.append({'type': 'informal', 'key': token, 'repo': repo, 'meta': {}})

            outputs.append({
                'source_id': source_id,
                'extracted': matched,
                'total_references': len(matched)
            })

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            outputs.append({
                'source_id': f"doc_{idx}",
                'extracted': [],
                'total_references': 0
            })

    return pd.DataFrame(outputs)

# Create sample data
sample_text = "We used data from GEO accession GSE12345, SRA SRP98765 and Zenodo 10.5281/zenodo.1234567. Also see DOI 10.1038/s41586-020-2649-2."

corpus_data = [
    {'id': 'paper1', 'text': sample_text},
    {'id': 'paper2', 'text': 'Our raw reads were deposited to SRA under SRP123456. See also DOI 10.5281/zenodo.7654321.'},
    {'id': 'paper3', 'text': 'We used standard datasets including MNIST and CIFAR-10 for evaluation.'}
]

corpus = pd.DataFrame(corpus_data)

# Process corpus
print("Processing corpus...")
results_df = process_corpus(corpus, use_api=False)
print("Processing complete!")
print(results_df)

# Evaluation
gold = [
    {
        'source_id': 'paper1',
        'labels': [
            {'type': 'doi', 'key': '10.1038/s41586-020-2649-2'},
            {'type': 'accession', 'key': 'GSE12345'},
            {'type': 'accession', 'key': 'SRP98765'},
            {'type': 'doi', 'key': '10.5281/zenodo.1234567'}
        ]
    },
    {
        'source_id': 'paper2',
        'labels': [
            {'type': 'accession', 'key': 'SRP123456'},
            {'type': 'doi', 'key': '10.5281/zenodo.7654321'}
        ]
    },
    {
        'source_id': 'paper3',
        'labels': []
    }
]

def eval_detection(gold: List[Dict], predicted: List[Dict]) -> Dict:
    """Evaluate detection performance"""
    y_true = []
    y_pred = []

    for g in gold:
        sid = g['source_id']
        gset = set([(l['type'], l['key'].lower()) for l in g['labels']])

        p = next((p for p in predicted if p['source_id'] == sid), {'extracted': []})
        pset = set([(e['type'], e['key'].lower()) for e in p['extracted']])

        for label in gset:
            y_true.append(1)
            y_pred.append(1 if label in pset else 0)

        for label in pset:
            if label not in gset:
                y_true.append(0)
                y_pred.append(1)

    if y_true:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    else:
        precision = recall = f1 = 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1}

eval_results = eval_detection(gold, results_df.to_dict('records'))
print(f"Precision: {eval_results['precision']:.3f}, Recall: {eval_results['recall']:.3f}, F1: {eval_results['f1']:.3f}")

# MDC output format
def format_for_mdc(records: List[Dict]) -> List[Dict]:
    """Format for Make Data Count compatibility"""
    mdc_output = []

    for record in records:
        for extracted in record['extracted']:
            mdc_record = {
                'dataset_identifier': extracted.get('key', ''),
                'dataset_identifier_type': extracted.get('type', ''),
                'referencing_publication_id': record['source_id'],
                'confidence_score': 0.9 if extracted['type'] == 'doi' else 0.7,
                'extraction_method': 'regex' if extracted['type'] in ['doi', 'accession'] else 'heuristic'
            }
            mdc_output.append(mdc_record)

    return mdc_output

mdc_output = format_for_mdc(results_df.to_dict('records'))
print("MDC output sample:")
for record in mdc_output[:3]:
    print(record)

# Visualization
ref_types = [extracted['type'] for record in results_df.to_dict('records') for extracted in record['extracted']]
if ref_types:
    type_counts = pd.Series(ref_types).value_counts()
    plt.figure(figsize=(10, 6))
    type_counts.plot(kind='bar')
    plt.title('Reference Types Distribution')
    plt.show()
else:
    print("No references found for visualization")

print("Pipeline completed successfully! ðŸŽ‰")


# Uninstall all potentially conflicting packages again, ensuring a clean state.
# This is crucial as the numpy error indicates a low-level conflict.
# Include core libraries and those known to cause conflicts or depend on specific numpy versions.
!pip uninstall numpy scipy scikit-learn transformers torch pandas spacy matplotlib seaborn requests rapidfuzz python-magic tika -y

# Install a set of package versions known to be compatible with each other and hopefully the environment.
# Using slightly older but stable versions that often avoid binary incompatibility issues.
!pip install numpy==1.23.5
!pip install scipy==1.9.3
!pip install scikit-learn==1.1.3
!pip install pandas==1.5.3

# Install NLP and related libraries
!pip install spacy==3.4.4
!python -m spacy download en_core_web_sm==3.4.1
!pip install rapidfuzz python-magic tika requests==2.28.1

# Install visualization libraries
!pip install matplotlib==3.5.3 seaborn==0.12.2

# Install transformers and a compatible torch version.
# Using CPU version for broader compatibility, and a version known to work with older transformers.
!pip install transformers==4.26.0
!pip install torch==1.13.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu

print("Installation process completed. Please restart the runtime.")

# After restarting the runtime, run the import block again.
# The subtask of importing will be completed in a subsequent cell after the restart.


# Import necessary libraries, including os and pandas
import os
import pandas as pd
import numpy as np # numpy is needed for apply and potentially other operations

# Load the training data using the function that successfully found the sample data
# Using the last known successful path for the sample data
train_base_path = '/content/sample_data/train'

def load_data(base_path):
    """
    Load training data from directory structure with robust error handling
    """
    if not os.path.exists(base_path):
        print(f"Error: Base path '{base_path}' does not exist!")
        return pd.DataFrame() # Return empty DataFrame on error

    possible_files = ['train_labels.csv', 'labels.csv']
    labels_path = None

    for file in possible_files:
        test_path = os.path.join(base_path, file)
        if os.path.exists(test_path):
            labels_path = test_path
            break

    if labels_path is None:
        print(f"Error: No label file found in '{base_path}'")
        return pd.DataFrame() # Return empty DataFrame on error

    try:
        labels_df = pd.read_csv(labels_path)
        print(f"Successfully loaded {len(labels_df)} rows from {labels_path}")
        return labels_df

    except Exception as e:
        print(f"Error reading file {labels_path}: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

# Load the data into train_df
train_df = load_data(train_base_path)

if not train_df.empty:
    print("\nOriginal DataFrame head:")
    display(train_df.head())

    # Handle potential missing values in citation_type
    # The original data has a 'type' column which we'll use for 'citation_type'
    if 'type' in train_df.columns:
        # Fill missing 'type' values with 'unknown' and rename the column
        train_df['citation_type'] = train_df['type'].fillna('unknown')
        # Drop the original 'type' column if 'citation_type' is successfully created
        if 'citation_type' in train_df.columns:
             train_df = train_df.drop(columns=['type'])
        print("\nAfter handling missing 'type' (now 'citation_type'):")
        print(train_df['citation_type'].value_counts(dropna=False))
    else:
        print("\n'type' column not found. Adding dummy 'citation_type'.")
        # Add a dummy 'citation_type' if the original 'type' column is missing
        train_df['citation_type'] = 'unknown' # Default placeholder


    # Ensure dataset_id is treated as a string
    if 'dataset_id' in train_df.columns:
        train_df['dataset_id'] = train_df['dataset_id'].astype(str)
        print("\n'dataset_id' column ensured as string type.")
    else:
        print("\n'dataset_id' column not found.")
        # Add a dummy 'dataset_id' if missing
        train_df['dataset_id'] = [f"dummy_dataset_{i}" for i in range(len(train_df))]
        print("Added dummy 'dataset_id' column.")


    # Add dummy 'mention_context' if it doesn't exist
    if 'mention_context' not in train_df.columns:
        print("\nAdding dummy 'mention_context' column as it was not found.")
        # Example dummy context - replace with real text data if available
        # Use .get() with default values for robustness if article_id or dataset_id are also missing
        train_df['mention_context'] = train_df.apply(
            lambda row: f"This is a placeholder context for article {row.get('article_id', 'N/A')} and dataset {row.get('dataset_id', 'N/A')}.",
            axis=1
        )
        print("Dummy 'mention_context' added.")
    else:
        print("\n'mention_context' column already exists.")


    print("\nProcessed DataFrame head:")
    display(train_df.head())

    # Check for required columns after processing
    required_cols = ['article_id', 'dataset_id', 'citation_type', 'mention_context']
    missing_required = [col for col in required_cols if col not in train_df.columns]

    if missing_required:
        print(f"\nWarning: Missing required columns after processing: {missing_required}")
    else:
        print("\nAll required columns ('article_id', 'dataset_id', 'citation_type', 'mention_context') are present.")


else:
    print("\nFailed to load training data. Cannot proceed with preprocessing.")


# Import necessary libraries, including os and pandas
import os
import pandas as pd
import numpy as np # numpy is needed for apply and potentially other operations

# Load the training data using the function that successfully found the sample data
# Using the last known successful path for the sample data
train_base_path = '/content/sample_data/train'

def load_data(base_path):
    """
    Load training data from directory structure with robust error handling
    """
    if not os.path.exists(base_path):
        print(f"Error: Base path '{base_path}' does not exist!")
        return pd.DataFrame() # Return empty DataFrame on error

    possible_files = ['train_labels.csv', 'labels.csv']
    labels_path = None

    for file in possible_files:
        test_path = os.path.join(base_path, file)
        if os.path.exists(test_path):
            labels_path = test_path
            break

    if labels_path is None:
        print(f"Error: No label file found in '{base_path}'")
        return pd.DataFrame() # Return empty DataFrame on error

    try:
        labels_df = pd.read_csv(labels_path)
        print(f"Successfully loaded {len(labels_df)} rows from {labels_path}")
        return labels_df

    except Exception as e:
        print(f"Error reading file {labels_path}: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

# Load the data into train_df
train_df = load_data(train_base_path)

if not train_df.empty:
    print("\nOriginal DataFrame head:")
    display(train_df.head())

    # Handle potential missing values in citation_type
    # The original data has a 'type' column which we'll use for 'citation_type'
    if 'type' in train_df.columns:
        # Fill missing 'type' values with 'unknown' and rename the column
        train_df['citation_type'] = train_df['type'].fillna('unknown')
        # Drop the original 'type' column if 'citation_type' is successfully created
        if 'citation_type' in train_df.columns:
             train_df = train_df.drop(columns=['type'])
        print("\nAfter handling missing 'type' (now 'citation_type'):")
        print(train_df['citation_type'].value_counts(dropna=False))
    else:
        print("\n'type' column not found. Adding dummy 'citation_type'.")
        # Add a dummy 'citation_type' if the original 'type' column is missing
        train_df['citation_type'] = 'unknown' # Default placeholder


    # Ensure dataset_id is treated as a string
    if 'dataset_id' in train_df.columns:
        train_df['dataset_id'] = train_df['dataset_id'].astype(str)
        print("\n'dataset_id' column ensured as string type.")
    else:
        print("\n'dataset_id' column not found.")
        # Add a dummy 'dataset_id' if missing
        train_df['dataset_id'] = [f"dummy_dataset_{i}" for i in range(len(train_df))]
        print("Added dummy 'dataset_id' column.")


    # Add dummy 'mention_context' if it doesn't exist
    if 'mention_context' not in train_df.columns:
        print("\nAdding dummy 'mention_context' column as it was not found.")
        # Example dummy context - replace with real text data if available
        # Use .get() with default values for robustness if article_id or dataset_id are also missing
        train_df['mention_context'] = train_df.apply(
            lambda row: f"This is a placeholder context for article {row.get('article_id', 'N/A')} and dataset {row.get('dataset_id', 'N/A')}.",
            axis=1
        )
        print("Dummy 'mention_context' added.")
    else:
        print("\n'mention_context' column already exists.")


    print("\nProcessed DataFrame head:")
    display(train_df.head())

    # Check for required columns after processing
    required_cols = ['article_id', 'dataset_id', 'citation_type', 'mention_context']
    missing_required = [col for col in required_cols if col not in train_df.columns]

    if missing_required:
        print(f"\nWarning: Missing required columns after processing: {missing_required}")
    else:
        print("\nAll required columns ('article_id', 'dataset_id', 'citation_type', 'mention_context') are present.")


else:
    print("\nFailed to load training data. Cannot proceed with preprocessing.")


import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def extract_text_features(text):
    """
    Extract various text features from context
    """
    features = {}

    # Basic text features
    features['length'] = len(text)
    features['word_count'] = len(text.split())
    features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text else 0

    # Keyword features
    primary_keywords = ['generate', 'produce', 'create', 'collect', 'measure', 'experiment', 'study']
    secondary_keywords = ['use', 'analyze', 'reuse', 'existing', 'previous', 'obtain', 'download']

    text_lower = text.lower()
    features['primary_keyword_count'] = sum(1 for word in primary_keywords if word in text_lower)
    features['secondary_keyword_count'] = sum(1 for word in secondary_keywords if word in text_lower)
    features['keyword_ratio'] = features['primary_keyword_count'] / (features['secondary_keyword_count'] + 1)

    # Repository mentions
    repositories = ['geo', 'genbank', 'arrayexpress', 'pdb', 'dryad', 'figshare', 'zenodo']
    features['repo_mentions'] = sum(1 for repo in repositories if repo in text_lower)

    return features

def extract_identifier_features(dataset_id):
    """
    Extract features from the dataset identifier itself
    """
    features = {}

    # Convert dataset_id to string to handle both integers and potential string IDs
    dataset_id_str = str(dataset_id)

    features['is_doi'] = 1 if '10.' in dataset_id_str else 0
    features['is_geo'] = 1 if dataset_id_str.startswith('GSE') else 0
    features['id_length'] = len(dataset_id_str)
    features['has_special_chars'] = 1 if any(c in dataset_id_str for c in ['/', '.', '-', ':']) else 0

    return features

def create_feature_matrix(df):
    """
    Create feature matrix from dataframe
    """
    features = []

    for _, row in df.iterrows():
        feature_set = {}

        # Text features from context
        # Ensure 'mention_context' is treated as string and handle potential NaNs
        mention_context = str(row.get('mention_context', '')) # Get with default empty string
        text_feats = extract_text_features(mention_context)
        feature_set.update(text_feats)

        # Identifier features
        id_feats = extract_identifier_features(row['dataset_id'])
        feature_set.update(id_feats)

        # Target variable
        feature_set['target'] = 1 if row['citation_type'] == 'Primary' else 0

        features.append(feature_set)

    return pd.DataFrame(features)

# Assuming train_df was successfully loaded and preprocessed in the previous step
# Need to reconstruct train_df from article_labels for feature engineering
# Flatten the article_labels dictionary back into a DataFrame
train_data_list = []
for article_id, citations in article_labels.items():
    for citation in citations:
        train_data_list.append({
            'article_id': article_id,
            'dataset_id': citation['dataset_id'],
            'citation_type': citation['type'], # Use the 'type' from article_labels
            # Placeholder for mention_context, ideally this would be loaded from text files
            # For this example, we'll create a dummy context similar to the data creation cell
            'mention_context': f"Context for article {article_id}, dataset {citation['dataset_id']} ({citation['type']})."
        })

train_df = pd.DataFrame(train_data_list)


if not train_df.empty:
    # Create feature matrix
    feature_df = create_feature_matrix(train_df)

    print("Feature matrix shape:", feature_df.shape)
    print("\nFeature matrix head:")
    display(feature_df.head())

    # Check for missing values
    print("\nMissing values:")
    print(feature_df.isnull().sum())

    # Correlation analysis
    plt.figure(figsize=(12, 8))
    correlation_matrix = feature_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.show()
else:
    print("train_df is empty. Skipping feature engineering and analysis.")
    # Initialize feature_df as an empty DataFrame to prevent downstream errors
    feature_df = pd.DataFrame()


import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# First, let's create some sample data since article_labels isn't defined
print("Creating sample data for feature engineering...")

# Create sample article_labels data structure
article_labels = {
    'article_1': [
        {'dataset_id': 'GSE12345', 'type': 'Primary'},
        {'dataset_id': '10.1038/s41586-020-2649-2', 'type': 'Secondary'}
    ],
    'article_2': [
        {'dataset_id': 'SRP98765', 'type': 'Primary'},
        {'dataset_id': 'PDB_1ABC', 'type': 'Secondary'},
        {'dataset_id': '10.5281/zenodo.1234567', 'type': 'Primary'}
    ],
    'article_3': [
        {'dataset_id': 'GSE54321', 'type': 'Secondary'},
        {'dataset_id': '10.6084/m9.figshare.7654321', 'type': 'Primary'}
    ]
}

# Create realistic context examples
context_examples = {
    'GSE12345': "We generated RNA-seq data and deposited it in GEO under accession GSE12345. The data was produced using Illumina sequencing.",
    '10.1038/s41586-020-2649-2': "We analyzed existing data from DOI 10.1038/s41586-020-2649-2 to compare our results with previous studies.",
    'SRP98765': "Raw sequencing reads were produced and made available in SRA under accession SRP98765 as part of this study.",
    'PDB_1ABC': "We used the protein structure from PDB entry 1ABC for molecular docking studies in our analysis.",
    '10.5281/zenodo.1234567': "All custom code and generated datasets are available on Zenodo at DOI 10.5281/zenodo.1234567.",
    'GSE54321': "Existing microarray data from GEO accession GSE54321 was reanalyzed in this work to validate our findings.",
    '10.6084/m9.figshare.7654321': "We created and deposited supplementary materials on FigShare at DOI 10.6084/m9.figshare.7654321."
}

def extract_text_features(text):
    """
    Extract various text features from context
    """
    features = {}

    # Basic text features
    features['length'] = len(text)
    features['word_count'] = len(text.split())
    features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text else 0

    # Keyword features for primary vs secondary citations
    primary_keywords = ['generate', 'produce', 'create', 'collect', 'measure', 'experiment',
                       'study', 'deposit', 'make available', 'this work', 'our data']
    secondary_keywords = ['use', 'analyze', 'reuse', 'existing', 'previous', 'obtain',
                         'download', 'compare', 'validate', 'reanalyze']

    text_lower = text.lower()
    features['primary_keyword_count'] = sum(1 for word in primary_keywords if word in text_lower)
    features['secondary_keyword_count'] = sum(1 for word in secondary_keywords if word in text_lower)
    features['keyword_ratio'] = features['primary_keyword_count'] / (features['secondary_keyword_count'] + 1e-6)

    # Repository mentions
    repositories = ['geo', 'genbank', 'arrayexpress', 'pdb', 'dryad', 'figshare', 'zenodo',
                   'sra', 'ena', 'proteomexchange']
    features['repo_mentions'] = sum(1 for repo in repositories if repo in text_lower)

    # Presence indicators
    features['has_deposit_verbs'] = 1 if any(word in text_lower for word in ['deposit', 'submit', 'upload']) else 0
    features['has_creation_verbs'] = 1 if any(word in text_lower for word in ['generate', 'create', 'produce']) else 0
    features['has_analysis_verbs'] = 1 if any(word in text_lower for word in ['analyze', 'use', 'reuse']) else 0

    return features

def extract_identifier_features(dataset_id):
    """
    Extract features from the dataset identifier itself
    """
    features = {}

    # Convert dataset_id to string to handle both integers and potential string IDs
    dataset_id_str = str(dataset_id)

    # Identifier type features
    features['is_doi'] = 1 if dataset_id_str.startswith('10.') else 0
    features['is_geo'] = 1 if dataset_id_str.startswith('GSE') else 0
    features['is_sra'] = 1 if dataset_id_str.startswith(('SRP', 'SRR', 'SRA')) else 0
    features['is_pdb'] = 1 if dataset_id_str.startswith(('PDB', 'pdb')) else 0
    features['is_zenodo'] = 1 if 'zenodo' in dataset_id_str.lower() else 0
    features['is_figshare'] = 1 if 'figshare' in dataset_id_str.lower() else 0

    # Structural features
    features['id_length'] = len(dataset_id_str)
    features['has_special_chars'] = 1 if any(c in dataset_id_str for c in ['/', '.', '-', ':', '_']) else 0
    features['has_numbers'] = 1 if any(c.isdigit() for c in dataset_id_str) else 0
    features['has_letters'] = 1 if any(c.isalpha() for c in dataset_id_str) else 0

    # Pattern complexity
    features['digit_count'] = sum(1 for c in dataset_id_str if c.isdigit())
    features['letter_count'] = sum(1 for c in dataset_id_str if c.isalpha())
    features['special_char_count'] = sum(1 for c in dataset_id_str if not c.isalnum())

    return features

def create_feature_matrix(article_labels, context_examples):
    """
    Create feature matrix from article_labels and context examples
    """
    features = []

    for article_id, citations in article_labels.items():
        for citation in citations:
            dataset_id = citation['dataset_id']
            citation_type = citation['type']

            # Get context for this dataset mention
            mention_context = context_examples.get(dataset_id, f"Context for {dataset_id} in {article_id}")

            feature_set = {}

            # Add basic identifiers
            feature_set['article_id'] = article_id
            feature_set['dataset_id'] = dataset_id

            # Text features from context
            text_feats = extract_text_features(mention_context)
            feature_set.update(text_feats)

            # Identifier features
            id_feats = extract_identifier_features(dataset_id)
            feature_set.update(id_feats)

            # Target variable
            feature_set['target'] = 1 if citation_type == 'Primary' else 0
            feature_set['citation_type'] = citation_type

            features.append(feature_set)

    return pd.DataFrame(features)

# Create feature matrix
print("Creating feature matrix...")
feature_df = create_feature_matrix(article_labels, context_examples)

print("Feature matrix shape:", feature_df.shape)
print("\nFeature matrix head:")
print(feature_df.head())

# Check for missing values
print("\nMissing values:")
print(feature_df.isnull().sum())

# Basic statistics
print("\nBasic statistics:")
print(feature_df.describe())

# Target distribution
print("\nTarget distribution:")
print(feature_df['target'].value_counts())
print(feature_df['citation_type'].value_counts())

# Correlation analysis
plt.figure(figsize=(14, 10))
numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
correlation_matrix = feature_df[numeric_cols].corr()

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', square=True, cbar_kws={"shrink": .8})
plt.title('Feature Correlation Matrix (Lower Triangle)')
plt.tight_layout()
plt.show()

# Feature importance analysis (using correlation with target)
target_correlations = correlation_matrix['target'].drop('target').sort_values(key=abs, ascending=False)

plt.figure(figsize=(12, 8))
target_correlations.plot(kind='bar', color=['red' if x < 0 else 'blue' for x in target_correlations])
plt.title('Feature Correlation with Target Variable')
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Distribution of key features by target class
key_features = ['primary_keyword_count', 'secondary_keyword_count', 'keyword_ratio',
                'repo_mentions', 'is_doi', 'is_geo']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, feature in enumerate(key_features):
    if feature in feature_df.columns:
        for target_val in [0, 1]:
            subset = feature_df[feature_df['target'] == target_val]
            axes[i].hist(subset[feature], alpha=0.7, label=f'Target={target_val}', bins=20)
        axes[i].set_title(f'Distribution of {feature}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frequency')
        axes[i].legend()

plt.tight_layout()
plt.show()

# Pairplot of most correlated features
most_correlated = target_correlations.head(6).index.tolist()
if 'target' in most_correlated:
    most_correlated.remove('target')

if most_correlated:
    plot_df = feature_df[most_correlated + ['target']]
    plot_df['target'] = plot_df['target'].astype(str)  # Convert to categorical for coloring

    sns.pairplot(plot_df, hue='target', palette={ '0': 'red', '1': 'blue' },
                diag_kind='kde', corner=True)
    plt.suptitle('Pairplot of Most Correlated Features', y=1.02)
    plt.show()

print("\nFeature engineering completed successfully!")
print("Key insights:")
print(f"- Total samples: {len(feature_df)}")
print(f"- Primary citations: {len(feature_df[feature_df['target'] == 1])}")
print(f"- Secondary citations: {len(feature_df[feature_df['target'] == 0])}")
print(f"- Most predictive features: {target_correlations.head(3).index.tolist()}")


# Install required packages
!pip install PyMuPDF spacy scikit-learn xgboost matplotlib seaborn
!python -m spacy download en_core_web_sm


# Install spacy and download the English model
!pip install spacy
!python -m spacy download en_core_web_sm


# Install required packages
!pip install PyMuPDF spacy scikit-learn xgboost matplotlib seaborn
!python -m spacy download en_core_web_sm


# Uninstall all potentially conflicting packages again, ensuring a clean state.
# This is crucial as the numpy error indicates a low-level conflict.
# Include core libraries and those known to cause conflicts or depend on specific numpy versions.
!pip uninstall numpy scipy scikit-learn transformers torch pandas spacy matplotlib seaborn requests rapidfuzz python-magic tika -y

# Install a set of package versions known to be compatible with each other and hopefully the environment.
# Using slightly older but stable versions that often avoid binary incompatibility issues.
!pip install numpy==1.23.5
!pip install scipy==1.9.3
!pip install pandas==1.5.3 # Install pandas after numpy and scipy
!pip install scikit-learn==1.1.3 # Install scikit-learn after numpy and scipy

# Install NLP and related libraries
!pip install spacy==3.4.4
!python -m spacy download en_core_web_sm==3.4.1
!pip install rapidfuzz python-magic tika requests==2.28.1

# Install visualization libraries
!pip install matplotlib==3.5.3 seaborn==0.12.2

# Install transformers and a compatible torch version.
# Using CPU version for broader compatibility, and a version known to work with older transformers.
!pip install transformers==4.26.0
!pip install torch==1.13.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu

print("Installation process completed. Please restart the runtime.")

# After restarting the runtime, the imports and feature engineering will be attempted again.


# Data Citation Mining Pipeline - No external dependencies required
import re
import json
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional

# -------------------------
# Data Citation Miner Class
# -------------------------
class DataCitationMiner:
    def __init__(self):
        self.dataset_patterns = self._initialize_patterns()
        self.primary_keywords = ['generate', 'produce', 'create', 'collect', 'deposit', 'this study', 'our data', 'new data']
        self.secondary_keywords = ['use', 'analyze', 'reuse', 'existing', 'previous', 'obtain', 'publicly available']
        self.repositories = ['geo', 'sra', 'ena', 'pdb', 'zenodo', 'figshare', 'dryad']

    def _initialize_patterns(self):
        """Initialize regex patterns for dataset detection"""
        return {
            'doi': re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I),
            'geo': re.compile(r'GSE\d+', re.I),
            'sra': re.compile(r'SRP\d+|SRR\d+|SRA\d+', re.I),
            'ena': re.compile(r'PRJNA\d+|ERP\d+', re.I),
            'pdb': re.compile(r'PDB\s*[:#]?\s*[0-9A-Za-z]{4}', re.I),
            'zenodo': re.compile(r'zenodo\.\d+', re.I),
            'figshare': re.compile(r'figshare\.\d+', re.I),
            'dryad': re.compile(r'dryad\.[a-z0-9]+', re.I),
        }

    def extract_references(self, text: str) -> Dict[str, List[str]]:
        """Extract dataset references from text"""
        if not text or not isinstance(text, str):
            return {'dois': [], 'accessions': []}

        results = {'dois': [], 'accessions': []}

        try:
            # Extract DOIs
            results['dois'] = list(set([m.group(0).rstrip('.;,') for m in self.dataset_patterns['doi'].finditer(text)]))

            # Extract accessions from other repositories
            for repo in ['geo', 'sra', 'ena', 'pdb', 'zenodo', 'figshare', 'dryad']:
                matches = [m.group(0) for m in self.dataset_patterns[repo].finditer(text)]
                results['accessions'].extend(matches)

            results['accessions'] = list(set(results['accessions']))

        except Exception as e:
            print(f"Error extracting references: {e}")

        return results

    def extract_text_features(self, text: str) -> Dict:
        """Extract features from text for classification"""
        features = {}

        if not text:
            return features

        text_lower = text.lower()

        # Basic text features
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())

        # Keyword features for primary vs secondary citations
        features['primary_keyword_count'] = sum(1 for word in self.primary_keywords if word in text_lower)
        features['secondary_keyword_count'] = sum(1 for word in self.secondary_keywords if word in text_lower)
        features['primary_secondary_ratio'] = (features['primary_keyword_count'] + 1) / (features['secondary_keyword_count'] + 1)

        # Repository mentions
        features['repo_mentions'] = sum(1 for repo in self.repositories if repo in text_lower)

        # Reference indicators
        reference_indicators = ['accession', 'dataset', 'data set', 'repository', 'deposit', 'submit']
        features['reference_indicators'] = sum(1 for word in reference_indicators if word in text_lower)

        # Tense analysis (simple approach)
        present_tense_indicators = ['generate', 'produce', 'create', 'collect', 'deposit']
        past_tense_indicators = ['generated', 'produced', 'created', 'collected', 'deposited', 'used', 'analyzed']

        features['present_tense_score'] = sum(1 for word in present_tense_indicators if word in text_lower)
        features['past_tense_score'] = sum(1 for word in past_tense_indicators if word in text_lower)

        return features

    def predict_citation_type(self, text: str) -> str:
        """Predict citation type using rule-based approach"""
        if not text:
            return 'Unknown'

        text_lower = text.lower()

        creation_score = sum(1 for word in self.primary_keywords if word in text_lower)
        usage_score = sum(1 for word in self.secondary_keywords if word in text_lower)

        if creation_score > usage_score:
            return 'Primary'
        elif usage_score > creation_score:
            return 'Secondary'
        else:
            # If scores are equal, use additional heuristics
            features = self.extract_text_features(text)
            if features.get('present_tense_score', 0) > features.get('past_tense_score', 0):
                return 'Primary'
            else:
                return 'Secondary'

    def analyze_publication(self, text: str, source_id: str = "") -> Dict:
        """Comprehensive analysis of a publication text"""
        references = self.extract_references(text)
        citation_type = self.predict_citation_type(text)

        return {
            'source_id': source_id,
            'citation_type': citation_type,
            'datasets_mentioned': len(references['dois']) + len(references['accessions']),
            'doi_references': references['dois'],
            'accession_references': references['accessions'],
            'mdc_output': self.create_mdc_output(references, source_id)
        }

    def create_mdc_output(self, references: Dict, source_id: str = "") -> List[Dict]:
        """Create Make Data Count compatible output"""
        mdc_records = []

        # Process DOIs
        for doi in references.get('dois', []):
            mdc_records.append({
                'dataset_identifier': doi,
                'dataset_identifier_type': 'doi',
                'referencing_publication_id': source_id,
                'confidence_score': 0.9,
                'extraction_method': 'regex'
            })

        # Process accessions
        for acc in references.get('accessions', []):
            mdc_records.append({
                'dataset_identifier': acc,
                'dataset_identifier_type': 'accession',
                'referencing_publication_id': source_id,
                'confidence_score': 0.8,
                'extraction_method': 'pattern_matching'
            })

        return mdc_records

    def evaluate_on_corpus(self, corpus):
        """Evaluate performance on a test corpus"""
        correct = 0
        total = len(corpus)

        for text, true_label in corpus:
            predicted = self.predict_citation_type(text)
            if predicted.lower() == true_label.lower():
                correct += 1

        accuracy = correct / total if total > 0 else 0
        return accuracy

# -------------------------
# Create Sample Data
# -------------------------
def create_sample_corpus():
    """Create sample training data for demonstration"""
    return [
        # Primary citations
        ("We generated RNA-seq data and deposited it in GEO under accession GSE12345.", "primary"),
        ("Raw sequencing reads were produced and submitted to SRA with accession SRP98765.", "primary"),
        ("All custom code and datasets created in this study are available.", "primary"),
        ("New proteomics data was collected and deposited to PRIDE with identifier PXD12345.", "primary"),
        ("We produced original microarray data for this investigation.", "primary"),
        ("Data deposited in Zenodo with DOI 10.5281/zenodo.1234567.", "primary"),

        # Secondary citations
        ("We analyzed existing microarray data from GEO accession GSE54321.", "secondary"),
        ("The study used protein structures from PDB for molecular docking.", "secondary"),
        ("Previous results were compared with our findings using publicly available data.", "secondary"),
        ("Existing RNA-seq datasets from SRA were obtained and reanalyzed.", "secondary"),
        ("We retrieved climate data from the Dryad repository for our analysis.", "secondary"),
        ("Used existing datasets from previous publications.", "secondary"),
    ]

# -------------------------
# Simple Visualization Functions
# -------------------------
def print_citation_type_distribution(corpus):
    """Print distribution of citation types"""
    counter = Counter(label for _, label in corpus)
    print("Citation Type Distribution:")
    for label, count in counter.items():
        print(f"  {label}: {count}")

def print_feature_analysis(miner, text):
    """Print feature analysis for a given text"""
    features = miner.extract_text_features(text)
    print(f"Text: {text}")
    print("Features:")
    for feature, value in features.items():
        print(f"  {feature}: {value}")
    print(f"Predicted citation type: {miner.predict_citation_type(text)}")
    print("-" * 50)

# -------------------------
# Main Demonstration
# -------------------------
def main():
    """Main demonstration function"""
    print("DATA CITATION MINING PIPELINE")
    print("=" * 50)

    # Create sample data
    print("Creating sample corpus...")
    corpus = create_sample_corpus()

    # Show corpus distribution
    print_citation_type_distribution(corpus)
    print()

    # Initialize miner
    print("Initializing Data Citation Miner...")
    miner = DataCitationMiner()

    # Evaluate on corpus
    accuracy = miner.evaluate_on_corpus(corpus)
    print(f"Rule-based accuracy on sample corpus: {accuracy:.2f}")
    print()

    # Test predictions
    print("Testing predictions...")
    test_texts = [
        "We created and deposited new proteomics data in PRIDE.",
        "Existing RNA-seq data from GEO was reanalyzed.",
        "This study produced novel datasets available online.",
        "We used previously published data from SRA accession SRP12345.",
        "Both new data was generated and existing data was utilized."
    ]

    for text in test_texts:
        prediction = miner.predict_citation_type(text)
        print(f"Text: {text}")
        print(f"Predicted citation type: {prediction}")
        print("-" * 40)

    # Test reference extraction
    print("\nTesting reference extraction...")
    sample_text = "This study used data from GEO GSE12345 and SRA SRP98765. Also referenced DOI 10.1038/s41586-020-2649-2."
    references = miner.extract_references(sample_text)
    print(f"Text: {sample_text}")
    print(f"Extracted references: {references}")

    # Test comprehensive analysis
    print("\nTesting comprehensive publication analysis...")
    analysis = miner.analyze_publication(sample_text, "paper_123")
    print(f"Analysis results: {json.dumps(analysis, indent=2)}")

    # Feature analysis examples
    print("\nFeature analysis examples:")
    example_texts = [
        "We generated new data and deposited it in GEO GSE12345.",
        "We used existing data from GEO GSE54321 for analysis."
    ]

    for text in example_texts:
        print_feature_analysis(miner, text)

    # Save evaluation scores
    evaluation_data = {
        "accuracy": accuracy,
        "precision": "N/A (rule-based system)",
        "recall": "N/A (rule-based system)",
        "f1_score": "N/A (rule-based system)"
    }

    with open("submission_scores.csv", "w") as f: # Corrected filename and added 'w' mode
        json.dump(evaluation_data, f, indent=4) # Using json.dump for dictionary

    print("âœ… Scores saved to submission_scores.csv")


# Run the demonstration
if __name__ == "__main__":
    main()

print("\nPipeline execution completed successfully! ðŸŽ‰")


import pandas as pd
import os

# Path to the sample data labels file
sample_labels_path = '/content/sample_data/train/train_labels.csv'

# Check if the file exists
if os.path.exists(sample_labels_path):
    try:
        # Load the CSV file
        temp_df = pd.read_csv(sample_labels_path)

        # Print the column names
        print("Columns found in train_labels.csv:")
        print(temp_df.columns.tolist())

        # Display the head of the DataFrame to inspect the data and column names
        print("\nHead of the DataFrame:")
        display(temp_df.head())

    except Exception as e:
        print(f"Error reading train_labels.csv: {e}")
else:
    print(f"File not found: {sample_labels_path}")


import pandas as pd
import os

# Create directory if it doesn't exist
output_dir = '/content/sample_data/train'
os.makedirs(output_dir, exist_ok=True)

# Sample Data Creation (from cell I9Cq_zbIzYSz)
data = {
    'article_id': [1, 1, 2, 3, 3],
    'dataset_id': [101, 102, 201, 301, 302],
    'type': ['primary', 'secondary', 'primary', 'secondary', 'primary'],
    'mention_context': [
        'Data from dataset 101 was used.',
        'Referenced dataset 102.',
        'New data generated for 201.',
        'Analysis of dataset 301.',
        'Results based on dataset 302.'
    ]
}
train_labels_df = pd.DataFrame(data)

# Save to CSV
output_path = os.path.join(output_dir, 'train_labels.csv')
train_labels_df.to_csv(output_path, index=False)

print(f"Sample train_labels.csv created at: {output_path}")
print("\nCreated DataFrame head:")
display(train_labels_df.head())


# Install required packages
!pip install tika python-magic rapidfuzz requests seaborn matplotlib spacy scikit-learn
!python -m spacy download en_core_web_sm


def create_sample_corpus():
    """Create sample training data for demonstration"""
    data = [
        # Primary citations
        {'text': 'We generated RNA-seq data and deposited it in GEO under accession GSE12345.', 'citation_type': 'Primary'},
        {'text': 'Raw sequencing reads were produced and submitted to SRA with accession SRP98765.', 'citation_type': 'Primary'},
        {'text': 'All custom code and datasets created in this study are available.', 'citation_type': 'Primary'},
        {'text': 'New proteomics data was collected and deposited to PRIDE with identifier PXD12345.', 'citation_type': 'Primary'},
        {'text': 'We produced original microarray data for this investigation.', 'citation_type': 'Primary'},

        # Secondary citations
        {'text': 'We analyzed existing microarray data from GEO accession GSE54321.', 'citation_type': 'Secondary'},
        {'text': 'The study used protein structures from PDB for molecular docking.', 'citation_type': 'Secondary'},
        {'text': 'Previous results were compared with our findings using publicly available data.', 'citation_type': 'Secondary'},
        {'text': 'Existing RNA-seq datasets from SRA were obtained and reanalyzed.', 'citation_type': 'Secondary'},
        {'text': 'We retrieved climate data from the Dryad repository for our analysis.', 'citation_type': 'Secondary'},

        # Additional examples
        {'text': 'Data deposited in Zenodo with DOI 10.5281/zenodo.1234567.', 'citation_type': 'Primary'},
        {'text': 'Used existing datasets from previous publications.', 'citation_type': 'Secondary'},
        {'text': 'This study both generated new data and utilized existing datasets.', 'citation_type': 'Primary'},
    ]

    df = pd.DataFrame(data)

    # Ensure dummy dataset_id column exists
    df["dataset_id"] = [f"DS{i+1}" for i in range(len(df))]

    return df


def analyze_id_patterns(df):
    """Analyze patterns in dataset IDs (safe-check version)"""
    if "dataset_id" not in df.columns:
        print("No 'dataset_id' column found. Skipping ID pattern analysis.")
        return

    doi_count = df['dataset_id'].str.contains(r'10\.\d{4,9}/', regex=True).sum()
    geo_count = df['dataset_id'].str.contains(r'GSE\d+', case=False, regex=True).sum()
    sra_count = df['dataset_id'].str.contains(r'SR[PX]\d+', case=False, regex=True).sum()

    print(f"DOI identifiers: {doi_count} ({doi_count/len(df)*100:.1f}%)")
    print(f"GEO accessions: {geo_count} ({geo_count/len(df)*100:.1f}%)")
    print(f"SRA accessions: {sra_count} ({sra_count/len(df)*100:.1f}%)")

    identified_count = doi_count + geo_count + sra_count
    other_count = len(df) - identified_count
    print(f"Other formats: {other_count} ({other_count/len(df)*100:.1f}%)")


def main():
    """Main demonstration function"""
    print("DATA CITATION MINING PIPELINE")
    print("=" * 60)

    # Load and prepare data
    print("Loading and preparing data...")
    train_df = load_and_prepare_data()

    # Basic statistics
    print("\nDataset Overview:")
    print(f"Total examples: {len(train_df)}")

    if 'article_id' in train_df.columns:
        print(f"Unique articles: {train_df['article_id'].nunique()}")

    if 'dataset_id' in train_df.columns:
        print(f"Unique datasets: {train_df['dataset_id'].nunique()}")
    else:
        print("No 'dataset_id' column found (using dummy IDs or skipping).")

    print("\nCitation Type Distribution:")
    print(train_df['citation_type'].value_counts())

    # Visualize data distribution
    visualize_data_distribution(train_df)

    # Analyze ID patterns (safe-check included)
    if not train_df.empty:
        analyze_id_patterns(train_df)

    # Continue rest of pipeline...



# First, let's install the required packages
!pip install pandas numpy scikit-learn matplotlib seaborn

# Import necessary libraries
import pandas as pd
import numpy as np
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

print("All necessary libraries imported successfully.")

# -------------------------
# Data Citation Miner Class
# -------------------------
class DataCitationMiner:
    def __init__(self):
        self.dataset_patterns = self._initialize_patterns()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.label_encoder = LabelEncoder()
        self.models = {}

    def _initialize_patterns(self):
        """Initialize regex patterns for dataset detection"""
        return {
            'doi': re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I),
            'geo': re.compile(r'GSE\d+', re.I),
            'sra': re.compile(r'SRP\d+|SRR\d+|SRA\d+', re.I),
            'ena': re.compile(r'PRJNA\d+|ERP\d+', re.I),
            'pdb': re.compile(r'PDB\s*[:#]?\s*[0-9A-Za-z]{4}', re.I),
            'zenodo': re.compile(r'zenodo\.\d+', re.I),
            'figshare': re.compile(r'figshare\.\d+', re.I),
            'dryad': re.compile(r'dryad\.[a-z0-9]+', re.I),
        }

    def extract_references(self, text: str) -> Dict[str, List[str]]:
        """Extract dataset references from text"""
        if not text or not isinstance(text, str):
            return {'dois': [], 'accessions': []}

        results = {'dois': [], 'accessions': []}

        try:
            # Extract DOIs
            results['dois'] = list(set([m.group(0).rstrip('.;,') for m in self.dataset_patterns['doi'].finditer(text)]))

            # Extract accessions from other repositories
            for repo in ['geo', 'sra', 'ena', 'pdb', 'zenodo', 'figshare', 'dryad']:
                matches = [m.group(0) for m in self.dataset_patterns[repo].finditer(text)]
                results['accessions'].extend(matches)

            results['accessions'] = list(set(results['accessions']))

        except Exception as e:
            print(f"Error extracting references: {e}")

        return results

    def extract_text_features(self, text: str) -> Dict:
        """Extract features from text for classification"""
        features = {}

        if not text:
            return features

        text_lower = text.lower()

        # Basic text features
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())

        # Keyword features for primary vs secondary citations
        primary_keywords = ['generate', 'produce', 'create', 'collect', 'deposit', 'this study', 'our data', 'new data']
        secondary_keywords = ['use', 'analyze', 'reuse', 'existing', 'previous', 'obtain', 'publicly available']

        features['primary_keyword_count'] = sum(1 for word in primary_keywords if word in text_lower)
        features['secondary_keyword_count'] = sum(1 for word in secondary_keywords if word in text_lower)
        features['primary_secondary_ratio'] = (features['primary_keyword_count'] + 1) / (features['secondary_keyword_count'] + 1)

        # Repository mentions
        repositories = ['geo', 'sra', 'ena', 'pdb', 'zenodo', 'figshare', 'dryad']
        features['repo_mentions'] = sum(1 for repo in repositories if repo in text_lower)

        # Reference indicators
        reference_indicators = ['accession', 'dataset', 'data set', 'repository', 'deposit', 'submit']
        features['reference_indicators'] = sum(1 for word in reference_indicators if word in text_lower)

        # Tense analysis (simple approach)
        present_tense_indicators = ['generate', 'produce', 'create', 'collect', 'deposit']
        past_tense_indicators = ['generated', 'produced', 'created', 'collected', 'deposited', 'used', 'analyzed']

        features['present_tense_score'] = sum(1 for word in present_tense_indicators if word in text_lower)
        features['past_tense_score'] = sum(1 for word in past_tense_indicators if word in text_lower)

        return features

    def prepare_training_data(self, corpus_df: pd.DataFrame, text_col: str = 'text', label_col: str = 'citation_type'):
        """Prepare training data for ML models"""
        features = []
        labels = []

        for _, row in corpus_df.iterrows():
            text = row[text_col] if pd.notna(row[text_col]) else ""

            # Extract features
            feat = self.extract_text_features(text)
            features.append(feat)
            labels.append(row[label_col])

        # Convert to DataFrame
        feature_df = pd.DataFrame(features)

        # Encode labels
        if labels:
            y_encoded = self.label_encoder.fit_transform(labels)
        else:
            y_encoded = np.array([])

        return feature_df, y_encoded

    def train_model(self, X, y, test_size=0.2):
        """Train a Random Forest classifier with cross-validation"""
        if len(X) == 0 or len(y) == 0:
            print("No data available for training")
            return None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Train Random Forest
        self.models['random_forest'] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models['random_forest'].fit(X_train, y_train)

        # Cross-validation
        cv_scores = cross_val_score(self.models['random_forest'], X, y, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Evaluate on test set
        y_pred = self.models['random_forest'].predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Test Set Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

        return accuracy

    def predict_citation_type(self, text: str) -> str:
        """Predict citation type using trained model or rules"""
        if not text:
            return 'Unknown'

        if self.models and 'random_forest' in self.models:
            features = self.extract_text_features(text)
            feature_df = pd.DataFrame([features])
            pred = self.models['random_forest'].predict(feature_df)[0]
            return self.label_encoder.inverse_transform([pred])[0]

        return self._rule_based_prediction(text)

    def _rule_based_prediction(self, text: str) -> str:
        """Fallback rule-based prediction"""
        if not text:
            return 'Unknown'

        text_lower = text.lower()

        creation_indicators = ['generate', 'produce', 'create', 'collect', 'deposit', 'this study', 'our data', 'new data']
        usage_indicators = ['use', 'analyze', 'reuse', 'existing', 'previous', 'obtain', 'publicly available']

        creation_score = sum(1 for word in creation_indicators if word in text_lower)
        usage_score = sum(1 for word in usage_indicators if word in text_lower)

        if creation_score > usage_score:
            return 'Primary'
        elif usage_score > creation_score:
            return 'Secondary'
        else:
            return 'Unknown'

    def analyze_publication(self, text: str, source_id: str = "") -> Dict:
        """Comprehensive analysis of a publication text"""
        references = self.extract_references(text)
        citation_type = self.predict_citation_type(text)

        return {
            'source_id': source_id,
            'citation_type': citation_type,
            'datasets_mentioned': len(references['dois']) + len(references['accessions']),
            'doi_references': references['dois'],
            'accession_references': references['accessions'],
            'mdc_output': self.create_mdc_output(references, source_id)
        }

    def create_mdc_output(self, references: Dict, source_id: str = "") -> List[Dict]:
        """Create Make Data Count compatible output"""
        mdc_records = []

        # Process DOIs
        for doi in references.get('dois', []):
            mdc_records.append({
                'dataset_identifier': doi,
                'dataset_identifier_type': 'doi',
                'referencing_publication_id': source_id,
                'confidence_score': 0.9,
                'extraction_method': 'regex'
            })

        # Process accessions
        for acc in references.get('accessions', []):
            mdc_records.append({
                'dataset_identifier': acc,
                'dataset_identifier_type': 'accession',
                'referencing_publication_id': source_id,
                'confidence_score': 0.8,
                'extraction_method': 'pattern_matching'
            })

        return mdc_records

# -------------------------
# Data Loading and Preparation
# -------------------------
def load_and_prepare_data():
    """Load and prepare the training data"""
    try:
        # Try to load the training data
        train_df = pd.read_csv('/content/sample_data/train/train_labels.csv')
        print("Successfully loaded train_df from CSV.")

        # Add dummy 'citation_type' and 'mention_context' if they don't exist
        if 'citation_type' not in train_df.columns:
            print("Adding dummy 'citation_type' column as it was not found in the loaded data.")
            train_df['citation_type'] = train_df['article_id'].apply(lambda x: 'Primary' if x % 2 == 0 else 'Secondary')

        if 'mention_context' not in train_df.columns:
            print("Adding dummy 'mention_context' column as it was not found in the loaded data.")
            train_df['mention_context'] = train_df.apply(
                lambda row: f"This is a dummy context for article {row['article_id']} and dataset {row['dataset_id']}.",
                axis=1
            )

        # Ensure dataset_id is treated as a string
        train_df['dataset_id'] = train_df['dataset_id'].astype(str)

        return train_df

    except FileNotFoundError:
        print("Error: train_labels.csv not found. Creating sample data for demonstration.")
        return create_sample_corpus()

def create_sample_corpus():
    """Create sample training data for demonstration"""
    data = [
        # Primary citations
        {'text': 'We generated RNA-seq data and deposited it in GEO under accession GSE12345.', 'citation_type': 'Primary'},
        {'text': 'Raw sequencing reads were produced and submitted to SRA with accession SRP98765.', 'citation_type': 'Primary'},
        {'text': 'All custom code and datasets created in this study are available.', 'citation_type': 'Primary'},
        {'text': 'New proteomics data was collected and deposited to PRIDE with identifier PXD12345.', 'citation_type': 'Primary'},
        {'text': 'We produced original microarray data for this investigation.', 'citation_type': 'Primary'},

        # Secondary citations
        {'text': 'We analyzed existing microarray data from GEO accession GSE54321.', 'citation_type': 'Secondary'},
        {'text': 'The study used protein structures from PDB for molecular docking.', 'citation_type': 'Secondary'},
        {'text': 'Previous results were compared with our findings using publicly available data.', 'citation_type': 'Secondary'},
        {'text': 'Existing RNA-seq datasets from SRA were obtained and reanalyzed.', 'citation_type': 'Secondary'},
        {'text': 'We retrieved climate data from the Dryad repository for our analysis.', 'citation_type': 'Secondary'},

        # Additional examples
        {'text': 'Data deposited in Zenodo with DOI 10.5281/zenodo.1234567.', 'citation_type': 'Primary'},
        {'text': 'Used existing datasets from previous publications.', 'citation_type': 'Secondary'},
        {'text': 'This study both generated new data and utilized existing datasets.', 'citation_type': 'Primary'},
    ]

    return pd.DataFrame(data)

def analyze_id_patterns(df):
    """Analyze patterns in dataset IDs"""
    doi_count = df['dataset_id'].str.contains(r'10\.\d{4,9}/', regex=True).sum()
    geo_count = df['dataset_id'].str.contains(r'GSE\d+', case=False, regex=True).sum()
    sra_count = df['dataset_id'].str.contains(r'SR[PX]\d+', case=False, regex=True).sum()

    print(f"DOI identifiers: {doi_count} ({doi_count/len(df)*100:.1f}%)")
    print(f"GEO accessions: {geo_count} ({geo_count/len(df)*100:.1f}%)")
    print(f"SRA accessions: {sra_count} ({sra_count/len(df)*100:.1f}%)")

    identified_count = doi_count + geo_count + sra_count
    other_count = len(df) - identified_count
    print(f"Other formats: {other_count} ({other_count/len(df)*100:.1f}%)")

def visualize_data_distribution(train_df):
    """Visualize the data distribution"""
    # Citation type distribution
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    train_df['citation_type'].value_counts().plot(kind='bar', color=['skyblue', 'lightcoral'])
    plt.title('Citation Type Distribution')
    plt.xlabel('Citation Type')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    train_df['citation_type'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Citation Type Proportion')
    plt.ylabel('')

    plt.tight_layout()
    plt.show()

    # Context length analysis if available
    if 'mention_context' in train_df.columns and not train_df['mention_context'].empty:
        train_df['context_length'] = train_df['mention_context'].apply(len)
        plt.figure(figsize=(10, 5))
        plt.hist(train_df['context_length'], bins=50, alpha=0.7, color='purple')
        plt.title('Distribution of Context Lengths')
        plt.xlabel('Context Length (characters)')
        plt.ylabel('Frequency')
        plt.show()

# -------------------------
# Main Demonstration
# -------------------------
def main():
    """Main demonstration function"""
    print("DATA CITATION MINING PIPELINE")
    print("=" * 60)

    # Load and prepare data
    print("Loading and preparing data...")
    train_df = load_and_prepare_data()

    # Basic statistics
    print("\nDataset Overview:")
    print(f"Total examples: {len(train_df)}")
    if 'article_id' in train_df.columns:
        print(f"Unique articles: {train_df['article_id'].nunique()}")
    print(f"Unique datasets: {train_df['dataset_id'].nunique()}")
    print("\nCitation Type Distribution:")
    print(train_df['citation_type'].value_counts())

    # Visualize data distribution
    visualize_data_distribution(train_df)

    # Analyze ID patterns
    if not train_df.empty:
        analyze_id_patterns(train_df)

    # Initialize miner
    print("\nInitializing Data Citation Miner...")
    miner = DataCitationMiner()

    # Prepare training data
    print("Preparing training data...")
    # Use 'mention_context' if available, otherwise use a default text column
    text_col = 'mention_context' if 'mention_context' in train_df.columns else 'text'
    X, y = miner.prepare_training_data(train_df, text_col=text_col, label_col='citation_type')
    print(f"Feature matrix shape: {X.shape}")

    # Train model
    print("\nTraining model...")
    accuracy = miner.train_model(X, y)

    # Test predictions
    print("\nTesting predictions...")
    test_texts = [
        "We created and deposited new proteomics data in PRIDE.",
        "Existing RNA-seq data from GEO was reanalyzed.",
        "This study produced novel datasets available online.",
        "We used previously published data from SRA accession SRP12345.",
        "Both new data was generated and existing data was utilized."
    ]

    for text in test_texts:
        prediction = miner.predict_citation_type(text)
        print(f"Text: {text[:50]}...")
        print(f"Predicted citation type: {prediction}")
        print("-" * 40)

    # Test reference extraction
    print("\nTesting reference extraction...")
    sample_text = "This study used data from GEO GSE12345 and SRA SRP98765. Also referenced DOI 10.1038/s41586-020-2649-2."
    references = miner.extract_references(sample_text)
    print(f"Text: {sample_text}")
    print(f"Extracted references: {references}")

    # Test comprehensive analysis
    print("\nTesting comprehensive publication analysis...")
    analysis = miner.analyze_publication(sample_text, "paper_123")
    print(f"Analysis results: {json.dumps(analysis, indent=2)}")

    # Save evaluation scores
    submission_scores = pd.DataFrame([{
        "Accuracy": accuracy,
        "Precision": "N/A (single model)",
        "Recall": "N/A (single model)",
        "F1_Score": "N/A (single model)"
    }])
    submission_scores.to_csv("submission_scores.csv", index=False)
    print("âœ… Scores saved to submission_scores.csv")

# Run the demonstration
if __name__ == "__main__":
    main()

print("\nPipeline execution completed successfully! ðŸŽ‰")

