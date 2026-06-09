import os
import re
import json
import warnings
from pathlib import Path
from tqdm.auto import tqdm


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


!pip install --no-deps --ignore-installed /kaggle/input/fitz-and-tools-whl/*.whl


import fitz  # PyMuPDF
import xml.etree.ElementTree as ET


from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
import lightgbm as lgb


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
warnings.filterwarnings("ignore")
tqdm.pandas()


BASE_PATH = Path("/kaggle/input/make-data-count-finding-data-references")
TRAIN_PATH = BASE_PATH / "train"
TEST_PATH = BASE_PATH / "test"
TRAIN_LABELS_PATH = BASE_PATH / "train_labels.csv"


SUBMISSION_PATH = Path("/kaggle/working/submission.csv")


train_labels_df = pd.read_csv(TRAIN_LABELS_PATH)


print("Training Labels Information:")
train_labels_df.info()
print("\nFirst 5 rows:")
print(train_labels_df.head())


plt.figure(figsize=(10, 6))
ax = sns.countplot(data=train_labels_df, x='type', order=['Primary', 'Secondary', 'Missing'], palette='viridis')
plt.title('Distribution of Citation Types in Training Data', fontsize=16)
plt.xlabel('Citation Type', fontsize=12)
plt.ylabel('Count', fontsize=12)

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 9), textcoords='offset points')

plt.show()


def get_id_type(dataset_id):
    if 'doi.org' in str(dataset_id) or str(dataset_id).startswith('10.'):
        return 'DOI'
    elif 'zenodo' in str(dataset_id):
        return 'Zenodo'
    elif re.match(r'GSE\d+', str(dataset_id), re.IGNORECASE):
        return 'GEO Accession (GSE)'
    elif re.match(r'E-MTAB-\d+', str(dataset_id), re.IGNORECASE):
        return 'ArrayExpress'
    elif re.match(r'CHEMBL\d+', str(dataset_id), re.IGNORECASE):
        return 'ChEMBL'
    elif 'github' in str(dataset_id):
        return 'GitHub'
    elif 'pdb' in str(dataset_id):
        return 'PDB'
    elif dataset_id == 'Missing':
        return 'Missing'
    else:
        return 'Other Accession ID'


labeled_citations = train_labels_df[train_labels_df['type'] != 'Missing'].copy()
labeled_citations['id_type'] = labeled_citations['dataset_id'].apply(get_id_type)


plt.figure(figsize=(12, 8))
ax = sns.countplot(data=labeled_citations, y='id_type', hue='type',
                   order=labeled_citations['id_type'].value_counts().index,
                   palette='magma')

plt.title('Types of Dataset IDs Found in Labels', fontsize=16)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Dataset ID Type', fontsize=12)
plt.legend(title='Citation Type')
plt.tight_layout()
plt.show()


def extract_text_from_xml(xml_path):
    """Extracts all text content from an XML file."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # Join all text nodes
        xml_text = " ".join(elem.text.strip() for elem in root.iter() if elem.text)
        return xml_text
    except ET.ParseError:
        return ""


def remove_references_section(text):
    """A heuristic to remove the references/bibliography section from a paper."""
    # Convert text to lowercase for consistent matching
    lower_text = text.lower()
    
    # Define patterns that often signal the start of a reference section
    # This list can be expanded for better accuracy
    reference_markers = [
        "references", "reference", "literature cited", "bibliography"
    ]
    
    # Find the last occurrence of any reference marker
    cut_index = -1
    for marker in reference_markers:
        # Search for the marker as a whole word, possibly at the start of a line
        pattern = r'^(?:\d+\.?\s*)?' + re.escape(marker) + r'$'
        matches = list(re.finditer(pattern, lower_text, re.MULTILINE))
        if matches:
            last_match_start = matches[-1].start()
            # Heuristic: only cut if the reference section is in the last ~40% of the doc
            if last_match_start > 0.6 * len(lower_text):
                cut_index = max(cut_index, last_match_start)
    if cut_index != -1:
        return text[:cut_index]
    return text
    
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF and attempts to remove the references section."""
    try:
        doc = fitz.open(pdf_path)
        full_text = "".join(page.get_text() for page in doc)
        doc.close()
        # Apply the heuristic to remove the reference section
        cleaned_text = remove_references_section(full_text)
        return cleaned_text
    except Exception:
        return ""


def get_text_for_article(article_id, base_folder):
    """Gets text for an article, preferring XML over PDF."""
    xml_path = base_folder / "XML" / f"{article_id}.xml"
    pdf_path = base_folder / "PDF" / f"{article_id}.pdf"
    
    if xml_path.exists():
        return extract_text_from_xml(xml_path)
    elif pdf_path.exists():
        return extract_text_from_pdf(pdf_path)
    return ""


def normalize_doi(doi_string):
    """Normalizes a DOI to the standard https://doi.org/ format."""
    # Strip leading/trailing whitespace and common punctuation
    doi_string = doi_string.strip('.,;()[]{} ')
    
    # Prepend the full URL if it's missing
    if doi_string.startswith('10.'):
        return f"https://doi.org/{doi_string}"
    elif doi_string.startswith('doi.org/'):
        return f"https://{doi_string}"
    elif doi_string.startswith('dx.doi.org/'):
        return f"https://{doi_string}"
    return doi_string


def find_potential_citations(text):
    """Finds potential dataset IDs in text using a dictionary of regex patterns."""
    
    # A more comprehensive set of patterns based on our EDA
    citation_patterns = {
        'doi': r"(?:https?://)?(?:dx\.)?doi\.org/(10\.\d{4,9}/[-._;()/:A-Z0-9]+)|(10\.\d{4,9}/[-._;()/:A-Z0-9]+)",
        'chembl': r'CHEMBL\d+',
        'gse': r'GSE\d+',
        'pdb': r'[1-9][A-Z0-9]{3}', # PDB IDs are 4 chars, starting with a number
        'arrayexpress': r'E-MEXP-\d+|E-MTAB-\d+',
        'sra': r'SRA\d+|PRJNA\d+|SRP\d+', # Sequence Read Archive patterns
        'zenodo': r'zenodo\.org/record/\d+',
        'github': r'github\.com/[\w\-]+/[\w\-]+',
    }
    
    found_citations = set()
    
    for key, pattern in citation_patterns.items():
        # Using re.finditer to get match objects for more control
        for match in re.finditer(pattern, text, re.IGNORECASE):
            candidate = match.group(0)
            
            # Special handling for DOI to capture the correct group
            if key == 'doi':
                # The regex has two capture groups, one for full URL-like DOIs and one for '10.xxx'
                doi_part = match.group(1) or match.group(2)
                if doi_part:
                    candidate = normalize_doi(doi_part)
            
            # Normalize other IDs if needed (e.g., lowercase for consistency)
            candidate = candidate.lower().strip('.,;() ')
            found_citations.add(candidate)
            
    return list(found_citations)


def create_features(text, citation, context_window=500):
    """
    Creates a dictionary of features for a given citation candidate.
    
    Args:
        text (str): The full text of the article.
        citation (str): The citation candidate string.
        context_window (int): Number of characters to look at before and after the citation.
    
    Returns:
        dict: A dictionary of features.
    """
    features = {}
    text_lower = text.lower()
    
    # Normalize citation for searching
    citation_search_term = citation.replace("https://doi.org/", "").lower()
    
    # Find the first occurrence to create context
    try:
        match_pos = text_lower.find(citation_search_term)
        if match_pos == -1:
            return {} # Should not happen if citation was found in text
        
        start = max(0, match_pos - context_window)
        end = min(len(text_lower), match_pos + len(citation_search_term) + context_window)
        context = text_lower[start:end]
    except:
        context = text_lower

    # 1. Keyword-based features
    primary_keywords = [
        "generated", "created", "produced", "our data", "this study", 
        "in-house", "we collected", "we measured", "newly collected"
    ]
    secondary_keywords = [
        "reused", "obtained from", "retrieved from", "existing data", "publicly available", 
        "previously published", "derived from", "third-party", "benchmark"
    ]
    
    features['primary_keyword_count'] = sum(1 for kw in primary_keywords if kw in context)
    features['secondary_keyword_count'] = sum(1 for kw in secondary_keywords if kw in context)

    # 2. Section-based features (heuristics)
    section_keywords = {
        'in_methods': ["method", "material"],
        'in_data_availability': ["data availability", "data access", "code availability"],
        'in_supplement': ["supplementary", "supporting information"],
    }
    for sec_name, sec_kws in section_keywords.items():
        features[sec_name] = any(kw in context for kw in sec_kws)

    # 3. Citation format features
    features['is_doi'] = 'doi.org' in citation
    features['is_github'] = 'github.com' in citation
    features['is_zenodo'] = 'zenodo.org' in citation
    features['is_accession'] = any(kw in citation for kw in ['gse', 'chembl', 'sra', 'prj', 'pdb'])

    # 4. Proximity features
    features['near_figure'] = 'figure' in context or 'fig.' in context
    features['near_table'] = 'table' in context
    return features


print("Preparing training data...")


true_labels_lookup = {}
for _, row in train_labels_df.iterrows():
    # Skip 'Missing' labels as they are not positive examples
    if row['type'] == 'Missing':
        continue
    
    article_id = row['article_id']
    dataset_id = row['dataset_id'].lower() # Normalize for matching
    citation_type = row['type']
    
    if article_id not in true_labels_lookup:
        true_labels_lookup[article_id] = {}
    true_labels_lookup[article_id][dataset_id] = citation_type


training_records = []
all_article_ids = train_labels_df['article_id'].unique()


for article_id in tqdm(all_article_ids, desc="Processing Training Articles"):
    text = get_text_for_article(article_id, TRAIN_PATH)
    if not text:
        continue
    found_citations = find_potential_citations(text)
    true_citations_for_article = true_labels_lookup.get(article_id, {})
    
    for citation in found_citations:
        features = create_features(text, citation)
        if not features:
            continue
        
        # Determine the label for this candidate
        label = true_citations_for_article.get(citation, 'Not_A_Citation')
        
        # Store all relevant information together
        training_records.append({
            'article_id': article_id,
            'citation': citation,
            'features': features,
            'label': label
        })


training_df = pd.DataFrame(training_records)
X_dicts = training_df['features'].tolist()
y_labels = training_df['label'].tolist()


print(f"\nGenerated {len(training_df)} training samples.")
print("Label distribution:")
print(training_df['label'].value_counts())


vectorizer = DictVectorizer(sparse=True)
X_vec = vectorizer.fit_transform(X_dicts)


label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y_labels)


print("\nLabel Encoding:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"{i}: {class_name}")


N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)


oof_preds = np.zeros((len(y_enc), len(label_encoder.classes_)))
models = []


lgb_params = {
    'objective': 'multiclass',
    'num_class': len(label_encoder.classes_),
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'seed': RANDOM_SEED,
    'n_jobs': -1,
    'verbose': -1,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
}


for fold, (train_idx, val_idx) in enumerate(skf.split(X_vec, y_enc)):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")
    X_train, y_train = X_vec[train_idx], y_enc[train_idx]
    X_val, y_val = X_vec[val_idx], y_enc[val_idx]
    
    model = lgb.LGBMClassifier(**lgb_params)
    
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='multi_logloss',
              callbacks=[lgb.early_stopping(100, verbose=False)])
    
    fold_preds = model.predict_proba(X_val)
    oof_preds[val_idx] = fold_preds
    models.append(model)
    
print("\n--- Training Complete ---")


oof_pred_labels = label_encoder.inverse_transform(np.argmax(oof_preds, axis=1))


oof_df = training_df.copy()
oof_df['pred_label'] = oof_pred_labels


print("OOF Classification Report:")
print(classification_report(oof_df['label'], oof_df['pred_label'], digits=4))


def calculate_competition_f1(true_df, pred_df):
    """Calculates F1 score based on competition rules."""
    
    # True positives are exact matches on article, dataset, and type
    merged_df = pd.merge(true_df, pred_df, on=['article_id', 'dataset_id', 'type'], how='inner')
    tp = len(merged_df)
    
    fp = len(pred_df) - tp
    fn = len(true_df) - tp
    
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    return f1, tp, fp, fn


oof_predictions_for_f1 = oof_df[oof_df['pred_label'].isin(['Primary', 'Secondary'])].copy()
oof_predictions_for_f1.rename(columns={'citation': 'dataset_id', 'pred_label': 'type'}, inplace=True)
# Normalize to lowercase for consistent matching
oof_predictions_for_f1['dataset_id'] = oof_predictions_for_f1['dataset_id'].str.lower()


ground_truth_df = train_labels_df[train_labels_df['type'] != 'Missing'].copy()
ground_truth_df['dataset_id'] = ground_truth_df['dataset_id'].str.lower()


f1, tp, fp, fn = calculate_competition_f1(
    ground_truth_df[['article_id', 'dataset_id', 'type']],
    oof_predictions_for_f1[['article_id', 'dataset_id', 'type']]
)


print("\nAccurate Competition F1 Score (on OOF predictions):")
print(f"TP: {tp}, FP: {fp}, FN: {fn}")
print(f"F1 Score: {f1:.4f}")


print("Generating predictions on the test set...")


test_pdf_files = os.listdir(TEST_PATH / "PDF")
test_xml_files = os.listdir(TEST_PATH / "XML")
test_article_ids = {f.replace('.pdf', '') for f in test_pdf_files} | {f.replace('.xml', '') for f in test_xml_files}


predictions = []
row_id = 0


for article_id in tqdm(list(test_article_ids), desc="Processing Test Articles"):
    text = get_text_for_article(article_id, TEST_PATH)
    if not text:
        continue
        
    found_citations = find_potential_citations(text)
    
    for citation in found_citations:
        features = create_features(text, citation)
        if not features:
            continue
        
        # Vectorize features using the same vectorizer from training
        X_test_vec = vectorizer.transform([features])
        
        # Average predictions from all models
        avg_pred_proba = np.mean([model.predict_proba(X_test_vec) for model in models], axis=0)
        
        # Get the class with the highest average probability
        pred_encoded = np.argmax(avg_pred_proba, axis=1)[0]
        citation_type = label_encoder.inverse_transform([pred_encoded])[0]
        
        # Only include 'Primary' and 'Secondary' in the submission
        if citation_type in ['Primary', 'Secondary']:
            predictions.append({
                "row_id": row_id,
                "article_id": article_id,
                "dataset_id": citation,
                "type": citation_type,
            })
            row_id += 1


submission_df = pd.DataFrame(predictions)


submission_df = submission_df.drop_duplicates(subset=['article_id', 'dataset_id', 'type'])


submission_df.to_csv('submission.csv', index=False)


print(f"\nSubmission file created with {len(submission_df)} predictions.")
print(submission_df.head())

