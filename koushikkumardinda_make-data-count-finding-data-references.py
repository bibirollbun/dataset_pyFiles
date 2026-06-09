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
import re
from pathlib import Path
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

# --- Paths and Constants ---
DATA_DIR = Path('/kaggle/input/make-data-count-finding-data-references')
TRAIN_DIR = DATA_DIR / 'train'
TEST_DIR = DATA_DIR / 'test'

# --- 1. Data Extraction and Preprocessing ---

def get_xml_text(xml_path):
    """Extracts text from an XML file."""
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'lxml-xml')
            text_parts = soup.find_all(['p', 'abstract', 'sec', 'title', 'ref-list'])
            full_text = ' '.join(p.get_text() for p in text_parts)
            return full_text
    except Exception:
        return ""

def get_article_doi(xml_path):
    """Extracts the article DOI from an XML file."""
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'lxml-xml')
            doi_tag = soup.find('article-id', {'pub-id-type': 'doi'})
            if doi_tag:
                return doi_tag.get_text()
    except Exception:
        pass
    return None

def extract_text_from_article(article_id, data_dir):
    """
    Extracts text from a given article, preferring XML over PDF.
    Returns the text and the article DOI.
    """
    xml_path = data_dir / 'XML' / f'{article_id}.xml'
    pdf_path = data_dir / 'PDF' / f'{article_id}.pdf'
    
    article_doi = None
    if xml_path.exists():
        text = get_xml_text(xml_path)
        article_doi = get_article_doi(xml_path)
        if not article_doi:
            # Fallback to article_id if DOI not found in XML
            article_doi = article_id
        return text, article_doi
    
    if pdf_path.exists():
        # NOTE: pdfplumber is not a standard Kaggle library.
        # You would need to install it if allowed or use a different parser.
        # For this example, we will treat PDFs as non-parsable to simulate
        # a real-world scenario where XML is prioritized.
        print(f"Warning: PDF file for {article_id} is not supported in this simplified script.")
    
    return "", None

# --- 2. Identifier Detection ---

DOI_REGEX = r'(?:doi:|https?://(?:dx\.)?doi\.org/)(10\.\d{4,9}/[^\s"\'<>()]+)'
ACCESSION_ID_REGEX = r'\b(GSE\d+|PDB\s?\d[a-zA-Z\d]{3}|E-MEXP-\d+|E-MTAB-\d+|PRJ[AEBD]\d+)\b'

def find_all_identifiers(text):
    """Finds all DOIs and Accession IDs in a given text."""
    found_dois = set(re.findall(DOI_REGEX, text, re.IGNORECASE))
    found_accessions = set(re.findall(ACCESSION_ID_REGEX, text, re.IGNORECASE))
    
    # Add full doi url for consistency
    full_dois = {f"https://doi.org/{d}" for d in found_dois}
    
    return list(full_dois.union(found_accessions))

def get_context(text, identifier, window_size=50):
    """Extracts a fixed-size context window around an identifier."""
    # Escape special characters for regex
    escaped_id = re.escape(identifier)
    
    # Find the start and end of the identifier in the text
    match = re.search(escaped_id, text, re.IGNORECASE)
    if not match:
        return ""
    
    start_pos = max(0, match.start() - window_size)
    end_pos = min(len(text), match.end() + window_size)
    
    return text[start_pos:end_pos]

# --- 3. Citation Type Classification ---

def train_model():
    """Trains a simple classifier on the training data."""
    
    # Load labels
    train_labels_df = pd.read_csv(TRAIN_DIR / '/kaggle/input/make-data-count-finding-data-references/train_labels.csv')
    
    # Feature engineering for training data
    contexts = []
    labels = []
    
    for _, row in train_labels_df.iterrows():
        article_id = row['article_id']
        dataset_id = row['dataset_id']
        citation_type = row['type']
        
        # NOTE: This is a simplification. A robust solution would handle
        # multiple identifiers per article and get context for each.
        text, _ = extract_text_from_article(article_id, TRAIN_DIR)
        
        if text and dataset_id:
            context = get_context(text, dataset_id)
            if context:
                contexts.append(context)
                labels.append(citation_type)

    if not contexts:
        print("Warning: No training data with valid contexts found.")
        return None, None
        
    # Vectorize the text data
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train = vectorizer.fit_transform(contexts)
    y_train = labels
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, vectorizer

# --- Main Logic ---

if __name__ == "__main__":
    
    # Train the model and vectorizer on the training data
    model, vectorizer = train_model()
    
    if not model or not vectorizer:
        print("Model training failed. Exiting.")
    else:
        submission_rows = []
        row_id = 0
        
        # Get list of all test articles (XML and PDF)
        test_article_files = set([p.stem for p in (TEST_DIR / 'XML').glob('*.xml')])
        
        # Process each test article
        for article_id in sorted(list(test_article_files)):
            print(f"Processing {article_id}...")
            
            # Step 1: Extract text and article DOI
            text, article_doi = extract_text_from_article(article_id, TEST_DIR)
            
            if not text:
                print(f"  - No parsable text found. Skipping.")
                continue
            
            # Step 2: Find all dataset identifiers
            found_identifiers = find_all_identifiers(text)
            
            if not found_identifiers:
                print(f"  - No identifiers found. Skipping.")
                continue
            
            # Step 3: Classify each identifier
            for dataset_id in found_identifiers:
                context = get_context(text, dataset_id)
                
                # Check if context is non-empty before prediction
                if context:
                    context_vector = vectorizer.transform([context])
                    citation_type = model.predict(context_vector)[0]
                else:
                    # Default to 'Secondary' if context is not found
                    citation_type = 'Secondary'
                
                submission_rows.append({
                    'row_id': row_id,
                    'article_id': article_doi,
                    'dataset_id': dataset_id,
                    'type': citation_type
                })
                row_id += 1
                
        # Step 4: Create and save the submission file
        submission_df = pd.DataFrame(submission_rows)
        submission_df.to_csv('submission.csv', index=False)
        
        print("\nSubmission file 'submission.csv' created successfully.")
        print(submission_df.head())


# import pandas as pd
# import re
# import os
# import lxml.etree as ET
# import fitz  # PyMuPDF library

# class DataReferenceExtractor:
#     """
#     A class to extract and classify data references from scientific papers.
#     Designed for an internet-disabled environment, relying on regex, keyword
#     matching, and robust file parsing.
#     """

#     def __init__(self, data_path, output_path, primary_keywords, secondary_keywords):
#         self.data_path = data_path
#         self.output_path = output_path
#         self.primary_keywords = primary_keywords
#         self.secondary_keywords = secondary_keywords
#         self.results = []
#         self.row_id = 0

#     def parse_text(self, article_id):
#         """
#         Parses the full text from either an XML or PDF file.
#         Prioritizes XML as the most reliable source.

#         Args:
#             article_id (str): The unique identifier for the article.

#         Returns:
#             str: The full text of the article.
#         """
#         # Try to parse XML first, as it's the most reliable source
#         xml_filepath = os.path.join(self.data_path, 'XML', f'{article_id}.xml')
#         if os.path.exists(xml_filepath):
#             try:
#                 tree = ET.parse(xml_filepath)
#                 # Find all text content within the article
#                 text = ' '.join(tree.xpath('//text()'))
#                 return text.strip()
#             except ET.ParseError as e:
#                 print(f"Error parsing XML for {article_id}: {e}. Falling back to PDF.")

#         # If XML fails or doesn't exist, try to parse PDF
#         pdf_filepath = os.path.join(self.data_path, 'PDF', f'{article_id}.pdf')
#         if os.path.exists(pdf_filepath):
#             try:
#                 with fitz.open(pdf_filepath) as doc:
#                     text = ""
#                     for page in doc:
#                         text += page.get_text()
#                 return text.strip()
#             except Exception as e:
#                 print(f"Error parsing PDF for {article_id}: {e}")

#         print(f"No usable file found for article_id: {article_id}")
#         return ""

#     def find_identifiers(self, text):
#         """
#         Finds all potential dataset identifiers using a robust set of regex patterns.
#         Includes DOIs and various Accession IDs.

#         Args:
#             text (str): The full text of the article.

#         Returns:
#             list: A list of unique dataset identifiers found.
#         """
#         found_ids = set()

#         # Regex patterns for various identifier types
#         # 1. DOIs: Captures standard DOIs with and without prefixes
#         doi_pattern = r'(?:https?://(?:dx\.)?doi\.org/|doi:)?(10\.\d{4,9}/[^\s/"]+)'
#         # 2. Accession IDs: Common patterns from the dataset description
#         accession_patterns = [
#             r'(GSE\d+)',                 # Gene Expression Omnibus
#             r'(PDB\s\w+)',               # Protein Data Bank
#             r'(E-MEXP-\d+)',             # ArrayExpress
#             r'(CHEMBL\d+)',              # ChEMBL
#             r'(PRJ[ENADBS]+\d+)'         # NCBI BioProject / ENA
#         ]

#         # Combine all patterns into a single list
#         patterns = [doi_pattern] + accession_patterns

#         for pattern in patterns:
#             matches = re.findall(pattern, text, re.IGNORECASE)
#             for match in matches:
#                 # For DOIs, ensure the prefix is standardized for the submission file
#                 if '10.' in match and 'doi.org' not in match:
#                     found_ids.add(f'https://doi.org/{match}')
#                 else:
#                     found_ids.add(match)

#         return list(found_ids)

#     def classify_type(self, context):
#         """
#         Classifies the citation type as 'Primary' or 'Secondary' based on keywords
#         in the surrounding text context. Simulates a trained model.

#         Args:
#             context (str): The text snippet surrounding the identifier.

#         Returns:
#             str: 'Primary', 'Secondary', or 'Unknown'.
#         """
#         context_lower = context.lower()

#         # Score the context based on keyword presence
#         primary_score = sum(1 for kw in self.primary_keywords if kw in context_lower)
#         secondary_score = sum(1 for kw in self.secondary_keywords if kw in context_lower)

#         # A simple heuristic for classification:
#         if primary_score > secondary_score:
#             return 'Primary'
#         elif secondary_score > primary_score:
#             return 'Secondary'
#         else:
#             # If scores are equal, or both are zero, classify based on a common pattern
#             # For this competition, many DOIs are primary, and accession IDs are often secondary.
#             # This is a heuristic to break ties.
#             if 'doi.org' in context_lower or '10.' in context_lower:
#                 return 'Primary'
#             else:
#                 return 'Secondary'

#     def process_article(self, article_id):
#         """
#         Processes a single article file to find and classify data references.

#         Args:
#             article_id (str): The unique identifier for the article.
#         """
#         full_text = self.parse_text(article_id)
#         if not full_text:
#             return

#         identifiers = self.find_identifiers(full_text)

#         # De-duplicate identifiers found within a single article
#         unique_identifiers = sorted(list(set(identifiers)))

#         for identifier in unique_identifiers:
#             # Find the context for classification
#             # Search for the exact identifier string in the text
#             # Handle both with and without the 'https://doi.org/' prefix
#             search_str = identifier.replace('https://doi.org/', '')
#             match = re.search(f'(.{{100}}){re.escape(search_str)}(.{{100}})', full_text, re.DOTALL | re.IGNORECASE)

#             if match:
#                 context = match.group(1) + match.group(2)
#                 citation_type = self.classify_type(context)

#                 # Append to results list
#                 self.results.append({
#                     'row_id': self.row_id,
#                     'article_id': article_id,
#                     'dataset_id': identifier,
#                     'type': citation_type
#                 })
#                 self.row_id += 1

#     def generate_submission(self, test_df):
#         """
#         Generates the final submission file.

#         Args:
#             test_df (pd.DataFrame): DataFrame containing the test article IDs.
#         """
#         # Process each article in the test set
#         for index, row in test_df.iterrows():
#             article_id = row['article_id']
#             self.process_article(article_id)

#         # Convert results to a DataFrame
#         if not self.results:
#             print("No data references found. Generating an empty submission file.")
#             submission_df = pd.DataFrame(columns=['row_id', 'article_id', 'dataset_id', 'type'])
#         else:
#             submission_df = pd.DataFrame(self.results)
#             # Ensure unique (article_id, dataset_id) pairs
#             submission_df.drop_duplicates(subset=['article_id', 'dataset_id'], inplace=True)
#             # Re-index the row_id
#             submission_df['row_id'] = range(len(submission_df))

#         # Save the final submission file
#         submission_df.to_csv(self.output_path, index=False)
#         print(f"Submission file created successfully at {self.output_path} with {len(submission_df)} rows.")

# if __name__ == '__main__':
#     # Define file paths and keywords
#     # These paths are typical for a Kaggle notebook
#     data_dir = '/kaggle/input/make-data-count-finding-data-references/test'
#     test_articles_csv = '/kaggle/input/make-data-count-finding-data-references/sample_submission.csv'
#     submission_file_path = 'submission.csv'

#     # Keywords for a rule-based classification
#     primary_keywords = [
#         'generated', 'collected', 'raw data for', 'our study', 'newly acquired', 'newly sequenced',
#         'is available at', 'can be accessed from', 'produced', 'deposited', 'provided in this study'
#     ]
#     secondary_keywords = [
#         'reused', 'derived from', 'publicly available', 're-analyzed', 'retrieved from',
#         'used from', 'existing data', 'published data', 'from a previous study'
#     ]

#     # Load the sample submission file to get the list of test article IDs
#     try:
#         test_df = pd.read_csv(test_articles_csv)
#         print(f"Loaded test articles from {test_articles_csv}")
#     except FileNotFoundError:
#         print(f"Error: The file {test_articles_csv} was not found.")
#         test_df = pd.DataFrame({'article_id': []})

#     # Initialize and run the extractor
#     extractor = DataReferenceExtractor(data_dir, submission_file_path, primary_keywords, secondary_keywords)
#     extractor.generate_submission(test_df)


import pandas as pd
import re
import os
import lxml.etree as ET
# The 'fitz' library (PyMuPDF) is not available in the offline Kaggle environment.
# PDF parsing will be skipped for articles without a corresponding XML file.

class DataReferenceExtractor:
    """
    A class to extract and classify data references from scientific papers.
    Designed for an internet-disabled environment, relying on regex, keyword
    matching, and robust file parsing.
    """

    def __init__(self, data_path, output_path, primary_keywords, secondary_keywords):
        self.data_path = data_path
        self.output_path = output_path
        self.primary_keywords = primary_keywords
        self.secondary_keywords = secondary_keywords
        self.results = []
        self.row_id = 0

    def parse_text(self, article_id):
        """
        Parses the full text from either an XML or PDF file.
        Prioritizes XML as the most reliable source.

        Args:
            article_id (str): The unique identifier for the article.

        Returns:
            str: The full text of the article.
        """
        # Try to parse XML first, as it's the most reliable source
        xml_filepath = os.path.join(self.data_path, 'XML', f'{article_id}.xml')
        if os.path.exists(xml_filepath):
            try:
                tree = ET.parse(xml_filepath)
                # Find all text content within the article
                text = ' '.join(tree.xpath('//text()'))
                return text.strip()
            except ET.ParseError as e:
                print(f"Error parsing XML for {article_id}: {e}. Skipping PDF parsing due to library unavailability.")
                return ""

        # Since 'fitz' is not available, we cannot parse the PDF.
        # Print a message and return an empty string.
        pdf_filepath = os.path.join(self.data_path, 'PDF', f'{article_id}.pdf')
        if os.path.exists(pdf_filepath):
            print(f"Skipping PDF for {article_id} due to 'fitz' library unavailability.")
        else:
            print(f"No usable file found for article_id: {article_id}")

        return ""

    def find_identifiers(self, text):
        """
        Finds all potential dataset identifiers using a robust set of regex patterns.
        Includes DOIs and various Accession IDs.

        Args:
            text (str): The full text of the article.

        Returns:
            list: A list of unique dataset identifiers found.
        """
        found_ids = set()

        # Regex patterns for various identifier types
        # 1. DOIs: Captures standard DOIs with and without prefixes
        doi_pattern = r'(?:https?://(?:dx\.)?doi\.org/|doi:)?(10\.\d{4,9}/[^\s/"]+)'
        # 2. Accession IDs: Common patterns from the dataset description
        accession_patterns = [
            r'(GSE\d+)',                 # Gene Expression Omnibus
            r'(PDB\s\w+)',               # Protein Data Bank
            r'(E-MEXP-\d+)',             # ArrayExpress
            r'(CHEMBL\d+)',              # ChEMBL
            r'(PRJ[ENADBS]+\d+)'         # NCBI BioProject / ENA
        ]

        # Combine all patterns into a single list
        patterns = [doi_pattern] + accession_patterns

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # For DOIs, ensure the prefix is standardized for the submission file
                if '10.' in match and 'doi.org' not in match:
                    found_ids.add(f'https://doi.org/{match}')
                else:
                    found_ids.add(match)

        return list(found_ids)

    def classify_type(self, context):
        """
        Classifies the citation type as 'Primary' or 'Secondary' based on keywords
        in the surrounding text context. Simulates a trained model.

        Args:
            context (str): The text snippet surrounding the identifier.

        Returns:
            str: 'Primary', 'Secondary', or 'Unknown'.
        """
        context_lower = context.lower()

        # Score the context based on keyword presence
        primary_score = sum(1 for kw in self.primary_keywords if kw in context_lower)
        secondary_score = sum(1 for kw in self.secondary_keywords if kw in context_lower)

        # A simple heuristic for classification:
        if primary_score > secondary_score:
            return 'Primary'
        elif secondary_score > primary_score:
            return 'Secondary'
        else:
            # If scores are equal, or both are zero, classify based on a common pattern
            # For this competition, many DOIs are primary, and accession IDs are often secondary.
            # This is a heuristic to break ties.
            if 'doi.org' in context_lower or '10.' in context_lower:
                return 'Primary'
            else:
                return 'Secondary'

    def process_article(self, article_id):
        """
        Processes a single article file to find and classify data references.

        Args:
            article_id (str): The unique identifier for the article.
        """
        full_text = self.parse_text(article_id)
        if not full_text:
            return

        identifiers = self.find_identifiers(full_text)

        # De-duplicate identifiers found within a single article
        unique_identifiers = sorted(list(set(identifiers)))

        for identifier in unique_identifiers:
            # Find the context for classification
            # Search for the exact identifier string in the text
            # Handle both with and without the 'https://doi.org/' prefix
            search_str = identifier.replace('https://doi.org/', '')
            match = re.search(f'(.{{100}}){re.escape(search_str)}(.{{100}})', full_text, re.DOTALL | re.IGNORECASE)

            if match:
                context = match.group(1) + match.group(2)
                citation_type = self.classify_type(context)

                # Append to results list
                self.results.append({
                    'row_id': self.row_id,
                    'article_id': article_id,
                    'dataset_id': identifier,
                    'type': citation_type
                })
                self.row_id += 1

    def generate_submission(self, test_df):
        """
        Generates the final submission file.

        Args:
            test_df (pd.DataFrame): DataFrame containing the test article IDs.
        """
        # Process each article in the test set
        for index, row in test_df.iterrows():
            article_id = row['article_id']
            self.process_article(article_id)

        # Convert results to a DataFrame
        if not self.results:
            print("No data references found. Generating an empty submission file.")
            submission_df = pd.DataFrame(columns=['row_id', 'article_id', 'dataset_id', 'type'])
        else:
            submission_df = pd.DataFrame(self.results)
            # Ensure unique (article_id, dataset_id) pairs
            submission_df.drop_duplicates(subset=['article_id', 'dataset_id'], inplace=True)
            # Re-index the row_id
            submission_df['row_id'] = range(len(submission_df))

        # Save the final submission file
        submission_df.to_csv(self.output_path, index=False)
        print(f"Submission file created successfully at {self.output_path} with {len(submission_df)} rows.")

if __name__ == '__main__':
    # Define file paths and keywords
    # These paths are typical for a Kaggle notebook
    data_dir = '/kaggle/input/make-data-count-finding-data-references/test'
    test_articles_csv = '/kaggle/input/make-data-count-finding-data-references/sample_submission.csv'
    submission_file_path = 'submission.csv'

    # Keywords for a rule-based classification
    primary_keywords = [
        'generated', 'collected', 'raw data for', 'our study', 'newly acquired', 'newly sequenced',
        'is available at', 'can be accessed from', 'produced', 'deposited', 'provided in this study'
    ]
    secondary_keywords = [
        'reused', 'derived from', 'publicly available', 're-analyzed', 'retrieved from',
        'used from', 'existing data', 'published data', 'from a previous study'
    ]

    # Load the sample submission file to get the list of test article IDs
    try:
        test_df = pd.read_csv(test_articles_csv)
        print(f"Loaded test articles from {test_articles_csv}")
    except FileNotFoundError:
        print(f"Error: The file {test_articles_csv} was not found.")
        test_df = pd.DataFrame({'article_id': []})

    # Initialize and run the extractor
    extractor = DataReferenceExtractor(data_dir, submission_file_path, primary_keywords, secondary_keywords)
    extractor.generate_submission(test_df)


