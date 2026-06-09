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


df = pd.read_csv("/kaggle/input/make-data-count-finding-data-references/sample_submission.csv")
df.head()


dftest = pd.read_csv("/kaggle/input/make-data-count-finding-data-references/train_labels.csv")
dftest.head()








# import pandas as pd
# import os
# import re
# from lxml import etree
# from tqdm import tqdm

# # Load train labels and sample submission
# train_labels = pd.read_csv('/kaggle/input/make-data-count-finding-data-references/train_labels.csv')
# sample_submission = pd.read_csv('/kaggle/input/make-data-count-finding-data-references/sample_submission.csv')

# # Display basic info
# print("Train Label Samples:")
# print(train_labels.head())

# # XML helper function to parse XML file and extract text
# def extract_text_from_xml(xml_path):
#     try:
#         with open(xml_path, 'rb') as file:
#             tree = etree.parse(file)
#             body = tree.xpath('//body')
#             text = " ".join([etree.tostring(elem, method="text", encoding="unicode") for elem in body])
#         return text
#     except Exception as e:
#         return ""

# # Build a simple keyword matching rule-based extractor
# def extract_datasets(text):
#     # Sample pattern to catch DOI-like and accession ID-like patterns
#     doi_pattern = r'(10\.\d{4,9}/[-._;()/:A-Z0-9]+)'
#     accession_pattern = r'\b(GSE\d+|E-MEXP-\d+|PRJ\w+\d+|PDB\s+\w+)\b'

#     dois = re.findall(doi_pattern, text, re.IGNORECASE)
#     accessions = re.findall(accession_pattern, text)

#     results = []

#     for doi in set(dois):
#         full_doi = f'https://doi.org/{doi}'
#         results.append((full_doi, 'Primary'))  # Assume Primary for baseline

#     for acc in set(accessions):
#         results.append((acc, 'Secondary'))  # Assume Secondary for baseline

#     return results

# # Iterate over test XML files
# test_dir = "/kaggle/input/make-data-count-finding-data-references/test/XML"
# article_files = os.listdir(test_dir)

# submission_data = []

# for file in tqdm(article_files):
#     article_id = file.replace('.xml', '')
#     xml_path = os.path.join(test_dir, file)

#     text = extract_text_from_xml(xml_path)
#     dataset_mentions = extract_datasets(text)

#     for dataset_id, dtype in dataset_mentions:
#         submission_data.append({
#             'article_id': article_id,
#             'dataset_id': dataset_id,
#             'type': dtype
#         })

# # Convert to DataFrame and add row_id
# submission_df = pd.DataFrame(submission_data).drop_duplicates()
# submission_df.reset_index(drop=True, inplace=True)
# submission_df.insert(0, 'row_id', submission_df.index)

# # Save submission
# submission_df.to_csv('submission.csv', index=False)
# print("Submission file saved.")
# submission_df.head()



import pandas as pd
import os
import re
from lxml import etree
from tqdm import tqdm

# Load data
train_labels = pd.read_csv('/kaggle/input/make-data-count-finding-data-references/train_labels.csv')
sample_submission = pd.read_csv('/kaggle/input/make-data-count-finding-data-references/sample_submission.csv')

# Path to XML test files
test_dir = "/kaggle/input/make-data-count-finding-data-references/test/XML"
article_files = os.listdir(test_dir)

# Function to extract text from XML
def extract_text_from_xml(xml_path):
    try:
        with open(xml_path, 'rb') as file:
            tree = etree.parse(file)
            body = tree.xpath('//body')
            text = " ".join([etree.tostring(elem, method="text", encoding="unicode") for elem in body])
            return text
    except Exception as e:
        return ""

# Improved function to classify dataset reference as Primary or Secondary
def classify_citation(text_snippet):
    lower_text = text_snippet.lower()
    if any(keyword in lower_text for keyword in [
        "our data", "we collected", "this study", "generated", "deposited", "submitted", "data generated"
    ]):
        return "Primary"
    if any(keyword in lower_text for keyword in [
        "available at", "downloaded from", "obtained from", "previous study", "reused from", "sourced from"
    ]):
        return "Secondary"
    return "Primary"  # Default fallback

# Function to extract DOIs and accession IDs
def extract_datasets(text):
    doi_pattern = r'(?:doi:)?(?:https?://doi.org/)?(10\.\d{4,9}/[-._;()/:A-Z0-9]+)'
    accession_pattern = r'\b(GSE\d+|E-MEXP-\d+|PRJ\w+\d+|PDB\s*\w+)\b'

    matches = []

    for match in re.finditer(doi_pattern, text, flags=re.IGNORECASE):
        span = text[max(0, match.start()-100):match.end()+100]  # context window
        doi = match.group(1)
        full_doi = f"https://doi.org/{doi}"
        citation_type = classify_citation(span)
        matches.append((full_doi, citation_type))

    for match in re.finditer(accession_pattern, text):
        acc = match.group(1)
        span = text[max(0, match.start()-100):match.end()+100]
        citation_type = classify_citation(span)
        matches.append((acc, citation_type))

    return matches

# Process all test articles
submission_data = []

for file in tqdm(article_files):
    article_id = file.replace('.xml', '')
    xml_path = os.path.join(test_dir, file)

    text = extract_text_from_xml(xml_path)
    if not text.strip():
        continue

    dataset_mentions = extract_datasets(text)

    for dataset_id, dtype in dataset_mentions:
        submission_data.append({
            'article_id': article_id,
            'dataset_id': dataset_id,
            'type': dtype
        })

# Final DataFrame
submission_df = pd.DataFrame(submission_data).drop_duplicates(subset=['article_id', 'dataset_id', 'type'])
submission_df.reset_index(drop=True, inplace=True)
submission_df.insert(0, 'row_id', submission_df.index)

# Save
submission_df.to_csv("submission.csv", index=False)
submission_df.head()





