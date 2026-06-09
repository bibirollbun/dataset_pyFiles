!pip install -q PyPDF2


import pandas as pd
import os
import PyPDF2

print("DATASET STRUCTURE:")
print("=" * 40)

base_path = '/kaggle/input/make-data-count-finding-data-references'
print("Available files:")
for item in os.listdir(base_path):
    print(f"  {item}")

print("\nTraining data structure:")
train_path = f"{base_path}/train"
for item in os.listdir(train_path):
    item_path = os.path.join(train_path, item)
    if os.path.isdir(item_path):
        count = len(os.listdir(item_path))
        print(f"  {item}: {count} files")


labels_df = pd.read_csv(f'{base_path}/train_labels.csv')

print("TRAINING LABELS OVERVIEW:")
print("=" * 40)
print(f"Total labels: {len(labels_df)}")
print(f"Unique articles: {labels_df['article_id'].nunique()}")
print()
print("Label structure:")
print(labels_df.head())
print()
print("Label type distribution:")
print(labels_df['type'].value_counts())


# Focus on a specific problematic example
print("SPECIFIC TEST CASE:")
print("=" * 40)

# The problematic example identified
test_article = '10.1016_j.cpc.2024.109087'
test_row = labels_df[labels_df['article_id'] == test_article]

print(f"Article ID: {test_article}")
print(f"Expected dataset reference: {test_row['dataset_id'].iloc[0]}")
print()
print("TEMPORAL ISSUE:")
print("- Article uploaded to arXiv: January 20, 2024")
print("- Dataset uploaded to Mendeley: January 29, 2024")
print("- Question: How can the article reference a dataset created 9 days later?")


# Check if the dataset reference actually appears in the PDF
print("PDF CONTENT ANALYSIS:")
print("=" * 40)

pdf_path = f'{base_path}/train/PDF/{test_article}.pdf'
dataset_doi = test_row['dataset_id'].iloc[0]
dataset_id = dataset_doi.split('/')[-1]  # Extract unique ID: 9gr6pxhfjm

print(f"Searching PDF for dataset reference...")
print(f"Full DOI: {dataset_doi}")
print(f"Dataset ID: {dataset_id}")
print()

try:
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        
        # Extract all text
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text()
        
        print(f"PDF text length: {len(full_text):,} characters")
        print()
        print("SEARCH RESULTS:")
        print(f"Contains dataset ID '{dataset_id}': {dataset_id in full_text}")
        print(f"Contains full DOI '{dataset_doi}': {dataset_doi in full_text}")
        print(f"Contains 'Mendeley': {'mendeley' in full_text.lower()}")
        print(f"Contains 'data availab': {'data availab' in full_text.lower()}")
        
except Exception as e:
    print(f"Error reading PDF: {e}")


# Check if the dataset reference appears in the XML version
print("XML CONTENT ANALYSIS:")
print("=" * 40)

xml_path = f'{base_path}/train/XML/{test_article}.xml'

try:
    with open(xml_path, 'r', encoding='utf-8') as file:
        xml_content = file.read()
    
    print(f"XML content length: {len(xml_content):,} characters")
    print()
    print("SEARCH RESULTS:")
    print(f"Contains dataset ID '{dataset_id}': {dataset_id in xml_content}")
    print(f"Contains full DOI '{dataset_doi}': {dataset_doi in xml_content}")
    print(f"Contains 'Mendeley': {'mendeley' in xml_content.lower()}")
    print(f"Contains 'data availab': {'data availab' in xml_content.lower()}")
    
except Exception as e:
    print(f"Error reading XML: {e}")





from tqdm import tqdm
import random

# Focus only on Primary labels for analysis - random sample of 50
print("PRIMARY LABEL ANALYSIS (RANDOM SAMPLE):")
print("=" * 50)

primary_labels = labels_df[labels_df['type'] == 'Primary'].copy()
print(f"Total Primary labels: {len(primary_labels)}")

# Random sample of 50
random.seed(42)  # For reproducibility
sample_primary = primary_labels.sample(n=50, random_state=42)
print(f"Analyzing random sample of {len(sample_primary)} Primary labels...")
print()

# Track results
found_in_pdf = 0
found_in_xml = 0
not_found_anywhere = 0
pdf_only = 0
xml_only = 0
both_formats = 0
analysis_errors = 0

# Sample detailed results for investigation
detailed_results = []

for idx, row in tqdm(sample_primary.iterrows(), total=len(sample_primary), desc="Analyzing Primary labels"):
    article_id = row['article_id']
    dataset_doi = row['dataset_id']
    
    # Extract dataset ID from DOI
    if 'doi.org' in dataset_doi:
        dataset_id = dataset_doi.split('/')[-1]
    else:
        dataset_id = dataset_doi
    
    pdf_found = False
    xml_found = False
    
    try:
        # Check PDF
        pdf_path = f'{base_path}/train/PDF/{article_id}.pdf'
        if os.path.exists(pdf_path):
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                pdf_text = ""
                for page in pdf_reader.pages:
                    pdf_text += page.extract_text()
                
                # Check for various forms of the reference
                pdf_found = (dataset_id in pdf_text or 
                           dataset_doi in pdf_text or
                           dataset_doi.replace('https://', '') in pdf_text)
        
        # Check XML
        xml_path = f'{base_path}/train/XML/{article_id}.xml'
        if os.path.exists(xml_path):
            with open(xml_path, 'r', encoding='utf-8') as file:
                xml_content = file.read()
                
                # Check for various forms of the reference
                xml_found = (dataset_id in xml_content or 
                           dataset_doi in xml_content or
                           dataset_doi.replace('https://', '') in xml_content)
        
        # Update counters
        if pdf_found:
            found_in_pdf += 1
        if xml_found:
            found_in_xml += 1
        if pdf_found and xml_found:
            both_formats += 1
        elif pdf_found and not xml_found:
            pdf_only += 1
        elif xml_found and not pdf_found:
            xml_only += 1
        elif not pdf_found and not xml_found:
            not_found_anywhere += 1
            
        # Store detailed results for not-found cases
        if not pdf_found and not xml_found:
            detailed_results.append({
                'article_id': article_id,
                'dataset_id': dataset_id,
                'dataset_doi': dataset_doi,
                'type': row['type']
            })
            
    except Exception as e:
        analysis_errors += 1

print("\nPRIMARY LABEL ANALYSIS RESULTS (Sample of 50):")
print(f"Found in PDF: {found_in_pdf}")
print(f"Found in XML: {found_in_xml}")
print(f"Found in both formats: {both_formats}")
print(f"Found in PDF only: {pdf_only}")
print(f"Found in XML only: {xml_only}")
print(f"Not found anywhere: {not_found_anywhere}")
print(f"Analysis errors: {analysis_errors}")
print()
print(f"Success rate: {((found_in_pdf + found_in_xml - both_formats) / len(sample_primary) * 100):.1f}%")

print(f"\nAll cases not found anywhere ({len(detailed_results)}):")
for i, result in enumerate(detailed_results):
    print(f"{i+1}. {result['article_id']} -> {result['dataset_id']}")

