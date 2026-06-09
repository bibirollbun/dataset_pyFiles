import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


path = "/kaggle/input/make-data-count-finding-data-references"

test_pdf = f"{path}/test/PDF"
test_xml = f"{path}/test/XML"
train_pdf = f"{path}/train/PDF"
train_xml = f"{path}/train/XML"
train_labels = f"{path}/train_labels.csv"


print("Total train Files:")
print("PDF files:", len(os.listdir(train_pdf)))
print("XML files:", len(os.listdir(train_xml)))

print("\nTotal test Files:")
print("PDF files:", len(os.listdir(test_pdf)))
print("XML files:", len(os.listdir(test_xml)))


train_pdf_id = set(f[:-4] for f in os.listdir(train_pdf) if f.endswith(".pdf"))
train_xml_id = set(f[:-4] for f in os.listdir(train_xml) if f.endswith(".xml"))

both = train_pdf_id & train_xml_id
only_pdf = train_pdf_id - train_xml_id
only_xml = train_xml_id - train_pdf_id

print(f"PDF files:       {len(train_pdf_id)}")
print(f"XML files:       {len(train_xml_id)}")
print(f"Matched:         {len(both)}")
print(f"Only in PDF:     {len(only_pdf)}")
print(f"Only in XML:     {len(only_xml)}")


test_pdf_id = set(f[:-4] for f in os.listdir(test_pdf) if f.endswith(".pdf"))
test_xml_id = set(f[:-4] for f in os.listdir(test_xml) if f.endswith(".xml"))

both = test_pdf_id & test_xml_id
onlly_pdf = test_pdf_id - test_xml_id
onlly_xml = test_xml_id - test_pdf_id

print(f"PDF files:       {len(test_pdf_id)}")
print(f"XML files:       {len(test_xml_id)}")
print(f"Matched:         {len(both)}")
print(f"Only in PDF:     {len(onlly_pdf)}")
print(f"Only in XML:     {len(onlly_xml)}")


labels = pd.read_csv(train_labels)
print(labels.head())


print(f"total rows: {labels.shape[0]}")
print(f"total columns: {labels.shape[1]}")


print("Data types:")
print(labels.dtypes)

print("\narticle_ids:", labels['article_id'].nunique())
print("dataset_ids:", labels['dataset_id'].nunique())
print("type:", labels['type'].nunique())


print("\nLabel Distribution")
print(labels["type"].value_counts())


plt.figure(figsize=(4, 3))
labels["type"].value_counts().plot(kind="bar")
plt.title("Primary vs Secondary Labels")
plt.ylabel("Count")
plt.show()


def extract_source(x):
    x = str(x)
    if x.startswith("https://doi.org/"):
        return x.split("/")[2]
    elif ":" in x:
        return x.split(":")[0]
    elif x[:3].isalpha():
        return x[:6]
    return "Unknown"


labels['source'] = labels['dataset_id'].apply(extract_source)
print("\ntop dataset sources:")
print(labels['source'].value_counts().head(20))


import xml.etree.ElementTree as ET

sample_xml = f"{train_xml}/10.1002_2017jc013030.xml"


try:
    tree = ET.parse(sample_xml)
    root = tree.getroot()
    text_xml = " ".join([elem.text.strip() for elem in root.iter() if elem.text and elem.text.strip()])
    print("sample XML extracted text:\n")
    print(text_xml[:1000])
except Exception as e:
    print("failed to extract:", e)


!pip install pdfplumber


import pdfplumber

sample_pdf1 = f"{train_pdf}/10.1002_2017jc013030.pdf"


text = ""
try:
    with pdfplumber.open(sample_pdf1) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    print("sample PDF extracted text:\n")
    print(text[:1000])
except Exception as e:
    print("failed to extract:", e)


!pip install pyMuPDF


import fitz

sample_pdf2 = f"{train_pdf}/10.1002_2017jc013030.pdf"


text1 =""
try:
    doc = fitz.open(sample_pdf2)
    for page in doc:
        text1 += page.get_text()
    print("Extracted text using PyMuPDF:\n")
    print(text1[:1000])
except Exception as e:
    print("failed to extract:", e)


import re


# Based on the Train_labels.csv taken

patterns = {
    "DOI": r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+(?<![\.\,])",
    "Accession ID": r"\b(?:GSE|E-GEOD|E-MTAB|PRJ[EAN]|EPI_ISL|EGAS|CHEMBL|EMPIAR|CVCL|ENS[A-Z]+|IPR|HPA|CP|ERR|SRR|K0\d{4})\d+\b",
    "PDB ID": r"\b(?!\d{4}\b)(?:[A-Z0-9]{4})\b",
    "Taxonomy ID": r"\b3\.\d+\.\d+\.\d+\b(?!\.\d)" 
}

compiled_patterns = {k: re.compile(v, re.IGNORECASE) for k, v in patterns.items()}


for source, content in zip(["XML", "pdfplumber", "PyMuPDF"], [text_xml, text, text1]):
    print(f"\n=== Matches from {source} ===")
    for label, regex in compiled_patterns.items():
        matches = regex.findall(content)
        print(f"{label} (first 10):", matches[:10])






















