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
import os

import xml.etree.ElementTree as ET

tree = ET.parse("/kaggle/input/make-data-count-finding-data-references/train/XML/10.1038_s41396-020-00885-8.xml")
root = tree.getroot()

# Например, выводим название тега и атрибуты корня
print(root.tag, root.attrib)

# Если хотим пройтись по всем тегам
for elem in root.iter():
    print(elem.tag, elem.text)


!pip install pymupdf


import os
import re
import fitz  # PyMuPDF
from bs4 import BeautifulSoup

# ======= Чтение текста =======
def extract_text_from_xml(xml_path):
    """Чтение текста из XML."""
    with open(xml_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "xml")
    return soup.get_text(separator=" ", strip=True)

def extract_text_from_pdf(pdf_path):
    """Чтение текста из PDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def get_article_text(article_id, xml_dir, pdf_dir):
    """
    Возвращает текст статьи по её article_id.
    Сначала пробует XML, потом PDF.
    """
    xml_file = os.path.join(xml_dir, f"{article_id}.xml")
    pdf_file = os.path.join(pdf_dir, f"{article_id}.pdf")

    if os.path.exists(xml_file):
        return extract_text_from_xml(xml_file)
    elif os.path.exists(pdf_file):
        return extract_text_from_pdf(pdf_file)
    else:
        raise FileNotFoundError(f"No XML or PDF found for {article_id}")

# ======= Поиск идентификаторов =======
def extract_dataset_ids(text):
    """
    Ищет DOI и accession IDs в тексте статьи.
    Возвращает словарь с найденными объектами.
    """
    results = {
        "doi": [],
        "accession": []
    }

    # Поиск DOI (форматы: https://doi.org/..., doi:..., 10.xxx/...)
    doi_pattern = r"(?:doi:\s*|https?://doi\.org/|dx\.doi\.org/)?(10\.\d{4,9}/\S+)"
    results["doi"] = re.findall(doi_pattern, text, flags=re.IGNORECASE)

    # Поиск accession IDs (GSE, E-MEXP, PDB, PRJ, и др.)
    accession_pattern = r"\b(?:GSE\d+|E\-\w+\-\d+|PDB\s+\w+|PRJ\w+\d+)\b"
    results["accession"] = re.findall(accession_pattern, text, flags=re.IGNORECASE)

    return results

# ======= Пример использования =======
xml_dir = "/kaggle/input/make-data-count-finding-data-references/train/XML"
pdf_dir = "/kaggle/input/make-data-count-finding-data-references/train/PDF"

article_id = "10.1038_s41396-020-00885-8"  # пример
text = get_article_text(article_id, xml_dir, pdf_dir)
ids_found = extract_dataset_ids(text)

print("DOI:", ids_found["doi"])
print("Accession IDs:", ids_found["accession"])



# ======= Нормализация идентификаторов =======
def normalize_id(identifier: str) -> str:
    """Удаляет префиксы вроде https://doi.org/ и приводит к нижнему регистру."""
    if not identifier:
        return ""
    identifier = identifier.strip().lower()
    identifier = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", identifier)
    identifier = re.sub(r"^doi:\s*", "", identifier)
    return identifier

# ======= Поиск идентификаторов =======
def extract_dataset_ids(text):
    doi_pattern = r"(?:doi:\s*|https?://doi\.org/|dx\.doi\.org/)?(10\.\d{4,9}/\S+)"
    accession_pattern = r"\b(?:gse\d+|e\-\w+\-\d+|pdb\s+\w+|prj\w+\d+)\b"
    dois = [normalize_id(m) for m in re.findall(doi_pattern, text, flags=re.IGNORECASE)]
    accessions = [m.lower() for m in re.findall(accession_pattern, text, flags=re.IGNORECASE)]
    return set(dois + accessions)



labels_df = pd.read_csv("/kaggle/input/make-data-count-finding-data-references/train_labels.csv")

results = []
for article_id, group in labels_df.groupby("article_id"):
    text = get_article_text(article_id, xml_dir, pdf_dir)
    found_ids = extract_dataset_ids(text)
    for _, row in group.iterrows():
        dataset_id = normalize_id(str(row["dataset_id"]))
        citation_type = row["type"]
        found = dataset_id in found_ids
        results.append({
            "article_id": article_id,
            "dataset_id": dataset_id,
            "type": citation_type,
            "found_in_text": found
        })

results_df = pd.DataFrame(results)
results_df.to_csv("train_with_found_flag.csv", index=False)

print(results_df.head(10))

