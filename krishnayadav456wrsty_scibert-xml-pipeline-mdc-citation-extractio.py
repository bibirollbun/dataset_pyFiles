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



import os
import pandas as pd
import xml.etree.ElementTree as ET
from collections import defaultdict
from fuzzywuzzy import fuzz, process


train_path = "/kaggle/input/make-data-count-finding-data-references/train_labels.csv"
test_xml_dir = "/kaggle/input/make-data-count-finding-data-references/test/XML"


df_train = pd.read_csv(train_path)
known_datasets = df_train['dataset_id'].unique().tolist()

print("Train labels sample:")
print(df_train.head())


def extract_mentions_from_xml(file_path):
    mentions = set()
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        for elem in root.iter():
            if elem.tag.lower() in ['ref', 'xref']:
                text = ''.join(elem.itertext()).strip()
                if text:
                    mentions.add(text)
    except ET.ParseError:
        print(f"â�Œ Failed to parse: {file_path}")
    return mentions

# Extract mentions from all test files
all_mentions = defaultdict(set)
xml_files = [f for f in os.listdir(test_xml_dir) if f.endswith(".xml")]
print(f"Total test XML files: {len(xml_files)}")

for fname in xml_files:
    path = os.path.join(test_xml_dir, fname)
    mentions = extract_mentions_from_xml(path)
    all_mentions[fname] = mentions


matches = []
for xml_file, mentions in all_mentions.items():
    article_id = xml_file.replace(".xml", "")
    for mention in mentions:
        best_match, score = process.extractOne(mention, known_datasets, scorer=fuzz.token_sort_ratio)
        if score >= 85:  
            matches.append({
                "article_id": article_id,
                "dataset_id": best_match
            })

# Create submission DataFrame
submission_df = pd.DataFrame(matches).drop_duplicates()

# ğŸ’¾ Save to CSV
submission_df.to_csv("submission.csv", index=False)
print(" Submission file created: submission.csv")
print(submission_df.head())


