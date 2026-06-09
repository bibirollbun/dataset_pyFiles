import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import re

from collections import Counter
import os, xml.etree.ElementTree as ET
from glob import glob


data_root = '/kaggle/input/make-data-count-finding-data-references'
xml_dir = f'{data_root}/train/XML'
pdf_dir = f'{data_root}/train/PDF'
labels_path = f'{data_root}/train_labels.csv'


labels_df = pd.read_csv(labels_path)
label_ids = set(labels_df['article_id'].unique())


xml_files = sorted(glob(f'{xml_dir}/*.xml'))
pdf_files = sorted(glob(f'{pdf_dir}/*.pdf'))
xml_ids = set([os.path.basename(f).replace(".xml", "") for f in xml_files])
pdf_ids = set([os.path.basename(f).replace(".pdf", "") for f in pdf_files])


def parse_and_analyze(xml_path):
    tag_counter = Counter()
    sentence_count = 0
    titles = []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        def strip_ns(tag):
            return tag.split('}')[-1] if '}' in tag else tag

        for elem in root.iter():
            tag = strip_ns(elem.tag)
            tag_counter[tag] += 1

            if tag == 's':
                sentence_count += 1

            elif tag == 'title':
                text = ''.join(elem.itertext()).strip()
                if text:
                    cleaned = text.lower()
                    if len(cleaned) > 3 and not cleaned.startswith('fig') and not cleaned.startswith('table'):
                        titles.append(cleaned)

        return tag_counter, sentence_count, titles

    except Exception as e:
        return 'broken', 0, []


sentence_lengths = []
keyword_hits = Counter()
tag_stats = Counter()
section_titles = []


keywords = ['dataset', 'doi', 'argo', 'ftp', 'figshare', 'https', 'data']

for f in xml_files:
    file_id = os.path.basename(f).replace(".xml", "")
    result = parse_and_analyze(f)
    if result[0] == 'broken':
        broken_xml_ids.append(file_id)
        continue

    tags, sent_count, titles = result
    tag_stats.update(tags)
    sentence_lengths.append(sent_count)
    section_titles.extend(titles)

    # Keyword check
    for kw in keywords:
        keyword_hits[kw] += sum(1 for t in titles if kw in t)


plt.figure(figsize=(8,4))
plt.hist(sentence_lengths, bins=30, color='skyblue', edgecolor='black')
plt.title('Sentence Count per Paper')
plt.xlabel('Number of Sentences')
plt.ylabel('Number of Papers')
plt.grid(True)
plt.show()


plt.figure(figsize=(8,4))
sns.barplot(x=list(keyword_hits.keys()), y=list(keyword_hits.values()), palette='viridis')
plt.title('Keyword Frequency in Section Titles')
plt.xlabel('Keyword')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


top_tags = tag_stats.most_common(15)
tags_df = pd.DataFrame(top_tags, columns=['Tag', 'Count'])
plt.figure(figsize=(8,4))
sns.barplot(data=tags_df, x='Tag', y='Count', palette='coolwarm')
plt.title('Top 15 XML Tags')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


missing_xml_ids = label_ids - xml_ids
broken_xml_ids = []


xml_ok_ids = xml_ids - set(broken_xml_ids)
usable_ids = xml_ok_ids.union(pdf_ids)
fallback_ids = missing_xml_ids.union(set(broken_xml_ids))
pdf_fallback_ids = fallback_ids.intersection(pdf_ids)

fallback_table = pd.DataFrame({
    'Missing XML': list(missing_xml_ids),
    'Has PDF': [i in pdf_ids for i in missing_xml_ids]
})


print(fallback_table)


print("\n======= SUMMARY REPORT =======")
print(f"Total labels in train_labels.csv : {len(label_ids)}")
print(f"Available XMLs               : {len(xml_ids)}")
print(f"Broken XMLs Detected         : {len(broken_xml_ids)}")
print(f"Missing XMLs from labels.csv : {len(missing_xml_ids)}")
print(f"PDFs available               : {len(pdf_ids)}")
print(f"Fallback PDFs usable         : {len(pdf_fallback_ids)}")

