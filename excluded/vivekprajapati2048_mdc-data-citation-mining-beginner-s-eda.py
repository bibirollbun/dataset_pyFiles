import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


TRAIN_PATH = '/kaggle/input/make-data-count-finding-data-references/train'
TEST_PATH = '/kaggle/input/make-data-count-finding-data-references/test'
LABELS_PATH = '/kaggle/input/make-data-count-finding-data-references/train_labels.csv'

TRAIN_PDF = os.path.join(TRAIN_PATH, 'PDF')
TRAIN_XML = os.path.join(TRAIN_PATH, 'XML')
TEST_PDF = os.path.join(TEST_PATH, 'PDF')
TEST_XML = os.path.join(TEST_PATH, 'XML')


labels_df = pd.read_csv(LABELS_PATH)
labels_df['has_dataset'] = labels_df['dataset_id'] != 'Missing'

print(f"ğŸ”� Train labels shape: {labels_df.shape}")
display(labels_df.head())


def load_files(pdf_path, xml_path, dataset_name):
    pdf_files = [f for f in os.listdir(pdf_path) if f.endswith('.pdf')]
    xml_files = [f for f in os.listdir(xml_path) if f.endswith('.xml')]
    df_pdf = pd.DataFrame({
        'article_id': [f.replace('.pdf', '') for f in pdf_files],
        'file': pdf_files,
        'path': [os.path.join(pdf_path, f) for f in pdf_files],
        'format': 'pdf',
        'dataset': dataset_name
    })
    df_xml = pd.DataFrame({
        'article_id': [f.replace('.xml', '') for f in xml_files],
        'file': xml_files,
        'path': [os.path.join(xml_path, f) for f in xml_files],
        'format': 'xml',
        'dataset': dataset_name
    })
    return pd.concat([df_pdf, df_xml], ignore_index=True)

train_files = load_files(TRAIN_PDF, TRAIN_XML, 'train')
test_files = load_files(TEST_PDF, TEST_XML, 'test')

print(f"ğŸ—‚ï¸� Train files shape: {train_files.shape}")
display(train_files.sample(3))


train_merged = train_files.merge(labels_df, how='left', on='article_id')
train_merged['has_dataset'] = train_merged['dataset_id'] != 'Missing'
test_merged = test_files.copy()


print(f"ğŸ“� Unique train articles: {train_merged['article_id'].nunique()}")
print(f"ğŸ”– PDF files: {train_merged[train_merged['format']=='pdf']['article_id'].nunique()}")
print(f"ğŸ”– XML files: {train_merged[train_merged['format']=='xml']['article_id'].nunique()}")
print(f"ğŸ“¦ Articles with datasets: {train_merged[train_merged['has_dataset']]['article_id'].nunique()}")


example_id = train_merged['article_id'].value_counts().idxmax()  # most frequent, likely has multiple rows

# Show all rows for that article
example_rows = train_merged[train_merged['article_id'] == example_id]
print(f"All rows for article_id = {example_id}:")
display(example_rows)


# Count citation types for this article
print("Citation type counts:")
print(example_rows['type'].value_counts())


# Show grouping by dataset_id and type
print("Each unique (dataset_id, type) pair:")
for (ds_id, typ), group in example_rows.groupby(['dataset_id', 'type']):
    print(f"  Dataset: {ds_id} | Citation Type: {typ} | Rows: {len(group)}")
    display(group)
    break  # remove or adjust break to see more groups


# Let's take an example from test set as well
display(test_merged.head(3))


# Pick a test article with multiple file formats, if any
test_id = test_merged['article_id'].value_counts().idxmax()

print(f"All files for test article_id = {test_id}:")
display(test_merged[test_merged['article_id'] == test_id])


plt.figure(figsize=(7,5))
sns.countplot(data=train_merged, x='format', palette='pastel')
plt.title('Articles by Format')
plt.xlabel('Format')
plt.ylabel('Number of Articles')
plt.show()


plt.figure(figsize=(8,5))
sns.countplot(data=train_merged, x='format', hue='has_dataset', palette='Set2')
plt.title('Dataset Availability by Format')
plt.xlabel('Format')
plt.ylabel('Number of Articles')
plt.legend(title='Has Dataset', labels=['No', 'Yes'])
plt.show()


plt.figure(figsize=(8,5))
sns.countplot(data=train_merged, x='type', order=['Primary', 'Secondary', 'Missing'], palette='muted')
plt.title('Citation Type Distribution')
plt.xlabel('Citation Type')
plt.ylabel('Count')
plt.show()


type_counts = train_merged['type'].value_counts()
explode = [0.08]*len(type_counts)
plt.figure(figsize=(7,7))
plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=150, explode=explode, colors=sns.color_palette('pastel'))
plt.title('Citation Type Share')
plt.show()


ds_per_article = train_merged[train_merged['has_dataset']].groupby('article_id')['dataset_id'].count().reset_index(name='dataset_count')
multi_ds_stats = ds_per_article['dataset_count'].value_counts().sort_index()
plt.figure(figsize=(10,5))
sns.barplot(x=multi_ds_stats.index, y=multi_ds_stats.values, palette='Blues')
plt.title('Distribution: Number of Datasets per Article')
plt.xlabel('Number of Datasets')
plt.ylabel('Number of Articles')
plt.show()


print("Top 5 articles with the most datasets:")
display(ds_per_article.sort_values('dataset_count', ascending=False).head())


ds_type_dist = train_merged.groupby(['type', 'has_dataset']).size().unstack().fillna(0)
ds_type_dist.plot(kind='bar', stacked=True, figsize=(8,6), colormap='Set3')
plt.title('Dataset Availability per Citation Type')
plt.xlabel('Citation Type')
plt.ylabel('Article Count')
plt.show()


unique_datasets = train_merged[train_merged['has_dataset']]['dataset_id'].nunique()
print(f"ğŸ”� Unique datasets cited: {unique_datasets}")


top_datasets = train_merged[train_merged['has_dataset']]['dataset_id'].value_counts().head()
print("ğŸ�¯ Top 5 most frequently cited datasets:")
display(top_datasets)


summary = {
    "Total Articles": train_merged['article_id'].nunique(),
    "Total Datasets": unique_datasets,
    "Articles with Dataset": train_merged[train_merged['has_dataset']]['article_id'].nunique(),
    "Primary Citations": (train_merged['type'] == 'Primary').sum(),
    "Secondary Citations": (train_merged['type'] == 'Secondary').sum(),
    "Missing Citations": (train_merged['type'] == 'Missing').sum(),
}
print("ğŸ�… EDA Summary:")
for k, v in summary.items():
    print(f"{k}: {v}")

