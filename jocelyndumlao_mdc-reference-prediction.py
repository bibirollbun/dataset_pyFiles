import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')


# Load data
train = pd.read_csv('/kaggle/input/make-data-count-finding-data-references/train_labels.csv')
print("Train shape:", train.shape)
print(train.dtypes)
train.head()


import matplotlib.pyplot as plt
import seaborn as sns

#sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 14, 'font.weight': 'bold'})

fig, axes = plt.subplots(2, 3, figsize=(22, 14))
fig.suptitle("Exploratory Data Analysis on Training Labels", fontsize=20, fontweight='bold')

# 1. Primary vs Secondary Types
sns.countplot(data=train, x='type', ax=axes[0, 0], palette='coolwarm')
axes[0, 0].set_title("Type Distribution", color='navy')
axes[0, 0].set_facecolor('#f0f8ff')

# 2. Unique Datasets per Article
article_dataset_counts = train.groupby('article_id')['dataset_id'].nunique()
sns.histplot(article_dataset_counts, bins=30, ax=axes[0, 1], color='skyblue')
axes[0, 1].set_title("Unique Dataset Count per Article", color='darkgreen')
axes[0, 1].set_facecolor('#f5f5dc')

# 3. Top 10 Most Referenced Datasets
top_datasets = train['dataset_id'].value_counts().head(10)
sns.barplot(y=top_datasets.index, x=top_datasets.values, ax=axes[0, 2], palette='viridis')
axes[0, 2].set_title("Top 10 Referenced Datasets", color='purple')
axes[0, 2].set_facecolor('#fdf6e3')

# 4. Dataset Prefix Distribution (e.g., zenodo, figshare)
train['prefix'] = train['dataset_id'].str.extract(r'https://doi.org/(.*?)/')
prefix_counts = train['prefix'].value_counts().head(10)
sns.barplot(x=prefix_counts.values, y=prefix_counts.index, ax=axes[1, 0], palette='cubehelix')
axes[1, 0].set_title("Top DOI Prefixes", color='darkblue')
axes[1, 0].set_facecolor('#f0fff0')

# 5. Dataset Reuse Count Distribution
reuse_counts = train['dataset_id'].value_counts()
sns.histplot(reuse_counts, bins=30, ax=axes[1, 1], color='tomato')
axes[1, 1].set_title("Dataset Reuse Frequency", color='darkred')
axes[1, 1].set_facecolor('#fefbd8')

# 6. Articles Mentioning Both Types
type_counts = train.groupby('article_id')['type'].nunique()
both_types_articles = (type_counts[type_counts > 1].count())
one_type_articles = (type_counts[type_counts == 1].count())
sns.barplot(x=['One Type', 'Both Primary & Secondary'], y=[one_type_articles, both_types_articles], ax=axes[1, 2], palette='Accent')
axes[1, 2].set_title("Articles Mentioning One vs Both Types", color='teal')
axes[1, 2].set_facecolor('#e6f2ff')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


train['doi_length'] = train['dataset_id'].apply(lambda x: len(str(x)))
plt.figure(figsize=(10, 6))
sns.histplot(train['doi_length'], bins=30, color='orchid', kde=True)
plt.title("DOI String Length Distribution", fontsize=16, fontweight='bold', color='darkmagenta')
plt.xlabel("Length of dataset_id string")
plt.ylabel("Frequency")
plt.gca().set_facecolor('#fff0f5')
plt.show()



print("ğŸ”¢ Top 5 DOI Prefixes:")
print(prefix_counts.head())

print("\nğŸ”� Average Dataset Reuse Count:", reuse_counts.mean())
print("ğŸ“š Articles with Both Types:", both_types_articles)
print("ğŸ“š Articles with Only One Type:", one_type_articles)



import glob
import xml.etree.ElementTree as ET
import re

# Helper: convert to full DOI format
def normalize_doi(raw_doi):
    if raw_doi.startswith("http"):
        return raw_doi
    return f"https://doi.org/{raw_doi}"

# Collect predictions
test_xmls = glob.glob("/kaggle/input/make-data-count-finding-data-references/test/XML/*.xml")

predictions = []
for path in test_xmls:
    article_id = os.path.basename(path).replace('.xml', '')
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        text = ET.tostring(root, encoding='unicode')

        found_dois = set(re.findall(r'(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', text, re.IGNORECASE))
        for doi in found_dois:
            full_doi = normalize_doi(doi)
            predictions.append((article_id, full_doi, "Primary"))  # Default heuristic type
    except:
        print(f"Error in parsing: {article_id}")

pred_df = pd.DataFrame(predictions, columns=['article_id', 'dataset_id', 'type'])
print("Prediction sample:")
pred_df.head()


pred_df = pred_df.drop_duplicates()
pred_df = pred_df.reset_index().rename(columns={"index": "row_id"})

submission_path = "/kaggle/working/submission.csv"
pred_df.to_csv(submission_path, index=False)

print("Submission file saved. Preview:")
pred_df.head()



# Plot: Number of predictions per article
pred_counts = pred_df['article_id'].value_counts()
plt.figure(figsize=(10, 6))
sns.histplot(pred_counts, bins=30, color='orange', kde=True)
plt.title("Predicted Dataset Counts per Article", fontsize=16, fontweight='bold', color='darkred')
plt.xlabel("Predicted Datasets per Article", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.gca().set_facecolor('#fff5ee')
plt.show()




