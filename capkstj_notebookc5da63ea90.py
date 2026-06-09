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


# from datasets import load_dataset

# ds = load_dataset("Yelp/yelp_review_full")


import pandas as pd
from datasets import load_dataset

# Load Yelp Review Full dataset
# Load Yelp dataset
yelp_dataset = load_dataset("Yelp/yelp_review_full")
yelp_df = pd.DataFrame(yelp_dataset['train'])
yelp_df = yelp_df[['text', 'label']]
yelp_df['source'] = 'Yelp'
yelp_df



# Load Sentiment Analysis on Movie Reviews dataset
# Assuming you have a TSV file named 'train.tsv'
amazon_df = pd.read_json('/kaggle/input/combine/Cell_Phones_and_Accessories_5.json', lines=True)
amazon_df = amazon_df[['reviewText', 'overall']]
amazon_df.rename(columns={'reviewText': 'text', 'overall': 'label'}, inplace=True)
amazon_df['label'] = amazon_df['label'].astype(int) - 1
amazon_df['source'] = 'Amazon'
amazon_df



# Load Amazon Product Reviews dataset
# Assuming you have a JSON lines file named 'amazon_reviews.jsonl'
kaggle_df = pd.read_csv('/kaggle/input/combine/train.tsv', sep='\t')
kaggle_df = kaggle_df[['Phrase', 'Sentiment']]
kaggle_df.rename(columns={'Phrase': 'text', 'Sentiment': 'label'}, inplace=True)
kaggle_df['source'] = 'rottenTomato'
kaggle_df


for df in [yelp_df, amazon_df, kaggle_df]:
    df.dropna(subset=['text', 'label'], inplace=True)

def balance_labels(df):
    label_counts = df['label'].value_counts()
    min_count = label_counts.min()
    balanced_df = df.groupby('label').apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)
    return balanced_df
yelp_df_balanced = balance_labels(yelp_df)
amazon_df_balanced = balance_labels(amazon_df)
kaggle_df_balanced = balance_labels(kaggle_df)

# Combine all balanced datasets
combined_df = pd.concat([yelp_df_balanced, amazon_df_balanced, kaggle_df_balanced], ignore_index=True)


# Optional: remove duplicates
combined_df.drop_duplicates(subset=['text', 'label', 'source'], inplace=True)
# Optional: Shuffle the combined dataset
combined_df = combined_df.sample(frac=1).reset_index(drop=True)
combined_df


import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)  # sharey=False to allow independent y-axis scaling

datasets = [yelp_df, amazon_df, kaggle_df]
titles = ['Yelp Label Distribution', 'Amazon Label Distribution', 'RottonTomato Label Distribution']

for ax, data, title in zip(axes, datasets, titles):
    sns.countplot(data=data, x='label', ax=ax)
    ax.set_title(title)
    # Set y-limit slightly above the max count
    counts = data['label'].value_counts()
    ax.set_ylim(0, counts.max() * 1.1)  # 10% padding above max count

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)  # sharey=False to allow independent y-axis scaling

datasets = [yelp_df_balanced, amazon_df_balanced, kaggle_df_balanced]
titles = ['Yelp Label Distribution', 'Amazon Label Distribution', 'RottonTomato Label Distribution']

for ax, data, title in zip(axes, datasets, titles):
    sns.countplot(data=data, x='label', ax=ax)
    ax.set_title(title)
    # Set y-limit slightly above the max count
    counts = data['label'].value_counts()
    ax.set_ylim(0, counts.max() * 1.1)  # 10% padding above max count

plt.tight_layout()
plt.show()



combined_df.to_csv("/kaggle/working/normalized_reviews.csv", index=False)


%cd /kaggle/working
from IPython.display import FileLink
FileLink('a/b/c.tgz')

