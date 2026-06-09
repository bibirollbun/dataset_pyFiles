# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     print(dirname)
import matplotlib.pyplot as plt
import seaborn as sns
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


taxonomy_df = pd.read_csv("/kaggle/input/birdclef-2025/taxonomy.csv")
taxonomy_df.head()


df = pd.read_csv("/kaggle/input/birdclef-2025/train.csv")
df.head()


for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")


print("Missing latitude:", df["latitude"].isna().sum())
print("Missing longitude:", df["longitude"].isna().sum())
missing_geo = df["latitude"].isna().mean() * 100
print(f"{missing_geo:.2f}% of recordings are missing location data.")


df["has_location"] = df["latitude"].notna()
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df[df["has_location"]], x="longitude", y="latitude", s=10, alpha=0.5)
plt.title("Geographic Distribution of Recordings")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()


df[["latitude", "longitude"]].describe()


rounded_locs = df[df["has_location"]].copy()
rounded_locs["lat_rounded"] = rounded_locs["latitude"].round(1)
rounded_locs["lon_rounded"] = rounded_locs["longitude"].round(1)

location_counts = rounded_locs.groupby(["lat_rounded", "lon_rounded"]).size().reset_index(name="count")

sns.histplot(location_counts["count"], bins=30, log_scale=(False, True))
plt.title("Distribution of Recordings per Geographic Region")
plt.xlabel("Number of Recordings (per ~10km grid)")
plt.ylabel("Count")
plt.show()


print("Number of unique primary labels:", df['primary_label'].nunique())

# Top and bottom species by frequency
primary_counts = df['primary_label'].value_counts()
print("\nTop 5 most frequent species:")
print(primary_counts.head())

print("\nBottom 5 least frequent species:")
print(primary_counts.tail())


print("Species with fewer than 10 samples:", (primary_counts < 10).sum())
print("Species with fewer than 50 samples:", (primary_counts < 50).sum())


import ast

df['secondary_labels'] = df['secondary_labels'].fillna("[]")
df['secondary_labels'] = df['secondary_labels'].apply(ast.literal_eval)

print("Parsed secondary_labels (first 5 rows):")
print(df['secondary_labels'].head())


df['n_secondary'] = df['secondary_labels'].apply(len)

print("\nDistribution of number of secondary labels per clip:")
print(df['n_secondary'].value_counts().sort_index())


all_secondaries = set([s for sublist in df['secondary_labels'] for s in sublist])
all_primaries = set(df['primary_label'].unique())

only_in_secondary = all_secondaries - all_primaries

print("\nSpecies that appear only as secondary labels:", len(only_in_secondary))
print("Examples:", list(only_in_secondary)[:5])


print("Unique values in 'type':", df['type'].unique()[:20])
print("\nClip count per type:")
print(df['type'].value_counts())


# Convert the stringified list into a real Python list.


import ast
from collections import Counter

df['type_parsed'] = df['type'].apply(ast.literal_eval)

type_counter = Counter([label for sublist in df['type_parsed'] for label in sublist if label.strip()])

print("Top 15 most common individual 'type' labels:")
for label, count in type_counter.most_common(15):
    print(f"{label}: {count}")


labels, counts = zip(*type_counter.most_common(15))
plt.figure(figsize=(10, 5))
plt.barh(labels[::-1], counts[::-1])
plt.xlabel("Count")
plt.title("Top 15 Individual 'Type' Labels")
plt.tight_layout()
plt.show()


print(df['rating'].describe())
print("\nUnique rating values and their counts:")
print(df['rating'].value_counts().sort_index())


rating_counts = df['rating'].value_counts().sort_index()

rating_df = pd.DataFrame(list(rating_counts.items()), columns=["Rating", "Count"])

plt.figure(figsize=(10, 6))
plt.bar(rating_df["Rating"], rating_df["Count"])
plt.xlabel("Rating")
plt.ylabel("Number of Clips")
plt.title("Distribution of Clip Ratings")
plt.xticks(rating_df["Rating"])
plt.tight_layout()
plt.show()




