!pip install py7zr
!pip install datasketch


import os

import pandas as pd
import py7zr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import BallTree
from datasketch import MinHash, MinHashLSHForest

from collections import defaultdict, Counter
import spacy
import string
import re

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

import warnings
warnings.filterwarnings('ignore')


PATH = "/kaggle/input/mercari-price-suggestion-challenge/"
WORKING_PATH = "/kaggle/working/"


# Extract the .7z file
with py7zr.SevenZipFile(f'{PATH}train.tsv.7z', mode='r') as z:
    z.extractall(path=WORKING_PATH)

# Read the extracted TSV file with encoding handling
train_df = pd.read_csv("/kaggle/working/train.tsv", sep="\t", encoding="utf-8")


print(f"Training Shape: {train_df.shape}")


train_df.info()


train_df.describe()


len(train_df["name"].unique())


missing_values = train_df.isnull().sum().reset_index()
missing_values.columns = ['column_name', 'missing_count']
missing_values


missing_values['missing_rate'] = (missing_values['missing_count'] / len(train_df)) * 100

missing_values = missing_values[missing_values['missing_count'] > 0]

# Plot the missing value rates
plt.figure(figsize=(10, 6))
sns.barplot(x=missing_values['column_name'], y=missing_values['missing_rate'], palette='viridis')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Missing Value Rate (%)')
plt.xlabel('Columns')
plt.title('Missing Value Rate by Column')
plt.show()


duplicate_count = train_df.duplicated().sum()
print(f"Nombre de doublons : {duplicate_count}")


train_df['name_length'] = train_df['name'].apply(lambda x: len(x) if pd.notnull(x) else 0)


train_df["name_length"].describe().head()


plt.figure(figsize=(8, 5))
plt.hist(train_df['name_length'], bins=50, color='blue', alpha=0.7)
plt.xlabel("Length of names")
plt.ylabel("Frequency")
plt.title("Distribution of name length")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


filtered_df = train_df[(train_df['name_length'] < 4) | (train_df['name_length'] > 40)][["name", "name_length"]]
filtered_df = filtered_df.sort_values(by="name_length")
filtered_df


name_length_counts = train_df['name'].str.len().value_counts().reset_index()
name_length_counts.columns = ['name_length', 'count']
name_length_counts = name_length_counts.sort_values(by="name_length")

print(name_length_counts.head(50))


missing_brand_ratio = train_df['brand_name'].isnull().sum() / len(train_df)
print(missing_brand_ratio)


distinct_brand_count = train_df['brand_name'].dropna().nunique()
print(distinct_brand_count)


# Filter non-null values in "brand_name"
filtered_df = train_df.dropna(subset=["brand_name"])

# Count occurrences and sort in descending order
brand_counts = filtered_df["brand_name"].value_counts().reset_index()

# Rename columns for clarity
brand_counts.columns = ["brand_name", "count"]

# Display the top 10 most frequent brands
print(brand_counts.head(10))


# Show first 50
print(brand_counts.head(50))

plt.figure(figsize=(10, 6))
sns.barplot(x='count', y='brand_name', data=brand_counts.head(20))
plt.title('Top 20 most common brands')
plt.xlabel('Number of occurrences')
plt.ylabel('Brand')
plt.show()


unique_brands = brand_counts["brand_name"].unique()
print(len(unique_brands))
print(unique_brands[20:50])


# Filter rows where "name" and "brand_name" are not null
filtered_df = train_df.dropna(subset=["name", "brand_name"])
filtered_df["name_length"] = filtered_df["name"].str.split().str.len()

# Count the number of products per brand and get the top 20 most frequent ones
top_20_brands = filtered_df["brand_name"].value_counts().nlargest(20).index

# Filter name lengths for the top 20 brands
filtered_lengths = {brand: filtered_df.loc[filtered_df["brand_name"] == brand, "name_length"] for brand in top_20_brands}

colors = plt.cm.get_cmap("tab20", len(top_20_brands)).colors

# Plot the histogram
plt.figure(figsize=(12, 6))
for i, (brand, lengths) in enumerate(filtered_lengths.items()):
    plt.hist(lengths, bins=50, alpha=0.6, label=brand, color=colors[i])

plt.title('Distribution of Product Name Lengths (Top 20 Brands)')
plt.xlabel('Product Name Length')
plt.ylabel('Frequency')
plt.legend(loc="upper right", fontsize="small", ncol=2)
plt.grid(True)
plt.show()


duplicates = train_df[train_df['name'].duplicated(keep=False)]
print(f"Duplicate values : {len(duplicates)}")
print(duplicates[['name', 'brand_name']].head(10))


train_df[train_df["name"]== 'Younique 3d fiber lash mascara'][['name','train_id']]


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(train_df.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.title("Missing Values Before Cleaning")

duplicates_before = train_df[train_df['name'].duplicated(keep=False)].shape[0]

train_df_cleaned = train_df.dropna().drop_duplicates(subset=["name"])

plt.subplot(1, 2, 2)
sns.heatmap(train_df_cleaned.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.title("Missing Values After Cleaning")

plt.show()

print(f"Duplicated rows before: {duplicates_before}")
print(f"Duplicated rows after: {train_df_cleaned['name'].duplicated().sum()}")


def clean_text(text):
    # Initialize a list to store cleaned words
    words = []

    # Process the sentence with spaCy
    doc = nlp(text.lower())  # Convert to lowercase + spaCy processing

    for token in doc:
        lemma = token.lemma_  # Get the base form (lemma) of the word

        # Check the conditions for keeping the word
        if (
            lemma not in nlp.Defaults.stop_words  # Remove stopwords
            and lemma not in string.punctuation  # Remove punctuation
            and not re.search(r'\d', lemma)  # Remove numbers
            and len(lemma) > 2  # Remove very short words (e.g., "t'", "d'")
            and token.pos_ not in ["PRON", "DET", "ADP", "CONJ"]  # Remove pronouns, determiners, prepositions, conjunctions
        ):
            words.append(lemma)  # Store the cleaned word

    # Join the cleaned words back into a string
    return ' '.join(words)


train_df_cleaned['cleaned_name'] = train_df_cleaned['name'].apply(clean_text)
train_df_cleaned.head()


words_list = [word for sentence in train_df_cleaned['cleaned_name'].tolist() for word in sentence.split()]

word_freq = Counter(words_list)

pd.DataFrame(word_freq.most_common(20), columns=["Word", "Frequency"])\
.set_index("Word")\
.plot(kind="bar", figsize=(10, 5), color="royalblue")

plt.title("Top 20 most frequent words")
plt.xlabel("Word")
plt.ylabel("Frequency")
plt.show()


top_20_brands = train_df_cleaned['brand_name'].value_counts().head(20)
print(top_20_brands)


for brand_name in list(top_20_brands.keys()):
    text = ' '.join(train_df_cleaned[train_df_cleaned['brand_name'] == brand_name]['cleaned_name'])
    print
    # Generate the word cloud
    wordcloud = WordCloud(collocations=False, width=800, height=400).generate(text)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for Intent: {brand_name}")
    plt.show()


def create_ngrams(string: str):
    result = []
    for n in range(3, 4):  # Using 3-grams (you can adjust n-range for different n-grams)
        ngrams = zip(*[string[i:] for i in range(n)])
        ngrams = [''.join(ngram) for ngram in ngrams if ' ' not in ngram]  # Skip ngrams with spaces
        result.extend(ngrams)
    return result

test_string = "mercari"
output = create_ngrams(test_string)
print("Generated N-grams:", output)


# Vectorization function for each brand
def vectorize(brand: str, df: pd.DataFrame):
    corpus = df[df['brand_name'] == brand]["cleaned_name"].tolist()
    
    # Check if the corpus is empty or contains very little data
    if not corpus or all(len(text.strip()) == 0 for text in corpus):
        print(f"Brand '{brand}' has no data or empty text for vectorization.")
        return None, corpus
    
    # Apply TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words='english')  
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
        return tfidf_matrix, corpus
    except ValueError as e:
        print(f"Error while vectorizing brand '{brand}': {str(e)}")
        return None, corpus


# Count the number of rows for each brand
brand_counts = train_df_cleaned['brand_name'].value_counts()

# Filter for brands with at least 1000 rows
valid_brands = brand_counts[brand_counts >= 1000]

# Select the two brands with the fewest rows
top_2_brands = valid_brands.nsmallest(2).index

# Filter the DataFrame to include only these two brands
train_df_cleaned = train_df_cleaned[train_df_cleaned['brand_name'].isin(top_2_brands)]


vbrand_list = list(train_df_cleaned['brand_name'].unique())
vnames = {}

for brand in vbrand_list:
    tfidf_matrix, corpus = vectorize(brand, train_df_cleaned)  
    vnames[brand] = {
        'tfidf_matrix': tfidf_matrix,  
        'corpus': corpus  
    }

df_vnames = pd.DataFrame.from_dict(vnames, orient='index')  
df_vnames.head()


def find_similar_names(brand: str, df: pd.DataFrame, top_k: int):
    tfidf_matrix, corpus = df_vnames.loc[brand, "tfidf_matrix"], df_vnames.loc[brand, "corpus"]
    
    # If the tfidf_matrix is None, return an empty DataFrame or handle accordingly
    if tfidf_matrix is None:
        print(f"Warning: TF-IDF matrix for brand '{brand}' is None.")
        return pd.DataFrame()  # Return an empty DataFrame
    
    train_ids = df[df['brand_name'] == brand]['train_id'].tolist()

    # Check if top_k is greater than the number of corpus items
    if top_k > len(corpus):
        print(f"Warning: top_k ({top_k}) is greater than the number of available items ({len(corpus)}). Adjusting top_k to {len(corpus) - 1}.")
        top_k = len(corpus) - 1  # Ensure top_k doesn't exceed the number of items
    tree = BallTree(tfidf_matrix.toarray())
    
    distances, indices = tree.query(tfidf_matrix.toarray(), k=top_k+1)

    results = []
    for idx, name in enumerate(corpus):
        for j in range(1, top_k + 1):  
            i = indices[idx, j]
            results.append({
                'name1': name,
                'id1': train_ids[idx],
                'name2': corpus[i],
                'id2': train_ids[i],
                'cosine_similarity': 1 - (distances[idx, j] ** 2) / 2,  # Cosine similarity
                'brand_name': brand
            })

    return pd.DataFrame(results)


all_results = []

for brand in train_df_cleaned['brand_name'].unique():
    top_k = 5  
    df_similar = find_similar_names(brand,train_df_cleaned, top_k)
    all_results.append(df_similar)

final_df = pd.concat(all_results, ignore_index=True)
final_df.to_csv('/kaggle/working/results_balltree.csv', index=False)
print(final_df.head())


def jaccard_similarity(set1, set2):
    """Calculates the Jaccard similarity between two sets."""
    return len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0
    
# Test the jaccard_similarity function
set1 = set("Acme Corp".split())
set2 = set("Acme Corporation".split())
print("Jaccard Similarity:", jaccard_similarity(set1, set2))


def get_minhash(text, num_perm=128):
    """Generates a MinHash signature for a given text."""
    m = MinHash(num_perm=num_perm)
    for shingle in text.split():  # Using words as shingles
        m.update(shingle.encode('utf8'))
    return m
# Test the get_minhash function
text = "Acme Corp"
minhash_signature = get_minhash(text)
print("MinHash Signature:", minhash_signature)


def find_similar_names(brand: str, df: pd.DataFrame, top_k: int, num_perm=128):
    """Finds similar names using MinHash LSH Forest."""
    brand_df = df[df['brand_name'] == brand]
    cleaned_names = brand_df['cleaned_name'].tolist()
    train_ids = brand_df['train_id'].tolist()

    # Build the LSH Forest
    lsh_forest = MinHashLSHForest(num_perm=num_perm)
    minhash_dict = {}

    for idx, name in enumerate(cleaned_names):
        minhash = get_minhash(name, num_perm)
        lsh_forest.add(idx, minhash)
        minhash_dict[idx] = minhash  # Store the MinHashes for precise Jaccard calculation

    lsh_forest.index()

    results = []
    for idx, name in enumerate(cleaned_names):
        minhash = minhash_dict[idx]
        similar_idxs = lsh_forest.query(minhash, top_k + 1)  # Includes the name itself

        for i in similar_idxs:
            if i != idx:  # Exclude the name itself
                sim_score = jaccard_similarity(set(cleaned_names[idx].split()), set(cleaned_names[i].split()))
                results.append({
                    'name1': name,
                    'id1': train_ids[idx],
                    'name2': cleaned_names[i],
                    'id2': train_ids[i],
                    'jaccard_similarity': sim_score,
                    'brand_name': brand
                })

    return pd.DataFrame(results)


all_results = []
for brand in train_df_cleaned['brand_name'].unique():
    top_k = 5  
    df_similar = find_similar_names(brand, train_df_cleaned[:100000], top_k)
    all_results.append(df_similar)

# ConcatÃ©nation des rÃ©sultats
final_df = pd.concat(all_results, ignore_index=True)
final_df.to_csv('/kaggle/working/results_lsh.csv', index=False)
print(final_df.head())

