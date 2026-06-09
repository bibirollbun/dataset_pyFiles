# ğŸ“¦ Step 1: Import libraries

import pandas as pd  
# For working with tabular data using DataFrames (e.g., reading CSVs, handling columns)

import numpy as np  
# Provides support for large, multi-dimensional arrays and matrices, and mathematical operations

import matplotlib.pyplot as plt  
# A core plotting library used to create static, animated, and interactive visualizations

import seaborn as sns  
# A statistical data visualization library built on top of matplotlib; helps create beautiful plots

import re  
# Regular expression operations used for text preprocessing (e.g., removing special characters or patterns)

import string  
# Provides string constants and functions, useful for removing punctuation

# Import scikit-learn modules for text processing and clustering
from sklearn.feature_extraction.text import TfidfVectorizer  
# Converts a collection of raw text documents into TF-IDF feature vectors

from sklearn.cluster import KMeans  
# A machine learning algorithm used for clustering data into groups

from sklearn.decomposition import PCA  
# Principal Component Analysis, used to reduce data dimensionality for visualization or efficiency

from sklearn.metrics import silhouette_score  
# A metric to evaluate clustering quality by measuring how well each point fits into its cluster

# Natural Language Toolkit (NLTK) for working with human language data
import nltk  
# A popular NLP library used for tasks like tokenization, stopwords, stemming, etc.

from nltk.corpus import stopwords  
# A predefined list of common stopwords (e.g., "the", "is", "and") that are usually removed in NLP

nltk.download('stopwords')  
# Downloads the stopwords corpus the first time you use it



# ğŸ“‚ Step 2: Load the dataset
df = pd.read_csv('/kaggle/input/quora-question-pairs/test.csv')


# Optional: View data types
print(df.dtypes)


df.head()


print(df[["question1", "question2"]])


df = df.dropna(subset=['question1', 'question2'])  # Remove rows with missing values


df = df.head(10000)  # For performance (optional)


# ğŸ”� Step 3: Text cleaning function

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)


# Apply cleaning
df['q1_clean'] = df['question1'].apply(clean_text)
df['q2_clean'] = df['question2'].apply(clean_text)


# âœ�ï¸� Step 4: Combine question1 and question2 into one corpus for clustering
combined_questions = pd.concat([df['q1_clean'], df['q2_clean']], ignore_index=True)


# ğŸ§® Step 5: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(combined_questions)


# ğŸ¤– Step 6: Apply K-Means Clustering

# Try different values for k
k = 10
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(X)

# Assign cluster labels
labels = kmeans.labels_

# Add to dataframe
clustered_df = pd.DataFrame({'question': combined_questions, 'cluster': labels})


# ğŸ“Š Step 7: Analyze cluster contents
for i in range(k):
    print(f"\nğŸ“Œ Cluster {i} sample questions:")
    print(clustered_df[clustered_df['cluster'] == i]['question'].head(3).to_string(index=False))



# ğŸ§ª Step 8 (Optional): Visualize with PCA (2D)

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X.toarray())

plt.figure(figsize=(10, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='tab10', s=10)
plt.title("K-Means Clustering of Questions (PCA Reduced)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Cluster")
plt.show()



# ğŸ“ˆ Step 9: Evaluate clustering with Silhouette Score
score = silhouette_score(X, labels)
print("Silhouette Score:", score)


