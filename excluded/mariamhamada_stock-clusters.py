import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




import pandas as pd
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
import nltk 

import os
import re
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk


nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


data_path = "/kaggle/input/transfer-learning-on-stack-exchange-tags/"
files = os.listdir(data_path)
print(files)



def read_csv_from_zip(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        
        file_name = z.namelist()[0]
        with z.open(file_name) as f:
            df = pd.read_csv(f)
    return df



travel_df   = read_csv_from_zip(data_path + "travel.csv.zip")
crypto_df   = read_csv_from_zip(data_path + "crypto.csv.zip")
diy_df      = read_csv_from_zip(data_path + "diy.csv.zip")
biology_df  = read_csv_from_zip(data_path + "biology.csv.zip")
robotics_df = read_csv_from_zip(data_path + "robotics.csv.zip")
cooking_df  = read_csv_from_zip(data_path + "cooking.csv.zip")


cooking_df


travel_df


crypto_df


diy_df 


biology_df


robotics_df


dfs = [travel_df, crypto_df, diy_df, biology_df, robotics_df, cooking_df]


df = pd.concat(dfs, axis=0, ignore_index=True)


df


df.shape


df.info()


df.isna().sum()


df.duplicated().sum()


df["content"].iloc[0]


df["content"].iloc[5]


df["content"].iloc[20]


df["tags"].value_counts()




def clean_text(text):
    # 1. Remove HTML tags 
    text = re.sub(r'<.*?>', '', text)
    
    # 2. Remove URLs (http/https/ftp)
    text = re.sub(r'http\S+|www\S+|ftp\S+', '', text)
    
    # 3. Remove newlines and backslashes
    text = re.sub(r'\\n', ' ', text)   # remove literal '\n'
    text = text.replace("\n", " ")     # remove actual newline characters
    
    # 4. Remove non-alphabetic characters (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 5. Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 6. Tokenize
    tokens = text.split()

    # 7. Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    # 8. Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # 9. Join back to string
    return " ".join(tokens)



max_len = df["content"].str.len().max()
print("Maximum length:", max_len)



df["title_tags"] = df["title"].astype(str) + " " + df["tags"].astype(str)

df



df.drop(["id" , "title" , "tags"] , axis=1 , inplace=True)


df


df["title_tags"].iloc[0]


df["title_tags"].iloc[19]


df["content"].iloc[7]


# Split into train/test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)



train_df.shape


train_df["title_tags"] = train_df["title_tags"].apply(clean_text)
train_df["content"] = train_df["content"].apply(clean_text)
test_df["title_tags"] = test_df["title_tags"].apply(clean_text)
test_df["content"] = test_df["content"].apply(clean_text)



train_df


train_df['content'].str.len().max()


test_df


train_df["content"]=train_df["content"]+" "+train_df["title_tags"]


train_df


train_df.drop("title_tags" , axis=1 , inplace=True)


train_df


test_df["content"]=test_df["content"]+" "+test_df["title_tags"]


test_df.drop("title_tags" , axis=1 , inplace=True)


test_df


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans



vectorizer = TfidfVectorizer(max_features=5000)  
X_train_tfidf = vectorizer.fit_transform(train_df["content"])
X_test_tfidf = vectorizer.transform(test_df["content"])


X_train_tfidf.toarray()


vectorizer.get_feature_names_out()


from sklearn.decomposition import PCA
import numpy as np

X_train_dense = X_train_tfidf.toarray()


pca = PCA(n_components=0.95, svd_solver='full')
X_pca = pca.fit_transform(X_train_dense)

print("Explained variance ratio (per component):", pca.explained_variance_ratio_)
print("Total variance explained:", np.sum(pca.explained_variance_ratio_))
print("Number of components selected:", pca.n_components_)




X_test_dense =X_test_tfidf.toarray()


X_test_pca = pca.transform(X_test_dense)




kmeans = KMeans(n_clusters=5, random_state=42)  
kmeans.fit(X_pca)


clusters = kmeans.predict(X_pca)


sil_score = silhouette_score(X_pca, clusters)
print("Silhouette Score:", sil_score)


#X_test_pca

test_clusters = kmeans.predict(X_test_pca)


sil_test_score = silhouette_score(X_test_pca, test_clusters)
print("Silhouette Score:", sil_score)


sample_data = X_pca[:500]  

linked = linkage(sample_data, method='ward')

plt.figure(figsize=(20, 8))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title("Hierarchical Clustering Dendrogram")
plt.show()


# Take a sample of first 500 points
sample_data = X_pca[:500]
sample_clusters = clusters[:500]   


plt.figure(figsize=(10, 7))
plt.scatter(sample_data[:, 0], sample_data[:, 1], 
            c=sample_clusters, cmap="tab10", s=40, alpha=0.7)


centroids_pca = kmeans.cluster_centers_
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
            c='black', s=200, marker='X', label="Centroids")

plt.title("KMeans Clustering on Reduced Data (first 500 samples)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.show()


