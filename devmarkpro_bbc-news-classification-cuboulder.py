!pip install gensim
!pip install plotly


import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from gensim.downloader import load
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression



glove_model = load("glove-wiki-gigaword-100")


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


import os

CONFIG = {
    "train_path": "./learn-ai-bbc/BBC_News_Train.csv",
    "test_path": "./learn-ai-bbc/BBC_News_Test.csv",
    "sample_solution_path": "./learn-ai-bbc/BBC_News_Sample_Solution.csv",
    "RANDOM_STATE": 42,
    "DEFAULT_TEST_SIZE": 0.2,
    "DEFAULT_TRAIN_SIZE": 0.8
}
# change traing and test path if the notebook is running in kaggle
if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == 'Interactive':
    print("running in Kaggle")
    CONFIG["train_path"] = "/kaggle/input/learn-ai-bbc/BBC News Train.csv"
    CONFIG["test_path"] = "/kaggle/input/learn-ai-bbc/BBC News Test.csv"
    CONFIG["sample_solution_path"] = "/kaggle/input/learn-ai-bbc/BBC News Sample Solution.csv"

print(CONFIG)


def load_data(train_path: str, test_path: str, sample_solution_path: str) -> tuple:
    """
    Load the train, test, and sample solution datasets.

    Args:
        train_path (str): Path to the training dataset.
        test_path (str): Path to the test dataset.
        sample_solution_path (str): Path to the sample solution dataset.

    Returns:
        tuple: A tuple containing the train, test, and sample solution DataFrames.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_solution_df = pd.read_csv(sample_solution_path)

    return train_df, test_df, sample_solution_df


train_df, test_df, sample_solution_df = load_data(
    train_path=CONFIG["train_path"],
    test_path=CONFIG["test_path"],
    sample_solution_path=CONFIG["sample_solution_path"]
)


print(train_df.shape)
print(test_df.shape)
print(sample_solution_df.shape)


train_df.head()


test_df.head()


sample_solution_df.head()


train_df['Category'].unique()


train_df.describe()


category_counts = train_df['Category'].value_counts(normalize=True)

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Count of Categories", "Proportion of Categories")
)

fig.add_trace(
    go.Histogram(
        x=train_df['Category'],
        name="Count"
    ),
    row=1, col=1
)

fig.add_trace(
    go.Bar(
        x=category_counts.index,
        y=category_counts.values,
        name="Proportion"
    ),
    row=1, col=2
)

fig.show()


# mean, min, max, and std of the length of the text in the training set
text_length = train_df['Text'].apply(len)
print(text_length.describe())

# Visualize the distribution of text lengths in the training set
fig = px.histogram(text_length, title='Distribution of Text Lengths in Training Set')
fig.update_xaxes(title='Text Length')
fig.update_yaxes(title='Count')
fig.show()


train_df["Text"].duplicated().sum()


train_df = train_df[~train_df["Text"].duplicated()]



def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def tokenize(text):
    return word_tokenize(text)

df = train_df.copy(deep=True)
df['text'] = df['Text'].astype(str)

df['clean_text'] = df['text'].apply(clean_text)


# hanling stop words

stop_words = set(stopwords.words('english'))

def remove_stop_words(tokens):
    return [word for word in tokens if word not in stop_words]

df['tokens'] = df['clean_text'].apply(tokenize).apply(remove_stop_words)




df


# number of unique tokens in the training set
unique_tokens = set()
for tokens in df['tokens']:
    unique_tokens.update(tokens)

print(f"Number of unique tokens in the training set: {len(unique_tokens)}")


def get_document_vector(tokens, embedding_index, dim):
    """
    Get the document vector for a list of tokens using the GloVe embedding index.
    If a token is not found in the embedding index, it is ignored.
    If no tokens are found, a zero vector of the specified dimension is returned.
    Args:
        tokens (list): List of tokens (words) from the document.
        embedding_index (dict): Dictionary mapping tokens to their GloVe vectors.
        dim (int): Dimension of the GloVe vectors (default is 100).
    Returns:
        np.ndarray: A vector representing the document, averaged from the GloVe vectors of the tokens.
        If no tokens are found in the embedding index, returns a zero vector of the specified
    """
    vecs = []
    for token in tokens:
        if token in embedding_index:
            vecs.append(embedding_index[token])
    if len(vecs) > 0:
        return np.mean(vecs, axis=0)
    else:
        return np.zeros(dim)

df['doc_vector'] = df['tokens'].apply(lambda x: get_document_vector(x, glove_model, dim=100))


df


X = np.stack(df['doc_vector'].values)
print(X.shape)
pca = PCA(n_components=5, random_state=CONFIG['RANDOM_STATE'])
X_reduced = pca.fit_transform(X)

fig = px.scatter(
    x=X_reduced[:, 0],
    y=X_reduced[:, 1],
    color=df['Category'],
    labels={'x': 'PCA Component 1', 'y': 'PCA Component 2', 'color': 'Category'},
    title='PCA of GloVe-based Document Vectors',
    width=800,
    height=600
)
fig.show()


svd = TruncatedSVD(n_components=5, random_state=CONFIG['RANDOM_STATE'])
document_topic_matrix = svd.fit_transform(X)
topic_word_matrix = svd.components_

print("Document-topic matrix shape:", document_topic_matrix.shape)
print("Topic-word matrix shape:", topic_word_matrix.shape)


fig = px.scatter_matrix(
    document_topic_matrix,
    dimensions=[0, 1, 2, 3, 4],
    color=df['Category'],
    labels=dict.fromkeys(range(5), 'Component'),
    title='SVD Components of Document Vectors',
    width=800,
    height=600
)
fig.update_traces(diagonal_visible=False)
fig.show()



X_train, X_test, y_train, y_test = train_test_split(
    np.vstack(df['doc_vector'].values), df['Category'], test_size=CONFIG['DEFAULT_TEST_SIZE'], random_state=CONFIG['RANDOM_STATE']
)



svd = TruncatedSVD(n_components=5, random_state=CONFIG['RANDOM_STATE'])
X_train_svd = svd.fit_transform(X_train)
X_test_svd = svd.transform(X_test)



kmeans = KMeans(n_clusters=df['Category'].nunique(), random_state=CONFIG['RANDOM_STATE'])
train_clusters = kmeans.fit_predict(X_train_svd)
test_clusters = kmeans.predict(X_test_svd)



def map_clusters_to_labels(tc, yt):
    """
    Map clusters to labels based on the majority label in each cluster.

    Args:
        tc (np.ndarray): Array of cluster assignments for training data.
        yt (pd.Series): Series of true labels for training data.

    Returns:
        dict: Dictionary mapping clusters to their majority labels.
    """

    cluster_to_label = {}
    for cluster in np.unique(tc):
        mask = tc == cluster # Create a mask for the current cluster
        majority_label = pd.Series(yt[mask]).mode()[0] # Get the most common label in the cluster

        cluster_to_label[cluster] = majority_label
    return cluster_to_label



cluster_to_label = map_clusters_to_labels(train_clusters, y_train)
# Predict final labels
y_train_pred = np.array([cluster_to_label[c] for c in train_clusters])
y_test_pred = np.array([cluster_to_label[c] for c in test_clusters])

print("Train accuracy:", accuracy_score(y_train, y_train_pred))
print("Test accuracy:", accuracy_score(y_test, y_test_pred))


cm = confusion_matrix(y_test, y_test_pred, labels=np.unique(y_test))


# plot the confusion matrix using plotly
fig = px.imshow(
    cm,
    text_auto=True,
    labels=dict(x="Predicted Category", y="True Category", color="Count"),
    x=np.unique(y_test),
    y=np.unique(y_test),
    color_continuous_scale='YlGn',
    # aspect="auto",
    title="Confusion Matrix",
    width=600,
    height=600
)
fig.show()


from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

param_grid = {
    'n_clusters': list(range(2, 11)),
    'init': ['k-means++', 'random'],
    'max_iter': [100, 200, 300],
    'n_init': [10, 20, 30]
}

best_score = -1
best_params = None

for params in ParameterGrid(param_grid):
    kmeans = KMeans(**params, random_state=CONFIG['RANDOM_STATE'])
    labels = kmeans.fit_predict(X_train_svd)
    score = silhouette_score(X_train_svd, labels)
    if score > best_score:
        best_score = score
        best_params = params

print("Best parameters:", best_params)
print("Best silhouette score:", best_score)



# Use the best parameters to fit the KMeans model
best_kmeans = KMeans(**best_params, random_state=CONFIG['RANDOM_STATE'])
best_kmeans.fit(X_train_svd)
train_clusters = best_kmeans.predict(X_train_svd)
test_clusters = best_kmeans.predict(X_test_svd)
cluster_to_label = map_clusters_to_labels(train_clusters, y_train)
# Predict final labels
y_train_pred = np.array([cluster_to_label[c] for c in train_clusters])
y_test_pred = np.array([cluster_to_label[c] for c in test_clusters])
print("Train accuracy after hyperparameter tuning:", accuracy_score(y_train, y_train_pred))
print("Test accuracy after hyperparameter tuning:", accuracy_score(y_test, y_test_pred))

# confusion matrix after hyperparameter tuning
cm = confusion_matrix(y_test, y_test_pred, labels=np.unique(y_test))
# plot the confusion matrix using plotly
fig = px.imshow(
    cm,
    text_auto=True,
    labels=dict(x="Predicted Category", y="True Category", color="Count"),
    x=np.unique(y_test),
    y=np.unique(y_test),
    color_continuous_scale='YlGn',
    # aspect="auto",
    title="Confusion Matrix after Hyperparameter Tuning",
    width=600,
    height=600
)
fig.show()



clf = LogisticRegression(max_iter=1000, random_state=CONFIG['RANDOM_STATE'])
clf.fit(X_train, y_train)

# Predictions
y_train_pred_supervised = clf.predict(X_train)
y_test_pred_supervised= clf.predict(X_test)




print("Train accuracy:", accuracy_score(y_train, y_train_pred_supervised))
print("Test accuracy:", accuracy_score(y_test, y_test_pred_supervised))
print(classification_report(y_test, y_test_pred_supervised))


cm = confusion_matrix(y_test, y_test_pred_supervised, labels=np.unique(y_test))


# plot the confusion matrix using plotly
fig = px.imshow(
    cm,
    text_auto=True,
    labels=dict(x="Predicted Category", y="True Category", color="Count"),
    x=np.unique(y_test),
    y=np.unique(y_test),
    color_continuous_scale='YlGn',
    # aspect="auto",
    title="Confusion Matrix",
    width=600,
    height=600
)
fig.show()


train_sizes = []
accuracies = []
train_fracs = [0.01, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.90, 0.95, 0.99, 0.999, 0.9999, 0.99999]

for frac in train_fracs:
    X_partial, _, y_partial, _ = train_test_split(X_train, y_train, train_size=frac, random_state=CONFIG['RANDOM_STATE'])
    clf.fit(X_partial, y_partial)
    y_test_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_test_pred)
    train_sizes.append(frac * 100)  # percent
    accuracies.append(acc)
    print(f"Train size: {frac*100:.3f}% -> Test accuracy: {acc:.6f}")



df_plot = pd.DataFrame({
    'Train Size (%)': train_sizes,
    'Test Accuracy': accuracies
})

fig = px.line(
    df_plot,
    x='Train Size (%)',
    y='Test Accuracy',
    markers=True,
    title='Test Accuracy vs Training Data Size (Supervised Learning)',
    labels={'Train Size (%)': 'Training Data Size (%)', 'Test Accuracy': 'Test Accuracy'},
    width=800,
    height=600
)
fig.update_layout(yaxis=dict(range=[0,1]))  # fix scale to [0,1]
fig.show()

