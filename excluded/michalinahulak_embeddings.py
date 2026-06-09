pip install -U sentence-transformers


import pandas as pd
import numpy as np
import warnings

from transformers import BertTokenizer, BertModel
import torch
from sentence_transformers import SentenceTransformer

from sklearn.manifold import TSNE
import plotly.express as px

warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


model = SentenceTransformer('all-MiniLM-L6-v2')


def preprocess_data(df):
    df['Podcast_Episode'] = df['Podcast_Name'] + " - " + df['Episode_Title']
    
    texts = df['Podcast_Episode'].tolist()
    
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
    
    return df, embeddings

def reduce_dimensions(embeddings, n_components=2):
    tsne = TSNE(n_components=n_components, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    return reduced_embeddings

def plot_interactive_tsne(reduced_embeddings, df):
    df['x'] = reduced_embeddings[:, 0]
    df['y'] = reduced_embeddings[:, 1]
    fig = px.scatter(df, x='x', y='y', hover_data={'Podcast_Episode': True}, 
                     title="t-SNE of Podcast Embeddings", labels={'Podcast_Episode': 'Podcast Name'})
    fig.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=2, color='DarkSlateGrey')))
    fig.show()


train, embeddings = preprocess_data(train)


embeddings


embeddings_df = pd.DataFrame(embeddings)
embeddings_df['id'] = train['id'].values  
embeddings_df.to_csv('podcast_embeddings_train.csv', index=False)


# reduced_embeddings = reduce_dimensions(embeddings)
# plot_interactive_tsne(reduced_embeddings, train)


test, embeddings_test = preprocess_data(test)

embeddings_test


embeddings_test = pd.DataFrame(embeddings_test)
embeddings_test['id'] = test['id'].values  
embeddings_test.to_csv('podcast_embeddings_test.csv', index=False)

