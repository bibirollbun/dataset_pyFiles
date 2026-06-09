# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go

import plotly.io as pio 
pio.renderers.default = 'iframe'

import plotly
plotly.offline.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px


import warnings
warnings.simplefilter(action='ignore', category=Warning)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install xlrd


!pip install openpyxl


df1 = pd.read_csv('/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv')
df1.tail()


df = pd.read_excel('/kaggle/input/widsdatathon2025/Data Dictionary.xlsx')
df.head()


apq = pd.read_excel('/kaggle/input/apq-only/APQ only.xlsx')
apq.head()


#By SRK https://www.kaggle.com/code/sudalairajkumar/simple-exploration-notebook-qiqc

from wordcloud import WordCloud, STOPWORDS

# Thanks : https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##
def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='white',
                    color_func=lambda *args, **kwargs: "black",       
                    stopwords = stopwords,
                       max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    width=800, 
                    height=400,
                    mask = mask)
    wordcloud.generate(str(text))
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'black', 
                                          'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  
    
plot_wordcloud(apq["APQ"], title="Academic Performance Questionnaire")


import plotly.io as pio
pio.renderers.default = 'iframe'


#By SRK https://www.kaggle.com/code/sudalairajkumar/simple-exploration-notebook-qiqc

from wordcloud import WordCloud, STOPWORDS

from plotly import tools

from collections import defaultdict
#train1_df = train_df[train_df["target"]==1]
#train0_df = train_df[train_df["target"]==0]

## custom function for ngram generation ##
def generate_ngrams(text, n_gram=1):
    token = [token for token in str(text).lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

## custom function for horizontal bar chart ##
def horizontal_bar_chart(df, color):
        trace = go.Bar(
        y=df["word"].values[::-1],
        x=df["wordcount"].values[::-1],
        showlegend=False,
        orientation = 'h',
        marker=dict(
            color=color,
        ),
    )
        return trace

## Get the bar chart from sincere questions ##
freq_dict = defaultdict(int)
for sent in apq["APQ"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(50), 'black')
# Creating two subplots
fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,
                          subplot_titles=["Frequent words of Academic Performance Questionnaire"]) 
                                         # "Frequent words of GPT Values"])
fig.append_trace(trace0, 1, 1)
#fig.append_trace(trace1, 1, 2)
fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
py.iplot(fig, filename='word-plots')


ehq = pd.read_excel('/kaggle/input/apq-only/EHQ.xlsx')
ehq.head()


sdq = pd.read_excel('/kaggle/input/apq-only/SDQ.xlsx')
sdq.head()


def fix(arg: str) -> str:
    for key, value in {'4 k': '4k', '8 k': '8k'}.items():
        arg = arg.replace(key, value)
    return arg

apq['APQ'] = apq['APQ'].apply(fix)


from matplotlib.pyplot import subplots
from matplotlib.pyplot import axis
from matplotlib.pyplot import imshow
from wordcloud import WordCloud
from wordcloud import STOPWORDS

FRACTION = 0.2
subplots(figsize=(12, 12))
text = ' '.join(apq.sample(frac=FRACTION, random_state=2023)['APQ'].values.tolist())
imshow(X=WordCloud(random_state=2023, height=1200, width=1200,background_color = '#66CDAA',colormap= "Purples", stopwords=STOPWORDS,).generate(text=text), )
axis('off')


from collections import Counter
from plotly.express import bar
count_df = pd.DataFrame.from_dict(Counter(text.split(',')), orient='index').reset_index().sort_values(ascending=False, by=0)
bar(data_frame=count_df.head(n=50), x='index', y=0)


%env TOKENIZERS_PARALLELISM=true
! pip install sentence-transformers


from arrow import now
from sentence_transformers import SentenceTransformer

# we can't use more features than we can visualize
MAX_FEATURES = 20 #Original was 300 Try what number works 
SAMPLE_SIZE = 43  #Original was 10000

model_start = now()
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
# we need to encode the essays to get the words' relationships to each other
embedding = model.encode(apq.sample(n=SAMPLE_SIZE, random_state=2023)['APQ'].values.tolist())
print('{}: got embeddings'.format(now()))
features = count_df.head(n=MAX_FEATURES)['index'].values.tolist()
feature_embeddings = model.encode(features)
print('model time: {}'.format(now() - model_start))


#Mike Delong https://www.kaggle.com/code/mikedelong/parse-prompts-with-sentencetransformer

# now we can project the feature vectors into 2-space to visualize 
from pandas import DataFrame
from plotly.express import scatter
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=2025, verbose=1, n_iter=250, perplexity = min(50, len(features)-1))
tsne_df = DataFrame(data=tsne.fit_transform(X=feature_embeddings), columns=['t0', 't1'])
tsne_df['word'] = features
tsne_df['count'] = count_df.head(n=MAX_FEATURES)[0].values.tolist()
scatter(data_frame=tsne_df, x='t0', y='t1', text='word', height=900, hover_data=['count'] ).update_traces(marker={'size': 1})


#https://scipy-lectures.org/packages/scikit-learn/

print(apq.shape)

n_samples, n_features = apq.shape
print(n_samples)

print(n_features)


!pip install sentence_transformers umap-learn -q


! pip install -q -U sentence-transformers
! pip install -q -U watermark
! pip install -q -U cluestar

import umap

from sentence_transformers import SentenceTransformer
from cluestar import plot_text
import matplotlib.pyplot as plt


#Binga https://www.kaggle.com/code/phanisrikanth/daigt-cluster-explore-7-prompts-dataset

# Convert essays to embeddings using sentence transformers library.
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
# model = SentenceTransformer('all-mpnet-base-v2', device='cuda')
# model = SentenceTransformer('/kaggle/input/thenlper-gte-large/').to("cuda:0") # use this if needed.

#Let's encode the input from training data.
train_embeddings = model.encode(apq['APQ'])


#Binga https://www.kaggle.com/code/phanisrikanth/daigt-cluster-explore-7-prompts-dataset

train_embeddings.shape


#Binga https://www.kaggle.com/code/phanisrikanth/daigt-cluster-explore-7-prompts-dataset

# Build a UMAP representation of the essay embeddings.
model = umap.UMAP(random_state=42)
train_umap_embeddings = model.fit_transform(train_embeddings)


#Binga https://www.kaggle.com/code/phanisrikanth/daigt-cluster-explore-7-prompts-dataset

# Plot the umap embeddings in 2D space. Add a legend with blues representing student essays and 1s representing AI generated essays.
plot_text(train_umap_embeddings, apq['APQ'], color_array=list(apq['APQ'].astype(str)))

