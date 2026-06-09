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


import pandas as pd
import numpy as np
import re
import nltk
import spacy
import string
import matplotlib.pyplot as plt
import seaborn as sns
#plotly
import plotly.offline as py
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
from plotly import tools
init_notebook_mode(connected=True)
import plotly.figure_factory as ff
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


train=pd.read_csv('../input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip',compression='zip',
                 header=0,delimiter='\t',quoting=0, doublequote=False, escapechar='\\')
train.head()


test=pd.read_csv('../input/word2vec-nlp-tutorial/testData.tsv.zip', compression='zip',
                header=0, delimiter='\t', quoting=0)
test.head()


unlabeled=pd.read_csv('../input/word2vec-nlp-tutorial/unlabeledTrainData.tsv.zip', compression='zip',
                 header=0,delimiter='\t',quoting=0, doublequote=False, escapechar='\\')
unlabeled.head()


def get_database_info(df):
    # first view of the database
    
    print("No of columns of database", df.shape[1])
    print("No of rows ", df.shape[0])
    print("Names of the columns", df.columns)
    print("Missing value counts", df.isnull().value_counts())
    print(df.describe().T)
    print(df.info())
    return df.head(5)
    


def calculate_missing_percentage(df):
    missing_stats=df.isnull().sum()/len(df)*100
    prod_count=pd.DataFrame(missing_stats.sort_index())
    plt.figure()
    # plot in barplot
    
    sns.barplot(x=missing_stats.index, y=missing_stats.values, alpha=0.8)
    plt.title("Percent Missing")
    plt.ylabel("Missing", fontsize=12)
    plt.xlabel("Feature", fontsize=12)
    plt.xticks(rotation=90)
    plt.show()
    


def draw_num_plot(df, column):
    # to draw the KDE plot
    
    plt.figure(figsize=(10,10))
    col = column
    grouped = df[col].value_counts().reset_index()
    grouped = grouped.rename(columns = {col : "count", "index" : col})

    ## plot
    trace = go.Pie(labels=grouped[col], values=grouped['count'], pull=[0.05, 0], marker=dict(colors=["#6ad49b", "#a678de"]))
    layout = go.Layout(title="", height=600, legend=dict(x=0.1, y=1.1))
    fig = go.Figure(data = [trace], layout = layout)
    iplot(fig)


def histogram_plot(df, col):
    df[col].plot(
    kind='hist',
    bins=50,
    title='Reviewers Age Distribution')
    
    


plt.style.use('seaborn-darkgrid')

orange_black = ['#fdc029', '#df861d', 'FF6347', '#aa3d01',
                '#a30e15', '#800000', '#171820']

plt.rcParams['figure.figsize'] = (10,5) 
plt.rcParams['figure.facecolor'] = '#FFFACD' 
plt.rcParams['axes.facecolor'] = 'FFFFE0' 
plt.rcParams['axes.grid'] = True 
plt.rcParams['grid.color'] = orange_black[3]
plt.rcParams['grid.linestyle'] = '--'


get_database_info(train)
calculate_missing_percentage(train)
draw_num_plot(train, 'sentiment')
histogram_plot(train, 'sentiment')


get_database_info(test)
calculate_missing_percentage(test)


get_database_info(unlabeled)
calculate_missing_percentage(unlabeled)


train['review_lower']=train['review'].str.lower()
train.head()


punc_to_remove=string.punctuation

def remove_punctuation(text):
    return text.translate(str.maketrans('','',punc_to_remove))

train["text_to_punc"]=train['review_lower'].apply(lambda text: remove_punctuation(text))



train.head()


from nltk.corpus import stopwords
",".join(stopwords.words("english"))


STOPWORDS=set(stopwords.words("english"))

def remove_stopword(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

train['text_to_stop']=train['review'].apply(lambda text: remove_stopword(text))
train.head()



train.drop(['review_lower', 'text_to_punc'], axis=1, inplace=True)


train.head()


train.head()


from collections import Counter
cnt=Counter()

for text in train['text_to_stop'].values:
    for word in text.split():
        cnt[word]+=1
        
cnt.most_common(10)


FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])
def remove_freqwords(text):
    """custom function to remove the frequent words"""
    return " ".join([word for word in str(text).split() if word not in FREQWORDS])

train["text_wo_stopfreq"] = train["text_to_stop"].apply(lambda text: remove_freqwords(text))
train.head()



n_rare_word=10
RAREWORDS=set([w for (w ,wc) in cnt.most_common()[:-n_rare_word-1:-1]])

def remove_rareword(text):
    return " ".join([word for word in str(text).split() if word not in RAREWORDS])

train['rare_text']=train['text_wo_stopfreq'].apply(lambda text: remove_rareword(text))
train.head(10)


from nltk.stem.porter import PorterStemmer

train.drop(["text_wo_stopfreq","rare_text"], axis=1, inplace=True)


train.head()


from nltk.stem.snowball import SnowballStemmer
SnowballStemmer.languages


from nltk.stem import WordNetLemmatizer

lematizer=WordNetLemmatizer()

def lemmatizer_words(text):
    return " ".join([lematizer.lemmatize(word) for word in text.split()])

train['lemma_text']=train['review'].apply(lambda text: lemmatizer_words(text))
train.head()


train['lemma_text'][0]


from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

lemmmatizer=WordNetLemmatizer()

wordnet_map={"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
def lemmatized_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lematizer.lemmatize(word , wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

train["text_lemmatized"] = train["review"].apply(lambda text: lemmatized_words(text))
train.head()


train['text_lemmatized'][0]


import re

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

train['text_to_html']=train['review'].apply(lambda text: remove_tags(text))
train.head()


train['text_to_html'][0]


def remove_urld(text):
    url_pattern=re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

train['no_url']=train['review'].apply(lambda text: remove_urld(text))    
train.head()


train['no_url'][0]


train['no_url'][1]


# https://gist.github.com/nealrs/96342d8231b75cf4bb82 
cList = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}


c_re = re.compile('(%s)' % '|'.join(cList.keys()))
def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text)


train['decontracted_word']=train['review'].apply(lambda text: expandContractions(text, c_re=c_re))
train.head()


from wordcloud import WordCloud

# Thanks : https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##
def plot_wordcloud(text, mask=None, max_words=1000, max_font_size=100, figure_size=(14.0,16.0), 
                   title = None, title_size=40, image_color=False):
    

    wordcloud = WordCloud(background_color='black',max_words = max_words,
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
    
plot_wordcloud(train["text_to_html"], title="Word Cloud of Review")


def plot_wordcloud(text, mask=None, max_words=400, max_font_size=120, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='white',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    mask = mask)
    wordcloud.generate(text)
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'green', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  
    
d = '../input/masks/masks-wordclouds/'


comments_text = str(train.text_to_html)
comments_mask = np.array(Image.open(d + 'comment.png'))
plot_wordcloud(comments_text, comments_mask, max_words=400, max_font_size=120, 
               title = 'Most common words in all of the Review', title_size=50)


n = round(train.shape[0]*0.01)
top_recommended_comments_text = str(train.nlargest(n, columns='sentiment').text_to_html)
upvote_mask = np.array(Image.open(d + 'upvote.png'))
plot_wordcloud(top_recommended_comments_text, upvote_mask, max_words=300000, max_font_size=300,
               title = 'Most common words in the top 1% most upvoted comments')

