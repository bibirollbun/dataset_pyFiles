import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
import json
from tqdm import tqdm
tqdm.pandas()
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

import os
import gc
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import gensim
from sklearn.model_selection import KFold

from IPython.display import SVG
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import svm
from scipy.sparse import hstack
from collections import defaultdict
import plotly.graph_objects as gobs

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB

from tqdm.auto import tqdm
from bs4 import BeautifulSoup
from collections import defaultdict
import re 
import scipy
from scipy import sparse
from IPython.display import display
from pprint import pprint
from matplotlib import pyplot as plt 
import time
import scipy.optimize as optimize
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn.linear_model import Ridge
import zipfile
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import string
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer() 
nltk.download('wordnet')



# åŠ è½½æ•°æ�®é›†
val= pd.read_csv('/kaggle/input/jigsaw-toxic-severity-rating/validation_data.csv')
train_csv_zip_path = '/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip'
with zipfile.ZipFile(train_csv_zip_path) as zf:
    zf.extractall('./')
train_csv_path = './train.csv'
train_csv_path = './train.csv'
sample_sub_path = '/kaggle/input/jigsaw-toxic-severity-rating/sample_submission.csv'
comments_to_score_path = '/kaggle/input/jigsaw-toxic-severity-rating/comments_to_score.csv'
val_path='/kaggle/input/jigsaw-toxic-severity-rating/validation_data.csv'
df_train = pd.read_csv("./train.csv")
df_train1 = pd.read_csv("./train.csv")
df_sub = pd.read_csv("/kaggle/input/jigsaw-toxic-severity-rating/comments_to_score.csv")

val.head()
df_train.head()


# æ•°æ�®æ¸…æ´—
def clean_text(text):
# ç”¨æ­£åˆ™è¡¨è¾¾å¼�æ›¿æ�¢HTMLæ ‡ç­¾ä¸ºç©ºæ ¼ï¼ˆä¾‹å¦‚"<p>Hello!</p>" â†’ "  Hello!  "ï¼‰
    text=re.sub('<.*?>', ' ', text)  
# åˆ é™¤æ‰€æœ‰æ ‡ç‚¹ç¬¦å�·
    text = text.translate(str.maketrans(' ',' ',string.punctuation))
# å�ªä¿�ç•™å­—æ¯�å’Œæ•°å­—ï¼Œç¤ºä¾‹ï¼šè¾“å…¥ "Hello123!" â†’ è¾“å‡º "Hello
    text = re.sub('[^a-zA-Z0-9]',' ',text)  
# å°†æ�¢è¡Œç¬¦æ›¿æ�¢ä¸ºç©ºæ ¼
    text = re.sub("\n"," ",text)
# è½¬æ�¢ä¸ºå…¨å°�å†™
    text = text.lower()
# å�»é™¤å¤šä½™ç©ºæ ¼ï¼šåˆ†å‰²å�•è¯�å��é‡�æ–°ç”¨å�•ç©ºæ ¼è¿�æ�¥æˆ�å­—ç¬¦ä¸²è¿”å›�
    text=' '.join(text.split())
    return text
    
# ä»�æ–‡æœ¬ä¸­ç§»é™¤å�œç”¨è¯�ï¼ˆå¦‚ "the", "is" ç­‰ï¼‰
def stopwords(input_text, stop_words):
     # å°†å�œç”¨è¯�åˆ—è¡¨è½¬ä¸ºé›†å�ˆæ��é«˜æ•ˆç�‡
    stop_words = set(stop_words)
    # ä½¿ç”¨NLTKçš„åˆ†è¯�å·¥å…·å°†æ–‡æœ¬æ‹†åˆ†ä¸ºå�•è¯�åˆ—è¡¨
    word_tokens = word_tokenize(input_text)
    # è¿‡æ»¤å�œç”¨è¯�ï¼ˆå�•æ¬¡å¾ªç�¯ï¼‰
    filtered_words = [word for word in word_tokens if word not in stop_words]
    # æ³¨æ„�è¿™é‡Œåº”è¯¥è¿”å›�ä¸€ä¸ªå�•è¯�åˆ—è¡¨è€Œä¸�æ˜¯å­—ç¬¦ä¸²
    return filtered_words  # è¿”å›�åˆ—è¡¨ï¼Œå¦‚ ["hello", "world"]


irrelevant_words = ['wiki','wikipedia','page']
# æ–‡æœ¬é¢„å¤„ç�†å‡½æ•°
def clean(data, text_column):
    # Step 1: åŸºç¡€æ¸…æ´—ï¼ˆå­—ç¬¦çº§ï¼‰
    data[text_column] = data[text_column].apply(clean_text)
    
    # Step 2: åˆ†è¯� + è¿‡æ»¤å�œç”¨è¯�å’Œæ— å…³è¯�ï¼ˆå�•è¯�çº§ï¼‰
    data[text_column] = data[text_column].apply(
        lambda x: [word for word in word_tokenize(x) 
                  if word not in stop_words 
                  and word not in irrelevant_words]
    )
    
    # Step 3: è¯�å½¢è¿˜å�Ÿï¼ˆå�•è¯�çº§ï¼‰
    data[text_column] = data[text_column].apply(
        lambda words: [lemmatizer.lemmatize(word, pos='v') for word in words]
    )

# æ¸…æ´—è®­ç»ƒé›†è¯„è®ºå†…å®¹
clean(df_train,"comment_text")
df_train.head()

# éªŒè¯�é›†æ¸…æ´—
clean(val,"less_toxic")
clean(val,"more_toxic")
val.head()


# ç»Ÿè®¡æ¯�ä¸ªæ ‡æ³¨è€…åœ¨éªŒè¯�é›†ä¸­æ ‡æ³¨äº†å¤šå°‘è¡Œæ•°æ�®ï¼Œå¹¶ç»˜åˆ¶ç›´æ–¹å›¾å±•ç¤ºæ ‡æ³¨è€…æ•°æ�®é‡�çš„åˆ†å¸ƒæƒ…å†µï¼šæ¨ªè½´ä¸ºè¡Œæ•°
w = val['worker'].value_counts() \
    .plot(kind='hist', bins=50,
          color="#e88366", figsize=(12, 5))
w.set_facecolor("white")  # è®¾ç½®å›¾è¡¨èƒŒæ™¯ä¸ºç™½è‰²
w.set_title('Frequeny of Worker in Validation Set', fontsize=20)
w.set_xlabel('Rows in Validation set for a Worker')


text1 = 'less_toxic'
text2 = "more_toxic"
# å¯¹æ¯”ä¸¤ç±»æ–‡æœ¬çš„å­—ç¬¦æ•°åˆ†å¸ƒ
def lplot(data, var, color, label):
    fig, ax=plt.subplots(1,1)
    length=data[var].apply(len)
    data["length"]=length
    length = data.loc[data["length"]<1000]["length"]
    ax.set_facecolor("white")
    sns.distplot(length, color=color)
    plt.suptitle(label)
    return plt

lplot(val,text1,"#eb9973", "Less Toxic Comments Length")
lplot(val, text2,"#db565d", "More Toxic Comments Length")
plt.show() 


# å¯¹æ¯”ä¸¤ç±»æ–‡æœ¬çš„å�•è¯�æ•°åˆ†å¸ƒ
def wplot(data, var,color, label):
    fig, ax=plt.subplots(1,1)
    words = val[var].apply(len)  # ç›´æ�¥å�–åˆ—è¡¨é•¿åº¦ä½œä¸ºå�•è¯�æ•°
    val['words'] = words
    words = val.loc[val['words']<200]['words']
    ax.set_facecolor("white")
    sns.distplot(words, color=color)
    plt.suptitle(label)
    return plt

wplot(val,text1,"#e88366", "Less Toxic Comments Words")
wplot(val, text2,"#cc4664", "More Toxic Comments Words")
plt.show() 


# ä¸¤ç±»æ–‡æœ¬å¹³å�‡å�•è¯�é•¿åº¦åˆ†æ��
def awdplt(data, var, color, label):
    fig, ax = plt.subplots(1,1)
    # è®¡ç®—å¹³å�‡å�•è¯�é•¿åº¦
    avg_word_len = data[var].apply(
    lambda x: (len(x.replace(" ", "")) / len(x.split())) if len(x.split())>0 else 0
     )
    val['avg_word_len'] = avg_word_len
    avg_word_len = val.loc[data['avg_word_len'] < 10]['avg_word_len']  # è¿‡æ»¤å¼‚å¸¸å€¼
    sns.distplot(avg_word_len, color=color)
    ax.set_facecolor("white")
    plt.suptitle(label)
    return plt


# # ç®±çº¿å›¾å±•ç¤ºä¸¤ç±»æ–‡æœ¬çš„å�•è¯�æ•°åˆ†å¸ƒå·®å¼‚ï¼ˆä¸­ä½�æ•°ã€�å››åˆ†ä½�æ•°ã€�å¼‚å¸¸å€¼ç­‰ï¼‰
# less_toxic_words = [len(sentence.split(' ')) for sentence in val['less_toxic']]
# more_toxic_words = [len(sentence.split(' ')) for sentence in val['more_toxic']]

# fig = go.Figure()
# fig.add_trace(go.Box(y=less_toxic_words, name='Less Toxic'))
# fig.add_trace(go.Box(y=more_toxic_words, name='More Toxic'))

# fig.update_layout(
#     title='Word Count Distribution Comparison',
#     yaxis_title='Word Count'
# )

# fig.show()



# å®šä¹‰n-gramç”Ÿæˆ�å‡½æ•°
def ngram(text, n_gram=1):
    if isinstance(text, list):
        # å¦‚æ�œä¼ å…¥çš„æ˜¯åˆ—è¡¨ï¼Œå°†åˆ—è¡¨ä¸­çš„å­—ç¬¦ä¸²æ‹¼æ�¥æˆ�ä¸€ä¸ªå¤§å­—ç¬¦ä¸²
        text = " ".join(text)
    tokens = word_tokenize(text.lower())  # ä½¿ç”¨ç¨³å�¥çš„åˆ†è¯�å™¨
    tokens = [t for t in tokens if t not in STOPWORDS and t.isalnum()]  # è¿‡æ»¤å�œç”¨è¯�å’Œé��å­—æ¯�æ•°å­—å­—ç¬¦
    ngrams = zip(*[tokens[i:] for i in range(n_gram)])  # ç”Ÿæˆ�N-gramç»„å�ˆ
    return [' '.join(ngram) for ngram in ngrams]   # è¿”å›�N-gramå­—ç¬¦ä¸²åˆ—è¡¨

N = 18  # å±•ç¤ºå‰�18ä¸ªé«˜é¢‘è¯�

# ç»Ÿè®¡ä½�æ¯’æ€§æ–‡æœ¬çš„å�•å­—è¯�é¢‘ç�‡
less_toxic_unigrams = defaultdict(int)
for tweet in val['less_toxic']:
    for word in ngram(tweet, 1):
        less_toxic_unigrams[word] += 1

# è½¬æ�¢ä¸ºDataFrameå¹¶æŒ‰é¢‘ç�‡é™�åº�æ�’åº�
df_less_toxic_unigrams = pd.DataFrame(sorted(less_toxic_unigrams.items(), key=lambda x: x[1])[::-1])
unigrams_less_100 = df_less_toxic_unigrams[:N]  # å�–å‰�Nä¸ª

# ç»Ÿè®¡é«˜æ¯’æ€§æ–‡æœ¬çš„å�•å­—è¯�é¢‘ç�‡
more_toxic_unigrams = defaultdict(int)
for tweet in val['more_toxic']:
    for word in ngram(tweet, 1):
        more_toxic_unigrams[word] += 1
        
df_more_toxic_unigrams = pd.DataFrame(sorted(more_toxic_unigrams.items(), key=lambda x: x[1])[::-1])
unigrams_more_100 = df_more_toxic_unigrams[:N]

# å�¯è§†åŒ–é«˜é¢‘å�•å­—è¯�åˆ†å¸ƒå¹¶ç»˜åˆ¶æ�¡å½¢å›¾
fig, axes = plt.subplots(ncols=2, figsize=(18, N//2), dpi=100)
plt.tight_layout()

sns.barplot(y=unigrams_less_100[0], x=unigrams_less_100[1], ax=axes[0], color='#eca479')
sns.barplot(y=unigrams_more_100[0], x=unigrams_more_100[1], ax=axes[1], color='#de5d5c')

for i in range(2):
    axes[i].spines['right'].set_visible(False)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
    axes[i].tick_params(axis='x', labelsize=13)
    axes[i].tick_params(axis='y', labelsize=13)
    axes[i].set_facecolor("white")

axes[0].set_title(f'Top {N} most common unigrams in less_toxic comments', fontsize=15)
axes[1].set_title(f'Top {N} most common unigrams in more_toxic comments', fontsize=15)

plt.show()


# ç»Ÿè®¡ä½�æ¯’æ€§æ–‡æœ¬çš„å�Œå­—è¯�é¢‘ç�‡
less_toxic_bigrams = defaultdict(int)
for tweet in val['less_toxic']:
    for word in ngram(tweet, 2):
        less_toxic_bigrams[word] += 1
        
df_less_toxic_bigrams = pd.DataFrame(sorted(less_toxic_bigrams.items(), key=lambda x: x[1])[::-1])

bigrams_less_100 = df_less_toxic_bigrams[:N]

# ç»Ÿè®¡é«˜æ¯’æ€§æ–‡æœ¬çš„å�Œå­—è¯�é¢‘ç�‡
more_toxic_bigrams = defaultdict(int)
for tweet in val['more_toxic']:
    for word in ngram(tweet, 2):
        more_toxic_bigrams[word] += 1
        
df_more_toxic_bigrams = pd.DataFrame(sorted(more_toxic_bigrams.items(), key=lambda x: x[1])[::-1])

bigrams_more_100 = df_more_toxic_bigrams[:N]

# å�¯è§†åŒ–é«˜é¢‘å�Œå­—è¯�åˆ†å¸ƒå¹¶ç»˜åˆ¶æ�¡å½¢å›¾
fig, axes = plt.subplots(ncols=2, figsize=(18, N//2), dpi=100)
plt.tight_layout()

sns.barplot(y=bigrams_less_100[0], x=bigrams_less_100[1], ax=axes[0], color='#e98d6b')
sns.barplot(y=bigrams_more_100[0], x=bigrams_more_100[1], ax=axes[1], color='#d14a61')

for i in range(2):
    axes[i].spines['right'].set_visible(False)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
    axes[i].tick_params(axis='x', labelsize=13)
    axes[i].tick_params(axis='y', labelsize=13)
    axes[i].set_facecolor("white")
    
axes[0].set_title(f'Top {N} most common bigrams in less_toxic comments', fontsize=15)
axes[1].set_title(f'Top {N} most common bigrams in more_toxic comments', fontsize=15)

plt.show()


random_score = {'obscene': 0.20, 'toxic': 0.40, 'threat': 0.6, 
                'insult': 0.65, 'severe_toxic': 0.9, 'identity_hate': 0.9}

# ä¸ºæ¯�ä¸ªæ¯’æ€§ç±»åˆ«èµ‹äºˆæ�ƒé‡�ï¼Œè°ƒæ•´æ ‡ç­¾å€¼
for category in random_score:
    df_train[category] = df_train[category] * random_score[category]

# è®¡ç®—ç»¼å�ˆæ¯’æ€§å¾—åˆ†ï¼ˆå�„åˆ—å�‡å€¼ï¼‰
df_train['score'] = df_train.loc[:, 'toxic':'identity_hate'].mean(axis=1)
df_train['y'] = df_train['score']

# ç»Ÿè®¡æ¯’æ€§æ ·æœ¬æ•°é‡�
min_len = (df_train['y'] > 0).sum()  

# éš�æœºæŠ½å�–ç­‰é‡�çš„é��æ¯’æ€§æ ·æœ¬,è§£å†³ç±»åˆ«ä¸�å¹³è¡¡é—®é¢˜
df_non_tox = df_train[df_train['y'] == 0].sample(n=min_len, random_state=201)  

# å�ˆå¹¶æ¯’æ€§æ ·æœ¬å’ŒæŠ½å�–çš„é��æ¯’æ€§æ ·æœ¬
df_train_new = pd.concat([df_train[df_train['y'] > 0], df_non_tox]) 
df_train_new.head()

# è®¡ç®—å�Ÿè®­ç»ƒé›†ä¸­æ¯’æ€§æ ·æœ¬å’Œæ­£å¸¸æ ·æœ¬çš„æ•°é‡�
n_samples_toxic = len(df_train[df_train['score'] != 0])  
n_samples_normal = len(df_train) - n_samples_toxic

# éš�æœºåˆ é™¤éƒ¨åˆ†ä¸­æ€§æ ·æœ¬ï¼ˆä»…ä¿�ç•™1/5ï¼‰
idx_to_drop = df_train[df_train['score'] == 0].index[n_samples_toxic//5:]
df_train = df_train.drop(idx_to_drop)
print(f'Reduced number of neutral text samples from {n_samples_normal} to {n_samples_toxic//5}.')
print(f'Total number of training samples: {len(df_train)}')


# æ ¹æ�®æ¯’æ€§å¾—åˆ†åˆ’åˆ†äº”ä¸ªæƒ…æ„Ÿç­‰çº§
# è®¡ç®—æ¯’æ€§å¾—åˆ†åˆ†ä½�æ•°é˜ˆå€¼ï¼ˆå�–å‰�20%åˆ†ä½�æ•°ï¼‰
df_targets = pd.DataFrame(pd.unique(df_train['score'].values), columns=['target_value']).sort_values(by='target_value', ascending = True).reset_index(drop=True)
THRESHOLD = df_targets['target_value'].quantile(q=0.2)

# å°†æ¯’æ€§å¾—åˆ†æ˜ å°„ä¸º1-5çº§æƒ…æ„Ÿæ ‡ç­¾
df_train['sentiment'] = df_train['score'].map(
    lambda x: 1 if x < THRESHOLD else 
              2 if x < THRESHOLD*2 else 
              3 if x < THRESHOLD*3 else 
              4 if x < THRESHOLD*4 else 5
)

# æœ€ç»ˆè®­ç»ƒæ•°æ�®ä»…ä¿�ç•™æ–‡æœ¬åˆ—å’Œç”Ÿæˆ�çš„æƒ…æ„Ÿæ ‡ç­¾åˆ—ã€‚
df_train = df_train[['comment_text','sentiment']].reset_index(drop=True)
df_train

# æ£€æŸ¥æ•°æ�®ç±»å�‹å¹¶è½¬æ�¢ä¸ºå­—ç¬¦ä¸²
for i, row in df_train.iterrows():
    if isinstance(row['comment_text'], list):
        df_train.at[i, 'comment_text'] = " ".join(row['comment_text'])

# åˆ�å§‹åŒ–TF-IDFå�‘é‡�åŒ–å™¨ï¼ˆè¿‡æ»¤è‹±æ–‡å�œç”¨è¯�ï¼‰
tf_idf_vect = TfidfVectorizer(analyzer='word', stop_words='english')
# å¯¹è®­ç»ƒé›†æ–‡æœ¬è¿›è¡Œå�‘é‡�åŒ–ï¼Œå°†æ–‡æœ¬è½¬æ�¢ä¸ºTF-IDFçŸ©é˜µï¼ˆæ¯�è¡Œè¡¨ç¤ºä¸€ä¸ªæ–‡æ¡£ï¼Œæ¯�åˆ—è¡¨ç¤ºä¸€ä¸ªè¯�çš„TF-IDFå€¼ï¼‰
X = tf_idf_vect.fit_transform(df_train['comment_text']).toarray()
X

# åŠ è½½æµ‹è¯•é›†å¹¶æ¸…æ´—æ–‡æœ¬
df_test = pd.read_csv(comments_to_score_path)
clean(df_test, "text")
df_test.head(3)

# æ£€æŸ¥æ•°æ�®ç±»å�‹å¹¶è½¬æ�¢ä¸ºå­—ç¬¦ä¸²
for i, row in df_test.iterrows():
    if isinstance(row['text'], list):
        df_test.at[i, 'text'] = " ".join(row['text'])

# ä½¿ç”¨è®­ç»ƒé›†çš„å�‘é‡�åŒ–å™¨è½¬æ�¢æµ‹è¯•é›†
Y = tf_idf_vect.transform(df_test['text']).toarray()
Y


clean(df_train1,"comment_text")


# åˆ›å»ºäºŒåˆ†ç±»æ ‡ç­¾ y
df_train1['y'] = (df_train1[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1) > 0 ).astype(int)
df_train_binary = df_train1[['comment_text', 'y']].rename(columns={'comment_text': 'text'})

word1=df_train_binary.loc[df_train_binary['y']==1,['text','y']]


df_lt=df_train_binary.loc[df_train_binary['y']==0]
df_train_binary['y'].value_counts(normalize=True)


toxic_len = (df_train_binary['y'] == 1).sum()
print(toxic_len)
df_train_balanced = df_train_binary[df_train_binary['y'] == 0].sample(n=toxic_len)
df_train_balanced['y'].value_counts(normalize=True)
df_train_b = pd.concat([df_train_binary[df_train_binary['y'] == 1], df_train_balanced])
df_train_b['y'].value_counts()


tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
# æ£€æŸ¥æ•°æ�®ç±»å�‹å¹¶è½¬æ�¢ä¸ºå­—ç¬¦ä¸²
for i, row in df_train_b.iterrows():
    if isinstance(row['text'], list):
        df_train_b.at[i, 'text'] = " ".join(row['text'])

features = tfidf.fit_transform(df_train_b['text']).toarray()
labels = df_train_b['y']
features.shape


models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

sns.set_style("whitegrid")
sns.boxplot(x='model_name', y='accuracy', data=cv_df,palette="flare")
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, linewidth=1,palette="flare")
plt.show()


ft_importance = pd.DataFrame({"Feature Importance":cv_df.groupby('model_name').accuracy.mean()}, index=cv_df.model_name)
X_cols=ft_importance.query('`Feature Importance` > 0.1').sort_values(by="Feature Importance", ascending=False)
col1=X_cols.drop_duplicates()
col1.style.background_gradient(cmap="flare")


from sklearn.calibration import CalibratedClassifierCV
svm = LinearSVC()
clf = CalibratedClassifierCV(svm) 
clf.fit(features, labels)

CalibratedClassifierCV(base_estimator=LinearSVC())

y_pred = clf.predict(features)
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(labels, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
cmap = "flare"
sns.heatmap(conf_mat, annot=True, fmt='d',cmap=cmap)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


clean(df_sub, "text")
X_sub = df_sub['text']
# æ£€æŸ¥æ•°æ�®ç±»å�‹å¹¶è½¬æ�¢ä¸ºå­—ç¬¦ä¸²
for i, row in df_sub.iterrows():
    if isinstance(row['text'], list):
        df_sub.at[i, 'text'] = " ".join(row['text'])

X_test= tfidf.transform(df_sub['text'])
y_test = clf.predict_proba(X_test)
df_sub['score'] = y_test[:, 1]
df_sub[['comment_id', 'score']].to_csv("submission.csv", index=False)

