# 必要なライブラリのインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')


# ZIPファイルの展開とデータの読み込み
!unzip -n ../input/konwinski-prize/data.a_zip
import zipfile
konwinski = zipfile.ZipFile('../input/konwinski-prize/data.a_zip')
konwinski.extractall()
train_data = pd.read_parquet("data/data.parquet")

# データの基本情報を表示
print("データセットの基本情報:")
print(train_data.info())


# リポジトリの分布を可視化
repo = train_data['repo'].value_counts()
plt.figure(figsize=(10, 6))
plt.pie(repo.values,
        labels=repo.index,
        autopct='%1.1f%%')
plt.title('Distribution of GitHub repositories')
plt.show()


# SWE-benchデータセットの読み込み
from datasets import load_dataset
swebench = load_dataset('princeton-nlp/SWE-bench', split='test')


# ワードクラウドの作成
from wordcloud import WordCloud, STOPWORDS

def plot_wordcloud(text, title=""):
    stopwords = set(STOPWORDS)
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=100,
        width=800,
        height=400
    ).generate(str(text))
    
    plt.figure(figsize=(24.0, 16.0))
    plt.imshow(wordcloud)
    plt.title(title, fontsize=40)
    plt.axis('off')
    plt.tight_layout()

# 問題文のワードクラウドを表示
plot_wordcloud(train_data["problem_statement"], "Frequently occurring words included in issue sentences")


# Fail to Passテストの単語分析
from collections import defaultdict

def generate_ngrams(text, n_gram=1):
    """テキストからn-gramを生成する関数"""
    text = str(text)
    token = [token for token in text.lower().split(" ") 
            if token != "" and token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

# 単語の頻度カウント
freq_dict = defaultdict(int)
for sent in train_data["FAIL_TO_PASS"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1

# 頻度順にソート
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]

# 上位20単語を表示
plt.figure(figsize=(12, 6))
sns.barplot(data=fd_sorted.head(20), x='wordcount', y='word')
plt.title('Top 20 most frequent words in the FAIL_TO_PASS test')
plt.show()

