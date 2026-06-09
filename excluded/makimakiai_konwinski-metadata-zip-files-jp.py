# 必要なライブラリのインポート
import numpy as np # 線形代数用
import pandas as pd # データ処理、CSV入出力用

import matplotlib.pyplot as plt 
import seaborn as sns

# Plotlyグラフ表示に必要な2行
import plotly.io as pio
pio.renderers.default = 'iframe'

import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px

# 警告を無視
import warnings
warnings.filterwarnings('ignore')

# ファイルの確認
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))





# ZIPファイルの展開
!unzip -n ../input/konwinski-prize/data.a_zip

# データの読み込み
import zipfile
konwinski = zipfile.ZipFile('../input/konwinski-prize/data.a_zip')
konwinski.extractall()

train_data = pd.read_parquet("data/data.parquet")


# データの確認
train_data.head()
train_data.info()

# 欠損値の確認
print("各カラムの欠損値数:\n", train_data.isnull().sum())


# リポジトリの円グラフ表示
repo = train_data['repo'].value_counts()
plt.pie(repo.values,
        labels=repo.index,
        autopct='%1.1f%%')
plt.title('GitHubリポジトリの分布')
plt.show()


# ワードクラウドの作成
from wordcloud import WordCloud, STOPWORDS

def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=False):
    # ストップワードの設定
    stopwords = set(STOPWORDS)
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)

    # ワードクラウドの生成
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
    
    # プロット設定
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask)
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud)
        plt.title(title, fontdict={'size': title_size, 'color': 'black', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off')
    plt.tight_layout()  
    
# 問題文のワードクラウドを表示
plot_wordcloud(train_data["problem_statement"], title="問題文に含まれる単語の分布")


from plotly import tools
from collections import defaultdict

# N-gramを生成するカスタム関数
def generate_ngrams(text, n_gram=1):
    text = str(text)
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

# 水平棒グラフ作成用のカスタム関数
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

# Fail to Passの単語頻度分析
freq_dict = defaultdict(int)
for sent in train_data["FAIL_TO_PASS"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(50), 'black')

# サブプロットの作成
fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,
                          subplot_titles=["Fail to Pass での頻出単語"]) 
fig.append_trace(trace1, 1, 2)
fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', 
                    title="単語出現頻度の分析")
py.iplot(fig, filename='word-plots')


# SWE-benchデータセットの読み込み
from datasets import load_dataset
swebench = load_dataset('princeton-nlp/SWE-bench', split='test')

