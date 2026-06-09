# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#Two lines Required to Plot Plotly
import plotly.io as pio
pio.renderers.default = 'iframe'

import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#JSON
import json


#df = pd.read_json('../input/cusersmarildownloadsdata-json/data.json', encoding = 'utf-8-sig')

#input_df = pd.read_json('../input/traffic-violations/data.json', lines=True, orient="columns")

#data= pd.read_json('../input/traffic-violations/data.json', lines=True)

#data= pd.read_json('../input/traffic-violations/data.json', lines=True, orient='records')

#data= pd.read_json('../input/traffic-violations/data.json', orient=str)

#data= pd.read_json('../input/traffic-violations/data.json', typ="series")

#df = pd.read_json(path_or_buf='/kaggle/input/en-to-ko/EN_TO_KO.json')


df4= pd.read_json('../input/openai-gpt-oss-20b-red-teaming/example-harmony-findings.json', typ="series")

df4.tail()


red = json.load(open('../input/openai-gpt-oss-20b-red-teaming/example-harmony-findings.json'))

df = pd.DataFrame(red["issue_summary"], index=[0,8])


df.tail()


df.iloc[0,0]


#By SRK https://www.kaggle.com/code/sudalairajkumar/simple-exploration-notebook-qiqc

from wordcloud import WordCloud, STOPWORDS

# Thanks : https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##
def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='red',
                    color_func=lambda *args, **kwargs: "white",      
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
    
plot_wordcloud(df["title"], title="Shrink advice")


df1 = pd.DataFrame(red["model"], index=[0,3])

df1.tail()


df2 = pd.DataFrame(red["steps_to_reproduce"])

df2.tail()


df5 = pd.DataFrame(red["environment"], index=[0,2])

df5.tail()


df3 = pd.DataFrame(red["harmony_response_walkthroughs"])

df3.tail()


#First row, first column 

df3.iloc[0,0]

