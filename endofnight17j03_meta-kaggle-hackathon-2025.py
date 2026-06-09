# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline

#Two lines Required to Plot Plotly
import plotly.io as pio
pio.renderers.default = 'iframe'

import plotly.graph_objects as go
import plotly.express as px


import warnings
warnings.simplefilter(action='ignore', category=Warning)

import cv2
from wordcloud import WordCloud, ImageColorGenerator

import urllib
import random

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv("/kaggle/input/2023-kaggle-ai-report/kaggle_writeups_20230510.csv", delimiter=',', encoding='utf8')
pd.set_option('display.max_columns', None)
df.tail()


df["Title of Writeup"].value_counts()


##Code by Taha07  https://www.kaggle.com/taha07/data-scientists-jobs-analysis-visualization/notebook

from wordcloud import WordCloud
from wordcloud import STOPWORDS
stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color = 'white',
                      color_func=lambda *args, **kwargs: "black",
                      height =2000,
                      width = 2000
                     ).generate(str(df["Title of Writeup"]))
plt.rcParams['figure.figsize'] = (12,12)
plt.axis("off")
plt.imshow(wordcloud)
plt.title("Kaggle Competitions Writeups titles")
plt.show()


#By SRK https://www.kaggle.com/code/sudalairajkumar/winning-solutions-of-kaggle-competitions

from IPython.core.display import HTML

data_path = "../input/"

competitions_df = pd.read_csv(data_path + "meta-kaggle/Competitions.csv")
comps_to_use = ["Featured", "Research", "Recruitment"]
competitions_df = competitions_df[competitions_df["HostSegmentTitle"].isin(comps_to_use)]
competitions_df["EnabledDate"] = pd.to_datetime(competitions_df["EnabledDate"], format="%m/%d/%Y %H:%M:%S")
competitions_df = competitions_df.sort_values(by="EnabledDate", ascending=False).reset_index(drop=True)
competitions_df.head()

forum_topics_df = pd.read_csv(data_path + "meta-kaggle/ForumTopics.csv")

comp_tags_df = pd.read_csv(data_path + "meta-kaggle/CompetitionTags.csv")
tags_df = pd.read_csv(data_path + "meta-kaggle/Tags.csv", usecols=["Id", "Name"])

def get_comp_tags(comp_id):
    temp_df = comp_tags_df[comp_tags_df["CompetitionId"]==comp_id]
    temp_df = pd.merge(temp_df, tags_df, left_on="TagId", right_on="Id")
    tags_str = "Tags : "
    for ind, row in temp_df.iterrows():
        tags_str += row["Name"] + ", "
    return tags_str.strip(", ")

def check_solution(topic):
    is_solution = False
    to_exclude = ["?", "submit", "why", "what", "resolution", "benchmark"]
    if "solution" in topic.lower():
        is_solution = True
        for exc in to_exclude:
            if exc in topic.lower():
                is_solution = False
    to_include = ["2nd place code", '"dance with ensemble" sharing']
    for inc in to_include:
        if inc in topic.lower():
            is_solution = True
    return is_solution

def get_discussion_results(forum_id, n):
    results_df = forum_topics_df[forum_topics_df["ForumId"]==forum_id]
    results_df["is_solution"] = results_df["Title"].apply(lambda x: check_solution(str(x)))
    results_df = results_df[results_df["is_solution"] == 1]
    results_df = results_df.sort_values(by=["Score","TotalMessages"], ascending=False).head(n).reset_index(drop=True)
    return results_df[["Title", "Id", "Score", "TotalMessages", "TotalReplies"]]

def render_html_for_comp(forum_id, comp_id, comp_name, comp_slug, comp_subtitle, n):
    results_df = get_discussion_results(forum_id, n)
    
    if len(results_df) < 1:
        return
    
    comp_tags = get_comp_tags(comp_id)
    
    comp_url = "https://www.kaggle.com/c/"+str(comp_slug)
    hs = """<style>
                .rendered_html tr {font-size: 12px; text-align: left;}
                th {
                text-align: left;
                }
            </style>
            <h3><font color="#FFD700"><a href="""+comp_url+""">"""+comp_name+"""</font></h3>
            <p>"""+comp_subtitle+"""</p>
            """
    
    if comp_tags != "Tags :":
        hs +=   """
            <p>"""+comp_tags+"""</p>
            """
    
    hs +=   """
            <table>
            <tr>
                <th><b>S.No</b></th>
                <th><b>Discussion Title</b></th>
                <th><b>Number of upvotes</b></th>
                <th><b>Total Replies</b></th>
            </tr>"""
    
    for i, row in results_df.iterrows():
        url = "https://www.kaggle.com/c/"+str(comp_slug)+"/discussion/"+str(row["Id"])
        hs += """<tr>
                    <td>"""+str(i+1)+"""</td>
                    <td><a href="""+url+""" target="_blank"><b>"""  +str(row['Title']) + """</b></a></td>
                    <td>"""+str(row['Score'])+"""</td>
                    <td>"""+str(row['TotalReplies'])+"""</td>
                    </tr>"""
    hs += "</table>"
    display(HTML(hs))

for ind, comp_row in competitions_df.iterrows():
    render_html_for_comp(comp_row["ForumId"], comp_row["Id"], comp_row["Title"], comp_row["Slug"], comp_row["Subtitle"], 12)

