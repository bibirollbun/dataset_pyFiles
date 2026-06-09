%pip install plotly[express] -q


import csv
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import warnings

from datetime import datetime
from IPython.display import HTML
from math import log, floor


pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_columns', 1000)
warnings.filterwarnings('ignore')
pio.renderers.default = 'iframe' #https://www.kaggle.com/discussions/product-announcements/549950


DATASET_ROOT_DIRECTORY = "/kaggle/input"
COVID_PANDEMIC_START = "2020-03-11"
COVID_PANDEMIC_END = "2023-05-05"
MILLION = 1_000_000
MILESTONE_MULTIPLE = 5_000_000
SLIDE_SIZE = 90
TOP_X = 30
ZSCORE_CUTOFF = 3.0


class Color:
    BLUE="blue"
    GREEN="green"    
    PURPLE="purple"
    RED="red"


df_forums = pd.read_csv(f"{DATASET_ROOT_DIRECTORY}/meta-kaggle/Forums.csv").rename(columns={"Id":"ForumId"})
df_users = pd.read_csv(f"{DATASET_ROOT_DIRECTORY}/meta-kaggle/Users.csv").rename(columns={"Id":"UserId"})


df_kaggle_events =  pd.read_csv(f"{DATASET_ROOT_DIRECTORY}/kaggle-staff-forum-topic-posts/events_posts.csv", encoding="utf8", quoting=csv.QUOTE_NONNUMERIC)


df_users["RegisterDate"] = pd.to_datetime(df_users["RegisterDate"]).dt.date
df_kaggle_events["EventDate"] = pd.to_datetime(df_kaggle_events["EventDate"]).dt.date


df_users = df_users[df_users["RegisterDate"] <= (df_users["RegisterDate"].max() - pd.Timedelta(days=1))].reset_index(drop=True)


HTML(df_kaggle_events.head().reset_index(drop=True).to_html(render_links=True, escape=False))


def aggregate_grouping_and_columns(df, group, column_operation_mappings):
    inner_df_agg = df.groupby(group).agg(column_operation_mappings).reset_index()
    inner_df_agg.columns = ['_'.join(col).strip() for col in inner_df_agg.columns.values]
    inner_df_agg.columns = [col[:-1] if col[-1] == "_" else col for col in inner_df_agg.columns.values]    
    inner_df_agg = inner_df_agg.sort_values(group).reset_index(drop=True)
    return inner_df_agg

def human_format(number):
    units = ['', 'K', 'M', 'G', 'T', 'P']
    k = 1000.0
    magnitude = int(floor(log(number, k)))
    return '%.2f%s' % (number / k**magnitude, units[magnitude])

def upvote_milestone_generator(df, entity_type="",verb={}, multiple=500000):
    milestones = {}
    quotient = int(df.shape[0] // multiple)
    milestones[0] = f"First ever {entity_type[0]} {verb}"
    for i in range(0,multiple*(quotient+1),multiple):
        if(i>0):
            milestones[i] = f"{human_format(i)} {entity_type[1]} {verb}"
    return milestones

def add_milestone(milestones, key, value):
    milestones[key] = value
    return dict(sorted(milestones.items()))

def milestones_extractor(df, milestones, comparison_column, export_columns):
    df_collection = []
    for k in milestones.keys():
        inner_df = df[df.index==df[df[comparison_column] >= k].index[0]].copy()
        inner_df["text"] = milestones[k]
        df_collection.append(inner_df)
        
    df_combined = pd.concat(df_collection)
    export_columns.append("text")
    df_combined = df_combined[export_columns].reset_index(drop=True)    
    
    return df_combined


df_users_agg = aggregate_grouping_and_columns(df_users, "RegisterDate",{"UserId":["nunique"]})
df_users_agg = df_users_agg.rename(columns={"UserId_nunique":"UniqueUsers"})
df_users_agg["RegisterDateYear"] = pd.to_datetime(df_users_agg["RegisterDate"]).dt.year
df_users_agg["RegisterDateMonth"] = pd.to_datetime(df_users_agg["RegisterDate"]).dt.month
df_users_agg["CumSum"] = df_users_agg["UniqueUsers"].cumsum()
df_users_agg["WindowSizeDayMean"] = df_users_agg["UniqueUsers"].transform(lambda x: x.rolling(SLIDE_SIZE, 1).mean().shift().bfill()).astype("float")
df_users_agg["WindowSizeDayStd"] = df_users_agg["UniqueUsers"].transform(lambda x: x.rolling(SLIDE_SIZE, 1).std().shift().bfill()).astype("float")


max_million = (df_users_agg["CumSum"].max() // MILLION) * MILLION
user_registration_milestones = upvote_milestone_generator(df_users, entity_type=["User","Users"], verb="Registered",multiple = MILESTONE_MULTIPLE)
user_registration_milestones = add_milestone(user_registration_milestones,MILLION,f"{human_format(MILLION)} Users Registered")
user_registration_milestones = add_milestone(user_registration_milestones,max_million,f"{human_format(max_million)} Users Registered")
df_user_registration_milestones = milestones_extractor(df_users_agg, user_registration_milestones,"CumSum", df_users_agg.columns.tolist())


def draw_graph(df_agg, x_axis, y_axis, title, height= 756, width=1344, font_size=20, log_y=False):
    fig = px.line(df_agg, x=x_axis, y=y_axis, markers=False, height=height, width=width, log_y=log_y, orientation="h")

    fig.update_layout(
        title=dict(text=title, font=dict(size=font_size), automargin=True)
    )
    
    return fig

def add_annotation(fig, x, y, text, font_size=10, opacity=0.8, color="red", ay=-100, textangle=45):
    fig.add_annotation(x=x
                        ,y=y
                        ,text=text
                        ,showarrow=True
                        ,ay=ay
                        ,arrowhead=1
                        ,arrowsize=1
                        ,arrowwidth=1
                        ,bgcolor="white"
                        ,xanchor="right" #if negative ay value , this should be right
                        ,bordercolor="#c7c7c7"
                        ,borderwidth=1
                        ,borderpad=1
                        ,opacity=opacity
                        ,textangle=textangle
                        ,font=dict(
                            family="Courier New",
                            size=font_size,
                            color=color
                ),                   

    )
    return fig

def add_covid_era(fig, start_x, end_y, title, color="red", opacity=0.25):
    fig.add_vrect(x0=start_x, x1=end_y, 
                  annotation_text=title, annotation_position="top left",
                  annotation=dict(font_size=20, font_family="Courier New"),
                  fillcolor=color, opacity=opacity, line_width=0)

    return fig


fig = draw_graph(df_users_agg,"RegisterDate","CumSum","Cumulative New Users Per Day",log_y=False)

for x, row in df_user_registration_milestones.iterrows():
    x_axis = row["RegisterDate"]
    y_axis = row["CumSum"]
    text = row["text"]
    fig = add_annotation(fig, x_axis, y_axis, text,13,1,"red",-20, 60)
    
fig = add_covid_era(fig, COVID_PANDEMIC_START, COVID_PANDEMIC_END, "COVID-19", Color.GREEN)    

fig.show()


df_users_agg["ZScore"] = round((df_users_agg["UniqueUsers"] - df_users_agg["WindowSizeDayMean"])/df_users_agg["WindowSizeDayStd"],4)
df_users_agg["ZScoreLabel"]  = "Z-score: " + df_users_agg["ZScore"].astype("str")
df_outlier_days = df_users_agg[df_users_agg["ZScore"] >= ZSCORE_CUTOFF].reset_index(drop=True)


df_outlier_days.nlargest(TOP_X,"ZScore").reset_index(drop=True)


columns = ['RegisterDate',
 'CumSum',
 'UniqueUsers']


df_users_agg[columns].nlargest(20, "UniqueUsers").reset_index(drop=True)


df_outlier_days.sort_values(["RegisterDate"], ascending=False).reset_index(drop=True).head(TOP_X)


fig = draw_graph(df_users_agg,"RegisterDate","UniqueUsers","New Users Per Day",log_y=False)

for x, row in df_outlier_days.iterrows():
    x_axis = row["RegisterDate"]
    y_axis = row["UniqueUsers"]
    text = row["ZScoreLabel"]
    fig = add_annotation(fig, x_axis, y_axis, text,13,1,Color.PURPLE,-20, 60)
    
fig = add_covid_era(fig, COVID_PANDEMIC_START, COVID_PANDEMIC_END, "COVID-19", Color.GREEN)    

fig.show()


df_kaggle_events = pd.merge(df_kaggle_events, df_users_agg, left_on=["EventDate"], right_on=["RegisterDate"])


HTML(df_kaggle_events[["EventDate","Clickable_Link","UniqueUsers","WindowSizeDayMean","ZScore"]].nlargest(TOP_X,"ZScore").reset_index(drop=True).to_html(render_links=True, escape=False))


fig = draw_graph(df_users_agg,"RegisterDate","UniqueUsers","New Users Per day with Significant Events",log_y=False)

for x, row in df_kaggle_events.iterrows():
    x_axis = row["RegisterDate"]
    y_axis = row["UniqueUsers"]
    text = row["Clickable_Link"]
    fig = add_annotation(fig, x_axis, y_axis, text,13,1,Color.BLUE,-20, 60)
    
fig = add_covid_era(fig, COVID_PANDEMIC_START, COVID_PANDEMIC_END, "COVID-19", Color.GREEN)
    
fig.show()


fig = draw_graph(df_users_agg,"RegisterDate","UniqueUsers","New Users Per day with Significant Events",log_y=False)

for x, row in df_kaggle_events.iterrows():
    x_axis = row["RegisterDate"]
    y_axis = row["UniqueUsers"]
    text = row["Clickable_Link"]
    fig = add_annotation(fig, x_axis, y_axis, text,13,1,Color.BLUE,-20, 60)
    
for x, row in df_kaggle_events.iterrows():
    x_axis = row["RegisterDate"]
    y_axis = row["UniqueUsers"]
    text = row["ZScoreLabel"]
    fig = add_annotation(fig, x_axis, y_axis, text,13,1,Color.PURPLE,20, -60)       
    
fig = add_covid_era(fig, COVID_PANDEMIC_START, COVID_PANDEMIC_END, "COVID-19", Color.GREEN)
    
fig.show()


print(f"Average Z-score: {round(df_kaggle_events['ZScore'].mean(),4)}")
print(f"Median Z-score: {round(df_kaggle_events['ZScore'].median(),4)}")

