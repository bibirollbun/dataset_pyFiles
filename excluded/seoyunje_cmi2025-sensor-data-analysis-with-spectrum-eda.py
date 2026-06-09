import os
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns 

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from colorama import Fore, Style
c_ = Fore.GREEN
st_ = Style.BRIGHT

import warnings
warnings.filterwarnings('ignore')

print(f"{c_}{st_}Import necessary Library!")


!pip -q install palettable 
import palettable.colorbrewer.qualitative as pbq


train_demo = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')
print(f'{c_}Shape of DataFrame: ', train_demo.shape)
print(f'{Fore.BLUE}{st_} => Average of Age: {np.mean(train_demo.age):.1f}')
print(f'{Fore.BLUE}{st_} => Average of Height(cm): {np.mean(train_demo.height_cm):.1f}cm')
print(f'{Fore.BLUE}{st_} => Average of shoulder_to_wrist_cm: {np.mean(train_demo.shoulder_to_wrist_cm):.1f}cm')
print(f'{Fore.BLUE}{st_} => Average of elbow_to_wrist_cm: {np.mean(train_demo.elbow_to_wrist_cm):.1f}cm')
print(display(train_demo))


print(f'{Fore.BLUE}{st_} => Female Subject: {train_demo[train_demo["sex"] == 0].shape[0]}')
print(f'{Fore.BLUE}{st_} => Male Subject: {train_demo[train_demo["sex"] == 1].shape[0]}')
print(f'{Fore.BLUE}{st_} => Adult Subject: {train_demo[train_demo["adult_child"] == 1].shape[0]}')
print(f'{Fore.BLUE}{st_} => Child Subject: {train_demo[train_demo["adult_child"] == 0].shape[0]}')

trace1 = go.Pie(
    labels= ["Child", "Adult"],
    values= train_demo['adult_child'].value_counts().sort_index(),
    hole=0.5,
    textinfo="label+percent",
    marker=dict(colors=['rgba(255,0,0,0.5)', 'rgba(0,0,255,0.5)']),
    showlegend=False,
)
trace2 = go.Pie(
    labels= ["Female", "Male"],
    values= train_demo['sex'].value_counts().sort_index(),
    hole=0.5,
    textinfo="label+percent",
    marker=dict(colors=['rgba(255,0,0,0.5)', 'rgba(0,0,255,0.5)']),
    showlegend=False,
)


fig = make_subplots(rows=1,cols=2, subplot_titles=("Child(0) vs Adult(1)", "Female(0) vs Male(1)"),
                    specs=[[{'type':'domain'},{'type': 'domain'}]])
fig.add_trace(trace1, 1, 1)
fig.add_trace(trace2, 1, 2)


fig.show(renderer='iframe')


print(f'\n{Fore.BLUE}{st_}From the bar chart, it’s clear that the proportions of females and males in the child and adult categories are not balanced.')

trace1 = go.Histogram(
    y = train_demo[train_demo['sex'] == 0]['adult_child'],
    name = 'Female',
    marker=dict(color='rgba(255,0,0,0.5)'),
    orientation='h',
)

trace2 = go.Histogram(
    y = train_demo[train_demo['sex'] == 1]['adult_child'],
    name = 'Male',
    marker=dict(color='rgba(0,0,255,0.5)'),
    orientation='h',
)


layout = go.Layout(
    title = "Distribution of Child vs Adult",
    barmode='group',
    bargap=0.5,
    yaxis = dict(title='Child(0) vs Adult(1)', ticklen = 5, zeroline=False),
    xaxis = dict(title='Count', ticklen = 5, zeroline=False),
    legend = dict(title='Sex', bordercolor='black', borderwidth=1),
)

fig = go.Figure(data=[trace1,trace2], layout=layout)

fig.show(renderer='iframe')


print(f'{Fore.BLUE}{st_} => Min of Age: {np.min(train_demo.age):.1f}')
print(f'{Fore.BLUE}{st_} => Max of Age: {np.max(train_demo.age):.1f}')

print("\nAs shown in the bar chart above, females tend to be concentrated in the child group (under 18), whereas males are more represented in the adult group (18 and older).")

trace1 = go.Histogram(
    x = train_demo[train_demo['sex'] == 0]['age'],
    name = 'Female',
    marker=dict(color='rgba(171, 50, 96, 0.6)'),
    xbins = dict(size=2.5)
    
)

trace2 = go.Histogram(
    x = train_demo[train_demo['sex'] == 1]['age'],
    name = 'Male',
    marker=dict(color='rgba(12, 50, 196, 0.6)'),
    xbins = dict(size=2.5)
)

layout = go.Layout(
    title = "Distribution of Age",
    barmode='overlay',
    xaxis = dict(title='Age', ticklen = 5, zeroline=False),
    yaxis = dict(title='Count', ticklen = 5, zeroline=False),
    legend = dict(title='Sex', bordercolor='black', borderwidth=1),
)

fig = go.Figure(data=[trace1,trace2], layout=layout)

fig.show(renderer='iframe')


print(f'{Fore.BLUE}{st_} => Left-handed Subject: {train_demo[train_demo["handedness"] == 0].shape[0]}')
print(f'{Fore.BLUE}{st_} => Right-handed Subject: {train_demo[train_demo["handedness"] == 1].shape[0]}')

print('\nAcross age groups, left- and right-handedness appear nearly equal in proportion. When analyzed by sex, the proportion of left-handed individuals stays the same, while that of right-handed individuals differs. This difference seems to stem from the larger number of males compared to females, rather than suggesting any correlation between sex and handedness.')

trace1 = go.Pie(
    labels=['Left_Handed','Right_Handed'],
    values= train_demo['handedness'].value_counts().sort_index(),
    textinfo="label+percent",
    hole=0.7,
    showlegend=False,
    marker=dict(colors=("rgba(255,0,0,0.5)","rgba(0,0,255,0.5)")),
)
trace2 = go.Histogram(
    x = train_demo[train_demo['sex']==0]['handedness'],
    name='Female',
    marker=dict(color='rgba(255,0,0,0.5)'),
)
trace3 = go.Histogram(
    x = train_demo[train_demo['sex']==1]['handedness'],
    name='Male',
    marker=dict(color='rgba(0,0,255,0.5)'),
)

fig = make_subplots(rows=1, cols=2,specs=[[{'type':'domain'},{'type':'xy'}]])

fig.add_trace(trace1, 1, 1)
fig.add_traces([trace2,trace3], 1, 2)

fig.update_layout(
    title='Distribution of Handedness',
    barmode='group',
    bargap=0.5,
    
    xaxis1 = dict(title='Left(0) vs Right(1)', ticklen = 5, zeroline=False),
    yaxis1 = dict(title='Count', ticklen = 5, zeroline=False),
    legend=dict(title='Sex', bordercolor='black', borderwidth=1),
)

fig.show(renderer='iframe')


print(f'{Fore.BLUE}{st_} => Average of Female Height(cm): {np.mean(train_demo[train_demo["sex"] == 0].height_cm):.1f}cm')
print(f'{Fore.BLUE}{st_} => Average of Male Height(cm): {np.mean(train_demo[train_demo["sex"] == 1].height_cm):.1f}cm')
print("\nIt’s interesting that the average height of women turned out to be higher than that of men, even though the proportion of children (who tend to be shorter) is much higher among women, while the proportion of adults is higher among men.")

trace1 = go.Box(
    y = train_demo[train_demo['sex'] == 0]['height_cm'],
    name = "female",
    marker = dict(color='rgba(255,0,0,0.6)'),
    showlegend=False
)
trace2 = go.Box(
    y = train_demo[train_demo['sex'] == 1]['height_cm'],
    name = "male",
    marker = dict(color='rgba(0,0,255,0.6)'),
    showlegend=False
)

trace3 = go.Box(
    y = train_demo[train_demo['adult_child'] == 0]['height_cm'],
    name = "child",
    marker = dict(color='rgba(255,0,0,0.6)'),
    showlegend=False
)
trace4 = go.Box(
    y = train_demo[train_demo['adult_child'] == 1]['height_cm'],
    name = "adult",
    marker = dict(color='rgba(0,0,255,0.6)'),
    showlegend=False
)

fig = make_subplots(rows=1, cols=2)

fig.add_traces([trace1, trace2], 1, 1)
fig.add_traces([trace3, trace4], 1, 2)

fig.update_layout(
    title='Distribution of Height',
    yaxis = dict(title='Height', ticklen = 5, zeroline=False),
)

fig.show(renderer="iframe")


trace1 = go.Scatter(
    x = train_demo[train_demo['adult_child'] == 0][train_demo['sex']==0]['height_cm'],
    y = train_demo[train_demo['adult_child'] == 0][train_demo['sex']==0]['shoulder_to_wrist_cm'],
    mode = 'markers',
    name = 'Female',
    marker=dict(color='rgba(255,0,0,0.5)'),
)
trace2 = go.Scatter(
    x = train_demo[train_demo['adult_child'] == 0][train_demo['sex']==1]['height_cm'],
    y = train_demo[train_demo['adult_child'] == 0][train_demo['sex']==1]['shoulder_to_wrist_cm'],
    mode = 'markers',
    name = 'Male',
    marker=dict(color='rgba(0,0,255,0.5)'),
)
trace3 = go.Scatter(
    x = train_demo[train_demo['adult_child'] == 1][train_demo['sex']==0]['height_cm'],
    y = train_demo[train_demo['adult_child'] == 1][train_demo['sex']==0]['shoulder_to_wrist_cm'],
    mode = 'markers',
    name = 'Female',
    marker=dict(color='rgba(255,0,0,0.5)'),
    showlegend=False,
)
trace4 = go.Scatter(
    x = train_demo[train_demo['adult_child'] == 1][train_demo['sex']==1]['height_cm'],
    y = train_demo[train_demo['adult_child'] == 1][train_demo['sex']==1]['shoulder_to_wrist_cm'],
    mode = 'markers',
    name = 'Male',
    marker=dict(color='rgba(0,0,255,0.5)'),
    showlegend=False,
)


fig = make_subplots(rows=1, cols=2, subplot_titles=("Child", "Adult"))

fig.add_traces([trace1,trace2], 1, 1)
fig.add_traces([trace3,trace4], 1, 2)


fig.update_layout(
    title=dict(
    text="<b>Correlation with Height and Shoulder_to_Wrist</b>",
    font=dict(size=20, family="Arial", color="black"),),
    xaxis1 = dict(title='Height', ticklen = 5, zeroline=False),
    yaxis1 = dict(title='Shoulder_to_Wrist', ticklen = 5, zeroline=False),
    xaxis2 = dict(title='Height', ticklen = 5, zeroline=False),
    yaxis2 = dict(title='Shoulder_to_Wrist', ticklen = 5, zeroline=False),
    legend=dict(title='Sex', bordercolor='black', borderwidth=1)
)

fig.show(renderer='iframe')


trace1 = go.Scatter(
    x = train_demo[train_demo['adult_child'] == 0][train_demo['sex']==0]['height_cm'],
    y = train_demo[train_demo['adult_child'] == 0][train_demo['sex']==0]['elbow_to_wrist_cm'],
    mode = 'markers',
    name = 'Female',
    marker=dict(color='rgba(255,0,0,0.5)'),
)
trace2 = go.Scatter(
    x = train_demo[train_demo['adult_child'] == 0][train_demo['sex']==1]['height_cm'],
    y = train_demo[train_demo['adult_child'] == 0][train_demo['sex']==1]['elbow_to_wrist_cm'],
    mode = 'markers',
    name = 'Male',
    marker=dict(color='rgba(0,0,255,0.5)'),
)
trace3 = go.Scatter(
    x = train_demo[train_demo['adult_child'] == 1][train_demo['sex']==0]['height_cm'],
    y = train_demo[train_demo['adult_child'] == 1][train_demo['sex']==0]['elbow_to_wrist_cm'],
    mode = 'markers',
    name = 'Female',
    marker=dict(color='rgba(255,0,0,0.5)'),
    showlegend=False,
)
trace4 = go.Scatter(
    x = train_demo[train_demo['adult_child'] == 1][train_demo['sex']==1]['height_cm'],
    y = train_demo[train_demo['adult_child'] == 1][train_demo['sex']==1]['elbow_to_wrist_cm'],
    mode = 'markers',
    name = 'Male',
    marker=dict(color='rgba(0,0,255,0.5)'),
    showlegend=False,
)


fig = make_subplots(rows=1, cols=2, subplot_titles=("Child", "Adult"))

fig.add_traces([trace1,trace2], 1, 1)
fig.add_traces([trace3,trace4], 1, 2)


fig.update_layout(
    title=dict(
    text="<b>Correlation with Height and Elbow_to_Wrist</b>",
    font=dict(size=20, family="Arial", color="black"),),
    xaxis1 = dict(title='Height', ticklen = 5, zeroline=False),
    yaxis1 = dict(title='Elbow_to_Wrist', ticklen = 5, zeroline=False),
    xaxis2 = dict(title='Height', ticklen = 5, zeroline=False),
    yaxis2 = dict(title='Elbow_to_Wrist', ticklen = 5, zeroline=False),
    legend=dict(title='Sex', bordercolor='black', borderwidth=1)
)

fig.show(renderer='iframe')


trace1 = go.Scatter(
    x = train_demo[train_demo['adult_child'] == 0][train_demo['sex']==0]['shoulder_to_wrist_cm'],
    y = train_demo[train_demo['adult_child'] == 0][train_demo['sex']==0]['elbow_to_wrist_cm'],
    mode = 'markers',
    name = 'Female',
    marker=dict(color='rgba(255,0,0,0.5)'),
)
trace2 = go.Scatter(
    x = train_demo[train_demo['adult_child'] == 0][train_demo['sex']==1]['shoulder_to_wrist_cm'],
    y = train_demo[train_demo['adult_child'] == 0][train_demo['sex']==1]['elbow_to_wrist_cm'],
    mode = 'markers',
    name = 'Male',
    marker=dict(color='rgba(0,0,255,0.5)'),
)
trace3 = go.Scatter(
    x = train_demo[train_demo['adult_child'] == 1][train_demo['sex']==0]['shoulder_to_wrist_cm'],
    y = train_demo[train_demo['adult_child'] == 1][train_demo['sex']==0]['elbow_to_wrist_cm'],
    mode = 'markers',
    name = 'Female',
    marker=dict(color='rgba(255,0,0,0.5)'),
    showlegend=False,
)
trace4 = go.Scatter(
    x = train_demo[train_demo['adult_child'] == 1][train_demo['sex']==1]['shoulder_to_wrist_cm'],
    y = train_demo[train_demo['adult_child'] == 1][train_demo['sex']==1]['elbow_to_wrist_cm'],
    mode = 'markers',
    name = 'Male',
    marker=dict(color='rgba(0,0,255,0.5)'),
    showlegend=False,
)


fig = make_subplots(rows=1, cols=2, subplot_titles=("Child", "Adult"))

fig.add_traces([trace1,trace2], 1, 1)
fig.add_traces([trace3,trace4], 1, 2)


fig.update_layout(
    title=dict(
    text="<b>Correlation with Shoulder_to_Wrist and Elbow_to_Wrist</b>",
    font=dict(size=20, family="Arial", color="black"),),
    xaxis1 = dict(title='Shoulder_to_Wrist', ticklen = 5, zeroline=False),
    yaxis1 = dict(title='Elbow_to_Wrist', ticklen = 5, zeroline=False),
    xaxis2 = dict(title='Shoulder_to_Wrist', ticklen = 5, zeroline=False),
    yaxis2 = dict(title='Elbow_to_Wrist', ticklen = 5, zeroline=False),
    legend=dict(title='Sex', bordercolor='black', borderwidth=1)
)

fig.show(renderer='iframe')


train_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
print(f"{c_}Shape of DataFrame: ", train_df.shape)
print(f"\n{Fore.BLUE}{st_}There are total {train_df['sequence_id'].nunique()} seqeunces")
print(f"{Fore.BLUE}{st_}There are total {train_df['subject'].nunique()} subjects")
print(f"{Fore.BLUE}{st_}On average Each Subject did {train_df['sequence_id'].nunique()/train_df['subject'].nunique():.1f} Gesture Experiments")
print(display(train_df))


print(f"\n{Fore.BLUE}{st_}There are total {train_df[train_df['sequence_type'] == 'Target']['gesture'].nunique()} Gesture Type in Target")
print(f"{Fore.BLUE}{st_}There are total {train_df[train_df['sequence_type'] == 'Non-Target']['gesture'].nunique()} Gesture Type in Non-Target")
print("\nAs you know, our task is twofold: first, to distinguish BFRB-like gestures from non-BFRB-like gestures, which are distributed in a 6:4 ratio; and second, to further classify the specific type of BFRB-like gesture.However, as shown in the chart below, each of the eight gesture types within the Target class accounts for exactly 12.5%, reflecting an even distribution across all gesture categories.")


tmp1 = train_df.groupby(['sequence_id']).agg('first')['sequence_type'].value_counts()
tmp2 = train_df[train_df['sequence_type'] == 'Target'].groupby(['sequence_id']).agg('first')['gesture'].value_counts()
tmp3 = train_df[train_df['sequence_type'] == 'Non-Target'].groupby(['sequence_id']).agg('first')['gesture'].value_counts()

trace1 = go.Pie(
    labels = tmp1.index,
    values = tmp1,
    hole = 0.7,
    textinfo = "label+percent",
    showlegend=False
)

trace2 = go.Pie(
    labels = tmp2.index,
    values = tmp2,
    hole = 0.7,
    textinfo = "percent",
    showlegend=False
)
trace3 = go.Pie(
    labels = tmp3.index,
    values = tmp3,
    hole = 0.7,
    textinfo = "percent",
    showlegend=False
)


fig = make_subplots(rows=1, cols=3, subplot_titles=("Target vs Non-Target", "Gesture in Target", "Gesture in Non-Target"),specs=[[{'type': 'domain'}, {'type': 'domain'},{'type': 'domain'}]])

fig.add_trace(trace1, 1,1)
fig.add_trace(trace2, 1,2)
fig.add_trace(trace3, 1,3)

fig.update_layout(
    title="Distribution of Target",
)

fig.show(renderer='iframe')


tmp = train_df.groupby(['subject','sequence_id'])['gesture'].agg("first").reset_index()
tmp2 = tmp.groupby(['subject'])['gesture'].value_counts().reset_index()
tmp3 = tmp2.groupby(['gesture'])['count'].agg('mean').reset_index()

trace = go.Bar(
    x = tmp3['gesture'],
    y = tmp3['count'],
    marker = dict(color = f'rgba(0,0,255,0.2)',
                 line=dict(color=f'rgb(255,255,0)', width=2)),
)

layout = go.Layout(
    title="The Average Gesture Count per Subject"
)

fig = go.Figure(data=trace, layout=layout)

fig.show(renderer='iframe')


tmp1 = train_df.groupby(['subject','sequence_id']).agg('max').reset_index()
tmp2 = train_df.groupby('subject')['sequence_id'].nunique().reset_index()

print(f"\n{Fore.BLUE}{st_}The Average of Sequence Lengths is {np.mean(tmp1['sequence_counter']):.2f}")
print(f"\n{Fore.BLUE}{st_}The Average of Number of Sequnces per subject is {np.mean(tmp2['sequence_id']):.2f}")

trace1 = go.Histogram(
    x = tmp1['sequence_counter'],
    opacity=0.75,
    showlegend=False,
) 

trace2 = go.Histogram(
    x = tmp2['sequence_id'],
    opacity=0.75,
    showlegend=False,
) 

fig = make_subplots(1,2, subplot_titles=("Distribution of Sequence Lengths", 'Distribution of Number of Sequences per Subject'))

fig.add_trace(trace1, 1, 1)
fig.add_trace(trace2, 1, 2)

fig.update_layout(
    xaxis1 = dict(title='Sequence Length (frames)', ticklen = 5, zeroline=False),
    yaxis1 = dict(title='Count', ticklen = 5, zeroline=False),
    xaxis2 = dict(title='Number of Sequences', ticklen = 5, zeroline=False),
    yaxis2 = dict(title='Count', ticklen = 5, zeroline=False),
)

fig.show(renderer='iframe')


print(f"✅ {Fore.BLUE}{st_}There are more outliers above 300 in the Target sequences than in the Non-Target sequences.\n")
print(f"✅ {Fore.BLUE}{st_}Looking first at the gestures within the Target category, the Neck-scratch gesture stands out, exhibiting sequence counts exceeding 300 — a pattern not observed in other gestures. In contrast, within the Non-Target category, the gestures Text on phone and Write name in air are notable.")


fig = make_subplots(rows=1, cols=3, subplot_titles=("Target vs Non-Target", "Gesture in Target", "Gesture in Non-Target"))

for seq_type, color in zip(['Target', 'Non-Target'], ['red', 'blue']):
    fig.add_trace(
        go.Box(
            y=train_df.loc[train_df['sequence_type'] == seq_type, 'sequence_counter'],
            name=seq_type,
            marker_color=color,
            boxmean='sd',
            showlegend=False,
        ), 1, 1
    )


for ges_type in train_df[train_df['sequence_type'] == 'Target']['gesture'].unique():
    fig.add_trace(
        go.Box(
            y=train_df.loc[train_df['gesture'] == ges_type, 'sequence_counter'],
            name=ges_type,
            boxmean='sd',
            showlegend=False,
        ), 1, 2
    )

for ges_type in train_df[train_df['sequence_type'] == 'Non-Target']['gesture'].unique():
    fig.add_trace(
        go.Box(
            y=train_df.loc[train_df['gesture'] == ges_type, 'sequence_counter'],
            name=ges_type,
            boxmean='sd',
            showlegend=False,
        ), 1, 3
    )

fig.update_layout(
    title="Correlation with Gesture and Sequence Counter",
    yaxis = dict(title='Number of Sequences', ticklen = 5, zeroline=False),
)

fig.show(renderer='iframe')


fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Orientation in Target Gesture", "Orientation in Non-Target Gesture")
)

colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']

tmp2 = tmp1[tmp1['sequence_type'] == 'Target']
for orien, col in zip(tmp2['orientation'].unique(), colors):
    counts = tmp2[tmp2['orientation'] == orien]['gesture'].value_counts()
    fig.add_trace(
        go.Bar(
            x=counts.index,
            y=counts.values,
            marker_color=col,
            name=f"{orien}",
        ),
        row=1, col=1
    )


tmp3 = tmp1[tmp1['sequence_type'] == 'Non-Target']
for orien, col in zip(tmp3['orientation'].unique(),colors):
    counts = tmp3[tmp3['orientation'] == orien]['gesture'].value_counts()
    fig.add_trace(
        go.Bar(
            x=counts.index,
            y=counts.values,
            marker_color = col,
            showlegend=False
        ),
        row=1, col=2
    )

fig.update_layout(
    barmode='group',
    xaxis1 = dict(title='Gestures', ticklen = 5, zeroline=False),
    yaxis1 = dict(title='Count', ticklen = 5, zeroline=False),
    xaxis2 = dict(title='Gestures', ticklen = 5, zeroline=False),
    yaxis2 = dict(title='Count', ticklen = 5, zeroline=False),
)

fig.show(renderer='iframe')



train_df.groupby(['subject','sequence_id'])['behavior'].nunique().value_counts()

## ✅ Every Sequence_id has 3 unique behavior including perform gesture
## ✅ But Sequence_id(SEQ_011975) has only 2 unique behavior and not incldue perform gesture
## ✅ So When training Dataset, I'll drop out SEQ_011975


train_df = train_df[train_df['sequence_id'] != 'SEQ_011975'].reset_index(drop=True)


train_df['behavior'].value_counts()

## ✅ There are 4 Behavior(Relaxes and moves hand to target location, moves hand to target location, hand at target location, performs gesture)
## ✅ But Every Sequence only use 3 Behavior 
## ✅ First Method: Relaxes and moves hand to target loaction -> hand at target location -> performs gesture
## ✅ Second Mtehod: Moves hand to target location -> hand at target location -> performs gesture


trace1 = go.Pie(
    labels= train_df['behavior'].value_counts().index,
    values= train_df['behavior'].value_counts(),
    hole=0.7,
    textinfo="label+percent",
    showlegend = False
)

tmp = train_df[train_df['sequence_id'] != 'SEQ_011975']

trace2 = go.Pie(
    labels= tmp.groupby(['subject','sequence_id'])['behavior'].unique().apply(tuple).value_counts().index,
    values= tmp.groupby(['subject','sequence_id'])['behavior'].unique().apply(tuple).value_counts(),
    hole=0.7,
    textinfo="percent",
    showlegend=False,
) 


fig = make_subplots(rows=1, cols=2, subplot_titles=("Behavior Types", "Behavior Flow in each sequence"), specs=[[{'type': 'domain'},{'type': 'domain'}]])

fig.add_trace(trace1, 1, 1)
fig.add_trace(trace2, 1, 2)

fig.show(renderer='iframe')


print(f'\n✅{Fore.BLUE}{st_} As shown in the graph, each gesture exhibits two behavior flows occurring in approximately equal proportions, roughly 50-50. This indicates that the competition organizers have artificially balanced the dataset to ensure an equal distribution of behavior flows for each gesture.')
print("\n✅ On average, each subject performs around 100 gesture measurements. Given this, it is highly likely that each identical gesture was executed with two different behavior flows at least twice.")
print("\n✅ Considering the two behavior types, four orientations, and eighteen gestures, the total combinations would be 2 × 4 × 18 = 144. However, since the ten Non-Target orientations have fewer than four orientations, it is likely that the average number of measurements per subject is around 100.")

tmp = train_df.groupby(['subject','sequence_id']).agg('max').reset_index().drop(columns='behavior')
tmp2 = train_df.groupby(['subject','sequence_id'])['behavior'].unique().apply(lambda x: " -> ".join(x)).reset_index()[['sequence_id','behavior']]
tmp1 = tmp.merge(tmp2, on='sequence_id', how='left')

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Behavior flow in Target Gesture", "Behavior flow in Non-Target Gesture")
)

colors = ['#636EFA', '#EF553B']

tmp2 = tmp1[tmp1['sequence_type'] == 'Target']
for i, (bh, col) in enumerate(zip(tmp2['behavior'].unique(), colors)):
    counts = tmp2[tmp2['behavior'] == bh]['gesture'].value_counts()
    fig.add_trace(
        go.Bar(
            x=counts.index,
            y=counts.values,
            marker_color=col,
            name=f"FLOW_{i+1}",
        ),
        row=1, col=1
    )


tmp3 = tmp1[tmp1['sequence_type'] == 'Non-Target']
for bh, col in zip(tmp3['behavior'].unique(),colors):
    counts = tmp3[tmp3['behavior'] == bh]['gesture'].value_counts()
    fig.add_trace(
        go.Bar(
            x=counts.index,
            y=counts.values,
            marker_color = col,
            showlegend=False
        ),
        row=1, col=2
    )

fig.update_layout(
    barmode='group',
    xaxis1 = dict(title='Gestures', ticklen = 5, zeroline=False),
    yaxis1 = dict(title='Count', ticklen = 5, zeroline=False),
    xaxis2 = dict(title='Gestures', ticklen = 5, zeroline=False),
    yaxis2 = dict(title='Count', ticklen = 5, zeroline=False),
)

fig.show(renderer='iframe')



tmp = train_df.groupby(['subject','sequence_id'])['behavior'].value_counts().unstack().reset_index().fillna(0)
tmp['seqeunce_length'] = tmp['Hand at target location'] + tmp['Moves hand to target location'] + tmp['Performs gesture'] + tmp['Relaxes and moves hand to target location']
tmp['Hand at target location Ratio'] = tmp['Hand at target location']/tmp['seqeunce_length'] * 100
tmp['Moves hand to target location Ratio'] = tmp['Moves hand to target location']/tmp['seqeunce_length'] * 100
tmp['Performs gesture Ratio'] = tmp['Performs gesture']/tmp['seqeunce_length'] * 100
tmp['Relaxes and moves hand to target location Ratio'] = tmp['Relaxes and moves hand to target location']/tmp['seqeunce_length'] * 100

train_df = train_df.merge(tmp[tmp.columns[1:]], on='sequence_id', how='left')
train_df['sequence_counter_pct'] = train_df['sequence_counter'] / train_df['seqeunce_length'] * 100

tmp = train_df.groupby(['subject','sequence_id'])['behavior'].unique().apply(lambda x: " -> ".join(x)).reset_index().rename(columns={'behavior': 'behavior_flow'})
train_df = train_df.merge(tmp[tmp.columns[1:]], on='sequence_id', how='left')


FLOW_1 = 'Relaxes and moves hand to target location -> Hand at target location -> Performs gesture'
FLOW_2 = 'Moves hand to target location -> Hand at target location -> Performs gesture'

print(f"{Fore.GREEN}{st_}### Flow: {FLOW_1} ###")
print(f"\n{Fore.BLUE}{st_}✅ The Average Percentage of Relaxes and moves hand to target location in Seqeunce: {np.mean(train_df[train_df['behavior_flow'] == FLOW_1]['Relaxes and moves hand to target location Ratio']):.1f}")
print(f"\n{Fore.BLUE}{st_}✅ The Average Percentage of Hand at target location in Seqeunce: {np.mean(train_df[train_df['behavior_flow'] == FLOW_1]['Hand at target location Ratio']):.1f}")
print(f"\n{Fore.BLUE}{st_}✅ The Average Percentage of Performs gesture in Seqeunce: {np.mean(train_df[train_df['behavior_flow'] == FLOW_1]['Performs gesture Ratio']):.1f}")

print(f"\n{Fore.GREEN}{st_}### Flow: {FLOW_2} ###")
print(f"\n{Fore.BLUE}{st_}✅ The Average Percentage of Moves hand to target location in Seqeunce: {np.mean(train_df[train_df['behavior_flow'] == FLOW_2]['Moves hand to target location Ratio']):.1f}")
print(f"\n{Fore.BLUE}{st_}✅ The Average Percentage of Hand at target location in Seqeunce: {np.mean(train_df[train_df['behavior_flow'] == FLOW_2]['Hand at target location Ratio']):.1f}")
print(f"\n{Fore.BLUE}{st_}✅ The Average Percentage of Performs gesture in Seqeunce: {np.mean(train_df[train_df['behavior_flow'] == FLOW_2]['Performs gesture Ratio']):.1f}")




from plotly.subplots import make_subplots
import plotly.graph_objects as go

print(f"\n{Fore.GREEN}{st_}### Flow1: {FLOW_1} ###")
print(f"\n{Fore.GREEN}{st_}### Flow2: {FLOW_2} ###")

fig = make_subplots(1, 2, subplot_titles=("Flow 1", "Flow 2"))

tmp1 = train_df.loc[(train_df['behavior_flow'] == FLOW_1) & (train_df['sequence_type'] == 'Target')]

for bh in tmp1['behavior'].unique():
    sub_df = tmp1[tmp1['behavior'] == bh].groupby('gesture')[f'{bh} Ratio'].mean().reset_index()

    fig.add_trace(
        go.Bar(
            y=sub_df['gesture'],
            x=sub_df[f'{bh} Ratio'],
            name=bh,
            orientation='h',
            showlegend=False,
        ), 1, 1
    )

tmp2 = train_df.loc[(train_df['behavior_flow'] == FLOW_2) & (train_df['sequence_type'] == 'Target')]

for bh in tmp2['behavior'].unique():
    sub_df = tmp2[tmp2['behavior'] == bh].groupby('gesture')[f'{bh} Ratio'].mean().reset_index()

    fig.add_trace(
        go.Bar(
            y=sub_df['gesture'],
            x=sub_df[f'{bh} Ratio'],
            name=bh,
            orientation='h',
            showlegend=False,
        ), 1, 2
    )

fig.update_layout(
    title="Distribution of Sequence Counter per Behavior in Target",
    barmode='stack',
)

for annotation in fig.layout.annotations:
    annotation.font.size = 12

fig.show(renderer='iframe')



from plotly.subplots import make_subplots
import plotly.graph_objects as go

print(f"\n{Fore.GREEN}{st_}### Flow1: {FLOW_1} ###")
print(f"\n{Fore.GREEN}{st_}### Flow2: {FLOW_2} ###")

fig = make_subplots(1, 2, subplot_titles=("Flow 1", "Flow 2"))

tmp1 = train_df.loc[(train_df['behavior_flow'] == FLOW_1) & (train_df['sequence_type'] != 'Target')]

for bh in tmp1['behavior'].unique():
    sub_df = tmp1[tmp1['behavior'] == bh].groupby('gesture')['sequence_counter_pct'].mean().reset_index()

    fig.add_trace(
        go.Bar(
            y=sub_df['gesture'],
            x=sub_df['sequence_counter_pct'],
            name=bh,
            orientation='h',
            showlegend=False,
        ), 1, 1
    )

tmp2 = train_df.loc[(train_df['behavior_flow'] == FLOW_2) & (train_df['sequence_type'] != 'Target')]

for bh in tmp2['behavior'].unique():
    sub_df = tmp2[tmp2['behavior'] == bh].groupby('gesture')['sequence_counter_pct'].mean().reset_index()

    fig.add_trace(
        go.Bar(
            y=sub_df['gesture'],
            x=sub_df['sequence_counter_pct'],
            name=bh,
            orientation='h',
            showlegend=False,
        ), 1, 2
    )

fig.update_layout(
    title="Distribution of Sequence Counter per Behavior in Non-Target",
    barmode='stack',
)

for annotation in fig.layout.annotations:
    annotation.font.size = 12

fig.show(renderer='iframe')



train_df['phase'].value_counts()

## ✅ Transition: Relaxes and moves hand to target location, Moves hand to target location, Hand at target location
## ✅ Gesture: Performs gesture
## We'll use two information(Transition, Gesture) when training model.


tmp = train_df.groupby(['subject','sequence_id'])['phase'].value_counts().unstack().reset_index()
tmp['sum'] = tmp['Gesture'] + tmp['Transition']
tmp['Gesture_Ratio'] = tmp['Gesture']/tmp['sum'] * 100
tmp['Transition_Ratio'] = tmp['Transition']/tmp['sum'] * 100

train_df = train_df.merge(tmp[['sequence_id','Gesture','Transition','Gesture_Ratio','Transition_Ratio']], on='sequence_id', how='left')


print(f"\n{Fore.BLUE}{st_} The Average Gesture Ratio in Sequence: {np.mean(train_df['Gesture_Ratio']):.1f}")
print(f"\n{Fore.BLUE}{st_} The Average Transition Ratio in Sequence: {np.mean(train_df['Transition_Ratio']):.1f}")

trace1 = go.Histogram(
    x = train_df['Gesture_Ratio'],
    opacity=0.5,
    marker_color='red',
    name = 'Gesture Ratio',
)

trace2 = go.Histogram(
    x = train_df['Transition_Ratio'],
    opacity=0.5,
    marker_color='blue',
    name = 'Transition Ratio',
)

fig = go.Figure()

fig.add_traces([trace1, trace2])

fig.update_layout(
    barmode='overlay',
    xaxis_title="Percent",
)


fig.show(renderer='iframe')


tmp = train_df.groupby(['subject','sequence_id']).agg('first').reset_index()


fig = make_subplots(1,2)


fig.add_trace(
        go.Box(
            x=tmp['gesture'],
            y=tmp['Transition_Ratio'],
            name='Transition_Ratio',
            boxmean='sd'
        ), 1, 1
    )
fig.add_trace(
        go.Box(
            x=tmp['gesture'],
            y=tmp['Gesture_Ratio'],
            name='Gesture Ratio',
            boxmean='sd'
        ), 1, 2
    )

fig.add_hrect(
    y0=95,
    y1=100,
    fillcolor="red",
    opacity=0.2,
    layer="below",
    line_width=0,
    row=1, col=1
)

fig.add_hrect(
    y0=85,
    y1=100,
    fillcolor="red",
    opacity=0.2,
    layer="below",
    line_width=0,
    row=1, col=2
)

fig.update_layout(
    title= "Distribution of Phase Ratio",
    boxmode='group' 
)

fig.show(renderer='iframe')



trace1 = go.Bar(
    y = tmp[tmp['Gesture_Ratio'] > 85]['sequence_id'],
    x = tmp[tmp['Gesture_Ratio'] > 85]['Gesture_Ratio'],
    orientation = 'h',
    name = 'Gesture',
    marker = dict(color='rgba(255,0,0,0.5)',
                 line=dict(color='red', width=1)),
)
trace2 = go.Bar(
    y = tmp[tmp['Gesture_Ratio'] > 85]['sequence_id'],
    x = tmp[tmp['Gesture_Ratio'] > 85]['Transition_Ratio'],
    orientation = 'h',
    name = 'Transition',
    marker = dict(color='rgba(0,0,255,0.5)',
                 line=dict(color='blue', width=1)),
)
trace3 = go.Bar(
    y = tmp[tmp['Transition_Ratio'] > 95]['sequence_id'],
    x = tmp[tmp['Transition_Ratio'] > 95]['Gesture_Ratio'],
    orientation = 'h',
    name = 'Gesture',
    marker = dict(color='rgba(255,0,0,0.5)',
                 line=dict(color='red', width=1)),
    showlegend=False,
)
trace4 = go.Bar(
    y = tmp[tmp['Transition_Ratio'] > 95]['sequence_id'],
    x = tmp[tmp['Transition_Ratio'] > 95]['Transition_Ratio'],
    orientation = 'h',
    name = 'Transition',
    marker = dict(color='rgba(0,0,255,0.5)',
                 line=dict(color='blue', width=1)),
    showlegend=False,
)

fig = make_subplots(rows=1, cols=2, subplot_titles=("Too many Gesture", "Too many Transition"))

fig.add_traces([trace1,trace2], 1, 1)
fig.add_traces([trace3,trace4], 1, 2)

fig.update_layout(
    barmode='stack'
)

fig.show(renderer='iframe')


gesture_ids = tmp[tmp['Gesture_Ratio'] > 85]['sequence_id'].values
transition_ids = tmp[tmp['Transition_Ratio'] > 95]['sequence_id'].values

print(f"\n{Fore.BLUE}{st_} {gesture_ids}")
print(f"\n{Fore.BLUE}{st_} {transition_ids}")

remove_ids = np.unique(np.concatenate([gesture_ids, transition_ids]))

train_df = train_df[~train_df['sequence_id'].isin(remove_ids)].reset_index(drop=True)


fig = go.Figure()

tmp1 = train_df[train_df['sequence_type'] == 'Target']

for bh in tmp1['phase'].unique():
    fig.add_trace(
        go.Box(
            x=tmp1[tmp1['phase'] == bh]['gesture'],
            y=tmp1[tmp1['phase'] == bh]['sequence_counter_pct'],
            name=bh,
            boxmean='sd'
        )
    )


fig.update_layout(
    title= "Distribution of Sequence Counter per Phase in Target",
    boxmode='group' 
)

fig.show(renderer='iframe')



fig = go.Figure()

tmp2 = train_df[train_df['sequence_type'] == 'Non-Target']

for bh in tmp2['phase'].unique():
    fig.add_trace(
        go.Box(
            x=tmp2[tmp2['phase'] == bh]['gesture'],
            y=tmp2[tmp2['phase'] == bh]['sequence_counter_pct'],
            name=bh,
            boxmean='sd'
        )
    )


fig.update_layout(
    title= "Distribution of Sequence Counter per Phase in Non-Target",
    boxmode='group' 
)

fig.show(renderer='iframe')



gc.collect()


acc_list = ['acc_x','acc_y','acc_z']
rot_list = ['rot_x', 'rot_y', 'rot_z','rot_w']

acc_color = ['red','yellow','orange']
rot_color = ['red', 'yellow', 'yellowgreen', 'skyblue']


train_df[acc_list + rot_list].info()


tmp = train_df.groupby(['sequence_id']).agg(
    rot_x_na_ratio = ('rot_x', lambda x: x.isna().mean() * 100)
).reset_index()

tmp['rot_x_na_ratio'].value_counts()


rot_nan_list = tmp[tmp['rot_x_na_ratio'] == 100]['sequence_id'].values
rot_nan_list


import cudf, cuml, cupy
print('RAPIDS', cudf.__version__)


from cuml import UMAP
umap = UMAP()

cols = acc_list + rot_list
stats = ['mean', 'min', 'max', 'std'] 

agg_dict = {col: stats for col in cols}
tmp = cudf.from_pandas(train_df.dropna().reset_index(drop=True))

tmp_stats = tmp.groupby('sequence_id').agg(agg_dict)

tmp_stats.columns = [f"{col}_{stat}" for col, stat in tmp_stats.columns]

tmp_stats = tmp_stats.reset_index()

tmp_meta = tmp[['sequence_id', 'gesture', 'sequence_type']]\
    .drop_duplicates(subset=['sequence_id'])\
    .dropna()

tmp = tmp_stats.merge(tmp_meta, on='sequence_id', how='left')

tmp = tmp.to_pandas()
tmp.loc[tmp['sequence_type'] != 'Target', 'gesture'] = 'Non-Target'
tmp_cudf = cudf.from_pandas(tmp)


%%time

features = tmp_cudf.drop(columns=['sequence_id','gesture','sequence_type'])

embed_2d = umap.fit_transform(features)
embed_2d = embed_2d.to_numpy()
embed_df = pd.DataFrame(embed_2d, columns=['x','y'])
embed_df['gesture'] = tmp['gesture'].values

fig = go.Figure()

for ges in tmp['gesture'].unique():
    tmp2 = embed_df[embed_df['gesture'] == ges]
    
    fig.add_trace(
        go.Scatter(
            x = tmp2['x'],
            y = tmp2['y'],
            mode = 'markers',
            marker=dict(size=6, opacity=0.7),
            name = ges,
        )
    )

fig.update_layout(
    title='Correlation with Gesture & Thermopile Sensor',
    legend=dict(title='Gesture Type', bordercolor='black', borderwidth=1)
    
)

fig.show(renderer='iframe')


fig = make_subplots(rows=1, cols=2, subplot_titles=("Transition","Gesture"))

tmp1 = train_df[train_df['phase'] == 'Transition']

for acc, col in zip(acc_list, acc_color):
    fig.add_trace(
        go.Box(
            x=tmp1['sequence_type'],
            y=tmp1[acc],
            name=acc,
            marker_color=col,
            boxmean='sd'
        ), 1, 1
        )
            
tmp2 = train_df[train_df['phase'] == 'Gesture']

for acc, col in zip(acc_list, acc_color):
    fig.add_trace(
        go.Box(
            x=tmp2['sequence_type'],
            y=tmp2[acc],
            name=acc,
            marker_color=col,
            boxmean='sd',
            showlegend=False
        ), 1, 2
        )

fig.update_layout(
    title= "Distribution of Accelerator",
    boxmode='group' 
)

fig.show(renderer="iframe")


# ============================================================================
# Outlier Removal for Specific Gesture
# ============================================================================
# In general, when 'phase' is 'Gesture', no gesture shows acc_z values over 20.
# However, the gesture 'Forehead - pull hairline' includes an outlier (acc_z = 23)
# specifically in sequence_id = 'SEQ_012112'. This outlier will be excluded
# to maintain consistency in gesture dynamics.
# =============================================

acc_z_outline = train_df.loc[
    (train_df['phase'] == 'Gesture') & 
    (train_df['acc_z'] > 20)
]['sequence_id'].values[0]

train_df = train_df[train_df['sequence_id'] != acc_z_outline].reset_index(drop=True)


seq_list = np.random.choice(train_df[train_df['behavior_flow'] == FLOW_1]['sequence_id'].unique(), 3)

titles = [seq for seq in seq_list]

fig = make_subplots(rows=3, cols=1, subplot_titles=titles)

for idx, seq_id in enumerate(seq_list, start=1):
    
    tmp = train_df[train_df['sequence_id'] == seq_id].reset_index()
    
    for acc, col in zip(acc_list, acc_color):
        fig.add_trace(
            go.Scatter(
                x = tmp.index,
                y = tmp[acc],
                mode = 'lines',
                name = acc,
                marker_color = col,
                showlegend = (idx == 1),
            ), idx, 1
        )
        
    fig.add_vrect(
        x0=0, x1=tmp.iloc[0]['Relaxes and moves hand to target location']-1,
        fillcolor="red",
        opacity=0.1,
        line_width=0,
        row=idx, col=1,
        annotation_text="Transition",
        annotation_position="top left"
    )
    fig.add_vrect(
        x0=tmp.iloc[0]['Relaxes and moves hand to target location']-1, x1=tmp.iloc[0]['Hand at target location'] + tmp.iloc[0]['Relaxes and moves hand to target location'] -1,
        fillcolor="green",
        opacity=0.1,
        line_width=0,
        row=idx, col=1,
        annotation_text="Pause",
        annotation_position="top right"
    )
    fig.add_vrect(
        x0=tmp.iloc[0]['Hand at target location'] + tmp.iloc[0]['Relaxes and moves hand to target location'] -1, x1=len(tmp)-1,
        fillcolor="blue",
        opacity=0.1,
        line_width=0,
        row=idx, col=1,
        annotation_text="Gesture",
        annotation_position="top right"
    )
fig.update_layout(
    title=FLOW_1,
)

fig.show(renderer="iframe")


seq_list = np.random.choice(train_df[train_df['behavior_flow'] == FLOW_2]['sequence_id'].unique(), 3)

titles = [seq for seq in seq_list]

fig = make_subplots(rows=3, cols=1, subplot_titles=titles)

for idx, seq_id in enumerate(seq_list, start=1):
    
    tmp = train_df[train_df['sequence_id'] == seq_id].reset_index()
    
    for acc, col in zip(acc_list, acc_color):
        fig.add_trace(
            go.Scatter(
                x = tmp.index,
                y = tmp[acc],
                mode = 'lines',
                name = acc,
                marker_color = col,
                showlegend = (idx == 1),
            ), idx, 1
        )
        
    fig.add_vrect(
        x0=0, x1=tmp.iloc[0]['Moves hand to target location']-1,
        fillcolor="red",
        opacity=0.1,
        line_width=0,
        row=idx, col=1,
        annotation_text="Transition",
        annotation_position="top left"
    )
    fig.add_vrect(
        x0=tmp.iloc[0]['Moves hand to target location']-1, x1=tmp.iloc[0]['Hand at target location'] + tmp.iloc[0]['Moves hand to target location'] -1,
        fillcolor="green",
        opacity=0.1,
        line_width=0,
        row=idx, col=1,
        annotation_text="Pause",
        annotation_position="top right"
    )
    fig.add_vrect(
        x0=tmp.iloc[0]['Hand at target location'] + tmp.iloc[0]['Moves hand to target location'] -1, x1=len(tmp)-1,
        fillcolor="blue",
        opacity=0.1,
        line_width=0,
        row=idx, col=1,
        annotation_text="Gesture",
        annotation_position="top right"
    )
fig.update_layout(
    title=FLOW_2,
)

fig.show(renderer="iframe")


fig = make_subplots(rows=1, cols=2, subplot_titles=("Transition","Gesture"))

tmp1 = train_df[train_df['phase'] == 'Transition']

for rot, col in zip(rot_list, rot_color):
    fig.add_trace(
        go.Box(
            x=tmp1['sequence_type'],
            y=tmp1[rot],
            name=rot,
            marker_color=col,
            boxmean='sd'
        ), 1, 1
        )
            
tmp2 = train_df[train_df['phase'] == 'Gesture']

for rot, col in zip(rot_list, rot_color):
    fig.add_trace(
        go.Box(
            x=tmp2['sequence_type'],
            y=tmp2[rot],
            name=rot,
            marker_color=col,
            boxmean='sd',
            showlegend=False
        ), 1, 2
        )

fig.update_layout(
    title= "Distribution of Rotation",
    boxmode='group' 
)

fig.show(renderer="iframe")


seq_list = np.random.choice(train_df[train_df['behavior_flow'] == FLOW_1]['sequence_id'].unique(), 3)

titles = [seq for seq in seq_list]

fig = make_subplots(rows=3, cols=1, subplot_titles=titles)

for idx, seq_id in enumerate(seq_list, start=1):
    
    tmp = train_df[train_df['sequence_id'] == seq_id].reset_index()
    
    for rot, col in zip(rot_list, rot_color):
        fig.add_trace(
            go.Scatter(
                x = tmp.index,
                y = tmp[rot],
                mode = 'lines',
                name = rot,
                marker_color = col,
                showlegend = (idx == 1),
            ), idx, 1
        )
        
    fig.add_vrect(
        x0=0, x1=tmp.iloc[0]['Relaxes and moves hand to target location']-1,
        fillcolor="red",
        opacity=0.1,
        line_width=0,
        row=idx, col=1,
        annotation_text="Transition",
        annotation_position="top left"
    )
    fig.add_vrect(
        x0=tmp.iloc[0]['Relaxes and moves hand to target location']-1, x1=tmp.iloc[0]['Hand at target location'] + tmp.iloc[0]['Relaxes and moves hand to target location'] -1,
        fillcolor="green",
        opacity=0.1,
        line_width=0,
        row=idx, col=1,
        annotation_text="Pause",
        annotation_position="top right"
    )
    fig.add_vrect(
        x0=tmp.iloc[0]['Hand at target location'] + tmp.iloc[0]['Relaxes and moves hand to target location'] -1, x1=len(tmp)-1,
        fillcolor="blue",
        opacity=0.1,
        line_width=0,
        row=idx, col=1,
        annotation_text="Gesture",
        annotation_position="top right"
    )
fig.update_layout(
    title=FLOW_1,
)

fig.show(renderer="iframe")


seq_list = np.random.choice(train_df[train_df['behavior_flow'] == FLOW_2]['sequence_id'].unique(), 3)

titles = [seq for seq in seq_list]

fig = make_subplots(rows=3, cols=1, subplot_titles=titles)

for idx, seq_id in enumerate(seq_list, start=1):
    
    tmp = train_df[train_df['sequence_id'] == seq_id].reset_index()
    
    for rot, col in zip(rot_list, rot_color):
        fig.add_trace(
            go.Scatter(
                x = tmp.index,
                y = tmp[rot],
                mode = 'lines',
                name = rot,
                marker_color = col,
                showlegend = (idx == 1),
            ), idx, 1
        )
        
    fig.add_vrect(
        x0=0, x1=tmp.iloc[0]['Moves hand to target location']-1,
        fillcolor="red",
        opacity=0.1,
        line_width=0,
        row=idx, col=1,
        annotation_text="Transition",
        annotation_position="top left"
    )
    fig.add_vrect(
        x0=tmp.iloc[0]['Moves hand to target location']-1, x1=tmp.iloc[0]['Hand at target location'] + tmp.iloc[0]['Moves hand to target location'] -1,
        fillcolor="green",
        opacity=0.1,
        line_width=0,
        row=idx, col=1,
        annotation_text="Pause",
        annotation_position="top right"
    )
    fig.add_vrect(
        x0=tmp.iloc[0]['Hand at target location'] + tmp.iloc[0]['Moves hand to target location'] -1, x1=len(tmp)-1,
        fillcolor="blue",
        opacity=0.1,
        line_width=0,
        row=idx, col=1,
        annotation_text="Gesture",
        annotation_position="top right"
    )
fig.update_layout(
    title=FLOW_2,
)

fig.show(renderer="iframe")


import librosa
from scipy import interpolate

def inter_signal(x):
    original_len = len(x)
    interp_func = interpolate.interp1d(np.linspace(0, 1, original_len), x, kind='linear')
    x_interp = interp_func(np.linspace(0, 1, 1024))
    return x_interp

def spectrogram_from_imu(df, seq_id, sr=64):
    tmp = df[df['sequence_id'] == seq_id].reset_index()

    img_acc = np.zeros((192,128,3), dtype='float32')
    img_rot = np.zeros((192,96,4), dtype='float32')

    for i, acc in enumerate(acc_list):
        x = tmp[acc].values
        m = np.nanmean(x)
        x = np.nan_to_num(x, nan=m)
        x = inter_signal(x)

        mel_spec = librosa.feature.melspectrogram(y=x, sr=sr, hop_length=len(x)//128, n_fft=128,
                                                  n_mels=192, fmin=0, fmax=sr//2)
        width = (mel_spec.shape[1]//16)*16
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:,:width]

        mel_spec_db = (mel_spec_db+40)/40
        img_acc[:,:,i] = mel_spec_db

    for i, rot in enumerate(rot_list):
        x = tmp[rot].values
        m = np.nanmean(x)
        x = np.nan_to_num(x, nan=m)
        x = inter_signal(x)

        mel_spec = librosa.feature.melspectrogram(y=x, sr=sr, hop_length=len(x)//96, n_fft=96,
                                                  n_mels=192, fmin=0, fmax=sr//2)
        width = (mel_spec.shape[1]//16)*16
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:,:width]

        mel_spec_db = (mel_spec_db+40)/40
        img_rot[...,i] = mel_spec_db

    

    return img_acc, img_rot


seq_list = np.random.choice(train_df['sequence_id'].unique(), 2)


plt.figure(figsize=(14, 6))

for i, seq in enumerate(seq_list):
    img_acc, img_rot = spectrogram_from_imu(train_df, seq)

    for j, acc in enumerate(acc_list):
        plt.subplot(2, 7, 7*i + j + 1)
        plt.imshow(img_acc[...,j], aspect='auto', origin='lower', cmap='jet')
        plt.title(f'{seq}_{acc}')
        plt.axis('off')

    for j, rot in enumerate(rot_list):
        plt.subplot(2, 7, 7*i + j + 4)
        plt.imshow(img_rot[...,j], aspect='auto', origin='lower', cmap='jet')
        plt.title(f'{seq}_{rot}')
        plt.axis('off')


plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 6))

for i, seq in enumerate(seq_list):
    img_acc, img_rot = spectrogram_from_imu(train_df, seq)

    img_acc = np.concatenate([img_acc[...,0], img_acc[...,1], img_acc[...,2]], axis=1) # (192, 384)
    img_rot = np.concatenate([img_rot[...,0], img_rot[...,1], img_rot[...,2], img_rot[...,3]], axis=1) # (192,384)

    img = np.concatenate([img_acc, img_rot], axis=0)

    plt.subplot(1,2,i+1)
    plt.imshow(img, aspect='auto', origin='lower', cmap='jet')

plt.tight_layout()
plt.show()



gc.collect()


thm_list = ['thm_1','thm_2','thm_3','thm_4','thm_5']
thm_color = ['#FF5733',  '#33C1FF',  '#33FF57', '#FFC300','#8E44AD'] 


train_df[thm_list].info()


tmp = train_df.groupby(['sequence_id']).agg(
    thm_1_na_ratio = ('thm_1', lambda x: x.isna().mean()*100),
    thm_2_na_ratio = ('thm_2', lambda x: x.isna().mean()*100),
    thm_3_na_ratio = ('thm_3', lambda x: x.isna().mean()*100),
    thm_4_na_ratio = ('thm_4', lambda x: x.isna().mean()*100),
    thm_5_na_ratio = ('thm_5', lambda x: x.isna().mean()*100),

).reset_index()


thm_1_na_list = tmp[tmp['thm_1_na_ratio'] > 50]['sequence_id'].values
thm_2_na_list = tmp[tmp['thm_2_na_ratio'] > 50]['sequence_id'].values
thm_3_na_list = tmp[tmp['thm_3_na_ratio'] > 50]['sequence_id'].values
thm_4_na_list = tmp[tmp['thm_4_na_ratio'] > 50]['sequence_id'].values
thm_5_na_list = tmp[tmp['thm_5_na_ratio'] > 50]['sequence_id'].values

all_ids = np.concatenate([thm_1_na_list, thm_2_na_list, thm_3_na_list, thm_4_na_list, thm_5_na_list])
thm_na_list = np.unique(all_ids)
len(thm_na_list)


cols = thm_list
stats = ['mean', 'min', 'max', 'std'] 

agg_dict = {col: stats for col in cols}
tmp = cudf.from_pandas(train_df.dropna().reset_index(drop=True))

tmp_stats = tmp.groupby('sequence_id').agg(agg_dict)

tmp_stats.columns = [f"{col}_{stat}" for col, stat in tmp_stats.columns]

tmp_stats = tmp_stats.reset_index()

tmp_meta = tmp[['sequence_id', 'gesture', 'sequence_type']]\
    .drop_duplicates(subset=['sequence_id'])\
    .dropna()

tmp = tmp_stats.merge(tmp_meta, on='sequence_id', how='left')

tmp = tmp.to_pandas()
tmp.loc[tmp['sequence_type'] != 'Target', 'gesture'] = 'Non-Target'
tmp_cudf = cudf.from_pandas(tmp)


%%time

features = tmp_cudf.drop(columns=['sequence_id','gesture','sequence_type'])

embed_2d = umap.fit_transform(features)
embed_2d = embed_2d.to_numpy()
embed_df = pd.DataFrame(embed_2d, columns=['x','y'])
embed_df['gesture'] = tmp['gesture'].values

fig = go.Figure()

for ges in tmp['gesture'].unique():
    tmp2 = embed_df[embed_df['gesture'] == ges]
    
    fig.add_trace(
        go.Scatter(
            x = tmp2['x'],
            y = tmp2['y'],
            mode = 'markers',
            marker=dict(size=6, opacity=0.7),
            name = ges,
        )
    )

fig.update_layout(
    title='Correlation with Gesture & IMU Devices',
    legend=dict(title='Gesture Type', bordercolor='black', borderwidth=1)
    
)

fig.show(renderer='iframe')


tmp = tmp.reset_index()


print(f"\n{Fore.BLUE}{st_} ☑️ All thermopile devices show a strong correlation (above 0.7) with each other, indicating closely related behavior. \n ☑️ However, Thermopile 3 stands out as it has relatively low correlation with all other devices, suggesting that it should be examined more carefully")


import plotly.express as px

features =  ['thm_1_mean','thm_2_mean','thm_3_mean','thm_4_mean','thm_5_mean']

corr_matrix = tmp[features].corr()

fig = px.imshow(corr_matrix,
                text_auto=True,
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                aspect='auto',
                title='Correlation Matrix')

fig.show(renderer="iframe")


def draw_bbox(image, box, label):
    output = image.copy()

    text_width, text_height = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]

    cv2.rectangle(output, (box[0], box[1]-text_height-5),
                  (box[0]+text_width-5, box[1]), (255,255,255), -1)
    cv2.putText(output, label, (box[0], box[1]-3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,200), 1, cv2.LINE_AA)
    
    return output


from matplotlib import animation, rc, cm
from matplotlib.colors import Normalize
import cv2

rc('animation', html='jshtml')

def get_colormap_color(value, vmin, vmax, cmap_name='coolwarm'):
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(norm(value))
    rgb = tuple(int(255 * x) for x in rgba[:3])
    return rgb  # (R, G, B)

def create_animation(sequence_id):
    ims = []

    tmp = train_df[train_df['sequence_id'] == sequence_id].reset_index()

    all_thm_values = tmp[[f'thm_{i}' for i in range(1, 6)]].values.flatten()
    vmin = np.nanmin(all_thm_values)
    vmax = np.nanmax(all_thm_values)


    for i, row in tqdm(tmp.iterrows(), desc='Create Thermopile Image'):

        img = np.zeros((256,256,3), dtype='float32')
        
        color1 = get_colormap_color(row['thm_1'], vmin, vmax)
        color2 = get_colormap_color(row['thm_2'], vmin, vmax)
        color3 = get_colormap_color(row['thm_3'], vmin, vmax)
        color4 = get_colormap_color(row['thm_4'], vmin, vmax)
        color5 = get_colormap_color(row['thm_5'], vmin, vmax)

        # Thm_1
        cv2.rectangle(img, (150, 70), (170, 90), color=color1, thickness=-1)
        cv2.rectangle(img, (150, 70), (170, 90), color=(0, 0, 255), thickness=1)
        img = draw_bbox(img, (150, 70, 170, 90), label=f'1: {row["thm_1"]:.1f}')
        # Thm_2
        cv2.rectangle(img, (150, 30), (170, 50), color=color2, thickness=-1)
        cv2.rectangle(img, (150, 30), (170, 50), color=(0, 0, 255), thickness=1)
        img = draw_bbox(img, (150, 30, 170, 50), label=f'2: {row["thm_2"]:.1f}')
        # Thm_3
        cv2.rectangle(img, (190, 100), (210, 120), color=color3, thickness=-1)
        cv2.rectangle(img, (190, 100), (210, 120), color=(0, 0, 255), thickness=1)
        img = draw_bbox(img, (190, 100, 210, 120), label=f'3: {row["thm_3"]:.1f}')
        # Thm_4
        cv2.rectangle(img, (150, 210), (170, 230), color=color4, thickness=-1)
        cv2.rectangle(img, (150, 210), (170, 230), color=(0, 0, 255), thickness=1)
        img = draw_bbox(img, (150, 210, 170, 230), label=f'4: {row["thm_4"]:.1f}')
        # Thm_5
        cv2.rectangle(img, (30, 100), (50, 120), color=color5, thickness=-1)
        cv2.rectangle(img, (30, 100), (50, 120), color=(0, 0, 255), thickness=1)
        img = draw_bbox(img, (30, 100, 50, 120), label=f'5: {row["thm_5"]:.1f}')

        ims.append(img)

    fig = plt.figure(figsize=(6,6))
    plt.axis('off')
    im = plt.imshow(ims[0].astype(np.uint8))

    def animate_func(i):
        im.set_array(ims[i].astype(np.uint8))
        return [im]
        
    plt.close(fig)

    return animation.FuncAnimation(fig, animate_func, frames=len(ims), interval = 1000 // 24)

seq_id = np.random.choice(train_df['sequence_id'].unique())
anim = create_animation(seq_id)
anim


print(f"\n{Fore.BLUE}{st_} ☑️ Unlike the other thermopiles, thm_3 has values of zero, which are considered outliers. This likely explains why the correlation of thm_3_mean with the means of other thermopiles is low in the heatmap above, as the presence of zeros may have distorted the mean values")


tmp = train_df[~train_df['sequence_id'].isin(thm_na_list)].reset_index(drop=True).copy()

plt.style.use('Solarize_Light2')

fig, axes = plt.subplots(2, 3, figsize=(12, 6))
axes = axes.flatten()
color_list = ['blue', 'red']

sns.set(style='whitegrid', context='notebook')
for i in range(5):
    sns.boxplot(
        x='phase', y=f'thm_{i+1}', data=tmp,
        palette=dict(zip(tmp['phase'].unique(), color_list)),
        ax=axes[i], showmeans=True
    )
    axes[i].set_title(f'thm_{i}')


axes[5].axis('off')

plt.suptitle('<Thermopile with Phase>', fontweight='bold')
plt.tight_layout()
plt.show()


print(f"\n{Fore.BLUE}{st_} ☑️ As is well understood, during the Pause Phase, Thermopile readings tend to remain stable with minimal fluctuations")

seq_list = np.random.choice(train_df[train_df['behavior_flow'] == FLOW_1]['sequence_id'].unique(), 3)

titles = [seq for seq in seq_list]

fig = make_subplots(rows=3, cols=1, subplot_titles=titles)

for idx, seq_id in enumerate(seq_list, start=1):
    
    tmp = train_df[train_df['sequence_id'] == seq_id].reset_index()
    
    for thm, col in zip(thm_list, thm_color):
        fig.add_trace(
            go.Scatter(
                x = tmp.index,
                y = tmp[thm],
                mode = 'lines',
                name = thm,
                marker_color = col,
                showlegend = (idx == 1),
            ), idx, 1
        )
        
    fig.add_vrect(
        x0=0, x1=tmp.iloc[0]['Relaxes and moves hand to target location']-1,
        fillcolor="red",
        opacity=0.1,
        line_width=0,
        row=idx, col=1,
        annotation_text="Transition",
        annotation_position="top left"
    )
    fig.add_vrect(
        x0=tmp.iloc[0]['Relaxes and moves hand to target location']-1, x1=tmp.iloc[0]['Hand at target location'] + tmp.iloc[0]['Relaxes and moves hand to target location'] -1,
        fillcolor="green",
        opacity=0.1,
        line_width=0,
        row=idx, col=1,
        annotation_text="Pause",
        annotation_position="top right"
    )
    fig.add_vrect(
        x0=tmp.iloc[0]['Hand at target location'] + tmp.iloc[0]['Relaxes and moves hand to target location'] -1, x1=len(tmp)-1,
        fillcolor="blue",
        opacity=0.1,
        line_width=0,
        row=idx, col=1,
        annotation_text="Gesture",
        annotation_position="top right"
    )
fig.update_layout(
    title=FLOW_1,
)

fig.show(renderer='iframe')


tmp = train_df[~train_df['sequence_id'].isin(thm_na_list)].reset_index(drop=True)
thm_3_0 = tmp[tmp['thm_3'] == 0]['sequence_id'].unique()
tmp = tmp[~tmp['sequence_id'].isin(thm_3_0)].reset_index(drop=True)

tmp = tmp[tmp['behavior'] == 'Hand at target location'].groupby('sequence_id').agg(
    pause_diff_1 = ('thm_1', lambda x: np.mean(np.abs(np.diff(x)))),
    pause_diff_2 = ('thm_2', lambda x: np.mean(np.abs(np.diff(x)))),
    pause_diff_3 = ('thm_3', lambda x: np.mean(np.abs(np.diff(x)))),
    pause_diff_4 = ('thm_4', lambda x: np.mean(np.abs(np.diff(x)))),
    pause_diff_5 = ('thm_5', lambda x: np.mean(np.abs(np.diff(x)))),
    gesture = ('gesture', lambda x: x.iloc[0]),
).reset_index()



fig = go.Figure()

for i, col in enumerate(thm_color, start=1):
    fig.add_trace(
        go.Box(
            y = tmp[f'pause_diff_{i}'],
            name = f'pause_diff_{i}',
            marker_color = col,
            boxmean='sd',)
    )

fig.add_hrect(
    y0=3,
    y1=4,
    fillcolor="red",
    opacity=0.2,
    layer="below",
    line_width=0
)

fig.update_layout(
    title="Ditribution of Thermopile Diff in Pause Phase"
)

fig.show(renderer="iframe")


seq_list = ['SEQ_012617']

titles = [seq for seq in seq_list]

fig = go.Figure()
    
tmp = train_df[train_df['sequence_id'] == seq_list[0]].reset_index()
    
for thm, col in zip(thm_list, thm_color):
        fig.add_trace(
            go.Scatter(
                x = tmp.index,
                y = tmp[thm],
                mode = 'lines',
                name = thm,
                marker_color = col,
            ), 
        )
        
fig.add_vrect(
        x0=0, x1=tmp.iloc[0]['Relaxes and moves hand to target location']-1,
        fillcolor="red",
        opacity=0.1,
        line_width=0,
        row=idx, col=1,
        annotation_text="Transition",
        annotation_position="top left"
    )
fig.add_vrect(
        x0=tmp.iloc[0]['Relaxes and moves hand to target location']-1, x1=tmp.iloc[0]['Hand at target location'] + tmp.iloc[0]['Relaxes and moves hand to target location'] -1,
        fillcolor="green",
        opacity=0.1,
        line_width=0,
        row=idx, col=1,
        annotation_text="Pause",
        annotation_position="top right"
    )
fig.add_vrect(
        x0=tmp.iloc[0]['Hand at target location'] + tmp.iloc[0]['Relaxes and moves hand to target location'] -1, x1=len(tmp)-1,
        fillcolor="blue",
        opacity=0.1,
        line_width=0,
        row=idx, col=1,
        annotation_text="Gesture",
        annotation_position="top right"
    )
fig.update_layout(
    title=FLOW_1,
)

fig.show(renderer='iframe')


gc.collect()


tof_list = [col for col in train_df.columns if 'tof' in col ]

tof_1_list = tof_list[:64]
tof_2_list = tof_list[64:128]
tof_3_list = tof_list[128:192]
tof_4_list = tof_list[192:256]
tof_5_list = tof_list[256:320]


def nan_ratio_mean(df, cols):
    return df[cols].isna().mean(axis=1).groupby(df['sequence_id']).mean() * 100

tof_1_na = nan_ratio_mean(train_df, tof_1_list).rename('tof_1_na_ratio')
tof_2_na = nan_ratio_mean(train_df, tof_2_list).rename('tof_2_na_ratio')
tof_3_na = nan_ratio_mean(train_df, tof_3_list).rename('tof_3_na_ratio')
tof_4_na = nan_ratio_mean(train_df, tof_4_list).rename('tof_4_na_ratio')
tof_5_na = nan_ratio_mean(train_df, tof_5_list).rename('tof_5_na_ratio')

tmp = pd.concat([tof_1_na, tof_2_na, tof_3_na, tof_4_na, tof_5_na], axis=1).reset_index()


tof_1_na_list = tmp[tmp['tof_1_na_ratio'] > 50]['sequence_id'].values
tof_2_na_list = tmp[tmp['tof_2_na_ratio'] > 50]['sequence_id'].values
tof_3_na_list = tmp[tmp['tof_3_na_ratio'] > 50]['sequence_id'].values
tof_4_na_list = tmp[tmp['tof_4_na_ratio'] > 50]['sequence_id'].values
tof_5_na_list = tmp[tmp['tof_5_na_ratio'] > 50]['sequence_id'].values

all_ids = np.concatenate([tof_1_na_list, tof_2_na_list, tof_3_na_list, tof_4_na_list, tof_5_na_list])
tof_na_list = np.unique(all_ids)
len(tof_na_list)


from cuml import UMAP
umap = UMAP()

cols = tof_list
stats = ['mean', 'min', 'max', 'std'] 

agg_dict = {col: stats for col in cols}
tmp = cudf.from_pandas(train_df.dropna().reset_index(drop=True))

tmp_stats = tmp.groupby('sequence_id').agg(agg_dict)

tmp_stats.columns = [f"{col}_{stat}" for col, stat in tmp_stats.columns]

tmp_stats = tmp_stats.reset_index()

tmp_meta = tmp[['sequence_id', 'gesture', 'sequence_type']]\
    .drop_duplicates(subset=['sequence_id'])\
    .dropna()

tmp = tmp_stats.merge(tmp_meta, on='sequence_id', how='left')

tmp = tmp.to_pandas()
tmp.loc[tmp['sequence_type'] != 'Target', 'gesture'] = 'Non-Target'
tmp_cudf = cudf.from_pandas(tmp)


%%time

features = tmp_cudf.drop(columns=['sequence_id','gesture','sequence_type'])

embed_2d = umap.fit_transform(features)
embed_2d = embed_2d.to_numpy()
embed_df = pd.DataFrame(embed_2d, columns=['x','y'])
embed_df['gesture'] = tmp['gesture'].values

fig = go.Figure()

for ges in tmp['gesture'].unique():
    tmp2 = embed_df[embed_df['gesture'] == ges]
    
    fig.add_trace(
        go.Scatter(
            x = tmp2['x'],
            y = tmp2['y'],
            mode = 'markers',
            marker=dict(size=6, opacity=0.7),
            name = ges,
        )
    )

fig.update_layout(
    title='Correlation with Gesture & Time of Flight',
    legend=dict(title='Gesture Type', bordercolor='black', borderwidth=1)
    
)

fig.show(renderer='iframe')


print(f"\n{Fore.BLUE}{st_} TOF devices that are physically closer to each other tend to have higher correlations. For example, TOF3 and TOF5, as well as TOF2 and TOF4, which are farther apart, show relatively lower correlations.")

train_df['tof_1_mean'] = np.mean(train_df[tof_1_list], axis=1)
train_df['tof_2_mean'] = np.mean(train_df[tof_2_list], axis=1)
train_df['tof_3_mean'] = np.mean(train_df[tof_3_list], axis=1)
train_df['tof_4_mean'] = np.mean(train_df[tof_4_list], axis=1)
train_df['tof_5_mean'] = np.mean(train_df[tof_5_list], axis=1)

features =  ['tof_1_mean','tof_2_mean','tof_3_mean','tof_4_mean','tof_5_mean']

corr_matrix = train_df[features].corr()

fig = px.imshow(corr_matrix,
                text_auto=True,
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                aspect='auto',
                title='Correlation Matrix')

fig.show(renderer="iframe")


def create_each_tof(sequence_id, tof_id):
    ims = []

    tmp = train_df[train_df['sequence_id'] == sequence_id].reset_index()

    for i, row in tqdm(tmp.iterrows(), desc='Create TOF Image'):

        img = np.zeros((128,128), dtype='float32')
        
        for i in range(8):
            for j in range(8):
                img[i*16:(i+1)*16,j*16:(j+1)*16] = row[f'tof_{tof_id}_v{8*i+j}']

        ims.append(img)

    fig = plt.figure(figsize=(6,6))
    plt.title(f'TOF Device: {tof_id}\n Sequence_id: {sequence_id}')
    plt.axis('off')
    im = plt.imshow(ims[0].astype(np.uint8), cmap='coolwarm', vmin=np.min(img), vmax=np.max(img))

    def animate_func(i):
        im.set_array(ims[i].astype(np.uint8))
        return [im]
        
    plt.close(fig)

    return animation.FuncAnimation(fig, animate_func, frames=len(ims), interval = 1000 // 24)

seq_id = np.random.choice(train_df['sequence_id'].unique())
anim = create_each_tof(seq_id, tof_id=1)
anim




def get_colormap_color(value, vmin, vmax, cmap_name='coolwarm'):
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(norm(value))
    rgb = tuple(int(255 * x) for x in rgba[:3])
    return rgb  # (R, G, B)

def create_tof(sequence_id):
    ims = []

    tmp = train_df[train_df['sequence_id'] == sequence_id].reset_index()

    all_thm_values = tmp[[f'thm_{i}' for i in range(1, 6)]].values.flatten()
    vmin = np.nanmin(all_thm_values)
    vmax = np.nanmax(all_thm_values)


    for i, row in tqdm(tmp.iterrows(), desc='Create Thermopile Image'):

        img = np.zeros((256,256,3), dtype='float32')
        
        color1 = get_colormap_color(row['tof_1_mean'], vmin, vmax)
        color2 = get_colormap_color(row['tof_2_mean'], vmin, vmax)
        color3 = get_colormap_color(row['tof_3_mean'], vmin, vmax)
        color4 = get_colormap_color(row['tof_4_mean'], vmin, vmax)
        color5 = get_colormap_color(row['tof_5_mean'], vmin, vmax)

        # Thm_1
        cv2.rectangle(img, (130, 70), (150, 90), color=color1, thickness=-1)
        cv2.rectangle(img, (130, 70), (150, 90), color=(255, 0, 0), thickness=1)
        img = draw_bbox(img, (130, 70, 150, 90), label=f'1: {row["tof_1_mean"]:.1f}')
        # Thm_2
        cv2.rectangle(img, (130, 30), (150, 50), color=color2, thickness=-1)
        cv2.rectangle(img, (130, 30), (150, 50), color=(255, 0, 0), thickness=1)
        img = draw_bbox(img, (130, 30, 150, 50), label=f'2: {row["tof_2_mean"]:.1f}')
        # Thm_3
        cv2.rectangle(img, (190, 120), (210, 140), color=color3, thickness=-1)
        cv2.rectangle(img, (190, 120), (210, 140), color=(255, 0, 0), thickness=1)
        img = draw_bbox(img, (190, 120, 210, 140), label=f'3: {row["tof_3_mean"]:.1f}')
        # Thm_4
        cv2.rectangle(img, (130, 210), (150, 230), color=color4, thickness=-1)
        cv2.rectangle(img, (130, 210), (150, 230), color=(255, 0, 0), thickness=1)
        img = draw_bbox(img, (130, 210, 150, 230), label=f'4: {row["tof_4_mean"]:.1f}')
        # Thm_5
        cv2.rectangle(img, (30, 120), (50, 140), color=color5, thickness=-1)
        cv2.rectangle(img, (30, 120), (50, 140), color=(255, 0, 0), thickness=1)
        img = draw_bbox(img, (30, 120, 50, 140), label=f'5: {row["tof_5_mean"]:.1f}')

        ims.append(img)

    fig = plt.figure(figsize=(6,6))
    plt.axis('off')
    im = plt.imshow(ims[0].astype(np.uint8))

    def animate_func(i):
        im.set_array(ims[i].astype(np.uint8))
        return [im]
        
    plt.close(fig)

    return animation.FuncAnimation(fig, animate_func, frames=len(ims), interval = 1000 // 24)

seq_id = np.random.choice(train_df['sequence_id'].unique())
anim = create_tof(seq_id)
anim



seq_list = np.random.choice(train_df[train_df['behavior_flow'] == FLOW_1]['sequence_id'].unique(), 3)

tof_mean_list = ['tof_1_mean','tof_2_mean','tof_3_mean','tof_4_mean','tof_5_mean']

titles = [seq for seq in seq_list]

fig = make_subplots(rows=3, cols=1, subplot_titles=titles)

for idx, seq_id in enumerate(seq_list, start=1):
    
    tmp = train_df[train_df['sequence_id'] == seq_id].reset_index()
    
    for tof, col in zip(tof_mean_list, thm_color):
        fig.add_trace(
            go.Scatter(
                x = tmp.index,
                y = tmp[tof],
                mode = 'lines',
                name = tof,
                marker_color = col,
                showlegend = (idx == 1),
            ), idx, 1
        )
        
    fig.add_vrect(
        x0=0, x1=tmp.iloc[0]['Relaxes and moves hand to target location']-1,
        fillcolor="red",
        opacity=0.1,
        line_width=0,
        row=idx, col=1,
        annotation_text="Transition",
        annotation_position="top left"
    )
    fig.add_vrect(
        x0=tmp.iloc[0]['Relaxes and moves hand to target location']-1, x1=tmp.iloc[0]['Hand at target location'] + tmp.iloc[0]['Relaxes and moves hand to target location'] -1,
        fillcolor="green",
        opacity=0.1,
        line_width=0,
        row=idx, col=1,
        annotation_text="Pause",
        annotation_position="top right"
    )
    fig.add_vrect(
        x0=tmp.iloc[0]['Hand at target location'] + tmp.iloc[0]['Relaxes and moves hand to target location'] -1, x1=len(tmp)-1,
        fillcolor="blue",
        opacity=0.1,
        line_width=0,
        row=idx, col=1,
        annotation_text="Gesture",
        annotation_position="top right"
    )
fig.update_layout(
    title=FLOW_1,
)

fig.show(renderer='iframe')


tmp = train_df[~train_df['sequence_id'].isin(tof_na_list)].reset_index(drop=True)

tmp = tmp[tmp['behavior'] == 'Hand at target location'].groupby('sequence_id').agg(
    pause_diff_1 = ('tof_1_mean', lambda x: np.mean(np.abs(np.diff(x)))),
    pause_diff_2 = ('tof_2_mean', lambda x: np.mean(np.abs(np.diff(x)))),
    pause_diff_3 = ('tof_3_mean', lambda x: np.mean(np.abs(np.diff(x)))),
    pause_diff_4 = ('tof_4_mean', lambda x: np.mean(np.abs(np.diff(x)))),
    pause_diff_5 = ('tof_5_mean', lambda x: np.mean(np.abs(np.diff(x)))),
    gesture = ('gesture', lambda x: x.iloc[0]),
).reset_index()


fig = go.Figure()

for i, col in enumerate(thm_color, start=1):
    fig.add_trace(
        go.Box(
            y = tmp[f'pause_diff_{i}'],
            name = f'pause_diff_{i}',
            marker_color = col,
            boxmean='sd',)
    )

fig.add_hrect(
    y0=80,
    y1=100,
    fillcolor="red",
    opacity=0.2,
    layer="below",
    line_width=0
)

fig.update_layout(
    title="Ditribution of TOF Diff in Pause Phase"
)

fig.show(renderer="iframe")


seq_list = ['SEQ_047121','SEQ_030299']

tof_mean_list = ['tof_1_mean','tof_2_mean','tof_3_mean','tof_4_mean','tof_5_mean']

titles = [seq for seq in seq_list]

fig = go.Figure()

tmp = train_df[train_df['sequence_id'] == seq_list[0]].reset_index()
    
for tof, col in zip(tof_mean_list, thm_color):
        fig.add_trace(
            go.Scatter(
                x = tmp.index,
                y = tmp[tof],
                mode = 'lines',
                name = tof,
                marker_color = col,
            ),
        )
        
fig.add_vrect(
        x0=0, x1=tmp.iloc[0]['Relaxes and moves hand to target location']-1,
        fillcolor="red",
        opacity=0.1,
        line_width=0,
        row=idx, col=1,
        annotation_text="Transition",
        annotation_position="top left"
    )
fig.add_vrect(
        x0=tmp.iloc[0]['Relaxes and moves hand to target location']-1, x1=tmp.iloc[0]['Hand at target location'] + tmp.iloc[0]['Relaxes and moves hand to target location'] -1,
        fillcolor="green",
        opacity=0.1,
        line_width=0,
        row=idx, col=1,
        annotation_text="Pause",
        annotation_position="top right"
    )
fig.add_vrect(
        x0=tmp.iloc[0]['Hand at target location'] + tmp.iloc[0]['Relaxes and moves hand to target location'] -1, x1=len(tmp)-1,
        fillcolor="blue",
        opacity=0.1,
        line_width=0,
        row=idx, col=1,
        annotation_text="Gesture",
        annotation_position="top right"
    )
fig.update_layout(
    title=FLOW_1,
)

fig.show(renderer='iframe')


import librosa
from scipy import interpolate

def inter_signal(x):
    original_len = len(x)
    interp_func = interpolate.interp1d(np.linspace(0, 1, original_len), x, kind='linear')
    x_interp = interp_func(np.linspace(0, 1, 1024))
    return x_interp

def spectrogram_from_tof(df, seq_id, sr=64):
    tmp = df[df['sequence_id'] == seq_id].reset_index()

    img = np.zeros((256,50,5), dtype='float32')

    for i, tof in enumerate(tof_mean_list):
        x = tmp[tof].values
        m = np.nanmean(x)
        x = np.nan_to_num(x, nan=m)
        x = inter_signal(x)

        mel_spec = librosa.feature.melspectrogram(y=x, sr=sr, hop_length=len(x)//50, n_fft=100,
                                                  n_mels=256, fmin=0, fmax=sr)
        width = (mel_spec.shape[1]//10)*10
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:,:width]

        mel_spec_db = (mel_spec_db+40)/40
        img[:,:,i] = mel_spec_db

    return img


seq_list = np.random.choice(train_df['sequence_id'].unique(), 2)

plt.figure(figsize=(14, 6))

for i, seq in enumerate(seq_list):
    img = spectrogram_from_tof(train_df, seq)

    for j, tof in enumerate(tof_mean_list):
        plt.subplot(2, 5, 5*i + j + 1)
        plt.imshow(img[...,j], aspect='auto', origin='lower', cmap='jet')
        plt.title(f'{seq}_{tof}')
        plt.axis('off')

plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 6))

for i, seq in enumerate(seq_list):
    img = np.zeros((256,256), dtype=np.float32)
    
    img_tof = spectrogram_from_tof(train_df, seq)
    img_tof = np.concatenate([img_tof[...,i] for i in range(5)], axis=1) 

    img[:, 3:253] = img_tof
    
    plt.subplot(1,2,i+1)
    plt.imshow(img, aspect='auto', origin='lower', cmap='jet')
    plt.axis('off')

plt.tight_layout()
plt.show()


# Behavior

## ✅ Every Sequence_id has 3 unique behavior including perform gesture
## ✅ But Sequence_id(SEQ_011975) has only 2 unique behavior and not incldue perform gesture

# Phase Ratio

## ✅ gesture_ids = tmp[tmp['Gesture_Ratio'] > 85]['sequence_id'].values
## ✅ transition_ids = tmp[tmp['Transition_Ratio'] > 95]['sequence_id'].values


# IMU: Accelerator + Rotation

## ✅ Accelerator: No Null
## ✅ Rotation: null(0%): 8089, null(100%): 50
## rot_nan_list(total: 50) 

# Thermopile Sensor

## ✅ thm_1: null(0%): 8034, null(100%): 104
## ✅ thm_2: null(0%): 8021, null(71.4%): 1, null(100%): 116
## ✅ thm_3: null(0%): 8038, null(100%): 100
## ✅ thm_4: null(0%): 8042, null(100%): 96
## ✅ thm_5: null(0%): 7655, null(77.6%): 271, null(100%): 482

## ✅ thm_na_list(total: 508)

## ✅ thm_outlier(in pause): ['SEQ_012617']


# Time-of-Flight Sensor

## ✅ tof_1: null(0%): 8042, null(100%): 96
## ✅ tof_2: null(0%): 8042, null(100%): 96
## ✅ tof_3: null(0%): 8042, null(100%): 96
## ✅ tof_4: null(0%): 8042, null(100%): 96
## ✅ tof_5: null(0%): 7703, null(77.6%): 1, null(100%): 434

## ✅ tof_na_list(total: 435)

## ✅ tof_outlier(in pause): ['SEQ_047121','SEQ_030299']


delete_id = []

# Behavior
delete_id.append('SEQ_011975')

# Phase Ratio
delete_id.extend(gesture_ids.tolist())
delete_id.extend(transition_ids.tolist())

# IMU Null
delete_id.extend(rot_nan_list)

# THM Null
delete_id.extend(thm_na_list)
delete_id.append('SEQ_012617')

# TOF Null
delete_id.extend(tof_na_list)
delete_id.extend(['SEQ_047121','SEQ_030299'])

delete_id = list(set(delete_id))


train_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')

print(f"\n{Fore.BLUE}{st_}Deleting Sequence id {len(delete_id)}")
print(f"\n{Fore.BLUE}{st_}total {train_df['sequence_id'].nunique()} seqeunces Before Cleaning")

train_df = train_df[~train_df['sequence_id'].isin(delete_id)].reset_index(drop=True)

print(f"\n{Fore.BLUE}{st_}total {train_df['sequence_id'].nunique()} seqeunces After Cleaning")


train_df.to_csv('/kaggle/working/train_cleaned.csv', index=False)

