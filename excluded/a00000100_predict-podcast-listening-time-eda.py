import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

!pip install --upgrade seaborn==0.13.2 --quiet
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = 'iframe'


# Input Directory
from pathlib import Path

input_path = "../input/"
print(f"\033[1;38;5;196m{input_path}\033[0m")

for path in sorted(Path(input_path).glob("*/")):
    if path.is_dir():
        print("  " + '\033[1;34m' + path.name + '\033[0m' + "/")
    else:
        print(path.name)
    if path.is_dir():
        for subpath in sorted(path.glob("*")):
            print("    " + subpath.name + ("/" if subpath.is_dir() else ""))


def load_data(**kwargs):
    return pd.read_csv('../input/playground-series-s5e4/train.csv', **kwargs), pd.read_csv('../input/playground-series-s5e4/test.csv', **kwargs)

train, test = load_data()


# Description
with open('../input/podcast-listening-time-prediction-dataset/podcast_dataset_info.txt','r') as f:
    string = f.read()

print(string)



import warnings
warnings.filterwarnings("ignore", message=".*invalid value encountered in.*", category=RuntimeWarning)


train.head()


train.info()


# "id" is the same as index
assert (train["id"]==train.index).all()


train, test = load_data(index_col = 'id')



from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

fig = make_subplots(rows=2, cols=4, shared_xaxes=True, row_heights=[0.775, 0.225], vertical_spacing=0.075, horizontal_spacing=0.05,
                    specs=[[{"secondary_y": True}] * 4, [{}] * 4],
                    column_titles=['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Listening_Time_minutes'])

fig.update_xaxes(linewidth=1, linecolor='grey', mirror=True, ticks='outside', showline=True)
fig.update_yaxes(linewidth=1, linecolor='grey', mirror=True, ticks='outside', showline=True)

for i, col in enumerate(['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Listening_Time_minutes']):
    q1, q2 = train[col].quantile([0.0005, 0.9995]) # 0.05%ile and 99.95%ile
    sampled = pd.concat([
        train.loc[train[col] < q1, col],
        train.loc[train[col] > q2, col],
        train.loc[train[col].between(q1, q2), col].sample(100_000, random_state=42)
    ])
    
    fig.add_trace(go.Histogram(x=sampled, nbinsx=50, marker_color='#636EFA', name='', marker=dict(line=dict(color='black', width=0.5))),
                  row=1, col=i+1)
    fig.add_trace(go.Box(x=sampled, marker_color='black', line_width=1, name='', fillcolor='#636EFA', hoverlabel=dict(bgcolor="royalblue")),
                  row=2, col=i+1)
    
    kde = gaussian_kde(sampled.dropna())
    x_vals = np.linspace(sampled.min(), sampled.max(), 1000)
    y_vals = kde(x_vals)

    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, line_color='blue', name=''), secondary_y=True, row=1, col=i+1)
    if col!='Listening_Time_minutes':
        n=4
        all_peaks, _ = find_peaks(y_vals)
        prominences = y_vals[all_peaks]
        top_n_peaks = all_peaks[np.argsort(prominences)[-n:]]

        valley_idxs, _ = find_peaks(-y_vals)
        valley_depths = y_vals[valley_idxs]
        bottom_n_valley_idxs = valley_idxs[np.argsort(valley_depths)[:n]]
        
        fig.add_trace(go.Scatter(x=x_vals[top_n_peaks], y=y_vals[top_n_peaks], mode='markers', line_color='red', name=''), secondary_y=True, row=1, col=i+1)
        fig.add_trace(go.Scatter(x=x_vals[bottom_n_valley_idxs], y=y_vals[bottom_n_valley_idxs], mode='markers', line_color='red', name=''), secondary_y=True, row=1, col=i+1)
    fig.update_yaxes(range=(0, y_vals.max()*1.06), showticklabels=False, ticks='', secondary_y=True, row=1, col=i+1)
    fig.update_yaxes(showticklabels=False, row=2, col=i+1)
    fig.update_xaxes(ticks='', row=1, col=i+1)
    fig.update_yaxes(range=[-0.3, 0.3], ticks='', row=2, col=i+1)

fig.update_layout(title = dict(text='Continuous numerical attributes', x = 0.5, y = 0.98, font_color = 'black'), plot_bgcolor="white", height=300,
                  showlegend=False, margin=dict(l=15, r=0, t=50, b=15))
for ann in fig.layout.annotations:
    ann.font.size = 13
fig.show()

train[['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Listening_Time_minutes']].describe().T


train['Episode_Title'] = train['Episode_Title'].str.extract(r'(\d+)')[0].astype(int)



fig = make_subplots(rows = 1, cols = 2, subplot_titles=["Number_of_Ads", "Episode_Title"], column_widths=[0.65, 0.35], horizontal_spacing=0.04)

cat, counts = np.unique(train["Number_of_Ads"].dropna(), return_counts=True)
perc = (counts / len(train["Number_of_Ads"].dropna()) * 100).round(1)
fig.add_trace(go.Bar(x=cat.astype(str), y=counts, width = 0.5, text = np.char.add(perc.astype(str), "%"), textfont_size = 11,
                     hovertext = counts, hoverinfo = "text",
                     textposition="outside", marker_color='#636EFA'), row=1, col=1)
fig.update_yaxes(range = [0, counts.max()*1.09], row = 1, col = 1)

vc = train.Episode_Title.value_counts().sort_index()
fig.add_trace(go.Bar(x=vc.index, y=vc.values, width = 1, marker_color='#636EFA', name=''), row=1, col=2)

fig.update_xaxes(linewidth=1, linecolor='grey', mirror=True, ticks='outside', showline=True)
fig.update_yaxes(linewidth=1, linecolor='grey', mirror=True, ticks='outside', showline=True)
fig.update_layout(height=300, showlegend=False, plot_bgcolor="white", margin=dict(l=35, r=10, t=50, b=5),
                  title_text="Discrete numerical attributes", title = dict(x = 0.5, y = 0.98, font_color = 'black'))
fig.show()



sort_order = {'Publication_Day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
              'Publication_Time': ['Morning', 'Afternoon', 'Evening', 'Night'],
              'Episode_Sentiment': ['Negative', 'Neutral', 'Positive']}

fig = make_subplots(rows = 2, cols = 4, subplot_titles=["Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment", "Podcast_Name"],
                    specs=[[{}, {}, {}, {}], [{"colspan": 3}, None, None, None]], horizontal_spacing=0.042, vertical_spacing=0.22)

for i, col in enumerate(["Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment", "Podcast_Name"], start=1):
    cat, counts = np.unique(train[col].dropna(), return_counts=True)
    if col not in sort_order.keys():
        if col == 'Podcast_Name':
            grouped_names = train.groupby(col, observed=False)['Genre'].value_counts().groupby(level=0, observed=False).head(1).sort_index(level='Genre').index
            sort_order[col] = list(grouped_names.get_level_values(col))
        else:
            sort_idx = np.argsort(-counts)
            sort_order[col] = list(cat[sort_idx])

    order = sort_order[col]
    train[col] = pd.Categorical(train[col], categories=order, ordered=True)
    
    sort_idx = np.argsort([order.index(t) if t in order else 999 for t in cat])
    cat, counts = cat[sort_idx], counts[sort_idx]
    
    if col=='Podcast_Name':
        color = px.colors.qualitative.Plotly[:len(sort_order['Genre'])]
        color = grouped_names.get_level_values('Genre').map({genre: color[i % len(color)] for i, genre in enumerate(sort_order['Genre'])})
        color_podcast_name_dict = dict(zip(sort_order['Podcast_Name'], list(color)))
    elif col=='Genre':
        color = px.colors.qualitative.Plotly[:len(sort_order[col])]
        color_genre_dict = dict(zip(sort_order['Genre'], list(color)))
    else:
        color = '#636EFA'
    perc = (counts / len(train[col].dropna()) * 100).round(1)

    fig.add_trace(go.Bar(x=cat.astype(str), y=counts, width = 0.5, text = perc.astype(str) if col!='Podcast_Name' else None,
                         hovertext = counts, hoverinfo = "text", textposition="outside", textfont=dict(size=11),
                         marker_color=color),
                         row=2 if col=='Podcast_Name' else 1, col=1 if col=='Podcast_Name' else i)
    if col!='Podcast_Name':
        fig.update_yaxes(range = [0, counts.max()*1.1], row = 1, col = i)

fig.update_layout(height=650, showlegend=False, plot_bgcolor="white",
                  uniformtext_mode='show', margin=dict(l=35, r=10, t=50, b=5),
                  title_text="Categorical attributes", title = dict(x = 0.5, y = 0.98, font_color = 'black'))
fig.update_xaxes(linewidth=1, linecolor='grey', mirror=True, ticks='outside', showline=True)
fig.update_yaxes(linewidth=1, linecolor='grey', mirror=True, ticks='outside', showline=True)
fig.show()

train[col] = pd.Categorical(train[col], categories=sort_order[col], ordered=True)


train.loc[:, train.isna().any()].isna().agg(['sum', 'mean'])



import missingno

ax = plt.subplots(figsize = (3, 7), dpi = 52)[1]
missingno.matrix(train.loc[:, train.isna().any()], label_rotation = 90, fontsize = 15, ax = ax, sparkline=False)
plt.show()



df = train.groupby(train['Episode_Length_minutes'].isnull()).apply(lambda x: x)

axes = plt.subplots(1, 3, figsize = (10, 2.25))[1]
for i, col in enumerate(['Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Listening_Time_minutes']):
    sns.kdeplot(df.loc[False, col], color="blue", fill=True, ax=axes[i], label='not NaN')
    sns.kdeplot(df.loc[True, col], color="red", fill=True, ax=axes[i], label='NaN')
    axes[i].set_xlabel(None)
    axes[i].set_ylabel(None)
    axes[i].set_title(col, fontsize = 10)
plt.tight_layout()
plt.legend()
plt.show()

df[['Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Listening_Time_minutes']].groupby(level=0).describe()



fig = make_subplots(rows = 1, cols = 2, subplot_titles=["Number_of_Ads", "Episode_Title"], column_widths=[0.30, 0.70], horizontal_spacing=0.04)

for isna, color in zip([False, True], ['#636EFA', 'red']):
    cat, counts = np.unique(df.loc[isna, "Number_of_Ads"].dropna(), return_counts=True)
    perc = (counts / len(df.loc[isna, "Number_of_Ads"].dropna()) * 100).round(1)
    fig.add_trace(go.Bar(x=cat.astype(str), y=perc, width = 0.4, text = np.char.add(perc.astype(str), "%"), textfont_size = 11,
                         hovertext = counts, hoverinfo = "text", name=('' if isna == 1 else "not ") + "NaN",
                         textposition="outside", marker_color=color, legendgroup=isna), row=1, col=1)
    fig.update_yaxes(range = [0, perc.max()*1.09], row = 1, col = 1)
    fig.update_xaxes(range = [-0.5, 3.5], row = 1, col = 1)

for isna, color in zip([False, True], ['#636EFA', 'red']):
    vc = df.loc[isna, 'Episode_Title'].value_counts().sort_index()
    fig.add_trace(go.Bar(x=vc.index, y=vc.values/vc.sum()*100, hovertext = vc.values, width = 0.4, marker_color=color,
                         name=('' if isna == 1 else "not ") + "NaN", showlegend=False, legendgroup=isna), row=1, col=2)

fig.update_xaxes(linewidth=1, linecolor='grey', mirror=True, ticks='outside', showline=True)
fig.update_yaxes(linewidth=1, linecolor='grey', mirror=True, ticks='outside', showline=True)
fig.update_layout(height=300, showlegend=True, plot_bgcolor="white", margin=dict(l=35, r=10, t=50, b=5))
fig.show()



fig = make_subplots(rows = 2, cols = 4, subplot_titles=["Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment", "Podcast_Name"],
                    specs=[[{}, {}, {}, {}], [{"colspan": 3}, None, None, None]], horizontal_spacing=0.042, vertical_spacing=0.22)
for i, col in enumerate(["Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment", "Podcast_Name"], start=1):
    for isna, color in zip([False, True], ['#636EFA', 'red']):
        cat, counts = np.unique(df.loc[isna, col].dropna(), return_counts=True)
        order = sort_order[col]
        sort_idx = np.argsort([order.index(t) if t in order else 999 for t in cat])
        cat, counts = cat[sort_idx], counts[sort_idx]
        perc = (counts / len(df.loc[isna, col].dropna()) * 100).round(1)
        fig.add_trace(go.Bar(x=cat.astype(str), y=perc, width = 0.4, textfont_size = 8,
                             hovertext = counts, hoverinfo = "text", name=('' if isna == 1 else "not ") + "NaN",
                             showlegend=True if i==1 else False, legendgroup=isna, marker_color=color),
                             row=2 if col=='Podcast_Name' else 1, col=1 if col=='Podcast_Name' else i)
fig.update_layout(height=650, showlegend=True, plot_bgcolor="white",
                  margin=dict(l=35, r=10, t=50, b=5))
fig.update_xaxes(linewidth=1, linecolor='grey', mirror=True, ticks='outside', showline=True)
fig.update_yaxes(linewidth=1, linecolor='grey', mirror=True, ticks='outside', showline=True)
fig.show()



sampled = train.sample(100_000, random_state = 42)

axes = plt.subplots(1, 2, figsize=(9.25, 4.25))[1]
axes[0].scatter(data=sampled, x='Host_Popularity_percentage', y="Episode_Length_minutes",
            c='royalblue', s=5, alpha=0.05)
axes[1].scatter(data=sampled, x='Guest_Popularity_percentage', y="Episode_Length_minutes",
            c='royalblue', s=5, alpha=0.05)
axes[0].set_xlabel('Host_Popularity_percentage')
axes[0].set_ylabel('Episode_Length_minutes')
axes[0].axis([20-5, 100+5, 5-5, 120+5])
axes[1].set_xlabel('Guest_Popularity_percentage')
axes[1].set_yticks([])
axes[1].axis([0-5, 100+5, 5-5, 120+5])
plt.subplots_adjust(wspace=0.05)
plt.show()



axes = plt.subplots(1, 2, figsize=(12, 2.75), gridspec_kw={'width_ratios': [1, 2.5]}, constrained_layout=True)[1]

for i, col in enumerate(["Number_of_Ads", "Episode_Title"]):
    sns.boxplot(data=train, x=col, y='Episode_Length_minutes', ax=axes[i], hue=col, palette="Set3", legend=False)
    sns.pointplot(data=train, x=col, y='Episode_Length_minutes', ax=axes[i], alpha = [0.6, 0.3][i], color='royalblue',
                  errorbar='sd', markers='o', estimator='median')
    axes[i].axhline(y=train['Episode_Length_minutes'].median(), color='r', linestyle='--', linewidth=0.5)
    axes[i].set_title(col)
    axes[i].set_xlabel("")
    axes[i].set_ylim(-10, 130)
    axes[i].set_ylabel('Episode_Length_minutes' if i == 0 else "")
axes[0].set_xlim(-0.5, 3.5)
axes[1].set_xticks(np.arange(0, 100, 10)-1)
plt.show()



axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)[1]

for i, col in enumerate(["Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]):
    sns.boxplot(data=train, x=col, y='Episode_Length_minutes', ax=axes[i], hue=col, palette=color_genre_dict if col == "Genre" else "Set3", order=None)
    sns.pointplot(data=train, x=col, y='Episode_Length_minutes', ax=axes[i], alpha = 0.6, color='royalblue',
                  errorbar='sd', markers='o', estimator='median')
    axes[i].axhline(y=train['Episode_Length_minutes'].median(), color='r', linestyle='--', linewidth=0.5)
    axes[i].set_title(col, fontsize=12)
    axes[i].tick_params(axis='x', rotation=-90)
    axes[i].set_xlabel("")
    axes[i].set_ylim(-10, 130)
    axes[i].set_ylabel('Episode_Length_minutes' if i == 0 else "")
plt.show()



plt.figure(figsize = (10, 3))
sns.boxplot(data=train, x="Podcast_Name", y='Episode_Length_minutes', hue="Podcast_Name", palette=color_podcast_name_dict, legend=False)
sns.pointplot(data=train, x='Podcast_Name', y='Episode_Length_minutes', alpha = 0.6, color='royalblue', errorbar='sd', markers='o', estimator='median')
plt.axhline(y=train['Episode_Length_minutes'].median(), color='r', linestyle='--', linewidth=0.5)    
plt.title("Podcast_Name")
plt.xlabel("")
plt.tick_params(axis='x', rotation=-90)
plt.xticks(fontsize=8)
plt.ylim(-10, 130)
plt.ylabel('Episode_Length_minutes')
plt.show()



df = train.groupby(train['Guest_Popularity_percentage'].isnull()).apply(lambda x: x)#.drop((False, 101637))

axes = plt.subplots(1, 3, figsize = (10, 2.25))[1]
for i, col in enumerate(['Episode_Length_minutes', 'Host_Popularity_percentage', 'Listening_Time_minutes']):
    sns.kdeplot(df.loc[False, col], color="blue", fill=True, ax=axes[i], label='not NaN')
    sns.kdeplot(df.loc[True, col], color="red", fill=True, ax=axes[i], label='NaN')
    axes[i].set_xlabel(None)
    axes[i].set_ylabel(None)
    axes[i].set_title(col, fontsize = 10)
plt.tight_layout()
plt.legend()
plt.show()

df[['Episode_Length_minutes', 'Host_Popularity_percentage', 'Listening_Time_minutes']].groupby(level=0).describe()



fig = make_subplots(rows = 1, cols = 2, subplot_titles=["Number_of_Ads", "Episode_Title"], column_widths=[0.30, 0.70], horizontal_spacing=0.04)

for isna, color in zip([False, True], ['#636EFA', 'red']):
    cat, counts = np.unique(df.loc[isna, "Number_of_Ads"].dropna(), return_counts=True)
    perc = (counts / len(df.loc[isna, "Number_of_Ads"].dropna()) * 100).round(1)
    fig.add_trace(go.Bar(x=cat.astype(str), y=perc, width = 0.4, text = np.char.add(perc.astype(str), "%"), textfont_size = 11,
                         hovertext = counts, hoverinfo = "text", name=('' if isna == 1 else "not ") + "NaN",
                         textposition="outside", marker_color=color, legendgroup=isna), row=1, col=1)
    fig.update_yaxes(range = [0, perc.max()*1.25], row = 1, col = 1)
    fig.update_xaxes(range = [-0.5, 3.5], row = 1, col = 1)

for isna, color in zip([False, True], ['#636EFA', 'red']):
    vc = df.loc[isna, 'Episode_Title'].value_counts().sort_index()
    fig.add_trace(go.Bar(x=vc.index, y=vc.values/vc.sum()*100, hovertext = vc.values, width = 0.4, marker_color=color,
                         name=('' if isna == 1 else "not ") + "NaN", showlegend=False, legendgroup=isna), row=1, col=2)

fig.update_xaxes(linewidth=1, linecolor='grey', mirror=True, ticks='outside', showline=True)
fig.update_yaxes(linewidth=1, linecolor='grey', mirror=True, ticks='outside', showline=True)
fig.update_layout(height=300, showlegend=True, plot_bgcolor="white", margin=dict(l=35, r=10, t=50, b=5),
                  title = dict(x = 0.5, y = 0.98, font_color = 'black'))
fig.show()



fig = make_subplots(rows = 1, cols = 4, subplot_titles=["Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"], horizontal_spacing=0.042)
for i, col in enumerate(["Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"], start=1):
    for isna, color in zip([False, True], ['#636EFA', 'red']):
        cat, counts = np.unique(df.loc[isna, col].dropna(), return_counts=True)
        if col == "Genre":
            sort_idx = np.argsort(-counts)
        else:
            order = sort_order[col]
            sort_idx = np.argsort([order.index(t) if t in order else 999 for t in cat])
        cat, counts = cat[sort_idx], counts[sort_idx]
        perc = (counts / len(df.loc[isna, col].dropna()) * 100).round(1)
        fig.add_trace(go.Bar(x=cat.astype(str), y=perc, width = 0.4, textfont_size = 8,
                             hovertext = counts, hoverinfo = "text", name=('' if isna == 1 else "not ") + "NaN",
                             showlegend=True if col=='Genre' else False, legendgroup=isna, marker_color=color), row=1, col=i)
fig.update_layout(height=300, showlegend=True, plot_bgcolor="white",
                  uniformtext_minsize=12, uniformtext_mode='show', margin=dict(l=35, r=10, t=50, b=5),
                  title = dict(x = 0.5, y = 0.98, font_color = 'black'))
fig.update_xaxes(linewidth=1, linecolor='grey', mirror=True, ticks='outside', showline=True)
fig.update_yaxes(linewidth=1, linecolor='grey', mirror=True, ticks='outside', showline=True)
fig.show()



plt.figure(figsize = (4, 2.75))
sns.kdeplot(train.loc[~train['Guest_Popularity_percentage'].isna(), 'Listening_Time_minutes'].sample(100_000, random_state=42), fill=True)
sns.kdeplot(train.loc[train['Guest_Popularity_percentage'].isna(), 'Listening_Time_minutes'], fill=True, color='r', label='NaN')
sns.kdeplot(train.loc[(train['Guest_Popularity_percentage']>70), 'Listening_Time_minutes'], fill=True, label='Guest popularity>70')

plt.legend(loc='lower right')
plt.show()



plt.figure(figsize=(5, 4.75))
plt.scatter(data=train.sample(100_000, random_state = 42), x='Guest_Popularity_percentage', y="Host_Popularity_percentage",
        c='royalblue', s=5, alpha=0.05)
plt.xlabel('Guest_Popularity_percentage')
plt.ylabel('Host_Popularity_percentage')
plt.axis([0-5, 100+5, 20-5, 100+5])
plt.show()



axes = plt.subplots(1, 2, figsize=(12, 3.2), gridspec_kw={'width_ratios': [1, 2.5]}, constrained_layout=True)[1]

for i, col in enumerate(["Number_of_Ads", "Episode_Title"]):
    sns.boxplot(data=train, x=col, y='Guest_Popularity_percentage', ax=axes[i], hue=col, palette="Set3", legend=False)
    sns.pointplot(data=train, x=col, y='Guest_Popularity_percentage', ax=axes[i], alpha = [0.6, 0.3][i], color='royalblue',
                  errorbar='sd', markers='o', estimator='median')
    axes[i].axhline(y=train['Guest_Popularity_percentage'].median(), color='r', linestyle='--', linewidth=0.5)
    axes[i].set_title(col)
    axes[i].set_xlabel("")
    axes[i].set_ylabel('Guest_Popularity_percentage' if i == 0 else "")
axes[0].set_xlim(-0.5, 3.5)
axes[1].set_xticks(np.arange(0, 100, 10)-1)
plt.show()



axes = plt.subplots(1, 4, figsize=(16, 4.5), constrained_layout=True)[1]

for i, col in enumerate(["Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]):
    sns.boxplot(data=train, x=col, y='Guest_Popularity_percentage', order=sort_order[col], ax=axes[i], hue=col,
                palette=color_genre_dict if col == "Genre" else "Set3")
    sns.pointplot(data=train, x=col, y='Guest_Popularity_percentage', ax=axes[i], alpha = 0.6, color='royalblue',
                  errorbar='sd', markers='o', estimator='median')
    axes[i].axhline(y=train['Guest_Popularity_percentage'].median(), color='r', linestyle='--', linewidth=0.5)
    axes[i].set_title(col, fontsize=12)
    axes[i].tick_params(axis='x', rotation=-90)
    axes[i].set_xlabel("")
    axes[i].set_ylabel('Guest_Popularity_percentage' if i == 0 else "")
plt.show()



plt.figure(figsize = (10, 3.5))
sns.boxplot(data=train, x="Podcast_Name", y='Guest_Popularity_percentage', hue="Podcast_Name", palette=color_podcast_name_dict, legend=False)
sns.pointplot(data=train, x="Podcast_Name", y='Guest_Popularity_percentage', alpha = 0.6, color='royalblue', errorbar='sd',
              markers='o', estimator='median')
plt.axhline(y=train['Guest_Popularity_percentage'].median(), color='r', linestyle='--', linewidth=0.5)
plt.title("Podcast_Name")
plt.xlabel("")
plt.tick_params(axis='x', rotation=-90)
plt.xticks(fontsize=8)
plt.ylabel('Guest_Popularity_percentage')
plt.show()



q1, q2 = train['Episode_Length_minutes'].quantile([0.001, 0.999]) # 0.1%ile and 99.9%ile
sampled = pd.concat([
    train.loc[train['Episode_Length_minutes'] < q1, 'Episode_Length_minutes'],
    train.loc[train['Episode_Length_minutes'] > q2, 'Episode_Length_minutes'],
    train.loc[train['Episode_Length_minutes'].between(q1, q2), 'Episode_Length_minutes'].sample(2500, random_state=42)
])

fig = go.Figure(go.Box(
    x=sampled, boxpoints='all', jitter=0.3, pointpos=-1.8, name='',
    hovertext = "Length: " + sampled.astype(str) + " mins<br>Index: " + sampled.index.astype(str),
    hoverlabel=dict(bgcolor="royalblue"),
    marker=dict(color='rgba(65, 105, 255, 0.6)', size=4),
    line=dict(color='royalblue',width=2), fillcolor='rgba(0, 0, 255, 0.2)'
))
fig.update_xaxes(title='Episode_Length_minutes', title_font_size=15)
fig.update_yaxes(range=[-0.58, 0.35])
fig.update_layout(margin=dict(l=0, r=0, t=10, b=15), height=200)
fig.show()


train['Episode_Length_minutes'] = train['Episode_Length_minutes'].clip(5, 120)



fig=go.Figure()
for col, color in zip(['Guest_Popularity_percentage', 'Host_Popularity_percentage'], (('royalblue', 'rgba(65, 105, 255, 0.6)', 'rgba(0, 0, 255, 0.2)'), ('red', 'rgba(255, 0, 0, 0.4)', 'rgba(255, 0, 0, 0.2)'))):
    q1, q2 = train[col].quantile([0.001, 0.999]) # 0.1%ile and 99.9%ile
    sampled = pd.concat([
        train.loc[train[col] < q1, col],
        train.loc[train[col] > q2, col],
        train.loc[train[col].between(q1, q2), col].sample(2500, random_state=42)
    ])
    
    fig.add_trace(go.Box(
        x=sampled, boxpoints='all', jitter=0.3, pointpos=-1.8, name=col, hoverinfo='text',
        hovertext = "Length: " + sampled.astype(str) + "%<br>Index: " + sampled.index.astype(str),
        hoverlabel=dict(bgcolor=color[0]),
        marker=dict(color=color[1], size=4),
        line=dict(color=color[0],width=2), fillcolor=color[2]
    ))

fig.update_layout(margin=dict(l=0, r=0, t=10, b=15), width=1000, height=300, showlegend=False)
fig.show()


train['Host_Popularity_percentage'] = train['Host_Popularity_percentage'].clip(20, 100)
train['Guest_Popularity_percentage'] = train['Guest_Popularity_percentage'].clip(0.01, 100)


train['Number_of_Ads'].value_counts().sort_index()


train['Number_of_Ads'] = train['Number_of_Ads'].clip(0, 3)



plt.figure(figsize=(4.5, 4.25))
plt.scatter(train["Episode_Length_minutes"], train["Listening_Time_minutes"], c='royalblue', s=5, alpha=0.5)
plt.plot([0, 120], [0, 120], c='k', ls='--')
plt.xlabel('Episode_Length_minutes')
plt.ylabel('Listening_Time_minutes')
plt.grid()
plt.show()


train.groupby(['Podcast_Name', 'Episode_Title'], observed=False).size().sort_index(level=[0, 1])


train.groupby('Podcast_Name', observed=False)['Genre'].value_counts().groupby(level=0, observed=False).head(2)



sampled = train.sample(100_000, random_state = 42)
corr = train[['Listening_Time_minutes', 'Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage']].corr().to_numpy()[0][1:]

axes = plt.subplots(1, 3, figsize=(13, 3.75))[1]
for i, col in enumerate(['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage']):
    axes[i].scatter(sampled[col], sampled['Listening_Time_minutes'], c='royalblue', alpha=0.005)
    axes[i].set_xlabel(col)
    axes[i].annotate("%.3f" % corr[i], (0.12, 0.93), xycoords='axes fraction', ha='center', va='center')
axes[0].set_ylabel('Listening_Time_minutes')

plt.show()



axes = plt.subplots(1, 3, figsize=(13, 3.5))[1]
for i, col in enumerate(['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage']):
    train[col+'_binned'] = pd.qcut(train[col], q=10)

    bucket_means = train.groupby(col+'_binned', observed=False)[col].mean().round(1)
    train[col+'_binned'] = train[col+'_binned'].map(bucket_means)

    sns.boxplot(data=train, x=col+'_binned', y='Listening_Time_minutes', ax = axes[i], hue=col+'_binned', palette="Set3", legend=False)
    sns.pointplot(data=train, x=col+'_binned', y='Listening_Time_minutes', ax=axes[i], alpha = 0.6, color='royalblue',
                  errorbar='sd', markers='o', estimator='median')
    if i!=0:
        axes[i].set_ylabel(None)
    axes[i].tick_params(axis='x', labelsize=7)
    axes[i].tick_params(axis='y', labelsize=9)
plt.show()
plt.close()



axes = plt.subplots(1, 3, figsize=(13, 3.5))[1]
for i, col in enumerate(['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage']):
    train[col+'_binned'] = pd.qcut(train[col], q=10)

    bucket_means = train.groupby(col+'_binned', observed=False)[col].mean().round(1)
    train[col+'_binned'] = train[col+'_binned'].map(bucket_means)

    sns.pointplot(data=train, x=col+'_binned', y='Listening_Time_minutes', ax=axes[i], alpha = 0.6, color='royalblue',
                  errorbar=None, markers='o', estimator='median')
    if i!=0:
        axes[i].set_ylabel(None)
    axes[i].tick_params(axis='x', labelsize=7)
    axes[i].tick_params(axis='y', labelsize=9)
    axes[i].grid()
plt.show()
plt.close()



axes = plt.subplots(1, 2, figsize=(12, 2.75), gridspec_kw={'width_ratios': [1, 2.25]}, constrained_layout=True)[1]

for i, col in enumerate(["Number_of_Ads", "Episode_Title"]):
    sns.boxplot(data=train, x=col, y='Listening_Time_minutes', ax=axes[i], hue=col, palette="Set3", legend=False)
    sns.pointplot(data=train, x=col, y='Listening_Time_minutes', ax=axes[i], alpha = [0.6, 0.3][i], color='royalblue',
                  errorbar='sd', markers='o', estimator='median')
    axes[i].set_title(col)
    axes[i].set_xlabel("")
    axes[i].set_ylabel('Listening_Time_minutes' if i == 0 else "")
axes[1].axhline(y=train['Listening_Time_minutes'].median(), color='r', linestyle='--', linewidth=0.5)
axes[0].set_xlim(-0.5, 3.5)
axes[1].set_xticks(np.arange(0, 100, 10)-1)
plt.show()



axes = plt.subplots(1, 2, figsize=(12, 2.75), gridspec_kw={'width_ratios': [1, 2.25]}, constrained_layout=True)[1]

for i, col in enumerate(["Number_of_Ads", "Episode_Title"]):
    sns.pointplot(data=train, x=col, y='Listening_Time_minutes', ax=axes[i], alpha=0.6, color='royalblue',
                  errorbar=None, markers='o', estimator='median')
    axes[i].set_ylabel('Listening_Time_minutes' if i == 0 else "")
    axes[i].grid()
axes[1].axhline(y=train['Listening_Time_minutes'].median(), color='r', linestyle='--', linewidth=0.5)
axes[0].set_xlim(-0.5, 3.5)
axes[1].set_xticks(np.arange(0, 100, 10)-1)
plt.show()



axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)[1]

for i, col in enumerate(["Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]):
    sns.boxplot(data=train, x=col, y='Listening_Time_minutes', order=sort_order[col], ax=axes[i], hue=col,
                palette=color_genre_dict if col == "Genre" else "Set3")
    sns.pointplot(data=train, x=col, y='Listening_Time_minutes', ax=axes[i], alpha = 0.6, color='royalblue',
                  errorbar='sd', markers='o', estimator='median')
    axes[i].axhline(y=train['Listening_Time_minutes'].median(), color='r', linestyle='--', linewidth=0.5)
    axes[i].set_title(col, fontsize=12)
    axes[i].tick_params(axis='x', rotation=-90)
    axes[i].set_xlabel("")
    axes[i].set_ylabel('Listening_Time_minutes' if i == 0 else "")
plt.show()



plt.figure(figsize = (10, 3))
sns.boxplot(data=train, x="Podcast_Name", y='Listening_Time_minutes', hue="Podcast_Name", palette=color_podcast_name_dict, legend=False)
sns.pointplot(data=train, x="Podcast_Name", y='Listening_Time_minutes', alpha = 0.6, color='royalblue',
              errorbar='sd', markers='o', estimator='median')
plt.axhline(y=train['Listening_Time_minutes'].median(), color='r', linestyle='--', linewidth=0.5)
plt.title("Podcast_Name")
plt.xlabel("")
plt.tick_params(axis='x', rotation=-90)
plt.xticks(fontsize=8)
plt.ylabel('Listening_Time_minutes')
plt.show()



axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)[1]

for i, col in enumerate(["Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]):
    sns.pointplot(data=train, x=col, y='Listening_Time_minutes', ax=axes[i], alpha = 0.6, color='royalblue',
                  errorbar=None, markers='o', estimator='median')
    axes[i].axhline(y=train['Listening_Time_minutes'].median(), color='r', linestyle='--', linewidth=0.5)
    axes[i].set_title(col, fontsize=12)
    axes[i].tick_params(axis='x', rotation=-90)
    axes[i].set_xlabel("")
    axes[i].set_ylabel('Listening_Time_minutes' if i == 0 else "")
plt.show()



plt.figure(figsize = (10, 3))
sns.pointplot(data=train, x="Podcast_Name", y='Listening_Time_minutes', alpha = 0.6, color='royalblue',
              errorbar=None, markers='o', estimator='median')
plt.axhline(y=train['Listening_Time_minutes'].median(), color='r', linestyle='--', linewidth=0.5)
plt.title("Podcast_Name")
plt.xlabel("")
plt.tick_params(axis='x', rotation=-90)
plt.xticks(fontsize=8)
plt.ylabel('Listening_Time_minutes')
plt.show()


# Feature importances
import catboost

X_train, y_train = train.iloc[:, :10], train["Listening_Time_minutes"]

model = catboost.CatBoostRegressor(iterations = 100, random_seed=42, thread_count=-1, verbose=False)
model.fit(X_train, y_train, cat_features = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment'])


pd.DataFrame(zip(model.feature_names_, model.feature_importances_)).sort_values(by=1, ascending=False).reset_index(drop=True)



sampled = train.sample(5_000, random_state = 42)
fig = go.Figure()

val = train.groupby('Number_of_Ads', observed=False)[['Episode_Length_minutes_binned', 'Listening_Time_minutes']].apply(lambda x: x)

for i, cat in enumerate(np.unique(train['Number_of_Ads'].dropna())):
    fig.add_trace(go.Scatter(x=sampled.loc[sampled['Number_of_Ads']==cat, 'Episode_Length_minutes'], y=sampled.loc[sampled['Number_of_Ads']==cat, 'Listening_Time_minutes'],
                  mode='markers', marker_color=px.colors.qualitative.Plotly[i], marker_opacity=0.2, opacity=0.4, legendgroup=cat, showlegend=False))
    val_ = val.loc[cat].groupby('Episode_Length_minutes_binned', observed=False).median()
    fig.add_trace(go.Scatter(x=val_.index.astype(float), y=val_.values.ravel(), mode='lines+markers', marker_color=px.colors.qualitative.Plotly[i],
                             name="Number_of_Ads={}".format(cat), legendgroup=cat))

fig.update_xaxes(title='Episode_Length_minutes', linewidth=1, linecolor='grey', mirror=True, ticks='outside', showline=True)
fig.update_yaxes(title='Listening_Time_minutes', linewidth=1, linecolor='grey', mirror=True, ticks='outside', showline=True)
fig.update_layout(width=525, height=350, showlegend=True, plot_bgcolor="white", margin=dict(l=35, r=10, t=30, b=5))
fig.show()



fig = go.Figure()

val = train.groupby('Host_Popularity_percentage_binned', observed=False)[['Episode_Length_minutes_binned', 'Listening_Time_minutes']].apply(lambda x: x)

for i, cat in enumerate(np.unique(train['Host_Popularity_percentage_binned'].dropna())):
    fig.add_trace(go.Scatter(x=sampled.loc[sampled['Host_Popularity_percentage_binned']==cat, 'Episode_Length_minutes'], y=sampled.loc[sampled['Host_Popularity_percentage_binned']==cat, 'Listening_Time_minutes'],
                  mode='markers', marker_color=px.colors.qualitative.Plotly[i], marker_opacity=0.2, opacity=0.4, legendgroup=cat, showlegend=False))
    val_ = val.loc[cat].groupby('Episode_Length_minutes_binned', observed=False).median()
    fig.add_trace(go.Scatter(x=val_.index.astype(float), y=val_.values.ravel(), mode='lines+markers', marker_color=px.colors.qualitative.Plotly[i],
                             name="Host_Popularity_percentage_binned={}".format(cat), legendgroup=cat))

fig.update_xaxes(title='Episode_Length_minutes', linewidth=1, linecolor='grey', mirror=True, ticks='outside', showline=True)
fig.update_yaxes(title='Listening_Time_minutes', linewidth=1, linecolor='grey', mirror=True, ticks='outside', showline=True)
fig.update_layout(width=600, height=350, showlegend=True, legend_font_size=9, plot_bgcolor="white", margin=dict(l=35, r=10, t=30, b=5))
fig.show()



fig = go.Figure()

val = train.groupby('Guest_Popularity_percentage_binned', observed=False)[['Episode_Length_minutes_binned', 'Listening_Time_minutes']].apply(lambda x: x)

for i, cat in enumerate(np.unique(train['Guest_Popularity_percentage_binned'].dropna())):
    fig.add_trace(go.Scatter(x=sampled.loc[sampled['Guest_Popularity_percentage_binned']==cat, 'Episode_Length_minutes'], y=sampled.loc[sampled['Guest_Popularity_percentage_binned']==cat, 'Listening_Time_minutes'],
                  mode='markers', marker_color=px.colors.qualitative.Plotly[i], marker_opacity=0.2, opacity=0.4, legendgroup=cat, showlegend=False))
    val_ = val.loc[cat].groupby('Episode_Length_minutes_binned', observed=False).median()
    fig.add_trace(go.Scatter(x=val_.index.astype(float), y=val_.values.ravel(), mode='lines+markers', marker_color=px.colors.qualitative.Plotly[i],
                             name="Guest_Popularity_percentage_binned={}".format(cat), legendgroup=cat))

fig.update_xaxes(title='Episode_Length_minutes', linewidth=1, linecolor='grey', mirror=True, ticks='outside', showline=True)
fig.update_yaxes(title='Listening_Time_minutes', linewidth=1, linecolor='grey', mirror=True, ticks='outside', showline=True)
fig.update_layout(width=600, height=350, showlegend=True, legend_font_size=9, plot_bgcolor="white", margin=dict(l=35, r=10, t=30, b=5))
fig.show()



fig = go.Figure()

val = train.groupby('Genre', observed=False)[['Episode_Length_minutes_binned', 'Listening_Time_minutes']].apply(lambda x: x)

for i, cat in enumerate(np.unique(train['Genre'].dropna())):
    fig.add_trace(go.Scatter(x=sampled.loc[sampled['Genre']==cat, 'Episode_Length_minutes'], y=sampled.loc[sampled['Genre']==cat, 'Listening_Time_minutes'],
                  mode='markers', marker_color=px.colors.qualitative.Plotly[i], marker_opacity=0.2, opacity=0.4, legendgroup=cat, showlegend=False))
    val_ = val.loc[cat].groupby('Episode_Length_minutes_binned', observed=False).median()
    fig.add_trace(go.Scatter(x=val_.index.astype(float), y=val_.values.ravel(), mode='lines+markers', marker_color=px.colors.qualitative.Plotly[i],
                             name="Genre={}".format(cat), legendgroup=cat))

fig.update_xaxes(title='Episode_Length_minutes', linewidth=1, linecolor='grey', mirror=True, ticks='outside', showline=True)
fig.update_yaxes(title='Listening_Time_minutes', linewidth=1, linecolor='grey', mirror=True, ticks='outside', showline=True)
fig.update_layout(width=525, height=350, showlegend=True, plot_bgcolor="white", margin=dict(l=35, r=10, t=30, b=5))
fig.show()



fig = go.Figure()

val = train.groupby('Episode_Sentiment', observed=False)[['Episode_Length_minutes_binned', 'Listening_Time_minutes']].apply(lambda x: x)

for i, cat in enumerate(np.unique(train['Episode_Sentiment'].dropna())):
    fig.add_trace(go.Scatter(x=sampled.loc[sampled['Episode_Sentiment']==cat, 'Episode_Length_minutes'], y=sampled.loc[sampled['Episode_Sentiment']==cat, 'Listening_Time_minutes'],
                  mode='markers', marker_color=px.colors.qualitative.Plotly[i], marker_opacity=0.2, opacity=0.4, legendgroup=cat, showlegend=False))
    val_ = val.loc[cat].groupby('Episode_Length_minutes_binned', observed=False).median()
    fig.add_trace(go.Scatter(x=val_.index.astype(float), y=val_.values.ravel(), mode='lines+markers', marker_color=px.colors.qualitative.Plotly[i],
                             name="Episode_Sentiment={}".format(cat), legendgroup=cat))

fig.update_xaxes(title='Episode_Length_minutes', linewidth=1, linecolor='grey', mirror=True, ticks='outside', showline=True)
fig.update_yaxes(title='Listening_Time_minutes', linewidth=1, linecolor='grey', mirror=True, ticks='outside', showline=True)
fig.update_layout(width=600, height=350, showlegend=True, plot_bgcolor="white", margin=dict(l=35, r=10, t=30, b=5))
fig.show()

