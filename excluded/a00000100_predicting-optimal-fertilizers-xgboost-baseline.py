import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', 100)

%pip install -q -U seaborn
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

%pip install -q kaleido==0.2.1
from IPython.display import Image, display

import plotly.io as pio
pio.renderers.default = 'iframe'

%pip install -q -U scikit-learn
from sklearn.model_selection import cross_val_score

%pip install -q -U xgboost
%pip install -q -U lightgbm
%pip install -q -U catboost


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


# MAP@3
from sklearn.utils.validation import check_consistent_length
from sklearn.metrics import make_scorer

def MAP3(y_true, y_pred):
    check_consistent_length(y_true, y_pred)
    score = 0.0
    for label, preds in zip(y_true, y_pred):
        for rank, pred in enumerate(preds.split()[:3], start=1):
            if pred == str(label):
                score += 1.0 / rank
                break
    return score / len(y_pred)

mean_average_precision_at_3 = make_scorer(MAP3, greater_is_better=True)


# Original dataset
data = pd.read_csv(input_path + 'fertilizer-prediction/Fertilizer Prediction.csv')


def load_data(**kwargs):
    path = input_path + 'playground-series-s5e6/'
    return pd.read_csv(path + 'train.csv', **kwargs), pd.read_csv(path + 'test.csv', **kwargs)

train, test = load_data()


data.head()


train.head()



from io import StringIO
from IPython.display import display, HTML

buffer = StringIO()

data.info(buf=buffer)

info = buffer.getvalue()

html = f"""
<div style="display: flex; gap: 20px;">
    <div>
        <div style="font-weight: bold; margin-bottom: 5px;">Original Dataset</div>
        <pre style="border: 1px solid #ccc; padding: 10px;">{info}</pre>
    </div>
</div>
"""

display(HTML(html))



from io import StringIO
from IPython.display import display, HTML

buffer1 = StringIO()
buffer2 = StringIO()

train.info(buf=buffer1)
test.info(buf=buffer2)

info1 = buffer1.getvalue()
info2 = buffer2.getvalue()

html = f"""
<div style="display: flex; gap: 20px;">
    <div>
        <div style="font-weight: bold; margin-bottom: 5px;">Train</div>
        <pre style="border: 1px solid #ccc; padding: 10px;">{info1}</pre>
    </div>
    <div>
        <div style="font-weight: bold; margin-bottom: 5px;">Test</div>
        <pre style="border: 1px solid #ccc; padding: 10px;">{info2}</pre>
    </div>
</div>
"""

display(HTML(html))


# "id" is the same as index
assert (train["id"] == train.index).all()
assert (test ["id"] == test .index + len(train)).all()


target = 'Fertilizer Name'
num_att = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
cat_att = ['Soil Type', 'Crop Type']


train, test = load_data(index_col='id')

X_train = train.drop(columns=target)
y_train = train[target]



def get_bar_trace(df, name, visible=True):
    val_counts = df[target].value_counts()
    perc = (val_counts.values / len(df[target]) * 100).round(1)
    return go.Bar(
        x=val_counts.index.astype(str),
        y=val_counts.values,
        width=0.5,
        text=np.char.add(perc.astype(str), "%"),
        textfont_size=11,
        hovertext=val_counts.values,
        hoverinfo="text",
        textposition="outside",
        marker_color="royalblue",
        name=name,
        visible=visible
    ), val_counts.max()

train_trace, train_max = get_bar_trace(train, "train", visible=True)
data_trace, data_max = get_bar_trace(data, "data", visible=False)

fig = make_subplots(rows=1, cols=1, subplot_titles=[target])
fig.add_trace(train_trace, row=1, col=1)
fig.add_trace(data_trace, row=1, col=1)

axis_style = dict(
    linewidth=1,
    linecolor="grey",
    mirror=True,
    ticks="outside",
    showline=True
)

fig.update_layout(
    width=550,
    height=300,
    showlegend=False,
    plot_bgcolor="white",
    uniformtext_minsize=12,
    uniformtext_mode="show",
    margin=dict(l=35, r=60, t=50, b=5),
    title_text="Target",
    title=dict(x=0.45, y=0.99, font_color="black"),
    updatemenus=[
        {
            "buttons": [
                {
                    "label": "Train",
                    "method": "update",
                    "args": [
                        {"visible": [True, False]},
                        {"yaxis": {
                            "range": [0, train_max * 1.15],
                            **axis_style
                        }}
                    ],
                },
                {
                    "label": "Original Data",
                    "method": "update",
                    "args": [
                        {"visible": [False, True]},
                        {"yaxis": {
                            "range": [0, data_max * 1.15],
                            **axis_style
                        }}
                    ],
                },
            ],
            "type": "buttons",
            "direction": "down",
            "x": 1.3,
            "y": 1.035,
            "showactive": True,
            "pad": {"r": 5, "t": 5},
            "font": {"size": 12},
            "borderwidth": 1,
        }
    ],
)

fig.update_yaxes(range=[0, train_max * 1.15], row=1, col=1, **axis_style)
fig.update_xaxes(**axis_style)

fig.show()



from sklearn.base import BaseEstimator

class CustomConstantClassifier(BaseEstimator):
    def __init__(self, constant):
        self.constant = constant
    
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        y_pred = [self.constant] * len(X)
        return np.array(y_pred)



top3 = train[target].value_counts().head(3).index.to_list()
constant = ' '.join(top3)

model = CustomConstantClassifier(constant)
model.fit(X_train, y_train)
tra_score = mean_average_precision_at_3(model, X_train, y_train)
print(f'Tra score: {tra_score: .5f}')

cv_scores = cross_val_score(model, X_train, y_train, scoring=mean_average_precision_at_3, cv=10, n_jobs=-1)
print(f'CV  score: {cv_scores.mean(): .5f}, std: {cv_scores.std().round(4)}')



def get_bar_traces(df, visible=False):
    traces = []
    y_maxes = []
    for i, att in enumerate(cat_att):
        val_counts = df[att].value_counts()
        perc = (val_counts.values / len(df[att]) * 100).round(1)
        trace = go.Bar(
            x=val_counts.index.astype(str),
            y=val_counts.values,
            width=0.5,
            text=np.char.add(perc.astype(str), '%'),
            textfont_size=11,
            hovertext=val_counts.values,
            hoverinfo='text',
            textposition='outside',
            marker_color='royalblue',
            name=f"{att}_{'train' if visible else 'data'}",
            visible=visible
        )
        traces.append((trace, val_counts.max()))
    return traces

train_traces = get_bar_traces(train, visible=True)
data_traces = get_bar_traces(data, visible=False)

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=cat_att,
    column_widths=[0.3, 0.7],
    horizontal_spacing=0.05
)

for i, (trace, _) in enumerate(train_traces):
    fig.add_trace(trace, row=1, col=i+1)
for i, (trace, _) in enumerate(data_traces):
    fig.add_trace(trace, row=1, col=i+1)

train_maxes = [m for _, m in train_traces]
data_maxes = [m for _, m in data_traces]

axis_style = dict(
    linewidth=1,
    linecolor='grey',
    mirror=True,
    ticks='outside',
    showline=True
)

fig.update_yaxes(range=[0, train_maxes[0] * 1.15], row=1, col=1, **axis_style)
fig.update_yaxes(range=[0, train_maxes[1] * 1.15], row=1, col=2, **axis_style)
fig.update_xaxes(**axis_style, tickangle=0)

fig.update_layout(
    width=1100,
    height=275,
    showlegend=False,
    plot_bgcolor='white',
    uniformtext_minsize=12,
    uniformtext_mode='show',
    margin=dict(l=35, r=70, t=50, b=5),
    title_text='Categorical attributes',
    title=dict(x=0.45, y=0.97, font_color='black'),
    updatemenus=[
        {
            "buttons": [
                {
                    "label": "Train",
                    "method": "update",
                    "args": [
                        {"visible": [True, True, False, False]},
                        {
                            "yaxis.range": [0, train_maxes[0] * 1.15],
                            "yaxis2.range": [0, train_maxes[1] * 1.15]
                        }
                    ]
                },
                {
                    "label": "Original Data",
                    "method": "update",
                    "args": [
                        {"visible": [False, False, True, True]},
                        {
                            "yaxis.range": [0, data_maxes[0] * 1.15],
                            "yaxis2.range": [0, data_maxes[1] * 1.15]
                        }
                    ]
                }
            ],
            "type": "buttons",
            "direction": "down",
            "x": 1.125,
            "y": 1.04,
            "showactive": True,
            "font": {"size": 12},
            "borderwidth": 1,
            "pad": {"r": 5, "t": 5}
        }
    ]
)

fig.show()


test[cat_att].agg(['unique']).T



n_cols = 3
n_features = len(num_att)
n_groups = (n_features + n_cols - 1) // n_cols
n_rows = n_groups * 3 - 1

fig = make_subplots(
    rows=n_rows,
    cols=n_cols,
    row_heights=[0.55 if i % 3 == 0 else 0.2 if i % 3 == 1 else 0.15 for i in range(n_rows)],
    vertical_spacing=0.03,
    horizontal_spacing=0.05,
    subplot_titles=num_att[:3] + [' '] * 6 + num_att[3:]
)

trace_visibility_train = []
trace_visibility_data = []

def add_traces(df, is_train):
    for i, att in enumerate(num_att):
        group = i // n_cols
        col_pos = i % n_cols
        hist_row = group * 3 + 1
        box_row = hist_row + 1

        x_data = df[att]
        bins = np.arange(x_data.min() - 0.5, x_data.max() + 1.5, 1)
        hist_vals, bin_edges = np.histogram(x_data, bins=bins)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        # Histogram
        fig.add_trace(go.Bar(
            x=bin_centers, y=hist_vals,
            showlegend=False, width=1, name=att,
            marker=dict(color='#636EFA', line=dict(color='black', width=1)),
            visible=is_train
        ), row=hist_row, col=col_pos + 1)

        trace_visibility_train.append(is_train)
        trace_visibility_data.append(not is_train)

        # KDE
        kde = sns.kdeplot(x_data, bw_adjust=2, cut=0)
        x_kde, y_kde = kde.get_lines()[0].get_data()
        plt.clf(); plt.close()
        fig.add_trace(go.Scatter(
            x=x_kde, y=y_kde * len(x_data) * (bin_edges[1] - bin_edges[0]),
            mode='lines', line=dict(color='royalblue'), name=att,
            showlegend=False, visible=is_train
        ), row=hist_row, col=col_pos + 1)

        trace_visibility_train.append(is_train)
        trace_visibility_data.append(not is_train)

        # Boxplot
        fig.add_trace(go.Box(
            x=x_data, orientation='h',
            fillcolor='#636EFA', name=att,
            line_width=1, line_color='black',
            boxpoints=False, showlegend=False,
            visible=is_train,
            hoverlabel=dict(bgcolor='royalblue', font_color='white')
        ), row=box_row, col=col_pos + 1)

        trace_visibility_train.append(is_train)
        trace_visibility_data.append(not is_train)

        xmax, xmin = x_data.max(), x_data.min()
        range_x = [xmin - 0.075 * (xmax - xmin), xmax + 0.075 * (xmax - xmin)]
        fig.update_xaxes(range=range_x, showticklabels=False, row=hist_row, col=col_pos + 1)
        fig.update_xaxes(range=range_x, row=box_row, col=col_pos + 1)
        fig.update_yaxes(range=[-0.33, 0.33], showticklabels=False, row=box_row, col=col_pos + 1)
add_traces(train, is_train=True)
add_traces(data, is_train=False)

for r in range(3, n_rows + 1, 3):
    for c in range(1, n_cols + 1):
        fig.update_xaxes(visible=False, row=r, col=c)
        fig.update_yaxes(visible=False, row=r, col=c)

fig.update_xaxes(linewidth=1, linecolor='grey', mirror=True, ticks='outside', showline=True)
fig.update_yaxes(linewidth=1.5, linecolor='grey', mirror=True, ticks='outside', showline=True)

fig.update_layout(
    height=3.8 * n_groups * 85,
    width=1150,
    title_text="Numerical attributes",
    title_x=0.48, title_y=0.99,
    title_font_color='black',
    showlegend=False,
    plot_bgcolor="white",
    margin=dict(t=50, l=50, r=0, b=30),
    updatemenus=[
        {
            "buttons": [
                {
                    "label": "Train",
                    "method": "update",
                    "args": [{"visible": trace_visibility_train}]
                },
                {
                    "label": "Original Data",
                    "method": "update",
                    "args": [{"visible": trace_visibility_data}]
                }
            ],
            "type": "buttons",
            "direction": "down",
            "x": 1.125,
            "y": 1.01,
            "showactive": True,
            "font": {"size": 12},
            "borderwidth": 1,
            "pad": {"r": 5, "t": 5}
        }
    ]
)

fig.show()

pd.concat([train[num_att].describe(), train[num_att].agg(['nunique'])]).T


test[num_att].agg(['min', 'max', 'nunique']).T



def generate_grouped_bar_traces(df, visible=False):
    traces = []
    y_maxes = []

    for i, att in enumerate(cat_att, start=1):
        group = df.groupby([att], observed=False)[target]
        perc = 100 * group.value_counts(normalize=True).unstack().reindex(df[att].value_counts().index)
        counts = group.value_counts().unstack().reindex(df[att].value_counts().index)
        
        max_count = counts.values.max()
        y_maxes.append(max_count)

        opacities = np.sqrt(counts / max_count)

        for j, fertilizer in enumerate(df[target].value_counts().index):
            trace = go.Bar(
                x=counts.index.astype(str),
                y=counts[fertilizer],
                name=fertilizer,
                marker=dict(
                    color=px.colors.qualitative.Plotly[j % 10],
                    opacity=opacities[fertilizer],
                    line=dict(width=0.5, color='black')
                ),
                hovertext=counts[fertilizer],
                hoverinfo='text',
                textposition="outside",
                showlegend=True if i == 1 else False,
                legendgroup=str(fertilizer),
                visible=visible
            )
            traces.append((trace, i))

    return traces, y_maxes

train_traces, train_maxes = generate_grouped_bar_traces(train, visible=True)
data_traces, data_maxes = generate_grouped_bar_traces(data, visible=False)

fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=cat_att,
    vertical_spacing=0.15,
    specs=[[{"colspan": 2}, None, None],
           [{"colspan": 3}, None, None]]
)

for (trace, row_idx), (_, col_idx) in zip(train_traces, [(1, 1)] * len(train_traces[:len(cat_att[0:1])*2]) + [(2, 1)] * (len(train_traces) - len(cat_att[0:1])*2)):
    fig.add_trace(trace, row=row_idx, col=1)
for (trace, row_idx), (_, col_idx) in zip(data_traces, [(1, 1)] * len(data_traces[:len(cat_att[0:1])*2]) + [(2, 1)] * (len(data_traces) - len(cat_att[0:1])*2)):
    fig.add_trace(trace, row=row_idx, col=1)

axis_style = dict(
    linewidth=1,
    linecolor='grey',
    mirror=True,
    ticks='outside',
    showline=True,
    tickangle=0
)

fig.update_yaxes(range=[0, train_maxes[0] * 1.1], row=1, col=1, **axis_style)
fig.update_yaxes(range=[0, train_maxes[1] * 1.1], row=2, col=1, **axis_style)
fig.update_xaxes(**axis_style)

n_classes = train[target].nunique()
n_vars = len(cat_att)
n_traces = n_classes * n_vars

fig.update_layout(
    height=425,
    width=1100,
    showlegend=True,
    plot_bgcolor="white",
    margin=dict(l=35, r=10, t=50, b=5),
    legend_title_text=target,
    legend=dict(tracegroupgap=1, x=0.67, y=1.05),
    updatemenus=[
        {
            "buttons": [
                {
                    "label": "Train",
                    "method": "update",
                    "args": [
                        {"visible": [True]*n_traces + [False]*n_traces},
                        {
                            "yaxis.range": [0, train_maxes[0] * 1.1],
                            "yaxis2.range": [0, train_maxes[1] * 1.1]
                        }
                    ]
                },
                {
                    "label": "Original Data",
                    "method": "update",
                    "args": [
                        {"visible": [False]*n_traces + [True]*n_traces},
                        {
                            "yaxis.range": [0, data_maxes[0] * 1.1],
                            "yaxis2.range": [0, data_maxes[1] * 1.1]
                        }
                    ]
                }
            ],
            "type": "buttons",
            "direction": "down",
            "x": 0.925,
            "y": 0.95,
            "showactive": True,
            "font": {"size": 12},
            "borderwidth": 1,
            "pad": {"r": 5, "t": 5}
        }
    ]
)

fig.show()



from sklearn.base import BaseEstimator

class GroupwiseClassifier(BaseEstimator):
    def __init__(self, features, bin_factor=1):
        self.features = features
        if bin_factor<=0 or bin_factor>1:
            raise ValueError('`bin_factor` must lie in the range (0, 1]')
        self.bin_factor = bin_factor
    
    def fit(self, X, y):
        X = X.copy()
        
        self.bin_edges_ = {}
        if self.bin_factor!=1:
            for col in X.select_dtypes('number').columns:
                X[col], bins = pd.qcut(X[col], q=int(X[col].nunique() * self.bin_factor), duplicates='drop', retbins=True)
                self.bin_edges_[col] = bins
        
        top3_per_group = (
            y.groupby([X[feature] for feature in self.features], observed=False)
             .value_counts()
             .groupby(level=list(range(len(self.features))), observed=False)
             .head(3)
             .index
        ).to_frame(index=False)
        
        self.top3_dict_ = (
            top3_per_group
            .groupby(self.features, observed=False)[y.name]
            .apply(lambda x: ' '.join(x))
            .to_dict()
        )
        return self
    
    def predict(self, X):
        X = X.copy()
        
        if self.bin_factor!=1:
            for col in X.select_dtypes('number').columns:
                X[col] = pd.cut(X[col], bins=self.bin_edges_[col], include_lowest=True)
        
        if len(self.features)==1:
            keys = list(X[self.features].values.ravel())
        else:
            keys = list(zip(*[X[feature] for feature in self.features]))
        
        y_pred = [self.top3_dict_.get(key, '') for key in keys]
        return np.array(y_pred)



from itertools import combinations

def RunBaselines(*, group_size, cat_features=None, num_features=None, mix_dtypes_only=True, bin_factor=1, cv=10):
    scores = pd.DataFrame(columns=['train', 'CV', 'std'])
    features = (cat_features or []) + (num_features or [])
    
    for group in combinations(features, group_size):
        if cat_features and num_features and mix_dtypes_only:
            if set(group).issubset(cat_features) or set(group).issubset(num_features):
                continue
        index = ' - '.join(group)
        model = GroupwiseClassifier(features=list(group), bin_factor=bin_factor)
        model.fit(X_train, y_train)
        
        tra_score = mean_average_precision_at_3(model, X_train, y_train)
        scores.loc[index, 'train'] = tra_score
        
        cv_scores = cross_val_score(model, X_train, y_train, scoring=mean_average_precision_at_3, cv=cv, n_jobs=-1)
        scores.loc[index, 'CV' ] = cv_scores.mean()
        scores.loc[index, 'std'] = cv_scores.std()
    return scores


RunBaselines(cat_features=cat_att, group_size=1).sort_values('CV', ascending=False)



n_cols, n_rows = 3, 2
colors = px.colors.qualitative.Plotly
categories = sorted(set(train[target]))

def generate_histogram_traces(df, visible=False):
    traces = []
    x_ranges = []

    for i, att in enumerate(num_att):
        bins = np.arange(df[att].min() - 0.5, df[att].max() + 1.5, 1)

        row = i // n_cols + 1
        col = i % n_cols + 1

        for j, cat in enumerate(categories):
            x_vals = df[df[target] == cat][att]
            trace = go.Histogram(
                x=x_vals,
                xbins=dict(start=bins[0], end=bins[-1], size=1),
                name=str(cat),
                marker=dict(
                    color=colors[j % len(colors)],
                    line=dict(width=0.5, color='black')
                ),
                opacity=0.7,
                showlegend=(i == 2),
                legendgroup=str(cat),
                visible=visible
            )
            traces.append((trace, row, col))

            xmax, xmin = x_vals.max(), x_vals.min()
            range_x = [xmin - 0.075 * (xmax - xmin), xmax + 0.075 * (xmax - xmin)]
            x_ranges.append((range_x, row, col))

    return traces, x_ranges

def get_kde_traces(df, visible=False):
    traces = []
    x_ranges = {}

    for i, att in enumerate(num_att):
        row = i // n_cols + 1
        col = i % n_cols + 1
        xmin_total, xmax_total = np.inf, -np.inf

        for j, cat in enumerate(categories):
            subset = df[df[target] == cat][att].dropna()
            if len(subset) < 2:
                continue

            fig_, ax = plt.subplots()
            sns.kdeplot(
                x=subset,
                cut=0,
                bw_adjust=1.5,
                fill=False,
                ax=ax,
                color=colors[j % len(colors)],
                linewidth=2
            )
            line = ax.lines[-1]
            x_vals = line.get_xdata()
            y_vals = line.get_ydata()
            plt.close(fig_)

            xmin_total = min(xmin_total, x_vals.min())
            xmax_total = max(xmax_total, x_vals.max())

            trace = go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                name=str(cat),
                legendgroup=str(cat),
                line=dict(width=2, color=colors[j % len(colors)]),
                showlegend=(i == 2),
                visible=visible
            )
            traces.append((trace, row, col))

        if np.isfinite(xmin_total) and np.isfinite(xmax_total):
            padding = 0.075 * (xmax_total - xmin_total)
            x_ranges[(row, col)] = [xmin_total - padding, xmax_total + padding]

    return traces, x_ranges

hist_train_traces, train_xranges = generate_histogram_traces(train, visible=True)
hist_data_traces, _ = generate_histogram_traces(data, visible=False)
kde_train_traces, _ = get_kde_traces(train, visible=False)
kde_data_traces, _ = get_kde_traces(data, visible=False)

fig = make_subplots(
    rows=n_rows, cols=n_cols,
    subplot_titles=num_att,
    horizontal_spacing=0.04,
    vertical_spacing=0.15
)

all_traces = hist_train_traces + hist_data_traces + kde_train_traces + kde_data_traces
for trace, row, col in all_traces:
    fig.add_trace(trace, row=row, col=col)

for i in range(len(num_att)):
    row = i // n_cols + 1
    col = i % n_cols + 1
    fig.update_xaxes(showline=True, linewidth=1, linecolor='grey', mirror=True, row=row, col=col)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='grey', mirror=True, row=row, col=col)

for (range_x, row, col) in train_xranges:
    fig.update_xaxes(range=range_x, row=row, col=col)

n_hist = len(hist_train_traces)
n_kde = len(kde_train_traces)

visibility = {
    "hist_train": [True]*n_hist + [False]*n_hist + [False]*n_kde + [False]*n_kde,
    "hist_data":  [False]*n_hist + [True]*n_hist + [False]*n_kde + [False]*n_kde,
    "kde_train":  [False]*n_hist + [False]*n_hist + [True]*n_kde + [False]*n_kde,
    "kde_data":   [False]*n_hist + [False]*n_hist + [False]*n_kde + [True]*n_kde,
}

current_mode = "hist_train"

fig.update_layout(
    barmode='stack',
    height=500,
    width=1150,
    plot_bgcolor='rgba(240,240,245,0.5)',
    paper_bgcolor='rgba(255,255,255,0)',
    margin=dict(t=60, l=40, r=20, b=40),
    legend=dict(title=target, x=1.01, y=1.01, font=dict(size=10), tracegroupgap=0),
    updatemenus=[
        {
            "buttons": [
                {
                    "label": "Hist – Train",
                    "method": "update",
                    "args": [{"visible": visibility["hist_train"]}]
                },
                {
                    "label": "KDE – Train",
                    "method": "update",
                    "args": [{"visible": visibility["kde_train"]}]
                },
                {
                    "label": "Hist – Original Data",
                    "method": "update",
                    "args": [{"visible": visibility["hist_data"]}]
                },
                {
                    "label": "KDE – Original Data",
                    "method": "update",
                    "args": [{"visible": visibility["kde_data"]}]
                }
            ],
            "type": "buttons",
            "direction": "down",
            "x": 1.175,
            "y": 0.36,
            "showactive": True,
            "font": {"size": 12},
            "borderwidth": 1,
            "pad": {"r": 5, "t": 5}
        }
    ]
)

fig.show()


RunBaselines(num_features=num_att, group_size=1, bin_factor=1).sort_values('CV', ascending=False)


# Cat × Cat interactions
def generate_soil_crop_traces(df, visible=False):
    group = df.groupby(['Crop Type'], observed=False)['Soil Type']
    perc = 100 * group.value_counts(normalize=True).unstack().reindex(df['Crop Type'].value_counts().index)
    counts = group.value_counts().unstack().reindex(df['Crop Type'].value_counts().index)
    
    max_count = counts.values.max()
    opacities = np.sqrt(counts / max_count)

    traces = []
    for i, soil in enumerate(df['Soil Type'].value_counts().index):
        trace = go.Bar(
            x=counts.index.astype(str),
            y=counts[soil],
            name=soil,
            marker=dict(
                color=px.colors.qualitative.Plotly[i % 10],
                opacity=opacities[soil],
                line=dict(width=0.5, color='black')
            ),
            hovertext=counts[soil],
            hoverinfo='text',
            textposition="outside",
            visible=visible,
            legendgroup=str(soil),
            showlegend=True
        )
        traces.append(trace)
    return traces, max_count

train_traces, train_max = generate_soil_crop_traces(train, visible=True)
data_traces, data_max = generate_soil_crop_traces(data, visible=False)
n_traces = len(train_traces)

fig = go.Figure()
for trace in train_traces + data_traces:
    fig.add_trace(trace)

axis_style = dict(
    linewidth=1,
    linecolor='grey',
    mirror=True,
    ticks='outside',
    showline=True,
    tickangle=0
)

fig.update_layout(
    height=225,
    width=1150,
    showlegend=True,
    plot_bgcolor="white",
    margin=dict(l=15, r=10, t=30, b=5),
    legend_title_text='Soil Type',
    legend=dict(tracegroupgap=1, x=1.01, y=1),
    xaxis_title='Crop Type',
    updatemenus=[
        {
            "buttons": [
                {
                    "label": "Train",
                    "method": "update",
                    "args": [
                        {"visible": [True]*n_traces + [False]*n_traces},
                        {"yaxis.range": [0, train_max * 1.1]}
                    ]
                },
                {
                    "label": "Original Data",
                    "method": "update",
                    "args": [
                        {"visible": [False]*n_traces + [True]*n_traces},
                        {"yaxis.range": [0, data_max * 1.1]}
                    ]
                }
            ],
            "type": "buttons",
            "direction": "down",
            "x": 1.26,
            "y": 0.8,
            "showactive": True,
            "font": {"size": 12},
            "borderwidth": 1,
            "pad": {"r": 5, "t": 5}
        }
    ]
)

fig.update_xaxes(**axis_style)
fig.update_yaxes(**axis_style)

fig.show()


# Num × Num interactions
df_train = train.sample(50_000, random_state=42)
df_data  = data.sample(50_000, random_state=42)

features   = num_att
target_col = target

classes = df_train[target_col].unique()
palette = px.colors.qualitative.Plotly
color_map = {c: palette[i % len(palette)] for i, c in enumerate(classes)}

n = len(features)
fig = make_subplots(
    rows=n-1, cols=n-1,
    shared_xaxes=True, shared_yaxes=True,
    horizontal_spacing=0, vertical_spacing=0,
    subplot_titles=features[1:]
)

def add_layer(df, visible: bool):
    start_index = len(fig.data)
    for r, y in enumerate(features[:-1],  start=1):
        for c, x in enumerate(features[1:], start=1):
            if r > c:
                continue
            fig.add_trace(
                go.Scattergl(
                    x=df[x], y=df[y],
                    mode="markers",
                    marker=dict(opacity=0.015, color='royalblue'),
                    showlegend=False,
                    visible=visible,
                    name='',
                ),
                row=r, col=c
            )
    return len(fig.data) - start_index

n_train = add_layer(df_train, visible=True)
n_data  = add_layer(df_data,  visible=False)

axis_style = dict(linewidth=1, linecolor="grey", mirror=True,
                  ticks="outside", showline=True, tickangle=0)

for r, y in enumerate(features[:-1],  start=1):
    for c, x in enumerate(features[1:], start=1):
        if r > c:
            continue
        rng_x = pd.concat([df_train[x], df_data[x]]).agg(['min', 'max']).values
        rng_y = pd.concat([df_train[y], df_data[y]]).agg(['min', 'max']).values

        fig.update_xaxes(range=rng_x, row=r, col=c, **axis_style)
        fig.update_yaxes(range=rng_y, row=r, col=c, **axis_style)

        ticks, title = '', None
        if r == c:
            fig.update_xaxes(showticklabels=True, row=r, col=c)
            fig.update_yaxes(showticklabels=True, row=r, col=c)
            ticks = 'outside'
            title = y
        if r == 1 and c == 1:
            fig.update_xaxes(matches=None, row=r, col=c)
            fig.update_yaxes(matches=None, row=r, col=c)

        fig.update_xaxes(linewidth=1, linecolor='grey',
                         mirror=True, ticks=ticks, showline=True, row=r, col=c)
        fig.update_yaxes(linewidth=1, linecolor='grey',
                         mirror=True, ticks=ticks, showline=True, row=r, col=c, title=title)

corr_train = train[features].corr()
corr_data  = data[features].corr()
ann_idx_train = []
ann_idx_data = []
subplot_title_idxs = []
for i, annotation in enumerate(fig.layout.annotations):
    if annotation.text in features:
        subplot_title_idxs.append(i)

for r, y in enumerate(features[:-1],  start=1):
    for c, x in enumerate(features[1:], start=1):
        if r > c:
            continue

        corr_val_train = corr_train.loc[y, x]
        corr_val_data  = corr_data.loc[y, x]

        x_mid = df_train[x].min() + 0.2 * (df_train[x].max() - df_train[x].min())
        y_mid = df_train[y].max() - 0.125 * (df_train[y].max() - df_train[y].min())

        subplot_idx = (r - 1) * (n - 1) + c
        xref = f"x{subplot_idx}"
        yref = f"y{subplot_idx}"

        ann_train = dict(
            x=x_mid, y=y_mid,
            text=f"{corr_val_train:.3f}",
            showarrow=False,
            font=dict(size=15),
            xref=xref, yref=yref,
            visible=True
        )
        ann_data = dict(
            x=x_mid, y=y_mid,
            text=f"{corr_val_data:.3f}",
            showarrow=False,
            font=dict(size=15),
            xref=xref, yref=yref,
            visible=False
        )
        fig.add_annotation(ann_train)
        fig.add_annotation(ann_data)
        ann_idx_train.append(len(fig.layout.annotations)-2)
        ann_idx_data.append(len(fig.layout.annotations)-1)

for i in subplot_title_idxs:
    fig.layout.annotations[i].font.size = 15
    fig.layout.annotations[i].y += 0.01

n_proxy = 0
visible_train =  [True]*n_train + [False]*n_data + [True]*n_proxy
visible_data  =  [False]*n_train + [True]*n_data + [True]*n_proxy

def make_visibility_args(trace_visibility, annotation_on, annotation_off):
    ann_visibility = [False] * len(fig.layout.annotations)
    for idx in annotation_on:
        ann_visibility[idx] = True
    for idx in annotation_off:
        ann_visibility[idx] = False
    for idx in subplot_title_idxs:
        ann_visibility[idx] = True

    updated_annotations = []
    for i, ann in enumerate(fig.layout.annotations):
        ann_dict = ann.to_plotly_json()
        ann_dict["visible"] = ann_visibility[i]
        updated_annotations.append(ann_dict)

    return [{"visible": trace_visibility}, {"annotations": updated_annotations}]

fig.update_layout(
    height=700, width=900,
    plot_bgcolor="white",
    margin=dict(t=45, b=20, l=20, r=200),
    showlegend=False,
    legend_title_text=target_col,
    legend=dict(tracegroupgap=1, x=1.03, y=1),
    updatemenus=[{
        "buttons": [
            {
                "label": "Train",
                "method": "update",
                "args": make_visibility_args(visible_train, ann_idx_train, ann_idx_data),
            },
            {
                "label": "Original Data",
                "method": "update",
                "args": make_visibility_args(visible_data, ann_idx_data, ann_idx_train),
            },
        ],
        "type": "buttons", "direction": "down",
        "x": 1.2, "y": 1.01,
        "showactive": True,
        "font": {"size": 12},
        "pad": {"r": 5, "t": 5},
        "borderwidth": 1
    }]
)

fig.show()


# Cat × Num interactions
n_cols, n_rows = 3, 2
colors = px.colors.qualitative.Plotly
categories = sorted(set(train['Soil Type']))

def generate_histogram_traces(df, visible=False):
    traces = []
    x_ranges = []

    for i, att in enumerate(num_att):
        bins = np.arange(df[att].min() - 0.5, df[att].max() + 1.5, 1)

        row = i // n_cols + 1
        col = i % n_cols + 1

        for j, cat in enumerate(categories):
            x_vals = df[df['Soil Type'] == cat][att]
            trace = go.Histogram(
                x=x_vals,
                xbins=dict(start=bins[0], end=bins[-1], size=1),
                name=str(cat),
                marker=dict(
                    color=colors[j % len(colors)],
                    line=dict(width=0.5, color='black')
                ),
                opacity=0.7,
                showlegend=(i == 2),
                legendgroup=str(cat),
                visible=visible
            )
            traces.append((trace, row, col))

            xmax, xmin = x_vals.max(), x_vals.min()
            range_x = [xmin - 0.075 * (xmax - xmin), xmax + 0.075 * (xmax - xmin)]
            x_ranges.append((range_x, row, col))

    return traces, x_ranges

def get_kde_traces(df, visible=False):
    traces = []
    x_ranges = {}

    for i, att in enumerate(num_att):
        row = i // n_cols + 1
        col = i % n_cols + 1
        xmin_total, xmax_total = np.inf, -np.inf

        for j, cat in enumerate(categories):
            subset = df[df['Soil Type'] == cat][att].dropna()
            if len(subset) < 2:
                continue

            fig_, ax = plt.subplots()
            sns.kdeplot(
                x=subset,
                cut=0,
                bw_adjust=1.5,
                fill=False,
                ax=ax,
                color=colors[j % len(colors)],
                linewidth=2
            )
            line = ax.lines[-1]
            x_vals = line.get_xdata()
            y_vals = line.get_ydata()
            plt.close(fig_)

            xmin_total = min(xmin_total, x_vals.min())
            xmax_total = max(xmax_total, x_vals.max())

            trace = go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                name=str(cat),
                legendgroup=str(cat),
                line=dict(width=2, color=colors[j % len(colors)]),
                showlegend=(i == 2),
                visible=visible
            )
            traces.append((trace, row, col))

        if np.isfinite(xmin_total) and np.isfinite(xmax_total):
            padding = 0.075 * (xmax_total - xmin_total)
            x_ranges[(row, col)] = [xmin_total - padding, xmax_total + padding]

    return traces, x_ranges

hist_train_traces, train_xranges = generate_histogram_traces(train, visible=True)
hist_data_traces, _ = generate_histogram_traces(data, visible=False)
kde_train_traces, _ = get_kde_traces(train, visible=False)
kde_data_traces, _ = get_kde_traces(data, visible=False)

fig = make_subplots(
    rows=n_rows, cols=n_cols,
    subplot_titles=num_att,
    horizontal_spacing=0.04,
    vertical_spacing=0.15
)

all_traces = hist_train_traces + hist_data_traces + kde_train_traces + kde_data_traces
for trace, row, col in all_traces:
    fig.add_trace(trace, row=row, col=col)

for i in range(len(num_att)):
    row = i // n_cols + 1
    col = i % n_cols + 1
    fig.update_xaxes(showline=True, linewidth=1, linecolor='grey', mirror=True, row=row, col=col)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='grey', mirror=True, row=row, col=col)

for (range_x, row, col) in train_xranges:
    fig.update_xaxes(range=range_x, row=row, col=col)

n_hist = len(hist_train_traces)
n_kde = len(kde_train_traces)

visibility = {
    "hist_train": [True]*n_hist + [False]*n_hist + [False]*n_kde + [False]*n_kde,
    "hist_data":  [False]*n_hist + [True]*n_hist + [False]*n_kde + [False]*n_kde,
    "kde_train":  [False]*n_hist + [False]*n_hist + [True]*n_kde + [False]*n_kde,
    "kde_data":   [False]*n_hist + [False]*n_hist + [False]*n_kde + [True]*n_kde,
}

current_mode = "hist_train"

fig.update_layout(
    barmode='stack',
    height=500,
    width=1150,
    plot_bgcolor='rgba(240,240,245,0.5)',
    paper_bgcolor='rgba(255,255,255,0)',
    margin=dict(t=60, l=40, r=20, b=40),
    legend=dict(title='Soil Type', x=1.01, y=1.01, font=dict(size=10), tracegroupgap=0),
    updatemenus=[
        {
            "buttons": [
                {
                    "label": "Hist – Train",
                    "method": "update",
                    "args": [{"visible": visibility["hist_train"]}]
                },
                {
                    "label": "KDE – Train",
                    "method": "update",
                    "args": [{"visible": visibility["kde_train"]}]
                },
                {
                    "label": "Hist – Original Data",
                    "method": "update",
                    "args": [{"visible": visibility["hist_data"]}]
                },
                {
                    "label": "KDE – Original Data",
                    "method": "update",
                    "args": [{"visible": visibility["kde_data"]}]
                }
            ],
            "type": "buttons",
            "direction": "down",
            "x": 1.17,
            "y": 0.36,
            "showactive": True,
            "font": {"size": 12},
            "borderwidth": 1,
            "pad": {"r": 5, "t": 5}
        }
    ]
)

fig.show()



n_cols, n_rows = 3, 2
colors = px.colors.qualitative.Plotly
categories = sorted(set(train['Crop Type']))

def generate_histogram_traces(df, visible=False):
    traces = []
    x_ranges = []

    for i, att in enumerate(num_att):
        bins = np.arange(df[att].min() - 0.5, df[att].max() + 1.5, 1)

        row = i // n_cols + 1
        col = i % n_cols + 1

        for j, cat in enumerate(categories):
            x_vals = df[df['Crop Type'] == cat][att]
            trace = go.Histogram(
                x=x_vals,
                xbins=dict(start=bins[0], end=bins[-1], size=1),
                name=str(cat),
                marker=dict(
                    color=colors[j % len(colors)],
                    line=dict(width=0.5, color='black')
                ),
                opacity=0.7,
                showlegend=(i == 2),
                legendgroup=str(cat),
                visible=visible
            )
            traces.append((trace, row, col))

            xmax, xmin = x_vals.max(), x_vals.min()
            range_x = [xmin - 0.075 * (xmax - xmin), xmax + 0.075 * (xmax - xmin)]
            x_ranges.append((range_x, row, col))

    return traces, x_ranges

def get_kde_traces(df, visible=False):
    traces = []
    x_ranges = {}

    for i, att in enumerate(num_att):
        row = i // n_cols + 1
        col = i % n_cols + 1
        xmin_total, xmax_total = np.inf, -np.inf

        for j, cat in enumerate(categories):
            subset = df[df['Crop Type'] == cat][att].dropna()
            if len(subset) < 2:
                continue

            fig_, ax = plt.subplots()
            sns.kdeplot(
                x=subset,
                cut=0,
                bw_adjust=1.5,
                fill=False,
                ax=ax,
                color=colors[j % len(colors)],
                linewidth=2
            )
            line = ax.lines[-1]
            x_vals = line.get_xdata()
            y_vals = line.get_ydata()
            plt.close(fig_)

            xmin_total = min(xmin_total, x_vals.min())
            xmax_total = max(xmax_total, x_vals.max())

            trace = go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                name=str(cat),
                legendgroup=str(cat),
                line=dict(width=2, color=colors[j % len(colors)]),
                showlegend=(i == 2),
                visible=visible
            )
            traces.append((trace, row, col))

        if np.isfinite(xmin_total) and np.isfinite(xmax_total):
            padding = 0.075 * (xmax_total - xmin_total)
            x_ranges[(row, col)] = [xmin_total - padding, xmax_total + padding]

    return traces, x_ranges

hist_train_traces, train_xranges = generate_histogram_traces(train, visible=True)
hist_data_traces, _ = generate_histogram_traces(data, visible=False)
kde_train_traces, _ = get_kde_traces(train, visible=False)
kde_data_traces, _ = get_kde_traces(data, visible=False)

fig = make_subplots(
    rows=n_rows, cols=n_cols,
    subplot_titles=num_att,
    horizontal_spacing=0.04,
    vertical_spacing=0.15
)

all_traces = hist_train_traces + hist_data_traces + kde_train_traces + kde_data_traces
for trace, row, col in all_traces:
    fig.add_trace(trace, row=row, col=col)

for i in range(len(num_att)):
    row = i // n_cols + 1
    col = i % n_cols + 1
    fig.update_xaxes(showline=True, linewidth=1, linecolor='grey', mirror=True, row=row, col=col)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='grey', mirror=True, row=row, col=col)

for (range_x, row, col) in train_xranges:
    fig.update_xaxes(range=range_x, row=row, col=col)

n_hist = len(hist_train_traces)
n_kde = len(kde_train_traces)

visibility = {
    "hist_train": [True]*n_hist + [False]*n_hist + [False]*n_kde + [False]*n_kde,
    "hist_data":  [False]*n_hist + [True]*n_hist + [False]*n_kde + [False]*n_kde,
    "kde_train":  [False]*n_hist + [False]*n_hist + [True]*n_kde + [False]*n_kde,
    "kde_data":   [False]*n_hist + [False]*n_hist + [False]*n_kde + [True]*n_kde,
}

current_mode = "hist_train"

fig.update_layout(
    barmode='stack',
    height=500,
    width=1150,
    plot_bgcolor='rgba(240,240,245,0.5)',
    paper_bgcolor='rgba(255,255,255,0)',
    margin=dict(t=60, l=40, r=20, b=40),
    legend=dict(title='Crop Type', x=1.01, y=1.01, font=dict(size=10), tracegroupgap=0),
    updatemenus=[
        {
            "buttons": [
                {
                    "label": "Hist – Train",
                    "method": "update",
                    "args": [{"visible": visibility["hist_train"]}]
                },
                {
                    "label": "KDE – Train",
                    "method": "update",
                    "args": [{"visible": visibility["kde_train"]}]
                },
                {
                    "label": "Hist – Original Data",
                    "method": "update",
                    "args": [{"visible": visibility["hist_data"]}]
                },
                {
                    "label": "KDE – Original Data",
                    "method": "update",
                    "args": [{"visible": visibility["kde_data"]}]
                }
            ],
            "type": "buttons",
            "direction": "down",
            "x": 1.17,
            "y": 0.36,
            "showactive": True,
            "font": {"size": 12},
            "borderwidth": 1,
            "pad": {"r": 5, "t": 5}
        }
    ]
)

fig.show()



def create_heatmap_frame(df, name, target):
    ct = pd.crosstab(
        [df["Soil Type"], df["Crop Type"]],
        df[target],
        normalize="index"
    ).T

    z = ct.values
    x = [f"{soil} - {crop}" for soil, crop in ct.columns]
    y = ct.index

    annotations = []
    z_min = z.min()
    z_max = z.max()
    threshold = (z_min + z_max) / 2

    for i, y_val in enumerate(y):
        for j, x_val in enumerate(ct.columns):
            annotations.append(
                dict(
                    x=f"{x_val[0]} - {x_val[1]}",
                    y=y_val,
                    text=f"{ct.iat[i, j]:.2f}",
                    showarrow=False,
                    font=dict(
                        color="black" if ct.iat[i, j] < threshold else "white",
                        size=9
                    ),
                    xanchor="center",
                    yanchor="middle"
                )
            )

    return {
        "data": [go.Heatmap(
            z=z,
            x=x,
            y=y,
            xgap=0.2, ygap=0.2,
            colorscale="YlGnBu",
            showscale=False
        )],
        "name": name,
        "layout": go.Layout(annotations=annotations)
    }

frame_train = create_heatmap_frame(train, "Train", target)
frame_data = create_heatmap_frame(data, "Data", target)

fig = go.Figure(
    data=frame_train["data"],
    layout=go.Layout(
        title="Fertilizer share",
        title_y=0.87,
        title_x=0.5,
        xaxis=dict(title="Soil Type - Crop Type"),
        yaxis=dict(title=target),
        width=1250,
        height=375,
        font=dict(size=10),
        margin=dict(l=20, r=0, t=70, b=0),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.995,
                y=1.25,
                showactive=True,
                buttons=[
                    dict(label="Train", method="animate", args=[["Train"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]),
                    dict(label="Original Data", method="animate", args=[["Data"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]),
                ],
                font_size=11,
            )
        ],
        annotations=frame_train["layout"].annotations
    ),
    frames=[go.Frame(**frame_train), go.Frame(**frame_data)]
)

fig.show()


# Cat × Cat interactions
RunBaselines(cat_features=cat_att, group_size=2)



df_train = train.sample(50_000, random_state=42)
df_data  = data.sample(50_000, random_state=42)

features   = num_att
target_col = target

classes = df_train[target_col].unique()
palette = px.colors.qualitative.Plotly
color_map = {c: palette[i % len(palette)] for i, c in enumerate(classes)}

n = len(features)
fig = make_subplots(
    rows=n-1, cols=n-1,
    shared_xaxes=True, shared_yaxes=True,
    horizontal_spacing=0, vertical_spacing=0,
    subplot_titles=features[1:]
)

def add_layer(df, visible: bool):
    start_index = len(fig.data)
    for r, y in enumerate(features[:-1],  start=1):
        for c, x in enumerate(features[1:], start=1):
            if r > c:
                continue
            for cls in classes:
                mask = df[target_col] == cls
                fig.add_trace(
                    go.Scattergl(
                        x=df.loc[mask, x], y=df.loc[mask, y],
                        mode="markers",
                        marker=dict(opacity=0.015, color=color_map[cls]),
                        legendgroup=str(cls),
                        showlegend=False,
                        visible=visible,
                        name=cls,
                    ),
                    row=r, col=c
                )
    return len(fig.data) - start_index

n_train = add_layer(df_train, visible=True)
n_data  = add_layer(df_data,  visible=False)

for cls in classes:
    fig.add_trace(
        go.Scattergl(
            x=[None], y=[None],
            mode="markers",
            marker=dict(color=color_map[cls], opacity=1),
            name=str(cls),
            legendgroup=str(cls),
            showlegend=True,
            visible=True
        ),
        row=1, col=1
    )
n_proxy = len(classes)

axis_style = dict(linewidth=1, linecolor="grey", mirror=True,
                  ticks="outside", showline=True, tickangle=0)

for r, y in enumerate(features[:-1],  start=1):
    for c, x in enumerate(features[1:], start=1):
        if r > c:
            continue
        rng_x = pd.concat([df_train[x], df_data[x]]).agg(['min', 'max']).values
        rng_y = pd.concat([df_train[y], df_data[y]]).agg(['min', 'max']).values

        fig.update_xaxes(range=rng_x, row=r, col=c, **axis_style)
        fig.update_yaxes(range=rng_y, row=r, col=c, **axis_style)

        ticks, title = '', None
        if r == c:
            fig.update_xaxes(showticklabels=True, row=r, col=c)
            fig.update_yaxes(showticklabels=True, row=r, col=c)
            ticks = 'outside'
            title = y
        if r == 1 and c == 1:
            fig.update_xaxes(matches=None, row=r, col=c)
            fig.update_yaxes(matches=None, row=r, col=c)
        fig.update_xaxes(linewidth=1, linecolor='grey',
                         mirror=True, ticks=ticks, showline=True, row=r, col=c)
        fig.update_yaxes(linewidth=1, linecolor='grey',
                         mirror=True, ticks=ticks, showline=True, row=r, col=c, title=title)

visible_train = [True]*n_train + [False]*n_data + [True]*n_proxy
visible_data  = [False]*n_train + [True]*n_data + [True]*n_proxy

fig.update_layout(
    height=700, width=925,
    plot_bgcolor="white",
    margin=dict(t=45, b=20, l=20, r=200),
    showlegend=True,
    legend_title_text=target_col,
    legend=dict(tracegroupgap=1, x=1.03, y=1),
    updatemenus=[{
        "buttons": [
            {"label": "Train", "method": "update",
             "args": [{"visible": visible_train}, {}]},
            {"label": "Original Data",  "method": "update",
             "args": [{"visible": visible_data},  {}]},
        ],
        "type": "buttons", "direction": "down",
        "x": 1.4, "y": 0.925,
        "showactive": True,
        "font": {"size": 12},
        "pad": {"r": 5, "t": 5},
        "borderwidth": 1
    }]
)

for annotation in fig.layout.annotations:
    annotation.font.size = 15
    annotation.y += 0.01

fig.show()


# Num × Num interactions
RunBaselines(num_features=num_att, group_size=2, bin_factor=1).sort_values('CV', ascending=False).head()


# Cat × Num interactions
RunBaselines(cat_features=cat_att, num_features=num_att, group_size=2, bin_factor=1).sort_values('CV', ascending=False).head()


# 3-feature interactions
groups = [('Soil Type', 'Crop Type', 'Phosphorous'),
          ('Soil Type', 'Crop Type', 'Moisture'),
          ('Soil Type', 'Crop Type', 'Nitrogen'),
          ('Crop Type', 'Moisture', 'Phosphorous'),
          ('Crop Type', 'Nitrogen', 'Phosphorous'),]

scores = pd.DataFrame(columns=['train', 'CV', 'std']).rename_axis('3-feature interactions', axis=1)

for group in groups:
    index = ' - '.join(group)
    model = GroupwiseClassifier(features=list(group), bin_factor=0.25)
    model.fit(X_train, y_train)
    
    tra_score = mean_average_precision_at_3(model, X_train, y_train)
    scores.loc[index, 'train'] = tra_score
    
    cv_scores = cross_val_score(model, X_train, y_train, scoring=mean_average_precision_at_3, cv=10, n_jobs=-1)
    scores.loc[index, 'CV' ] = cv_scores.mean()
    scores.loc[index, 'std'] = cv_scores.std()

scores.sort_values('CV', ascending=False).head()


# 4-feature interactions
groups = [('Soil Type', 'Crop Type', 'Potassium', 'Phosphorous'),
          ('Soil Type', 'Crop Type', 'Temparature', 'Phosphorous'),
          ('Soil Type', 'Crop Type', 'Temparature', 'Moisture'),
          ('Soil Type', 'Crop Type', 'Temparature', 'Potassium'),
          ('Soil Type', 'Crop Type', 'Nitrogen', 'Phosphorous')]

scores = pd.DataFrame(columns=['train', 'CV', 'std']).rename_axis('4-feature interactions', axis=1)

for group in groups:
    index = ' - '.join(group)
    model = GroupwiseClassifier(features=list(group), bin_factor=0.25)
    model.fit(X_train, y_train)
    
    tra_score = mean_average_precision_at_3(model, X_train, y_train)
    scores.loc[index, 'train'] = tra_score
    
    cv_scores = cross_val_score(model, X_train, y_train, scoring=mean_average_precision_at_3, cv=10, n_jobs=-1)
    scores.loc[index, 'CV' ] = cv_scores.mean()
    scores.loc[index, 'std'] = cv_scores.std()

scores.sort_values('CV', ascending=False).head()


import sklearn
sklearn.set_config(transform_output='pandas')

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgbm
import catboost


target = 'Fertilizer Name'
num_att = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
cat_att = ['Soil Type', 'Crop Type']


train, test = load_data(index_col='id')

X_train = train.drop(columns=target)
y_train = train[target]



from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder

class Top3LabelClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, encode_target=False):
        self.estimator = estimator
        self.encode_target = encode_target
    def fit(self, X, y, **kwargs):
        self.estimator_ = clone(self.estimator)
        if self.encode_target:
            self.label_encoder_ = LabelEncoder()
            y = self.label_encoder_.fit_transform(y)
        self.estimator_.fit(X, y, **kwargs)
        return self
    def predict(self, X):
        scores = self.estimator_.predict_proba(X)
        top3_scores = np.argsort(scores, axis=1)[:, -3:][:, ::-1]
        classes = self.label_encoder_.classes_ if self.encode_target else self.estimator_.classes_
        top3_classes = classes[top3_scores]
        y_pred = np.array([' '.join(map(str, row)) for row in top3_classes])
        return y_pred
    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


# Random forest
rf_model = Pipeline([
    ('1hot_encode', make_column_transformer((OneHotEncoder(sparse_output=False, dtype=int), cat_att),
                                            remainder='passthrough', verbose_feature_names_out=False)),
    ('model', Top3LabelClassifier(estimator=RandomForestClassifier(n_estimators=1000, max_depth=11, max_features=0.7,
                                                                   max_leaf_nodes=255, max_samples=0.9,
                                                                   random_state=42, n_jobs=-1)))
])
rf_model.fit(X_train, y_train)
print(f"Train score: {mean_average_precision_at_3(rf_model, X_train, y_train): .5f}")

cv_scores = cross_val_score(rf_model, X_train, y_train, scoring=mean_average_precision_at_3, cv=3, n_jobs=-1)
print(f"CV-score: {cv_scores.mean(): .5f}, std: {cv_scores.std(): .5f}")


# XGBoost
xgb_model = Pipeline([
    ('to_category', make_column_transformer((FunctionTransformer(lambda x: x.astype('category')), cat_att),
                                            remainder='passthrough', verbose_feature_names_out=False)),
    ('model', Top3LabelClassifier(xgb.XGBClassifier(n_estimators=1000, objective='mutli:softprob', subsample=0.8,
                                                    max_leaves=255, max_depth=11, max_bin=127,
                                                    learning_rate=0.1, eval_metric='mlogloss', colsample_bytree=0.8,
                                                    enable_categorical=True, n_jobs=-1, random_state=42), encode_target=True))
])

xgb_model.fit(X_train, y_train)
print(f"Train score: {mean_average_precision_at_3(xgb_model, X_train, y_train): .5f}")

cv_scores = cross_val_score(xgb_model, X_train, y_train, scoring=mean_average_precision_at_3, cv=3, n_jobs=-1)
print(f"CV-score: {cv_scores.mean(): .5f}, std: {cv_scores.std(): .5f}")


# CatBoost
catboost_model = Pipeline([
    ('1hot_encode', make_column_transformer((OneHotEncoder(sparse_output=False, dtype=int), cat_att),
                                            remainder='passthrough', verbose_feature_names_out=False)),
    ('model', Top3LabelClassifier(catboost.CatBoostClassifier(iterations=1000, depth=6, border_count=254,
                                                              rsm=1, learning_rate=0.1, random_seed=42,
                                                              thread_count=-1, verbose=False)))
])

catboost_model.fit(X_train, y_train)
print(f"Train score: {mean_average_precision_at_3(catboost_model, X_train, y_train): .5f}")

cv_scores = cross_val_score(catboost_model, X_train, y_train, scoring=mean_average_precision_at_3, cv=3, n_jobs=-1)
print(f"CV-score: {cv_scores.mean(): .5f}, std: {cv_scores.std(): .5f}")


# LightGBM
lgbm_model = Pipeline([
    ('to_category', make_column_transformer((FunctionTransformer(lambda x: x.astype('category')), cat_att),
                                            remainder='passthrough', verbose_feature_names_out=False)),
    ('model', Top3LabelClassifier(lgbm.LGBMClassifier(n_estimators=1000, max_leaves=255, max_depth=7,
                                                      max_bin=63, learning_rate=0.1, eval_metric='mlogloss',
                                                      subsample=0.9, colsample_bytree=0.8,
                                                      n_jobs=-1, random_state=42, verbose=-1)))
])
lgbm_model.fit(X_train, y_train)
print(f"Train score:{mean_average_precision_at_3(lgbm_model, X_train, y_train): .5f}")

cv_scores = cross_val_score(lgbm_model, X_train, y_train, scoring=mean_average_precision_at_3, cv=3, n_jobs=-1)
print(f"CV-score: {cv_scores.mean(): .5f}, std: {cv_scores.std(): .5f}")



import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, message='Falling back to prediction using DMatrix')


# XGBoost
preprocessor = make_column_transformer((FunctionTransformer(lambda x: x.astype('category')), cat_att),
                                       remainder='passthrough', verbose_feature_names_out=False)
preprocessor.fit(X_train)

label_encoder = LabelEncoder()
label_encoder.fit(y_train)

xgb_model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', Top3LabelClassifier(xgb.XGBClassifier(n_estimators=10000, early_stopping_rounds=50, tree_method='hist',
                                                    learning_rate=0.03, subsample=0.8, colsample_bytree=0.3,
                                                    max_bin=127, max_depth=7, objective='multi:softprob',
                                                    eval_metric='mlogloss', enable_categorical=True,
                                                    device='cuda', random_state=42, n_jobs=-1), encode_target=True))
])

cv_scores = np.empty(3)
splitter = StratifiedKFold(n_splits=3)

for i, (tra_idx, val_idx) in enumerate(splitter.split(X_train, y_train)):
    X_tra, X_val = X_train.iloc[tra_idx], X_train.iloc[val_idx]
    y_tra, y_val = y_train.iloc[tra_idx], y_train.iloc[val_idx]
    X_val_t, y_val_t = preprocessor.transform(X_val), label_encoder.transform(y_val)
    
    xgb_model.fit(
        X_tra, y_tra,
        model__eval_set=[(X_val_t, y_val_t)],
        model__verbose=True
    )
    cv_scores[i] = mean_average_precision_at_3(xgb_model, X_val, y_val)

print(f'CV-score:{cv_scores.mean(): .5f}, std:{cv_scores.std(): .5f}')


# XGBoost + Data augmentation
cv_scores = np.empty(3)
splitter = StratifiedKFold(n_splits=3)

for i, (tra_idx, val_idx) in enumerate(splitter.split(X_train, y_train)):
    X_tra, X_val = X_train.iloc[tra_idx], X_train.iloc[val_idx]
    y_tra, y_val = y_train.iloc[tra_idx], y_train.iloc[val_idx]
    X_val_t, y_val_t = preprocessor.transform(X_val), label_encoder.transform(y_val)
    
    # Augmentation
    for k in range(5):
        X_tra = pd.concat([X_tra, data.drop(columns=target)], axis=0, ignore_index=True)
        y_tra = pd.concat([y_tra, data[target]], axis=0, ignore_index=True)
    
    xgb_model.fit(
        X_tra, y_tra,
        model__eval_set=[(X_val_t, y_val_t)],
        model__verbose=False
    )
    cv_scores[i] = mean_average_precision_at_3(xgb_model, X_val, y_val)

print(f'CV-score:{cv_scores.mean(): .5f}, std:{cv_scores.std(): .5f}')


# XGBoost + FE(categorizing numerical attributes) + Data augmentation
test_probas = np.zeros((len(test), y_train.nunique()))

preprocessor = make_column_transformer((FunctionTransformer(lambda x: x.astype('category').add_suffix('_cat')), num_att),
                                       (FunctionTransformer(lambda x: x.astype('category')), cat_att),
                                       ('passthrough', num_att),
                                       remainder='passthrough', verbose_feature_names_out=False)
preprocessor.fit(X_train)

xgb_model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', Top3LabelClassifier(xgb.XGBClassifier(n_estimators=10000, early_stopping_rounds=50, tree_method='hist',
                                                    learning_rate=0.03, subsample=0.8, colsample_bytree=0.3,
                                                    max_bin=127, max_depth=7, objective='multi:softprob',
                                                    eval_metric='mlogloss', enable_categorical=True,
                                                    device='cuda', random_state=42, n_jobs=-1), encode_target=True))
])

cv_scores = np.empty(3)
splitter = StratifiedKFold(n_splits=3)

for i, (tra_idx, val_idx) in enumerate(splitter.split(X_train, y_train)):
    X_tra, X_val = X_train.iloc[tra_idx], X_train.iloc[val_idx]
    y_tra, y_val = y_train.iloc[tra_idx], y_train.iloc[val_idx]
    X_val_t, y_val_t = preprocessor.transform(X_val), label_encoder.transform(y_val)
    
    # Augmentation
    for k in range(5):
        X_tra = pd.concat([X_tra, data.drop(columns=target)], axis=0, ignore_index=True)
        y_tra = pd.concat([y_tra, data[target]], axis=0, ignore_index=True)
    
    xgb_model.fit(
        X_tra, y_tra,
        model__eval_set=[(X_val_t, y_val_t)],
        model__verbose=False
    )
    cv_scores[i] = mean_average_precision_at_3(xgb_model, X_val, y_val)
    # Test predictions
    test_probas += xgb_model.predict_proba(test)

print(f'CV-score:{cv_scores.mean(): .5f}, std: {cv_scores.std(): .5f}')


top3_scores = np.argsort(test_probas, axis=1)[:, -3:][:, ::-1]
classes = xgb_model['model'].label_encoder_.classes_
top3_classes = classes[top3_scores]
preds = np.array([' '.join(map(str, row)) for row in top3_classes])


y_test_pred = test.reset_index()[['id']].copy()
y_test_pred[target] = preds

y_test_pred.head()


y_test_pred.to_csv('submission.csv', index=False)

