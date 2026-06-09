%reset -f

import warnings
from collections import namedtuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata
from sklearn.preprocessing import RobustScaler
from lifelines import CoxPHFitter, KaplanMeierFitter, NelsonAalenFitter
from lifelines.utils import concordance_index

warnings.simplefilter('ignore')


def score(solution, submission):
    for col in submission.columns:
        if not pd.api.types.is_numeric_dtype(submission[col]):
            raise
    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)
    merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)
    metric_list = []
    for race in merged_df_race_dict.keys():
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]
        c_index_race = concordance_index(
                        merged_df_race['efs_time'],
                        -merged_df_race['prediction'],
                        merged_df_race['efs'])
        metric_list.append(c_index_race)
    return float(np.mean(metric_list)-np.sqrt(np.var(metric_list)))


for efs_time,   efs,      pred in [
    [[1, 1],   [0, 1],   [1, 0],],
    [[1, 1],   [0, 1],   [0, 1],],
    [[1, 0],   [0, 1],   [1, 0],],
    [[0, 1],   [0, 1],   [1, 0],],
        ]:
    try:
        c_index = concordance_index(efs_time, pred, efs)
        print(f'{c_index:.3f}')
    except ZeroDivisionError as e:
        print(e)


train = pd.read_csv('../input/equity-post-HCT-survival-predictions/train.csv')

def plot_y_transformation(Y):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharey=True)
    sns.histplot(Y, y='y', hue='efs', ax=axes[0, 0])
    sns.scatterplot(Y, x='efs_time', y='y', hue='efs', ax=axes[0, 1])
    sns.scatterplot(Y[Y['efs'] == 0], x='efs_time', y='y', label='efs=0', ax=axes[1, 0])
    axes[1, 0].legend()
    sns.scatterplot(Y[Y['efs'] == 1], x='efs_time', y='y', label='efs=1', color=sns.color_palette()[1], ax=axes[1, 1])
    axes[1, 1].legend()
    plt.show()


Y = train[['efs_time', 'efs']]

Y['y'] = pd.concat(
        [
            Y['efs_time'],
            1 - Y['efs']
        ],
        axis=1
    ).apply(tuple, axis=1
    ).rank(ascending=False)

solution = train[['race_group', 'efs', 'efs_time']]
submission = Y[['y']].rename(columns={'y': 'prediction'})
print(score(solution, submission))

plot_y_transformation(Y)


Y = train[['efs_time', 'efs']]

Y['y'] = Y['efs_time'].values
mx = Y.loc[Y['efs'] == 1, 'y'].max()
mn = Y.loc[Y['efs'] == 0, 'y'].min()
Y.loc[Y['efs'] == 0, 'y'] = Y.loc[Y['efs'] == 0,'y'] + mx - mn
Y['y'] = Y['y'].rank()
Y.loc[Y['efs'] == 0, 'y'] += len(Y) // 2
Y['y'] = -Y['y'] / Y['y'].max()

solution = train[['race_group', 'efs', 'efs_time']]
submission = Y[['y']].rename(columns={'y': 'prediction'})
print(score(solution, submission))

plot_y_transformation(Y)


Y = train[['efs', 'efs_time']]

f = NelsonAalenFitter(label='y')
f.fit(Y['efs_time'], event_observed=Y['efs'])
Y = Y.join(f.cumulative_hazard_, on='efs_time')
Y['y'] = -Y['y']

solution = train[['race_group', 'efs', 'efs_time']]
submission = Y[['y']].rename(columns={'y': 'prediction'})
print(score(solution, submission))

plot_y_transformation(Y)


Y = train[['efs', 'efs_time']]

f = KaplanMeierFitter(label='y')
f.fit(Y['efs_time'], event_observed=Y['efs'])
Y = Y.join(f.survival_function_, on='efs_time')

solution = train[['race_group', 'efs', 'efs_time']]
submission = Y[['y']].rename(columns={'y': 'prediction'})
print(score(solution, submission))

plot_y_transformation(Y)


X = train.drop(columns=['ID', 'efs_time', 'efs'])
Xf = X.select_dtypes('float')
Xf = Xf.fillna(Xf.median())
Xc = X.select_dtypes('object')

for c in Xc.columns:
    Xc[c], _ = Xc[c].factorize(use_na_sentinel=False)
    Xc[c] = Xc[c].astype('category')

Xc = pd.get_dummies(Xc.select_dtypes('category'), drop_first=True)
X = pd.concat([Xf, Xc], axis=1)

Y = train[['efs', 'efs_time']]

f = CoxPHFitter()
f.fit(pd.concat([X, Y], axis=1), 'efs_time', 'efs')
Y['y'] = f.predict_partial_hazard(X)

solution = train[['race_group', 'efs', 'efs_time']]
submission = Y[['y']].rename(columns={'y': 'prediction'})
print(score(solution, submission))

plot_y_transformation(Y)


Y = train[['ID', 'efs', 'efs_time']]

f = KaplanMeierFitter()
f.fit(Y['efs_time'], event_observed=Y['efs'])
Y = Y.join(f.survival_function_, on='efs_time')

Ranked = namedtuple('Ranked', Y.columns.to_list() + ['rank'])
ranked = []
rank = 0

Yc = Y[Y['efs'] == 0].sort_values('efs_time')
Yc_rows = Yc.itertuples(index=False)
censored = next(Yc_rows)
censored_last_efs_time = -1

def add_censored():
    global rank, censored, censored_last_efs_time
    if censored_last_efs_time < censored.efs_time:
        censored_last_efs_time = censored.efs_time
        rank += 1
    ranked.append(Ranked(**censored._asdict(), rank=rank))
    try:
        censored = next(Yc_rows)
    except StopIteration:
        censored = None

Ye = Y[Y['efs'] == 1].sort_values('efs_time')
Ye_rows = Ye.itertuples(index=False)
event = next(Ye_rows)
event_last_efs_time = -1

def add_event():
    global rank, event, event_last_efs_time
    if event_last_efs_time < event.efs_time:
        event_last_efs_time = event.efs_time
        rank += 1
    ranked.append(Ranked(**event._asdict(), rank=rank))
    try:
        event = next(Ye_rows)
    except StopIteration:
        event = None

while True:
    if event is None and censored is None:
        break
    if event is None:
        add_censored()
        continue
    if censored is None:
        add_event()
        continue
    if event.efs_time <= censored.efs_time:
        add_event()
        continue
    # probability that censored had event before event['efs_time']
    p = (1 - event.KM_estimate / censored.KM_estimate)
    if 0.5 > p:
        add_event()
        continue
    add_censored()

Y = pd.DataFrame(ranked, columns=Ranked._fields)
Y['y'] = Y['rank'].rank(ascending=False)
Y = Y.sort_values('ID').reset_index(drop=True)

solution = train[['race_group', 'efs', 'efs_time']]
submission = Y[['y']].rename(columns={'y': 'prediction'})
print(score(solution, submission))

plot_y_transformation(Y)




