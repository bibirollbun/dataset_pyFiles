%%capture
!pip install lifelines


%reset -f

import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('../input/equity-post-HCT-survival-predictions/train.csv'
                   ).set_index('ID')

train["y"] = train.efs_time.values
mx = train.loc[train.efs==1,"efs_time"].max()
mn = train.loc[train.efs==0,"efs_time"].min()
train.loc[train.efs==0,"y"] = train.loc[train.efs==0,"y"] + mx - mn
train.y = train.y.rank()
train.loc[train.efs==0,"y"] += len(train)//2
train.y = train.y / train.y.max()

plt.hist(train.loc[train.efs==1,"y"],bins=100,label="efs=1", color='darkorange')
plt.hist(train.loc[train.efs==0,"y"],bins=100,label="efs=0", color='saddlebrown')
plt.xlabel("y")
plt.ylabel("density")
plt.legend()
plt.show()

def plot(efs, color,logx=False,  ax=None):
    return train.loc[train['efs'] == efs, ['efs_time', 'y']].sort_values('y'
            ).plot.scatter('y', 'efs_time', label=f'efs={efs}', logx=logx, color=color, ax=ax)

ax = plot(1, 'darkorange')
_ = plot(0, 'saddlebrown', ax=ax)


import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
import lightgbm as lgb
from lifelines.utils import concordance_index

# Competition score function
# https://www.kaggle.com/code/metric/eefs-concordance-index
# changed to not mutate the passed in dataframes by deleteing the ID column
# Instead I .set_index('ID') before passing in
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

Xt = train.drop(columns=['efs', 'efs_time', 'y'])
Xf = Xt.select_dtypes('float')
Xc = Xt.select_dtypes('object').astype('category')
X = pd.concat([Xf, Xc], axis=1)

y = train.y

scores = []

for i, (fit_i, pred_i) in enumerate(KFold(shuffle=True).split(X)):
    fit_d = lgb.Dataset(X.iloc[fit_i], y.iloc[fit_i])
    params = {'verbosity': -1}
    m = lgb.train(params, fit_d)
    solution = train.iloc[pred_i][['race_group', 'efs', 'efs_time']]
    submission = pd.DataFrame(-m.predict(X.iloc[pred_i]), index=pred_i, columns=['prediction'])
    scores.append(score(solution, submission))

print(sum(scores) / len(scores))


%reset -f

import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('../input/equity-post-HCT-survival-predictions/train.csv'
                   ).set_index('ID')

train["y"] = -train.efs_time.values
mx = train.loc[train.efs==1,"y"].max()
mn = train.loc[train.efs==0,"y"].min()
train.loc[train.efs==0,"y"] = train.loc[train.efs==0,"y"] + mx - mn
train.y = train.y.rank()
train.loc[train.efs==0,"y"] += len(train)//2
train.y = train.y / train.y.max()

plt.hist(train.loc[train.efs==1,"y"],bins=100,label="efs=1", color='olive')
plt.hist(train.loc[train.efs==0,"y"],bins=100,label="efs=0", color='slategray')
plt.xlabel("y")
plt.ylabel("density")
plt.legend()

def plot(efs, color,logx=False,  ax=None):
    return train.loc[train['efs'] == efs, ['efs_time', 'y']].sort_values('y'
            ).plot.scatter('y', 'efs_time', label=f'efs={efs}', logx=logx, color=color, ax=ax)

ax = plot(1, 'olive')
_ = plot(0, 'slategray', ax=ax)


import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
import lightgbm as lgb
from lifelines.utils import concordance_index

# Competition score function
# https://www.kaggle.com/code/metric/eefs-concordance-index
# changed to not mutate the passed in dataframes by deleteing the ID column
# Instead I .set_index('ID') before passing in
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

Xt = train.drop(columns=['efs', 'efs_time', 'y'])
Xf = Xt.select_dtypes('float')
Xc = Xt.select_dtypes('object').astype('category')
X = pd.concat([Xf, Xc], axis=1)

y = train.y

scores = []

for i, (fit_i, pred_i) in enumerate(KFold(shuffle=True).split(X)):
    fit_d = lgb.Dataset(X.iloc[fit_i], y.iloc[fit_i])
    params = {'verbosity': -1}
    m = lgb.train(params, fit_d)
    solution = train.iloc[pred_i][['race_group', 'efs', 'efs_time']]
    submission = pd.DataFrame(-m.predict(X.iloc[pred_i]), index=pred_i, columns=['prediction'])
    scores.append(score(solution, submission))

print(sum(scores) / len(scores))




