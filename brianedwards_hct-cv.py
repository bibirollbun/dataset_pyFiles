%%capture
!pip install lifelines


import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import make_scorer
from lifelines.utils import concordance_index

train_csv = pd.read_csv('../input/equity-post-HCT-survival-predictions/train.csv')
train = train_csv.set_index('ID')

for col in train.columns:
    if col in ['efs', 'efs_time']:
        continue
    if train[col].dtype != 'object':
        train[col] = train[col].fillna(train[col].median())
        continue
    train[col] = train[col].fillna('unknown')
    mean_efs_by_cat = train[[col, 'efs']].set_index(col).squeeze().groupby(col).mean()
    train[col] = train[col].map({category: code for code, category in enumerate(mean_efs_by_cat.sort_values().index)})

X = train.drop(columns=['efs', 'efs_time'])
X = X.fillna(X.median())
y = train['efs']
solution = train[['race_group', 'efs', 'efs_time']]

def score(y, y_pred, **kwargs):
    submission = pd.DataFrame(y_pred, index=train.index, columns=['prediction'])
    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)
    merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)
    metric_list = []
    for race in merged_df_race_dict.keys():
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]
        c_index_race = concordance_index(merged_df_race['efs_time'],
                                         -merged_df_race['prediction'],
                                         merged_df_race['efs'])
        metric_list.append(c_index_race)
    return float(np.mean(metric_list)-np.sqrt(np.var(metric_list)))

cv = RidgeCV(alphas=np.logspace(-3, 3), scoring=make_scorer(score))
cv.fit(X, y)
print(cv.alpha_)




