%%capture
!pip install lifelines


from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit
from lifelines.utils import concordance_index


def score(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    event_label = 'efs'
    interval_label = 'efs_time'
    prediction_label = 'prediction'
    for col in submission.columns:
        if not pd.api.types.is_numeric_dtype(submission[col]):
            raise Exception(f'Submission column {col} must be a number')
    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)
    merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)
    metric_list = []
    for race in merged_df_race_dict.keys():
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]
        c_index_race = concordance_index(
                        merged_df_race[interval_label],
                        -merged_df_race[prediction_label],
                        merged_df_race[event_label])
        metric_list.append(c_index_race)
    return float(np.mean(metric_list)-np.sqrt(np.var(metric_list)))


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


class Model:
    def __init__(self, name, alpha=1.0):
        self.name = name
        X = train.drop(columns=['efs', 'efs_time'])
        self.X = X.fillna(X.median())
        self.y = train['efs']
        self.m = Ridge(alpha)

    def fit(self, fit_i):
        self.m.fit(self.X.iloc[fit_i], self.y.iloc[fit_i])

    def predict(self, pred_i):
        return pd.DataFrame(self.m.predict(self.X.iloc[pred_i]),
                            index=pred_i,
                            columns=['prediction'])


models = [
    Model('alpha=1.0'),
    Model('alpha=138.9', 138.9),    
]


scores_by_model = defaultdict(list)

for fit_i, pred_i in ShuffleSplit().split(train):
    solution = train_csv[['ID', 'race_group', 'efs', 'efs_time']].set_index('ID').iloc[pred_i]
    
    for m in models:
        m.fit(fit_i)
        submission = m.predict(pred_i)
        scores_by_model[m.name].append(score(solution, submission))

leaderboard = []

for model_name, scores in scores_by_model.items():
    leaderboard.append((sum(scores) / len(scores), model_name))

for mean_score, model_name in sorted(leaderboard, reverse=True):
    print(f'{round(mean_score, 6):.5f} {model_name}')




