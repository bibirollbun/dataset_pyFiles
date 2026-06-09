%reset -f

import warnings; warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
import xgboost as xgb
from lifelines import NelsonAalenFitter
from lifelines.utils import concordance_index

train = pd.read_csv('../input/equity-post-HCT-survival-predictions/train.csv').set_index('ID')
test = pd.read_csv('../input/equity-post-HCT-survival-predictions/test.csv').set_index('ID')

X_ = pd.concat([train.drop(columns=['efs', 'efs_time']), test])
Xf = X_.select_dtypes('float')
Xc = X_.select_dtypes('object').astype('category')
for col in Xc:
    if Xc[col].isna().any():
        Xc[col] = Xc[col].cat.add_categories('Missing').fillna('Missing')
X_ = pd.concat([Xf, Xc], axis=1)

naf = NelsonAalenFitter(label='y')
naf.fit(train['efs_time'], event_observed=train['efs'])
y = -train[['efs_time']].join(naf.cumulative_hazard_, on='efs_time')['y']

train = pd.concat([train[['efs', 'efs_time']], X_[:len(train)], y], axis=1)
test = X_[len(train):]


def encode_with_leave_out_one(Xc, train, y, i_fold):
    
    # these are local to the function and shadow globals
    # which isn't the prettiest, but it kept me from rewriting and introducing new bugs
    Xc = Xc.iloc[i_fold]
    train = train.iloc[i_fold]
    y = y.iloc[i_fold]
    
    fold_cat_rows = 1
    
    oof_mean_y_by_cat_test = {}
    for col in Xc:
        oof_mean_y_by_cat_test[col] = train.groupby(col)['y'].mean()
    
    oof_mean_y_by_cat_train = {}
    for col in Xc:
        y_sums = train.groupby(col)['y'].sum().to_dict()
        y_counts = train.groupby(col)['y'].count().to_dict()
        for cat in train[col].cat.categories:
            if col == 'dri_score' and cat.startswith('High'):
                print(f'{col}, "{cat}", sum={y_sums[cat]}, count={y_counts[cat]}, oof_mean_y={((y_sums[cat] - train["y"]) / (y_counts[cat] - 1)).iloc[0]}')
                print(f'{(y_sums[cat] - train["y"]) / (y_counts[cat] - 1)}')
                print()
            oof_mean_y_by_cat_train[(col, cat)] = (y_sums[cat] - train['y']) / (y_counts[cat] - 1)
    
    med_cat_rows = {}
    for col in Xc:
        med_cat_rows[col] = int(Xc[col].value_counts().median())
    
    oof_mean_y_test = y.mean()
    
    oof_mean_y_train = {}
    y_sum = y.sum()
    for col in Xc:
        oof_mean_y_train[col] = (y_sum - train['y']) / (len(train) - 1)
    
    n = fold_cat_rows
    m = lambda col: med_cat_rows[col] / len(Xc) * 10
    Xc_loo = pd.DataFrame(index=Xc.index)
    for col in Xc:
        for cat in train[col].cat.categories:
            loo = (n * oof_mean_y_by_cat_train[(col, cat)] + m(col) * oof_mean_y_train[col]) / (n + m(col))
            Xc_loo.loc[Xc.index[Xc[col] == cat], col] = loo
        loo = (n * oof_mean_y_by_cat_test[col] + m(col) * oof_mean_y_test) / (n + m(col))
        Xc_loo.loc[len(train):, col] = Xc.join(loo, on=col)['y']
    return Xc_loo


def calc_score(oof):
    merged_df = train[['race_group', 'efs_time', 'efs']].assign(prediction=oof)
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


kfold = KFold(shuffle=True, random_state=1729)
m = xgb.XGBRegressor(enable_categorical=True)
oof = np.zeros(len(train))

for fold_n, (i_fold, i_oof) in enumerate(kfold.split(train.index)):
    print(f'fold: {fold_n}')
    Xc_loo = encode_with_leave_out_one(Xc, train, y, i_fold)
    X = pd.concat([Xf, Xc_loo], axis=1)
    m.fit(X.iloc[i_fold], y.iloc[i_fold])
    oof[i_oof] = m.predict(X.iloc[i_oof])

score = calc_score(oof)
print(f'score: {score:.4f}')


submission = pd.DataFrame(m.predict(X[len(train):]), index=test.index, columns=['prediction'])
submission.to_csv('submission.csv')




