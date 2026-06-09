%reset -f

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
import xgboost as xgb
from lifelines import NelsonAalenFitter
from lifelines.utils import concordance_index

warnings.simplefilter('ignore')


train = pd.read_csv('../input/equity-post-HCT-survival-predictions/train.csv'
                   ).set_index('ID')
train.index = train.index.astype('int32')


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


lb = pd.read_csv('../input/hct-leaderboard/equity-post-HCT-survival-predictions-publicleaderboard-2025-02-13.csv')

def lb_pct(score):
    return np.searchsorted(np.sort(lb['Score']), score) / len(lb['Score'])


X_raw = train.drop(columns=['efs', 'efs_time'])


X_float = X_raw.select_dtypes('float').astype('float32')
X_float = X_float.fillna(X_float.median())


naf = NelsonAalenFitter(label='y')
naf.fit(train['efs_time'], event_observed=train['efs'])
y = -train[['efs_time']].join(naf.cumulative_hazard_, on='efs_time')['y']


kfold = KFold(shuffle=True, random_state=1729)


def fit_predict(X_cat):
    X = pd.concat([X_float, X_cat], axis=1)
    m = xgb.XGBRegressor(enable_categorical=True)
    oof = np.zeros(len(train))
    for i_fold, i_oof in kfold.split(train.index):
        m.fit(X.iloc[i_fold], y.iloc[i_fold])
        oof[i_oof] = m.predict(X.iloc[i_oof])
    return oof


X_cat_raw = X_raw.select_dtypes('object').astype('category')

for col in X_cat_raw:
    X_cat_raw[col] = X_cat_raw[col].cat.add_categories('Missing').fillna('Missing')


oof = fit_predict(X_cat_raw)
score = calc_score(oof)
print(f'score: {score:.4f}, leaderboard percentile: {lb_pct(score):.4f}')
del oof, score


def encode_with_labels():
    X_cat = pd.DataFrame(index=train.index)
    for col in X_cat_raw:
        X_cat[col], _ = X_cat_raw[col].factorize(use_na_sentinel=False)
        X_cat[col] = X_cat[col].astype('int32').astype('category')
    return X_cat


oof = fit_predict(encode_with_labels())
score = calc_score(oof)
print(f'score: {score:.4f}, leaderboard percentile: {lb_pct(score):.4f}')
del oof, score


def encode_one_hot():
    X_cat = pd.DataFrame(index=train.index)
    for col in X_cat_raw:
        X_cat[col] = X_cat_raw[col].str.replace(r'[\[\]<]', '_', regex=True)
    return pd.get_dummies(X_cat)


oof = fit_predict(encode_one_hot())
score = calc_score(oof)
print(f'score: {score:.4f}, leaderboard percentile: {lb_pct(score):.4f}')
del oof, score


def encode_from_target(debug=False):
    X_cat = pd.DataFrame(index=train.index)

    for col in X_cat_raw:

        by_cat = pd.concat([
                X_cat_raw.assign(y=y).groupby(col)['y'].mean(),
                X_cat_raw[col].value_counts()],
            axis=1)

        by_cat['code'] = by_cat.apply(lambda row: 
                (row['count'] * row['y'] + by_cat['count'].median() * y.mean())
                    / (row['count'] + by_cat['count'].median()),
            axis=1)

        X_cat[col] = X_cat_raw[col].to_frame().join(by_cat['code'], on=col)['code']
        
        if debug and col == 'dri_score':
            print(f'{col}: m={int(by_cat["count"].median())}, overall_mean={y.mean():.4f}')
            display(by_cat.sort_index().head())

    return X_cat

encode_from_target(debug=True);


oof = fit_predict(encode_from_target())
score = calc_score(oof)
print(f'score: {score:.4f}, leaderboard percentile: {lb_pct(score):.4f}')
del oof, score


def encode_from_target_kfold(debug=False):
    X_cat = pd.DataFrame(index=train.index)
    
    for n_fold, (i_fold, i_oof) in enumerate(kfold.split(train.index)):
        oof_mean_y = y.iloc[i_oof].mean()

        for col in X_cat_raw:

            by_cat = pd.concat([
                    X_cat_raw.loc[i_fold, col].value_counts().rename('fold_count'),
                    X_cat_raw.loc[i_oof, col].value_counts().rename('oof_count'),
                    X_cat_raw.iloc[i_oof].assign(y=y.iloc[i_oof]).groupby(col)['y'].mean().rename('oof_mean_y')],
                axis=1)

            m = int(by_cat['oof_count'].median())

            by_cat['code'] = by_cat.apply(lambda row: 
                (row['fold_count'] * row['oof_mean_y'] + m * oof_mean_y)
                    / (row['fold_count'] + m),
            axis=1)

            X_cat[col] = X_cat_raw[col].to_frame().join(by_cat['code'], on=col)['code']
            
            if debug and n_fold == 0 and col == 'dri_score':
                print(f'{col}: m={m}, oof_mean_y={oof_mean_y:.4f}')
                display(by_cat.sort_index().head())

    return X_cat

encode_from_target_kfold(debug=True);


oof = fit_predict(encode_from_target_kfold())
score = calc_score(oof)
print(f'score: {score:.4f}, leaderboard percentile: {lb_pct(score):.4f}')
del oof, score


def encode_from_target_loo(debug=False):
    X_cat = pd.DataFrame(index=train.index)
    
    for col in X_cat_raw:
        fold_count = 1
        med_cat_count = int(X_cat_raw[col].value_counts().median())
        mean_y = y.mean()

        data = pd.DataFrame({
            'category': X_cat_raw[col],
            'y': y
        })

        cat_sums = data.groupby('category')['y'].sum()
        cat_counts = data['category'].value_counts()

        means = ((cat_sums[data['category']].values - data['y'].values) / 
                (cat_counts[data['category']].values - 1))
        
        X_cat[col] = (fold_count * means + med_cat_count * mean_y) / (fold_count + med_cat_count)
        
        if debug and col == 'dri_score':
            print(f'{col}: m={med_cat_count}, mean_y={mean_y:.4f}')
            print(X_cat.head())
            
    return X_cat

encode_from_target_loo(debug=True);


oof = fit_predict(encode_from_target_loo())
score = calc_score(oof)
print(f'score: {score:.4f}, leaderboard percentile: {lb_pct(score):.4f}')
del oof, score




