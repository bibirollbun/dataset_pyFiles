import warnings
warnings.simplefilter('ignore')
import os
import gc
import time
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
import polars as pl
from tqdm import tqdm
from sklearn.model_selection import KFold
import lightgbm as lgb
from lightgbm.callback import early_stopping


%%time

train = pl.read_csv('/kaggle/input/user-retention-prediction/train.csv')
display(train)


%%time

train = train.with_columns(
    pl.from_epoch(pl.col('Timestamp') + 8*3600, time_unit='s').alias('datetime')
).with_columns(pl.col('datetime').dt.date().alias('date'))

train_df = train.filter(pl.col('date') <= pl.date(2018, 10, 13))
train_ids = train_df.filter(pl.col('date') == pl.date(2018, 10, 13)).select(pl.col('ID')).unique()

label_df = train.filter(
    pl.col('date') >= pl.date(2018, 10, 14)
).group_by(['ID']).agg(pl.col('date').n_unique().alias('label'))

label_df = train_ids.join(label_df, on='ID', how='left').with_columns(pl.col('label').fill_null(0))

display(label_df)
display(label_df['label'].value_counts(sort=True, normalize=True))


%%time

df_train = label_df.join(
    train_df.group_by('ID').agg(
        pl.col('ActionId').count().alias('action_count'),
        pl.col('ActionId').n_unique().alias('action_nunique'),
        pl.col('ActionType').n_unique().alias('actiontype_nunique'),
        pl.col('date').n_unique().alias('active_days'),
        pl.col('ActionId').last().alias('last_action')
    ), on='ID', how='left'
).join(
    train_df.group_by(['ID', 'date']).agg(
        (pl.col('Timestamp').max() - pl.col('Timestamp').min()).alias('time_span_daily')
    ).group_by('ID').agg(
        pl.col('time_span_daily').mean().alias('mean_time_span_daily'),
        pl.col('time_span_daily').max().alias('max_time_span_daily'),
        pl.col('time_span_daily').min().alias('min_time_span_daily'),
        pl.col('time_span_daily').std().alias('std_time_span_daily'),
    ), on='ID', how='left'
).join(
    train_df.filter(
        pl.col('date').is_between(pl.date(2018, 10, 12)-pl.duration(days=6), pl.date(2018, 10, 12))
    ).group_by('ID').agg(
        pl.col('date').n_unique().alias('lastweek_active_days')
    ), on='ID', how='left'
).with_columns(
    pl.col('lastweek_active_days').fill_null(0),
    (pl.col('action_count') / pl.col('active_days')).alias('action_per_day')
)

display(df_train)


%%time

df_train = df_train.to_pandas()
feature_names = [c for c in df_train.columns if c not in ['ID', 'label']]

kf = KFold(n_splits=5, random_state=42, shuffle=True)
models = []
oof_pred = np.zeros(len(df_train))
for i, (train_index, valid_index) in enumerate(kf.split(df_train, df_train['label'])):
    print(f'Fold {i} ...')
    x_valid = df_train.loc[valid_index, feature_names]
    y_valid = df_train.loc[valid_index, 'label']
    x_train = df_train.loc[train_index, feature_names]
    y_train = df_train.loc[train_index, 'label']
    
    model = lgb.LGBMRegressor(
        objective='regression',
        max_depth=8, 
        num_leaves=64,
        min_child_samples=64,
        n_estimators=1000,
        learning_rate=0.05, 
        verbose=-1,
        importance_type='gain'
    )
    model.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        callbacks=[early_stopping(stopping_rounds=100)]
    )
    oof_pred[valid_index] = model.predict(x_valid)
    models.append(model)
    del model; gc.collect()


feature_importances = np.mean([m.feature_importances_ for m in models], axis=0)
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

display(importance_df)


def smape(y_true, y_pred):
    scores = []
    for t, p in zip(y_true, y_pred):
        if t == 0 and p == 0:
            scores.append(0)
        else:
            scores.append(2 * abs(t - p) / (abs(t) + abs(p)))
    return 100 * np.mean(scores)


print('LastWeek:', smape(df_train['label'], df_train['lastweek_active_days']))
print('LGBMPred:', smape(df_train['label'], oof_pred))


ratios = np.array([0.165077, 0.071544, 0.061548, 0.055787, 0.053056, 0.058299, 0.083344, 0.451345])
cumulative_ratios = np.cumsum(ratios[:-1])
quantiles = np.quantile(oof_pred, cumulative_ratios)
bins = np.concatenate([[oof_pred.min()], quantiles, [oof_pred.max()]])
oof_pred_pp = np.digitize(oof_pred, bins, right=True) - 1

df_train['pred'] = oof_pred_pp
print('LGBMPred:', smape(df_train['label'], df_train['pred']))


%%time

test_df = train.filter(
    pl.col('date') >= pl.date(2018, 9, 28)
)

df_test = train.filter(pl.col('date') == pl.date(2018, 10, 20)).select(pl.col('ID')).unique().join(
    test_df.group_by('ID').agg(
        pl.col('ActionId').count().alias('action_count'),
        pl.col('ActionId').n_unique().alias('action_nunique'),
        pl.col('ActionType').n_unique().alias('actiontype_nunique'),
        pl.col('date').n_unique().alias('active_days'),
        pl.col('ActionId').last().alias('last_action')
    ), on='ID', how='left'
).join(
    test_df.group_by(['ID', 'date']).agg(
        (pl.col('Timestamp').max() - pl.col('Timestamp').min()).alias('time_span_daily')
    ).group_by('ID').agg(
        pl.col('time_span_daily').mean().alias('mean_time_span_daily'),
        pl.col('time_span_daily').max().alias('max_time_span_daily'),
        pl.col('time_span_daily').min().alias('min_time_span_daily'),
        pl.col('time_span_daily').std().alias('std_time_span_daily'),
    ), on='ID', how='left'
).join(
    test_df.filter(
        pl.col('date').is_between(pl.date(2018, 10, 19)-pl.duration(days=6), pl.date(2018, 10, 19))
    ).group_by('ID').agg(
        pl.col('date').n_unique().alias('lastweek_active_days')
    ), on='ID', how='left'
).with_columns(
    pl.col('lastweek_active_days').fill_null(0),
    (pl.col('action_count') / pl.col('active_days')).alias('action_per_day')
)

display(df_test)


%%time

df_test = df_test.to_pandas()

pred_test = np.zeros(len(df_test))
for model in models:
    pred_test += model.predict(df_test[feature_names]) / kf.n_splits


ratios = np.array([0.113103, 0.065773, 0.047192, 0.055159, 0.054760, 0.076450, 0.121039, 0.466478])
cumulative_ratios = np.cumsum(ratios[:-1])
quantiles = np.quantile(pred_test, cumulative_ratios)
bins = np.concatenate([[pred_test.min()], quantiles, [pred_test.max()]])
pred_test_pp = np.digitize(pred_test, bins, right=True) - 1

df_test['pred'] = pred_test_pp
df_test[['ID', 'pred']].to_csv('submission.csv', index=False)

