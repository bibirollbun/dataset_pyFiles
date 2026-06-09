import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


import kagglehub
metric = kagglehub.package_import('jiazhuang/cmi-2025-metric')


def get_competition_score(true, pred):
    assert len(true) == len(pred)
    N = len(true)
    true = pd.DataFrame({'id': range(N), 'gesture': true})
    pred = pd.DataFrame({'id': range(N), 'gesture': pred})
    return metric.score(true, pred, 'id')


DATA_ROOT = '/kaggle/input/cmi-detect-behavior-with-sensor-data/'

train_df = pd.read_csv(f'{DATA_ROOT}/train.csv')
train_demo_df = pd.read_csv(f'{DATA_ROOT}/train_demographics.csv')


def q25(x):
    return x.quantile(0.25)

def q75(x):
    return x.quantile(0.75)

def kurt(x):
    return x.kurt()

agg_funs = ['mean', 'std', 'min', 'max', q25, q75, 'skew', kurt]
seq_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']


agg_train_df = train_df.groupby(['sequence_id', 'subject']).agg({c: agg_funs for c in seq_cols})

agg_train_df.columns = [x[0] + '_' + x[1] for x in agg_train_df.columns]
agg_train_df.reset_index(inplace=True)


agg_train_df = pd.merge(
    agg_train_df,
    train_demo_df,
    on='subject',
    how='left'
)


agg_train_df['label'] = agg_train_df.sequence_id.map(train_df.groupby('sequence_id').gesture.apply(lambda x: x.iloc[0]))


agg_train_df.head()


feat_cols = [c for c in agg_train_df.columns if c not in {'sequence_id', 'subject', 'label'}]


pd.Series(feat_cols).to_csv('feat_cols.txt', index=False, header=False)


out = pd.read_csv('feat_cols.txt', names=['feat']).feat.tolist()


out == feat_cols


import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
import joblib


label_encoder = LabelEncoder()
label_encoder.fit(agg_train_df.label)
agg_train_df['label_code'] = label_encoder.transform(agg_train_df.label)


joblib.dump(label_encoder, 'label_encoder.joblib')


params = {
    'objective': 'multiclass',
    'num_class': 18,
    'metric': ['multi_logloss', 'multi_error'],
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'subsample_freq': 1,
    'verbose': -1,
}

oof = np.zeros(agg_train_df.shape[0], dtype=np.int32)
kfold = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (trn_idx, val_idx) in enumerate(kfold.split(agg_train_df.index, y=agg_train_df.label_code, groups=agg_train_df.subject)):
    trn_data = lgb.Dataset(agg_train_df.loc[trn_idx, feat_cols], agg_train_df.loc[trn_idx, 'label_code'])
    val_data = lgb.Dataset(agg_train_df.loc[val_idx, feat_cols], agg_train_df.loc[val_idx, 'label_code'])

    clf = lgb.train(params, trn_data, 10000, valid_sets=[trn_data, val_data], callbacks=[lgb.log_evaluation(100), lgb.early_stopping(400)])
    clf.save_model(f'lgb_model_fold{fold}.txt')

    val_pred = clf.predict(agg_train_df.loc[val_idx, feat_cols]).argmax(axis=-1)
    val_true = agg_train_df.loc[val_idx, 'label_code'].values
    print(f'Fold {fold} Cmpetition Score:', get_competition_score(label_encoder.inverse_transform(val_true), label_encoder.inverse_transform(val_pred)))

    oof[val_idx] = val_pred


oof_score = get_competition_score(
    agg_train_df.label, label_encoder.inverse_transform(oof)
)


print('5 Fold CV:', oof_score)




