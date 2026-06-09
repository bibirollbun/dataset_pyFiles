# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
# /kaggle/input/cat-in-the-dat-ii/sample_submission.csv
# /kaggle/input/cat-in-the-dat-ii/train.csv
# /kaggle/input/cat-in-the-dat-ii/test.csv

import os
import numpy as np
import polars as pl
from sklearn import linear_model, metrics, preprocessing, model_selection


df = pl.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')
df.shape, df.head(5)


df = df.with_columns(pl.lit(-1).alias('kfold'))
df = df.sample(fraction=1.0, shuffle=True)
n = df.height
y = df.get_column('target').to_numpy()
kf = model_selection.StratifiedKFold(n_splits=5)
kfold_arr = np.full(n, -1, dtype=np.int32)
for fold, (tr_idx, va_idx) in enumerate(kf.split(X=np.zeros((n, 1)), y=y)):
    kfold_arr[va_idx] = fold
df = df.with_columns(pl.Series('kfold', kfold_arr))
df.shape, df.head(3)


def make_ohe():
    try:
        return preprocessing.OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    except TypeError:
        return preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=True)

# def run_fold_polars(dfp: pl.DataFrame, fold: int):
#     features = [c for c in dfp.columns if c not in ('id','target','kfold')]

#     d = dfp.with_columns([pl.col(c).cast(pl.Utf8).fill_null('NONE').alias(c) for c in features])

#     tr = d.filter(pl.col('kfold') != fold)
#     va = d.filter(pl.col('kfold') == fold)

#     full_feat = pl.concat([tr.select(features), va.select(features)], how='vertical')
#     ohe = make_ohe()
#     ohe.fit(full_feat.to_numpy())

#     X_tr = ohe.transform(tr.select(features).to_numpy())
#     X_va = ohe.transform(va.select(features).to_numpy())
#     y_tr = tr.select('target').to_numpy().ravel()
#     y_va = va.select('target').to_numpy().ravel()

#     model = linear_model.LogisticRegression(max_iter=3000)
#     model.fit(X_tr, y_tr)
#     va_pred = model.predict_proba(X_va)[:, 1]
#     auc = metrics.roc_auc_score(y_va, va_pred)
#     print(f'Fold = {fold}, AUC = {auc}')
#     return auc

# folds = sorted(pl.Series(df.get_column('kfold')).unique().to_list())
# auc_scores = [run_fold_polars(df, f) for f in folds]
# print('CV mean AUC:', float(np.mean(auc_scores)))
# print('CV std AUC:', float(np.std(auc_scores)))


train_df = pl.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')
test_df = pl.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')
features = [c for c in train_df.columns if c not in ('id','target')]

train_df = train_df.with_columns([pl.col(c).cast(pl.Utf8).fill_null('NONE').alias(c) for c in features])
test_df = test_df.with_columns([pl.col(c).cast(pl.Utf8).fill_null('NONE').alias(c) for c in features])

ohe = make_ohe()
full_feat = pl.concat([train_df.select(features), test_df.select(features)], how='vertical')
ohe.fit(full_feat.to_numpy())

X_tr = ohe.transform(train_df.select(features).to_numpy())
X_te = ohe.transform(test_df.select(features).to_numpy())
y_tr = train_df.select('target').to_numpy().ravel()

final_model = linear_model.LogisticRegression(max_iter=3000)
final_model.fit(X_tr, y_tr)
test_pred = final_model.predict_proba(X_te)[:, 1]

sub = pl.DataFrame({
    'id': test_df.get_column('id'),
    'target': test_pred
})

sub.write_csv('ohe_logreg_submission.csv')


sub.shape, sub.head(10)

