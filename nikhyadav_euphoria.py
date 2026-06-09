# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(f"train shape: {train.shape}")
print(f"test shape: {test.shape}")

missing_train = train.isnull().sum() / len(train) * 100
missing_test = test.isnull().sum() / len(test) * 100
print(missing_train)
print(missing_test)


from scipy.stats import ks_2samp

# distribution-shift
for col in train.columns:
    stat, p_value = ks_2samp(train[col].dropna(), test[col].dropna())
    print(f"{col}: KS={stat:.4f}, p={p_value:.4f}")



skewness = train.drop(['id', 'Y'], axis=1).skew().sort_values(ascending=False)

skewness.plot(kind='barh', figsize=(10, 6), color='coral')
plt.show()



corr = train.drop(['id', 'Y'], axis=1).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(14, 12))
sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, vmin=-1, vmax=1, square=True)
plt.show()


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
test_ids = test['id']


print("inf vals")
Xt = train.drop(['id', 'Y'], axis=1).copy()
Xs = test.drop(['id'], axis=1).copy()
y = train['Y'].map({True: 1, False: 0})
inf_rep = {}
for c in Xt.columns:
    cmax = Xt[c].replace([np.inf, -np.inf], np.nan).max()
    cmin = Xt[c].replace([np.inf, -np.inf], np.nan).min()
    if pd.notna(cmax) and pd.notna(cmin):
        Xt[c] = Xt[c].replace(np.inf, 10 * cmax)
        Xt[c] = Xt[c].replace(-np.inf, 10 * cmin)
        Xs[c] = Xs[c].replace(np.inf, 10 * cmax)
        Xs[c] = Xs[c].replace(-np.inf, 10 * cmin)
        inf_rep[c] = (10 * cmax, 10 * cmin)

Xt = Xt.drop(['x_3'], axis=1, errors='ignore')
Xs = Xs.drop(['x_3'], axis=1, errors='ignore')

Xt['x_9_m'] = Xt['x_9'].isna().astype(int)
Xs['x_9_m'] = Xs['x_9'].isna().astype(int)

shift_feats = ['x_6', 'x_8', 'x_9']
drop_feats = ['x_10', 'x_11', 'x_12', 'x_13', 'x_14']
Xt = Xt.drop(columns=drop_feats, errors='ignore')
Xs = Xs.drop(columns=drop_feats, errors='ignore')

print("imp")
Xt['x_5_m'] = Xt['x_5'].isna().astype(int)
Xs['x_5_m'] = Xs['x_5'].isna().astype(int)
imp = SimpleImputer(strategy='median')
cols = [c for c in Xt.columns if not c.endswith('_m')]
Xt[cols] = imp.fit_transform(Xt[cols])
Xs[cols] = imp.transform(Xs[cols])

if 'x_9' in Xt.columns:
    Xt['x_9'] = np.log1p(Xt['x_9'].abs())
    Xs['x_9'] = np.log1p(Xs['x_9'].abs())
    print("x_9")

safe = ['x_1', 'x_21', 'x_2', 'x_4']
cnt = 0
for i in range(len(safe)):
    for j in range(i+1, len(safe)):
        a = safe[i]
        b = safe[j]
        Xt[f'{a}_x_{b}'] = Xt[a] * Xt[b]
        Xs[f'{a}_x_{b}'] = Xs[a] * Xs[b]
        Xt[f'{a}_d_{b}'] = Xt[a] / (np.abs(Xt[b]) + 1e-8)
        Xs[f'{a}_d_{b}'] = Xs[a] / (np.abs(Xs[b]) + 1e-8)
        cnt += 2

Xt['t4_sum'] = Xt[safe].sum(axis=1)
Xt['t4_mean'] = Xt[safe].mean(axis=1)
Xt['t4_std'] = Xt[safe].std(axis=1)
Xt['t4_max'] = Xt[safe].max(axis=1)
Xt['t4_min'] = Xt[safe].min(axis=1)

Xs['t4_sum'] = Xs[safe].sum(axis=1)
Xs['t4_mean'] = Xs[safe].mean(axis=1)
Xs['t4_std'] = Xs[safe].std(axis=1)
Xs['t4_max'] = Xs[safe].max(axis=1)
Xs['t4_min'] = Xs[safe].min(axis=1)

print("qtf")
sc = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=42)
sc.fit(Xt)

Xt_s = pd.DataFrame(sc.transform(Xt), columns=Xt.columns, index=Xt.index)
Xs_s = pd.DataFrame(sc.transform(Xs), columns=Xs.columns, index=Xs.index)

print("feat sel")
sel = SelectKBest(mutual_info_classif, k=min(30, Xt_s.shape[1]))
sel.fit(Xt_s, y)

mi = pd.DataFrame({'f': Xt_s.columns, 'mi': sel.scores_}).sort_values('mi', ascending=False)
print(mi.head(15).to_string(index=False))

Xt_f = sel.transform(Xt_s)
Xs_f = sel.transform(Xs_s)
sel_feats = Xt_s.columns[sel.get_support()].tolist()



from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb


X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
    X_train_final, y_train, test_size=0.2, random_state=42, stratify=y_train
)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 12,
    'max_depth': 3,
    'learning_rate': 0.012,
    'n_estimators': 1500,
    'scale_pos_weight': 3.39,
    'subsample': 0.55,
    'subsample_freq': 2,
    'colsample_bytree': 0.55,
    'min_child_samples': 60,
    'reg_alpha': 10.0,
    'reg_lambda': 4.0,
    'min_gain_to_split': 0.04,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

model = lgb.LGBMClassifier(**params)
model.fit(
    X_train_split, y_train_split,
    eval_set=[(X_valid_split, y_valid_split)],
    callbacks=[lgb.early_stopping(100, verbose=True)]
)

print(f"best: {model.best_iteration_}")
print("done")

print("validation")
validation_predictions = model.predict_proba(X_valid_split)[:, 1]
validation_auc = roc_auc_score(y_valid_split, validation_predictions)
print(f"AUC: {validation_auc:.6f}")

print("test")
test_predictions = model.predict_proba(X_test_final)[:, 1]

submission = pd.DataFrame({'id': test_ids, 'prediction': test_predictions})
submission.to_csv('submission_gbm.csv', index=False)


