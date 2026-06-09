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


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
orig = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


orig.rename(columns={'Diabetes_binary': 'diagnosed_diabetes'}, inplace=True)
orig['id'] = orig.index
orig = orig[train.columns.to_list()]


train = train.merge(orig, how='outer')


print(f'Train shape: {train.shape}')
print(f'Test shape: {test.shape}')
print(f'Orig shape: {orig.shape}')


cat_cols = train.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')


all_data = pd.concat([train, test, orig], ignore_index=True)
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    le.fit(all_data[col].astype(str))
    encoders[col] = le
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    orig[col] = le.transform(orig[col].astype(str))
train = pd.concat([train, orig], ignore_index=True)  # Now dtype-aligned


train.drop_duplicates(inplace=True)
print('Nulls total:', train.isnull().sum().sum())
print('Duplicates:', len(train) - train.drop_duplicates().shape[0])
print('Target balance:', train['diagnosed_diabetes'].value_counts(normalize=True))


num_cols = train.select_dtypes(include=[np.number]).columns.drop(['id', 'diagnosed_diabetes'])
ks_results = {  
                col: ks_2samp(train[col].dropna(), 
                orig[col].dropna()) 
                for col in ['age', 'bmi', 'systolic_bp']
             }


ks_results


pd.DataFrame({k: [v.statistic, v.pvalue] for k,v in ks_results.items()}, index=['KS_stat','p_value']).T.round(4)


print('\nDiabetes rate by gender:\n')
pd.crosstab(train['gender'], train['diagnosed_diabetes'], normalize='index')


target_corr = train[num_cols].corrwith(train['diagnosed_diabetes']).abs().sort_values(ascending=False)
print('\nTop |corr| with target:\n', target_corr)


plt.figure(figsize=(8, 6))
corr_mat = train[num_cols[:8]].corr()
sns.heatmap(corr_mat, annot=True, cmap='RdBu_r', center=0)
plt.title("Correlation Heatmap (Numerical Features)")
plt.show()


plt.figure(figsize=(7, 5))
sns.histplot(data=train, x='bmi', hue='diagnosed_diabetes', kde=True, bins=30)
plt.title("BMI Distribution by Diabetes")
plt.show()


plt.figure(figsize=(7, 5))
sns.boxplot(data=train, x='diagnosed_diabetes', y='age')
plt.title("Age Distribution by Diabetes")
plt.show()


plt.figure(figsize=(7, 5))
ctab = pd.crosstab(train['gender'], train['diagnosed_diabetes'], normalize='index') * 100
ctab.plot(kind='bar')
plt.title("Diabetes Percentage by Gender")
plt.ylabel("Percentage")
plt.legend(title="Diabetes")
plt.show()


target_corr = train.corr()['diagnosed_diabetes'].drop('diagnosed_diabetes')
top_corr = target_corr.abs().sort_values(ascending=False).head(8)
plt.figure(figsize=(8, 5))
sns.barplot(x=top_corr.index, y=top_corr.values)
plt.title("Top Features Correlated with Target")
plt.xticks(rotation=45)
plt.show()


# sns.pairplot(
#     train[['age', 'bmi', 'systolic_bp', 'hdl_cholesterol', 'diagnosed_diabetes']],
#     hue='diagnosed_diabetes',
#     diag_kind='kde'
# )
# plt.show()


class TargetEncoder(BaseEstimator, TransformerMixin):
    # Exact as provided; no changes
    def __init__(self, cols_to_encode, aggs=['mean'], cv=5, smooth='auto', drop_original=False):
        self.cols_to_encode = cols_to_encode
        self.aggs = aggs
        self.cv = cv
        self.smooth = smooth
        self.drop_original = drop_original
        self.mappings_ = {}
        self.global_stats_ = {}
    def fit(self, X, y):
        temp_df = X.copy()
        temp_df['target'] = y
        for agg_func in self.aggs:
            self.global_stats_[agg_func] = y.agg(agg_func)
        for col in self.cols_to_encode:
            self.mappings_[col] = {}
            for agg_func in self.aggs:
                mapping = temp_df.groupby(col)['target'].agg(agg_func)
                self.mappings_[col][agg_func] = mapping
        return self
    def transform(self, X):
        X_transformed = X.copy()
        for col in self.cols_to_encode:
            for agg_func in self.aggs:
                new_col_name = f'TE_{col}_{agg_func}'
                map_series = self.mappings_[col][agg_func]
                X_transformed[new_col_name] = X[col].map(map_series)
                X_transformed[new_col_name].fillna(self.global_stats_[agg_func], inplace=True)
        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)
        return X_transformed
    def fit_transform(self, X, y):
        self.fit(X, y)
        encoded_features = pd.DataFrame(index=X.index)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            temp_df_train = X_train.copy()
            temp_df_train['target'] = y_train
            for col in self.cols_to_encode:
                for agg_func in self.aggs:
                    new_col_name = f'TE_{col}_{agg_func}'
                    fold_global_stat = y_train.agg(agg_func)
                    mapping = temp_df_train.groupby(col)['target'].agg(agg_func)
                    if agg_func == 'mean':
                        counts = temp_df_train.groupby(col)['target'].count()
                        m = self.smooth
                        if self.smooth == 'auto':
                            variance_between = mapping.var()
                            avg_variance_within = temp_df_train.groupby(col)['target'].var().mean()
                            if variance_between > 0:
                                m = avg_variance_within / variance_between
                            else:
                                m = 0
                        smoothed_mapping = (counts * mapping + m * fold_global_stat) / (counts + m)
                        encoded_values = X_val[col].map(smoothed_mapping)
                    else:
                        encoded_values = X_val[col].map(mapping)
                    encoded_features.loc[X_val.index, new_col_name] = encoded_values.fillna(fold_global_stat)
        X_transformed = X.copy()
        for col in encoded_features.columns:
            X_transformed[col] = encoded_features[col]
        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)
        return X_transformed


X = train.drop(columns=['diagnosed_diabetes', 'id'])  # Exclude id/target; base ~24 feats
y = train['diagnosed_diabetes']
global_mean = y.mean()  # For fills


int_cols = X.select_dtypes(include=['int', 'float']).columns.tolist()
len(int_cols)


features = [col for col in test.columns if col != 'id']  # 24 feats
new_features = []
global_mean = y.mean()  # Pre-compute
for col in features:
    # Mean
    tmp = orig.groupby(col)['diagnosed_diabetes'].mean()
    new_name = f'orig_mean_{col}'  # Descriptive prefix
    train[new_name] = train[col].map(tmp).fillna(global_mean)
    test[new_name] = test[col].map(tmp).fillna(global_mean)
    new_features.append(new_name)
    # Count
    tmp_cnt_name = f'orig_cnt_{col}'
    cnt_map = orig[col].value_counts()
    train[tmp_cnt_name] = train[col].map(cnt_map).fillna(0)
    test[tmp_cnt_name] = test[col].map(cnt_map).fillna(0)
    new_features.append(tmp_cnt_name)
print(f'Augmented features: {len(new_features)}')  # 48


aug_df = train[new_features]
X = pd.concat([X, aug_df], axis=1)
test_aug = test.drop('id', axis=1)
test_aug = pd.concat([test_aug, test[new_features]], axis=1)

print(f'Total pre-TE features: {X.shape[1]}')


# Post-FE: Update int_cols (numerical only; aug means/cnts are float/int)
int_cols = [col for col in X.select_dtypes(include=['int', 'float']).columns if col.startswith(('orig_mean_', 'orig_cnt_')) or col not in cat_cols]

# Global TE for consistent test mappings (CV-internal prevents leakage)
global_te = TargetEncoder(int_cols, cv=5, smooth='auto', aggs=['mean'], drop_original=False)
global_te.fit_transform(X, y)
test_te_base = global_te.transform(test_aug)  # Use test_aug (post-FE)

# Pseudo-Labeling (2 iters; bridges synthetic-real shift; thresh=0.5 for confident labels)
def pseudo_label(X_train, y_train, test_df, te_base, n_iters=2, thresh=0.5):
    X_full = pd.concat([X_train, test_df], ignore_index=True)
    y_full = pd.concat([y_train, pd.Series([np.nan] * len(test_df))], ignore_index=True)
    for _ in range(n_iters):
        full_te = TargetEncoder(int_cols, cv=5, smooth='auto', aggs=['mean'])
        # Fit on labeled only (no leak)
        labeled_mask = ~y_full.isna()
        X_full_te = full_te.fit_transform(X_full[labeled_mask], y_full[labeled_mask])
        # Predict unlabeled
        unlabeled_te = full_te.transform(X_full[~labeled_mask])
        # Base model for pseudo (CatBoost; reg-tuned)
        pseudo_m = CatBoostClassifier(n_estimators=2000, depth=5, learning_rate=0.02, l2_leaf=3, subsample=0.8, random_seed=123, verbose=0, task_type='GPU')
        pseudo_m.fit(X_full_te, y_full[labeled_mask])
        pseudo_probs = pseudo_m.predict_proba(unlabeled_te)[:, 1]
        y_pseudo = (pseudo_probs > thresh).astype(int)
        y_full[~labeled_mask] = y_pseudo
    return X_full_te, y_full[labeled_mask], unlabeled_te, y_pseudo  # Return labeled for CV

X_ps, y_ps, test_ps_te, y_test_ps = pseudo_label(X, y, test_aug, test_te_base, n_iters=2)

# CV Setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

# OOF holders for calibration/ensemble
oof_preds = {}
test_preds_dict = {}
cv_aucs = {}

# 1. CatBoost (Reg-Tuned: l2_leaf=3, subsample=0.8; GPU; Pseudo-Integrated)
cat_oof = np.zeros(len(X_ps))
cat_test = np.zeros(len(test_aug))
cat_aucs = []
for tr_idx, val_idx in skf.split(X_ps, y_ps):
    X_tr, X_val = X_ps.iloc[tr_idx], X_ps.iloc[val_idx]
    y_tr, y_val = y_ps.iloc[tr_idx], y_ps.iloc[val_idx]
    fold_te = TargetEncoder(int_cols, cv=5, smooth='auto', aggs=['mean'], drop_original=False)
    X_tr_te = fold_te.fit_transform(X_tr, y_tr)
    X_val_te = fold_te.transform(X_val).reindex(columns=X_tr_te.columns, fill_value=global_te.global_stats_['mean'])
    test_te = test_ps_te.reindex(columns=X_tr_te.columns, fill_value=global_te.global_stats_['mean'])
    cat_m = CatBoostClassifier(n_estimators=2000, depth=5, learning_rate=0.02, l2_leaf=3, subsample=0.8, eval_metric='AUC', random_seed=123, use_best_model=True, verbose=0, early_stopping_rounds=200, task_type='GPU')
    cat_m.fit(X_tr_te, y_tr, eval_set=(X_val_te, y_val))
    val_p = cat_m.predict_proba(X_val_te)[:, 1]
    cat_aucs.append(roc_auc_score(y_val, val_p))
    cat_oof[val_idx] = val_p
    cat_test += cat_m.predict_proba(test_te)[:, 1] / 5
cv_aucs['CatBoost'] = np.mean(cat_aucs)
oof_preds['CatBoost'] = cat_oof
test_preds_dict['CatBoost'] = cat_test / len(skf)  # Normalized
print(f'CatBoost CV AUC: {cv_aucs["CatBoost"]:.5f} ± {np.std(cat_aucs):.5f}')

# 2. LGBM (Reg-Tuned: reg_lambda=1.0, subsample=0.8; GPU; Pseudo)
lgbm_oof = np.zeros(len(X_ps))
lgbm_test = np.zeros(len(test_aug))
lgbm_aucs = []
for tr_idx, val_idx in skf.split(X_ps, y_ps):
    X_tr, X_val = X_ps.iloc[tr_idx], X_ps.iloc[val_idx]
    y_tr, y_val = y_ps.iloc[tr_idx], y_ps.iloc[val_idx]
    fold_te = TargetEncoder(int_cols, cv=5, smooth='auto', aggs=['mean'], drop_original=False)
    X_tr_te = fold_te.fit_transform(X_tr, y_tr)
    X_val_te = fold_te.transform(X_val).reindex(columns=X_tr_te.columns, fill_value=global_te.global_stats_['mean'])
    test_te = test_ps_te.reindex(columns=X_tr_te.columns, fill_value=global_te.global_stats_['mean'])
    X_tr_te_np = X_tr_te.values
    X_val_te_np = X_val_te.values
    test_te_np = test_te.values
    lgbm_m = LGBMClassifier(n_estimators=2000, max_depth=5, learning_rate=0.02, reg_lambda=1.0, subsample=0.8, random_state=123, verbose=-1, device='gpu')
    lgbm_m.fit(X_tr_te_np, y_tr)
    val_p = lgbm_m.predict_proba(X_val_te_np)[:, 1]
    lgbm_aucs.append(roc_auc_score(y_val, val_p))
    lgbm_oof[val_idx] = val_p
    lgbm_test += lgbm_m.predict_proba(test_te_np)[:, 1] / 5
cv_aucs['LGBM'] = np.mean(lgbm_aucs)
oof_preds['LGBM'] = lgbm_oof
test_preds_dict['LGBM'] = lgbm_test
print(f'LGBM CV AUC: {cv_aucs["LGBM"]:.5f} ± {np.std(lgbm_aucs):.5f}')

# 3. XGBoost (Reg-Tuned: reg_lambda=1.0, subsample=0.8; GPU Hist; Pseudo)
xgb_oof = np.zeros(len(X_ps))
xgb_test = np.zeros(len(test_aug))
xgb_aucs = []
for tr_idx, val_idx in skf.split(X_ps, y_ps):
    X_tr, X_val = X_ps.iloc[tr_idx], X_ps.iloc[val_idx]
    y_tr, y_val = y_ps.iloc[tr_idx], y_ps.iloc[val_idx]
    fold_te = TargetEncoder(int_cols, cv=5, smooth='auto', aggs=['mean'], drop_original=False)
    X_tr_te = fold_te.fit_transform(X_tr, y_tr)
    X_val_te = fold_te.transform(X_val).reindex(columns=X_tr_te.columns, fill_value=global_te.global_stats_['mean'])
    test_te = test_ps_te.reindex(columns=X_tr_te.columns, fill_value=global_te.global_stats_['mean'])
    X_tr_te_np = X_tr_te.values
    X_val_te_np = X_val_te.values
    test_te_np = test_te.values
    xgb_m = XGBClassifier(n_estimators=2000, max_depth=5, learning_rate=0.02, reg_lambda=1.0, subsample=0.8, random_state=123, eval_metric='auc', tree_method='hist')
    xgb_m.fit(X_tr_te_np, y_tr)
    val_p = xgb_m.predict_proba(X_val_te_np)[:, 1]
    xgb_aucs.append(roc_auc_score(y_val, val_p))
    xgb_oof[val_idx] = val_p
    xgb_test += xgb_m.predict_proba(test_te_np)[:, 1] / 5
cv_aucs['XGBoost'] = np.mean(xgb_aucs)
oof_preds['XGBoost'] = xgb_oof
test_preds_dict['XGBoost'] = xgb_test
print(f'XGBoost CV AUC: {cv_aucs["XGBoost"]:.5f} ± {np.std(xgb_aucs):.5f}')

# Top-3 Ensemble (weighted by CV AUC; calibrated on OOF blend)
top3 = ['CatBoost', 'LGBM', 'XGBoost']
weights = np.array([cv_aucs[m] for m in top3])
weights /= weights.sum()
oof_blend = np.column_stack([oof_preds[m] for m in top3]) @ weights
test_blend = np.column_stack([test_preds_dict[m] for m in top3]) @ weights

# Calibration (isotonic on OOF blend)
from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(oof_blend, y_ps.values)  # Fit on pseudo-labeled y
ensemble_cal = iso.predict(test_blend)

# Submission (clip [0,1]; ensemble for LB win)
submission = pd.DataFrame({'id': sample['id'], 'diagnosed_diabetes': np.clip(ensemble_cal, 0, 1)})
submission.to_csv('submission.csv', index=False)
print(f'Ensemble CV AUC (blend proxy): {roc_auc_score(y_ps, oof_blend):.5f}')
print('Submission saved: submission.csv (LB target 0.75+)')

