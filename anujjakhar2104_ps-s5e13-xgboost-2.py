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


target_corr = train.corr()['diagnosed_diabetes'].drop('diagnosed_diabetes')
top_corr = target_corr.abs().sort_values(ascending=False).head(8)
plt.figure(figsize=(8, 5))
sns.barplot(x=top_corr.index, y=top_corr.values)
plt.title("Top Features Correlated with Target")
plt.xticks(rotation=45)
plt.show()


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


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)


models = {
    'CatBoost': CatBoostClassifier(n_estimators=1000, depth=6, learning_rate=0.05, random_state=123, verbose=0, task_type='GPU'),
    'LGBM': LGBMClassifier(n_estimators=1000, max_depth=6, learning_rate=0.05, random_state=123, verbose=-1, device='gpu'),
    'XGB': XGBClassifier(n_estimators=1000, max_depth=6, learning_rate=0.05, random_state=123, eval_metric='auc'), 
    'RF': RandomForestClassifier(n_estimators=500, max_depth=10, random_state=123),  
    'GB': GradientBoostingClassifier(n_estimators=500, max_depth=6, random_state=123), 
    'LR': LogisticRegression(random_state=123, max_iter=1000), 
    'SVM': SVC(probability=True, random_state=123), 
    'KNN': KNeighborsClassifier(n_neighbors=5) 
}


cv_results = {}
test_preds = {name: np.zeros(len(test)) for name in models}


global_te = TargetEncoder(int_cols, cv=5, smooth='auto', aggs=['mean'], drop_original=False)
global_te.fit_transform(X, y)  # Fit on full 
test_te_base = global_te.transform(test)  # Consistent test features


xgb_auc_scores = []
xgb_test_pred = np.zeros(len(test))
for i, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    fold_te = TargetEncoder(int_cols, cv=5, smooth='auto', aggs=['mean'], drop_original=False)
    X_tr_te = fold_te.fit_transform(X_tr, y_tr)
    X_val_te = fold_te.transform(X_val).reindex(columns=X_tr_te.columns, fill_value=global_te.global_stats_['mean'])
    test_te = test_te_base.reindex(columns=X_tr_te.columns, fill_value=global_te.global_stats_['mean'])
    xgb_model = XGBClassifier(n_estimators=1100, max_depth=5, learning_rate=0.015, random_state=123, eval_metric='auc', tree_method='hist')
    xgb_model.fit(X_tr_te, y_tr)
    val_pred = xgb_model.predict_proba(X_val_te)[:, 1]
    xgb_auc_scores.append(roc_auc_score(y_val, val_pred))
    xgb_test_pred += xgb_model.predict_proba(test_te)[:, 1] / 5
print(f'XGBoost CV AUC: {np.mean(xgb_auc_scores):.5f} ± {np.std(xgb_auc_scores):.5f}')
pd.DataFrame({'id': sample['id'], 'diagnosed_diabetes': xgb_test_pred}).to_csv('submission.csv', index=False)


# cat_auc_scores = []
# cat_test_pred = np.zeros(len(test))
# for i, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
#     X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
#     y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
#     fold_te = TargetEncoder(int_cols, cv=5, smooth='auto', aggs=['mean'], drop_original=False)
#     X_tr_te = fold_te.fit_transform(X_tr, y_tr)
#     X_val_te = fold_te.transform(X_val).reindex(columns=X_tr_te.columns, fill_value=global_te.global_stats_['mean'])
#     test_te = test_te_base.reindex(columns=X_tr_te.columns, fill_value=global_te.global_stats_['mean'])
#     train_pool = Pool(X_tr_te, y_tr)
#     test_pool = Pool(X_val_te, y_val)
#     cat_model = CatBoostClassifier(n_estimators=11061, depth=5, learning_rate=0.015, eval_metric='AUC', random_seed=123, use_best_model=True, verbose=0, early_stopping_rounds=400, task_type='GPU')
#     cat_model.fit(train_pool, eval_set=test_pool)
#     val_pred = cat_model.predict_proba(X_val_te)[:, 1]
#     cat_auc_scores.append(roc_auc_score(y_val, val_pred))
#     cat_test_pred += cat_model.predict_proba(test_te)[:, 1] / 5
# print(f'CatBoost CV AUC: {np.mean(cat_auc_scores):.5f} ± {np.std(cat_auc_scores):.5f}')
# pd.DataFrame({'id': sample['id'], 'diagnosed_diabetes': cat_test_pred}).to_csv('submission.csv', index=False)







