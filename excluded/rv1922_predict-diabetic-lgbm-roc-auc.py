import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")


print("Loading Data...")
train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
orig = pd.read_csv("/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv")
sample = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


train.head()


orig['id'] = orig.index
orig = orig[train.columns.to_list()]

train = train.merge(orig, how='outer')
train.shape


class TargetEncoder(BaseEstimator, TransformerMixin):
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
                            m = avg_variance_within / variance_between if variance_between > 0 else 0
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


def bmi_classification(score):
    if score >= 30:
        return 'Obese'
    elif score >= 25:
        return 'Overwieght'
    elif score >= 18.5:
        return 'Normal'
    else:
        return 'Underweight'


for d in [train, test, orig]:
    d['bmi_category'] = d['bmi'].apply(bmi_classification)


new_features = []
features = test.columns.to_list()

for col in features:
    tmp = orig.groupby(col)['diagnosed_diabetes'].mean()
    new_name = 'orig_' + str(col)
    tmp.name = new_name
    train = train.merge(tmp, how='left', on=col)
    test = test.merge(tmp, how='left', on=col)
    new_features.append(new_name)

    tmp_cnt_name = 'orig_cnt_' + str(col)
    tmp_cnt = orig.groupby(col).size().reset_index(name=tmp_cnt_name)
    train = train.merge(tmp_cnt, how='left', on=col)
    test = test.merge(tmp_cnt, how='left', on=col)
    new_features.append(tmp_cnt_name)

new_base = features + new_features
print(f'New features: {len(new_base)}')


objs = train.select_dtypes(include='object').columns.to_list()
for obj in objs:
    train[obj] = train[obj].astype('category')
    test[obj] = test[obj].astype('category')


objects = train.select_dtypes('category').columns
for obj in objects:
    le = LabelEncoder()
    le.fit(pd.concat([train[obj], test[obj]], axis=0).astype(str))
    train[obj] = le.transform(train[obj].astype(str))
    test[obj] = le.transform(test[obj].astype(str))


CAT_params = {
    'n_estimators': 12000,
    'depth': 3,
    'learning_rate': 0.01,
    'eval_metric': 'AUC',
    'random_seed': 123,
    'use_best_model': True,
    'verbose': 0,
    'early_stopping_rounds': 300,
    'task_type': 'GPU'
}


X = train.drop(columns=['diagnosed_diabetes'])
y = train['diagnosed_diabetes']
int_cols = X.select_dtypes(include=['int', 'float']).columns.to_list()


folds = 10
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

cat_test_pred = np.zeros(len(test))
cat_auc_scores = []

for i, (train_index, test_index) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    TE = TargetEncoder(cols_to_encode=int_cols, cv=5, smooth='auto', aggs=['mean'], drop_original=False)
    X_train = TE.fit_transform(X_train, y_train)
    X_test = TE.transform(X_test)
    test = TE.transform(test)

    cat_model = CatBoostClassifier(**CAT_params)
    cat_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    cat_test_pred += cat_model.predict_proba(test)[:, 1] / folds
    cat_y_pred = cat_model.predict_proba(X_test)[:, 1]
    cat_auc_score = roc_auc_score(y_test, cat_y_pred)
    cat_auc_scores.append(cat_auc_score)

    print(f'Fold {i} CatBoost AUC score: {cat_auc_score:.5f}')

plt.plot(cat_auc_scores, marker='D', linestyle='-')
plt.hlines(y=np.mean(cat_auc_scores), xmin=0, xmax=folds - 1, linestyles='--')
plt.xlabel('Fold')
plt.ylabel('AUC')
plt.title('AUC Scores')
plt.show()


submission = pd.DataFrame({
    'id': sample['id'],
    'diagnosed_diabetes': cat_test_pred
})

submission.to_csv('submission.csv', index=False)
submission.head()

