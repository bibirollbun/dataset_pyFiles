import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


data.shape


data.head()


data.describe()


data.info()


missing_table = pd.DataFrame({
    'Missing Values': data.isna().sum(),
    'Percentage (%)': (data.isnull().mean() * 100).round(2)
})

print(missing_table.sort_values(by='Missing Values', ascending=False))


data.nunique()


data['y'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, explode=(0,0.1), colors=['#4c72b0', '#aec7e8'])
plt.title('Target distribution')
plt.ylabel('')
plt.show()


sns.histplot(data=data, x="age", kde=True, bins=70)


plt.figure(figsize=(12, 5))
sns.countplot(data=data, x="job")
plt.xlabel('Type of job')
plt.ylabel('')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


sns.countplot(data=data, x="marital")
plt.xlabel('Marital status')
plt.ylabel('')
plt.tight_layout()
plt.show()


sns.countplot(data=data, x="education")
plt.xlabel('Level of education')
plt.ylabel('')
plt.tight_layout()
plt.show()


data['default'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, explode=(0,0.1), colors=['#4c72b0', '#aec7e8'])
plt.title('Has credit in default?')
plt.ylabel('')
plt.show()


sns.histplot(data=data, x="balance", kde=True, bins=50)


data['housing'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, explode=(0,0.1), colors=['#4c72b0', '#aec7e8'])
plt.title('Has a housing loan?')
plt.ylabel('')
plt.show()


data['loan'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, explode=(0,0.1), colors=['#4c72b0', '#aec7e8'])
plt.title('Has a personal loan?')
plt.ylabel('')
plt.show()


sns.countplot(data=data, x="contact")
plt.xlabel('Type of communication contact')
plt.ylabel('')
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 5))
sns.countplot(data=data, x="day")
plt.xlabel('Last contact day of the month')
plt.ylabel('')
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 5))
sns.countplot(data=data, x="month")
plt.xlabel('Last contact month of the year')
plt.ylabel('')
plt.tight_layout()
plt.show()


sns.histplot(data=data, x="duration", kde=True, bins=50)


sns.histplot(data=data, x="campaign", kde=True, bins=50)


plt.figure(figsize=(12, 5))
sns.countplot(data=data, x="poutcome")
plt.xlabel('Outcome of the previous marketing campaign')
plt.ylabel('')
plt.tight_layout()
plt.show()


plt.figure(figsize=(16, 6))

data_corr = data.corr(numeric_only=True)

heatmap = sns.heatmap(data_corr.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12})

plt.show()


from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import shuffle


X = data.drop(['y', 'id'], axis=1)
y = data['y']


numerical_pipeline = Pipeline([
    ('scaler', StandardScaler())
])


categorical_pipeline = Pipeline([
    ('encoder', OrdinalEncoder())
])


preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, make_column_selector(dtype_include=['int64', 'float64'])),
    ('cat', categorical_pipeline, make_column_selector(dtype_include=['object']))
])


le = LabelEncoder()
y = le.fit_transform(y)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
      pass

    def fit(self, X, y = None):
      return self

    def transform(self, X):
        X = X.copy()

        X['long_call'] = X['duration'] > 200
        X['balance_positive'] = (X['balance'] > 0).astype(int)
        X['campaign_multiple'] = (X['campaign'] > 2).astype(int)
        X['age_bin'] = pd.cut(X['age'], bins=[17, 30, 60, 100], labels=['young', 'middle', 'senior'])

        X['duration_log'] = np.log1p(X['duration'])
        X['campaign_log'] = np.log1p(X['campaign'])
        X['pdays_log'] = np.log1p(X['pdays'] + 1)
        X['previous_log'] = np.log1p(X['previous'])
        X['balance_sqrt'] = np.sqrt(X['balance'] - X['balance'].min() + 1)
        X['age_squared'] = X['age'] ** 2

        return X


import xgboost as xgb
from collections import Counter
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
import lightgbm as lgb


counter = Counter(y)
scale_pos_weight = counter[0] / counter[1]


parameters_lgbm={'n_estimators': 793,
                 'num_leaves': 135, 
                 'min_child_samples': 128, 
                 'learning_rate': 0.04658385614607383, 
                 'colsample_bytree': 1.0, 
                 'reg_alpha': 0.0009765625, 
                 'reg_lambda': 16.24395021787904, 
                 'max_bin': 255, 
                 'force_col_wise':True,
                 'device':'gpu',
                 'n_jobs':-1,
                 "objective" : "binary:logistic",
                 'metric': 'auc'
                } 


params = {
    'objective': "binary",
    'metric': 'auc',
    'verbosity': -1,
    'boosting_type': "gbdt",
    'learning_rate': 0.01,
    'max_depth': 20,
    'num_leaves': 200,
    'max_bin': 400,
    'subsample': 0.85,
    'colsample_bytree': 0.7,
    'subsample_freq': 1,
    'reg_alpha': 6.0,
    'reg_lambda': 4.0,
    'min_child_samples': 25,
    'min_split_gain': 0.001,
    'n_jobs': -1,
    'lambda_l1': 0.5,
    'lambda_l2': 0.3
}



pipeline = Pipeline([
    ('features', FeatureEngineer()),
    ('preprocessing', preprocessor)
    ])


X_preprocessed = pipeline.fit_transform(X)
X_test_preprocessed = pipeline.transform(test_data)


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_preprocessed))
test_preds = np.zeros(len(X_test_preprocessed))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_preprocessed, y)):
    print(f"Fold {fold + 1}")

    X_train, X_val = X_preprocessed[train_idx], X_preprocessed[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    lgb_model = lgb.train(
        parameters_lgbm,
        train_set=dtrain,
        num_boost_round=1000,
        valid_sets=[dval]
        )

    oof_preds[val_idx] = lgb_model.predict(X_val)
    test_preds += lgb_model.predict(X_test_preprocessed) / skf.n_splits

# Calculate AUC
cv_roc = roc_auc_score(y, oof_preds)
print(f"Cross-Validation ROC AUC: {cv_roc:.4f}")



submission = pd.DataFrame({
    'id': test_data["id"],
    'y': test_preds
})


submission


submission.to_csv('submission.csv', index=False)




