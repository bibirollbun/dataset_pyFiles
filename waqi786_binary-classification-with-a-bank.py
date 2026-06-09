


# Basic libraries
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Models
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer

# ML algorithms
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# Ensemble
from sklearn.ensemble import StackingClassifier

import warnings
warnings.filterwarnings('ignore')



train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')

print(train.shape, test.shape)
train.head()



print(train.isnull().sum())
print(test.isnull().sum())

# Target distribution
sns.countplot(x='y', data=train)
plt.title("Target Distribution")
plt.show()

# Numeric summary
train.describe()



def feature_engineering(df):
    # Day and month into single datetime-like feature
    month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
                 'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
    df['month_num'] = df['month'].map(month_map)
    
    # Combine campaign and previous
    df['campaign_previous'] = df['campaign'] + df['previous']
    
    # Duration ratio
    df['duration_campaign_ratio'] = df['duration'] / (df['campaign']+1)
    
    # Balance categories
    df['balance_cat'] = pd.cut(df['balance'], bins=[-9999,0,1000,5000,100000], labels=['low','medium','high','very_high'])
    
    return df

train = feature_engineering(train)
test = feature_engineering(test)



categorical_cols = train.select_dtypes(include=['object']).columns.tolist()

# Remove columns that might not exist in test/train simultaneously (just in case)
if 'y' in categorical_cols:
    categorical_cols.remove('y')

le = LabelEncoder()
for col in categorical_cols:
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))



X = train.drop(['y','id'], axis=1)
y = train['y']
X_test = test.drop('id', axis=1)



n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)



lgb_preds = np.zeros(len(X_test))
oof = np.zeros(len(X))

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'n_estimators': 10000,
    'max_depth': -1,
    'num_leaves': 64,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"FOLD {fold+1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = lgb.LGBMClassifier(**params)
    
    # Use callbacks for early stopping
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='auc',
              callbacks=[lgb.early_stopping(stopping_rounds=500),
                         lgb.log_evaluation(200)])
    
    oof[val_idx] = model.predict_proba(X_val, num_iteration=model.best_iteration_)[:,1]
    lgb_preds += model.predict_proba(X_test, num_iteration=model.best_iteration_)[:,1] / n_splits

print("OOF ROC AUC:", roc_auc_score(y, oof))



submission = sample.copy()
submission['y'] = lgb_preds
submission.to_csv('submission.csv', index=False)
submission.head()


