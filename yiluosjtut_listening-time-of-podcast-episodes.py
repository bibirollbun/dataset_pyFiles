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


# %% Section 1: Imports & Setup


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

sns.set_style('whitegrid')
pd.options.display.max_columns = 20



# %% Section 2: Data Loading & Initial Checks
# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

print("Train shape:", train.shape)
print("Test shape :", test.shape)
display(train.head())
display(train.info())


# %% Section 3: Exploratory Data Analysis (EDA)
# 3.1 Missing values overview
missing = train.isnull().sum()
print("Missing values in train:")
display(missing[missing > 0])

# 3.2 Distribution of numeric features
num_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
            'Guest_Popularity_percentage', 'Number_of_Ads', 'Listening_Time_minutes']
plt.figure(figsize=(12, 8))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(train[col], bins=30, kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()

# 3.3 Correlation heatmap (numeric)
corr = train[num_cols].corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Numeric Feature Correlation')
plt.show()

# 3.4 Categorical feature analysis boxplots
cat_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
plt.figure(figsize=(12, 8))
for i, col in enumerate(cat_cols, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=train[col], y=train['Listening_Time_minutes'])
    plt.xticks(rotation=45)
    plt.title(f'Listening Time by {col}')
plt.tight_layout()
plt.show()


# %% Section 4: Feature Engineering
# 4.1 Impute numeric columns
numeric_imputer = SimpleImputer(strategy='median')
train[num_cols[:-1]] = numeric_imputer.fit_transform(train[num_cols[:-1]])
test[num_cols[:-1]]  = numeric_imputer.transform(test[num_cols[:-1]])

# 4.2 Encode categorical columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col]  = le.transform(test[col].astype(str))
    label_encoders[col] = le

# 4.3 Prepare feature matrices
drop_cols = ['id', 'Podcast_Name', 'Episode_Title', 'Listening_Time_minutes']
X = train.drop(columns=drop_cols)
y = train['Listening_Time_minutes']
X_test = test.drop(columns=['id', 'Podcast_Name', 'Episode_Title'])


# %% Section 5: Model Training with CV & Stacking
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Prepare OOF and test predictions
base_models = ['lgb', 'ridge', 'rf']
oof_preds = pd.DataFrame(np.zeros((len(X), len(base_models))), columns=base_models)
test_preds = pd.DataFrame(np.zeros((len(X_test), len(base_models))), columns=base_models)

# LightGBM params
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 5000,
    'learning_rate': 0.08,
    'max_depth': 4,
    'reg_alpha': 0.8,
    'reg_lambda': 4,
    'colsample_bytree': 0.7,
    'subsample': 0.7,
    'random_state': 42,
    'verbose': -1
}

for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    # 5.1 LightGBM
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='rmse',
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_preds.loc[val_idx, 'lgb'] = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration_)
    test_preds['lgb'] += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration_) / n_folds

    # 5.2 Ridge Regression
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_tr, y_tr)
    oof_preds.loc[val_idx, 'ridge'] = ridge_model.predict(X_val)
    test_preds['ridge'] += ridge_model.predict(X_test) / n_folds

    # 5.3 Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf_model.fit(X_tr, y_tr)
    oof_preds.loc[val_idx, 'rf'] = rf_model.predict(X_val)
    test_preds['rf'] += rf_model.predict(X_test) / n_folds

    print(f"Fold {fold} completed")

# Clip predictions
oof_preds = oof_preds.clip(0, 120)
test_preds = test_preds.clip(0, 120)


# %% Section 6: Evaluate Base Models
for name in base_models:
    rmse = mean_squared_error(y, oof_preds[name], squared=False)
    print(f"{name.upper()} CV RMSE: {rmse:.4f}")

# %% Section 7: Stacking & Final Submission
meta_model = Ridge(alpha=1.0)
meta_model.fit(oof_preds, y)
final_preds = meta_model.predict(test_preds).clip(0, 120)

submission = pd.DataFrame({'id': test['id'], 'Listening_Time_minutes': final_preds})
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Final submission saved to /kaggle/working/submission.csv")




