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


# ğŸ“¦ Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
# To ignore warinings
import warnings
warnings.filterwarnings('ignore')


# ğŸ”½ Load Data
df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


df_sub.head()


df_train.head()


# ğŸ§¹ Drop ID column
df_train.drop(columns=['id'], inplace=True)
df_test.drop(columns=['id'], inplace=True)


# ğŸ”� Separate common columns between train and test
common_cols = df_train.columns.intersection(df_test.columns)


# ğŸ”� Separate numeric and categorical features
numerical_cols = df_train[common_cols].select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df_train[common_cols].select_dtypes(include=['object']).columns



# ğŸ”§ Fill Missing Values
# Numeric
df_train[numerical_cols] = df_train[numerical_cols].fillna(df_train[numerical_cols].median())
df_test[numerical_cols] = df_test[numerical_cols].fillna(df_train[numerical_cols].median())
# Categorical
df_train[categorical_cols] = df_train[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]))
df_test[categorical_cols] = df_test[categorical_cols].apply(lambda x: x.fillna(df_train[x.name].mode()[0]))



# ğŸ§  Quick Glance at the Data
print(df_train.info())
print(df_train.describe())


# ğŸ“Š 1. Distribution of Target
plt.figure(figsize=(8, 5))
sns.histplot(df_train['Listening_Time_minutes'], kde=True, bins=30)
plt.title('Distribution of Listening_Time_minutes Variable')
plt.xlabel('Target')
plt.ylabel('Frequency')
plt.show()


# ğŸ“‰ 2. Missing Value Matrix
msno.matrix(df_train)
plt.title("Missing Data Pattern - Training Set")
plt.show()


# ğŸ“Š 3. Correlation Heatmap (Numerical)
plt.figure(figsize=(12, 8))
corr = df_train[numerical_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap (Numerical Features)')
plt.show()


# ğŸ“ˆ 4. Boxplot of Categorical Features vs Target
for col in categorical_cols:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=col, y='Listening_Time_minutes', data=df_train)
    plt.title(f'{col} vs. Listening_Time_minutes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ğŸ“‰ 5. Histograms of Numerical Features
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df_train[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()


for col in categorical_cols:
    plt.figure(figsize=(10, 5))
    sns.violinplot(x=col, y='Listening_Time_minutes', data=df_train, inner='box')
    plt.title(f'Violin Plot: {col} vs Target')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



for col in categorical_cols[:3]:  # Avoid overload â€“ limit to 3
    plt.figure(figsize=(10, 5))
    sns.stripplot(x=col, y='Listening_Time_minutes', data=df_train, jitter=True, alpha=0.5)
    plt.title(f'Strip Plot: {col} vs Listening_Time_minutes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



for col in categorical_cols:
    if df_train[col].nunique() > 5 and df_train[col].nunique() < 20:
        plt.figure(figsize=(12, 6))
        sns.boxenplot(x=col, y='Listening_Time_minutes', data=df_train)
        plt.title(f'Boxen Plot: {col} vs Target')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()



for col in categorical_cols:
    category_target_mean = df_train.groupby(col)['Listening_Time_minutes'].mean().sort_values()
    plt.figure(figsize=(10, 4))
    sns.barplot(x=category_target_mean.index, y=category_target_mean.values, palette="viridis")
    plt.title(f'Mean Target by {col}')
    plt.xticks(rotation=45)
    plt.ylabel('Mean Target')
    plt.tight_layout()
    plt.show()



target_corr = df_train[numerical_cols].corrwith(df_train['Listening_Time_minutes']).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=target_corr.values, y=target_corr.index, palette='coolwarm')
plt.title('Numerical Features Correlation with Target')
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.show()



for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df_train[col])
    plt.title(f'Boxplot: {col}')
    plt.tight_layout()
    plt.show()



plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, cmap='vlag', linewidths=0.5, fmt=".2f", center=0)
plt.title("Feature Correlation Heatmap")
plt.show()



# Copy data to avoid modifying original
train = df_train.copy()
test = df_test.copy()

# Save the target and drop it temporarily
target = train['Listening_Time_minutes']
train.drop(columns=['Listening_Time_minutes'], inplace=True)

# Tag origin for later split
train['source'] = 'train'
test['source'] = 'test'

# Combine datasets
full_data = pd.concat([train, test], axis=0).reset_index(drop=True)



# Feature: Episode number from title
full_data['Episode_Number'] = full_data['Episode_Title'].str.extract(r'(\d+)').astype(float)

# Drop high-cardinality strings
full_data.drop(columns=['Podcast_Name', 'Episode_Title'], inplace=True)

# One-hot encoding for categoricals
cat_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
full_data = pd.get_dummies(full_data, columns=cat_cols, drop_first=True)

# Fill NA (if any left)
full_data = full_data.fillna(full_data.median(numeric_only=True))



# Split back
train = full_data[full_data['source'] == 'train'].drop(columns=['source'])
test = full_data[full_data['source'] == 'test'].drop(columns=['source'])

# Add back the target
train['Listening_Time_minutes'] = target



X = train.drop(columns=['Listening_Time_minutes'])
y = train['Listening_Time_minutes']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_val)
rmse = mean_squared_error(y_val, y_pred, squared=False)
r2 = r2_score(y_val, y_pred)

print(f"ğŸ“‰ Validation RMSE: {rmse:.4f}")
print(f"ğŸ“ˆ Validation RÂ²: {r2:.4f}")



importances = model.feature_importances_
features = X.columns

feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp.values[:15], y=feat_imp.index[:15])
plt.title('Top 15 Feature Importances')
plt.tight_layout()
plt.show()



# Predict on actual test set
test_preds = model.predict(test)

# Create submission
df_sub['Listening_Time_minutes'] = test_preds
df_sub.to_csv('submission.csv', index=False)
print("âœ… Submission file saved as 'submission.csv'")



pip install lightgbm xgboost optuna



import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import StackingRegressor



# ğŸ“¥ Load data
df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')



# ğŸ”§ Feature Engineering
target = df_train['Listening_Time_minutes']
df_train.drop(['id', 'Listening_Time_minutes'], axis=1, inplace=True)
df_test.drop('id', axis=1, inplace=True)

df_train['source'] = 'train'
df_test['source'] = 'test'

combined = pd.concat([df_train, df_test])
combined['Episode_Number'] = combined['Episode_Title'].str.extract(r'(\d+)').astype(float)
combined.drop(columns=['Podcast_Name', 'Episode_Title'], inplace=True)

cat_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
combined = pd.get_dummies(combined, columns=cat_cols, drop_first=True)



# Fill missing values
combined.fillna(combined.median(numeric_only=True), inplace=True)



# Split back
train = combined[combined['source'] == 'train'].drop(columns='source')
test = combined[combined['source'] == 'test'].drop(columns='source')



# ğŸ“‰ Log-transform the target to handle skewness
target_log = np.log1p(target)



# ğŸ“Š Define Cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)



# ğŸ”® Hyperparameter tuning using Optuna for LightGBM
def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
    }
    lgb_model = lgb.LGBMRegressor(**params, n_estimators=1000)
    score = -np.mean(cross_val_score(lgb_model, train, target_log, cv=cv, scoring='neg_root_mean_squared_error'))
    return score

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)
best_params = study.best_params





best_params


# ğŸ“¦ Define base models with tuned LGB
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
xgboost = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, objective='reg:squarederror', random_state=42)
lightgbm = lgb.LGBMRegressor(**best_params, n_estimators=1000)



# ğŸ§  Stacking Regressor
stack = StackingRegressor(
    estimators=[
        ('rf', rf),
        ('xgb', xgboost),
        ('lgb', lightgbm)
    ],
    final_estimator=Ridge(alpha=1.0)
)



# ğŸ“ˆ Fit model on full data
stack.fit(train, target_log)



# ğŸ”� Predict on test set and inverse log
preds = stack.predict(test)
preds_final = np.expm1(preds)



# ğŸ“„ Save Submission
df_sub['Listening_Time_minutes'] = preds_final
df_sub.to_csv('submission.csv', index=False)
print("âœ… Submission file created!")





dsa

