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


df_train=pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')


df_test=pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
df_test.head()



df_sub=pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
df_sub.head


df_test.shape


df_train.head()


df_train.shape


df_train.isnull().sum()


df_train['Sex'].unique()


df_train['Sex'].value_counts()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# 3. Distribution of the Target Variable ('Calories')
plt.figure(figsize=(10, 6))
sns.histplot(df_train['Calories'], kde=True)
plt.title('Distribution of Calories')
plt.xlabel('Calories')
plt.ylabel('Frequency')
plt.show()

# 4. Distribution of Numerical Features
numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(df_train[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

# 5. Box Plots for Numerical Features (to detect outliers)
plt.figure(figsize=(15, 10))
sns.boxplot(data=df_train[numerical_features], orient="v")
plt.title('Box Plots of Numerical Features')
plt.ylabel('Value')
plt.show()

# 6. Relationship between Numerical Features and Target Variable ('Calories')
for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df_train[feature], y=df_train['Calories'])
    plt.title(f'{feature} vs Calories')
    plt.xlabel(feature)
    plt.ylabel('Calories')
    plt.show()

# 7. Relationship between 'Sex' and 'Calories'
plt.figure(figsize=(8, 6))
sns.boxplot(x=df_train['Sex'], y=df_train['Calories'])
plt.title('Sex vs Calories')
plt.xlabel('Sex')
plt.ylabel('Calories')
plt.show()





# 8. Correlation Matrix
numerical_df = df_train.select_dtypes(include=['number']) # Select only numerical columns
correlation_matrix = numerical_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()






sns.pairplot(df_train[numerical_features])
plt.title('Pairwise Relationships between Numerical Features')
plt.show()

# 4. categorical feature ('sex') or target variable ('calories') ke beech sambandh ka vishleshan karen
plt.figure(figsize=(8, 6))
sns.violinplot(x='Sex', y='Calories', data=df_train)
plt.title('Sex vs Calories (Violin Plot)')
plt.xlabel('Sex')
plt.ylabel('Calories')
plt.show()



# Remove duplicates & reduce synthetic leakage
cols = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
df_train = df_train.drop_duplicates(subset=df_train.columns, keep='first').reset_index(drop=True)
df_train = df_train.groupby(by=cols)['Calories'].min().reset_index()




import time
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


# Encode 'Sex'
le = LabelEncoder()
df_train['Sex'] = le.fit_transform(df_train['Sex'])  # male=1, female=0
df_test['Sex'] = le.transform(df_test['Sex'])


# Feature Engineering
def add_bmi_intensity(df):
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
    df['Intensity'] = df['Heart_Rate'] / df['Duration']
    return df

def add_cross_terms(df, features):
    df = df.copy()
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            name = f"{features[i]}_x_{features[j]}"
            df[name] = df[features[i]] * df[features[j]]
    return df

numerical_features = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]

df_train = add_bmi_intensity(df_train.copy()) # Use .copy() to avoid modifying the original DataFrame unintentionally
df_train = add_cross_terms(df_train, numerical_features)

df_test = add_bmi_intensity(df_test.copy())
df_test = add_cross_terms(df_test, numerical_features)

print("Shape of df_train after feature engineering:", df_train.shape)
print("Shape of df_test after feature engineering:", df_test.shape)
print("\nFirst few rows of df_train with new features:")
print(df_train.head())


# Prepare matrices
X = df_train.drop(columns=["Calories"])
y = np.log1p(df_train["Calories"])
X_test = df_test.drop(columns=["id"])


# Stratified KFold based on Duration bins
n_bins = 10
duration_binned = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', subsample=None)\
                    .fit_transform(df_train[["Duration"]]).astype(int).flatten()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_log_error
import time

# Ensemble CV
cat_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))
oof_preds = np.zeros(len(X))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, duration_binned)):
    print(f"\\nğŸ”� Fold {fold+1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    start = time.time()

    # --- CatBoost ---
    cat = CatBoostRegressor(
        verbose=0,
        random_state=42
    )
    cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
    oof_preds[val_idx] += cat.predict(X_val) * 0.5
    cat_preds += cat.predict(X_test) / skf.n_splits

    # --- XGBoost ---
    xgb = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=10,
        subsample=0.9,
        colsample_bytree=0.7,
        gamma=0.01,
        max_delta_step=2,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=100,
        eval_metric="rmse",
        verbosity=0,
        random_state=42
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
    oof_preds[val_idx] += xgb.predict(X_val) * 0.5
    xgb_preds += xgb.predict(X_test) / skf.n_splits

    fold_rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_val), np.expm1(oof_preds[val_idx])))
    print(f"ğŸ“‰ Fold {fold+1} RMSLE: {fold_rmsle:.5f}")
    print(f"â�±ï¸� Time: {time.time() - start:.1f} sec")


# Final Evaluation
final_rmsle = np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(oof_preds)))
print(f"\\nâœ… Final OOF RMSLE: {final_rmsle:.5f}")




# Final submission
final_preds = 0.5 * cat_preds + 0.5 * xgb_preds
df_sub["Calories"] = np.clip(np.expm1(final_preds), 1, 314)
df_sub.to_csv("submission.csv", index=False)
print("\nğŸ“� submission.csv saved.")
df_sub.head()




