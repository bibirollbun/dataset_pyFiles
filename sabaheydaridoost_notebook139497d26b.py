! pip install lifelines


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold, KFold
from lifelines.utils import concordance_index
from sklearn.metrics import mean_squared_error, accuracy_score
from lifelines import KaplanMeierFitter


MAIN_PATH='/kaggle/input/equity-post-HCT-survival-predictions/'


train = pd.read_csv(f'{MAIN_PATH}train.csv')
test = pd.read_csv(f'{MAIN_PATH}test.csv')


train.info()


train.describe()


plt.figure(figsize=(10, 6))
sns.histplot(train["efs_time"], bins=50, kde=True)
plt.title("Distribution of efs_time")
plt.show()


plt.figure(figsize=(10, 6))
sns.countplot(x=train["efs"], palette="viridis")
plt.title("Distribution of efs")
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x=train["efs"], y=train["efs_time"], palette="coolwarm")
plt.title("Boxplot of efs_time by efs")
plt.show()


plt.figure(figsize=(10, 6))
sns.violinplot(x=train["race_group"], y=train["efs_time"], palette="pastel")
plt.title("Violin Plot of efs_time by Race Group")
plt.xticks(rotation=45)
plt.show()


train.drop(columns=["ID"], inplace=True)


train.isnull().sum()


train = train.dropna(subset=['efs', 'efs_time'])


X = train.drop(columns=["efs_time", "efs"])
y = train[["efs_time", "efs"]].copy()


# Kaplan-Meier Label Engineering
kmf = KaplanMeierFitter()
kmf.fit(y['efs_time'], y['efs'])
y['km_label'] = kmf.survival_function_at_times(y['efs_time']).values
y.loc[y['efs'] == 0, 'km_label'] -= 0.2


not_valid_chars = ['[', ']', '<', '(', ')', '+', ',', '/']

new_cols = []
for col in X.columns.values:
    name = ''
    for c in col:
        if c not in not_valid_chars:
            name += c
    new_cols.append(str(name))
X.columns = new_cols


y["efs"] = y["efs"].astype(bool)


categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()


num_imputer = SimpleImputer(strategy="mean")
cat_imputer = SimpleImputer(strategy="most_frequent")
X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])


encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]))
X_encoded.columns = encoder.get_feature_names_out()
X = X.drop(columns=categorical_cols).reset_index(drop=True)
X = pd.concat([X, X_encoded], axis=1)


scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])


# kmf = KaplanMeierFitter()
# # kmf.fit(y['efs_time'], y['efs'])
# y['km_label'] = kmf.survival_function_at_times(y['efs_time']).values
# y.loc[y['efs'] == 0, 'km_label'] -= 0.2


# # K-Fold Cross Validation
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# mse_scores = []

# for i, (train_indexes, val_indexes) in enumerate(kf.split(X)):
#     X_train, X_test = X.iloc[train_indexes], X.iloc[val_indexes]
#     y_train, y_test = y.iloc[train_indexes], y.iloc[val_indexes]

#     model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
#     model.fit(X_train, y_train["km_label"])

#     preds_km = model.predict(X_test)
#     mse_scores.append(mean_squared_error(y_test["km_label"], preds_km))
#     print(f'Fold #{i} MSE: {mse_scores[-1]}')

# final_mse = np.mean(mse_scores)
# print(f'Final Mean MSE: {final_mse:.4f}')

