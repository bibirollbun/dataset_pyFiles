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


import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

train = pd.read_csv("/kaggle/input/tabular-playground-series-feb-2021/train.csv")
test = pd.read_csv("/kaggle/input/tabular-playground-series-feb-2021/test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()


train.info()


train.describe().T


cont_features = [col for col in train.columns if 'cont' in col]


long_df = train[cont_features].melt(var_name='feature', value_name='value')


plt.figure(figsize=(18, 8))
sns.boxenplot(data=long_df, x='feature', y='value', palette='coolwarm')
plt.title("Boxenplot of Continuous Features")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()



cont_features = [col for col in train.columns if 'cont' in col]


corr_matrix = train[cont_features].corr()


plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap of Continuous Features")
plt.tight_layout()
plt.show()



cat_features = [col for col in train.columns if 'cat' in col]


for col in cat_features:
    plt.figure(figsize=(6, 3))
    sns.countplot(data=train, x=col, order=train[col].value_counts().index, palette='Set2')
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



from sklearn.preprocessing import LabelEncoder, StandardScaler

label_encoders = {}
for col in cat_features:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le 

scaler = StandardScaler()
train[cont_features] = scaler.fit_transform(train[cont_features])
test[cont_features] = scaler.transform(test[cont_features])



from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

X = train.drop(['target', 'id'], axis=1)
y = train['target']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_valid)

# Evaluation
r2 = r2_score(y_valid, xgb_preds)
mse = mean_squared_error(y_valid, xgb_preds)
print(f"XGBoost R2 Score: {r2:.4f}")
print(f"XGBoost MSE: {mse:.4f}")




from lightgbm import LGBMRegressor

lgb_model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
lgb_model.fit(X_train, y_train)
lgb_preds = lgb_model.predict(X_valid)

# Evaluation
print(f"LightGBM R²: {r2_score(y_valid, lgb_preds):.4f}")
print(f"LightGBM MSE: {mean_squared_error(y_valid, lgb_preds):.4f}")




from catboost import CatBoostRegressor

cat_model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, verbose=0, random_seed=42)
cat_model.fit(X_train, y_train)
cat_preds = cat_model.predict(X_valid)

print(f"CatBoost R²: {r2_score(y_valid, cat_preds):.4f}")
print(f"CatBoost MSE: {mean_squared_error(y_valid, cat_preds):.4f}")




importances = xgb_model.feature_importances_
features = X_train.columns


plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()



!pip install shap
import shap

explainer = shap.Explainer(xgb_model, X_train)
shap_values = explainer(X_valid)



shap.plots.beeswarm(shap_values, max_display=15)




final_preds = cat_model.predict(test.drop(['id'], axis=1))

submission = pd.DataFrame({
    'id': test['id'],
    'target': final_preds
})


submission.to_csv('submission.csv', index=False)
submission.head()





