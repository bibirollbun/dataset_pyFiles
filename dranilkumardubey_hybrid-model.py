import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import StandardScaler, LabelEncoder

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})


print(train.head())
print(train.describe())
print(train.isnull().sum())

sns.histplot(train['Calories'], kde=True)
plt.title("Target Distribution")
plt.show()

sns.heatmap(train.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


from scipy.stats import zscore

z_scores = zscore(train.select_dtypes(include=[np.number]))
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
train = train[filtered_entries]


# Combine for uniform preprocessing
test['Calories'] = np.nan
full_data = pd.concat([train, test], sort=False)

# Encode categorical features
label = LabelEncoder()
full_data['Sex'] = label.fit_transform(full_data['Sex'])

# Drop IDs or unrelated columns if any
full_data.drop(columns=['id'], inplace=True)

# Split back
train = full_data[full_data['Calories'].notnull()]
test = full_data[full_data['Calories'].isnull()].drop('Calories', axis=1)
X = train.drop('Calories', axis=1)
y = train['Calories']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test)


xgb = XGBRegressor(random_state=42)
xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6],
    'learning_rate': [0.05, 0.1]
}

xgb_cv = GridSearchCV(xgb, xgb_params, cv=3, scoring='neg_root_mean_squared_error')
xgb_cv.fit(X_train, y_train)

xgb_pred = xgb_cv.predict(X_val)
xgb_rmsle = np.sqrt(mean_squared_log_error(y_val, xgb_pred))
print("XGBoost RMSLE:", xgb_rmsle)


cat = CatBoostRegressor(verbose=0, random_state=42)
cat_params = {
    'depth': [4, 6],
    'iterations': [200, 300],
    'learning_rate': [0.05, 0.1]
}

cat_cv = GridSearchCV(cat, cat_params, cv=3, scoring='neg_root_mean_squared_error')
cat_cv.fit(X_train, y_train)

cat_pred = cat_cv.predict(X_val)
cat_rmsle = np.sqrt(mean_squared_log_error(y_val, cat_pred))
print("CatBoost RMSLE:", cat_rmsle)


final_pred = (0.55 * xgb_cv.predict(X_test_scaled) +
              0.45 * cat_cv.predict(X_test_scaled))

submission['Calories'] = final_pred
submission.to_csv("submission.csv", index=False)



submission




