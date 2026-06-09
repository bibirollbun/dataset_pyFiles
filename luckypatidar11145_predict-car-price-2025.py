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


train = pd.read_csv("/kaggle/input/predict-supercars-prices-2025/supercars_train.csv")


pd.set_option('display.max_columns', None)


train.head(1)


print(train.info())
print(train.isnull().sum())
train.describe()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
sns.histplot(train['price'], bins=50, kde=True)
plt.title("Price Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x=train['price'])
plt.title("Price Outliers")
plt.show()



num_cols = train.select_dtypes(include=['int64','float64']).columns.drop(['price'])
train[num_cols].hist(figsize=(15,12), bins=30)
plt.suptitle("Histograms of Numerical Features")
plt.show()



plt.figure(figsize=(12,8))
corr = train[num_cols.tolist() + ['price']].corr()
sns.heatmap(corr, annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Top correlated features with price
print(corr['price'].sort_values(ascending=False))



cat_cols = train.select_dtypes(include=['object']).columns.drop(['id'])

for col in cat_cols:
    plt.figure(figsize=(10,5))
    sns.countplot(y=col, data=train, order=train[col].value_counts().index[:15])
    plt.title(f"Top categories in {col}")
    plt.show()



for col in ['brand', 'color', 'engine_config', 'transmission', 'drivetrain', 'interior_material', 'model']:
    plt.figure(figsize=(12,6))
    sns.boxplot(x=col, y='price', data=train)
    plt.xticks(rotation=45)
    plt.title(f"Price Distribution by {col}")
    plt.show()



plt.figure(figsize=(12,6))
sns.boxplot(data=train[['horsepower','torque','weight_kg','mileage','price']])
plt.title("Outliers in Key Numerical Features")
plt.show()



# feature engineering
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

train = train.drop(columns=['id','tire_brand'])

train['damage_cost'] = train['damage_cost'].fillna(0)
train['damage_type'] = train['damage_type'].fillna("None")

train['last_service_date'] = pd.to_datetime(train['last_service_date'], errors='coerce')
train['days_since_service'] = (pd.Timestamp.today() - train['last_service_date']).dt.days
train = train.drop(columns=['last_service_date'])  # drop raw date

for col in ['brand', 'model']:
    freq = train[col].value_counts()
    train[col] = train[col].map(freq)

low_card_cols = ['color','engine_config','transmission','drivetrain','market_region','interior_material','brake_type','damage_type']

train = pd.get_dummies(train, columns=low_card_cols, drop_first=True)

train['power_to_weight'] = train['horsepower'] / train['weight_kg']
train['torque_to_weight'] = train['torque'] / train['weight_kg']

flag_cols = ['carbon_fiber_body','aero_package','limited_edition','has_warranty','non_original_parts','damage']

train[flag_cols] = train[flag_cols].astype('category')
num_cols = ['year','horsepower','torque','weight_kg','zero_to_60_s','top_speed_mph','mileage','num_owners','warranty_years','damage_cost','days_since_service','power_to_weight','torque_to_weight']

scaler = StandardScaler()
train[num_cols] = scaler.fit_transform(train[num_cols])

le = LabelEncoder()
train['service_history'] = le.fit_transform(train['service_history'])



train.head()


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import xgboost as xgb

X = train.drop(columns=["price"])
y = train["price"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)

rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_val)

print("Random Forest Results")
print("MAE:", mean_absolute_error(y_val, rf_preds))
print("RMSE:", np.sqrt(mean_squared_error(y_val, rf_preds)))
print("R²:", r2_score(y_val, rf_preds))

xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=50,
              verbose=False)

xgb_preds = xgb_model.predict(X_val)

print("\nXGBoost Results")
print("MAE:", mean_absolute_error(y_val, xgb_preds))
print("RMSE:", np.sqrt(mean_squared_error(y_val, xgb_preds)))
print("R²:", r2_score(y_val, xgb_preds))

test_data = pd.read_csv("/kaggle/input/predict-supercars-prices-2025/supercars_test.csv")
test_ids = test_data["id"]
X_test_final = pd.get_dummies(test_data.drop(columns=["id"]), drop_first=True)
X_test_final = X_test_final.reindex(columns=X.columns, fill_value=0)
X_test_final = imputer.transform(X_test_final)

final_preds = xgb_model.predict(X_test_final)

submission = pd.DataFrame({
    "id": test_ids,
    "target": final_preds
})


sample_sub = pd.read_csv("/kaggle/input/predict-supercars-prices-2025/sample_submission.csv")

submission = sample_sub.copy()
submission["target"] = final_preds
submission["target"] = submission["target"].astype(float)
submission.to_csv("submission.csv", index=False)
print("submission.csv created successfully")
submission.head()




