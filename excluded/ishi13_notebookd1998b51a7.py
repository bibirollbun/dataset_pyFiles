# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/train.csv")
test = pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/test.csv")
val = pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/val.csv")


print(train.info())


print(train.describe())


print(train.isnull().sum())


train.shape


test.info()


test.shape


test.describe


sns.histplot(train['Lap_Time_Seconds'], kde=True)
plt.title("Distribution of Lap Time")
plt.show()


sns.boxplot(x=train['Lap_Time_Seconds'])
plt.title("Boxplot of Lap Time")
plt.show()


numeric_features = train.select_dtypes(include=[np.number])
sns.heatmap(numeric_features.corr(), annot=True, fmt=".2f")
plt.title("Feature Correlation")
plt.show()


train['Race_Status'] = train['position'].map({-1: 'DNF', -2: 'DNS', -3: 'DSQ', -4: 'DNQ'}).fillna('Finished')
train['Finish_Rate'] = train['finishes'] / train['starts']
train['Points_per_Race'] = train['Championship_Points'] / train['starts']


cols_to_drop = ['Unique ID', 'Rider_ID', 'rider', 'team', 'bike', 'rider_name', 'team_name', 'bike_name', 'position']
train.drop(columns=cols_to_drop, inplace=True)


categorical_cols = train.select_dtypes(include='object').columns
train = pd.get_dummies(train, columns=categorical_cols, drop_first=True)


scaler = StandardScaler()
numeric_cols = train.select_dtypes(include=[np.number]).drop(columns='Lap_Time_Seconds').columns
train[numeric_cols] = scaler.fit_transform(train[numeric_cols])



X = train.drop("Lap_Time_Seconds", axis=1)
y = train["Lap_Time_Seconds"]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


model = XGBRegressor()
model.fit(X_train, y_train)
preds = model.predict(X_valid)


print("MAE:", mean_absolute_error(y_valid, preds))
print("RMSE:", np.sqrt(mean_squared_error(y_valid, preds)))
print("R2 Score:", r2_score(y_valid, preds))


val['Race_Status'] = val['position'].map({-1: 'DNF', -2: 'DNS', -3: 'DSQ', -4: 'DNQ'}).fillna('Finished')
val['Finish_Rate'] = val['finishes'] / val['starts']
val['Points_per_Race'] = val['Championship_Points'] / val['starts']


cols_to_drop = ['Unique ID', 'Rider_ID', 'rider', 'team', 'bike',
'rider_name', 'team_name', 'bike_name', 'position']
val.drop(columns=cols_to_drop, inplace=True)


val = pd.get_dummies(val, drop_first=True)
val = val.reindex(columns=X_train.columns, fill_value=0)


val[numeric_cols] = scaler.transform(val[numeric_cols]) 


y_val_true = val_original_target = pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/val.csv")["Lap_Time_Seconds"]
y_val_pred = model.predict(val)


mae = mean_absolute_error(y_val_true, y_val_pred)
rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred))
r2 = r2_score(y_val_true, y_val_pred)

print("Validation on val.csv:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R² Score:", r2)


test['Race_Status'] = test['position'].map({-1: 'DNF', -2: 'DNS', -3: 'DSQ', -4: 'DNQ'}).fillna('Finished')
test['Finish_Rate'] = test['finishes'] / test['starts']
test['Points_per_Race'] = test['Championship_Points'] / test['starts']


cols_to_drop = ['Unique ID', 'Rider_ID', 'rider', 'team', 'bike',
'rider_name', 'team_name', 'bike_name', 'position']
test.drop(columns=cols_to_drop, inplace=True)


test = pd.get_dummies(test, drop_first=True)
test = test.reindex(columns=X_train.columns, fill_value=0) # X_train is from train/test split


test[numeric_cols] = scaler.transform(test[numeric_cols])


predictions = model.predict(test)


submission = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/sample_submission.csv')
submission['Lap_Time_Seconds'] = predictions
submission.to_csv('my_submission.csv', index=False)




