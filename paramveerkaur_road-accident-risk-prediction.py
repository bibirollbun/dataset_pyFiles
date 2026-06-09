import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


data = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
data.head()


test_data = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
test_data.head()


le = LabelEncoder()
data["road_type"]=le.fit_transform(data["road_type"])
data["lighting"]=le.fit_transform(data["lighting"])
data["weather"]=le.fit_transform(data["weather"])
data["road_signs_present"]=le.fit_transform(data["road_signs_present"])
data["public_road"]=le.fit_transform(data["public_road"])
data["time_of_day"]=le.fit_transform(data["time_of_day"])
data["holiday"]=le.fit_transform(data["holiday"])
data["school_season"]=le.fit_transform(data["school_season"])
data.head()


le = LabelEncoder()
test_data["road_type"]=le.fit_transform(test_data["road_type"])
test_data["lighting"]=le.fit_transform(test_data["lighting"])
test_data["weather"]=le.fit_transform(test_data["weather"])
test_data["road_signs_present"]=le.fit_transform(test_data["road_signs_present"])
test_data["public_road"]=le.fit_transform(test_data["public_road"])
test_data["time_of_day"]=le.fit_transform(test_data["time_of_day"])
test_data["holiday"]=le.fit_transform(test_data["holiday"])
test_data["school_season"]=le.fit_transform(test_data["school_season"])
test_data.head()


data = data.drop("id", axis=1)
data.head()


data.info()


data.isnull().sum()


data.duplicated().sum()


print(data.shape)
data = data.drop_duplicates()
print(data.shape)


X = data.drop("accident_risk", axis=1)
y = data["accident_risk"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


rf = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 150, 200, 250],
    'max_depth': [5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring='neg_root_mean_squared_error', 
    verbose=2
)


grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
best_rf = grid_search.best_estimator_


y_pred = best_rf.predict(X_test)
y_pred = y_pred.clip(0, 1)


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")


test_ids = test_data["id"]
test_data = test_data.drop("id", axis=1)
print(test_ids)


scaler = MinMaxScaler()
test_data = scaler.fit_transform(test_data)
predicted_risk = best_rf.predict(test_data)
predicted_risk = predicted_risk.clip(0,1)
predicted_risk = np.round(predicted_risk,3)
print(predicted_risk)


result = pd.DataFrame({'id':test_ids, 'accident_risk':predicted_risk})
result.to_csv('submission.csv', index=False)
print(result)

