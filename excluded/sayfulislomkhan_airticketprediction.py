import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# Fayllarni yuklash
train = pd.read_csv("/kaggle/input/aviachipta-narxini-bashorat-qilish/train_data.csv")
test = pd.read_csv("/kaggle/input/aviachipta-narxini-bashorat-qilish/test_data.csv")
sample = pd.read_csv("/kaggle/input/aviachipta-narxini-bashorat-qilish/sample_solution.csv")


x = train.drop("price", axis=1)
y = train['price']


# Train-test ajratish (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)


# Ustunlarni ajratish
nums = ['duration', 'days_left']
cats = ['airline', 'flight', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']


# Pipeline yaratish
num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, nums),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cats)
])


# Ma'lumotlarni o‘zgartirish
X_train_pre = full_pipeline.fit_transform(X_train)
X_val_pre = full_pipeline.transform(X_val)


# Modelni yaratish
random_forest_m = RandomForestRegressor(random_state=42)
random_forest_m.fit(X_train_pre, y_train)


# Validation set uchun bashorat qilish
val_pred = random_forest_m.predict(X_val_pre)


# Modelni baholash (RMSE)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Validation RMSE: {rmse:.2f}")


# Hyperparameter Tuning (GridSearchCV)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}


grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)


grid_search.fit(X_train_pre, y_train)


# Eng yaxshi model
best_model = grid_search.best_estimator_
print("Eng yaxshi parametrlari:", grid_search.best_params_)


# Test to‘plami uchun bashorat qilish
test_pre = full_pipeline.transform(test)
test_predict = best_model.predict(test_pre)


# Natijalarni saqlash
sample['price'] = test_predict
sample.to_csv('prediction.csv', index=False)
print("Bashoratlar saqlandi!")

