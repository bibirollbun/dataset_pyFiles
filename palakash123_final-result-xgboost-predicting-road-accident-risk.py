import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os 



BASE_DIR = os.getcwd()
if BASE_DIR.endswith('notebooks'):
    BASE_DIR = os.path.dirname(BASE_DIR)

train_path = "/kaggle/input/akashpal-accident-risk-processed-data/preprocessed_train.csv"
test_path = "/kaggle/input/akashpal-accident-risk-processed-data/preprocessed_test.csv"




train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
print(f"Train shape: {train.shape}, Test shape: {test.shape}")



#splitting the features and target 

X= train.drop("accident_risk",axis=1)
y = train['accident_risk']


#splitting the data for validation 

X_train , X_val , y_train , y_val = train_test_split(X,y,test_size=0.20,random_state=4294967295)
print("Train size: {X_train.shape},Validation size:{X_val.shape}")


#creating a very base model 

rf = RandomForestRegressor(random_state=4294967295)
rf.fit(X_train,y_train)


#now lets do  predictions and evaluations
y_pred = rf.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_val, y_pred)




print(f"Model Performance:")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor


models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=4294967295),
    "Gradient Boosting": GradientBoostingRegressor(random_state=4294967295),
    "XGBoost": XGBRegressor(random_state=4294967295, n_estimators=200)
}


for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    rmse = (mean_squared_error(y_val, preds)) ** 0.5
    r2 = r2_score(y_val, preds)
    print(f"{name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")


import joblib, os

best_model = models["XGBoost"]
model_path = os.path.join(BASE_DIR, "models", "xgboost_model.pkl")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(best_model, model_path)
print(f"Best model saved at: {model_path}")



#prediction on test_data 

test = pd.read_csv(test_path)


#dropping the target col if accidently included 

if 'accident_risk' in test.columns:
    test = test.drop('accident_risk',axis=1)

# Predict using the best model
test_preds = best_model.predict(test)


# Create submission file
submission = pd.DataFrame({
    "id": range(len(test_preds)),
    "accident_risk": test_preds
})


submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"✅ Submission file created at: {submission_path}")





