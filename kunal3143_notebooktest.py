import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import gc


df = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/train.csv', low_memory=False)


for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category')


df['team_name'] = df['team_name'].cat.add_categories('Unknown').fillna('Unknown')
df = df.drop(columns=['rider_name'], errors='ignore')


X = df.drop(columns=['Lap_Time_Seconds'])
y = df['Lap_Time_Seconds']


for col in X.select_dtypes(include=['category']).columns:
    X[col] = X[col].cat.codes


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42)



model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    eval_metric='rmse',          
    early_stopping_rounds=20, 
    # other parameters...
    random_state=42
)


model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],    # Keep eval_set in fit()
    verbose=False
)
del X_train, y_train
gc.collect()


preds = model.predict(X_val)
rmse = mean_squared_error(y_val, preds, squared=False)
print(f"XGBoost RMSE: {rmse:.4f}")

