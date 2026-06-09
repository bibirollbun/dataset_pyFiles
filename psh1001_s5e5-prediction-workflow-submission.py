import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_squared_log_error,
    r2_score
)
import lightgbm as lgb

# FEATURE ENGINEERING (basic) 
df=pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df = df.drop(columns=['id'])  

# 1.2 Targetâ€�encode Sex by mean Calories
sex_mean = df.groupby('Sex')['Calories'].mean()
df['Sex_enc'] = df['Sex'].map(sex_mean)
df = df.drop(columns=['Sex'])

# 1.3 Create BMI feature (Weight in kg / Height in mÂ²)
df['Height_m'] = df['Height'] / 100
df['BMI'] = df['Weight'] / (df['Height_m'] ** 2)

# 1.4 (Optional) Interaction: Duration Ã— Heart_Rate
df['Dur_x_HR'] = df['Duration'] * df['Heart_Rate']

# 1.5 Drop intermediate columns if desired
df = df.drop(columns=['Height_m'])

# TRAIN/TEST SPLIT
X = df.drop(columns=['Calories'])
y = df['Calories']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.005, random_state=42
)

# LOG-TRANSFORM TARGET (for RMSLE)
y_train_log = np.log1p(y_train)
y_test_log  = np.log1p(y_test)

#  LIGHTGBM DATASETS 
lgb_train = lgb.Dataset(X_train, label=y_train_log)
lgb_test  = lgb.Dataset(X_test,  label=y_test_log, reference=lgb_train)

# TRAIN MODEL
params = {
    'objective': 'regression',
    'metric':    'rmse',
    'learning_rate': 0.018,
    'max_depth':     8,
    'num_leaves':    64,
    'feature_fraction': 0.9,
    'bagging_fraction':  0.9,
    'bagging_freq':      1,
    'device_type': 'gpu',      # use GPU
    'gpu_platform_id': 0,
    'gpu_device_id':   0,
    'verbose':  -1
}

# 2. Prepare datasets as before
lgb_train = lgb.Dataset(X_train, label=y_train_log)
lgb_test  = lgb.Dataset(X_test,  label=y_test_log, reference=lgb_train)

# 3. Train with callbacks for early stopping and logging
model = lgb.train(
    params,
    lgb_train,
    num_boost_round=1200,
    valid_sets=[lgb_test],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=50)
    ]
)

# 4. Predict & inverse log-transform
y_pred_log = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = np.expm1(y_pred_log)
y_pred = np.maximum(y_pred, 0)

# 5. Evaluate
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_squared_log_error,
    r2_score
)
import numpy as np

mse   = mean_squared_error(y_test, y_pred)
rmse  = np.sqrt(mse)
mae   = mean_absolute_error(y_test, y_pred)
rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
r2    = r2_score(y_test, y_pred)

print(f"MSE:   {mse:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"MAE:   {mae:.4f}")
print(f"RMSLE: {rmsle:.4f}")
print(f"RÂ²:    {r2:.4f}")


dft=pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
dft_orig = dft.copy()

dft = dft.drop(columns=['id'])  # drop raw id
dft['Sex_enc'] = dft['Sex'].map(sex_mean)
dft = dft.drop(columns=['Sex'])
dft['Height_m'] = dft['Height'] / 100
dft['BMI']      = dft['Weight'] / (dft['Height_m'] ** 2)
dft['Dur_x_HR'] = dft['Duration'] * dft['Heart_Rate']
dft = dft.drop(columns=['Height_m'])
y_pred_log = model.predict(dft, num_iteration=model.best_iteration)
y_pred     = np.expm1(y_pred_log).clip(min=0)
submission = pd.DataFrame({
    'id':       dft_orig['id'],
    'Calories': y_pred
})
submission.to_csv('submission.csv', index=False)
print("Wrote submission.csv with", len(submission), "rows")





