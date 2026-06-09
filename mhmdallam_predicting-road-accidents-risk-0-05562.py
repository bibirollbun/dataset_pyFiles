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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


train.head()


train.info()


train.describe().round(2)


train.duplicated().sum()



train.isnull().sum()


print("Numeric Col Names",train.select_dtypes(include=['number']).columns)
print("_"*90)
print("Categorical Col Names",train.select_dtypes(include=['object']).columns)
print("_"*90)
print("binary Col Names",train.select_dtypes(include=['bool']).columns)



num_col= ['num_lanes', 'curvature', 'speed_limit','num_reported_accidents']

target = 'accident_risk'

cat_col = ['road_type', 'lighting', 'weather', 'time_of_day']

bin_col = ['road_signs_present', 'public_road', 'holiday', 'school_season']


for col in cat_col:
    print(f"Unique categories in '{col}' column: {train[col].unique()}")
    print("*"*60)


weather_risk = train.groupby('weather')['accident_risk'].mean().sort_values(ascending=False)
print(f"\nHighest Risk Weather: {weather_risk.index[0]} (Avg Risk: {weather_risk.iloc[0]:.3f})")
print(f"Lowest Risk Weather: {weather_risk.index[-1]} (Avg Risk: {weather_risk.iloc[-1]:.3f})")


road_risk = train.groupby('road_type')['accident_risk'].mean().sort_values(ascending=False)
print(f"\nHighest Risk Road Type: {road_risk.index[0]} (Avg Risk: {road_risk.iloc[0]:.3f})")
print(f"Lowest Risk Road Type: {road_risk.index[-1]} (Avg Risk: {road_risk.iloc[-1]:.3f})")


time_risk = train.groupby('time_of_day')['accident_risk'].mean().sort_values(ascending=False)
print(f"\nHighest Risk Time: {time_risk.index[0]} (Avg Risk: {time_risk.iloc[0]:.3f})")
print(f"Lowest Risk Time: {time_risk.index[-1]} (Avg Risk: {time_risk.iloc[-1]:.3f})")


lighting_risk = train.groupby('lighting')['accident_risk'].mean().sort_values(ascending=False)
print(f"\nHighest Risk lighting: {lighting_risk.index[0]} (Avg Risk: {lighting_risk.iloc[0]:.3f})")
print(f"Lowest Risk lighting: {lighting_risk.index[-1]} (Avg Risk: {lighting_risk.iloc[-1]:.3f})")


for col in cat_col:
    train[col] = train[col].astype('category')


train.info()


features = ['road_type','num_lanes','curvature','speed_limit','lighting','weather',
            'road_signs_present','public_road','time_of_day','holiday','school_season',
            'num_reported_accidents']



X = train[features]
y = train[target]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=42)


from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


params = {
    "tree_method": "hist",
    "device": "cuda",
    "learning_rate": 0.0126,
    "n_estimators": 803,
    "max_depth": 11,
    "subsample": 0.801,
    "colsample_bytree": 0.813,
    "reg_alpha": 1.60,
    "reg_lambda": 7.52,
    "verbosity": 1,
    "eval_metric": "rmse",
    "random_state": 42,
    "enable_categorical": True
}
models = {
         'XGBoost': XGBRegressor(**params),
    
}
results = {}


for name, model in models.items():  
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(Y_test, y_pred)
    mse = mean_squared_error(Y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, y_pred)
    
    # Store results
    results[name] = {  
        'model': model,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'predictions': y_pred
    }
    
    print(f"{name:20} | R2: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")


plt.figure(figsize=(8,6))
plt.scatter(Y_test, y_pred, alpha=0.3)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Accident Risk")
plt.ylabel("Predicted Accident Risk")
plt.title("Predicted vs Actual - Test Set")
plt.show()


test = test.drop(columns=["id"])

for col in cat_col:
    test[col] = test[col].astype('category')


y_test_pred = model.predict(test)




submission = pd.DataFrame({
    "id": submission.id,          
    "accident_risk": y_test_pred    
})


submission.to_csv("sub1.csv", index=False)





