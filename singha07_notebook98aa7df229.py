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


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


train_df.sample(5)


train_df.info()


cat_cols = train_df.select_dtypes(include = ['object','bool']).columns


cat_cols


for col in cat_cols:
    
    print(train_df[col].value_counts())
    print('-' * 30)


train_df.duplicated().sum()


train_df.drop(columns = ['id'],inplace = True)
id = test_df['id'].copy()
test_df.drop(columns = ['id'], inplace = True)


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()

for col in train_df.select_dtypes(include = ['object','bool']).columns:
    train_df[col] = lb.fit_transform(train_df[col])

for col in test_df.select_dtypes(include = ['object','bool']).columns:
    test_df[col] = lb.fit_transform(test_df[col])


train_df.sample(5)


corr = train_df.corr()

# Plot the heatmap
plt.figure(figsize=(15,10))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap", fontsize=16)
plt.show()


X = train_df.drop(columns = ['accident_risk'])
y = train_df['accident_risk']


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)


from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import xgboost as xgb

xg = xgb.XGBRegressor(
    n_estimators=300,        # more trees → better learning (but may overfit)
    learning_rate=0.05,      # smaller learning rate → slower, more stable learning
    max_depth=6,             # control tree depth (default=6)
    subsample=0.8,           # use 80% of samples for each tree → prevents overfitting
    colsample_bytree=0.8,    # use 80% of features for each tree
    random_state=42,
    reg_lambda=1.0,          # L2 regularization
    reg_alpha=0.1,           # L1 regularization
    gamma=0.1                # minimum loss reduction for further partition
)
xg.fit(x_train, y_train)

y_pred = xg.predict(x_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")



from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import xgboost as xgb

# Base XGBoost model
xg = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    reg_lambda=1.0,
    reg_alpha=0.1,
    gamma=0.1
)

# Bagging ensemble on top of XGBoost
bagging_xg = BaggingRegressor(
    base_estimator=xg,     # The base model
    n_estimators=10,       # Number of bootstrapped models
    max_samples=0.8,       # 80% of samples per estimator
    max_features=1.0,      # Use all features
    bootstrap=True,        # Sampling with replacement
    n_jobs=-1,             # Use all CPU cores
    random_state=42
)

# Fit the ensemble
bagging_xg.fit(x_train, y_train)

# Predictions
y_pred = bagging_xg.predict(x_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ Bagging XGBoost Performance:")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")



xg.fit(X,y)


y_pred = xg.predict(test_df)


submission = pd.DataFrame({
    'id': id,
    'accident_risk': y_pred
})



submission.head(5)


submission.to_csv('submission.csv', index=False)


