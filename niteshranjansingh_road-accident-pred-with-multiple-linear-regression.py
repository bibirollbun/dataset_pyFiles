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


train_df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


train_df.head()


train_df.shape, test_df.shape


train_df.info()


train_df.describe()





train_df.isnull().sum()


corr = train_df.corr(numeric_only = True)


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12,8))
sns.heatmap(corr,annot=True,cmap='coolwarm')
plt.show()


columns=['road_type', 'lighting', 'weather', 'time_of_day']
df_train_encoded = pd.get_dummies(train_df, columns = columns, drop_first = True)
for col in df_train_encoded.select_dtypes(include='bool').columns:
    df_train_encoded[col] = df_train_encoded[col].astype(int)
df_train_en = df_train_encoded.drop('id',axis=1 )    
df_train_en.info()


columns=['road_type', 'lighting', 'weather', 'time_of_day']
df_test_encoded = pd.get_dummies(test_df, columns = columns, drop_first = True)
for col in df_test_encoded.select_dtypes(include='bool').columns:
    df_test_encoded[col] = df_test_encoded[col].astype(int)
test_df_en = df_test_encoded.drop('id',axis=1 )    
test_df_en.info()


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(20,12))
sns.heatmap(df_train_en.corr(),annot=True,cmap='coolwarm')
plt.show()


### from sklearn.model_selection import train_test_split
"""
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Split train into training & validation
X = df_train_en.drop('accident_risk', axis=1)
y = df_train_en['accident_risk']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict on validation set
y_val_pred = model.predict(X_val_scaled)

# Evaluate
mse = mean_squared_error(y_val, y_val_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_val_pred)

print("Validation MSE:", mse)
print("Validation RMSE:", rmse)
print("Validation RÂ²:", r2)
"""


#from sklearn.model_selection import train_test_split

#X = df_train_en.drop('accident_risk', axis=1)
#y = df_train_en['accident_risk']

#X_train, X_val, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train features & target
X_train = df_train_en.drop('accident_risk', axis=1)
y_train = df_train_en['accident_risk']

# Test features
X_test = test_df_en.copy()


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit on train
X_test_scaled = scaler.transform(X_test)        # transform test


from sklearn.linear_model import LinearRegression
lin_model = LinearRegression()
lin_model.fit(X_train_scaled,y_train)
y_pred = lin_model.predict(X_test_scaled)


print(lin_model.intercept_,lin_model.coef_)


# Clip predictions to [0,1] just in case
y_pred_clipped = np.clip(y_pred, 0, 1)


import pandas as pd

submission = pd.DataFrame({
    'id': test_df['id'],         # id column from test set
    'accident_risk': y_pred_clipped  # predicted values
})



submission.to_csv('submission_0.csv', index=False)


test_df.shape


y_pred.shape


from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha=0.001)
lasso_model.fit(X_train_scaled,y_train)
y_pred_lasso = lasso_model.predict(X_test_scaled)
y_pred_clipped_lasso = np.clip(y_pred_lasso, 0, 1)
submission = pd.DataFrame({
    'id': test_df['id'],         # id column from test set
    'accident_risk': y_pred_clipped_lasso  # predicted values
})
submission.to_csv('submission_1.csv', index=False)


from sklearn.linear_model import LassoCV
lassoCV_model = LassoCV(cv=5,random_state=42)
lassoCV_model.fit(X_train_scaled,y_train)
y_pred_lassoCV = lassoCV_model.predict(X_test_scaled)
y_pred_clipped_lassoCV = np.clip(y_pred_lassoCV, 0, 1)
submission = pd.DataFrame({
    'id': test_df['id'],         # id column from test set
    'accident_risk': y_pred_clipped_lassoCV  # predicted values
})
submission.to_csv('submission_2.csv', index=False)


print(lassoCV_model.alpha_)


# ----------------------------
# 1ï¸�âƒ£ Import Libraries
# ----------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# ----------------------------
# 2ï¸�âƒ£ Load Data (replace with your files)
# ----------------------------
#train_df = pd.read_csv('train.csv')
#test_df = pd.read_csv('test.csv')

# ----------------------------
# 3ï¸�âƒ£ Encode categorical features (if not already encoded)
# ----------------------------
categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
test_df = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)

# ----------------------------
# 4ï¸�âƒ£ Separate features and target
# ----------------------------
X = train_df.drop('accident_risk', axis=1)
y = train_df['accident_risk']

# Align test set columns with train set
X_test = test_df[X.columns]

# ----------------------------
# 5ï¸�âƒ£ Train/Validation split
# ----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 6ï¸�âƒ£ Optional: Feature scaling (LightGBM can handle unscaled data, so optional)
# ----------------------------
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_val_scaled = scaler.transform(X_val)
# X_test_scaled = scaler.transform(X_test)

# ----------------------------
# 7ï¸�âƒ£ Train LightGBM
# ----------------------------
lgb_model = lgb.LGBMRegressor(
    n_estimators=5000,
    learning_rate=0.01,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
"""
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    early_stopping_rounds=100,
    verbose=100
)
"""
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(stopping_rounds=100)]  # instead of early_stopping_rounds
)

# ----------------------------
# 8ï¸�âƒ£ Evaluate on validation set
# ----------------------------
y_val_pred = lgb_model.predict(X_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"Validation RMSE: {rmse_val:.5f}")

# ----------------------------
# 9ï¸�âƒ£ Predict on test set
# ----------------------------
y_test_pred = lgb_model.predict(X_test)
y_test_pred_clipped = np.clip(y_test_pred, 0, 1)  # Ensure predictions are in [0,1]

# ----------------------------
# ğŸ”Ÿ Create submission CSV
# ----------------------------
submission = pd.DataFrame({
    'id': test_df['id'],
    'accident_risk': y_test_pred_clipped
})
submission.to_csv('submission_lgbm.csv', index=False)

print("Submission file saved: submission_lgbm.csv")





