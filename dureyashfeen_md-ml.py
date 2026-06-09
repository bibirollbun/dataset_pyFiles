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


# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import warnings
warnings.filterwarnings('ignore')
sns.set(style='whitegrid')



# load the data 
df_tr = pd.read_csv("/kaggle/input/molecular-machine-learning/train.csv")
df_ts = pd.read_csv("/kaggle/input/molecular-machine-learning/test.csv")
submission_file = pd.read_csv("/kaggle/input/molecular-machine-learning/sample_submission.csv")


df_tr.info()


df_tr.describe()


df_tr.columns


df_tr.isnull().sum()


# Outlier imputation function
def impute_outliers(df):
    df_out = df.copy()
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()
    if "T80" in numeric_cols:
        numeric_cols.remove("T80")  # exclude target from imputation
    for col in numeric_cols:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = (df_out[col] < lower) | (df_out[col] > upper)
        median = df_out[col].median()
        df_out.loc[outliers, col] = median
    return df_out

# Clean both train and test
df_tr = impute_outliers(df_tr)
df_ts = impute_outliers(df_ts)


plt.figure(figsize=(8,5))
sns.histplot(df_tr['T80'], bins=50, kde=True)
plt.title('Target Variable (T80) Distribution')
plt.show()


# Filter only numeric columns for correlation
numeric_df = df_tr.select_dtypes(include=[np.number])

# Now compute correlation safely
plt.figure(figsize=(14,10))
corr = numeric_df.corr()
sns.heatmap(corr[['T80']].sort_values(by='T80', ascending=False).head(30), annot=True, cmap='coolwarm')
plt.title('Top Correlated Features with T80')
plt.show()


# ====================================================
# ğŸ§¹ Preprocessing
# ====================================================
# Drop non-numeric or non-useful columns
df_tr = df_tr.drop(['Batch_ID', 'Smiles'], axis=1)

# Separate features and target
X = df_tr.drop('T80', axis=1)
y = df_tr['T80']
feature_columns = X.columns  # Save column names

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Process test data
df_ts = df_ts.drop(['Batch_ID', 'Smiles'], axis=1)
df_ts = df_ts[feature_columns]  # Use saved column names
X_test_scaled = scaler.transform(df_ts)


# Models # ğŸ¤– Model Training:
rf = RandomForestRegressor(n_estimators=200, random_state=42)
xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1)
lgb = LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42)

# Stacking
stacked_model = StackingRegressor(
    estimators=[('xgb', xgb), ('lgb', lgb)],
    final_estimator=rf,
    n_jobs=-1
)

# Train stacked model
stacked_model.fit(X_train, y_train)


val_preds = stacked_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
r2 = r2_score(y_val, val_preds)
print(f"\nValidation RMSE: {rmse:.4f}")
print(f"Validation R^2 Score: {r2:.4f}")


# Predict and save submission
submission_preds = stacked_model.predict(X_test_scaled)
submission_file["T80"] = submission_preds
submission_file.to_csv("submission.csv", index=False)




