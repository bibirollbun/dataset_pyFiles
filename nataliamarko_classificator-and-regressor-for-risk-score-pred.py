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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, mean_absolute_error
import catboost
import xgboost as xgb
import seaborn as sns


import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
data.head()


# drop columns
df = data.drop(columns=["ID", "efs", "efs_time"], errors="ignore").copy()

# Identify numeric & categorical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns

# Fill missing values
df[num_cols] = df[num_cols].fillna(-1)
df[cat_cols] = df[cat_cols].fillna("missing")

# Convert object columns to category dtype for CatBoost
df[cat_cols] = df[cat_cols].astype("category")

df.head(2)



categ_features = cat_cols.to_list()

# Define Features & Targets
X_class = df.copy()
y_class = data["efs"].copy()  # Target for classification

X_reg = df[data["efs"] == 1].copy()  # Use only `efs = 1` for regression
y_reg = data.loc[data["efs"] == 1, "efs_time"].copy()  # Predict `efs_time`

# Split into train & test
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)



# Initialize CatBoostClassifier
clf = catboost.CatBoostClassifier(iterations=50, depth=8, learning_rate=0.1, random_seed=42, verbose=100)

# Train classifier
clf.fit(X_train_class, y_train_class, cat_features=categ_features, eval_set=(X_test_class, y_test_class))

# Predict
y_pred_class = clf.predict(X_test_class)

# Evaluate
acc = accuracy_score(y_test_class, y_pred_class)
print(f"Classification Accuracy: {acc:.4f}")



# Initialize CatBoostRegressor
reg = catboost.CatBoostRegressor(iterations=100, depth=10, learning_rate=0.1, random_seed=42, verbose=100)

# Train regressor only on `efs = 1` cases
reg.fit(X_train_reg, y_train_reg, cat_features=categ_features, eval_set=(X_test_reg, y_test_reg))
# Predict
y_pred_reg = reg.predict(X_test_reg)

# Evaluate
mae = mean_absolute_error(y_test_reg, y_pred_reg)
print(f"Regression MAE: {mae:.4f}")




p_efs = clf.predict_proba(X_test_class)[:, 1]  # Probability of event occurrence
t_efs_full = reg.predict(X_test_class)  # Predict on entire test set

# Ensure efs_time = 0 where probability is low (avoids shape mismatch)
t_efs_full[p_efs < 0.5] = 0  

# Compute Risk Score
max_efs_time = data["efs_time"].max()
risk_score = p_efs * np.log1p(max_efs_time - t_efs_full)  # Now, shapes match!

plt.figure(figsize=(8,5))
sns.scatterplot(x=t_efs_full, y=risk_score)
plt.xlabel("Predicted efs_time")
plt.ylabel("Risk Score")
plt.title("Risk Score vs. Predicted efs_time")
plt.show()



# Step 1: Get Predictions
p_efs = clf.predict_proba(X_test_class)[:, 1]  # Probability of event occurrence

# Step 2: Fix `t_efs` Predictions
t_efs_log = reg.predict(X_test_class)
t_efs_log = np.clip(t_efs_log, None, 6)  # Prevent extreme values before exponentiation
t_efs = np.expm1(t_efs_log)  # Convert back from log scale
t_efs = np.clip(t_efs, 0, max_efs_time)  # Keep within valid range

# Step 3: Apply Quadratic Risk Scaling
risk_score = p_efs * ((max_efs_time - t_efs) ** 2 / max_efs_time)

# Step 4: Visualize Again
plt.figure(figsize=(8,5))
sns.scatterplot(x=t_efs, y=risk_score)
plt.xlabel("Predicted efs_time")
plt.ylabel("Risk Score")
plt.title("Risk Score vs. Predicted efs_time (Quadratic Scaling)")
plt.show()



# X_class, y_class, X_reg, y_reg are already defined
# Initialize a new CatBoostClassifier
clf = catboost.CatBoostClassifier(iterations=50, depth=8, learning_rate=0.1, random_seed=42, verbose=100)
clf.fit(X_class, y_class, cat_features=categ_features)

# Initialize a new CatBoostRegressor
reg = catboost.CatBoostRegressor(iterations=100, depth=10, learning_rate=0.1, random_seed=42, verbose=100)
reg.fit(X_reg, y_reg, cat_features=categ_features)



test_data = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv") 
def test_preprocess(df):
    ids = df["ID"].copy()
    # drop columns
    df = df.drop(columns=["ID"], errors="ignore").copy()

    # Identify numeric & categorical columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Fill missing values
    df[num_cols] = df[num_cols].fillna(-1)
    df[cat_cols] = df[cat_cols].fillna("missing")
    
    # Convert object columns to category dtype for CatBoost
    df[cat_cols] = df[cat_cols].astype("category")
    return df, ids

X_test_final, ids = test_preprocess(test_data)
X_test_final


# Step 1: Get Predictions
p_efs = clf.predict_proba(X_test_final)[:, 1]  # Probability of event occurrence
t_efs = reg.predict(X_test_final)  # Predicted duration

# Step 2: Normalize `t_efs` Before Squaring
max_efs_time = data["efs_time"].max()
scaled_t_efs = (max_efs_time - t_efs) / max_efs_time  # Normalize to [0,1]

# Step 3: Compute Risk Score with Quadratic Scaling
risk_score = p_efs * (scaled_t_efs ** 2)  # Keeps values in [0,1]

# Step 6: Visualize Risk Score
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
sns.scatterplot(x=t_efs, y=risk_score)
plt.xlabel("Predicted efs_time")
plt.ylabel("Risk Score")
plt.title("Risk Score vs. Predicted efs_time (Fixed Scaling)")
plt.show()



risk_score


# submission
submission = pd.DataFrame({
    "ID": ids,
    "prediction": risk_score
})

submission.to_csv("/kaggle/working/submission.csv", index=False)





