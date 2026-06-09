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


"""
# Kaggle Playground Competition: Predicting Backpack Prices 

###  Introduction**
In this Kaggle competition, the goal is to predict backpack prices based on various attributes.
I used **XGBoost** to improve the score, optimizing hyperparameters to reduce memory consumption.

**Techniques used:**
- Data cleaning and preprocessing
- Encoding categorical variables
- Feature normalization
- Machine Learning Model: **Optimized XGBoost**
- Early stopping to prevent overfitting

"""

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb

#  Loading datasets
train_csv = "/kaggle/input/playground-series-s5e2/train.csv"
test_csv = "/kaggle/input/playground-series-s5e2/test.csv"
extra_training_csv = "/kaggle/input/playground-series-s5e2/training_extra.csv"


#  Reading data
df_train = pd.read_csv(train_csv)
df_train_extra = pd.read_csv(extra_training_csv)
df_test = pd.read_csv(test_csv)

#  Merging train and extra datasets
df_combined = pd.concat([df_train, df_train_extra], ignore_index=True).drop_duplicates()

#  Handling missing values
for col in ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]:
    df_combined[col].fillna(df_combined[col].mode()[0], inplace=True)
df_combined["Weight Capacity (kg)"].fillna(df_combined["Weight Capacity (kg)"].median(), inplace=True)

#  Encoding categorical variables
df_combined = pd.get_dummies(df_combined, columns=["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"], drop_first=True)

#  Splitting features and target
X = df_combined.drop(columns=['Price'])
y = df_combined['Price']

#  Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Training optimized XGBoost model
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    verbosity=1,
    early_stopping_rounds=10
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=10
)

#  Model evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"\n RMSE with Optimized XGBoost: {rmse}")

# ðŸ“Œ Generating submission file
df_test_clean = df_test.drop(columns=["id"])
df_test_clean = pd.get_dummies(df_test_clean, columns=["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"], drop_first=True)
missing_cols = set(X.columns) - set(df_test_clean.columns)
for col in missing_cols:
    df_test_clean[col] = 0  
df_test_clean = df_test_clean[X.columns]
df_test_scaled = scaler.transform(df_test_clean)

test_predictions = model.predict(df_test_scaled)
submission = pd.DataFrame({"id": df_test["id"], "Price": test_predictions})
submission.to_csv("submission.csv", index=False)

print("\n Submission file 'submission.csv' created successfully!")

"""
#  Conclusion
 XGBoost model improved the score compared to Random Forest.
 By optimizing hyperparameters, RMSE was reduced.
 The submission.csv file is ready for Kaggle!

 If you have suggestions or ideas for improvement, leave a comment! 
"""





