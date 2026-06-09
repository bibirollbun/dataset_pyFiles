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


# Step 1: Import all required libraries for data handling, visualization, preprocessing, and modeling.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import xgboost as xgb








# Step 2: Load train, test, and sample submission files and inspect their structure.

train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
display(train.head())


# Step 3: Understand the dataset — check feature types, nulls, and target distribution.

print(train.info())
print("\nTotal missing values:\n", train.isnull().sum())

plt.figure(figsize=(6,4))
sns.histplot(train["accident_risk"], bins=40, kde=True)
plt.title("Target Distribution: Accident Risk")
plt.xlabel("Accident Risk")
plt.show()
#Insight:The target (accident_risk) lies between 0 and 1, so a regression model is appropriate.



# Step 4: Separate features (X) from labels (y) and prepare test data similarly.

X = train.drop(["id", "accident_risk"], axis=1)
y = train["accident_risk"]

X_test = test.drop(["id"], axis=1)
test_id = test["id"]


# Step 5: Build preprocessing pipelines for numeric and categorical columns.

num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(exclude=np.number).columns

numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

categorical_transformer = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)
#Insight:Scaling numeric features and encoding categorical variables ensures that the model handles both types efficiently.



# Step 6: Combine the preprocessor and model into one unified pipeline.

xgb_model = xgb.XGBRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    tree_method="hist"
)

model = Pipeline(steps=[("preprocessor", preprocessor),
                        ("regressor", xgb_model)])
#Insight:
#tree_method="hist" speeds up large dataset training.
#The pipeline simplifies the workflow — preprocessing is automatically applied before modeling.



# Step 7: Perform 5-Fold Cross-Validation to evaluate model stability.

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold+1}")
    
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    oof_preds[valid_idx] = preds
    
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    print(f"  Fold {fold+1} RMSE: {rmse:.5f}")
    
    # Averaged predictions for test data
    test_preds += model.predict(X_test) / kf.n_splits
#Insight:Cross-validation helps generalize performance by evaluating the model across different data splits.



# Step 8: Compute overall cross-validation RMSE and visualize performance.

overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"\nOverall CV RMSE: {overall_rmse:.5f}")

plt.figure(figsize=(6,4))
sns.scatterplot(x=y, y=oof_preds, alpha=0.4)
plt.title("Predicted vs Actual Risk")
plt.xlabel("Actual Accident Risk")
plt.ylabel("Predicted Accident Risk")
plt.show()
#Insight:The scatterplot checks how well predicted risks align with actual labels.
#A dense cluster around the diagonal line indicates strong performance.



# Step 9: Identify which features most influence predictions.

xgb_booster = model.named_steps["regressor"].get_booster()
importance_df = pd.DataFrame(
    xgb_booster.get_score(importance_type="gain").items(),
    columns=["Feature", "Importance"]
).sort_values("Importance", ascending=False)

importance_df.head(10)
#Insight:Most impactful features (like weather, curvature, surface condition) help improve interpretability and feature engineering.




# Step 10: Create and export the final submission CSV file for Kaggle.

submission = pd.DataFrame({
    "id": test_id,
    "accident_risk": test_preds
})
submission.to_csv("submission.csv", index=False)
print("Submission file generated successfully ✅")
#Insight:This file is directly upload-ready for Kaggle’s competition submission page.

