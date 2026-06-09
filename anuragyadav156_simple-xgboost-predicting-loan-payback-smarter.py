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


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
!pip install XGBoost
import xgboost as xgb
!pip install lightgbm
import lightgbm as lgb
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
warnings.filterwarnings("ignore")
traindf = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
testdf = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
print(traindf.shape, testdf.shape)


traindf.sample(10)


traindf.info()


testdf.info()


traindf.describe(include = "all")


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate, KFold
print(traindf.duplicated().sum(),  "\n",  testdf.duplicated().sum())



numeric_cols = traindf.select_dtypes(include=['int64', 'float64']).columns.drop("loan_paid_back")

# Loop through each numerical column
for col in numeric_cols:
    Q1 = traindf[col].quantile(0.25)
    Q3 = traindf[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = traindf[(traindf[col] < lower_bound) | (traindf[col] > upper_bound)]
    print(f"{col}: {len(outliers)} outliers")
    outliers_for_test = testdf[(testdf[col] < lower_bound) | (testdf[col] > upper_bound)]
    print(f"{col}: {len(outliers_for_test)} outliers_for_test")


for col in numeric_cols:
    plt.figure(figsize=(5, 4))
    plt.boxplot(traindf[col].dropna())
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.show()


from sklearn.base import BaseEstimator, TransformerMixin
class IQRClipper(BaseEstimator, TransformerMixin):
    """Clips numerical columns to within 1.5*IQR of Q1 and Q3."""
    def __init__(self, factor=1.5):
        self.factor = factor
        self.bounds_ = {}

    def fit(self, X, y=None):
        # Compute bounds for each column
        X_df = pd.DataFrame(X)
        for col in X_df.columns:
            Q1 = X_df[col].quantile(0.25)
            Q3 = X_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - self.factor * IQR
            upper = Q3 + self.factor * IQR
            self.bounds_[col] = (lower, upper)
        return self

    def transform(self, X):
        # Apply clipping using fitted bounds
        X_df = pd.DataFrame(X).copy()
        for col, (lower, upper) in self.bounds_.items():
            X_df[col] = X_df[col].clip(lower, upper)
        return X_df.values  # return numpy array for sklearn compatibility



#Split features & target
X = traindf.drop("loan_paid_back", axis=1)
y = traindf["loan_paid_back"]



# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Remove 'id' from numerical columns if it's present
if 'id' in numerical_cols:
    numerical_cols.remove('id')

#Preprocessing pipelines
"""numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
"""
numeric_transformer = Pipeline(steps=[
    ('outlier_clip', IQRClipper(factor=1.5)),
    ('scaler', StandardScaler())
])


categorical_transformer = Pipeline(steps=[
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])


from xgboost import XGBRegressor
from sklearn.metrics import make_scorer
models = {
    "XGBoost": XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42,
        tree_method='hist',
        n_jobs=-1,
        objective='reg:squarederror'
    )
}

# Extract model from dictionary
xgb_model = models["XGBoost"]

# Build pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb_model)
])

# Define RMSE scorer
roc_scorer = make_scorer(roc_auc_score, greater_is_better=False)

# Cross-validation
cv = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_validate(
    pipeline,
    X, y,
    cv=cv,
    scoring={'ROC': roc_scorer},
    return_train_score=False
)

# Results
roc_scores = np.sqrt(-cv_scores['test_ROC'])  # Convert negative MSE to RMSE

print("XGBoost Model Evaluation")
print(f" Avg RMSE: {roc_scores.mean():.2f}")
print(f" Std Dev: {roc_scores.std():.2f}")

# Fit the pipeline on full training data
pipeline.fit(X, y)

# Predict on test data
final_preds = pipeline.predict(testdf)

# Create submission file
submission = pd.DataFrame({
    'id': testdf['id'],
    'loan_paid_back': final_preds
})

submission.to_csv('submission.csv', index=False)
print("submission file 'submission.csv' created using XGBoost pipeline!")



submission["loan_paid_back"] = submission["loan_paid_back"].round(2)
submission




