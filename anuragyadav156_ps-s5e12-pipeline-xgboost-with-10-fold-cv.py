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


from sklearn.model_selection import cross_validate, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


traindf = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
testdf = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
print(traindf.shape, testdf.shape)


traindf.info()


testdf.info()


print(traindf.duplicated().sum(),  "\n",  testdf.duplicated().sum())



import matplotlib.pyplot as plt
numeric_cols = traindf.select_dtypes(include=['int64', 'float64']).columns.drop("diagnosed_diabetes")
for col in numeric_cols:
    plt.figure(figsize=(5, 4))
    plt.boxplot(traindf[col].dropna())
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.show()



X = traindf.drop("diagnosed_diabetes", axis=1)
y = traindf["diagnosed_diabetes"]


# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Remove 'id' from numerical columns if it's present
if 'id' in numerical_cols:
    numerical_cols.remove('id')

#Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])


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
rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Cross-validation
cv = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_validate(
    pipeline,
    X, y,
    cv=cv,
    scoring={'RMSE': rmse_scorer, 'R2': 'r2'},
    return_train_score=False
)


# Results
rmse_scores = np.sqrt(-cv_scores['test_RMSE'])  # Convert negative MSE to RMSE
r2_scores = cv_scores['test_R2']

print("XGBoost Model Evaluation")
print(f" Avg RMSE: {rmse_scores.mean():.4f}")
print(f" Std Dev: {rmse_scores.std():.4f}")
print(f" Avg R²: {r2_scores.mean():.4f}")

# Fit the pipeline on full training data
pipeline.fit(X, y)

# Predict on test data
final_preds = pipeline.predict(testdf)

# Create submission file
submission = pd.DataFrame({
    'id': testdf['id'],
    'y': final_preds
})

submission.to_csv('submission.csv', index=False)
print("\n✅ Submission file 'submission.csv' created using XGBoost pipeline!")

