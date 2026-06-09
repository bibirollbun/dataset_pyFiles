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


# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import warnings
warnings.filterwarnings("ignore")


# File paths (Kaggle format)
train_path = "/kaggle/input/playground-series-s5e10/train.csv"
test_path = "/kaggle/input/playground-series-s5e10/test.csv"


# Load data
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)


# Basic shape & info
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nTrain columns:", list(train.columns))
print("\nTrain Info:")
print(train.info())


print("\nTrain Data Types:\n", train.dtypes.value_counts())
print("\nMissing Values (Train):\n", train.isnull().sum()[train.isnull().sum() > 0])
print("\nMissing Values (Test):\n", test.isnull().sum()[test.isnull().sum() > 0])


# Drop ID from both (it’s just an identifier)
if "id" in train.columns:
    train.drop(columns=["id"], inplace=True)
if "id" in test.columns:
    test.drop(columns=["id"], inplace=True)


# Visualize missing values
plt.figure(figsize=(10,5))
msno.bar(train, color="skyblue")
plt.title("Missing Values in Train Data")
plt.show()


num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = train.select_dtypes(exclude=[np.number]).columns.tolist()

print("\nNumeric Columns:", num_cols)
print("\nCategorical Columns:", cat_cols)


print("\nSummary Statistics (Numeric):")
display(train[num_cols].describe().T)


target = "accident_risk"

plt.figure(figsize=(6,4))
sns.histplot(train[target], kde=True, bins=30, color='steelblue')
plt.title("Distribution of Accident Risk")
plt.xlabel("accident_risk")
plt.show()


print("\nAccident Risk Summary:")
print(train[target].describe())


train[num_cols].hist(bins=25, figsize=(15, 10), color='steelblue', edgecolor='black')
plt.suptitle("Numerical Feature Distributions")
plt.show()


plt.figure(figsize=(12,10))
corr = train[num_cols].corr()
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Heatmap")
plt.show()


high_corr = corr.unstack().sort_values(ascending=False)
high_corr = high_corr[(high_corr != 1) & (abs(high_corr) > 0.8)]
print("\nHighly correlated pairs:\n", high_corr.head(10))


common_cols = [c for c in train.columns if c in test.columns]

train_means = train[common_cols].mean(numeric_only=True)
test_means = test[common_cols].mean(numeric_only=True)
diff = (train_means - test_means).abs().sort_values(ascending=False)
print("\nTop mean differences between Train and Test:\n", diff.head(10))


from scipy import stats

z_scores = np.abs(stats.zscore(train[num_cols].select_dtypes(include=[np.number])))
outliers = (z_scores > 3).sum(axis=0)
print("\nOutliers per numerical feature:\n", outliers[outliers > 0])


# Boxplots for selected features
plt.figure(figsize=(15, 8))
for i, col in enumerate(num_cols[:6]):  # limit to 6 for readability
    plt.subplot(2, 3, i + 1)
    sns.boxplot(x=train[col], color='lightcoral')
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()


if len(cat_cols) > 0:
    for col in cat_cols:
        plt.figure(figsize=(6,4))
        sns.countplot(y=train[col], palette="pastel")
        plt.title(f"Distribution of {col}")
        plt.show()
else:
    print("\nNo categorical columns found.")


print("\nSkewness of numerical features:")
print(train[num_cols].skew().sort_values(ascending=False).head(10))

print("\nKurtosis of numerical features:")
print(train[num_cols].kurt().sort_values(ascending=False).head(10))


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import numpy as np
import pandas as pd


# Features and target
X = train.drop(columns=[target])
y = train[target]


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Identify numeric & categorical columns
num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()


# Preprocessing for numeric & categorical
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])


# # Function to train & evaluate
# def train_and_evaluate(model, X_tr, y_tr, X_vl, y_vl):
#     pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                                ('model', model)])
#     pipeline.fit(X_tr, y_tr)
#     y_pred = pipeline.predict(X_vl)
#     mse = mean_squared_error(y_vl, y_pred)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(y_vl, y_pred)
#     print(f"{model.__class__.__name__} -> RMSE: {rmse:.4f}, R2: {r2:.4f}")
#     return pipeline

def train_and_evaluate(model, X_tr, y_tr, X_vl, y_vl, tree_model=False):
    """
    Trains a model using a pipeline (preprocessing + model), evaluates it on validation set,
    and optionally prints feature importances for tree-based models.
    
    Parameters:
        model: sklearn or xgboost model
        X_tr, y_tr: training features and target
        X_vl, y_vl: validation features and target
        tree_model: bool, if True prints feature importances (only for tree-based models)
    
    Returns:
        pipeline: trained sklearn Pipeline object
    """
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    pipeline.fit(X_tr, y_tr)
    
    y_pred = pipeline.predict(X_vl)
    mse = mean_squared_error(y_vl, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_vl, y_pred)
    print(f"{model.__class__.__name__} -> RMSE: {rmse:.4f}, R2: {r2:.4f}")
    
    # Feature importances for tree models
    if tree_model:
        try:
            importances = pipeline.named_steps['model'].feature_importances_
            # Handle one-hot encoded features
            if cat_features:
                ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat']
                cat_feature_names = ohe.get_feature_names_out(cat_features)
                all_features = np.concatenate([num_features, cat_feature_names])
            else:
                all_features = num_features
            fi_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
            fi_df = fi_df.sort_values(by='Importance', ascending=False).head(10)
            print("\nTop Feature Importances:")
            display(fi_df)
        except Exception as e:
            print("Feature importance could not be displayed:", e)
    
    return pipeline


# Linear models
lin_reg = train_and_evaluate(LinearRegression(), X_train, y_train, X_val, y_val)
ridge = train_and_evaluate(Ridge(alpha=1.0), X_train, y_train, X_val, y_val)
lasso = train_and_evaluate(Lasso(alpha=0.01), X_train, y_train, X_val, y_val)


rf = train_and_evaluate(RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_split=10, random_state=42),
                        X_train, y_train, X_val, y_val, tree_model=True)


gbr = train_and_evaluate(GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=5, random_state=42),
                         X_train, y_train, X_val, y_val, tree_model=True)


xgbr = train_and_evaluate(xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=5,
                                           subsample=0.8, colsample_bytree=0.8, random_state=42),
                          X_train, y_train, X_val, y_val, tree_model=True)


# Reload test with IDs (since you dropped it earlier)
test_with_id = pd.read_csv(test_path)

# Use your best model, say XGBRegressor (replace with the model that gave best R2)
best_model = xgbr  # or rf, gbr, etc.

# Predict on test data
test_preds = best_model.predict(test)

# Create submission DataFrame
submission = pd.DataFrame({
    "id": test_with_id["id"],
    "accident_risk": test_preds
})

# Save to CSV
submission.to_csv("submission.csv", index=False)

print("✅ Submission file created: submission.csv")
submission.head()

