import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os
# XGBoost
from xgboost import XGBRegressor
# LightGBM
from lightgbm import LGBMRegressor
# CatBoost
from catboost import CatBoostRegressor
# Gradient Boosting from scikit-learn
from sklearn.ensemble import GradientBoostingRegressor
# AdaBoost Regressor
from sklearn.ensemble import AdaBoostRegressor
pd.set_option("display.max_columns",None)
%matplotlib inline


df=pd.read_csv("/kaggle/input/sweet-regression-prediction-of-chocolate-sales/data_training.csv")


df.drop(columns=["Id","id"],axis=1,inplace=True)


df.head()


df.shape


df.info()


df.isnull().sum()


df.describe()


df.head()


numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
fig, axes = plt.subplots(5, 5, figsize=(20, 20))
axes = axes.ravel()
for idx, col in enumerate(numerical_cols):
    if idx < len(axes):
        axes[idx].hist(df[col], bins=5, alpha=0.7, edgecolor='black')
        axes[idx].set_title(f'Histogram: {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
plt.tight_layout()
plt.show()


categorical_cols = ['Tone_of_Ad', 'Weather', 'Coffee_Consumption']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, col in enumerate(categorical_cols):
    df[col].value_counts().plot(kind='bar', ax=axes[idx], alpha=0.7, edgecolor='black')
    axes[idx].set_title(f'Bar Plot: {col}')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Count')
    axes[idx].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(5, 5, figsize=(20, 20))
axes = axes.ravel()
sales_col = 'sales'
for idx, col in enumerate(numerical_cols):
    if col != sales_col and idx < len(axes):
        axes[idx].scatter(df[col], df[sales_col], alpha=0.7)
        axes[idx].set_title(f'{col} vs. {sales_col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel(sales_col)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, col in enumerate(categorical_cols):
    sns.boxplot(x=col, y=sales_col, data=df, ax=axes[idx])
    axes[idx].set_title(f'{col} vs. {sales_col}')
    axes[idx].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 10))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', center=0, square=True, fmt='.2f')
plt.title('Correlation Heatmap (Numerical Features)')
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, col in enumerate(categorical_cols):
    df_grouped = df.groupby(col)['sales'].mean().reset_index()
    sns.barplot(x=col, y='sales', data=df_grouped, ax=axes[idx], alpha=0.7)
    axes[idx].set_title(f'Mean Sales by {col}')
    axes[idx].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 5))
plt.plot(df.index, df['sales'], marker='o', linewidth=2, markersize=8)
plt.title('Sales Trend Over Observations')
plt.xlabel('Observation Index')
plt.ylabel('Sales')
plt.grid(True, alpha=0.3)
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, col in enumerate(categorical_cols):
    df[col].value_counts().plot(kind='pie', ax=axes[idx], autopct='%1.1f%%', startangle=90)
    axes[idx].set_title(f'Pie: {col}')
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, col in enumerate(categorical_cols):
    sns.violinplot(x=col, y=sales_col, data=df, ax=axes[idx])
    axes[idx].set_title(f'Violin: {col} vs. {sales_col}')
    axes[idx].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()


df.head()


y = df["sales"]
X = df.drop(columns=["sales"],axis=1)


numeric_features = [
    "Web_GRP", "TV_GRP", "Facebook_GRP", "No_of_Web_Banners", "Avg_Temperature",
    "No_of_Rabbits", "No_of_iPhone_14_Sold", "No_of_Big_Cities",
    "Health_Index", "Sustainability_Index", "Choc_Capital_Distance",
    "No_of_Competitors", "Time_in_Region", "Percent_Internet_Access",
    "Percent_Uni_Degrees", "Percent_Unemployed", "Avg_No_of_Cust_Complaints",
    "Avg_Customer_Age"
]

categorical_features = ["Tone_of_Ad", "Weather"]
ordinal_features = ["Coffee_Consumption"]

binary_features = ["Network_Five_G", "Import_Regulations", "Gender"]


# Pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

ordinal_transformer = Pipeline(steps=[
    ("ordinal", OrdinalEncoder(categories=[["low", "medium", "high"]]))
])

binary_transformer = "passthrough" 


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


X.shape


preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features),("cat", categorical_transformer, categorical_features),
        ("ord", ordinal_transformer, ordinal_features),
        ("bin", binary_transformer, binary_features)])


model = Pipeline(steps=[("preprocessor", preprocessor),("regressor", GradientBoostingRegressor())])
model.fit(X_train, y_train)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

preds = model.predict(X_valid)

# Metrics
mae = mean_absolute_error(y_valid, preds)
mse = mean_squared_error(y_valid, preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_valid, preds)
mape = mean_absolute_percentage_error(y_valid, preds)

print("Validation Metrics:")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")
print(f"MAPE : {mape:.4f}")



plt.figure(figsize=(15, 6))
sns.scatterplot(x=y_valid, y=preds, alpha=0.6, edgecolor=None)
sns.regplot(x=y_valid, y=preds, scatter=False, color="red", line_kws={"linewidth":2})

min_val = min(y_valid.min(), preds.min())
max_val = max(y_valid.max(), preds.max())
plt.plot([min_val, max_val], [min_val, max_val], color="green", linestyle="--", linewidth=2, label="Perfect Fit")

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.legend()
plt.show()


test_df=pd.read_csv("/kaggle/input/sweet-regression-prediction-of-chocolate-sales/data_test.csv")


test_df.head()


X_test = test_df.drop(columns=["Id","id"],errors="ignore")
test_predictions = model.predict(X_test)



submission = pd.DataFrame({"Id": test_df["Id"],"Expected": test_predictions})


submission.to_csv("submission.csv", index=False)
print("Submission file saved!")


sub_df=pd.read_csv("/kaggle/working/submission.csv")


sub_df.head()


df.head()


y = df["sales"]
X = df.drop(columns=["sales"], errors="ignore")

numeric_features = [
    "Web_GRP", "TV_GRP", "Facebook_GRP", "No_of_Web_Banners",
    "Avg_Temperature", "No_of_Rabbits", "No_of_iPhone_14_Sold",
    "No_of_Big_Cities", "Health_Index", "Sustainability_Index",
    "Choc_Capital_Distance", "No_of_Competitors", "Time_in_Region",
    "Percent_Internet_Access", "Percent_Uni_Degrees", "Percent_Unemployed",
    "Avg_No_of_Cust_Complaints", "Avg_Customer_Age"
]
categorical_features = ["Tone_of_Ad", "Weather"]
ordinal_features = ["Coffee_Consumption"]
binary_features = ["Network_Five_G", "Import_Regulations", "Gender"]


X[numeric_features] = X[numeric_features].fillna(X[numeric_features].median())
X[categorical_features + ordinal_features] = X[categorical_features + ordinal_features].ffill()
X[binary_features] = X[binary_features].fillna(0)

X[numeric_features] = StandardScaler().fit_transform(X[numeric_features])
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

ordinal_encoder = OrdinalEncoder(categories=[["low","medium","high"]])
X[ordinal_features] = ordinal_encoder.fit_transform(X[ordinal_features].astype(str))

bool_cols = X.select_dtypes(include='bool').columns
X[bool_cols] = X[bool_cols].astype(float)



X


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


model = GradientBoostingRegressor()
model.fit(X_train, y_train)


y_pred = model.predict(X_valid)
print("MAE:", mean_absolute_error(y_valid, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_valid, y_pred)))
print("R2:", r2_score(y_valid, y_pred))


plt.figure(figsize=(15, 6))
sns.scatterplot(x=y_valid, y=y_pred, alpha=0.6, edgecolor=None)
sns.regplot(x=y_valid, y=y_pred, scatter=False, color="red", line_kws={"linewidth":2})

min_val = min(y_valid.min(), y_pred.min())
max_val = max(y_valid.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color="green", linestyle="--", linewidth=2, label="Perfect Fit")

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.legend()
plt.show()


test_df=pd.read_csv("/kaggle/input/sweet-regression-prediction-of-chocolate-sales/data_test.csv")


test_df.head()


X_test = test_df.drop(columns=["Id","id"], errors="ignore")
X_test[numeric_features] = X_test[numeric_features].fillna(X[numeric_features].median())
X_test[categorical_features + ordinal_features] = X_test[categorical_features + ordinal_features].ffill()
X_test[binary_features] = X_test[binary_features].fillna(0)

scaler = StandardScaler().fit(X[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

X_test = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)

for col in X.columns:
    if col not in X_test.columns:
        X_test[col] = 0

X_test = X_test[X.columns]

ordinal_encoder = OrdinalEncoder(categories=[["low","medium","high"]])
ordinal_encoder.fit(df[ordinal_features].astype(str))
X_test[ordinal_features] = ordinal_encoder.transform(X_test[ordinal_features].astype(str))
bool_cols = X_test.select_dtypes(include='bool').columns
X_test[bool_cols] = X_test[bool_cols].astype(float)


y_pred_test = model.predict(X_test)

# Create submission
submission = pd.DataFrame({"Id": test_df["Id"],"Expected": y_pred_test})

submission.to_csv("gb_submission.csv", index=False)


sub_gb=pd.read_csv("/kaggle/working/gb_submission.csv")
sub_gb.head()




