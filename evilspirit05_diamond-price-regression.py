import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
import os
from tqdm import tqdm
from tabulate import tabulate
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# XGBoost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
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


df=pd.read_csv("/kaggle/input/predicting-the-price-of-diamond/train.csv")


df.drop(columns=["id"],axis=1,inplace=True)


df.head()


df["cut"] = df["cut"].apply(lambda x: x if x in ["Ideal", "Premium", "Very Good"] else "Very Good")


df["cut"].value_counts()


top_colors = ["G", "E", "F", "H", "D"]
df["color"] = df["color"].apply(lambda x: x if x in top_colors else "D")


df["color"].value_counts()


top_clarity = ["SI1", "VS2", "SI2", "VS1", "VVS2"]

# Combine the rest into "VS1" (last of top 5)
df["clarity"] = df["clarity"].apply(lambda x: x if x in top_clarity else "VVS2")


df["clarity"].value_counts()


df.head()


X = df.drop(columns=["price"])
y = df["price"]


numeric = X.select_dtypes(include=["int64","float64"]).columns.tolist()
cat = X.select_dtypes(include=["object"]).columns.tolist()

scaler = MinMaxScaler()
X[numeric] = scaler.fit_transform(X[numeric])

ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[cat] = ordinal_encoder.fit_transform(X[cat])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model_final = CatBoostRegressor(iterations=1000, learning_rate=0.005, depth=6, random_state=42, verbose=500)
model_final.fit(X_train, y_train)


y_test_pred = model_final.predict(X_test)
print("\nTest Set Metrics:")
print("R2:", r2_score(y_test, y_test_pred))
print("MAE:", mean_absolute_error(y_test, y_test_pred))
print("MSE:", mean_squared_error(y_test, y_test_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))


test_df=pd.read_csv("/kaggle/input/predicting-the-price-of-diamond/test.csv")


test_df.head()


test_df.shape


test_df["cut"] = test_df["cut"].apply(lambda x: x if x in ["Ideal", "Premium", "Very Good"] else "Very Good")
top_colors = ["G", "E", "F", "H", "D"]
test_df["color"] = test_df["color"].apply(lambda x: x if x in top_colors else "D")
top_clarity = ["SI1", "VS2", "SI2", "VS1", "VVS2"]
test_df["clarity"] = test_df["clarity"].apply(lambda x: x if x in top_clarity else "VVS2")


X_test_copy = test_df.drop(columns=["id"]).copy()
X_test_copy[numeric] = scaler.transform(X_test_copy[numeric])
X_test_copy[cat] = ordinal_encoder.transform(X_test_copy[cat])


predictions = model_final.predict(X_test_copy)
submission = pd.DataFrame({"id": test_df["id"], "price": predictions})
submission.to_csv("submission.csv", index=False)
print("✅ Submission file created")


df=pd.read_csv("/kaggle/input/predicting-the-price-of-diamond/train.csv")


df.head()


df.drop(columns=["id"],axis=1,inplace=True)


df.isnull().sum()


df["cut"].value_counts()


df["color"].value_counts()


df["clarity"].value_counts()


df = df[df['cut'] != 'D']


valid_cuts = ['Ideal', 'Premium', 'Very Good', 'Good', 'Fair']
valid_colors = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
valid_clarities = ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1']
df = df[df['cut'].isin(valid_cuts) & df['color'].isin(valid_colors) & df['clarity'].isin(valid_clarities)]


for col in ['carat', 'depth', 'table', 'x', 'y', 'z', 'price']:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df[col] = df[col].clip(lower_bound, upper_bound)


cut_map = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
color_map = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}
clarity_map = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}

df['cut'] = df['cut'].map(cut_map)
df['color'] = df['color'].map(color_map)
df['clarity'] = df['clarity'].map(clarity_map)

# Frequency encoding
cut_freq = df['cut'].value_counts(normalize=True)
color_freq = df['color'].value_counts(normalize=True)
clarity_freq = df['clarity'].value_counts(normalize=True)
df['cut_freq'] = df['cut'].map(cut_freq)
df['color_freq'] = df['color'].map(color_freq)
df['clarity_freq'] = df['clarity'].map(clarity_freq)


# Interaction features
df['cut_clarity'] = df['cut'] * df['clarity']
df['cut_color'] = df['cut'] * df['color']
df['color_clarity'] = df['color'] * df['clarity']

# Step 3: Numerical Feature Transformations
# Volume (handle zero values)
df['volume'] = df['x'] * df['y'] * df['z']
df['volume'] = df['volume'].replace(0, df['volume'].median())

# Ratios
df['depth_to_table'] = df['depth'] / df['table'].replace(0, df['table'].median())
df['carat_per_volume'] = df['carat'] / df['volume']
df['xy_ratio'] = df['x'] / df['y'].replace(0, df['y'].median())

# Shape approximation
df['is_round'] = (df['xy_ratio'].between(0.95, 1.05)).astype(int)

# Polynomial features
df['carat_squared'] = df['carat'] ** 2
df['volume_squared'] = df['volume'] ** 2
df['depth_squared'] = df['depth'] ** 2

# Log transformation
df['log_carat'] = np.log1p(df['carat'])


df.head()


df.shape


# Define numerical and categorical columns
numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z', 'price', 'cut_freq', 'color_freq', 
                  'clarity_freq', 'volume', 'depth_to_table', 'carat_per_volume', 'xy_ratio', 
                  'carat_squared', 'volume_squared', 'depth_squared', 'log_carat']
categorical_cols = ['cut', 'color', 'clarity', 'cut_clarity', 'cut_color', 'color_clarity', 'is_round']

# Filter columns to ensure they exist in the dataset
numerical_cols = [col for col in numerical_cols if col in df.columns]
categorical_cols = [col for col in categorical_cols if col in df.columns]


n_cols = 4  # Number of columns in subplot grid
n_rows = int(np.ceil(len(numerical_cols) / n_cols))  # Calculate rows needed
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3), constrained_layout=True)
axes = axes.flatten()  # Flatten for easier indexing

for i, col in enumerate(numerical_cols):
    sns.histplot(df[col], kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')

# Remove empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.show()


print("\nPlotting Count Plots for Categorical Features...")
n_cols = 3
n_rows = int(np.ceil(len(categorical_cols) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3), constrained_layout=True)
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    sns.countplot(x=col, data=df, ax=axes[i])
    axes[i].set_title(f'Count Plot of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')
    axes[i].tick_params(axis='x', rotation=45)

# Remove empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])


plt.show()


print("\nPlotting Scatter Plots vs Price...")
n_cols = 4
n_rows = int(np.ceil((len(numerical_cols) - 1) / n_cols))  # Exclude 'price'
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3), constrained_layout=True)
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    if col != 'price':
        sns.scatterplot(x=col, y='price', data=df, ax=axes[i])
        # Highlight position 1
        axes[i].scatter(df.iloc[1][col], df.iloc[1]['price'], color='red', s=100, label='Position 1')
        axes[i].set_title(f'{col} vs Price')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Price')
        axes[i].legend()

# Remove empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.show()


print("\nPlotting Box Plots for Categorical Features vs Price...")
n_cols = 3
n_rows = int(np.ceil(len(categorical_cols) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3), constrained_layout=True)
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    sns.boxplot(x=col, y='price', data=df, ax=axes[i])
    axes[i].set_title(f'{col} vs Price')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Price')
    axes[i].tick_params(axis='x', rotation=45)

# Remove empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.savefig('/kaggle/working/box_price.png')
plt.show()


print("\nPlotting Correlation Heatmap...")
plt.figure(figsize=(15, 8))
corr_matrix = df[numerical_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()


scaler = StandardScaler()
continuous_features = ['carat', 'depth', 'table', 'x', 'y', 'z', 'volume', 'depth_to_table', 
                       'carat_per_volume', 'xy_ratio', 'log_carat', 'carat_squared', 
                       'volume_squared', 'depth_squared']
df[continuous_features] = scaler.fit_transform(df[continuous_features])


X = df.drop(['price'], axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


import xgboost as xgb
model = CatBoostRegressor(iterations=1000,learning_rate=0.005,depth=6,random_seed=42,verbose=300)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nTest RMSE: {rmse:.4f}")
print(f"Test R2 Score: {r2:.4f}")


test_df=pd.read_csv("/kaggle/input/predicting-the-price-of-diamond/test.csv")


test_df.head()


test_df.shape


test_df["cut"].value_counts()


test_df["color"].value_counts()


test_df["clarity"].value_counts()


valid_cuts = ['Ideal', 'Premium', 'Very Good', 'Good', 'Fair']
valid_colors = ['D', 'E', 'F', 'G', 'H', 'I', 'J',"VVS2","SI1"]
valid_clarities = ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1',"VS3","F","SI3","VVS3"]
test_df = test_df[test_df['cut'].isin(valid_cuts) & test_df['color'].isin(valid_colors) & test_df['clarity'].isin(valid_clarities)]


for col in ['carat', 'depth', 'table', 'x', 'y', 'z']:
    q1, q3 = test_df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    test_df[col] = test_df[col].clip(lower_bound, upper_bound)


cut_map = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
color_map = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7,"VVS2":8,"SI1":9}
clarity_map = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8,"VS3":9,"F":10,
              "SI3":11,"VVS3":12}


test_df['cut'] = test_df['cut'].map(cut_map)
test_df['color'] = test_df['color'].map(color_map)
test_df['clarity'] = test_df['clarity'].map(clarity_map)


cut_freq = test_df['cut'].value_counts(normalize=True)
color_freq = test_df['color'].value_counts(normalize=True)
clarity_freq = test_df['clarity'].value_counts(normalize=True)
test_df['cut_freq'] = test_df['cut'].map(cut_freq)
test_df['color_freq'] = test_df['color'].map(color_freq)
test_df['clarity_freq'] = test_df['clarity'].map(clarity_freq)

test_df['cut_clarity'] = test_df['cut'] * test_df['clarity']
test_df['cut_color'] = test_df['cut'] * test_df['color']
test_df['color_clarity'] = test_df['color'] * test_df['clarity']


test_df['volume'] = test_df['x'] * test_df['y'] * test_df['z']
test_df['volume'] = test_df['volume'].replace(0, test_df['volume'].median())

# Ratios
test_df['depth_to_table'] = test_df['depth'] / test_df['table'].replace(0, test_df['table'].median())
test_df['carat_per_volume'] = test_df['carat'] / test_df['volume']
test_df['xy_ratio'] = test_df['x'] / test_df['y'].replace(0, test_df['y'].median())

# Shape approximation
test_df['is_round'] = (test_df['xy_ratio'].between(0.95, 1.05)).astype(int)

# Polynomial features
test_df['carat_squared'] = test_df['carat'] ** 2
test_df['volume_squared'] = test_df['volume'] ** 2
test_df['depth_squared'] = test_df['depth'] ** 2

# Log transformation
test_df['log_carat'] = np.log1p(test_df['carat'])



test_df[continuous_features] = scaler.transform(test_df[continuous_features])


test_df.shape


Id=test_df.id


test_df.drop(columns=["id"],axis=1,inplace=True)


pred=model.predict(test_df)
sub=pd.DataFrame({"id":Id,"price":pred})
sub.to_csv("3rd_submission.csv",index=False)




