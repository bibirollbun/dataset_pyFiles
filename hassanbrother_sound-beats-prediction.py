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


# Standard library for handling code warnings
from warnings import filterwarnings

# Data handling
import pandas as pd
import numpy as np

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Model evaluation
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split

# Preprocessing
from sklearn.impute import SimpleImputer # Impute values
from sklearn.preprocessing import (
    StandardScaler,      # normalize features (mean=0, var=1)
    RobustScaler,        # scale features robust to outliers
    FunctionTransformer, # apply custom transformations
    PolynomialFeatures   # generate polynomial/interaction terms
)

# Column and pipeline utilities
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import make_pipeline, Pipeline

# Linear models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# ensemble models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

# Custom transformers
from sklearn.base import TransformerMixin, BaseEstimator

# Suppress warnings
filterwarnings('ignore')

# Seaborn theme 
sns.set_theme(style="whitegrid", palette="muted")

# Matplotlib defaults
plt.rcParams["figure.figsize"] = (15, 6)   # default figure size
plt.rcParams["font.size"] = 12             # default font size
plt.rcParams["axes.titlesize"] = 14        # title size
plt.rcParams["axes.labelsize"] = 12        # axis label size
plt.rcParams["legend.fontsize"] = 11       # legend size
plt.rcParams["figure.dpi"] = 300           # resolution


train_data = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv', index_col='id')
train_data.head()


train_data.info()


train_data.describe().T


fig, axes = plt.subplots(5, 2, figsize=(14, 18))
axes = axes.ravel()

for i, col in enumerate(train_data.select_dtypes(include="number").columns):
    ax = axes[i]
    data = train_data[col]
    

    sns.histplot(data, kde=True, ax=ax, color="purple")
    ax.set_title(col, fontsize=12)

    mean = data.mean()
    median = data.median()
    std = data.std()
    skew = data.skew()

    ax.axvline(mean, color="red", linestyle="--", linewidth=1.5, label="Mean")
    ax.axvline(median, color="blue", linestyle=":", linewidth=1.5, label="Median")
    
    ax.axvspan(mean-std, mean+std, color="purple", alpha=0.1, label="±1 Std")
    
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    textstr = (f"Mean: {mean:.2f}\n"
               f"Median: {median:.2f}\n"
               f"Std: {std:.2f}\n"
               f"Skew: {skew:.2f}")
    ax.text(0.98, 0.95, textstr, transform=ax.transAxes, 
            fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6))

plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 8))

corr = train_data.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(
    corr,
    mask=mask,
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f",
    linewidths=.5,
    cbar_kws={"shrink": .8, "label": "Correlation Strength"}
)

plt.title("Correlation Matrix", fontsize=16, weight="bold")
plt.show()


fig, axes = plt.subplots(5, 2, figsize=(14, 18))
axes = axes.ravel()

target = 'BeatsPerMinute'

for i, col in enumerate(train_data.columns):
    ax = axes[i]
    
    sns.scatterplot(x=train_data[col], y=train_data[target], ax=ax, color="purple", s=50, alpha=0.7)
    
    ax.set_xlabel(col)
    ax.set_ylabel(target)
    ax.set_title(f"{col} vs {target}", fontsize=12)
    break

plt.tight_layout()
plt.show()


class LogSkewedTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, skew_threshold=1.0):
        self.skew_threshold = skew_threshold
        self.skewed_cols_ = []

    def fit(self, X, y=None):
        self.skewed_cols_ = [col for col in X.columns if abs(X[col].skew()) >= self.skew_threshold]
        return self

    def transform(self, X):
        X_trans = X.copy()
        for col in self.skewed_cols_:
            X_trans[col] = X_trans[col].fillna(X_trans[col].median())
            X_trans[col] = np.log1p(X_trans[col])
        return X_trans


numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler())
])


d2_model = Pipeline([
    ('log_skew_trf', LogSkewedTransformer(skew_threshold=1.0)),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

X = train_data.drop('BeatsPerMinute', axis=1)
y = train_data['BeatsPerMinute']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

d2_scores = cross_val_score(d2_model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
print("CV RMSE Scores:", d2_scores)
print("Mean CV RMSE:", d2_scores.mean())
print("Std CV RMSE:", d2_scores.std())


d3_model = Pipeline([
    ('log_skew_trf', LogSkewedTransformer(skew_threshold=1.0)),
    ('poly', PolynomialFeatures(degree=3, include_bias=False)),
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

d3_scores = cross_val_score(d3_model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
print("CV RMSE Scores:", d3_scores)
print("Mean CV RMSE:", d3_scores.mean())
print("Std CV RMSE:", d3_scores.std())


rfr_model = Pipeline([
    ('log_skew_trf', LogSkewedTransformer(skew_threshold=1.0)),
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1))
])

rfr_scores = cross_val_score(rfr_model, X, y, cv=5, scoring='neg_root_mean_squared_error')
print("CV RMSE Scores:", rfr_scores)
print("Mean CV RMSE:", rfr_scores.mean())
print("Std CV RMSE:", rfr_scores.std())


xgb_model = Pipeline([
    ('log_skew_trf', LogSkewedTransformer(skew_threshold=1.0)),
    ('scaler', StandardScaler()),
    ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.05))
])

xgb_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
print("CV RMSE Scores:", xgb_scores)
print("Mean CV RMSE:", xgb_scores.mean())
print("Std CV RMSE:", xgb_scores.std())


abr_model = Pipeline([
    ('log_skew_trf', LogSkewedTransformer(skew_threshold=1.0)),
    ('scaler', StandardScaler()),
    ('regressor', AdaBoostRegressor(n_estimators=70))
])

abr_scores = cross_val_score(abr_model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
print("CV RMSE Scores:", abr_scores)
print("Mean CV RMSE:", abr_scores.mean())
print("Std CV RMSE:", abr_scores.std())


rfr_model.fit(X, y)

test_data = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv', index_col='id')


y_pred = rfr_model.predict(test_data)

submission = pd.DataFrame({
    "id": test_data.index,           
    "BeatsPerMinute": y_pred
})

submission.to_csv("submission(2).csv", index=False)




