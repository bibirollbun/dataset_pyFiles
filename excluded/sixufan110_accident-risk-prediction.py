# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
df.info()


X = df.drop(['id', 'accident_risk'], axis=1)
y = df['accident_risk']

cat_features = X.select_dtypes(include='object').columns
num_features = X.select_dtypes(exclude='object').columns


# build data preprocessing pipeline

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), cat_features)
    ],
    remainder='passthrough'  # keep numeric features
)

X_encoded = preprocessor.fit_transform(X)


# fit a linear model

lr = LinearRegression()
lr.fit(X_encoded, y)
y_pred = lr.predict(X_encoded)
residuals = y - y_pred


# residual analysis

plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted accident risk')
plt.ylabel('Residuals')
plt.title('Residual Plot: Linear Regression')
plt.show()


# create dummy feature and interactions

from sklearn.preprocessing import PolynomialFeatures

X_baseline = pd.get_dummies(X, columns=cat_features, drop_first=True)

poly = PolynomialFeatures(
    degree=2, # pair all features
    interaction_only=True, # avoids squared terms
    include_bias=False # no constant column
)

X_inter_fitted = poly.fit_transform(X_baseline)

# create column names for interactions
feature_names = poly.get_feature_names_out(X_baseline.columns)

X_inter = pd.DataFrame(X_inter_fitted, columns=feature_names)


# compare baseline and interactions for better cv

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

# Baseline model
rf = RandomForestRegressor(random_state=42)
baseline = np.mean(cross_val_score(rf, X_baseline, y, cv=5, scoring='r2'))

# Model with interactions
rf_inter = RandomForestRegressor(random_state=42)
interaction = np.mean(cross_val_score(rf_inter, X_inter, y, cv=5, scoring='r2'))

print("Baseline R²:", baseline)
print("With Interactions R²:", interaction)




