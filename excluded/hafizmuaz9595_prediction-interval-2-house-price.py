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


!pip install -q quantile-forest 2>/dev/null


import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder



random_state = 0
np.random.seed(random_state)

base_path = "/kaggle/input/prediction-interval-competition-ii-house-price/"
alpha = 0.1



def interval_score(y_true, lower, upper, alpha=0.1, return_coverage=False):
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    width = upper - lower
    penalty_lower = 2 / alpha * (lower - y_true)
    penalty_upper = 2 / alpha * (y_true - upper)

    score = width.copy()
    score += np.where(y_true < lower, penalty_lower, 0)
    score += np.where(y_true > upper, penalty_upper, 0)

    if return_coverage:
        inside = (y_true >= lower) & (y_true <= upper)
        coverage = np.mean(inside)
        return np.mean(score), coverage

    return np.mean(score)



df = pd.read_csv(base_path + "dataset.csv", index_col="id", parse_dates=["sale_date"])
df_test = pd.read_csv(base_path + "test.csv", index_col="id", parse_dates=["sale_date"])


X = df.drop("sale_price", axis=1)
y = df["sale_price"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=random_state)
X_test = df_test.copy()



cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
X_val[cat_cols] = encoder.transform(X_val[cat_cols])
X_test[cat_cols] = encoder.transform(X_test[cat_cols])


num_cols = X_train.select_dtypes(include="number").columns.tolist()
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

def impute(df, cols, imputer, fit=False):
    if fit:
        return pd.DataFrame(imputer.fit_transform(df[cols]), columns=cols, index=df.index)
    else:
        return pd.DataFrame(imputer.transform(df[cols]), columns=cols, index=df.index)

X_train[num_cols] = impute(X_train, num_cols, num_imputer, fit=True)
X_val[num_cols] = impute(X_val, num_cols, num_imputer)
X_test[num_cols] = impute(X_test, num_cols, num_imputer)

X_train[cat_cols] = impute(X_train, cat_cols, cat_imputer, fit=True)
X_val[cat_cols] = impute(X_val, cat_cols, cat_imputer)
X_test[cat_cols] = impute(X_test, cat_cols, cat_imputer)


class SaleDateEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, date_column="sale_date"):
        self.date_column = date_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.assign(sale_week=X["sale_date"].dt.isocalendar().week)
        return X.drop(columns=["sale_date"])

saledate_encoder = SaleDateEncoder()
X_train = saledate_encoder.fit_transform(X_train)
X_val = saledate_encoder.transform(X_val)
X_test = saledate_encoder.transform(X_test)



qrf = RandomForestQuantileRegressor(
    n_estimators=100,
    max_features=0.333,
    max_samples_leaf=1,
    random_state=random_state,
)
qrf.fit(X_train, y_train)



quantiles = [alpha / 2, 1 - alpha / 2]
y_val_pred = qrf.predict(X_val, quantiles=quantiles)
y_val_pred = pd.DataFrame(y_val_pred, columns=["pi_lower", "pi_upper"])


# Heatmap of correlation matrix
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(20,10))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# Relationship between sale_price and some features
sns.scatterplot(data=df, x='sqft', y='sale_price')
plt.title("sqft vs sale_price")
plt.show()
sns.boxplot(data=df, x='grade', y='sale_price')
plt.title("grade vs sale_price")
plt.show()


quantiles


mws, coverage = interval_score(
    y_val, y_val_pred["pi_lower"], y_val_pred["pi_upper"],
    alpha=alpha, return_coverage=True
)

print("Mean Score:", round(mws, 2))
print("Coverage:", round(coverage * 100, 1), "%")



test_preds = qrf.predict(X_test, quantiles=quantiles)

sample_submission = pd.read_csv(base_path + "sample_submission.csv")
sample_submission["pi_lower"] = test_preds[:, 0]
sample_submission["pi_upper"] = test_preds[:, 1]

sample_submission.to_csv("submission.csv", index=False)
sample_submission.head()


