import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('tableau-colorblind10') # mpl.style.available
import seaborn as sns
sns.set_style("white")

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin

random_state = 42
np.random.seed(random_state)

base_path = '/kaggle/input/prediction-interval-competition-ii-house-price/'
alpha = 0.1  # the specified competition alpha (i.e., 90% coverage)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv(base_path + 'dataset.csv', index_col='id', parse_dates=['sale_date'])
test = pd.read_csv(base_path + 'test.csv', index_col='id', parse_dates=['sale_date'])
print(f"Train: {train.shape[0]} rows, {train.shape[1]} columns")
print(f"Test: {test.shape[0]} rows, {test.shape[1]} columns")


train.head()


test.head()


train['sale_nbr'] = train['sale_nbr'].fillna(train['sale_nbr'].median())
test['sale_nbr'] = test['sale_nbr'].fillna(test['sale_nbr'].median())

# zoning was used to impute the missing values in subdivision and submarket due to the apparent relationship among them
def fill_subdivision(df):
    nan_idx = df.loc[df['subdivision'].isna()].index
    for idx in nan_idx:
        zoning = df.loc[idx, 'zoning']
        if pd.notna(zoning):
            try:
                df.loc[idx, 'subdivision'] = df.loc[df['zoning'] == zoning, 'subdivision'].mode()[0]
            except KeyError:
                df.loc[idx, 'subdivision'] = 'Unknown'
        else:
            df.loc[idx, 'subdivision'] = 'Unknown'
    return df

train = fill_subdivision(train)
test = fill_subdivision(test)

def fill_submarket(df):
    nan_idx = df.loc[df['submarket'].isna()].index
    for idx in nan_idx:
        zoning = df.loc[idx, 'zoning']
        if pd.notna(zoning):
            try:
                df.loc[idx, 'submarket'] = df.loc[df['zoning'] == zoning, 'submarket'].mode()[0]
            except KeyError:
                df.loc[idx, 'submarket'] = 'Unknown'
        else:
            df.loc[idx, 'submarket'] = 'Unknown'
    return df

train = fill_submarket(train)
test = fill_submarket(test)


def winkler_score(y_true, lower, upper, alpha=0.1, return_coverage=False):
    """Compute the Winkler Interval Score for prediction intervals.

    Args:
        y_true (array-like): True observed values.
        lower (array-like): Lower bounds of prediction intervals.
        upper (array-like): Upper bounds of prediction intervals.
        alpha (float): Significance level (e.g., 0.1 for 90% intervals).
        return_coverage (bool): If True, also return empirical coverage.

    Returns:
        score (float): Mean Winkler Score.
        coverage (float, optional): Proportion of true values within intervals.
    """
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


train_sorted = train.sort_values(by='sale_price')

val_proportion = .25
val_step = int(1/val_proportion)
val_indices = np.arange(np.random.randint(0, val_step), train_sorted.shape[0], val_step)

val = train_sorted.iloc[val_indices]
train = train_sorted.drop(index=val_indices).reset_index(drop=True)
val = val.reset_index(drop=True)


# Sale Date
train['year'] = train['sale_date'].dt.year
train['month'] = train['sale_date'].dt.month
val['year'] = val['sale_date'].dt.year
val['month'] = val['sale_date'].dt.month
test['year'] = test['sale_date'].dt.year
test['month'] = test['sale_date'].dt.month

# Sale Warning
train['sale_warning'] = train['sale_warning'].str.strip()
train['sale_warning'] = train['sale_warning'].str.split(' ')
val['sale_warning'] = val['sale_warning'].str.strip()
val['sale_warning'] = val['sale_warning'].str.split(' ')
test['sale_warning'] = test['sale_warning'].str.strip()
test['sale_warning'] = test['sale_warning'].str.split(' ')

train['sale_warning'] = train['sale_warning'].apply(lambda x: np.nan if x[0] == '' else x)
train['sale_warning_len'] = train['sale_warning'].str.len()
train['sale_warning_len'] = train['sale_warning_len'].fillna(0).astype(np.int64)
train['sale_warning'] = train['sale_warning'].apply(lambda x: x if isinstance(x, list) else [0])
val['sale_warning'] = val['sale_warning'].apply(lambda x: np.nan if x[0] == '' else x)
val['sale_warning_len'] = val['sale_warning'].str.len()
val['sale_warning_len'] = val['sale_warning_len'].fillna(0).astype(np.int64)
val['sale_warning'] = val['sale_warning'].apply(lambda x: x if isinstance(x, list) else [0])
test['sale_warning'] = test['sale_warning'].apply(lambda x: np.nan if x[0] == '' else x)
test['sale_warning_len'] = test['sale_warning'].str.len()
test['sale_warning_len'] = test['sale_warning_len'].fillna(0).astype(np.int64)
test['sale_warning'] = test['sale_warning'].apply(lambda x: x if isinstance(x, list) else [0])

train['sale_warning_len'] = train['sale_warning_len'].astype(np.int64)
val['sale_warning_len'] = val['sale_warning_len'].astype(np.int64)
test['sale_warning_len'] = test['sale_warning_len'].astype(np.int64)

# Year
train['year_built_sale'] = train['year'] - train['year_built']
val['year_built_sale'] = val['year'] - val['year_built']
test['year_built_sale'] = test['year'] - test['year_built']

def year_reno_sale(row):
    if row['year_reno'] > 0:
        return row['year'] - row['year_reno']
    else:
        return row['year_built_sale']

train['year_reno_sale'] = train.apply(year_reno_sale, axis=1)
val['year_reno_sale'] = val.apply(year_reno_sale, axis=1)
test['year_reno_sale'] = test.apply(year_reno_sale, axis=1)


X_train, X_val = train.drop(['sale_warning', "sale_price"], axis=1), val.drop(["sale_warning", "sale_price"], axis=1)
y_train, y_val = train["sale_price"], val["sale_price"]

X_test = test.drop(["sale_warning"], axis=1)

# Ordinal encoding.
cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
X_val[cat_cols] = encoder.transform(X_val[cat_cols])
X_test[cat_cols] = encoder.transform(X_test[cat_cols])

# Imputation.
num_cols = X_train.select_dtypes(include="number").columns.tolist()

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

def impute(df, cols, imputer, fit=False):
    """Helper function for imputation."""
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
    """Encode sale date as a week of the year feature."""

    def __init__(self, date_column="sale_date"):
        self.date_column = date_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.assign(
            **{
                "sale_week": lambda x: x["sale_date"].dt.isocalendar().week,
            }
        ).drop(columns=["sale_date"])
        return X

    def fit_transform(self, X):
        return self.fit(self, X).transform(X)

# Sale date encoding.
saledate_encoder = SaleDateEncoder(date_column="sale_date")
X_train = saledate_encoder.fit_transform(X_train)
X_val = saledate_encoder.transform(X_val)
X_test = saledate_encoder.transform(X_test)

# HISTGBR
hgbr_l = HistGradientBoostingRegressor(
    max_iter=1000,
    max_depth=5,
    max_leaf_nodes=31,
    min_samples_leaf=1,
    l2_regularization=0.1,
    learning_rate=0.1,
    random_state=random_state,
    loss="quantile",
    quantile=alpha/2
)
hgbr_l.fit(X_train, y_train)

hgbr_u = HistGradientBoostingRegressor(
    max_iter=1000,
    max_depth=5,
    max_leaf_nodes=31,
    min_samples_leaf=1,
    l2_regularization=0.1,
    learning_rate=0.1,
    random_state=random_state,
    loss="quantile",
    quantile=1 - alpha/2
)
hgbr_u.fit(X_train, y_train)

y_val_pred_l = hgbr_l.predict(X_val)
y_val_pred_u = hgbr_u.predict(X_val)

y_val_pred = pd.DataFrame({
    "pi_lower": y_val_pred_l,
    "pi_upper": y_val_pred_u
})

mws, coverage = winkler_score(
    y_val,
    y_val_pred["pi_lower"],
    y_val_pred["pi_upper"],
    alpha=alpha,
    return_coverage=True,
)

print("Mean Winkler Score:", round(mws, 2))
print("Coverage:", round(coverage * 100, 1), "%")

y_val_pred.reset_index(drop=True, inplace=True)
y_val.reset_index(drop=True, inplace=True)

nonconf_score = np.maximum(y_val_pred['pi_lower'] - y_val, y_val - y_val_pred['pi_upper'])

q = nonconf_score.quantile(1 - alpha)
print(q)

test_preds_l = hgbr_l.predict(X_test) - q
test_preds_u = hgbr_u.predict(X_test) + q
test_preds = pd.DataFrame({
    "pi_lower": test_preds_l,
    "pi_upper": test_preds_u
})

sample_submission = pd.read_csv(base_path + "sample_submission.csv")
sample_submission['pi_lower'] = test_preds["pi_lower"]
sample_submission['pi_upper'] = test_preds["pi_upper"]

sample_submission.to_csv("submission.csv", index=False)

