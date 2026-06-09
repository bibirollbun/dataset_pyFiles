!pip install -q quantile-forest 2>/dev/null  # package for quantile regression forests


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold


import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import warnings
warnings.filterwarnings("ignore")
import lightgbm as lgb
import xgboost as xgb

random_state = 0
np.random.seed(random_state)

# Competition variables.
base_path = "/kaggle/input/prediction-interval-competition-ii-house-price/"
alpha = 0.1  # the specified competition alpha (i.e., 90% coverage)


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


df = pd.read_csv(base_path + "dataset.csv", index_col="id", parse_dates=["sale_date"])
df_test = pd.read_csv(base_path + "test.csv", index_col="id", parse_dates=["sale_date"])


# Split features and target.
X = df.drop("sale_price", axis=1)
y = df["sale_price"]
# Split train/val and test.
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=random_state
)
X_test = df_test.copy()


# Ordinal encoding.
cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
X_val[cat_cols] = encoder.transform(X_val[cat_cols])
X_test[cat_cols] = encoder.transform(X_test[cat_cols])


# Imputation.
num_cols = X_train.select_dtypes(include="number").columns.tolist()

num_imputer = SimpleImputer(strategy="mean")
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
                "month": lambda x: x['sale_date'].dt.month,
                "year": lambda x: x['sale_date'].dt.year,
                'day': lambda x: x['sale_date'].dt.weekday
                
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


#Huấn luyện mô hình QRF
qrf = RandomForestQuantileRegressor(
    n_estimators=100,
    max_features=0.333,
    max_samples_leaf=1,
    random_state=random_state,
)
qrf.fit(X_train, y_train)


#Huấn luyện mô hình LightGBM
def train_quantile_model_lgb(alpha):
    model = lgb.LGBMRegressor(objective='quantile',
                              alpha=alpha, n_estimators=1500, 
                              device='gpu')
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='quantile',
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    return model

model_lgb_lower = train_quantile_model_lgb(0.05)
model_lgb_upper = train_quantile_model_lgb(0.95)


#Huấn luyện mô hình XGBoost
def train_quantile_model_xgb(alpha):
    params_xgb = {
        'objective': 'reg:quantileerror',
        'quantile_alpha':alpha,
        'learning_rate': 0.05,
        'max_depth': 8,              
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 3000,
        'random_state': 42,
        'tree_method' : 'gpu_hist'
    }
    model = xgb.XGBRegressor(**params_xgb)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=1000,
        verbose=100
     )
    return model

model_xgb_lower = train_quantile_model_xgb(0.05)
model_xgb_upper = train_quantile_model_xgb(0.95)


quantiles = [alpha / 2, 1 - alpha / 2]

y_val_pred = qrf.predict(X_val, quantiles=quantiles)
y_val_pred = pd.DataFrame(y_val_pred, columns=["pi_lower", "pi_upper"])

mws, coverage = winkler_score(
    y_val,
    y_val_pred["pi_lower"],
    y_val_pred["pi_upper"],
    alpha=alpha,
    return_coverage=True,
)

print('Kết quả đánh giá mô hình QRF:')
print("Mean Winkler Score:", round(mws, 2))
print("Coverage:", round(coverage * 100, 1), "%")


val_lgb_lower = model_lgb_lower.predict(X_val)
val_lgb_upper = model_lgb_upper.predict(X_val)

mws, coverage = winkler_score(
    y_val,
    val_lgb_lower,
    val_lgb_upper,
    alpha=alpha,
    return_coverage=True,
)

print("Kết quả đánh giá mô hình LightGBM:")
print("Mean Winkler Score:", round(mws, 2))
print("Coverage:", round(coverage * 100, 1), "%")


val_xgb_lower = model_xgb_lower.predict(X_val)
val_xgb_upper = model_xgb_upper.predict(X_val)

mws, coverage = winkler_score(
    y_val,
    val_xgb_lower,
    val_xgb_upper,
    alpha=alpha,
    return_coverage=True,
)

print("Kết quả đánh giá mô hình XGBoost:")
print("Mean Winkler Score:", round(mws, 2))
print("Coverage:", round(coverage * 100, 1), "%")


# Predict intervals on test set.
test_lower = model_lgb_lower.predict(X_test)
test_upper = model_lgb_upper.predict(X_test)

sample_submission = pd.read_csv(base_path + "sample_submission.csv")
sample_submission["pi_lower"] = test_lower
sample_submission["pi_upper"] = test_upper
sample_submission.to_csv("submission.csv", index=False)

sample_submission

