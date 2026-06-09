import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px


from sklearn.preprocessing import StandardScaler , MinMaxScaler , LabelEncoder , OrdinalEncoder
from sklearn.model_selection import train_test_split , GridSearchCV , RandomizedSearchCV , KFold , StratifiedKFold
from sklearn.impute import SimpleImputer # works like fillna 
from sklearn.linear_model import LinearRegression , Ridge , Lasso
# from sklearn.ensemble import AdaBoostRegressor , RandomForestRegressor
# from xgboost import XGBRegressor
# from catboost import CatBoostRegressor
# from lightgbm import LGBMRegressor
# from quantile_forest import RandomForestQuantileRegressor
# from mapie.regression import MapieQuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor

import math
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import warnings
warnings.filterwarnings('ignore')

import os


# train_df = pd.read_csv('/home/nex/Downloads/prediction-interval-competition-ii-house-price/dataset.csv')
# test_df = pd.read_csv('/home/nex/Downloads/prediction-interval-competition-ii-house-price/test.csv')
# submission_df = pd.read_csv('/home/nex/Downloads/prediction-interval-competition-ii-house-price/sample_submission.csv')

train_df = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv")
test_df = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/test.csv")
submission_df= pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/sample_submission.csv")


print(f'train data shape {train_df.shape}')
print(f'test data shape {test_df.shape}')
print(f'submission data shape {submission_df.shape}')


print(f"train data info : {train_df.info()}")



print(f"columns of train data : {train_df.columns}")
print("=="*30)
print(f"columns of test data : {test_df.columns}")


print(f"descriptive statistics : {train_df.describe()}")


train_df.columns


def preprocessing(df):
    imputer = SimpleImputer(strategy="mean")
    le = LabelEncoder()
    oencdr = OrdinalEncoder()

    # Convert to datetime and extract date/month
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df["month"] = df["sale_date"].dt.month
    df["day"] = df["sale_date"].dt.day

    # Now drop unwanted columns
    col = ["id", "sale_date", "sale_warning", "latitude", "longitude"]
    df = df.drop(columns=col)

    # Impute and encode
    df["sale_nbr"] = imputer.fit_transform(df[["sale_nbr"]])
    df["join_status"] = le.fit_transform(df["join_status"])
    df["city"] = le.fit_transform(df["city"])
    df["zoning"] = le.fit_transform(df["zoning"])
    df["subdivision"] = le.fit_transform(df["subdivision"])
    df["garb_sqft"] = imputer.fit_transform(df[["garb_sqft"]])
    df["gara_sqft"] = imputer.fit_transform(df[["gara_sqft"]])
    df["submarket"] = df["submarket"].dropna(inplace=True)
    df["submarket"] = oencdr.fit_transform(df[["submarket"]])


    return df



train_df = preprocessing(train_df)


train_df


train_df.info()


train_df.isnull().sum()


train_df.describe()


corr = train_df.corr()
sns.heatmap(corr)


X = train_df.drop(columns = ["sale_price"])
y = train_df["sale_price"]


print(f"shape of X : {X.shape}")
print("=="*20)
print(f"shape of y : {y.shape}")


random_state = 42 
np.random.seed(random_state)


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


# # finding best parameters for the model ( long process and will take more >= 1 hour)

# param_grid = [
#     {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.01},
#     {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1},
#     {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.01},
#     {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1},
#     {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.01},
#     {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.1}
# ]

# alpha = 0.1  # 90% prediction interval

# best_score = np.inf  # Initialize here
# best_params = None
# cv = KFold(n_splits=5, shuffle=True, random_state=42)

# for params in param_grid:
#     fold_scores = []
#     for train_idx, val_idx in cv.split(X):
#         X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#         y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#         # Lower quantile model
#         lower_model = GradientBoostingRegressor(
#             loss='quantile', alpha=alpha/2,
#             n_estimators=params['n_estimators'],
#             max_depth=params['max_depth'],
#             learning_rate=params['learning_rate'],
#             random_state=42
#         )
#         lower_model.fit(X_train, y_train)
#         y_lower = lower_model.predict(X_val)

#         # Upper quantile model
#         upper_model = GradientBoostingRegressor(
#             loss='quantile', alpha=1-alpha/2,
#             n_estimators=params['n_estimators'],
#             max_depth=params['max_depth'],
#             learning_rate=params['learning_rate'],
#             random_state=42
#         )
#         upper_model.fit(X_train, y_train)
#         y_upper = upper_model.predict(X_val)

#         score = winkler_score(y_val, y_lower, y_upper, alpha)
#         fold_scores.append(score)

#     # Calculating mean AFTER all folds (resultant score)
#     mean_score = np.mean(fold_scores)
#     print(f"Params: {params}, Mean Winkler Score: {mean_score:.4f}")

#     # Updating best parameters
#     if mean_score < best_score:
#         best_score = mean_score
#         best_params = params

# print("\nBest Params:", best_params, "Best Winkler Score:", best_score)



alpha = 0.1
best_params = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1
}

lower_model = GradientBoostingRegressor(
    loss='quantile', alpha=alpha/2,
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
)
lower_model.fit(X, y)

upper_model = GradientBoostingRegressor(
    loss='quantile', alpha=1-(alpha/2),
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
)
upper_model.fit(X, y)


test_df= preprocessing(test_df)


lower_pi = lower_model.predict(test_df)
upper_pi = upper_model.predict(test_df)


submission_df = pd.DataFrame({
    "id" : submission_df["id"],
    "pi_lower": np.round(lower_pi,2),
    "pi_upper": np.round(upper_pi,2)
})

submission_df.to_csv("submission.csv" , index=False)


submission_df




