import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv", index_col=0)
predict = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv", index_col=0)

print(train.info())
print("\nMissing values in train:", train.isnull().sum().sum())
print("Missing values in test:", predict.isnull().sum().sum())

# basic statistics
print(train['accident_risk'].describe())


class RoadAccidentFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self

    def fit(self, df, y=None):
        df= df.copy()

        return self

    def transform(self, df):
        df= df.copy()

        df["CurxSpeed"] = df["curvature"] * df["speed_limit"]
        df["Curxlames"] = df["curvature"] * df["num_lanes"]
        df["Speedxlames"] = df["num_lanes"] * df["speed_limit"]

        for col in list(df.select_dtypes(include="object").columns):
            df[col] = df[col].astype("category")
            
        for col in list(df.select_dtypes(include="bool").columns):
            df[col] = df[col].astype("int")
            
        return df


y = train.iloc[:, -1].copy()
X = train.iloc[:, 1:-1].copy()
X_pred = predict



features = RoadAccidentFeatures().fit_transform(X)



features.info()


cv = KFold(n_splits=5, shuffle=True, random_state=42)

numericTransformer = Pipeline([
    ("scaler", MinMaxScaler())
])
numericTransformer.set_output(transform="pandas")

categoryTransformer = Pipeline([
    ("oe", OrdinalEncoder()),
])
categoryTransformer.set_output(transform="pandas")

preprocessorLine = ColumnTransformer([
    ("cat", categoryTransformer, make_column_selector(dtype_include='category')),
    ("num", numericTransformer, make_column_selector(dtype_include=['int64', 'float64'])),
])
preprocessorLine.set_output(transform="pandas")

xgb = Pipeline([
    ("RARPF", RoadAccidentFeatures()),
    ("preprocessor", preprocessorLine),
    ("xgb", XGBRegressor(
        random_state=42,
        n_estimators=1000,
        verbose=False
    ))
])

cat = Pipeline([
    ("RARPF", RoadAccidentFeatures()),
    ("preprocessor", preprocessorLine),
    ("cat", CatBoostRegressor(
        random_state=42,
        n_estimators=1000,
        #cat_features=features.select_dtypes(include="category").columns.to_list(),
        verbose=False
    ))
])

lgbm = Pipeline([
    ("RARPF", RoadAccidentFeatures()),
    ("preprocessor", preprocessorLine),
    ("lgbm", LGBMRegressor(
        random_state=42,
        n_estimators=1000,
        verbose=-1
    ))
])



list(range(len(features.select_dtypes(include="category").columns)))


cat_score = np.sqrt(-1 * cross_val_score(cat, X, y, cv=cv, scoring="neg_mean_squared_error", error_score='raise'))
xgb_score = np.sqrt(-1 * cross_val_score(xgb, X, y, cv=cv, scoring="neg_mean_squared_error"))
lgbm_score = np.sqrt(-1 * cross_val_score(lgbm, X, y, cv=cv, scoring="neg_mean_squared_error"))

print("XGB:", xgb_score, "- Average score: ", xgb_score.mean())
print("CAT:", cat_score, "- Average score: ", cat_score.mean())
print("LGB:", lgbm_score, "- Average score: ", lgbm_score.mean())


cat.fit(X, y)
df_final = pd.DataFrame(data=cat.predict(X_pred), index=X_pred.index, columns=["accident_risk"])
df_final


df_final.to_csv("/kaggle/working/" + "submission.csv")

