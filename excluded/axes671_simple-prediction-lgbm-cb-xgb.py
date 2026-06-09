!pip install hillclimbers


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from hillclimbers import climb_hill, partial

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from matplotlib import style

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = [8, 8]


train = pd.read_csv(r"/kaggle/input/playground-series-s5e1/train.csv")
test = pd.read_csv(r"/kaggle/input/playground-series-s5e1/test.csv")
sample = pd.read_csv(r"/kaggle/input/playground-series-s5e1/sample_submission.csv")


train


train = train.dropna()


sns.histplot(np.log1p(train['num_sold']), binwidth=0.5, kde=True)


def policy_date(df):
    df["date"] = pd.to_datetime(df["date"])

    df["Year"] = df["date"].dt.year
    df["Month"] = df["date"].dt.month
    df["Day"] = df["date"].dt.day
    df["DOW"] = df["date"].dt.day_of_week

    Ymin = np.min(df["Year"])
    Ymax = np.max(df["Year"])

    df["Ysin"] = np.sin(2 * np.pi * (df["Year"] - Ymin) / (Ymax - Ymin))
    df["Ycos"] = np.cos(2 * np.pi * (df["Year"] - Ymin) / (Ymax - Ymin))

    df["Msin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Mcos"] = np.cos(2 * np.pi * df["Month"] / 12)

    df["Wsin"] = np.sin(2 * np.pi * df["DOW"] / 6)
    df["Wcos"] = np.cos(2 * np.pi * df["DOW"] / 6)

    df["Dsin"] = np.sin(2 * np.pi * df["Day"] / 31)
    df["Dcos"] = np.cos(2 * np.pi * df["Day"] / 31)

    df = df.drop(columns=["id", "date", "Year", "Month", "Day", "DOW"])

    return df


train = policy_date(train)
test = policy_date(test)


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

Ys = train.groupby("Ysin")["num_sold"].mean()
Yc = train.groupby("Ycos")["num_sold"].mean()
sns.lineplot(Ys, ax=ax1, label="cos")
sns.lineplot(Yc, ax=ax1, label="sin")
ax1.set_xlabel("Year")
Ms = train.groupby("Msin")["num_sold"].mean()
Mc = train.groupby("Mcos")["num_sold"].mean()
Yc = train.groupby("Ycos")["num_sold"].mean()
sns.lineplot(Ms, ax=ax2)
sns.lineplot(Mc, ax=ax2)
ax2.set_xlabel("Month")
Ws = train.groupby("Wsin")["num_sold"].mean()
Wc = train.groupby("Wcos")["num_sold"].mean()
sns.lineplot(Ws, ax=ax3)
sns.lineplot(Wc, ax=ax3)
ax3.set_xlabel("Week")
Ds = train.groupby("Dsin")["num_sold"].mean()
Dc = train.groupby("Dcos")["num_sold"].mean()
sns.lineplot(Ds, ax=ax4)
sns.lineplot(Dc, ax=ax4)
ax4.set_xlabel("Day")
plt.legend()
plt.tight_layout()


train.isna().sum().sort_values(ascending=False)


train = train.fillna(value=0.001)


X = train.drop(columns=["num_sold"])
y = train["num_sold"]
y = np.log1p(y)


X.dtypes


numericals = X.dtypes[X.dtypes != "object"].index
categoricals = X.dtypes[X.dtypes == "object"].index

num_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        # ('scaler', StandardScaler())
    ]
)

cat_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder()),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, numericals),
        ("cat", cat_pipeline, categoricals),
    ]
)

X_processed = preprocessor.fit_transform(X)
test_processed = preprocessor.transform(test)


Xt, Xv, yt, yv = train_test_split(X_processed, y, random_state=42, shuffle=True)


X_processed


model_lgb = LGBMRegressor()
model_lgb.fit(Xt, yt)
pred_lgb2 = model_lgb.predict(Xv)
mean_absolute_percentage_error(np.expm1(yv), np.expm1(pred_lgb2))


model_cat = CatBoostRegressor(silent=True)
model_cat.fit(Xt, yt)
pred_cat2 = model_cat.predict(Xv)
mean_absolute_percentage_error(np.expm1(yv), np.expm1(pred_cat2))


model_xgb = XGBRegressor()
model_xgb.fit(Xt, yt)
pred_xgb2 = model_xgb.predict(Xv)
mean_absolute_percentage_error(np.expm1(yv), np.expm1(pred_xgb2))


data = pd.DataFrame(yv)
data['predxgb'] = pred_xgb2
data['predlgb'] = pred_lgb2
data['predcat'] = pred_cat2
data['sum3'] = (pred_xgb2 + pred_lgb2 + pred_cat2) / 3


data


data['num_sold'] = np.expm1(data['num_sold'])
data['predxgb'] = np.expm1(pred_xgb2)
data['predlgb'] = np.expm1(pred_lgb2)
data['predcat'] = np.expm1(pred_cat2)
data['sum3'] =    np.expm1(data['sum3'])


sns.scatterplot(data=data, x='num_sold', y='predxgb', alpha=0.4)
sns.scatterplot(data=data, x='num_sold', y='predlgb', alpha=0.4)
sns.scatterplot(data=data, x='num_sold', y='predcat', alpha=0.4)
sns.scatterplot(data=data, x='num_sold', y='sum3', alpha=0.9)
sns.lineplot(x=(0, 4500), y=(0, 4500), color='black')


data


mean_absolute_percentage_error(data['num_sold'], data['sum3'])


model_cat.fit(X_processed, y)
model_xgb.fit(X_processed, y)
model_lgb.fit(X_processed, y)


test_pred_cat = model_cat.predict(test_processed)
test_pred_xgb = model_xgb.predict(test_processed)
test_pred_lgb = model_lgb.predict(test_processed)


avg_test = (test_pred_cat + test_pred_xgb + test_pred_lgb) / 3


sample['num_sold'] = avg_test


# sample.to_csv('kaggle1.csv', index=None)


def fit_predict(model,):
    scores = []
    oof_pred = np.zeros((X_processed.shape[0]))
    test_pred = np.zeros((test_processed.shape[0]))

    skf = KFold(n_splits=5, random_state=42, shuffle=True)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = pd.DataFrame(X_processed).iloc[train_idx], pd.DataFrame(X_processed).iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = clone(model)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        oof_pred[val_idx] = y_pred

        temp_test_pred = model.predict(test_processed)
        test_pred += temp_test_pred / 5

        score = mean_absolute_percentage_error(np.expm1(y_val), np.expm1(y_pred))
        scores.append(score)

    overall_score = mean_absolute_percentage_error(np.expm1(y), np.expm1(oof_pred))

    print(
        f"\n Overall: {overall_score:.5f} | Average score: {np.mean(scores):.5f} ± {np.std(scores):.5f}"
    )
    return oof_pred, test_pred


oof_pred_probs = {}
test_pred_probs = {}
oof_pred_probs["XGBoost"], test_pred_probs["XGBoost"] = fit_predict(model_xgb)
oof_pred_probs["LGBM"], test_pred_probs["LGBM"] = fit_predict(model_lgb)
oof_pred_probs["CAT"], test_pred_probs["CAT"] = fit_predict(model_cat)


# hill_climb_test_pred_probs, hill_climb_oof_pred_probs = climb_hill(
#     train=train,
#     oof_pred_df=pd.DataFrame(oof_pred_probs),
#     test_pred_df=pd.DataFrame(test_pred_probs),
#     target='num_sold',
#     objective="minimize",
#     eval_metric=partial(mean_absolute_percentage_error),
#     negative_weights=True,
#     precision=0.001,
#     plot_hill=True,
#     plot_hist=True,
#     return_oof_preds=True) 


# hill_climb_test_pred_probs


sample['num_sold'] = np.expm1((
    test_pred_probs["CAT"]
    + test_pred_probs["XGBoost"]
    + test_pred_probs["LGBM"]
) / 3)
sample.to_csv('kaggle3.csv', index=None)




