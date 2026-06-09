!pip install tubesml


import numpy as np
import pandas as pd

import tubesml as tml
from tubesml.base import BaseTransformer, fit_wrapper, transform_wrapper

from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb
import lightgbm as lgb

import matplotlib.pyplot as plt
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

df.head()


_ = tml.list_missing(df)
print("_"*40)
_ = tml.list_missing(df_test)


df.describe()


df_test.describe()


df.hist(bins=20, figsize=(15,10), grid=False)
plt.show()


df_test.hist(bins=20, figsize=(15,10), grid=False)
plt.show()


df["target"] = np.log1p(df["Calories"])
tml.plot_distribution(df, "target")


num_cor = tml.plot_correlations(data=df.select_dtypes('number'), target="target", annot=True)
num_cor


for c in num_cor.index[1:].tolist():
    if c == "id" or c == "Calories":
        continue
    tml.plot_bivariate(data=df, x=c, y="target", hue="Sex")


tml.segm_target(data=df, target='target', cat="Sex")


NFOLDS = 5

kfolds = KFold(n_splits=NFOLDS, shuffle=True, random_state=32)


class BaselineModel(BaseTransformer):
    def __init__(self):
        super().__init__()
        self.mean_target = None
        
    @fit_wrapper
    def fit(self, X, y):
        self.mean_target = y.mean()

    def predict(self, X):
        return np.array([self.mean_target] * len(X))


base_model = BaselineModel()
oof, _ = tml.cv_score(data=df, target=df["target"], estimator=base_model, cv=kfolds)

oof = np.expm1(oof)

base_score = np.sqrt(mean_squared_log_error(y_true=df["Calories"], y_pred=oof))

round(base_score, 5)


training_cols = [c for c in df if c not in ["id", "Calories", "target"]]
target = df["target"]

training_cols


processing_pipe = Pipeline([("dummies", tml.Dummify(drop_first=True)),
                            ("scaler", tml.DfScaler(method="robust"))])


model_pipe = Pipeline([('processing', processing_pipe),
                       ('model', Lasso(alpha=0.01, random_state=35))])


oof, res = tml.cv_score(data=df[training_cols], target=target, estimator=model_pipe, cv=kfolds, imp_coef=True)

oof = np.expm1(oof)

model_score = np.sqrt(mean_squared_log_error(y_true=df["Calories"], y_pred=oof))

print(round(model_score, 5))
print(f"{round((model_score - base_score) / base_score, 3) * 100} %")

lasso_score = model_score

tml.plot_feat_imp(res['feat_imp'])

tml.plot_regression_predictions(data=df, true_label=df["Calories"], pred_label=oof,
                                feature=['Duration', 'Heart_Rate'], hue='Sex')


model_pipe = Pipeline([('processing', processing_pipe),
                       ('model', Ridge(random_state=35))])


oof, res = tml.cv_score(data=df[training_cols], target=target, estimator=model_pipe, cv=kfolds, imp_coef=True)

oof = np.expm1(oof)

model_score = np.sqrt(mean_squared_log_error(y_true=df["Calories"], y_pred=oof))

print(round(model_score, 5))
print(f"{round((model_score - base_score) / base_score, 3) * 100} %")

ridge_score = model_score

tml.plot_feat_imp(res['feat_imp'])

tml.plot_regression_predictions(data=df, true_label=df["Calories"], pred_label=oof,
                                feature=['Duration', 'Heart_Rate'], hue='Sex')


model_pipe = Pipeline([('processing', processing_pipe),
                       ('model', RandomForestRegressor(n_jobs=-1, random_state=366))])


oof, res = tml.cv_score(data=df[training_cols], target=target, estimator=model_pipe, cv=kfolds, imp_coef=True)

oof = np.expm1(oof)

model_score = np.sqrt(mean_squared_log_error(y_true=df["Calories"], y_pred=oof))

print(round(model_score, 5))
print(f"{round((model_score - base_score) / base_score, 3) * 100} %")

score_forest = model_score

tml.plot_feat_imp(res['feat_imp'])

tml.plot_regression_predictions(data=df, true_label=df["Calories"], pred_label=oof,
                                feature=['Duration', 'Heart_Rate'], hue='Sex')


model_pipe = Pipeline([('processing', processing_pipe),
                       ('model', xgb.XGBRegressor(n_estimators=10000,
                                                  early_stopping_rounds=100,
                                                  n_jobs=-1,
                                                  random_state=33))])

fit_params = {'verbose': False}

oof, res = tml.cv_score(data=df[training_cols], target=target, estimator=model_pipe, cv=kfolds,
                        imp_coef=True, early_stopping=True, fit_params=fit_params)

oof = np.expm1(oof)

model_score = np.sqrt(mean_squared_log_error(y_true=df["Calories"], y_pred=oof))

print(res["iterations"])
print(round(model_score, 5))
print(f"{round((model_score - base_score) / base_score, 3) * 100} %")

xgb_score = model_score

tml.plot_feat_imp(res['feat_imp'])

tml.plot_regression_predictions(data=df, true_label=df["Calories"], pred_label=oof,
                                feature=['Duration', 'Heart_Rate'], hue='Sex')


model_pipe = Pipeline([('processing', processing_pipe),
                       ('model', lgb.LGBMRegressor(n_estimators=10000,
                                                             n_jobs=-1,
                                                             random_state=354,
                                                             verbose=-1))])

callbacks = [lgb.early_stopping(100, verbose=0)]
fit_params = {"callbacks":callbacks}

oof, res = tml.cv_score(data=df[training_cols], target=target, estimator=model_pipe, cv=kfolds,
                        imp_coef=True, early_stopping=True, fit_params=fit_params)

oof = np.expm1(oof)

model_score = np.sqrt(mean_squared_log_error(y_true=df["Calories"], y_pred=oof))

print(res["iterations"])
print(round(model_score, 5))
print(f"{round((model_score - base_score) / base_score, 3) * 100} %")

lgb_score = model_score

tml.plot_feat_imp(res['feat_imp'])

tml.plot_regression_predictions(data=df, true_label=df["Calories"], pred_label=oof,
                                feature=['Duration', 'Heart_Rate'], hue='Sex')


from sklearn.base import RegressorMixin, BaseEstimator

class ModelBySex(RegressorMixin, BaseEstimator):
    def __init__(self, model_male, model_female):
        super().__init__()
        self.model_male = model_male
        self.model_female = model_female

    def fit(self, X, y):
        X = X.copy().reset_index(drop=True)
        male_X = X[X["Sex"] == "male"]
        female_X = X[X["Sex"] == "female"]
        male_y = pd.Series(y.iloc[male_X.index].values.ravel())
        female_y = pd.Series(y.iloc[female_X.index].values.ravel())

        self.model_male.fit(male_X[[c for c in X if c!="Sex"]], male_y)
        self.model_female.fit(female_X[[c for c in X if c!="Sex"]], female_y)

        return self

    def predict(self, X):
        Xpr = X.copy()
        Xpr.loc[Xpr["Sex"] == "male", "prediction"] = self.model_male.predict(X[X["Sex"] == "male"][[c for c in X if c!="Sex"]])
        Xpr.loc[Xpr["Sex"] == "female", "prediction"] = self.model_female.predict(X[X["Sex"] == "female"][[c for c in X if c!="Sex"]])

        return Xpr["prediction"].values


model = ModelBySex(model_male=lgb.LGBMRegressor(n_estimators=800,
                                                n_jobs=-1,
                                                random_state=354,
                                                verbose=-1),
                  model_female=lgb.LGBMRegressor(n_estimators=800,
                                                n_jobs=-1,
                                                random_state=354,
                                                verbose=-1))

oof, res = tml.cv_score(data=df[training_cols], target=target, estimator=model, cv=kfolds)

oof = np.expm1(oof)

model_score = np.sqrt(mean_squared_log_error(y_true=df["Calories"], y_pred=oof))

print(round(model_score, 5))
print(f"{round((model_score - base_score) / base_score, 3) * 100} %")

by_sex_score = model_score

tml.plot_regression_predictions(data=df, true_label=df["Calories"], pred_label=oof,
                                feature=['Duration', 'Heart_Rate'], hue='Sex')


tmp = df.copy()
tmp["W_H_ratio"] = tmp["Weight"] / tmp["Height"]
tmp["HR_D_ratio"] = tmp["Heart_Rate"] / tmp["Duration"]
tmp["HR_W_ratio"] = tmp["Heart_Rate"] / tmp["Weight"]
tmp["T_HR_ratio"] = tmp["Body_Temp"] / tmp["Heart_Rate"]
tmp["T_D_ratio"] = tmp["Body_Temp"] / tmp["Duration"]
tmp.loc[tmp["Sex"] == "male", "BMR"] = tmp["Weight"] * 10 + 6.25 * tmp["Height"] - 5 * tmp["Age"] + 5
tmp.loc[tmp["Sex"] == "female", "BMR"] = tmp["Weight"] * 10 + 6.25 * tmp["Height"] - 5 * tmp["Age"] - 161


num_cor = tml.plot_correlations(data=tmp.select_dtypes('number'), target="target", annot=True)
num_cor


for c in ["T_HR_ratio", "T_D_ratio", "HR_D_ratio", "HR_W_ratio", "W_H_ratio", "BMR"]:
    tml.plot_bivariate(data=tmp, x=c, y="target", hue="Sex")


class FE(BaseTransformer):
    def __init__(self, W_H=False, HR_D=False, HR_W=False, T_HR=False, T_D=False, BMR=False):
        super().__init__()
        self.W_H = W_H
        self.HR_D = HR_D
        self.HR_W = HR_W
        self.T_HR = T_HR
        self.T_D = T_D
        self.BMR = BMR

    @fit_wrapper
    def fit(self, X, y=None):
        return self


    @transform_wrapper
    def transform(self, X, y=None):
        Xtr = X.copy()
        if self.W_H:
            Xtr["W_H_ratio"] = Xtr["Weight"] / Xtr["Height"]
        if self.HR_D:
            Xtr["HR_D_ratio"] = Xtr["Heart_Rate"] / Xtr["Duration"]
        if self.HR_W:
            Xtr["HR_W_ratio"] = Xtr["Heart_Rate"] / Xtr["Weight"]
        if self.T_HR:
            Xtr["T_HR_ratio"] = Xtr["Body_Temp"] / Xtr["Heart_Rate"]
        if self.T_D:
            Xtr["T_D_ratio"] = Xtr["Body_Temp"] / Xtr["Duration"]
        if self.BMR:
            Xtr.loc[tmp["Sex"] == "male", "BMR"] = Xtr["Weight"] * 10 + 6.25 * Xtr["Height"] - 5 * Xtr["Age"] + 5
            Xtr.loc[tmp["Sex"] == "female", "BMR"] = Xtr["Weight"] * 10 + 6.25 * Xtr["Height"] - 5 * Xtr["Age"] - 161

        return Xtr


processing_pipe = Pipeline([("dummies", tml.Dummify(drop_first=True)),
                            ("fe", FE(W_H=False, #0.9
                                      T_HR=False, #0.9% better
                                      T_D=False, # 1% better
                                      HR_D=True, # 1.1%
                                      HR_W=False, #0.8
                                      BMR=False # 0.9
                                     ))])


model_pipe = Pipeline([('processing', processing_pipe),
                       ('model', lgb.LGBMRegressor(n_estimators=10000,
                                                   subsample=0.9,
                                                   num_leaves=10,
                                                   reg_lambda=10,
                                                   reg_alpha=1,
                                                   n_jobs=-1,
                                                   random_state=354,
                                                   verbose=-1))])

callbacks = [lgb.early_stopping(100, verbose=0)]
fit_params = {"callbacks":callbacks}

oof, res = tml.cv_score(data=df[training_cols], target=target, estimator=model_pipe, cv=kfolds,
                        imp_coef=True, early_stopping=True, fit_params=fit_params)

oof = np.expm1(oof)

model_score = np.sqrt(mean_squared_log_error(y_true=df["Calories"], y_pred=oof))

print(res["iterations"])
print(round(model_score, 5))
print(f"{round((model_score - lgb_score) / lgb_score, 3) * 100} %")

tml.plot_feat_imp(res['feat_imp'])

tml.plot_regression_predictions(data=df, true_label=df["Calories"], pred_label=oof,
                                feature=['Duration', 'Heart_Rate'], hue='Sex')


processing_pipe = Pipeline([("dummies", tml.Dummify(drop_first=True)),
                            ("fe", FE(W_H=False, #0.9
                                      T_HR=False, #0.9% better
                                      T_D=False, # 1% better
                                      HR_D=True, # 1.1%
                                      HR_W=False, #0.8
                                      BMR=False # 0.9
                                     ))])

model_pipe = model_pipe = Pipeline([('processing', processing_pipe),
                       ('model', lgb.LGBMRegressor(n_estimators=2000,
                                                   subsample=0.9,
                                                   num_leaves=10,
                                                   reg_lambda=10,
                                                   reg_alpha=1,
                                                   n_jobs=-1,
                                                   random_state=354,
                                                   verbose=-1))])

tmp = df.sample(10000)
tmp_target = tmp["target"]

tml.plot_learning_curve(estimator=model_pipe, X=tmp[training_cols], y=tmp_target, cv=kfolds,
                        scoring="neg_mean_squared_error",
                        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)


processing_pipe = Pipeline([("dummies", tml.Dummify(drop_first=True)),
                            ("fe", FE(W_H=False,
                                      T_HR=False,
                                      T_D=False,
                                      HR_D=True,
                                      HR_W=False,
                                      BMR=False
                                     ))])

model_pipe = model_pipe = Pipeline([('processing', processing_pipe),
                       ('model', lgb.LGBMRegressor(n_estimators=2400,
                                                   subsample=0.9,
                                                   num_leaves=10,
                                                   reg_lambda=10,
                                                   reg_alpha=1,
                                                   n_jobs=-1,
                                                   random_state=354,
                                                   verbose=-1))])

model_pipe.fit(df[training_cols], target)
predictions = model_pipe.predict(df_test[training_cols])

predictions = np.expm1(predictions)


sub = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
sub["Calories"] = predictions

sub.to_csv("submission.csv", index=False)
sub.head()

