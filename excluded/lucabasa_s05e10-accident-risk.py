!pip install tubesml


import numpy as np 
import pandas as pd

import tubesml as tml
from tubesml.base import BaseTransformer, fit_wrapper, transform_wrapper

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, Ridge

import lightgbm as lgb
import xgboost as xgb

import optuna
from optuna.samplers import TPESampler

import matplotlib.pyplot as plt
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")

_ = tml.list_missing(df_train)

print(df_train["accident_risk"].mean())

df_train.head()


df_original = pd.read_csv("/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv")

_ = tml.list_missing(df_original)

print(df_original["accident_risk"].mean())

df_original.head()


for col in df_train.select_dtypes(exclude="number"):
    print(col)
    print(df_train[col].value_counts(dropna=False))
    print("\n")


df_train.describe().T


df_train.columns


class GeneralCleaner(BaseTransformer):
    
    @transform_wrapper
    def transform(self, X, y=None):
        Xtr = X.copy()

        for col in ["road_signs_present", "public_road", "holiday", "school_season"]:
            Xtr[col] = Xtr[col].astype(int)
        
        return Xtr


tmp = df_train.copy()

gencl = GeneralCleaner()
tmp = gencl.fit_transform(tmp)

num_cor = tml.plot_correlations(data=tmp.select_dtypes('number'), target="accident_risk", annot=True)
num_cor


for col in df_train.select_dtypes(exclude="number"):
    tml.segm_target(data=df_train, cat=col, target="accident_risk")


for col in num_cor.index[:4]:
    if col == "accident_risk":
        tml.plot_distribution(df_train, col)
    else:
        tml.plot_bivariate(df_train, col, "accident_risk")


N_FOLDS = 5
kfolds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=13)


class BaselineModel(BaseEstimator, RegressorMixin):
    def __init__(self, by_cols=None):
        self.by_cols = by_cols
        if by_cols is not None:
            self.means = {}
        else:
            self.means = None

    def fit(self, X, y):
        if self.by_cols is not None:
            tmp = X.copy()
            tmp["target"] = y
            self.means = tmp.groupby(self.by_cols)["target"].mean().to_dict()
        else:
            self.means = y.mean()

    def predict(self, X):
        if self.by_cols is not None:
            return X[self.by_cols[0]].map(self.means).to_numpy()
        else:
            return pd.Series([self.means] * len(X)).to_numpy()


basemodel = BaselineModel()

cvscore = tml.CrossValidate(data=df_train, target=df_train["accident_risk"], estimator=basemodel, cv=kfolds)

baseline_oof, _ = cvscore.score()

print(f"RMSE baseline: {np.sqrt(mean_squared_error(y_true=df_train['accident_risk'], y_pred=baseline_oof))}")

tml.plot_regression_predictions(data=df_train, true_label=df_train['accident_risk'], pred_label=baseline_oof, hue="lighting")


basemodel = BaselineModel(by_cols=["lighting"])

cvscore = tml.CrossValidate(data=df_train, target=df_train["accident_risk"], estimator=basemodel, cv=kfolds)

baseline_oof, _ = cvscore.score()

print(f"RMSE baseline: {np.sqrt(mean_squared_error(y_true=df_train['accident_risk'], y_pred=baseline_oof))}")

tml.plot_regression_predictions(data=df_train, true_label=df_train['accident_risk'], pred_label=baseline_oof, hue="lighting")


train, test = tml.make_test(df_train, 0.2, random_state=55)

# df_full = pd.concat([train.drop("id", axis=1), df_original], ignore_index=True)

df_full = train.drop(["id"], axis=1).copy()

target = df_full["accident_risk"]
df = df_full.drop(["accident_risk"], axis=1)

test_target = test["accident_risk"]
test = test.drop(["id", "accident_risk"], axis=1)

basemodel = BaselineModel(by_cols=["lighting"])

cvscore = tml.CrossValidate(data=df, target=target, estimator=basemodel, cv=kfolds)

baseline_oof, _ = cvscore.score()

baseline_score = np.sqrt(mean_squared_error(y_true=target, y_pred=baseline_oof))

print(f"RMSE baseline: {baseline_score}")

result_compare = {"score": baseline_score, 
                  "oof": baseline_oof}


def train_predict(data, test_data, target, estimator, cv, test_target=None, fit_params=None, early_stopping=False, shap=True, result_compare=None):
    cv_score = tml.CrossValidate(data=data, target=target, test=test_data,
                                 estimator=estimator, cv=cv, fit_params=fit_params, early_stopping=early_stopping,
                                shap=shap, imp_coef=True)

    oof, pred, result_dict = cv_score.score()

    score = np.sqrt(mean_squared_error(y_true=target, y_pred=oof))
    print(f"OOF score: {score}")

    if early_stopping:
        print(result_dict["iterations"])

    if shap:
        tml.plot_feat_imp(result_dict["feat_imp"], imp="both", n=10)
        to_plot = result_dict["feat_imp"].head(10)["Feature"].to_list()
        tml.plot_shap_values(result_dict["shap_values"], features=to_plot)
    else:
        tml.plot_feat_imp(result_dict["feat_imp"], imp="standard", n=10)

    tml.plot_regression_predictions(data=data, true_label=target, pred_label=oof, hue="lighting", feature=["curvature", "speed_limit"])

    if result_compare is not None:
        baseline = result_compare["score"]
        ref_oof = result_compare["oof"]
    
        print(f"This is {round((score - baseline) / baseline * 100, 3)}% better than the baseline.")
    
        comparison = tml.CompareModels(data=data, true_label=target.reset_index(drop=True),
                                       pred_1=oof, pred_2=ref_oof,
                                       kfold=cv,
                                       regression=True,
                                       probabilities=True,
                                       metric_func=mean_squared_error)
    
        comparison.compare_metrics()
        comparison.compare_predictions()
        comparison.statistical_significance()

    if test_target is not None:
        score = np.sqrt(mean_squared_error(y_true=test_target, y_pred=pred))
        print(f"Test score: {score}")
        tml.plot_regression_predictions(data=test_data, true_label=test_target, pred_label=pred, hue="lighting", feature=["curvature", "speed_limit"])

    return oof, pred, result_dict


class GeneralCleaner(BaseTransformer):
    
    @transform_wrapper
    def transform(self, X, y=None):
        Xtr = X.copy()

        for col in ["road_signs_present", "public_road", "holiday", "school_season"]:
            Xtr[col] = Xtr[col].astype(int)
        
        return Xtr


class Categorizer(BaseTransformer):
    def __init__(self, categories=None):
        super().__init__()
        self.categories = categories

    @transform_wrapper
    def transform(self, X, y=None):
        Xtr = X.copy()

        if self.categories is None:
            cats = Xtr.columns
        else:
            cats = self.categories

        for cat in cats:
            if cat in Xtr.columns:
                Xtr[cat] = Xtr[cat].astype("category")
        
        return Xtr

class ColumnRename(BaseTransformer):
    def __init__(self, suffix="_1"):
        super().__init__()
        self.suffix = suffix

    @transform_wrapper
    def transform(self, X, y=None):
        Xtr = X.copy()

        Xtr.columns = [f"{col}{self.suffix}" for col in Xtr.columns]

        return Xtr


class FeatureEng(BaseTransformer):
    def __init__(self, road_risk=True, curv_speed=True):
        self.road_risk = road_risk
        self.curv_speed = curv_speed
    

    @transform_wrapper
    def transform(self, X, y=None):
        Xtr = X.copy()

        if self.road_risk:
            Xtr["road_risk"] = (0.3 * X["curvature"] +
                            0.2 * (X["lighting"] == "night").astype(int) +
                            0.1 * (X["weather"] != "clear").astype(int) +
                            0.2 * (X["speed_limit"] >= 60).astype(int) +
                            0.1 * (X["num_reported_accidents"] > 2).astype(int))

        if self.curv_speed:
            Xtr["curv_speed"] = Xtr["curvature"] * Xtr["speed_limit"]
        
        return Xtr


cat_pipe = Pipeline([('sel', tml.DtypeSel('category')),
                     ('imputer', tml.DfImputer(fill_value="unknown", strategy="constant")),
                     # ("te", tml.TargetEncoder()),
                     ("dummies", tml.Dummify(drop_first=True))])

num_pipe = Pipeline([('sel', tml.DtypeSel('numeric')),
                     ('imputer', tml.DfImputer(strategy='mean')),
                     ])

processing_pipe = tml.FeatureUnionDf(transformer_list=[('num', num_pipe),
                                                       ('cat', cat_pipe)])

processing = Pipeline([("gc", GeneralCleaner()),
                       ("fe", FeatureEng(curv_speed=False)),
                       ("pipe", processing_pipe),
                       ("poly", tml.DfPolynomial(interaction_only=True, degree=2)),
                       ("scaler", tml.DfScaler()),
                       # ("pca", tml.DfPCA(n_components=0.9, compress=True))
                      ])

full_pipe = Pipeline([("proc", processing), 
                      ("model", Ridge(random_state=43))])


oof_ridge, pred_ridge, res_ridge = train_predict(data=df, target=target, test_data=test, test_target=test_target,
                                                 estimator=full_pipe, cv=kfolds, result_compare=result_compare)

ridge_score = np.sqrt(mean_squared_error(y_true=target, y_pred=oof_ridge))
result_compare = {"score": ridge_score, 
                  "oof": oof_ridge}


cat_pipe = Pipeline([('sel', tml.DtypeSel('category')),
                     ('imputer', tml.DfImputer(fill_value="unknown", strategy="constant")),
                     ("te", tml.TargetEncoder(agg_func="mean")),
                    ])

num_pipe = Pipeline([('sel', tml.DtypeSel('numeric')),
                     ('imputer', tml.DfImputer(strategy='mean')),
                     ])

processing_pipe = tml.FeatureUnionDf(transformer_list=[('num', num_pipe),
                                                       ('cat_means', cat_pipe)
                                                      ])

processing = Pipeline([("gc", GeneralCleaner()),
                       ("fe", FeatureEng(road_risk=True, curv_speed=False)),
                       ("pipe", processing_pipe),
                      ])

full_pipe = Pipeline([("proc", processing), 
                      ("model", lgb.LGBMRegressor(n_jobs=-1, random_state=23,
                                                  colsample_bytree=0.91,
                                                  min_child_weight=30,
                                                  reg_lambda=16,
                                                  reg_alpha=1.3,
                                                  subsample=0.9,
                                                  num_leaves=27,
                                                  n_estimators=10000, verbose=-1))])

callbacks = [lgb.early_stopping(100, verbose=0)]
fit_params_lgb = {"callbacks":callbacks, "eval_metric": "rmse"}

oof_lgb, pred_lgb, res_lgb = train_predict(data=df, target=target, test_data=test, test_target=test_target,
                                                 estimator=full_pipe, cv=kfolds, result_compare=result_compare,
                                                    fit_params=fit_params_lgb, early_stopping=True)

lgb_score = np.sqrt(mean_squared_error(y_true=target, y_pred=oof_lgb))
result_compare = {"score": lgb_score, 
                  "oof": oof_lgb}


cat_pipe = Pipeline([('sel', tml.DtypeSel('category')),
                     ('imputer', tml.DfImputer(fill_value="unknown", strategy="constant")),
                     ("te", tml.TargetEncoder()),
                     #("dummies", tml.Dummify(drop_first=True))
                    ])

num_pipe = Pipeline([('sel', tml.DtypeSel('numeric')),
                     ('imputer', tml.DfImputer(strategy='mean')),
                     ])

processing_pipe = tml.FeatureUnionDf(transformer_list=[('num', num_pipe),
                                                       ('cat', cat_pipe)])

processing = Pipeline([("gc", GeneralCleaner()),
                       ("fe", FeatureEng()),
                       ("pipe", processing_pipe),
                       #("poly", tml.DfPolynomial(interaction_only=True)),
                       ])

full_pipe = Pipeline([("proc", processing), 
                      ("model", xgb.XGBRegressor(n_jobs=-1, random_state=23,
                                                  n_estimators=10000,
                                                 reg_alpha=2,
                                                 reg_lambda=17,
                                                 min_child_weight=1,
                                                 max_depth=16,
                                                early_stopping_rounds=100,
                                                  eval_metric="rmse",))])

fit_params_xgb = {'verbose': False}

oof_xgb, pred_xgb, res_xgb = train_predict(data=df, target=target, test_data=test, test_target=test_target,
                                                 estimator=full_pipe, cv=kfolds, result_compare=result_compare,
                                                    fit_params=fit_params_xgb, early_stopping=True)


 def objective(trial, data=df, target=target):
    param = {
        # "num_leaves": trial.suggest_int("num_leaves", 10, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 60),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 100.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 100.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 1),
        'subsample': trial.suggest_float('subsample', 0.4, 1),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3 , 300),
        "road_risk": trial.suggest_categorical("road_risk", [True, False]),
        "curv_speed": trial.suggest_categorical("curv_speed", [True, False])
    }
    cat_pipe = Pipeline([('sel', tml.DtypeSel('category')),
                     ('imputer', tml.DfImputer(fill_value="unknown", strategy="constant")),
                     ("te", tml.TargetEncoder(agg_func="mean")),
                    ])

    num_pipe = Pipeline([('sel', tml.DtypeSel('numeric')),
                         ('imputer', tml.DfImputer(strategy='mean')),
                         ])
    
    processing_pipe = tml.FeatureUnionDf(transformer_list=[('num', num_pipe),
                                                           ('cat_means', cat_pipe)
                                                          ])
    
    processing = Pipeline([("gc", GeneralCleaner()),
                           ("fe", FeatureEng(road_risk=param["road_risk"], curv_speed=param["curv_speed"])),
                           ("pipe", processing_pipe),
                          ])

    full_pipe = Pipeline([("proc", processing), 
                      ("model", xgb.XGBRegressor(n_jobs=-1, random_state=23,
                                                 reg_lambda=param['reg_lambda'],
                                                      reg_alpha=param['reg_alpha'],
                                                      subsample=param['subsample'],
                                                      max_depth=param['max_depth'],
                                                      min_child_weight=param['min_child_weight'],
                                                      colsample_bytree=param["colsample_bytree"],
                                                 colsample_bylevel=param["colsample_bylevel"],
                                                early_stopping_rounds=100,
                                                  eval_metric="rmse",))])

    fit_params = {'verbose': False}

    cv_score = tml.CrossValidate(data=data, target=target, 
                                 estimator=full_pipe, cv=kfolds, fit_params=fit_params, early_stopping=True,
                                 shap=False, imp_coef=False)

    oof, _ = cv_score.score()
    
    rmse = np.sqrt(mean_squared_error(y_true=target, y_pred=oof))
    
    return rmse


# sampler = TPESampler(seed=645)  # Make the sampler behave in a deterministic way.

# study = optuna.create_study(direction='minimize', sampler=sampler)
# optuna.logging.set_verbosity(optuna.logging.WARNING)
# study.optimize(objective, n_trials=200, n_jobs=-1)
# print('Number of finished trials:', len(study.trials))
# print('Best trial:', study.best_trial.params)


# study.trials_dataframe().sort_values('value').head(10)


#df_train = pd.concat([df_train.drop("id", axis=1), df_original], ignore_index=True)

df = df_train.drop(["accident_risk", "id"], axis=1)
target = df_train["accident_risk"]

df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
df_test = df_test.drop("id", axis=1)

df_test.head()


cat_pipe = Pipeline([('sel', tml.DtypeSel('category')),
                     ('imputer', tml.DfImputer(fill_value="unknown", strategy="constant")),
                     ("te", tml.TargetEncoder(agg_func="mean")),
                    ])

num_pipe = Pipeline([('sel', tml.DtypeSel('numeric')),
                     ('imputer', tml.DfImputer(strategy='mean')),
                     ])

processing_pipe = tml.FeatureUnionDf(transformer_list=[('num', num_pipe),
                                                       ('cat_means', cat_pipe)
                                                      ])

processing = Pipeline([("gc", GeneralCleaner()),
                       ("fe", FeatureEng(road_risk=True, curv_speed=False)),
                       ("pipe", processing_pipe),
                      ])

full_pipe = Pipeline([("proc", processing), 
                      ("model", lgb.LGBMRegressor(n_jobs=-1, random_state=23,
                                                  colsample_bytree=0.91,
                                                  min_child_weight=30,
                                                  reg_lambda=16,
                                                  reg_alpha=1.3,
                                                  subsample=0.9,
                                                  num_leaves=27,
                                                  n_estimators=10000, verbose=-1))])

callbacks = [lgb.early_stopping(100, verbose=0)]
fit_params_lgb = {"callbacks":callbacks, "eval_metric": "rmse"}


oof_final, pred_final, res_final = train_predict(data=df, target=target, test_data=df_test, 
                                                 estimator=full_pipe, cv=kfolds,
                                                fit_params=fit_params_lgb, early_stopping=True)


sub = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")
sub["accident_risk"] = pred_final


sub.to_csv("submission.csv", index=False)

sub

