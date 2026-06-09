!pip install tubesml


import numpy as np 
import pandas as pd

import tubesml as tml
from tubesml.base import BaseTransformer, fit_wrapper, transform_wrapper

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
%matplotlib inline

import lightgbm as lgb
import xgboost as xgb

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
df_original = pd.read_csv("/kaggle/input/bank-marketing-dataset-full/bank-full.csv", sep=";")
df_original["y"] = df_original["y"].map({"no": 0, "yes": 1}).astype(int)
df_test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
_ = tml.list_missing(df_train)
_ = tml.list_missing(df_original)
print(df_train["id"].nunique() - len(df_train))
df_train.head()


df_original.head()


df_train.info()


print(df_train["y"].mean(), df_original["y"].mean())


for col in df_train.select_dtypes(exclude="number"):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))

    df_train.groupby("y")[col].value_counts().unstack().T.plot(kind="bar", stacked=True, ax=ax[0])
    df_train.groupby(col)["y"].mean().plot(kind="bar", ax=ax[1])
    plt.show()


df_train.describe().T


df_original.describe().T


class GeneralCleaner(BaseTransformer):
    def __init__(self, remove_outliers=True, margin=0.005):
        self.remove_outliers = remove_outliers
        self.margin = margin
    
    @transform_wrapper
    def transform(self, X, y=None):
        Xtr = X.copy()

        if self.remove_outliers:
            for col in ["duration", "balance","campaign", "previous", "pdays"]:
                lower_bound = Xtr[col].quantile(0 + self.margin)
                upper_bound = Xtr[col].quantile(1 - self.margin)
                Xtr[col] = Xtr[col].clip(lower=lower_bound, upper=upper_bound)

        for col in ["duration", "campaign", "previous"]:
            Xtr[col] = np.log1p(Xtr[col])

        Xtr["pdays"] = np.log1p(Xtr["pdays"] + 1)
        # Xtr["previous"] = np.log1p(Xtr["previous"] + 1)

        Xtr["month"] = Xtr["month"].map({"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5,
                                         "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10,
                                         "nov": 11, "dec": 12}).astype(int)
        
        return Xtr


gencl = GeneralCleaner()

tmp = gencl.fit_transform(df_train)


num_cor = tml.plot_correlations(data=tmp.select_dtypes('number'), target="y", annot=True)
num_cor


for col in num_cor.index[1:]:
    if col == "id":
        continue
    tml.segm_target(data=tmp, cat="y", target=col)


train, test = tml.make_test(df_train, test_size=0.2, random_state=324)


df_total = pd.concat([train, df_original], ignore_index=True)


oof_baseline = np.where((train["duration"] > 200), 1, 0)
oof_baseline_total = np.where((df_total["duration"] > 200), 1, 0)
pred_test = np.where((test["duration"] > 200), 1, 0)

baseline_score = roc_auc_score(y_true=train["y"], y_score=oof_baseline)
baseline_score_total = roc_auc_score(y_true=df_total["y"], y_score=oof_baseline_total)
baseline_test = roc_auc_score(y_true=test["y"], y_score=pred_test)

print(f"Baseline OOF score: \t{baseline_score}")
print(f"Baseline total OOF score: \t{baseline_score_total}")
print(f"Baseline test score: \t{baseline_test}")


class GeneralCleaner(BaseTransformer):
    def __init__(self, remove_outliers=True, margin=0.005):
        self.remove_outliers = remove_outliers
        self.margin = margin
    
    @transform_wrapper
    def transform(self, X, y=None):
        Xtr = X.copy()

        if self.remove_outliers:
            for col in ["duration", "balance","campaign", "previous", "pdays"]:
                lower_bound = Xtr[col].quantile(0 + self.margin)
                upper_bound = Xtr[col].quantile(1 - self.margin)
                Xtr[col] = Xtr[col].clip(lower=lower_bound, upper=upper_bound)

        for col in ["duration", "campaign", "previous"]:
            Xtr[col] = np.log1p(Xtr[col])

        Xtr["pdays"] = np.log1p(Xtr["pdays"] + 1)
        # Xtr["previous"] = np.log1p(Xtr["previous"] + 1)

        Xtr["month"] = Xtr["month"].map({"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5,
                                         "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10,
                                         "nov": 11, "dec": 12}).astype(int)
        
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


N_FOLDS = 5
kfolds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=13)

df = train.drop(["id", "y"], axis=1)
target = train["y"]

total = df_total.drop(["id", "y"], axis=1)
target_total = df_total["y"]

test_target = test["y"]

# test = df_test.drop("id", axis=1)


def train_predict(data, test_data, target, estimator, cv, fit_params=None, early_stopping=False, shap=True, total=False):
    cv_score = tml.CrossValidate(data=data, target=target, test=test_data,
                                 estimator=estimator, cv=cv, fit_params=fit_params, early_stopping=early_stopping,
                                shap=shap, imp_coef=True, predict_proba=True, regression=False)

    oof, pred, result_dict = cv_score.score()

    binary_oof = (oof > 0.5).astype(int)
    score = roc_auc_score(y_true=target, y_score=oof)

    if early_stopping:
        print(result_dict["iterations"])

    if shap:
        tml.plot_feat_imp(result_dict["feat_imp"], imp="both", n=10)
        to_plot = result_dict["feat_imp"].head(10)["Feature"].to_list()
        tml.plot_shap_values(result_dict["shap_values"], features=to_plot)
    else:
        tml.plot_feat_imp(result_dict["feat_imp"], imp="standard", n=10)

    tml.eval_classification(data=data, target=target, preds=oof, proba=True, thrs=0.5, plot=True, feat="duration")

    if total:
        baseline = baseline_score_total
        ref_oof = oof_baseline_total
    else:
        baseline = baseline_score
        ref_oof = oof_baseline

    print(f"This is {round((score - baseline) / baseline * 100, 3)}% better than the baseline.")

    comparison = tml.CompareModels(data=data, true_label=target.reset_index(drop=True),
                                   pred_1=oof, pred_2=ref_oof,
                                   kfold=cv,
                                   regression=False,
                                   probabilities=True,
                                   metric_func=roc_auc_score)

    comparison.compare_metrics()
    comparison.compare_predictions()
    comparison.statistical_significance()

    tml.eval_classification(data=test_data, target=test_target, preds=pred, proba=True, thrs=0.5, plot=True, feat="duration")

    return oof, pred, result_dict


cat_pipe = Pipeline([('sel', tml.DtypeSel('category')),
                     ('imputer', tml.DfImputer(fill_value="unknown", strategy="constant")),
                     ("dummies", tml.Dummify(drop_first=True))])

num_pipe = Pipeline([('sel', tml.DtypeSel('numeric')),
                     ('imputer', tml.DfImputer(strategy='mean')),
                     ])

processing_pipe = tml.FeatureUnionDf(transformer_list=[('num', num_pipe),
                                                       ('cat', cat_pipe)])

processing = Pipeline([("gc", GeneralCleaner(remove_outliers=True)),
                       ("pipe", processing_pipe),
                       ("poly", tml.DfPolynomial(interaction_only=True)),
                       ("scaler", tml.DfScaler())])

full_pipe = Pipeline([("proc", processing), 
                      ("model", LogisticRegression(random_state=45, max_iter=5000))])


oof_logit, pred_logit, res_logit = train_predict(data=df, target=target, test_data=test, estimator=full_pipe, cv=kfolds)


cat_pipe = Pipeline([('sel', tml.DtypeSel('category')),
                     ('imputer', tml.DfImputer(fill_value="unknown", strategy="constant")),
                     ("dummies", tml.Dummify(drop_first=True))])

num_pipe = Pipeline([('sel', tml.DtypeSel('numeric')),
                     ('imputer', tml.DfImputer(strategy='mean')),
                     ("poly", tml.DfPolynomial(interaction_only=True))])

processing_pipe = tml.FeatureUnionDf(transformer_list=[('num', num_pipe),
                                                       ('cat', cat_pipe)])

processing = Pipeline([("gc", GeneralCleaner(remove_outliers=True)),
                       ("pipe", processing_pipe),
                       ("scaler", tml.DfScaler())])

full_pipe = Pipeline([("proc", processing), 
                      ("model", LogisticRegression(random_state=45, max_iter=5000))])


oof_logit_total, pred_logit_total, res_logit_total = train_predict(data=total, target=target_total, test_data=test, estimator=full_pipe, cv=kfolds, total=True)


baseline_score = roc_auc_score(y_true=train["y"], y_score=oof_logit)
oof_baseline = oof_logit

baseline_score_total = roc_auc_score(y_true=df_total["y"], y_score=oof_logit_total)
oof_baseline_total = oof_logit_total


cats = ['job', 'marital', 'education',
        'default', 'housing', 'loan', 'contact',
        'month', 'poutcome']

cat_pipe = Pipeline([('sel', tml.DtypeSel('category')),
                     ("te", tml.TargetEncoder())
                    ])

num_pipe = Pipeline([('sel', tml.DtypeSel('numeric')),
                     ('imputer', tml.DfImputer(strategy='mean'))])

processing_pipe = tml.FeatureUnionDf(transformer_list=[('num', num_pipe),
                                                       ('cat', cat_pipe)])

processing = Pipeline([("gc", GeneralCleaner(remove_outliers=False)),
                       ("pipe", processing_pipe)
                       ])

full_pipe = Pipeline([("processing", processing),
                      ("model", lgb.LGBMClassifier(random_state=2345,
                                                   n_estimators=10000,
                                                   learning_rate=0.05,
                                                   min_child_samples=9,
                                                   subsample=0.8,
                                                   colsample_bytree=0.5,
                                                   num_leaves=100,
                                                   max_depth=10,
                                                   max_bin=3600,
                                                   reg_alpha=0.79,
                                                   reg_lambda=3,
                                                   verbose=-1))])

callbacks = [lgb.early_stopping(100, verbose=0)]
fit_params = {"callbacks":callbacks, "eval_metric": "auc"}


oof_lgb, pred_lgb, res_lgb = train_predict(data=df, target=target, test_data=test,
                                           estimator=full_pipe, cv=kfolds, shap=False,
                                           fit_params=fit_params, early_stopping=True)


cat_pipe = Pipeline([('sel', tml.DtypeSel('category')),
                     ("te", tml.TargetEncoder())
                    ])

num_pipe = Pipeline([('sel', tml.DtypeSel('numeric')),
                     ('imputer', tml.DfImputer(strategy='mean'))])

processing_pipe = tml.FeatureUnionDf(transformer_list=[('num', num_pipe),
                                                       ('cat', cat_pipe)])

processing = Pipeline([("gc", GeneralCleaner(remove_outliers=False)),
                       ("pipe", processing_pipe)
                       ])

full_pipe = Pipeline([("processing", processing),
                      ("model", lgb.LGBMClassifier(random_state=2345,
                                                   n_estimators=10000,
                                                   learning_rate=0.05,
                                                   min_child_samples=9,
                                                   subsample=0.8,
                                                   colsample_bytree=0.5,
                                                   num_leaves=100,
                                                   max_depth=10,
                                                   max_bin=3600,
                                                   reg_alpha=0.79,
                                                   reg_lambda=3,
                                                   n_jobs=-1,
                                                   verbose=-1))])

callbacks = [lgb.early_stopping(100, verbose=0)]
fit_params = {"callbacks":callbacks, "eval_metric": "auc"}


oof_lgb_total, pred_lgb_total, res_lgb_total = train_predict(data=total, target=target_total, test_data=test,
                                           estimator=full_pipe, cv=kfolds, shap=False,
                                           fit_params=fit_params, early_stopping=True, total=True)


baseline_score = roc_auc_score(y_true=train["y"], y_score=oof_lgb)
oof_baseline = oof_lgb

baseline_score_total = roc_auc_score(y_true=df_total["y"], y_score=oof_lgb_total)
oof_baseline_total = oof_lgb_total


cat_pipe = Pipeline([('sel', tml.DtypeSel('category')),
                     ('imputer', tml.DfImputer(fill_value="unknown", strategy="constant")),
                     ("dummies", tml.Dummify(drop_first=True))])

num_pipe = Pipeline([('sel', tml.DtypeSel('numeric')),
                     ('imputer', tml.DfImputer(strategy='mean'))])

processing_pipe = tml.FeatureUnionDf(transformer_list=[('num', num_pipe),
                                                       ('cat', cat_pipe)])

processing = Pipeline([("gc", GeneralCleaner(remove_outliers=True)),
                       ("pipe", processing_pipe),
                       ])

full_pipe = Pipeline([("processing", processing),
                      ("model", xgb.XGBClassifier(n_estimators=10000,
                                                  random_state=324,
                                                  learning_rate=0.1,
                                                  subsample=0.8,
                                                  colsample_bytree=0.7,
                                                  reg_alpha=2,
                                                  # subsample=0.8,
                                                  # reg_lambda=10,
                                                  # reg_alpha=10,
                                                  # max_depth=10,
                                                  # colsample_bytree=0.6,
                                                  early_stopping_rounds=100,
                                                  eval_metric="auc",
                                                  n_jobs=-1))])


fit_params = {'verbose': False}

oof_xgb, pred_xgb, res_xgb = train_predict(data=df, target=target, test_data=test,
                                           estimator=full_pipe, cv=kfolds, shap=True,
                                           fit_params=fit_params, early_stopping=True)


cat_pipe = Pipeline([('sel', tml.DtypeSel('category')),
                     ('imputer', tml.DfImputer(fill_value="unknown", strategy="constant")),
                     ("dummies", tml.Dummify(drop_first=True))])

num_pipe = Pipeline([('sel', tml.DtypeSel('numeric')),
                     ('imputer', tml.DfImputer(strategy='mean'))])

processing_pipe = tml.FeatureUnionDf(transformer_list=[('num', num_pipe),
                                                       ('cat', cat_pipe)])

processing = Pipeline([("gc", GeneralCleaner(remove_outliers=True)),
                       ("pipe", processing_pipe),
                       ])

full_pipe = Pipeline([("processing", processing),
                      ("model", xgb.XGBClassifier(n_estimators=10000,
                                                  random_state=324,
                                                  subsample=0.8,
                                                  reg_lambda=10,
                                                  reg_alpha=15,
                                                  max_depth=10,
                                                  colsample_bytree=0.6,
                                                  early_stopping_rounds=100,
                                                  eval_metric="auc",
                                                  n_jobs=-1))])

fit_params = {'verbose': False}

oof_xgb_total, pred_xgb_total, res_xgb_total = train_predict(data=total, target=target_total, test_data=test,
                                           estimator=full_pipe, cv=kfolds, shap=True,
                                           fit_params=fit_params, early_stopping=True, total=True)


# tmp = df.copy()
# tmp = processing.fit_transform(tmp, target)

# tmp["true_label"] = target
# tmp["prediction"] = oof_lgb

# analysis = tml.ErrorAnalyzer(data=tmp.sample(100000), prediction_column="prediction", true_label="true_label")
# analysis.fit()

# viz = tml.VisualizeError(analysis=analysis)
# viz.plot_feature_importance(n=10, imp="both")
# viz.plot_error_rates(n=10)
# viz.plot_pdp(n=10)


# full_pipe = Pipeline([("proc", processing), 
#                       ("model", lgb.LGBMClassifier(random_state=2345,
#                                                    n_estimators=1500,
#                                                    learning_rate=0.05,
#                                                    min_child_samples=9,
#                                                    subsample=0.8,
#                                                    colsample_bytree=0.5,
#                                                    num_leaves=100,
#                                                    max_depth=10,
#                                                    max_bin=3600,
#                                                    reg_alpha=0.79,
#                                                    reg_lambda=3,
#                                                    n_jobs=-1,
#                                                    verbose=-1))])

# tml.plot_learning_curve(estimator=full_pipe, X=df, y=target, cv=kfolds,
#                         train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)


# cat_pipe = Pipeline([('sel', tml.DtypeSel('category')),
#                      ('imputer', tml.DfImputer(fill_value="unknown", strategy="constant")),
#                      ("dummies", tml.Dummify(drop_first=True))])

# num_pipe = Pipeline([('sel', tml.DtypeSel('numeric')),
#                      ('imputer', tml.DfImputer(strategy='mean')),
#                      ])

# processing_pipe = tml.FeatureUnionDf(transformer_list=[('num', num_pipe),
#                                                        ('cat', cat_pipe)])

# processing = Pipeline([("gc", GeneralCleaner(remove_outliers=True)),
#                        ("pipe", processing_pipe),
#                        ("poly", tml.DfPolynomial(interaction_only=True)),
#                        ("scaler", tml.DfScaler())])

# full_pipe = Pipeline([("proc", processing), 
#                       ("model", LogisticRegression(random_state=45,
#                                                    max_iter=5000))])


# grid_params = {
#                'model__C': np.arange(1, 20),
#                'proc__gc__remove_outliers': [True, False],
#               }

# res, bp, be = tml.grid_search(data=df, target=target, estimator=full_pipe,
#                               cv=kfolds, random=3, param_grid=grid_params, 
#                               scoring='roc_auc')
# print(bp)

# res.head(10)


def train_predict(data, test_data, target, estimator, cv, fit_params=None, early_stopping=False, shap=True, total=False):
    cv_score = tml.CrossValidate(data=data, target=target, test=test_data,
                                 estimator=estimator, cv=cv, fit_params=fit_params, early_stopping=early_stopping,
                                shap=shap, imp_coef=True, predict_proba=True, regression=False)

    oof, pred, result_dict = cv_score.score()

    binary_oof = (oof > 0.5).astype(int)
    score = roc_auc_score(y_true=target, y_score=oof)

    if early_stopping:
        print(result_dict["iterations"])

    if shap:
        tml.plot_feat_imp(result_dict["feat_imp"], imp="both", n=10)
        to_plot = result_dict["feat_imp"].head(10)["Feature"].to_list()
        tml.plot_shap_values(result_dict["shap_values"], features=to_plot)
    else:
        tml.plot_feat_imp(result_dict["feat_imp"], imp="standard", n=10)

    tml.eval_classification(data=data, target=target, preds=oof, proba=True, thrs=0.5, plot=True, feat="duration")

    return oof, pred, result_dict


cat_pipe = Pipeline([('sel', tml.DtypeSel('category')),
                     ("te", tml.TargetEncoder())
                    ])

num_pipe = Pipeline([('sel', tml.DtypeSel('numeric')),
                     ('imputer', tml.DfImputer(strategy='mean'))])

processing_pipe = tml.FeatureUnionDf(transformer_list=[('num', num_pipe),
                                                       ('cat', cat_pipe)])

processing = Pipeline([("gc", GeneralCleaner(remove_outliers=False)),
                       ("pipe", processing_pipe)
                       ])

full_pipe = Pipeline([("processing", processing),
                      ("model", lgb.LGBMClassifier(random_state=2345,
                                                   n_estimators=10000,
                                                   learning_rate=0.05,
                                                   min_child_samples=9,
                                                   subsample=0.8,
                                                   colsample_bytree=0.5,
                                                   num_leaves=100,
                                                   max_depth=10,
                                                   max_bin=3600,
                                                   reg_alpha=0.79,
                                                   reg_lambda=3,
                                                   n_jobs=-1,
                                                   verbose=-1))])

callbacks = [lgb.early_stopping(100, verbose=0)]
fit_params = {"callbacks":callbacks, "eval_metric": "auc"}

df_total = pd.concat([df_train, df_original], ignore_index=True)

df = df_total.drop(["id", "y"], axis=1)
target = df_total["y"]

test = df_test.drop(["id"], axis=1)


oof_lgb_final, pred_lgb_final, res_lgb_final = train_predict(data=df, target=target, test_data=test,
                                           estimator=full_pipe, cv=kfolds, shap=False,
                                           fit_params=fit_params, early_stopping=True, total=False)


sub = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
sub["y"] = pred_lgb_final

sub.to_csv("submission.csv", index=False)

sub

