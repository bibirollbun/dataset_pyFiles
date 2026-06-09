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

import lightgbm as lgb
import xgboost as xgb

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



df_train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
_ = tml.list_missing(df_train)
print(df_train["id"].nunique() - len(df_train))
df_train.head()


df_train.info()


for col in df_train.select_dtypes(exclude="number"):
    if col == "id":
        continue
    print(col)
    print(df_train[col].value_counts(dropna=False))
    print("\n")


df_train.describe().T


class GeneralCleaner(BaseTransformer):
    
    @transform_wrapper
    def transform(self, X, y=None):
        Xtr = X.copy()

        Xtr["Stage_fear"] = Xtr["Stage_fear"].map({"Yes": 1, "No": 0})
        Xtr["Drained_after_socializing"] = Xtr["Drained_after_socializing"].map({"Yes": 1, "No": 0})
        
        return Xtr

class Categorizer(BaseTransformer):
    def __init__(self, cats=None):
        super().__init__()
        self.cats = cats

    
    @transform_wrapper
    def transform(self, X, y=None):
        Xtr = X.copy()
        if self.cats is None:
            cols = X.columns
        else:
            cols = self.cats
        for c in cols:
            Xtr[c] = Xtr[c].astype("category")

        return Xtr


tmp = df_train.copy()

tmp["target"] = tmp["Personality"].map({"Introvert": 0, "Extrovert": 1})

gencl = GeneralCleaner()
tmp = gencl.fit_transform(tmp)

num_cor = tml.plot_correlations(data=tmp.select_dtypes('number'), target="target", annot=True)
num_cor



for col in num_cor.index[1:]:
    tml.segm_target(data=tmp, cat="Personality", target=col)


oof_baseline = np.where((df_train["Stage_fear"] == "Yes") | (df_train["Drained_after_socializing"] == "Yes"), 0, 1)

baseline_score = accuracy_score(y_true=tmp["target"], y_pred=oof_baseline)

baseline_score


N_FOLDS = 5
kfolds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=13)

df = df_train.drop(["id", "Personality"], axis=1)
target = df_train["Personality"].map({"Introvert": 0, "Extrovert": 1}).astype(int)

test = df_test.drop("id", axis=1)


def train_predict(data, test_data, target, estimator, cv, fit_params=None, early_stopping=False, shap=True):
    cv_score = tml.CrossValidate(data=data, target=target, test=test_data,
                                 estimator=estimator, cv=cv, fit_params=fit_params, early_stopping=early_stopping,
                                shap=shap, imp_coef=True, predict_proba=True, regression=False)

    oof, pred, result_dict = cv_score.score()

    binary_oof = (oof > 0.5).astype(int)
    score = accuracy_score(y_true=target, y_pred=binary_oof)

    if early_stopping:
        print(result_dict["iterations"])

    if shap:
        tml.plot_feat_imp(result_dict["feat_imp"], imp="both", n=10)
        to_plot = result_dict["feat_imp"].head(10)["Feature"].to_list()
        tml.plot_shap_values(result_dict["shap_values"], features=to_plot)
    else:
        tml.plot_feat_imp(result_dict["feat_imp"], imp="standard", n=10)

    tml.eval_classification(data=df, target=target, preds=oof, proba=True, thrs=0.5, plot=True)

    print(f"This is {round((score - baseline_score) / baseline_score * 100, 3)}% better than the baseline.")

    comparison = tml.CompareModels(data=df, true_label=target.reset_index(drop=True),
                                   pred_1=oof, pred_2=oof_baseline,
                                   kfold=cv,
                                   regression=False,
                                   probabilities=True,
                                   metric_func=roc_auc_score)

    comparison.compare_metrics()
    comparison.compare_predictions()
    comparison.statistical_significance()

    return oof, pred, result_dict


cat_pipe = Pipeline([('sel', tml.DtypeSel('category')),
                     ('imputer', tml.DfImputer(fill_value="missing", strategy="constant")),
                     ("dummies", tml.Dummify(drop_first=True))])

num_pipe = Pipeline([('sel', tml.DtypeSel('numeric')),
                     ('imputer', tml.DfImputer(strategy='mean'))])

processing_pipe = tml.FeatureUnionDf(transformer_list=[('num', num_pipe),
                                                       ('cat', cat_pipe)])

processing = Pipeline([("pipe", processing_pipe),
                       ("poly", tml.DfPolynomial(degree=3, interaction_only=True)),
                       ("scaler", tml.DfScaler())])

full_pipe = Pipeline([("proc", processing), 
                      ("model", LogisticRegression(random_state=45, max_iter=5000))])


oof_logit, pred_logit, res_logit = train_predict(data=df, target=target, test_data=test, estimator=full_pipe, cv=kfolds)


cat_pipe = Pipeline([('sel', tml.DtypeSel('category')),
                     ('imputer', tml.DfImputer(fill_value="missing", strategy="constant")),
                     ("dummies", tml.Dummify(drop_first=True))])

num_pipe = Pipeline([('sel', tml.DtypeSel('numeric')),
                     ('imputer', tml.DfImputer(strategy='mean'))])

processing_pipe = tml.FeatureUnionDf(transformer_list=[('num', num_pipe),
                                                       ('cat', cat_pipe)])

processing = Pipeline([("pipe", processing_pipe),
                       ("poly", tml.DfPolynomial(interaction_only=True))])

full_pipe = Pipeline([("proc", processing), 
                      ("model", RandomForestClassifier(n_estimators=1000, random_state=467))])


oof_forest, pred_forest, res_forest = train_predict(data=df, target=target, test_data=test, estimator=full_pipe, cv=kfolds, shap=False)


cat_pipe = Pipeline([('sel', tml.DtypeSel('category')),
                     #('imputer', tml.DfImputer(strategy='most_frequent', add_indicator=True)),
                     ("dummies", tml.Dummify(drop_first=True))
                    ])

num_pipe = Pipeline([('sel', tml.DtypeSel('numeric')),
                     ('imputer', tml.DfImputer(strategy='mean'))])

processing_pipe = tml.FeatureUnionDf(transformer_list=[('num', num_pipe),
                                                       ('cat', cat_pipe)])

processing = Pipeline([("pipe", processing_pipe),
                       ("poly", tml.DfPolynomial(interaction_only=True))])

full_pipe = Pipeline([("processing", processing), 
                      ("model", lgb.LGBMClassifier(random_state=2345,
                                                   n_estimators=10000,
                                                   reg_alpha=10,
                                                   reg_lambda=10,
                                                   verbose=-1))])

callbacks = [lgb.early_stopping(100, verbose=0)]
fit_params = {"callbacks":callbacks}


oof_lgb, pred_lgb, res_lgb = train_predict(data=df, target=target, test_data=test,
                                           estimator=full_pipe, cv=kfolds, shap=True,
                                           fit_params=fit_params, early_stopping=True)


cat_pipe = Pipeline([('sel', tml.DtypeSel('category')),
                     ('imputer', tml.DfImputer(fill_value="missing", strategy="constant")),
                     ("dummies", tml.Dummify(drop_first=True))])

num_pipe = Pipeline([('sel', tml.DtypeSel('numeric')),
                     ('imputer', tml.DfImputer(strategy='mean'))])

processing_pipe = tml.FeatureUnionDf(transformer_list=[('num', num_pipe),
                                                       ('cat', cat_pipe)])

processing = Pipeline([("pipe", processing_pipe),
                       ("poly", tml.DfPolynomial(interaction_only=True))])

full_pipe = Pipeline([("proc", processing), 
                      ("model", xgb.XGBClassifier(n_estimators=10000,
                                                  max_depth=20,
                                                  subsample=0.5,
                                                  reg_lambda=10,
                                                  reg_alpha=10,
                                                  random_state=324,
                                                  early_stopping_rounds=100))])

fit_params = {'verbose': False}

oof_xgb, pred_xgb, res_xgb = train_predict(data=df, target=target, test_data=test,
                                           estimator=full_pipe, cv=kfolds, shap=True,
                                           fit_params=fit_params, early_stopping=True)


tmp = df.copy()
tmp = processing_pipe.fit_transform(tmp, target)

tmp["true_label"] = target
tmp["prediction"] = oof_xgb

analysis = tml.ErrorAnalyzer(data=tmp, prediction_column="prediction", true_label="true_label")
analysis.fit()

viz = tml.VisualizeError(analysis=analysis)
viz.plot_feature_importance(n=10, imp="both")
viz.plot_error_rates(n=4)
viz.plot_pdp(n=4)


full_pipe = Pipeline([("proc", processing), 
                      ("model", xgb.XGBClassifier(n_estimators=70,
                                                  max_depth=20,
                                                  subsample=0.5,
                                                  reg_lambda=10,
                                                  reg_alpha=10,
                                                  random_state=324))])

tml.plot_learning_curve(estimator=full_pipe, X=df, y=target, cv=kfolds,
                        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)


cat_pipe = Pipeline([('sel', tml.DtypeSel('category')),
                     ('imputer', tml.DfImputer(fill_value="missing", strategy="constant")),
                     ("dummies", tml.Dummify(drop_first=True))])

num_pipe = Pipeline([('sel', tml.DtypeSel('numeric')),
                     ('imputer', tml.DfImputer(strategy='mean'))])

processing_pipe = tml.FeatureUnionDf(transformer_list=[('num', num_pipe),
                                                       ('cat', cat_pipe)])

processing = Pipeline([("pipe", cat_pipe),
                       ("poly", tml.DfPolynomial(degree=2, interaction_only=True)),
                       ("scaler", tml.DfScaler())])

full_pipe = Pipeline([("proc", processing), 
                      ("model", LogisticRegression(random_state=45, max_iter=5000))])


oof_logit_cat, pred_logit_cat, res_logit_cat = train_predict(data=df, target=target, test_data=test, estimator=full_pipe, cv=kfolds)


cat_pipe = Pipeline([('sel', tml.DtypeSel('category')),
                     ('imputer', tml.DfImputer(fill_value="missing", strategy="constant")),
                     ("dummies", tml.Dummify(drop_first=True))])

num_pipe = Pipeline([('sel', tml.DtypeSel('numeric')),
                     ('imputer', tml.DfImputer(strategy='mean'))])

processing_pipe = tml.FeatureUnionDf(transformer_list=[('num', num_pipe),
                                                       ('cat', cat_pipe)])

processing = Pipeline([("pipe", num_pipe),
                       ("poly", tml.DfPolynomial(degree=3, interaction_only=True)),
                       ("scaler", tml.DfScaler())])

full_pipe = Pipeline([("proc", processing), 
                      ("model", LogisticRegression(random_state=45, max_iter=5000))])


oof_logit_num, pred_logit_num, res_logit_num = train_predict(data=df, target=target, test_data=test, estimator=full_pipe, cv=kfolds)


comparison = tml.CompareModels(data=df, true_label=target.reset_index(drop=True),
                                   pred_1=oof_logit_cat, pred_2=oof_logit_num,
                                   kfold=kfolds,
                                   regression=False,
                                   probabilities=True,
                                   metric_func=roc_auc_score)

comparison.compare_metrics()
comparison.compare_predictions()
comparison.statistical_significance()





sub = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
preds = (pred_xgb > 0.5).astype(int)
sub["Personality"] = preds
sub["Personality"] = sub["Personality"].map({1: "Extrovert", 0: "Introvert"})

sub.to_csv("submission.csv", index=False)

sub




