!pip install tubesml


from itertools import combinations, product

import numpy as np 
import pandas as pd 

import tubesml as tml

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
tmp = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')

df_train = pd.concat([df, tmp], ignore_index=True)
df_test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

df_train.head()


_ = tml.list_missing(df_train)
print("_"*40)
_ = tml.list_missing(df_test)


df_train["id"].nunique() - len(df_train)


train, test = tml.make_test(train=df_train, test_size=0.2, random_state=43)

print(train["Brand"].value_counts(normalize=True))
print(test["Brand"].value_counts(normalize=True))


tml.plot_distribution(data=train, column="Price")


corr = tml.plot_correlations(data=train.select_dtypes("number"), target="Price", annot=True)


cats = tml.find_cats(data=train, target="Price")
for cat in cats:
    tml.segm_target(data=train, target="Price", cat=cat)


def _generate_combinations(lst):
    comb = []
    for i in range(1, len(lst) + 1):
        comb.extend(combinations(lst, i))
    return comb

def make_baseline(data_train, data_test, target, columns, functions, evaluation, sqrt=True):
    col_combs = _generate_combinations(columns)
    all_combs = list(product(col_combs, functions))

    for func in functions:
        vals_tot = data_train[target].agg(func)
        prediction_train = [vals_tot] * len(data_train)
        prediction_test = [vals_tot] * len(data_test)
    
        error_train = evaluation(y_true=data_train[target], y_pred=prediction_train)
        error_test = evaluation(y_true=data_test[target], y_pred=prediction_test)
    
        if sqrt:
            error_train = np.sqrt(error_train)
            error_test = np.sqrt(error_test)
    
        print(f"Baseline with {func}:")
        print(f"\t Train: {round(error_train, 3)}")
        print(f"\t Train: {round(error_test, 3)}")

    for comb in all_combs:
        if len(comb[0]) == 1:
            cols = comb[0][0]
        else:
            cols = list(comb[0])
        
        vals = data_train.groupby(cols)[target].agg(comb[1]).reset_index().rename(columns={target: "Prediction"})
        vals_tot = data_train[target].agg(comb[1])
        tmp = pd.merge(data_train, vals, on=cols, how="left")
        prediction_train = tmp["Prediction"].fillna(vals_tot)
        tmp = pd.merge(data_test, vals, on=cols, how="left")
        prediction_test = tmp["Prediction"].fillna(vals_tot)
        
        error_train = evaluation(y_true=data_train[target], y_pred=prediction_train)
        error_test = evaluation(y_true=data_test[target], y_pred=prediction_test)

        if sqrt:
            error_train = np.sqrt(error_train)
            error_test = np.sqrt(error_test)

        print(f"Baseline over {cols} with {comb[1]}:")
        print(f"\t Train: {round(error_train, 3)}")
        print(f"\t Test: {round(error_test, 3)}")


make_baseline(data_train=train, data_test=test, target="Price", columns=["Brand", "Color", "Size", "Style", "Material"], functions=["mean", "median"], evaluation=mean_squared_error)


tot_vals = df_train["Price"].mean()
vals = df_train.groupby(['Brand', 'Color', 'Material'])["Price"].mean().reset_index()
tmp = pd.merge(df_test, vals, on=['Brand', 'Color', 'Material'], how="left").fillna(tot_vals)

base_sub = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
base_sub["Price"] = tmp["Price"]

base_sub.to_csv("baseline_submission.csv", index=False)
base_sub.head()


cat_pipe = Pipeline([('sel', tml.DtypeSel('category')),
                     ('imputer', tml.DfImputer(strategy='most_frequent'))])
num_pipe = Pipeline([('sel', tml.DtypeSel('numeric')),
                     ('imputer', tml.DfImputer(strategy='mean'))])

full_pipe = tml.FeatureUnionDf(transformer_list=[('num', num_pipe),
                                                 ('cat', cat_pipe)])

cleaning_pipe = Pipeline([('pipe', full_pipe),
                          ('dummy', tml.Dummify(drop_first=True))])

cleaning_pipe


tmp = cleaning_pipe.fit_transform(train)
tmp.head()


N_FOLDS = 5  # FIXME: experiment on this
kfolds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=4398)

training_cols = [c for c in train if c not in ["id", "Price"]]
target = train["Price"]

training_cols


model_pipe = Pipeline([('processing', cleaning_pipe),
                       ('scaler', tml.DfScaler()),
                       ('model', Ridge(random_state=34))])


oof, res = tml.cv_score(data=train[training_cols], target=target, estimator=model_pipe, cv=kfolds, imp_coef=True)

print(round(np.sqrt(mean_squared_error(y_true=target, y_pred=oof)), 3))

tml.plot_feat_imp(res['feat_imp'], n=10)

tml.plot_regression_predictions(data=train[training_cols], true_label=target, pred_label=oof, feature=["Weight Capacity (kg)"])


model_pipe = Pipeline([('processing', cleaning_pipe),
                       ('scaler', tml.DfScaler()),
                       ('model', Lasso(alpha=0.5, random_state=34))])

oof, res = tml.cv_score(data=train[training_cols], target=target, estimator=model_pipe, cv=kfolds, imp_coef=True)

print(round(np.sqrt(mean_squared_error(y_true=target, y_pred=oof)), 3))

tml.plot_feat_imp(res['feat_imp'], n=10)

tml.plot_regression_predictions(data=train[training_cols], true_label=target, pred_label=oof, feature=["Weight Capacity (kg)"])


model_pipe = Pipeline([('processing', cleaning_pipe),
                       ('model', DecisionTreeRegressor(max_depth=10, random_state=545))])

oof, res = tml.cv_score(data=train[training_cols], target=target, estimator=model_pipe, cv=kfolds, imp_coef=True)

print(round(np.sqrt(mean_squared_error(y_true=target, y_pred=oof)), 3))

tml.plot_feat_imp(res['feat_imp'], n=10)

tml.plot_regression_predictions(data=train[training_cols], true_label=target, pred_label=oof, feature=["Weight Capacity (kg)"])


model_pipe = Pipeline([('processing', cleaning_pipe),
                       ('model', RandomForestRegressor(max_features="sqrt", max_depth=10, random_state=545, n_jobs=-1))])

oof, res = tml.cv_score(data=train[training_cols], target=target, estimator=model_pipe, cv=kfolds, imp_coef=True)

print(round(np.sqrt(mean_squared_error(y_true=target, y_pred=oof)), 3))

tml.plot_feat_imp(res['feat_imp'], n=10)

tml.plot_regression_predictions(data=train[training_cols], true_label=target, pred_label=oof, feature=["Weight Capacity (kg)"])


model_pipe = Pipeline([('processing', cleaning_pipe),
                       ('model', ExtraTreesRegressor(max_features="sqrt", max_depth=10, random_state=545, n_jobs=-1))])

oof, res = tml.cv_score(data=train[training_cols], target=target, estimator=model_pipe, cv=kfolds, imp_coef=True)

print(round(np.sqrt(mean_squared_error(y_true=target, y_pred=oof)), 3))

tml.plot_feat_imp(res['feat_imp'], n=10)

tml.plot_regression_predictions(data=train[training_cols], true_label=target, pred_label=oof, feature=["Weight Capacity (kg)"])


from tubesml.base import BaseTransformer, transform_wrapper

class Categorizer(BaseTransformer):
    def __init__(self, cats):
        super().__init__()
        self.cats = cats

    
    @transform_wrapper
    def transform(self, X, y=None):
        Xtr = X.copy()
        for c in self.cats:
            Xtr[c] = Xtr[c].astype("category")

        return Xtr


class Interactions(BaseTransformer):
    @transform_wrapper
    def transform(self, X, y=None):
        Xtr = X.copy()
        Xtr["color_brand"] = Xtr["Color"].astype(str) + "_" + X["Brand"].astype(str)
        Xtr["color_material"] = Xtr["Color"].astype(str) + "_" + X["Material"].astype(str)
        Xtr["brand_material"] = Xtr["Material"].astype(str) + "_" + X["Brand"].astype(str)
        Xtr["color_brand"] = Xtr["color_brand"].astype("category")
        Xtr["color_material"] = Xtr["color_material"].astype("category")
        Xtr["brand_material"] = Xtr["brand_material"].astype("category")

        return Xtr


CATS = ['Brand',
 'Material',
 'Size',
 'Laptop Compartment',
 'Waterproof',
 'Style',
 'Color']


model_pipe = Pipeline([('processing', cleaning_pipe),
                       ('model', xgb.XGBRegressor(n_estimators=1000, max_depth=3, subsample=0.8,
                                                  enable_categorical=True,
                                                  early_stopping_rounds=100,
                                                  random_state=33))])

fit_params = {'verbose': False}

oof, res = tml.cv_score(data=train[training_cols], target=target, estimator=model_pipe, cv=kfolds, imp_coef=True, early_stopping=True, fit_params=fit_params)

print(round(np.sqrt(mean_squared_error(y_true=target, y_pred=oof)), 3))
print(res["iterations"])

tml.plot_feat_imp(res['feat_imp'], n=10)

tml.plot_regression_predictions(data=train[training_cols], true_label=target, pred_label=oof, feature=["Weight Capacity (kg)"])


model_pipe = Pipeline([("categorizer", Categorizer(cats=CATS)),
                       ('model', lgb.LGBMRegressor(n_estimators=1000,
                                                   random_state=354,
                                                   verbose=-1))])


callbacks = [lgb.early_stopping(100, verbose=0)]
fit_params = {"callbacks":callbacks}

oof, res = tml.cv_score(data=train[training_cols], target=target,
                        estimator=model_pipe, cv=kfolds,
                        imp_coef=True, early_stopping=True, fit_params=fit_params)

print(round(np.sqrt(mean_squared_error(y_true=target, y_pred=oof)), 3))
print(res["iterations"])

tml.plot_feat_imp(res['feat_imp'], n=10)

tml.plot_regression_predictions(data=train[training_cols], true_label=target,
                                pred_label=oof, feature=["Weight Capacity (kg)"])


model_pipe = Pipeline([("clean", full_pipe),
                       ("categorizer", Categorizer(cats=CATS)),
                       ('model', cb.CatBoostRegressor(random_state=34,
                                                      cat_features=CATS,
                                                      early_stopping_rounds=100,
                                                      iterations=1000))])

fit_params = {"verbose":False}

oof, res = tml.cv_score(data=train[training_cols], target=target,
                        estimator=model_pipe, cv=kfolds, imp_coef=True,
                        early_stopping=True, fit_params=fit_params)

print(round(np.sqrt(mean_squared_error(y_true=target, y_pred=oof)), 3))
print(res["iterations"])

tml.plot_feat_imp(res['feat_imp'], n=10)

tml.plot_regression_predictions(data=train[training_cols], true_label=target,
                                pred_label=oof, feature=["Weight Capacity (kg)"])


class OrdinalFeatures(BaseTransformer):
    def __init__(self, add_interactions=False):
        super().__init__()
        self.add_interactions = add_interactions
        
    @transform_wrapper
    def transform(self, X, y=None):
        Xtr = X.copy()
        Xtr["Laptop Compartment"] = Xtr["Laptop Compartment"].map({"No": 1, "Yes": 2})
        Xtr["Waterproof"] = Xtr["Waterproof"].map({"No": 1, "Yes": 2})
        Xtr["Size"] = Xtr["Size"].map({"Small": 1, "Medium": 2, "Large": 3})

        if self.add_interactions:
            Xtr["Laptop_wc"] = Xtr["Laptop Compartment"] * Xtr["Weight Capacity (kg)"]
            Xtr["Waterproof_wc"] = Xtr["Waterproof"] * Xtr["Weight Capacity (kg)"]
            Xtr["Size_wc"] = Xtr["Size"] * Xtr["Weight Capacity (kg)"]

        return Xtr


class PrepEncoder(BaseTransformer):
    def __init__(self, to_encode, means=True, medians=True, stds=True):
        super().__init__()
        self.to_encode = to_encode
        self.means = means
        self.medians = medians
        self.stds = stds

    @transform_wrapper
    def transform(self, X, y=None):
        cols = self.to_encode
        Xtr = X.copy()
        if self.means:
            for col in cols:
                Xtr[f"{col}_mean"] = Xtr[col]
        if self.medians:
            for col in cols:
                Xtr[f"{col}_median"] = Xtr[col]
        if self.stds:
            for col in cols:
                Xtr[f"{col}_std"] = Xtr[col]

        return Xtr

class Interactions(BaseTransformer):
    @transform_wrapper
    def transform(self, X, y=None):
        Xtr = X.copy()
        Xtr["color_brand"] = Xtr["Color"].astype(str) + "_" + X["Brand"].astype(str)
        Xtr["color_material"] = Xtr["Color"].astype(str) + "_" + X["Material"].astype(str)
        Xtr["brand_material"] = Xtr["Material"].astype(str) + "_" + X["Brand"].astype(str)
        Xtr["brand_size"] = Xtr["Size"].astype(str) + "_" + X["Brand"].astype(str)
        Xtr["material_size"] = Xtr["Size"].astype(str) + "_" + X["Material"].astype(str)
        Xtr["brand_style"] = Xtr["Style"].astype(str) + "_" + X["Brand"].astype(str)
        Xtr["color_style"] = Xtr["Style"].astype(str) + "_" + X["Color"].astype(str)
        Xtr["style_size"] = Xtr["Size"].astype(str) + "_" + X["Style"].astype(str)
        # Xtr["color_brand"] = Xtr["color_brand"].astype("category")
        # Xtr["color_material"] = Xtr["color_material"].astype("category")
        # Xtr["brand_material"] = Xtr["brand_material"].astype("category")

        return Xtr


tmp = train.copy()
to_encode = ['Brand',
 'Material',
 'Style',
 'Color']+ ["color_brand", "color_material", "brand_material",
            "brand_size", "material_size", "brand_style", "color_style"]
trf = Pipeline([("nans", full_pipe),
                ("ordinal", OrdinalFeatures()),
                ("interactions", Interactions()),
                ("prep_te", PrepEncoder(to_encode=to_encode)),
                ("te_mean", tml.TargetEncoder(to_encode=[f"{c}_mean" for c in to_encode],
                                              agg_func="mean")),
                ("te_median", tml.TargetEncoder(to_encode=[f"{c}_median" for c in to_encode],
                                              agg_func="median")),
                ("te_std", tml.TargetEncoder(to_encode=[f"{c}_std" for c in to_encode],
                                              agg_func="std")),
               ])
            
res = trf.fit_transform(tmp, tmp["Price"])
res.head()


for cat in ["color_brand", "color_material", "brand_material", "brand_size",
            "material_size", "brand_style", "color_style", "style_size"]:
    tml.segm_target(data=res, target="Price", cat=cat)


corr = tml.plot_correlations(data=res.select_dtypes("number"), target="Price", annot=True)


corr[:10]


class Interactions(BaseTransformer):
    @transform_wrapper
    def transform(self, X, y=None):
        Xtr = X.copy()
        Xtr["color_brand"] = Xtr["Color"].astype(str) + "_" + X["Brand"].astype(str)
        Xtr["color_material"] = Xtr["Color"].astype(str) + "_" + X["Material"].astype(str)
        Xtr["brand_material"] = Xtr["Material"].astype(str) + "_" + X["Brand"].astype(str)
        Xtr["brand_size"] = Xtr["Size"].astype(str) + "_" + X["Brand"].astype(str)
        Xtr["material_size"] = Xtr["Size"].astype(str) + "_" + X["Material"].astype(str)
        Xtr["brand_style"] = Xtr["Style"].astype(str) + "_" + X["Brand"].astype(str)
        Xtr["color_style"] = Xtr["Style"].astype(str) + "_" + X["Color"].astype(str)
        #Xtr["style_size"] = Xtr["Size"].astype(str) + "_" + X["Style"].astype(str)
        # Xtr["color_brand"] = Xtr["color_brand"].astype("category")
        # Xtr["color_material"] = Xtr["color_material"].astype("category")
        # Xtr["brand_material"] = Xtr["brand_material"].astype("category")

        return Xtr


CATS = ['Brand',
 'Material',
 # 'Size',
 # 'Laptop Compartment',
 # 'Waterproof',
 'Style',
 'Color'] + ["color_brand", "color_material", "brand_material",
            "brand_size", "material_size", "brand_style", "color_style"]

to_encode = ['Brand',
 'Material',
 'Style',
 'Color']+ ["color_brand", "color_material", "brand_material",
            "brand_size", "material_size", "brand_style", "color_style"]

processing = Pipeline([#("nans", full_pipe),
                ("ordinal", OrdinalFeatures()),
                ("interactions", Interactions()),
                ("prep_te", PrepEncoder(to_encode=to_encode, medians=False)),
                ("te_mean", tml.TargetEncoder(to_encode=[f"{c}_mean" for c in to_encode],
                                              agg_func="mean")),
                # ("te_median", tml.TargetEncoder(to_encode=[f"{c}_median" for c in to_encode],
                #                               agg_func="median")),
                ("te_std", tml.TargetEncoder(to_encode=[f"{c}_std" for c in to_encode],
                                              agg_func="std")),
                ("categorizer", Categorizer(cats=CATS))       
               ])

model_pipe = Pipeline([("processing", processing),
                       # ("poly", tml.DfPolynomial(to_interact=["brand_material_mean",
                       #                                        "color_brand_mean",
                       #                                        "color_material_mean",
                       #                                        "brand_size_mean",
                       #                                        "material_size_mean"],
                                                # interaction_only=True)),
                       ('model', lgb.LGBMRegressor(n_estimators=1000,
                                                   random_state=354,
                                                   verbose=-1, n_jobs=3))])


callbacks = [lgb.early_stopping(100, verbose=0)]
fit_params = {"callbacks":callbacks}

oof, res = tml.cv_score(data=train[training_cols],
                        target=target, estimator=model_pipe,
                        cv=kfolds, imp_coef=True,
                        early_stopping=True, fit_params=fit_params)

print(round(np.sqrt(mean_squared_error(y_true=target, y_pred=oof)), 3))
print(res["iterations"])

tml.plot_feat_imp(res['feat_imp'], n=10)

tml.plot_regression_predictions(data=train[training_cols],
                                true_label=target, pred_label=oof,
                                feature=["Weight Capacity (kg)"])


# import optuna
# from optuna.samplers import TPESampler

# def objective(trial, data=train[training_cols], target=target):
    
#     param = {
#         'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 100.0),
#         'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 100.0),
#         'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
#         'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
    #     'num_leaves': trial.suggest_int('num_leaves', 2, 300),
    #     'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
    #     'cat_l2': trial.suggest_float('cat_l2', 1e-3, 10)
    # }
    
    # model = Pipeline([("categorizer", Categorizer(cats=CATS)),
    #                   ('model', lgb.LGBMRegressor(n_estimators=5000, 
    #                                                subsample=param['subsample'],
    #                                                reg_lambda=param['reg_lambda'], 
    #                                                reg_alpha=param['reg_alpha'],
    #                                                colsample_bytree=param['colsample_bytree'],
    #                                                num_leaves=param['num_leaves'], 
    #                                                min_child_weight=param['min_child_weight'],
    #                                               cat_l2=param["cat_l2"],
#                                                    random_state=5, n_jobs=-1))])  
    
#     callbacks = [lgb.early_stopping(100, verbose=0)]
#     fit_params = {"callbacks":callbacks}

#     oof, res = tml.cv_score(data=data, target=target, estimator=model, cv=kfolds, imp_coef=False,
#                                    early_stopping=True, fit_params=fit_params)
    
#     rmse = np.sqrt(mean_squared_error(y_true=target, y_pred=oof))
    
#     return rmse

# sampler = TPESampler(seed=645)  # Make the sampler behave in a deterministic way.

# study = optuna.create_study(direction='minimize', sampler=sampler)
# study.optimize(objective, n_trials=100, n_jobs=3)
# print('Number of finished trials:', len(study.trials))
# print('Best trial:', study.best_trial.params)


CATS = ['Brand',
 'Material',
 'Style',
 'Color'] + ["color_brand", "color_material", "brand_material",
            "brand_size", "material_size", "brand_style", "color_style"]

to_encode = ['Brand',
 'Material',
 'Style',
 'Color']+ ["color_brand", "color_material", "brand_material",
            "brand_size", "material_size", "brand_style", "color_style"]

processing = Pipeline([
                ("ordinal", OrdinalFeatures()),
                ("interactions", Interactions()),
                ("prep_te", PrepEncoder(to_encode=to_encode, medians=False)),
                ("te_mean", tml.TargetEncoder(to_encode=[f"{c}_mean" for c in to_encode],
                                              agg_func="mean")),
                ("te_std", tml.TargetEncoder(to_encode=[f"{c}_std" for c in to_encode],
                                              agg_func="std")),
                ("categorizer", Categorizer(cats=CATS))       
               ])

model_pipe = Pipeline([("processing", processing),
                       ('model', lgb.LGBMRegressor(n_estimators=100,
                                                   reg_alpha=35,
                                                   reg_lambda=7,
                                                   subsample=0.6,
                                                   colsample__bytree=0.7,
                                                   min_child_weight=27,
                                                   num_leaves=30,
                                                   random_state=354,
                                                   verbose=-1, n_jobs=3))])
model_pipe.fit(train[training_cols], target)
oof = model_pipe.predict(test[training_cols])

print(round(np.sqrt(mean_squared_error(y_true=test["Price"], y_pred=oof)), 3))

tml.plot_regression_predictions(data=test[training_cols], true_label=test["Price"], pred_label=oof, feature=["Weight Capacity (kg)"], hue="Brand")


CATS = ['Brand',
 'Material',
 'Style',
 'Color'] + ["color_brand", "color_material", "brand_material",
            "brand_size", "material_size", "brand_style", "color_style"]

to_encode = ['Brand',
 'Material',
 'Style',
 'Color']+ ["color_brand", "color_material", "brand_material",
            "brand_size", "material_size", "brand_style", "color_style"]

processing = Pipeline([
                ("ordinal", OrdinalFeatures()),
                ("interactions", Interactions()),
                ("prep_te", PrepEncoder(to_encode=to_encode, medians=False)),
                ("te_mean", tml.TargetEncoder(to_encode=[f"{c}_mean" for c in to_encode],
                                              agg_func="mean")),
                ("te_std", tml.TargetEncoder(to_encode=[f"{c}_std" for c in to_encode],
                                              agg_func="std")),
                ("categorizer", Categorizer(cats=CATS))       
               ])

model_pipe = Pipeline([("processing", processing),
                       ('model', lgb.LGBMRegressor(n_estimators=100,
                                                   reg_alpha=35,
                                                   reg_lambda=7,
                                                   subsample=0.6,
                                                   colsample__bytree=0.7,
                                                   min_child_weight=27,
                                                   num_leaves=30,
                                                   random_state=354,
                                                   verbose=-1, n_jobs=3))])

model_pipe.fit(df_train[training_cols], df_train["Price"])
predictions = model_pipe.predict(df_test[training_cols])


sub = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
sub["Price"] = predictions

sub.to_csv("submission.csv", index=False)
sub.head()




