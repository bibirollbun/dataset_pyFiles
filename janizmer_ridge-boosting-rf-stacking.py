#basics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import copy
warnings.filterwarnings("ignore")
import time
import json
random_state=42

#preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#feature engineering
from sklearn.feature_selection import mutual_info_regression

#transformers and pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

#algorithms
from lightgbm import LGBMRegressor, early_stopping
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
import optuna

#model evaluation
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.base import clone

#stacking
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer

#feature selection
from sklearn.feature_selection import RFECV


train = pd.read_csv('/kaggle/input/raw-data/train.csv', index_col=[0])
test = pd.read_csv('/kaggle/input/raw-data/test.csv', index_col=[0])
train.head(5)


#reserved for pipieline
pipe_train = train.copy()
pipe_test = test.copy()

#for data analysis
train_df = train.copy()
test_df = test.copy()
train_df.head(5)


train_df.describe().T


numerical_features = [feature for feature in train_df.columns if feature!='BeatsPerMinute']
target = "BeatsPerMinute"


sampled_train = train_df.sample(frac = 0.01)


fig, ax = plt.subplots(3, 3, figsize=(40, 40))
for var, subplot in zip(numerical_features, ax.flatten()):
    sns.scatterplot(x=var, y='BeatsPerMinute', data = sampled_train, ax=subplot, hue = 'BeatsPerMinute')


fig, ax = plt.subplots(3,3, figsize=(40,40))
for var, subplot in zip(numerical_features, ax.flatten()):
    sns.histplot(x=var, data=sampled_train, ax=subplot)


def extended_describe(df):
    desc = df.describe().T
    desc["median"] = df.median()
    desc["~zeros_count"] = ((df < 0.05) & (df > 0)).sum().values
    desc["~zeros_percent"] = (((df < 0.05) & (df > 0)).sum() / len(df) * 100).values
    desc["mode"] = [df[col].mode().iloc[0] if not df[col].mode().empty else np.nan for col in df.columns]
    desc.drop(columns=["25%", "50%", "75%"], inplace=True)
    return desc

extended_describe(train_df)


# Display correlations between numerical features and target on heatmap.
sns.set(font_scale=1.1)
corr_train = sampled_train.corr(method="spearman")
mask = np.triu(corr_train.corr(method="spearman"))
plt.figure(figsize=(20,20))
sns.heatmap(corr_train, annot=True, fmt='.2f', cmap='viridis', square=True, mask=mask, linewidths=1)


# Mutual Information score
y_sampled = sampled_train.BeatsPerMinute
features_sampled = sampled_train[numerical_features]

mutual_info = mutual_info_regression(features_sampled, y_sampled, random_state=random_state)
mutual_info_s = pd.Series(mutual_info)
mutual_info_s.index = features_sampled.columns
mutual_info_s = pd.DataFrame(mutual_info_s.sort_values(ascending=False), columns = ['Num Feature'])
mutual_info_s.style.background_gradient("cool")


plt.figure(figsize=(16,10))
sns.histplot(x='BeatsPerMinute', data=sampled_train, bins=20, color='Skyblue', kde=True)
plt.title('BeatsPerMinute distribution')
plt.ylabel('Frequency')
plt.xlabel('BeatsPerMinute')
plt.show()


#Custom transformer

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, add_attributes=True):
        self.add_attributes = add_attributes
        self.new_feature_names = []
    def fit(self, X, y=None):
        #Placeholder
        return self
    
    def transform(self, X):
        # Placeholder
        return X
    
    def get_feature_names_out(self, input_features=None):
        # Output feature names
        if self.add_attributes:
            return self.new_feature_names
        else:
            return list(input_features) if input_features is not None else []
FeatureEngineer = FeatureEngineer()


numerical_transformer = StandardScaler()


cross_valid = KFold(n_splits=10, random_state=random_state, shuffle=True)


def cv_score(model,
    data,
    cross_valid,
    label='BeatsPerMinute',
    early_stopping_rounds=50):
    
    if not hasattr(data, "copy") or not hasattr(data, "columns"):
        raise TypeError("data must be a pandas DataFrame")
    if label not in data.columns:
        raise KeyError(f"target column '{label}' not found")
        
    X = data.drop(columns=[label]).copy()
    y = data[label]
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.select_dtypes(include=[np.number]).copy()
    X = X.fillna(X.median(numeric_only=True))
    
    #Preparing variables for predictions
    val_predictions = np.zeros(len(X))
    val_true = np.zeros(len(y))
    val_scores, train_scores = [], []
    for fold, (train_idx, val_idx) in enumerate(cross_valid.split(X, y)):
        X_train, y_train = X.iloc[train_idx].reset_index(drop=True), y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx].reset_index(drop=True), y.iloc[val_idx]
        m = clone(model)
        
        if hasattr(m, "n_jobs"):
            m.set_params(n_jobs=1)
            
        # Early stopping logic
        fit_params = {}
        if early_stopping_rounds is not None:
            # XGBoost
            if isinstance(m, XGBRegressor):
                pass
            #         fit_params['early_stopping_rounds'] = early_stopping_rounds
            #         fit_params['eval_set'] = [(X_val, y_val)]
            #         fit_params['verbose'] = False
                # except TypeError:
                #     from xgboost.callback import EarlyStopping
                #     fit_params['callbacks'] = [EarlyStopping(rounds=early_stopping_rounds, save_best=True)]
                #     fit_params['eval_set'] = [(X_val, y_val)]
        
            # CatBoost
            elif isinstance(m, CatBoostRegressor):
                fit_params['early_stopping_rounds'] = early_stopping_rounds
                fit_params['eval_set'] = [(X_val, y_val)]
                fit_params['verbose'] = False
        
            # LightGBM
            elif isinstance(m, LGBMRegressor):
                pass
                # from lightgbm import early_stopping
                # fit_params['callbacks'] = [early_stopping(stopping_rounds=early_stopping_rounds)]
                # fit_params['eval_set'] = [(X_val, y_val)]
        if fit_params:    
            m.fit(X_train, y_train, **fit_params)
        else:
            m.fit(X_train, y_train)
            
        train_preds = m.predict(X_train)
        val_preds = m.predict(X_val)

        val_predictions[val_idx] = val_preds
        val_true[val_idx] = y_val.values

        # Metics(RMSE)
        train_score = np.sqrt(mean_squared_error(y_train, train_preds))
        val_score   = np.sqrt(mean_squared_error(y_val, val_preds))

        train_scores.append(train_score)
        val_scores.append(val_score)
        
        print(f"fold {fold + 1} rmse={val_score:.5f}", flush=True)
        
    # Result  is mean of RMSE on validation and training folds
    print(f"Val RMSE: {np.mean(val_scores):.5f} ± {np.std(val_scores):.5f} | "
          f"Train RMSE: {np.mean(train_scores):.5f} ± {np.std(train_scores):.5f}")

    return val_scores, val_predictions, val_true


val_scores_df, val_predictions_df, val_true_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


ridge_optuna_params = {'max_iter': 969,
                       'alpha': 996.2276353844511,
                       'tol': 4.41828902390748e-05}
ridge_tuned = Ridge(**ridge_optuna_params, random_state=random_state)
ridge_pipe = Pipeline(steps=[
    #('FeatureEngineer', FeatureEngineer),
    ('numerical_transformer', numerical_transformer),
    ('ridge', ridge_tuned)
])

ridge_pipe


%%time
val_scores_df['RIDGE'], val_predictions_df['RIDGE'], val_true_df['RIDGE'] = cv_score(ridge_pipe, data = pipe_train, cross_valid=cross_valid)
pipe_train.head()


val_predictions_df['RIDGE'].min()


# Find best params with Optuna
# def objective(trial, data, cv):
#     params = {
#         "max_depth": trial.suggest_int("max_depth", 3, 10),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
#         "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
#         "alpha": trial.suggest_float("alpha", 0.0, 1.0),
#         "gamma": trial.suggest_float("gamma", 0.0, 5.0),
#     }

#     model = XGBRegressor(random_state=42, **params)
#     print(type(model))
#     scores = cv_score(model, data, cross_valid=cv)
#     val_score = scores[0]
#     return np.mean(val_score)

# study = optuna.create_study(direction="minimize")
# def save_best_params(study, trial):
#     with open("xgboost_bestparams.json", "w") as f:
#         json.dump(study.best_params, f, indent=4)
# study.optimize(lambda trial: objective(trial, data = pipe_train, cv = cross_valid), n_trials=20, callbacks=[save_best_params])

# print("Best RMSE:", study.best_trial.value)
# print("Best params:", study.best_trial.params)

# XGB regressor with optuna best params
with open("/kaggle/input/xgboost-bestparams/xgboost_bestparams.json", "r") as f:
    xgb_optuna_params = json.load(f)

xgb_tuned = XGBRegressor(
    **xgb_optuna_params,
    random_state=random_state
 )

xgb_pipe = Pipeline(steps=[
    ('FeatureEngineer', FeatureEngineer),
    ('xgb_tuned', xgb_tuned)
])

xgb_pipe


val_scores_df['XGBOOST'], val_predictions_df['XGBOOST'], val_true_df['XGBOOST'] = cv_score(xgb_pipe, data = pipe_train,cross_valid=cross_valid)


lgbm_optuna_params = {
    'n_estimators': 934,
    'max_depth': 16,
    'learning_rate': 0.003717017250694586,
    'min_data_in_leaf': 56,
    'subsample': 0.939684807121632,
    'max_bin': 420,
    'feature_fraction': 0.8416988483664828
}

lgbm_tuned = LGBMRegressor(
    **lgbm_optuna_params,
    random_state=random_state,
    verbose=-1
)

lgbm_pipe = Pipeline(steps=[
    ('FeatureEngineer', FeatureEngineer),
    ('lgbm_tuned', lgbm_tuned)
])

lgbm_pipe


val_scores_df['GBM'], val_predictions_df['GBM'], val_true_df['GBM'] = cv_score(lgbm_pipe, data = pipe_train, cross_valid=cross_valid)


catb_optuna_params = {
    'iterations': 100,
    'colsample_bylevel': 0.884050283064001,
    'learning_rate': 0.03501796397486782,
    'random_strength': 0.01644595544178421,
    'border_count': 300,
    'depth': 6,
    'l2_leaf_reg': 7,
    'grow_policy': 'SymmetricTree',
    'boosting_type': 'Plain',
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.8615585051787144
}

catb_tuned = CatBoostRegressor(
    **catb_optuna_params,
    random_state=random_state,
    logging_level='Silent'
)

catb_pipe = Pipeline(steps=[
    ('FeatureEngineer', FeatureEngineer),
    ('catb_tuned', catb_tuned)
])

catb_pipe


val_scores_df['CB'], val_predictions_df['CB'], val_true_df['CB'] = cv_score(catb_pipe, data = pipe_train, cross_valid=cross_valid)


transposed_df = val_scores_df.transpose()
transposed_df.columns = ['fold-1','fold2','fold3','fold4','fold5', 'fold-6','fold7','fold8','fold9','fold10' ]
transposed_df['Mean'] = transposed_df.mean(axis=1)
transposed_df['Std'] = transposed_df.std(axis=1)
transposed_df.sort_values(by = 'Mean', ascending=True).style.background_gradient('Dark2_r')


sns.set(font_scale=1.1)
correlation_train = val_predictions_df.corr()
mask = np.triu(np.ones_like(correlation_train, dtype=bool))
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_train,
            annot=True,
            fmt='.3f',
            cmap='coolwarm',
            vmin=-1, vmax=1, center=0,
            square=True,
            mask=mask,
            linewidths=1,
            cbar=True)
plt.show()


from sklearn.pipeline import make_pipeline
from optuna.samplers import TPESampler

# Ridge regressor as meta regressor. It uses oof_predictions_df.
def objective(trial):
    max_iter = trial.suggest_int("max_iter", 100, 4000)
    alpha = trial.suggest_float("alpha", 1e-4, 1000, log=True)
    tol = trial.suggest_float("tol", 1e-6, 1e-3, log=True)

    ensemble_regressor = Ridge(max_iter=max_iter, alpha=alpha, tol=tol)
    ensemble_pipeline = make_pipeline(StandardScaler(), ensemble_regressor)

    ss = cross_valid
    
    score = cross_val_score(ensemble_pipeline, val_predictions_df, pipe_train.BeatsPerMinute, scoring= "neg_root_mean_squared_error",  cv=ss)
    return score.mean()

sampler = TPESampler(seed=random_state)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=100)



ensemble_params = {
    'max_iter': 969,
    'alpha': 996.2276353844511,
    'tol': 4.41828902390748e-05
}

ridge_ensemble = Ridge(
    **ensemble_params,
    random_state=random_state
)

ensemble_pipe = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('ridge_ensemble', ridge_ensemble)
])


min_features_to_select = 1

pipeline = Pipeline([
    ('rfecv', RFECV(
        estimator=ensemble_pipe,                    
        step=1,
        cv = cross_valid,
        scoring="neg_root_mean_squared_error",
        min_features_to_select=min_features_to_select,
        n_jobs=1,
        importance_getter=lambda est: est.named_steps['ridge_ensemble'].coef_
    ))
])

# Fit
pipeline.fit(val_predictions_df, pipe_train.BeatsPerMinute)

# Results
rfecv_step = pipeline.named_steps['rfecv']
mean_scores = rfecv_step.cv_results_["mean_test_score"]  # negative RMSE
best_idx = np.argmax(mean_scores)
best_rmse = -mean_scores[best_idx]

support = rfecv_step.support_
selected_models = np.array(val_predictions_df.columns)[support]

print("Best CV RMSE:", best_rmse)
print('Number of evaluated models:', val_predictions_df.shape[1])
print('Number of selected models:', support.sum())
print('Selected Models for Ensemble:', selected_models)


val_predictions_df.head()


X_train = pipe_train.drop(columns=['BeatsPerMinute'])
X_train = X_train.replace([np.inf, -np.inf], np.nan).select_dtypes(include=[np.number])
X_train = X_train.fillna(X_train.median(numeric_only=True))
y_train = pipe_train['BeatsPerMinute'].astype(float).values

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
catb_tuned.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100, verbose=False)
lgbm_tuned.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[early_stopping(100)])
ridge_tuned.fit(X_train, y_train)

X_test = pipe_test.drop(columns=['BeatsPerMinute'], errors='ignore')
X_test = X_test.replace([np.inf, -np.inf], np.nan).select_dtypes(include=[np.number])
X_test = X_test.reindex(columns=X_train.columns)
X_test = X_test.fillna(X_train.median(numeric_only=True))

cols = ['RIDGE','CB','GBM']
X_meta = val_predictions_df[cols].astype(float)
y_meta = pipe_train.loc[val_predictions_df.index, 'BeatsPerMinute'].astype(float)

meta = make_pipeline(SimpleImputer(strategy="median"),
                     StandardScaler(),
                     Ridge(**study.best_params))
meta.fit(X_meta, y_meta)

pred_cb_test = catb_tuned.predict(X_test)
pred_gbm_test = lgbm_tuned.predict(X_test)
pred_ridge_test = ridge_tuned.predict(X_test)

P_test = pd.DataFrame({'CB': pred_cb_test, 'GBM': pred_gbm_test, 'RIDGE': pred_ridge_test}, index=X_test.index).astype(float)
pred_final = meta.predict(P_test[cols])
out = pd.DataFrame({'id': X_test.index, 'prediction': pred_final})
out.to_csv('/kaggle/working/attempt.csv', index=False)

