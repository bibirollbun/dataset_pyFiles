import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb
warnings.filterwarnings('ignore')


# Histogram matching for making the same distributions for original data
def to_distribution(mask, sample):
    sorted_mask = sorted(mask.values)

    mask_quantiles = np.linspace(0, 1, len(mask))
    sample_quantiles = np.argsort(np.argsort(sample)) / (len(sample) - 1)
    
    result = np.interp(sample_quantiles, mask_quantiles, sorted_mask)
    return result


# Making OOF-preds for all GBDTs
def trees_training(models, cv, X_train, y_train, X_test, y_test):
    results = {mod: {"oof_preds": np.zeros(len(X_train)),
                    "y_preds": np.zeros(len(X_test)),
                    "oof_scores": np.zeros(cv),
                    "y_scores": np.zeros(cv)
                    } 
               for mod in models}
    kf = KFold(n_splits = cv, shuffle = True)
    
    for fold_num, (idx_train, idx_test) in enumerate(kf.split(X_train, y_train)):
        X_fold_train, y_fold_train = X_train.iloc[idx_train], y_train.iloc[idx_train]
        X_fold_test, y_fold_test = X_train.iloc[idx_test], y_train.iloc[idx_test]

        X_fold_train, X_fold_val, y_fold_train, y_fold_val = train_test_split(X_fold_train, y_fold_train, test_size = 0.1)

        X_test_copy = X_test.copy()

        for name, model in models.items():
            if name == 'xgb':
                model.fit(X_fold_train, y_fold_train,
                          eval_set = [(X_fold_val, y_fold_val)],
                          early_stopping_rounds = 150,
                          verbose = False)
            elif name == 'lgb':
                model.fit(X_fold_train, y_fold_train, 
                          eval_set = [(X_fold_val, y_fold_val)],
                          callbacks=[lgb.early_stopping(stopping_rounds=150),
                                    lgb.log_evaluation(period=0)]
                         )
                         
            elif name == 'ctb':
                model.fit(X_fold_train, y_fold_train,
                          eval_set = [(X_fold_val, y_fold_val)]
                         )

            oof_preds = model.predict(X_fold_test)
            y_preds = model.predict(X_test_copy)
            oof_score = np.sqrt(mean_squared_error(y_fold_test, oof_preds))
            y_score = np.sqrt(mean_squared_error(y_test, y_preds))

            results[name]['oof_preds'][idx_test] = oof_preds
            results[name]['y_preds'] += y_preds / cv
            results[name]['oof_scores'][fold_num] = oof_score
            results[name]['y_scores'][fold_num] = y_score

    return results


df_train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df_orig = pd.read_csv('/kaggle/input/calories-burnt-prediction/calories.csv')

df_orig.columns = df_train.columns

df_train['Sex'] = df_train['Sex'].apply(lambda sex: 0 if sex == 'female' else 1)
df_orig['Sex'] = df_orig['Sex'].apply(lambda sex: 0 if sex == 'female' else 1)

df_train['Intensity'] = df_train['Heart_Rate'] / df_train['Duration']
df_orig['Intensity'] = df_train['Heart_Rate'] / df_train['Duration']

df_train.drop(columns = ['id'], inplace = True)
df_orig.drop(columns = ['id'], inplace = True)

for col in df_orig.columns:
    df_orig[col] = to_distribution(df_train[col], df_orig[col])


X = df_train.drop(columns = ['Calories'])
y = np.log1p(df_train['Calories'])
df_orig['Calories'] = np.log1p(df_orig['Calories'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


X_train = pd.concat([X_train, df_orig.drop(columns = ['Calories'])]).reset_index(drop = True)
y_train = pd.concat([y_train, df_orig['Calories']]).reset_index(drop = True)
X_test = X_test.reset_index(drop = True)
y_test = y_test.reset_index(drop = True)


xgbmodel = xgb.XGBRegressor(
    n_estimators = 7500,
    learning_rate = 0.01,
    max_depth = 6,
    subsample = 0.8,
    colsample_bytree = 0.8,
    alpha = 0,
    verbosity = 0,
    reg_lambda = 0.1,
    device = 'gpu'
)

lgbmodel = lgb.LGBMRegressor(n_estimators = 7500,
                               learning_rate = 0.01,
                               device='gpu',
                               verbosity = -1,
                               num_leaves = 31,
                               max_depth = -1,
                               min_child_samples = 20,
                               subsample = 0.8,
                               colsample_bytree = 0.8,
                               reg_alpha = 0.1,
                               reg_lambda = 0.1,
                               verbose = -1
                          )

catboostmodel = ctb.CatBoostRegressor(
    iterations = 9000,
    learning_rate = 0.01,
    early_stopping_rounds = 150,
    task_type = 'GPU',
    depth = 6,
    l2_leaf_reg = 3,
    bootstrap_type = 'Bayesian',
    bagging_temperature = 1.0,
    train_dir = None,
    logging_level='Silent'
)


models = {
          "xgb": xgbmodel,
          "lgb": lgbmodel,
          "ctb": catboostmodel
         }


results = trees_training(models, 5, X_train, y_train, X_test, y_test)


X_final_train, X_val, y_final_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = 42)


xgbmodel.fit(X_final_train, y_final_train,
              eval_set = [(X_val, y_val)],
              early_stopping_rounds = 150,
              verbose = False)

lgbmodel.fit(X_final_train, y_final_train, 
                          eval_set = [(X_val, y_val)],
                          callbacks=[lgb.early_stopping(stopping_rounds=150),
                                    lgb.log_evaluation(period=0)],
                         )

catboostmodel.fit(X_final_train, y_final_train,
                          eval_set = [(X_val, y_val)]
                 )


X_train_stack = pd.DataFrame([results[model]['oof_preds'] for model in models]).transpose()
X_test_stack = pd.DataFrame([results[model]['y_preds'] for model in models]).transpose()


ridgemodel = RidgeCV(alphas = np.linspace(0.01, 50, 200), scoring = 'neg_root_mean_squared_error', cv = 5)
ridgemodel.fit(X_train_stack, y_train)
preds = ridgemodel.predict(X_test_stack)


ridgemodel.alpha_


ridgemodel.coef_


np.sqrt(mean_squared_error(y_test, preds))


df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


df_test['Sex'] = df_test['Sex'].apply(lambda sex: 0 if sex == 'female' else 1)
df_test['Intensity'] = df_test['Heart_Rate'] / df_test['Duration']


df_test.drop(columns = ['id'], inplace = True)


xgb_preds = xgbmodel.predict(df_test)
lgb_preds = lgbmodel.predict(df_test)
ctb_preds = catboostmodel.predict(df_test)


X_preds = pd.DataFrame([xgb_preds, lgb_preds, ctb_preds]).transpose()


pred = ridgemodel.predict(X_preds)


submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


submission['Calories'] = np.expm1(pred)
submission.to_csv('submission.csv', index = False)


submission

