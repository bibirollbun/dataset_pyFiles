import os
import shutil
import scipy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer

import warnings 
warnings.filterwarnings('ignore')

import lightgbm
import xgboost
import catboost


# LOAD DATA

train_df = pd.read_csv(r'/kaggle/input/playground-series-s5e10/train.csv').drop(columns = 'id')
test_df  = pd.read_csv(r'/kaggle/input/playground-series-s5e10/test.csv').drop(columns = 'id')

train_df.shape, test_df.shape


# LOAD EXTERNAL DATA (FROM OTHER NOTEBOOKS)

tabm_residual_oof  = pd.read_csv(r'/kaggle/input/s5e10-tabm-over-residuals/oof_tabm_overresid.csv')
tabm_residual_test = pd.read_csv(r'/kaggle/input/s5e10-tabm-over-residuals/test_tabm_overresid.csv')

realmlp_oof = pd.read_csv('/kaggle/input/pg-s5e10-realmlp-cv-0-055936-lb-0-05549/realmlp_oof.csv')
realmlp_test = pd.read_csv('/kaggle/input/pg-s5e10-realmlp-cv-0-055936-lb-0-05549/realmlp.csv')

xgb_residual_oof = pd.DataFrame(np.load(r'/kaggle/input/xgb-boosting-over-residuals-cv-0-05595/oof.npy'), columns = ['xgb_residual'])
xgb_residual_test = pd.read_csv(r'/kaggle/input/xgb-boosting-over-residuals-cv-0-05595/submission.csv')

single_tabm_oof = pd.read_csv(r'/kaggle/input/s5e10-single-tabm-tuned/oof_tabm_plus_origcol_tuned.csv')
single_tabm_test = pd.read_csv(r'/kaggle/input/s5e10-single-tabm-tuned/test_tabm_plus_origcol_tuned.csv')

xgb_diff_seed_oof = pd.read_csv(r'/kaggle/input/s5e10-xgb-origcol-20seeds/oof_xgb_plus_origcol.csv')
xgb_diff_seed_test = pd.read_csv(r'/kaggle/input/s5e10-single-tabm-tuned/test_tabm_plus_origcol_tuned.csv')

xgb_rich_oof  = pd.read_csv(r'/kaggle/input/feature-rich-single-xgb-cv-0-05594/oof_xgb_enhanced_meta_20251011_184833_cv0.055942.csv')
xgb_rich_test = pd.read_csv(r'/kaggle/input/feature-rich-single-xgb-cv-0-05594/test_xgb_enhanced_meta_20251011_184833_cv0.055942.csv')

nn_stacking_oof = pd.read_csv(r'/kaggle/input/s5e10-nn-stacking-baseline/oof_nn_ensemble.csv')
nn_stacking_test = pd.read_csv(r'/kaggle/input/s5e10-nn-stacking-baseline/test_nn_ensemble.csv')

single_ydf_oof = pd.read_csv(r'/kaggle/input/road-risk-single-ydf/YDF_oof.csv')
single_ydf_test = pd.read_csv(r'/kaggle/input/road-risk-single-ydf/YDF_test.csv')

ridge_stacking_oof = pd.read_csv(r'/kaggle/input/accident-prediction-stacking-model/meta_model/oof/oof_ridge_L2_V12.csv')
ridge_stacking_test = pd.read_csv(r'/kaggle/input/accident-prediction-stacking-model/meta_model/test/ridge_L2_V12.csv')

# MERGE ALL EXTERNAL DATA
external_oof = pd.concat((tabm_residual_oof, realmlp_oof, xgb_residual_oof, single_tabm_oof, xgb_diff_seed_oof, xgb_rich_oof, 
                          nn_stacking_oof, single_ydf_oof, ridge_stacking_oof), axis = 1)

external_oof = external_oof.drop(columns = 'id')     # --> DROP ALL id FEATURES
external_oof.columns = ['tabm_residual', 'realmlp', 'xgb_residual', 'single_tabm', 'xgb_diff_seed20', 'xgb_rich',
                        'nn_stacking', 'single_ydf', 'ridge_stacking']  # RENAME COLUMNS


external_test = pd.concat((tabm_residual_test, realmlp_test, xgb_residual_test, single_tabm_test, xgb_diff_seed_test, xgb_rich_test, 
                           nn_stacking_test, single_ydf_test, ridge_stacking_test), axis = 1)

external_test = external_test.drop(columns = 'id')     # --> DROP ALL id FEATURES
external_test.columns = ['tabm_residual', 'realmlp', 'xgb_residual', 'single_tabm', 'xgb_diff_seed20', 'xgb_rich', 
                         'nn_stacking', 'single_ydf', 'ridge_stacking']  # RENAME COLUMNS


external_oof.shape, external_test.shape


# MERGING TRAIN DATA WITH EXTERNAL DATA

train_df = pd.concat((train_df, external_oof), axis = 1)
test_df = pd.concat((test_df, external_test), axis = 1)

display(train_df)
display(test_df)


# CHECK ACCIDENT RISK DISTRIBUTION

sns.histplot(train_df['accident_risk'], kde = True, bins = 50, color = 'orange')


# FEATURE ENGINEERING
def f(X):
    return \
    0.3 * X["curvature"] + \
    0.2 * (X["lighting"] == "night").astype(int) + \
    0.1 * (X["weather"] != "clear").astype(int) + \
    0.2 * (X["speed_limit"] >= 60).astype(int) + \
    0.1 * (X["num_reported_accidents"] > 2).astype(int)

def clip(f):
    def clip_f(X):
        sigma = 0.05
        mu = f(X)
        a, b = -mu/sigma, (1-mu)/sigma
        Phi_a, Phi_b = scipy.stats.norm.cdf(a), scipy.stats.norm.cdf(b)
        phi_a, phi_b = scipy.stats.norm.pdf(a), scipy.stats.norm.pdf(b)
        return mu*(Phi_b-Phi_a)+sigma*(phi_a-phi_b)+1-Phi_b
    return clip_f

train = clip(f)(train_df)
test = clip(f)(test_df)

train_df['score'] = train
test_df['score']  = test

train_df


# LABEL ENCODING

# GET CAT COLS
categorical_cols = train_df.select_dtypes(include = 'object').columns

encoders = {}

# DO ENCODE FOR EACH CATEGORICAL COLS
for col in categorical_cols:
    train_df[col], uniques = train_df[col].factorize()
    encoders[col] = dict(zip(uniques, range(len(uniques))))

    test_df[col] = test_df[col].map(encoders[col])

train_df.head(4)


# SPLIT

# PREDICT RESIDUAL
#train_df['residual'] = train_df['accident_risk'] - train_df['score']

x = train_df.drop(labels = ['accident_risk'], axis = 1)
y = train_df['accident_risk']

x_test = test_df.copy()

x.shape, y.shape, x_test.shape


# CHECK RESIDUAL DISTRIBUTION

#sns.kdeplot(train_df['residual'], fill = False)


# LGBM

train_scores, val_scores, test_lgbm = [], [], []
oof_lgbm = np.zeros(len(x))

skfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 2025)

# SKFOLD
for i, (train_index, val_index), in enumerate(skfold.split(x, pd.qcut(y, q = 10).cat.codes)):

    # SPLIT DATA
    x_train, x_val = x.iloc[train_index], x.iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # DEFINE LGBM
    lgbm = lightgbm.LGBMRegressor(boosting_type = 'gbdt', 
                                  n_estimators = 20_000, 
                                  learning_rate = 0.01, 
                                  num_leaves = 244, 
                                  random_state = 2025, 
                                  verbose = -1,
                                  device = 'cpu',
                                  n_jobs = -1)
    
    lgbm.fit(x_train, y_train, eval_set = (x_val, y_val),
             eval_metric = 'rmse',
             callbacks = [lightgbm.early_stopping(stopping_rounds = 300)])

    # PREDICT
    y_train_predict = lgbm.predict(x_train)
    y_val_predict   = lgbm.predict(x_val)
    y_test_predict = lgbm.predict(x_test)

    # PREDICT RMSE
    train_rmse = mean_squared_error(y_train, y_train_predict, squared = False)
    val_rmse   = mean_squared_error(y_val, y_val_predict, squared = False)

    print(f'Fold {i+1} 🚀 : 1️⃣ Train RMSE = {train_rmse}, 2️⃣ Val RMSE = {val_rmse}')

    # PLOT IMPORTANCE
    lightgbm.plot_importance(booster = lgbm, importance_type = 'gain', title = f'Feature Importance Fold {i+1}')
    plt.show()

    # PUSH SCORE
    train_scores.append(train_rmse)
    val_scores.append(val_rmse)

    # STORE OOF VAL AND TEST PREDICT
    oof_lgbm[val_index] = y_val_predict
    test_lgbm.append(y_test_predict)

    
print(f'\n🎉 Train Fold Prediction 1️⃣: {np.mean(train_scores)}')
print(f'🎉 Val Fold Prediction     2️⃣: {np.mean(val_scores)}\n')

print(f'🎉 std Train Fold Prediction 3️⃣: {np.std(train_scores, ddof = 0)}')
print(f'🎉 std Val Fold Prediction   4️⃣: {np.std(val_scores, ddof = 0)}')


# SPLIT IMPORTANCES
lightgbm.plot_importance(lgbm, importance_type = 'split', figsize = (5, 5))


# XGBOOST

train_scores, val_scores, test_xgb = [], [], []
oof_xgb = np.zeros(len(x))

skfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 2025)

# SKFOLD
for i, (train_index, val_index), in enumerate(skfold.split(x, pd.qcut(y, q = 10).cat.codes)):

    # SPLIT DATA
    x_train, x_val = x.iloc[train_index], x.iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # DEFINE XGBOOST
    xgb = xgboost.XGBRegressor(n_estimators = 10_000,
                               learning_rate = 0.01,
                               max_depth = 10,
                               subsample = 1,
                               early_stopping_rounds = 300)

    xgb.fit(x_train, y_train, eval_set = [(x_val, y_val)], verbose = 500)

    # PREDICT
    y_train_predict = xgb.predict(x_train)
    y_val_predict   = xgb.predict(x_val)
    y_test_predict = xgb.predict(x_test)

    # PREDICT RMSE
    train_rmse = mean_squared_error(y_train, y_train_predict, squared = False)
    val_rmse   = mean_squared_error(y_val, y_val_predict, squared = False)

    print(f'Fold {i+1} 🚀 : 1️⃣ Train RMSE = {train_rmse}, 2️⃣ Val RMSE = {val_rmse}')

    # PLOT IMPORTANCE
    xgboost.plot_importance(booster = xgb, importance_type = 'gain')
    plt.show()

    # PUSH SCORE
    train_scores.append(train_rmse)
    val_scores.append(val_rmse)

    # STORE OOF VAL AND TEST PREDICT
    oof_xgb[val_index] = y_val_predict
    test_xgb.append(y_test_predict)

    
print(f'\n🎉 Train Fold Prediction 1️⃣: {np.mean(train_scores)}')
print(f'🎉 Val Fold Prediction     2️⃣: {np.mean(val_scores)}\n')

print(f'🎉 std Train Fold Prediction 3️⃣: {np.std(train_scores, ddof = 0)}')
print(f'🎉 std Val Fold Prediction   4️⃣: {np.std(val_scores, ddof = 0)}')


# XGBOOST SPLIT IMPORTANCES
xgboost.plot_importance(booster = xgb, importance_type = 'weight')


# CATBOOST

train_scores, val_scores, test_cat = [], [], []
oof_cat = np.zeros(len(x))

skfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 2025)

# SKFOLD
for i, (train_index, val_index), in enumerate(skfold.split(x, pd.qcut(y, q = 10).cat.codes)):

    # SPLIT DATA
    x_train, x_val = x.iloc[train_index], x.iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # DEFINE CATBOOST
    cat = catboost.CatBoostRegressor(learning_rate=0.1,
                                     depth=6,
                                     iterations = 8000,
                                     grow_policy='SymmetricTree',
                                     loss_function = 'RMSE',
                                     bootstrap_type = 'Bayesian',
                                     leaf_estimation_method = 'Newton'
                                    )

    cat.fit(x_train, y_train, eval_set = [(x_val, y_val)],
            verbose = 0,
            early_stopping_rounds = 300)

    # PREDICT
    y_train_predict = cat.predict(x_train)
    y_val_predict   = cat.predict(x_val)
    y_test_predict = cat.predict(x_test)

    # PREDICT RMSE
    train_rmse = mean_squared_error(y_train, y_train_predict, squared = False)
    val_rmse   = mean_squared_error(y_val, y_val_predict, squared = False)

    print(f'Fold {i+1} 🚀 : 1️⃣ Train RMSE = {train_rmse}, 2️⃣ Val RMSE = {val_rmse}')

    # PLOT FEATURE IMPORTANCES
    feature_importances = cat.get_feature_importance(prettified=True)
    fi_df = pd.DataFrame({"Feature": x_train.columns, "Importance": cat.get_feature_importance()}).sort_values(by="Importance", ascending=False)
    sns.barplot(fi_df, x = 'Importance', y = 'Feature')
    plt.show()
    
    # PUSH SCORE
    train_scores.append(train_rmse)
    val_scores.append(val_rmse)

    # STORE OOF VAL AND TEST PREDICT
    oof_cat[val_index] = y_val_predict
    test_cat.append(y_test_predict)

    
print(f'\n🎉 Train Fold Prediction 1️⃣: {np.mean(train_scores)}')
print(f'🎉 Val Fold Prediction     2️⃣: {np.mean(val_scores)}\n')

print(f'🎉 std Train Fold Prediction 3️⃣: {np.std(train_scores, ddof = 0)}')
print(f'🎉 std Val Fold Prediction   4️⃣: {np.std(val_scores, ddof = 0)}')


# CATBOOST FEATURE IMPORTANCES

feature_importances = cat.get_feature_importance(prettified=True)

# CONVERT TO DATAFRAME
fi_df = pd.DataFrame({"Feature": x_train.columns, "Importance": cat.get_feature_importance()}).sort_values(by="Importance", ascending=False)

# VISUALIZE FEATURE IMPORTANCES
sns.barplot(fi_df, x = 'Importance', y = 'Feature')


# CHECK TEST DISTRIBUTION 3 MODELS

plt.figure(figsize = (13, 6))


# OOF PREDICTION DISTRIBUTION COMPARISON
plt.subplot(1, 2, 1)
sns.kdeplot(oof_lgbm, label = 'LightGBM', fill = False)
sns.kdeplot(oof_xgb, label = 'XGBoost', fill = False)
sns.kdeplot(oof_cat, label = 'CatBoost', fill = False)
sns.kdeplot(train_df['accident_risk'], label = 'True Label', fill = False)
plt.title('OOF Prediction Distribution Comparison')
plt.legend()

# TEST PREDICTION DISTRIBUTION COMPARISON
plt.subplot(1, 2, 2)
sns.kdeplot(np.mean(test_lgbm, axis = 0), label = 'LightGBM', fill = False, color = 'orange')
sns.kdeplot(np.mean(test_xgb, axis = 0), label = 'XGBoost', fill = False)
sns.kdeplot(np.mean(test_cat, axis = 0), label = 'CatBoost', fill = False)
plt.title('Test Prediction Distribution Comparison')
plt.legend()

plt.show()


# CHECK TEST PRED
test_lgbm


# CREATE DATAFRAME FROM OOF AND TEST PREDICTION

# MERGE OOF VAL
oof_model = pd.concat([pd.DataFrame(oof_lgbm, columns=["lgbm"]), 
                       pd.DataFrame(oof_xgb, columns=["xgb"]), 
                       pd.DataFrame(oof_cat, columns=["cat"])], axis=1)

# MERGE TEST PRED
test_prediction = pd.concat([pd.DataFrame(np.mean(test_lgbm, axis = 0), columns = ["lgbm"]),
                             pd.DataFrame(np.mean(test_xgb, axis = 0), columns = ["xgb"]),
                             pd.DataFrame(np.mean(test_cat, axis = 0), columns = ["cat"])], axis = 1)

print('OOF data :')
display(oof_model)

print('\nTest Prediction :')
display(test_prediction)


# CREATE OUTPUT FOLDER

os.makedirs('meta_model/oof', exist_ok = True)
os.makedirs('meta_model/test', exist_ok = True)


# ENSEMBLE MODEL (MEAN AVERAGE)

oof_avg  = np.mean(oof_model, axis = 1) #  + train_df['score']   # GET OOF MEAN AND CONVERT BACK TO NORMAL
test_avg = np.mean(test_prediction, axis = 1) # + test_df['score']  # GET TEST AVG AND CONVERT BACK TO NORMAL

submission = pd.read_csv(r'/kaggle/input/playground-series-s5e10/sample_submission.csv')
submission['accident_risk'] = test_avg

display(submission)

# SAVE OOF AND TEST SUBMISSION
oof_avg.to_frame(name = 'mean_L1').to_csv(r'meta_model/oof/oof_mean_L2_V13.csv', index = False) 
submission.to_csv(r'meta_model/test/mean_L2_V13.csv', index = False)


# META-MODEL - RIDGE

# STORE SCORES, OOF, AND TEST
train_scores, val_scores, test_ridge = [], [], []
oof_ridge = np.zeros(len(x))

skfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 2025)

# SKFOLD
for i, (train_index, val_index), in enumerate(skfold.split(x, pd.qcut(y, q = 10).cat.codes)):

    # SPLIT DATA
    x_train, x_val = oof_model.iloc[train_index], oof_model.iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # DEFINE RIDGE
    ridge = Ridge(alpha = 1.0, max_iter = 10_000)
    ridge.fit(x_train, y_train)

    # PREDICT
    y_train_predict = ridge.predict(x_train)
    y_val_predict   = ridge.predict(x_val)
    y_test_predict  = ridge.predict(test_prediction)

    # PREDICT MAE
    train_rmse = mean_squared_error(y_train, y_train_predict, squared = False)
    val_rmse   = mean_squared_error(y_val, y_val_predict, squared = False)

    print(f'Fold {i+1} 🚀 : 1️⃣ Train RMSE = {train_rmse}, 2️⃣ Val RMSE = {val_rmse}')

    # PUSH SCORE
    train_scores.append(train_rmse)
    val_scores.append(val_rmse)

    # STORE OOF VAL AND TEST PREDICT
    oof_ridge[val_index] = y_val_predict
    test_ridge.append(y_test_predict)

    
print(f'\n🎉 Train Fold Prediction 1️⃣: {np.mean(train_scores)}')
print(f'🎉 Val Fold Prediction     2️⃣: {np.mean(val_scores)}\n')

print(f'🎉 std Train Fold Prediction 3️⃣: {np.std(train_scores, ddof = 0)}')
print(f'🎉 std Val Fold Prediction   4️⃣: {np.std(val_scores, ddof = 0)}')


# SAVE OOF AND SUBMISSION

#oof_ridge  = oof_ridge + train_df['score']
test_ridge = np.mean(test_ridge, axis = 0)# + test_df['score']

# SUBMISSION
submission = pd.read_csv(r'/kaggle/input/playground-series-s5e10/sample_submission.csv')
submission['accident_risk'] = test_ridge

display(submission)

# SAVE
pd.DataFrame(oof_ridge, columns = ['ridge_L2']).to_csv(r'meta_model/oof/oof_ridge_L2_V13.csv', index = False)
submission.to_csv(r'meta_model/test/ridge_L2_V13.csv', index = False)


# TOP MODEL - RIDGE

top_model = pd.DataFrame(ridge.coef_, columns = ['top_model'], index = ['lgbm', 'xgboost', 'catboost'])

sns.barplot(x = top_model.index, y = top_model['top_model'], palette = 'pastel')
plt.title('Best Model on Meta-Model Ridge')


# META-MODEL - ELASTIC NET

train_scores, val_scores, test_elastic = [], [], []
oof_elastic = np.zeros(len(x))

skfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 2025)

# SKFOLD
for i, (train_index, val_index), in enumerate(skfold.split(x, pd.qcut(y, q = 10).cat.codes)):

    # SPLIT DATA
    x_train, x_val = oof_model.iloc[train_index], oof_model.iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # DEFINE CATBOOST
    elastic = ElasticNet(alpha = 0.0001230508455517883, l1_ratio = 0.002111156035744549, max_iter = 10_000)

    elastic.fit(x_train, y_train)

    # PREDICT
    y_train_predict = elastic.predict(x_train)
    y_val_predict   = elastic.predict(x_val)
    y_test_predict = elastic.predict(test_prediction)

    # PREDICT MAE
    train_rmse = mean_squared_error(y_train, y_train_predict, squared = False)
    val_rmse   = mean_squared_error(y_val, y_val_predict, squared = False)

    print(f'Fold {i+1} 🚀 : 1️⃣ Train RMSE = {train_rmse}, 2️⃣ Val RMSE = {val_rmse}')

    # PUSH SCORE
    train_scores.append(train_rmse)
    val_scores.append(val_rmse)

    # STORE OOF VAL AND TEST PREDICT
    oof_elastic[val_index] = y_val_predict
    test_elastic.append(y_test_predict)

    
print(f'\n🎉 Train Fold Prediction 1️⃣: {np.mean(train_scores)}')
print(f'🎉 Val Fold Prediction     2️⃣: {np.mean(val_scores)}\n')

print(f'🎉 std Train Fold Prediction 3️⃣: {np.std(train_scores, ddof = 0)}')
print(f'🎉 std Val Fold Prediction   4️⃣: {np.std(val_scores, ddof = 0)}')


# SAVE OOF AND SUBMISSION

# CONVERT BACK TO NORMAL VALUES
#oof_elastic  = oof_elastic + train_df['score']
test_elastic = np.mean(test_elastic, axis = 0)# + test_df['score']

# SUBMISSION
submission = pd.read_csv(r'/kaggle/input/playground-series-s5e10/sample_submission.csv')
submission['accident_risk'] = test_elastic

display(submission)

# SAVE
pd.DataFrame(oof_elastic, columns = ['elastic_L2']).to_csv(r'meta_model/oof/oof_elastic_L2_V13.csv', index = False)
submission.to_csv(r'meta_model/test/elastic_L2_V13.csv', index = False)


# TOP MODEL ON ELASTIC NET

top_model = pd.DataFrame(elastic.coef_, columns = ['top_model'], index = ['lgbm', 'xgboost', 'catboost'])

sns.barplot(x = top_model.index, y = top_model['top_model'], palette = 'pastel')
plt.title('Best Model on Meta-Model Elastic Net')


# META-MODEL - RANDOM FOREST

train_scores, val_scores, test_rf = [], [], []
oof_rf = np.zeros(len(x))

skfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 2025)
0
# SKFOLD
for i, (train_index, val_index), in enumerate(skfold.split(x, pd.qcut(y, q = 10).cat.codes)):

    # SPLIT DATA
    x_train, x_val = oof_model.iloc[train_index], oof_model.iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # DEFINE CATBOOST
    rf = RandomForestRegressor(
        n_estimators = 700, 
        max_depth = 7,      
        random_state= 2025,
        n_jobs=-1           
    )

    rf.fit(x_train, y_train)

    # PREDICT
    y_train_predict = rf.predict(x_train)
    y_val_predict   = rf.predict(x_val)
    y_test_predict = rf.predict(test_prediction)

    # PREDICT MAE
    train_rmse = mean_squared_error(y_train, y_train_predict, squared = False)
    val_rmse   = mean_squared_error(y_val, y_val_predict, squared = False)

    print(f'Fold {i+1} 🚀 : 1️⃣ Train RMSE = {train_rmse}, 2️⃣ Val RMSE = {val_rmse}')

    # PUSH SCORE
    train_scores.append(train_rmse)
    val_scores.append(val_rmse)

    # STORE OOF VAL AND TEST PREDICT
    oof_rf[val_index] = y_val_predict
    test_rf.append(y_test_predict)

    
print(f'\n🎉 Train Fold Prediction 1️⃣: {np.mean(train_scores)}')
print(f'🎉 Val Fold Prediction     2️⃣: {np.mean(val_scores)}\n')

print(f'🎉 std Train Fold Prediction 3️⃣: {np.std(train_scores, ddof = 0)}')
print(f'🎉 std Val Fold Prediction   4️⃣: {np.std(val_scores, ddof = 0)}')


# SAVE OOF AND SUBMISSION

# CONVERT BACK TO NORMAL VALUES
#oof_rf  = oof_rf + train_df['score']
test_rf = np.mean(test_rf, axis = 0)# + test_df['score']

# SUBMISSION
submission = pd.read_csv(r'/kaggle/input/playground-series-s5e10/sample_submission.csv')
submission['accident_risk'] = test_rf

display(submission)

# SAVE
pd.DataFrame(oof_rf, columns = ['rf_L2']).to_csv(r'meta_model/oof/oof_rf_L2_V13.csv', index = False)
submission.to_csv(r'meta_model/test/rf_L2_V13.csv', index = False)


# FEATURE IMPORTANCE - RANDOM FOREST

feature_importance = pd.DataFrame({'feature': oof_model.columns, 'importance': rf.feature_importances_}).sort_values(by='importance', ascending=False)

sns.barplot(x = feature_importance['feature'], y = feature_importance['importance'], palette = 'pastel')
plt.title('Best Model on Meta-Model Elastic Net')


# META-MODEL MEAN AVERAGE

meta_avg_oof  = (oof_ridge + oof_elastic + oof_rf) / 3 
meta_avg_test = (test_ridge + test_elastic + test_rf) / 3 

submission = pd.read_csv(r'/kaggle/input/playground-series-s5e10/sample_submission.csv')
submission['accident_risk'] = meta_avg_test

display(submission)

# SAVE
pd.DataFrame(meta_avg_oof, columns = ['mean_L3']).to_csv(r'meta_model/oof/oof_mean_L3_V13.csv', index = False)
submission.to_csv(r'meta_model/test/mean_L3_V13.csv', index = False)


# TEST DISTRIBUTION COMPARISON
sns.kdeplot(train_df['accident_risk'], label = 'True Label (Train)', fill = False)
sns.kdeplot(test_ridge, label = 'Meta: Ridge', fill = False)
sns.kdeplot(test_elastic, label = 'Meta: ElasticNet', fill = False)
sns.kdeplot(test_rf, label = 'Meta: RandomForest', fill = False)
sns.kdeplot(test_avg, label = 'AVG Base models', fill = False)
sns.kdeplot(meta_avg_test, label = 'AVG Meta-models', fill = False)

plt.title('Test Prediction Distribution Comparison')
plt.legend()
plt.show()


# SAVE BASE MODEL 

# CONVERT BACK RESIDUAL TO NORMAL VALUES
lgbm_oof = oof_lgbm# + train_df['score']
xgb_oof  = oof_xgb# + train_df['score']
cat_oof  = oof_cat#+ train_df['score']

# CONVERT BACK RESIDUAL TO NORMAL VALUES
lgbm_test = np.mean(test_lgbm, axis = 0)# + test_df['score']
xgb_test  = np.mean(test_xgb, axis = 0)#  + test_df['score']
cat_test  = np.mean(test_cat, axis = 0)#  + test_df['score']

# CREATE OUTPUT DIRECTORY
os.makedirs('base_model/oof', exist_ok = True)
os.makedirs('base_model/test', exist_ok = True)

# SAVE CATBOOST
pd.DataFrame(cat_oof, columns = ['cat_ag_oof']).to_csv(r'base_model/oof/oof_cat_multi-oof_V13.csv', index = False)  # SAVE OOF
submission = pd.read_csv(r'/kaggle/input/playground-series-s5e10/sample_submission.csv')
submission['accident_risk'] = cat_test
submission.to_csv(r'base_model/test/cat_multi-oof_V13.csv', index = False)

# SAVE XGBOOST
pd.DataFrame(xgb_oof, columns = ['xgb_ag_oof']).to_csv(r'base_model/oof/oof_xgb_multi-oof_V13.csv', index = False)  # SAVE OOF
submission = pd.read_csv(r'/kaggle/input/playground-series-s5e10/sample_submission.csv')
submission['accident_risk'] = xgb_test
submission.to_csv(r'base_model/test/xgb_multi-oof_V13.csv', index = False)

# SAVE LGBM
pd.DataFrame(lgbm_oof, columns = ['lgbm_ag_oof']).to_csv(r'base_model/oof/oof_lgbm_multi-oof_V13.csv', index = False)  # SAVE OOF
submission = pd.read_csv(r'/kaggle/input/playground-series-s5e10/sample_submission.csv')
submission['accident_risk'] = lgbm_test
submission.to_csv(r'base_model/test/lgbm_multi-oof_V13.csv', index = False)

print(f'Data Saved!\n')
submission

