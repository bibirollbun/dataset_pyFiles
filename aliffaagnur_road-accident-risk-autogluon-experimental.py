%%capture
!pip install autogluon.tabular scikit-learn==1.5.2
#!pip install "autogluon.tabular[tabpfn]==1.4.0"


import os
import scipy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')


# LOAD DATA

train_df = pd.read_csv(r'/kaggle/input/playground-series-s5e10/train.csv').drop(columns = 'id')
test_df  = pd.read_csv(r'/kaggle/input/playground-series-s5e10/test.csv').drop(columns = 'id')

print(f'Train : {train_df.shape[0]}')
print(f'Test  : {test_df.shape[0]}')


# LOAD EXTERNAL DATA (FROM OTHER NOTEBOOKS)

tabm_residual_oof  = pd.read_csv(r'/kaggle/input/s5e10-tabm-over-residuals/oof_tabm_overresid.csv')
tabm_residual_test = pd.read_csv(r'/kaggle/input/s5e10-tabm-over-residuals/test_tabm_overresid.csv')
print(f'Success Load Tabm Residual!')

realmlp_oof = pd.read_csv('/kaggle/input/pg-s5e10-realmlp-cv-0-055936-lb-0-05549/realmlp_oof.csv')
realmlp_test = pd.read_csv('/kaggle/input/pg-s5e10-realmlp-cv-0-055936-lb-0-05549/realmlp.csv')
print(f'Success Load RealMLP!')

xgb_residual_oof = pd.DataFrame(np.load(r'/kaggle/input/xgb-boosting-over-residuals-cv-0-05595/oof.npy'), columns = ['xgb_residual'])
xgb_residual_test = pd.read_csv(r'/kaggle/input/xgb-boosting-over-residuals-cv-0-05595/submission.csv')
print(f'Success Load XGB Residual!')

single_tabm_oof = pd.read_csv(r'/kaggle/input/s5e10-single-tabm-tuned/oof_tabm_plus_origcol_tuned.csv')
single_tabm_test = pd.read_csv(r'/kaggle/input/s5e10-single-tabm-tuned/test_tabm_plus_origcol_tuned.csv')
print(f'Success Load Single Tabm!')

xgb_diff_seed_oof = pd.read_csv(r'/kaggle/input/s5e10-xgb-origcol-20seeds/oof_xgb_plus_origcol.csv')
xgb_diff_seed_test = pd.read_csv(r'/kaggle/input/s5e10-single-tabm-tuned/test_tabm_plus_origcol_tuned.csv')
print(f'Success Load XGB * 20 Seed!')

xgb_rich_oof  = pd.read_csv(r'/kaggle/input/feature-rich-single-xgb-cv-0-05594/oof_xgb_enhanced_meta_20251011_184833_cv0.055942.csv')
xgb_rich_test = pd.read_csv(r'/kaggle/input/feature-rich-single-xgb-cv-0-05594/test_xgb_enhanced_meta_20251011_184833_cv0.055942.csv')
print(f'Success Load XGB Rich Features!')

nn_stacking_oof = pd.read_csv(r'/kaggle/input/s5e10-nn-stacking-baseline/oof_nn_ensemble.csv')
nn_stacking_test = pd.read_csv(r'/kaggle/input/s5e10-nn-stacking-baseline/test_nn_ensemble.csv')
print(f'Success Load NN Stacking!')

single_ydf_oof = pd.read_csv(r'/kaggle/input/road-risk-single-ydf/YDF_oof.csv')
single_ydf_test = pd.read_csv(r'/kaggle/input/road-risk-single-ydf/YDF_test.csv')
print(f'Success Load Single YDF!')

# MERGE ALL EXTERNAL DATA
external_oof = pd.concat((tabm_residual_oof, realmlp_oof, xgb_residual_oof, single_tabm_oof, xgb_diff_seed_oof, xgb_rich_oof, 
                          nn_stacking_oof, single_ydf_oof), axis = 1)

external_oof = external_oof.drop(columns = 'id')     # --> DROP ALL id FEATURES
external_oof.columns = ['tabm_residual', 'realmlp', 'xgb_residual', 'single_tabm', 'xgb_diff_seed20', 'xgb_rich',
                        'nn_stacking', 'single_ydf']  # RENAME COLUMNS


external_test = pd.concat((tabm_residual_test, realmlp_test, xgb_residual_test, single_tabm_test, xgb_diff_seed_test, xgb_rich_test, 
                           nn_stacking_test, single_ydf_test), axis = 1)

external_test = external_test.drop(columns = 'id')     # --> DROP ALL id FEATURES
external_test.columns = ['tabm_residual', 'realmlp', 'xgb_residual', 'single_tabm', 'xgb_diff_seed20', 'xgb_rich', 
                         'nn_stacking', 'single_ydf']  # RENAME COLUMNS


external_oof.shape, external_test.shape





# MERGING TRAIN DATA WITH EXTERNAL DATA

merge_train = pd.concat((train_df, external_oof), axis = 1)
merge_test = pd.concat((test_df, external_test), axis = 1)

display(merge_train)
display(merge_test)


# CHECK TARGET DISTRIBUTION

sns.histplot(train_df['accident_risk'], kde = True, color = 'orange', bins = 50)


# CHECK CORRELATION BETWEEN OOF

# GET ALL OOF AUTOGLUON COLUMNS
autogluon_features = merge_train.iloc[:, 13:].columns

# HEATMAP CORRELATION
plt.figure(figsize = (10, 8))
corr_matrix = merge_train[autogluon_features].corr(method = 'spearman')
sns.heatmap(corr_matrix, annot = False, cmap = 'coolwarm', fmt = '.2f')


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

train = clip(f)(merge_train)
test = clip(f)(merge_test)

merge_train['score'] = train
merge_test['score']  = test

display(merge_train.shape)
display(merge_test.shape)

merge_train


# AUTOGLUON

# DEFINE AUTOGLUON
predictor = TabularPredictor(label = 'accident_risk',
                             problem_type = 'regression',
                             eval_metric = 'rmse')

# TRAIN AUTOGLUON
predictor.fit(merge_train,
              presets = 'best_quality',
              time_limit = 3600 * 11,
              auto_stack = True,
              use_bag_holdout = True,
              #num_bag_folds = 5,
              #num_bag_sets = 3,
              num_cpus = 4,
              verbosity = 2,
              ag_args_ensemble = {'use_orig_features' : False},
              ag_args_fit={'early_stopping_rounds': 300, 'num_cpus': 4, 'num_gpus': 0},
              #ag_args_fit={'time_limit': 900}
             )


%%time
# COMPARE MODELS
predictor.leaderboard(silent = True)


# CHECK FEATURE IMPORTANCES 

importance_df = predictor.feature_importance(merge_train[:500])

importance_df.style.background_gradient(subset=['importance', 'stddev'], cmap='Blues')


# PLOT FEATURE IMPORTANCE

imp = importance_df['importance'].sort_values(ascending=True)

plt.figure(figsize=(6, 8))
imp.plot(kind='barh', color='steelblue')
plt.title('Feature Importance (AutoGluon)')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()


# CHECKING BEST MODEL 

best_model = predictor.model_best

print(f'Best Model : {best_model}')


%%time
# CHECK SUBMISSION

# TEST DATA PREDICTION
y_test = predictor.predict(merge_test)

submission = pd.read_csv(r'/kaggle/input/playground-series-s5e10/sample_submission.csv')

submission['accident_risk'] = y_test

submission


# GET OOF (OUT-OF-FOLD) PREDICTION

# GET OOF
oof_predictions = predictor.predict_oof()

# CONVERT TO DATAFRAME
y_pred = oof_predictions.to_frame(name = 'oof_prediction')  # ---> RETURN DATAFRAME
oof_df = pd.DataFrame(y_pred)

oof_df


# EVALUATION

rmse = mean_squared_error(train_df['accident_risk'], oof_predictions, squared = False)
r2   = r2_score(train_df['accident_risk'], oof_predictions)

print(f'RMSE : {rmse}')
print(f'R2   : {r2}"')


# RESIDUAL PLOT

y_true = merge_train['accident_risk']

plt.figure(figsize = (12, 5))

# ACTUAL VS PREDICTED DATA
plt.subplot(1, 2, 1)
plt.scatter(x = y_true, y = y_pred, alpha = 0.6)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)

plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Actual vs Predicted Data')

# RESIDUAL PLOT
residual = y_true - y_pred['oof_prediction'].values

plt.subplot(1, 2, 2)
plt.scatter(x = y_pred, y = residual, alpha = 0.6)
plt.axhline(y=0, color='r', linestyle='--') 
plt.xlabel("Predicted values")
plt.ylabel("Residuals (y_true - y_pred)")
plt.title("Residual Plot")

plt.show()


# DISTRIBUTION COMPARISON

plt.figure(figsize = (12, 5))

plt.subplot(1, 2, 1)
sns.kdeplot(merge_train['accident_risk'], label = 'True Label', fill = False)
sns.kdeplot(oof_predictions, label = 'Predicted Label (OOF)', fill = False)
plt.title('True vs Predicted Accident Risk Distribution')
plt.legend()

plt.subplot(1, 2, 2)
sns.kdeplot(y_test, label = 'Test Acccident Risk', fill = False)
plt.title('Test Accident Risk Distribution')
plt.legend()

plt.show()


# SAVE SUBMISSION
submission.to_csv(r'autogluon_experiment11.csv', index = False)
oof_df.to_csv(r'oof_autogluon_experiment11.csv', index = False)

