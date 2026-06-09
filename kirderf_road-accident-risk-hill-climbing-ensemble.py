


!pip install hillclimbers -q


import pandas as pd
import numpy as np
from hillclimbers import climb_hill, partial
from sklearn.metrics import mean_squared_error





# LOAD DATA

train_df = pd.read_csv(r'/kaggle/input/playground-series-s5e10/train.csv')
test_df  = pd.read_csv(r'/kaggle/input/playground-series-s5e10/test.csv')

# LOAD ONLY THE BEST AUTOGLUON OOF
autogluon_oof = pd.read_csv(r'/kaggle/input/autogluon-oof/AutoGluon OOF/OOF/oof_autogluon11.csv')
autogluon_test = pd.read_csv(r'/kaggle/input/autogluon-oof/AutoGluon OOF/Test/submission_autogluon11.csv')
train_df['autogluon_oof11'] = autogluon_oof['oof_prediction']
test_df['autogluon_oof11']  = autogluon_test['accident_risk']

train_df = train_df.iloc[:,-2:]
test_df = test_df.iloc[:,-1:]

print(f'Train : {train_df.shape[0]}')
print(f'Test  : {test_df.shape[0]}')


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

ridge_stacking_oof = pd.read_csv(r'/kaggle/input/accident-prediction-stacking-model/meta_model/oof/oof_ridge_L2_V13.csv')
ridge_stacking_test = pd.read_csv(r'/kaggle/input/accident-prediction-stacking-model/meta_model/test/ridge_L2_V13.csv')

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

merge_train = pd.concat((train_df, external_oof), axis = 1)
merge_test = pd.concat((test_df, external_test), axis = 1)




merge_train


Objective="minimize"
def custom_score(y_true, y_pred):
    return mean_squared_error(y_true, y_pred,squared=False)
Eval_metric=partial(custom_score)
Negative_weights=False
Precision=0.1
test_preds, oof_preds = climb_hill(
                 train=merge_train[['accident_risk']], 
                 oof_pred_df=merge_train.drop(columns = 'accident_risk'), 
                 test_pred_df=merge_test,
                 target='accident_risk',
                 objective=Objective, 
                 eval_metric=Eval_metric,
                 negative_weights=Negative_weights, 
                 precision=Precision,
                 return_oof_preds=True,
                 plot_hill=True,
                 plot_hist=True
            )






submission = pd.read_csv(r'/kaggle/input/playground-series-s5e10/sample_submission.csv')

submission['accident_risk'] = test_preds

submission


# SAVE SUBMISSION
submission.to_csv('submission.csv', index = False)
pd.DataFrame(oof_preds).to_csv('oof.csv', index=False)

