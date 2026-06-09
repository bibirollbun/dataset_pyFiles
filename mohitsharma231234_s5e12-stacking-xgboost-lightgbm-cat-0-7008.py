import numpy as np 
import pandas as pd


oof_lgb = np.load("/kaggle/input/s5e12-feature-en-lightgbm-0-6995/lgb_oof.npy")
oof_xgb = np.load("/kaggle/input/s5e12-xgboost-diabetes-predictic-0-7006/xgb_oof.npy")
oof_cat = np.load("/kaggle/input/fork-of-s5e12-catboost-feature-eng-0-7004/cat_oof.npy")
oof_nn = np.load("/kaggle/input/s5e12-ann-cv-deep-learning-0-69823/nn_oof.npy")


lgb_preds = np.load("/kaggle/input/s5e12-feature-en-lightgbm-0-6995/lgb_test.npy")
xgb_preds = np.load("/kaggle/input/s5e12-xgboost-diabetes-predictic-0-7006/xgb_preds.npy")
cat_preds = np.load("/kaggle/input/fork-of-s5e12-catboost-feature-eng-0-7004/cat_preds.npy")
nn_preds = np.load("/kaggle/input/s5e12-ann-cv-deep-learning-0-69823/nn_preds.npy")


train_d = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
org_df = pd.read_csv("/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv")
train_columns = train_d.columns.tolist()
new_df = org_df.reindex(columns = train_columns)
train_d = pd.concat([train_d,new_df],ignore_index = True)



y = train_d['diagnosed_diabetes']


print(oof_lgb.shape)
print(oof_xgb.shape)
print(oof_cat.shape)


from sklearn.linear_model import LogisticRegression
stack_train = np.vstack([oof_lgb,oof_xgb,oof_cat,oof_nn]).T
stack_test = np.vstack([lgb_preds,xgb_preds,cat_preds,nn_preds]).T
log_model = LogisticRegression(max_iter = 2000)

log_model.fit(stack_train,y)

preds = log_model.predict_proba(stack_test)[:,1]


# Submission 
submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
submission['diagnosed_diabetes'] = preds
submission.to_csv("submission.csv", index=False)


submission.head()




