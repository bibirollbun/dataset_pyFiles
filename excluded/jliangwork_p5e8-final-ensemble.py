import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


y_val = pd.read_csv("/kaggle/input/p5e8-yval/y_val_B.csv").to_numpy().ravel()
y_train = pd.read_csv("/kaggle/input/p5e8-yval/y_train_B.csv").to_numpy().ravel()
y_fulltrain = np.hstack([y_train, y_val])


xgb_trainA = pd.read_csv("/kaggle/input/p5e8-xgb-models/xgb_trainA_predictions_stage1.csv").to_numpy().ravel()[45211:]
xgb_trainB = pd.read_csv("/kaggle/input/p5e8-xgb-models/xgb_trainB_predictions_stage1.csv").to_numpy().ravel()
xgb_valA = pd.read_csv("/kaggle/input/p5e8-xgb-models/xgb_valA_predictions_stage1.csv").to_numpy().ravel()
xgb_valB = pd.read_csv("/kaggle/input/p5e8-xgb-models/xgb_valB_predictions_stage1.csv").to_numpy().ravel()
xgb_fulltrainA = pd.read_csv("/kaggle/input/p5e8-xgb-models/xgb_fulltrainA_predictions_final.csv").to_numpy().ravel()[45211:]
xgb_fulltrainB = pd.read_csv("/kaggle/input/p5e8-xgb-models/xgb_fulltrainB_predictions_final.csv").to_numpy().ravel()
xgb_testA = pd.read_csv("/kaggle/input/p5e8-xgb-models/xgb_testA_predictions_final.csv").to_numpy().ravel()
xgb_testB = pd.read_csv("/kaggle/input/p5e8-xgb-models/xgb_testB_predictions_final.csv").to_numpy().ravel()

nn_trainA = pd.read_csv("/kaggle/input/p5e8-nn-models/nn_trainA_predictions_stage1.csv").to_numpy().ravel()[45211:]
nn_trainB = pd.read_csv("/kaggle/input/p5e8-nn-models/nn_trainB_predictions_stage1.csv").to_numpy().ravel()
nn_valA = pd.read_csv("/kaggle/input/p5e8-nn-models/nn_valA_predictions_stage1.csv").to_numpy().ravel()
nn_valB = pd.read_csv("/kaggle/input/p5e8-nn-models/nn_valB_predictions_stage1.csv").to_numpy().ravel()
nn_fulltrainA = pd.read_csv("/kaggle/input/p5e8-nn-models/nn_fulltrainA_predictions_final.csv").to_numpy().ravel()[45211:]
nn_fulltrainB = pd.read_csv("/kaggle/input/p5e8-nn-models/nn_fulltrainB_predictions_final.csv").to_numpy().ravel()
nn_testA = pd.read_csv("/kaggle/input/p5e8-nn-models/nn_testA_predictions_final.csv").to_numpy().ravel()
nn_testB = pd.read_csv("/kaggle/input/p5e8-nn-models/nn_testB_predictions_final.csv").to_numpy().ravel()

annoy_trainA = pd.read_csv("/kaggle/input/p5e8-annoy-models/annoy_trainA_predictions_stage1.csv").to_numpy().ravel()[45211:]
xgb_trainB = pd.read_csv("/kaggle/input/p5e8-xgb-models/xgb_trainB_predictions_stage1.csv").to_numpy().ravel()
annoy_trainB = pd.read_csv("/kaggle/input/p5e8-annoy-models/annoy_trainB_predictions_stage1.csv").to_numpy().ravel()
annoy_valA = pd.read_csv("/kaggle/input/p5e8-annoy-models/annoy_valA_predictions_stage1.csv").to_numpy().ravel()
annoy_valB = pd.read_csv("/kaggle/input/p5e8-annoy-models/annoy_valB_predictions_stage1.csv").to_numpy().ravel()
annoy_fulltrainA = pd.read_csv("/kaggle/input/p5e8-annoy-models/annoy_fulltrainA_predictions_final.csv").to_numpy().ravel()[45211:]
xgb_trainB = pd.read_csv("/kaggle/input/p5e8-xgb-models/xgb_trainB_predictions_stage1.csv").to_numpy().ravel()
annoy_fulltrainB = pd.read_csv("/kaggle/input/p5e8-annoy-models/annoy_fulltrainB_predictions_final.csv").to_numpy().ravel()
annoy_testA = pd.read_csv("/kaggle/input/p5e8-annoy-models/annoy_testA_predictions_final.csv").to_numpy().ravel()
annoy_testB = pd.read_csv("/kaggle/input/p5e8-annoy-models/annoy_testB_predictions_final.csv").to_numpy().ravel()


train_dictionary = {"xgbA": xgb_trainA, "xgbB": xgb_trainB, 
                    "nnA": nn_trainA, "nnB": nn_trainB,
                    "annoyA": annoy_trainA, "annoyB": annoy_trainB
                   }

val_dictionary = {"xgbA": xgb_valA, "xgbB": xgb_valB, 
                    "nnA": nn_valA, "nnB": nn_valB,
                   "annoyA": annoy_valA, "annoyB": annoy_valB
                 }

test_dictionary = {"xgbA": xgb_testA, "xgbB": xgb_testB, 
                    "nnA": nn_testA, "nnB": nn_testB,
                    "annoyA": annoy_testA, "annoyB": annoy_testB
                  }

fulltrain_dictionary = {"xgbA": xgb_fulltrainA, "xgbB": xgb_fulltrainB, 
                    "nnA": nn_fulltrainA, "nnB": nn_fulltrainB,
                    "annoyA": annoy_fulltrainA, "annoyB": annoy_fulltrainB
                       }

X_train = pd.DataFrame.from_dict(train_dictionary)
X_val = pd.DataFrame.from_dict(val_dictionary)
X_test = pd.DataFrame.from_dict(test_dictionary)
X_full_train = pd.DataFrame.from_dict(fulltrain_dictionary)


from sklearn.metrics import roc_auc_score


import xgboost as xgb

ensemble_xgb_model = xgb.XGBClassifier(
    objective = 'binary:logistic',
    eval_metric = 'auc',
    n_estimators = 2000,
    learning_rate = 0.005,
    max_depth = 2,
    #max_leaves = 2,
    subsample = 0.1,
    random_state = 1,
    #grow_policy = 'lossguide',
    scale_pos_weight = 7,
    reg_lambda = 0.000,
    colsample_bytree = 0.34,
)

ensemble_xgb_model.fit(X_train, y_train, 
               eval_set=[(X_train, y_train), (X_val, y_val)], 
               early_stopping_rounds = 200, 
               verbose = 10)


ensemble_xgb_model.get_booster().get_score(importance_type='gain')


if False:
    from sklearn.linear_model import LogisticRegression
    
    ensemble_lr_model = LogisticRegression(penalty = 'l1', 
                                           solver = 'liblinear',
                                           random_state=1, max_iter = 1000, class_weight = "balanced", C = .1)
    ensemble_lr_model.fit(X_train, y_train)
    y_pred = ensemble_lr_model.predict_proba(X_val)[:, 1]
    stacked_score = roc_auc_score(y_val, y_pred)
    print(f"The ROC AUC score for the stacked model on validation data is: {stacked_score:.4f}")



ensemble_xgb_model = xgb.XGBClassifier(
    objective = 'binary:logistic',
    eval_metric = 'auc',
    n_estimators = 1500,
    learning_rate = 0.005,
    max_depth = 2,
    #max_leaves = 2,
    subsample = 0.1,
    random_state = 1,
    #grow_policy = 'lossguide',
    scale_pos_weight = 7,
    reg_lambda = 0.000,
    colsample_bytree = 0.34,
)

ensemble_xgb_model.fit(X_full_train, y_fulltrain, 
               eval_set=[(X_full_train, y_fulltrain)], 
               verbose = 10)

final_prediction = ensemble_xgb_model.predict_proba(X_test)[ : , 1]
submission['y'] = final_prediction
submission.to_csv("submission.csv", index = False)


submission.head(20)




