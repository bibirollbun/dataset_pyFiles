import pandas as pd
#data_path
data_path = '/kaggle/input/porto-seguro-safe-driver-prediction/'

train = pd.read_csv(data_path + 'train.csv', index_col='id')
test  = pd.read_csv(data_path + 'test.csv', index_col='id')
submission = pd.read_csv(data_path + 'sample_submission.csv', index_col='id')


all_data = pd.concat([train, test], ignore_index=True)
all_data = all_data.drop('target', axis = 1)


all_data


all_features = all_data.columns   #Overall features
all_features


from sklearn.preprocessing import OneHotEncoder

cat_features = [feature for feature in all_features if 'cat' in feature]
onehot_encoder = OneHotEncoder()
encoded_cat_matrix = onehot_encoder.fit_transform(all_data[cat_features])

# <Compressed Sparse Row sparse matrix of dtype 'float64'	with 20832392 stored elements and shape (1488028, 184)>
encoded_cat_matrix  # The number of features increased by the number of categorical features's distinct value.


all_data['num_missing'] = (all_data == -1).sum(axis = 1)


remaining_features = [feature for feature in all_features if ('cat' not in feature and 'calc' not in feature)]
remaining_features.append('num_missing')


#features ending with `ind`
ind_features = [feature for feature in all_features if 'ind' in feature]
is_first_feature = True

for ind_feauture in ind_features:
  if is_first_feature:
    all_data['mix_ind'] = all_data[ind_feauture].astype(str)+ '_'
    is_first_feature = False
  else:
    all_data['mix_ind'] += all_data[ind_feauture].astype(str) + '_'


all_data['mix_ind']


all_data['ps_ind_02_cat'].value_counts()


all_data['ps_ind_02_cat'].value_counts().to_dict()


cat_count_features = []
for feature in cat_features + ['mix_ind']:
  val_counts_dict = all_data[feature].value_counts().to_dict()
  all_data[f'{feature}_count'] = all_data[feature].apply(lambda x: val_counts_dict[x])

  cat_count_features.append(f'{feature}_count')


cat_count_features


from scipy import sparse

drop_features = ['ps_ind_14', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_car_14']

all_data_remaining = all_data[remaining_features + cat_count_features].drop(drop_features, axis =1)
all_data_sprs = sparse.hstack([sparse.csr_matrix(all_data_remaining), encoded_cat_matrix], format='csr')


all_data_sprs


num_train = len(train)

X = all_data_sprs[:num_train]
X_test = all_data_sprs[num_train:]
y = train['target'].values


import numpy as np

def eval_gini(y_true, y_pred):
  """
  Compute gini coefficient
  """
  assert y_true.shape == y_pred.shape

  n_samples = y_true.shape[0]
  L_mid     = np.linspace(1/n_samples, 1, n_samples) # The values on the diagonal line

  #1)Gini coefficient for predicted values
  pred_order = y_true[y_pred.argsort()]
  L_pred = np.cumsum(pred_order) / np.sum(pred_order)  #Lorenzo curve
  G_pred = np.sum(L_mid - L_pred)   # Gini coefficient for predicted values

  #2)Gini coefficient where predictions are perfect
  true_order = y_true[y_true.argsort()]
  L_true     = np.cumsum(true_order) / np.sum(true_order)
  G_true     = np.sum(L_mid - L_true)  #Gini coefficient where predictions are perfect

  #normalized gini coefficient
  return G_pred / G_true


def lightgbm_gini(preds, dtrain):
  """
  Gini function for LightGBM
  """
  labels = dtrain.get_label()
  gini_score = eval_gini(labels, preds)
  return 'gini', gini_score, True

def xgboost_gini(preds, dtrain):
  """
  Gini function for XGBoost
  """
  labels = dtrain.get_label()
  gini_score = eval_gini(labels, preds)
  return 'gini', gini_score


!pip install lightgbm


import lightgbm as lgb
max_param_lgb = {
    'bagging_fraction': 0.6    # (subsample)Data sampling ratio to use for trainning the individual trees.To enable bagging, set the bagging_fraction parameter to a value other than 0
    ,'feature_fraction': 0.6   # Feature sampling ratio to use for trainning the individual trees.
    ,'lambda_l1': 0.7
    ,'lambda_l2': 0.9
    ,'min_child_samples': 9 # Minimum number of data points in a leaf. Controls overfitting
    ,'min_child_weight': 36 # Minimum sum of instance weight (hessian) needed in a leaf
    ,'num_leaves': 40       # Maximum number of leaves in one tree. Larger = more complex model
    ,'objective': 'binary'
    ,'learning_rate': 0.005
    ,'bagging_freq': 1   # bagging frequency - Decide how many iterations to perform bagging. 0 - no bagging, 1 - On each iterations, trains the trees with the new sampling data
    ,'force_row_wise': True # a parameter that improves memory efficiency whem memory capacity is insufficient
    ,'random_state': 1991
}


help(lgb.train)


from sklearn.model_selection import StratifiedKFold
from lightgbm.callback import early_stopping, log_evaluation

folds = StratifiedKFold(n_splits = 5, shuffle=True, random_state=1991)

#A one-dimensional array containing the probabilities predicted by the model trained using the OOF method for the validation data target values
oof_val_preds_lgb = np.zeros(X.shape[0])
# A one-dimensional array containing the probabilities predicted by the mode trained using the OOF method for the test data target values
oof_test_preds_lgb = np.zeros(X_test.shape[0])

for idx, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
  print('#'* 40, f"Fold {idx + 1} / Fold {folds.n_splits}", '#'* 40)

  X_train, y_train = X[train_idx], y[train_idx]   # train dataset
  X_valid, y_valid = X[valid_idx], y[valid_idx]  # validation dataset

  #LightGBM dedicated dataset declaration
  dtrain = lgb.Dataset(X_train, y_train)
  dvalid = lgb.Dataset(X_valid, y_valid)

  #train model
  lgb_model = lgb.train(params = max_param_lgb,
                        train_set=dtrain,
                        num_boost_round=2500 ,
                        valid_sets=dvalid ,
                        feval=lightgbm_gini,
                        callbacks=[lgb.early_stopping(stopping_rounds=300), lgb.log_evaluation(period=100)]
                        )
  #Predict probabilities using test dataset
  oof_test_preds_lgb += lgb_model.predict(X_test) / folds.n_splits

  #Predict probabilities validation data target values for model performance evaluation
  oof_val_preds_lgb[valid_idx] += lgb_model.predict(X_valid)

  # Gini Coefficient for prediction probability of  validation data
  gini_score = eval_gini(y_valid, oof_val_preds_lgb[valid_idx])
  print(f"Fold {idx + 1}  Gini Coeffcient : {gini_score}\n")


!pip install xgboost


import xgboost as xgb
max_params_xgb = {
    'colsample_bytree': 0.8843124587484356,
    'gamma': 10.452246227672624,
    'max_depth': 7,
    'min_child_weight': 6.494091293383359,
    'reg_alpha': 8.551838810159788,
    'reg_lambda': 1.3814765995549108,
    'scale_pos_weight': 1.423280772455086,
    'subsample': 0.7001630536555632,
    'objective': 'binary:logistic',
    'learning_rate': 0.02,
    'random_state': 1991
}


from sklearn.model_selection import StratifiedKFold
from lightgbm.callback import early_stopping, log_evaluation

folds = StratifiedKFold(n_splits = 5, shuffle=True, random_state=1991)

#A one-dimensional array containing the probabilities predicted by the model trained using the OOF method for the validation data target values
oof_val_preds_xgb = np.zeros(X.shape[0])
# A one-dimensional array containing the probabilities predicted by the mode trained using the OOF method for the test data target values
oof_test_preds_xgb = np.zeros(X_test.shape[0])

for idx, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
  print('#'* 40, f"Fold {idx + 1} / Fold {folds.n_splits}", '#'* 40)

  X_train, y_train = X[train_idx], y[train_idx]   # train dataset
  X_valid, y_valid = X[valid_idx], y[valid_idx]  # validation dataset

  #XGBboost dedicated dataset declaration
  dtrain = xgb.DMatrix(X_train, y_train)
  dvalid = xgb.DMatrix(X_valid, y_valid)
  dtest  = xgb.DMatrix(X_test)

  #train model
  xgb_model = xgb.train(
                        params = max_params_xgb,
                        dtrain = dtrain,
                        num_boost_round = 2500,
                        evals = [(dvalid, 'valid')],
                        maximize = True,
                        custom_metric= xgboost_gini,
                        early_stopping_rounds = 300,
                        verbose_eval = 100
                        )
  # Boositng iteration range set
  best_iter = xgb_model.best_iteration
  #Predict probabilities using test dataset
  oof_test_preds_xgb += xgb_model.predict(dtest, iteration_range = (0, best_iter))/ folds.n_splits

  #Predict probabilities validation data target values for model performance evaluation
  oof_val_preds_xgb[valid_idx] += xgb_model.predict(dvalid, iteration_range = (0, best_iter))

  # Gini Coefficient for prediction probability of  validation data
  gini_score = eval_gini(y_valid, oof_val_preds_xgb[valid_idx])
  print(f"Fold {idx + 1}  Gini Coeffcient : {gini_score}\n")


print(f"LightGBM OOF Validation Data Gini Coeff : {eval_gini(y, oof_val_preds_lgb)}")


print(f"XGBoost OOF Validation Data Gini Coeff : {eval_gini(y, oof_val_preds_xgb)}")


#weighted ratio : 50%
oof_test_preds = oof_test_preds_lgb * 0.5 + oof_test_preds_xgb * 0.5


submission['target'] = oof_test_preds
submission.to_csv('submission.csv')

