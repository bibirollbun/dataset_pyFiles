import pandas as pd
#data_path
data_path = '/kaggle/input/porto-seguro-safe-driver-prediction/'

train = pd.read_csv(data_path + 'train.csv', index_col='id')
test  = pd.read_csv(data_path + 'test.csv', index_col='id')
submission = pd.read_csv(data_path + 'sample_submission.csv', index_col='id')


all_data = pd.concat([train, test], ignore_index = True)
all_data = all_data.drop(['target'], axis = 1)

all_features = all_data.columns


from sklearn.preprocessing import OneHotEncoder

cat_features = [feature for feature in all_features if 'cat' in feature]

#OneHotEncoder
onehot_encoder = OneHotEncoder()
encoded_cat_matrix = onehot_encoder.fit_transform(all_data[cat_features])
encoded_cat_matrix


all_data['num_missing'] = (all_data == -1).sum(axis = 1)


all_data.head()


remaining_features = [feature for feature in all_features if ('cat' not in feature and 'calc' not in feature)]
remaining_features.append('num_missing')


remaining_features


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
for feature in cat_features+['mix_ind']:
  val_counts_dict = all_data[feature].value_counts().to_dict()
  all_data[f'{feature}_count'] = all_data[feature].apply(lambda  x: val_counts_dict[x])

  cat_count_features.append(f'{feature}_count')


cat_count_features


from scipy import sparse

drop_features = ['ps_ind_14', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_car_14']

all_data_remaining = all_data[remaining_features + cat_count_features].drop(drop_features, axis = 1)


all_data_sprs = sparse.hstack([sparse.csr_matrix(all_data_remaining), encoded_cat_matrix], format='csr')


num_train = len(train)
X = all_data_sprs[:num_train]
X_test = all_data_sprs[num_train:]
y = train['target'].values


!pip install --upgrade xgboost


!pip install bayesian-optimization


import xgboost as xgb
from xgboost.callback import EarlyStopping, EvaluationMonitor
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size =0.2, random_state=0)

#Dataset for Bayesian Optimization
bayes_dtrain = xgb.DMatrix(X_train, y_train)
bayes_dvalid = xgb.DMatrix(X_valid, y_valid)


param_bounds = {
    'max_depth': (4,8),
    'subsample': (0.6, 0.9),
    'colsample_bytree': (0.7, 1.0),
    'min_child_weight': (5,7),
    'reg_alpha': (7,9),
    'reg_lambda': (1.1,1.5),
    'gamma': (8, 11),
    'scale_pos_weight':(1.4, 1.6)
}

#Fixed hyperparameter
fixed_params = {'objective': 'binary:logistic',
                'learning_rate': 0.02,
                'random_state': 1991
                }


import numpy as np
def eval_gini(y_true, y_pred):
  """
  Computation function for Gini Coefficient
  """
  assert y_true.shape == y_pred.shape

  n_samples = y_true.shape[0]

  L_mid = np.linspace(1/ n_samples, 1, n_samples)  # the values on the diagonal line

  #1) Gini coefficinet for predicted values
  pred_order = y_true[y_pred.argsort()]
  L_pred = np.cumsum(pred_order)/ np.sum(pred_order)
  G_pred = np.sum(L_mid - L_pred)

  #2) Gini Coefficient when the prediction is perfect
  true_order = y_true[y_true.argsort()]
  L_true     = np.cumsum(true_order) / np.sum(true_order)
  G_true     = np.sum(L_mid - L_true)   # Gini Coefficient when the prediction is perfect

  # Normalized Gini Coefficient
  return G_pred / G_true


def gini(preds, dtrain):
  '''
  Computation function for Gini Coefficient with XGBoost
  '''
  labels = dtrain.get_label()
  gini_score = eval_gini(labels, preds)
  return 'gini', gini_score


def eval_function(max_depth, subsample, colsample_bytree, min_child_weight, reg_alpha, reg_lambda, gamma, scale_pos_weight):
  """
  Computation function for Evaluation metrics(Gini Coefficient)
  """
  params = {
      'max_depth':int(round(max_depth)),
      'subsample':subsample,
      'colsample_bytree':colsample_bytree,
      'min_child_weight':min_child_weight,
      'reg_alpha':reg_alpha,
      'reg_lambda':reg_lambda,
      'gamma':gamma,
      'scale_pos_weight':scale_pos_weight
  }
  #fixed hyperparameter add
  params.update(fixed_params)

  print(f"Hyperparameter : {params}")

  # Train XGBoost
  # `feval` parameter name was replaced into `custom_metric` under 3.0.5 version
  xgb_model = xgb.train(params = params,
                        dtrain=bayes_dtrain,
                        num_boost_round=2000,
                        evals = [(bayes_dvalid,'bayes_dvalid')],
                        maximize=True, # the larger Gini coefficient , the better.
                        #feval=gini,
                        custom_metric=gini,
                        early_stopping_rounds = 200,
                        verbose_eval = 100
                        )
  best_iter = xgb_model.best_iteration  # The Optimal iteration

  # Predict with valid data
  preds = xgb_model.predict(bayes_dvalid, iteration_range=(0, best_iter))

  # Compute Gini Coefficient
  gini_score = eval_gini(y_valid, preds)
  print(f"Gini Coefficient  : {gini_score}\n")

  return gini_score


# import xgboost as xgb
# print(xgb.train)
# print(xgb.__file__)


# !pip uninstall xgboost
# !pip install --no-cache-dir xgboost==3.0.5


#help(xgb.train)


from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(f=eval_function,
                                 pbounds = param_bounds ,
                                 random_state = 0)
optimizer.maximize(init_points = 3, n_iter = 6)


optimizer.max


max_params = optimizer.max['params']
max_params


max_params['max_depth'] = int(round(max_params['max_depth'])) #max_depth's data type should be integer type.
max_params.update(fixed_params)
max_params


from sklearn.model_selection import StratifiedKFold

folds = StratifiedKFold(n_splits = 5, shuffle=True, random_state=1991)

# A one-dimensional array containing the probabilities predicted 
# by the model trainned using OOF method for the validation data target values

oof_val_preds = np.zeros(X.shape[0])

# A one-dimensional array containing the probabilities predicted by the model
# trainned using OOF method for the test data target values
oof_test_preds = np.zeros(X_test.shape[0])

for idx, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
  print("#"* 40, f'Fold {idx + 1} / Fold : {folds.n_splits}', '#'* 40)

  # train data, validation data
  X_train, y_train = X[train_idx], y[train_idx]
  X_valid, y_valid = X[valid_idx], y[valid_idx]

  #Dataset creation for XGBoost dedicated dataset
  dtrain = xgb.DMatrix(X_train, y_train)
  dvalid = xgb.DMatrix(X_valid, y_valid)
  dtest  = xgb.DMatrix(X_test)

  #XGBBoost model train
  xgb_model = xgb.train(params = max_params,
                        dtrain= dtrain,
                        num_boost_round = 2000,
                        evals = [(dvalid, 'valid')],
                        maximize=True,
                        custom_metric=gini,
                        early_stopping_rounds= 200,
                        verbose_eval = 100)
  # Booting iteration cout setting for optimal model performance
  best_iter = xgb_model.best_iteration

  # OOF Prediction with the test data
  oof_test_preds += xgb_model.predict(dtest, iteration_range=(0, best_iter))/ folds.n_splits

  # Predict validation data target values for model performance validation
  oof_val_preds[valid_idx] += xgb_model.predict(dvalid, iteration_range=(0, best_iter))

  # Compute normalized Gini coefficient for predictive probabilities with validation data
  gini_score = eval_gini(y_valid, oof_val_preds[valid_idx])
  print(f"Fold {idx + 1} Gini Coefficient : {gini_score}\n")


submission['target'] = oof_test_preds
submission.to_csv('submission.csv')

