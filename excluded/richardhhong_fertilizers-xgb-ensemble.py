import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter('ignore')

SEED = 30
K = 10
NUM_CLASSES = 7


df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df_original = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")


le = LabelEncoder()
le.fit(df_train['Fertilizer Name'])

def make_features(df, test=False, original=False):
    df_temp = df.copy()
    if not original:
        df_temp.drop(columns=['id'], inplace=True)
    cat_cols = df_temp.select_dtypes(include=['object']).columns
    df_temp[cat_cols] = df_temp[cat_cols].astype('category')

    # adding binning of numerical features
    numerical_features = [col for col in df_temp.select_dtypes(include=['int64', 'float64']).columns]
    for col in numerical_features:
        df_temp[f'{col}_Binned'] = df_temp[col].astype(str).astype('category')

    if not test:
        df_temp['Fertilizer Name'] = le.transform(df_temp['Fertilizer Name'])
    
    return df_temp


def mapk(actual, predicted, k=3):
    total_score = 0.0
    actual = le.inverse_transform(actual)
    for a, p in zip(actual, predicted):
        if a in p[:k]:
            index = p.index(a)
            total_score += 1.0 / (index + 1)
    return total_score / len(actual)


df_train1 = make_features(df_train)
df_original1 = make_features(df_original, original=True)
df_test1 = make_features(df_test, test=True)


initial_params = {
    "tree_method": "gpu_hist",
    "predictor": "gpu_predictor",
    'seed': SEED,
    'enable_categorical': True,
    'early_stopping_rounds': 100
}


X = df_train1.drop(columns=['Fertilizer Name'])
y = df_train1['Fertilizer Name']

X_original = df_original1.drop(columns=['Fertilizer Name'])
y_original = df_original1['Fertilizer Name']

X_original_copy = X_original.copy()
y_original_copy = y_original.copy()


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=SEED)


model_params = [
    # from https://www.kaggle.com/code/hahahaj/single-xgb
    [{
        'objective': 'multi:softprob',  
        'num_class': 7, 
        'max_depth': 7,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'max_bin': 128,
        'colsample_bytree': 0.3, 
        'colsample_bylevel': 1,  
        'colsample_bynode': 1,  
        'tree_method': 'hist',  
        'random_state': SEED,
        'eval_metric': 'mlogloss',
        'device': "cuda",
        'enable_categorical':True,
        'n_estimators':10000,
        'early_stopping_rounds':50,
    }, 7], # weight of original

    # my own hyperparams from tuning without using original data
    [{'learning_rate': 0.03,
       'max_depth': 11,
       'subsample': 0.8,
       'colsample_bytree': 0.30000000000000004,
       'max_bin': 1551,
       'min_child_weight': 3,
       'gamma': 0.0,
       'lambda': 0.0013312723042592412,
       'alpha': 0.6136573473631746,
       'max_delta_step': 8,
       'n_estimators': 3000,
       'enable_categorical': True,
       'early_stopping_rounds': 100,
       'random_state': SEED,
       'device': "cuda"
     }, 7],

    # my own hyperparams from tuning optuna with original data
     [{'learning_rate': 0.05,
       'max_depth': 7,
       'subsample': 0.8,
       'colsample_bytree': 0.4,
       'max_bin': 1323,
       'min_child_weight': 4,
       'gamma': 0.03,
       'lambda': 0.00953514350973168,
       'alpha': 0.6191568184269528,
       'max_delta_step': 3,
       "n_estimators": 3000,
       "enable_categorical": True,
       'early_stopping_rounds': 100,
       'random_state': SEED,
       'device': "cuda"
       }, 3]
    

]


# Setting Objects Containing Models and Iteration Parameters
kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)

base_models = [XGBClassifier(**(params[0])) for params in model_params]
original_iterations = [params[1] for params in model_params]

N_MODELS = len(base_models)


# # Making OOF Predictions
# oof_train = np.zeros((len(X_train), N_MODELS * NUM_CLASSES))
# val_preds = np.zeros((len(X_val), N_MODELS * NUM_CLASSES))

# for m_idx, model in enumerate(base_models):
#     val_fold_preds = []
#     scores = []
#     fold = 0

#     for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
#         X_original_fold = pd.concat([X_original_copy] * original_iterations[m_idx])
#         y_original_fold = pd.concat([y_original_copy] * original_iterations[m_idx])

#         X_train_fold = pd.concat([X_train.iloc[train_idx], X_original_fold]).reset_index(drop=True)
#         y_train_fold = pd.concat([y_train.iloc[train_idx], y_original_fold]).reset_index(drop=True)
#         X_val_fold = X_train.iloc[val_idx]
#         y_val_fold = y_train.iloc[val_idx]


#         model.fit(
#             X_train_fold,
#             y_train_fold,
#             eval_set=[(X_val_fold, y_val_fold)],
#             verbose=250,
#         )

#         # Predict on validation fold and store in OOF matrix
#         probas = model.predict_proba(X_val_fold)
#         oof_train[val_idx, m_idx*NUM_CLASSES:(m_idx+1)*NUM_CLASSES] = probas

#         # Predict on test data and save for averaging
#         val_pred = model.predict_proba(X_val)
#         val_fold_preds.append(val_pred)

#         val_fold_pred = model.predict_proba(X_val_fold)
#         val_fold_pred = np.argsort(val_fold_pred, axis=1)[:, -3:][:, ::-1]
#         val_fold_pred = [[le.classes_[j] for j in row] for row in val_fold_pred]
#         score = mapk(y_val_fold, val_fold_pred)
#         scores.append(score)

#         print(f"XGB Model {m_idx+1} fold {fold} val score: {score}")

#     avg_score = np.mean(scores)
#     print(f"========== XGB Model {m_idx+1} Average Val Score: {avg_score} ==========")
#     print('\n')
#     val_preds[:, m_idx*NUM_CLASSES:(m_idx+1)*NUM_CLASSES] = np.mean(val_fold_preds, axis=0)


# # saving so we don't have to run oof loop every time
# np.save('oof_train.npy', oof_train)
# np.save('val_preds.npy', val_preds)


# Using Data From Earlier Run
oof_train = np.load('/kaggle/input/fertilizers-oof-train-data/oof_train.npy')
val_preds = np.load('/kaggle/input/fertilizers-oof-train-data/val_preds.npy')


from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import GridSearchCV


# No Tuned Regularization
lr_meta_model = LogisticRegression(max_iter=1000, random_state=SEED)
lr_meta_model.fit(oof_train, y_train)
meta_val_preds = lr_meta_model.predict_proba(val_preds)
meta_val_preds = np.argsort(meta_val_preds, axis=1)[:, -3:][:, ::-1]
meta_val_preds = [[le.classes_[j] for j in row] for row in meta_val_preds]
score = mapk(y_val, meta_val_preds)
print(f"Logistic Regression Meta Model Val Score: {score}")


# With Tuned Regularization
lr_meta_model = LogisticRegression(max_iter=1000, penalty='l2', C=10.0, random_state=SEED)
lr_meta_model.fit(oof_train, y_train)
meta_val_preds = lr_meta_model.predict_proba(val_preds)
meta_val_preds = np.argsort(meta_val_preds, axis=1)[:, -3:][:, ::-1]
meta_val_preds = [[le.classes_[j] for j in row] for row in meta_val_preds]
score = mapk(y_val, meta_val_preds)
print(f"Logistic Regression Meta Model Val Score: {score}")


kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)

oof_meta_train = np.zeros((len(oof_train), NUM_CLASSES))

for fold, (train_idx, val_idx) in enumerate(kf.split(oof_train, y_train), 1):
    X_train_fold = oof_train[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = oof_train[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    model = LogisticRegression(max_iter=1000, penalty='l2', C=10.0, random_state=SEED)
    model.fit(X_train_fold, y_train_fold)
    oof_meta_train[val_idx] = model.predict_proba(X_val_fold)
    meta_val_preds = model.predict_proba(X_val_fold)
    meta_val_preds = np.argsort(meta_val_preds, axis=1)[:, -3:][:, ::-1]
    meta_val_preds = [[le.classes_[j] for j in row] for row in meta_val_preds]
    score = mapk(y_val_fold, meta_val_preds)
    print(f'Fold {fold} MAPK Score: {score}')


import lightgbm as lgb


# lgb_meta_model = lgb.LGBMClassifier(
#     device='gpu',
#     verbosity=-1
# )
# lgb_meta_model.fit(oof_train, y_train)
# meta_val_preds = lgb_meta_model.predict_proba(val_preds)
# meta_val_preds = np.argsort(meta_val_preds, axis=1)[:, -3:][:, ::-1]
# meta_val_preds = [[le.classes_[j] for j in row] for row in meta_val_preds]
# score = mapk(y_val, meta_val_preds)
# print(f"LGB Meta Model Val Score: {score}")


# optuna for lgb
import optuna

def objective(trial):
    params = {
        "device": 'gpu',
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 13),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        'early_stopping_rounds':50,
        'verbosity': -1,
        "n_estimators": 1000,
        "random_state": SEED
    }

    lr_meta_val_preds = lr_meta_model.predict_proba(val_preds)

    scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(oof_meta_train, y_train), 1):
        X_train_fold = oof_meta_train[train_idx]
        y_train_fold = y_train.iloc[train_idx]
        X_val_fold = oof_meta_train[val_idx]
        y_val_fold = y_train.iloc[val_idx]
        
        bonus_meta_model = lgb.LGBMClassifier(**params)
        bonus_meta_model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)])
        meta_val_preds = bonus_meta_model.predict_proba(X_val_fold)
        meta_val_preds = np.argsort(meta_val_preds, axis=1)[:, -3:][:, ::-1]
        meta_val_preds = [[le.classes_[j] for j in row] for row in meta_val_preds]
        score = mapk(y_val_fold, meta_val_preds)
        print(score)
        scores.append(score)
    
    return np.mean(score)


# study = optuna.create_study(direction='maximize',
#                             sampler = optuna.samplers.RandomSampler(seed=SEED),
#                             study_name = "BIG BLUE FIN TUNA!!")
# study.optimize(objective, n_trials=50, show_progress_bar=True)



# best_params = study.best_params
# print(f'Best Trial Params: {best_params}')

# print(f'Best Trial Value: {study.best_trial.value}')


df_test1 = make_features(df_test, test=True)


# Store models and predictions
test_preds = np.zeros((len(df_test1), N_MODELS * NUM_CLASSES))

for i in range(N_MODELS):
    model = base_models[i]

    # Extend training data with original data replicated as needed
    X_original_repeated = pd.concat([X_original_copy] * original_iterations[i])
    y_original_repeated = pd.concat([y_original_copy] * original_iterations[i])
    X_train_model = pd.concat([X_train, X_original_repeated]).reset_index(drop=True)
    y_train_model = pd.concat([y_train, y_original_repeated]).reset_index(drop=True)

    # Train the model
    model.fit(X_train_model, y_train_model, eval_set=[(X_val, y_val)], verbose=250)

    # Predict probabilities and store in the correct slice
    test_preds[:, i * NUM_CLASSES:(i + 1) * NUM_CLASSES] = model.predict_proba(df_test1)


best_params = {'num_leaves': 38,
               'max_depth': 9,
               'learning_rate': 0.016686659437902658,
               'min_child_samples': 18,
               'subsample': 0.5154109456308481,
               'subsample_freq': 10,
               'colsample_bytree': 0.6284188576231671,
               'lambda_l1': 3.267468632569532e-08,
               'lambda_l2': 3.085844376113324e-05,
               'min_split_gain': 0.3570967475164035,
               "device": 'gpu',
               'early_stopping_rounds':50,
               'verbosity': -1,
               "n_estimators": 1000,
               "random_state": SEED
              }

bonus_meta_model = lgb.LGBMClassifier(**best_params)

lr_meta_val_preds = lr_meta_model.predict_proba(val_preds)

bonus_meta_model.fit(oof_meta_train, y_train, eval_set=[(lr_meta_val_preds, y_val)])
meta_val_preds = bonus_meta_model.predict_proba(lr_meta_val_preds)
meta_val_preds = np.argsort(meta_val_preds, axis=1)[:, -3:][:, ::-1]
meta_val_preds = [[le.classes_[j] for j in row] for row in meta_val_preds]
score = mapk(y_val, meta_val_preds)
print(f"LGB Meta Meta Model Val Score: {score}")


y_test_pred = lr_meta_model.predict_proba(test_preds)
y_test_pred = bonus_meta_model.predict_proba(y_test_pred)

y_test_pred = np.argsort(y_test_pred, axis=1)[:, -3:][:, ::-1]
y_test_pred = [[le.classes_[j] for j in row] for row in y_test_pred]
y_test_pred = [' '.join(row) for row in y_test_pred]

submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
submission['Fertilizer Name'] = y_test_pred
submission.to_csv('submission.csv', index=False)
submission.head()

