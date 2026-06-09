import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
%matplotlib inline
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier, plot_importance
from catboost import CatBoostClassifier


pd.options.display.max_rows=100
pd.options.display.max_columns=None


cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)


train_data = pd.read_csv('/kaggle/input/playground-series-s4e3/train.csv')
original_data = pd.read_csv('/kaggle/input/faulty-steel-plates/faults.csv')
TARGET_FEATURES = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains','Dirtiness', 'Bumps', 'Other_Faults']
test_data = pd.read_csv('/kaggle/input/playground-series-s4e3/test.csv')


train_data.drop(['id'],axis = 1,inplace = True)
test_data.drop(['id'],inplace = True,axis = 1)
train_data = pd.concat([train_data,original_data],axis = 0)
train_data = pd.concat([train_data],axis = 0)
train_data.reset_index(drop=True, inplace=True)


train_data = train_data[train_data[TARGET_FEATURES].sum(axis=1) <= 1]
train_data['Outside_Global_Index'] = np.where(train_data['Outside_Global_Index']==0.7, 0.5, train_data['Outside_Global_Index'])
targets_bin = train_data[TARGET_FEATURES]
y_xgb = targets_bin


train_data['Target'] = np.argmax(train_data[TARGET_FEATURES].values, axis=1) + 1
train_data.loc[train_data[TARGET_FEATURES].sum(axis=1) == 0, 'Target'] = 0


train_data.drop(TARGET_FEATURES, inplace=True,axis =1)


def feature_engineering(data):
    data['Ratio_Length_Thickness'] = data['Length_of_Conveyer'] / data['Steel_Plate_Thickness']
    data['Normalized_Steel_Thickness'] = (data['Steel_Plate_Thickness'] - data['Steel_Plate_Thickness'].min()) / (data['Steel_Plate_Thickness'].max() - data['Steel_Plate_Thickness'].min())


    data['width'] = data['X_Maximum'] - data['X_Minimum']
    data['height'] = data['Y_Maximum'] - data['Y_Minimum']
    data['rel_width'] = data['width'] / data['Steel_Plate_Thickness']
    data['rel_height'] = data['height'] / data['Steel_Plate_Thickness']
    data['X_Range*Pixels_Areas'] = (data['X_Maximum'] - data['X_Minimum']) * data['Pixels_Areas']
    data['log_luminosity'] = np.log1p(data['Sum_of_Luminosity'])
    data['Compactness'] = (data['X_Perimeter']+data['Y_Perimeter'])*(data['X_Perimeter']+data['Y_Perimeter']) / data['Pixels_Areas']
    data['Luminosity_fluctuation'] = (data['Maximum_of_Luminosity'] + data['Minimum_of_Luminosity'])/2 - (data['Sum_of_Luminosity']/data['Pixels_Areas'])*(data['Sum_of_Luminosity']/data['Pixels_Areas'])
    data['is_large_area'] = (data['Pixels_Areas'] > data['Pixels_Areas'].quantile(0.8)).astype(int)
    return data


train_data = feature_engineering(train_data)
test_data = feature_engineering(test_data)
features_to_drop = ['Y_Minimum', 'Steel_Plate_Thickness', 'Sum_of_Luminosity', 'Edges_X_Index', 'SigmoidOfAreas', 'Luminosity_Index', 'TypeOfSteel_A300']


train_data = train_data.drop(features_to_drop,axis = 1)
test_data = test_data.drop(features_to_drop,axis = 1)


X = train_data.drop(['Target'], axis=1)  
y = train_data['Target']  


X


RETRAIN_XGB_MODEL = False
def objective(trial):
    # Define hyperparameters to tune
    param = {
        'grow_policy': trial.suggest_categorical('grow_policy', ["depthwise", "lossguide"]),
        'multi_strategy': trial.suggest_categorical('multi_strategy', ["one_output_per_tree"]),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
        'gamma' : trial.suggest_float('gamma', 1e-5, 0.5, log=True),
        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        
    }
    param['n_estimators'] = 3000
    param['early_stopping_rounds'] = 50
    param['booster'] = 'gbtree'
    param["verbosity"] = 0
    param['tree_method'] = "gpu_hist"
    param['device'] = "cuda:0"
    param['n_gpus'] = 1
    
    auc_scores = []

    for train_idx, valid_idx in cv.split(X, y):

        X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]
        y_xgb_train_fold, y_xgb_valid_fold = y_xgb.iloc[train_idx], y_xgb.iloc[valid_idx]
                
        # Create and fit the model
        model = XGBClassifier(**param)
        model.fit(X_train_fold, y_xgb_train_fold, eval_set=[(X_valid_fold, y_xgb_valid_fold)],verbose=False)

        # Predict class probabilities
        y_prob = model.predict_proba(X_valid_fold)

        # Compute the AUC for each class and take the average
        average_auc = roc_auc_score(targets_bin.iloc[valid_idx], y_prob, multi_class="ovr", average="macro")
        auc_scores.append(average_auc)

    # Return the average AUC score across all folds
    return np.mean(auc_scores)


if RETRAIN_XGB_MODEL:
    xgb_study = optuna.create_study(direction='maximize',study_name = "xgb_model_training")
    xgb_study.optimize(objective, n_trials=100)  # Adjust the number of trials as necessary
    # Output the optimization results
    print(f"Best trial average AUC: {study.best_value:.4f}")
    print(study.best_params)
    for key, value in study.best_params.items():
        print(f"{key}: {value}")


RETRAIN_LGBM_MODEL = False
# Define the objective function for Optuna optimization
def objective(trial):
    # Define parameters to be optimized for the LGBMClassifier
    
    param = {
    'objective': 'multiclass',  # Equivalent to multi:softmax but needs num_class as well
    'num_class': 8,  # Specify the number of classes if your task is multi-class classification
    'learning_rate': trial.suggest_float('learning_rate', 0.0025, 0.05),
    'n_estimators': 6000,
    'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
    'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
    'max_depth': trial.suggest_int('max_depth', 3, 15),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
    'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
    'device_type': 'cpu',
    'num_leaves': trial.suggest_int('num_leaves', 4, 2048),
    "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    "verbosity": -1,
    "early_stopping_rounds": 50,
    }

    auc_scores = []

    for train_idx, valid_idx in cv.split(X, y):

        X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

        # Create and fit the model
        model = LGBMClassifier(**param)
        model.fit(X_train_fold, y_train_fold, eval_set=[(X_valid_fold, y_valid_fold)])

        # Predict class probabilities
        y_prob = model.predict_proba(X_valid_fold)

        # Compute the AUC for each class and take the average
        average_auc = roc_auc_score(targets_bin.iloc[valid_idx], y_prob[:, 1:], multi_class="ovr", average="macro")
        auc_scores.append(average_auc)

    # Return the average AUC score across all folds
    return np.mean(auc_scores)

# Run Optuna optimization
if RETRAIN_LGBM_MODEL:
    #lgbm_study = optuna.create_study(direction='maximize',study_name = "lgbm_model_training", storage="sqlite:///db.sqlite3")
    lgbm_study.optimize(objective, n_trials=500)  # Adjust the number of trials as necessary
    # Output the optimization results
    print(f"Best trial average AUC: {study.best_value:.4f}")
    print(study.best_params)
    for key, value in study.best_params.items():
        print(f"{key}: {value}")


RETRAIN_CATBOOST_MODEL = False

def objective(trial):
    # Define parameters to be optimized for the CatBoostClassifier
    param = {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1),
        'n_estimators': 2000,
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.001, 10.0, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
#         "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.99),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "bootstrap_type": "Bernoulli",
        "early_stopping_rounds": 100,
        "task_type": 'CPU',
        "verbose": False
    }

    param['task_type'] = "GPU"
    param['devices'] = '0'
    auc_scores = []

    for train_idx, valid_idx in cv.split(X, y):
        X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

        # Create and fit the model
        model = CatBoostClassifier(**param)
        model.fit(X_train_fold, y_train_fold, eval_set=[(X_valid_fold, y_valid_fold)])

        # Predict class probabilities
        y_prob = model.predict_proba(X_valid_fold)

        # Compute the AUC for each class and take the average
        average_auc = roc_auc_score(targets_bin.iloc[valid_idx], y_prob[:, 1:], multi_class="ovr", average="macro")
        auc_scores.append(average_auc)

    # Return the average AUC score across all folds
    return np.mean(auc_scores)
if RETRAIN_CATBOOST_MODEL:
    # Run Optuna optimization
    catboost_study = optuna.create_study(direction='maximize', study_name="catboost_model_training")
    catboost_study.optimize(objective, n_trials=150)  # Adjust the number of trials as necessary
    # Output the optimization results
    print(f"Best trial average AUC: {study.best_value:.4f}")
    print(study.best_params)
    for key, value in study.best_params.items():
        print(f"{key}: {value}")


RETRAIN_HGBC_MODEL = False

def objective(trial):
    # Define hyperparameters to tune
    param = {
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
        'max_iter': trial.suggest_int('max_iter', 100, 2500),  # Equivalent to n_estimators
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'l2_regularization': trial.suggest_float('l2_regularization', 1e-8, 10.0, log=True),  # Equivalent to reg_lambda
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20, 300),
        'max_bins': trial.suggest_int('max_bins', 25, 255),
    }
    
    auc_scores = []

    for train_idx, valid_idx in cv.split(X, y):
        X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

        # Create and fit the model
        model = HistGradientBoostingClassifier(**param)
        model.fit(X_train_fold, y_train_fold)

        # Predict class probabilities
        y_prob = model.predict_proba(X_valid_fold)

        # Compute the AUC for each class and take the average
        average_auc = roc_auc_score(targets_bin.iloc[valid_idx], y_prob[:, 1:], multi_class="ovr", average="macro")
        auc_scores.append(average_auc)

    # Return the average AUC score across all folds
    return np.mean(auc_scores)


if RETRAIN_HGBC_MODEL:
    # Example usage with Optuna
    hgbc_study = optuna.create_study(direction='maximize', study_name="HistGradientBoostingClassifier_model_training")
    hgbc_study.optimize(objective, n_trials=100)  # Adjust the number of trials as necessary

    # Output the optimization results
    print(f"Best trial average AUC: {study.best_value:.4f}")
    print(hgbc_study.best_params)
    for key, value in study.best_params.items():
        print(f"{key}: {value}")


lgbm_params = {
    "objective": "multiclass",
    'num_class': 8,
    "boosting_type": "gbdt",
    "verbosity": -1,
    "early_stopping_rounds": 50,
    'n_estimators': 3000,
    'learning_rate': 0.008508269501456054,
    'lambda_l1': 7.365461205850494e-08,
    'lambda_l2': 4.2565699722054004e-05,
    'max_depth': 5,
    'colsample_bytree': 0.3001271251186239,
    'subsample': 0.5392120673070296,
    'min_child_weight': 4,
    'num_leaves': 394,
    'min_child_samples': 96,
}

cat_params = {
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "bootstrap_type": "Bernoulli",
    "early_stopping_rounds": 100,
    "verbose": False,
    'n_estimators': 2000,
    'learning_rate': 0.030297706218000695,
    'l2_leaf_reg': 0.532839791504174,
    'depth': 5,
    'colsample_bylevel': 0.1250483657354198,
    'subsample': 0.838540736360766
}

xgb_params = {
    'grow_policy': 'depthwise',
    'multi_strategy': 'one_output_per_tree',
    'n_estimators': 3000,
    'early_stopping_rounds': 50,
    'learning_rate': 0.005063786261534407,
    'gamma': 0.03360259160829512,
    'subsample': 0.9006931840971558,
    'colsample_bytree': 0.3005895848895487,
    'max_depth': 8,
    'min_child_weight': 6,
    'lambda': 0.46049449681503624,
    'alpha': 0.002529132375391616,
}

hgbc_params = {
    'learning_rate': 0.06212471625814388,
    'max_iter': 1387,
    'max_depth': 4,
    'l2_regularization': 4.393355351649873e-06,
    'min_samples_leaf': 175,
    'max_bins': 108
}


def compute_oof(X, y, model, model_name, cv):
    
    cv_oof = pd.DataFrame(np.zeros((len(y), len(TARGET_FEATURES))), columns=TARGET_FEATURES)
    prob_predictions_test = []
    for train_idx, valid_idx in cv.split(X, y):
        X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]
        y_xgb_train_fold, y_xgb_valid_fold = y_xgb.iloc[train_idx], y_xgb.iloc[valid_idx]

        prob_predictions = []
        if model_name == 'XGB':
            model.fit(X_train_fold, y_xgb_train_fold, eval_set=[(X_valid_fold, y_xgb_valid_fold)],verbose=False)
            cv_oof.iloc[valid_idx, :] = model.predict_proba(X_valid_fold)
            prob_predictions_test.append(model.predict_proba(test_data))
        elif model_name == 'LGBM':
            model.fit(X_train_fold, y_train_fold, eval_set=[(X_valid_fold, y_valid_fold)])
            cv_oof.iloc[valid_idx, :] = model.predict_proba(X_valid_fold)[:, 1:]
            prob_predictions_test.append(model.predict_proba(test_data)[:, 1:])
        elif model_name == 'CAT':
            model.fit(X_train_fold, y_train_fold, eval_set=[(X_valid_fold, y_valid_fold)])
            cv_oof.iloc[valid_idx, :] = model.predict_proba(X_valid_fold)[:, 1:]
            prob_predictions_test.append(model.predict_proba(test_data)[:, 1:])
        else:
            model.fit(X_train_fold, y_train_fold)
            cv_oof.iloc[valid_idx, :] = model.predict_proba(X_valid_fold)[:, 1:]
            prob_predictions_test.append(model.predict_proba(test_data)[:, 1:])
    preds = np.mean(prob_predictions_test, axis=0)
    return cv_oof, preds


cat_oof, cat_preds = compute_oof(X, y, CatBoostClassifier(**cat_params), 'CAT', cv)


xgb_oof, xgb_preds = compute_oof(X, y, XGBClassifier(**xgb_params), 'XGB', cv)
lgbm_oof, lgbm_preds = compute_oof(X, y, LGBMClassifier(**lgbm_params), 'LGBM', cv)


hgbc_oof, hgbc_preds = compute_oof(X, y, HistGradientBoostingClassifier(**hgbc_params), 'HGBC', cv)


MinMax = 0
# 0 None
# 1 MinMax min to 0 max to 1
# 2 scale a row sum to 1
if MinMax == 1:
    xgb_oof = MinMaxScaler().fit_transform(xgb_oof)
    lgbm_oof = MinMaxScaler().fit_transform(lgbm_oof)
    cat_oof = MinMaxScaler().fit_transform(cat_oof)
    hgbc_oof = MinMaxScaler().fit_transform(hgbc_oof)
    xgb_preds = MinMaxScaler().fit_transform(xgb_preds)
    lgbm_preds = MinMaxScaler().fit_transform(lgbm_preds)
    cat_preds = MinMaxScaler().fit_transform(cat_preds)
    hgbc_preds = MinMaxScaler().fit_transform(hgbc_preds)
elif MinMax == 2:
    xgb_oof = xgb_oof.div(xgb_oof.sum(axis=1), axis=0)
    lgbm_oof = lgbm_oof.div(lgbm_oof.sum(axis=1), axis=0)
    cat_oof = cat_oof.div(cat_oof.sum(axis=1), axis=0)
    hgbc_oof = hgbc_oof.div(hgbc_oof.sum(axis=1), axis=0)
    xgb_oof = np.array(xgb_oof)
    lgbm_oof = np.array(lgbm_oof)
    cat_oof = np.array(cat_oof)
    hgbc_oof = np.array(hgbc_oof)
    xgb_preds = xgb_preds / xgb_preds.sum(axis=1, keepdims=True)
    lgbm_preds = lgbm_preds / lgbm_preds.sum(axis=1, keepdims=True)
    cat_preds = cat_preds / cat_preds.sum(axis=1, keepdims=True)
    hgbc_preds = hgbc_preds / hgbc_preds.sum(axis=1, keepdims=True)
else:
    xgb_oof = np.array(xgb_oof)
    lgbm_oof = np.array(lgbm_oof)
    cat_oof = np.array(cat_oof)
    hgbc_oof = np.array(hgbc_oof)
    xgb_preds = np.array(xgb_preds)
    lgbm_preds = np.array(lgbm_preds)
    cat_preds = np.array(cat_preds)
    hgbc_preds = np.array(hgbc_preds)


xgb_score = roc_auc_score(targets_bin, xgb_oof, multi_class='ovr')
lgbm_score = roc_auc_score(targets_bin, lgbm_oof, multi_class='ovr')
cat_score = roc_auc_score(targets_bin, cat_oof, multi_class='ovr')
hgbc_score = roc_auc_score(targets_bin, hgbc_oof, multi_class='ovr')

print(f"oof roc-auc score for XGB  model: {xgb_score:0.5f}")
print(f"oof roc-auc score for LGBM model: {lgbm_score:0.5f}")
print(f"oof roc-auc score for CAT  model: {cat_score:0.5f}")
print(f"oof roc-auc score for HGBC model: {hgbc_score:0.5f}")


from functools import partial
from scipy.optimize import minimize

blend = np.zeros((xgb_oof.shape[0], xgb_oof.shape[1]))
preds = np.zeros((xgb_preds.shape[0], xgb_preds.shape[1]))

initial_weights = np.array([0.25, 0.25, 0.25, 0.25])

def calculate_roc_auc(weights, oof_1, oof_2, oof_3, oof_4, target):
    # Normalize weights
    weights /= np.sum(weights)
    
    # Calculate weighted sum
    weighted_sum = oof_1 * weights[0] + oof_2 * weights[1] + oof_3 * weights[2] + oof_4 * weights[3]
    
    # Calculate ROC AUC score
    score = roc_auc_score(target, weighted_sum, multi_class='ovr')
    
    # Return negative score to maximize during optimization
    return -score

# Define bounds for each weight (greater than or equal to 0)
bounds = [(0, None), (0, None), (0, None), (0, None)]

for k in range(len(TARGET_FEATURES)):
    result = minimize(partial(calculate_roc_auc, 
                              oof_1=xgb_oof[:, k],
                              oof_2=lgbm_oof[:, k],
                              oof_3=cat_oof[:, k],
                              oof_4=hgbc_oof[:, k],
                              target=targets_bin.iloc[:, k]), 
                      initial_weights, 
                      method='Nelder-Mead',
                      bounds=bounds)
    
    optimal_weights = result.x / np.sum(result.x)
    
    # Update print statement and calculation for four models
    print(f"Class {TARGET_FEATURES[k]} optimal weights: XGB={optimal_weights[0]:0.3f}, LGBM={optimal_weights[1]:0.3f}, CAT={optimal_weights[2]:0.3f}, HGBC={optimal_weights[3]:0.3f}")
    blend[:, k] = (xgb_oof[:, k] * optimal_weights[0] + lgbm_oof[:, k] * optimal_weights[1] +
                   cat_oof[:, k] * optimal_weights[2] + hgbc_oof[:, k] * optimal_weights[3])
    preds[:, k] = (xgb_preds[:, k] * optimal_weights[0] + lgbm_preds[:, k] * optimal_weights[1] +
                   cat_preds[:, k] * optimal_weights[2] + hgbc_preds[:, k] * optimal_weights[3])


blend_score = roc_auc_score(targets_bin, blend, multi_class='ovr')
print(f"oof roc-auc score for blend of models: {blend_score:0.5f}")


submission = pd.read_csv('/kaggle/input/playground-series-s4e3/sample_submission.csv')
submission.iloc[:, 1:] = preds


submission.to_csv("submission.csv", index=False)


submission

