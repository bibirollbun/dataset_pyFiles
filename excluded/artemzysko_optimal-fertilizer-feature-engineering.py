import os

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from optuna.exceptions import ExperimentalWarning
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ExperimentalWarning)
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")


train_data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_data = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
original_data = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
submission_data = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


print("train_data shape :",train_data.shape)
print("test_data shape :",test_data.shape)
print("original_data shape :",original_data.shape)
print("submission_data shape :",submission_data.shape)


train_data = train_data.drop("id", axis=1)
test_data = test_data.drop("id", axis=1)
# cheat: merge original and competition dataset :D
train_data_merged = pd.concat([train_data, original_data], ignore_index=True)
train_data_merged = train_data_merged.drop_duplicates()
print("shape of the data :",train_data_merged.shape)

test_data_copy = test_data.copy()
train_data_merged_copy = train_data_merged.copy()


train_data_merged.columns


# # ==========================
# # 1. Interaction & Derived Features
# # ==========================
# for df in [train_data, test_data]:
#     df['N_to_P'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-5)
#     df['N_to_K'] = df['Nitrogen'] / (df['Potassium'] + 1e-5)
#     df['P_to_K'] = df['Phosphorous'] / (df['Potassium'] + 1e-5)
#     df['Total_NPK'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
#     df['Climate_Index'] = (df['Temparature'] + df['Humidity']) / 2
#     df['Water_Stress'] = df['Humidity'] - df['Moisture']

# # ==========================
# # 2. Group-Based Features (on already-encoded 'Crop Type')
# # ==========================
# crop_group_means = train_data.groupby('Crop Type')[['Nitrogen', 'Phosphorous', 'Potassium']].mean()

# for df in [train_data, test_data]:
#     df['Crop_N_mean'] = df['Crop Type'].map(crop_group_means['Nitrogen'])
#     df['Crop_P_mean'] = df['Crop Type'].map(crop_group_means['Phosphorous'])
#     df['Crop_K_mean'] = df['Crop Type'].map(crop_group_means['Potassium'])

# return train_data, test_data


cat_cols = test_data_copy.select_dtypes(exclude='number').columns.tolist()
feature_les = {col: LabelEncoder() for col in cat_cols}
target_le = LabelEncoder()

# fit cat featuers encoders
for col in cat_cols:
    feature_les[col].fit(train_data_merged_copy[col])

target_le.fit(train_data_merged_copy['Fertilizer Name'])


soil_group_means = train_data_merged_copy\
    .groupby('Soil Type')[['Moisture', 'Humidity', 'Nitrogen', 'Potassium', 'Phosphorous']]\
    .mean()
soil_group_means


crop_group_means = train_data_merged_copy\
    .groupby('Crop Type')[['Moisture', 'Humidity', 'Nitrogen', 'Potassium', 'Phosphorous']]\
    .mean()
crop_group_means


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

def label_encoding(df):
    df = df.copy()
    for col in cat_cols:
        df[col] = feature_les[col].transform(df[col])
    return df

def manual_feature_engineering(df):
    df = df.copy()
    # df = df.drop('id', axis=1)
    # df["NPK_sum"] = df["Nitrogen"] + df["Phosphorous"] + df["Potassium"]
    # df["Nit_to_Phos_ratio"] = df["Nitrogen"] / (df["Phosphorous"] + 1e-5)
    # df["Nit_to_Pot_ratio"] = df["Nitrogen"] / (df["Potassium"] + 1e-5)
    # df["Pot_to_Phos_ratio"] = df["Potassium"] / (df["Phosphorous"] + 1e-5)
    df['Climate_Index'] = (df['Temparature'] + df['Humidity']) / 2
    df['Water_Stress'] = df['Humidity'] - df['Moisture']
    # df['Crop_N_mean'] = df['Crop Type'].map(crop_group_means['Nitrogen'])
    # df['Crop_P_mean'] = df['Crop Type'].map(crop_group_means['Phosphorous'])
    # df['Crop_K_mean'] = df['Crop Type'].map(crop_group_means['Potassium'])

    df['Crop_N_diff'] = df['Nitrogen'] - df['Crop Type'].map(crop_group_means['Nitrogen'])
    df['Crop_P_diff'] = df['Phosphorous'] - df['Crop Type'].map(crop_group_means['Phosphorous'])
    df['Crop_K_diff'] = df['Potassium'] - df['Crop Type'].map(crop_group_means['Potassium'])
    
    # df['Soil_N_mean'] = df['Soil Type'].map(soil_group_means['Nitrogen'])
    # df['Soil_P_mean'] = df['Soil Type'].map(soil_group_means['Phosphorous'])
    # df['Soil_K_mean'] = df['Soil Type'].map(soil_group_means['Potassium'])
    
    return df


preprocessing_pipeline = Pipeline([
    ('manual_features', FunctionTransformer(manual_feature_engineering)),
    ('label_encoders', FunctionTransformer(label_encoding)),
])


preprocessing_pipeline.transform(train_data_merged_copy).head()


train_preprocessed = preprocessing_pipeline.transform(train_data_merged_copy)

X = train_preprocessed.drop('Fertilizer Name', axis=1).to_numpy()
y = target_le.transform(train_preprocessed['Fertilizer Name'])

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(X.shape, y.shape)
print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)


print(target_le.classes_, len(target_le.classes_))
y[:25]


def map3_score(predicted_top3: np.ndarray,   # shape = (n_val, 3), dtype = object or int
               y_true_fold: np.ndarray,      # shape = (n_val,)
              ) -> float:
    # print(type(predicted_top3), type(y_true_fold))
    
    n_val = y_true_fold.shape[0]
    total_score = 0.0

    for i in range(n_val):
        true_label = y_true_fold[i]
        top3_preds = predicted_top3[i].tolist()  # convert row to a Python list

        try:
            # .index(...) returns 0-based position. Add +1 to get 1-based rank.
            rank = top3_preds.index(true_label) + 1
            if rank <= 3:
                total_score += 1.0 / rank
            # If rank > 3, that cannot happen here, because top3_preds has exactly 3 items.
        except ValueError:
            # true_label not in top-3  score += 0
            pass

    return total_score / n_val


import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score


def xgboost_objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 40),
        'min_child_weight': trial.suggest_float('min_child_weight', 0, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'subsample': trial.suggest_float('subsample', 0, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0, 1),
        'eta': trial.suggest_float('eta', 0, 1),
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0)
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    map3_scores = []

    for train_index, val_index in cv.split(X_train, y_train):
        x_train_fold = X_train[train_index]
        x_val_fold = X_train[val_index]
        y_train_fold = y_train[train_index]
        y_val_fold = y_train[val_index]

        # x_train_fold = X_train.iloc[train_index]
        # x_val_fold = X_train.iloc[val_index]
        
        # y_train_fold = y_train.iloc[train_index]
        # y_val_fold = y_train.iloc[val_index]
        
        model = XGBClassifier(
            **params,
            verbosity=0,
            objective='multi:softprob',
            enable_categorical=True,
            tree_method="gpu_hist",
            gpu_id=0, 
            predictor="gpu_predictor",
            n_jobs=-1,
            random_seed=42
        )
        
        model.fit(x_train_fold, y_train_fold, eval_set=[(x_val_fold, y_val_fold)],
              early_stopping_rounds=50, verbose=False)

        pred_proba = model.predict_proba(x_val_fold)
        top3_index = np.argsort(pred_proba, axis=1)[:, -3:][:, ::-1]
        class_labs = model.classes_
        top3_labs = class_labs[top3_index]

        # fold_map3 = map3_score(top3_labs, y_val_fold.to_numpy())
        fold_map3 = map3_score(top3_labs, y_val_fold)
        map3_scores.append(fold_map3)
        mean_map3 = np.mean(map3_scores)

    return mean_map3


# study = optuna.create_study(direction="maximize", sampler=TPESampler(n_startup_trials=30, seed=42, multivariate=True))
# study.optimize(xgboost_objective, n_trials=50, n_jobs=1)
# print("Best trial:")
# print(study.best_trial.params)


# XGBoost best params (25 trials)
# Best trial:
# xgb_params = {'learning_rate': 0.01596950334578271, 'max_depth': 20, 
#  'min_child_weight': 8.324426408004218, 'gamma': 1.0616955533913808, 
#  'alpha': 0.005337032762603957, 'subsample': 0.18340450985343382, 
#  'colsample_bytree': 0.3042422429595377, 'eta': 0.5247564316322378, 
#  'n_estimators': 489, 'lambda': 0.014618962793704957}

# Best trial: (climate index, water stress)
# {'learning_rate': 0.01596950334578271, 'max_depth': 20, 'min_child_weight': 8.324426408004218, 'gamma': 1.0616955533913808, 'alpha': 0.005337032762603957, 'subsample': 0.18340450985343382, 'colsample_bytree': 0.3042422429595377, 'eta': 0.5247564316322378, 'n_estimators': 489, 'lambda': 0.014618962793704957}


# 0.32671568575008797 (climate index, water stress, crop-groups-pnk)
# Best trial:
# {'learning_rate': 0.01596950334578271, 'max_depth': 20, 'min_child_weight': 8.324426408004218, 'gamma': 1.0616955533913808, 'alpha': 0.005337032762603957, 'subsample': 0.18340450985343382, 'colsample_bytree': 0.3042422429595377, 'eta': 0.5247564316322378, 'n_estimators': 489, 'lambda': 0.014618962793704957}





def lgbm_objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_child_weight': trial.suggest_float('min_child_weight', 0, 10),
        'min_split_gain': trial.suggest_float('min_split_gain', 0, 5),  # instead of gamma
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000)
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    map3_scores = []

    for train_index, val_index in cv.split(X_train, y_train):
        x_train_fold = X_train[train_index]
        x_val_fold = X_train[val_index]
        y_train_fold = y_train[train_index]
        y_val_fold = y_train[val_index]

        model = LGBMClassifier(
            **params,
            objective='multiclass',
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
            device='gpu'
        )

        model.fit(x_train_fold, y_train_fold,
                  eval_set=[(x_val_fold, y_val_fold)])

        pred_proba = model.predict_proba(x_val_fold)
        top3_index = np.argsort(pred_proba, axis=1)[:, -3:][:, ::-1]
        class_labs = model.classes_
        top3_labs = class_labs[top3_index]

        fold_map3 = map3_score(top3_labs, y_val_fold)
        map3_scores.append(fold_map3)

    return np.mean(map3_scores)


# study_lgbm = optuna.create_study(direction="maximize", sampler=TPESampler(n_startup_trials=30, seed=42, multivariate=True))
# study_lgbm.optimize(lgbm_objective, n_trials=25, n_jobs=1)
# print("Best trial:")
# print(study_lgbm.best_trial.params)


# LGBM best params
# Best trial:
# {'learning_rate': 0.267255063036884, 'max_depth': 17, 'min_child_weight': 6.420316461542877, 'min_split_gain': 0.42069982497524416, 'reg_alpha': 0.004431133722857619, 'reg_lambda': 3.928409513479229, 'subsample': 0.803214529829795, 'colsample_bytree': 0.5045985258083148, 'n_estimators': 191}



test_data_preprocessed_np = preprocessing_pipeline.transform(test_data_copy).to_numpy()


xgb_params = {'learning_rate': 0.01596950334578271, 'max_depth': 20, 
 'min_child_weight': 8.324426408004218, 'gamma': 1.0616955533913808, 
 'alpha': 0.005337032762603957, 'subsample': 0.18340450985343382, 
 'colsample_bytree': 0.3042422429595377, 'eta': 0.5247564316322378, 
 'n_estimators': 489, 'lambda': 0.014618962793704957}

xgb_classifier = XGBClassifier(
    **xgb_params,
    verbosity=0,
    objective='multi:softprob',
    enable_categorical=True,
    tree_method="gpu_hist",
    gpu_id=0, 
    predictor="gpu_predictor",
    n_jobs=-1,
    random_seed=42
)

xgb_classifier.fit(X, y, eval_set=[(X_valid, y_valid)],
              early_stopping_rounds=50, verbose=False)


pred_proba = xgb_classifier.predict_proba(test_data_preprocessed_np)
top3_index = np.argsort(pred_proba, axis=1)[:, -3:][:, ::-1]
class_labs = xgb_classifier.classes_
top3_labs = class_labs[top3_index]


top3_result_strings = np.array(list(map(
    lambda x: ' '.join(target_le.inverse_transform(x)), top3_labs)))


top3_result_strings.shape, test_data.shape, submission_data.shape


submission = pd.DataFrame({
    'id': submission_data['id'].values,
    'Fertilizer Name': top3_result_strings
})
submission.to_csv('/kaggle/working/submission.csv',index=False)
submission




