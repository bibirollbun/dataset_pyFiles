!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


!cp /kaggle/input/metric/metric.py ./
from metric import score
import numpy as np, pandas as pd
from scipy.stats import rankdata 
from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import quantile_transform
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
import catboost as cb
from lightgbm import LGBMRegressor
import lightgbm as lgb
import os
import pickle
import joblib
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")



test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")


plt.hist(train.loc[train.efs==1,"efs_time"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"efs_time"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Time of Observation, efs_time")
plt.ylabel("Density")
plt.title("Times of Observation. Either time to event, or time observed without event.")
plt.legend()
plt.show()


# def update_target_with_probabilities(df, probability_func, target_name, time_col='efs_time', event_col='efs'):
#     df[target_name] = 0.0
#     race_groups = df['race_group'].unique()
#     probs_dict = {}
#     for race in race_groups:
#         race_df = df[df['race_group'] == race]
#         probs_dict[race] = probability_func(race_df,time_col, event_col)
#     for race in race_groups:
#         # Assign probabilities to the target column
#         df.loc[df['race_group'] == race, target_name] = probs_dict[race]
#     return df[target_name]

# def KaplanMeier(in_data, time_col='efs_time', event_col='efs'):
#     kmf = KaplanMeierFitter()
#     kmf.fit(durations=in_data[time_col], event_observed=in_data[event_col])
#     return kmf.survival_function_at_times(in_data[time_col]).values.flatten()

# train['y'] = update_target_with_probabilities(train, KaplanMeier, target_name='y')
# train.loc[train.efs==0,"y"]-=0.15


# def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
#     kmf = KaplanMeierFitter()
#     kmf.fit(df[time_col], df[event_col])
#     y = kmf.survival_function_at_times(df[time_col]).values
#     return y
# train["y"] = transform_survival_probability(train, time_col='efs_time', event_col='efs')

def transform_quantile(time, event):
    transformed = np.full(len(time), np.nan)
    transformed_dead = quantile_transform(- time[event == 1].values.reshape(-1, 1)).ravel()
    transformed_censored = quantile_transform(- time[event == 0].values.reshape(-1, 1)).ravel()
    transformed[event == 1] = transformed_dead
    transformed[event == 0] = transformed_censored*0.01 - 0.3
    return transformed
# train["y"] = transform_quantile(time=train.efs_time, event=train.efs)

train['y'] = 0.0
race_groups = train['race_group'].unique()
probs_dict = {}
for race in race_groups:
    race_df = train[train['race_group'] == race]
    probs_dict[race] = transform_quantile(time=race_df.efs_time, event=race_df.efs)
for race in race_groups:
    train.loc[train['race_group'] == race,'y'] = probs_dict[race]



# train["y"] = train.efs_time.values
# mx = train.loc[train.efs==1,"efs_time"].max()
# mn = train.loc[train.efs==0,"efs_time"].min()
# train.loc[train.efs==0,"y"] = train.loc[train.efs==0,"y"] + mx - mn
# train.y = train.y / train.y.max()
# train.y = -np.log( train.y )


# asian_df=train[train['race_group'] == "White"]


# plt.hist(asian_df.loc[asian_df.efs==1,"y"],bins=100,label="efs=1, Yes Event")
# plt.hist(asian_df.loc[asian_df.efs==0,"y"],bins=100,label="efs=0, Maybe Event")
# # plt.xlim((0,0.01))
# plt.xlabel("Transformed Target y")
# plt.ylabel("Density")
# plt.title("Transformed Target y using both efs and efs_time.")
# plt.legend()
# plt.show()


plt.hist(train.loc[train.efs==1,"y"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"y"],bins=100,label="efs=0, Maybe Event")
# plt.xlim((0,0.01))
plt.xlabel("Transformed Target y")
plt.ylabel("Density")
plt.title("Transformed Target y using both efs and efs_time.")
plt.legend()
plt.show()


RMV = ["ID","efs","efs_time","y"]
FEATURES = [c for c in train.columns if not c in RMV]

CATS = []
for c in FEATURES:
    num_unique = train[c].nunique()
    if num_unique < 100:
        CATS.append(c)
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")
combined = pd.concat([train,test],axis=0,ignore_index=True)

encoders = {}
for c in FEATURES:
    if c in CATS:
        encoders[c] = LabelEncoder()
        combined[c] = encoders[c].fit_transform(combined[c].astype(str))
        combined[c] = combined[c].astype("int32")
        combined[c] = combined[c].astype("category")
    else:
        if combined[c].dtype=="float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype=="int64":
            combined[c] = combined[c].astype("int32")
    
train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()


with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)


FOLDS = 10
skf = StratifiedKFold(n_splits=10,shuffle=True, random_state=1)

oof_xgb = np.zeros(len(train))
pred_xgb = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(skf.split(train, train.race_group)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"y"]
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"y"]
    x_test = test[FEATURES].copy()

    model_xgb = XGBRegressor(
        device="cuda",
        max_depth=3,  
        colsample_bytree=0.5,  
        subsample=0.8,  
        n_estimators=20000,  
        learning_rate=0.01,  
        enable_categorical=True,
        min_child_weight=80,
        early_stopping_rounds=500,
    )
    
    model_xgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],  
        verbose=1000 
    )

    # INFER OOF
    oof_xgb[test_index] = model_xgb.predict(x_valid)
    # INFER TEST
    pred_xgb += model_xgb.predict(x_test)

    model_path = os.path.join('/kaggle/working/', f"xgb_fold_{i+1}.pkl")
    with open(model_path, "wb") as file:
        pickle.dump(model_xgb, file)
        print(f"Model for Fold {i+1} saved at {model_path}")

# COMPUTE AVERAGE TEST PREDS
pred_xgb /= FOLDS




# def evaluate_fold(y_va_pred,X_va,idx_va,fold):
#     """Compute and print the metrics (concordance index) per race group for a single fold.

#     Global variables:
#     - train, X_va, idx_va
#     - The metrics are saved in the global list all_scores.
#     """
#     race_groups = np.unique(train.race_group)
#     metric_list = []
#     for race in race_groups:
#         mask = X_va.race_group.values == race
#         c_index_race = concordance_index(
#             train.efs_time.iloc[idx_va][mask],
#             - y_va_pred[mask],
#             train.efs.iloc[idx_va][mask]
#         )
#         # print(f"# {race:42} {c_index_race:.3f}")
#         metric_list.append(c_index_race)
#     fold_score = np.mean(metric_list) - np.sqrt(np.var(metric_list))
#     print(f"# Total fold {fold}:{' ':29} {fold_score:.3f} mean={np.mean(metric_list):.3f} std={np.std(metric_list):.3f}")
#     return fold_score


# def objective(trial, X_train, y_train,X_val,y_val):
#     params = {
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'learning_rate': trial.suggest_categorical(
#             'learning_rate', [round(0.001 * i, 3) for i in range(1, 101)]),
#         'min_child_weight': trial.suggest_categorical('min_child_weight', [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200]),
#         'subsample': trial.suggest_categorical('subsample', [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]),
#         'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]),
#         'n_estimators': 20000,
#         'early_stopping_rounds': 500
#     }
    
#     model = xgb.XGBRegressor(**params, device='cuda', enable_categorical=True)
    
#     # Fit with validation set for early stopping
#     model.fit(
#         X_train, y_train, 
#         eval_set=[(X_val, y_val)],
#         verbose=0
#     )
    
#     # Return validation score
#     return -mean_squared_error(y_val, model.predict(X_val))

# X = train[FEATURES]
# y = train['y']
# groups = train['race_group']

# # Initialize result containers
# oof_predictions = np.zeros(len(train))
# test_predictions = np.zeros(len(test))
# fold_scores = []

# # Outer cross-validation
# outer_cv = StratifiedKFold(n_splits=10,shuffle=True, random_state=1)

# for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X,groups)):
#     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
#     groups_train = groups.iloc[train_idx]
    
#     # Create an Optuna study
#     study = optuna.create_study(direction='maximize')
#     study.optimize(
#         lambda trial: objective(trial, X_train, y_train,X_val,y_val), 
#         n_trials=50
#     )
    
#     # Best parameters from Optuna
#     best_params = study.best_params
#     print(f"Fold {fold} best params:{best_params}")
#     best_params.update({
#         'device': 'cuda', 
#         'enable_categorical': True,
#         'n_estimators': 20000,
#         'early_stopping_rounds': 500
#     })
    
#     # Train final model with best params
#     model = xgb.XGBRegressor(**best_params)
#     model.fit(
#         X_train, y_train, 
#         eval_set=[(X_val, y_val)],
#         verbose=1000
#     )

#     model_path = f'xgb_fold_{fold+1}.joblib'
#     joblib.dump(model, model_path)
#     print(f"Saved model for Fold {fold+1} to {model_path}")
    
#     # Predict on validation set
#     val_preds = model.predict(X_val)
#     oof_predictions[val_idx] = val_preds
    
#     # Evaluate fold
#     fold_score = evaluate_fold(val_preds, X_val, val_idx,fold)
#     fold_scores.append(fold_score)
    
#     # Predict on test set
#     test_predictions += model.predict(test[FEATURES]) / outer_cv.n_splits

# # Print results
# print("Fold Scores:", fold_scores)
# print("Mean Score:", np.mean(fold_scores))
# print("Score Standard Deviation:", np.std(fold_scores))


y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_xgb
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost KaplanMeier =",m)


np.save('oof_xgb.npy', oof_xgb)


oof_xgb = np.load('/kaggle/input/684-oof/oof_pred_684.npy')


test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")

test['L1_preds'] = pred_xgb
train['L1_preds'] = oof_xgb

def transform_quantile(time, event):
    transformed = np.full(len(time), np.nan)
    transformed_dead = quantile_transform(- time[event == 1].values.reshape(-1, 1)).ravel()
    transformed_censored = quantile_transform(- time[event == 0].values.reshape(-1, 1)).ravel()
    transformed[event == 1] = transformed_dead
    transformed[event == 0] = transformed_censored*0.01 - 0.3
    return transformed
# train["y"] = transform_quantile(time=train.efs_time, event=train.efs)

train['y'] = 0.0
race_groups = train['race_group'].unique()
probs_dict = {}
for race in race_groups:
    race_df = train[train['race_group'] == race]
    probs_dict[race] = transform_quantile(time=race_df.efs_time, event=race_df.efs)
for race in race_groups:
    train.loc[train['race_group'] == race,'y'] = probs_dict[race]

RMV = ["ID","efs","efs_time","y"]
FEATURES = [c for c in train.columns if not c in RMV]

CATS = []
for c in FEATURES:
    num_unique = train[c].nunique()
    if num_unique < 100:
        CATS.append(c)
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")
combined = pd.concat([train,test],axis=0,ignore_index=True)

for c in FEATURES:
    if c in CATS:
        combined[c] = encoders[c].transform(combined[c].astype(str))
        combined[c] = combined[c].astype("int32")
        combined[c] = combined[c].astype("category")
    else:
        if combined[c].dtype=="float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype=="int64":
            combined[c] = combined[c].astype("int32")
    
train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()

FOLDS = 10
skf = StratifiedKFold(n_splits=10,shuffle=True, random_state=1)

oof_xgb2 = np.zeros(len(train))
pred_xgb2 = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(skf.split(train, train.race_group)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"y"]
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"y"]
    x_test = test[FEATURES].copy()

    model_xgb = XGBRegressor(
        device="cuda",
        max_depth=2,  
        colsample_bytree=0.5,  
        subsample=0.8,  
        n_estimators=10000,  
        learning_rate=0.005,  
        enable_categorical=True,
        min_child_weight=64,
        early_stopping_rounds=500
    )
    
    model_xgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],  
        verbose=1000
    )

    # INFER OOF
    oof_xgb2[test_index] = model_xgb.predict(x_valid)
    # INFER TEST
    pred_xgb2 += model_xgb.predict(x_test)

    model_path = os.path.join('/kaggle/working/', f"xgbl2_fold_{i+1}.pkl")
    with open(model_path, "wb") as file:
        pickle.dump(model_xgb, file)
        print(f"Model for Fold {i+1} saved at {model_path}")

# COMPUTE AVERAGE TEST PREDS
pred_xgb2 /= FOLDS





y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_xgb2
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost KaplanMeier =",m)

np.save('oof_xgb2.npy', oof_xgb2)


sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
sub.prediction =pred_xgb2
sub.to_csv("submission.csv",index=False)



import matplotlib.pyplot as plt
import numpy as np

# Get feature importance from the last trained model
feature_importance = model_xgb.get_booster().get_score(importance_type="weight")

# Convert to sorted list
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

# Extract names and values
feature_names = [x[0] for x in sorted_features]
importance_values = [x[1] for x in sorted_features]

# Plot
plt.figure(figsize=(12, 6))
plt.barh(feature_names[::-1], importance_values[::-1], color='blue')
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importances (XGBoost)")
plt.show()


