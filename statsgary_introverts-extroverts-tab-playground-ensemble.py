!pip install modelviz --quiet


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
import optuna
import warnings
import modelviz
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


print("Train shape:", train.shape)
print("Test shape:", test.shape)


print(train.head())
print(train.isnull().sum())


ID_COL = 'id'
TARGET_COL = 'Personality'
NUMERICAL_COLS = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                  'Friends_circle_size', 'Post_frequency']
CATEGORICAL_COLS = ['Stage_fear', 'Drained_after_socializing']


RANDOM_STATE = 42
N_SPLITS = 3
N_TRIALS = 30


for col in NUMERICAL_COLS:
    train[col].fillna(train[col].median(), inplace=True)
    test[col].fillna(test[col].median(), inplace=True)


for col in CATEGORICAL_COLS:
    train[col].fillna(train[col].mode()[0], inplace=True)
    test[col].fillna(test[col].mode()[0], inplace=True)


X = train.drop([TARGET_COL, ID_COL], axis=1)
y = train[TARGET_COL]
X_test = test.drop([ID_COL], axis=1)


for col in CATEGORICAL_COLS:
    X[col] = X[col].fillna('missing')
    X_test[col] = X_test[col].fillna('missing')
    le = LabelEncoder()
    all_values = pd.concat([X[col], X_test[col]], axis=0).astype(str)
    le.fit(all_values)
    X[col] = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_labs = le.fit_transform(y) 
np.unique(y_labs)


print(le.classes_)


le.inverse_transform([0,1])


scaler = StandardScaler()
X[NUMERICAL_COLS] = scaler.fit_transform(X[NUMERICAL_COLS])
X_test[NUMERICAL_COLS] = scaler.transform(X_test[NUMERICAL_COLS])


# --- LightGBM Tuning ---
def objective_lgbm(trial):
    param = {
        'objective': 'multiclass',
        'num_class': len(np.unique(y_labs)),
        'metric': 'multi_logloss',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 15, 255),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'random_state': RANDOM_STATE
    }
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, valid_idx in cv.split(X, y):
        model = LGBMClassifier(**param)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[valid_idx])
        scores.append(accuracy_score(y.iloc[valid_idx], preds))
    return np.mean(scores)

study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgbm, n_trials=N_TRIALS)
BEST_LGB_PARAMS = study_lgb.best_params



print('y_labs type:', type(y_labs), 'dtype:', y_labs.dtype, 'shape:', y_labs.shape)
print('Unique:', set(y_labs[:20]))



# --- XGBoost Tuning ---
def objective_xgb(trial):
    param = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y_labs)),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': RANDOM_STATE,
        'tree_method': 'hist'
    }
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, valid_idx in cv.split(X, y_labs):
        model = XGBClassifier(
            **param,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        model.fit(X.iloc[train_idx], y_labs[train_idx])
        probs = model.predict_proba(X.iloc[valid_idx])
        preds = np.argmax(probs, axis=1)
        scores.append(accuracy_score(y_labs[valid_idx], preds))
    return np.mean(scores)

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS)
BEST_XGB_PARAMS = study_xgb.best_params



from catboost import CatBoostClassifier


# --- CatBoost Tuning ---
def objective_cat(trial):
    param = {
        'loss_function': 'MultiClass',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'random_state': RANDOM_STATE,
        'iterations': 1000,
        'verbose': False
    }
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, valid_idx in cv.split(X, y_labs):
        model = CatBoostClassifier(**param)
        model.fit(X.iloc[train_idx], y_labs[train_idx])
        preds = model.predict(X.iloc[valid_idx])
        scores.append(accuracy_score(y_labs[valid_idx], preds))
    return np.mean(scores)

study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(objective_cat, n_trials=N_TRIALS)
BEST_CAT_PARAMS = study_cat.best_params



n_classes = len(np.unique(y_labs))

BEST_LGB_PARAMS = study_lgb.best_params
BEST_LGB_PARAMS.update({'objective':'multiclass', 'num_class':n_classes, 'random_state':42})

BEST_XGB_PARAMS = study_xgb.best_params
BEST_XGB_PARAMS.update({'objective':'multi:softprob', 'num_class':n_classes, 'use_label_encoder':False, 'eval_metric':'mlogloss', 'random_state':42})

BEST_CAT_PARAMS = study_cat.best_params
BEST_CAT_PARAMS.update({'loss_function':'MultiClass', 'random_state':42})


n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds_lgb = np.zeros((X.shape[0], n_classes))
oof_preds_xgb = np.zeros((X.shape[0], n_classes))
oof_preds_cat = np.zeros((X.shape[0], n_classes))
test_preds_lgb = np.zeros((X_test.shape[0], n_classes))
test_preds_xgb = np.zeros((X_test.shape[0], n_classes))
test_preds_cat = np.zeros((X_test.shape[0],n_classes))



for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_labs)):
    print(f"Fold {fold+1}/{n_splits}") 

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_labs[train_idx], y_labs[val_idx]

    # LightGBM
    lgb_model = LGBMClassifier(**BEST_LGB_PARAMS, verbose=0)
    lgb_model.fit(X_train, y_train)
    oof_preds_lgb[val_idx] = lgb_model.predict_proba(X_val)
    test_preds_lgb += lgb_model.predict_proba(X_test) / n_splits

    # XGBoost
    xgb_model = XGBClassifier(**BEST_XGB_PARAMS, verbose=0)
    xgb_model.fit(X_train, y_train)
    oof_preds_xgb[val_idx] = xgb_model.predict_proba(X_val)
    test_preds_xgb += xgb_model.predict_proba(X_test) / n_splits

    # CatBoost
    cat_model = CatBoostClassifier(**BEST_CAT_PARAMS, verbose=0)
    cat_model.fit(X_train, y_train)
    oof_preds_cat[val_idx] = cat_model.predict_proba(X_val)
    test_preds_cat += cat_model.predict_proba(X_test) / n_splits



# Stack predictions for meta-model
stacked_X = np.concatenate([oof_preds_lgb, oof_preds_xgb, oof_preds_cat], axis=1)
stacked_test = np.concatenate([test_preds_lgb, test_preds_xgb, test_preds_cat], axis=1)

# Meta-model: Logistic Regression (can try LGBM/Ridge as well)
meta_model = LogisticRegression(max_iter=2000,  random_state=42)
meta_model.fit(stacked_X, y_labs)
final_preds = meta_model.predict(stacked_test)

# Cross-validated score of the stack
cv_stack = cross_val_score(meta_model, stacked_X, y_labs, cv=5, scoring='f1')
print(f"Stacked Ensemble CV Accuracy: {cv_stack.mean():.4f}")


class_names = le.classes_
int_to_label = {i: label for i, label in enumerate(class_names)}

submission = pd.DataFrame(
    {"id": test['id'],
    "Personality": [int_to_label[x] for x in final_preds]}
)
submission


submission.to_csv('submission.csv')

