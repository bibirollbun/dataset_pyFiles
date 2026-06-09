import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score,train_test_split, StratifiedKFold
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, StackingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
import itertools
import optuna


df = pd.read_csv(f'/kaggle/input/playground-series-s5e7/train.csv')
df_test = pd.read_csv(f'/kaggle/input/playground-series-s5e7/test.csv')


Null_bi_col = ['Stage_fear', 'Drained_after_socializing']
for i in Null_bi_col:
    df[i] = df[i].map({
        'Yes': 1,
        'No' : 0
    })
    df_test[i] = df_test[i].map({
        'Yes': 1,
        'No' : 0
    })
df['Personality'] = df['Personality'].map({
    'Introvert' : 0,
    'Extrovert' : 1
})


# Use KNN imputer 
X = df.drop(columns=['Personality', 'id'])
X_test = df_test.drop(columns=['id'])
imputer = KNNImputer(n_neighbors=5, weights='uniform')
df_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
df_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)


df_set = pd.concat([df_imputed, df['Personality']], axis = 1)


X = df_set.drop(columns = ['Personality'])
Y = df_set['Personality']
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)


import optuna
def objective(trial):
    # Tham số cho XGBoost
    xgb_params = {
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3),
        'gamma': trial.suggest_float('xgb_gamma', 0.0, 3.0),
        'max_depth': trial.suggest_int('xgb_max_depth', 6, 15),
        'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 10),
        'subsample': trial.suggest_float('xgb_subsample', 0.8, 1.0),
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.1, 1.0),
        'reg_lambda': trial.suggest_float('xgb_reg_lambda', 0.1, 2.0),
        'reg_alpha': trial.suggest_float('xgb_reg_alpha', 0.1, 2.0),
        'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 1000),
        'objective': 'multi:softprob',
        'num_class': len(set(y_train)),  # nếu là multi-class
        'use_label_encoder': False,
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'n_jobs': -1
    }

    # Tham số cho Random Forest
    rf_params = {
        'n_estimators': trial.suggest_int('rf_n_estimators', 100, 1000),
        'criterion': trial.suggest_categorical('rf_criterion', ["gini", "entropy", "log_loss"]),
        'max_depth': trial.suggest_int('rf_max_depth', 6, 15),
        'min_samples_split': trial.suggest_float('rf_min_samples_split', 0.0, 1.0),
        'max_features': trial.suggest_categorical('rf_max_features', ["sqrt", "log2", None]),
        'random_state': 42,
        'n_jobs': -1
    }

    # Trọng số soft voting
    w_rf = trial.suggest_float("weight_rf", 0.1, 2.0)
    w_xgb = trial.suggest_float("weight_xgb", 0.1, 2.0)

    # Khởi tạo mô hình
    xgb_model = XGBClassifier(**xgb_params)
    rf_model = RandomForestClassifier(**rf_params)

    voting_clf = VotingClassifier(
        estimators=[("rf", rf_model), ("xgb", xgb_model)],
        voting="soft",
        weights=[w_rf, w_xgb],
        n_jobs=-1
    )

    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    return acc

# Tạo study và chạy tối ưu
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Best parameters:", study.best_trial.params)
print("Best score:", study.best_value)



rf_params = {
    'n_estimators': 626,
    'criterion': 'log_loss',
    'max_depth': 13,
    'min_samples_split': 0.8321156372191643,
    'max_features': 'log2',
    'random_state': 42,
    'n_jobs': -1
}
xgb_params = {
    'learning_rate': 0.07297004610144715,
    'gamma': 1.7253356371825426,
    'max_depth': 7,
    'min_child_weight': 4,
    'subsample': 0.8714485363200863,
    'colsample_bytree': 0.42344206351773384,
    'reg_lambda': 0.7342401441632284,
    'reg_alpha': 0.5546660715967193,
    'n_estimators': 271,
    'objective': 'multi:softprob',
    'num_class': len(set(y_train)),
    'use_label_encoder': False,
    'eval_metric': 'mlogloss',
    'random_state': 42,
    'n_jobs': -1
}

rf_model = RandomForestClassifier(**rf_params)
xgb_model = XGBClassifier(**xgb_params)

voting_clf = VotingClassifier(
        estimators=[("rf", rf_model), ("xgb", xgb_model)],
        voting="soft",
        weights=[1.3281795373334073, 1.5235020381631403],
        n_jobs=-1
)

voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(acc)


pred_vote = voting_clf.predict(df_test_imputed)
submission_df = pd.DataFrame({
    'id': df_test['id'],
    'Personality': pred_vote 
})
submission_df['Personality'] = submission_df['Personality'].map({
    0: 'Introvert',
    1: 'Extrovert'
})
submission_df.to_csv(f'submission.csv', index = False)


submission_df.head(5)

