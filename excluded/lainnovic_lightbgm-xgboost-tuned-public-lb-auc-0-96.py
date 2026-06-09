#update 1 , now I have integrated 2 hyper parameter tuned models 
#note the running time might be high as the tuning params are pretty strong , you can edit as you wish 


#hey guys this is going to be my first ever notebook upload, as of now we are not 
#going to use any external data 

#we are going to make a baseline model which will be updated periodically 

train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')



#looking at the data first
print(train.head(10))


#complete data breakdown 


def data_summary(df):
    summary = pd.DataFrame({
        'dtype': df.dtypes,
        'n_unique': df.nunique(),
        'n_missing': df.isnull().sum(),
        'pct_missing': 100 * df.isnull().mean()
    })
    summary['sample_values'] = df.apply(lambda col: col.unique()[:5])
    return summary.sort_values(by='pct_missing', ascending=False)

data_summary(train)



#as we can see , there are no NAN values so this reduce our work by 20-25% but that being said 
#there are values stated as unknown ... so we might have to do something bout that 





import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier
import optuna
import warnings
warnings.filterwarnings('ignore')


# ===================== Preprocessing =====================
train = train.drop(columns=['id'])
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])  # Apply same encoder to test
    label_encoders[col] = le

X = train.drop(columns=['y'])
y = train['y']

# Save test ids and drop id column
test_ids = test['id']
test = test.drop(columns='id')

# ===================== Optuna Tuning: LightGBM =====================
def objective_lgb(trial):
    params = {
        'n_estimators': 300,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 16, 64),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
        'objective': 'binary',
        'verbosity': -1
    }

    aucs = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[early_stopping(15), log_evaluation(0)]
        )
        preds = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, preds))
    return np.mean(aucs)

study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgb, n_trials=15)
best_params_lgb = study_lgb.best_params
best_params_lgb.update({'n_estimators': 300, 'objective': 'binary', 'verbosity': -1})

# ===================== Optuna Tuning: XGBoost =====================
def objective_xgb(trial):
    params = {
        'n_estimators': 300,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
        'use_label_encoder': False,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'verbosity': 0
    }

    aucs = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, preds))
    return np.mean(aucs)

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=15)
best_params_xgb = study_xgb.best_params
best_params_xgb.update({
    'n_estimators': 300,
    'use_label_encoder': False,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'verbosity': 0
})

# ===================== Final Training =====================
lgb_final = LGBMClassifier(**best_params_lgb)
xgb_final = XGBClassifier(**best_params_xgb)

lgb_final.fit(X, y)
xgb_final.fit(X, y)

# ===================== Predict on Test and Average =====================
test_preds_lgb = lgb_final.predict_proba(test)[:, 1]
test_preds_xgb = xgb_final.predict_proba(test)[:, 1]
final_preds = (test_preds_lgb + test_preds_xgb) / 2

# ===================== Save Submission =====================
submission = pd.DataFrame({
    'id': test_ids,
    'y': final_preds
})
submission.to_csv("submission.csv", index=False)
print("✅ Submission file 'submission.csv' saved!")


























