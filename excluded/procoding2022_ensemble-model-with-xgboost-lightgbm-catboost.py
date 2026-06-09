import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col='id')


test['winddirection'] = test['winddirection'].fillna(test['winddirection'].median())


def create_features(df):
    df = df.copy()
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df['temp_ratio'] = df['temparature'] / (df['maxtemp'] + 1e-6)
    df['humid_pressure_interaction'] = df['humidity'] * df['pressure']
    df['cloud_sunshine_ratio'] = df['cloud'] / (df['sunshine'] + 1)
    df['temp_humidity_index'] = (df['temparature'] * df['humidity']) / 100
    df['pressure_temp_humidity'] = (df['pressure'] * df['temparature']) / (df['humidity'] + 1e-6)
    return df

train = create_features(train)
test = create_features(test)


X = train.drop(columns=['rainfall'])
y = train['rainfall']
X_test = test.copy()
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }
    model = XGBClassifier(**params, random_state=42, use_label_encoder=False, eval_metric='logloss')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    for train_idx, val_idx in skf.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict_proba(X[val_idx])[:, 1]
        auc_scores.append(roc_auc_score(y[val_idx], y_pred))
    return np.mean(auc_scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
best_params = study.best_trial.params


xgb = XGBClassifier(**best_params, random_state=42, use_label_encoder=False, eval_metric='logloss')
lgb = LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=7, random_state=42)
cat = CatBoostClassifier(n_estimators=500, learning_rate=0.05, depth=7, verbose=False, random_state=42)


stacking_model = StackingClassifier(estimators=[
    ('xgb', xgb),
    ('lgb', lgb),
    ('cat', cat)
], final_estimator=XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42))

stacking_model.fit(X, y)


test_predictions = stacking_model.predict_proba(X_test)[:, 1]


submission = pd.DataFrame({
    'id': test.index,
    'rainfall': test_predictions
})


submission


submission.to_csv('submission.csv', index=False)

