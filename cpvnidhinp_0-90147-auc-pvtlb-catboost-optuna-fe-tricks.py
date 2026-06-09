# IMPORT LIBRARIES 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import warnings
warnings.filterwarnings('ignore')

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

# LOAD DATA
train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

y = train['rainfall']
X = train.drop(['id', 'rainfall'], axis=1)
X_test = test.drop(['id'], axis=1)

#  FEATURE ENGINEERING
def add_features(df):
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df['dewpoint_diff'] = df['temparature'] - df['dewpoint']
    df['wind_x'] = df['windspeed'] * np.cos(np.radians(df['winddirection']))
    df['wind_y'] = df['windspeed'] * np.sin(np.radians(df['winddirection']))
    df['cloud_sun_ratio'] = df['cloud'] / (df['sunshine'] + 1)
    df['sun_intensity'] = df['sunshine'] * df['temparature']
    df['humidity_pressure'] = df['humidity'] * df['pressure']
    df['humidity_temp_ratio'] = df['humidity'] / (df['temparature'] + 1)
    return df

X = add_features(X)
X_test = add_features(X_test)

# SELECT TOP 20 FEATURES WITH CATBOOST
cat_feat_model = CatBoostClassifier(iterations=700, learning_rate=0.03, depth=6, random_seed=42, verbose=0)
cat_feat_model.fit(X, y)
importances = pd.Series(cat_feat_model.get_feature_importance(Pool(X, label=y)), index=X.columns)
top_features = importances.sort_values(ascending=False).head(20).index.tolist()

X_top = X[top_features]
X_test_top = X_test[top_features]

#  DEFINE OPTUNA OBJECTIVE
def objective(trial):
    params = {
        'iterations': 700,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),        
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'verbose': 0,
        'random_seed': 42,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC'
    }

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for train_idx, val_idx in kf.split(X_top, y):
        X_tr, X_val = X_top.iloc[train_idx], X_top.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, preds))

    return np.mean(aucs)

#  OPTUNA TUNING
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("Best AUC:", study.best_value)
print("Best Params:", study.best_params)

# TRAIN FINAL MODEL WITH BEST PARAMS
best_params = study.best_params
best_params.update({
    'iterations': 700,
    'verbose': 0,
    'random_seed': 42
})

final_model = CatBoostClassifier(**best_params)
final_model.fit(X_top, y)

# PREDICT ON TEST
final_probs = final_model.predict_proba(X_test_top)[:, 1]

# OPTIONAL THRESHOLD OPTIMIZATION
def find_best_threshold(y_true, preds):
    best_thresh, best_f1 = 0.5, 0
    for t in np.arange(0.1, 0.9, 0.01):
        f1 = f1_score(y_true, (preds > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh, best_f1

# Evaluate on train set using CV preds
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_top))

for train_idx, val_idx in kf.split(X_top, y):
    X_tr, X_val = X_top.iloc[train_idx], X_top.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = CatBoostClassifier(**best_params)
    model.fit(X_tr, y_tr)
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

best_thresh, best_f1 = find_best_threshold(y, oof_preds)
auc_score = roc_auc_score(y, oof_preds)
print(f"\n Tuned CatBoost ROC AUC: {auc_score:.4f} | Best Threshold: {best_thresh:.2f} | F1: {best_f1:.4f}")

# FINAL PREDICTIONS + SUBMISSION
final_preds = (final_probs > best_thresh).astype(int)

submission = pd.DataFrame({
    'id': test['id'],
    'rainfall': final_probs 
})
#submission.to_csv('/kaggle/working/sub_10.csv', index=False)
submission.head(10)


