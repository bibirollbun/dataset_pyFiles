# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


FRAC = 1
orig = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv").set_index("id")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
extra = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
train = pd.concat([orig, extra])
train = train.sample(frac=FRAC, random_state=42).reset_index(drop=True)

train.head()


from sklearn.preprocessing import LabelEncoder

y = train["Fertilizer Name"]
le = LabelEncoder()
y_enc = le.fit_transform(y)
X = train.drop("Fertilizer Name", axis = 1)

X_enc = X.copy()
cat_cols = X_enc.astype("category")
for col in cat_cols:
    X_enc[col] = X_enc[col].astype("category").cat.codes
    test[col] = test[col].astype("category").cat.codes



def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        score = 0.0
        for i in range(min(k, len(p))):
            if p[i] == a:
                score += 1.0 / (i + 1)
                break  
        return score
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])




from sklearn.model_selection import KFold
from xgboost import XGBClassifier
import optuna



# CV setup
seed1 = 42
cv = KFold(n_splits=3, shuffle=True, random_state=seed1)

# Base params
params = {
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'num_class': len(le.classes_),
    'use_label_encoder': False,
    'verbosity': 0,
    'tree_method' : "gpu_hist",
    'predictor' : "gpu_predictor",
    'n_jobs' : -1
    
}

# Optuna objective
def objective(trial):
    tuned_params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 20),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 100.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-3, 100.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 100.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000)
    }

    full_params = params.copy()
    full_params.update(tuned_params)
    fold_scores = []
    
    for fold, (idx_train, idx_valid) in enumerate(cv.split(X_enc)):
        X_train, y_train = X_enc.iloc[idx_train], y_enc[idx_train]
        X_valid, y_valid = X_enc.iloc[idx_valid], y_enc[idx_valid]

        model = XGBClassifier(
            **full_params,
            early_stopping_rounds=100,
            random_state=seed1
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )

        val_probs = model.predict_proba(X_valid)
        top3_val_indices = np.argsort(val_probs, axis=1)[:, ::-1][:, :3]
        top3_val_labels = le.inverse_transform(top3_val_indices.ravel()).reshape(top3_val_indices.shape)

        true_labels = le.inverse_transform(y_valid)
        fold_scores.append(mapk(true_labels.tolist(), top3_val_labels.tolist(), k=3))

    return np.mean(fold_scores)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("Best MAP@3:", study.best_value)
print("Best Parameters:", study.best_params)

# Save best params
params.update(study.best_params)




X_test = test[X.columns].copy()

FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

accum_test_probs = np.zeros((len(X_test), len(le.classes_)))
predicted_oof_topk = np.empty((len(X_enc), 3), dtype=object)



for fold, (idx_train, idx_valid) in enumerate(kf.split(X_enc)):
    print(f"Training fold {fold+1}...")
    X_train, y_train = X_enc.iloc[idx_train], y_enc[idx_train]
    X_valid, y_valid = X_enc.iloc[idx_valid], y_enc[idx_valid]

    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )

    val_probs = model.predict_proba(X_valid)
    test_probs = model.predict_proba(X_test)

    accum_test_probs += test_probs

    top3_val_indices = np.argsort(val_probs, axis=1)[:, ::-1][:, :3]
    top3_val_labels = le.inverse_transform(top3_val_indices.ravel()).reshape(top3_val_indices.shape)
    predicted_oof_topk[idx_valid] = top3_val_labels

    fold_mapk = mapk(le.inverse_transform(y_valid).tolist(), top3_val_labels.tolist(), k=3)
    print(f"Fold {fold+1} MAP@3: {fold_mapk:.5f}")

avg_test_probs = accum_test_probs / FOLDS

avg_test_probs_df = pd.DataFrame(avg_test_probs, columns=le.classes_)

top3_indices = np.argsort(avg_test_probs, axis=1)[:, ::-1][:, :3]
top3_labels = le.inverse_transform(top3_indices.ravel()).reshape(top3_indices.shape)
final_preds = [" ".join(row) for row in top3_labels]

cv_mapk = mapk(y.tolist(), predicted_oof_topk.tolist(), k=3)
print(f"CV MAP@3: {cv_mapk:.5f}")



submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': final_preds
})

submission.to_csv('submission.csv', index=False)
print("Submission created:")
print(submission.head())

