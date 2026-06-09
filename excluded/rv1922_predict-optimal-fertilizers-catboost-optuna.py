from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import os
import warnings
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier, Pool
import optuna
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
original = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


train = train.drop("id", axis=1)
test = test.drop("id", axis=1)
original = original.drop("id", axis=1)
train = pd.concat([train, original], ignore_index=True)
train = train.drop_duplicates()


train.head()


train.info()


cat_cols = [col for col in train.select_dtypes(include=['object', 'category']).columns if col != "Fertilizer Name"]
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])


fer_label_enc = LabelEncoder()
train["Fertilizer Name"] = fer_label_enc.fit_transform(train["Fertilizer Name"])


train.head()


X = train.drop(columns=["Fertilizer Name"])
y = train["Fertilizer Name"]
X_test = test


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)


def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk([a], [pred[i] for i in range(k)], k) for a, pred in zip(actual, predicted)])


def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 100.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_strength': trial.suggest_float('random_strength', 1e-9, 10.0),
        'task_type': 'GPU',
        'devices': '0',
        'eval_metric': 'MultiClass',
        'verbose': False,
        'early_stopping_rounds': 100
    }

    train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_cols)

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=valid_pool)

    probs = model.predict_proba(X_valid)

    class_labels = model.classes_
    top_preds = [
        [class_labels[i] for i in np.argsort(p)[::-1][:3]] for p in probs
    ]

    return mapk(y_valid.values, top_preds, k=3)


#study = optuna.create_study(direction='maximize')
#study.optimize(objective, n_trials=50)


#print("\nâœ… Best Trial:")
#print(f"MAP@3 Score: {study.best_value:.5f}")
#print("Best Parameters:")


FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
num_classes = len(np.unique(y))

# Initialize arrays for out-of-fold and test predictions
oof = np.zeros((len(X), num_classes))
pred_prob = np.zeros((len(X_test), num_classes))

# CatBoost parameters
best_params_cat = {
    'iterations': 4764,
    'learning_rate': 0.06093272183703829,
    'depth': 6,
    'l2_leaf_reg': 0.8590705802376157,
    'bagging_temperature': 0.5015570553007467,
    'border_count': 124,
    'random_strength': 2.645404610746076,
    'task_type': 'GPU',
    'eval_metric': 'MultiClass',
    'verbose': False,
    'early_stopping_rounds': 100
}

# XGBoost parameters
xgb_params = {
    'max_depth': 13,
    'colsample_bytree': 0.30440196038980377,
    'subsample': 0.5302363702993608,
    'n_estimators': 4500,
    'learning_rate': 0.043509813901570604,
    'gamma': 0.34649185501450364,
    'max_delta_step': 8,
    'reg_alpha': 2.0136709028472195,
    'reg_lambda': 3.131760778539737,
    'early_stopping_rounds': 100,
    'objective': 'multi:softprob',
    'random_state': 13,
    'enable_categorical': True,
    'tree_method': 'hist',
    'device': 'cuda'
}

# K-Fold Cross-Validation
for fold, (train_idx, valid_idx) in enumerate(kf.split(X), 1):
    print(f"\n{'='*10} Fold {fold} {'='*10}")
    
    # Split data
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # --- Train CatBoost ---
    train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_cols)
    model_cat = CatBoostClassifier(**best_params_cat)
    model_cat.fit(train_pool, eval_set=valid_pool)
    oof_cat_fold = model_cat.predict_proba(X_valid)
    test_pred_cat_fold = model_cat.predict_proba(X_test)

    # --- Train XGBoost ---
    model_xgb = XGBClassifier(**xgb_params)
    model_xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    oof_xgb_fold = model_xgb.predict_proba(X_valid)
    test_pred_xgb_fold = model_xgb.predict_proba(X_test)

    # Average predictions from both models
    oof[valid_idx] = (oof_cat_fold + oof_xgb_fold) / 2
    pred_prob += (test_pred_cat_fold + test_pred_xgb_fold) / 2 / FOLDS

    # Compute MAP@3 for current fold
    class_labels = model_cat.classes_  # Assuming both models have same class order
    top_preds = [[class_labels[i] for i in np.argsort(p)[::-1][:3]] for p in oof[valid_idx]]
    actual = [[label] for label in y_valid]
    map3_score = mapk(actual, top_preds, k=3)
    print(f"âœ… Fold {fold} MAP@3 Score: {map3_score:.5f}")

# --- Final Metrics ---
logloss_score = log_loss(y, oof)
print(f"\nðŸ“Š Overall Log Loss: {logloss_score:.5f}")


top_3_preds = np.argsort(pred_prob, axis=1)[:, -3:][:, ::-1]
top_3_labels = fer_label_enc.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)
submission = pd.DataFrame({
    'id': submission['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})
submission.to_csv('submission.csv', index=False)
print("âœ… Submission file saved as 'submission.csv'")


print("\nSubmission Preview:")
print(submission.head())

