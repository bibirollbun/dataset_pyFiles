import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import top_k_accuracy_score
from collections import Counter


LOCAL = False


if LOCAL:
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
else:
    train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


train.head()


train.info()


train['Fertilizer Name'].value_counts()


train['Fertilizer Name'].value_counts(normalize=True) * 100


import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer
from tqdm import tqdm

# === Загрузка и кодирование ===
X = train.drop(columns=['Fertilizer Name', 'id'])
y = train['Fertilizer Name']

# Label encode target
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# Encode Soil Type и Crop Type
cat_features = ['Soil Type', 'Crop Type']
encoders = {}
for col in cat_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# === Метрика MAP@3 ===
def map3(preds, labels):
    score = 0.0
    for i, pred in enumerate(preds):
        if labels[i] in pred:
            rank = pred.index(labels[i]) + 1
            score += 1.0 / rank
    return score / len(labels)

# === Optuna Objective ===
def objective(trial):
    params = {
    'n_estimators': trial.suggest_int('n_estimators', 100, 600),
    'max_depth': trial.suggest_int('max_depth', 3, 12),
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
    'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0),
    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    'gamma': trial.suggest_float('gamma', 0.0, 5.0),
    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
    'max_delta_step': trial.suggest_int('max_delta_step', 0, 5),
    'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
    'eval_metric': 'mlogloss',
    'random_state': 42
}

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    map3_scores = []

    for train_idx, val_idx in skf.split(X, y_encoded):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        val_probs = model.predict_proba(X_val)
        top3 = np.argsort(val_probs, axis=1)[:, -3:][:, ::-1]
        score = map3(top3.tolist(), y_val.tolist())
        map3_scores.append(score)

    return np.mean(map3_scores)

# === Оптимизация ===
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Best MAP@3:", study.best_value)
print("Best Params:", study.best_params)

# === Обучение финальной модели на всех данных ===
final_model = XGBClassifier(
    **study.best_params,
    use_label_encoder=False,
    eval_metric='mlogloss',
    tree_method='hist',
    random_state=42
)
final_model.fit(X, y_encoded)



# Подготовка теста
X_test = test.drop(columns=['id'])
for col in cat_features:
    X_test[col] = encoders[col].transform(X_test[col])

probs_test = final_model.predict_proba(X_test)
top3_test = np.argsort(probs_test, axis=1)[:, -3:][:, ::-1]
top3_labels = le_target.inverse_transform(top3_test.ravel()).reshape(-1, 3)

submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': [' '.join(row) for row in top3_labels]
})
submission.to_csv('submission.csv', index=False)


