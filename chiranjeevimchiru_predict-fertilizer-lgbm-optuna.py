import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
import optuna
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import label_ranking_average_precision_score
from tqdm import tqdm


# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")

# Label encoding target
le = LabelEncoder()
train['Fertilizer Name Enc'] = le.fit_transform(train['Fertilizer Name'])


features = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']
categorical_features = ['Soil Type', 'Crop Type']



X_train, X_valid, y_train, y_valid = train_test_split(
    train[features], train['Fertilizer Name Enc'], test_size=0.2, random_state=42, stratify=train['Fertilizer Name Enc']
)
# Prepare Pools
train_pool = Pool(X_train, y_train, cat_features=categorical_features)
valid_pool = Pool(X_valid, y_valid, cat_features=categorical_features)


def mapk(actual, predicted, k=3):
    """
    Computes the mean average precision at k.
    actual: true labels
    predicted: predicted probabilities (2D array)
    """
    top_k_preds = np.argsort(predicted, axis=1)[:, -k:][:, ::-1]
    score = 0.0
    for i in range(len(actual)):
        if actual[i] in top_k_preds[i]:
            rank = np.where(top_k_preds[i] == actual[i])[0][0] + 1
            score += 1.0 / rank
    return score / len(actual)


def objective(trial):
    params = {
        "iterations": 10,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "random_strength": trial.suggest_float("random_strength", 1e-9, 10.0),
        "loss_function": "MultiClass",
        "eval_metric": "TotalF1",
        "verbose": 0,
        "random_seed": 42,
        "task_type": "CPU"
    }

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=50, use_best_model=True)
    preds = model.predict_proba(X_valid)
    return mapk(y_valid.values, preds, k=3)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)


best_params = study.best_params
best_params.update({
    "iterations": 10,
    "loss_function": "MultiClass",
    "verbose": 100,
    "random_seed": 42
})

final_model = CatBoostClassifier(**best_params)
final_model.fit(train[features], train['Fertilizer Name Enc'], cat_features=categorical_features)


test_preds_proba = final_model.predict_proba(test[features])
top_3_preds = np.argsort(test_preds_proba, axis=1)[:, -3:][:, ::-1]
top_3_labels = np.array([le.inverse_transform(row) for row in top_3_preds
])


submission = pd.DataFrame({
    "id": test["id"],
    "Fertilizer Name": [' '.join(row) for row in top_3_labels]
})

submission
submission.to_csv("submission.csv", index=False)





