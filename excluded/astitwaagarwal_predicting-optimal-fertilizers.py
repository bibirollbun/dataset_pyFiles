import pandas as pd
import numpy as np
import optuna
import csv

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import category_encoders as ce


def map3(preds, true):
    """
    Compute Mean Average Precision at 3.
    """
    top_3 = np.argsort(preds, axis=1)[:, -3:][:, ::-1]
    score = 0
    for i, t in enumerate(true):
        if t == top_3[i][0]:
            score += 1 / 1
        elif t == top_3[i][1]:
            score += 1 / 2
        elif t == top_3[i][2]:
            score += 1 / 3
    return score / len(true)


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
train_more = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer-Prediction.csv")


train = pd.concat([train, train_more])


train.drop(columns=['id'], inplace=True)
test_ids = test['id']
test.drop(columns=['id'], inplace=True)


X = train.drop(columns='Fertilizer Name')
y = train['Fertilizer Name']


encoder = ce.OrdinalEncoder()
X_encoded = encoder.fit_transform(X)
test_encoded = encoder.transform(test)


le = LabelEncoder()
y_encoded = le.fit_transform(y)


X_train, X_val, y_train, y_val = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1200),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 6),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y_encoded)),
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'random_state': 42,
        'verbosity': 0
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train, 
              eval_set=[(X_val, y_val)], 
              early_stopping_rounds=20, 
              verbose=False)
    
    preds = model.predict_proba(X_val)
    return map3(preds, y_val)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)


best_params = study.best_params
best_params.update({
    'objective': 'multi:softprob',
    'num_class': len(np.unique(y_encoded)),
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'random_state': 42
})


final_model = XGBClassifier(**best_params)
final_model.fit(X_encoded, y_encoded)


test_preds = final_model.predict_proba(test_encoded)


top_3 = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]
top_3_labels = np.array([le.inverse_transform(row) for row in top_3])
fertilizer_preds = [' '.join(map(str, preds)) for preds in top_3_labels]


submission_rows = list(zip(test_ids, fertilizer_preds))

with open("submission.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "Fertilizer Name"])
    writer.writerows(submission_rows)




