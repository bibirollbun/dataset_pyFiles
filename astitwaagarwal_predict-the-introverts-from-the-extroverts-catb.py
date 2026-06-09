import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from catboost import CatBoostClassifier, Pool
import optuna


train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
submission_df = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


binary_map = {'Yes': 1, 'No': 0}
train_df['Stage_fear'] = train_df['Stage_fear'].map(binary_map)
train_df['Drained_after_socializing'] = train_df['Drained_after_socializing'].map(binary_map)
train_df['Personality'] = train_df['Personality'].map({'Introvert': 0, 'Extrovert': 1})


test_df['Stage_fear'] = test_df['Stage_fear'].map(binary_map)
test_df['Drained_after_socializing'] = test_df['Drained_after_socializing'].map(binary_map)


train_df.fillna(train_df.median(numeric_only=True), inplace=True)
test_df.fillna(test_df.median(numeric_only=True), inplace=True)


X = train_df.drop(['id', 'Personality'], axis=1)
y = train_df['Personality']
X_test = test_df.drop('id', axis=1)


def add_features(df):
    df['Social_to_Friend_Ratio'] = df['Social_event_attendance'] / (df['Friends_circle_size'] + 1)
    df['Posts_per_Friend'] = df['Post_frequency'] / (df['Friends_circle_size'] + 1)
    df['Time_to_GoOut_Ratio'] = df['Time_spent_Alone'] / (df['Going_outside'] + 1)
    df['Social_Score'] = (
        df['Social_event_attendance'] +
        df['Going_outside'] +
        df['Post_frequency'] +
        df['Friends_circle_size']
    )
    return df


X = add_features(X)
X_test = add_features(X_test)


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


def objective(trial):
    params = {
        'iterations': trial.suggest_int("iterations", 300, 1000),
        'depth': trial.suggest_int("depth", 4, 8),
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3),
        'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 1, 10),
        'random_strength': trial.suggest_float("random_strength", 1e-9, 10),
        'bagging_temperature': trial.suggest_float("bagging_temperature", 0.0, 1.0),
        'border_count': trial.suggest_int("border_count", 32, 255),
        'verbose': 0,
        'loss_function': 'Logloss',
        'task_type': 'GPU'
    }

    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)
    preds = model.predict(X_valid)
    return accuracy_score(y_valid, preds)



study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
best_params = study.best_params


final_model = CatBoostClassifier(**best_params, loss_function='Logloss', task_type='GPU', verbose=0)
final_model.fit(X_train, y_train)


val_probs = final_model.predict_proba(X_valid)[:, 1]
best_f1, best_threshold = 0, 0.5
for t in np.arange(0.3, 0.7, 0.01):
    preds = (val_probs > t).astype(int)
    f1 = f1_score(y_valid, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"Best threshold: {best_threshold:.2f} | Best F1 Score: {best_f1:.4f}")


final_model.fit(X, y)
test_probs = final_model.predict_proba(X_test)[:, 1]
test_preds = (test_probs > best_threshold).astype(int)
submission_df['Personality'] = np.where(test_preds == 1, 'Extrovert', 'Introvert')


submission_df.to_csv("submission.csv", index=False)
print("submission.csv saved")




