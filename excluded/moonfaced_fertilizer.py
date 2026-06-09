import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import optuna
from optuna.samplers import TPESampler
import warnings

warnings.filterwarnings("ignore")

# --- Custom MAP@3 Metric ---
def map_at_k(y_true, y_pred_proba, encoder, k=3):
    y_true_decoded = encoder.inverse_transform(y_true)
    top_k_pred_indices = np.argsort(y_pred_proba, axis=1)[:, -k:][:, ::-1]
    top_k_pred_decoded = encoder.inverse_transform(top_k_pred_indices.flatten()).reshape(top_k_pred_indices.shape)

    scores = []
    for i in range(len(y_true)):
        true_label = y_true_decoded[i]
        predicted = top_k_pred_decoded[i]
        score = 0
        for j in range(k):
            if predicted[j] == true_label:
                score += 1.0 / (j + 1.0)
                break
        scores.append(score)
    return np.mean(scores)

# --- Load data ---
train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
original_df = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')

# --- Combine ---
required_cols = [col for col in train_df.columns if col != 'id']
original_df_filtered = original_df[required_cols]
combined_train_df = pd.concat([train_df.drop('id', axis=1), original_df_filtered], ignore_index=True)

# --- Feature Engineering ---
combined_train_df['N_P'] = combined_train_df['Nitrogen'] * combined_train_df['Phosphorous']
combined_train_df['K_H'] = combined_train_df['Potassium'] * combined_train_df['Humidity']
combined_train_df['N2'] = combined_train_df['Nitrogen'] ** 2
combined_train_df['P2'] = combined_train_df['Phosphorous'] ** 2
combined_train_df['log_Moisture'] = np.log1p(combined_train_df['Moisture'])
combined_train_df['Temp_Hum'] = combined_train_df['Temparature'] * combined_train_df['Humidity']
combined_train_df['N_over_K'] = combined_train_df['Nitrogen'] / (combined_train_df['Potassium'] + 1)

test_df['N_P'] = test_df['Nitrogen'] * test_df['Phosphorous']
test_df['K_H'] = test_df['Potassium'] * test_df['Humidity']
test_df['N2'] = test_df['Nitrogen'] ** 2
test_df['P2'] = test_df['Phosphorous'] ** 2
test_df['log_Moisture'] = np.log1p(test_df['Moisture'])
test_df['Temp_Hum'] = test_df['Temparature'] * test_df['Humidity']
test_df['N_over_K'] = test_df['Nitrogen'] / (test_df['Potassium'] + 1)


# --- Preprocessing ---
numerical_features = [
    'Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous',
    'N_P', 'K_H', 'N2', 'P2', 'log_Moisture', 'Temp_Hum', 'N_over_K'
]
categorical_features = ['Soil Type', 'Crop Type']
preprocessor = ColumnTransformer([
    ('num', 'passthrough', numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

le = LabelEncoder()
y = le.fit_transform(combined_train_df['Fertilizer Name'])
X = preprocessor.fit_transform(combined_train_df.drop('Fertilizer Name', axis=1))
X = X.toarray() if hasattr(X, 'toarray') else X
num_classes = len(le.classes_)
X_test = preprocessor.transform(test_df.drop('id', axis=1))
X_test = X_test.toarray() if hasattr(X_test, 'toarray') else X_test

# --- Optuna objective ---
def objective(trial):
    params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "num_class": num_classes,
        "use_label_encoder": False,
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
        "max_depth": trial.suggest_int("max_depth", 7, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 2),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 2),
        "random_state": 42
    }

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X, y):
        model = xgb.XGBClassifier(**params)
        model.fit(X[train_idx], y[train_idx])
        y_pred_proba = model.predict_proba(X[val_idx])
        score = map_at_k(y[val_idx], y_pred_proba, le, k=3)
        scores.append(score)

    return np.mean(scores)

# --- Run Optuna study ---
study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=30)

# --- Train final model on full data ---
best_params = study.best_params
best_params.update({
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "num_class": num_classes,
    "use_label_encoder": False,
    "random_state": 42
})

final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X, y)

final_model.save_model("final_model_optuna_xgb.json")  # или .model
import joblib
joblib.dump(le, "label_encoder.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")

# --- Predict on test set ---
y_test_proba = final_model.predict_proba(X_test)
top3_idx = np.argsort(y_test_proba, axis=1)[:, -3:][:, ::-1]
top3_labels = [" ".join(le.inverse_transform(row)) for row in top3_idx]

submission_df = pd.DataFrame({
    'id': test_df['id'],
    'Fertilizer Name': top3_labels
})
submission_df.to_csv("submission_optuna_xgb.csv", index=False)

# --- Print results ---
print("\nBest Params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")
print("Best MAP@3:", study.best_value)
print(submission_df.head())





