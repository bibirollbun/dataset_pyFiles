import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer
import optuna



# Load data
df_sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")

# 2nd dataset
original = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')
# Ths Orignial data is mentioned on data section of competitions


# Drop ID
df_train = df_train.drop(columns=['id'])
df_test = df_test.drop(columns=['id'])


df_train = pd.concat([df_train, original], ignore_index=True)



df_train.info()


duplicate_rows = df_train.duplicated()
print(f"Number of duplicate rows: {duplicate_rows.sum()}")



df_train.isnull().sum()


# Ordinal encode categorical features
cat_cols = df_train.select_dtypes(include='object').columns
cat_cols = cat_cols[cat_cols != 'Fertilizer Name']
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df_train[cat_cols] = ordinal_encoder.fit_transform(df_train[cat_cols].astype(str))
df_test[cat_cols] = ordinal_encoder.transform(df_test[cat_cols].astype(str))


# Encode target
le = LabelEncoder()
df_train['Fertilizer Name'] = le.fit_transform(df_train['Fertilizer Name'])


df_train['Fertilizer Name']



# Features and target
X = df_train.drop(columns=['Fertilizer Name'])
y = df_train['Fertilizer Name']


# Train/validation split
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# MAP@3 scoring function
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
    return np.mean([apk([a], p, k) for a, p in zip(actual, predicted)])

# Optuna objective
def objective(trial):
    params = {
    "objective": "multi:softprob",
    "num_class": len(np.unique(train_y)),
    "eval_metric": "mlogloss",
    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.06),
    "max_depth": trial.suggest_int("max_depth", 5, 10),
    "subsample": trial.suggest_float("subsample", 0.7, 0.9),
    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
    "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.7, 0.9),
    "colsample_bynode": trial.suggest_float("colsample_bynode", 0.6, 0.8),
    "n_estimators": 1000,
    "verbosity": 0,
    "random_state": 42,
    "early_stopping_rounds": 50,
    
    # these two for GPU:
    "tree_method": "gpu_hist",
    "device": "cuda",
}


    model = XGBClassifier(**params)
    model.fit(train_X, train_y,
              eval_set=[(valid_X, valid_y)],
              verbose=False)
    pred_probs = model.predict_proba(valid_X)
    top_3_preds = np.argsort(pred_probs, axis=1)[:, -3:][:, ::-1]
    score = mapk(valid_y, top_3_preds)
    return score

# Run Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=25)

print("âœ… Best MAP@3:", study.best_value)
print("ğŸ”§ Best params:", study.best_params)

# Training final model with best params
best_params = study.best_params
best_params.update({
    "objective": "multi:softprob",
    "num_class": len(np.unique(y)),
    "eval_metric": "mlogloss",
    "n_estimators": 1000,
    "verbosity": 0,
    "random_state": 42,
})
final_model = XGBClassifier(**best_params)
final_model.fit(X, y)

# Predict on test
test_probs = final_model.predict_proba(df_test)
top_3_preds = np.argsort(test_probs, axis=1)[:, -3:][:, ::-1]
top_3_labels = le.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)

# Create submission
submission = pd.DataFrame({
    "id": df_sub["id"],
    "Fertilizer Name": [' '.join(row) for row in top_3_labels]
})
submission.to_csv("submission.csv", index=False)
print("ğŸ“� Submission saved to 'submission.csv'")





