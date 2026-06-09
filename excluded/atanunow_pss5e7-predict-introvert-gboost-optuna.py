import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import optuna

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor


import warnings
warnings.filterwarnings('ignore')


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
original_df = pd.read_csv('/kaggle/input/extrovert-introvert-dataset/personality_datasert.csv')
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


num_col = [col for col in train_df.select_dtypes(['number']).columns if col!='id']
cat_col = [col for col in train_df.select_dtypes(['object']).columns if col!='Personality']
target_col = "Personality"
id_col = 'id'


combined_df = pd.concat([train_df.drop(columns=id_col, axis=1),original_df], ignore_index=True).drop_duplicates()


le = LabelEncoder()
tgt_le = LabelEncoder()


non_impute_columns = [target_col]
features = combined_df.drop(columns=non_impute_columns)


encoders = {}

for col in cat_col:
    temp_col = combined_df[col].fillna("NaN_Placeholder")
    combined_df[col] = le.fit_transform(temp_col)  # Encode the column
    combined_df[col] = combined_df[col].where(temp_col != "NaN_Placeholder", np.nan) # Restore NaN values
    encoders[col] = {cls: le.transform([cls])[0] for cls in le.classes_ if cls != "NaN_Placeholder"}


numerical_columns = list(combined_df.select_dtypes('number').columns)


# Impute numerical columns
imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=42), random_state=42)
combined_df[numerical_columns] = imputer.fit_transform(combined_df[numerical_columns])


target_col = 'Personality'
tgt_le = LabelEncoder()

X = combined_df.drop(target_col, axis=1).values  # Ensure numpy array for compatibility
y = tgt_le.fit_transform(combined_df[target_col])


def oof_gradient_boosting(model, X, y, n_splits=5):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    feature_importances = np.zeros(X.shape[1])
    fold_scores = []
    fold_models = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
        print(f"Training fold {fold + 1}/{n_splits}...")
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        oof_preds[valid_idx] = preds
        fold_score = accuracy_score(y_valid, preds)
        fold_scores.append(fold_score)
        fold_models.append(model)

        # Accumulate feature importance
        feature_importances += model.feature_importances_ / n_splits

    best_fold_idx = np.argmax(fold_scores)  # Find the fold with the highest accuracy
    best_model = fold_models[best_fold_idx]  # Retrieve the best model

    print(f"Average CV Accuracy: {np.mean(fold_scores):.4f}")
    print(f"Best Fold: {best_fold_idx + 1}, Accuracy: {fold_scores[best_fold_idx]:.4f}")
    return oof_preds, feature_importances, best_model


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "random_state": 42
    }

    model = GradientBoostingClassifier(**params)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, valid_idx in kf.split(X, y):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        score = accuracy_score(y_valid, preds)
        scores.append(score)

    return np.mean(scores)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)


best_params = study.best_params
print("Best Hyperparameters:", best_params)


final_model = GradientBoostingClassifier(**best_params)
oof_preds, feature_importances, best_model = oof_gradient_boosting(final_model, X, y)

