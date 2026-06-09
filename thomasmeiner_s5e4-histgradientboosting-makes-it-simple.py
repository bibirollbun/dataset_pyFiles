!pip install --upgrade scikit-learn


import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, KFold


train = pd.read_csv(r'/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv(r'/kaggle/input/playground-series-s5e4/test.csv')
submission = pd.read_csv(r'/kaggle/input/playground-series-s5e4/sample_submission.csv')

target_col = "Listening_Time_minutes"


X = train.drop(columns=[target_col, "id"])  
y = train[target_col]

numeric_features = ["Episode_Length_minutes", 
                    "Host_Popularity_percentage", 
                    "Guest_Popularity_percentage", 
                    "Number_of_Ads"]

categorical_features = ["Podcast_Name", 
                        "Episode_Title", 
                        "Genre", 
                        "Publication_Day", 
                        "Publication_Time", 
                        "Episode_Sentiment"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", HistGradientBoostingRegressor(random_state=42))
])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    model_pipeline, 
    X, 
    y, 
    cv=kf, 
    scoring="neg_root_mean_squared_error"
)

rmse_scores = -scores
print(f"Cross-validated RMSE (mean): {rmse_scores.mean():.2f}")
print(f"Cross-validated RMSE (std):  {rmse_scores.std():.2f}")



X = train.drop(columns=[target_col, "id"])
y = train[target_col]

for col in X.select_dtypes(include="object"):
    X[col] = X[col].astype("category")


model = HistGradientBoostingRegressor(
    categorical_features="from_dtype",  # automatically detect categorical columns
    random_state=42
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    model,
    X,
    y,
    cv=kf,
    scoring="neg_root_mean_squared_error"
)

rmse_scores = -scores
print(f"Cross-validated RMSE (mean): {rmse_scores.mean():.2f}")
print(f"Cross-validated RMSE (std):  {rmse_scores.std():.2f}")



import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

import optuna
import time


X_full = train.drop(columns=[target_col, "id"])
y_full = train[target_col]

for col in X_full.select_dtypes(include="object"):
    X_full[col] = X_full[col].astype("category")


outer_kf = KFold(n_splits=5, shuffle=True, random_state=42)

outer_scores = []
outer_models = []

for fold_id, (train_idx, valid_idx) in enumerate(outer_kf.split(X_full)):
    print(f"\n=== Outer Fold {fold_id+1} / 5 ===")
    
    X_train, X_valid = X_full.iloc[train_idx], X_full.iloc[valid_idx]
    y_train, y_valid = y_full.iloc[train_idx], y_full.iloc[valid_idx]
    

    def objective(trial):
        """
        Given a trial, sample hyperparameters and evaluate
        via 5-fold CV on (X_train, y_train).
        Return the mean RMSE across inner folds.
        """
        # Hyperparameter search space:
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.2, log=True)
        max_iter = trial.suggest_int("max_iter", 50, 1000)
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 2, 64)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 100)
        l2_regularization = trial.suggest_float("l2_regularization", 1e-10, 1e3, log=True)
        
        model = HistGradientBoostingRegressor(
            categorical_features="from_dtype",
            random_state=42,
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization
        )
        
        inner_kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            scoring="neg_root_mean_squared_error",
            cv=inner_kf,
            n_jobs=-1
        )
        
        return -cv_scores.mean() 
    
    study = optuna.create_study(direction="minimize")
    study.optimize(
        objective,
        timeout=3600,     # 1 hour max
        n_trials=100     
    )
    
    best_params = study.best_params
    print("Best params:", best_params)
    
    best_model = HistGradientBoostingRegressor(
        categorical_features="from_dtype",
        random_state=42,
        **best_params
    )
    best_model.fit(X_train, y_train)
    
    y_pred_valid = best_model.predict(X_valid)
    mse_valid = mean_squared_error(y_valid, y_pred_valid)  # returns MSE
    rmse_valid = np.sqrt(mse_valid) 
    print(f"Outer Fold {fold_id+1} RMSE: {rmse_valid:.4f}")
    
    outer_scores.append(rmse_valid)
    outer_models.append(best_model)

mean_rmse = np.mean(outer_scores)
std_rmse = np.std(outer_scores)
print("\n=== Final Nested CV Results ===")
print(f"Mean RMSE across outer folds: {mean_rmse:.4f}")
print(f"Std  RMSE across outer folds: {std_rmse:.4f}")


X_test = test.drop(columns=["id"])
for col in X_test.select_dtypes(include="object"):
    X_test[col] = X_test[col].astype("category")

y_preds_ensemble = np.mean([m.predict(X_test) for m in outer_models], axis=0)

submission[target_col] = y_preds_ensemble
submission.to_csv("submission.csv", index=False)
submission




