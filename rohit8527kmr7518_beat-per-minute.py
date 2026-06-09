import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import PowerTransformer, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool

from warnings import filterwarnings
filterwarnings('ignore')


df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
print(f"shape:{df.shape}")
df.head(5)


overview = df.describe().T
overview['missing'] = df.isnull().sum()
overview['unique'] = df.nunique()
print(overview)


numeric_cols = df.columns.drop('id')  # exclude ID
plt.figure(figsize=(20,12))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 4, i)
    sns.histplot(df[col], bins=50, kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()


plt.figure(figsize=(20,12))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 4, i)
    sns.boxplot(x=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()


corr = df[numeric_cols].corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()


corr = df.corr()['BeatsPerMinute'].sort_values(ascending=False)
print(corr)


sns.pairplot(df.sample(5000), vars=['RhythmScore','Energy','MoodScore','TrackDurationMs','BeatsPerMinute'])
plt.show()


# Separate features and target
X = df.drop(["id", "BeatsPerMinute"], axis=1)
y = df["BeatsPerMinute"]
X_test = test_df.drop("id", axis=1)

# Define columns for preprocessing
skewed_features = ["VocalContent", "AcousticQuality", "InstrumentalScore", "LivePerformanceLikelihood"]
standardize_features = ["AudioLoudness"]

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("skewed", PowerTransformer(method="yeo-johnson"), skewed_features),
        ("standardize", StandardScaler(), standardize_features),
    ],
    remainder="passthrough"  # Keep other columns unchanged
)

# Fit and transform training data; transform test data
X_processed = preprocessor.fit_transform(X)
X_test_processed = preprocessor.transform(X_test)

# Retrieve transformed feature names
feature_names = preprocessor.get_feature_names_out()

# Convert arrays back to DataFrame for easier handling
X_processed = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)

X_processed.head()


models = {
    "xgb": XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    ),
    "lgbm": LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ),
    "cat": CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        verbose=0,
        random_state=42
    )
}


# Define 5-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize dictionaries for OOF predictions, test predictions, and CV scores
oof_predictions = {name: np.zeros(len(X_processed)) for name in models.keys()}
test_predictions = {name: np.zeros((len(X_test_processed), kf.get_n_splits())) for name in models.keys()}
cv_scores = {name: [] for name in models.keys()}


# # K-Fold Cross-validation training
# for model_name, model in models.items():
#     print(f"Training {model_name}...")
    
#     for fold, (train_idx, val_idx) in enumerate(kf.split(X_processed, y)):
#         X_train, X_val = X_processed.iloc[train_idx], X_processed.iloc[val_idx]
#         y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
#         model.fit(X_train, y_train)
        
#         oof_predictions[model_name][val_idx] = model.predict(X_val)

#         test_predictions[model_name][:, fold] = model.predict(X_test_processed)
        
#         fold_rmse = mean_squared_error(y_val, oof_predictions[model_name][val_idx], squared=False)
#         cv_scores[model_name].append(fold_rmse)
#         print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}")
    
#     print(f"{model_name} CV RMSE: {np.mean(cv_scores[model_name]):.4f}\n")


# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# def objective(trial):
#     # Hyperparameter search space
#     params = {
#         "iterations": 3000,  # high iterations with early stopping
#         "depth": trial.suggest_int("depth", 5, 12),
#         "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.08),
#         "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1, 20),
#         "bagging_temperature": trial.suggest_uniform("bagging_temperature", 0, 2),
#         "border_count": trial.suggest_int("border_count", 32, 255),
#         "random_strength": trial.suggest_uniform("random_strength", 1e-9, 10),
#         "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
#         "eval_metric": "RMSE",
#         "task_type": "GPU",
#         "devices": "0:1",  # safely use both T4 GPUs
#         "verbose": 0,
#     }

#     fold_rmse = []

#     # 5-Fold CV
#     for train_idx, val_idx in kf.split(X_processed, y):
#         X_train, X_val = X_processed.iloc[train_idx], X_processed.iloc[val_idx]
#         y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#         train_pool = Pool(X_train, y_train)
#         val_pool = Pool(X_val, y_val)

#         model = CatBoostRegressor(**params, early_stopping_rounds=200)
#         model.fit(train_pool, eval_set=val_pool)

#         preds = model.predict(X_val)
#         rmse = mean_squared_error(y_val, preds, squared=False)
#         fold_rmse.append(rmse)

#     # Average RMSE across folds
#     return np.mean(fold_rmse)


# # Create Optuna study
# study = optuna.create_study(direction="minimize", study_name="catboost_optuna_gpu_sequential")

# # Run trials SEQUENTIALLY (no n_jobs) to avoid GPU conflicts
# study.optimize(objective, n_trials=150)  # Increase n_trials for better tuning

# # Best result
# print("Best RMSE:", study.best_value)
# print("Best hyperparameters:", study.best_params)



best_params = {
    'depth': 12,
    'learning_rate': 0.03999699823212053,
    'l2_leaf_reg': 9.49661933298946,
    'bagging_temperature': 0.373103774520797,
    'border_count': 118,
    'random_strength': 4.039987340929386,
    'grow_policy': 'Lossguide',
    'task_type': 'GPU',
    'devices': '0:1',   # use both T4 GPUs
    'verbose': 50
}


train_pool = Pool(X_processed, y)

x_test_processed = preprocessor.transform(test_df.drop("id", axis=1))
x_test_processed = pd.DataFrame(x_test_processed, columns=feature_names, index=test_df.index)
test_pool = Pool(x_test_processed)


# ======= Train final CatBoost model =======
final_model = CatBoostRegressor(
    iterations=5000,
    early_stopping_rounds=300,
    **best_params
)
final_model.fit(train_pool)


final_test_preds = final_model.predict(test_pool)


submission = pd.DataFrame({
    "id": test_df["id"],
    "BeatsPerMinute": final_test_preds
})

submission.to_csv("submission.csv", index=False)
print("Submission saved as submission.csv!")


submission.to_csv("/kaggle/working/submission.csv", index=False)


submission_file = pd.read_csv('/kaggle/working/submission.csv')
submission.head()




