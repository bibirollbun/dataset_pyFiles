import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt



df_train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")

# Features & Target
X = df_train.drop(columns=["id"])
y = X.pop("BeatsPerMinute")  # <-- assuming BeatsPerMinute is target

X_test = df_test.drop(columns=["id"])


def objective(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "n_estimators": 10000  # high, let early stopping decide
    }

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    rmse_scores = []

    for tr_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**params)

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=200),
                lgb.log_evaluation(100)
            ]
        )

        preds = model.predict(X_val, num_iteration=model.best_iteration_)
        rmse = mean_squared_error(y_val, preds, squared=False)
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)



print("ðŸ”Ž Running Optuna optimization...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=15)

print("âœ… Best params:", study.best_params)


final_model = lgb.LGBMRegressor(**study.best_params)

final_model.fit(
    X, y,
    eval_set=[(X, y)],
    callbacks=[lgb.early_stopping(stopping_rounds=250)]
)



lgb.plot_importance(final_model, max_num_features=20, importance_type="gain")
plt.show()



preds_test = final_model.predict(X_test, num_iteration=final_model.best_iteration_)
submission = pd.DataFrame({"id": df_test["id"], "BeatsPerMinute": preds_test})
submission.to_csv("submission.csv", index=False)
submission.head()


import seaborn as sns
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
final_model.fit(X_train, y_train)
y_val_pred = final_model.predict(X_val)

# Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_val, y=y_val_pred, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
plt.xlabel("Actual BeatsPerMinute")
plt.ylabel("Predicted BeatsPerMinute")
plt.title("Actual vs Predicted")
plt.show()

# Residual Plot
residuals = y_val - y_val_pred
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_val_pred, y=residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted BeatsPerMinute")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# -------------------------
# Helper Functions
# -------------------------
def plot_binned_features(df, numerical_cols, bin_sizes=[10, 15, 20]):
    n_features = len(numerical_cols)
    n_bins = len(bin_sizes)
    
    fig, axes = plt.subplots(n_features, n_bins + 1, figsize=(16, 4 * n_features), tight_layout=True)
    fig.suptitle('Distribution of Original vs. Binned Features', fontsize=18, y=1.02)
    
    for i, col in enumerate(numerical_cols):
        ax_orig = axes[i, 0]
        sns.histplot(data=df, x=col, kde=True, ax=ax_orig, color='skyblue')
        ax_orig.set_title(f'Original: {col}')
        
        for j, size in enumerate(bin_sizes):
            binned_col = f'{col}_bin_{size}'
            df[binned_col] = pd.cut(df[col], bins=size, labels=False)
            ax_bin = axes[i, j + 1]
            sns.histplot(data=df, x=binned_col, discrete=True, ax=ax_bin, color='salmon')
            ax_bin.set_title(f'Binned: {binned_col}')
    
    plt.show()

def plot_numerical_features(train_df, test_df, numerical_cols):
    n_features = len(numerical_cols)
    fig, axes = plt.subplots(n_features, 2, figsize=(16, 4 * n_features), tight_layout=True)
    fig.suptitle('Distribution of Numerical Features (Train vs. Test)', fontsize=16, y=1.02)
    
    if n_features == 1:
        axes = np.array([axes])
        
    for i, col in enumerate(numerical_cols):
        ax_train = axes[i, 0]
        sns.histplot(data=train_df, x=col, kde=True, ax=ax_train, color='skyblue')
        ax_train.set_title(f'Train: {col}')
        
        ax_test = axes[i, 1]
        sns.histplot(data=test_df, x=col, kde=True, ax=ax_test, color='salmon')
        ax_test.set_title(f'Test: {col}')
    
    plt.show()




# Example usage
numerical_cols = X.columns.tolist()
plot_binned_features(df_train, numerical_cols[:3])  # first 3 features only for readability
plot_numerical_features(df_train, df_test, numerical_cols[:3])

