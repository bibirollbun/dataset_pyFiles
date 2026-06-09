import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import optuna


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv',index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv',index_col='id')
submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


train.head()


train.info()


train.describe()


num_df = train.select_dtypes(include=np.number)
cat_df = train.select_dtypes(include='object')


for col in num_df.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=train,x=col,hue='Personality')


for col in cat_df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=cat_df,x=col)


sns.heatmap(num_df.corr(),annot=True)


le = LabelEncoder()
train["Personality_encoded"] = le.fit_transform(train["Personality"])


X = train.drop(columns=["Personality", "Personality_encoded"])
y = train["Personality_encoded"]


combined = pd.concat([X, test], axis=0)
cat_cols = combined.select_dtypes(include="object").columns.tolist()
encoder = OrdinalEncoder()
combined[cat_cols] = encoder.fit_transform(combined[cat_cols])

X = combined.iloc[:len(X)].reset_index(drop=True)
test = combined.iloc[len(X):].reset_index(drop=True)


X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nFull Training set size for Optuna: {len(X_train_full)} samples")
print(f"Held-out Test set size for final evaluation: {len(X_test)} samples")


def objective(trial):
    """
    Objective function for Optuna to optimize XGBoost hyperparameters.
    Uses StratifiedKFold for cross-validation.
    """
    
    param = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000), 
        "max_depth": trial.suggest_int("max_depth", 3, 9), 
        "eta": trial.suggest_float("eta", 1e-3, 0.3, log=True), 
        "subsample": trial.suggest_float("subsample", 0.6, 1.0), 
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0), 
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10), 
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True), 
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True), 
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "early_stopping_rounds": 50, 
    }

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full, y_train_full)):
        X_train_fold, X_val_fold = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_train_fold, y_val_fold = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

        model = xgb.XGBClassifier(**param, use_label_encoder=False)
        model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_val_fold, y_val_fold)],
                  verbose=False) 

        y_pred_fold = model.predict(X_val_fold)
        accuracies.append(accuracy_score(y_val_fold, y_pred_fold))


    return np.mean(accuracies)


print("\nStarting Optuna Hyperparameter Optimization...")
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50) 

print("\nOptuna optimization finished.")
print(f"Number of finished trials: {len(study.trials)}")
print(f"Best trial: {study.best_trial.value:.4f} (Accuracy)")
print("Best hyperparameters found:")
best_params = study.best_trial.params
for key, value in best_params.items():
    print(f"  {key}: {value}")


print("\nTraining final model with best hyperparameters...")
final_xgb_model = xgb.XGBClassifier(**best_params, objective='binary:logistic', eval_metric='logloss',
                                    use_label_encoder=False, random_state=42)
final_xgb_model.fit(X_train_full, y_train_full)


print("\nPerforming optimal threshold adjustment on the held-out test set...")

y_test_pred_proba = final_xgb_model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.0, 1.01, 0.01)
accuracies_test = []
precisions_test = []
recalls_test = []
f1_scores_test = []

for threshold in thresholds:
    y_pred_thresholded = (y_test_pred_proba >= threshold).astype(int)
    accuracies_test.append(accuracy_score(y_test, y_pred_thresholded))
    precisions_test.append(precision_score(y_test, y_pred_thresholded, zero_division=0))
    recalls_test.append(recall_score(y_test, y_pred_thresholded, zero_division=0))
    f1_scores_test.append(f1_score(y_test, y_pred_thresholded, zero_division=0))



plt.figure(figsize=(8, 5))
plt.plot(thresholds, accuracies_test, label='Accuracy', color='blue')
plt.plot(thresholds, precisions_test, label='Precision', color='green', linestyle='--')
plt.plot(thresholds, recalls_test, label='Recall', color='red', linestyle='--')
plt.plot(thresholds, f1_scores_test, label='F1-Score', color='purple', linestyle=':')

plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Final Model Performance Metrics vs. Probability Threshold on Test Set')
plt.legend()
plt.grid(True)
plt.show()



optimal_threshold_idx_test = np.argmax(accuracies_test)
optimal_threshold_final = thresholds[optimal_threshold_idx_test]
max_accuracy_final = accuracies_test[optimal_threshold_idx_test]

print(f"\nOptimal Threshold for Accuracy on Test Set: {optimal_threshold_final:.2f}")
print(f"Maximum Accuracy achieved on Test Set: {max_accuracy_final:.4f}")

y_test_pred_default = (y_test_pred_proba >= 0.5).astype(int)
default_test_accuracy = accuracy_score(y_test, y_test_pred_default)
print(f"Test Accuracy with default threshold (0.5): {default_test_accuracy:.4f}")

print("\n--- Final Model Predictions with Optimal Threshold ---")
final_predictions = (y_test_pred_proba >= optimal_threshold_final).astype(int)
print("First 10 actual labels:", y_test.head(10).tolist())
print("First 10 optimal predictions:", final_predictions[:10].tolist())



test_preds = final_xgb_model.predict_proba(test)[:, 1]
final_preds = (test_preds > optimal_threshold_final).astype(int)
submission["Personality"] = le.inverse_transform(final_preds)
submission.to_csv("submission.csv", index=False)
submission.head()

