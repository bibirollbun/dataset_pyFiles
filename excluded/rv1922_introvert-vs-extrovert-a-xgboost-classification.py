pip install --upgrade scikit-learn


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
import xgboost as xgb
import optuna
import warnings
from sklearn.metrics import roc_curve
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
original = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_datasert.csv')


original.head()


train.head()


train.info()


train = train.drop(['id'], axis=1)


train = pd.concat([train, original], ignore_index=True)
train = train.drop_duplicates()


cat_cols = [col for col in train.select_dtypes(include=['object', 'category']).columns if col != "Personality"]
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])


le = LabelEncoder()
train['Personality'] = le.fit_transform(train['Personality'])


train.head()


X = train.drop(['Personality'], axis=1)
y = train['Personality']
X_test_final = test.drop(columns=["id"])


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1, random_state=42)


def objective(trial):
    # Suggest hyperparameters
    param = {
        'verbosity': 0,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',               # use AUC in training
        'tree_method': 'gpu_hist',          # if you have a GPU; else 'hist'
        'predictor': 'gpu_predictor',       # else 'cpu_predictor'
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }

    # Instantiate model
    model = xgb.XGBClassifier(**param, use_label_encoder=False)

    # 5‑fold stratified CV, scoring by AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        model, X, y,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )
    # return mean AUC
    return scores.mean()


#study = optuna.create_study(
#    direction="maximize",
#    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
#)

#study.optimize(objective, n_trials=50, timeout=3600)


#print("Best composite score:", study.best_value)
#print("Best trial params:")
#for key, val in study.best_params.items():
    #print(f"  {key}: {val}")


test_ids = test['id']
X_final_test = test.drop(['id'], axis=1)


best_params = {
    'n_estimators': 1650,
    'max_depth': 6,
    'learning_rate': 0.06543137487135454,
    'subsample': 0.6752474115840645,
    'colsample_bytree': 0.7631792786851659,
    'gamma': 2.242810732343882,
    'reg_alpha': 7.351432998635316,
    'reg_lambda': 0.012487325656562683,
    'min_child_weight': 7,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',            # or 'hist' if no GPU
    'use_label_encoder': False
}


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

oof_preds   = np.zeros(len(X))
oof_proba   = np.zeros(len(X))
test_preds  = np.zeros(len(X_test_final))

fold_accuracies = []
fold_roc_aucs   = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"--- Fold {fold} ---")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    val_proba = model.predict_proba(X_val)[:, 1]

    oof_preds[val_idx] = val_preds
    oof_proba[val_idx] = val_proba

    acc   = accuracy_score(y_val, val_preds)
    auc   = roc_auc_score(y_val, val_proba)
    fold_accuracies.append(acc)
    fold_roc_aucs.append(auc)

    print(f"Accuracy: {acc:.4f} | ROC AUC: {auc:.4f}")

    test_preds += model.predict_proba(X_test_final)[:, 1]

test_preds /= skf.n_splits
final_preds = (test_preds > 0.5).astype(int)

oof_acc = accuracy_score(y, oof_preds)
oof_auc = roc_auc_score(y, oof_proba)

mean_acc = np.mean(fold_accuracies)
std_acc  = np.std(fold_accuracies)
mean_auc = np.mean(fold_roc_aucs)
std_auc  = np.std(fold_roc_aucs)

print("\n=== CV Summary ===")
print(f"Fold Accuracies: {fold_accuracies}")
print(f" → Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
print(f"Fold ROC AUCs:   {fold_roc_aucs}")
print(f" → Mean ROC AUC: {mean_auc:.4f} ± {std_auc:.4f}")

print("\n=== OOF (all folds) ===")
print(f"Overall OOF Accuracy: {oof_acc:.4f}")
print(f"Overall OOF ROC AUC:  {oof_auc:.4f}")


folds = list(range(1, len(fold_accuracies) + 1))
df = pd.DataFrame({
    'fold': folds,
    'accuracy': fold_accuracies,
    'roc_auc': fold_roc_aucs
})


plt.figure(figsize=(8, 4))
sns.lineplot(x='fold', y='accuracy', data=df, marker='o')
plt.title('StratifiedKFold Accuracies by Fold')
plt.xlabel('Fold Number')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.xticks(folds)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 4))
sns.lineplot(x='fold', y='roc_auc', data=df, marker='o')
plt.title('StratifiedKFold ROC AUC by Fold')
plt.xlabel('Fold Number')
plt.ylabel('ROC AUC')
plt.ylim(0, 1)
plt.xticks(folds)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


conf_matrix = confusion_matrix(y, oof_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (OOF)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()


%%time
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, max_num_features=15, importance_type='gain', height=0.6)
plt.title('Top 15 Feature Importances')
plt.tight_layout()
plt.show()


submission["Personality"] = le.inverse_transform(final_preds)
submission.to_csv("submission.csv", index=False)
print("✅ Submission file saved as 'submission.csv'")


submission.head()

