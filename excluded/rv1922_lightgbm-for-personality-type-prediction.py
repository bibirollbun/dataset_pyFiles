import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
import optuna
from catboost import CatBoostClassifier, Pool
import warnings
from sklearn.metrics import roc_curve
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


train.head()


train.info()


cat_cols = [col for col in train.select_dtypes(include=['object', 'category']).columns if col != "Personality"]
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])


le = LabelEncoder()
train['Personality'] = le.fit_transform(train['Personality'])


train.head()


cat_feats = ["Stage_fear", "Drained_after_socializing", "Personality"]


X = train.drop(['Personality', 'id'], axis=1)
y = train['Personality']
X_test_final = test.drop(columns=["id"])


def objective(trial):
    # Hyperparameter search space
    params = {
        'iterations': trial.suggest_int('iterations', 1000, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-4, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 5.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),
        'task_type': 'GPU',
        'devices': '0',
        'verbose': False,
        'allow_writing_files': False
    }

    # Split the data once for consistency across trials
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    train_pool = Pool(X_train, y_train)
    valid_pool = Pool(X_valid, y_valid)

    model = CatBoostClassifier(**params)
    
    model.fit(
        train_pool,
        eval_set=valid_pool,
        early_stopping_rounds=100,
        use_best_model=True,
    )
    
    preds = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, preds)
    
    return auc  # Optuna will maximize this


'''
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name='catboost-gpu-earlystop'
    )
study.optimize(objective, n_trials=100, timeout=60*60)'''


#print("Best AUC:   ", study.best_value)
#print("Best params:", study.best_trial.params)


test_ids = test['id']
X_final_test = test.drop(['id'], axis=1)


params = {
    'iterations': 1801,
    'learning_rate': 0.03369834440916496,
    'depth': 3,
    'l2_leaf_reg': 0.07418785693594356,
    'bagging_temperature': 0.9791465941479747,
    'border_count': 78,
    'scale_pos_weight': 2.0516656140482588,
    'random_strength': 0.26964285169499846,
    'task_type': 'GPU',
    'devices': '0',
    'verbose': False,
    'allow_writing_files': False
}


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Arrays to accumulate OOF and test predictions
oof_preds   = np.zeros(len(X))
oof_proba   = np.zeros(len(X))
test_preds  = np.zeros(len(X_test_final))

# Lists to store perâ€�fold metrics
fold_accuracies = []
fold_roc_aucs   = []

# ===== training loop =====
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"--- Fold {fold} ---")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train)

    # Predictions on validation
    val_preds = model.predict(X_val)
    val_proba = model.predict_proba(X_val)[:, 1]

    # Accumulate OOF
    oof_preds[val_idx] = val_preds
    oof_proba[val_idx] = val_proba

    # Compute metrics for this fold
    acc   = accuracy_score(y_val, val_preds)
    auc   = roc_auc_score(y_val, val_proba)
    fold_accuracies.append(acc)
    fold_roc_aucs.append(auc)

    print(f"Accuracy: {acc:.4f} | ROC AUC: {auc:.4f}")

    # Predict on the test set
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
print(f" â†’ Mean Accuracy: {mean_acc:.4f} Â± {std_acc:.4f}")
print(f"Fold ROC AUCs:   {fold_roc_aucs}")
print(f" â†’ Mean ROC AUC: {mean_auc:.4f} Â± {std_auc:.4f}")

print("\n=== OOF (all folds) ===")
print(f"Overall OOF Accuracy: {oof_acc:.4f}")
print(f"Overall OOF ROC AUC:  {oof_auc:.4f}")


conf_matrix = confusion_matrix(y, oof_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (OOF)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()


submission["Personality"] = le.inverse_transform(final_preds)
submission.to_csv("submission.csv", index=False)
print("âœ… Submission file saved as 'submission.csv'")


submission.head()

