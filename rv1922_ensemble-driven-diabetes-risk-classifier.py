import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
original = pd.read_csv("/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv")
submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
print("Loading Data...")


train.head()


train['is_train'] = 1
test['is_train'] = 0
df = pd.concat([train, test], axis=0).reset_index(drop=True)


cat_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    df[col] = df[col].astype('category')


X = df[df['is_train'] == 1].drop(['diagnosed_diabetes', 'is_train'], axis=1)
y = df[df['is_train'] == 1]['diagnosed_diabetes']
test_data = df[df['is_train'] == 0].drop(['diagnosed_diabetes', 'is_train'], axis=1)


class_counts = y.value_counts()
total = len(y)
positive_class = class_counts[1]
negative_class = class_counts[0]
scale_pos_weight = negative_class / positive_class
print(f"\nCalculated scale_pos_weight: {scale_pos_weight:.2f}")


best_params = {
    'learning_rate': 0.059216255749261655,
    'num_leaves': 26,
    'max_depth': 4,
    'lambda_l1': 1.3404844864067962,
    'lambda_l2': 3.1381681073903975e-07,
    'min_child_samples': 95,
    'subsample': 0.9745291249731525,
    'colsample_bytree': 0.5645863195919457,
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'n_jobs': -1,
    'random_state': 42,
    'n_estimators': 5000,
    'scale_pos_weight': scale_pos_weight 
}


print("--- Training Final Model (10-Fold CV) ---")

FOLDS = 10
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test_data))

# book-keeping for plots
fold_aucs = []
fold_best_iters = []
fold_eval_results = []      
fold_fprs = []             
fold_tprs = []              
fold_labels = []

start_time = time.time()
for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y), 1):
    t0 = time.time()
    print(f"\n--- Fold {fold}/{FOLDS} ---")
    X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]
    
    model = lgb.LGBMClassifier(**best_params)
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(0)
    ]
    
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=callbacks
    )
    
    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    test_preds += model.predict_proba(test_data)[:, 1] / FOLDS
    
    auc_score = roc_auc_score(y_val, val_preds)
    fold_aucs.append(auc_score)
    
    best_iter = getattr(model, 'best_iteration_', None)
    fold_best_iters.append(best_iter)
    
    evals_result = {}
    try:
        evals_result = model.evals_result_  # dict: {'valid_0': {'auc': [..]}}
        val_auc_history = evals_result.get('valid_0', {}).get('auc', [])
    except Exception:
        val_auc_history = []
    fold_eval_results.append(val_auc_history)
    
    # ROC curve points for validation set
    fpr, tpr, _ = roc_curve(y_val, val_preds)
    fold_fprs.append(fpr)
    fold_tprs.append(tpr)
    fold_labels.append(f'Fold {fold} (AUC={auc_score:.5f})')
    
    # Feature importances (gain if available, else split)
    try:
        fi = model.booster_.feature_importance(importance_type='gain')
    except Exception:
        fi = model.feature_importances_
    fold_feature_importances.append(fi)
    
    elapsed = time.time() - t0
    print(f"Fold {fold} AUC: {auc_score:.5f}")
    print(f"Fold {fold} best_iteration: {best_iter}")
    print(f"Fold {fold} time: {elapsed:.1f}s")

total_elapsed = time.time() - start_time
# Final OOF AUC
final_oof_auc = roc_auc_score(y, oof_preds)
print(f"\nFinal OOF AUC: {final_oof_auc:.5f}")
print(f"Total training time: {total_elapsed:.1f}s")


plt.figure(figsize=(10, 8))
# 1) ROC curves per fold
for fpr, tpr, lbl in zip(fold_fprs, fold_tprs, fold_labels):
    plt.plot(fpr, tpr, label=lbl)
plt.plot([0,1],[0,1],'--', linewidth=0.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Per-Fold ROC curves (validation)')
plt.legend(loc='lower right', fontsize='small')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 8))
for idx, auc_hist in enumerate(fold_eval_results, 1):
    if len(auc_hist) == 0:
        continue
    iters = np.arange(1, len(auc_hist)+1)
    plt.plot(iters, auc_hist, label=f"Fold {idx} (best_iter={fold_best_iters[idx-1]})")
plt.xlabel('Iteration')
plt.ylabel('Validation AUC')
plt.title('Learning Curves (validation AUC per iteration)')
plt.legend(loc='lower right', fontsize='small')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,5))
plt.hist(oof_preds, bins=50)
plt.xlabel('OOF predicted probability')
plt.ylabel('Count')
plt.title('Distribution of OOF predicted probabilities')
plt.tight_layout()
plt.show()


summary_df = pd.DataFrame({
    'fold': np.arange(1, FOLDS+1),
    'val_auc': fold_aucs,
    'best_iteration': fold_best_iters
})
print("\nPer-fold summary:")
print(summary_df.to_string(index=False))

print(f"\nFinal OOF AUC: {final_oof_auc:.5f}")


submission = pd.DataFrame({
    "id": pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")["id"],
    "diagnosed_diabetes": test_preds
})

submission_filename = "submission.csv"
submission.to_csv(submission_filename, index=False)
print(f"Saved {submission_filename}")


submission.head()

