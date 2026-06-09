import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
import xgboost as xgb

train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col='id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col='id')
original_df = pd.read_csv('/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv') 
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


print("Train")
display(train_df.head())

print("Test")
display(test_df.head())

print("Original")
display(original_df.head())


print(f"\nTrain: {train_df.shape}")
print(f"Test: {test_df.shape}")
print(f"Original: {original_df.shape}")

print("\nEksik veri oranları (train):")
print(train_df.isnull().mean().sort_values(ascending=False))

print("\nEksik veri oranları (test):")
print(test_df.isnull().mean().sort_values(ascending=False))

print("\nEksik veri oranları (original):")
print(original_df.isnull().mean().sort_values(ascending=False))


train_df['group'] = train_df.index // 365


original_df.columns = original_df.columns.str.strip()

if 'rainfall' in original_df.columns:
    original_df['rainfall'] = original_df['rainfall'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)


print(f"Birleştirme öncesi eğitim verisi satır sayısı: {train_df.shape[0]}")
print(f"Orijinal veri satır sayısı: {original_df.shape[0]}")


original_df_copy = original_df.copy()
original_df_copy['group'] = 6  

common_cols = list(set(train_df.columns) & set(original_df_copy.columns))
train_concat = pd.concat([train_df[common_cols], original_df_copy[common_cols]], axis=0)

print(f"Birleştirme sonrası eğitim verisi satır sayısı: {train_concat.shape[0]}")


FEATURE_COLS = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 
                'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
original_rainfall_mean = original_df['rainfall'].mean()
train_encoded = train_df.copy()
train_encoded = train_encoded.fillna(train_encoded.mean())

for col in FEATURE_COLS:
    if col in original_df.columns:
        target_means = original_df.groupby(col)['rainfall'].mean().to_dict()
        new_col_name = f"{col}_encoded"
        train_encoded[new_col_name] = train_encoded[col].map(target_means)
        train_encoded[new_col_name] = train_encoded[new_col_name].fillna(original_rainfall_mean)


def train_and_evaluate_model(model, X, y, groups, model_name, cv_folds=None):
    if cv_folds is None:
        cv_folds = GroupKFold(n_splits=6)
    oof_predictions = np.zeros(len(X))
    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(cv_folds.split(X, y, groups)):
        train_mask = groups.iloc[train_idx] != 6
        train_idx_filtered = train_idx[train_mask]
        X_train_fold = X.iloc[train_idx_filtered]
        y_train_fold = y.iloc[train_idx_filtered]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        model.fit(X_train_fold, y_train_fold)
        if hasattr(model, 'predict_proba'):
            val_pred = model.predict_proba(X_val_fold)[:, 1]
        else:
            val_pred = model.decision_function(X_val_fold)
        oof_predictions[val_idx] = val_pred
        val_mask = groups.iloc[val_idx] != 6
        if val_mask.sum() > 0:
            fold_score = roc_auc_score(y_val_fold[val_mask], val_pred[val_mask])
            cv_scores.append(fold_score)
            print(f"Fold {fold+1}: AUC = {fold_score:.5f}")
    non_group6_mask = groups != 6
    overall_cv = roc_auc_score(y[non_group6_mask], oof_predictions[non_group6_mask])
    print(f"{model_name} - Genel CV AUC: {overall_cv:.5f}")
    return oof_predictions, overall_cv, model, cv_scores

def make_test_predictions(model, X_test):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X_test)[:, 1]
    else:
        return model.decision_function(X_test)

def create_submission(predictions, filename, threshold=0.5):
    submission = sample_submission.copy()
    submission['rainfall'] = (predictions > threshold).astype(int)
    submission.to_csv(filename, index=False)
    print(f"✓ {filename} oluşturuldu")


print("="*50)
print("MODEL 1: XGBoost (Orijinal Veri Yeni Satırlar Olarak)")
print("="*50)

X_concat = train_concat.drop(['rainfall', 'group'], axis=1)
y_concat = train_concat['rainfall']
groups_concat = train_concat['group']

xgb_params = {
    'max_depth': 3,
    'colsample_bytree': 0.9,
    'subsample': 0.9,
    'eval_metric': 'auc',
    'objective': 'binary:logistic',
    'random_state': 42,
    'n_estimators': 100
}
xgb_model = xgb.XGBClassifier(**xgb_params)
xgb_oof, xgb_cv_score, xgb_trained, xgb_cv_scores = train_and_evaluate_model(
    xgb_model, X_concat, y_concat, groups_concat, "XGBoost"
)

test_for_xgb = test_df[X_concat.columns].copy().fillna(0)
xgb_test_pred = make_test_predictions(xgb_trained, test_for_xgb)
create_submission(xgb_test_pred, 'xgb_submission.csv')


print("="*50)
print("MODEL 2: SVM (Orijinal Veri Target Encoding ile)")
print("="*50)

X_encoded = train_encoded.drop(['rainfall', 'group'], axis=1)
y_encoded = train_encoded['rainfall']
groups_encoded = train_encoded['group']

svm_model = SVC(C=0.1, kernel='poly', degree=1, probability=True, random_state=42)
svm_oof, svm_cv_score, svm_trained, svm_cv_scores = train_and_evaluate_model(
    svm_model, X_encoded, y_encoded, groups_encoded, "SVM"
)

test_encoded = test_df.copy()
for col in FEATURE_COLS:
    if col in original_df.columns and col in test_df.columns:
        target_means = original_df.groupby(col)['rainfall'].mean().to_dict()
        new_col_name = f"{col}_encoded"
        test_encoded[new_col_name] = test_encoded[col].map(target_means)
        test_encoded[new_col_name] = test_encoded[new_col_name].fillna(original_rainfall_mean)
for col in X_encoded.columns:
    if col not in test_encoded.columns:
        test_encoded[col] = 0
test_encoded = test_encoded.fillna(0)

svm_test_pred = make_test_predictions(svm_trained, test_encoded[X_encoded.columns])
create_submission(svm_test_pred, 'svm_submission.csv')


print("="*50)
print("ENSEMBLE: XGBoost + SVM (Eşit Ağırlık)")
print("="*50)

ensemble_test_pred = 0.5 * xgb_test_pred + 0.5 * svm_test_pred
create_submission(ensemble_test_pred, 'submission.csv')

ensemble_oof_common = 0.5 * xgb_oof[:len(svm_oof)] + 0.5 * svm_oof
non_group6_mask_common = (groups_concat[:len(svm_oof)] != 6)
ensemble_cv_auc = roc_auc_score(y_concat[:len(svm_oof)][non_group6_mask_common], ensemble_oof_common[non_group6_mask_common])

print(f"Ensemble (XGBoost + SVM) OOF CV AUC: {ensemble_cv_auc:.5f}")


plt.figure(figsize=(12,5))
plt.plot(range(1, len(xgb_cv_scores)+1), xgb_cv_scores, marker='o', label='XGBoost')
plt.plot(range(1, len(svm_cv_scores)+1), svm_cv_scores, marker='o', label='SVM')
plt.xlabel('Fold')
plt.ylabel('AUC')
plt.title('Fold Bazında Model Performansı (AUC)')
plt.legend()
plt.grid(True)
plt.show()


model_names = ['XGBoost', 'SVM', 'Ensemble (XGB+SVM)']
cv_scores = [xgb_cv_score, svm_cv_score, ensemble_cv_auc]

plt.figure(figsize=(7,5))
bars = plt.bar(model_names, cv_scores, color=['skyblue', 'orange', 'green'])
plt.ylim(0.8, 1)
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.005, 
             f"{cv_scores[i]:.4f}", ha='center', fontsize=12, weight='bold')
plt.title('Model CV (AUC) Karşılaştırması')
plt.ylabel('CV AUC')
plt.show()



plt.figure(figsize=(10,5))
plt.hist(xgb_test_pred, bins=40, alpha=0.5, label='XGBoost', color='skyblue')
plt.hist(svm_test_pred, bins=40, alpha=0.5, label='SVM', color='orange')
plt.hist(ensemble_test_pred, bins=40, alpha=0.5, label='Ensemble', color='green')
plt.title('Test Tahminlerinin Dağılımı')
plt.xlabel('Tahmin (Probability)')
plt.ylabel('Frekans')
plt.legend()
plt.show()

