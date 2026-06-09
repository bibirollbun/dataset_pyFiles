import warnings
warnings.simplefilter('ignore')


import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('/kaggle/input/predicting-the-price-of-diamond/train.csv')
orig = pd.read_csv('/kaggle/input/diamonds/diamonds.csv')

print('Train Shape:', train.shape)
print('Original Shape:', orig.shape)


FEATURES = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
CATS = ['cut', 'color', 'clarity']
TARGET = 'price'


train['is_orig'] = 0
orig['is_orig'] = 1

combined_df = pd.concat([
    train[FEATURES + ['is_orig']],
    orig[FEATURES + ['is_orig']]
], ignore_index=True)

combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

print('\nCombined Shape for Adversarial Validation:', combined_df.shape)
combined_df.head()


X = combined_df[FEATURES].copy()
y = combined_df['is_orig']


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
feature_importances = pd.DataFrame(index=FEATURES)


print("\n--- Starting Adversarial Validation ---")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}/{N_SPLITS}")

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    X_train[CATS] = X_train[CATS].astype('category')
    X_val[CATS] = X_val[CATS].astype('category')

    model = XGBClassifier(
        random_state=42,
        n_estimators=2000,      
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='auc',          
        enable_categorical=True,
        device='cpu'
    )

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=50,
              verbose=0) 

    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    
    feature_importances[f'fold_{fold+1}'] = model.feature_importances_

    auc = roc_auc_score(y_val, val_preds)
    print(f"  AUC Score for Fold {fold+1}: {auc:.4f}")

print("--- Adversarial Validation Finished ---\n")
overall_auc = roc_auc_score(y, oof_preds)

print(f"Overall OOF AUC Score: {overall_auc:.4f}")

if overall_auc > 0.8:
    print("\n[Conclusion] AUC > 0.8: The datasets are quite DIFFERENT.")
elif overall_auc > 0.6:
    print("\n[Conclusion] 0.6 < AUC <= 0.8: The datasets have some differences.")
else:
    print("\n[Conclusion] AUC <= 0.6: The datasets are very SIMILAR.")


feature_importances['mean'] = feature_importances.mean(axis=1)
feature_importances.sort_values('mean', ascending=False, inplace=True)

plt.figure(figsize=(10, 6))
sns.barplot(x='mean', y=feature_importances.index, data=feature_importances)
plt.title('Feature Importances in Adversarial Validation')
plt.xlabel('Mean Importance')
plt.ylabel('Features')
plt.grid(True, axis='x')
plt.show()

print("\nTop 5 features used to distinguish the datasets:")
print(feature_importances['mean'].head())


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


X = orig[FEATURES].copy()
y = orig[TARGET]


%%time
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(orig))

print("\n--- Starting Cross-Validation ---")
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Fold {fold+1}/{N_SPLITS}")

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    X_train[CATS] = X_train[CATS].astype('category')
    X_val[CATS] = X_val[CATS].astype('category')

    model = XGBRegressor(
        random_state=42,
        n_estimators=10000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True, 
        device='cuda'
    )

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=100,
              verbose=500)

    val_preds = model.predict(X_val)
    oof_preds[val_idx] = val_preds

    r2 = r2_score(y_val, val_preds)
    print(f"  R2 Score for Fold {fold+1}: {r2:.4f}")

print("--- Cross-Validation Finished ---\n")
overall_r2 = r2_score(y, oof_preds)
overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))

print(f"Overall OOF R2 Score: {overall_r2:.4f}")
print(f"Overall OOF RMSE: {overall_rmse:.4f}")


X = pd.concat([train[FEATURES], orig[FEATURES]]).copy()
y = pd.concat([train[TARGET], orig[TARGET]])


%%time
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))

print("\n--- Starting Cross-Validation ---")
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Fold {fold+1}/{N_SPLITS}")

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    X_train[CATS] = X_train[CATS].astype('category')
    X_val[CATS] = X_val[CATS].astype('category')

    model = XGBRegressor(
        random_state=42,
        n_estimators=10000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True, 
        device='cpu'
    )

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=100,
              verbose=500)

    val_preds = model.predict(X_val)
    oof_preds[val_idx] = val_preds

    r2 = r2_score(y_val, val_preds)
    print(f"  R2 Score for Fold {fold+1}: {r2:.4f}")

print("--- Cross-Validation Finished ---\n")
overall_r2 = r2_score(y, oof_preds)
overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))

print(f"Overall OOF R2 Score: {overall_r2:.4f}")
print(f"Overall OOF RMSE: {overall_rmse:.4f}")


X = orig[FEATURES].copy()
y = orig[TARGET]
X_test = train[FEATURES].copy()


%%time
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

print("\n--- Starting Cross-Validation ---")
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Fold {fold+1}/{N_SPLITS}")

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    X_train[CATS] = X_train[CATS].astype('category')
    X_val[CATS] = X_val[CATS].astype('category')
    X_test[CATS] = X_test[CATS].astype('category')

    model = XGBRegressor(
        random_state=42,
        n_estimators=10000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True, 
        device='cpu'
    )

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=100,
              verbose=500)

    val_preds = model.predict(X_val)
    test_preds += model.predict(X_test) / 5
    oof_preds[val_idx] = val_preds

    r2 = r2_score(y_val, val_preds)

print("--- Cross-Validation Finished ---\n")
overall_r2 = r2_score(train[TARGET], test_preds)
overall_rmse = np.sqrt(mean_squared_error(train[TARGET], test_preds))

print(f"Overall OOF R2 Score: {overall_r2:.4f}")
print(f"Overall OOF RMSE: {overall_rmse:.4f}")




