import warnings
warnings.simplefilter('ignore')


import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('/kaggle/input/predicting-the-price-of-diamond/train.csv')
test = pd.read_csv('/kaggle/input/predicting-the-price-of-diamond/test.csv')
orig = pd.read_csv('/kaggle/input/diamonds/diamonds.csv')
print('Train Shape:', train.shape)
print('Test Shape:', test.shape)
print('Orig Shape:', orig.shape)
train.head(3)


TARGET = 'price'
CATS = ['cut', 'color', 'clarity']
NUMS = ['carat', 'depth', 'table', 'x', 'y', 'z']
FEATURES = NUMS + CATS
print(len(FEATURES), 'Features:', FEATURES)


from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor


X = train[FEATURES].copy()
y = train[TARGET]
X_test = test[FEATURES].copy()


%%time
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

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
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.8,
        enable_categorical=True, 
        device='cpu'
    )

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=100,
              verbose=500)

    val_preds = model.predict(X_val)

    test_preds += model.predict(X_test) / N_SPLITS
    oof_preds[val_idx] = val_preds

    r2 = r2_score(y_val, val_preds)
    print(f"  R2 Score for Fold {fold+1}: {r2:.4f}")

print("--- Cross-Validation Finished ---\n")
overall_r2 = r2_score(y, oof_preds)
overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))

print(f"Overall OOF R2 Score: {overall_r2:.4f}")
print(f"Overall OOF RMSE: {overall_rmse:.4f}")


importances = model.feature_importances_
feature_names = FEATURES

feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

plt.figure(figsize=(12, 10))
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20))
plt.title('Top 20 Feature Importances (Last Fold Model)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


pd.DataFrame({'id': train.id, TARGET: oof_preds}).to_csv('oof_xgb_no_orig.csv', index=False)
pd.DataFrame({'id': test.id, TARGET: test_preds}).to_csv('test_xgb_no_orig.csv', index=False)


X = train[FEATURES].copy()
y = train[TARGET]
X_test = test[FEATURES].copy()

X_orig = orig[FEATURES].copy()
y_orig = orig[TARGET]


N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds_orig = np.zeros(len(train))
test_preds_orig = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Fold {fold+1}/{N_SPLITS}")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    X_train_aug = pd.concat([X_train, X_orig], ignore_index=True)
    y_train_aug = pd.concat([y_train, y_orig], ignore_index=True)

    X_train_aug[CATS] = X_train_aug[CATS].astype('category')
    X_val[CATS] = X_val[CATS].astype('category')
    X_test[CATS] = X_test[CATS].astype('category')

    model = XGBRegressor(
        random_state=42,
        n_estimators=10000,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.8,
        enable_categorical=True,
        device='cpu' # 'cuda' for GPU, 'cpu' for CPU
    )
    
    model.fit(X_train_aug, y_train_aug,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=100,
              verbose=500)
    
    val_preds = model.predict(X_val)
    test_preds_orig += model.predict(X_test) / N_SPLITS
    oof_preds_orig[val_idx] = val_preds

    r2 = r2_score(y_val, val_preds)
    print(f"  R2 Score for Fold {fold+1}: {r2:.4f}")

print("--- Cross-Validation Finished ---\n")
overall_r2_orig = r2_score(y, oof_preds_orig)
overall_rmse_orig = np.sqrt(mean_squared_error(y, oof_preds_orig))
print(f"Overall OOF R2 Score (With Orig Data): {overall_r2_orig:.4f}")
print(f"Overall OOF RMSE (With Orig Data): {overall_rmse_orig:.4f}")


importances = model.feature_importances_
feature_names = FEATURES

feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

plt.figure(figsize=(12, 10))
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20))
plt.title('Top 20 Feature Importances (Last Fold Model)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


pd.DataFrame({'id': train.id, TARGET: oof_preds_orig}).to_csv('oof_xgb_orig_row.csv', index=False)
pd.DataFrame({'id': test.id, TARGET: test_preds_orig}).to_csv('test_xgb_orig_row.csv', index=False)


X = train[FEATURES].copy()
y = train[TARGET]
X_test = test[FEATURES].copy()


new_feature_cols = []
for col in FEATURES:
    target_mean_map = orig.groupby(col)[TARGET].mean()
    
    new_col_name = f'orig_te_{col}'
    
    X[new_col_name] = X[col].map(target_mean_map)
    X_test[new_col_name] = X_test[col].map(target_mean_map)
    
    global_mean = orig[TARGET].mean()
    X[new_col_name].fillna(global_mean, inplace=True)
    X_test[new_col_name].fillna(global_mean, inplace=True)
    
    new_feature_cols.append(new_col_name)
    print(f"Created TE feature: {new_col_name}")

FEATURES_WITH_TE = FEATURES + new_feature_cols
print(f"\nTotal number of features is now: {len(FEATURES_WITH_TE)}")


N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds_te = np.zeros(len(train))
test_preds_te = np.zeros(len(test))

X_train_data = X[FEATURES_WITH_TE]
X_test_data = X_test[FEATURES_WITH_TE]

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_data, y)):
    print(f"Fold {fold+1}/{N_SPLITS}")
    X_train, y_train = X_train_data.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X_train_data.iloc[val_idx], y.iloc[val_idx]

    X_train[CATS] = X_train[CATS].astype('category')
    X_val[CATS] = X_val[CATS].astype('category')
    X_test_data[CATS] = X_test_data[CATS].astype('category')
    
    model = XGBRegressor(
        random_state=42,
        n_estimators=10000,
        learning_rate=0.01,
        max_depth=3,
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
    test_preds_te += model.predict(X_test_data) / N_SPLITS
    oof_preds_te[val_idx] = val_preds

    r2 = r2_score(y_val, val_preds)
    print(f"  R2 Score for Fold {fold+1}: {r2:.4f}")

print("--- Cross-Validation Finished ---\n")
overall_r2_te = r2_score(y, oof_preds_te)
overall_rmse_te = np.sqrt(mean_squared_error(y, oof_preds_te))
print(f"Overall OOF R2 Score (With Target Encoding): {overall_r2_te:.4f}")
print(f"Overall OOF RMSE (With Target Encoding): {overall_rmse_te:.4f}")


importances = model.feature_importances_
feature_names = FEATURES_WITH_TE

feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

plt.figure(figsize=(12, 10))
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20))
plt.title('Top 20 Feature Importances (Last Fold Model)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


pd.DataFrame({'id': train.id, TARGET: oof_preds_te}).to_csv('oof_xgb_orig_col.csv', index=False)
pd.DataFrame({'id': test.id, TARGET: test_preds_te}).to_csv('test_xgb_orig_col.csv', index=False)

