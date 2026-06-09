from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("/kaggle/input/black-jack-smart-effect-of-removal-ml/train.csv")
test = pd.read_csv("/kaggle/input/black-jack-smart-effect-of-removal-ml/test.csv")

X = train.drop(["ev", "id"], axis=1).copy()
y = train["ev"].copy()
X_test = test.drop(columns='id').copy()

FOLDS = 7
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_pred = np.zeros(len(X))
fold_mse = []
test_preds = np.zeros((len(X_test), FOLDS))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = XGBRegressor(
        n_estimators=10000,
        learning_rate=0.02,
        early_stopping_rounds=100,
        colsample_bytree=0.5,
        subsample=0.8,
        random_state=42,
        verbosity=0
    )
    
    model.fit(
        X_train, y_train,
        #early_stopping_rounds=100,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    y_pred = model.predict(X_val)
    mse_fold = mean_squared_error(y_val, y_pred)
    fold_mse.append(mse_fold)
    oof_pred[val_idx] = y_pred
    
    test_preds[:, fold - 1] = model.predict(X_test)

overall_mse = mean_squared_error(y, oof_pred)
print(f"\nOverall OOF MSE: {overall_mse:.8f}")

final_test_pred = test_preds.mean(axis=1)


importance_weight = model.get_booster().get_score(importance_type='weight')
importance_weight = {k: v for k, v in sorted(importance_weight.items(), key=lambda item: item[1], reverse=True)}

importance_gain = model.get_booster().get_score(importance_type='gain')
importance_gain = {k: v for k, v in sorted(importance_gain.items(), key=lambda item: item[1], reverse=True)}

importance_cover = model.get_booster().get_score(importance_type='cover')
importance_cover = {k: v for k, v in sorted(importance_cover.items(), key=lambda item: item[1], reverse=True)}

df_importance_weight = pd.DataFrame({'feature': list(importance_weight.keys()), 
                                    'importance': list(importance_weight.values())})
df_importance_gain = pd.DataFrame({'feature': list(importance_gain.keys()), 
                                  'importance': list(importance_gain.values())})
df_importance_cover = pd.DataFrame({'feature': list(importance_cover.keys()), 
                                   'importance': list(importance_cover.values())})

for df in [df_importance_weight, df_importance_gain, df_importance_cover]:
    df['importance'] = df['importance'] / df['importance'].sum()

plt.figure(figsize=(12, 6))
plt.barh(df_importance_gain['feature'], df_importance_gain['importance'])
plt.xlabel('Importance (Gain)')
plt.title('XGBoost Feature Importance (Gain)')
plt.gca().invert_yaxis()  # Display highest importance at the top
plt.tight_layout()
plt.show()

df_corr = train.drop(columns='id').copy()
corr = df_corr.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation")
plt.show()


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Standardize the data (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run PCA without limiting components to see explained variance
pca = PCA()
pca.fit(X_scaled)

# Plot explained variance ratio
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.7)
plt.step(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), where='mid', color='red')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio / Cumulative Explained Variance')
plt.title('Explained Variance by Principal Components')
plt.grid(True)
plt.tight_layout()
plt.show()

# Print cumulative explained variance for different numbers of components
cumulative = np.cumsum(pca.explained_variance_ratio_)
for i, var in enumerate(cumulative):
    print(f"{i+1} components: {var:.4f} cumulative variance")


# Create PCA-transformed datasets with different numbers of components
results = []
component_numbers = [2, 3, 4, 5, 6, 7, 8, 9, 10] 

for n_components in component_numbers:
    # Create PCA with n components
    pca_n = PCA(n_components=n_components)
    X_pca = pca_n.fit_transform(X_scaled)

    X_train, X_val = X_pca[:-3000], X_pca[-3000:]
    y_train, y_val = y.iloc[:-3000], y.iloc[-3000:]

    model = XGBRegressor(
        n_estimators=10000,
        learning_rate=0.02,
        early_stopping_rounds=100,
        colsample_bytree=0.5,
        subsample=0.8,
        random_state=42,
        verbosity=0
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )


    y_pred = model.predict(X_val)
    mse_pca = mean_squared_error(y_val, y_pred)
    print(f'{n_components} components MSE: {mse_pca}')

    results.append({'n_components': n_components, 'mse': mse_pca})

original_mse=0.00000153
linear_regression_mse=0.0000095965
# Plot results
plt.figure(figsize=(10, 6))
plt.plot([r['n_components'] for r in results], [r['mse'] for r in results], 'o-')
plt.axhline(y=original_mse, color='r', linestyle='--', label='Original MSE (no PCA)')
plt.axhline(y=linear_regression_mse, color='k', linestyle='--', label='Linear MSE (no PCA)')
plt.xlabel('Number of Principal Components')
plt.ylabel('Mean Squared Error')
plt.title('Model Performance vs. Number of Principal Components')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


sub = pd.read_csv("/kaggle/input/black-jack-smart-effect-of-removal-ml/sample_submission.csv")
sub.ev = final_test_pred
sub.to_csv("submission.csv", index=False)

