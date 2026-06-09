import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.simplefilter('ignore')


train = pd.read_csv("/kaggle/input/black-jack-smart-effect-of-removal-ml/train.csv")
test = pd.read_csv("/kaggle/input/black-jack-smart-effect-of-removal-ml/test.csv")
print("Train Shape:", train.shape)
print("Test  Shape:", test.shape)
train.head(3)

TARGET = "ev"


from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


X = train.drop([TARGET, "id"], axis=1).copy()
y = train[TARGET].copy()
X_test = test.drop(columns='id').copy()


FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_pred = np.zeros(len(X))
fold_mse = []
test_preds = np.zeros((len(X_test), FOLDS))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
    print(f"Training fold {fold} ...")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = CatBoostRegressor(
        iterations=20000,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        od_wait=100,
        verbose=False,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        verbose=5000
    )
    
    y_pred = model.predict(X_val)
    mse_fold = mean_squared_error(y_val, y_pred)
    fold_mse.append(mse_fold)
    oof_pred[val_idx] = y_pred
    print(f"Fold {fold} MSE: {mse_fold:.11f}")
    
    test_preds[:, fold - 1] = model.predict(X_test)

overall_mse = mean_squared_error(y, oof_pred)
print(f"\nOverall OOF MSE: {overall_mse:.11f}")

final_test_pred = test_preds.mean(axis=1)
print("\nFinal test predictions (first 10 samples):")
print(final_test_pred[:10])


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,6))

axes[0].hist(y, bins=50, edgecolor='k', alpha=0.7)
axes[0].set_title("Distribution of y (Target)")
axes[0].set_xlabel("Target Value")
axes[0].set_ylabel("Frequency")
axes[0].grid(True)

axes[1].hist(final_test_pred, bins=50, edgecolor='k', alpha=0.7)
axes[1].set_title("Distribution of Final Test Predictions")
axes[1].set_xlabel("Predicted Value")
axes[1].set_ylabel("Frequency")
axes[1].grid(True)

plt.tight_layout()
plt.show()


sub = pd.read_csv("/kaggle/input/black-jack-smart-effect-of-removal-ml/sample_submission.csv")
sub.ev = final_test_pred
sub.to_csv("submission.csv", index=False)
sub.head(3)

