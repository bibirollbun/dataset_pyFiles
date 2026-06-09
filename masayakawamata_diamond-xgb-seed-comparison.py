import warnings
warnings.simplefilter('ignore')


import pandas as pd, numpy as np

train = pd.read_csv('/kaggle/input/predicting-the-price-of-diamond/train.csv')
test = pd.read_csv('/kaggle/input/predicting-the-price-of-diamond/test.csv')
orig = pd.read_csv('/kaggle/input/diamonds/diamonds.csv')

train.head(3)


TARGET = 'price'
CATS = ['cut', 'color', 'clarity']
NUMS = ['carat', 'depth', 'table', 'x', 'y', 'z']
FEATURES = CATS + NUMS

# Prepare the data
X = train[FEATURES]
y = train[TARGET]
X_test = test[FEATURES]

X_test[CATS] = X_test[CATS].astype('category')


from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns


xgb_params = {
    'learning_rate': 0.01,
    'max_depth': 3,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'device': 'cuda',
    'n_estimators': 10000,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'enable_categorical': True,
}

FIXED_SPLIT_SEED = 42 
kf = KFold(n_splits=5, shuffle=True, random_state=FIXED_SPLIT_SEED)

model_seeds = list(range(20))
results = {}

for seed in model_seeds:
    print(f"--- Starting training for model_seed={seed} ---")
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"  Fold {fold+1}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_train[CATS] = X_train[CATS].astype('category')
        X_val[CATS] = X_val[CATS].astype('category')

        model = xgb.XGBRegressor(**xgb_params, random_state=seed)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=200,
                  verbose=False)

        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / kf.n_splits

    r2 = r2_score(y, oof_preds)
    results[seed] = r2
    print(f"R2 score for model_seed={seed}: {r2:.6f}\n")

print("--- Final R2 Scores per Model Seed (Splits Fixed) ---")
for seed, r2 in results.items():
    print(f"Model Seed = {seed}: R2 = {r2:.6f}")

best_seed = max(results, key=results.get)
print(f"\nThe best model seed is {best_seed} with an R2 score of {results[best_seed]:.6f}.")

plt.figure(figsize=(12, 7))
sns.lineplot(x=list(results.keys()), y=list(results.values()), marker='o', color='royalblue')
plt.title('CV R2 Score vs. Model Random Seed (Fixed Splits)', fontsize=16)
plt.xlabel('Model Random Seed', fontsize=12)
plt.ylabel('CV R2 Score', fontsize=12)
plt.xticks(model_seeds)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('r2_vs_model_seed_fixed_split.png')
plt.show()




