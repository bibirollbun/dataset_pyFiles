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
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns


xgb_params = {
    'learning_rate': 0.01,
    'max_depth': 3,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'device': 'cuda',
    'n_estimators': 20000,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'enable_categorical': True,
    'early_stopping_rounds': 500
}

# Experiment with different weights by iterating through k values
k_values = list(range(1, 21))
results = {}

for k in k_values:
    print(f"--- Starting training for k={k} ---")
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"  Fold {fold+1}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Augment the training data by concatenating the 'orig' dataset
        X_train_aug = pd.concat([X_train, orig[FEATURES]], axis=0)
        y_train_aug = pd.concat([y_train, orig[TARGET]], axis=0)

        X_train_aug[CATS] = X_train_aug[CATS].astype('category')
        X_val[CATS] = X_val[CATS].astype('category')

        # Create sample weights
        # Assign weight 'k' to original training data and weight '1' to augmented data
        sample_weight = np.concatenate([np.full(len(X_train), k), np.ones(len(orig))])

        # Initialize and train the model
        model = XGBRegressor(**xgb_params)
        model.fit(X_train_aug, y_train_aug,
                  eval_set=[(X_val, y_val)],
                  sample_weight=sample_weight,
                  verbose=False) # Suppress training logs

        # Make predictions
        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / kf.n_splits

    # Evaluate the R2 score for the current k
    r2 = r2_score(y, oof_preds)
    results[k] = r2
    print(f"R2 score for k={k}: {r2:.6f}\n")

# Display the final results
print("--- Final R2 Scores ---")
for k, r2 in results.items():
    print(f"k = {k}: R2 = {r2:.6f}")

# Identify and print the best k value
best_k = max(results, key=results.get)
print(f"\nThe best k value is {best_k} with an R2 score of {results[best_k]:.6f}.")

# Plot the results
plt.figure(figsize=(10, 6))
sns.lineplot(x=list(results.keys()), y=list(results.values()), marker='o')
plt.title('R2 Score vs. Weight Multiplier (k)')
plt.xlabel('Weight Multiplier (k)')
plt.ylabel('R2 Score')
plt.grid(True)
plt.savefig('r2_vs_k.png')




