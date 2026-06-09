import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load Data
train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')

# Split features and target
X = train.drop(columns=['Listening_Time_minutes'])
y = train['Listening_Time_minutes']
X_test = test.copy()

# Feature Engineering Function
def feature_engineering(df):
    df = df.copy()
    
    # Example placeholder: you can add real feature logic here
    # df['new_feature'] = df['feature1'] / (df['feature2'] + 1)

    df.fillna(-999, inplace=True)  # Replace missing values
    return df

# Apply feature engineering
X = feature_engineering(X)
X_test = feature_engineering(X_test)

# One-hot encoding
X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)

# Align columns
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# LightGBM Parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': 42
}

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
preds = np.zeros(len(X_test))
scores = []

# Training Loop
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nâœ… Fold {fold + 1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        num_boost_round=10000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=200)
        ]
    )


    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    rmse = mean_squared_error(y_val, val_preds, squared=False)
    print(f"ðŸŽ¯ Fold {fold + 1} RMSE: {rmse:.5f}")
    scores.append(rmse)

    preds += model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits

# Average RMSE
print(f"\nðŸ“Š Average RMSE: {np.mean(scores):.5f}")

# Submission File
submission = pd.DataFrame({
    'id': test['id'],
    'Listening_Time_minutes': preds
})
submission.to_csv('submission.csv', index=False)
print("âœ… Submission file saved: submission.csv")


