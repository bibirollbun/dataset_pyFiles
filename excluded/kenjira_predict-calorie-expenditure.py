import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder


# === STEP 1: Load Data ===
train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


# === STEP 2: Feature Engineering ===
for df in [train_df, test_df]:
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['HRxDuration'] = df['Heart_Rate'] * df['Duration']
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])  # 0/1 encoding

# === STEP 3: Prepare Features ===
features = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI', 'HRxDuration']
X = train_df[features]
y = train_df['Calories']
X_test = test_df[features]


# === STEP 4: Cross-Validation Setup ===
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

# === STEP 5: Training with Cross-Validation ===
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Fold {fold + 1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBRegressor(
            n_estimators=5000,
            learning_rate=0.01,
            max_depth=8,
            min_child_weight=10,
            subsample=0.85,
            colsample_bytree=0.7,
            gamma=0.05,
            reg_alpha=0.5,
            reg_lambda=1.0,
            early_stopping_rounds=200,
            eval_metric="rmse",
            random_state=42
        )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    val_preds = model.predict(X_val)
    oof_preds[val_idx] = val_preds
    test_preds += model.predict(X_test) / kf.n_splits

# === STEP 6: Evaluate RMSLE ===
rmsle = np.sqrt(mean_squared_log_error(y, np.maximum(0, oof_preds)))
print(f"\n✅ Final CV RMSLE: {rmsle:.5f}")


# === STEP 7: Feature Importance ===
import matplotlib.pyplot as plt
import seaborn as sns
FOLDS = 5
importances = np.zeros(len(features))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = XGBRegressor(
        n_estimators=5000,
        learning_rate=0.01,
        max_depth=8,
        min_child_weight=10,
        subsample=0.85,
        colsample_bytree=0.7,
        gamma=0.05,
        reg_alpha=0.5,
        reg_lambda=1.0,
        early_stopping_rounds=200,
        eval_metric="rmse",
        random_state=42
    )

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=0)

    importances += model.feature_importances_

# === เฉลี่ยและแสดงผล ===
importances /= FOLDS
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.barh([features[i] for i in sorted_idx], importances[sorted_idx])
plt.gca().invert_yaxis()
plt.title("Average Feature Importance (XGBoost)")
plt.xlabel("Importance Score")
plt.show()


# display the distribution of predicted calories
plt.figure(figsize=(10, 5))
plt.hist(test_preds, bins=100)
plt.title("Distribution of Predicted Calories (Test Set)")
plt.xlabel("Calories")
plt.ylabel("Frequency")
plt.show()


# === STEP 8: Submission File ===
submission = pd.DataFrame({
    'id': test_df['id'],
    'Calories': np.maximum(0, test_preds)  # Ensure no negative predictions
})
submission.head()
submission.to_csv("submission_model1.csv", index=False)
print("✅ Saved submission_model1.csv")
submission.head()

