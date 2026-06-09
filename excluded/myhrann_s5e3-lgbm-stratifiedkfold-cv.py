import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


train.head()


target_col = "rainfall"  # Modify this if needed
id_col = "id"  # Modify this if needed


X = train.drop(columns=[target_col, id_col])  # Features
y = train[target_col]  # Target variable
X_test = test.drop(columns=[id_col])  # Test features


categorical_cols = X.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])


n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

predictions = np.zeros(len(X_test))  # To store averaged predictions
cv_scores = []


params = {
    "objective": "binary",
    "metric": "binary_error",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": 42,
    "verbosity": -1
}


for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"Training fold {fold+1}...")
    
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=500,
        # early_stopping_rounds=50,
        # verbose_eval=50
    )


y_valid_pred = model.predict(X_valid)
y_valid_binary = (y_valid_pred > 0.5).astype(int)
accuracy = accuracy_score(y_valid, y_valid_binary)
cv_scores.append(accuracy)


# Store test predictions
predictions += model.predict(X_test) / n_splits


final_predictions = (predictions > 0.5).astype(int)


submission = pd.DataFrame({
    id_col: test[id_col],
    target_col: final_predictions
})

submission.to_csv("submission.csv", index=False)

print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean accuracy: {np.mean(cv_scores):.4f}")
print("submission.csv has been created!")

