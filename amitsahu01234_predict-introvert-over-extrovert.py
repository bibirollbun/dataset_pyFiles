import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

train = pd.read_csv("/kaggle/input/introvert-extrovert-datasets/train.csv")
test = pd.read_csv("/kaggle/input/introvert-extrovert-datasets/test.csv")

label_enc = LabelEncoder()
train['Personality'] = label_enc.fit_transform(train['Personality'])  # Extrovert=0, Introvert=1

X = train.drop(columns=["id", "Personality"])
y = train["Personality"]
X_test = test.drop(columns=["id"])

for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])

final_preds = np.zeros(len(X_test))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
    print(f"\n Fold {fold + 1}")
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = LGBMClassifier(
        objective='binary',
        learning_rate=0.05,
        num_leaves=64,
        max_depth=7,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=20,
        n_estimators=200,    
        verbosity=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds_valid = model.predict(X_valid)
    acc = accuracy_score(y_valid, preds_valid)
    print(f" Fold {fold + 1} Accuracy: {acc:.4f}")

    final_preds += model.predict_proba(X_test)[:, 1] / cv.n_splits
final_preds_binary = (final_preds > 0.5).astype(int)
final_labels = label_enc.inverse_transform(final_preds_binary)

submission = pd.DataFrame({
    'id': test['id'],
    'Personality': final_labels
})
submission.to_csv("submission.csv", index=False)
print("\n submission.csv saved successfully!")


