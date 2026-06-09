import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')



test_id = test["id"]
train.drop("id", axis=1, inplace=True)
test.drop("id", axis=1, inplace=True)


# Split features and target
X = train.drop("Fertilizer Name", axis=1)
y = train["Fertilizer Name"]


# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# Encode categorical features
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    test[col] = le.transform(test[col])


# K-Fold CV setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros((test.shape[0], len(label_encoder.classes_)))
val_losses = []


# Train and evaluate
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded), 1):
    print(f"Fold {fold}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    val_probs = model.predict_proba(X_val)
    loss = log_loss(y_val, val_probs)
    print(f"Log Loss: {loss:.5f}")
    val_losses.append(loss)


test_preds += model.predict_proba(test)


# Average test preds
test_preds /= skf.n_splits



# Get top-3 class predictions
top3_indices = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]
top3_labels = label_encoder.inverse_transform(top3_indices.ravel()).reshape(top3_indices.shape)
top3_preds = [' '.join(row) for row in top3_labels]


# Prepare submission
submission = pd.DataFrame({
    "id": test_id,
    "Fertilizer Name": top3_preds
})
submission.to_csv("rf_submission.csv", index=False)

print(f"\nMean CV Log Loss: {np.mean(val_losses):.5f}")
print("Random Forest submission saved.")


submission.head(5)

