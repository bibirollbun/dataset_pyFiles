import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import log_loss, top_k_accuracy_score

from xgboost import XGBClassifier


categorical_features = ['Soil Type', 'Crop Type']
numeric_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
target_column = 'Fertilizer Name'

train_df1 = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
train_df2 = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
train_df = pd.concat([train_df1, train_df2], ignore_index=True)

X_train = train_df[categorical_features + numeric_features]
y_train = train_df[target_column]

test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
X_test = test_df[categorical_features + numeric_features]

le = LabelEncoder()
y_encoded = le.fit_transform(y_train)


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)
numeric_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numeric_transformer, numeric_features)
    ]
)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb_model)
])


# For storing predictions
n_classes = len(np.unique(y_encoded))
oof_preds = np.zeros((X_train.shape[0], n_classes))
test_preds = np.zeros((X_test.shape[0], n_classes))

# Cross-validation
for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train, y_encoded)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
    y_tr, y_val = y_encoded[train_idx], y_encoded[valid_idx]

    fold_pipeline = clone(pipeline)
    fold_pipeline.fit(X_tr, y_tr)

    # OOF predictions
    oof_preds[valid_idx] = fold_pipeline.predict_proba(X_val)

    # Test predictions
    test_preds += fold_pipeline.predict_proba(X_test) / kf.n_splits

    # Metrics
    logloss = log_loss(y_val, oof_preds[valid_idx])
    top3 = top_k_accuracy_score(y_val, oof_preds[valid_idx], k=3)
    print(f"Fold {fold + 1} - Log Loss: {logloss:.4f}, Top-3 Accuracy: {top3:.4f}")


# Get class names in the label encoder's order
class_indices = np.arange(oof_preds.shape[1])
class_names = le.inverse_transform(class_indices)

# Create column names for probabilities
prob_columns = [f"prob_{label}" for label in class_names]

# Save OOF probabilities
preds_oof_df = pd.DataFrame(oof_preds, columns=prob_columns)
preds_oof_df.to_csv("oof_probs.csv", index=False)

# Save test probabilities (preserve original test_df)
preds_test_df = pd.DataFrame(test_preds, columns=prob_columns)
preds_test_df.insert(0, "id", test_df["id"].values)
preds_test_df.to_csv("test_probs.csv", index=False)
preds_test_df.head()


# Get top-3 predictions (indices) for OOF and test sets
oof_top3 = np.argsort(oof_preds, axis=1)[:, -3:][:, ::-1]
test_top3 = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]

# Convert top-3 indices to label names
oof_top3_labels_flat = le.inverse_transform(oof_top3.flatten())
test_top3_labels_flat = le.inverse_transform(test_top3.flatten())

# Reshape back to (n_samples, 3)
oof_top3_labels = oof_top3_labels_flat.reshape(-1, 3)
test_top3_labels = test_top3_labels_flat.reshape(-1, 3)

# Join top-3 labels into space-separated strings
oof_top3_strings = [" ".join(row) for row in oof_top3_labels]
test_top3_strings = [" ".join(row) for row in test_top3_labels]

# Create DataFrames for top-3 predictions
oof_top3_df = pd.DataFrame({"Fertilizer Name": oof_top3_strings})
test_top3_df = pd.DataFrame({
    "id": test_df["id"].values,
    "Fertilizer Name": test_top3_strings
})

# Save DataFrames to CSV files
oof_top3_df.to_csv("oof_top3.csv", index=False)
test_top3_df.to_csv("test_top3.csv", index=False)

test_top3_df.head()


def mapk(true_labels, pred_labels, k=3):
    total_score = 0.0
    for true, preds in zip(true_labels, pred_labels):
        try:
            rank = preds.index(true) + 1
            total_score += 1.0 / rank
        except ValueError:
            pass
    return total_score / len(true_labels)


# Compute overall Top-3 Accuracy (already have oof_preds and true labels)
top3_acc = top_k_accuracy_score(le.transform(y_train), oof_preds, k=3)

# Prepare for MAP@3 using existing oof_top3_labels
true_labels = y_train.tolist()  # Already in string form
pred_labels_list = [list(row) for row in oof_top3_labels]

map3 = mapk(true_labels, pred_labels_list, k=3)

print(f"\nOOF Top-3 Accuracy: {top3_acc:.4f}")
print(f"OOF MAP@3: {map3:.4f}")


# Create a submission DataFrame with test IDs and top-3 predictions
submission_df = pd.DataFrame({
    "id": test_df["id"].values,
    "Fertilizer Name": test_top3_strings
})

# Save to submission.csv
submission_df.to_csv("submission.csv", index=False)
submission_df.head()

