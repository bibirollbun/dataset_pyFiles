import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

X = train.drop(columns=["id", "Personality"])
y = train["Personality"]
X_test = test.drop(columns=["id"])

# Columns
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()

# Preprocessing pipelines
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))
])
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# Classifiers
rf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
xgb = XGBClassifier(n_estimators=250, learning_rate=0.07, max_depth=7, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
lgb = LGBMClassifier(n_estimators=250, learning_rate=0.07, max_depth=7, random_state=42)

# Voting ensemble
ensemble = VotingClassifier(
    estimators=[("rf", rf), ("xgb", xgb), ("lgb", lgb)],
    voting="soft"
)

# Full pipeline
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", ensemble)
])

# K-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
final_preds = np.zeros((X_test.shape[0], 2))  # for soft voting probas

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"Fold {fold+1} Accuracy: {acc:.4f}")
    accuracies.append(acc)

    # Soft voting for test set
    final_preds += pipeline.predict_proba(X_test)

# Final hard prediction
avg_acc = np.mean(accuracies)
print(f"\nğŸ”� Average CV Accuracy: {avg_acc:.4f}")

# Make final predictions
labels = np.array(pipeline.classes_)
final_labels = labels[np.argmax(final_preds, axis=1)]
submission["Personality"] = final_labels
submission.to_csv("submission.csv", index=False)
print("âœ… Submission file saved as 'submission.csv'")


