import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

# Drop ID column from features
X = train.drop(columns=["id", "Personality"])
y = train["Personality"]
X_test = test.drop(columns=["id"])

# Identify column types
num_cols = X.select_dtypes(include=["float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Imputers
num_imputer = SimpleImputer(strategy="mean")
cat_imputer = SimpleImputer(strategy="most_frequent")


# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("imputer", num_imputer),
        ("scaler", StandardScaler())
    ]), num_cols),
    ("cat", SimpleImputer(strategy='most_frequent'), cat_cols)
])

# Convert categorical columns
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(cat_imputer.fit_transform(X[[col]]).ravel())
    X_test[col] = le.transform(cat_imputer.transform(X_test[[col]]).ravel())

# Apply preprocessing to training and test data
X_processed = preprocessor.fit_transform(X)
X_test_processed = preprocessor.transform(X_test)


# Final input data
X_final = pd.DataFrame(X_processed)
X_test_final = pd.DataFrame(X_test_processed)

# Label Encode target
target_le = LabelEncoder()
y_encoded = target_le.fit_transform(y)


X_final


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

models = {
    "RandomForest": RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
),
    "XGBoost": XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.2,
    reg_alpha=0.5,
    reg_lambda=1,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
),
    "LightGBM": LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.2,
    reg_lambda=0.4,
    random_state=42,
    verbosity=0,
    n_jobs=-1
),
    "LogisticRegression": LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='saga',
    max_iter=2000,
    class_weight='balanced',
    random_state=42
),
    "CatBoost": CatBoostClassifier(
    iterations=300,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3,
    border_count=64,
    random_state=42,
    verbose=0,
    task_type='CPU'  # or 'GPU' if available
)}



# Train and store predictions
predictions = {}
for name, model in models.items():
    model.fit(X_final, y_encoded)
    preds = model.predict(X_test_final)
    preds_labels = target_le.inverse_transform(preds)
    predictions[name] = pd.DataFrame({
        "id": test["id"],
        "Personality": preds_labels
    })



for name, df in predictions.items():
    df.to_csv(f"{name}_submission.csv", index=False)



# Combine all model predictions
ensemble_df = pd.DataFrame({"id": test["id"]})
for name, df in predictions.items():
    ensemble_df[name] = df["Personality"]

# Majority voting
def majority_vote(row):
    return row.mode()[0]

ensemble_df["Personality"] = ensemble_df.drop(columns=["id"]).apply(majority_vote, axis=1)

# Save ensemble submission
ensemble_df[["id", "Personality"]].to_csv("Ensemble_submission.csv", index=False)




comparison = pd.DataFrame()
for name, df in predictions.items():
    comparison[name] = df["Personality"]

comparison["Agreement"] = comparison.apply(lambda row: row.nunique() == 1, axis=1)
agreement_ratio = comparison["Agreement"].mean()

print(f"Model agreement on test set: {agreement_ratio:.2%}")


