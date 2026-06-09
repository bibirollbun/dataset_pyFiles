import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier

plt.style.use("seaborn-v0_8")
sns.set_palette("crest")



train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)


train.head()


sns.countplot(x="loan_paid_back", data=train, palette="Set2")
plt.title("Target Distribution â€” Loan Paid Back (0 = No, 1 = Yes)", fontsize=13)
plt.show()

train["loan_paid_back"].value_counts(normalize=True)



num_cols = train.select_dtypes(include=['int64', 'float64']).columns.drop(['id', 'loan_paid_back'])
cat_cols = train.select_dtypes(include=['object']).columns

# Distribution of numerical features
train[num_cols].hist(figsize=(12, 8), bins=30, color="#00bcd4")
plt.suptitle("Numerical Feature Distributions", fontsize=14)
plt.show()





# Correlation
plt.figure(figsize=(10, 6))
sns.heatmap(train[num_cols].corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap", fontsize=13)
plt.show()


for col in cat_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(y=col, data=train, order=train[col].value_counts().index[:10], palette="viridis")
    plt.title(f"Top Categories in {col}")
    plt.show()



X = train.drop(columns=["loan_paid_back"])
y = train["loan_paid_back"]

num_features = X.select_dtypes(include=['int64', 'float64']).columns.drop("id")
cat_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ]
)


model = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=31,
    random_state=42,
    subsample=0.8,
    colsample_bytree=0.8
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    auc_scores.append(auc)
    print(f"Fold {fold+1} AUC: {auc:.4f}")

print("\\nâœ… Average CV AUC:", np.mean(auc_scores))



pipeline.fit(X, y)
test_preds = pipeline.predict_proba(test)[:, 1]

submission = sample.copy()
submission["loan_paid_back"] = test_preds
submission.to_csv("submission.csv", index=False)

print("âœ… Submission file created successfully!")
submission.head()



model = pipeline.named_steps["model"]

# Get encoded feature names
feature_names = (
    list(num_features) +
    list(pipeline.named_steps["preprocessor"]
         .transformers_[1][1]
         .named_steps["encoder"]
         .get_feature_names_out(cat_features))
)

importances = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False).head(20)

plt.figure(figsize=(10, 6))
sns.barplot(y="Feature", x="Importance", data=importances, palette="crest")
plt.title("Top 20 Feature Importances (LightGBM)")
plt.show()

