import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,roc_auc_score, confusion_matrix, classification_report)

import warnings
warnings.filterwarnings("ignore")



# Training and test datasets
df = pd.read_csv("/kaggle/input/heart-disease-prediction-dataquest/heart_train.csv")
test_df = pd.read_csv("/kaggle/input/heart-disease-prediction-dataquest/heart_test.csv")
sample_sub = pd.read_csv("/kaggle/input/heart-disease-prediction-dataquest/sample_submission.csv")

df.info()
df.head()


# Target distribution
sns.countplot(x='HeartDisease', data=df)
plt.title("Heart Disease Distribution")
plt.show()

# Correlation heatmap (only numeric columns)
plt.figure(figsize=(10, 6))
corr_matrix = df.select_dtypes(include=["int64", "float64"]).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()



# Split features and target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Identify types
categorical_features = X.select_dtypes(include="object").columns.tolist()
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Transformers
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")

# Column transformer
preprocessor = ColumnTransformer([
    ("num", numerical_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



# Logistic Regression
lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, solver='liblinear', random_state=42))
])

# Random Forest
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Gradient Boosting
gb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(n_estimators=100, random_state=42))
])



pipelines = {
    "Logistic Regression": lr_pipeline,
    "Random Forest": rf_pipeline,
    "Gradient Boosting": gb_pipeline
}

results = {}

for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    proba = pipe.predict_proba(X_val)[:, 1]
    
    results[name] = {
        "Accuracy": accuracy_score(y_val, preds),
        "Precision": precision_score(y_val, preds),
        "Recall": recall_score(y_val, preds),
        "F1 Score": f1_score(y_val, preds),
        "ROC AUC": roc_auc_score(y_val, proba)
    }

# Show results
results_df = pd.DataFrame(results).T
results_df.sort_values("Accuracy", ascending=False)



# Use the best model (change as needed)
final_model = rf_pipeline
final_model.fit(X, y)  # Train on full data

# Prepare test features
X_test = test_df[X.columns]

# Predict
test_preds = final_model.predict(X_test)

# Format submission
submission = pd.DataFrame({
    "id": test_df.index,
    "HeartDisease": test_preds
})

submission.to_csv("submission.csv", index=False)
submission.head()


