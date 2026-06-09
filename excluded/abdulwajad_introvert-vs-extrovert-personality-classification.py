# ğŸ“¦ Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression


# ğŸ“¥ Load the Data
train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


import matplotlib.pyplot as plt
import seaborn as sns

# Set a beautiful theme
sns.set(style="whitegrid", palette="pastel")

# Distribution of target variable
plt.figure(figsize=(8, 5))
sns.countplot(data=train_df, x="Personality", order=train_df["Personality"].value_counts().index)
plt.title("Distribution of Personality Types")
plt.xlabel("Personality")
plt.ylabel("Count")
plt.show()



# ğŸ§¹ Drop ID
train_df.drop(columns='id', inplace=True)
test_ids = test_df['id']
test_df.drop(columns='id', inplace=True)


# ğŸ�¯ Target and Features
X = train_df.drop(columns='Personality')
y = train_df['Personality']
X_test = test_df.copy()

# ğŸ”� Identify Column Types
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()


# Plot distribution of numeric features
X[numeric_cols].hist(bins=30, figsize=(15, 10), edgecolor='black')
plt.suptitle("Numeric Feature Distributions", fontsize=16)
plt.show()



plt.figure(figsize=(10, 8))
corr = X[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap of Numeric Features")
plt.show()



# ğŸ”� Identify Column Types
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()


for col in numeric_cols:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=train_df, x="Personality", y=col)
    plt.title(f"{col} vs Personality")
    plt.show()



# ğŸ› ï¸� Preprocessing Pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# ğŸ”— Combine Transformers
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)


# âš™ï¸� Full Pipeline with Classifier
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# ğŸ§ª Train-test split (for local evaluation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ğŸš‚ Train the model
clf.fit(X_train, y_train)


# ğŸ§¾ Evaluate on validation set
y_pred = clf.predict(X_val)
print(classification_report(y_val, y_pred))



import plotly.graph_objects as go
from sklearn.metrics import classification_report

# Get classification report as a dictionary
report_dict = classification_report(y_val, y_pred, output_dict=True)

# Convert to DataFrame for easier handling
report_df = pd.DataFrame(report_dict).transpose()

# Round metrics
report_df = report_df.round(3)

# Reorder: Bring class labels first, then avg/total rows
ordered_rows = [label for label in y.unique() if label in report_df.index] + ["accuracy", "macro avg", "weighted avg"]
report_df = report_df.loc[ordered_rows]

# Plot as table
fig = go.Figure(data=[go.Table(
    header=dict(
        values=["Class/Metric", "Precision", "Recall", "F1-Score", "Support"],
        fill_color='indigo',
        font=dict(color='white', size=14),
        align='left'
    ),
    cells=dict(
        values=[
            report_df.index,
            report_df["precision"],
            report_df["recall"],
            report_df["f1-score"],
            report_df.get("support", [""] * len(report_df))  # Accuracy row doesn't have support
        ],
        fill_color='lavender',
        align='left',
        font=dict(size=12)
    )
)])

fig.update_layout(
    title_text="ğŸ“‹ Classification Report Summary",
    title_x=0.5
)

fig.show()



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_val, y_pred, labels=clf.named_steps['classifier'].classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.named_steps['classifier'].classes_)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Blues")
plt.title("Confusion Matrix - Validation Set")
plt.show()



# ğŸ“Š Predict on test set
test_preds = clf.predict(X_test)


# ğŸ“� Prepare submission
submission['Personality'] = test_preds
submission.to_csv("submission.csv", index=False)
print("âœ… Submission file saved as submission.csv")


