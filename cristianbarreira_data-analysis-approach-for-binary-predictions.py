# Main Libraries
import pandas as pd
pd.set_option('display.max_columns', None)

import numpy as np

import warnings
warnings.filterwarnings("ignore")


# Visualization Libraries
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set(style="whitegrid")

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ML Libraries
from catboost import CatBoostClassifier

import xgboost as xgb
from xgboost import XGBClassifier

import lightgbm as lgb
from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin


# Load datasets
df_train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
df_test  = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

# Display dataset shape
print("Train Dataset shape:", df_train.shape)
print("Test Dataset shape:", df_test.shape)

# Preview train data
df_train


# Preview test data
df_test


df_train.info()


df_train.describe()


df_train.duplicated().sum()


# Differentiate numerical and categorical data
numerical_cols = df_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = df_train.select_dtypes(include=['object', 'bool']).columns.tolist()

print("Numerical Columns:", numerical_cols)
print("Categorical Columns:", categorical_cols)


# With this graph, is possible to see how many clients had suscribed, and the 'conversion rate' of the overall campaign data

plt.figure(figsize=(6, 4))
sns.countplot(data=df_train, x='y', palette='pastel', edgecolor='black')
plt.title('Distribution of Subscription to Bank Deposit', fontsize=12)
plt.xlabel('Subscribed to Bank Deposit ', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Display normalized value counts (as proportions)
print("\n Subscription to Term Deposit Value Counts (Proportions):")
print(df_train['y'].value_counts(normalize=True).round(4))


# Distribution analysis of Categorical Variables

# Loop through each categorical variable
for col in categorical_cols:
    # Calculate metrics
    agg = df_train.groupby(col).agg(
        clients=("y", "count"),
        conversion=("y", "sum")
    ).reset_index()
    agg["conversion_rate"] = agg["conversion"] / agg["clients"]

    # Interactive graph to show all the data available in the visualization
    fig = px.bar(
        agg, x=col, y="clients",
        title=f"Distribution and Conversion Rate by {col}",
        text="clients"
    )

    # Add conversion rate plot
    fig.add_scatter(
        x=agg[col],
        y=agg["conversion_rate"],
        mode="lines+markers",
        name="Conversion Rate",
        yaxis="y2"
    )

    # Secondary axis for conversion rate
    fig.update_layout(
        yaxis=dict(title="Number of Clients"),
        yaxis2=dict(title="Conversion Rate", overlaying="y", side="right"),
        hovermode="x unified"
    )

    # Plot interactive graph
    fig.show()


print(numerical_cols)



# As we dont need to analyze 'id' and 'y', lets drop them

numerical_cols.remove('id')
numerical_cols.remove('y')
print(numerical_cols)


# Distribution analysis of Numerical Variables

for col in numerical_cols:
    # Bin column into intervals
    df_train['bin'] = pd.cut(df_train[col], bins=30)
    # Conversion rate per bin
    conv_rate = df_train.groupby('bin')['y'].mean().reset_index()
    conv_rate['bin_center'] = conv_rate['bin'].apply(lambda x: x.mid)

    # Create figure with secondary axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    hist = go.Histogram(
        x=df_train[col],
        nbinsx=30,
        name="Count",
        marker_color="skyblue",
        opacity=0.7
    )
    fig.add_trace(hist, secondary_y=False)

    # Conversion rate line
    line = go.Scatter(
        x=conv_rate['bin_center'],
        y=conv_rate['y'],
        name="Conversion Rate",
        mode="lines+markers",
        line=dict(color="red")
    )
    fig.add_trace(line, secondary_y=True)

    # Layout
    fig.update_layout(
        title=f"Distribution of {col} with Conversion Rate",
        template="simple_white",
        bargap=0.05
    )
    fig.update_xaxes(title_text=col)
    fig.update_yaxes(title_text="Count", secondary_y=False)
    fig.update_yaxes(title_text="Conversion Rate", secondary_y=True)

    fig.show()

    # Descriptive Stats
    print(f'\n Descriptive Stats for {col}:\n')
    print(df_train[col].describe(), '\n' + '-'*40)




# Separate features and target
X = df_train.drop(columns=["y", "id"])
y = df_train["y"]

# Split into train/validation, lets take a 20 % as validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessing: scale numerical columns, one-hot encode categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Full pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train
pipeline.fit(X_train, y_train)

# Validate
y_val_pred = pipeline.predict(X_val)
y_val_proba = pipeline.predict_proba(X_val)[:, 1]

# Metrics
report = classification_report(y_val, y_val_pred)
roc_auc = roc_auc_score(y_val, y_val_proba)

print("=== Evaluation on Train Dataset ===")
print(report)
print("ROC AUC:", roc_auc)


numerical_cols_2 = numerical_cols.copy()
numerical_cols_2.remove('duration')
print(numerical_cols)
print(numerical_cols_2)


# --- Custom Transformer for Feature Engineering ---
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # pdays: never contacted vs contacted
        if "pdays" in X.columns:
            X["pdays_contacted"] = X["pdays"].apply(lambda v: 0 if v == -1 else 1)

        # balance per age (wealth index)
        if "balance" in X.columns and "age" in X.columns:
            X["balance_per_age"] = X["balance"] / (X["age"]+1)

        return X

# Extended with engineered features (without 'duration')
engineer = FeatureEngineer()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols_2 + ["balance_per_age"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols + ["pdays_contacted"])
    ],
    remainder="drop"
)

model_pipeline = Pipeline(steps=[
    ("engineer", engineer),
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

# Train-validation split
X = df_train.drop(columns=["y", "id"])
y = df_train["y"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit pipeline
model_pipeline.fit(X_train, y_train)
y_val_pred = model_pipeline.predict(X_val)
y_val_proba = model_pipeline.predict_proba(X_val)[:, 1]

# Metrics
report_eng = classification_report(y_val, y_val_pred)
roc_auc_eng = roc_auc_score(y_val, y_val_proba)

(report_eng, roc_auc_eng)
print("=== Evaluation on Train Dataset (Improved feature engineering) ===")
print(report_eng)
print("ROC AUC:", roc_auc_eng)


# --- Custom Transformer for Feature Engineering ---
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # pdays: never contacted vs contacted
        if "pdays" in X.columns:
            X["pdays_contacted"] = X["pdays"].apply(lambda v: 0 if v == -1 else 1)

        # balance per age (wealth index)
        if "balance" in X.columns and "age" in X.columns:
            X["balance_per_age"] = X["balance"] / (X["age"]+1)

        return X

# Extended with engineered features (with 'duration')
engineer = FeatureEngineer()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols + ["balance_per_age"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols + ["pdays_contacted"])
    ],
    remainder="drop"
)

model_pipeline = Pipeline(steps=[
    ("engineer", engineer),
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

# Train-validation split
X = df_train.drop(columns=["y", "id"])
y = df_train["y"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit pipeline
model_pipeline.fit(X_train, y_train)
y_val_pred = model_pipeline.predict(X_val)
y_val_proba = model_pipeline.predict_proba(X_val)[:, 1]

# Metrics
report_eng = classification_report(y_val, y_val_pred)
roc_auc_eng = roc_auc_score(y_val, y_val_proba)

(report_eng, roc_auc_eng)
print("=== Evaluation on Train Dataset (Improved feature engineering) ===")
print(report_eng)
print("ROC AUC:", roc_auc_eng)


# Store the pipelines created

# ---------------- Models to Compare ----------------
models = {
    "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "CatBoost": CatBoostClassifier(verbose=0, class_weights=[1, (y==0).sum()/(y==1).sum()]),
    "XGBoost": XGBClassifier(eval_metric="logloss", use_label_encoder=False, scale_pos_weight=(y==0).sum()/(y==1).sum()),
    "LightGBM": LGBMClassifier(class_weight="balanced")
}

results = {}

fitted_pipelines = {}

for name, clf in models.items():
    pipeline = Pipeline(steps=[
        ("engineer", FeatureEngineer()),
        ("preprocessor", preprocessor),
        ("model", clf)
    ])

    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    y_val_proba = pipeline.predict_proba(X_val)[:, 1]

    report = classification_report(y_val, y_val_pred, output_dict=True)
    roc_auc = roc_auc_score(y_val, y_val_proba)

    results[name] = {
        "roc_auc": roc_auc,
        "precision_1": report["1"]["precision"],
        "recall_1": report["1"]["recall"],
        "f1_1": report["1"]["f1-score"],
        "accuracy": report["accuracy"]
    }

    # ðŸ”‘ keep the trained pipeline for later predictions
    fitted_pipelines[name] = pipeline

# Results summary
results_df = pd.DataFrame(results).T
print(results_df)


# Apply to the 'test' dataset

X_test = df_test.drop(columns=["id"])  # drop 'id' since it's not a predictor

test_predictions = {}

for name, pipe in fitted_pipelines.items():
    # Predicted class (0 or 1)
    y_test_pred = pipe.predict(X_test)

    # Predicted probability of conversion
    y_test_proba = pipe.predict_proba(X_test)[:, 1]

    test_predictions[name] = pd.DataFrame({
        "id": df_test["id"],
        f"{name}_pred": y_test_pred,
        f"{name}_proba": y_test_proba
    })



# Merge predictions into one DataFrame
df_predictions = df_test[["id"]].copy()

for name, pred_df in test_predictions.items():
    df_predictions = df_predictions.merge(pred_df, on="id")

df_predictions.head()


# Pick best model by ROC AUC
best_model_name = results_df["roc_auc"].idxmax()
best_model_name


# Grab only the best model results
best_model_predictions = test_predictions[best_model_name]

best_model_predictions.head(10)


# Save it to the submission file
submission = best_model_predictions.copy()
submission = submission.rename(columns={'CatBoost_proba': "y"})
submission = submission.drop(columns=["CatBoost_pred"])
submission.head(10)


submission.to_csv('submission.csv', index=False)
print("Submission created")

