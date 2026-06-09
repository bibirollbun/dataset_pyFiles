from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV


train_df=pd.read_csv(r'/kaggle/input/playground-series-s5e3/train.csv')
test_df=pd.read_csv(r'/kaggle/input/playground-series-s5e3/test.csv')
print(train_df.shape,test_df.shape)


# Display summary statistics
print("\nTrain Summary:")
print(train_df.describe())


# Check for missing values
missing_values_train = train_df.isnull().sum()
missing_values_test = test_df.isnull().sum()
print("\nMissing Values in Train Dataset:")
print(missing_values_train)
print("\nMissing Values in Test Dataset:")
print(missing_values_test)


# Feature Selection - Drop 'id' column
target = 'rainfall'
id_col = 'id'
features = [col for col in train_df.columns if col not in [target, id_col]]

# Handling Missing Values
imputer = SimpleImputer(strategy='median')
train_df[features] = imputer.fit_transform(train_df[features])
test_df[features] = imputer.transform(test_df[features])


# Pipeline-aware imports
target = 'rainfall'
id_col = 'id'
features = [col for col in train_df.columns if col not in [target, id_col]]

# Train/Valid Split BEFORE any preprocessing
X = train_df[features].copy()
y = train_df[target].copy()
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Preprocessing pipeline (NO DATA LEAKAGE)
preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ("scaler", StandardScaler())
])

X_train_trans = preprocessor.fit_transform(X_train)
X_valid_trans = preprocessor.transform(X_valid)
X_test_trans = preprocessor.transform(test_df[features])

# ğŸ”� Feature Selection
mi_scores = mutual_info_classif(X_train_trans, y_train)
selected = np.argsort(mi_scores)[-20:]
X_train_trans = X_train_trans[:, selected]
X_valid_trans = X_valid_trans[:, selected]
X_test_trans = X_test_trans[:, selected]

gb = GradientBoostingClassifier(random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.7, 0.8, 1.0],
    'max_features': ['sqrt', 'log2']
}

#  Models to Train
models = {
    "Logistic Regression": LogisticRegressionCV(
        # Cs=np.logspace(-4, 4, 20),
        cv=5,
        Cs=np.logspace(-3, 3, 30),
        scoring='f1',
        penalty='elasticnet',
        solver='saga',
        class_weight='balanced',
        l1_ratios=[0.1, 0.3, 0.5, 0.7, 0.9],
        max_iter=10000,
        n_jobs=-1
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        use_label_encoder=False
    )
}

#  Evaluation Functions
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='plasma')
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()

def plot_model_scores(scores, metric_name):
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(scores.keys()), y=list(scores.values()), palette="viridis")
    plt.xlabel("Models")
    plt.ylabel(metric_name)
    plt.title(f"Model Comparison - {metric_name}")
    plt.show()

#  Training & Evaluation
auc_scores, accuracy_scores, f1_scores = {}, {}, {}

for name, model in models.items():
    model.fit(X_train_trans, y_train)
    y_pred = model.predict(X_valid_trans)
    y_pred_proba = model.predict_proba(X_valid_trans)[:, 1]

    auc_scores[name] = roc_auc_score(y_valid, y_pred_proba)
    accuracy_scores[name] = accuracy_score(y_valid, y_pred)
    f1_scores[name] = f1_score(y_valid, y_pred)

    print(f"{name} AUC-ROC Score: {auc_scores[name]:.4f}")
    plot_confusion_matrix(confusion_matrix(y_valid, y_pred), name)

# Plot Comparison
plot_model_scores(auc_scores, "AUC-ROC Score")
plot_model_scores(accuracy_scores, "Accuracy Score")
plot_model_scores(f1_scores, "F1 Score")


# Final Submission with Stacking Model
final_model = models["Logistic Regression"]
final_model.fit(np.vstack((X_train_trans, X_valid_trans)), np.concatenate((y_train, y_valid)))
test_df[target] = final_model.predict_proba(X_test_trans)[:, 1]
submission = test_df[[id_col, target]]
submission_path = "/kaggle/working/submission_lr_f1.csv"
submission.to_csv(submission_path, index=False)
print(f" Submission file saved at: {submission_path}")

