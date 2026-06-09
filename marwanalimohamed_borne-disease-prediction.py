# Clean uninstall first
!pip uninstall -y scikit-learn imbalanced-learn

# Force reinstall correct compatible versions
!pip install --quiet scikit-learn==1.3.2 imbalanced-learn==0.11.0 xgboost lightgbm statsmodels


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier

from imblearn.over_sampling import SMOTE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBClassifier


train = pd.read_csv("/kaggle/input/playground-series-s3e13/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s3e13/test.csv")


train.head(5)


print("Train shape:", train.shape)


print("\nTrain info:")
print(train.info())


print("Missing values in train:")
print(train.isna().sum())


print("Duplicated rows in train:", train.duplicated().sum())


print("\nStatistical summary of train:")
train.describe()


# assume last column in train is the target
target = train.columns[-1]
print("Target column:", target)


plt.figure(figsize=(8,5))
sns.countplot(x='prognosis', data=train)
plt.title("Target variable distribution")
plt.xticks(rotation=55)
plt.show()


# Correlation heatmap للـ numeric columns
plt.figure(figsize=(12,8))
corr = train.corr(numeric_only=True)
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Correlation heatmap")
plt.show()


# Variance Inflation Factor (VIF) for multicollinearity
X_numeric = train.drop(columns=["prognosis"]).select_dtypes(include=[np.number])
vif_data = pd.DataFrame()
vif_data["feature"] = X_numeric.columns
vif_data["VIF"] = [variance_inflation_factor(X_numeric.values, i) for i in range(len(X_numeric.columns))]
print("\nVariance Inflation Factors (VIF):\n", vif_data.sort_values("VIF", ascending=False))


# Features & Target
X = train.drop(columns=["prognosis"])
y = train["prognosis"]

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)


# Check imbalance
from collections import Counter
if max(Counter(y).values()) / min(Counter(y).values()) > 2:
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)
    print("Applied SMOTE. New class distribution:", Counter(y))
else:
    print("No significant imbalance detected.")


# Outlier Detection (IQR Method)
# =========================
Q1 = X_numeric.quantile(0.25)
Q3 = X_numeric.quantile(0.75)
IQR = Q3 - Q1
outliers = ((X_numeric < (Q1 - 1.5 * IQR)) | (X_numeric > (Q3 + 1.5 * IQR))).sum()
print("\nNumber of outliers per feature:\n", outliers)


use_robust = (outliers.sum() > 0.05 * X_numeric.size)
if use_robust:
    print("Using RobustScaler due to outliers.")
    scaler = RobustScaler()
else:
    print("Using StandardScaler.")
    scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("\nBefore SMOTE:\n", pd.Series(y_train).value_counts())
print("\nAfter SMOTE:\n", pd.Series(y_resampled).value_counts())


models = {
"Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "KNN": KNeighborsClassifier(n_neighbors=7),  # doesn't support class_weight
    "SVM": SVC(C=1, kernel="rbf", class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42)
}


param_grid = {
    "Logistic Regression": {"C": [0.1, 1, 10]},
    "KNN": {"n_neighbors": [3, 5, 7]},
    "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    "Decision Tree": {"max_depth": [None, 5, 10]},
    "Random Forest": {"n_estimators": [50, 100], "max_depth": [None, 10]}
}


for name, model in models.items():
    print(f"\n=== {name} (Before Tuning) ===")
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Accuracy
    print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Validation Accuracy:", accuracy_score(y_test, y_test_pred))
    
    # Classification report
    print("\nClassification Report:\n", classification_report(y_test, y_test_pred, target_names=le.classes_))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap="Blues", xticks_rotation=90)
    plt.title(f"Confusion Matrix - {name} (Before Tuning)")
    plt.show()


best_models = {}
for name, model in models.items():
    print(f"\n=== {name} (Hyperparameter Tuning) ===")
    grid = GridSearchCV(model, param_grid[name], cv=3, n_jobs=-1, scoring="accuracy")
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_
    print(f"Best Params: {grid.best_params_}")
    
    # Predictions
    y_train_pred = best_models[name].predict(X_train)
    y_test_pred = best_models[name].predict(X_test)
    
    # Accuracy
    print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Validation Accuracy:", accuracy_score(y_test, y_test_pred))
    
    # Classification report
    print("\nClassification Report:\n", classification_report(y_test, y_test_pred, target_names=le.classes_))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap="Blues", xticks_rotation=90)
    plt.title(f"Confusion Matrix - {name} (After Tuning)")
    plt.show()



voting_clf = VotingClassifier(
    estimators=[
        ("lr", models["Logistic Regression"]),
        ("knn", models["KNN"]),
        ("svm", models["SVM"]),
        ("dt", models["Decision Tree"]),
        ("rf", models["Random Forest"])
    ],
    voting="hard"
)


voting_clf.fit(X_resampled, y_resampled)
y_test_pred = voting_clf.predict(X_test)

print("\n=== Voting Classifier (Ensemble) ===")
print("Validation Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

cm = confusion_matrix(y_test, y_test_pred, labels=np.unique(y_test))
plt.figure(figsize=(10,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test), cmap="Blues")
plt.title("Confusion Matrix - Voting Classifier")
plt.show()


stacking_clf = StackingClassifier(
    estimators=[
        ("lr", models["Logistic Regression"]),
        ("knn", models["KNN"]),
        ("svm", models["SVM"]),
        ("dt", models["Decision Tree"]),
        ("rf", models["Random Forest"])
    ],
    final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
    n_jobs=-1
)


stacking_clf.fit(X_resampled, y_resampled)
y_test_pred = stacking_clf.predict(X_test)

print("\n=== Stacking Classifier ===")
print("Validation Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

cm = confusion_matrix(y_test, y_test_pred, labels=np.unique(y_test))
plt.figure(figsize=(10,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test),
            cmap="Blues")
plt.title("Confusion Matrix - Stacking Classifier")
plt.show()


xgb_clf = XGBClassifier(
    objective="multi:softmax",
    num_class=len(np.unique(y)),
    eval_metric="mlogloss",
    use_label_encoder=False,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)


xgb_clf.fit(X_resampled, y_resampled)
y_test_pred = xgb_clf.predict(X_test)

print("\n=== XGBoost Classifier ===")
print("Validation Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

cm = confusion_matrix(y_test, y_test_pred, labels=np.unique(y_test))
plt.figure(figsize=(10,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test),
            cmap="Blues")
plt.title("Confusion Matrix - XGBoost Classifier")
plt.show()


param_grid_lr = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["liblinear", "saga"]
}

grid_lr = GridSearchCV(
    LogisticRegression(max_iter=1000, class_weight="balanced"),
    param_grid_lr,
    cv=3,
    n_jobs=-1
)


grid_lr.fit(X_resampled, y_resampled)
print("\nBest Logistic Regression Params:", grid_lr.best_params_)
y_test_pred = grid_lr.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))


param_grid_knn = {
    "n_neighbors": [3, 5, 7, 9, 11],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}

grid_knn = GridSearchCV(
    KNeighborsClassifier(),
    param_grid_knn,
    cv=3,
    n_jobs=-1
)


grid_knn.fit(X_resampled, y_resampled)
print("\nBest KNN Params:", grid_knn.best_params_)
y_test_pred = grid_knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))


param_grid_svm = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf", "poly"],
    "gamma": ["scale", "auto"]
}

grid_svm = GridSearchCV(
    SVC(class_weight="balanced"),
    param_grid_svm,
    cv=3,
    n_jobs=-1
)


grid_svm.fit(X_resampled, y_resampled)
print("\nBest SVM Params:", grid_svm.best_params_)
y_test_pred = grid_svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))


param_grid_dt = {
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"]
}

grid_dt = GridSearchCV(
    DecisionTreeClassifier(class_weight="balanced"),
    param_grid_dt,
    cv=3,
    n_jobs=-1
)


grid_dt.fit(X_resampled, y_resampled)
print("\nBest Decision Tree Params:", grid_dt.best_params_)
y_test_pred = grid_dt.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))


param_grid_rf = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "bootstrap": [True, False]
}

grid_rf = RandomizedSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=42),
    param_grid_rf,
    n_iter=10,
    cv=3,
    n_jobs=-1,
    random_state=42
)


grid_rf.fit(X_resampled, y_resampled)
print("\nBest Random Forest Params:", grid_rf.best_params_)
y_test_pred = grid_rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))


param_grid_xgb = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

grid_xgb = RandomizedSearchCV(
    XGBClassifier(
        objective="multi:softmax",
        num_class=len(np.unique(y)),
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42
    ),
    param_grid_xgb,
    n_iter=10,
    cv=3,
    n_jobs=-1,
    random_state=42
)


grid_xgb.fit(X_resampled, y_resampled)
print("\nBest XGBoost Params:", grid_xgb.best_params_)
y_test_pred = grid_xgb.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))


train = pd.read_csv("/kaggle/input/playground-series-s3e13/train.csv")
X = train.drop(columns=["id", "prognosis"], errors="ignore")
y = train["prognosis"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

best_model = XGBClassifier(random_state=42)
best_model.fit(X_scaled, y_encoded)

test = pd.read_csv("/kaggle/input/playground-series-s3e13/test.csv")
test_ids = test["id"]
X_test = test.drop(columns=["id"], errors="ignore")
X_test_scaled = scaler.transform(X_test)

y_test_pred = best_model.predict(X_test_scaled)
y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)

submission = pd.DataFrame({
    "id": test_ids,
    "prognosis": y_test_pred_labels
})

submission.to_csv("submission.csv", index=False)
print("submission.csv is Ready")


submission.head(5)




