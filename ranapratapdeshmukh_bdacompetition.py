!pip install -U scikit-learn imbalanced-learn --quiet


!pip install --upgrade --quiet scikit-learn xgboost lightgbm catboost


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

print("Train Data Shape:", train.shape)
print("Test Data Shape:", test.shape)
display(train.head())
display(test.head())


# Missing values
print("Missing values in Train:")
print(train.isna().sum())

print("\nMissing values in test dataset:")
print(test.isna().sum())

# Visualize rainfall distribution
sns.countplot(x='rainfall', data=train)
plt.title("Rainfall Class Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(train.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


# Fill NA with mean
train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(), inplace=True)

# Drop columns
train.drop(columns=['maxtemp', 'mintemp'], inplace=True)
test.drop(columns=['maxtemp', 'mintemp'], inplace=True)


poly = PolynomialFeatures(degree=3, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(train.drop(columns=['rainfall']))
X_test_poly = poly.transform(test)

# Extract feature names (optional)
feature_names = poly.get_feature_names_out(train.drop(columns=['rainfall']).columns)
X = pd.DataFrame(X_poly, columns=feature_names)
X_test = pd.DataFrame(X_test_poly, columns=feature_names)
y = train['rainfall']


smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

sns.countplot(x=y)
plt.title("Balanced Target Distribution after SMOTE")
plt.show()


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


param_grids = {
    "RandomForest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20]
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2]
    },
    "LightGBM": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2]
    },
    "CatBoost": {
        "iterations": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2]
    },
    "LogisticRegression": {
        "C": [0.1, 1, 10],
        "penalty": ["l2"]
    }
}

best_model = None
best_val_accuracy = 0

for name, params in param_grids.items():
    print(f"Tuning {name}...")
    if name == "RandomForest":
        model = RandomForestClassifier(random_state=42)
    elif name == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    elif name == "LightGBM":
        model = LGBMClassifier(random_state=42)
    elif name == "CatBoost":
        model = CatBoostClassifier(verbose=0, random_state=42)
    elif name == "LogisticRegression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    
    grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model_instance = grid_search.best_estimator_
    
    train_acc = accuracy_score(y_train, best_model_instance.predict(X_train))
    val_acc = accuracy_score(y_val, best_model_instance.predict(X_val))
    
    print(f"{name} Best Params: {grid_search.best_params_}")
    print(f"{name} Train Accuracy: {train_acc}")
    print(f"{name} Validation Accuracy: {val_acc}")
    print(classification_report(y_val, best_model_instance.predict(X_val)))
    
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        best_model = best_model_instance

print(f"\nğŸ�† Best Performing Individual Model:\n{best_model}")


stacking_clf = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=200, learning_rate=0.1, eval_metric='logloss', random_state=42)),
        ('lgbm', LGBMClassifier(n_estimators=200, learning_rate=0.1, random_state=42))
    ],
    final_estimator=LogisticRegression(),
    passthrough=True
)
stacking_clf.fit(X_train, y_train)

stacking_val_acc = accuracy_score(y_val, stacking_clf.predict(X_val))
print(f"\nğŸ“Š Stacking Classifier Validation Accuracy: {stacking_val_acc:.4f}")


# rainfall_prob = best_model.predict_proba(X_test)[:, 1]
# submission = pd.DataFrame({'id': test.index, 'rainfall': rainfall_prob})
# submission.to_csv("submission.csv", index=False)


# rainfall_prob = stacking_clf.predict_proba(X_test)[:, 1]
# submission = pd.DataFrame({'id': test.index, 'rainfall': rainfall_prob})
# submission.to_csv("submission.csv", index=False)

