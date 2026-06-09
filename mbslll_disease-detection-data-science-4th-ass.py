import numpy as np
import pandas as pd

import joblib
import os
os.mkdir('./predictions/')
os.mkdir('./models/')

from sklearn.model_selection import GridSearchCV

from sklearn.inspection import permutation_importance

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, classification_report

import seaborn as sns
import matplotlib.pylab as plt
plt.style.use('ggplot')

pd.set_option('display.max_columns', 35)


df = pd.read_csv('../input/datascience-4-competition/train.csv')


df


plt.figure(figsize=(8, 6))
sns.heatmap(df.corr())


sns.barplot(df['label'].value_counts(normalize=True))


X, y = df.drop(columns=['label', 'ID', 'feature57', 'feature58', 'feature60', 'feature61']), df['label']


# Perform stratified train/test split to preserve class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


models = {
    "SVM": SVC(probability=True),
    "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
    "Naive Bayes": BernoulliNB(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(estimator=DecisionTreeClassifier(), random_state=42),
    "XGBoost": XGBClassifier(random_state=42),
    "LDA": LinearDiscriminantAnalysis(),
}


param_grids = {
    "Logistic Regression": [
        {
            'penalty': ['l1'],
            'solver': ['liblinear', 'saga'],
            'C': [0.01, 0.1, 1.0, 10.0]
        },
        {
            'penalty': ['l2'],
            'solver': ['liblinear', 'lbfgs', 'sag', 'saga'],
            'C': [0.01, 0.1, 1.0, 10.0]
        },
        {
            'penalty': ['elasticnet'],
            'solver': ['saga'],
            'C': [0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.0, 0.5, 1.0]
        },
        {
            'penalty': [None],
            'solver': ['lbfgs', 'sag', 'saga']
        }
    ],

    "Naive Bayes": {
        'alpha': [0.1, 0.5, 1.0, 5.0, 10.0],
        'fit_prior': [True, False]
    },
    "SVM": {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    },
    "Random Forest": {
        'n_estimators': [200, 300],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2],
        'max_features': ['log2'],
        'bootstrap': [True]
    },
    "AdaBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0],
        'estimator__max_depth': [1, 3, 5],
        'estimator__min_samples_split': [2, 5],
        'estimator__min_samples_leaf': [1, 2]
    },
    "XGBoost": {
        'learning_rate': [0.03, 0.1],
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'min_child_weight': [1, 3],
        'subsample': [0.7, 1.0],
        'colsample_bytree': [0.7, 1.0],
        'gamma': [0, 0.5],
        'reg_alpha': [0, 1.0],
        'reg_lambda': [1.0]
    },
    "LDA": [
        {
            'solver': ['svd'],  # svd must be used without shrinkage
        },
        {
            'solver': ['lsqr', 'eigen'],
            'shrinkage': [None, 'auto', 0.0, 0.5, 1.0]  # valid only with these solvers
        }
    ]
}


best_models = {}
best_params = {}

for name in models:
    print(f"\nğŸ”� Running GridSearchCV for {name}...")
    
    grid_search = GridSearchCV(
        estimator=models[name],
        param_grid=param_grids[name],
        scoring='accuracy',
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    best_models[name] = grid_search.best_estimator_
    best_params[name] = grid_search.best_params_  # ğŸ”� Save the parameters
    
    print(f"âœ… Best Params for {name}: {grid_search.best_params_}")
    print(f"ğŸ�¯ Best Score: {grid_search.best_score_:.4f}")

# Save each model
for name, model in best_models.items():
    joblib.dump(model, f"./models/{name.replace(' ', '_')}_best_model.pkl")


best_models = {}
for name in models:
    model_path = f"./models/{name.replace(' ', '_')}_best_model.pkl"
    best_models[name] = joblib.load(model_path)


test_df = pd.read_csv('../input/datascience-4-competition/test.csv')
X_test_kaggle = test_df[[f'feature{i}' for i in range(64)]].drop(columns=['feature57', 'feature58', 'feature60', 'feature61'])
X_test_kaggle_scaled = scaler.transform(X_test_kaggle)


# Combine loaded models for soft voting
voting_clf = VotingClassifier(
    estimators=[
        (k, v) for k,v in best_models.items()
    ],
    voting='soft',  # Use soft voting
    n_jobs=-1
)

# Fit ensemble on training data
voting_clf.fit(X_train_scaled, y_train)

# Predictions
y_train_pred = voting_clf.predict(X_train_scaled)
y_test_pred = voting_clf.predict(X_test_scaled)

# Accuracy
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

name = 'Voting'

# Output results
print(f"\nğŸ§  VotingClassifier:")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")
print("Classification Report (Test):")
print(classification_report(y_test, y_test_pred))


y_pred = voting_clf.predict(X_test_kaggle_scaled)

# Create a DataFrame with predictions
pred_df = pd.DataFrame({
    'ID': np.arange(y_pred.shape[0]),
    'label': y_pred
})

# Clean model name for filename (no spaces, special chars)
safe_name = name.replace(" ", "_").replace("/", "_")

# Save to CSV
pred_df.to_csv(f"./predictions/{safe_name}_predictions.csv", index=False)

print(f"Saved predictions for {name} to {safe_name}_predictions.csv")


# Train and evaluate
for name, model in best_models.items():
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Accuracy
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    # Output results
    print(f"\n{name}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print("Classification Report (Test):")
    print(classification_report(y_test, y_test_pred))


for name, model in best_models.items():
    y_pred = model.predict(X_test_kaggle_scaled)

    
    # Create a DataFrame with predictions
    pred_df = pd.DataFrame({
        'ID': np.arange(y_pred.shape[0]),
        'label': y_pred
    })

    # pred_df.reset_index(drop=True, inplace=True)
    # print(pred_df)

    # Clean model name for filename (no spaces, special chars)
    safe_name = name.replace(" ", "_").replace("/", "_")

    # Save to CSV
    pred_df.to_csv(f"./predictions/{safe_name}_predictions.csv", index=False)

    print(f"Saved predictions for {name} to {safe_name}_predictions.csv")


# Compute mean and std
importances = best_models['Random Forest'].feature_importances_
std = np.std([tree.feature_importances_ for tree in best_models['Random Forest'].estimators_], axis=0)

# Create Series and sort top features
forest_importances = pd.Series(importances, index=X_train.columns)
top_n = 10
top_features = forest_importances.sort_values(ascending=False).head(top_n)
top_std = std[top_features.index.map(lambda x: X_train.columns.get_loc(x))]

# Plot
plt.figure(figsize=(10, 6))
plt.bar(x=top_features.index, height=top_features.values, yerr=top_std, capsize=5)
plt.title("Feature Importances Using MDI (Random Forest)")
plt.ylabel("Mean Decrease in Impurity")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Run permutation importance
result = permutation_importance(
    best_models['Random Forest'],
    X_test_scaled,
    y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=2
)

# Create a DataFrame for top N features
importances = result.importances_mean
std = result.importances_std
feature_names = X_train.columns

# Sort and select top 10
indices = np.argsort(importances)[::-1]
top_n = 10
top_indices = indices[:top_n]
top_features = feature_names[top_indices]
top_importances = importances[top_indices]
top_std = std[top_indices]

# Plot with error bars
plt.figure(figsize=(10, 6))
plt.bar(x=top_features, height=top_importances, yerr=top_std, capsize=5)
plt.title("Feature Importances Using Permutation (Random Forest)")
plt.ylabel("Mean Accuracy Decrease")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

