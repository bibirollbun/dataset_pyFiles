import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# To ignore warinings
import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Load the training data
try:
    train_df = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/train.csv', header = None)
    labels_df = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/trainLabels.csv', header = None)
    test_df = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/test.csv', header = None)
except FileNotFoundError:
    print("Make sure 'train.csv', 'trainLabels.csv', and 'test.csv' are in the current directory.")
    exit()

# Separate features (X) and target (y)
X = train_df.values
y = labels_df.values.ravel()

print("Training data shape:", X.shape)
print("Training labels shape:", y.shape)
print("Test data shape:", test_df.shape)


# Check class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title('Class Distribution')
plt.xlabel('Class (0 or 1)')
plt.ylabel('Number of Samples')
plt.show()

# Check feature distributions (let's look at the first few)
plt.figure(figsize=(15, 10))
for i in range(5):
    plt.subplot(2, 3, i + 1)
    sns.histplot(X[:, i], kde=True)
    plt.title(f'Feature {i+1} Distribution')
plt.tight_layout()
plt.show()

# Check for correlations (using a sample to avoid computational overhead for all 40 features initially)
sample_indices = np.random.choice(X.shape[1], size=10, replace=False)
correlation_matrix_sample = pd.DataFrame(X[:, sample_indices]).corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix_sample, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix of a Sample of Features')
plt.show()


# Split data for initial evaluation (optional, cross-validation is the main evaluation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def preprocess_data(X_train, X_val=None, X_test=None, scaler_type='standard'):
    """Applies scaling to the training, validation, and test sets."""
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid scaler_type. Choose 'standard' or 'minmax'.")

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

# Apply StandardScaler
X_train_scaled, X_val_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_val, test_df.values, scaler_type='standard')

print("Scaled training data shape:", X_train_scaled.shape)
if X_val_scaled is not None:
    print("Scaled validation data shape:", X_val_scaled.shape)
print("Scaled test data shape:", X_test_scaled.shape)


models = {
    'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear'),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42, probability=True), # probability=True for ROC curve
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', model)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1_weighted') # Using F1-weighted for potential imbalance
    results[name] = cv_scores.mean()
    print(f'{name}: Mean F1 = {cv_scores.mean():.4f}, Std F1 = {cv_scores.std():.4f}')

# Display model comparison table
model_comparison = pd.DataFrame.from_dict(results, orient='index', columns=['Mean F1 Score'])
model_comparison = model_comparison.sort_values(by='Mean F1 Score', ascending=False)
print("\nModel Comparison (Mean F1 Score):\n", model_comparison)


param_grids = {
    'Logistic Regression': {
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'classifier__penalty': ['l1', 'l2']
    },
    'Random Forest': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 3, 5]
    },
    'SVM': {
        'classifier__C': [0.1, 1, 10],
        'classifier__gamma': ['scale', 'auto', 0.1, 1]
    },
    'Gradient Boosting': {
        'classifier__n_estimators': [50, 100],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__max_depth': [3, 5]
    }
}

best_models = {}
for name, model in models.items():
    print(f"\nTuning hyperparameters for {name}:")
    pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', model)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grids[name], cv=cv, scoring='f1_weighted', n_jobs=-1, verbose=1)
    grid_search.fit(X, y)
    best_models[name] = grid_search.best_estimator_
    print(f"Best parameters for {name}: {grid_search.best_params_}")

print("\nBest Models:\n", best_models)


plt.figure(figsize=(12, 8))
for name, best_model in best_models.items():
    y_pred_proba = best_model.predict_proba(X_val_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    auc_score = roc_auc_score(y_val, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 8))
for name, best_model in best_models.items():
    y_pred_proba = best_model.predict_proba(X_val_scaled)[:, 1]
    precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f'{name} (PR AUC = {pr_auc:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison')
plt.legend()
plt.grid(True)
plt.show()

# Confusion Matrices
for name, best_model in best_models.items():
    y_pred = best_model.predict(X_val_scaled)
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative (0)', 'Positive (1)'], 
                yticklabels=['Negative (0)', 'Positive (1)'])
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Feature Importance (for tree-based models)
if 'Random Forest' in best_models:
    rf_model = best_models['Random Forest'].named_steps['classifier']
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances (Random Forest)")
    plt.bar(range(10), importances[indices[:10]], align="center")
    plt.xticks(range(10), indices[:10] + 1) # Feature numbers
    plt.xlim([-1, 10])
    plt.xlabel("Feature Number")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()

if 'Gradient Boosting' in best_models:
    gb_model = best_models['Gradient Boosting'].named_steps['classifier']
    importances = gb_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances (Gradient Boosting)")
    plt.bar(range(10), importances[indices[:10]], align="center")
    plt.xticks(range(10), indices[:10] + 1) # Feature numbers
    plt.xlim([-1, 10])
    plt.xlabel("Feature Number")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# Load the training and test data
try:
    train_df = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/train.csv', header=None)
    labels_df = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/trainLabels.csv', header=None)
    test_df = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/test.csv', header=None)
except FileNotFoundError:
    print("Make sure 'train.csv', 'trainLabels.csv', and 'test.csv' are in the current directory.")
    exit()

# Separate features (X) and target (y)
X = train_df.values
y = labels_df.values.ravel()  # Use .ravel() to flatten the array

# Create the final pipeline with the best SVM model
final_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('classifier', SVC(C=10, probability=True, random_state=42))  # Best SVM hyperparameters
])

# Train the final pipeline on the entire training data
final_pipeline.fit(X, y)

# Make predictions on the test data
test_predictions = final_pipeline.predict(test_df.values)

# Create a submission file
id_column = [i for i in range(1, 9001)]  # Generate the sequence 1 to 9000
submission_df = pd.DataFrame({'Id': id_column, 'Solution': test_predictions})
submission_df.to_csv('submission.csv', index=False)
print("\nSubmission file 'submission.csv' created successfully!")
print("\nFirst few rows of the submission file:")
print(submission_df.head())


# Separate features (X) and target (y)
X = train_df.values
y = labels_df.values.ravel()  # Use .ravel() to flatten the array
X_test = test_df.values

# Combine train and test data for GMM fitting
X_all = np.vstack((X, X_test))

# Apply Gaussian Mixture Model for feature engineering
n_components = 4  # You can tune this parameter
gmm = GaussianMixture(n_components=n_components, random_state=42)
gmm.fit(X_all)

# Create new features based on GMM probabilities
X_train_gmm = gmm.predict_proba(X)
X_test_gmm = gmm.predict_proba(X_test)

# Define the SVM pipeline
svm_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('classifier', SVC(random_state=42, probability=True))
])

# Define the parameter grid for GridSearchCV
param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__gamma': ['scale', 'auto', 0.1, 1]
}

# Perform GridSearchCV with StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search_svm = GridSearchCV(svm_pipeline, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1, verbose=1)

# Train the GridSearchCV on the GMM-transformed training data
grid_search_svm.fit(X_train_gmm, y)

# Get the best SVM model
best_svm = grid_search_svm.best_estimator_
print("Best SVM parameters:", grid_search_svm.best_params_)

# Make predictions on the GMM-transformed test data
test_predictions = best_svm.predict(X_test_gmm)

# Create the submission file
id_column = [i for i in range(1, 9001)]
submission_df = pd.DataFrame({'Id': id_column, 'Solution': test_predictions})
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully!")


# Separate features (X) and target (y)
X = train_df.values
y = labels_df.values.ravel()
X_test = test_df.values
X_all = np.vstack((X, X_test))

# Apply Gaussian Mixture Model
n_components = 4
gmm = GaussianMixture(n_components=n_components, random_state=42)
gmm.fit(X_all)
clusters = gmm.predict(X_all)

# Visualize using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_all)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='viridis')
plt.title(f'GMM Visualization in 2D (PCA), {n_components} components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Visualize using t-SNE (can be slower)
tsne = TSNE(n_components=2, random_state=42, n_iter=300)
X_tsne = tsne.fit_transform(X_all)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=clusters, palette='viridis')
plt.title(f'GMM Visualization in 2D (t-SNE), {n_components} components')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()

