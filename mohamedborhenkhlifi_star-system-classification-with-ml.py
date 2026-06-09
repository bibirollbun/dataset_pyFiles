# Data manipulation and visualization
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Model selection and evaluation
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Advanced ensemble methods
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/orbyx-ml-challenge-star-system-classification/train.csv")
test = pd.read_csv("/kaggle/input/orbyx-ml-challenge-star-system-classification/test.csv")
sample_submission = pd.read_csv("/kaggle/input/orbyx-ml-challenge-star-system-classification/sample_submission.csv")


print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()


train.describe()


train.info()


#determine the number of null values in each feature
train.isnull().sum()


numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns

train[numeric_cols].hist(figsize=(15,10), bins=30)
plt.suptitle("Numerical Feature Distributions")
plt.show()


categorical_cols = train.select_dtypes(include=['object']).columns
for col in categorical_cols:
    train[col].value_counts().plot(kind='bar', figsize=(6,3))
    plt.title(f"Distribution of {col}")
    plt.show()


def corolation_Matrix(df):
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()


corolation_Matrix(train)


train_processed=train.copy()


# Fill numeric with median
for col in numeric_cols:
    train_processed[col] = train_processed[col].fillna(train_processed[col].median())


 # One-Hot Encode categorical variables
train_processed = pd.get_dummies(
    train_processed,
    columns=categorical_cols,
    drop_first=True
)


train_processed.shape


train_processed


# target
y = train_processed["system_type"]


# feature engineering
train_processed['stellar_density'] = train_processed['star_mass'] / (train_processed['star_size'] ** 3)


features_to_exclude = ["star_size", "system_type"]
features = [col for col in train_processed.columns if col not in features_to_exclude]
    
train_processed = train_processed[features]
train_processed.shape


# scaling
numeric_cols = [col for col in numeric_cols if col not in features_to_exclude]
scaler = StandardScaler()
train_processed[numeric_cols] = scaler.fit_transform(train_processed[numeric_cols])
    
X = train_processed


X.head()


from sklearn.model_selection import cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=42)
# Define Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(
        eval_metric='logloss',  
        use_label_encoder=False
    ),
    "CatBoost": CatBoostClassifier(verbose=0),
    "Extra Trees": ExtraTreesClassifier(  
        n_estimators=100, 
        random_state=1
    ),
    "LightGBM": LGBMClassifier(verbose=-1), 
    "SVM": SVC(probability=True), 
    "K-Neighbors": KNeighborsClassifier(),
}

# Train and cross-validate each model
cv_results = {}

print("\nModel Training and Cross-Validation:")
for name, model in models.items():
    # Use 'accuracy' scoring instead of 'neg_mean_squared_error'
    scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=5)
    cv_results[name] = {
        'mean_accuracy': scores.mean(),
        'std_accuracy': scores.std()
    }
    model.fit(X_train, y_train)
    print(f"{name} trained. Mean CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# Display sorted cross-validation results (highest accuracy first)
print("\nCross-Validation Results (Sorted by Mean Accuracy):")
sorted_cv_results = sorted(cv_results.items(), key=lambda x: x[1]['mean_accuracy'], reverse=True)
for name, result in sorted_cv_results:
    print(f"{name} - Mean Accuracy: {result['mean_accuracy']:.4f}, Std Dev: {result['std_accuracy']:.4f}")


# CatBoost Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

print("=== CatBoost Hyperparameter Tuning ===")

catboost_param_grid = {
    'iterations': [100, 200, 500],
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'l2_leaf_reg': [1, 3, 5]
}

catboost = CatBoostClassifier(verbose=0, random_state=42)
catboost_grid = GridSearchCV(
    catboost, 
    catboost_param_grid, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
catboost_grid.fit(X_train, y_train)

print(f"Best CatBoost Parameters: {catboost_grid.best_params_}")
print(f"Best CatBoost CV Accuracy: {catboost_grid.best_score_:.4f}")

# Store the best model
best_catboost = catboost_grid.best_estimator_


# LightGBM Hyperparameter Tuning
from sklearn.model_selection import RandomizedSearchCV

print("\n=== LightGBM Randomized Search ===")

lgbm_param_dist = {
    'n_estimators': [50, 100, 150],  # Fewer values
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [20, 31, 40],
    'min_child_samples': [20, 30, 40],
    'subsample': [0.8, 0.9, 1.0]
}

lgbm = LGBMClassifier(
    random_state=42,
    verbose=-1,
    n_jobs=1
)

lgbm_random = RandomizedSearchCV(
    lgbm,
    lgbm_param_dist,
    n_iter=20,  
    cv=3,  
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

lgbm_random.fit(X_train, y_train)

print(f"Best LightGBM Parameters: {lgbm_random.best_params_}")
print(f"Best LightGBM CV Accuracy: {lgbm_random.best_score_:.4f}")

best_lgbm = lgbm_random.best_estimator_


# Evaluate Models

models = {
    "LightGBM (Tuned)": best_lgbm,
    "CatBoost (Tuned)": best_catboost,
}

print("\nCross-Validation Results:")
print("=" * 50)

cv_results = {}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=5)
    cv_results[name] = {
        'mean_accuracy': scores.mean(),
        'std_accuracy': scores.std()
    }
    print(f"{name} - Mean CV Accuracy: {scores.mean():.4f}, Std Dev: {scores.std():.4f}")

# Evaluate the best models on the test set
print("\n" + "=" * 50)
print("Test Set Evaluation:")
print("=" * 50)

for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} - Accuracy: {accuracy:.4f}")


# Create an ensemble using VotingClassifier
ensemble_model = VotingClassifier(estimators=[
    ('LightGBM (Tuned)', best_lgbm),
    ('CatBoost (Tuned)', best_catboost)
], voting='soft')  # 'soft' for probability-based voting

# Fit the ensemble model
ensemble_model.fit(X_train, y_train)

# Evaluate only the ensemble model using cross-validation
print("\nCross-Validation Results:")
print("=" * 50)

models = {
    'Ensemble (LightGBM + CatBoost)': ensemble_model
}

cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=5)
    cv_results[name] = {
        'mean_accuracy': scores.mean(),
        'std_accuracy': scores.std()
    }
    print(f"{name} - Mean CV Accuracy: {scores.mean():.4f}, Std Dev: {scores.std():.4f}")

# Evaluate the ensemble model on the test set
print("\n" + "=" * 50)
print("Test Set Evaluation:")
print("=" * 50)

for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} - Accuracy: {accuracy:.4f}")


# divide features into numerical and categorical 
numeric_cols = test.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = test.select_dtypes(include=['object']).columns


test_processed=test.copy()
# Fill numeric with median
for col in numeric_cols:
    test_processed[col] = test_processed[col].fillna(test_processed[col].median())


 # One-Hot Encode categorical variables
test_processed = pd.get_dummies(
    test_processed,
    columns=categorical_cols,
    drop_first=True
)


test_processed['stellar_density'] = test_processed['star_mass'] / (test_processed['star_size'] ** 3)


features_to_exclude = ["star_size","id"]
features = [col for col in test_processed.columns if col not in features_to_exclude]
    
test_processed = test_processed[features]


numeric_cols = [col for col in numeric_cols if col not in features_to_exclude]
scaler = StandardScaler()
test_processed[numeric_cols] = scaler.fit_transform(test_processed[numeric_cols])


def prepare_final_model(X,y):
    final_model = ensemble_model 
    final_model.fit(X, y)
    return final_model
final_model = prepare_final_model(X, y)


# Make predictions on test data
test_predictions = final_model.predict(test_processed)


# Create submission
submission = sample_submission.copy()
submission["system_type"] = test_predictions

# Save submission file
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")




