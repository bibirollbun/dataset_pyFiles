# ğŸ“¦ Install dependencies
# We use `ydata-profiling` (formerly `pandas-profiling`) to automatically generate a comprehensive EDA report.

!pip install -U ydata-profiling[notebook] catboost > /dev/null 2>&1


# ============================================================
# ğŸ“¦ IMPORTS FOR MACHINE LEARNING PIPELINES
# Organized imports for a typical ML project: data handling, visualization,
# model training, and evaluation.
# Each section is clearly documented for readability and maintainability.
# ============================================================

# ============================================================
# ğŸš€ Core & OS Utilities
# These modules provide fundamental operating system access and utilities.
# Useful for interacting with the filesystem, environment variables, and generating randomness.
# ============================================================
import os                # Standard library module for interacting with the operating system (e.g., paths, files, env vars) ğŸ�š
import random            # Standard library module for generating random numbers and choices ğŸ�²

# ============================================================
# ğŸ”¢ Numerical & Data Handling
# Essential libraries for numerical computations and data manipulation.
# Numpy: efficient numerical arrays and matrix operations.
# Pandas: high-level data structures and data manipulation tools.
# YData Profiling: for automated exploratory data analysis reports.
# ============================================================
import numpy as np       # Numerical computation library; provides ndarray and linear algebra capabilities âš¡
import pandas as pd       # Data manipulation and analysis library; provides DataFrame and Series abstractions ğŸ§™â€�â™‚ï¸�
from ydata_profiling import ProfileReport  # Automated EDA report generator; explores datasets and outputs a rich HTML report ğŸ§ª

# ============================================================
# ğŸŒ� Kaggle & Data Access
# Library to fetch datasets and notebooks directly from Kaggle.
# Useful when participating in Kaggle competitions or using Kaggle-hosted datasets.
# ============================================================
import kagglehub         # Seamless integration with Kaggle datasets and models ğŸ“¥

# ============================================================
# ğŸ“Š Visualization
# Libraries for creating visualizations of data and model results.
# Matplotlib: low-level, customizable plotting.
# Seaborn: statistical data visualization built on top of matplotlib.
# ============================================================
import matplotlib.pyplot as plt  # Core Python plotting library for creating a wide variety of static, animated, and interactive plots ğŸ“ˆ
import seaborn as sns            # Statistical data visualization library with beautiful default styles and functions ğŸ�¨

# ============================================================
# ğŸ”� Machine Learning Utilities
# Scikit-learn utilities for data splitting, cross-validation, scaling, encoding, and pipelines.
# These help prepare data and build robust workflows for training and evaluating models.
# ============================================================
from sklearn.model_selection import (
    train_test_split,      # Split arrays or matrices into random train and test subsets âœ‚ï¸�
    StratifiedKFold,       # K-Folds cross-validator with preserved class distribution âš–ï¸�
    cross_validate,        # Evaluate a score by cross-validation with multiple metrics ğŸ› ï¸�
    GridSearchCV           # Exhaustive search over specified parameter values for an estimator ğŸ”�
)

from sklearn.preprocessing import (
    LabelEncoder,          # Encode categorical labels as integers; useful for target or ordinal variables ğŸ” 
    MinMaxScaler           # Normalize features to a given range (default: [0,1])

)

from sklearn.pipeline import Pipeline  # Utility to chain multiple data preprocessing and modeling steps into a single workflow ğŸ¤�

# ============================================================
# ğŸ¤– Classic Machine Learning Models
# Standard scikit-learn estimators covering linear, tree-based, instance-based, and probabilistic approaches.
# Useful for benchmarking and building interpretable models.
# ============================================================
from sklearn.linear_model import LogisticRegression            # Linear classifier using logistic loss; good baseline model ğŸš¦
from sklearn.tree import DecisionTreeClassifier                # Decision tree for classification tasks ğŸŒ³
from sklearn.neighbors import KNeighborsClassifier             # K-Nearest Neighbors classifier; instance-based learning ğŸ“�
from sklearn.naive_bayes import GaussianNB                     # Naive Bayes classifier for Gaussian-distributed features ğŸ¤“
from sklearn.svm import SVC                                    # Support Vector Classifier for finding maximum-margin hyperplanes ğŸš§

from sklearn.ensemble import (
    RandomForestClassifier,      # Ensemble of decision trees trained on bootstrapped samples with feature randomness ğŸŒ²ğŸŒ²
    ExtraTreesClassifier,        # Ensemble of extremely randomized trees (faster and more variance reduction) ğŸ“°ğŸŒ³
    AdaBoostClassifier,          # Adaptive boosting to combine weak learners into a strong classifier âš¡ğŸ¤–
    GradientBoostingClassifier   # Gradient boosting for building strong learners incrementally ğŸ�†
)

from sklearn.neural_network import MLPClassifier  # Multi-layer Perceptron neural network classifier ğŸ§ 

# ============================================================
# ğŸ§® Advanced Gradient Boosting Libraries
# Popular third-party gradient boosting implementations.
# Known for performance and flexibility, especially on structured/tabular data.
# ============================================================
from xgboost import XGBClassifier            # eXtreme Gradient Boosting; highly optimized and widely used ğŸš€
from lightgbm import LGBMClassifier          # LightGBM; fast and efficient gradient boosting implementation ğŸ’¨
from catboost import CatBoostClassifier      # CatBoost; handles categorical features natively and prevents overfitting ğŸ�±

# ============================================================
# ğŸ“ˆ Evaluation Metrics
# Metrics for assessing the performance of classification models.
# Includes both overall accuracy and more nuanced, class-wise measures.
# Also includes tools for generating ROC and precision-recall curves.
# ============================================================
from sklearn.metrics import (
    accuracy_score,                  # Proportion of correctly classified instances ğŸ�¯
    classification_report,           # Precision, recall, f1-score per class, plus overall averages ğŸ“‹
    roc_auc_score,                   # Area under the Receiver Operating Characteristic curve ğŸ“�
    f1_score,                        # Harmonic mean of precision and recall; balances false positives/negatives âš–ï¸�
    precision_score,                 # Proportion of predicted positives that are correct âœ…
    recall_score,                    # Proportion of actual positives correctly identified ğŸ”�
    confusion_matrix,                # Tabulate predicted vs. actual class counts ğŸ”·
    roc_curve,                       # Compute points for ROC curve to visualize TPR vs. FPR ğŸ“Š
    precision_recall_curve,          # Compute precision-recall trade-off points ğŸ“ˆ
    auc                              # Compute area under a curve (e.g., ROC or PR) ğŸ”·
)



# Set notebook display options
%matplotlib inline
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


# Seed for reproducibility

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(42)


# Login to kaggle
kagglehub.login()


# Download competition dataset
DATA_DIR = kagglehub.competition_download('playground-series-s5e7')

# Load training dataset
train_path = os.path.join(DATA_DIR, 'train.csv')
train_df = pd.read_csv(train_path)

# Verify load
print("Train shape:", train_df.shape)


train_df


# Display first few rows
train_df.head()


# ğŸ“� Generate an automated EDA report using ydata-profiling
# Pass the training dataframe and set a custom title for the report
profile = ProfileReport(train_df, title="ğŸ“Š Training Data Profiling Report")

# ğŸ‘€ Display the profiling report inline
profile


# Visualize distribution of categorical data
def plot_cat_cols_dist(cols):
    for col in cols:
        plt.figure()
        train_df[col].value_counts(dropna=False).plot.bar()
        plt.title(f'Category Distribution: {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.show()


cat_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
plot_cat_cols_dist(cat_cols)


# 2. Encoding

from sklearn.preprocessing import LabelEncoder

# Label encoding with Scikit-learn
le = LabelEncoder()

# Boolean Yes/No â†’ 1/0
bool_cols = ['Stage_fear', 'Drained_after_socializing']
for col in bool_cols:
    train_df[col] = le.fit_transform(train_df[col])

# Introvert/Extrovert â†’ 1/0
train_df["Personality"] = le.fit_transform(train_df["Personality"])

plot_cat_cols_dist(['Personality', 'Stage_fear', 'Drained_after_socializing'])


# ğŸ”¢ Median imputation for numeric columns
numeric_cols = [
    'Time_spent_Alone',
    'Social_event_attendance',
    'Going_outside',
    'Friends_circle_size',
    'Post_frequency'
]
for col in numeric_cols:
    train_df[col] = train_df[col].fillna(train_df[col].mode()[0])


# âœ… Mode imputation for boolean columns and encoding Yes/No â†’ 1/0
bool_cols = ['Stage_fear', 'Drained_after_socializing']
for col in bool_cols:
    train_df[col] = train_df[col].fillna(train_df[col].mode()[0])



train_df.isna().sum()



train_df


# ------------------------------------------------------------
# ğŸ”„ Scale all features in the training dataset to the [0,1] range
# MinMaxScaler rescales each feature individually by subtracting the minimum and dividing by the range (max - min).
# This is often beneficial for algorithms sensitive to feature scales, like KNN, SVM, and neural networks.
# ------------------------------------------------------------

# Instantiate the scaler object
scaler = MinMaxScaler()

# Fit the scaler on the training data and transform it in one step.
# `fit_transform()` computes the min and max per column and applies the scaling.
# The result is converted back into a pandas DataFrame with the original column names preserved.
train_df = pd.DataFrame(
    scaler.fit_transform(train_df),   # Scaled numpy array
    columns=train_df.columns          # Preserve original feature names
)

# Display the scaled training DataFrame
train_df



# ğŸ—‘ï¸� Drop 'id' column (irrelevant for modeling)
train_df = train_df.drop(columns=["id"])

# ğŸª“ Separate features and target
X = train_df.drop('Personality', axis=1)
y = train_df['Personality']

# ğŸ”€ Train-test split (80% train / 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ğŸ“� Print data shapes
print("âœ… Shape of training data:", X_train.shape)
print("âœ… Shape of test data:", X_test.shape)


# CV splitter
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# models + search spaces
search_spaces = {
    "Logistic Regression": (
        LogisticRegression(random_state=42, max_iter=1000),
        [
            # liblinear â†’ l1 or l2, dual only with l2
            {
                "solver":  ["liblinear"],
                "penalty": ["l2"],
                "C":       [0.1, 1, 10],
                "dual":    [True, False],
                "class_weight": [None, "balanced"]
            },
            {
                "solver":  ["liblinear"],
                "penalty": ["l1"],
                "C":       [0.1, 1, 10],
                "dual":    [False],
                "class_weight": [None, "balanced"]
            },

            # saga â†’ elasticnet (needs both C & l1_ratio)
            {
                "solver":    ["saga"],
                "penalty":   ["elasticnet"],
                "C":         [0.1, 1, 10],
                "l1_ratio":  [0.0, 0.5, 1.0],
                "class_weight": [None, "balanced"]

            },
            # saga â†’ pure l1 or l2 (C matters, l1_ratio is ignored)
            {
                "solver":  ["saga"],
                "penalty": ["l1", "l2"],
                "C":       [0.1, 1, 10],
                "class_weight": [None, "balanced"]
            },
            # saga â†’ no penalty (only solver & penalty; C/l1_ratio wonâ€™t be used)
            {
                "solver":  ["saga"],
                "penalty": [None],
                "class_weight": [None, "balanced"]

            },

            # other solvers (lbfgs, newton-cg, sag) â†’ l2 or no penalty, no dual
            {
                "solver":  ["lbfgs", "newton-cg", "sag"],
                "penalty": ["l2"],
                "C":       [0.1, 1, 10],
                "class_weight": [None, "balanced"]

            },
            {
                "solver":  ["lbfgs", "newton-cg", "sag"],
                "penalty": [None],
                "class_weight": [None, "balanced"]
            },
        ]
    ),

    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        {
            "criterion":         ["gini", "entropy"],
            "max_depth":         [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "class_weight": [None, "balanced"],
        }
    ),

    "KNN": (
        KNeighborsClassifier(),
        {
            "n_neighbors": [3, 5, 7],
            "weights":     ["uniform", "distance"],
        }
    ),

    "Gaussian NB": (
        GaussianNB(),
        {
            "var_smoothing": [1e-9, 1e-8, 1e-7],
        }
    ),

    "SVM (RBF)": (
        SVC(kernel="rbf", probability=True, random_state=42),
        {
            "C":     [0.1, 1, 10],
            "gamma": ["scale", "auto"],
            "class_weight": [None, "balanced"]
        }
    ),

    "Random Forest": (
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {
            "n_estimators": [100, 200],
            "max_depth":    [None, 10, 20],
            "max_features": ["sqrt", "log2"],
            "class_weight": [None, "balanced"]
        }
    ),

    "Extra Trees": (
        ExtraTreesClassifier(random_state=42, n_jobs=-1),
        {
            "n_estimators": [100, 200],
            "max_depth":    [None, 10, 20],
            "max_features": ["sqrt", "log2"],
            "class_weight": [None, "balanced"]
        }
    ),

    "AdaBoost": (
        AdaBoostClassifier(random_state=42),
        {
            "n_estimators":   [50, 100],
            "learning_rate":  [0.1, 1.0],
        }
    ),

    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "n_estimators":  [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth":     [3, 5],
        }
    ),

    "XGBoost": (
        XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        {
            "n_estimators":  [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth":     [3, 5],
        }
    ),

    "LightGBM": (
        LGBMClassifier(random_state=42),
        {
            "n_estimators":  [100, 200],
            "learning_rate": [0.05, 0.1],
            "num_leaves":    [31, 63],
        }
    ),

    "CatBoost": (
        CatBoostClassifier(verbose=0, random_state=42),
        {
            "iterations":    [100, 200],
            "depth":         [6, 8],
            "learning_rate": [0.05, 0.1],
        }
    ),

    "MLP": (
        MLPClassifier(random_state=42, max_iter=500),
        {
            "hidden_layer_sizes": [(100,), (100,100)],
            "alpha":              [1e-4, 1e-3],
            "learning_rate_init": [1e-3, 1e-2],
        }
    ),
}

records = []

for name, (estimator, param_grid) in search_spaces.items():
    print(f"ğŸ”· Tuning {name}â€¦")

    # No scaler, just the estimator
    search = GridSearchCV(
        estimator,
        param_grid=param_grid or {},    # empty dict if no hyper-params
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    search.fit(X_train, y_train)

    records.append({
        "model": name,
        "best_params": search.best_params_,
        "best_mean_cv_auc": search.best_score_
    })

df_results = (
    pd.DataFrame(records)
      .sort_values("best_mean_cv_auc", ascending=False)
      .reset_index(drop=True)
)


df_results




# Best row
best_row = df_results.iloc[2]

best_model_name = best_row['model']
best_params = best_row['best_params']

print(f"ğŸ�† Best model: {best_model_name}")
print(f"ğŸ”§ Best params: {best_params}")


# Get base estimator *class* and create fresh instance
base_estimator_class = type(search_spaces[best_model_name][0])
base_estimator_params = search_spaces[best_model_name][0].get_params()

# Create a fresh estimator of the same type, with same defaults
best_classifier = base_estimator_class(**base_estimator_params)

# Set best params
best_classifier.set_params(**best_params)

# Train final model
best_classifier.fit(X_train, y_train)



from sklearn.base import clone
from sklearn.ensemble import VotingClassifier

# 1. Grab the top-4 by best_mean_cv_auc
top4 = (
    df_results
    .sort_values('best_mean_cv_auc', ascending=False)
    .head(4)
    .reset_index(drop=True)
)

# 2. Build a list of (name, estimator) tuples
estimators = []
for _, row in top4.iterrows():
    name   = row['model']
    params = row['best_params']
    # prototype is your original classifier instance
    proto  = search_spaces[name][0]

    # clone to get a fresh estimator
    clf = clone(proto)

    # set the best hyperparameters
    clf.set_params(**params)

    estimators.append((name, clf))

# 3. Create the VotingClassifier
voting_clf = VotingClassifier(
    estimators=estimators,
    voting='soft',  # average predicted probabilities
    weights=top4['best_mean_cv_auc'].tolist()
)

# 4. Fit the ensemble
voting_clf.fit(X_train, y_train)



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve,
    auc
)


def evaluate_classifier(model, X_test, y_test,
                        threshold=0.5,
                        figsize_confusion=(5, 4),
                        figsize_roc=(6, 5),
                        figsize_pr=(6, 5),
                        print_report=True):
    """
    Evaluate a binary classification model with adjustable threshold.

    Parameters
    ----------
    model : estimator
        Any fitted classifier with predict and predict_proba.
    X_test : array-like
        Test features.
    y_test : array-like
        True binary labels for X_test.
    threshold : float
        Decision threshold for class 1.
    ...

    """
    # 1. Get predicted probabilities
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_proba = model.decision_function(X_test)

    # 2. Predict labels based on threshold
    y_pred = (y_proba >= threshold).astype(int)

    # 3. Compute metrics
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_proba)

    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
    pr_auc    = auc(rec_curve, prec_curve)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }

    print(f"âœ… Threshold:      {threshold:.2f}")
    print("âœ… Accuracy:      {:.3f}".format(accuracy))
    print("âœ… Precision:     {:.3f}".format(precision))
    print("âœ… Recall:        {:.3f}".format(recall))
    print("âœ… F1-score:      {:.3f}".format(f1))
    print("âœ… ROC AUC:       {:.3f}".format(roc_auc))
    print("âœ… PR AUC:        {:.3f}".format(pr_auc))

    if print_report:
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=figsize_confusion)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred Introvert', 'Pred Extrovert'],
                yticklabels=['True Introvert', 'True Extrovert'])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=figsize_roc)
    plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0,1], [0,1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

    baseline = np.mean(y_test == 1)
    plt.figure(figsize=figsize_pr)
    plt.plot(rec_curve, prec_curve, lw=2, label=f"PR curve (AUC = {pr_auc:.3f})")
    plt.hlines(baseline, 0, 1, linestyle='--', color='gray',
               label=f"Baseline = {baseline:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precisionâ€“Recall Curve")
    plt.legend(loc="lower left")
    plt.show()




evaluate_classifier(best_classifier, X_test, y_test)


evaluate_classifier(voting_clf, X_test, y_test)


# ğŸ“„ Load the sample submission file

# Build the path to the sample submission CSV
sample_submission_path = os.path.join(DATA_DIR, 'sample_submission.csv')

# ğŸ“¥ Read the sample submission into a DataFrame
sample_submission_df = pd.read_csv(sample_submission_path)

# ğŸ‘€ Display the sample submission template
sample_submission_df


# ğŸ“„ Load the test dataset

# Build the path to the test dataset CSV
test_path = os.path.join(DATA_DIR, 'test.csv')

# ğŸ“¥ Read the test dataset into a DataFrame
test_df = pd.read_csv(test_path)

# ğŸ‘€ Display the test dataset
test_df


# ğŸ§¹ Clean and preprocess the test dataset

test_df_cleaned = test_df.copy()

# ğŸ”¢ Fill numeric columns in test set with medians from train set
for col in numeric_cols:
    median_val = train_df[col].median()
    test_df_cleaned[col] = test_df_cleaned[col].fillna(median_val)

# âœ… Standardize, fill, and encode boolean columns
for col in bool_cols:
    mode_val = train_df[col].mode()[0]

    # ğŸ§½ Clean string values: strip spaces & title-case
    test_df_cleaned[col] = test_df_cleaned[col].astype(str).str.strip().str.title()

    # Replace empty/invalid strings with np.nan
    test_df_cleaned[col] = test_df_cleaned[col].replace(['', 'Nan', 'nan', 'None'], np.nan)

    # Fill missing with mode value from train
    test_df_cleaned[col] = test_df_cleaned[col].fillna(mode_val)

    # Encode Yes/No â†’ 1/0
    test_df_cleaned[col] = test_df_cleaned[col].map({'Yes': 1, 'No': 0})

    # ğŸ”„ If unknown values remain (still NaNs), fill with mode again
    if test_df_cleaned[col].isna().any():
        mode_val_numeric = test_df_cleaned[col].mode()[0]
        test_df_cleaned[col] = test_df_cleaned[col].fillna(mode_val_numeric)

# ğŸ†” Extract test IDs and test features
test_ids = test_df_cleaned['id']
test_features = test_df_cleaned.drop(columns=['id'])

# ğŸ”� Check for any remaining missing values
test_features.isna().sum()


# ğŸ¤– Make predictions on the test features
test_preds = voting_clf.predict(test_features)

# ğŸ”„ Convert numeric predictions to string labels (0 â†’ Extrovert, 1 â†’ Introvert)
personality_labels = ['Extrovert' if p == 0 else 'Introvert' for p in test_preds]


# ğŸ“„ Build the submission DataFrame
# Combine test IDs with predicted personality labels
submission_df = pd.DataFrame({
    'id': test_ids,
    'Personality': personality_labels
})

# ğŸ‘€ Display the submission DataFrame
submission_df


# ğŸ’¾ Save the submission DataFrame as a CSV file (without index column)
submission_df.to_csv("submission.csv", index=False)

# âœ… Confirmation message
print("ğŸ“� Submission file saved as submission.csv")

