import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score



file_path = "/kaggle/input/ml-with-python-course-project/train.csv"

try:
    df = pd.read_csv(file_path)
    print("File loaded from: ", file_path)
except FileNotFoundError:
    print("File is not found under: ", file_path)
except Exception as e:
    print("An expected error occurred: ", e)


print(df.head())


print(df.info())


print(df.describe())


print(df.isnull().sum())


print(df[df['salary'].isnull()]['status'].value_counts())
print(df[df['salary'] == 0]['status'].value_counts())


duplicate_rows = df[df.duplicated()]
print(duplicate_rows)


df = df.drop('sl_no', axis=1)
print(df.columns)


target_variable = 'status'
print(f"Unique values: {df[target_variable].unique()}")


numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_cols:
    unique_count = df[col].nunique()
    print(f"Count of unique values in '{col}': {unique_count}")
    print(f"Unique values: {df[col].unique()}")
    print()


categorical_cols = df.select_dtypes(include=['object', 'category']).columns
categorical_cols = [col for col in categorical_cols if col != target_variable] 

for col in categorical_cols:
    unique_count = df[col].nunique()
    print(f"Count of unique values in '{col}': {unique_count}")
    print(f"Unique values: {df[col].unique()}")
    print()


binary_numeric_cols = [col for col in numerical_cols if df[col].nunique() == 2]
numerical_cols = [col for col in numerical_cols if col not in binary_numeric_cols]

print("Binary Columns:")
print(binary_numeric_cols)
print()
print("Numerical Columns:")
print(numerical_cols)


binary_categorical_cols = [col for col in categorical_cols if df[col].nunique() == 2]
multi_categorical_cols = [col for col in categorical_cols if col not in binary_categorical_cols]

print("Binary Categorical Columns:")
print(binary_categorical_cols)
print()
print("Multi-Class Categorical Columns:")
print(multi_categorical_cols)


categorical_cols = ['gender', 'ssc_b', 'hsc_b', 'workex', 'specialisation', 'hsc_s', 'degree_t']


df[numerical_cols].hist(bins=30, figsize=(15,12))
plt.suptitle("Distributions of Numerical Columns (before transformation)", fontsize=18)
plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 12))
for i, col in enumerate(categorical_cols, start=1):
    plt.subplot(3, 3, i)
    sns.countplot(data=df, x=col)
    plt.title(f"Distribution of {col}")

plt.suptitle("Distributions of Categorical Features", fontsize=18)
plt.tight_layout()
plt.show()


# Set up subplot grid
plt.figure(figsize=(15, 12))

for i, col in enumerate(categorical_cols, start=1):
    # Compute % distribution
    cross_tab = pd.crosstab(df[col], df['status'], normalize='index') * 100
    cross_tab.plot(kind='bar', stacked=True, ax=plt.subplot(3, 3, i))
    
    plt.title(f"Placement % by {col}")
    plt.ylabel("Percentage")
    plt.xlabel(col)
    plt.ylim(0, 100)
    plt.xticks(rotation=0)

plt.suptitle("Percentage of Placement by Categorical Feature", fontsize=18)
plt.tight_layout()
plt.show()


sns.countplot(x="status", data=df)
plt.title("Class Distribution of Target Variable")
plt.xlabel("Status")
plt.ylabel("Count")
plt.show()

print(df["status"].value_counts(normalize=True).round(2))


df_cleaned = df.copy()


df_cleaned[target_variable] = df_cleaned[target_variable].map({'Placed': 1, 'Not Placed': 0})
print(df_cleaned[target_variable].unique())


print(binary_categorical_cols)


df_cleaned['ssc_b'] = df_cleaned['ssc_b'].map({'Central': 1, 'Others': 0})
print(df_cleaned['ssc_b'].unique())

df_cleaned['hsc_b'] = df_cleaned['hsc_b'].map({'Central': 1, 'Others': 0})
print(df_cleaned['hsc_b'].unique())

df_cleaned['workex'] = df_cleaned['workex'].map({'Yes': 1, 'No': 0})
print(df_cleaned['workex'].unique())

df_cleaned['specialisation'] = df_cleaned['specialisation'].map({'Mkt&Fin': 1, 'Mkt&HR': 0})
print(df_cleaned['specialisation'].unique())


# One-hot encode mutli cat columns
multi_categorical_cols = ['hsc_s', 'degree_t']
df_temp = pd.get_dummies(df_cleaned, columns=multi_categorical_cols, drop_first=True) # dropped to prevent multicollinearity


# fill missing values of salary
df_temp['salary'] = df_temp['salary'].fillna(0)


df_temp.head()


df_temp = df_temp[[col for col in df_temp.columns if col != 'status'] + ['status']]
df_temp.head()


corr_matrix = df_temp.corr().abs() # Calculate absolute values 

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()


numerical_cols = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']

plt.figure(figsize=(15, 12))
for i, col in enumerate(numerical_cols, start=1):
    plt.subplot(3, 3, i)
    sns.boxplot(data=df_temp, x=col)
    plt.title(f"Boxplot of {col}")

plt.tight_layout()
plt.show()


def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"{column}: {len(outliers)} outlier(s)")
    print(f"Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}\n")
    return outliers, lower_bound, upper_bound


for col in ['hsc_p', 'degree_p']:
    detect_outliers_iqr(df_temp, col) # use temp df


df_cleaned.head()


df_cleaned.drop(columns='salary', inplace=True)


df_cleaned['hsc_p'] = df_cleaned['hsc_p'].clip(lower=42.75, upper=91.15)
df_cleaned['degree_p'] = df_cleaned['degree_p'].clip(upper=88.5)

print("Verified after capping")
for col in ['hsc_p', 'degree_p']:
    detect_outliers_iqr(df_cleaned, col)


X = df_cleaned.drop(columns='status')
y = df_cleaned['status']

RANDOM_STATE=42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
print("Train data shape")
print(X_train.shape, y_train.shape)

print("Test data shape")
print(X_test.shape, y_test.shape)


X_train.info()


X_train.head()


print(y_train.value_counts())


counter = Counter(y_train)
negative = counter[0]  # 'Not Placed'
positive = counter[1]  # 'Placed'
scale_pos_weight = negative / positive

print(f"scale_pos_weight: {scale_pos_weight:.2f}")


SCALE_POS_WEIGHT = 0.46


binary_categorical_cols = ['gender', 'ssc_b', 'hsc_b', 'workex', 'specialisation']
numerical_cols = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']
multi_class_cols = ['hsc_s', 'degree_t']


# Numerical: scale
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Multi-class categoricals: one-hot encode
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))  # drop='first' to avoid multicollinearity
])

# Column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, multi_class_cols)
    ],
    remainder='passthrough'  # keep binary columns as they are
)


X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)


models = {
    'Logistic Regression': LogisticRegression(C=1.0, solver='liblinear', max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_split=4, class_weight='balanced', random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=RANDOM_STATE),
    'SVM': SVC(C=1.0, kernel='rbf', probability=True, class_weight='balanced'),  # probability=True needed for ROC, Voting
    'KNN': KNeighborsClassifier(n_neighbors=5, weights='uniform'),
    'XGBoost': XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        scale_pos_weight = SCALE_POS_WEIGHT,
        random_state=RANDOM_STATE
    )
}


trained_models = {}

for name, model in models.items():
    model.fit(X_train_transformed, y_train)
    trained_models[name] = model


results = []

for name, model in trained_models.items():
    y_pred = model.predict(X_test_transformed)
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1Score': f1_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results).sort_values(by='F1Score', ascending=False)
results_df = results_df.round(2)

display(results_df)


n_models = len(trained_models)
cols = 3
rows = math.ceil(n_models / cols)

plt.figure(figsize=(cols * 5, rows * 4))

for i, (name, model) in enumerate(trained_models.items(), start=1):
    y_pred = model.predict(X_test_transformed)
    cm = confusion_matrix(y_test, y_pred)

    plt.subplot(rows, cols, i)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"{name}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

plt.tight_layout()
plt.suptitle("Confusion Matrices for All Models", fontsize=18, y=1.05)
plt.show()


for name, model in trained_models.items():
    y_pred = model.predict(X_test_transformed)
    print(f"Model: {name}")
    print(classification_report(y_test, y_pred))
    print()


n_models = len(trained_models)
cols = 3
rows = math.ceil(n_models / cols)

plt.figure(figsize=(cols * 5, rows * 4))

for i, (name, model) in enumerate(trained_models.items(), start=1):
    plt.subplot(rows, cols, i)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_transformed)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test_transformed)
    else:
        plt.title(f"{name}\n(no ROC)")
        continue

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {name}")
    plt.legend(loc="lower right")

plt.tight_layout()
plt.suptitle("ROC Curves for All Models", fontsize=16, y=1.05)
plt.show()


plt.figure(figsize=(15, 10))

for i, (name, model) in enumerate(trained_models.items(), start=1):
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test_transformed)[:, 1]
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test_transformed)
    else:
        continue

    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    avg_precision = average_precision_score(y_test, y_scores)

    plt.subplot(2, 3, i)
    plt.plot(recall, precision, label=f"AP = {avg_precision:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve: {name}")
    plt.legend()

plt.tight_layout()
plt.suptitle("Precision-Recall Curves for All Models", fontsize=16, y=1.05)
plt.show()


# Define parameter grid
rf_param_grid = {
    'n_estimators': [100, 200], # Number of trees in the forest
    'max_depth': [None, 5, 6, 7], # Max depth of each tree (None = full growth)
    'min_samples_split': [2, 5, 10], # Min samples to split a node
    'min_samples_leaf': [1, 2, 4], # Min samples required at a leaf
    'max_features': ['sqrt', 'log2'], # Features to consider at each split
    'class_weight': [None, 'balanced'], # To compare with balanced and none
    'bootstrap': [True, False] # Whether bootstrap samples are used
}

# Use StratifiedKFold
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Random Forest GridSearchCV
rf_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=RANDOM_STATE),
    param_grid=rf_param_grid,
    cv=stratified_kfold,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

# Fit the model
rf_search.fit(X_train_transformed, y_train)

# Print best hyperparameters found
print("Best RF params", rf_search.best_params_)


# Define parameter grid
lr_param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],  # Regularization strength (lower = more regularization)
    'penalty': ['l2'],                          # L2 regularization
    'solver': ['liblinear', 'lbfgs'],           # Optimization algorithm
    'max_iter': [200, 300, 500],                # Max training iterations
    'tol': [1e-3, 1e-4],                        # Convergence threshold
    'class_weight': [None, 'balanced'],         # Handle class imbalance
    'fit_intercept': [True, False]              # Include bias term or not
}

# Use StratifiedKFold
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Logistic Regression GridSearchCV
lr_search = GridSearchCV(
    estimator=LogisticRegression(random_state=RANDOM_STATE),
    param_grid=lr_param_grid,
    cv=stratified_kfold,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

# Fit the model
lr_search.fit(X_train_transformed, y_train)

# Print best hyperparameters found
print("Best LR params:", lr_search.best_params_)


# Define parameter grid
xgb_param_grid = {
    'n_estimators': [50, 100, 200],         # Try higher capacity
    'max_depth': [3, 5, 7],                 # Common safe range
    'learning_rate': [0.01, 0.05, 0.1],     # Safer range to avoid overfitting
    'subsample': [0.6, 0.8, 1.0],           # Controls row sampling (regularization)
    'colsample_bytree': [0.6, 0.8, 1.0],    # Controls feature sampling (often underused)
    'gamma': [0, 1, 5],                     # Controls minimum loss reduction for a split
}

# XGBClassifier GridSearchCV
xgb_search = GridSearchCV(
    estimator=XGBClassifier(eval_metric='logloss', scale_pos_weight=SCALE_POS_WEIGHT, random_state=RANDOM_STATE),
    param_grid=xgb_param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

# Fit the model
xgb_search.fit(X_train_transformed, y_train)

# Print best hyperparameters found
print("Best XGB params:", xgb_search.best_params_)


# Define parameter grid
dt_param_grid = {
    'criterion': ['gini'],                      # Split quality metric
    'max_depth': [None, 15, 20],                # Max tree depth
    'min_samples_split': [2, 5],                # Min samples to split an internal node
    'min_samples_leaf': [1],                    # Keep leaves small to avoid underfitting
    'max_features': [None, 'sqrt', 'log2'],     # Number of features to consider at each split
    'splitter': ['best'],                       # Strategy to choose the split
    'class_weight': [None, 'balanced']          # Adjusts for class imbalance
}

# Use StratifiedKFold
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Decision Tree GridSearchCV
dt_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
    param_grid=dt_param_grid,
    cv=stratified_kfold,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

# Fit the model
dt_search.fit(X_train_transformed, y_train)

# Print best hyperparameters found
print("Best Decision Tree params:", dt_search.best_params_)


tuned_rf = rf_search.best_estimator_
tuned_lr = lr_search.best_estimator_
tuned_xgb = xgb_search.best_estimator_
tuned_dt = dt_search.best_estimator_


tuned_models = {
    "Tuned RF": tuned_rf,
    "Tuned LR": tuned_lr,
    "Tuned XGB": tuned_xgb,
    "Tuned DT": tuned_dt
}


print("Before Tuning")
print(results_df[0:4])


# Show tuned results
tuned_results = []

for name, model in tuned_models.items():
    y_pred = model.predict(X_test_transformed)
    tuned_results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    })

tuned_results_df = pd.DataFrame(tuned_results).sort_values(by='F1 Score', ascending=False)
tuned_results_df = tuned_results_df.round(2)

print("After Tuning")
print(tuned_results_df)


n_models = len(tuned_models)
cols = 2
rows = math.ceil(n_models / cols)

plt.figure(figsize=(cols * 5, rows * 4))

for i, (name, model) in enumerate(tuned_models.items(), start=1):
    y_pred = model.predict(X_test_transformed)
    cm = confusion_matrix(y_test, y_pred)

    plt.subplot(rows, cols, i)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"{name}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

plt.tight_layout()
plt.suptitle("Confusion Matrices for Tuned Models", fontsize=16, y=1.05)
plt.show()


for name, model in tuned_models.items():
    y_pred = model.predict(X_test_transformed)
    print(f"Model: {name}")
    print(classification_report(y_test, y_pred))


voting_clf_soft = VotingClassifier(
    estimators=[
        ('rf', rf_search.best_estimator_),
        ('xgb', xgb_search.best_estimator_),
        ('lr', lr_search.best_estimator_),
        ('dt', dt_search.best_estimator_)
    ],
    voting='soft'
)

voting_clf_hard = VotingClassifier(
    estimators=[
        ('rf', tuned_rf),
        ('xgb', tuned_xgb),
        ('lr', tuned_lr),
        ('dt', tuned_dt)
    ],
    voting='hard'
)


voting_clf_soft.fit(X_train_transformed, y_train)
voting_clf_hard.fit(X_train_transformed, y_train)


y_pred_voting_soft = voting_clf_soft.predict(X_test_transformed)

print("Voting Classifier (soft) Performance:")
print(classification_report(y_test, y_pred_voting_soft))


y_pred_voting_hard = voting_clf_hard.predict(X_test_transformed)

print("Voting Classifier (soft) Performance:")
print(classification_report(y_test, y_pred_voting_hard))


final_models = {
    'Voting (Soft)': voting_clf_soft,
    'Voting (Hard)': voting_clf_hard,
    'Tuned RF': rf_search.best_estimator_,
    'Tuned XGB': xgb_search.best_estimator_,
    'Tuned LR': lr_search.best_estimator_,
    'Tuned DT': dt_search.best_estimator_
}


results = []

for name, model in final_models.items():
    y_pred = model.predict(X_test_transformed)
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1Score': f1_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results).sort_values(by='F1Score', ascending=False)
results_df = results_df.round(2)

display(results_df)


for name, model in final_models.items():
    y_pred = model.predict(X_test_transformed)
    print(f"Model: {name}")
    print(classification_report(y_test, y_pred))


n_models = len(final_models)
cols = 3
rows = math.ceil(n_models / cols)

plt.figure(figsize=(cols * 5, rows * 4))

for i, (name, model) in enumerate(final_models.items(), start=1):
    y_pred = model.predict(X_test_transformed)
    cm = confusion_matrix(y_test, y_pred)

    plt.subplot(rows, cols, i)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"{name}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

plt.tight_layout()
plt.suptitle("Confusion Matrices for Final Models", fontsize=16, y=1.05)
plt.show()

