# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Import neccessary libraries
import seaborn as sns  # Data visualization library based on matplotlib
import matplotlib.pyplot as plt  # Plotting library for creating visualizations
import warnings  # Module for issuing and managing warning messages
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder  # Preprocessing tools for scaling and encoding data
from sklearn.model_selection import train_test_split  # Function for splitting data into training and testing sets
from sklearn.linear_model import LogisticRegression  # Logistic regression model for classification tasks
from sklearn.ensemble import RandomForestClassifier  # Random Forest classifier from scikit-learn
from xgboost import XGBClassifier  # XGBoost classifier, optimized for gradient boosting
from catboost import CatBoostClassifier  # CatBoost classifier, optimized for categorical features
from sklearn.model_selection import StratifiedKFold  #  cross-validation for model evaluation
from sklearn.svm import SVC  # Support Vector Classifier from scikit-learn
import xgboost as xgb  # Importing full XGBoost library for additional functionalities
import lightgbm as lgb # Import LightGBM
from tabulate import tabulate  # For displaying data in a table format
from sklearn.metrics import precision_recall_curve, roc_auc_score, classification_report, accuracy_score  # Evaluation metrics for model performance
from sklearn.metrics import roc_curve  # Computes the Receiver Operating Characteristic (ROC) curve for evaluating classification performance


# Load the dataset
df_train = pd.read_csv("/kaggle/input/playground-series-s4e10/train.csv")  # Load the training dataset from a CSV file
df_test = pd.read_csv("/kaggle/input/playground-series-s4e10/test.csv")  # Load the test dataset from a CSV file
df_train  # Display the training dataset


# Check the sizre of datasets
print(df_train.shape)
print(df_test.shape)


df_train.info()


df_test.info()


# Visualizing the distribution of the target variable 'loan_status'
ax = sns.countplot(x=df_train["loan_status"])
plt.title("Loan Status Distribution")

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', fontsize=12, color='black')
    
plt.show()


# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Visualizing the probability density of loan status
plt.figure(figsize=(10, 6))
sns.kdeplot(df_train[df_train['loan_status'] == 0]['loan_amnt'], label='Non-Default', fill=True)
sns.kdeplot(df_train[df_train['loan_status'] == 1]['loan_amnt'], label='Default', fill=True)
plt.title('KDE of Loan Amount by Loan Status')
plt.xlabel('Loan Amount')
plt.ylabel('Density')
plt.legend()
plt.show()


numerical_features = [
    "person_age", "person_income", "person_emp_length", 
    "loan_amnt", "loan_int_rate", "loan_percent_income", 
    "cb_person_cred_hist_length"
]
warnings.simplefilter(action='ignore', category=FutureWarning)
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(numerical_features):
    sns.histplot(df_train[col], bins=30, kde=True, ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")

plt.tight_layout()
plt.show()



# Plot distributions of categorical features
categorical_features = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
axes = axes.flatten()

for i, col in enumerate(categorical_features):
    sns.countplot(y=df_train[col], palette="coolwarm", ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")

plt.tight_layout()
plt.show()



# Encode categorical variables
# Label encoding for binary categorical variables 
label_encoder = LabelEncoder()
df_train.loc[:, 'cb_person_default_on_file'] = label_encoder.fit_transform(df_train['cb_person_default_on_file'])
df_test.loc[:, 'cb_person_default_on_file'] = label_encoder.transform(df_test['cb_person_default_on_file'])

df_train['cb_person_default_on_file'] = df_train['cb_person_default_on_file'].astype(int)
df_test['cb_person_default_on_file'] = df_test['cb_person_default_on_file'].astype(int)



# One-hot encoding for other categorical variables
df_train = pd.get_dummies(df_train, columns=['person_home_ownership', 'loan_intent', 'loan_grade'], drop_first=True)
df_test = pd.get_dummies(df_test, columns=['person_home_ownership', 'loan_intent', 'loan_grade'], drop_first=True)

# Feature Scaling 
scaler = StandardScaler()
df_train[numerical_features] = scaler.fit_transform(df_train[numerical_features])
df_test[numerical_features] = scaler.transform(df_test[numerical_features])


df_train.shape,df_test.shape


# Splitting data into features (X) and target (y)
X = df_train.drop(columns=["loan_status", "id"], axis=1)  # Remove target variable
y = df_train["loan_status"]


# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


# Store history
history = {
    'logistic_train': [], 'logistic_test': [],
    'random_forest_train': [], 'random_forest_test': [],
    'catboost_train': [], 'catboost_test': [],
    'xgboost_train': [], 'xgboost_test': [],
    'lightgbm_train': [], 'lightgbm_test': []  # Added LightGBM tracking
}
# Save the number of models
models = []
# Number of epochs
epochs = 20
# Create a list of increasing C values for Logistic Regression
C_values = np.logspace(-2, 2, epochs)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for epoch, C in enumerate(C_values):
    # Logistic Regression
    lr_model = LogisticRegression(C=C, max_iter=2000, penalty='l2', random_state=42)
    lr_model.fit(X_train, y_train)
    models.append(('Logistic Regression', lr_model))
    train_acc_lr = accuracy_score(y_train, lr_model.predict(X_train))
    test_acc_lr = accuracy_score(y_test, lr_model.predict(X_test))
    history['logistic_train'].append(train_acc_lr)
    history['logistic_test'].append(test_acc_lr)
    
    # Random Forest (varying n_estimators as proxy for complexity)
    rf_model = RandomForestClassifier(n_estimators=50 + epoch * 10, 
                                      min_samples_split=5, 
                                      min_samples_leaf=2, 
                                      max_depth=10,
                                      random_state=42)
    rf_model.fit(X_train, y_train)
    models.append(('Random Forest', rf_model))
    train_acc_rf = accuracy_score(y_train, rf_model.predict(X_train))
    test_acc_rf = accuracy_score(y_test, rf_model.predict(X_test))
    history['random_forest_train'].append(train_acc_rf)
    history['random_forest_test'].append(test_acc_rf)
    
    # CatBoost with early stopping
    cb_model = CatBoostClassifier(iterations=200, 
                                  learning_rate=0.02 + (epoch * 0.015), 
                                  depth=6,
                                  random_seed=42, 
                                  verbose=0,
                                  early_stopping_rounds=10)
    cb_model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=0)
    models.append(('CatBoost', cb_model))
    train_acc_cb = accuracy_score(y_train, cb_model.predict(X_train))
    test_acc_cb = accuracy_score(y_test, cb_model.predict(X_test))
    history['catboost_train'].append(train_acc_cb)
    history['catboost_test'].append(test_acc_cb)
    
    # XGBoost with early stopping
    xgb_model = xgb.XGBClassifier(max_depth=3 + (epoch // 4), 
                                  learning_rate=0.1, 
                                  n_estimators=200, 
                                  subsample=0.8,
                                  colsample_bytree=0.8,
                                  random_state=42,
                                  early_stopping_rounds=10, 
                                  verbosity=0)
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], 
                verbose=False)
    models.append(('XGBoost', xgb_model))
    train_acc_xgb = accuracy_score(y_train, xgb_model.predict(X_train))
    test_acc_xgb = accuracy_score(y_test, xgb_model.predict(X_test))
    history['xgboost_train'].append(train_acc_xgb)
    history['xgboost_test'].append(test_acc_xgb)

    # LightGBM with early stopping
    lgbm_model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05 + (epoch * 0.01),
        num_leaves=31 + (epoch * 2),
        max_depth=-1,  # -1 means no limit
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=10,
        verbose=-1  # -1 means silent/quiet
    )
    lgbm_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='logloss',
    )
    models.append(('LightGBM', lgbm_model))
    train_acc_lgbm = accuracy_score(y_train, lgbm_model.predict(X_train))
    test_acc_lgbm = accuracy_score(y_test, lgbm_model.predict(X_test))
    history['lightgbm_train'].append(train_acc_lgbm)
    history['lightgbm_test'].append(test_acc_lgbm)
    
    print(f"Epoch {epoch+1}/{epochs} - "
          f"LR Train: {train_acc_lr:.4f}, Test: {test_acc_lr:.4f} | "
          f"RF Train: {train_acc_rf:.4f}, Test: {test_acc_rf:.4f} | "
          f"CB Train: {train_acc_cb:.4f}, Test: {test_acc_cb:.4f} | "
          f"XGB Train: {train_acc_xgb:.4f}, Test: {test_acc_xgb:.4f} | "
          f"LGBM Train: {train_acc_lgbm:.4f}, Test: {test_acc_lgbm:.4f}")


# Create figure and axes properly
fig, axs = plt.subplots(3, 1, figsize=(14, 18))

# Chart 1: Logistic Regression & Random Forest
axs[0].plot(range(1, epochs + 1), history['logistic_train'], 'o-', label='Logistic Train', color='#1f77b4')
axs[0].plot(range(1, epochs + 1), history['logistic_test'], 'o--', label='Logistic Test', color='#1f77b4')
axs[0].plot(range(1, epochs + 1), history['random_forest_train'], 's-', label='RF Train', color='#2ca02c')
axs[0].plot(range(1, epochs + 1), history['random_forest_test'], 's--', label='RF Test', color='#2ca02c')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Accuracy')
axs[0].set_title('Model Accuracy: Logistic Regression & Random Forest')
axs[0].legend()
axs[0].grid(True)
axs[0].set_ylim(0.75, 1.0)  # Set y-axis limits

# Chart 2: CatBoost & XGBoost
axs[1].plot(range(1, epochs + 1), history['catboost_train'], '^-', label='CatBoost Train', color='#ff7f0e')
axs[1].plot(range(1, epochs + 1), history['catboost_test'], '^--', label='CatBoost Test', color='#ff7f0e')
axs[1].plot(range(1, epochs + 1), history['xgboost_train'], 'd-', label='XGBoost Train', color='#9467bd')
axs[1].plot(range(1, epochs + 1), history['xgboost_test'], 'd--', label='XGBoost Test', color='#9467bd')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].set_title('Model Accuracy: CatBoost & XGBoost')
axs[1].legend()
axs[1].grid(True)
axs[1].set_ylim(0.90, 1.0)  # Set y-axis limits

# Chart 3: LightGBM
axs[2].plot(range(1, epochs + 1), history['lightgbm_train'], '*-', label='LightGBM Train', color='#d62728')
axs[2].plot(range(1, epochs + 1), history['lightgbm_test'], '*--', label='LightGBM Test', color='#d62728')
axs[2].set_xlabel('Epoch')
axs[2].set_ylabel('Accuracy')
axs[2].set_title('Model Accuracy: LightGBM')
axs[2].legend()
axs[2].grid(True)
axs[2].set_ylim(0.90, 1.0)  # Set y-axis limits

# Adjust layout and show the plots
plt.tight_layout()
plt.show()


# Collect the model performance metrics into a list of dictionaries
results = [
    {'Model': 'Logistic Regression', 
     'Accuracy': accuracy_score(y_test, lr_model.predict(X_test)),
     'ROC-AUC': roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1])},
    
    {'Model': 'Random Forest', 
     'Accuracy': accuracy_score(y_test, rf_model.predict(X_test)),
     'ROC-AUC': roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])},
    
    {'Model': 'CatBoost', 
     'Accuracy': accuracy_score(y_test, cb_model.predict(X_test)),
     'ROC-AUC': roc_auc_score(y_test, cb_model.predict_proba(X_test)[:, 1])},
    
    {'Model': 'XGBoost', 
     'Accuracy': accuracy_score(y_test, xgb_model.predict(X_test)),
     'ROC-AUC': roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])},

    {'Model': 'LightGBM', 
     'Accuracy': accuracy_score(y_test,  lgbm_model.predict(X_test)),
     'ROC-AUC': roc_auc_score(y_test,  lgbm_model.predict_proba(X_test)[:, 1])}
]

# Create a DataFrame for better display
results_df = pd.DataFrame(results)

# Display results as a table using tabulate
print(tabulate(results_df, headers='keys', tablefmt='grid'))

# Alternatively, display the DataFrame directly
results_df


# Ensemble Predictions
ensemble_predictions = np.zeros(X_test.shape[0])
ensemble_test_predictions = np.zeros(df_test.shape[0])


# Normalize the predictions by dividing by the number of models
ensemble_predictions /= len(models)
ensemble_test_predictions /= len(models)


# Ensure the test dataset make sense the columns with train dataset.
df_test = df_test[X_train.columns] 

# Apply the model and weight its probability
for name, model in models:
    predictions = model.predict_proba(X_test)[:, 1]  
    ensemble_predictions = predictions 
    
    test_predictions = model.predict_proba(df_test)[:, 1]
    ensemble_test_predictions = test_predictions


# Convert probability to class labels [0,1]
ensemble_predictions_labels = (ensemble_predictions > 0.5).astype(int)
ensemble_test_labels = (ensemble_test_predictions > 0.5).astype(int)


# Evaluate the ensemble performance
accuracy = accuracy_score(y_test, ensemble_predictions_labels)
roc_auc = roc_auc_score(y_test, ensemble_predictions)
print(f"Ensemble Accuracy: {accuracy}")
print(f"ROC AUC Score: {roc_auc}")
print(classification_report(y_test, ensemble_predictions_labels))


# Visualizing ROC Curve for the Ensemble Model
fpr, tpr, _ = roc_curve(y_test, ensemble_predictions)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Ensemble Model (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Ensemble Model')
plt.legend(loc="lower right")
plt.show()


# Prepare the submission file
submission = pd.DataFrame({'id': df_test.index, 'loan_status': ensemble_test_labels})
submission.to_csv('submission.csv', index=False)

print("Submission file saved!")


print(submission)


# Load the test dataset to check IDs
df_test = pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')
print(df_test[['id']].head())


# Prepare the submission file with predicted probabilities and correct IDs
submission = pd.DataFrame({'id': df_test['id'], 'loan_status': ensemble_test_predictions})
submission.to_csv('/kaggle/working/submission.csv', index=False)


# Check the final result
submission.head

