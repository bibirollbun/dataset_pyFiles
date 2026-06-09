# Required libraries for multi-class classification and hyperparameter tuning

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn libraries:
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,  roc_curve


# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier

# For handling warnings
import warnings
warnings.filterwarnings("ignore")

print("Libraries have been loaded successfully")

# # 8. Display all rows and columns: (uncomment if you want the whole output in the cells, I don't prefer it while uploading my notebook on Kaggle or Github)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


# Setting a style for the plots
sns.set_theme(style="whitegrid")


# ğŸš€ Step 1: Loading the Data
print("Loading the dataset... ğŸ•µï¸�â€�â™‚ï¸�")
df_train = pd.read_csv("/kaggle/input/bank-churn-pre-processed-dataset/train_preprocessed.csv")
df_test = pd.read_csv("/kaggle/input/bank-churn-pre-processed-dataset/test_preprocessed.csv")
submission = pd.read_csv("/kaggle/input/bank-churn-pre-processed-dataset/sample_submission.csv")
print("Dataset loaded successfully!")


print("The preprocessed training dataset:")
display(df_train.head())

print("The preprocessed testing dataset:")
display(df_test.head())

print("The submission dataset:")
display(submission.head())



# check the shape of each:

print(f"The shape of the training dataset: {df_train.shape}")

print(f"The shape of the testing dataset: {df_test.shape}")

print(f"The shape of the submission dataset: {submission.shape}")


# check the columns of each and print them in a presentable form:

print("The columns of the training dataset:")

for i, col in enumerate(df_train.columns):
    print(f"{i+1}. {col}")

print("\n\nThe columns of the testing dataset:")
for i, col in enumerate(df_test.columns):
    print(f"{i+1}. {col}")

print("\n\nThe columns of the submission dataset:")
for i, col in enumerate(submission.columns):
    print(f"{i+1}. {col}")



# find the missing values in each dataset:

print("The missing values in the training dataset:")
display(df_train.isnull().sum())

print("---------------------------------------------------------------")

print("The missing values in the testing dataset:")
display(df_test.isnull().sum())




df_train.columns


# remove the following columns as we have already transformed and encoded them.
# Also, we will remove the target column from the training dataset.
# the columns to be removed are: 'id, CustomerId', Surname, 'CreditScore', 'Geography', 'Age', 'Balance', 'EstimatedSalary', 'Exited'


# drop the columns from the training dataset:

df_train.drop(['id', 'CustomerId', 'Surname', 'CreditScore', 'Age', 'Balance', 'EstimatedSalary'], axis=1, inplace=True)



# Scale the features using standard scaler:

# Scaling :
columns_to_scale = df_train.iloc[:, :-1].columns.drop('Exited')
min_max_scalers = {}

for col in columns_to_scale:
    # Create a new standard Scalar for the column
    scaler = StandardScaler()

    # Fit and transform the data
    df_train[col] = scaler.fit_transform(df_train[[col]])

    # Store the scaler in the dictionary
    min_max_scalers[col] = scaler

print("The training dataset after scaling:")
display(df_train.head())


df = df_train.copy()



# split the data into training and testing sets

X = df.drop('Exited', axis=1)
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"The shape of the training set: {X_train.shape} and the shape of the testing set: {X_test.shape}")



# Train RandomForestClassifier and extract feature importance

from sklearn.ensemble import RandomForestClassifier

# Train the RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Extract feature importances
feature_importances = rf_classifier.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.show()

# Step 9: Analyze the top features
top_features = importance_df.head(10)
print(top_features)


# #Install scikit learn using this cell if the below code don't work at your end.

# !pip install scikit-learn==1.1.3



%%time
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

# Define hyperparameter grids
params_xgb = {
    'n_estimators': [100, 300],  # Reduced upper limit
    'learning_rate': [0.01, 0.1],
    'max_depth': [3],  # Keeping only lower depth for efficiency
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'scale_pos_weight': [1]  # Removing 2 to reduce search space
}



params_lgb = {
    'n_estimators': [100, 300],
    'learning_rate': [0.01, 0.1],
    'max_depth': [5],  # Keeping only 5 to reduce complexity
    'num_leaves': [31],
    'min_child_samples': [50],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'scale_pos_weight': [1]
}

params_rf = {
    'n_estimators': [100, 300],
    'max_depth': [5, None],  # Removing 10 for efficiency
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}


# params_lr = {
#     'C': [0.1, 1, 10],
#     'solver': ['liblinear']
# }

# List of models
models = {
    "XGBoost": (XGBClassifier(random_state=42), params_xgb),
    "LightGBM": (LGBMClassifier(random_state=42, verbose=-1), params_lgb),
    "RandomForest": (RandomForestClassifier(random_state=42), params_rf),
    # "LogisticRegression": (LogisticRegression(random_state=42), params_lr)
}

# Train & evaluate models
for model_name, (model, params) in models.items():
    print(f"\nğŸ”� Training {model_name}...\n")

    try:
        grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Best Model Evaluation
        y_pred = grid_search.best_estimator_.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        # Print Results
        print(f"âœ… Model: {model_name}")
        print(f"Cross-validation Accuracy: {grid_search.best_score_:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision Score: {precision:.4f}")
        print("\n" + "_" * 80 + "\n")

        # Confusion Matrix
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.show()

    except Exception as e:
        print(f"â�Œ Error for model {model_name}: {e}")


# Dictionary to store model results
best_models = {}

# Train & evaluate models
for model_name, (model, params) in models.items():
    print(f"\nğŸ”� Training {model_name}...\n")

    try:
        grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Store best model & parameters
        best_models[model_name] = {
            "Best Params": grid_search.best_params_,
            "Best CV Accuracy": grid_search.best_score_,
            "Test Accuracy": accuracy_score(y_test, grid_search.best_estimator_.predict(X_test))
        }

    except Exception as e:
        print(f"â�Œ Error for model {model_name}: {e}")

# Print best model and parameters
best_model_name = max(best_models, key=lambda x: best_models[x]["Test Accuracy"])
best_model_info = best_models[best_model_name]

print("\nğŸ�¯ Best Model Summary ğŸ�¯\n")
print(f"ğŸ�† Best Model: {best_model_name}")
print(f"âœ… Best Cross-Validation Accuracy: {best_model_info['Best CV Accuracy']:.4f}")
print(f"ğŸ“Š Test Accuracy: {best_model_info['Test Accuracy']:.4f}")
print("âš™ï¸� Best Hyperparameters:")
for param, value in best_model_info["Best Params"].items():
    print(f"   ğŸ”¹ {param}: {value}")


