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


#import libraries:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import shap

# from ydata_profiling import ProfileReport 


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold, cross_val_score


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
import optuna

import warnings
warnings.simplefilter("ignore")
pd.options.mode.chained_assignment = None  

pd.set_option('display.max_columns', None)


train  = pd.read_csv(f"/kaggle/input/stock-pledge-defaults-prediction/train.csv")
test   = pd.read_csv(f"/kaggle/input/stock-pledge-defaults-prediction/test.csv")

train.head()


train.columns


print(train.shape)
print(test.shape)


train.dtypes


# Data Type

# Select only categorical columns
categorical_columns = train.select_dtypes(include=['object', 'category']).columns

# Print categorical columns
print(categorical_columns)



# Stock Code is a unique identifier - it's meaningless in terms of prediction- however, for better identification we can make it index


train.set_index('Stock code', inplace=True)
train.head()


# Doing the Same for test set:

test.set_index('Stock code', inplace=True)
test.head()


#how is P/E ratio related to the target

train['P/E ratio'].value_counts()


# Frequency Encoding:

# Compute frequency encoding
freq_encoding = train['P/E ratio'].value_counts(normalize=True)

# Map frequencies to the column
train['P/E ratio'] = train['P/E ratio'].map(freq_encoding)
train.head()


# Compute frequency encoding
freq_encoding = test['P/E ratio'].value_counts(normalize=True)

# Map frequencies to the column
test['P/E ratio'] = test['P/E ratio'].map(freq_encoding)
test.head()


def show_columns_with_missing_values(df):
    """
    Displays columns in the dataset that contain missing values.
    
    Parameters:
        df (pd.DataFrame): The dataframe to check for missing values.
    
    Returns:
        None (prints columns with missing values and their counts)
    """
    missing_cols = df.columns[df.isnull().any()]
    if missing_cols.empty:
        print("No missing values in the dataset.")
    else:
        print("Columns with missing values:")
        print(df[missing_cols].isnull().sum())
    return


show_columns_with_missing_values(train)


show_columns_with_missing_values(test)


train['IsDefault'] = train['IsDefault'].astype(int)
train['IsDefault'].value_counts()


tar =train['IsDefault'].value_counts(normalize = True).reset_index()
#tar



sns.barplot(x=tar['IsDefault'], y=tar['proportion']*100, data=tar)
plt.xlabel('IsDefault')
plt.ylabel('proportion(%)')
plt.title('Class Imbalance in Stock Train Data')
plt.show()


col_miss = ['Tobin Q', 'Debt financing costs', 'Enterprise age', 'Goodwill impairment ratio','Asset quality index',           
'SG&A Expense','Number of key audit matters']


def plot_histograms_with_kde(df):
    """
    Plots histograms with KDE for each column in col_miss, overlaying distributions for target == 0 and target == 1.
    
    Parameters:
    df (pd.DataFrame): Dataframe containing the data.
    col_miss (list): List of column names to plot.
    target_col (str): Name of the target column (default is 'target').
    """
    target_col = 'IsDefault'
    for col in col_miss:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[df[target_col] == 0][col], kde=True, color='blue', label='Target = 0', bins=30, stat='density')
        sns.histplot(df[df[target_col] == 1][col], kde=True, color='red', label='Target = 1', bins=30, stat='density')
        
        plt.title(f'Distribution of {col} by {target_col}')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.legend()
        plt.show()
        plt.close()


plot_histograms_with_kde(train)


for col in col_miss:
    plt.figure(figsize=(8, 5))
    sns.histplot(test[col], kde=True, color='blue',  bins=30, stat='density')
    
   
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    plt.close()
    


#Delete

# existing_cols = [col for col in col_miss if col in train.columns]
# train_= train.drop(columns=existing_cols, errors='ignore')
# print(train_.shape)


# existing_cols = [col for col in col_miss if col in test.columns]
# test_= test.drop(columns=existing_cols, errors='ignore')
# print(test_.shape)


# Missing Value imputation:


col_miss = ['Tobin Q', 'Debt financing costs', 'Enterprise age', 'Goodwill impairment ratio','Asset quality index',           
'SG&A Expense','Number of key audit matters']


test[col_miss] = test[col_miss].fillna(test[col_miss].median())
test.shape


train.shape


#Test-Train:

y = train['IsDefault']
X = train.drop(columns= 'IsDefault')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train.shape


X_test.shape


smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# Initialize XGBoost Model
# Adding scale_pos_weight to handle class imbalance
ratio = np.sum(y_train == 0) / np.sum(y_train == 1)
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Train the model on the training data
model.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred = model.predict(X_test)


# Evaluate the model

f1 = f1_score(y_test, y_pred, average='binary')

# Output the performance metrics
print(f'F1 Score: {f1:.4f}')


# Feature Importance


#Feature Importance - Get values and sort them
feature_importance = model.feature_importances_
sorted_indices = np.argsort(feature_importance)[::-1]  # Sort indices in descending order

# Sort feature names and importance values accordingly
sorted_features = np.array(X_train_resampled.columns)[sorted_indices]
sorted_importance = feature_importance[sorted_indices]

# Plot Feature Importance (Sorted)
plt.figure(figsize=(30, 20))
plt.barh(sorted_features, sorted_importance)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("XGBoost Feature Importance")
plt.gca().invert_yaxis()  # Invert y-axis for highest importance on top
plt.show()


# Initialize the SHAP explainer

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test) 

#  Visualize the SHAP summary plot
shap.summary_plot(shap_values, X_test, plot_type="bar")


# Initialize KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)


# Initialize a DataFrame to accumulate SHAP importances
shap_importances = pd.DataFrame(0, index=X_train_resampled.columns, columns=['importance'])

# Loop through each fold
for train_index, test_index in kf.split(X):
    X_train_resampled, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train_resampled, y_test = y[train_index], y[test_index]
    
    
    
    # Compute SHAP values using TreeExplainer (efficient for tree-based models)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test) 
    
    # Compute mean absolute SHAP value for each feature in this fold
    #fold_importance = np.abs(shap_values).mean(axis=0)
    fold_importance = np.abs(shap_values.values).mean(axis=0)
    
    # Aggregate importances
    shap_importances['importance'] += fold_importance

# Average the SHAP values across folds
shap_importances['importance'] /= kf.get_n_splits()

# Sort features by their average importance
top_features = shap_importances.sort_values(by='importance', ascending=False)
print("Top features based on average absolute SHAP values across folds:")
print(top_features)


# Keep only those features which have non zero SHAP values:

# # Select top features
top_40_features = top_features.head(40)
top_40_features


# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # Now, evaluate different subsets of features using cross-validation
# n_features_list = [5, 10, 20, 30, 40, 50, 60]  # adjust based on your total features
# cv_scores = []

# for n_features in n_features_list:
#     selected_feats = top_features.index[:n_features]
#     model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=ratio)
#     scores = cross_val_score(model, X_train[selected_feats], y_train, scoring='f1', cv=skf)
#     cv_scores.append(np.mean(scores))


    



# # Plotting the performance vs. number of top features
# plt.figure(figsize=(8, 5))
# plt.plot(n_features_list, cv_scores, marker='o')
# plt.xlabel('Number of Features')
# plt.ylabel('Mean Cross-Validation Score')
# plt.title('Model Performance vs. Number of Features')
# plt.grid(True)
# plt.show()


# # Select top features
# top_5_features = top_features.head(5)
# top_5_features


# Selecting Features based on SHAP values

# Extract SHAP values from shap.Explanation object
# shap_values_array = np.abs(shap_values.values).mean(axis=0)

# # Create a DataFrame for feature importance
# feature_importance_df = pd.DataFrame({
#     'Feature': X_test.columns,
#     'Mean_ABS_SHAP': shap_values_array
# })

# # Sort by importance in descending order
# feature_importance_df = feature_importance_df.sort_values(by='Mean_ABS_SHAP', ascending=False)

# # Select top features
# top_30_features = feature_importance_df.head(30)


# top_30_features



selected_features = top_40_features.index.tolist()
selected_features


X_train_selected = X_train_resampled[selected_features]
X_test_selected = X_test[selected_features]

print(X_train_selected.shape)
print(X_test_selected.shape)


### Plot Heat Map




# # Compute correlation matrix
# corr_matrix = X_train.corr()

# # Plot heatmap
# plt.figure(figsize=(12, 8))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Feature Correlation Matrix")
# plt.show()


# # Get highly correlated features
# threshold = 0.70
# high_corr_features = set()

# for i in range(len(corr_matrix.columns)):
#     for j in range(i):
#         if abs(corr_matrix.iloc[i, j]) > threshold:
#             high_corr_features.add(corr_matrix.columns[i])



# # Drop correlated features from dataset
# X_train_subset = X_train.drop(columns=high_corr_features)
# X_test_subset = X_test.drop(columns=high_corr_features)



# print(X_train_subset.shape)
# print(X_train_subset.columns)

# print(X_test_subset.shape)
# print(X_test_subset.columns)


#Final Top features:

feat = X_train_selected.columns
feat


# Adding scale_pos_weight to handle class imbalance
ratio = np.sum(y_train == 0) / np.sum(y_train == 1)
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Train the model on the training data
model.fit(X_train_selected, y_train_resampled)

# Make predictions
y_pred = model.predict(X_test_selected)



# Evaluate the model

f1 = f1_score(y_test, y_pred, average='binary')

# Output the performance metrics

print(f'F1 Score: {f1:.4f}')



# # Define the hyperparameter grid
# param_grid = {
#     'scale_pos_weight': [1, 2, 3],
#     'max_depth': [3, 5, 7],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'n_estimators': [100, 200, 300]
# }

# # Set up StratifiedKFold to preserve the class distribution in each fold
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # Define F1 score as the evaluation metric
# f1_scorer = make_scorer(f1_score, pos_label=1)

# # Set up GridSearchCV with the defined parameters and scorer
# grid_search = GridSearchCV(
#     estimator=model,
#     param_grid=param_grid,
#     scoring=f1_scorer,
#     cv=skf,
#     n_jobs=-1
# )

# # Fit GridSearchCV on your training data
# grid_search.fit(X_train_selected, y_train)

# # Output the best parameters and best F1 score found during tuning
# print("Best parameters:", grid_search.best_params_)
# print("Best F1 score:", grid_search.best_score_)



# # Get the best estimator from GridSearchCV
# best_model = grid_search.best_estimator_

# # Use the best model to make predictions on your test set
# y_pred = best_model.predict(X_test_selected)

# # Evaluate the model

# f1 = f1_score(y_test, y_pred, average='binary')

# # Output the performance metrics

# print(f'F1 Score: {f1:.4f}')



def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "use_label_encoder": False,  # Avoid warnings for XGBClassifier
        "eval_metric": "logloss"       # Required to prevent warnings
    }
    
    model = xgb.XGBClassifier(**params)
    
    # Set up StratifiedKFold for cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Perform cross validation using the selected features
    scores = cross_val_score(model, X_train_selected, y_train_resampled, scoring='f1', cv=skf, n_jobs=-1)
    
    # Return the average F1 score across folds
    return scores.mean()

# Create and run the Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best trial:")
trial = study.best_trial
print("  F1 Score: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# # Get the best hyperparameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)


# # Retrain the model with the best parameters
final_model = xgb.XGBClassifier(**best_params, objective="binary:logistic", eval_metric="logloss", random_state=42)
final_model.fit(X_train_selected, y_train_resampled)  


# # Make predictions
y_pred = final_model.predict(X_test_selected)

# Evaluate the model

f1 = f1_score(y_test, y_pred, average='binary')

# Output the performance metrics

print(f'F1 Score: {f1:.4f}')



#subset the test/inference data:

test_subset = test[feat]
test_subset.shape


#Prediction

test_pred = model.predict(test_subset)
test_pred



submission = pd.DataFrame({'Stock code': test_subset.index, 'IsDefault': test_pred})
submission.to_csv('/kaggle/working/submission.csv', index=False)
display(submission)


# Assuming X_train_selected and X_test_selected are pandas DataFrames
overlap = set(X_train_selected.index).intersection(set(X_test_selected.index))
if overlap:
    print("Warning: There is an overlap between training and test sets!")
else:
    print("No overlap detected between training and test sets.")





