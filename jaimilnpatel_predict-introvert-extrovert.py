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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')



train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')



# Display the first few rows of the training data
print("Training Data Head:")
display(train_df.head())

# Display the info of the training data
print("\nTraining Data Info:")
display(train_df.info())

# Display the first few rows of the test data
print("\nTest Data Head:")
display(test_df.head())

# Display the info of the test data
print("\nTest Data Info:")
display(test_df.info())


import matplotlib.pyplot as plt
import seaborn as sns

# Visualize the distribution of 'Time_spent_Alone'
plt.figure(figsize=(8, 5))
sns.histplot(data=train_df, x='Time_spent_Alone', hue='Personality', kde=True)
plt.title('Distribution of Time_spent_Alone by Personality')
plt.show()

# Visualize the distribution of 'Friends_circle_size'
plt.figure(figsize=(8, 5))
sns.histplot(data=train_df, x='Friends_circle_size', hue='Personality', kde=True)
plt.title('Distribution of Friends_circle_size by Personality')
plt.show()

# Visualize the distribution of 'Social_event_attendance'
plt.figure(figsize=(8, 5))
sns.histplot(data=train_df, x='Social_event_attendance', hue='Personality', kde=True)
plt.title('Distribution of Social_event_attendance by Personality')
plt.show()

# Visualize the distribution of 'Going_outside'
plt.figure(figsize=(8, 5))
sns.histplot(data=train_df, x='Going_outside', hue='Personality', kde=True)
plt.title('Distribution of Going_outside by Personality')
plt.show()


print("Percentage of missing values in train_df:")
display(train_df.isnull().sum() / len(train_df) * 100)

print("\nPercentage of missing values in test_df:")
display(test_df.isnull().sum() / len(test_df) * 100)


# Impute missing numerical values with the median from the training data
# Create missingness indicators before imputing
for col in numerical_cols:
    if train_df[col].isnull().any():
        train_df[f'{col}_was_missing'] = train_df[col].isnull().astype(int)
        test_df[f'{col}_was_missing'] = test_df[col].isnull().astype(int)

# Then impute with median
for col in numerical_cols:
    if train_df[col].isnull().any():
        median_val = train_df[col].median()
        train_df[col].fillna(median_val, inplace=True)
        if col in test_df.columns:
            test_df[col].fillna(median_val, inplace=True)

# Verify that there are no remaining missing values
print("Percentage of missing values in train_df after imputation:")
display(train_df.isnull().sum() / len(train_df) * 100)

print("\nPercentage of missing values in test_df after imputation:")
display(test_df.isnull().sum() / len(test_df) * 100)


# Identify categorical columns excluding 'Personality'
categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
if 'Personality' in categorical_cols:
    categorical_cols.remove('Personality')

# Create countplots for each categorical column
for col in categorical_cols:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=train_df, x=col, hue='Personality')
    plt.title(f'Distribution of {col} by Personality')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()


# Create scatter plots for relevant numerical features
numerical_cols = train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numerical_cols.remove('id') # Exclude the 'id' column

# Pairs to visualize based on potential relationships or interest
pairs_to_plot = [
    ('Time_spent_Alone', 'Friends_circle_size'),
    ('Social_event_attendance', 'Going_outside'),
    ('Time_spent_Alone', 'Social_event_attendance'),
    ('Friends_circle_size', 'Going_outside'),
    ('Post_frequency', 'Friends_circle_size')
]

for x_col, y_col in pairs_to_plot:
    if x_col in numerical_cols and y_col in numerical_cols:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=train_df, x=x_col, y=y_col, hue='Personality', alpha=0.6)
        plt.title(f'Scatter Plot of {x_col} vs {y_col} by Personality')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.show()

# Generate and visualize the correlation matrix
correlation_matrix = train_df[numerical_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()


# Explore the relationship between 'Post_frequency' and 'Personality'
plt.figure(figsize=(8, 5))
sns.boxplot(data=train_df, x='Personality', y='Post_frequency')
plt.title('Distribution of Post_frequency by Personality')
plt.show()

# Analyze the distribution of numerical features based on 'Stage_fear'
numerical_cols = train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numerical_cols.remove('id') # Exclude the 'id' column

for col in numerical_cols:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=train_df, x='Stage_fear', y=col, hue='Personality')
    plt.title(f'Distribution of {col} by Stage_fear and Personality')
    plt.show()

# Analyze the distribution of numerical features based on 'Drained_after_socializing'
for col in numerical_cols:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=train_df, x='Drained_after_socializing', y=col, hue='Personality')
    plt.title(f'Distribution of {col} by Drained_after_socializing and Personality')
    plt.show()


# Explore interactions: Average of numerical features by combinations of categorical features and Personality
categorical_interaction_cols = ['Stage_fear', 'Drained_after_socializing']
numerical_interaction_cols = train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numerical_interaction_cols.remove('id')

for num_col in numerical_interaction_cols:
    for cat_col1 in categorical_interaction_cols:
        for cat_col2 in categorical_interaction_cols:
            if cat_col1 != cat_col2:
                print(f"\nAverage of {num_col} by {cat_col1}, {cat_col2}, and Personality:")
                display(train_df.groupby([cat_col1, cat_col2, 'Personality'])[num_col].mean().unstack())

# Analyze the relationship between 'Post_frequency' and other numerical features by Personality
numerical_cols_for_pairplot = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
sns.pairplot(train_df, vars=numerical_cols_for_pairplot, hue='Personality', diag_kind='kde')
plt.suptitle('Pair Plot of Numerical Features by Personality', y=1.02)
plt.show()


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Separate features and target in the training data
train_features = train_df.drop('Personality', axis=1)
train_target = train_df['Personality']

# Identify categorical and numerical features (excluding 'Personality' and 'id')
categorical_features = train_features.select_dtypes(include=['object']).columns.tolist()
numerical_features = train_features.select_dtypes(include=['float64', 'int64']).columns.tolist()

if 'id' in numerical_features:
    numerical_features.remove('id')

# Create a column transformer for preprocessing
# It applies OneHotEncoder to categorical features and StandardScaler to numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Keep other columns (like 'id')
)

# Apply the preprocessor to the training features
train_processed = preprocessor.fit_transform(train_features)

# Apply the preprocessor to the test data
test_processed = preprocessor.transform(test_df)

# Convert the processed arrays back to DataFrames
# Get the new column names after one-hot encoding
ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
new_feature_names = numerical_features + list(ohe_feature_names)

# Add 'id' back to the processed dataframes
train_processed_df = pd.DataFrame(train_processed, columns=new_feature_names + ['id'])
test_processed_df = pd.DataFrame(test_processed, columns=new_feature_names + ['id'])

# Re-add the target variable to the training dataframe
train_processed_df['Personality'] = train_target.values


# Verify the processed dataframes
print("Processed Training Data Head:")
display(train_processed_df.head())

print("\nProcessed Training Data Info:")
display(train_processed_df.info())

print("\nProcessed Test Data Head:")
display(test_processed_df.head())

print("\nProcessed Test Data Info:")
display(test_processed_df.info())



# Create interaction feature: Time_spent_Alone * Drained_after_socializing_Yes
train_processed_df['Time_spent_Alone_x_Drained_Yes'] = train_processed_df['Time_spent_Alone'] * train_processed_df['Drained_after_socializing_Yes']
test_processed_df['Time_spent_Alone_x_Drained_Yes'] = test_processed_df['Time_spent_Alone'] * test_processed_df['Drained_after_socializing_Yes']

# Create interaction feature: Friends_circle_size * Social_event_attendance
train_processed_df['Friends_circle_size_x_Social_event_attendance'] = train_processed_df['Friends_circle_size'] * train_processed_df['Social_event_attendance']
test_processed_df['Friends_circle_size_x_Social_event_attendance'] = test_processed_df['Friends_circle_size'] * test_processed_df['Social_event_attendance']

# Create interaction feature: Going_outside * Social_event_attendance
train_processed_df['Going_outside_x_Social_event_attendance'] = train_processed_df['Going_outside'] * train_processed_df['Social_event_attendance']
test_processed_df['Going_outside_x_Social_event_attendance'] = test_processed_df['Going_outside'] * test_processed_df['Social_event_attendance']

# Create interaction feature: Post_frequency * Friends_circle_size
train_processed_df['Post_frequency_x_Friends_circle_size'] = train_processed_df['Post_frequency'] * train_processed_df['Friends_circle_size']
test_processed_df['Post_frequency_x_Friends_circle_size'] = test_processed_df['Post_frequency'] * test_processed_df['Friends_circle_size']


# Document the newly created features
print("Newly created features:")
print("- Time_spent_Alone_x_Drained_Yes: Interaction between Time_spent_Alone and feeling Drained_after_socializing (Yes).")
print("- Friends_circle_size_x_Social_event_attendance: Interaction between Friends_circle_size and Social_event_attendance.")
print("- Going_outside_x_Social_event_attendance: Interaction between Going_outside and Social_event_attendance.")
print("- Post_frequency_x_Friends_circle_size: Interaction between Post_frequency and Friends_circle_size.")


# Display the head of the modified dataframes
print("\nModified train_processed_df Head:")
display(train_processed_df.head())

print("\nModified test_processed_df Head:")
display(test_processed_df.head())


# Model Selection Reasoning:

# 1. Logistic Regression:
# - Simple and interpretable linear model.
# - Good baseline model to establish initial performance.
# - Suitable for binary classification.
# - Less sensitive to feature scaling (although we have already scaled).
# - Can provide probability estimates.

# 2. RandomForestClassifier:
# - Ensemble method based on decision trees.
# - Can capture non-linear relationships and feature interactions.
# - Generally robust to outliers and missing values (though handled here).
# - Provides feature importance, which can be useful for understanding key predictors.
# - Tends to perform well on a variety of datasets.

# 3. GradientBoostingClassifier (LightGBM or XGBoost):
# - Another powerful ensemble method.
# - Builds trees sequentially, correcting errors of previous trees.
# - Often achieves high accuracy and is widely used in competitions.
# - Can handle complex relationships and interactions.
# - LightGBM and XGBoost are known for their speed and efficiency.

print("Selected Models:")
print("- Logistic Regression: As a simple, interpretable baseline.")
print("- RandomForestClassifier: To capture non-linearities and interactions, and for feature importance.")
print("- GradientBoostingClassifier (e.g., LightGBM/XGBoost): For potentially higher accuracy and handling complex patterns.")


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
# from xgboost import XGBClassifier # Alternative gradient boosting model


# Separate features (X) and target (y) from the training data
# Drop the 'id' column as it's not a feature for training
X_train = train_processed_df.drop(['Personality', 'id'], axis=1)
y_train = train_processed_df['Personality']

# Instantiate the selected models
model_lr = LogisticRegression(random_state=42)
model_rf = RandomForestClassifier(random_state=42)
model_gbm = LGBMClassifier(random_state=42)
# model_xgb = XGBClassifier(random_state=42) # Instantiate XGBoost if chosen

# Train each model
print("Training Logistic Regression model...")
model_lr.fit(X_train, y_train)
print("Logistic Regression model trained.")

print("\nTraining RandomForestClassifier model...")
model_rf.fit(X_train, y_train)
print("RandomForestClassifier model trained.")

print("\nTraining LGBMClassifier model...")
model_gbm.fit(X_train, y_train)
print("LGBMClassifier model trained.")

# print("\nTraining XGBClassifier model...")
# model_xgb.fit(X_train, y_train)
# print("XGBClassifier model trained.")

# Store the trained models (already stored in the variables above)
print("\nTrained models stored in variables: model_lr, model_rf, model_gbm")


from sklearn.metrics import accuracy_score

# Make predictions on the training data
y_pred_lr = model_lr.predict(X_train)
y_pred_rf = model_rf.predict(X_train)
y_pred_gbm = model_gbm.predict(X_train)

# Calculate accuracy for each model
accuracy_lr = accuracy_score(y_train, y_pred_lr)
accuracy_rf = accuracy_score(y_train, y_pred_rf)
accuracy_gbm = accuracy_score(y_train, y_pred_gbm)

# Print the accuracy scores
print(f"Training Accuracy - Logistic Regression: {accuracy_lr:.4f}")
print(f"Training Accuracy - RandomForestClassifier: {accuracy_rf:.4f}")
print(f"Training Accuracy - LGBMClassifier: {accuracy_gbm:.4f}")


from sklearn.model_selection import GridSearchCV

# Define parameter grids for the best-performing models (RandomForestClassifier and LGBMClassifier)

# Parameter grid for RandomForestClassifier
# Reduced the grid size for faster execution
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Parameter grid for LGBMClassifier
# Reduced the grid size for faster execution
param_grid_gbm = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'num_leaves': [31, 62],
    'max_depth': [10, -1]
}


# Instantiate GridSearchCV for RandomForestClassifier
grid_search_rf = GridSearchCV(estimator=model_rf, param_grid=param_grid_rf, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)

# Instantiate GridSearchCV for LGBMClassifier
grid_search_gbm = GridSearchCV(estimator=model_gbm, param_grid=param_grid_gbm, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)

# Fit GridSearchCV to the training data for RandomForestClassifier
print("Performing Grid Search for RandomForestClassifier...")
grid_search_rf.fit(X_train, y_train)
print("Grid Search for RandomForestClassifier completed.")

# Print the best parameters found for RandomForestClassifier
print("\nBest parameters for RandomForestClassifier:")
print(grid_search_rf.best_params_)

# Fit GridSearchCV to the training data for LGBMClassifier
print("\nPerforming Grid Search for LGBMClassifier...")
grid_search_gbm.fit(X_train, y_train)
print("Grid Search for LGBMClassifier completed.")

# Print the best parameters found for LGBMClassifier
print("\nBest parameters for LGBMClassifier:")
print(grid_search_gbm.best_params_)

# Train the best RandomForestClassifier model with the best parameters
best_model_rf = grid_search_rf.best_estimator_
print("\nTraining the best RandomForestClassifier model...")
best_model_rf.fit(X_train, y_train)
print("Best RandomForestClassifier model trained.")

# Train the best LGBMClassifier model with the best parameters
best_model_gbm = grid_search_gbm.best_estimator_
print("\nTraining the best LGBMClassifier model...")
best_model_gbm.fit(X_train, y_train)
print("Best LGBMClassifier model trained.")

# Evaluate the tuned models on the training data
y_pred_tuned_rf = best_model_rf.predict(X_train)
accuracy_tuned_rf = accuracy_score(y_train, y_pred_tuned_rf)

y_pred_tuned_gbm = best_model_gbm.predict(X_train)
accuracy_tuned_gbm = accuracy_score(y_train, y_pred_tuned_gbm)

# Print the accuracy scores of the tuned models
print(f"\nTraining Accuracy - Tuned RandomForestClassifier: {accuracy_tuned_rf:.4f}")
print(f"Training Accuracy - Tuned LGBMClassifier: {accuracy_tuned_gbm:.4f}")


from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

# Define the ensemble model (VotingClassifier with 'soft' voting)
# Using the best-performing tuned models: best_model_rf and best_model_gbm
# Ensure that the models support predict_proba for 'soft' voting
ensemble_model = VotingClassifier(
    estimators=[('rf', best_model_rf), ('gbm', best_model_gbm)],
    voting='soft'  # Use 'soft' voting for weighted probability averaging
)

# Train the ensemble model on the training data
print("Training the Ensemble Model (VotingClassifier)...")
ensemble_model.fit(X_train, y_train)
print("Ensemble Model trained.")

# Evaluate the performance of the ensemble model on the training data
y_pred_ensemble_train = ensemble_model.predict(X_train)
accuracy_ensemble_train = accuracy_score(y_train, y_pred_ensemble_train)
precision_ensemble_train = precision_score(y_train, y_pred_ensemble_train, pos_label='Extrovert')
recall_ensemble_train = recall_score(y_train, y_pred_ensemble_train, pos_label='Extrovert')
f1_ensemble_train = f1_score(y_train, y_pred_ensemble_train, pos_label='Extrovert')
roc_auc_ensemble_train = roc_auc_score(y_train, ensemble_model.predict_proba(X_train)[:, 1])


print("\nEvaluation Metrics for Ensemble Model on Training Data:")
print(f"Accuracy: {accuracy_ensemble_train:.4f}")
print(f"Precision: {precision_ensemble_train:.4f}")
print(f"Recall: {recall_ensemble_train:.4f}")
print(f"F1-Score: {f1_ensemble_train:.4f}")
print(f"ROC AUC: {roc_auc_ensemble_train:.4f}")


# Perform cross-validation for the ensemble model
cv_scores_ensemble = cross_val_score(ensemble_model, X_train, y_train, cv=5, scoring='accuracy')
print("\nCross-validation scores for Ensemble Model:")
print(cv_scores_ensemble)
print(f"Mean CV Accuracy: {cv_scores_ensemble.mean():.4f}")
print(f"Standard Deviation of CV Accuracy: {cv_scores_ensemble.std():.4f}")

# Compare performance with individual models
print("\nComparison of Model Performance:")
print(f"Tuned RandomForestClassifier Training Accuracy: {accuracy_tuned_rf:.4f}")
print(f"Tuned LGBMClassifier Training Accuracy: {accuracy_tuned_gbm:.4f}")
print(f"Ensemble Model Training Accuracy: {accuracy_ensemble_train:.4f}")


import matplotlib.pyplot as plt
import seaborn as sns

# Get feature importances from the tuned RandomForestClassifier
rf_importances = best_model_rf.feature_importances_

# Get feature importances from the tuned LGBMClassifier
gbm_importances = best_model_gbm.feature_importances_

# Create a pandas Series for feature importances with feature names as index
feature_names = X_train.columns
feature_importances_rf_series = pd.Series(rf_importances, index=feature_names)
feature_importances_gbm_series = pd.Series(gbm_importances, index=feature_names)

# Sort feature importances in descending order
sorted_feature_importances_rf = feature_importances_rf_series.sort_values(ascending=False)
sorted_feature_importances_gbm = feature_importances_gbm_series.sort_values(ascending=False)

# Print the sorted feature importances
print("Feature Importances (RandomForestClassifier):")
display(sorted_feature_importances_rf)

print("\nFeature Importances (LGBMClassifier):")
display(sorted_feature_importances_gbm)

# Visualize the sorted feature importances
plt.figure(figsize=(12, 7))
sns.barplot(x=sorted_feature_importances_rf.values, y=sorted_feature_importances_rf.index)
plt.title('Feature Importances (RandomForestClassifier)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

plt.figure(figsize=(12, 7))
sns.barplot(x=sorted_feature_importances_gbm.values, y=sorted_feature_importances_gbm.index)
plt.title('Feature Importances (LGBMClassifier)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Analyze and discuss the top features
print("\nAnalysis of Top Features:")
print("Based on both RandomForestClassifier and LGBMClassifier feature importances, the most important features for predicting personality are consistently related to:")
print("- Stage_fear: Both 'Stage_fear_No' and 'Stage_fear_Yes' are highly important, indicating that having or not having stage fear is a strong predictor.")
print("- Drained_after_socializing: Similarly, 'Drained_after_socializing_No' and 'Drained_after_socializing_Yes' are among the top features, highlighting the significance of feeling drained after social interactions.")
print("- Time_spent_Alone: This numerical feature is also consistently ranked high, aligning with the EDA insight that Introverts tend to spend more time alone.")
print("- Post_frequency: This feature also shows moderate importance, suggesting a link between social media activity and personality.")
print("\nOther features like 'Going_outside', 'Social_event_attendance', and 'Friends_circle_size' are also important, reflecting the social aspects differentiating extroverts and introverts. The engineered interaction features generally have lower importance compared to the original features, although 'Time_spent_Alone_x_Drained_Yes' shows some relevance.")


# Select the best-performing model
best_model = best_model_gbm

# Prepare test data by dropping 'id' column
X_test = test_processed_df.drop('id', axis=1)

# Predict using the model
test_pred_labels = best_model.predict(X_test)

# Load sample submission to get the correct ID format/order
sub = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

# Create final submission DataFrame
submission = pd.DataFrame({
    'id': sub['id'],  # preserve order from sample submission
    'Personality': test_pred_labels
})

# Save to CSV for submission
submission.to_csv('submission.csv', index=False)

# Optional: preview top rows
print("✅ Submission file created:")
display(submission.head())


