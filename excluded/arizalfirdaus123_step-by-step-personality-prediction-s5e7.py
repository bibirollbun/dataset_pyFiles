# Core libraries for data manipulation and visualization
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Scikit-learn for preprocessing, modeling, and metrics
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform

# The machine learning model
from lightgbm import LGBMClassifier

# Kaggle-specific imports
import kagglehub

print("Libraries imported successfully!")


# import kagglehub
# kagglehub.login()


# playground_series_s5e7_path = kagglehub.competition_download('playground-series-s5e7')

# print('Data source import complete.')


# Load the three essential datasets from the competition
# -- Colab --
# train = pd.read_csv(f"{playground_series_s5e7_path}/train.csv")
# test = pd.read_csv(f"{playground_series_s5e7_path}/test.csv")
# sample_submission = pd.read_csv(f"{playground_series_s5e7_path}/sample_submission.csv")

train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

print("Data loaded successfully!")

original = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_datasert.csv")

print("External dataset loaded successfully!")


# Display the first 5 rows of the training data to validate it's loaded correctly
print("Top 5 rows of the training data:")
train.head()


# Print all column names in the training data
print("Column Names:")
print(train.columns)


# Check the data type of each column
print("\nData Types:")
print(train.dtypes)


# Set plot style
sns.set_style("whitegrid")

# Create a count plot of the target variable
plt.figure(figsize=(6, 4))
sns.countplot(x='Personality', data=train, palette='viridis')
plt.title('Target Variable Distribution', fontsize=14)
plt.show()


# Select only the feature columns (excluding 'id' and the target)
features = train.drop(['id', 'Personality'], axis=1).columns

plt.figure(figsize=(16, 12))
for i, feature in enumerate(features):
    plt.subplot(4, 5, i + 1)
    sns.histplot(train[feature], kde=True, bins=30, color='skyblue')
    plt.title(feature, fontsize=10)
    plt.xlabel('')
    plt.ylabel('')
plt.tight_layout()
plt.suptitle('Distribution of Input Features', y=1.02, fontsize=16)
plt.show()


# Create a fresh copy to work with for correlation analysis
train_corr = train.copy()

# Define mappings for categorical columns
feature_columns_to_map = ['Stage_fear', 'Drained_after_socializing']
yes_no_mapping = {'Yes': 1, 'No': 0}
personality_mapping = {'Extrovert': 1, 'Introvert': 0}

# Apply mappings
for col in feature_columns_to_map:
    train_corr[col] = train_corr[col].map(yes_no_mapping)
train_corr['Personality'] = train_corr['Personality'].map(personality_mapping)

print("Mapping complete. Now handling missing values for correlation matrix...")

# Fill missing values with the mode for this temporary dataframe
for col in feature_columns_to_map:
    mode_value = train_corr[col].mode()[0]
    train_corr[col] = train_corr[col].fillna(mode_value)
    print(f"NaNs in '{col}' filled with mode: {int(mode_value)}")

print("-" * 30)

# Generate and plot the correlation matrix
corr_matrix = train_corr.drop('id', axis=1).corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Features', fontsize=16)
plt.show()


print("Merging with external dataset...")
# Rename the 'Personality' column in the original data to avoid conflicts
df_original = original.rename(columns={'Personality': 'match_p'})

# Define the key columns for merging
merge_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
              'Going_outside', 'Drained_after_socializing', 
              'Friends_circle_size', 'Post_frequency']

df_original = df_original.drop_duplicates(subset=merge_cols)

# Merge with the training and test data
train = train.merge(df_original, how='left', on=merge_cols)
test = test.merge(df_original, how='left', on=merge_cols)

train = train.drop_duplicates(subset=['id'], keep='first')
test = test.drop_duplicates(subset=['id'], keep='first')

print(f"Train shape after merge: {train.shape}")
print(f"Test shape after merge: {test.shape}")

print("\nPerforming Feature Engineering...")
# Create a flag feature for matched data
train['match_p_is_null'] = train['match_p'].isna().astype(int)
test['match_p_is_null'] = test['match_p'].isna().astype(int)

# Fill NaNs in match_p with an 'unknown' category for later encoding
train['match_p'] = train['match_p'].fillna('unknown')
test['match_p'] = test['match_p'].fillna('unknown')

print("New features 'match_p_is_null' and 'match_p' have been created.")


# Define mappings
yes_no_mapping = {'Yes': 1, 'No': 0}

# Mapping
personality_target_mapping = {'Extrovert': 1, 'Introvert': 0}
personality_feature_mapping = {'Extrovert': 2, 'Introvert': 1, 'unknown': 0}

# Apply mappings to train and test sets
train['Stage_fear'] = train['Stage_fear'].map(yes_no_mapping)
test['Stage_fear'] = test['Stage_fear'].map(yes_no_mapping)

train['Drained_after_socializing'] = train['Drained_after_socializing'].map(yes_no_mapping)
test['Drained_after_socializing'] = test['Drained_after_socializing'].map(yes_no_mapping)

# Apply mapping for the new feature 'match_p'
train['match_p'] = train['match_p'].map(personality_feature_mapping)
test['match_p'] = test['match_p'].map(personality_feature_mapping)

# Apply mapping for the target variable 'Personality' in the train set
train['Personality'] = train['Personality'].map(personality_target_mapping)

print("All categorical columns have been mapped to numeric values.")


# Define our feature set (X) and target (y)
X = train.drop(['id', 'Personality'], axis=1)
y = train['Personality']
X_test = test.drop('id', axis=1)

for col in X.select_dtypes(include=np.number).columns:
    mean_value = X[col].mean()
    X[col] = X[col].fillna(mean_value)
    X_test[col] = X_test[col].fillna(mean_value)

# Initialize and apply the scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

print("Data has been separated, imputed, and scaled successfully.")


# Initialize and train the baseline model
model = LGBMClassifier(
    objective='binary',
    metric='auc',
    random_state=42
)

model.fit(X_scaled, y)

print("Baseline model training complete.")


# Calculate predictions on the training set for accuracy evaluation
train_predictions = model.predict(X_scaled)

# Calculate accuracy on the training set
accuracy = accuracy_score(y, train_predictions)

print(f"Accuracy on the training set: {accuracy:.4f}")


print("Starting Randomizer search...")

# Initialize the LGBMClassifier for tuning
lgbm = LGBMClassifier(objective='binary', metric='auc', random_state=42, device='gpu')

# Define the parameter distribution for RandomizedSearchCV
param_dist = {
    'verbose' : [-1],
    'n_estimators': [100, 200, 300, 400, 500],
    'learning_rate': uniform(0.01, 0.1),
    'num_leaves': [20, 31, 40, 50, 60],
    'max_depth': [-1, 10, 20, 30],
    'min_child_samples': [10, 20, 30, 40, 50],
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.7, 0.3),
}

# Initialize and run RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=lgbm,
    param_distributions=param_dist,
    n_iter=100,
    scoring='roc_auc',
    cv=3,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_scaled, y)

print("\nRandomized search complete.")
print(f"Best parameters found: {random_search.best_params_}")
print(f"Best cross-validated AUC score: {random_search.best_score_:.4f}")


# Define the best parameters from the search
best_params = random_search.best_params_

# Initialize the LGBMClassifier with the best parameters
lgbm_tuned = LGBMClassifier(objective='binary', metric='auc', random_state=42, device='gpu', **best_params)

# Initialize 5-Fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []
# Initialize an array to store the test set probability predictions
test_predictions = np.zeros((len(X_test_scaled), 2)) # (number of test rows, 2 classes)

print("Starting Stratified K-Fold Cross-Validation and Prediction...")

# Loop through each fold
for fold, (train_index, val_index) in enumerate(skf.split(X_scaled, y)):
    print(f"--- Fold {fold+1} ---")
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Train the model on the fold's training data
    lgbm_tuned.fit(X_train, y_train)

    # Predict probabilities on the validation data
    y_pred_proba = lgbm_tuned.predict_proba(X_val)[:, 1]

    # Calculate and store the AUC score for the fold
    auc = roc_auc_score(y_val, y_pred_proba)
    print(f"Fold {fold+1} AUC: {auc:.4f}")
    auc_scores.append(auc)
    
    # Add the probability predictions from this fold's model to the total test predictions
    # and divide by the number of splits to average them
    test_predictions += lgbm_tuned.predict_proba(X_test_scaled) / skf.n_splits

print("-" * 30)
print(f"Average AUC across all folds: {np.mean(auc_scores):.4f}")
print("Cross-validation and test prediction complete.")


# Get the final class predictions by taking the argmax of the averaged probabilities
final_predictions_binary = np.argmax(test_predictions, axis=1)
print("Generated final binary predictions from ensembled probabilities.")

# Create a reverse mapping to convert 0/1 back to text labels
reverse_mapping = {1: 'Extrovert', 0: 'Introvert'}
final_predictions_text = pd.Series(final_predictions_binary).map(reverse_mapping)
print("Converted binary predictions to text labels ('Extrovert'/'Introvert').")

# Display the first 5 text predictions to verify
print("\nFirst 5 generated text labels:")
print(final_predictions_text.head())


# Create the submission DataFrame using the sample submission as a template
submission_df = pd.DataFrame({'id': test['id'], 'Personality': final_predictions_text})

# Save the final submission file
submission_df.to_csv('submission_tuned.csv', index=False)

print("Submission file 'submission_tuned.csv' created successfully!")
display(submission_df.head())

