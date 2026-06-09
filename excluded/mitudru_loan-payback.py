import pandas as pd

train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')

print('Train DataFrame Head:')
print(train_df.head())
print('\nTest DataFrame Head:')
print(test_df.head())
print('\nSample Submission DataFrame Head:')
print(sample_submission_df.head())


print('Train DataFrame Info:')
train_df.info()
print('\nTest DataFrame Info:')
test_df.info()


print('Missing values in Train DataFrame:')
print((train_df.isnull().sum() / len(train_df)) * 100)

print('\nMissing values in Test DataFrame:')
print((test_df.isnull().sum() / len(test_df)) * 100)


print('Descriptive Statistics for Numerical Features in Train DataFrame:')
print(train_df.describe())


print('Unique Values and Counts for Categorical Features in Train DataFrame:')
categorical_cols = train_df.select_dtypes(include='object').columns

for col in categorical_cols:
    print(f"\n--- Column: {col} ---")
    print(train_df[col].value_counts())



import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# List of numerical features to plot
numerical_features = ['annual_income', 'loan_amount', 'credit_score', 'interest_rate', 'debt_to_income_ratio']

# Create subplots for each numerical feature
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_features):
    plt.subplot(2, 3, i + 1) # Adjust subplot grid as needed
    sns.histplot(train_df[col], kde=True, bins=50)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# List of categorical features to plot
categorical_features = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']

# Create subplots for each categorical feature
plt.figure(figsize=(20, 20)) # Increased figure size to accommodate more plots and labels
for i, col in enumerate(categorical_features):
    plt.subplot(3, 2, i + 1) # Adjust subplot grid as needed
    sns.countplot(data=train_df, y=col, order=train_df[col].value_counts().index, palette='viridis') # Using y for horizontal bars for better readability, and ordering by count
    plt.title(f'Count of {col}')
    plt.xlabel('Count')
    plt.ylabel(col)

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

categorical_features = [
    'gender', 'marital_status', 'education_level',
    'employment_status', 'loan_purpose', 'grade_subgrade'
]

plt.figure(figsize=(20, 20))

for i, col in enumerate(categorical_features):
    plt.subplot(3, 2, i + 1)
    
    sns.countplot(
        data=train_df,
        y=col,
        order=train_df[col].value_counts().index,
        palette='viridis'
    )
    
    plt.title(f'Count of {col}')
    plt.xlabel('Count')
    plt.ylabel(col)
    
    # Remove legend explicitly
    plt.legend([], [], frameon=False)

plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

plt.figure(figsize=(8, 6))

sns.countplot(
    data=train_df,
    x='loan_paid_back',
    palette='viridis',
    hue='loan_paid_back'   # OK to keep this if you want two different colors
)

# Remove the legend explicitly
plt.legend([], [], frameon=False)

plt.title('Distribution of Loan Paid Back (Target Variable)')
plt.xlabel('Loan Paid Back')
plt.ylabel('Count')
plt.xticks([0, 1], ['Not Paid Back (0)', 'Paid Back (1)'])

plt.show()

print('\nValue Counts for loan_paid_back:')
print(train_df['loan_paid_back'].value_counts())

print('\nPercentage Distribution for loan_paid_back:')
print(train_df['loan_paid_back'].value_counts(normalize=True) * 100)



import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Select only numerical columns for correlation matrix
numerical_df = train_df.select_dtypes(include=['float64', 'int64'])

# Calculate the correlation matrix
correlation_matrix = numerical_df.corr()

# Plotting the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Features in Train DataFrame')
plt.show()



import pandas as pd
from sklearn.preprocessing import StandardScaler

# Separate target variable from train_df before preprocessing
X_train = train_df.drop('loan_paid_back', axis=1)
y_train = train_df['loan_paid_back']

# Identify categorical columns
categorical_cols_train = X_train.select_dtypes(include='object').columns
categorical_cols_test = test_df.select_dtypes(include='object').columns

# Ensure both lists are identical for consistency
# If they are not, it means one dataframe has a category that the other doesn't for a given column.
# For now, we'll assume they are similar and get dummies for both.
# A more robust approach would be to concatenate both, apply get_dummies, then split.
# However, for this task, we will apply separately and then align.

# Apply one-hot encoding to categorical features in both train and test datasets
X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols_train, drop_first=True)
test_df_encoded = pd.get_dummies(test_df, columns=categorical_cols_test, drop_first=True)

# Align columns - this is crucial to ensure both dataframes have the same columns after one-hot encoding
X_train_aligned, test_df_aligned = X_train_encoded.align(test_df_encoded, join='outer', axis=1, fill_value=0)

# Identify numerical columns (excluding 'id' which is not a feature for scaling)
numerical_cols_train = X_train_aligned.select_dtypes(include=['int64', 'float64']).columns.drop('id')
numerical_cols_test = test_df_aligned.select_dtypes(include=['int64', 'float64']).columns.drop('id')

# Initialize StandardScaler
scaler = StandardScaler()

# Fit on training data numerical columns and transform both train and test
X_train_aligned[numerical_cols_train] = scaler.fit_transform(X_train_aligned[numerical_cols_train])
test_df_aligned[numerical_cols_test] = scaler.transform(test_df_aligned[numerical_cols_test])

# Re-add 'id' column to the preprocessed dataframes if needed later, but for training, it's usually dropped
# We will keep it for now as it's not explicitly stated to drop it permanently

# Display the first few rows of the preprocessed dataframes
print('Preprocessed X_train (features) Head:')
print(X_train_aligned.head())
print('\nPreprocessed Test DataFrame Head:')
print(test_df_aligned.head())

print(f'Shape of X_train_aligned: {X_train_aligned.shape}')
print(f'Shape of y_train: {y_train.shape}')
print(f'Shape of test_df_aligned: {test_df_aligned.shape}')


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Create new features for train_df
train_df['loan_to_income_ratio'] = train_df['loan_amount'] / train_df['annual_income']
train_df['interest_rate_per_credit_score'] = train_df['interest_rate'] / train_df['credit_score']

# Create new features for test_df
test_df['loan_to_income_ratio'] = test_df['loan_amount'] / test_df['annual_income']
test_df['interest_rate_per_credit_score'] = test_df['interest_rate'] / test_df['credit_score']

print("New features created for train_df. Head of relevant columns:")
print(train_df[['loan_to_income_ratio', 'interest_rate_per_credit_score', 'loan_amount', 'annual_income', 'interest_rate', 'credit_score']].head())

print("\nNew features created for test_df. Head of relevant columns:")
print(test_df[['loan_to_income_ratio', 'interest_rate_per_credit_score', 'loan_amount', 'annual_income', 'interest_rate', 'credit_score']].head())


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

plt.figure(figsize=(10, 6))
sns.kdeplot(data=train_df, x='loan_to_income_ratio', hue='loan_paid_back', fill=True, common_norm=False, palette='viridis')
plt.title('Distribution of Loan to Income Ratio by Loan Paid Back Status')
plt.xlabel('Loan to Income Ratio')
plt.ylabel('Density')
plt.legend(title='Loan Paid Back', labels=['Not Paid Back (0)', 'Paid Back (1)'])
plt.show()


from sklearn.preprocessing import StandardScaler

# Initialize StandardScaler
scaler_new_features = StandardScaler()

# Define the new features to be scaled
new_features = ['loan_to_income_ratio', 'interest_rate_per_credit_score']

# Fit on training data and transform both train and test
# Using .copy() to avoid SettingWithCopyWarning
train_df_scaled_new_features = scaler_new_features.fit_transform(train_df[new_features])
test_df_scaled_new_features = scaler_new_features.transform(test_df[new_features])

# Add the scaled new features to X_train_aligned and test_df_aligned
X_train_aligned['loan_to_income_ratio_scaled'] = train_df_scaled_new_features[:, 0]
X_train_aligned['interest_rate_per_credit_score_scaled'] = train_df_scaled_new_features[:, 1]

test_df_aligned['loan_to_income_ratio_scaled'] = test_df_scaled_new_features[:, 0]
test_df_aligned['interest_rate_per_credit_score_scaled'] = test_df_scaled_new_features[:, 1]

# Display the first few rows and shape of the modified dataframes
print('Modified X_train_aligned (features) Head:')
print(X_train_aligned.head())
print(f'Shape of X_train_aligned: {X_train_aligned.shape}')

print('\nModified Test DataFrame Head:')
print(test_df_aligned.head())
print(f'Shape of test_df_aligned: {test_df_aligned.shape}')


import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

# Calculate scale_pos_weight for handling class imbalance
neg_count = y_train.value_counts()[0.0]
pos_count = y_train.value_counts()[1.0]
scale_pos_weight_value = neg_count / pos_count

print(f"Class 0 count: {neg_count}")
print(f"Class 1 count: {pos_count}")
print(f"Scale Pos Weight: {scale_pos_weight_value:.2f}")

# 1. Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 2. Initialize LGBMClassifier
lgbm_model = lgb.LGBMClassifier(
    objective='binary',
    random_state=42,
    n_estimators=1000, # Increased estimators for potentially better performance
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    reg_alpha=0.1,
    reg_lambda=0.1,
    scale_pos_weight=scale_pos_weight_value, # Handle class imbalance
    n_jobs=-1 # Use all available cores
)

roc_auc_scores = []

print("\nStarting Stratified K-Fold Cross-Validation...")

# 3. Perform cross-validation
for fold, (train_index, val_index) in enumerate(skf.split(X_train_aligned, y_train)):
    print(f"\nFold {fold+1}/")
    X_train_fold, X_val_fold = X_train_aligned.iloc[train_index], X_train_aligned.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    # Train the model on the training data of the current fold
    lgbm_model.fit(X_train_fold, y_train_fold)

    # Predict probabilities on the validation set
    y_pred_proba = lgbm_model.predict_proba(X_val_fold)[:, 1]

    # Calculate ROC AUC score for the current fold
    fold_roc_auc = roc_auc_score(y_val_fold, y_pred_proba)
    roc_auc_scores.append(fold_roc_auc)
    print(f"ROC AUC for Fold {fold+1}: {fold_roc_auc:.4f}")

# Print the average ROC AUC score and its standard deviation
print(f"\nAverage ROC AUC across folds: {np.mean(roc_auc_scores):.4f}")
print(f"Standard Deviation of ROC AUC across folds: {np.std(roc_auc_scores):.4f}")

print("\nTraining final model on the entire training dataset...")
# Train the final model on the entire X_train_aligned and y_train dataset
final_lgbm_model = lgb.LGBMClassifier(
    objective='binary',
    random_state=42,
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    reg_alpha=0.1,
    reg_lambda=0.1,
    scale_pos_weight=scale_pos_weight_value,
    n_jobs=-1
)
final_lgbm_model.fit(X_train_aligned, y_train)
print("Final model trained successfully.")


import pandas as pd

# 1. Predict probabilities on the preprocessed test set
X_test_features = test_df_aligned.copy()

# Predict probabilities for the positive class (class 1)
test_predictions_proba = final_lgbm_model.predict_proba(X_test_features)[:, 1]

# 2. Create a new pandas DataFrame for submission
submission_df = pd.DataFrame({
    'id': test_df['id'],  
    'loan_paid_back': test_predictions_proba
})

# 5. Save the submission DataFrame to the Kaggle output location
output_path = "/kaggle/working/submission.csv"
submission_df.to_csv(output_path, index=False)

print(f"Submission file saved to: {output_path}")
print("\nFirst 5 rows of submission_df:")
print(submission_df.head())


