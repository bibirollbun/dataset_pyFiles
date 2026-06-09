import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# Define the base path to the dataset folder in Google Drive
base_path = '/kaggle/input/playground-series-s5e7'

# Define the paths to the individual CSV files
train_path = f'{base_path}/train.csv'
test_path = f'{base_path}/test.csv'
sample_submission_path = f'{base_path}/sample_submission.csv'

# Read the CSV files into pandas DataFrames
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_submission_df = pd.read_csv(sample_submission_path)


# Check the shape of the training and test datasets
print("Shape of train_df:", train_df.shape)
print("Shape of test_df:", test_df.shape)

# Look at basic info about data types and memory usage for the training dataset
print("\nInfo of train_df:")
display(train_df.info())

# Look at basic info about data types and memory usage for the test dataset
print("\nInfo of test_df:")
display(test_df.info())


# Calculate missing values for train_df
train_missing = train_df.isnull().sum()
train_missing_percentage = (train_missing / len(train_df)) * 100

# Calculate missing values for test_df
test_missing = test_df.isnull().sum()
test_missing_percentage = (test_missing / len(test_df)) * 100

# Create a DataFrame to summarize missing values
missing_summary = pd.DataFrame({
    'Train Missing Count': train_missing,
    'Train Missing Percentage (%)': train_missing_percentage,
    'Test Missing Count': test_missing,
    'Test Missing Percentage (%)': test_missing_percentage
})

# Display the missing values summary table
display(missing_summary)


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
import numpy as np

# Identify numerical and categorical columns (excluding the target 'Personality' from imputation)
numerical_cols = train_df.select_dtypes(include=np.number).columns.tolist()
# Remove 'id' from numerical columns as it's an identifier and not a feature for imputation
numerical_cols.remove('id')
categorical_cols = train_df.select_dtypes(include='object').columns.tolist()
# Remove 'Personality' from categorical columns as it's the target variable
if 'Personality' in categorical_cols:
    categorical_cols.remove('Personality')
display(numerical_cols)
display(categorical_cols)


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

def impute_datasets(train_df, test_df, numerical_cols, categorical_cols):
    # Initiate MICE for numerical features and mode imputer for categorical features
    iterative_imputer = IterativeImputer(max_iter=10, random_state=0, min_value=0)
    mode_imputer = SimpleImputer(strategy='most_frequent')

    train_df[numerical_cols] = iterative_imputer.fit_transform(train_df[numerical_cols])
    train_df[categorical_cols] = mode_imputer.fit_transform(train_df[categorical_cols])

    test_df[numerical_cols] = iterative_imputer.transform(test_df[numerical_cols])
    test_df[categorical_cols] = mode_imputer.transform(test_df[categorical_cols])

    return train_df, test_df

train_df, test_df = impute_datasets(train_df, test_df, numerical_cols, categorical_cols)

# Verify that there are no more missing values in the relevant columns
print("Missing values after imputation (train_df):")
display(train_df[numerical_cols + categorical_cols].isnull().sum())

print("\nMissing values after imputation (test_df):")
display(test_df[numerical_cols + categorical_cols].isnull().sum())



# Count the occurrences of each class in the target variable
class_counts = train_df['Personality'].value_counts()

# Calculate the percentage split
class_percentages = train_df['Personality'].value_counts(normalize=True) * 100

print("Class Distribution Counts:")
print(class_counts)

print("\nClass Distribution Percentages:")
print(class_percentages)


# Separate numerical & categorical features
numerical_features = train_df.select_dtypes(include=np.number).columns.tolist()
numerical_features.remove('id')
categorical_features = train_df.select_dtypes(include='object').columns.tolist()
categorical_features.remove('Personality')

display(numerical_features)
display(categorical_features)


print("Unique values for categorical features:")
for col in categorical_features:
    print(f"\nFeature: {col}")
    print(train_df[col].unique())


import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for the plots
sns.set(style="whitegrid")

# Create boxplots for all numerical features
plt.figure(figsize=(15, 7))
for i, col in enumerate(numerical_features):
    plt.subplot(2, 3, i + 1) # Adjust the subplot grid as needed
    sns.boxplot(y=train_df[col])
    plt.title(f'Boxplot of {col}')
    plt.ylabel(col)

plt.tight_layout()
plt.show()


# Plot histograms for all numerical features
plt.figure(figsize=(15, 7))
for i, col in enumerate(numerical_features):
    plt.subplot(2, 3, i + 1) # Adjust the subplot grid as needed
    sns.histplot(train_df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import math

# Set the style for the plots
sns.set(style="whitegrid")
print("Analyzing Numerical Features vs. Personality:")

# Calculate grid dimensions
n_features = len(numerical_features)
n_cols = 3  # You can adjust this based on your preference
n_rows = math.ceil(n_features / n_cols)

# Create subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
fig.suptitle('Numerical Features Distribution by Personality Type', fontsize=16, y=0.98)

# Flatten axes array for easier indexing (handles both 1D and 2D cases)
if n_rows == 1:
    axes = [axes] if n_cols == 1 else axes
else:
    axes = axes.flatten()

# Create violin plots for each numerical feature
for i, feature in enumerate(numerical_features):
    sns.violinplot(x='Personality', y=feature, data=train_df, ax=axes[i])
    axes[i].set_title(f'{feature} Distribution by Personality Type')
    axes[i].set_xlabel('Personality')
    axes[i].set_ylabel(feature)
    
    # Rotate x-axis labels if needed for better readability
    axes[i].tick_params(axis='x', rotation=45)

# Hide any unused subplots
for i in range(n_features, len(axes)):
    axes[i].set_visible(False)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


print("\nAnalyzing Categorical Features vs. Personality:")

# Create cross-tabulations and analyze the distribution of personality within each category
for feature in categorical_features:
    print(f"\nFeature: {feature}")
    cross_tab = pd.crosstab(train_df[feature], train_df['Personality'])
    display(cross_tab)

    # Calculate percentages within each category
    cross_tab_percentage = pd.crosstab(train_df[feature], train_df['Personality'], normalize='index') * 100
    print(f"Percentage distribution of Personality within each {feature} category:")
    display(cross_tab_percentage)


# Calculate the correlation matrix for numerical features
correlation_matrix = train_df[numerical_features].corr()

print("Correlation Matrix of Numerical Features:")
display(correlation_matrix)

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()


print("\nAnalyzing Relationships Among Categorical Features:")

# Create cross-tabulations for pairs of categorical features
for i in range(len(categorical_features)):
    for j in range(i + 1, len(categorical_features)):
        feature1 = categorical_features[i]
        feature2 = categorical_features[j]
        print(f"\nRelationship between {feature1} and {feature2}:")
        cross_tab_cat = pd.crosstab(train_df[feature1], train_df[feature2])
        display(cross_tab_cat)


print("\nAnalyzing Mixed Relationships (Numerical and Categorical Features):")

# Calculate optimal grid dimensions
total_plots = len(categorical_features) * len(numerical_features)
n_cols = min(len(numerical_features), 4)  # Limit columns for readability
n_rows = math.ceil(total_plots / n_cols)

# Create one large grid
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
fig.suptitle('Mixed Relationships: Numerical vs Categorical Features', fontsize=18, y=0.98)

# Flatten axes array
axes = axes.flatten() if total_plots > 1 else [axes]

plot_idx = 0

# Create all combinations
for cat_feature in categorical_features:
    for num_feature in numerical_features:
        if plot_idx < len(axes):
            sns.boxplot(x=cat_feature, y=num_feature, data=train_df, ax=axes[plot_idx])
            axes[plot_idx].set_title(f'{num_feature} by {cat_feature}', fontsize=11, fontweight='bold')
            axes[plot_idx].set_xlabel(cat_feature, fontsize=9)
            axes[plot_idx].set_ylabel(num_feature, fontsize=9)
            axes[plot_idx].tick_params(axis='x', rotation=45, labelsize=8)
            axes[plot_idx].tick_params(axis='y', labelsize=8)
            plot_idx += 1

# Hide unused subplots
for i in range(plot_idx, len(axes)):
    axes[i].axis('off')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.94)  # Make room for main title
plt.show()


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import xgboost as xgb

def create_xgboost_pipeline(df, target_col='personality'):
    """
    Simple XGBoost pipeline for personality classification

    Parameters:
    - df: DataFrame with features and target
    - target_col: name of target column

    Returns:
    - pipeline: fitted sklearn pipeline
    - cv_results: cross-validation results
    """

    # Define feature columns
    numerical_features = ['Time_spent_Alone', 'Social_event_attendance',
                         'Going_outside', 'Friends_circle_size', 'Post_frequency']
    categorical_features = ['Stage_fear', 'Drained_after_socializing']

    # Prepare features
    X = df[numerical_features + categorical_features].copy()
    y = df[target_col]

    # Encode binary categorical features (Yes/No -> 1/0)
    le_stage = LabelEncoder()
    le_drained = LabelEncoder()

    X['Stage_fear'] = le_stage.fit_transform(X['Stage_fear'])
    X['Drained_after_socializing'] = le_drained.fit_transform(X['Drained_after_socializing'])

    # Encode target
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)

    # Calculate class weights for imbalanced data (0.28 introvert, 0.72 extrovert)
    # Assuming 0=introvert, 1=extrovert after encoding
    scale_pos_weight = 0.72 / 0.28  # ≈ 2.57

    # Create XGBoost model with class balancing
    xgb_model = xgb.XGBClassifier(
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )

    # Create simple pipeline (no preprocessing needed since we already encoded)
    pipeline = Pipeline([
        ('classifier', xgb_model)
    ])

    # Stratified K-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }

    # Perform cross-validation
    cv_results = cross_validate(
        pipeline, X, y_encoded,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )

    # Fit final pipeline
    pipeline.fit(X, y_encoded)

    return pipeline, cv_results

def print_cv_results(cv_results):
    """Print cross-validation results"""
    print("Cross-Validation Results:")
    print("-" * 40)

    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        scores = cv_results[f'test_{metric}']
        print(f"{metric.upper():>10}: {scores.mean():.4f} ± {scores.std():.4f}")

# Run pipeline
pipeline, cv_results = create_xgboost_pipeline(train_df, target_col='Personality')

# Print results
print_cv_results(cv_results)



# Prepare the test data

X_test = test_df[numerical_features + categorical_features].copy()
# We need the fitted LabelEncoders from the training step
# Since they were not returned by the create_xgboost_pipeline function,
# we will quickly refit them on the training data to ensure consistency
le_stage = LabelEncoder()
le_drained = LabelEncoder()

# Fit on the training data's categorical columns
le_stage.fit(train_df['Stage_fear'])
le_drained.fit(train_df['Drained_after_socializing'])

# Transform the test data's categorical columns
X_test['Stage_fear'] = le_stage.transform(X_test['Stage_fear'])
X_test['Drained_after_socializing'] = le_drained.transform(X_test['Drained_after_socializing'])

# Make predictions on the test data using the trained pipeline
test_predictions_encoded = pipeline.predict(X_test)

# Decode the predictions back to original labels ('Introvert', 'Extrovert')
# We need the fitted LabelEncoder for the target variable from the training step
# Similar to the feature encoders, we will refit the target encoder
le_target = LabelEncoder()
le_target.fit(train_df['Personality'])

test_predictions = le_target.inverse_transform(test_predictions_encoded)

# Create the submission DataFrame
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'Personality': test_predictions
})

# Display the head of the submission DataFrame
print("\nSubmission DataFrame Head:")
display(submission_df.head())

# Save the submission DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully.")

