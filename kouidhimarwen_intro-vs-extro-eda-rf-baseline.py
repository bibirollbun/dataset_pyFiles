# ğŸ“¦ Install dependencies
# We use `ydata-profiling` (formerly `pandas-profiling`) to automatically generate a comprehensive EDA report.

!pip install -U ydata-profiling[notebook] > /dev/null 2>&1


# ğŸš€ Core Python Libraries
import os                 # Interact with the operating system
import random             # Generate reproducible random numbers

# ğŸŒ� Kaggle & Data Access
import kagglehub          # Access pretrained models or datasets via KaggleHub

# ğŸ”¢ Numerical & Data Analysis
import numpy as np        # Numerical computations
import pandas as pd       # Data manipulation and analysis
from ydata_profiling import ProfileReport  # Automated EDA reports

# ğŸ“Š Visualization
import matplotlib.pyplot as plt            # Basic plotting

# ğŸ”� Machine Learning Utilities
from sklearn.model_selection import train_test_split  # Split data for training and validation
from sklearn.preprocessing import LabelEncoder        # Encode categorical variables

# ğŸ¤– Machine Learning Models & Evaluation
from sklearn.ensemble import RandomForestClassifier   # Ensemble model for classification
from sklearn.metrics import accuracy_score, classification_report  # Evaluation metrics



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


# Download competition dataset
DATA_DIR = kagglehub.competition_download('playground-series-s5e7')

# Load training dataset
train_path = os.path.join(DATA_DIR, 'train.csv')
train_df = pd.read_csv(train_path)

# Verify load
print("Train shape:", train_df.shape)


# Display first few rows
train_df.head()


# ğŸ“� Generate an automated EDA report using ydata-profiling
# Pass the training dataframe and set a custom title for the report
profile = ProfileReport(train_df, title="ğŸ“Š Training Data Profiling Report")

# ğŸ‘€ Display the profiling report inline
profile



# ğŸ—‘ï¸� Drop 'id' column (irrelevant for modeling)
train_df_cleaned = train_df.drop(columns=["id"])

# ğŸ”§ Impute missing values

# ğŸ”¢ Median imputation for numeric columns
numeric_cols = [
    'Time_spent_Alone',
    'Social_event_attendance',
    'Going_outside',
    'Friends_circle_size',
    'Post_frequency'
]
for col in numeric_cols:
    train_df_cleaned[col] = train_df_cleaned[col].fillna(train_df_cleaned[col].median())

# âœ… Mode imputation for boolean columns and encoding Yes/No â†’ 1/0
bool_cols = ['Stage_fear', 'Drained_after_socializing']
for col in bool_cols:
    train_df_cleaned[col] = train_df_cleaned[col].fillna(train_df_cleaned[col].mode()[0])
    train_df_cleaned[col] = train_df_cleaned[col].map({'Yes': 1, 'No': 0})

# ğŸ�·ï¸� Encode target variable 'Personality' (0 = Extrovert, 1 = Introvert)
train_df_cleaned['Personality'] = LabelEncoder().fit_transform(train_df_cleaned['Personality'])

# ğŸª“ Separate features and target
X = train_df_cleaned.drop('Personality', axis=1)
y = train_df_cleaned['Personality']

# ğŸ”€ Train-test split (80% train / 20% test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ğŸ“� Print data shapes
print("âœ… Shape of training data:", X_train.shape)
print("âœ… Shape of test data:", X_test.shape)



train_df_cleaned.isna().sum()



# ğŸŒ² Fit a baseline Random Forest model

# ğŸ¤– Initialize a Random Forest classifier with a fixed random state for reproducibility
clf = RandomForestClassifier(random_state=42)

# ğŸ�‹ï¸� Train the classifier on the training set
clf.fit(X_train, y_train)



# Predict on test set
y_pred = clf.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("Baseline Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Extrovert', 'Introvert']))



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
    median_val = train_df_cleaned[col].median()
    test_df_cleaned[col] = test_df_cleaned[col].fillna(median_val)

# âœ… Standardize, fill, and encode boolean columns
for col in bool_cols:
    mode_val = train_df_cleaned[col].mode()[0]

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
test_preds = clf.predict(test_features)

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

