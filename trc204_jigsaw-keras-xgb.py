import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import string # Import the string module

# Import common scikit-learn modules
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# Set some display options for pandas DataFrames
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Set the style for matplotlib plots
plt.style.use('seaborn-v0_8-darkgrid')

# Set the default figure size for plots
plt.rcParams['figure.figsize'] = (10, 6)

# Set the default font size for plots
plt.rcParams['font.size'] = 12

# Set the default color palette for seaborn plots
sns.set_palette('viridis')


dftrain = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
dftest = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')




def clean_text(text):
    """Removes punctuation and converts text to lowercase."""
    if isinstance(text, str):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text
    return text

# Apply the cleaning function to the relevant columns
dftrain['body_cleaned'] = dftrain['body'].apply(clean_text)
dftest['body_cleaned'] = dftest['body'].apply(clean_text)

# Define X and y
X = dftrain['body_cleaned']
y = dftrain['rule_violation']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

display(dftrain.head())
display(dftest.head())


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print("TF-IDF features created.")
print("Shape of X_train_tfidf:", X_train_tfidf.shape)
print("Shape of X_test_tfidf:", X_test_tfidf.shape)


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X)

# Transform the test data
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print("TF-IDF features created.")
print("Shape of X_train_tfidf:", X_train_tfidf.shape)
print("Shape of X_test_tfidf:", X_test_tfidf.shape)


# List the column names
print("Columns in dftrain:")
print(dftrain.columns.tolist())

# Based on the column names and the task of identifying rule violations,
# the most relevant columns to focus on are likely:
# - 'body': This contains the main text content which is the primary source of information about potential rule violations.
# - 'rule': This column explicitly states the rule that was violated, which can be useful for understanding the context of the violation.
# - 'subreddit': The subreddit where the post or comment was made might also provide context relevant to rule violations, as rules can vary between subreddits.
# - 'rule_violation': This is the target variable we are trying to predict.

print("\nSuggested columns to focus on for identifying rule violations:")
print("- 'body': Contains the main text content.")
print("- 'rule': States the violated rule (useful for context).")
print("- 'subreddit': Provides context as rules can vary.")
print("- 'rule_violation': The target variable.")


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression # Placeholder estimator

# Select features and target
features = ['body_cleaned', 'rule', 'subreddit']
target = 'rule_violation'

X = dftrain[features]
y = dftrain[target]

# Create a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(max_features=5000, stop_words='english'), 'body_cleaned'),
        ('categorical', OneHotEncoder(handle_unknown='ignore'), ['rule', 'subreddit'])
    ],
    remainder='passthrough' # Keep other columns if any, although not expected in this case
)

# Create a pipeline with the preprocessor and a placeholder estimator
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LogisticRegression())]) # Placeholder

# Fit the preprocessor to prepare the data
X_processed = preprocessor.fit_transform(X)

print("Features and target selected.")
print("ColumnTransformer created.")
print("Pipeline created with preprocessor.")
print("Preprocessor fitted on the data.")
print("Shape of processed data:", X_processed.shape)


from sklearn.model_selection import train_test_split

# Split the preprocessed data into training and testing sets
X_train_processed, X_test_processed, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# Print the shapes of the resulting sets
print("Shape of X_train_processed:", X_train_processed.shape)
print("Shape of X_test_processed:", X_test_processed.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


from xgboost import XGBClassifier

# Initialize XGBClassifier
xgb_model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)

# Train the model
xgb_model.fit(X_train_processed, y_train)

print("XGBoost Classifier initialized and trained.")


# Use the trained xgb_model to make predictions on the test data
y_pred = xgb_model.predict(X_test_processed)

# Calculate the accuracy, precision, recall, and F1-score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Calculate the ROC AUC score
y_pred_proba = xgb_model.predict_proba(X_test_processed)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print the calculated metrics
print("Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")


# Apply the cleaning function to the 'body' column of dftest
dftest['body_cleaned'] = dftest['body'].apply(clean_text)

# Select the features from dftest that were used for training
X_test_final = dftest[features] # 'features' variable was defined earlier

# Apply the preprocessor to the dftest data
# The preprocessor was fitted on the training data in cell 66df3614
X_test_processed_final = preprocessor.transform(X_test_final)

# Make predictions on the processed test data
test_predictions = xgb_model.predict(X_test_processed_final)

# Create a submission DataFrame
submission_df = pd.DataFrame({'row_id': dftest['row_id'], 'rule_violation': test_predictions})

# Save the submission DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)

print("Predictions made on dftest and saved to submission.csv")

