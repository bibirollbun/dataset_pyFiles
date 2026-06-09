pip install sentence-transformers


# --- 1. Import Necessary Libraries ---
# pandas is used for data manipulation and loading CSV files.
# BaseEstimator and TransformerMixin are base classes for creating custom scikit-learn components.
# SentenceTransformer is the library for using state-of-the-art language models like MiniLM.
# Pipeline and ColumnTransformer are for building organized, robust machine learning workflows.
# RandomForestClassifier is the classification model we will use.
import numpy as np 
import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


train_df_path = "../input/jigsaw-agile-community-rules/train.csv"
train_df = pd.read_csv(train_df_path, index_col='row_id')
train_df.head()


test_df_path = "../input/jigsaw-agile-community-rules/test.csv"
test_df = pd.read_csv(test_df_path, index_col='row_id')
test_df.head()


# --- 2. Create a Custom Scikit-Learn Wrapper for the Language Model ---
# This class makes the SentenceTransformer model compatible with a scikit-learn Pipeline.
# Scikit-learn's tools expect every step to have .fit() and .transform() methods.
class SentenceTransformerEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.model.encode(X.tolist())


# --- 4. Feature Engineering: Create a Rich Context Column ---
# Define all columns that contain text to be used as context for the model.
text_cols = ['body', 'rule', 'subreddit', 'positive_example_1', 'positive_example_2', 'negative_example_1', 'negative_example_2']


# --- Process Training Data ---
# Fill any potential missing text values with an empty string
for col in text_cols:
    train_df[col] = train_df[col].fillna('')

# Create a single 'combined_text' column for the training set
train_df['combined_text'] = (
    "SUBREDDIT: " + train_df['subreddit'] +
    " | RULE: " + train_df['rule'] +
    " | BODY: " + train_df['body'] +
    " | GOOD EXAMPLE 1: " + train_df['positive_example_1'] +
    " | GOOD EXAMPLE 2: " + train_df['positive_example_2'] +
    " | BAD EXAMPLE 1: " + train_df['negative_example_1'] +
    " | BAD EXAMPLE 2: " + train_df['negative_example_2']
)

# --- Process Testing Data (Apply the same logic) ---
for col in text_cols:
    test_df[col] = test_df[col].fillna('')

# Create the 'combined_text' column for the testing set
test_df['combined_text'] = (
    "SUBREDDIT: " + test_df['subreddit'] +
    " | RULE: " + test_df['rule'] +
    " | BODY: " + test_df['body'] +
    " | GOOD EXAMPLE 1: " + test_df['positive_example_1'] +
    " | GOOD EXAMPLE 2: " + test_df['positive_example_2'] +
    " | BAD EXAMPLE 1: " + test_df['negative_example_1'] +
    " | BAD EXAMPLE 2: " + test_df['negative_example_2']
)



# --- 5. Define Training and Testing Sets ---
# X represents the features (the input to the model).
# y represents the target variable (what the model is trying to predict).
# Define your training sets
X_train = train_df[['combined_text']]
y_train = train_df['rule_violation']

# The test set only contains features; the target is what we will predict.
# Define your testing set (features only)
X_test = test_df[['combined_text']]


# --- 6. Build the Full Machine Learning Pipeline ---
# The ColumnTransformer applies our custom SentenceTransformerEmbedder to the 'combined_text' column.
preprocessor = ColumnTransformer(
    transformers=[('text_embedder', SentenceTransformerEmbedder(), 'combined_text')],
    remainder='passthrough'
)

# The Pipeline chains the preprocessing step and the final classification model together.
# This ensures that data flows cleanly from raw text to a final prediction.
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])


# --- 7. Train the Model ---
# The .fit() method trains the entire pipeline on the training data.
# It first preprocesses the text into embeddings and then trains the RandomForestClassifier on them.
print("Training the full context model...")
full_pipeline.fit(X_train, y_train)


# --- 8. Make Predictions on the Unseen Test Set ---
# The .predict() method applies the trained pipeline to the new, unseen test data.
print("\nMaking predictions on the test set...")
test_predictions = full_pipeline.predict(X_test)


# Display the results
print("\nPredictions generated successfully!")
print("First 5 predictions:", test_predictions[:5])

