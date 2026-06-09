import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import re
import nltk
from nltk.corpus import stopwords
from xgboost import XGBRegressor
from sentence_transformers import SentenceTransformer

# Install sentence-transformers library if you haven't already
# !pip install -q sentence-transformers

# Download NLTK data if not already present
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

def preprocess_text(text):
    """
    Cleans and preprocesses the input text by lowercasing, removing special characters,
    and filtering out stopwords.
    """
    # Handle empty or None input
    if not isinstance(text, str) or text.strip() == "":
        return ""
    
    text = text.lower()  # Lowercasing
    text = re.sub(r"\r\n", " ", text)  # Replace \r\n with space
    text = re.sub(r"[^a-z\s]", "", text)  # Remove punctuation and numbers
    tokens = text.split()  # Tokenization
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]  # Stop word removal
    return " ".join(tokens)

train_df = pd.read_csv("/kaggle/input/internal-selection-bdc-2025-its/df_train.csv")
test_df = pd.read_csv("/kaggle/input/internal-selection-bdc-2025-its/df_test.csv")
# Preprocess essay text
train_df["processed_essay"] = train_df["essay"].apply(preprocess_text)
test_df["processed_essay"] = test_df["essay"].apply(preprocess_text)

# --- Sentence Transformer Vectorization ---
print("Loading Sentence Transformer model...")
# Using a popular, lightweight model. You can choose others like 'all-mpnet-base-v2' for potentially higher accuracy.
st_model = SentenceTransformer('all-mpnet-base-v2') 

print("Encoding training essays...")
train_embeddings = st_model.encode(train_df["processed_essay"].tolist(), show_progress_bar=True)
print("Encoding test essays...")
test_embeddings = st_model.encode(test_df["processed_essay"].tolist(), show_progress_bar=True)

# Convert embeddings to DataFrame
X_train_embeddings_df = pd.DataFrame(train_embeddings)
X_test_embeddings_df = pd.DataFrame(test_embeddings)

# Add essay length as a feature
train_df["essay_length"] = train_df["essay"].apply(len)
test_df["essay_length"] = test_df["essay"].apply(len)

# Combine features (Sentence Embeddings + Essay Length)
X_train = pd.concat([X_train_embeddings_df, train_df[["essay_length"]].reset_index(drop=True)], axis=1)
X_test = pd.concat([X_test_embeddings_df, test_df[["essay_length"]].reset_index(drop=True)], axis=1)

# XGBoost requires feature names to be strings
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)


# Define target columns
target_columns = ["task_achievement", "coherence_and_cohesion", "lexical_resource", "grammatical_range"]

# Define hyperparameters for each target column (these were likely tuned for TF-IDF, may need re-tuning)
hyperparameters = {
    "task_achievement": {
        'n_estimators': 218,
        'max_depth': 10,
        'learning_rate': 0.29868097549890427,
        'subsample': 0.9849705118931866,
        'colsample_bytree': 0.7971024950432997
    },
    "coherence_and_cohesion": {
        'n_estimators': 258,
        'max_depth': 9,
        'learning_rate': 0.29954325131884296,
        'subsample': 0.9973298665104741,
        'colsample_bytree': 0.7588513290921837
    },
    "lexical_resource": {
        'n_estimators': 218,
        'max_depth': 10,
        'learning_rate': 0.2615527107935345,
        'subsample': 0.9847920986369385,
        'colsample_bytree': 0.7660934250748141
    },
    "grammatical_range": {
        'n_estimators': 282,
        'max_depth': 9,
        'learning_rate': 0.29944689993813284,
        'subsample': 0.9746520209478066,
        'colsample_bytree': 0.8968903286556082
    }
}

# Dictionary to store trained models
models = {}
predictions = {}

# Train models for each target column with specified hyperparameters
for col in target_columns:
    print(f"\nTraining model for {col}...")
    
    # Create a copy of train_df and drop rows with missing values for the current target
    temp_train_df = train_df.dropna(subset=[col]).copy()
    
    # Align X_train with the rows remaining after dropping NaNs
    X_train_temp = X_train.loc[temp_train_df.index].reset_index(drop=True)
    y_train_temp = temp_train_df[col].reset_index(drop=True)
    
    # Train model with specified hyperparameters
    model = XGBRegressor(**hyperparameters[col], random_state=42, verbosity=0)
    model.fit(X_train_temp, y_train_temp)
    models[col] = model
    
    # Generate predictions for the test set
    predictions[col] = model.predict(X_test)

# Create submission DataFrame
submission_df = pd.DataFrame({
    "ID": range(1, len(X_test) + 1),
    "task_achievement": predictions["task_achievement"],
    "coherence_and_cohesion": predictions["coherence_and_cohesion"],
    "lexical_resource": predictions["lexical_resource"],
    "grammatical_range": predictions["grammatical_range"],
})

# Save submission file
submission_df.to_csv("submission_sentence_transformer_mpnet_base2.csv", index=False)
print("\nSubmission file created successfully at submission_sentence_transformer.csv")


