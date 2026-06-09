import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from gensim.models import Word2Vec
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk


# Download NLTK data
nltk.data.path.append('data/nltk_data')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# Check environment if in Kaggle
extract_folder_path = '/kaggle/input/amazon-fine-food-reviews' if os.path.exists('/kaggle/input/amazon-fine-food-reviews') else './data'

# Load the review data
review_file_path = os.path.join(extract_folder_path, 'Reviews.csv')
review_data = pd.read_csv(review_file_path)

# Displaying the first few rows of the dataset
review_data.head()


# Statistical summary of the dataset
review_data.describe()


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset for sentiment analysis.
    
    - Take the first 10,000 rows
    - Keep only 'Text' and 'Score' columns
    - Convert 'Score' to binary labels: 1 if Score >= 4, else 0
    - Clean the 'Text' column: lowercase, remove punctuation, tokenize, remove stopwords, lemmatize
    - Return DataFrame with 'Cleaned_Text', 'Tokens', and 'Label'

    Args:
        df: Original DataFrame with reviews
    Returns:
        Preprocessed DataFrame
    """
    # Take first 10,000 rows and select columns
    df = df.head(10000)[['Text', 'Score']].copy()
    
    # Convert Score to binary labels
    df['Label'] = (df['Score'] >= 4).astype(int)
    
    # Prepare stopwords and lemmatizer
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    def clean_text(text: str) -> Tuple[str, List[str]]:
        """
        Clean and tokenize text.
        Args:
            text: Original text string
        Returns:
            cleaned_text: Cleaned text string
            tokens: List of tokens
        """
        # Lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords and lemmatize
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()]
        # Join back for TF-IDF
        cleaned_text = ' '.join(tokens)
        return cleaned_text, tokens
    
    # Apply cleaning
    df[['Cleaned_Text', 'Tokens']] = df['Text'].apply(lambda x: pd.Series(clean_text(x)))
    
    return df[['Cleaned_Text', 'Tokens', 'Label']]

# Apply preprocessing
cleaned_df = preprocess(review_data)
cleaned_df.head()


def vectorize_tfidf(texts: List[str]) -> np.ndarray:
    """
    Vectorize texts using TF-IDF.
    
    Args:
        texts: List of cleaned text strings
    
    Returns:
        Sparse matrix of TF-IDF features
    """
    vectorizer = TfidfVectorizer(max_features=5000)  # Limit features for efficiency
    return vectorizer.fit_transform(texts)

def vectorize_w2v(tokens_list: List[List[str]], vector_size=100) -> Tuple[np.ndarray, Word2Vec]:
    """
    Train Word2Vec and vectorize documents by averaging word vectors.
    
    Args:
        tokens_list: List of tokenized texts
        vector_size: Dimension of word vectors
    
    Returns:
        Numpy array of document vectors, trained Word2Vec model
    """
    # Train Word2Vec model
    model = Word2Vec(sentences=tokens_list, vector_size=vector_size, window=5, min_count=1, workers=4)
    
    def get_doc_vector(tokens: List[str]) -> np.ndarray:
        """
        Get the average Word2Vec vector for a document.
        Args:
            tokens: List of tokens in the document
        Returns:
            Averaged vector
        """
        vectors = [model.wv[word] for word in tokens if word in model.wv]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(vector_size)
    
    # Get document vectors
    document_vectors = np.array([get_doc_vector(tokens) for tokens in tokens_list])
    return document_vectors, model

# Extract features
feature_matrix_tfidf = vectorize_tfidf(cleaned_df['Cleaned_Text'])
feature_matrix_w2v, w2v_model = vectorize_w2v(cleaned_df['Tokens'])

# Labels
y = cleaned_df['Label']

print(f"TF-IDF shape: {feature_matrix_tfidf.shape}")
print(f"Word2Vec shape: {feature_matrix_w2v.shape}")
print(f"Labels shape: {y.shape}")


def train_model(feature_matrix: np.ndarray, labels: pd.Series) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier on the given features and labels.
    
    Args:
        feature_matrix: Feature matrix
        labels: Labels
    
    Returns:
        Trained model
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(feature_matrix, labels)
    return model

# Train models
model_tfidf = train_model(feature_matrix_tfidf, y)
model_w2v = train_model(feature_matrix_w2v, y)

print("Models trained successfully.")


def evaluate_model(model: RandomForestClassifier, feature_matrix: np.ndarray, labels: pd.Series, cv=4) -> (float, float):
    """
    Evaluate model using 4-fold cross-validation.
    
    Args:
        model: Trained model
        feature_matrix: Feature matrix
        labels: Labels
        cv: Number of folds
    
    Returns:
        Mean accuracy and standard deviation
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    accuracy_scores = cross_val_score(model, feature_matrix, labels, cv=kf, scoring='accuracy')
    return accuracy_scores.mean(), accuracy_scores.std()

# Evaluate models
mean_acc_tfidf, std_acc_tfidf = evaluate_model(model_tfidf, feature_matrix_tfidf, y)
mean_acc_w2v, std_acc_w2v = evaluate_model(model_w2v, feature_matrix_w2v, y)

# Print results
print(f"TF-IDF: Mean Accuracy = {mean_acc_tfidf:.4f}, Std Dev = {std_acc_tfidf:.4f}")
print(f"Word2Vec: Mean Accuracy = {mean_acc_w2v:.4f}, Std Dev = {std_acc_w2v:.4f}")

# Create comparison table
import pandas as pd
results_df = pd.DataFrame({
    'Model': ['TF-IDF', 'Word2Vec'],
    'Mean Accuracy': [mean_acc_tfidf, mean_acc_w2v],
    'Std Dev': [std_acc_tfidf, std_acc_w2v]
})
results_df


# Bar chart comparing accuracies
models = results_df['Model']
accuracies = results_df['Mean Accuracy']

plt.figure(figsize=(8, 5))
plt.bar(models, accuracies, color=['skyblue', 'lightgreen'])
plt.ylabel('Mean Accuracy')
plt.title('Comparison of Sentiment Classification Models')
plt.ylim(0.75, 0.85)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.001, f"{v:.4f}", ha='center', fontweight='bold')
plt.show()


# Load test data
extract_folder_path = '/kaggle/input/hw2-sentiment-analysis' if os.path.exists('/kaggle/input/hw2-sentiment-analysis') else './data'
test_file_path = os.path.join(extract_folder_path, 'test.csv')
test_data = pd.read_csv(test_file_path)
print(f"Test data shape: {test_data.shape}")
test_data.head()

# Preprocess test data (only clean text, no labels)
def preprocess_test(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the test dataset.

    Args:
        df: Original test DataFrame with reviews
    Returns:
        Preprocessed DataFrame with 'Cleaned_Text'
    """
    df = df.copy()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    def clean_text(text: str) -> Tuple[str, List[str]]:
        """
        Clean and tokenize text.
        Args:
            text: Original text string
        Returns:
            cleaned_text: Cleaned text string
            tokens: List of tokens
        """
        # Lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords and lemmatize
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()]
        # Join back for TF-IDF
        cleaned_text = ' '.join(tokens)
        return cleaned_text, tokens
    
    # Apply cleaning
    df[['Cleaned_Text', 'Tokens']] = df['Text'].apply(lambda x: pd.Series(clean_text(x)))
    return df

test_cleaned = preprocess_test(test_data)


# Vectorize test data using the same TF-IDF vectorizer
# We need to fit on training data, but since we used fit_transform, we need to refit or use the same vectorizer
# For simplicity, refit on training cleaned text
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_vectorizer.fit(cleaned_df['Cleaned_Text'])
feature_matrix_test_tfidf = tfidf_vectorizer.transform(test_cleaned['Cleaned_Text'])

# Predict using the better model (TF-IDF)
predictions = model_tfidf.predict(feature_matrix_test_tfidf)


# Create submission DataFrame (1-based index)
submission = pd.DataFrame({
    'ID': np.arange(1, len(predictions) + 1),
    'Score': predictions
})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")
submission.head()


# Vectorize test data using Word2Vec
def vectorize_test_w2v(tokens_list: List[List[str]], model: Word2Vec) -> np.ndarray:
    """
    Vectorize test documents using a pre-trained Word2Vec model.
    
    Args:
        tokens_list: List of tokenized texts
        model: Trained Word2Vec model
    
    Returns:
        Numpy array of document vectors
    """
    def get_doc_vector(tokens: List[str]) -> np.ndarray:
        """
        Get the average Word2Vec vector for a document.
        Args:
            tokens: List of tokens in the document
        Returns:
            Averaged vector
        """
        vectors = [model.wv[word] for word in tokens if word in model.wv]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(model.vector_size)
    
    return np.array([get_doc_vector(tokens) for tokens in tokens_list])

feature_matrix_test_w2v = vectorize_test_w2v(test_cleaned['Tokens'], w2v_model)

# Predict using Word2Vec model
predictions_w2v = model_w2v.predict(feature_matrix_test_w2v)


# Create Word2Vec submission DataFrame (1-based index)
submission_w2v = pd.DataFrame({
    'ID': np.arange(1, len(predictions_w2v) + 1),
    'Score': predictions_w2v
})

# Save to CSV
submission_w2v.to_csv('submission_w2v.csv', index=False)
print("Word2Vec submission file created: submission_w2v.csv")

# Simple comparison: number of positive predictions
print(f"TF-IDF positive predictions: {sum(predictions)}")
print(f"Word2Vec positive predictions: {sum(predictions_w2v)}")

submission_w2v.head()

