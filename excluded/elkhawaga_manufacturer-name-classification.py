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


train_data = pd.read_csv("/kaggle/input/manufacturer-name-clustering/train.csv")
test_data = pd.read_csv("/kaggle/input/manufacturer-name-clustering/test.csv")


train_data.head()


test_data.head()


len(train_data)


import re

# RE-REVISED: Define the list of legal suffixes
LEGAL_FORMS = [
    # Complex German/EU Suffixes (most complex first)
    r'ag\s+and\s+co\s+kg', r'gmbh\s+and\s+co\s+kg', 
    r'ag\s+co\s+kg', r'gmbh\s+co\s*kg', 
    
    # Common single-word legal forms
    r'ag', r'gmbh', r'ug', r'se', r'ohg', r'kg', r'gbr', 
    r'inc', r'ltd', r'corp', r'co', r'llc', r'sa', r'spa', r'bv', r'nv', r'plc',
    
    # NEW: Single-letter forms with spaces/dots that survive punctuation removal
    r's\s+a', r's\s+l', r'l\s+t\s+d', r'i\s+n\s+c', r'a\s+g',
]

# Look for these forms at the end of the string, with optional spaces/punctuation before them.
SUFFIX_PATTERN = r'(?:' + '|'.join(LEGAL_FORMS) + r')\s*$'




def clean_manufacturer_name_final(name):
    # 1. Convert to lowercase
    name = name.lower()
    
    # 2. Normalize punctuation and spacing aggressively
    # This turns "Nestlé S.A." into "nestlé s a" 
    name = re.sub(r'[^\w\s]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip() 
    
    # 3. Use regex to remove legal suffixes repeatedly
    # The loop is essential to catch chained suffixes
    for _ in range(5): 
        # Look for the pattern at the very end of the string ($)
        name = re.sub(SUFFIX_PATTERN, '', name, flags=re.IGNORECASE).strip()
    
    # 4. Final Cleanup (remove remnants of the ampersand conversion)
    name = re.sub(r'\s+and\s+co$', '', name).strip()
    name = re.sub(r'\s+and\s*company$', '', name).strip()
    
    return name


# --- Final Check ---
final_tests = [
    'Siemens AG & Co. KG',             # Complex German
    'Apple Inc.',                      # Simple English
    'Bayerische Motoren Werke AG',     # AG at end
    'Daimler Truck AG',                # AG at end
    'BASF SE',                         # SE at end
    'General Electric Co.',            # Co. at end
    'Nestlé S.A.',                     # S.A.
    'Continental AG & Co KG'           # AG & Co KG combined
]

print("--- FINAL CLEANING TEST ---")
for original_name in final_tests:
    cleaned_name = clean_manufacturer_name_final(original_name)
    print(f"Original: {original_name:<30} | Cleaned: {cleaned_name}")




# --- Final Final Check ---
final_tests = [
    'Siemens AG & Co. KG',             # Complex German
    'Apple Inc.',                      # Simple English
    'Bayerische Motoren Werke AG',     # AG at end
    'Daimler Truck AG',                # AG at end
    'BASF SE',                         # SE at end
    'General Electric Co.',           # Co. at end
    'Nestlé S.A.',                     # S.A.
    'Continental AG & Co KG',          # AG & Co KG combined
    'Daimler-Benz AG',                 # Hyphen
]

print("--- FINAL CLEANING TEST (Goal: Remove ALL Legal Forms) ---")
for original_name in final_tests:
    cleaned_name = clean_manufacturer_name_final(original_name)
    print(f"Original: {original_name:<30} | Cleaned: {cleaned_name}")


# Assuming df_train and df_test are loaded
# df_train = pd.read_csv('train.csv')

train_data['NAME_CLEANED'] = train_data['WORD'].apply(clean_manufacturer_name_final)
test_data['NAME_CLEANED'] = test_data['WORD'].apply(clean_manufacturer_name_final)



train_data.head()


import gensim.downloader as api
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- 0. Load the Pre-Trained Word Embeddings ---
# This is a one-time download
# NOTE: If you run this in a Kaggle notebook, the model might be pre-loaded,
#       but run it if you are in your local environment.
try:
    word_vectors = api.load("glove-wiki-gigaword-50")
    print("GloVe model loaded successfully.")
except ValueError:
    print("GloVe model not found. Ensure you are connected to the internet to download it.")



# --- 1. Define the Vectorization Function (Averaging Embeddings) ---

def get_sentence_vector(sentence, model):
    """Calculates a document vector by averaging word embeddings."""
    # Ensure the input is a string
    sentence = str(sentence)
    
    # Get vectors for words found in the model's vocabulary
    vectors = [model[word] for word in sentence.split() if word in model.key_to_index]
    
    # If no valid words were found (e.g., empty string after cleaning), return a zero vector
    if not vectors:
        return np.zeros(model.vector_size)
        
    # Average the vectors along the 0-axis (row-wise)
    return np.mean(vectors, axis=0)


# --- 2. Create Feature Matrices (X) and Target (Y) ---

# Assuming 'df_train' and 'df_test' are your DataFrames 
# AND both have a 'NAME_CLEANED' column from your previous successful step.

# a) Training Data (X and Y)
X_train_embeddings = np.array([
    get_sentence_vector(name, word_vectors) for name in train_data['NAME_CLEANED']
])
y_train = train_data['TARGET'].values

print(f"\nX_train Shape: {X_train_embeddings.shape}")
print(f"y_train Shape: {y_train.shape}")


# b) Test Data (X for prediction)
X_test_embeddings = np.array([
    get_sentence_vector(name, word_vectors) for name in test_data['NAME_CLEANED']
])
print(f"X_test Shape: {X_test_embeddings.shape}")


# --- 3. Train the Random Forest Model ---

# Split training data for local validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_embeddings, y_train, test_size=0.2, random_state=42
)

# Initialize and Train the Model
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
print("\nStarting Model Training...")
model.fit(X_tr, y_tr)
print("Model Training Complete.")





import numpy as np

# Assuming X_train_embeddings is the numpy array created by your vectorization step:
# X_train_embeddings = np.array([get_sentence_vector(name, word_vectors) for name in df_train['NAME_CLEANED']])

# --- Check the Training Data ---

# 1. Calculate the magnitude (length) of each vector. 
#    A vector of all zeros will have a magnitude of 0.
#    We use np.linalg.norm(axis=1) to compute the Euclidean distance (magnitude) for each row.
vector_magnitudes = np.linalg.norm(X_train_embeddings, axis=1)

# 2. Count how many vectors have a magnitude greater than zero.
valid_embedding_count = np.sum(vector_magnitudes > 1e-6) # Use a small tolerance instead of exact zero

# 3. Calculate the total number of rows (samples)
total_rows = X_train_embeddings.shape[0]

# 4. Print the results
print(f"--- Embedding Validation Check (Training Data) ---")
print(f"Total rows (manufacturer names): {total_rows}")
print(f"Rows with a valid (non-zero) embedding: {valid_embedding_count}")
print(f"Rows with zero embedding (Out-of-Vocabulary): {total_rows - valid_embedding_count}")

# 5. Percentage Check
valid_percentage = (valid_embedding_count / total_rows) * 100
print(f"Percentage of valid embeddings: {valid_percentage:.2f}%")

# You should aim for a high percentage (e.g., >95%). 
# If the number is low, your model is essentially classifying based on zero vectors, which is bad.


train_data['WORD'].head(10)


from sklearn.feature_extraction.text import TfidfVectorizer

# --- 1. Vectorize the Names ---

# a) Initialize TF-IDF Vectorizer
# max_features is set to limit the vocabulary size (optional, but good practice)
vectorizer = TfidfVectorizer(
    analyzer='word', 
    token_pattern=r'\S+', # Treats sequences of non-whitespace characters as a single token (good for manufacturer names)
    ngram_range=(1, 2),  # Use single words (unigrams) and two-word phrases (bigrams)
    max_features=5000  
)

# b) Fit on Training Data (Learn the vocabulary from the training set)
X_train_tfidf = vectorizer.fit_transform(train_data['NAME_CLEANED'])

# c) Transform the Training and Test Data
X_test_tfidf = vectorizer.transform(test_data['NAME_CLEANED'])

# Define the target variable (Y)
y_train = train_data['TARGET'].values

print(f"\nVocabulary Size (Number of Features): {len(vectorizer.vocabulary_)}")
print(f"X_train_tfidf Shape: {X_train_tfidf.shape}")
print(f"X_test_tfidf Shape: {X_test_tfidf.shape}")

# NOTE: The first dimension (rows) is the total number of samples, and the 
# second dimension (columns) is the number of features (tokens).


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- 2. Train the Random Forest Model ---

# a) Split Training Data for Local Validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_tfidf, y_train, test_size=0.2, random_state=42
)

# b) Initialize and Train the Model
model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, max_depth=30)
print("\nStarting Model Training (TF-IDF)...")
model.fit(X_tr, y_tr)
print("Model Training Complete.")

# c) Evaluate Performance
val_predictions = model.predict(X_val)
accuracy = accuracy_score(y_val, val_predictions)

print(f"\nValidation Accuracy (TF-IDF): {accuracy:.4f}")
print("\nClassification Report (TF-IDF):\n", classification_report(y_val, val_predictions))




# Re-initialize and Train the Model with class_weight='balanced'
model_balanced = RandomForestClassifier(
    n_estimators=300, 
    random_state=42, 
    n_jobs=-1, 
    max_depth=30,
    
    # NEW PARAMETER: Tells the model to weight smaller classes more heavily
    class_weight='balanced' 
)

print("\nStarting Model Training with Class Weight Balancing...")
model_balanced.fit(X_tr, y_tr)

# Evaluate the new model
val_predictions_balanced = model_balanced.predict(X_val)
print(f"\nAccuracy (Balanced): {accuracy_score(y_val, val_predictions_balanced):.4f}")
print("\nClassification Report (Balanced):\n", classification_report(y_val, val_predictions_balanced))


# Predict the cluster number for the test set
final_predictions = model_balanced.predict(X_test_tfidf)

# Create and Save the Submission File ---

# Create the submission DataFrame
submission_df = pd.DataFrame({
    'ID': test_data['ID'],
    'TARGET': final_predictions
})

# Save to CSV for submission
submission_filename = 'submission.csv'
submission_df.to_csv(submission_filename, index=False)

print(f"\nSubmission file '{submission_filename}' created successfully.")
print("This file is ready to be uploaded to the Kaggle competition page!")




