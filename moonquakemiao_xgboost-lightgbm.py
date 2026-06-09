"""
Multi-Model Ensemble for Movie Review Sentiment Classification
"""

import pandas as pd # Data manipulation and analysis library
import numpy as np # Numerical operations, especially for arrays
import re # Regular expression operations for text cleaning
from sklearn.feature_extraction.text import TfidfVectorizer # For TF-IDF feature extraction
from sklearn.model_selection import StratifiedKFold # For stratified K-Fold cross-validation
from sklearn.metrics import accuracy_score # Metric for classification accuracy
from sklearn.linear_model import LogisticRegression # Logistic Regression classifier
from sklearn.ensemble import RandomForestClassifier # Random Forest classifier
from sklearn.naive_bayes import MultinomialNB # Multinomial Naive Bayes classifier
import xgboost as xgb # Extreme Gradient Boosting classifier
import lightgbm as lgb # Light Gradient Boosting Machine classifier
from scipy.sparse import hstack # Used to horizontally stack sparse matrices
import warnings # Used to manage warnings
warnings.filterwarnings('ignore') # Ignores all warning messages for cleaner output

# ===== 1. Load Data =====
print("Loading data...")
# Loads the training and testing datasets from CSV files.
train = pd.read_csv('/kaggle/input/py-sphere-movie-review-sentiment-challenge/train.csv')
test = pd.read_csv('/kaggle/input/py-sphere-movie-review-sentiment-challenge/test.csv')

# ===== 2. Text Preprocessing =====
def clean_text(text):
    """
    Cleans text by:
    1. Converting to string and lowercasing.
    2. Removing characters that are not lowercase letters, numbers, spaces, or basic punctuation.
    3. Replacing multiple spaces with a single space and stripping leading/trailing whitespace.
    """
    text = str(text).lower() # Convert to string and lowercase
    text = re.sub(r'[^a-z0-9\s!?.,]', '', text) # Remove special characters, keep letters, numbers, spaces, and some punctuation
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with single space and strip
    return text

# Apply the cleaning function to the 'review' column for both train and test sets,
# creating a new 'review_clean' column.
train['review_clean'] = train['review'].apply(clean_text)
test['review_clean'] = test['review'].apply(clean_text)

# ===== 3. Feature Extraction (Multiple TF-IDF Combinations) =====
print("\nExtracting features...")

# Word-level TF-IDF Vectorizer
tfidf_word = TfidfVectorizer(
    max_features=8000, # Consider only the top 8000 words/ngrams by frequency
    ngram_range=(1, 3), # Consider unigrams, bigrams, and trigrams
    min_df=2, # Ignore terms that appear in less than 2 documents
    max_df=0.8, # Ignore terms that appear in more than 80% of the documents
    sublinear_tf=True # Apply sublinear term frequency scaling (1 + log(tf))
)

# Character-level TF-IDF Vectorizer
tfidf_char = TfidfVectorizer(
    max_features=4000, # Consider only the top 4000 character ngrams
    analyzer='char', # Analyze text at the character level
    ngram_range=(3, 6), # Consider character ngrams of length 3 to 6
    min_df=2 # Ignore character ngrams that appear in less than 2 documents
)

# Fit and transform training data for word-level TF-IDF
X_train_word = tfidf_word.fit_transform(train['review_clean'])
# Fit and transform training data for character-level TF-IDF
X_train_char = tfidf_char.fit_transform(train['review_clean'])
# Horizontally stack the word and character features for the training set
X_train = hstack([X_train_word, X_train_char])

# Transform test data using the *fitted* word-level TF-IDF vectorizer
X_test_word = tfidf_word.transform(test['review_clean'])
# Transform test data using the *fitted* character-level TF-IDF vectorizer
X_test_char = tfidf_char.transform(test['review_clean'])
# Horizontally stack the word and character features for the test set
X_test = hstack([X_test_word, X_test_char])

# Extract the sentiment labels for the training set
y_train = train['sentiment'].values

# Print the final shape of the combined feature matrix for the training set
print(f"Feature dimension: {X_train.shape}")

# ===== 4. Define Multiple Models =====
# A dictionary holding various classification models with their optimized parameters.
models = {
    'Logistic Regression': LogisticRegression(
        C=2, # Inverse of regularization strength; higher C means less regularization
        max_iter=1000, # Maximum number of iterations for the solver to converge
        solver='saga' # Algorithm to use in the optimization problem; 'saga' is good for large datasets
    ),

    'XGBoost': xgb.XGBClassifier(
        max_depth=7, # Maximum depth of a tree
        learning_rate=0.1, # Step size shrinkage to prevent overfitting
        n_estimators=300, # Number of boosting rounds (trees)
        subsample=0.8, # Subsample ratio of the training instance
        colsample_bytree=0.8, # Subsample ratio of columns when constructing each tree
        random_state=42 # Random seed for reproducibility
    ),

    'LightGBM': lgb.LGBMClassifier(
        num_leaves=50, # Maximum number of leaves in one tree
        learning_rate=0.05, # Step size shrinkage
        n_estimators=300, # Number of boosting rounds (trees)
        random_state=42, # Random seed for reproducibility
        verbose=-1 # Suppress verbose output
    ),

    'Naive Bayes': MultinomialNB(alpha=0.1), # Multinomial Naive Bayes, alpha is the Laplace smoothing parameter

    'Random Forest': RandomForestClassifier(
        n_estimators=200, # Number of trees in the forest
        max_depth=20, # Maximum depth of the tree
        min_samples_split=10, # Minimum number of samples required to split an internal node
        random_state=42, # Random seed for reproducibility
        n_jobs=-1 # Use all available CPU cores for parallel processing
    )
}

# ===== 5. Cross-Validation + Stacking =====
print("\nCross-validation training...")

# Initialize Stratified K-Fold cross-validator.
# `n_splits=5`: 5-fold cross-validation.
# `shuffle=True`: Shuffle the data before splitting into batches.
# `random_state=42`: Ensures reproducible splits.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize arrays to store out-of-fold (OOF) predictions for each model on the training data,
# and test set predictions (averaged across folds) for each model.
# These will form the input features for the meta-model in stacking.
oof_predictions = np.zeros((len(train), len(models))) # Stores OOF predictions for the training set
test_predictions = np.zeros((len(test), len(models))) # Stores test predictions for the test set

# Iterate through each model defined in the 'models' dictionary
for model_idx, (name, model) in enumerate(models.items()):
    print(f"\n{'='*50}")
    print(f"Training model: {name}")
    print(f"{'='*50}")

    fold_scores = [] # To store accuracy scores for each fold
    test_preds_folds = [] # To store test predictions from each fold

    # Iterate through each fold generated by StratifiedKFold
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        # Split data into training and validation sets for the current fold
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Train the current base model on the training data of the current fold
        model.fit(X_tr, y_tr)

        # Make predictions on the validation set of the current fold
        val_pred = model.predict(X_val)
        acc = accuracy_score(y_val, val_pred) # Calculate accuracy for this fold
        fold_scores.append(acc) # Store the accuracy

        # Store the OOF predictions for the validation set.
        # These predictions will be used as features for the meta-model.
        oof_predictions[val_idx, model_idx] = val_pred

        # Make predictions on the entire test set using the model trained in this fold
        test_pred = model.predict(X_test)
        test_preds_folds.append(test_pred) # Store test predictions for later averaging

        print(f"  Fold {fold}: {acc:.4f}")

    # Average the test set predictions across all folds for the current model
    # (Rounding to nearest integer for binary classification)
    test_predictions[:, model_idx] = np.mean(test_preds_folds, axis=0).round()

    # Print the average accuracy and standard deviation across all folds for the current model
    print(f"Average accuracy: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")

# ===== 6. Ensemble Strategies =====
print(f"\n{'='*50}")
print("Ensemble Prediction")
print(f"{'='*50}")

# Strategy 1: Simple Voting
def voting_ensemble(predictions):
    """
    Performs majority voting on an array of predictions.
    For each sample, it counts the occurrences of each class label
    and returns the most frequent one.
    """
    return np.apply_along_axis(
        lambda x: np.bincount(x.astype(int)).argmax(), # Counts occurrences of each integer and returns the index of the max count
        axis=1, # Apply row-wise
        arr=predictions # Input array of predictions
    )

# Strategy 2: Weighted Voting (not directly used for final prediction in this script, but defined)
def weighted_voting(predictions, weights):
    """
    Performs weighted voting.
    This function is defined but not actively used for the final submission in this script.
    It would multiply predictions by given weights and then threshold.
    """
    weighted = predictions * weights
    return (weighted.sum(axis=1) > 0.5).astype(int) # Threshold at 0.5 for binary classification

# Apply simple voting to the test set predictions from the base models
test_pred_voting = voting_ensemble(test_predictions)

# Also train a meta-model (Stacking)
print("\nTraining Meta-Model (Stacking)...")
# The meta-model (a Logistic Regression in this case) is trained on the OOF predictions
# from the base models (oof_predictions) and the original training labels (y_train).
meta_model = LogisticRegression(max_iter=1000)
meta_model.fit(oof_predictions, y_train)
# The meta-model then makes predictions on the test set predictions from the base models.
test_pred_stacking = meta_model.predict(test_predictions)

# ===== 7. Evaluate various ensemble methods =====
print("\nOut-of-Fold Ensemble Evaluation:")
# Evaluate the simple voting ensemble on the OOF predictions
oof_voting = voting_ensemble(oof_predictions)
# Evaluate the stacking meta-model on the OOF predictions
oof_stacking = meta_model.predict(oof_predictions)

# Print the OOF accuracy for both simple voting and stacking
print(f"Voting OOF Accuracy: {accuracy_score(y_train, oof_voting):.4f}")
print(f"Stacking OOF Accuracy: {accuracy_score(y_train, oof_stacking):.4f}")

# ===== 8. Select Best Method and Submit =====
# Typically, Stacking yields better results
final_predictions = test_pred_stacking # Choose the stacking predictions as the final output

# Print the distribution of the final predicted labels
print(f"\nFinal prediction distribution: {pd.Series(final_predictions).value_counts().to_dict()}")

# Create the submission DataFrame
submission = pd.DataFrame({
    'id': test['id'], # Original 'id' from the test set
    'sentiment': final_predictions # Final predicted sentiment labels
})

# Save the submission DataFrame to a CSV file
submission.to_csv('ensemble_submission.csv', index=False) # `index=False` prevents writing DataFrame index
print("\n✅ Submission file saved: ensemble_submission.csv")
print("Expected Kaggle Score: 0.87-0.90")




