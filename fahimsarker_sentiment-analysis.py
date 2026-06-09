# Import necessary libraries
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")


# Load the training data
train_data = pd.read_csv('/kaggle/input/aiquest-bangla-sentiment-analysis-competition/train.csv')

# Text preprocessing function
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Apply preprocessing to the text column
train_data['text'] = train_data['text'].apply(preprocess_text)

# Encode the sentiment labels
label_encoder = LabelEncoder()
train_data['sentiment'] = label_encoder.fit_transform(train_data['sentiment'])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_data['text'], train_data['sentiment'], test_size=0.2, random_state=42)


# Use TF-IDF for feature extraction
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  # Use unigrams and bigrams
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)


# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_tfidf_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)


# Step 5: Train and Evaluate Multiple ML Models
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "Naive Bayes": MultinomialNB(),
    "XGBoost": XGBClassifier(random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

# Evaluate each model
results = {}
for name, model in models.items():
    model.fit(X_train_tfidf_resampled, y_train_resampled)
    y_val_pred = model.predict(X_val_tfidf)
    f1 = f1_score(y_val, y_val_pred, average='macro')
    results[name] = f1
    print(f"{name} Macro F1 Score: {f1:.4f}")

# Display results in a DataFrame
results_df = pd.DataFrame(list(results.items()), columns=["Model", "Macro F1 Score"])
print("\nModel Performance Summary:")
print(results_df.sort_values(by="Macro F1 Score", ascending=False))


# Step 6: Fine-Tune the Best Model (Naive Bayes)
# Define the parameter grid for Naive Bayes
param_grid_nb = {
    'alpha': [0.1, 0.5, 1.0, 2.0],  # Smoothing parameter
    'fit_prior': [True, False]  # Whether to learn class prior probabilities
}

# Initialize the Naive Bayes model
nb = MultinomialNB()

# Perform Grid Search
grid_search_nb = GridSearchCV(nb, param_grid_nb, cv=5, scoring='f1_macro', n_jobs=-1)
grid_search_nb.fit(X_train_tfidf_resampled, y_train_resampled)

# Best parameters and score
print("\nBest Parameters for Naive Bayes:", grid_search_nb.best_params_)
print("Best Macro F1 Score for Naive Bayes:", grid_search_nb.best_score_)

# Predict on the validation set with the best model
best_nb = grid_search_nb.best_estimator_
y_val_pred_nb = best_nb.predict(X_val_tfidf)
print("Validation Macro F1 Score (Naive Bayes):", f1_score(y_val, y_val_pred_nb, average='macro'))


# Load the test data
test_data = pd.read_csv('/kaggle/input/aiquest-bangla-sentiment-analysis-competition/train.csv')
test_data['text'] = test_data['text'].apply(preprocess_text)

X_test_tfidf = tfidf_vectorizer.transform(test_data['text'])

test_data['sentiment'] = best_nb.predict(X_test_tfidf)
test_data['sentiment'] = label_encoder.inverse_transform(test_data['sentiment'])
test_data[['id', 'sentiment']].to_csv('submission.csv', index=False)

print("\nPredictions saved to 'submission.csv'.")

