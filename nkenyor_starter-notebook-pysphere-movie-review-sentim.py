


# # Py-Sphere Movie Review Sentiment Challenge

# Welcome to the **Py-Sphere Movie Review Sentiment Challenge**. In this notebook, you'll find step-by-step guidanceâ€”from loading and exploring data to building a baseline model and preparing your submission.

# ---

# ## 1. Setup & Imports
# ```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report



# Adjust paths if needed
train = pd.read_csv('/kaggle/input/py-sphere-movie-review-sentiment-challenge/train.csv')
test = pd.read_csv('/kaggle/input/py-sphere-movie-review-sentiment-challenge/test.csv')
sample_submission = pd.read_csv('/kaggle/input/py-sphere-movie-review-sentiment-challenge/sample_submission.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)



# 3. Explore & Visualize

# Sample display
display(train.head(), test.head())

# Distribution of sentiment labels
sns.countplot(x='sentiment', data=train)
plt.title('Sentiment Label Distribution')
plt.show()



# 4. Basic Preprocessing

def preprocess_text(text):
    return text.lower()

train['clean_review'] = train['review'].apply(preprocess_text)
test['clean_review'] = test['review'].apply(preprocess_text)



# 5. Trainâ€“Validation Split

X_train, X_valid, y_train, y_valid = train_test_split(
    train['clean_review'], train['sentiment'], test_size=0.5, random_state=42
)



# 6. Feature Engineering: TF-IDF

tfidf = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2))
tfidf.fit(X_train)

X_train_tfidf = tfidf.transform(X_train)
X_valid_tfidf = tfidf.transform(X_valid)
X_test_tfidf = tfidf.transform(test['clean_review'])



# 7. Baseline Model: Logistic Regression

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_valid_tfidf)
print("Validation Accuracy:", accuracy_score(y_valid, y_pred))
print("Classification Report:\n", classification_report(y_valid, y_pred))



# 8. Predict on Test Set (fixed)
test_preds = model.predict(X_test_tfidf)

# Build submission from test file directly
submission = pd.DataFrame({
    "id": test["id"],  # Use IDs from the actual test file
    "sentiment": test_preds.astype(int)
})




# Save
submission.to_csv("submission.csv", index=False)
print("Saved `submission.csv`!")
submission.head()


