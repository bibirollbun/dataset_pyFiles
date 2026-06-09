import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import warnings
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def basic_clean(text):
    """
    Simplistic text cleaning: lowercasing and removing non-alphanumeric characters.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


train_df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test_df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
sample_submission = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")


train_df['Misconception'] = train_df['Misconception'].fillna('NA').astype(str)
train_df['target_cat'] = train_df['Category'] + ":" + train_df['Misconception']

le_target = LabelEncoder()
train_df['target_encoded'] = le_target.fit_transform(train_df['target_cat'])
target_classes = le_target.classes_


train_df['combined_text'] = (
    train_df['QuestionText'].astype(str) + " " +
    train_df['MC_Answer'].astype(str) + " " +
    train_df['StudentExplanation'].astype(str)
)
test_df['combined_text'] = (
    test_df['QuestionText'].astype(str) + " " +
    test_df['MC_Answer'].astype(str) + " " +
    test_df['StudentExplanation'].astype(str)
)

train_df['cleaned_text'] = train_df['combined_text'].apply(basic_clean)
test_df['cleaned_text'] = test_df['combined_text'].apply(basic_clean)


print("Creating TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))

# Fit TF-IDF on combined train and test text
all_text = pd.concat([train_df['cleaned_text'], test_df['cleaned_text']], axis=0)
tfidf_vectorizer.fit(all_text)

X_train_tfidf = tfidf_vectorizer.transform(train_df['cleaned_text'])
X_test_tfidf = tfidf_vectorizer.transform(test_df['cleaned_text'])
y_train = train_df['target_encoded']

print(f"TF-IDF feature shape (Train): {X_train_tfidf.shape}")
print(f"TF-IDF feature shape (Test): {X_test_tfidf.shape}")


print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=500, random_state=42, solver='saga', n_jobs=-1) # Use 'saga' for large datasets
model.fit(X_train_tfidf, y_train)


print("Making predictions...")
test_probas = model.predict_proba(X_test_tfidf)

# Get top 3 predictions
top3_indices = test_probas.argsort(axis=1)[:, -3:][:, ::-1] # Sort descending and take top 3

test_predictions_labels = []
for indices in top3_indices:
    pred_labels = [target_classes[i] for i in indices]
    test_predictions_labels.append(' '.join(pred_labels))


sample_submission['Category:Misconception'] = test_predictions_labels
submission_filename = "submission.csv"
sample_submission.to_csv(submission_filename, index=False)
print(f"\nSubmission file created: {submission_filename}")
print("Simplified model training complete.")

