import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Load data
train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')

# Encode labels
le = LabelEncoder()
y = le.fit_transform(train['Category'])

# Build pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
    ('svm', SVC(kernel='linear', probability=True))
])

# Train model
pipeline.fit(train['QuestionText'], y)

# Predict probabilities
probas = pipeline.predict_proba(test['QuestionText'])

# Map class indices to expected labels
label_map = {
    'Correct': None,
    'Neither': None,
    'Misconception': None
}

# Update map with true index from LabelEncoder
for idx, class_name in enumerate(le.classes_):
    if "Correct" in class_name:
        label_map['Correct'] = idx
    elif "Neither" in class_name:
        label_map['Neither'] = idx
    elif "Misconception" in class_name:
        label_map['Misconception'] = idx

# Create final submission strings
submission = pd.DataFrame()
submission['row_id'] = test['row_id']
submission['Category:Misconception'] = [
    f"True_Correct:{probas[i][label_map['Correct']]:.4f} "
    f"False_Neither:{probas[i][label_map['Neither']]:.4f} "
    f"False_Misconception:{probas[i][label_map['Misconception']]:.4f}"
    for i in range(len(test))
]

# Save submission
submission.to_csv("submission.csv", index=False)

