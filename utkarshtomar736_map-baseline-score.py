import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Load data
train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')


# Preprocessing (including NaN fix)
train['StudentExplanation'] = train['StudentExplanation'].fillna('no_explanation')
train['QuestionText'] = train['QuestionText'].fillna('')
train['MC_Answer'] = train['MC_Answer'].fillna('')
train['Category'] = train['Category'].fillna('UnknownCategory')
train['Misconception'] = train['Misconception'].fillna('UnknownMisconception')

# Create target
train['target'] = train['Category'] + ':' + train['Misconception']

# Group rare classes
label_counts = train['target'].value_counts()
rare_labels = label_counts[label_counts < 2].index
train['target'] = train['target'].apply(lambda x: 'Other' if x in rare_labels else x)

# Combine text features
train['text'] = train['QuestionText'] + ' ' + train['MC_Answer'] + ' ' + train['StudentExplanation']

# Prepare X and y
X = train['text']
y = train['target']

# Split (now safe)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)


# Train model
model = LogisticRegression(max_iter=300, multi_class='ovr', n_jobs=-1, random_state=42)
model.fit(X_train_tfidf, y_train)

# Predict probabilities on validation (for ranking)
y_val_proba = model.predict_proba(X_val_tfidf)

# Custom MAP@3 evaluation function (since there's one true label per sample)
def compute_map_at_3(y_true, y_proba, classes, k=3):
    
    # Get top k predicted indices per sample (descending probability)
    top_k_preds = np.argsort(y_proba, axis=1)[:, -k:][:, ::-1]  # Highest first
    
    aps = []
    for i in range(len(y_true)):
        true_label_idx = np.where(classes == y_true.iloc[i])[0][0]  # Index of true label
        relevant_ranks = np.where(top_k_preds[i] == true_label_idx)[0] + 1  # Ranks where true appears (1-based)
        
        if len(relevant_ranks) > 0:
            rank = relevant_ranks[0]  # First occurrence (since duplicates ignored)
            ap = 1.0 / rank  # Precision at first correct
        else:
            ap = 0.0
        aps.append(ap)
    
    return np.mean(aps)


# Compute and print local MAP@3
classes = np.array(model.classes_)
map3 = compute_map_at_3(y_val, y_val_proba, classes)
print(f'Local Validation MAP@3: {map3:.4f}')


test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')

# Preprocess test data
test['StudentExplanation'] = test['StudentExplanation'].fillna('no_explanation')
test['QuestionText'] = test['QuestionText'].fillna('')
test['MC_Answer'] = test['MC_Answer'].fillna('')
test['text'] = test['QuestionText'] + ' ' + test['MC_Answer'] + ' ' + test['StudentExplanation']

# Vectorize test text
X_test_tfidf = vectorizer.transform(test['text'])

# Predict probabilities
test_proba = model.predict_proba(X_test_tfidf)

# Get top 3 predictions per sample
top_k = 3
top_k_indices = np.argsort(test_proba, axis=1)[:, -top_k:][:, ::-1]  # Highest prob first
predictions = []
for i in range(len(test)):
    top_labels = [classes[idx] for idx in top_k_indices[i]]
    predictions.append(' '.join(top_labels)) 


# Create submission DataFrame
submission = pd.DataFrame({
    'row_id': test['row_id'],  
    'Category:Misconception': predictions
})


submission.to_csv("submission.csv", index=False)

