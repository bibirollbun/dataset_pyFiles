pip install pandas numpy matplotlib seaborn scikit-learn nltk


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Try multiple classifier imports with fallbacks
try:
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier()
except ImportError:
    try:
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(max_iter=1000)
        print("Using LogisticRegression as fallback")
    except ImportError:
        from sklearn.svm import LinearSVC
        classifier = LinearSVC()
        print("Using LinearSVC as fallback")

# Load data
train_df = pd.read_csv('/kaggle/input/kaggle-community-olympiad-unmasking-fakes/complete.csv')
test_df = pd.read_csv('/kaggle/input/kaggle-community-olympiad-unmasking-fakes/validate.csv')

# Check available columns in test_df
print("Columns in test_df:", test_df.columns.tolist())

# Simple text classification pipeline
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X = vectorizer.fit_transform(train_df['Statement'])
y = train_df['Label']

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
classifier.fit(X_train, y_train)

# Evaluate
val_predictions = classifier.predict(X_val)
print(classification_report(y_val, val_predictions))
print(f"Accuracy: {accuracy_score(y_val, val_predictions):.2f}")

# Make predictions
test_X = vectorizer.transform(test_df['Statement'])
test_predictions = classifier.predict(test_X)

# Create submission - using the correct ID column name
# First try common column names for ID
id_column = None
for possible_id in ['id', 'ID', 'Id', 'statement_id']:
    if possible_id in test_df.columns:
        id_column = possible_id
        break

if id_column is None:
    # If no ID column found, use index
    submission = pd.DataFrame({
        'label': test_predictions
    })
    print("No ID column found - using index as identifier")
else:
    submission = pd.DataFrame({
        'id': test_df[id_column],
        'label': test_predictions
    })

submission.to_csv('submission.csv', index=False)
print("Submission file created!")


# First, let's fix the scikit-learn version
!pip install --upgrade scikit-learn==1.0.2
import sklearn
print("Scikit-learn version:", sklearn.__version__)

# Now implement fine-tuning with version-safe code
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Manual grid search implementation (avoids GridSearchCV import issues)
def manual_grid_search(X_train, y_train, X_val, y_val):
    best_score = 0
    best_params = {}
    best_model = None
    
    # Define parameter combinations to try
    param_combinations = [
        {'max_features': 5000, 'ngram_range': (1, 1), 'C': 0.1},
        {'max_features': 10000, 'ngram_range': (1, 2), 'C': 1},
        {'max_features': 15000, 'ngram_range': (1, 3), 'C': 10},
        {'max_features': 10000, 'ngram_range': (1, 2), 'C': 0.1},
        {'max_features': 15000, 'ngram_range': (1, 2), 'C': 1}
    ]
    
    for params in param_combinations:
        # Vectorize text
        vectorizer = TfidfVectorizer(
            max_features=params['max_features'],
            ngram_range=params['ngram_range'],
            stop_words='english'
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        X_val_vec = vectorizer.transform(X_val)
        
        # Train model
        model = LogisticRegression(
            C=params['C'],
            max_iter=1000,
            solver='liblinear',
            penalty='l2'
        )
        model.fit(X_train_vec, y_train)
        
        # Evaluate
        val_predictions = model.predict(X_val_vec)
        score = accuracy_score(y_val, val_predictions)
        
        if score > best_score:
            best_score = score
            best_params = params
            best_model = (vectorizer, model)
    
    return best_model, best_params, best_score

# Create train/validation split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    train_df['Statement'],
    train_df['Label'],
    test_size=0.2,
    random_state=42
)

# Run manual grid search
best_model, best_params, best_score = manual_grid_search(X_train, y_train, X_val, y_val)
vectorizer, model = best_model

print("\nBest parameters found:")
print(best_params)
print(f"Best validation accuracy: {best_score:.2f}")

# Evaluate on full training data
X_full_vec = vectorizer.transform(train_df['Statement'])
y_full_pred = model.predict(X_full_vec)
print("\nFull training set performance:")
print(classification_report(train_df['Label'], y_full_pred))

# Create final submission
test_vec = vectorizer.transform(test_df['Statement'])
test_predictions = model.predict(test_vec)
submission = pd.DataFrame({
    'ID': test_df['ID'],
    'Label': test_predictions
})
submission.to_csv('optimized_submission.csv', index=False)
print("\nOptimized submission file created!")


# First, reset the environment to stable versions
!pip install --upgrade scikit-learn==0.24.2 pandas numpy

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load data
train_df = pd.read_csv('/kaggle/input/kaggle-community-olympiad-unmasking-fakes/complete.csv')
test_df = pd.read_csv('/kaggle/input/kaggle-community-olympiad-unmasking-fakes/validate.csv')

# Basic but effective feature engineering
def add_simple_features(df):
    df['word_count'] = df['Statement'].apply(lambda x: len(str(x).split()))
    df['char_count'] = df['Statement'].apply(lambda x: len(str(x)))
    df['excl_quest_count'] = df['Statement'].apply(lambda x: str(x).count('!') + str(x).count('?'))
    return df

train_df = add_simple_features(train_df)
test_df = add_simple_features(test_df)

# Text vectorization - using only basic parameters
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words='english'
)

# Prepare features
X_text = vectorizer.fit_transform(train_df['Statement'])
X_test_text = vectorizer.transform(test_df['Statement'])

# Simple logistic regression model (most stable classifier)
model = LogisticRegression(
    C=1,
    max_iter=1000,
    solver='liblinear',
    penalty='l2'
)

# Train model
model.fit(X_text, train_df['Label'])

# Evaluate
train_predictions = model.predict(X_text)
print("Training Performance:")
print(classification_report(train_df['Label'], train_predictions))

# Create submission
test_predictions = model.predict(X_test_text)
submission = pd.DataFrame({
    'ID': test_df['ID'],
    'label': test_predictions
})
submission.to_csv('stable_submission.csv', index=False)
print("\nSubmission file created!")


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Supercharged Feature Engineering (No External Dependencies)
def create_power_features(df):
    # Basic text stats
    df['word_count'] = df['Statement'].apply(lambda x: len(str(x).split()))
    df['char_count'] = df['Statement'].apply(lambda x: len(str(x)))
    df['avg_word_len'] = df['char_count'] / df['word_count']
    
    # Advanced punctuation analysis
    df['excl_quest'] = df['Statement'].apply(lambda x: str(x).count('!') + str(x).count('?'))
    df['quote_count'] = df['Statement'].apply(lambda x: str(x).count('"'))
    df['number_count'] = df['Statement'].apply(lambda x: sum(c.isdigit() for c in str(x)))
    
    # Political context features
    df['is_republican'] = df['Political Party Affiliation'].apply(
        lambda x: 1 if str(x).lower() == 'republican' else 0)
    df['is_democrat'] = df['Political Party Affiliation'].apply(
        lambda x: 1 if str(x).lower() == 'democrat' else 0)
    
    # Contextual flags
    df['has_state'] = df['State'].apply(lambda x: 0 if pd.isna(x) else 1)
    df['has_job_title'] = df['Speaker Job Title'].apply(lambda x: 0 if pd.isna(x) else 1)
    
    return df

# Apply feature engineering
train_df = create_power_features(train_df)
test_df = create_power_features(test_df)

# 2. Robust Text Vectorization
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    stop_words='english',
    min_df=3,
    max_df=0.9
)

# Prepare features
X_text = vectorizer.fit_transform(train_df['Statement'])
X_test_text = vectorizer.transform(test_df['Statement'])

# Combine with engineered features
feature_cols = ['word_count', 'char_count', 'avg_word_len', 'excl_quest', 
                'quote_count', 'number_count', 'is_republican', 'is_democrat',
                'has_state', 'has_job_title']

X_engineered = train_df[feature_cols].values
X_test_engineered = test_df[feature_cols].values

# Final feature matrix
X_final = np.hstack([X_text.toarray(), X_engineered])
X_test_final = np.hstack([X_test_text.toarray(), X_test_engineered])

# 3. Optimized Logistic Regression
model = LogisticRegression(
    C=0.8,
    max_iter=2000,
    solver='liblinear',
    penalty='l2',
    class_weight='balanced'
)

# Train model
model.fit(X_final, train_df['Label'])

# Evaluate
train_preds = model.predict(X_final)
print("Enhanced Training Performance:")
print(classification_report(train_df['Label'], train_preds))

# 4. Create Submission
test_preds = model.predict(X_test_final)
submission = pd.DataFrame({
    'ID': test_df['ID'],
    'label': test_preds
})
submission.to_csv('ultimate_submission.csv', index=False)
print("\nUltimate submission file created!")

