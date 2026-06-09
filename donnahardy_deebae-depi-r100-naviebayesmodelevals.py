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


#Library Download
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report



# Read Initial Dataset file
df = pd.read_csv('/kaggle/input/initial-news-dataset/Initial_Dataset_from_train export 2025-01-28 09-29-19.csv')


# Preprocess the text
df['text'] = df['text'].str.lower()  # Convert Uppercase to lowercase
df['text'] = df['text'].str.replace(r'[^a-z\s]', '', regex=True)  # Remove special characters and numbers

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Vectorize the text using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred, zero_division=0))



from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report, accuracy_score
import pandas as pd
import re

def preprocess_text(text):
    text = text.lower()  # Convert Uppercase to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters and numbers
    return text

def evaluate_alpha(alpha):
    # Read Initial Dataset file
    df = pd.read_csv('/kaggle/input/initial-news-dataset/Initial_Dataset_from_train export 2025-01-28 09-29-19.csv')
    
    # Preprocess the text
    df['text'] = df['text'].apply(preprocess_text)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    
    # Create a pipeline with TF-IDF vectorizer and Naive Bayes classifier with adjustable alpha
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('nb', MultinomialNB(alpha=alpha))
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print(f'Trial Balanced Accuracy with alpha={alpha}: {balanced_acc}')
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred, zero_division=0))

# Example usage with alpha=0.5
evaluate_alpha(alpha=0.5)




from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report, accuracy_score
import pandas as pd
import re

def preprocess_text(text):
    text = text.lower()  # Convert Uppercase to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters and numbers
    return text

def evaluate_alpha(alpha):
    # Read Initial Dataset file
    df = pd.read_csv('/kaggle/input/initial-news-dataset/Initial_Dataset_from_train export 2025-01-28 09-29-19.csv')
    
    # Preprocess the text
    df['text'] = df['text'].apply(preprocess_text)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    
    # Create a pipeline with TF-IDF vectorizer and Naive Bayes classifier with adjustable alpha
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('nb', MultinomialNB(alpha=alpha))
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print(f'Trial Balanced Accuracy with alpha={alpha}: {balanced_acc}')
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred, zero_division=0))

# Example usage with alpha=0.6
evaluate_alpha(alpha=0.6)




from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report, accuracy_score
import pandas as pd
import re

def preprocess_text(text):
    text = text.lower()  # Convert Uppercase to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters and numbers
    return text

def evaluate_alpha(alpha):
    # Read Initial Dataset file
    df = pd.read_csv('/kaggle/input/initial-news-dataset/Initial_Dataset_from_train export 2025-01-28 09-29-19.csv')
    
    # Preprocess the text
    df['text'] = df['text'].apply(preprocess_text)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    
    # Create a pipeline with TF-IDF vectorizer and Naive Bayes classifier with adjustable alpha
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('nb', MultinomialNB(alpha=alpha))
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print(f'Trial Balanced Accuracy with alpha={alpha}: {balanced_acc}')
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred, zero_division=0))

# Example usage with alpha=0.7
evaluate_alpha(alpha=0.7)




from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report, accuracy_score
import pandas as pd
import re

def preprocess_text(text):
    text = text.lower()  # Convert Uppercase to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters and numbers
    return text

def evaluate_alpha(alpha):
    # Read Initial Dataset file
    df = pd.read_csv('/kaggle/input/initial-news-dataset/Initial_Dataset_from_train export 2025-01-28 09-29-19.csv')
    
    # Preprocess the text
    df['text'] = df['text'].apply(preprocess_text)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    
    # Create a pipeline with TF-IDF vectorizer and Naive Bayes classifier with adjustable alpha
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('nb', MultinomialNB(alpha=alpha))
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print(f'Trial Balanced Accuracy with alpha={alpha}: {balanced_acc}')
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred, zero_division=0))

# Example usage with alpha=0.8
evaluate_alpha(alpha=0.8)




from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report, accuracy_score
import pandas as pd
import re

def preprocess_text(text):
    text = text.lower()  # Convert Uppercase to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters and numbers
    return text

def evaluate_alpha(alpha):
    # Read Initial Dataset file
    df = pd.read_csv('/kaggle/input/initial-news-dataset/Initial_Dataset_from_train export 2025-01-28 09-29-19.csv')
    
    # Preprocess the text
    df['text'] = df['text'].apply(preprocess_text)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    
    # Create a pipeline with TF-IDF vectorizer and Naive Bayes classifier with adjustable alpha
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('nb', MultinomialNB(alpha=alpha))
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print(f'Trial Balanced Accuracy with alpha={alpha}: {balanced_acc}')
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred, zero_division=0))

# Example usage with alpha=0.9
evaluate_alpha(alpha=0.9)




# Combine the text data and predictions into a new DataFrame
results_df = pd.DataFrame({
    'text': X_test,
    'predicted_label': y_pred,
    'true_label': y_test
})

# Reset index to align the DataFrame properly
results_df.reset_index(drop=True, inplace=True)

# Save the resulting DataFrame to a CSV file
results_df.to_csv('submission.csv', index=False)

print("Predictions saved to 'submission.csv'")

