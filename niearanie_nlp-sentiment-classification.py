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


!pip install nltk


!pip install sckit learn


train_df = "/kaggle/input/ml-olympiad-tfugsurabaya-2024/train.tsv"
test_df = "/kaggle/input/ml-olympiad-tfugsurabaya-2024/test.tsv"
sample_submission = "/kaggle/input/ml-olympiad-tfugsurabaya-2024/sample_submission.csv"


import pandas as pd
train_df = pd.read_csv('/kaggle/input/ml-olympiad-tfugsurabaya-2024/train.tsv', sep='\t')
print("Shape of train_df:", train_df.shape)
train_df.head()
print("First 5 rows of train_df:")
print(train_df.head())

print("\nInfo of train_df:")
train_df.info()

print("\nDescriptive statistics of train_df:")
print(train_df.describe())

test_df = pd.read_csv('/kaggle/input/ml-olympiad-tfugsurabaya-2024/test.tsv', sep='\t')
print("Shape of test_df:", test_df.shape)
test_df.head()
print("Info of test_df:")
test_df.info()

print("\nDescriptive statistics of test_df:")
print(test_df.describe())

sample_submission_df = pd.read_csv('/kaggle/input/ml-olympiad-tfugsurabaya-2024/sample_submission.csv')
print("Shape of sample_submission_df:", sample_submission_df.shape)
sample_submission_df.head()

print("Info of sample_submission_df:")
sample_submission_df.info()

print("\nDescriptive statistics of sample_submission_df:")
print(sample_submission_df.describe())


import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download necessary NLTK data (if not already downloaded)
nltk.download('stopwords')
nltk.download('punkt')

print("NLTK libraries imported and data downloaded.")
def preprocess_text(text):
    # 1. Convert text to lowercase
    text = text.lower()

    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 3. Remove numbers
    text = re.sub(r'\d+', '', text)

    # 4. Tokenize the text
    tokens = word_tokenize(text)

    # 5. Remove stopwords and apply stemming
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    processed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]

    # 6. Join the processed tokens back into a single string
    return " ".join(processed_tokens)

print("preprocess_text function defined.")
train_df['PROCESSED_REVIEW'] = train_df['REVIEW'].apply(preprocess_text)
print("First 5 rows of train_df with 'PROCESSED_REVIEW' column:")
print(train_df.head())


import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download necessary NLTK data (if not already downloaded)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

print("NLTK libraries imported and data downloaded.")
train_df['PROCESSED_REVIEW'] = train_df['REVIEW'].apply(preprocess_text)
print("First 5 rows of train_df with 'PROCESSED_REVIEW' column:")
print(train_df.head())

test_df['PROCESSED_REVIEW'] = test_df['REVIEW'].apply(preprocess_text)
print("First 5 rows of test_df with 'PROCESSED_REVIEW' column:")
print(test_df.head())


X = train_df['PROCESSED_REVIEW']
y = train_df['LABEL']

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit the vectorizer on the training data (X)
X_vectorized = vectorizer.fit_transform(X)

print("TF-IDF Vectorizer initialized and fitted.")
print("Shape of vectorized X (X_vectorized):", X_vectorized.shape)
X_vectorized = vectorizer.transform(X)
test_X_vectorized = vectorizer.transform(test_df['PROCESSED_REVIEW'])

print("Training data (X) transformed. Shape:", X_vectorized.shape)
print("Test data (test_df['PROCESSED_REVIEW']) transformed. Shape:", test_X_vectorized.shape)


from sklearn.model_selection import train_test_split

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Print the shapes of the resulting sets
print("Shape of X_train:", X_train.shape)
print("Shape of X_val:", X_val.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_val:", y_val.shape)


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Required scikit-learn modules imported.")


classifier = MultinomialNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_val)

print("Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))


classifier = MultinomialNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_val)

print("Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred, zero_division=0))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))


from sklearn.linear_model import LogisticRegression

print("LogisticRegression module imported.")


logistic_reg_classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
logistic_reg_classifier.fit(X_train, y_train)

y_pred_lr = logistic_reg_classifier.predict(X_val)

print("Logistic Regression Accuracy:", accuracy_score(y_val, y_pred_lr))
print("\nLogistic Regression Classification Report:\n", classification_report(y_val, y_pred_lr, zero_division=0))
print("\nLogistic Regression Confusion Matrix:\n", confusion_matrix(y_val, y_pred_lr))


logistic_reg_classifier = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
logistic_reg_classifier.fit(X_train, y_train)

y_pred_lr = logistic_reg_classifier.predict(X_val)

print("Logistic Regression Accuracy:", accuracy_score(y_val, y_pred_lr))
print("\nLogistic Regression Classification Report:\n", classification_report(y_val, y_pred_lr, zero_division=0))
print("\nLogistic Regression Confusion Matrix:\n", confusion_matrix(y_val, y_pred_lr))


from sklearn.svm import SVC

print("SVC module imported.")

# Initialize, train, and evaluate a Support Vector Machine (SVM) classifier
svm_classifier = SVC(random_state=42)
svm_classifier.fit(X_train, y_train)

y_pred_svm = svm_classifier.predict(X_val)

print("\nSupport Vector Machine Accuracy:", accuracy_score(y_val, y_pred_svm))
print("\nSupport Vector Machine Classification Report:\n", classification_report(y_val, y_pred_svm, zero_division=0))
print("\nSupport Vector Machine Confusion Matrix:\n", confusion_matrix(y_val, y_pred_svm))

# Collect metrics for all models
metrics = {
    'Naive Bayes': {
        'Accuracy': accuracy_score(y_val, classifier.predict(X_val)),
        'Precision': classification_report(y_val, classifier.predict(X_val), output_dict=True, zero_division=0)['weighted avg']['precision'],
        'Recall': classification_report(y_val, classifier.predict(X_val), output_dict=True, zero_division=0)['weighted avg']['recall'],
        'F1-Score': classification_report(y_val, classifier.predict(X_val), output_dict=True, zero_division=0)['weighted avg']['f1-score']
    },
    'Logistic Regression': {
        'Accuracy': accuracy_score(y_val, logistic_reg_classifier.predict(X_val)),
        'Precision': classification_report(y_val, logistic_reg_classifier.predict(X_val), output_dict=True, zero_division=0)['weighted avg']['precision'],
        'Recall': classification_report(y_val, logistic_reg_classifier.predict(X_val), output_dict=True, zero_division=0)['weighted avg']['recall'],
        'F1-Score': classification_report(y_val, logistic_reg_classifier.predict(X_val), output_dict=True, zero_division=0)['weighted avg']['f1-score']
    },
    'SVM': {
        'Accuracy': accuracy_score(y_val, y_pred_svm),
        'Precision': classification_report(y_val, y_pred_svm, output_dict=True, zero_division=0)['weighted avg']['precision'],
        'Recall': classification_report(y_val, y_pred_svm, output_dict=True, zero_division=0)['weighted avg']['recall'],
        'F1-Score': classification_report(y_val, y_pred_svm, output_dict=True, zero_division=0)['weighted avg']['f1-score']
    }
}

# Create a DataFrame for comparison
comparison_df = pd.DataFrame(metrics).T
print("\nModel Comparison Table:\n")
print(comparison_df)

