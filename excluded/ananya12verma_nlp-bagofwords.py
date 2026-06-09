# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk  # For text processing
from wordcloud import WordCloud

# Setting display options for better readability
pd.set_option('display.max_colwidth', 100)


import pandas as pd
import zipfile

# Unzipping and loading labeled training data
with zipfile.ZipFile('../input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip', 'r') as z:
    with z.open('labeledTrainData.tsv') as f:
        train = pd.read_csv(f, delimiter='\t')

# Unzipping and loading test data
with zipfile.ZipFile('../input/word2vec-nlp-tutorial/testData.tsv.zip', 'r') as z:
    with z.open('testData.tsv') as f:
        test = pd.read_csv(f, delimiter='\t', quoting=3)

# Unzipping and loading unlabeled training data
with zipfile.ZipFile('../input/word2vec-nlp-tutorial/unlabeledTrainData.tsv.zip', 'r') as z:
    with z.open('unlabeledTrainData.tsv') as f:
        unlabeled_train = pd.read_csv(f, delimiter='\t', quoting=3)  

# Loading sample submission
sample_submission = pd.read_csv('../input/word2vec-nlp-tutorial/sampleSubmission.csv')

# Display the first few rows of each file
print("ğŸ“„ Labeled Training Data Sample:")
display(train.head())

print("\nğŸ“„ Test Data Sample:")
display(test.head())

print("\nğŸ“„ Unlabeled Training Data Sample:")
display(unlabeled_train.head())

print("\nğŸ“„ Sample Submission Format:")
display(sample_submission.head())



# Check the shape of the data
print(f"Training data shape: {train.shape}")
print(f"Test data shape: {test.shape}")

# Display column information
train.info()

# Check for missing values
missing_values = train.isnull().sum()
print("\nMissing values in training data:")
print(missing_values)



# Distribution of positive and negative reviews
sns.countplot(data=train, x='sentiment', palette='viridis')
plt.title('Distribution of Positive and Negative Reviews')
plt.xlabel('Sentiment (0 = Negative, 1 = Positive)')
plt.ylabel('Count')
plt.show()

# Generate word clouds for positive and negative reviews
positive_reviews = ' '.join(train[train['sentiment'] == 1]['review'])
negative_reviews = ' '.join(train[train['sentiment'] == 0]['review'])

# Word Cloud for Positive Reviews
wordcloud_positive = WordCloud(width=800, height=400, max_words=100, background_color='white').generate(positive_reviews)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Positive Reviews')
plt.show()

# Word Cloud for Negative Reviews
wordcloud_negative = WordCloud(width=800, height=400, max_words=100, background_color='white').generate(negative_reviews)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Negative Reviews')
plt.show()



import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data if not already downloaded
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
!unzip /usr/share/nltk_data/corpora/wordnet.zip -d /usr/share/nltk_data/corpora/
# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Define a function for text cleaning and lemmatization
def clean_and_lemmatize_text(review):
    # Remove HTML tags
    review = BeautifulSoup(review, 'html.parser').get_text()
    
    # Remove non-letter characters and convert to lowercase
    review = re.sub("[^a-zA-Z]", " ", review).lower()
    
    # Tokenize the review
    words = word_tokenize(review)
    
    # Remove stopwords and apply lemmatization
    stop_words = set(stopwords.words("english"))
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    # Join words back into a single string
    cleaned_review = " ".join(words)
    return cleaned_review

# Apply the cleaning and lemmatization function to the training data
train['cleaned_review'] = train['review'].apply(clean_and_lemmatize_text)

# Display a few cleaned reviews to verify
print("âœ… Sample Cleaned and Lemmatized Reviews:")
display(train[['review', 'cleaned_review']].head())



from sklearn.feature_extraction.text import CountVectorizer

# Initialize CountVectorizer for Bag of Words
vectorizer = CountVectorizer(max_features=5000)  # Limit to top 5000 features for efficiency

# Fit and transform the cleaned reviews into feature vectors
train_bow = vectorizer.fit_transform(train['cleaned_review'])

# Display the shape of the resulting feature matrix
print("âœ… Bag of Words Feature Matrix Shape:", train_bow.shape)



from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the cleaned reviews into TF-IDF feature vectors
train_tfidf = tfidf_vectorizer.fit_transform(train['cleaned_review'])

# Display the shape of the resulting TF-IDF matrix
print("âœ… TF-IDF Feature Matrix Shape:", train_tfidf.shape)



# ğŸ“Œ Importing necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ğŸ“Œ Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
train_tfidf = tfidf_vectorizer.fit_transform(train['cleaned_review'])

# ğŸ“Œ Initialize CountVectorizer for Bag of Words
bow_vectorizer = CountVectorizer(max_features=5000)  # Limit to top 5000 features for efficiency
train_bow = bow_vectorizer.fit_transform(train['cleaned_review'])

# ğŸ“Œ Splitting the data into training and testing sets using TF-IDF features
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(train_tfidf, train['sentiment'], test_size=0.2, random_state=42)

# ğŸ“Œ Splitting the data into training and testing sets using Bag of Words features
X_train_bow, X_test_bow, _, _ = train_test_split(train_bow, train['sentiment'], test_size=0.2, random_state=42)

# ğŸ“Œ Training Logistic Regression using TF-IDF features
log_reg_tfidf = LogisticRegression(max_iter=200, random_state=42)
log_reg_tfidf.fit(X_train_tfidf, y_train)

# ğŸ“Œ Making predictions and evaluating the model (TF-IDF)
y_pred_tfidf = log_reg_tfidf.predict(X_test_tfidf)
accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)
print(f"\nâœ… Logistic Regression (TF-IDF) Accuracy: {accuracy_tfidf:.4f}")
print("ğŸ“Œ Classification Report (TF-IDF):\n", classification_report(y_test, y_pred_tfidf))
print("ğŸ“Œ Confusion Matrix (TF-IDF):\n", confusion_matrix(y_test, y_pred_tfidf))

# ğŸ“Œ Training Logistic Regression using Bag of Words features
log_reg_bow = LogisticRegression(max_iter=200, random_state=42)
log_reg_bow.fit(X_train_bow, y_train)

# ğŸ“Œ Making predictions and evaluating the model (BoW)
y_pred_bow = log_reg_bow.predict(X_test_bow)
accuracy_bow = accuracy_score(y_test, y_pred_bow)
print(f"\nâœ… Logistic Regression (Bag of Words) Accuracy: {accuracy_bow:.4f}")
print("ğŸ“Œ Classification Report (Bag of Words):\n", classification_report(y_test, y_pred_bow))
print("ğŸ“Œ Confusion Matrix (Bag of Words):\n", confusion_matrix(y_test, y_pred_bow))



# ğŸ“Œ Importing necessary libraries for ROC Curve
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# ğŸ“Œ Predict probabilities for TF-IDF features
y_pred_proba_tfidf = log_reg_tfidf.predict_proba(X_test_tfidf)[:, 1]

# ğŸ“Œ Compute ROC Curve for TF-IDF
fpr_tfidf, tpr_tfidf, _ = roc_curve(y_test, y_pred_proba_tfidf)
auc_tfidf = roc_auc_score(y_test, y_pred_proba_tfidf)

# ğŸ“Œ Predict probabilities for Bag of Words features
y_pred_proba_bow = log_reg_bow.predict_proba(X_test_bow)[:, 1]

# ğŸ“Œ Compute ROC Curve for Bag of Words
fpr_bow, tpr_bow, _ = roc_curve(y_test, y_pred_proba_bow)
auc_bow = roc_auc_score(y_test, y_pred_proba_bow)

# ğŸ“Š Plotting the ROC Curves
plt.figure(figsize=(10, 6))
plt.plot(fpr_tfidf, tpr_tfidf, color='blue', label=f'TF-IDF ROC curve (AUC = {auc_tfidf:.2f})')
plt.plot(fpr_bow, tpr_bow, color='green', label=f'BoW ROC curve (AUC = {auc_bow:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.title('ROC Curve for Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.show()



# ğŸ“Œ Apply the preprocessing function to the test data
test['cleaned_review'] = test['review'].apply(clean_and_lemmatize_text)

# ğŸ“Œ Transform the preprocessed test data using trained TF-IDF and BoW vectorizers
test_tfidf = tfidf_vectorizer.transform(test['cleaned_review'])
test_bow = bow_vectorizer.transform(test['cleaned_review'])

# ğŸ“Œ Choose the model for final submission (change if needed)
final_model = log_reg_tfidf  # Use TF-IDF model (or switch to log_reg_bow for BoW)
final_features = test_tfidf  # Use TF-IDF features (or switch to test_bow for BoW)

# ğŸ“Œ Predicting on test data for submission
test_predictions = final_model.predict(final_features)

# ğŸ“Œ Create a submission DataFrame with string IDs as per the sample format
submission = pd.DataFrame({
    'id': sample_submission['id'],           
    'sentiment': test_predictions
})

# ğŸ“Œ Save the submission file
submission_file = 'logistic_reg_submission.csv'
submission.to_csv(submission_file, index=False)
print(f"\nğŸ“� Submission file saved as: {submission_file}")



# ğŸ“Œ Importing necessary libraries for Random Forest
from sklearn.ensemble import RandomForestClassifier

# ğŸ“Œ Initialize Random Forest Classifier
rf_clf_tfidf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf_bow = RandomForestClassifier(n_estimators=100, random_state=42)

# ğŸ“Œ Training Random Forest using TF-IDF features
rf_clf_tfidf.fit(X_train_tfidf, y_train)

# ğŸ“Œ Making predictions and evaluating the model (TF-IDF)
y_pred_tfidf_rf = rf_clf_tfidf.predict(X_test_tfidf)
accuracy_tfidf_rf = accuracy_score(y_test, y_pred_tfidf_rf)
print(f"\nâœ… Random Forest (TF-IDF) Accuracy: {accuracy_tfidf_rf:.4f}")
print("ğŸ“Œ Classification Report (TF-IDF):\n", classification_report(y_test, y_pred_tfidf_rf))
print("ğŸ“Œ Confusion Matrix (TF-IDF):\n", confusion_matrix(y_test, y_pred_tfidf_rf))

# ğŸ“Œ Training Random Forest using Bag of Words features
rf_clf_bow.fit(X_train_bow, y_train)

# ğŸ“Œ Making predictions and evaluating the model (BoW)
y_pred_bow_rf = rf_clf_bow.predict(X_test_bow)
accuracy_bow_rf = accuracy_score(y_test, y_pred_bow_rf)
print(f"\nâœ… Random Forest (Bag of Words) Accuracy: {accuracy_bow_rf:.4f}")
print("ğŸ“Œ Classification Report (Bag of Words):\n", classification_report(y_test, y_pred_bow_rf))
print("ğŸ“Œ Confusion Matrix (Bag of Words):\n", confusion_matrix(y_test, y_pred_bow_rf))



# ğŸ“Œ Importing necessary libraries for ROC Curve
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# ğŸ“Œ ROC Curve for TF-IDF
fpr_tfidf, tpr_tfidf, _ = roc_curve(y_test, y_pred_tfidf_rf )
auc_tfidf = roc_auc_score(y_test, y_pred_tfidf_rf )

# ğŸ“Œ ROC Curve for Bag of Words
fpr_bow, tpr_bow, _ = roc_curve(y_test, y_pred_bow_rf)
auc_bow = roc_auc_score(y_test, y_pred_bow_rf)

# ğŸ“Š Plotting the ROC Curves
plt.figure(figsize=(10, 6))
plt.plot(fpr_tfidf, tpr_tfidf, color='blue', label=f'TF-IDF ROC curve (AUC = {auc_tfidf:.2f})')
plt.plot(fpr_bow, tpr_bow, color='green', label=f'BoW ROC curve (AUC = {auc_bow:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.title('ROC Curve for Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# ğŸ“Œ Predicting on test data using Random Forest (TF-IDF)
test_tfidf_rf = rf_clf_tfidf.predict(test_tfidf)

# ğŸ“Œ Create a submission DataFrame with string IDs as per the sample format
submission_tfidf_rf = pd.DataFrame({
    'id': sample_submission['id'],           
    'sentiment': test_tfidf_rf               # Use correct predictions
})

# ğŸ“Œ Save the TF-IDF submission file
tfidf_submission_file = 'rf_tfidf_submission.csv'
submission_tfidf_rf.to_csv(tfidf_submission_file, index=False)
print(f"\nğŸ“� Submission file saved as: {tfidf_submission_file}")



# ğŸ“Œ Import necessary libraries
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm  # For progress bar

# ğŸ“Œ Initialize SVM Classifier with verbose and max_iter for progress and time control
svm_clf_tfidf = SVC(kernel='rbf', probability=True, random_state=42, verbose=True, max_iter=1000)
svm_clf_bow = SVC(kernel='rbf', probability=True, random_state=42, verbose=True, max_iter=1000)

# ğŸ“Œ Training SVM using TF-IDF features with progress bar
print("\nâ�³ Training SVM with TF-IDF features...")
for _ in tqdm(range(1)):  # Dummy loop to show progress bar
    svm_clf_tfidf.fit(X_train_tfidf, y_train)

# ğŸ“Œ Making predictions and evaluating the model (TF-IDF)
y_pred_tfidf_svm = svm_clf_tfidf.predict(X_test_tfidf)
accuracy_tfidf_svm = accuracy_score(y_test, y_pred_tfidf_svm)
print(f"\nâœ… SVM (TF-IDF) Accuracy: {accuracy_tfidf_svm:.4f}")
print("ğŸ“Œ Classification Report (TF-IDF):\n", classification_report(y_test, y_pred_tfidf_svm))
print("ğŸ“Œ Confusion Matrix (TF-IDF):\n", confusion_matrix(y_test, y_pred_tfidf_svm))

# ğŸ“Œ Training SVM using Bag of Words features with progress bar
print("\nâ�³ Training SVM with Bag of Words features...")
for _ in tqdm(range(10)):  # Dummy loop to show progress bar
    svm_clf_bow.fit(X_train_bow, y_train)

# ğŸ“Œ Making predictions and evaluating the model (BoW)
y_pred_bow_svm = svm_clf_bow.predict(X_test_bow)
accuracy_bow_svm = accuracy_score(y_test, y_pred_bow_svm)
print(f"\nâœ… SVM (Bag of Words) Accuracy: {accuracy_bow_svm:.4f}")
print("ğŸ“Œ Classification Report (Bag of Words):\n", classification_report(y_test, y_pred_bow_svm))
print("ğŸ“Œ Confusion Matrix (Bag of Words):\n", confusion_matrix(y_test, y_pred_bow_svm))



# ğŸ“Œ Importing necessary libraries for ROC Curve
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# ğŸ“Œ ROC Curve for TF-IDF (SVM)
fpr_tfidf_svm, tpr_tfidf_svm, _ = roc_curve(y_test, y_pred_tfidf_svm)
auc_tfidf_svm = roc_auc_score(y_test, y_pred_tfidf_svm)

# ğŸ“Œ ROC Curve for Bag of Words (SVM)
fpr_bow_svm, tpr_bow_svm, _ = roc_curve(y_test, y_pred_bow_svm)
auc_bow_svm = roc_auc_score(y_test, y_pred_bow_svm)

# ğŸ“Š Plotting the ROC Curves
plt.figure(figsize=(10, 6))
plt.plot(fpr_tfidf_svm, tpr_tfidf_svm, color='blue', label=f'TF-IDF ROC curve (AUC = {auc_tfidf_svm:.2f})')
plt.plot(fpr_bow_svm, tpr_bow_svm, color='green', label=f'BoW ROC curve (AUC = {auc_bow_svm:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.title('ROC Curve for SVM')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.show()



# ğŸ“Œ Importing necessary libraries for DNN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ğŸ“Œ Building the DNN model for TF-IDF
dnn_tfidf = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_tfidf.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(set(y_train)), activation='softmax')
])

dnn_tfidf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ğŸ“Œ Training the DNN on TF-IDF features
dnn_tfidf.fit(X_train_tfidf, y_train, epochs=20, batch_size=32, verbose=1)

# ğŸ“Œ Evaluating the DNN (TF-IDF)
y_pred_tfidf_dnn = dnn_tfidf.predict(X_test_tfidf).argmax(axis=1)
accuracy_tfidf_dnn = accuracy_score(y_test, y_pred_tfidf_dnn)
print(f"\nâœ… DNN (TF-IDF) Accuracy: {accuracy_tfidf_dnn:.4f}")
print("ğŸ“Œ Classification Report (TF-IDF):\n", classification_report(y_test, y_pred_tfidf_dnn))
print("ğŸ“Œ Confusion Matrix (TF-IDF):\n", confusion_matrix(y_test, y_pred_tfidf_dnn))



# ğŸ“Œ Import necessary libraries
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# ğŸ“Œ Get predicted probabilities for TF-IDF
y_pred_tfidf_prob = dnn_tfidf.predict(X_test_tfidf)

# ğŸ“Œ ROC Curve for DNN (TF-IDF)
fpr_tfidf, tpr_tfidf, _ = roc_curve(y_test, y_pred_tfidf_prob[:, 1])
roc_auc_tfidf = auc(fpr_tfidf, tpr_tfidf)

plt.plot(fpr_tfidf, tpr_tfidf, label='TF-IDF (AUC = {:.2f})'.format(roc_auc_tfidf))


# ğŸ“Œ Plot settings
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for DNN')
plt.legend(loc="lower right")
plt.grid()
plt.show()



# ğŸ“Œ Predicting on test data using DNN (TF-IDF)
test_tfidf_dnn = dnn_tfidf.predict(test_tfidf).argmax(axis=1)

# ğŸ“Œ Create a submission DataFrame with string IDs as per the sample format
submission_dnn_tfidf = pd.DataFrame({
    'id': sample_submission['id'],           
    'sentiment': test_tfidf_dnn               # Use DNN TF-IDF predictions
})

# ğŸ“Œ Save the TF-IDF DNN submission file
dnn_tfidf_submission_file = 'dnn_tfidf_submission.csv'
submission_dnn_tfidf.to_csv(dnn_tfidf_submission_file, index=False)
print(f"\nğŸ“� Submission file saved as: {dnn_tfidf_submission_file}")


