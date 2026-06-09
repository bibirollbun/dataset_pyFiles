import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


df = pd.read_csv("/kaggle/input/aiquest-bangla-sentiment-analysis-competition/train.csv")

df.head()


df.shape


df.info()


df.isnull().sum()


df['sentiment'] = df['sentiment'].map({'positive': 0, 'negative': 1, 'neutral': 2})





df.head()


# Function to clean text
def clean_text(text):
    text = re.sub(r'[^\u0980-\u09FF\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

# Apply
df['cleaned_text'] = df['text'].apply(clean_text)


# Function to tokenize text
def tokenize_text(text):
    return text.split()

# Apply
df['tokens'] = df['cleaned_text'].apply(tokenize_text)


# Custom Bengali stopwords list
bn_stopwords = set([
    "এবং", "কিন্তু", "হয়", "হতে", "হয়েছে", "হচ্ছে", "হয়েছেন", "হল", "হলে", "হলো", 
    "আমি", "আমরা", "তুমি", "তোমরা", "সে", "তারা", "এই", "যে", "কি", "কী", "কেন", 
    "কোথায়", "কখন", "কিভাবে", "যেমন", "সব", "একটি", "একটা", "এক", "দুই", "তিন", 
    "চার", "পাঁচ", "দশ", "কয়েক", "অনেক", "সে", "তার", "তাদের", "আপনি", "আপনার", 
    "আমার", "আমাদের", "তোমার", "তোমাদের", "তার", "তাদের", "এটা", "এটি", "এইটা", 
    "এইটি", "সেটা", "সেটি", "যে", "যারা", "যার", "যাদের", "কেউ", "কাউকে", "কিছু", 
    "কোনো", "কোন", "কিছুই", "সবার", "সকল", "সেই", "তেমন", "সেখানে", "এখানে", "সুতরাং", 
    "অথবা", "বা", "নাকি", "যদি", "তবে", "তাহলে", "নয়", "না", "হ্যাঁ", "ঠিক", "ঠিকই", 
    "বেশ", "খুব", "অনেক", "অবশ্য", "মাত্র", "শুধু", "এখন", "আগে", "পরে", "কিছুক্ষণ", 
    "সবসময়", "প্রায়", "প্রতি", "মধ্যে", "ভিতরে", "বাইরে", "উপর", "নিচে", "সম্পর্কে", 
    "জন্য", "দিকে", "পাশে", "বিনা", "ছাড়া", "যেমন", "তেমন", "এত", "এতো", "এতটাই", 
    "কত", "কতো", "কতটাই", "কিছুটা", "কিছুই", "কিছুক্ষণ", "কিছুদিন", "কিছুবার", 
    "কিছুজন", "কিছুটা", "কিছুই", "কিছুক্ষণ", "কিছুদিন", "কিছুবার", "কিছুজন"
])


# Function to remove stopwords
def remove_stopwords(tokens):
    return [word for word in tokens if word not in bn_stopwords]


# Apply stopword removal
df['tokens'] = df['tokens'].apply(remove_stopwords)


df['processed_text'] = df['tokens'].apply(lambda x: ' '.join(x))


df = df.drop(columns = ['id', 'text', 'cleaned_text', 'tokens'])


# Initialize TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=5000)





df.head()


X = tfidf.fit_transform(df['processed_text']).toarray()
y = df['sentiment']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def calculate_results(y_true, y_pred):
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results


reg = LogisticRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)


lr_results = calculate_results(y_true=y_test,y_pred=y_pred)
lr_results


from sklearn.svm import SVC

# Train an SVM model
svm_model = SVC()
svm_model.fit(X_train, y_train)


# Predict on the test set
y_pred_svm = svm_model.predict(X_test)


# Evaluate model performance
svm_results = calculate_results(y_true=y_test, y_pred=y_pred_svm)
svm_results


from sklearn.ensemble import RandomForestClassifier
# Train a Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)


#  Predict on the test set
y_pred_rf = rf_model.predict(X_test)


# Evaluate model performance
rf_results = calculate_results(y_true=y_test, y_pred=y_pred_rf)
rf_results


import xgboost as xgb

# Train an XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)


# Predict on the test set
y_pred_xgb = xgb_model.predict(X_test)


# Evaluate model performance
xgb_results = calculate_results(y_true=y_test, y_pred=y_pred_xgb)
xgb_results


all_model_results = pd.DataFrame({"Logistic_Regression": lr_results,
                                  "SVM": svm_results,
                                  "Random Forest": rf_results,
                                  "Gradient Boosting": xgb_results})
all_model_results = all_model_results.transpose()
all_model_results


# Save the DataFrame to a CSV file
all_model_results.to_csv("model_comparison_results.csv", index=True)

