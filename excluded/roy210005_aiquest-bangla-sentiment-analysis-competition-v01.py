import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix,precision_recall_fscore_support


df=pd.read_csv('/kaggle/input/aiquest-bangla-sentiment-analysis-competition/train.csv')


df.head()


df.tail()


df.shape


df.info()


df.isnull()


df.isnull().sum()


#statistical measures
df.describe()


df['sentiment'].value_counts()


print(list(df.columns))


df.columns=df.columns.str.strip().str.lower()
df.columns
df.head()


df['text']


#maping
df['sentiment']=df['sentiment'].map({'positive':0,'negative':1,'neutral':2})
df.head()


#nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


#load bangla stopwordss
stpwrd=set(stopwords.words('bengali'))
print(stpwrd)


#Manually_add_more_Bengali_stopwords
bn_stpwrd = set([
    "এবং", "কিন্তু", "হয়", "হতে", "হয়েছে", "হচ্ছে", "হয়েছেন", "হল", "হলে", "হলো",
    "আমি", "আমরা", "তুমি", "তোমরা", "সে", "তারা", "এই", "যে", "কি", "কী", "কেন",
    "কোথায়", "কখন", "কিভাবে", "যেমন", "সব", "একটি", "একটা", "এক", "দুই", "তিন",
    "চার", "পাঁচ", "দশ", "কয়েক", "অনেক", "তার", "তাদের", "আপনি", "আপনার",
    "আমার", "আমাদের", "তোমার", "তোমাদের", "এটা", "এটি", "এইটা", "এইটি", "সেটা", "সেটি",
    "যারা", "যার", "যাদের", "কেউ", "কাউকে", "কিছু", "কোনো", "কোন", "কিছুই", "সবার",
    "সকল", "সেই", "তেমন", "সেখানে", "এখানে", "সুতরাং", "অথবা", "বা", "নাকি", "যদি",
    "তবে", "তাহলে", "নয়", "না", "হ্যাঁ", "ঠিক", "বেশ", "খুব", "অবশ্য", "মাত্র", "শুধু",
    "এখন", "আগে", "পরে", "কিছুক্ষণ", "সবসময়", "প্রায়", "প্রতি", "মধ্যে", "ভিতরে",
    "বাইরে", "উপর", "নিচে", "সম্পর্কে", "জন্য", "দিকে", "পাশে", "বিনা", "ছাড়া", "এত",
    "এতো", "এতটাই", "কত", "কতো", "কতটাই", "কিছুটা", "কিছুদিন", "কিছুবার", "কিছুজন"
])


def clean_text(text):
  text=re.sub(r'[^\u0980-\u09FF]',' ',text)
  words= text.split()
  words= [word for word in words if word not in stpwrd]
  return ' '.join(words)


#text_cleaning
df['cleaned_text']=df['text'].apply(clean_text)


#tokenization
def tokenize_text(text):
  return text.split()

df['tokens']= df['cleaned_text'].apply(tokenize_text)

#stopwads_remove
def remove_stpwrds(tokens):
  return[word for word in tokens if word not in bn_stpwrd]
df['tokens']= df['tokens'].apply(remove_stpwrds)


#join tokens to form clean_text
df['clean_text']= df['tokens'].apply(lambda x: ' '.join(x))


#drop unnecessary colums
df= df.drop(columns=['id','text','cleaned_text','tokens'])
print(df.columns)


#tf idf vectorizer
tfidf= TfidfVectorizer(max_features=5000)
X= tfidf.fit_transform(df['clean_text']).toarray()
y= df['sentiment']
df.head()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

#train text split
X_train,X_test, y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)

#define_funcation
def results(y_true, y_pred):
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")

    return {"accuracy": model_accuracy,
                     "precision": model_precision,
                     "recall": model_recall,
                     "f1": model_f1}

logistic_reg = LogisticRegression(max_iter=1000)
logistic_reg.fit(X_train, y_train)
y_pred = logistic_reg.predict(X_test)
logistic_dict=results(y_test,y_pred)

print(f"Accuracy :{logistic_dict['accuracy']:.2f}%")
print(f"Precision:{logistic_dict['precision']:.4f}")
print(f"Recall   :{logistic_dict['recall']:.4f}")
print(f"F1 Score :{logistic_dict['f1']:.4f}")



from sklearn.svm import SVC

#train smv model
svm_model= SVC()
svm_model.fit(X_train,y_train)
#predict test
y_pred=svm_model.predict(X_test)


svm_dict=results(y_test,y_pred)

print(f"Accuracy:{svm_dict['accuracy']:.2f}%")
print(f"Precision:{svm_dict['precision']:.4f}")
print(f"Recall:{svm_dict['recall']:.4f}")
print(f"F1 Score:{svm_dict['f1']:.4f}")



from sklearn.ensemble import RandomForestClassifier

random_forest= RandomForestClassifier()
random_forest.fit(X_train,y_train)

y_pred=random_forest.predict(X_test)

random_forest_dict= results(y_test, y_pred)

print(f"Accuracy:{random_forest_dict['accuracy']:.2f}%")
print(f"Precision:{random_forest_dict['precision']:.4f}")
print(f"Recall:{random_forest_dict['recall']:.4f}")
print(f"F1 Score:{random_forest_dict['f1']:.4f}")


from xgboost import XGBClassifier

xgb_model= XGBClassifier(eval_metric='mlogloss')
xgb_model.fit(X_train,y_train)

y_pred_xgb= xgb_model.predict(X_test)

xgb_dict= results(y_test,y_pred_xgb)

print(f"Accuracy:{xgb_dict['accuracy']:.2f}%")
print(f"Precision:{xgb_dict['precision']:.4f}")
print(f"Recall:{xgb_dict['recall']:.4f}")
print(f"F1 Score:{xgb_dict['f1']:.4f}")




import pandas as pd
results= pd.DataFrame({
    "Logistic_Regression": [logistic_dict['accuracy'], logistic_dict['precision'], logistic_dict['recall'], logistic_dict['f1']],
    "SVM": [svm_dict['accuracy'], svm_dict['precision'], svm_dict['recall'], svm_dict['f1']],
    "Random Forest": [random_forest_dict['accuracy'], random_forest_dict['precision'], random_forest_dict['recall'], random_forest_dict['f1']],
    "XGBoost": [xgb_dict['accuracy'], xgb_dict['precision'], xgb_dict['recall'], xgb_dict['f1']]},
    index=["Accuracy", "Precision", "Recall", "F1 Score"])

results = results.transpose()
results



