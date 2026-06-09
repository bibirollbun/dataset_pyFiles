import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import f1_score


stop_words = set(stopwords.words("english"))
nltk.download('omw-1.4')


df_train = pd.read_csv('/kaggle/input/comments-classification/Dataset/train.csv')
df_test = pd.read_csv('/kaggle/input/comments-classification/Dataset/test.csv')


df_train.head()


df_train.psychotic_depression.value_counts()


df_train.isnull().sum()


def stemming(text):
    stemmer = SnowballStemmer("english")
    text = text.split()
    text = [stemmer.stem(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def Removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lemmatization(text):
    lemmatizer= WordNetLemmatizer()
    text = text.split()
    text=[lemmatizer.lemmatize(y) for y in text]
    return " " .join(text)

def lower_case(text):
    text = text.split()
    text=[y.lower() for y in text]
    return " " .join(text)
            
def normalize_text(df):
    #df.comment_text=df.comment_text.apply(lambda text : lower_case(text))
    df.comment_text=df.comment_text.apply(lambda text : remove_stop_words(text))
    #df.comment_text=df.comment_text.apply(lambda text : Removing_numbers(text))
    #df.comment_text=df.comment_text.apply(lambda text : lemmatization(text))
    #df.comment_text=df.comment_text.apply(lambda text : stemming(text))
    return df

def normalized_sentence(sentence):
    #sentence= lower_case(sentence)
    sentence= remove_stop_words(sentence)
    #sentence= Removing_numbers(sentence)
    #sentence= lemmatization(sentence)
    #sentence= stemming(sentence)
    return sentence

# even though theres all this preprocessing possibilities...pretty much everything reduced the f1 score...so just removing stop words


normalized_sentence("JUST SOME PSSYYYCCHOOO TESTTTT!!!@@456 __")


df_train= normalize_text(df_train)
df_test= normalize_text(df_test)


X = df_train["comment_text"]
y = df_train["psychotic_depression"]

# Stratified split
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


tfidf = TfidfVectorizer(
    ngram_range=(1,2),
    min_df=3,
    max_features=10000
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf   = tfidf.transform(X_val)


# 1. Logistic Regression
modelLR = LogisticRegression(max_iter=1000, class_weight="balanced")
modelLR.fit(X_train_tfidf, y_train)

y_val_pred_lr = modelLR.predict(X_val_tfidf)
print("Logistic Regression F1:", f1_score(y_val, y_val_pred_lr))
print(classification_report(y_val, y_val_pred_lr))

# 2. Linear SVM
modelSVM = SGDClassifier(loss="hinge", class_weight="balanced", max_iter=1000)
modelSVM.fit(X_train_tfidf, y_train)

y_val_pred_svm = modelSVM.predict(X_val_tfidf)
print("Linear SVM F1:", f1_score(y_val, y_val_pred_svm))
print(classification_report(y_val, y_val_pred_svm))

# 3. Multinomial Naive Bayes
modelNB = MultinomialNB() 
modelNB.fit(X_train_tfidf, y_train)

y_val_pred_nb = modelNB.predict(X_val_tfidf)
print("Naive Bayes F1:", f1_score(y_val, y_val_pred_nb))
print(classification_report(y_val, y_val_pred_nb))

# 4. Random Forest
modelRF = RandomForestClassifier(
    n_estimators=300,          
    max_depth=10,             
    class_weight='balanced',  
    random_state=42,
    n_jobs=-1
)
modelRF.fit(X_train_tfidf, y_train)

y_val_pred_rf = modelRF.predict(X_val_tfidf)
print("Random Forest F1:", f1_score(y_val, y_val_pred_rf))
print(classification_report(y_val, y_val_pred_rf))

# XGBoost
neg = np.sum(y_val == 0)
pos = np.sum(y_val == 1)
ratio = neg / pos

modelXGB = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=ratio,  # for imbalaced class
    random_state=42,
    n_jobs=-1,
)
modelXGB.fit(X_train_tfidf, y_train)

y_val_pred_xg = modelXGB.predict(X_val_tfidf)
print("XGBoost F1:", f1_score(y_val, y_val_pred_xg))
print(classification_report(y_val, y_val_pred_xg))


# Majority voting
final_pred1 = np.round((y_val_pred_lr + y_val_pred_nb + y_val_pred_rf) / 3) # focusing on high recall for class "1"
print("Ensemble F1:", f1_score(y_val, final_pred1))
print(classification_report(y_val, final_pred1)) 

final_pred2 = np.round((y_val_pred_lr + y_val_pred_svm + y_val_pred_nb + y_val_pred_rf + y_val_pred_xg) / 5)
print("Ensemble F1:", f1_score(y_val, final_pred2))
print(classification_report(y_val, final_pred2))

# Weighted voting
final_pred = np.round((0.3*y_val_pred_lr + 0.2*y_val_pred_svm + 0.3*y_val_pred_nb + 0.2*y_val_pred_xg)) #Pretty random weights...didnt experiment much
print("Weighted Ensemble F1:", f1_score(y_val, final_pred))
print(classification_report(y_val, final_pred)) 


test = pd.read_csv('/kaggle/input/comments-classification/Dataset/test.csv')
test.isnull().sum()
test= normalize_text(test)


X_test = test["comment_text"]
X_test_tfidf= tfidf.transform(X_test)
y_test_pred_LR = modelLR.predict(X_test_tfidf)
y_test_pred_SVM = modelSVM.predict(X_test_tfidf)
y_test_pred_NB = modelNB.predict(X_test_tfidf)
y_test_pred_RF = modelRF.predict(X_test_tfidf)
y_test_pred_XGB = modelXGB.predict(X_test_tfidf)


# Majority voting
final_pred_maj_vote1 = np.round((y_test_pred_LR +y_test_pred_NB + y_test_pred_RF) / 3)
final_pred_maj_vote2 = np.round((y_test_pred_LR + y_test_pred_SVM + y_test_pred_NB + y_test_pred_RF + y_test_pred_XGB) / 5)

# Weighted voting
final_pred_weigh_vote = np.round((0.3*y_test_pred_LR + 0.2*y_test_pred_SVM + 0.3*y_test_pred_NB + 0.2*y_test_pred_XGB))


ids = list(range(1, len(y_test_pred_LR) + 1))
submission_LR = pd.DataFrame({
    "ID": ids,
    "psychotic_depression": y_test_pred_LR
})

submission_SVM = pd.DataFrame({
    "ID": ids,
    "psychotic_depression": y_test_pred_SVM
})

submission_NB = pd.DataFrame({
    "ID": ids,
    "psychotic_depression": y_test_pred_NB
})

submission_rf = pd.DataFrame({
    "ID": ids,
    "psychotic_depression": y_test_pred_RF
})

submission_xgb = pd.DataFrame({
    "ID": ids,
    "psychotic_depression":  y_test_pred_XGB
})

submission_maj_vote1 = pd.DataFrame({
    "ID": ids,
    "psychotic_depression": final_pred_maj_vote1
})

submission_maj_vote2 = pd.DataFrame({
    "ID": ids,
    "psychotic_depression": final_pred_maj_vote2
})

submission_weigh_vote = pd.DataFrame({
    "ID": ids,
    "psychotic_depression": final_pred_weigh_vote
})


# Kaggle score for each model

submission_LR = submission_LR.sort_values(by="ID")
submission_LR.to_csv("submission_LR.csv", index=False) # 0.49

submission_SVM = submission_SVM.sort_values(by="ID")
submission_SVM.to_csv("submission_SVM.csv", index=False) # 0.49

submission_NB = submission_NB.sort_values(by="ID")
submission_NB.to_csv("submission_NB.csv", index=False) # 0.52

submission_rf = submission_rf.sort_values(by="ID")
submission_rf.to_csv("submission_rf.csv", index=False) # 0.57

submission_xgb = submission_xgb.sort_values(by="ID")
submission_xgb.to_csv("submission_xgb.csv", index=False) # 0.47

submission_maj_vote1 = submission_maj_vote1.sort_values(by="ID")
submission_maj_vote1.to_csv("submission_maj_vote1.csv", index=False) # 0.52

submission_maj_vote2 = submission_maj_vote2.sort_values(by="ID")
submission_maj_vote2.to_csv("submission_maj_vote2.csv", index=False) # 0.50

submission_weigh_vote = submission_weigh_vote.sort_values(by="ID")
submission_weigh_vote.to_csv("submission_weigh_vote.csv", index=False) # 0.49




