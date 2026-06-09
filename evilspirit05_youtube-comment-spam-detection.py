! pip install Unidecode


import pandas as pd
import numpy as np
import re
import nltk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Reshape

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
confusion_matrix, classification_report)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score, accuracy_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
warnings.filterwarnings('ignore')
from unidecode import unidecode
# Télécharger NLTK data (exécutez une fois)
nltk.download('stopwords')
nltk.download('punkt')
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
import tensorflow as tf
tf.config.optimizer.set_jit(False)  # Disable XLA completely


!rm -rf /kaggle/working/*


df=pd.read_csv("/kaggle/input/spams-dans-les-commentaires-yt/train_yt.csv")


df.head()


df.drop(columns=["Unnamed: 0","COMMENT_ID","DATE"],axis=1,inplace=True)


df.shape


df.isnull().sum()


df["text"]=df["AUTHOR"]+" "+df["CONTENT"]+" "+df["VIDEO_NAME"]


df.drop(columns=["AUTHOR","CONTENT","VIDEO_NAME"],axis=1,inplace=True)


df.head()


df.isnull().sum()


# def clean_text(text):
#     if pd.isnull(text):
#         return ""
#     text = str(text)
#     text = text.lower()
#     # remove accents
#     text = unidecode(text)
#     # remove urls and mentions
#     text = re.sub(r"http\S+|www\S+", ' ', text)
#     text = re.sub(r"@\w+", ' ', text)
#     # keep letters and numbers
#     text = re.sub(r"[^a-z0-9\s]", ' ', text)
#     # remove digits (optional)
#     text = re.sub(r"\d+", ' ', text)
#     # collapse spaces
#     text = re.sub(r"\s+", ' ', text).strip()
#     return text

# df['text'] = df['text'].astype(str).apply(clean_text)


X = df['text']  # Ou combinez avec features si vous voulez
y = df['CLASS']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train split: {X_train.shape}, Val: {X_val.shape}")


# from nltk.corpus import stopwords
# english_stop = stopwords.words('english')
# vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

# X_train_tfidf = vectorizer.fit_transform(X_train)
# X_val_tfidf = vectorizer.transform(X_val)


# lr = LogisticRegression(solver='saga', penalty='l2', C=1.0, max_iter=2000, random_state=SEED)
# lr.fit(X_train_tfidf, y_train)
# y_pred = lr.predict(X_val_tfidf)



# acc = accuracy_score(y_val, y_pred)
# print(f"Accuracy sur Validation: {acc:.4f}")
# print("\nClassification Report:")
# print(classification_report(y_val, y_pred))


test=pd.read_csv("/kaggle/input/spams-dans-les-commentaires-yt/test_yt.csv")
test.drop(columns=["COMMENT_ID","DATE"],axis=1,inplace=True)
test.rename(columns={"Unnamed: 0":"ID"},inplace=True)
Id=test.ID
test.drop(columns=["ID"],axis=1,inplace=True)
test["text"]=test["AUTHOR"]+" "+test["CONTENT"]+" "+test["VIDEO_NAME"]
test.drop(columns=["AUTHOR","CONTENT","VIDEO_NAME"],axis=1,inplace=True)


test.head()


# test['text'] = test['text'].astype(str).apply(clean_text)


# preds=vectorizer.transform(test["text"])
# predict=lr.predict(preds)
# submission = pd.DataFrame({'ID': Id, 'CLASS': predict})
# submission.to_csv('submission.csv', index=False)
# print('Saved submission.csv')
# submission.head()


# vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1,2))

# X_train_tfidf = vectorizer.fit_transform(X_train)
# X_val_tfidf = vectorizer.transform(X_val)


# from catboost import CatBoostClassifier
# from sklearn.naive_bayes import MultinomialNB

# mnb=MultinomialNB(alpha=1.0) 
# mnb.fit(X_train_tfidf, y_train)
# y_pred = mnb.predict(X_val_tfidf)


# print("Validation Accuracy:", accuracy_score(y_val, y_pred))
# print(classification_report(y_val, y_pred))


# tf=vectorizer.transform(test["text"])
# predict=mnb.predict(tf)
# submission = pd.DataFrame({'ID': Id, 'CLASS': predict})
# submission.to_csv('mnb_submission.csv', index=False)
# print('Saved submission.csv')
# submission.head()


# vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

# X_train_tfidf = vectorizer.fit_transform(X_train)
# X_val_tfidf = vectorizer.transform(X_val)


# from sklearn.svm import LinearSVC

# svc = LinearSVC()
# svc.fit(X_train_tfidf, y_train)
# y_pred = svc.predict(X_val_tfidf)
# acc = accuracy_score(y_val, y_pred)
# print(f"Accuracy sur Validation: {acc:.4f}")
# print("\nClassification Report:")
# print(classification_report(y_val, y_pred))


# tf=vectorizer.transform(test["text"])
# predict= svc.predict(tf)
# submission = pd.DataFrame({'ID': Id, 'CLASS': predict})
# submission.to_csv('svm_submission.csv', index=False)
# print('Saved submission.csv')
# submission.head()


X = df['text']  
y = df['CLASS']

test = pd.read_csv("/kaggle/input/spams-dans-les-commentaires-yt/test_yt.csv")
test.rename(columns={"Unnamed: 0": "ID"}, inplace=True)
Id = test.ID
test["text"] = test["AUTHOR"] + " " + test["CONTENT"] + " " + test["VIDEO_NAME"]
test_text = test["text"]


vectorizer = TfidfVectorizer(ngram_range=(1,2),max_features=5000) #ngram_range=(1, 2)
X_tfidf = vectorizer.fit_transform(X)
test_tfidf = vectorizer.transform(test_text)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scores = []
acc_scores = []
test_preds = np.zeros(len(test))
fold = 1


for train_idx, val_idx in skf.split(X_tfidf, y):
    print(f"Starting Fold {fold}.......................................")
    
    X_train_fold = X_tfidf[train_idx]
    y_train_fold = y.iloc[train_idx].values
    
    X_val_fold = X_tfidf[val_idx]
    y_val_fold = y.iloc[val_idx].values
    
    input_dim = X_train_fold.shape[1]
    
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=0.1)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
    
    history = model.fit(
        X_train_fold, y_train_fold,
        validation_data=(X_val_fold, y_val_fold),
        epochs=200,
        batch_size=256,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    y_pred_val = (model.predict(X_val_fold) > 0.5).astype(int).flatten()
    f1 = f1_score(y_val_fold, y_pred_val)
    acc = accuracy_score(y_val_fold, y_pred_val)
    
    print(f"Fold {fold} - F1: {f1:.4f} - Accuracy: {acc:.4f}\n")
    
    f1_scores.append(f1)
    acc_scores.append(acc)
    
    fold_test_preds = model.predict(test_tfidf).flatten()
    test_preds += fold_test_preds / skf.n_splits
    
    fold += 1


print("Cross-Validation Results")
print(f"Mean F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"Mean Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")


final_predict = (test_preds >= 0.5).astype(int)
submission = pd.DataFrame({'ID': Id, 'CLASS': final_predict})
submission.to_csv('neu_submission.csv', index=False)
print('Saved neu_submission.csv')
submission.head()





