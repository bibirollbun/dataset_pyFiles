
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib



from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score , classification_report
from sklearn.svm import LinearSVC



train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
test  = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')


train.head()


train['StudentExplanation'] = train['StudentExplanation'].fillna('No explaination')
test['StudentExplanation'] = test['StudentExplanation'].fillna('No explaination')
train['QuestionText'] = train['QuestionText'].fillna('')
test['QuestionText'] = test['QuestionText'].fillna('')
train['MC_Answer'] = train['MC_Answer'].fillna('')
test['MC_Answer'] = test['MC_Answer'].fillna('')

train['Category'] = train['Category'].fillna('UnknownCategory')
train['Misconception'] = train['Misconception'].fillna('UnknownMisconception')


train['text'] = (train['QuestionText']+" " + train['MC_Answer'] + " " + train['StudentExplanation'])
train['text'][0]


x = train['text']
y = train['Category']

x_train , x_val, y_train , y_val = train_test_split( x ,y , random_state = 42 , test_size = 0.2)


vectorizer = TfidfVectorizer(max_features = 5000)
x_train_tfidf = vectorizer.fit_transform(x_train)
x_val_tfidf   = vectorizer.transform(x_val)


from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)


num_classes = len(le.classes_)
y_train_cat = to_categorical(y_train_enc, num_classes)
y_val_cat = to_categorical(y_val_enc, num_classes)


model = Sequential()
model.add(Embedding(20000, 128, input_length=200))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(num_classes, activation='softmax'))

max_words = 20000
max_len = 200
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(x_train)
X_train_seq = pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen=max_len)
X_val_seq = pad_sequences(tokenizer.texts_to_sequences(x_val), maxlen=max_len)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_seq, y_train_cat, validation_data=(X_val_seq, y_val_cat), epochs=3, batch_size=64)


test.head()


test['text'] = (test['QuestionText'] +" "+test['MC_Answer']+" "+test['StudentExplanation'])




test["text"] = (
    test["QuestionText"].fillna("") + " " +
    test["MC_Answer"].fillna("") + " " +
    test["StudentExplanation"].fillna("")
)

X_test_seq = pad_sequences(tokenizer.texts_to_sequences(test["text"]), maxlen=max_len)

y_pred_probs = model.predict(X_test_seq)
y_pred = y_pred_probs.argmax(axis=1)
cat_labels = le.inverse_transform(y_pred)

mis_labels = ["NA"] * len(cat_labels)

final_preds = [f"{c}:{m}" for c, m in zip(cat_labels, mis_labels)]

submission = pd.DataFrame({
    "row_id": test.index,
    "Category:Misconception": final_preds
})

submission.to_csv("submission.csv", index=False)
sub = pd.read_csv('submission.csv')
sub


sub = pd.read_csv('submission.csv')
sub






















