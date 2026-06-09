import tensorflow as tf
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Dropout, Activation, Embedding, Bidirectional,SimpleRNN,Normalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
import nltk
import warnings
warnings.filterwarnings('ignore')
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))


train_df=pd.read_csv('train.csv')
test_df=pd.read_csv('test.csv')
train_df.sample(3)


print(f"Train DataSet Info : \n {train_df.info()}")
print("--------------------------------------------")
print(f"Test DataSet Info : \n {test_df.info()}")


print(f"Train DataSet Dublicated : {train_df['body'].duplicated().sum()}")
print(f"Test DataSet Dublicated : {test_df['body'].duplicated().sum()}")


train_df=train_df.drop_duplicates(subset=['body'])
test_df=test_df.drop_duplicates(subset=['body'])


dataset=train_df[['body','rule_violation']]
dataset.head()


label=[0,1]

for index,value in enumerate(label):
    d_=train_df[train_df['rule_violation']==value]
    print(f"Body : \n {d_['body'].values[0]}")
    print(f"Rule : \n {d_['rule'].values[0]}")
    print(f"Rule Violation : \n {d_['rule_violation'].values[0]}")
    print("-----------------------------------------------------")


dataset['rule_violation'].value_counts().plot(kind='bar')


url_pattern = r'https?://\S+|www\.\S+'
html_pattern = r'<.*?>'
pattern = r'[^a-zA-z\s]'
between_list=r'\[[^\]]*\]'
many_space='  +'
dataset['url']=dataset['body'].apply(lambda x:len(re.findall(url_pattern,x)))
dataset.head()


url_check={}
for i in label:
   d_init=dataset[dataset['rule_violation']==i]
   url_check[f'When_{i}']=d_init['url'].sum()
pd.Series(url_check).plot(kind='bar',title="Number Of Links In 1 an 0")


dataset['body']=dataset['body'].apply(lambda x:re.sub(url_pattern,'',x))
dataset['body']=dataset['body'].apply(lambda x:re.sub(pattern,' ',x))
dataset['body']=dataset['body'].apply(lambda x:re.sub('\n','',x))
dataset['body']=dataset['body'].apply(lambda x:re.sub(html_pattern,'',x))
dataset['body']=dataset['body'].apply(lambda x:re.sub(between_list,'',x))
dataset['body']=dataset['body'].apply(lambda x:re.sub(many_space,' ',x))
dataset.head()


dataset['number_of_words']=dataset['body'].apply(lambda x:len(x.split()))
print(f"Max Number Of Words : {dataset['number_of_words'].max()}")
print(f"Min Number Of Words : {dataset['number_of_words'].min()}")


dataset=dataset[dataset['number_of_words']>=4]
print(f"Max Number Of Words : {dataset['number_of_words'].max()}")
print(f"Min Number Of Words : {dataset['number_of_words'].min()}")


text=" ".join(dataset['body'].values)
text


vocap_size=len(set(text.split()))
oov_token='<OOV>'
max_seq=dataset['number_of_words'].max()
embedding_dim=64
test_pre=0.2


tokenization=Tokenizer(num_words=vocap_size,oov_token=oov_token)
# Split Dataset For Train and Test
x_train,x_test,y_train,y_test=train_test_split(dataset['body'],dataset['rule_violation'],test_size=test_pre,random_state=44)
#Apply Tokenization
tokenization.fit_on_texts(x_train)
tokenization.fit_on_texts(x_test)


tokenization.word_index


# Apply Text TO Sequances
x_train_seq=tokenization.texts_to_sequences(x_train)
x_test_seq=tokenization.texts_to_sequences(x_test)
# Reset index to access elements by integer index
x_train = x_train.reset_index(drop=True)
x_test = x_test.reset_index(drop=True)
print(f"Train AS Words : {x_train[0]}")
print(f"Train Sequence : {x_train_seq[0]}")
print(f"Train Len {len(x_train[0].split())}  , {len(x_train_seq[0])}")
print("------------------------------------")
print(f"Test AS Words : {x_test[0]}")
print(f"Test Sequence : {x_test_seq[0]}")
print(f"Train Len {len(x_test[0].split())}  , {len(x_test_seq[0])}")


x_train_padding=pad_sequences(x_train_seq,maxlen=max_seq,padding='post',truncating='post')
x_test_padding=pad_sequences(x_test_seq,maxlen=max_seq,padding='post',truncating='post')
print(f"Train AS Words : {x_train[0]}")
print(f"Train Sequence : {x_train_seq[0]}")
print(f"Train Padding : {x_train_padding[0]}")
print(f"Train Len {len(x_train[0].split())}  , Train Seq {len(x_train_seq[0])} , Padding {len(x_train_padding[0])}")
print("------------------------------------")
print(f"Test AS Words : {x_test[0]}")
print(f"Test Sequence : {x_test_seq[0]}")
print(f"Train Padding : {x_test_padding[0]}")
print(f"Test Len {len(x_test[0].split())}  , Test Seq {len(x_test_seq[0])}, Padding {len(x_train_padding[0])}")


y_train=np.array(y_train)
y_test=np.array(y_test)


print(f"x_train_padding_shape : {x_train_padding.shape}")
print(f"y_train_shape : {y_train.shape}")
print(f"x_test_padding_shape : {x_test_padding.shape}")
print(f"y_test_shape : {y_test.shape}")


type(x_train_padding),type(y_train)


lstm_steps=128
dropout=0.35
epochs=20
num_class=1
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,decay=1e-6)
loss=tf.keras.losses.BinaryCrossentropy()
metrics=['accuracy']

model=Sequential()
model.add(Embedding(vocap_size,embedding_dim))
model.add(Dropout(dropout))
model.add(Bidirectional(LSTM(lstm_steps,return_sequences=False)))
model.add(Dropout(dropout))
model.add(Dense(num_class,activation='sigmoid'))

model.summary()


model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
model.fit(x_train_padding,y_train,epochs=epochs,validation_data=(x_test_padding,y_test))


print(f"Train Accuracy : {model.evaluate(x_train_padding,y_train)[1]}")
print(f"Test Accuracy : {model.evaluate(x_test_padding,y_test)[1]}")


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Assuming 'dataset' DataFrame and 'body', 'rule_violation' columns are available from previous cells

# Split the data into training and testing sets
X = dataset['body']
y = dataset['rule_violation']
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(X, y, test_size=0.2, random_state=44)

# Initialize TfidfVectorizer
tfidf = TfidfVectorizer()

# Fit and transform the training data, transform the testing data
X_train_tfidf = tfidf.fit_transform(X_train_tfidf)
X_test_tfidf = tfidf.transform(X_test_tfidf)

# Initialize and train the RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=500,max_depth=50, random_state=44)
random_forest.fit(X_train_tfidf, y_train_tfidf)

# Make predictions and evaluate the model
y_pred_tfidf = random_forest.predict(X_test_tfidf)
print(f"Score For Training : {random_forest.score(X_train_tfidf,y_train_tfidf)}")
print(f"Score For Testing : {random_forest.score(X_test_tfidf,y_test_tfidf)}")
print(f"Accuracy with RandomForest and TF-IDF: {accuracy_score(y_test_tfidf, y_pred_tfidf)}")


def predict_rule_violation(text):
   pre_tfidf=tfidf.transform([text])
   return random_forest.predict(pre_tfidf)[0]
test_df['predict_rule_violation']=test_df['body'].apply(predict_rule_violation)
test_df.head()


submission=test_df[['row_id','predict_rule_violation']]
submission.to_csv('submission.csv',index=False)

