import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


train=pd.read_csv("/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip", header=0, delimiter="\t")
test=pd.read_csv("/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip", header=0, delimiter="\t")
unlabeled_train=pd.read_csv("/kaggle/input/word2vec-nlp-tutorial/unlabeledTrainData.tsv.zip", header=0, delimiter="\t", quoting=3)


train.head()


train.shape


train.info()


train.isnull().sum()


train['sentiment'].value_counts()


sns.countplot(x=train['sentiment'], palette=['salmon', 'skyblue'])
plt.title('Positive vs. Negative Reviews')
plt.xlabel('0=Negative, 1=Positive')
plt.ylabel('Review');


test.head()


test.shape


test.info()


test.isnull().sum()


unlabeled_train.head()


unlabeled_train.shape


unlabeled_train.info()


unlabeled_train.isnull().sum()


def process_dataframes(df, column_name='review'):
    df['text_clean']=df[column_name].str.lower() #Lowercase
    df['text_clean']=df['text_clean'].str.replace(r'<.*?>', ' ', regex=True) #HTML tags
    df['text_clean']=df['text_clean'].str.replace(r'https?://\S+|www\.\S+', ' ', regex=True) #URL
    df['text_clean']=df['text_clean'].str.replace(r'\[.*?\]', ' ', regex=True) #Square brackets
    df['text_clean']=df['text_clean'].str.replace(r'\w*\d\w*', ' ', regex=True) #Words containing numbers
    df['text_clean']=df['text_clean'].str.replace(r'\s\d+\s', ' ', regex=True) #Numbers
    df['text_clean']=df['text_clean'].str.replace(r'[^a-zA-Z]', ' ', regex=True) #Special characters
    df['text_clean']=df['text_clean'].str.replace(r'\n', ' ', regex=True) #Newline
    df['text_clean']=df['text_clean'].str.replace(r'\s+', ' ', regex=True) #Spaces
    return df

train=process_dataframes(train)
test=process_dataframes(test)
unlabeled_train=process_dataframes(unlabeled_train)


train.head()


test.head()


unlabeled_train.head()


nltk.download('stopwords')
stop_words=list(stopwords.words('english'))


# TF-IDF
x_train_text=train['text_clean']
y_train=train['sentiment']
x_test_text=test['text_clean']

vectorizer=TfidfVectorizer(stop_words=stop_words,max_features=5000,ngram_range=(1, 2))

x_train_vector=vectorizer.fit_transform(x_train_text)
x_test_vector=vectorizer.transform(x_test_text)


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd

def compare_models_cv(x, y, cv=5, scoring='roc_auc'):
    
    modeller = [
        BernoulliNB(),
        LogisticRegression(solver='liblinear', random_state=42), 
        MultinomialNB(),
        RandomForestClassifier(random_state=42),
        GradientBoostingClassifier(random_state=42),
        AdaBoostClassifier(random_state=42)]
    
    isimler = [
        "BernoulliNB", 
        "Logistic Regression", 
        "MultinomialNB", 
        "Random Forest",
        "Gradient Boosting", 
        "AdaBoost"]
    
    cv_sonuclari = []
    
    print(f"âœ… Modeller {cv} katlÄ± Ã‡apraz DoÄŸrulama ({scoring} metriÄŸi) ile karÅŸÄ±laÅŸtÄ±rÄ±lÄ±yor...")
    
    for model, isim in zip(modeller, isimler):
        print(f"    -> {isim} deneniyor...")
        
        # n_jobs=-1 tÃ¼m Ã§ekirdekleri kullanarak iÅŸlemi hÄ±zlandÄ±rÄ±r
        cv_skorlari = cross_val_score(model, x, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        # Hata dÃ¼zeltildi: .append() kullanÄ±ldÄ± ve girinti ayarlandÄ±
        cv_sonuclari.append({ 
            'Model': isim,
            f'Ortalama CV {scoring.upper()}': np.mean(cv_skorlari),
            f'CV {scoring.upper()} Std. Dev.': np.std(cv_skorlari) 
        })
        
    metrics_df = pd.DataFrame(cv_sonuclari)
    metrics_df.sort_values(f'Ortalama CV {scoring.upper()}', ascending=False, inplace=True)
    metrics_df.reset_index(drop=True, inplace=True)
    
    print("\n\nğŸ�† KarÅŸÄ±laÅŸtÄ±rma TamamlandÄ±.")
    
    return metrics_df


compare_models_cv(x_train_vector, y_train, cv=5)


final_model=LogisticRegression(solver='liblinear', random_state=42)


final_model.fit(x_train_vector, y_train)
predictions=final_model.predict(x_test_vector)


submission = pd.DataFrame({"id": test["id"],"sentiment": predictions})
submission.to_csv("submission.csv", index=False)


nltk.download('punkt')


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


train_df=pd.read_csv("/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip", header=0, delimiter="\t")
test_df=pd.read_csv("/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip", header=0, delimiter="\t")
unlabeled_df=pd.read_csv("/kaggle/input/word2vec-nlp-tutorial/unlabeledTrainData.tsv.zip", header=0, delimiter="\t", quoting=3)


 all_reviews=pd.concat([train_df['review'], unlabeled_df['review']]).reset_index(drop=True)


all_reviews.head()


all_reviews.shape


from bs4 import BeautifulSoup
import re


def review_to_sentences(review, tokenizer):
    review_text = BeautifulSoup(review, "html.parser").get_text() #HTML
    raw_sentences = tokenizer.tokenize(review_text.strip()) #tokenize
    
    sentences = [] #word list
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            clean_sentence = re.sub("[^a-zA-Z]", " ", raw_sentence)
            sentences.append(clean_sentence.lower().split()) #Lowercase
    return sentences

sentences = []
for review in all_reviews:
    sentences += review_to_sentences(review, tokenizer)


len(sentences)


sentences[0]


from gensim.models import word2vec
import logging


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#Parameters
num_features = 300    
min_word_count = 40   
num_workers = 4       
context = 10          
downsampling = 1e-3  

model_w2v = word2vec.Word2Vec(sentences,workers=num_workers,vector_size=num_features,
                              min_count=min_word_count,window=context,sample=downsampling)

model_name = "300features_40minwords_10context"
model_w2v.save(model_name)

print("\nTest: woman + king - man = ?")
print(model_w2v.wv.most_similar(positive=['woman', 'king'], negative=['man']))

print("\nTest: 'bad':")
print(model_w2v.wv.most_similar("bad"))


#Feature Engineering
def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0
    index2word_set = set(model.wv.index_to_key)
    
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1
            featureVec = np.add(featureVec, model.wv[word])
    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):    
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    
    for review in reviews:
        if counter % 5000 == 0:
            print(f"Review: {counter}")
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter = counter + 1
    return reviewFeatureVecs


def quick_clean(df):
    df['clean_text'] = df['review'].str.lower()
    df['clean_text'] = df['clean_text'].str.replace(r'<.*?>', ' ', regex=True) 
    df['clean_text'] = df['clean_text'].str.replace(r'[^a-zA-Z]', ' ', regex=True) 
    df['clean_text'] = df['clean_text'].str.replace(r'\s+', ' ', regex=True) 
    return df

train_df=quick_clean(train_df)
test_df=quick_clean(test_df)


clean_train_reviews = []
for review in train_df["review"]:
    clean_train_reviews.append( review_to_sentences(review, tokenizer) )

train_words=[text.split() for text in train_df['clean_text']]
test_words=[text.split() for text in test_df['clean_text']]

trainDataVecs=getAvgFeatureVecs(train_words, model_w2v, num_features)
testDataVecs=getAvgFeatureVecs(test_words, model_w2v, num_features)


trainDataVecs.shape


testDataVecs.shape


y_train=train_df['sentiment']


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd

def compare_models_w2v(X, y, cv=5, scoring='roc_auc'):
    
    modeller = [
        LogisticRegression(solver='liblinear', random_state=42),
        RandomForestClassifier(n_estimators=100, random_state=42),
        GradientBoostingClassifier(random_state=42),
        AdaBoostClassifier(random_state=42)]
    
    isimler = [
        "Logistic Regression", 
        "Random Forest",
        "Gradient Boosting", 
        "AdaBoost"]
    
    cv_sonuclari = []
    
    print(f"âœ… Modeller Word2Vec verisi ({cv}) test ediliyor...")
    
    for model, isim in zip(modeller, isimler):
        print(f"   -> {isim} deneniyor...")
        try:
            cv_skorlari = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            cv_sonuclari.append({
                'Model': isim,
                f'Ortalama CV {scoring.upper()}': np.mean(cv_skorlari),
                f'Std. Dev.': np.std(cv_skorlari)
            })
        except Exception as e:
            print(f"      âš ï¸� Hata ({isim}): {e}")
        
    metrics_df = pd.DataFrame(cv_sonuclari)
    if not metrics_df.empty:
        metrics_df.sort_values(f'Ortalama CV {scoring.upper()}', ascending=False, inplace=True)
        metrics_df.reset_index(drop=True, inplace=True)
    
    print("\n KarÅŸÄ±laÅŸtÄ±rma TamamlandÄ±.")
    return metrics_df


compare_models_w2v(trainDataVecs, y_train, cv=5)


final_model=LogisticRegression(solver='liblinear', random_state=42)
final_model.fit(trainDataVecs, y_train)
predictions=final_model.predict(testDataVecs)
submission = pd.DataFrame({"id": test_df["id"],"sentiment": predictions})
submission.to_csv("submission.csv", index=False)


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Parameters
MAX_WORDS = 20000       
MAX_LEN = 200           
EMBEDDING_DIM = 300     

#Tokenizer
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(train_df['clean_text']) 
word_index = tokenizer.word_index

#Text to Sequences
x_train_seq = tokenizer.texts_to_sequences(train_df['clean_text'])
x_test_seq = tokenizer.texts_to_sequences(test_df['clean_text'])

x_train_pad = pad_sequences(x_train_seq, maxlen=MAX_LEN)
x_test_pad = pad_sequences(x_test_seq, maxlen=MAX_LEN)

y_train = train_df['sentiment'].values


x_train_pad.shape


num_words = min(MAX_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

found_count = 0
for word, i in word_index.items():
    if i >= MAX_WORDS:
        continue
    if word in model_w2v.wv:
        embedding_matrix[i] = model_w2v.wv[word]
        found_count += 1


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

model_lstm = Sequential()
model_lstm.add(Embedding(
    input_dim=num_words,
    output_dim=EMBEDDING_DIM,
    weights=[embedding_matrix], 
    input_length=MAX_LEN,
    trainable=False))

#LSTM 
model_lstm.add(Bidirectional(LSTM(units=128, return_sequences=False)))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(1, activation='sigmoid'))

model_lstm.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

model_lstm.summary()

history = model_lstm.fit(x_train_pad,y_train,batch_size=128,epochs=5,validation_split=0.2)


from tensorflow.keras.optimizers import Adam

model_lstm.layers[0].trainable = True
model_lstm.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.00005), metrics=['accuracy'])
model_lstm.summary() 

history_fine = model_lstm.fit(x_train_pad,y_train,batch_size=128,epochs=2,validation_split=0.2)


y_pred_prob = model_lstm.predict(x_test_pad, batch_size=128, verbose=1)
y_pred = (y_pred_prob >= 0.5).astype(int)

submission=pd.DataFrame({"id": test_df["id"],"sentiment": y_pred.flatten()})
submission.to_csv("submission.csv", index=False)


import pickle
from tensorflow.keras.models import save_model

model_lstm.save("sentiment_lstm_model.h5")
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

