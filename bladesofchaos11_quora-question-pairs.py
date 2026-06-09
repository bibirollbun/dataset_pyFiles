import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import zipfile


import gensim
from gensim.models import Word2Vec,KeyedVectors
from nltk import sent_tokenize
from gensim.utils import simple_preprocess


zip_files = ['/kaggle/input/quora-question-pairs/train.csv.zip', 
             '/kaggle/input/quora-question-pairs/test.csv.zip', 
             '/kaggle/input/quora-question-pairs/sample_submission.csv.zip']

working_dir = '/kaggle/working/'

for zip_file in zip_files:
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(working_dir)

print("Unzipped files.")


train_df = pd.read_csv("/kaggle/working/train.csv")
test_df = pd.read_csv("/kaggle/working/test.csv")


train_df.head()


print("Train dataframe:", train_df.shape)
print("Test dataframe:", test_df.shape)


train_df.info()


# drop missing values
train_df = train_df.dropna()
train_df.isnull().sum()


# duplicate rows
train_df.duplicated().sum()


# Distribution of duplicate vs non-duplicate pairs

print(train_df['is_duplicate'].value_counts())
print((train_df['is_duplicate'].value_counts()/train_df['is_duplicate'].count())*100)
train_df['is_duplicate'].value_counts().plot(kind='bar')


# Repeated questions

qid = pd.Series(train_df['qid1'].tolist() + train_df['qid2'].tolist())
print('Number of unique questions',np.unique(qid).shape[0])
x = qid.value_counts()>1
print('Number of questions getting repeated',x[x].shape[0])


# Log histogram of repeated questions

plt.hist(qid.value_counts().values,bins=160, color='r')
plt.yscale('log')
plt.show()


# Creating corpus

corpus = train_df['question1'].tolist() + train_df['question2'].tolist()

story = []
for sent in tqdm(corpus):
    story.append(simple_preprocess(sent))


import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download("stopwords")


stemmer = PorterStemmer()
def process_word_list(word_list):
    meaningful_words = [
        stemmer.stem(word)  # Applying stemming
        for word in word_list
        if word.lower() not in stopwords.words("english")  # Removing stopwords
    ]
    return meaningful_words

cleaned_story = [process_word_list(sublist) for sublist in tqdm(story)]


word2vec_model = gensim.models.Word2Vec(window = 5, min_count = 2, vector_size=30)
word2vec_model.build_vocab(cleaned_story)
word2vec_model.train(cleaned_story, 
            total_examples = word2vec_model.corpus_count, 
            epochs = word2vec_model.epochs)


train_df["Ques1_Cleaned"] = cleaned_story[:len(train_df)]
train_df["Ques2_Cleaned"] = cleaned_story[len(train_df):]


sample_data = train_df.copy() 
sample_data.reset_index(inplace = True, drop = True)


from textblob import TextBlob

def text_analysis(word_list):
    # Join the list of words into a text string
    text = " ".join(word_list)

    # Create a TextBlob object
    blob = TextBlob(text)

    # Sentiment Analysis
    polarity_score = blob.sentiment.polarity
    subjectivity_score = blob.sentiment.subjectivity

    # Word Metrics
    word_count = len(word_list)
    average_word_length = sum(len(word) for word in word_list) / word_count

    # Sentence Metrics
    sentence_count = len(blob.sentences)
    average_sentence_length = word_count / sentence_count

    # FOG Index
    complex_word_count = len([word for word in word_list if len(word) > 3])
    fog_index = 0.4 * (average_sentence_length + complex_word_count / word_count)
    
    return np.array([average_word_length, word_count, average_sentence_length, fog_index, complex_word_count])


# Converting text columns into vectors

vec1 = np.zeros((len(sample_data), 30))
vec2 = np.zeros((len(sample_data), 30))
other_features = np.zeros((len(sample_data), 10))

for i in tqdm(range(len(sample_data))):
    
    text1 = sample_data["Ques1_Cleaned"][i]
    text2 = sample_data["Ques2_Cleaned"][i]
    
    if len(text1) != 0:
        vec1[i] = (sum([word2vec_model.wv[word] for word in text1 if word in word2vec_model.wv.index_to_key]) / len(text1))
        blob1 = text_analysis(text1)
    else:
        vec1[i] = np.zeros((30,))
        blob1 = np.zeros((5,))
    if len(text2) != 0:
        vec2[i] = (sum([word2vec_model.wv[word] for word in text2 if word in word2vec_model.wv.index_to_key]) / len(text2))
        blob2 = text_analysis(text2)
    else:
        vec2i = np.zeros((30,))
        blob2 = np.zeros((5,))
    other_features[i] = np.concatenate([blob1, blob2])


sample_data = pd.concat([sample_data, 
                        pd.DataFrame(vec1, columns = np.arange(0, 30)),
                        pd.DataFrame(vec2, columns = np.arange(30, 60)),
                        pd.DataFrame(other_features, columns = np.arange(60, 70))], axis = 1)
sample_data.head()


pip install scikit-learn==1.2.2


pip install imbalanced-learn==0.12.2


# SMOTE Oversampling (because data is imbalanced)

from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='auto', random_state=42)
xresampled, yresampled = smote.fit_resample(sample_data.iloc[:,8:], sample_data["is_duplicate"])
xresampled.shape, yresampled.shape


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

X_train, X_test, y_train, y_test = train_test_split(xresampled, yresampled, test_size = 0.2)
X_train.shape, y_train.shape


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_model.score(X_test, y_test)


import xgboost as xg

xg_model = xg.XGBClassifier()
xg_model.fit(X_train, y_train)
xg_model.score(X_test, y_test)


y_pred_rf = rf_model.predict(X_test)
y_pred_xg = xg_model.predict(X_test)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
conf_matrix_xg = confusion_matrix(y_test, y_pred_xg)
class_report_rf = pd.DataFrame(classification_report(y_test, y_pred_rf, output_dict = True))
class_report_xg = pd.DataFrame(classification_report(y_test, y_pred_xg, output_dict = True))


plt.figure(figsize=(4, 4))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Random Forest Classifier')
plt.show()


plt.figure(figsize=(4, 4))
sns.heatmap(conf_matrix_xg, annot=True, fmt='d', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('XGBoost')
plt.show()


class_report_rf


class_report_xg


def inference(sample1, sample2, model):
    inf_text1 = process_word_list(sample1.split())
    inf_text2 = process_word_list(sample2.split())

    inf_vec1 = np.zeros((1, 30))
    inf_vec2 = np.zeros((1, 30))
    inf_other_features = np.zeros((1, 10))

    if len(inf_text1) != 0:
        inf_vec1[0] = (sum([word2vec_model.wv[word] for word in inf_text1 if word in word2vec_model.wv.index_to_key]) / len(inf_text1))
        inf_blob1 = text_analysis(inf_text1)
    else:
        inf_vec1[0] = np.zeros((30,))
        inf_blob1 = np.zeros((5,))
    if len(inf_text2) != 0:
        inf_vec2[0] = (sum([word2vec_model.wv[word] for word in inf_text2 if word in word2vec_model.wv.index_to_key]) / len(inf_text2))
        inf_blob2 = text_analysis(inf_text2)
    else:
        inf_vec2[0] = np.zeros((30,))
        inf_blob2 = np.zeros((5,))
    inf_other_features[0] = np.concatenate([inf_blob1, inf_blob2])

    inf_data = pd.DataFrame({})
    inf_data = pd.concat([inf_data, 
                            pd.DataFrame(inf_vec1, columns = np.arange(0, 30)),
                            pd.DataFrame(inf_vec2, columns = np.arange(30, 60)),
                            pd.DataFrame(inf_other_features, columns = np.arange(60, 70))], axis = 1)
    output = model.predict(inf_data)
    if out == 0:
        print("Not Duplicate")
    else:
        print("Duplicate")



from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
xresampled_dl, yresampled_dl = ros.fit_resample(train_df.iloc[:,3:5], train_df["is_duplicate"])
xresampled_dl.shape, yresampled_dl.shape


que1 = np.array(xresampled_dl['question1'].tolist())
que2 = np.array(xresampled_dl['question2'].tolist())

docx = []
for i in tqdm(range(len(que1))):
    docx.append(que1[i] + " " + que2[i])


lengths = []
for sent in tqdm(docx):
    lengths.append(len(sent.split()))


sns.kdeplot(lengths)


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

tokenizer = Tokenizer(oov_token = "<nothing>")
tokenizer.fit_on_texts(docx)
sequences = tokenizer.texts_to_sequences(docx)
sequences = pad_sequences(sequences, padding = "post", maxlen = 35)


import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(sequences.max()+1, output_dim = 30, input_length = 35))
model.add(tf.keras.layers.SimpleRNN(32, return_sequences = False))
model.add(tf.keras.layers.Dense(1, activation = "sigmoid"))
model.summary()


model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics =["accuracy"])
history = model.fit(sequences, yresampled_dl,
         epochs = 20, batch_size = 1000,
         validation_split = 0.2)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

max_seq_length = 35
vocab_size = sequences.max() + 1

lstm_model = Sequential()
lstm_model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_seq_length))
lstm_model.add(Bidirectional(LSTM(64, return_sequences=False)))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(1, activation="sigmoid"))
lstm_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
lstm_model.summary()


history2 = lstm_model.fit(sequences, yresampled_dl,
         epochs = 20, batch_size = 1000,
         validation_split = 0.2)

