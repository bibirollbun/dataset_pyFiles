#Importing all the libraries to be used
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split


train =pd.read_csv("/kaggle/input/depi-r-2-competition-1/xy_train.csv")
data_test=pd.read_csv("/kaggle/input/depi-r-2-competition-1/x_test.csv")


train.shape


train.head()


train.info()


# drop ID column for train
train.drop('ID',axis =1, inplace = True)
train.head()


#Palette
print(train["label"].unique())
#first of all let us evaluate the target and find out if our data is imbalanced or not
plt.figure(figsize=(12,8))
fg = sns.countplot(x= train["label"])
fg.set_title("Count Plot of Classes")
fg.set_xlabel("Classes")
fg.set_ylabel("Number of Data points")


train['label'] = train['label'].apply(lambda x: 1 if x == 2 else x)
print(train["label"].unique())        


#checking for null and duplicated values
print('the sum of null values\n',train.isnull().sum())
print('the sum of duplicated values\n',train.duplicated().sum())


train.drop_duplicates(inplace= True)
train.duplicated().sum()


# Remove all the special characters, single characters,and  extra white space from text , and use text lower case 
data_test['Cleaned_Text'] = data_test['text'].str.lower()
# Remove any caracter not in English
data_test['Cleaned_Text'] = data_test['Cleaned_Text'].apply(lambda x: re.sub('[^a-z |\s]', "",str(x)))
data_test['Cleaned_Text'] = data_test['Cleaned_Text'].apply(lambda x: re.sub(r"[\W]"," ",str(x)))
data_test['Cleaned_Text'] = data_test['Cleaned_Text'].apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', ' ', str(x))) 
data_test['Cleaned_Text'] = data_test['Cleaned_Text'].apply(lambda x: re.sub(r'[\s+]', ' ', str(x)))

# Remove all single characters from text
print('*************** text before cleaning ************')
print(data_test['text'])
print('\n *********** text after cleaning ***************')
print(data_test['Cleaned_Text'])


# Remove all the special characters, single characters,and  extra white space from text , and use text lower case 
train['Cleaned_Text'] = train['text'].str.lower()
# Remove any caracter not in English
train['Cleaned_Text'] = train['Cleaned_Text'].apply(lambda x: re.sub('[^a-z |\s]', "",str(x)))
train['Cleaned_Text'] = train['Cleaned_Text'].apply(lambda x: re.sub(r"[\W]"," ",str(x)))
train['Cleaned_Text'] = train['Cleaned_Text'].apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', ' ', str(x))) 
train['Cleaned_Text'] = train['Cleaned_Text'].apply(lambda x: re.sub(r'[\s+]', ' ', str(x)))

# Remove all single characters from text
print('*************** text before cleaning ************')
print(train['text'])
print('\n *********** text after cleaning ***************')
print(train['Cleaned_Text'])


train["Tokenize_Text"]=train.apply(lambda data: nltk.word_tokenize(data["Cleaned_Text"]), axis=1)
train.head(5)


data_test["Tokenize_Text"]= data_test.apply(lambda data: nltk.word_tokenize(data["Cleaned_Text"]), axis=1)
data_test['Tokenize_Text'].head(5)


nltk.download('stopwords')  #stop words
stop_words = set(stopwords.words('english'))

train['filtered_words'] = train["Tokenize_Text"].apply(
    lambda tokens: ' '.join(word for word in tokens if word not in stop_words))
train.head(5)


data_test['filtered_words'] = data_test["Tokenize_Text"].apply(
    lambda tokens: ' '.join(word for word in tokens if word not in stop_words))
data_test.head(5)



import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Tokenize and Pad Sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train['filtered_words'])
sequences = tokenizer.texts_to_sequences(train['filtered_words'])
word_index = tokenizer.word_index
padded_sequences = pad_sequences(sequences, padding='post')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, train['label'], test_size=0.2, random_state=42)

#  Create a Keras model with an Embedding layer
vocab_size = len(word_index) + 1  # Adding 1 because of reserved 0 index
embedding_dim = 5
max_length = len(padded_sequences[0])

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid') 
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, verbose=1)



# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


data_test.head()


#Tokenize and Pad the test data
test_sequences = tokenizer.texts_to_sequences(data_test['filtered_words'])  # Tokenize test data
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post')  # Pad test data

# Predict using the trained model
predictions = model.predict(test_padded)  

#  Convert predictions to binary labels 
predicted_labels = (predictions[:,0] > 0.4).astype(int)


test = pd.DataFrame({"ID":data_test["ID"].values})
test["label"] = predicted_labels
test.to_csv("submit.csv", index=False)

