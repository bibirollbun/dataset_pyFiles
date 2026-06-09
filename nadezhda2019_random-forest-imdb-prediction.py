import warnings
warnings.filterwarnings("ignore")


import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
from numpy import array
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split



#!pip show tensorflow
# !pip show keras



df = pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip', delimiter='\t')
tsv_file_path = '/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip'

df_un = pd.read_csv(tsv_file_path, delimiter='\t', quoting=3)
tsv_file_path = '/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip'

df_test = pd.read_csv(tsv_file_path, delimiter='\t', quoting=3)



# можно также для удаления тегов использовать библиотеку BeautifulSoup
#from bs4 import BeautifulSoup             
# используем регулярные выражения
def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)
    # альтернатива :    sentence = BeautifulSoup(sen).get_text() 

    # Удалить все символы, не являющиеся буквами
    sentence = re.sub('[^a-zA-Z]', # шаблон - что заменяем, ^a-zA-Z - все символы не являющиеся буквами
                        ' ',      # шаблон - на что заменяем
                      sentence) #  где заменяем

    # Удалить единственный символ (регулярное выражение: пробел, буква, пробел)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Удалим множественные пробелы (>=1 пробела заменяем на ровно 1 пробел)
    sentence = re.sub(r'\s+', ' ', sentence)
    
    words = sentence.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))   

# удаляем теги
TAG_RE = re.compile(r'<[^>]+>')

# функция для удаления тегов

def remove_tags(text):
    return TAG_RE.sub('', text)


X_train = []
sentences = list(df['review'])
for sen in sentences:
    X_train.append(preprocess_text(sen))


X_test = []
sentences = list(df_test['review'])
for sen in sentences:
    X_test.append(preprocess_text(sen))


X_train[0]


df.head()


y = df['sentiment']


y_train = np.array(list(map(lambda x: 1 if x==1 else 0, y)))


df['sentiment'].value_counts().plot(kind = 'bar')


print("Creating the bag of words...\n")
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(X_train)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()


# Take a look at the words in the vocabulary
#vocab = vectorizer
#print(vocab)


'''
import numpy as np
# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)
vocab = vectorizer
# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
    print(count, tag)
'''


print("Training the random forest...")
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit(train_data_features, y)


test_data_features = vectorizer.transform(X_test)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":df_test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )




