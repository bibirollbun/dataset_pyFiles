# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train=pd.read_parquet('/kaggle/input/wsdm-cup-multilingual-chatbot-arena/train.parquet')
test=pd.read_parquet('/kaggle/input/wsdm-cup-multilingual-chatbot-arena/test.parquet')


train.head()


train.shape


train.info()


train.describe()


train.isnull().sum()


test.isnull().sum()


train.duplicated().sum()


import matplotlib.pyplot as plt
import seaborn as sns
# Visualize the distribution of the 'winner' column (model_a vs. model_b)
sns.countplot(x='winner', data=train)
plt.title('Distribution of Winner (model_a vs. model_b)')
plt.show()



# Visualize the top 10 languages
top_languages = train['language'].value_counts().head(10)
sns.barplot(x=top_languages.index, y=top_languages.values)
plt.title('Top 10 Most Frequent Languages')
plt.xlabel('Language')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()



# Visualize the most frequent models in model_a and model_b
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.countplot(x='model_a', data=train, order=train['model_a'].value_counts().index[:10])
plt.title('Top 10 Most Frequent Models in model_a')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.countplot(x='model_b', data=train, order=train['model_b'].value_counts().index[:10])
plt.title('Top 10 Most Frequent Models in model_b')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()



from collections import Counter
import nltk
from tqdm import tqdm  # To track progress

nltk.download('punkt')

# Initialize word counters
prompt_freq = Counter()
response_a_freq = Counter()
response_b_freq = Counter()

# Tokenize row by row for efficiency
for _, row in tqdm(train.iterrows(), total=len(train)):
    prompt_freq.update(nltk.word_tokenize(row['prompt']))
    response_a_freq.update(nltk.word_tokenize(row['response_a']))
    response_b_freq.update(nltk.word_tokenize(row['response_b']))

# Print the top 10 most common words in each column
print("Top 10 words in 'prompt':", prompt_freq.most_common(10))
print("Top 10 words in 'response_a':", response_a_freq.most_common(10))
print("Top 10 words in 'response_b':", response_b_freq.most_common(10))




import nltk
from nltk.corpus import stopwords
import string
from tqdm import tqdm

nltk.download('punkt')
nltk.download('stopwords')



# Load stopwords for English + common languages
languages = ['english', 'russian', 'french', 'spanish', 'german']

# Collect stopwords from multiple languages
multi_lang_stopwords = set()
for lang in languages:
    try:
        multi_lang_stopwords.update(stopwords.words(lang))
    except:
        print(f"Stopwords for {lang} not found in NLTK")

# Add punctuation to the stopwords list
multi_lang_stopwords.update(string.punctuation)



from collections import Counter
import nltk
from nltk.corpus import stopwords
import string
from tqdm import tqdm

nltk.download('punkt')
nltk.download('stopwords')

# Load stopwords for multiple languages
languages = ['english', 'russian', 'french', 'spanish', 'german']
multi_lang_stopwords = set()

for lang in languages:
    try:
        multi_lang_stopwords.update(stopwords.words(lang))
    except:
        print(f"Stopwords for {lang} not found in NLTK")

# Add punctuation and digits to stopwords
multi_lang_stopwords.update(string.punctuation)  # Remove punctuation
multi_lang_stopwords.update(map(str, range(10)))  # Remove numbers ('0' to '9')

# Initialize word counters
prompt_freq = Counter()
response_a_freq = Counter()
response_b_freq = Counter()

# Tokenize and filter row by row
for _, row in tqdm(train.iterrows(), total=len(train)):
    prompt_tokens = [word.lower() for word in nltk.word_tokenize(row['prompt']) 
                     if word.lower() not in multi_lang_stopwords and word.isalpha()]
    
    response_a_tokens = [word.lower() for word in nltk.word_tokenize(row['response_a']) 
                         if word.lower() not in multi_lang_stopwords and word.isalpha()]
    
    response_b_tokens = [word.lower() for word in nltk.word_tokenize(row['response_b']) 
                         if word.lower() not in multi_lang_stopwords and word.isalpha()]

    prompt_freq.update(prompt_tokens)
    response_a_freq.update(response_a_tokens)
    response_b_freq.update(response_b_tokens)

# Print the top 10 most common meaningful words in each column
print("Top 10 meaningful words in 'prompt':", prompt_freq.most_common(10))
print("Top 10 meaningful words in 'response_a':", response_a_freq.most_common(10))
print("Top 10 meaningful words in 'response_b':", response_b_freq.most_common(10))



test.info()


test.describe()


test.shape


test


from collections import Counter
import nltk
from tqdm import tqdm  # To track progress

nltk.download('punkt')

# Initialize word counters
prompt_freq = Counter()
response_a_freq = Counter()
response_b_freq = Counter()

# Tokenize row by row for efficiency
for _, row in tqdm(test.iterrows(), total=len(train)):
    prompt_freq.update(nltk.word_tokenize(row['prompt']))
    response_a_freq.update(nltk.word_tokenize(row['response_a']))
    response_b_freq.update(nltk.word_tokenize(row['response_b']))

# Print the top 10 most common words in each column
print("Top 10 words in 'prompt':", prompt_freq.most_common(10))
print("Top 10 words in 'response_a':", response_a_freq.most_common(10))
print("Top 10 words in 'response_b':", response_b_freq.most_common(10))



from collections import Counter
import nltk
from nltk.corpus import stopwords
import string
from tqdm import tqdm

nltk.download('punkt')
nltk.download('stopwords')

# Load stopwords for multiple languages
languages = ['english', 'russian', 'french', 'spanish', 'german']
multi_lang_stopwords = set()

for lang in languages:
    try:
        multi_lang_stopwords.update(stopwords.words(lang))
    except:
        print(f"Stopwords for {lang} not found in NLTK")

# Add punctuation and digits to stopwords
multi_lang_stopwords.update(string.punctuation)  # Remove punctuation
multi_lang_stopwords.update(map(str, range(10)))  # Remove numbers ('0' to '9')

# Initialize word counters
prompt_freq = Counter()
response_a_freq = Counter()
response_b_freq = Counter()

# Tokenize and filter row by row
for _, row in tqdm(test.iterrows(), total=len(train)):
    prompt_tokens = [word.lower() for word in nltk.word_tokenize(row['prompt']) 
                     if word.lower() not in multi_lang_stopwords and word.isalpha()]
    
    response_a_tokens = [word.lower() for word in nltk.word_tokenize(row['response_a']) 
                         if word.lower() not in multi_lang_stopwords and word.isalpha()]
    
    response_b_tokens = [word.lower() for word in nltk.word_tokenize(row['response_b']) 
                         if word.lower() not in multi_lang_stopwords and word.isalpha()]

    prompt_freq.update(prompt_tokens)
    response_a_freq.update(response_a_tokens)
    response_b_freq.update(response_b_tokens)

# Print the top 10 most common meaningful words in each column
print("Top 10 meaningful words in 'prompt':", prompt_freq.most_common(10))
print("Top 10 meaningful words in 'response_a':", response_a_freq.most_common(10))
print("Top 10 meaningful words in 'response_b':", response_b_freq.most_common(10))


from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

# Combine all the text columns from the test and train data into one for fitting the TF-IDF model
all_text = pd.concat([train['prompt'], train['response_a'], train['response_b']])

# Fit the TF-IDF model on all the text
tfidf_vectorizer.fit(all_text)

# Transform each column into TF-IDF features
train_prompt_tfidf = tfidf_vectorizer.transform(train['prompt'])
train_response_a_tfidf = tfidf_vectorizer.transform(train['response_a'])
train_response_b_tfidf = tfidf_vectorizer.transform(train['response_b'])

# For the test data (ensure same vectorizer transformation)
test_prompt_tfidf = tfidf_vectorizer.transform(test['prompt'])
test_response_a_tfidf = tfidf_vectorizer.transform(test['response_a'])
test_response_b_tfidf = tfidf_vectorizer.transform(test['response_b'])

# Optionally, check the shape of transformed data (number of features and rows)
print(train_prompt_tfidf.shape, train_response_a_tfidf.shape, train_response_b_tfidf.shape)



from scipy.sparse import hstack

# Combine TF-IDF features for prompt, response_a, and response_b into one feature vector
train_features = hstack([train_prompt_tfidf, train_response_a_tfidf, train_response_b_tfidf])
test_features = hstack([test_prompt_tfidf, test_response_a_tfidf, test_response_b_tfidf])

# Check the shape of the combined feature matrix
print(train_features.shape, test_features.shape)



# Assuming you already have the TF-IDF features from previous steps

# 1. Separate the features (X) and labels (y) for the training data
X_train = hstack([train_prompt_tfidf, train_response_a_tfidf, train_response_b_tfidf])
y_train = train['winner']  # Assuming 'label' is the column with target labels in the train data

# 2. Separate the features for the test data (without target column)
X_test = hstack([test_prompt_tfidf, test_response_a_tfidf, test_response_b_tfidf])

# 3. Train the model (Random Forest Classifier as an example)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# 4. Make predictions on the test set
y_pred = model.predict(X_test)

# 5. (Optional) Print predictions for the test data (since there's no true label)
print("Predictions for the test data:")
print(y_pred)




# Assuming test data has a column 'id' to identify each row (or use the row index if no ID)
submission = pd.DataFrame({
    'id': test['id'],  # Replace 'id' with the actual identifier column in your test data
    'winner': y_pred  # Predictions from the model
})

# Save the predictions to a CSV file
submission.to_csv('submission.csv', index=False)

print("Submission file created: 'submission.csv'")



submission.head()




