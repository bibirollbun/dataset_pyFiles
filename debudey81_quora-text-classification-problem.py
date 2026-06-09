# Dataset Link - https://www.kaggle.com/c/quora-insincere-questions-classification


# import os
# os.environ['KAGGLE_CONFIG_DIR'] = '.'


# # Download Data from Kaggle
# !chmod 600 ./kaggle.json
# !kaggle competitions download -c quora-insincere-questions-classification -f train.csv -p data


# !kaggle competitions download -c quora-insincere-questions-classification -f test.csv -p data
# !kaggle competitions download -c quora-insincere-questions-classification -f sample_submission.csv -p data


import os
os.listdir('/kaggle/input/quora-insincere-questions-classification')


data_dir = '/kaggle/input/quora-insincere-questions-classification'


# Explore the Data Using Pandas
train_fname = data_dir + '/train.csv'
test_fname = data_dir + '/test.csv'
sample_fname = data_dir + '/sample_submission.csv'


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


raw_df = pd.read_csv(train_fname)
raw_df.head()


# check the dimension of the data
print(f'The shape of the data is {raw_df.shape}')


# check the distribution of target variable
raw_df['target'].value_counts()


raw_df['target'].value_counts(normalize=True).plot(kind='bar')
plt.show()


# check few sincere questions
raw_df[raw_df['target'] == 0]['question_text'].values[:10]


# check few insincere questions
raw_df[raw_df['target'] == 1]['question_text'].values[:10]


test_df = pd.read_csv(test_fname)
test_df.head()


print(f'Shape of the test data is',test_df.shape)


sub_df = pd.read_csv(sample_fname)
sub_df.head()


print(f'Shape of the submission data is',sub_df.shape)


sub_df['prediction'].value_counts()


# Create sample dataset
sample_size = 100000
sample_df = raw_df.sample(sample_size,random_state=42)


q1 = sample_df['question_text'].values[0]
print(q1)


# import nltk
# nltk.download('punkt_tab')


from nltk.tokenize import word_tokenize


word_tokenize(q1)


q1_tok = word_tokenize(q1)


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


english_stopwords = stopwords.words('english')


print(f'Total number of stop words are {len(english_stopwords)}')


def remove_stopwords(tokens):
  return [word for word in tokens if word.lower() not in english_stopwords]


q1_stp = remove_stopwords(q1_tok)
q1_stp


q1_tok


from nltk.stem.snowball import SnowballStemmer


stemmer = SnowballStemmer('english')


stemmer.stem('running')


stemmer.stem('runs')


q1_stm = [stemmer.stem(word) for word in q1_stp]
q1_stm


q1_stp


small_df = sample_df[:5]


small_df['question_text'].values


from sklearn.feature_extraction.text import CountVectorizer


small_vec = CountVectorizer()
small_vec.fit(small_df['question_text'])


small_vec.vocabulary_


# check feature names
small_vec.get_feature_names_out()


## Transform documents into Vectors
vectors = small_vec.transform(small_df['question_text'])
vectors.toarray()


vectors.shape


import re
from nltk.stem import PorterStemmer
def preprocess_text(text):
    """
    Preprocess the input text by tokenizing, converting to lowercase, removing punctuation,
    filtering out stopwords, and applying stemming.

    Parameters:
        text (str): The text to be processed.

    Returns:
        List[str]: A list of processed tokens.
    """
    # Step 1: Tokenize the text into individual words
    tokens = word_tokenize(text)

    # Step 2: Convert each token to lowercase
    tokens = [token.lower() for token in tokens]

    # Step 3: Remove punctuation and special characters from each token.
    # The regex pattern '[^a-zA-Z0-9]' matches any character that is not alphanumeric.
    tokens = [re.sub(r'[^a-zA-Z0-9]', '', token) for token in tokens]

    # Remove any tokens that may have become empty after removing punctuation
    tokens = [token for token in tokens if token]

    # Step 4: Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Step 5: Apply stemming to each token using Porter Stemmer
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens


# Example usage:
sample_text = "Here's an example sentence, showcasing the functionality: preprocessing text!"
processed_tokens = preprocess_text(sample_text)
print(processed_tokens)


# Configure Count Vectorize Parameters
vectorizer = CountVectorizer(tokenizer=preprocess_text, max_features=1000)


%%time
vectorizer.fit(sample_df['question_text'])


# check feature length
len(vectorizer.get_feature_names_out())


vectorizer.get_feature_names_out()[:100]


%%time
inputs = vectorizer.transform(sample_df['question_text'])


print(inputs.shape)


sample_df['question_text'].values[0]


# display the first row
inputs[0].toarray()


%%time
test_inputs = vectorizer.transform(test_df['question_text'])


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(inputs, sample_df['target'], test_size=0.2, random_state=42, stratify=sample_df['target'])


print(f'X_train shape is {X_train.shape}')
print(f'X_val shape is {X_val.shape}')


from sklearn.linear_model import LogisticRegression


# fit the model
model = LogisticRegression()
model.fit(X_train, y_train)


# evaluate on validation data
y_pred = model.predict(X_val)


# Model Performance Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print(f'Accuracy: {round(accuracy_score(y_val, y_pred),2)}')
print(f'Precision: {round(precision_score(y_val, y_pred),2)}')
print(f'Recall: {round(recall_score(y_val, y_pred),2)}')
print(f'F1 Score: {round(f1_score(y_val, y_pred),2)}')


# confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_val, y_pred)


# Plot ROC Curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_val, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


test_inputs.shape


test_preds = model.predict(test_inputs)


sub_df['prediction'] = test_preds


sub_df['prediction'].value_counts()


sub_df.to_csv('submission.csv',index=False)


!head submission.csv




