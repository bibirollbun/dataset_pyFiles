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


import pandas as pd

raw_df = pd.read_csv("/kaggle/input/quora-insincere-questions-classification/train.csv")
raw_df


sincere_df = raw_df[raw_df.target == 0]
sincere_df.question_text.values[:10]


insincere_df = raw_df[raw_df.target == 1]
insincere_df.question_text.values[:10]


raw_df.target.value_counts(normalize=True)


raw_df.target.value_counts(normalize=True).plot(kind='bar')


test_df = pd.read_csv("/kaggle/input/quora-insincere-questions-classification/test.csv")
test_df


sub_df = pd.read_csv("/kaggle/input/quora-insincere-questions-classification/sample_submission.csv")
sub_df


SAMPLE_SIZE = 100_000

sample_df = raw_df.sample(SAMPLE_SIZE, random_state=42)

sample_df


q0 = sincere_df.question_text.values[1]
q0


q1 = raw_df[raw_df.target == 1].question_text.values[0]
q1


import nltk
from nltk.tokenize import word_tokenize


q0


word_tokenize(q0)


q1


word_tokenize(q1)


q0_tok = word_tokenize(q0)
q1_tok = word_tokenize(q1)


q1_tok


from nltk.corpus import stopwords


english_stopwords = stopwords.words('english')

", ".join(english_stopwords)


def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in english_stopwords]


q0_tok


q0_stp = remove_stopwords(q0_tok)

q0_stp


q1_stp = remove_stopwords(q1_tok)

q1_stp


from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer(language='english')

words_to_stem = ['running', 'jumped', 'happily', 'quickly', 'foxes', 'going', 'fighting', 'crying', 'killed']

stemmed_words = [stemmer.stem(word) for word in words_to_stem]

print("Original words:", words_to_stem)
print("Stemmed words:", stemmed_words)


q0_stm = [stemmer.stem(word) for word in q0_stp]


q0_stp


q0_stm


q1_stm = [stemmer.stem(word) for word in q1_stp]


q1_stp


q1_stm


small_df = sample_df[:5]
small_df


small_df.question_text.values


from sklearn.feature_extraction.text import CountVectorizer


# Create a CountVectorizer Object
small_vect = CountVectorizer()

# Fit 
small_vect.fit(small_df.question_text)

# Print the generated vocabulary
print("Vocabulary:", small_vect.get_feature_names_out())


vectors = small_vect.transform(small_df.question_text)
vectors


small_df.question_text.values[0]


vectors[0].toarray()


vectors.toarray()


stemmer = SnowballStemmer(language='english')


def tokenize(text):
    return [stemmer.stem(word) for word in word_tokenize(text)]


tokenize('What is the really (dealing) here?')


vectorizer = CountVectorizer(lowercase=True, 
                             tokenizer=tokenize,
                             stop_words=english_stopwords,
                             max_features=1000)


%%time
vectorizer.fit(sample_df.question_text)


len(vectorizer.vocabulary_)


vectorizer.get_feature_names_out()[:100]


%%time
inputs = vectorizer.transform(sample_df.question_text)


inputs.shape


sample_df.question_text.values[0]


test_df


%%time
test_inputs = vectorizer.transform(test_df.question_text)


test_inputs.shape


from sklearn.model_selection import train_test_split


train_inputs, val_inputs, train_targets, val_targets = train_test_split(inputs, sample_df.target, test_size=0.3, random_state=42)


print(f"Train shape: {train_inputs.shape}")
print(f"train targets: {train_targets.shape}")
print(f"Validation shape: {val_inputs.shape}")
print(f"val_targets: {val_targets.shape}")
print(f"Test shape: {test_inputs.shape}")


from sklearn.linear_model import LogisticRegression


model = LogisticRegression(solver='sag',max_iter=1000)


%%time
model.fit(train_inputs, train_targets)


train_preds = model.predict(train_inputs)

train_preds[:5]


train_targets[:5]


train_probs = model.predict_proba(train_inputs)
train_probs


model.classes_


pd.Series(train_preds).value_counts()


pd.Series(train_targets).value_counts()


from sklearn.metrics import accuracy_score


accuracy_score(train_targets, train_preds)


# Get probability predictions instead of class predictions
val_pred_proba = model.predict_proba(val_inputs)[:, 1]  # Probability of class 1

# Calculate ROC-AUC
from sklearn.metrics import roc_auc_score
auc_score = roc_auc_score(val_targets, val_pred_proba)
print(f"LogisticRegression Model ROC-AUC score: {auc_score:.4f}")


# Check for overfitting - compare train vs validation performance
train_accuracy = accuracy_score(train_targets, train_preds)
val_accuracy = accuracy_score(val_targets, model.predict(val_inputs))

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation ROC-AUC: {auc_score:.4f}")


from sklearn.metrics import f1_score


f1_score(train_targets, train_preds)


f1_score(train_targets, np.zeros(len(train_targets)))


random_preds = np.random.choice((0, 1), len(train_targets))
f1_score(train_targets, random_preds)


sincere_df.question_text.values[:10]


sincere_df.target.values[:10]


model.predict(vectorizer.transform(sincere_df.question_text.values[:10]))


insincere_df.question_text.values[:10]


insincere_df.target.values[:10]


model.predict(vectorizer.transform(insincere_df.question_text.values[:10]))


from sklearn.ensemble import RandomForestClassifier


model_1 = RandomForestClassifier(n_estimators=500, random_state=42, bootstrap=True,max_features=0.7,max_depth=10)


%%time
model_1.fit(train_inputs, train_targets)


train_preds = model_1.predict(train_inputs)

train_preds[:5]


from sklearn.metrics import accuracy_score


accuracy_score(train_targets, train_preds)


f1_score(train_targets, train_preds)


sincere_df.target.values[:10]


model_1.predict(vectorizer.transform(sincere_df.question_text.values[:10]))


test_preds = model_1.predict(test_inputs)
test_preds[:5]


sub_df


sub_df['prediction'] = test_preds

# Verify the update
print("Updated submission preview:")
print(sub_df.head())
print(f"\nSubmission shape: {sub_df.shape}")

# Save the updated submission
sub_df.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")




