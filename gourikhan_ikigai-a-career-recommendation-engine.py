# Set a seed value
from numpy.random import seed
seed(101)

import pandas as pd
import numpy as np
import os

import pickle
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Don't Show Warning Messages
import warnings
warnings.filterwarnings('ignore')


# Read the data

df_questions = \
pd.read_csv('../input/data-science-for-good-careervillage/questions.csv')
df_answers = \
pd.read_csv('../input/data-science-for-good-careervillage/answers.csv')
df_professionals = \
pd.read_csv('../input/data-science-for-good-careervillage/professionals.csv')

df_comments = \
pd.read_csv('../input/data-science-for-good-careervillage/comments.csv')
df_tags = \
pd.read_csv('../input/data-science-for-good-careervillage/tags.csv')
df_tag_users = \
pd.read_csv('../input/data-science-for-good-careervillage/tag_users.csv')

#print(df_questions.shape)
#print(df_answers.shape)
#print(df_professionals.shape)
#print(df_comments.shape)
#print(df_tags.shape)
#print(df_tag_users.shape)


# Check what folders are available

os.listdir('../input')


# Load the pickled dataframe

path_1 = '../input/data-prep-for-career-village-recsys/df_qa_prof.pickle'

df_qa_prof = pickle.load(open(path_1,'rb'))

# check the shape
df_qa_prof.shape


df_qa_prof.head(2)


# Define a function to clean the text

def process_text(x):
    
    # remove the hash sign
    x = x.replace("#", "")
    
    # remove the dash sign with a space
    #x = x.replace("-", " ")
    
    # Remove HTML
    x = BeautifulSoup(x).get_text()
    
    # convert words to lower case
    x = x.lower()
    
    # remove the word question
    x = x.replace("question", "")
    
    # remove the word career
    x = x.replace("career", "")
    
    # remove the word study
    x = x.replace("study", "")
    
    # remove the word student
    x = x.replace("student", "")
    
    # remove the word school
    x = x.replace("school", "")
    
    # Remove non-letters
    x = re.sub("[^a-zA-Z]"," ", x)
    
    # Remove stop words
    # Convert words to lower case and split them
    words = x.split()
    stops = stopwords.words("english")
    x_list = [w for w in words if not w in stops]
    # convert the list to a string
    x = ' '.join(x_list)
    
    return x


###################################

QUESTION_INDEX = 1710

###################################


# =========================================== #
# Please check that QUESTION_INDEX = None in the above cell before entering
# your own question.

my_question_title = "How do I become a data scientist?"

my_question_body = "I want to be a data scientist. What subjects should I study? #data-science"

# =========================================== #


# Code to process the question

# if Option 1 is chosen
if QUESTION_INDEX != None:
    
    QUESTION_INDEX = int(QUESTION_INDEX)
    
    student_id = df_qa_prof.loc[QUESTION_INDEX, 'questions_author_id']
    # Get the question info from the dataset.
    # The text has already been cleaned above.
    question_id = df_qa_prof.loc[QUESTION_INDEX, 'questions_id']
    question_title = df_qa_prof.loc[QUESTION_INDEX, 'questions_title']
    question_body = df_qa_prof.loc[QUESTION_INDEX, 'questions_body']
    # question_text is clean text that is used in the models
    question_text = df_qa_prof.loc[QUESTION_INDEX, 'quest_text'] 

# if Option 2 is chosen
else:
    student_id = 33333333 # dummy id that's needed for the final selection code
    # get the input question
    question_id = 'My Question'
    question_title = my_question_title
    question_body = my_question_body
    # Clean the text using the process_text() function.
    # question_text is clean text that is used in the models
    question_text = process_text(question_title) + ' ' + process_text(question_body)
    

# Print the question
print('Question id: ', question_id)
print('Question Title: ', question_title)
print('\n')
print('Question Body:\n ', question_body)








# load df_professionals
path_2 = '../input/data-prep-for-career-village-recsys/df_professionals.pickle'
df_professionals = pickle.load(open(path_2,'rb'))

# replace all missing values with nothing
df_professionals = df_professionals.fillna('')

# Create a dictionary of tag id's and tag names
keys = list(df_tags['tags_tag_id'])
values = list(df_tags['tags_tag_name'])
tags_dict = dict(zip(keys, values))

# Change the tag id numbers to tag names that we can read
df_tag_users['tag_name'] = df_tag_users['tag_users_tag_id'].map(tags_dict)

df_tag_users.head()


# get a list of professionals
prof_list = list(df_professionals['professionals_id'])
# filter out the professionals from df_tag_users
df_prof_tag_users = df_tag_users[df_tag_users['tag_users_user_id'].isin(prof_list)]

df_prof_tag_users.shape


# drop the tag_users_tag_id column
df_prof_tag_users = df_prof_tag_users.drop('tag_users_tag_id', axis=1)

# replace missing values with nothing - just be be safe
df_prof_tag_users =df_prof_tag_users.fillna('')

# add a space to the end of each tag name
def add_space(x):
    x = x + ' '
    
    return x

df_prof_tag_users['tag_name'] = df_prof_tag_users['tag_name'].apply(add_space)

# groupby tag_users_user_id and sum() the tags
df_prof_tag_users = df_prof_tag_users.groupby('tag_users_user_id').sum()

# reset the index
df_prof_tag_users = df_prof_tag_users.reset_index()

# check how many professionals follow tags
num_followers = len(df_prof_tag_users['tag_users_user_id'])

# Are there professionals who don't follow any tags?

num_profs = df_professionals['professionals_id'].nunique()
num_tag_followers = df_prof_tag_users['tag_users_user_id'].nunique()

num_not_followers = num_profs - num_tag_followers

print(num_followers, 'professionals follow tags.')
print(num_not_followers, 'professionals do not follow tags.')

df_prof_tag_users.head()


# https://www.youtube.com/watch?v=h4hOPGo4UVU

# Change column name in df_prof_tag_users. 
# For the merge to work the column called professionals_id needs to be in
# both dataframes.
new_names = ['professionals_id', 'tags_followed']
df_prof_tag_users.columns = new_names

# perform the left merge
df_profs = pd.merge(df_professionals,df_prof_tag_users, 
                   on='professionals_id', how='left')

# replace missing values with nothing
df_profs = df_profs.fillna('')

print('We now have a combined dataframe containing the tag info and profile info for all professionals.')

df_profs.head()


# Create the new column by summing the strings from each seperate column.
df_profs['prof_info'] = df_profs['professionals_headline'] + ' ' \
+ df_profs['professionals_industry'] + ' ' + df_profs['tags_followed']

# clean the text using the process_text() function defined above
df_profs['prof_info'] = df_profs['prof_info'].apply(process_text)

print('The prof_info column contains the combined profile info of each professional.')
df_profs.head()


# copy a row from df_profs
df_row1 = df_profs[df_profs.index == 0] 
# set all values to nothing
df_row1.loc[:,:] = ''
# reset the index
df_row1 = df_row1.reset_index(drop=True)
    
# Assign the prof_info in this row to be the same as the question.
# We do this because later we will compare this question to all other rows
# in the prof_info column.
df_row1.loc[0, 'prof_info'] = question_text

# Concat df_row to df_profs
# The question will be the first row
df_profs = pd.concat([df_row1, df_profs], axis=0).reset_index(drop=True)

print('The Question, in processed form, is now located at the top of the prof_info column.')

df_profs.head(2)



# Select the data we want to use. 
# This column has our new question at the top.
data = df_profs['prof_info']

# instantiate vectorizer
vect = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.5)

# learn the 'vocabulary' of the data
vect.fit(data)

# Transform the data into a document term matrix.
# Keep in mind that the output type is a sparse matrix.
prof_dtm = vect.transform(data)

#prof_dtm.shape


# check what features have been created
#vect.get_feature_names()


# https://stackoverflow.com/questions/12118720/
# python-tf-idf-cosine-to-find-document-similarity

# prof_dtm[0:1] This selects the first row of prof_info column.
# We are saying: Tell me how similar every row is to the first row.
cosine_similarities = linear_kernel(prof_dtm[0:1], prof_dtm)

# The line of code commented out below would give us the cosine similarity score
# of every row to every other row, just like a correlation matrix.
# But there's no need for this and the RAM needed for this calculation
# would cause this kernel to crash.

# cosine_similarities = linear_kernel(prof_dtm, prof_dtm)

# Quick check: The first value should be 1.0 because it's the
# comparison of the question to itself.
cosine_similarities


# create a dataframe
df_cosine_matrix = pd.DataFrame(cosine_similarities)

# get the column names from df_train
cols = list(df_profs['professionals_id'])

# Change the name of the first column. This is the score for the Question
cols[0] = 'question_cosine_score'

# rename the columns in the dataframe
df_cosine_matrix.columns = cols

# Add the professionals id values as a new column.
# This is identical to answers_author_id.
#df_cosine_matrix['answers_id'] = df_train['answers_author_id']

# set the answers_id column as the index
#df_cosine_matrix.set_index('answers_id', inplace=True)

# transpose the dataframe
df = df_cosine_matrix.T

# rename the column
new_col = ['cosine_score_for_each_prof_id']
df.columns = new_col

# sort the cosine similarity values in descending order
df = df.sort_values('cosine_score_for_each_prof_id',axis=0, ascending=False)

# check the top 10 cosine scores
df.head(10)


# Set the cosine similarity threshold
MODEL_1_THRESHOLD = 0.13

# filter out all rows that have a cosine_score >= THRESHOLD
df = df[df['cosine_score_for_each_prof_id'] >= MODEL_1_THRESHOLD]

# remove the first row because this row is the question we asked
df = df[1:]

num_professionals = len(df)

print('Number of professionals chosen: ', num_professionals)

print('This is a sample of the professionals the model has selected.')

# Print the id's of the professionals who have been 
# selected as well as the associated cosine scores
df.head(10)


# reset the index
df.reset_index(inplace=True)

# rename the columns
new_names = ['prof_id', 'cosine_score_for_each_prof_id']
df.columns = new_names

# create a list with all answer id values from df
prof_list = list(df['prof_id'])

# display the list
#prof_list


print('Question id: ', question_id)
print('Question Title: ', question_title)
print('\n')
print('Question Body:\n ', question_body)


# Print the profiles the professionals who can answer this question. 
# Note: If you are running this kernel you may need to scroll the output otherwise
# you might mistakenly think that the text shown is all there is.

print('\n')
print('Model 1')
print('Number of professionals selected: ', len(prof_list))
print('== Printing info on each professional who was selected ==')

# set the index
df_professionals = df_professionals.set_index('professionals_id')

# set the index of df_profs to be the question id
df_profs = df_profs.set_index('professionals_id')

# Create an empty list to store the professional id's that are
# associated with the answers that have been selected,
model_1_list = []


for prof_id in prof_list:
    
    print('\n')
    
    # print the professional's id (i.e. their name)
    # get the prof id of the person who wrote the answer
    
    print('==> Professional id: ', prof_id)
    model_1_list.append(prof_id)
    
    
    # print their job title:
    title = df_professionals.loc[prof_id, 'professionals_headline']
    print('Title: ', title)
    
    # print the industry they work in
    industry = df_professionals.loc[prof_id, 'professionals_industry']
    print('Industry: ', industry)
    
    # Print the tags that are followed
    tags = df_profs.loc[prof_id,'tags_followed']
    print('==Tags being followed:\n',tags)
    


model_1_list











# load df_qa_prof
df_qa_prof = pickle.load(open(path_1,'rb'))
# load df_professionals
df_professionals = pickle.load(open(path_2,'rb'))


print(df_qa_prof.shape)
print(df_professionals.shape)


# copy a row from df_qa_prof
df_row2 = df_qa_prof[df_qa_prof.index == 0] 
# set all values to nothing
df_row2.loc[:,:] = ''
# reset the index
df_row2 = df_row2.reset_index(drop=True)
    
# Assign the answer_text in this row to be the same as the question.
# We do this because later we will compare this question to all other rows
# in the answer_text column.
df_row2.loc[0, 'answers_text'] = question_text

# Concat df_row2 to df_qa_prof.
# The question will be at the top of the first row.
df_qa_prof = pd.concat([df_row2, df_qa_prof], axis=0).reset_index(drop=True)

df_qa_prof.head(2)


# Select the data we want to use. Note we are comparing the question to answers.
# We need to vectorize the answers_text column.
# This column has our new question at the top.
data = df_qa_prof['answers_text']

# instantiate vectorizer
vect = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.5)

# learn the vocabulary of the data
vect.fit(data)

# Transform the data to a document term matrix.
# The output type is a sparse matrix.
prof_dtm = vect.transform(data)

prof_dtm.shape


# https://stackoverflow.com/questions/12118720/
# python-tf-idf-cosine-to-find-document-similarity

# prof_dtm[0:1] This selects the first row of prof_info column.
# We are saying: Tell me how similar every row is to the first row.
cosine_similarities = linear_kernel(prof_dtm[0:1], prof_dtm)

# The line below would give us the cosine similarity of every row to every other row,
# just like a correlation matrix.
# But there's no need for this and the RAM needed for this calculation
# would cause this kernel to crash.
# cosine_similarities = linear_kernel(prof_dtm, prof_dtm)

# Quick check: The first value should be 1.0 because it's the
# comparison of the question to itself.
cosine_similarities


# create a dataframe
df_cosine_matrix = pd.DataFrame(cosine_similarities)

# get the column names from df_train
cols = list(df_qa_prof['answers_id'])

# Change the name of the first column. This is the score for the Question
cols[0] = 'question_cosine_score'

# rename the columns in the dataframe
df_cosine_matrix.columns = cols

# Add the professionals id values as a new column.
# This is identical to answers_author_id.
df_cosine_matrix['answers_id'] = df_qa_prof['answers_author_id']

# set the answers_id column as the index
df_cosine_matrix.set_index('answers_id', inplace=True)

# transpose the dataframe
df = df_cosine_matrix.T

# rename the column
new_col = ['cosine_score_for_each_answer_id']
df.columns = new_col

# sort the cosine similarity values in descending order
df = df.sort_values('cosine_score_for_each_answer_id',axis=0, ascending=False)

# check the top 20 cosine scores
df.head(20)



# Set the cosine similarity threshold
MODEL_2_THRESHOLD = 0.1

# filter out all rows that have a cosine_score >= THRESHOLD
df = df[df['cosine_score_for_each_answer_id'] >= MODEL_2_THRESHOLD]

# remove the first row because this row is the question we asked
df = df[1:]

num_answers = len(df)

print('Number of answers chosen: ', num_answers)

print('This is a sample of the answers the model has selected.')

# print the answers that have been selected as well as the associated cosine scores
df.head(10)


# reset the index
df.reset_index(inplace=True)

# rename the columns
new_names = ['answers_id', 'cosine_score_for_each_answer_id']
df.columns = new_names

# create a list with all answer id values from df
answer_list = list(df['answers_id'])

# display the list
#answer_list


print('Question id: ', question_id)
print('Question Title: ', question_title)
print('\n')
print('Question Body:\n ', question_body)


# Print info on the professionals who can answer this question

#print('\n')
print('Model 2')
print('Number of professionals selected: ', len(answer_list))
print('== Printing info on each professional who was selected ==')

# set the index
df_professionals = df_professionals.set_index('professionals_id')



# Create an empty list to store the professional id's that are
# associated with the answers that have been selected,
model_2_list = []

# set the index of df_train to be the question id
df_qa_prof = df_qa_prof.set_index('answers_id')

for ans_id in answer_list:
    
    # print the professional's id (i.e. their name)
    # get the prof id of the person who wrote the answer
    prof_id = df_qa_prof.loc[ans_id, 'answers_author_id']
    print('\n')
    print('==> Professional id: ', prof_id)
    model_2_list.append(prof_id)
    
    
    # print their job title:
    title = df_professionals.loc[prof_id, 'professionals_headline']
    print('Title: ', title)
    
    # print the industry they work in
    industry = df_professionals.loc[prof_id, 'professionals_industry']
    print('Industry: ', industry)
    
    # Print the answer that they wrote which was similar the question being asked
    answer = df_qa_prof.loc[ans_id,'answers_body']
    print('==Answer given to similar question:\n',answer)
    


# uncomment the next line to print the list of professional id's

# model_2_list











# load df_qa_prof
df_qa_prof = pickle.load(open(path_1,'rb'))
# load df_professionals
df_professionals = pickle.load(open(path_2,'rb'))


print(df_qa_prof.shape)
print(df_professionals.shape)


# copy a row from df_qa_prof
df_row2 = df_qa_prof[df_qa_prof.index == 0] 
# set all values to nothing
df_row2.loc[:,:] = ''
# reset the index
df_row2 = df_row2.reset_index(drop=True)
    
# Assign the answer_text in this row to be the same as the question.
# We do this because later we will compare this question to all other rows
# in the answer_text column.
df_row2.loc[0, 'answers_text'] = question_text

# Concat df_row2 to df_qa_prof.
# The question will be at the top of the first row.
df_qa_prof = pd.concat([df_row2, df_qa_prof], axis=0).reset_index(drop=True)

df_qa_prof.head(2)


# Select the data we want to use. Note we are comparing the question to answers.
# We need to vectorize the answers_body column.
# This column has our new question at the top.
data = df_qa_prof['answers_text']

# instantiate vectorizer
vect = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.5)

# learn the vocabulary of the data
vect.fit(data)

# transform the data to a document term matrix
prof_dtm = vect.transform(data)

prof_dtm.shape


from sklearn.decomposition import TruncatedSVD

# Initialize
tsvd = TruncatedSVD(n_components=200, random_state=101)

# Fit
tsvd.fit(prof_dtm)

# Transform
# This returns a type numpy array and not a sparse matrix type as with tfidf.
prof_dtm = tsvd.transform(prof_dtm)

prof_dtm.shape


# create a dataframe
df = pd.DataFrame(prof_dtm)

df.head()


# https://stackoverflow.com/questions/12118720/
# python-tf-idf-cosine-to-find-document-similarity

# prof_dtm[0:1] This selects the first row of prof_info column. Note this slicing 
# is for a sparse matrix.
# We are saying: Tell me how similar every row is to the first row.
# Note that we are using cosine_similarity here and not linear_kernel.
cosine_similarities = cosine_similarity(prof_dtm[0:1], prof_dtm)

# The line below would give us the cosine similarity of every row to every other row,
# just like a correlation matrix.
# But there's no need for this and the RAM needed for this calculation
# would cause this kernel to crash.
# cosine_similarities = linear_kernel(prof_dtm, prof_dtm)

# Quick check: The first value should be 1.0 because it's the
# comparison of the question to itself.
cosine_similarities


# create a dataframe
df_cosine_matrix = pd.DataFrame(cosine_similarities)

# get the column names from df_train
cols = list(df_qa_prof['answers_id'])

# Change the name of the first column. This is the score for the Question
cols[0] = 'question_cosine_score'

# rename the columns in the dataframe
df_cosine_matrix.columns = cols

# Add the professionals id values as a new column.
# This is identical to answers_author_id.
df_cosine_matrix['answers_id'] = df_qa_prof['answers_author_id']

# set the answers_id column as the index
df_cosine_matrix.set_index('answers_id', inplace=True)

# transpose the dataframe
df = df_cosine_matrix.T

# rename the column
new_col = ['cosine_score_for_each_answer_id']
df.columns = new_col

# sort the cosine similarity values in descending order
df = df.sort_values('cosine_score_for_each_answer_id',axis=0, ascending=False)

# check the top 20 cosine scores
df.head(20)



# Set the cosine similarity threshold
MODEL_3_THRESHOLD = 0.65

# filter out all rows that have a cosine_score >= THRESHOLD
df = df[df['cosine_score_for_each_answer_id'] >= MODEL_3_THRESHOLD]

# remove the first row because this row is the question we asked
df = df[1:]

num_answers = len(df)

print('Number of answers chosen: ', num_answers)

print('This is a sample of the answers the model has selected.')

# print the answers that have been selected as well as the associated cosine scores
df.head(10)


# reset the index
df.reset_index(inplace=True)

# rename the columns
new_names = ['answers_id', 'cosine_score_for_each_answer_id']
df.columns = new_names

# create a list with all answer id values from df
answer_list = list(df['answers_id'])

# display the list
#answer_list


print('Question id: ', question_id)
print('Question Title: ', question_title)
print('\n')
print('Question Body:\n ', question_body)


# Print info on the professionals who can answer this question

#print('\n')
print('Model 3')
print('Number of professionals selected: ', len(answer_list))
print('== Printing info on each professional who was selected ==')

# set the index
df_professionals = df_professionals.set_index('professionals_id')



# Create an empty list to store the professional id's that are
# associated with the answers that have been selected,
model_3_list = []

# set the index of df_train to be the question id
df_qa_prof = df_qa_prof.set_index('answers_id')

for ans_id in answer_list:
    
    # print the professional's id (i.e. their name)
    # get the prof id of the person who wrote the answer
    prof_id = df_qa_prof.loc[ans_id, 'answers_author_id']
    print('\n')
    print('==> Professional id: ', prof_id)
    model_3_list.append(prof_id)
    
    
    # print their job title:
    title = df_professionals.loc[prof_id, 'professionals_headline']
    print('Title: ', title)
    
    # print the industry they work in
    industry = df_professionals.loc[prof_id, 'professionals_industry']
    print('Industry: ', industry)
    
    # Print the answer that they wrote which was similar the question being asked
    answer = df_qa_prof.loc[ans_id,'answers_body']
    print('==Answer given to similar question:\n',answer)


# uncomment the next line to print the list of professional id's

# model_3_list








# load df_qa_prof
df_qa_prof = pickle.load(open(path_1,'rb'))
# load df_professionals
df_professionals = pickle.load(open(path_2,'rb'))


print(df_qa_prof.shape)
print(df_professionals.shape)


# We will use GloVe vectors that have a standard length of 200
EMBED_LENGTH = 200


# copy a row from df_qa_prof
df_row3 = df_qa_prof[df_qa_prof.index == 0] 
# set all values to nothing
df_row3.loc[:,:] = ''
# reset the index
df_row3 = df_row3.reset_index(drop=True)
    
# Assign the answer_text in this row to be the same as the question.
# We do this because later we will compare this question to all other rows
# in the answer_text column.
df_row3.loc[0, 'answers_text'] = question_text

# Concat df_row to df_qa_prof
# The question will be the first row
df_qa_prof = pd.concat([df_row3, df_qa_prof], axis=0).reset_index(drop=True)



# Create a new column showing the length of each answer
df_qa_prof['answer_length'] = df_qa_prof['answers_body'].apply(len)

print('The answers_text column is the document corpus.')
df_qa_prof.head(2)


# Create a corpus of documents
corpus_text_list = list(df_qa_prof['answers_text'])



# Instantiate the tokenizer.
# Note that this is a word tokenizer.
t = Tokenizer()

# create a dictionary where the word is the key and a number is the value
t.fit_on_texts(corpus_text_list)

# How many words are there in our corpus vocabulary?

vocab_size = len(t.word_index)
print('Vocab size: ', vocab_size)


# These are all the words in the vocabulary of our corpus.
# Each word is assigned an index starting at 1.

t.word_index


# Add 1 to the number of words in the vocabulary
vocab_size = len(t.word_index) + 1
vocab_size


# convert the text to sequences of numbers
encoded_docs = t.texts_to_sequences(corpus_text_list)

# Print the list of lists
#print(encoded_docs)



# Let's look at the text lengths to decide what max_length to use
print('Min length: ', df_qa_prof['answer_length'].min())
print('Max length: ',df_qa_prof['answer_length'].max())
print('Mean length: ',df_qa_prof['answer_length'].mean())
print('Median length: ',df_qa_prof['answer_length'].median())
print('Mode lengths: ',df_qa_prof['answer_length'].mode()) # value that appears most often

# Set the max_length 
max_length = 500

# Pad each list so they all have the same length
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

# (num_answers, max_length)
padded_docs.shape  


# source: https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

# We will use pre-trained GloVe emedding vectors from Kaggle Datasets that
# have been imported into this kernel.
# https://www.kaggle.com/rtatman/glove-global-vectors-for-word-representation

# Load the pre-trained GloVe vectors
# Set the path to glove.6B.200d.txt
path = '../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt'

embeddings_index = dict()
f = open(path)

for line in f:
    # Note: use split(' ') instead of split() if you get an error.
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix
embedding_matrix = np.zeros((vocab_size, EMBED_LENGTH))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print('The result is a matrix of embeddings.')
print('Words are the rows, the features are the columns.')

# The result is a matrix of embeddings only for words in our data.
# Words are the rows, the features are the columns.




# Note that the words are on the index column
df_glove_embeddings = pd.DataFrame(embedding_matrix)

# get all the dictionary keys as a list
word_dict = t.word_index

# get a list of keys
keys = list(word_dict.keys())

# Insert a dummy_word at the first position.
# The dummy_word exists because our dict key:value pairs
# start from word:1 and not word:0.
keys.insert(0, 'dummy_word')

# transpose the dataframe so that the words become the columns
df_glove_embeddings = df_glove_embeddings.T

# set the names of the columns
df_glove_embeddings.columns = keys


# convert the dataframe back to the original form
df_glove_embeddings = df_glove_embeddings.T

# reset the index
df_glove_embeddings = df_glove_embeddings.reset_index(drop=False)

# change the name of the first column to 'words'
column_names = list(df_glove_embeddings.columns)
column_names[0] = 'words'
df_glove_embeddings.columns = column_names

print('This is the embeddings in a dataframe.')
df_glove_embeddings.head(10)


# create an empty matrix
encoding_mat = np.zeros((len(padded_docs), EMBED_LENGTH))

for i in range(0,len(padded_docs)):
    # select the document
    padded_doc = padded_docs[i]
    # create an empty encoding list
    encoding = np.zeros(EMBED_LENGTH)
    # select a document
    for item in padded_doc:
        # Here we are adding the vectors together.
        # This selects a row from embedding_matrix.
        # The output is a list.
        encoding = encoding + embedding_matrix[item] # item is an integer value
        
    # Insert the encoding to encoding_mat
    # Here we are averaging the encodings by dividing by the length.
    encoding_mat[i] = encoding/max_length

# check the shape of the matrix
encoding_mat.shape


# Display the embedding matrix
# The words are the rows and the features are the columns.

# Every row represents one answer that has been encoded as a vector
df_encoding_mat = pd.DataFrame(encoding_mat)

print('This is the embedding matrix. Each row represents one answer that has been encoded as a vector.')
print('Row 0 is the question.')
df_encoding_mat.head()


# check the shape of the embedding matrix
encoding_mat.shape


# reshape the encoding matrix to (num_samples, num_features)
encoding_mat = encoding_mat #.reshape(max_length,EMBED_LENGTH) 
# reshape the base_document i.e. the one we will compare to all others
base_doc = encoding_mat[0].reshape(1,EMBED_LENGTH)

# calculate the cosine similarity
cosine_similarities = cosine_similarity(base_doc, encoding_mat)

# The following would compute a cosine similiarity matrix comapring every
# doc to every other doc, like a correlation matrix.
# This uses a lot of RAM.
#cosine_similarities = cosine_similarity(encoding_mat, encoding_mat)

cosine_similarities.shape


# flatten the matrix
cosine_similarities = cosine_similarities.flatten()

#Check: The first value should be 1.0 because the 
# question is being compared to itself.
cosine_similarities


# create a dataframe
df_cosine_matrix = pd.DataFrame(cosine_similarities)

# transpose the dataframe
df_cosine_matrix = df_cosine_matrix.T

# get the column names from df_train
cols = list(df_qa_prof['answers_id'])

# Change the name of the first column. This is the score for the Question
cols[0] = 'question_cosine_score'

# rename the columns in the dataframe
df_cosine_matrix.columns = cols

# Add the professionals id values as a new column.
# This is identical to answers_author_id.
df_cosine_matrix['answers_id'] = df_qa_prof['answers_author_id']

# set the answers_id column as the index
df_cosine_matrix.set_index('answers_id', inplace=True)

# transpose the dataframe
df = df_cosine_matrix.T

# rename the column
new_col = ['cosine_score_for_each_answer_id']
df.columns = new_col

# sort the cosine similarity values in descending order
df = df.sort_values('cosine_score_for_each_answer_id',axis=0, ascending=False)

# check the top 20 cosine scores
df.head(20)






# Set the cosine similarity threshold
MODEL_4_THRESHOLD = 0.94


# filter out all rows that have a cosine_score >= THRESHOLD
df_selected = df[df['cosine_score_for_each_answer_id'] >= MODEL_4_THRESHOLD]

# remove the first row because this row is the question we asked
df_selected = df_selected[1:]

num_answers = len(df_selected)

print('Number of answers chosen: ', num_answers)

print('This is a sample of the answers the model has selected.')

# print the answers that have been selected as well as the associated cosine scores
df_selected.head(10)


# reset the index
df_selected.reset_index(inplace=True)

# rename the columns
new_names = ['answers_id', 'cosine_score_for_each_answer_id']
df_selected.columns = new_names

# create a list with all answer id values from df
answer_list = list(df_selected['answers_id'])

# display the list
# answer_list


print('Question id: ', question_id)
print('Question Title: ', question_title)
print('\n')
print('Question Body:\n ', question_body)


# Print info on the professionals who can answer this question

#print('\n')
print('Model 4')
print('Number of professionals selected: ', len(answer_list)) # correct this. there could be duplicates
print('== Printing info on each professional who was selected ==')

# set the index
df_professionals = df_professionals.set_index('professionals_id')

# Create an empty list to store the professional id's that are
# associated with the answers that have been selected,
model_4_list = []

# set the index of df_train to be the question id
df_qa_prof = df_qa_prof.set_index('answers_id')


for ans_id in answer_list:
    
    # print the professional's id (i.e. their name)
    # get the prof id of the person who wrote the answer
    prof_id = df_qa_prof.loc[ans_id, 'answers_author_id']
    print('\n')
    print('==> Professional id: ', prof_id)
    model_4_list.append(prof_id)
    
    print('Answer id: ', ans_id)
    
    
    # print their job title:
    title = df_professionals.loc[prof_id, 'professionals_headline']
    print('Title: ', title)
    
    # print the industry they work in
    industry = df_professionals.loc[prof_id, 'professionals_industry']
    print('Industry: ', industry)
    
    # Print the answer that they wrote which was similar the question being asked
    answer = df_qa_prof.loc[ans_id,'answers_body']
    
    print('==Answer given to similar question:\n',answer)


# uncomment the next line to print the list of professional id's

# model_4_list








# Note that there could be duplicate professional id's
# in these lists.

print('Model 1 Tags: ', len(model_1_list))
print('Model 2 Tfidf: ',len(model_2_list))
print('Model 3 TSVD: ',len(model_3_list))
print('Model 4 GloVe: ',len(model_4_list))



# Join all the lists
combined_list = model_1_list + model_2_list + model_3_list + model_4_list
# Create a dataframe containing all professionals
df_selected = pd.DataFrame(combined_list, columns=['professionals_id'])

# Drop any duplicate id's.
# Because model 2 and model 3 select professionals based on answers, 
# there is a possibility that the same professional could be selected 
# multiple times bcause they gave several answers that matched the Question.

# remove the duplicates
df_selected = df_selected.drop_duplicates('professionals_id')
# get the total number of professionals
total = len(df_selected)

print(total, 'professionals are able to answer the Question.')



def new_member(x):
    # get the value from df_professionals
    num_days_member = df_professionals.loc[x, 'num_days_member']

    if num_days_member <= 30:
        return 1
    else:
        return 0

df_selected['new_member'] = df_selected['professionals_id'].apply(new_member)


# Get the id of the student asking the question.
# student_id variable was captured above.

def past_interaction(x):
    # Filter out all the questions this professional has answered in the past
    df_past = df_qa_prof[df_qa_prof['answers_author_id'] == x]

    # Get a list of stuents who've asked the above questions
    student_list = list(df_past['questions_author_id'])

    # Check if the student asking this question is in student_list
    if student_id in student_list:
        return 1 # there was a past interaction
    else:
        return 0 # there has been no past interaction

# create a new column that shows if there was a past interaction
df_selected['past_interaction'] = \
df_selected['professionals_id'].apply(past_interaction)


df_selected.head()


# Has this professional answered a question within
# 30 days of the most recent answer posted on CareerVillage?
# Yes --> send email

# convert the answers_date_added to pandas datetime
df_answers['answers_date_added'] = \
pd.to_datetime(df_answers['answers_date_added'])

# get the date of the most recent answer
newest_answer_date = df_answers['answers_date_added'].max()

# Get the number of days a question was answered from the most recent answer posted
# on CareerVillage.

def days_from_newest_answer(x):
    
    num_days = (newest_answer_date - x).days
    
    return num_days

# create a new column
df_answers['days_from_newest_answer'] = \
df_answers['answers_date_added'].apply(days_from_newest_answer)

# filter out all rows where days_from_newest_answer <= 30
df_filtered = df_answers[df_answers['days_from_newest_answer'] <= 30]

# Drop duplicate professional id's because some professionals
# may have abswered multiple questions in that time period.
df_filtered = df_filtered.drop_duplicates('answers_author_id')

# get a list of professionals that made these recent answers
prof_list = list(df_filtered['answers_author_id'])


def recent_answer(x):
    if x in prof_list:
        return 1
    else:
        return 0

# create a new column
df_selected['recent_answer'] = \
df_selected['professionals_id'].apply(recent_answer)


# Has this professional made a comment within
# 30 days of the most recent answer posted on CareerVillage?
# Yes --> send email

# convert the answers_date_added to pandas datetime
df_comments['comments_date_added'] = pd.to_datetime(df_comments['comments_date_added'])

# Get the number of days a question was answered from the most recent answer posted
# on CareerVillage.

def days_from_newest_answer(x):
    
    num_days = (newest_answer_date - x).days
    
    return num_days

# create a new column
df_comments['days_from_newest_answer'] = \
df_comments['comments_date_added'].apply(days_from_newest_answer)

# filter out all rows where days_from_newest_answer <= 30
df_filtered = df_comments[df_comments['days_from_newest_answer'] <= 30]

# Drop duplicate professional id's because some professionals
# may have made multiple comments in that time period.
df_filtered = df_filtered.drop_duplicates('comments_author_id')

# get a list of professionals that made these recent comments
prof_list = list(df_filtered['comments_author_id'])

# add a new column to df_selected
def recent_comment(x):
    if x in prof_list:
        return 1
    else:
        return 0

df_selected['recent_comment'] = df_selected['professionals_id'].apply(recent_comment)



# sum up the row scores for each professional in df_selected
def sum_rows(row):
    
    total = row['new_member'] + row['recent_answer'] + \
    row['recent_comment'] + row['past_interaction']
    
    return total
    
df_selected['total_score'] = df_selected.apply(sum_rows, axis=1)


# filter out rows where the score > 0
df_send_email = df_selected[df_selected['total_score'] > 0]



final_selection_list = list(df_send_email['professionals_id'])

num_selected = len(final_selection_list)

print('=== Final Results ===\n')

print(num_selected, 'professionals are likely to respond to the email.')

#print('These are their names:\n', final_selection_list)

print('These are their scores.')

# Print the list of professionals that have a high likelihood of
# responding to an email notification
df_send_email.head(20)





print('This shows which model selected each chosen professional:\n')

for prof_id in final_selection_list:
    if prof_id in model_1_list:

        print('Model 1 Tags: ', prof_id)

for prof_id in final_selection_list:
    if prof_id in model_2_list:

        print('Model 2 Tfidf: ', prof_id)
        
for prof_id in final_selection_list:
    if prof_id in model_3_list:

        print('Model 3 TSVD: ', prof_id)

for prof_id in final_selection_list:
    if prof_id in model_4_list:

        print('Model 4 GloVe: ', prof_id)


# End of Recommender System
#====================================================================#








import pandas as pd
# The next two lines causes all the text to appear. Sentences are not truncated.
# All columns and all rows are displayed. Nothing is hidden.
# Note: this must be in the same cell as import pandas as pd
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None)


results_dict = {
'question_index': [777,999,2043,2487,3618,1710,1000,'custom question','custom question'],
'question_title': ['I want to major in computer science. What classes should I take?',
                 'What is required to become a firefighter?',
                 'What exactly, is the difference between a psychologist and psychiatrist?',
                 'How do I decide what career I want to choose?',
                 'What are the best ways to maintain a work and school balance?',
                 'I want to become an army officer. What can I do to become an army officer?',
                 'What are the challenges I may face pursuing a career in science, and how can I stand out among the rest?',
                 'How do I become a data scientist?', 'How do I become a plumber?'],
'Model_1_Tags': ['rec: 481 fp: 0','rec: 2 fp: 0','rec: 1 fp: 1','rec: 0 fp: 0','rec: 1 fp: 1','rec: 21 fp: 2','rec: 0 fp: 0','rec: 105 fp: 1','rec: 4 fp: 0'],
'Model_2_Tfidf': ['rec: 392 fp: 12','rec: 10 fp: 1','rec: 9 fp: 0','rec: 2 fp: 1','rec: 1 fp: 1','rec: 32 fp: 0','rec: 0 fp: 0','rec: 99 fp: 10','rec: 4 fp: 4'],
'Model_3_TSVD': ['rec: 140 fp: 1','rec: 0 fp: 0','rec: 0 fp: 0','rec: 6 fp: 1','rec: 0 fp: 0','rec: 46 fp: 0','rec: 0 fp: 0','rec: 108 fp: 8','rec: 7 fp: 7'],
'Model_4_GloVe': ['rec: 53 fp: 0','rec: 0 fp: 0','rec: 0 fp: 0','rec: 34 fp: 4','rec: 1588 fp: 0','rec: 0 fp: 0','rec: 26 fp: 0','rec: 0 fp: 0','rec: 0 fp: 0'],
'Final_Filter_Output': ['rec: 29 fp: 1','rec: 2 fp: 1','rec: 1 fp: 1','rec: 2 fp: 0','rec: 54 fp: 0','rec: 9 fp: 0','rec: 3 fp: 0','rec: 14 fp: 1','rec: 2 fp: 1']
  
}

    
df_results = pd.DataFrame(results_dict)

#df_results.head(10)


df_results.head(10)

