# Import important packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data Preprocessing
import re
import spacy

# Modeling
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC


# Load the data
train = pd.read_csv('/kaggle/input/learn-ai-bbc/BBC News Train.csv')
test = pd.read_csv('/kaggle/input/learn-ai-bbc/BBC News Test.csv')

# Take a look at the training set
print(train.info())
train.head()


# Check the ratio of the categories
labels = train['Category'].value_counts().index
values = train['Category'].value_counts().values

plt.pie(values, labels = labels, colors = ['lightsalmon', 'skyblue', 'wheat', 'lavender', 'lightgrey'], autopct = '%1.0f%%')
plt.title('Ratio of the Categories');


# Take a look at the text in each category
pd.set_option('display.max_colwidth', 200)
for category, group in train.groupby('Category'):
    print(f'Category: {category}')
    print(group['Text'].head())
    print()


fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 4))
money_regex = r'[€£¥$]\d+(?:[\.,]\d+)?[a-zA-Z]*'
money_labels = train[train['Text'].str.contains(money_regex)]['Category'].value_counts().index
money_values = train[train['Text'].str.contains(money_regex)]['Category'].value_counts().values
ax1.pie(money_values, labels = money_labels, colors = ['lightsalmon', 'skyblue', 'wheat', 'lavender', 'lightgrey'], autopct = '%1.0f%%')
ax1.set_title('Ratio of Cateroies with money_regex')

score_regex = r'\d+\s*-\s*\d+'
score_labels = train[train['Text'].str.contains(score_regex)]['Category'].value_counts().index
score_values = train[train['Text'].str.contains(score_regex)]['Category'].value_counts().values
ax2.pie(score_values, labels = score_labels, colors = ['lightsalmon', 'skyblue', 'wheat', 'lavender', 'lightgrey'], autopct = '%1.0f%%')
ax2.set_title('Ratio of Cateroies with score_regex');


# Check if any text contains a capital letter
print('Number of capital letters:', train['Text'].str.contains(r'[A-Z]').sum())

# Check if any text contains a URL
print('Number of URL:', train['Text'].str.contains(r'http').sum())
      
# Check if there is duplicated rows with same text and same category
print('Duplicated rows with same text and category:', train.duplicated(subset=['Text', 'Category']).sum())


# Remove the rows with duplicated 'Text' and 'Category', only keep the first record
train = train.drop_duplicates(subset = ['Text', 'Category'])


# Check if there is duplicated rows with same text but different category
print('Duplicated rows with different category:', train.duplicated(subset = ['Text']).sum())


# Text cleaning function
def text_cleaning(data):
    data['Text'] = data['Text'].str.replace(money_regex, 'dollar', regex = True)   # replace money-related pattens with 'dollar'
    data['Text'] = data['Text'].str.replace(score_regex, 'score', regex = True)    # replace score-related patterns with 'score'
    data['Text'] = data['Text'].apply(lambda x: re.sub(r'\w*\d+\w*', '', x))       # remove digit and words contains digit
    data['Text'] = data['Text'].apply(lambda x: re.sub(r'\s+', ' ', x))            # remove extra space

    return data


# Clean training and test set
train = text_cleaning(train)
test = text_cleaning(test)


# Map category in training set to integer
category_mapping = {'business': 1, 'tech': 2, 'politics': 3, 'sport': 4, 'entertainment': 5} 
train['category_int'] = train['Category'].map(category_mapping)


# Load the pre-trained English model from spaCy and disable unnecessary compenents to improve efficiency
nlp = spacy.load('en_core_web_sm', disable = ['ner', 'parser'])

# Define a function to tokenize the text data, remove stop words and punctuation, and lemmatize the tokens
def process_helper(text):          
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc 
              if not (token.is_stop or token.is_punct) and len(token) > 2]   
    return tokens

# Define a function to process the text data for modeling
def text_processing(data):
    data['Tokens'] = data['Text'].apply(process_helper)               
    data['Clean_text'] = data['Tokens'].apply(lambda x: ' '.join(x))   # Convert tokens into space-separated strings for later use in feature extraction
    return data


# Apply text_processing function on both training and test set
train = text_processing(train)
test = text_processing(test)


# Count the number of words in each article
train['word_count'] = train['Tokens'].apply(lambda x: len(x))

# Plot the word count of different categories
sns.boxplot(data = train, x = 'Category', color = 'skyblue', y = 'word_count')
plt.title('Word Count by Category');


# Create a CountVectorizer
cnt_vt = CountVectorizer()

# Fit and transform the 'clean_text' to create the document-term matrix
word_count_matrix = cnt_vt.fit_transform(train['Clean_text'])

# Convert the document-term matrix to a DataFrame
cnt_df = pd.DataFrame(word_count_matrix.toarray(), columns = cnt_vt.get_feature_names_out(),
                   index = train['Category'])

# Sum the word frequency by category
word_counts = cnt_df.groupby('Category').sum()
word_counts = word_counts.reset_index()


# Define a function to plot top words for a given category
def plot_top_words(category, n, ax):
    df = word_counts[word_counts['Category'] == category].drop('Category', axis = 1).T
    df.columns = ['Frequency']
    df = df.sort_values(by = 'Frequency', ascending = False).head(n)

    sns.barplot(x = df['Frequency'], y = df.index, color = 'skyblue', ax = ax)
    ax.set_title(f'Top words in {category}')
    ax.set_ylabel('')
    
    return ax


# Plot the categories
fig, ax = plt.subplots(nrows = 1, ncols = 5, figsize = (20, 6))
plt.subplots_adjust(wspace = 0.3)
for i, category in enumerate(train['Category'].unique()):
    plot_top_words(category, 20, ax[i])


# Define a function to remove top meaningless common words from the tokens
def rm_words(data, words_to_rm):
    data['Tokens'] = data['Tokens'].apply(lambda tokens: [word for word in tokens if word not in words_to_rm])
    for word in words_to_rm:
        data['Clean_text'] = data['Clean_text'].str.replace(word, '')

    return data

train = rm_words(train, ['say', 'new', 'year'])
test = rm_words(test, ['say', 'new', 'year'])


# Split the training set
tra, val = train_test_split(train, test_size = 0.2, random_state = 101)


# Create a TfidfVectorizer
tfidf_vt1 = TfidfVectorizer()

# Fit the 'clean_text' of 'tra' dataset to the TF-IDF vectorizer
tfidf_vt1.fit(tra['Clean_text'])

# Transform the 'clean_text' of 'tra' dataset to a document-term matrix of TF-IDF features
tfidf_mx_tra = tfidf_vt1.transform(tra['Clean_text'])


nmf_model1 = NMF(n_components = 5, random_state = 42)
nmf_model1.fit(tfidf_mx_tra)
w1 = nmf_model1.transform(tfidf_mx_tra)       # document-topic matrix
h1 = nmf_model1.components_                   # topic-word matrix


# Define a function to extract top words for each topic from the topic-word matrix(H)
def get_topic_words(tfidf_vectorizer, H_matrix, n):
    words = np.array(tfidf_vectorizer.get_feature_names_out())
    top_words = lambda t: [words[i] for i in np.argsort(t)[:-n-1:-1]]
    topic_vocabs = [top_words(t) for t in H_matrix]
    topics = [','.join(v) for v in topic_vocabs]
    return topics

# Get top words in each topics
get_topic_words(tfidf_vt1, h1, 15)


# Define a function to get the prediction results and accuracy
def pred_results(w, y_true, columns, mapping):
    df = pd.DataFrame(np.round(w, 2), columns = columns)
    df['y_pred'] = df.idxmax(axis = 1)
    df['y_true'] = y_true.reset_index(drop=True)

    # Replace string categories to integer
    df['y_pred'] = df['y_pred'].map(mapping)
    
    # Calculate accuracy score
    acc = accuracy_score(df['y_true'], df['y_pred'])

    return acc


# Get the accuracy score on the tra dataset
columns_nmf1 = ['sport', 'politics', 'business', 'entertainment', 'tech']
tra_acc_nmf1 = pred_results(w1, tra['category_int'], columns_nmf1, category_mapping)

print('Accuracy Score on the training set:', round(tra_acc_nmf1, 2))


# Transform the 'clean_text' of 'val' dataset to a document-term matrix of TF-IDF features
tfidf_mx_val = tfidf_vt1.transform(val['Clean_text'])

# predit topics for validation data
w2 = nmf_model1.transform(tfidf_mx_val)       # document-topic matrix

# Get the accuracy score on the tra dataset
val_acc_nmf1 = pred_results(w2, val['category_int'], columns_nmf1, category_mapping)

print('Accuracy Score on the validation set:', round(val_acc_nmf1, 2))


# Get TF-IDF matrix of the entire training set
tfidf_vt = TfidfVectorizer(ngram_range = (1, 2))
tfidf_mx = tfidf_vt.fit_transform(train['Clean_text'])

# Create a DataFrame to hold the TF-IDF matrix, then group by 'Category' to sum the TF-IDF values of each word within each category
tfidf_df = pd.DataFrame(tfidf_mx.toarray(), columns = tfidf_vt.get_feature_names_out(),
                        index = train['Category'])

tfidf_category = tfidf_df.groupby('Category').sum().reset_index()

# Define a function to get the top n words for each category
def top_words(category, n):
    df = tfidf_category[tfidf_category['Category'] == category].drop('Category', axis = 1).T
    df.columns = ['Frequency']
    df = df.sort_values(by = 'Frequency', ascending = False).head(n)
    return df.index

# Extract top 20 words for each category
top_words_lst = []
for category in train['Category'].unique():
    top_words_lst.append(top_words(category, 20))

top_words_lst


# Create a dictionary for top words in each category
category_dict = {
    'business' : top_words_lst[0].to_list(),
    'tech': top_words_lst[1].to_list(),
    'politics': top_words_lst[2].to_list(),
    'sport': top_words_lst[3].to_list(),
    'entertainment': top_words_lst[4].to_list()
    }

# Define a function to assign category according to the count of matching words between word_list and the category_dict
def assign_category(word_lists, category_dict):
    categories = []
    for words in word_lists:
        match_counts = {}
        for category, top_words in category_dict.items():
            match_counts[category] = len([w for w in words if w in top_words])
        category = max(match_counts, key = match_counts.get)
        categories.append(category)
    return categories


# Define a function of scoring for GridSearchCV
def accuracy(estimator, X, y):
    tfidf = estimator.named_steps['tfidf']
    tfidf_mx = tfidf.transform(X)
    nmf = estimator.named_steps['nmf']
    w = nmf.transform(tfidf_mx)
    h = nmf.components_

    # Get the top word in each topic
    topics = get_topic_words(tfidf, h, 5)
    topics_lst = [w.split(',') for w in topics]
    columns = assign_category(topics_lst, category_dict)

    # Get accuracy score
    accu= pred_results(w, y, columns, category_mapping)

    return accu


# Create tfidf-matrix using 'best_params'
tfidf_vt2 = TfidfVectorizer(max_df = 0.8, min_df = 2, ngram_range = (1, 2))
tfidf_vt2.fit(tra['Clean_text'])
tfidf_mx_tra2 = tfidf_vt2.transform(tra['Clean_text'])

# Train NMF model using 'best_params'
nmf_model2 = NMF(n_components = 5, init = 'nndsvdar', solver = 'mu', beta_loss= 'kullback-leibler', random_state = 0)
nmf_model2.fit(tfidf_mx_tra2)
w2 = nmf_model2.transform(tfidf_mx_tra2)
h2 = nmf_model2.components_

# Get top words in each topics
topics = get_topic_words(tfidf_vt2, h2, 5)
topics_lst = [w.split(',') for w in topics]

# Get the accuracy score on the tra dataset
columns = assign_category(topics_lst, category_dict)
tra_acc_nmf2 = pred_results(w2, tra['category_int'], columns, category_mapping)

print('Accuracy Score on the training set:', round(tra_acc_nmf2, 2))

# Transform the 'clean_text' of 'val' dataset to a document-term matrix of TF-IDF features
tfidf_mx_val2 = tfidf_vt2.transform(val['Clean_text'])

# predit topics for validation data
val_w2 = nmf_model2.transform(tfidf_mx_val2)       # document-topic matrix

# Get the accuracy score on the val dataset
val_acc_nmf2 = pred_results(val_w2, val['category_int'], columns, category_mapping)

print('Accuracy Score on the validation set:', round(val_acc_nmf2, 2))

# Transform the 'clean_text' of test dataset to a document-term matrix of TF-IDF features
tfidf_mx_test1 = tfidf_vt2.transform(test['Clean_text'])

# I will Use nmf_model2 to predict the category in the test set
test_w1 = nmf_model2.transform(tfidf_mx_test1)       # document-topic matrix

# Define a funciton to get the predicted category of the test set
def pred_category(w, columns, data):
    df = pd.DataFrame(np.round(w, 2), columns = columns)
    df['Category'] = df.idxmax(axis = 1)
    df['ArticleId'] = data['ArticleId']

    return df[['ArticleId', 'Category']]

# Get the result
solution_nmf_mu_80 = pred_category(test_w1, columns, test)


# Create tfidf-matrix of the whole training set using best_params
tfidf_vt_train = TfidfVectorizer(max_df = 0.8, min_df = 2, ngram_range = (1, 2))
tfidf_vt_train.fit(train['Clean_text'])
tfidf_mx_train = tfidf_vt_train.transform(train['Clean_text'])

# Train NMF model using best_params
nmf_model = NMF(n_components = 5, init = 'nndsvdar', solver = 'mu', beta_loss= 'kullback-leibler', random_state = 1)
nmf_model.fit(tfidf_mx_train)
w_train = nmf_model.transform(tfidf_mx_train)
h_train = nmf_model.components_

# Get top words in each topics
topics = get_topic_words(tfidf_vt_train, h_train, 5)
topics_lst = [w.split(',') for w in topics]

# Get the accuracy score on the train dataset
columns = assign_category(topics_lst, category_dict)
train_accuracy = pred_results(w_train, train['category_int'], columns, category_mapping)

print('Accuracy Score on the entire training set:', round(train_accuracy, 2))

# Transform the 'clean_text' of test dataset to a document-term matrix of TF-IDF features
tfidf_mx_test2 = tfidf_vt_train.transform(test['Clean_text'])

# Predict the category in the test set
test_w2 = nmf_model.transform(tfidf_mx_test2)       # document-topic matrix

# Get the result
solution_nmf_mu_100 = pred_category(test_w2, columns, test)


# Split the training set 
tra_30, val_70 = train_test_split(train, test_size = 0.7, random_state = 1, stratify = train['category_int'])
tra_40, val_60 = train_test_split(train, test_size = 0.6, random_state = 1, stratify = train['category_int'])
tra_50, val_50 = train_test_split(train, test_size = 0.5, random_state = 1, stratify = train['category_int'])
tra_60, val_40 = train_test_split(train, test_size = 0.4, random_state = 1, stratify = train['category_int'])
tra_70, val_30 = train_test_split(train, test_size = 0.3, random_state = 1, stratify = train['category_int'])
tra_80, val_20 = train_test_split(train, test_size = 0.2, random_state = 1, stratify = train['category_int'])
tra_90, val_10 = train_test_split(train, test_size = 0.1, random_state = 1, stratify = train['category_int'])


# Train the models and return the results
tra_lst = [tra_30, tra_40, tra_50, tra_60, tra_70, tra_80, tra_90]
val_lst = [val_70, val_60, val_50, val_40, val_30, val_20, val_10]
tra_accuracy = []
val_accuracy = []

for i in range(len(tra_lst)):
    tfidf = TfidfVectorizer(max_df = 0.8, min_df = 2, ngram_range = (1, 2))
    tfidf.fit(tra_lst[i]['Clean_text'])
    tfidf_mx = tfidf.transform(tra_lst[i]['Clean_text'])

    nmf = NMF(n_components = 5, init = 'nndsvdar', solver = 'mu', beta_loss= 'kullback-leibler', random_state = 2)
    nmf.fit(tfidf_mx)
    w = nmf.transform(tfidf_mx)
    h = nmf.components_

    topics = get_topic_words(tfidf, h, 5)
    topics_lst = [w.split(',') for w in topics]

    columns = assign_category(topics_lst, category_dict)
    tra_accu = pred_results(w, tra_lst[i]['category_int'], columns, category_mapping)
    tra_accuracy.append(round(tra_accu, 3))

    tfidf_mx_val = tfidf.transform(val_lst[i]['Clean_text'])
    w_val = nmf.transform(tfidf_mx_val)
    val_accu = pred_results(w_val, val_lst[i]['category_int'], columns, category_mapping)
    val_accuracy.append(round(val_accu, 3))


import warnings

# Create a dataframe for the NMF accuracy on different subsets of data
nmf_results = pd.DataFrame({
    'Train Size': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] * 2,
    'Metrics': ['train'] * 7 + ['test'] * 7,
    'Accuracy': tra_accuracy + val_accuracy
})

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", message="use_inf_as_na option is deprecated") 
warnings.filterwarnings("ignore", message = "When grouping with a length-1 list-like")

# Plot the results
sns.lineplot(data = nmf_results, x = 'Train Size', y = 'Accuracy', hue = 'Metrics', palette = ['skyblue', 'lightsalmon'])
plt.legend(title = '')
plt.title('NMF Accuracy on Varying Dataset Sizes')
plt.show();


# Create tfidf-matrix using best_params
tfidf_vt_40 = TfidfVectorizer(max_df = 0.8, min_df = 5, ngram_range = (1, 2), max_features = 3000)
tfidf_vt_40.fit(tra_40['Clean_text'])
tfidf_mx_tra40 = tfidf_vt_40.transform(tra_40['Clean_text'])

# Train NMF model using best_params
nmf_model_40 = NMF(n_components = 5, init = 'nndsvdar', solver = 'mu', beta_loss= 'kullback-leibler', random_state = 42)
nmf_model_40.fit(tfidf_mx_tra40)
w40 = nmf_model_40.transform(tfidf_mx_tra40)
h40 = nmf_model_40.components_

# Get top words in each topics
topics = get_topic_words(tfidf_vt_40, h40, 5)
topics_lst = [w.split(',') for w in topics]

# Get the accuracy score on the tra_40 dataset
columns = assign_category(topics_lst, category_dict)
tra_acc_nmf40 = pred_results(w40, tra_40['category_int'], columns, category_mapping)

print('Accuracy Score on the training set:', round(tra_acc_nmf40, 2))

# Transform the 'clean_text' of 'val_60' dataset to a document-term matrix of TF-IDF features
tfidf_mx_val40 = tfidf_vt_40.transform(val_60['Clean_text'])

# predit topics for validation data
val_w40 = nmf_model_40.transform(tfidf_mx_val40)       # document-topic matrix

# Get the accuracy score on the val_40 dataset
val_acc_nmf40 = pred_results(val_w40, val_60['category_int'], columns, category_mapping)

print('Accuracy Score on the validation set:', round(val_acc_nmf40, 2))


# Transform the 'clean_text' of test dataset to a document-term matrix of TF-IDF features
tfidf_mx_test40 = tfidf_vt_40.transform(test['Clean_text'])

# Predict topics for test data
test_w40 = nmf_model_40.transform(tfidf_mx_test40)       # document-topic matrix

# Get the result
solution_nmf_mu_40 = pred_category(test_w40, columns, test)


# Determine the number of components needed to reach 95% explained variance
tfidf_vt3 = TfidfVectorizer(max_df = 0.8, max_features = 8000)  # set max feature to improve efficiency
tfidf_vt3.fit(tra['Clean_text'])
tfidf_mx_tra80 = tfidf_vt3.transform(tra['Clean_text'])

svd = TruncatedSVD(n_components=8000)
svd.fit(tfidf_mx_tra80)
cum_explained_var = np.cumsum(svd.explained_variance_ratio_)
n_comp = np.argmax(cum_explained_var >= 0.95) + 1

# Apply TruncatedSVD with the optimal number of components
svd = TruncatedSVD(n_components = n_comp)
svd.fit(tfidf_mx_tra80)
tra_svd = svd.transform(tfidf_mx_tra80)

# Fit the 'tra_svd' to the LinearSVC model
clf = LinearSVC(random_state = 42)
clf.fit(tra_svd, tra['category_int'])

# Transform the tfidf matrix of validataion set by TruncatedSVD
tfidf_mx_val80 = tfidf_vt3.transform(val['Clean_text'])
val_svd = svd.transform(tfidf_mx_val80)
y_pred = clf.predict(val_svd)

# Calculate the accuracy
print('The accuracy of the LinearSVC model:', round(accuracy_score(val['category_int'], y_pred), 2))


SVC_tra_accuracy = []
SVC_val_accuracy = []

for i in range(len(tra_lst)):
    # Get tfidf matrix of the train set
    tfidf = TfidfVectorizer(max_df = 0.8, max_features = 8000)
    tfidf.fit(tra_lst[i]['Clean_text'])
    tfidf_mx_tra = tfidf.transform(tra_lst[i]['Clean_text'])

    # Determine the number of components needed to reach 95% explained variance
    svd = TruncatedSVD(n_components=8000)
    svd.fit(tfidf_mx_tra)
    cum_explained_var = np.cumsum(svd.explained_variance_ratio_)
    n_comp = np.argmax(cum_explained_var >= 0.95) + 1

    # Apply TruncatedSVD with the optimal number of components
    svd = TruncatedSVD(n_components = n_comp)
    svd.fit(tfidf_mx_tra)
    tra_svd = svd.transform(tfidf_mx_tra)

    # Train the LinearSVC model on the train set
    clf = LinearSVC(random_state = 42)
    clf.fit(tra_svd, tra_lst[i]['category_int'])

    # Calculate the accuracy of the train set
    tra_pred = clf.predict(tra_svd)
    tra_accu = round(accuracy_score(tra_lst[i]['category_int'], tra_pred), 4)
    SVC_tra_accuracy.append(tra_accu)

    # Get tfidf matrix of the validation set
    tfidf_mx_val = tfidf.transform(val_lst[i]['Clean_text'])

    # Transform the tfidf matrix of validataion set by TruncatedSVD
    val_svd = svd.transform(tfidf_mx_val)

    # Calculate the accuracy of the validation set
    val_pred = clf.predict(val_svd)
    val_accu = round(accuracy_score(val_lst[i]['category_int'], val_pred), 4)
    SVC_val_accuracy.append(val_accu)


# Create a dataframe for the SVC model results
svc_results = pd.DataFrame({
    'Train Size': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] * 2,
    'Metrics': ['train'] * 7 + ['test'] * 7,
    'Accuracy': SVC_tra_accuracy + SVC_val_accuracy
})

# Plot the results of SVC models
sns.lineplot(data = svc_results, x = 'Train Size', y = 'Accuracy', hue = 'Metrics', palette = ['skyblue', 'lightsalmon'])
plt.legend(title = '')
plt.title('LinearSVC Accuracy on Varying Dataset Sizes');


# Train the SVC model with 90% of the training set and get the results 

# Get tfidf matrix of the train set
tfidf_90 = TfidfVectorizer(max_df = 0.8, max_features = 8000)
tfidf_90.fit(tra_90['Clean_text'])
tfidf_mx_tra90 = tfidf_90.transform(tra_90['Clean_text'])

# Determine the number of components needed to reach 95% explained variance
svd = TruncatedSVD(n_components=8000)
svd.fit(tfidf_mx_tra90)
cum_explained_var = np.cumsum(svd.explained_variance_ratio_)
n_comp = np.argmax(cum_explained_var >= 0.95) + 1

# Apply TruncatedSVD with the optimal number of components
svd = TruncatedSVD(n_components = n_comp)
svd.fit(tfidf_mx_tra90)
tra_svd90 = svd.transform(tfidf_mx_tra90)

# Train the LinearSVC on the 90% of training set
clf_90 = LinearSVC(random_state = 42)
clf_90.fit(tra_svd90, tra_90['category_int'])

# Get tfidf matrix of the test set
tfidf_mx_test90 = tfidf_90.transform(test['Clean_text'])

# Transform the tfidf matrix of test set by TruncatedSVD
test_svd90 = svd.transform(tfidf_mx_test90)

# Create a dataframe for solution
solution_svc90 = pd.DataFrame({
    'ArticleId': test['ArticleId'],
    'Category': clf_90.predict(test_svd90)
})

# Map integer category to string
int_category = {1: 'business', 2: 'tech', 3: 'politics', 4: 'sport', 5: 'entertainment'}
solution_svc90['Category'] = solution_svc90['Category'].map(int_category)

# Save the solution
solution_svc90.to_csv('solution_svc90.csv', index = False)


# Train the SVC model with the entire training set and get the results 

# Get tfidf matrix of the train set
tfidf = TfidfVectorizer(max_df = 0.8, max_features = 8000)
tfidf.fit(train['Clean_text'])
tfidf_mx_train = tfidf.transform(train['Clean_text'])

# Determine the number of components needed to reach 95% explained variance
svd = TruncatedSVD(n_components=8000)
svd.fit(tfidf_mx_train)
cum_explained_var = np.cumsum(svd.explained_variance_ratio_)
n_comp = np.argmax(cum_explained_var >= 0.95) + 1

# Apply TruncatedSVD with the optimal number of components
svd = TruncatedSVD(n_components = n_comp)
svd.fit(tfidf_mx_train)
train_svd = svd.transform(tfidf_mx_train)

# Train the LinearSVC on the 90% of training set
clf_100 = LinearSVC(random_state = 42)
clf_100.fit(train_svd, train['category_int'])

# Get tfidf matrix of the validation set
tfidf_mx_test = tfidf.transform(test['Clean_text'])

# Transform the tfidf matrix of test set by TruncatedSVD
test_svd = svd.transform(tfidf_mx_test)

# Create a dataframe for solution
solution_svc100 = pd.DataFrame({
    'ArticleId': test['ArticleId'],
    'Category': clf_100.predict(test_svd)
})

# Map integer category to string
int_category = {1: 'business', 2: 'tech', 3: 'politics', 4: 'sport', 5: 'entertainment'}
solution_svc100['Category'] = solution_svc100['Category'].map(int_category)

