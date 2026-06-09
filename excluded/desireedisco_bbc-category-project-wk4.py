import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
import nltk
import itertools
from collections import Counter

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics import accuracy_score#, make_scorer
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')
nltk.download('averaged_perceptron_tagger_eng')


labeled_data = pd.read_csv("/kaggle/input/learn-ai-bbc/BBC News Train.csv")
test = pd.read_csv("/kaggle/input/learn-ai-bbc/BBC News Test.csv")

print(labeled_data.head())
print(test.head())


print(labeled_data.info())
print(labeled_data.describe(include=['object']))


# drop duplicates that are duplicated in text and category (I would not drop or drop both if the categories were different)
labeled_data = labeled_data.drop_duplicates(subset=['Text', 'Category'])
labeled_data.reset_index(drop=True, inplace=True)

# check to make sure we removed all duplicates
print(labeled_data.describe(include=['object']))


# get counts for each category
category_counts = labeled_data['Category'].value_counts()

# percentage of each class
category_percentages = labeled_data['Category'].value_counts(normalize=True) * 100

fig, ((ax1, ax2)) = plt.subplots(figsize=(11, 5), nrows=1, ncols=2)

sns.histplot(labeled_data['Category'], ax=ax1)

# category distribution
ax2.pie(category_percentages, labels=category_percentages.index, autopct='%1.1f%%', startangle=90)
ax2.set_title('Distribution of Articles by Category')
ax2.axis('equal')
plt.show()


# get word count in 'Text' column to look at word count distribution between the different categories
labeled_data['text_len'] = labeled_data['Text'].apply(lambda x: len(x.split()))


# ploting the boxplot for word count distribution
plt.figure(figsize=(15, 6))
sns.boxplot(x='text_len', y='Category', data=labeled_data)
plt.title('Distribution of Word Count by Category', fontsize=14)
plt.ylabel('Category', fontsize=12)
plt.xlabel('Word Count', fontsize=12)
plt.show()


def clean_text(text):
  # truncate string
  text = str(text[:1000])

  # remove punctuation
  punct_pattern = r'[^\w\s]+'
  text = re.sub(punct_pattern, '', text)

  # remove numbers
  num_pattern = r'[0-9]+'
  text = re.sub(num_pattern, '', text)

  # remove spaces
  text = re.sub(' +', ' ', text)

  # tokenize text
  tokens = word_tokenize(text)
    
  # remove stop words and join back to text string
  stop_words = stopwords.words('english')
  stop_words_removed = [word for word in tokens if word not in stop_words and len(word)>2]
  
  clean_text = ' '.join(stop_words_removed)
  return clean_text


# clean text and remove stop words
labeled_data.loc[:,'text_clean'] = labeled_data.loc[:,'Text'].apply(clean_text)


# get NLTK word list
word_list = set(words.words())

# get lemmas from wordnet
# get all synsets from wordnet.all_synsets() then get the lemma names from each synset
# this will help us expand the words list with different words and we just want to add the lemmas
# when we check the words we will take down to lemma form if there is not a direct match right away
wordnet_lemmas = set(lemma.name() for synset in wordnet.all_synsets() for lemma in synset.lemmas())

# combine lists
word_list_master = word_list.union(wordnet_lemmas)

print(len(word_list_master)) # the count should be 327276


# initiate a wordnet lemmatizer
wnl = WordNetLemmatizer()

# method to check if word is a recognized word
def check_words(text):
    # this is a part of speach dictionary to match coding between NLTK
    # maps part-of-speech tags (like 'NN' for noun, 'VB' for verb) to simplified tags ('n', 'v', 'a', 'r') used by WordNet
    pos_ref = {'NN': 'n', 'NNP': 'n', 'NNPS': 'n', 'NNS': 'n', 'JJ': 'a', 'JJR': 'a', 'JJS': 'a', 'RB': 'r', 'RBR': 'r', 'RBS': 'r', 'RP': 'r', 'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v', 'VBP': 'v', 'VBZ': 'v'}
    token_lst = text.split()
    
    # list of tokens with pos tags using nltk.pos_tag method
    lst_pos_tags = nltk.pos_tag(token_lst)

    # set of words not recognized
    not_in_words = set()
    
    # add recognized words to list
    recog_words = []

    
    # loop through tokens
    for token in lst_pos_tags:
        word = token[0]
            
        # get wordnet pos from NLTK pos tag
        pos = pos_ref.get(token[1])
        
        # if word in master word list add to recog_words
        if word in word_list_master:
            recog_words.append(word)
            pass

        # if word is not recognized right away check is lemma and see if it is recognized
        elif wnl.lemmatize(word) in word_list_master:
            recog_words.append(word)
            pass
        
        # otherwise look at token and pos tag
        else:
            
            # is pos tag is one from the dictionary above then check lemma using lematize with the pos information
            if pos is not None:
                lemma = wnl.lemmatize(word, pos)
                
                if lemma in word_list_master:
                    recog_words.append(word)
                    pass
                
                # if the token is still not recognized then check to see if it is ascii
                else:
                    if word.isascii():
                        not_in_words.add(word)
  
           
            # for tokens not labeled with pos that is in above pos_ref dictionary then check to see if the token is isascii
            else:
                if word.isascii():
                    not_in_words.add(word)

    recog_words = ' '.join(recog_words)
    not_recog_words = ' '.join(not_in_words)

    # return recog_words_to_text and the not_in_words set as columns
    return pd.Series({'text_recognized': recog_words, 'text_not_recog': not_recog_words})


# test check_words methon
text1 = labeled_data.loc[1,'text_clean']
#print(text1)

good, bad = check_words(text1)
print(good)
print(bad)


labeled_data[['text_recognized', 'text_not_recog']] = labeled_data['text_clean'].apply(lambda x: pd.Series(check_words(x)))


test.loc[:,'text_clean'] = test.loc[:,'Text'].apply(clean_text)
test[['text_recognized', 'text_not_recog']] = test['text_clean'].apply(lambda x: pd.Series(check_words(x)))


# method to get top 10 words - use Counter
def get_top_n_words(category, column, n=None):
    # list containing all the extracted tokens
    token_collection = [token for text in labeled_data.loc[labeled_data['Category']==category, column] for token in text.split() if len(text) > 0]
    # get the top n words
    top_words = Counter(token_collection).most_common(n)
    words = []
    freq = []
    for word, count in top_words:
        words.append(word)
        freq.append(count)
    # the words and freq lists with the top words and their frequencies
    return np.array(words), np.array(freq)


# use barplots to show top words for title and text for each label
# I am using the tokenized text columns
fig, ((ax1, ax2, ax3, ax4, ax5)) = plt.subplots(1, 5, figsize=(15, 5), constrained_layout=True)

# I choose to show the top 20 words
words, freq = get_top_n_words('business', 'text_recognized', 20)
sns.barplot(x=freq, y=words, palette='Spectral', ax=ax1)
ax1.set_title('Business - Top Words')
ax1.set_xlabel('Word Count')

words, freq = get_top_n_words('tech', 'text_recognized', 20)
sns.barplot(x=freq, y=words, palette='Spectral', ax=ax2)
ax2.set_title('Tech - Top Words')
ax2.set_xlabel('Word Count')

words, freq = get_top_n_words('politics', 'text_recognized', 20)
sns.barplot(x=freq, y=words, palette='Spectral', ax=ax3)
ax3.set_title('Politics - Top Words')
ax3.set_xlabel('Word Count')

words, freq = get_top_n_words('sport', 'text_recognized', 20)
sns.barplot(x=freq, y=words, palette='Spectral', ax=ax4)
ax4.set_title('Sport - Top Words')
ax4.set_xlabel('Word Count')

words, freq = get_top_n_words('entertainment', 'text_recognized', 20)
sns.barplot(x=freq, y=words, palette='Spectral', ax=ax5)
ax5.set_title('Entertainment - Top Words')
ax5.set_xlabel('Word Count')

plt.show()


# convert text to TF-IDF
vectorizer = TfidfVectorizer(stop_words = 'english',
                             norm = 'l2',
                             encoding = 'latin-1',
                             ngram_range = (1,2),
                             max_df = 0.8)

categories = labeled_data['Category'].unique()
category_documents = []

# group into categories to see the difference in vocab for each category
for category in categories:
    cat_df = labeled_data[labeled_data['Category'] == category]
    category_documents.append(' '.join(list(cat_df['text_recognized'])))


tfidf_matrix = vectorizer.fit_transform(category_documents)
feature_names = vectorizer.get_feature_names_out()


df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
df_tfidf['label_'] = list(categories)

top_words_by_class = {}
for class_label in df_tfidf['label_'].unique():
  class_df = df_tfidf[df_tfidf['label_'] == class_label]
  # Sum TF-IDF scores for each word in this class, excluding the 'label' column
  word_scores = class_df.drop('label_', axis=1).sum(axis=0)
  top_words_by_class[class_label] = word_scores


# use barplots to show top words for title and text for each label
fig, ((ax1, ax2, ax3, ax4, ax5)) = plt.subplots(1, 5, figsize=(15, 5), constrained_layout=True)

N = 20 # Number of top words to retrieve

# I choose to show the top 20 words
biz_words = top_words_by_class['business'].nlargest(N)
sns.barplot(x=biz_words.values, y=biz_words.index, palette='Spectral', ax=ax1)
ax1.set_title('Business - Top Words')
ax1.set_xlabel('Word Count')
ax1.set_ylabel(None)

tech_words = top_words_by_class['tech'].nlargest(N)
sns.barplot(x=tech_words.values, y=tech_words.index, palette='Spectral', ax=ax2)
ax2.set_title('Tech - Top Words')
ax2.set_xlabel('Word Count')
ax2.set_ylabel(None)

pol_words = top_words_by_class['politics'].nlargest(N)
sns.barplot(x=pol_words.values, y=pol_words.index, palette='Spectral', ax=ax3)
ax3.set_title('Politics - Top Words')
ax3.set_xlabel('Word Count')
ax3.set_ylabel(None)

sport_words = top_words_by_class['sport'].nlargest(N)
sns.barplot(x=sport_words.values, y=sport_words.index, palette='Spectral', ax=ax4)
ax4.set_title('Sport - Top Words')
ax4.set_xlabel('Word Count')
ax4.set_ylabel(None)

ent_words = top_words_by_class['entertainment'].nlargest(N)
sns.barplot(x=ent_words.values, y=ent_words.index, palette='Spectral', ax=ax5)
ax5.set_title('Entertainment - Top Words')
ax5.set_xlabel('Word Count')
ax5.set_ylabel(None)

plt.show()


tfidf_ind = TfidfVectorizer(sublinear_tf=True, max_df=.80, min_df=3, norm='l2', ngram_range=(1, 3), stop_words='english')
tfidf_ind_matrix = tfidf_ind.fit_transform(labeled_data['text_recognized'])
features = tfidf_ind_matrix.toarray()
labels = labeled_data['Category']
features.shape


nmf = NMF(n_components=5, random_state=0)
tfidfdist = nmf.fit_transform(tfidf_ind_matrix)

tfidfmaintopics = np.argmax(tfidfdist, axis=1)
tfidfdist = pd.DataFrame(tfidfdist, columns=["Topic "+str(i) for i in range(1,6)])
tfidfdist["NMF_Category"]=tfidfmaintopics

# top words per category
words = tfidf_ind.get_feature_names_out()
for i, topic in enumerate(nmf.components_):
    top_words = [words[j] for j in topic.argsort()[-10:]]
    print(f"Topic {i}: {', '.join(top_words)}")

plt.figure(figsize=(10,6))

sns.heatmap(pd.crosstab(tfidfdist["NMF_Category"], labeled_data['Category']), annot=True, fmt="d", annot_kws={"fontsize":15}, cmap="Blues", vmin=0)
plt.title("\nTF-IDF + NMF\n", fontsize=25)
plt.xlabel("Real Category")
plt.ylabel("NMF Label (Predicted Category)")

plt.show()


# we need to create a custom NMF function to predict the category labels in order to score by accuracy in gridsearch
class NMF_custom(BaseEstimator, ClassifierMixin):

    def __init__(self, n_components = 5, init = None, solver = 'cd', beta_loss = 'frobenius', l1_ratio = 0.1, max_iter = 200):

        self.n_components = n_components
        self.init = init
        self.solver = solver
        self.beta_loss = beta_loss
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.components_ = None
        self.label_dict = None

    def fit(self, X, y=None):

        self.nmf = NMF(n_components = self.n_components,
                       init = self.init,
                       solver = self.solver,
                       l1_ratio = self.l1_ratio,
                       max_iter = self.max_iter)
        self.nmf.fit(X)
        self.components_ = self.nmf.components_
        self.label_dict = self.get_labels(y, self.nmf.transform(X))

        return self

    # custom predict to attach category label
    def predict(self, X):
        y_pred = []
        W = self.nmf.transform(X)
        y_pred_id = np.argmax(W, axis=1)
        for id_ in y_pred_id:
            y_pred.append(self.label_dict[id_])
        
        return y_pred

    # get category labels for NMF group
    def get_labels(self, y_true, W):
        y_pred = np.argmax(W, axis=1)
        label_dict = {}
        l = np.unique(y_true)
        t = y_true.values
        mx = 0
        p_best = None
        for p in itertools.permutations(range(len(l))):
            c = np.array([l[p.index(x)] for x in y_pred])
            v = np.mean(c == t)
            if v > mx:
                mx = v
                p_best = p
        for i in range(len(l)):
            label_dict[i] = l[p_best.index(i)]
        return label_dict
        


def gridsearch_text_NMF(train_text):
    # number of unique categories
    n_unq_cat = 5

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', sublinear_tf=True, norm='l2')),
        ('nmf', NMF_custom())
    ])

    # parameter grid to use in GridSearchCV
    param_grid = {
        # TfidfVectorizer parameters
        'tfidf__max_df': [0.2, 0.3, 0.4, 0.5, 0.6], # max document frequency for a term.
        'tfidf__min_df': [3, 5],       # min document frequency for a term.
        'tfidf__ngram_range': [(1, 2), (1, 3)], # unigrams and bigrams or unigrams, bigrams, and trigrams.
        'tfidf__max_features': [None, 1000, 5000],
        
        # NMF parameters
        'nmf__n_components': [n_unq_cat],
        'nmf__init': ['nndsvda'],
        'nmf__solver': ['mu'],
        'nmf__beta_loss': ['frobenius', 'kullback-leibler'],
        'nmf__l1_ratio': [0.1]}

    # create GridSearchCV object with custom scoring function
    grid_search = GridSearchCV(
        pipeline,
        param_grid = param_grid,
        scoring='accuracy',
        return_train_score = True,
        error_score='raise',
        cv = 3)

    # fit training data
    grid_search = grid_search.fit(train_text, labeled_data['Category'])

    #print("\nDetailed Cross-Validation Results:")
    #for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
    #  print(f"Mean: {mean_score:.3f} for {params}")

    # best parameters and score
    print("\nBest parameters found: ", grid_search.best_params_)
    print("Best score found: ", grid_search.best_score_)

    # get the best NMF model from the fitted pipeline.
    best_nmf_model = grid_search.best_estimator_.named_steps['nmf']

    # get the TF-IDF feature names (words).
    tfidf_vectorizer = grid_search.best_estimator_.named_steps['tfidf']
    feature_names = tfidf_vectorizer.get_feature_names_out()

    print("\nTopic Dictionary:")
    for topic_idx, topic in enumerate(best_nmf_model.components_):
      top_words_indices = topic.argsort()[:-11:-1]
      top_words = [feature_names[i] for i in top_words_indices]
      print(f"Topic {topic_idx}: {', '.join(top_words)}")  

    return grid_search


#grid_search = gridsearch_text_NMF(labeled_data['Text'])

#Best parameters found:  {'nmf__beta_loss': 'frobenius', 'nmf__init': 'nndsvda', 'nmf__l1_ratio': 0.1, 
#'nmf__n_components': 5, 'nmf__solver': 'mu', 'tfidf__max_df': 0.2, 'tfidf__max_features': None, 
#'tfidf__min_df': 3, 'tfidf__ngram_range': (1, 3)}
#Best score found:  0.9562499999999999

#Topic Dictionary:
#Topic 0: growth, market, economy, company, firm, 2004, economic, shares, bank, sales
#Topic 1: game, win, england, cup, team, play, players, match, season, coach
#Topic 2: labour, election, blair, party, minister, mr blair, prime, prime minister, tories, brown
#Topic 3: film, best, awards, actor, award, films, star, actress, oscar, director
#Topic 4: users, technology, mobile, use, software, digital, phone, net, online, microsoft

# test score #32 -- .95918


#grid_search = gridsearch_text_NMF(labeled_data['text_clean'])
#Best parameters found:  {'nmf__beta_loss': 'frobenius', 'nmf__init': 'nndsvda', 'nmf__l1_ratio': 0.1, 
#'nmf__n_components': 5, 'nmf__solver': 'mu', 'tfidf__max_df': 0.3, 'tfidf__max_features': None, 'tfidf__min_df': 3, 
#'tfidf__ngram_range': (1, 2)}
#Best score found:  0.9222222222222222

#Topic Dictionary:
#Topic 0: growth, year, firm, market, economy, company, shares, sales, bank, oil
#Topic 1: england, game, win, cup, world, play, ireland, match, coach, chelsea
#Topic 2: labour, election, blair, party, brown, government, minister, tory, tony blair, tony
#Topic 3: film, best, awards, actor, award, actress, director, films, star, comedy
#Topic 4: people, mobile, users, phone, internet, net, microsoft, software, technology, online
# test score #33 -- .92925


#grid_search = gridsearch_text_NMF(labeled_data['text_recognized'])
#Best parameters found:  {'nmf__beta_loss': 'frobenius', 'nmf__init': 'nndsvda', 'nmf__l1_ratio': 0.1, 
#'nmf__n_components': 5, 'nmf__solver': 'mu', 'tfidf__max_df': 0.2, 'tfidf__max_features': None, 'tfidf__min_df': 3, 
#'tfidf__ngram_range': (1, 3)}
#Best score found:  0.90625

#Topic Dictionary:
#Topic 0: growth, firm, market, company, shares, economy, sales, bank, deal, oil
#Topic 1: game, win, cup, world, play, match, team, coach, players, season
#Topic 2: labour, election, blair, party, minister, brown, government, tony blair, tony, prime
#Topic 3: film, best, awards, actor, award, actress, director, star, films, comedy
#Topic 4: people, mobile, users, phone, internet, net, technology, software, broadband, online
# test score #34 -- .92244


def gridsearch_supervised(train_text):
    
    # number of unique categories
    n_unq_cat = 5

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', norm='l2', ngram_range=(1, 3), min_df=3, sublinear_tf=True)),
        ('classifier', LogisticRegression()) # placeholder, will be replaced in param_grid
    ])

    lr = LogisticRegression(solver='liblinear', max_iter=10000, class_weight='balanced', random_state=29)
    lsvc = LinearSVC(C=1.0, dual=False, loss='squared_hinge', class_weight=None, max_iter=10000)
    rfc = RandomForestClassifier(n_estimators=300, max_depth=30, min_samples_split=5, min_samples_leaf=1, bootstrap=True, class_weight='balanced', random_state=29, n_jobs=-1)
    mnb = MultinomialNB()
    svm = SVC(kernel = 'rbf', max_iter=10000)

    # Parameter grid to use in GridSearchCV
    param_grid = [
    {
        'tfidf__max_df': [0.2, 0.3, 0.4],
        'classifier': [lr],
        'classifier__C': [1.0, 10],
        'classifier__penalty': ['l1', 'l2'],
    },
    {
        'tfidf__max_df': [0.2, 0.3, 0.4],
        'classifier': [lsvc],
        'classifier__C': [0.1, 1.0],
    },
    {
        'tfidf__max_df': [0.2, 0.3, 0.4],
        'classifier': [rfc],
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20]
    },
    {
        'tfidf__max_df': [0.2, 0.3, 0.4],
        'classifier': [mnb],
        'classifier__alpha': [0.01, 0.1, 1.0],
    },
    {
        'tfidf__max_df': [0.2, 0.3, 0.4],
        'classifier': [svm],
        'classifier__C': [0.1, 1, 10],
        'classifier__gamma': [0.1, 1],
    }]

    # Create GridSearchCV object with custom scoring function
    grid_search = GridSearchCV(
          pipeline,
          param_grid = param_grid,
          scoring ='accuracy',
          return_train_score = True,
          error_score='raise',
          cv = 3)


    # fit training data
    grid_search = grid_search.fit(labeled_data['Text'], labeled_data['Category'])

    print("\nDetailed Cross-Validation Results:")
    #for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
    #  print(f"Mean: {mean_score:.3f} for {params}")

    # 5. Display the best parameters and score
    print("\nBest parameters found: ", grid_search.best_params_)
    print("Best score found: ", grid_search.best_score_)

    return grid_search


#grid_search = gridsearch_supervised(labeled_data['Text'])
#Best parameters found:  {'classifier': LogisticRegression(class_weight='balanced', max_iter=10000, random_state=29,
#solver='liblinear'), 'classifier__C': 1.0, 'classifier__penalty': 'l2', 'tfidf__max_df': 0.2, 
#'tfidf__min_df': 3, 'tfidf__ngram_range': (1, 3)}
#Best score found:  0.9777777777777779
#test #35 - .98231


#grid_search = gridsearch_supervised(labeled_data['text_clean'])
#Best parameters found:  {'classifier': LogisticRegression(class_weight='balanced', max_iter=10000, random_state=29,
#                   solver='liblinear'), 'classifier__C': 1.0, 'classifier__penalty': 'l2', 'tfidf__max_df': 0.2, 'tfidf__min_df': 3, 'tfidf__ngram_range': (1, 3)}
#Best score found:  0.9777777777777779
#test #36 - .97142


#grid_search = gridsearch_supervised(labeled_data['text_recognized'])
#Best parameters found:  {'classifier': LogisticRegression(class_weight='balanced', max_iter=10000, random_state=29,
#                   solver='liblinear'), 'classifier__C': 1.0, 'classifier__penalty': 'l2', 'tfidf__max_df': 0.2}
#Best score found:  0.9777777777777779
#test #37 - .97551




#best_super_pipeline = grid_search.best_estimator_
#test['Category'] = best_super_pipeline.predict(test['text_clean'])

#sub_df = test[['ArticleId', 'Category']]
#print(sub_df)

# save submission
#try:
#    sub_df.to_csv('submission.csv', index=False)
#except PermissionError:
#    print("Could not write the submission file.")


# number of unique categories
n_unq_cat = 5

pipeline_nmf = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_df=0.2, min_df=3, norm='l2',  sublinear_tf=True)),
    ('classifier', NMF_custom(n_components=n_unq_cat, init='nndsvda', solver='mu', beta_loss='frobenius', l1_ratio=0.1))
])

pipeline_lr = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_df=0.2, min_df=3, norm='l2',  sublinear_tf=True)),
    ('classifier', LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=10000, class_weight='balanced', random_state=29))
])


train_sizes = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]

nmf_train_scores = []
nmf_test_scores = []
lr_train_scores = []
lr_test_scores = []


for size_ratio in train_sizes:
    X_train_subset, X_test_subset, y_train_subset, y_test_subset = train_test_split(labeled_data['Text'], labeled_data['Category'], train_size=size_ratio, random_state=29, stratify=labeled_data['Category'])
    pipeline_nmf.fit(X_train_subset, y_train_subset)
    nmf_train_predict = pipeline_nmf.predict(X_train_subset)
    nmf_train_score = accuracy_score(y_train_subset, nmf_train_predict)
    nmf_train_scores.append(nmf_train_score)

    nmf_test_predict = pipeline_nmf.predict(X_test_subset)
    nmf_test_score = accuracy_score(y_test_subset, nmf_test_predict)
    nmf_test_scores.append(nmf_test_score)

    pipeline_lr.fit(X_train_subset, y_train_subset)
    lr_train_predict = pipeline_lr.predict(X_train_subset)
    lr_train_score = accuracy_score(y_train_subset, lr_train_predict)
    lr_train_scores.append(lr_train_score)

    lr_test_predict = pipeline_lr.predict(X_test_subset)
    lr_test_score = accuracy_score(y_test_subset, lr_test_predict)
    lr_test_scores.append(lr_test_score)


results_df = pd.DataFrame({'NMF Train': nmf_train_scores, 'NMF Test': nmf_test_scores, 'LR Train': lr_train_scores, 'LR Test': lr_test_scores})
results_df.index = train_sizes

print(results_df)

