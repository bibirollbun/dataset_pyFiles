import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
from itertools import permutations


from wordcloud import WordCloud


import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token, Span
#spacy.require_gpu()


from tqdm import tqdm
tqdm.pandas(desc = 'Processing Data Frame')


from sklearn.base import TransformerMixin

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import NMF
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#build paths to the data
data_dir = '/kaggle/input/learn-ai-bbc'
train_file = 'BBC News Train.csv'
test_file = 'BBC News Test.csv'

train_file_path = os.path.join(data_dir, train_file)#join train path
test_file_path = os.path.join(data_dir, test_file)#join test path

train = pd.read_csv(train_file_path) #import training data
test = pd.read_csv(test_file_path) #import test data


train.head() 


train['Text'][20]


test.head()


test['Text'][11]


print(f'The training dataset has {train.shape[0]} rows and {train.shape[1]} columns')
print(f'The test dataset has {test.shape[0]} rows and {test.shape[1]} columns')


train.info()


test.info()


train['Category'].unique()


train = train.set_index('ArticleId')


train.Category.value_counts()


gini_max = 1 - 1/5
gini = 1 - np.power(train.Category.value_counts()/train.count().max(), 2).sum()
gini_str = f"The training data has a Gini Index of {gini:.3f}. This is very close to the maximum value for 5 classes of {gini_max} so it can be concluded that the classes are well balanced."
print(gini_str)


#set seaborn palette to a 5 color wheel
sns.set_palette('husl', 5) 
#plot the category value counts as a pie chart
train.Category.value_counts().plot.pie(autopct = '%.1f%%', explode = [.05]*5)
#remove the y label because it looks goofy
plt.ylabel('')
#add a title
plt.title('Proportions of Category Labels')


#sum articles with duplicate `Text` values
train.duplicated(subset = 'Text').sum()


#print out the first 4 pairs of duplicate articles
train[train.duplicated(subset = 'Text', keep = False)].sort_values(by = 'Text').head(8)


#save the indices of the first instance of each duplicated article
duplicate_indices = train[train.duplicated(subset = 'Text')].index
#drop the first appearance of each duplicate from the training data
train = train.drop(duplicate_indices)


train.Category.value_counts().plot.pie(autopct = '%.1f%%', explode = [.05] * 5)
#i don't know why a pie chart even needs a y label, it doesn't even have a y axis
plt.ylabel('')
plt.title('Proportions of category labels after deduplication')


#create a column with the word count of each article
train['word_count'] = train.Text.apply(lambda T: len(T.split(' ')))


#create 2 x 1 figure with a shared x axis
fig, ax = plt.subplots(2,1, figsize = (10,5), sharex = True)
#log transform the x axis
plt.semilogx()
#add a title
plt.title('Kernel Density Estimation and Boxplot of Word Count by Category')
#plot kernel density estimation for each category in the top plot
sns.kdeplot(train, x = 'word_count', hue = 'Category', fill = 'Category', ax = ax[0])
#plot boxplots for each category in the bottom plot
sns.boxplot(train, x = 'word_count', y = 'Category', ax = ax[1])


from statsmodels.formula.api import ols
import statsmodels.api as sm
mod = ols('word_count ~ Category', data = train)
mod.fit().summary()


#helper function to plot word clouds for each category
def plot_word_clouds(df, col):
    '''
    df: dataframe object with articles
    col: string identifying the article column
    returns None
    '''

    #group df by Category and concatenate all of the articles in each category
    text_by_cat = df.groupby('Category')[col].apply(' '.join)
    #create a 3 x 2 figure with shared x and y axes
    fig, axes = plt.subplots(3, 2, figsize = (16, 8), sharex = True, sharey = True, facecolor = '#B1D6EB')
    #flatten the axes list
    axes = [ax for ax_list in axes for ax in ax_list]
    #create dictionary to hold word clouds
    clouds = {}
    #iterate over categories with text
    for i, (cat, text) in enumerate(text_by_cat.items()):
        #create WordCloud object
        word_cloud = WordCloud(width = 400, height = 125, 
                               background_color = '#55AFE2',
                               colormap = 'copper',
                               collocations = False
                               )
        #plot word cloud on the ith axis                    
        axes[i].imshow(word_cloud.generate(text))
        #set title to Category
        axes[i].set_title(f'{cat.capitalize()} Word Cloud')
        #do not show x and y axis
        axes[i].axis('off')
    #turn off the unused axis
    axes[5].axis('off')


plot_word_clouds(train, 'Text')


#load the spacy small english pipeline
nlp_sm = spacy.load('en_core_web_sm')
#tell spacy to merge entities tagged by the 'ner' pipeline element into one token
nlp_sm.add_pipe('merge_entities')


#convert the articles into spacy Docs
train['Docs_sm'] = train['Text'].progress_apply(nlp_sm)


#convert the test articles into spacy Docs
test['Docs_sm'] = test['Text'].progress_apply(nlp_sm)


#nlp_lg = spacy.load('/kaggle/working/spacy/en_core_web_lg.nlp', disable = ['parser', 'attribute-ruler'])
#nlp_lg.add_pipe('merge_entities')
#train['Docs_lg'] = train['Text'].progress_apply(nlp_lg)


#A preprocessor class for cleaning the docs
class Preprocessor(TransformerMixin):
    def __init__(self, tags = None):
        '''
        tags: ['tag1', 'tag2', ... , 'tagn']
        returns: None
        '''
        self.tags = tags
        pass

    #to be called by  GridSearchCV
    def set_params(self, **params):
        '''
        **params: {attribute_name: attribute_value}
        returns: None
        '''
        for key, value in params.items():
            setattr(self, key, value)

    
    def filter_token(self, token):
        '''
        token: a spaCy token object
        returns: a boolean
        '''
        #remove stopwords, punctuation and spaces
        if token.is_stop or token.is_punct or token.is_space:
            return False
        else:
            return True

    #
    def clean(self, doc):
        '''
        doc: a spaCy Doc object
        returns: a list of spacy tokens
        '''
        token_list = [token for token in doc if self.filter_token(token)]
        return token_list

    
    def annotate(self, tokens):
        '''
        tokens: a list of spaCy tokens
        returns: a string
        example output: 'GPE United_States_of_America'
        '''
        #if token was tagged as an entity replace whitespace with '_', else use the lemma
        text = ['_'.join(token.text.split()) if token.ent_type_ else token.lemma_ for token in tokens]
        
        #if self.tags isn't empty prepend each tokens tag to its text
        if self.tags:
            tags = {tag: [getattr(token, tag) for token in doc] for tag in self.tags}
            tags = [' '.join(tags) for tags in zip(*tags.values())]
            annotated = [' '.join((tag, text)) for tag, text in zip(tags, text)]
            return ' '.join(annotated)
        else:
            return ' '.join(text)

    #nothing to fit
    def fit(self, df, y = None):
        return self

    #clean and then annotate each doc
    def transform(self, docs, y = None):
        cleaned = docs.apply(self.clean)
        return cleaned.apply(self.annotate)
        


#initialize a preprocessor
P = Preprocessor()
#clean the docs for analysis
train['Clean'] = P.fit_transform(train['Docs_sm'])


plot_word_clouds(train, 'Clean')


#initialize a vectorizer
T = TfidfVectorizer()

#vectorize the cleaned articles
vectors = pd.DataFrame(T.fit_transform(train['Clean']).toarray(),
                       index = train.index, 
                       columns = T.get_feature_names_out())

#sum frequencies by category
vectors = pd.concat([train[['Category']], vectors], axis = 1).groupby('Category').sum().T


#extract the top words from each category
for cat in vectors.columns:
    cat_str = f'The top {cat} words after TfIdf vectorization are:'
    print(cat_str)
    print('\t'.join([word for word in vectors[vectors[cat] > 5].sort_values(cat, ascending = False).index]))
    print('\n')


#plot top word frequencies
vectors2 = vectors.reset_index(names = 'word')
fig, axes = plt.subplots(5, 1, sharex = True, figsize = (10,20))
for cat, ax in zip(train['Category'].unique(), axes):
    df = vectors2[vectors2[cat] > 5].sort_values(cat, ascending = False)
    sns.barplot(df, x = cat, y = 'word', ax = ax)


#split the training articles into training and validation sets
X_train, X_test, y_train, y_test = tts(train['Docs_sm'], train['Category'], test_size = .2, random_state = 42)


#a Class for predicting labels from the NMF W matrix
class Predictor(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, W, y):
        '''
        W: m_articles x n_categories numpy array from svc
        y: m_articles x 1 numpy array of labels
        returns: self
        '''
        self.W = W
        self.yt = y

        #the predicted class is the column with the largest value for each row
        self.preds = self.W.argmax(axis = 1)
        self.labels = y.unique()

        #generate permutations of class labels
        self.perms = permutations(self.labels)

        #initialize best permutation
        self.best = (0, self.labels)

        #score each permutation and update best
        for perm in self.perms:
            yp = [perm[pred] for pred in self.preds]
            score = accuracy_score(self.yt, yp)
            if score > self.best[0]:
                self.best = (score, perm)

        #save best permutation
        self.label_mapping = self.best[1]
        
        return self
        
    def predict(self, W):
        yp = W.argmax(axis = 1)
        yp = [self.label_mapping[y] for y in yp]
        return yp
        
    def score(self, W, yt):
        yp = self.predict(W)
        return accuracy_score(yt, yp)

    def fit_predict(self, W, y):
        self.fit(W, y)
        return self.predict(W)


#dict for storing pipelines
pipelines = {}


pipelines['nmf'] = Pipeline([('Preprocessing', Preprocessor()),
                            ('Vectorizing', TfidfVectorizer()),
                            ('Decomposing', NMF(n_components = 5)),
                            ('Predicting', Predictor())])

#fit nmf pipeline to the training set
pipelines['nmf'].fit(X_train, y_train)

#score the nmf pipeline on the validation set
print(f'Before parameter tuning, the NMF pipeline achieved a training accuracy of {pipelines["nmf"].score(X_train, y_train):.3f} and a test accuracy of {pipelines["nmf"].score(X_test, y_test):.3f}')


fig, ax = plt.subplots(1,2, sharey= True, figsize = (10,5))
ConfusionMatrixDisplay.from_predictions(pipelines['nmf'].predict(X_train), y_train, ax = ax[0], normalize = 'true')
ConfusionMatrixDisplay.from_predictions(pipelines['nmf'].predict(X_test), y_test, ax = ax[1], normalize  = 'true')
ax[0].set_title('Confusion Matrix for \n NMF Training Predictions')
ax[1].set_title('Confusion Matrix for \n NMF Validation Predictions')

ax[0].tick_params(rotation = 45)
ax[1].tick_params(rotation = 45)


pipelines['svc'] = Pipeline([('Preprocessing', Preprocessor()), 
                   ('Vectorizing', TfidfVectorizer()),
                   ('Classifying', SVC())])

#fit the svc pipelinee to the training set
pipelines['svc'].fit(X_train, y_train)

#score the svc pipeline on the validation set
pipelines['svc'].score(X_test, y_test)
print(f'Before parameter tuning, the NMF pipeline achieved a training accuracy of {pipelines["svc"].score(X_train, y_train):.3f} and a test accuracy of {pipelines["svc"].score(X_test, y_test):.3f}')


fig, ax = plt.subplots(1,2, sharey= True, figsize = (10,5))
ConfusionMatrixDisplay.from_predictions(pipelines['svc'].predict(X_train), y_train, ax = ax[0], normalize = 'true')
ConfusionMatrixDisplay.from_predictions(pipelines['svc'].predict(X_test), y_test, ax = ax[1], normalize  = 'true')
ax[0].tick_params(rotation = 45)
ax[1].tick_params(rotation = 45)

ax[0].set_title('Confusion Matrix for \n SVC Training Predictions')
ax[1].set_title('Confusion Matrix for \n SVC Validation Predictions')


#dict for storing parameter grids
param_grids = {}

param_grids['nmf'] = [{}]


param_grids['nmf'][0]['Preprocessing__tags'] =  [None, ['entity_type_']]
                                                
param_grids['nmf'][0]['Vectorizing__ngram_range'] = [(1,1), (1,2)]
param_grids['nmf'][0]['Vectorizing__max_features'] = [10000,20000]
param_grids['nmf'][0]['Vectorizing__max_df'] = [1, .5]
param_grids['nmf'][0]['Vectorizing__min_df'] = [2]
param_grids['nmf'][0]['Vectorizing__norm'] = ['l1', 'l2']
param_grids['nmf'][0]['Vectorizing__sublinear_tf'] = [True, False]

param_grids['nmf'][0]['Decomposing__init'] = ['nndsvda']
param_grids['nmf'][0]['Decomposing__solver'] = ['mu']
param_grids['nmf'][0]['Decomposing__beta_loss'] = ['frobenius', 'kullback-leibler']
param_grids['nmf'][0]['Decomposing__alpha_W'] = [0, .1]
param_grids['nmf'][0]['Decomposing__alpha_H'] = ['same']
param_grids['nmf'][0]['Decomposing__l1_ratio'] = [0,.5,1]
param_grids['nmf'][0]['Decomposing__random_state'] = [42]


gscvs = {}
gscvs['nmf'] = [None]
two_fold = KFold(2, shuffle = True, random_state = 42)
gscvs['nmf'][0] = GridSearchCV(pipelines['nmf'], param_grids['nmf'][0], cv = two_fold, verbose = 1)
gscvs['nmf'][0].fit(X_train, y_train)


cv_results = {}
cv_results['nmf'] = [None]
cv_results['nmf'][0] = pd.DataFrame(gscvs['nmf'][0].cv_results_)
cv_results['nmf'][0].sort_values('rank_test_score')[['rank_test_score',
                                                    'mean_test_score',
                                                    'std_test_score']].head(8)


sns.lineplot(cv_results['nmf'][0], x = 'rank_test_score', y = 'mean_test_score')
plt.title('K-Fold Cross Validation Score (K = 2) vs Cross Validation Rank for NMF Pipeline Parameters')
plt.ylabel('Mean Cross Validation Accuracy (n = 2)')
plt.xlabel('Cross Validation Rank')


best_estimators = {}
best_estimators['nmf'] = [None]
best_estimators['nmf'][0] = gscvs['nmf'][0].best_estimator_


gscvs['nmf'][0].best_params_


best_estimators['nmf'][0].score(X_test, y_test)
print(f'After 1 gridsearch, the NMF pipeline achieved a training accuracy of {best_estimators["nmf"][0].score(X_train, y_train):.3f} and a test accuracy of {best_estimators["nmf"][0].score(X_test, y_test):.3f}')


fig, ax = plt.subplots(1,2, sharey= True, figsize = (10,5))
ConfusionMatrixDisplay.from_predictions(best_estimators['nmf'][0].predict(X_train), y_train, ax = ax[0], normalize = 'true')
ConfusionMatrixDisplay.from_predictions(best_estimators['nmf'][0].predict(X_test), y_test, ax = ax[1], normalize  = 'true')
ax[0].tick_params(rotation = 45)
ax[1].tick_params(rotation = 45)

ax[0].set_title('Confusion Matrix for \n NMF Training Predictions after 1 iteration')
ax[1].set_title('Confusion Matrix for \n NMF Validation Predictions after 1 iteration')


param_grids['nmf'].append(None)
param_grids['nmf'][1] = {key: [val] for key, val in gscvs['nmf'][0].best_params_.items()}


param_grids['nmf'][1]['Vectorizing__ngram_range'] = [(1,2),(1,3)]
param_grids['nmf'][1]['Vectorizing__max_features'] = [5000, 10000]
param_grids['nmf'][1]['Vectorizing__max_df'] = [.5, .25]

param_grids['nmf'][1]['Decomposing__alpha_W'] = [0, .1]
param_grids['nmf'][1]['Decomposing__alpha_H'] = [0, .1]

param_grids['nmf'][1]


gscvs['nmf'].append(None)
gscvs['nmf'][1] = GridSearchCV(pipelines['nmf'], param_grids['nmf'][1], cv = two_fold, verbose = 1)


gscvs['nmf'][1].fit(X_train, y_train)


cv_results['nmf'].append(None)
cv_results['nmf'][1] = pd.DataFrame(gscvs['nmf'][1].cv_results_)
cv_results['nmf'][1].sort_values('rank_test_score')[['rank_test_score',
                                                    'mean_test_score',
                                                    'std_test_score']].head(8)


sns.lineplot(cv_results['nmf'][1], x = 'rank_test_score', y = 'mean_test_score')
plt.title('K-Fold Cross Validation Score (K = 2) vs Cross Validation Rank for NMF Pipeline Parameters')
plt.ylabel('Mean Cross Validation Accuracy (n = 2)')
plt.xlabel('Cross Validation Rank')


best_estimators['nmf'].append(None)
best_estimators['nmf'][1] = gscvs['nmf'][1].best_estimator_


gscvs['nmf'][1].best_params_


print(f'After 2 gridsearches, the NMF pipeline achieved a training accuracy of {best_estimators["nmf"][1].score(X_train, y_train):.3f} and a test accuracy of {best_estimators["nmf"][1].score(X_test, y_test):.3f}')


fig, ax = plt.subplots(1,2, sharey= True, figsize = (10,5))
ConfusionMatrixDisplay.from_predictions(best_estimators['nmf'][1].predict(X_train), y_train, ax = ax[0], normalize = 'true')
ConfusionMatrixDisplay.from_predictions(best_estimators['nmf'][1].predict(X_test), y_test, ax = ax[1], normalize  = 'true')
ax[0].tick_params(rotation = 45)
ax[1].tick_params(rotation = 45)
ax[0].set_title('Confusion Matrix for \n NMF Training Predictions after 2 iterations')
ax[1].set_title('Confusion Matrix for \n NMF Validation Predictions after 2 iterations')


param_grids['nmf'].append(None)
param_grids['nmf'][2] = {key: [val] for key, val in gscvs['nmf'][1].best_params_.items()}


param_grids['nmf'][2]['Vectorizing__max_df'] = [.25, .125]
param_grids['nmf'][2]['Vectorizing__max_features'] = [10000, 20000]
param_grids['nmf'][2]['Vectorizing__ngram_range'] = [(1,3), (1,4)]

param_grids['nmf'][2]['Decomposing__alpha_H'] = [.1,.2]
param_grids['nmf'][2]['Decomposing__alpha_W'] = [.1,.2]
param_grids['nmf'][2]


gscvs['nmf'].append(None)
gscvs['nmf'][2] = GridSearchCV(pipelines['nmf'], param_grids['nmf'][2], cv = two_fold, verbose = 1)
gscvs['nmf'][2].fit(X_train, y_train)


cv_results['nmf'].append(None)
cv_results['nmf'][2] = pd.DataFrame(gscvs['nmf'][2].cv_results_)
cv_results['nmf'][2].sort_values('rank_test_score')[['rank_test_score', 
                                                     'mean_test_score',
                                                    'std_test_score']].head(8)


sns.lineplot(cv_results['nmf'][2], x = 'rank_test_score', y = 'mean_test_score')
plt.title('K-Fold Cross Validation Score (K = 2) vs Cross Validation Rank for NMF Pipeline Parameters')
plt.ylabel('Mean Cross Validation Accuracy (n = 2)')
plt.xlabel('Cross Validation Rank')


best_estimators['nmf'].append(None)
best_estimators['nmf'][2] = gscvs['nmf'][2].best_estimator_
gscvs['nmf'][2].best_params_


print(f'After 3 gridsearches, the NMF pipeline achieved a training accuracy of {best_estimators["nmf"][2].score(X_train, y_train):.3f} and a test accuracy of {best_estimators["nmf"][2].score(X_test, y_test):.3f}')


fig, ax = plt.subplots(1,2, sharey= True, figsize = (10,5))
ConfusionMatrixDisplay.from_predictions(best_estimators['nmf'][2].predict(X_train), y_train, ax = ax[0], normalize = 'true')
ConfusionMatrixDisplay.from_predictions(best_estimators['nmf'][2].predict(X_test), y_test, ax = ax[1], normalize  = 'true')
ax[0].tick_params(rotation = 45)
ax[1].tick_params(rotation = 45)
ax[0].set_title('Confusion Matrix for \n NMF Training Predictions after 3 iterations')
ax[1].set_title('Confusion Matrix for \n NMF Validation Predictions after 3 iterations')


param_grids['nmf'].append(None)
param_grids['nmf'][3] = {key: [val] for key, val in gscvs['nmf'][2].best_params_.items()}

param_grids['nmf'][3]['Vectorizing__max_features'] = [20000, 40000]
param_grids['nmf'][3]['Vectorizing__ngram_range'] = [(1,3), (1,4)]

param_grids['nmf'][3]['Decomposing__alpha_W'] = [.2, .4]
param_grids['nmf'][3]


gscvs['nmf'].append(None)
gscvs['nmf'][3] = GridSearchCV(pipelines['nmf'], param_grids['nmf'][3], cv = two_fold, verbose = 1)
gscvs['nmf'][3].fit(X_train, y_train)


cv_results['nmf'].append(None)
cv_results['nmf'][3] = pd.DataFrame(gscvs['nmf'][3].cv_results_)
cv_results['nmf'][3].sort_values('rank_test_score')[['rank_test_score', 'mean_test_score', 'std_test_score','params']]


sns.lineplot(cv_results['nmf'][3], x = 'rank_test_score', y = 'mean_test_score')
plt.title('K-Fold Cross Validation Score (K = 2) vs Cross Validation Rank for NMF Pipeline Parameters')
plt.ylabel('Mean Cross Validation Accuracy (n = 2)')
plt.xlabel('Cross Validation Rank')


best_estimators['nmf'].append(None)
best_estimators['nmf'][3] = gscvs['nmf'][3].best_estimator_
gscvs['nmf'][3].best_params_


print(f'After 4 gridsearches, the NMF pipeline achieved a training accuracy of {best_estimators["nmf"][3].score(X_train, y_train):.3f} and a test accuracy of {best_estimators["nmf"][3].score(X_test, y_test):.3f}')


fig, ax = plt.subplots(1,2, sharey= True, figsize = (10,5))
ConfusionMatrixDisplay.from_predictions(best_estimators['nmf'][3].predict(X_train), y_train, ax = ax[0], normalize = 'true')
ConfusionMatrixDisplay.from_predictions(best_estimators['nmf'][3].predict(X_test), y_test, ax = ax[1], normalize  = 'true')
ax[0].tick_params(rotation = 45)
ax[1].tick_params(rotation = 45)
ax[0].set_title('Confusion Matrix for \n NMF Training Predictions after 4 iterations')
ax[1].set_title('Confusion Matrix for \n NMF Validation Predictions after 4 iterations')


test['Category'] = best_estimators['nmf'][3].predict(test['Docs_sm'])
test[['ArticleId','Category']].set_index('ArticleId').to_csv('nmf_submission.csv')


param_grids['svc'] = [{}]


param_grids['svc'][0]['Preprocessing__tags'] =  [None, ['entity_type_']]
                                                
param_grids['svc'][0]['Vectorizing__ngram_range'] = [(1,1), (1,2)]
param_grids['svc'][0]['Vectorizing__max_features'] = [10000,20000]
param_grids['svc'][0]['Vectorizing__max_df'] = [1, .5]
param_grids['svc'][0]['Vectorizing__min_df'] = [2]
param_grids['svc'][0]['Vectorizing__norm'] = ['l1', 'l2']
param_grids['svc'][0]['Vectorizing__sublinear_tf'] = [True, False]

param_grids['svc'][0]['Classifying__C'] = [1]
param_grids['svc'][0]['Classifying__kernel'] = ['linear', 'rbf']
param_grids['svc'][0]['Classifying__gamma'] = ['scale', 'auto']
param_grids['svc'][0]['Classifying__class_weight'] = [None, 'balanced']
param_grids['svc'][0]


gscvs['svc'] = [None]
gscvs['svc'][0] = GridSearchCV(pipelines['svc'], param_grids['svc'][0], cv = two_fold, verbose = 1, n_jobs = -1)
gscvs['svc'][0].fit(X_train, y_train)


cv_results['svc'] = [None]
cv_results['svc'][0] = pd.DataFrame(gscvs['svc'][0].cv_results_)
cv_results['svc'][0].sort_values('rank_test_score').dropna(subset = 'mean_test_score').tail()


gscvs['svc'][0].best_params_


sns.lineplot(cv_results['svc'][0], x = 'rank_test_score', y = 'mean_test_score')
plt.title('K-Fold Cross Validation Score (K = 2) vs Cross Validation Rank for SVC Pipeline Parameters')
plt.ylabel('Mean Cross Validation Accuracy (n = 2)')
plt.xlabel('Cross Validation Rank')


best_estimators['svc'] = [None]
best_estimators['svc'][0] = gscvs['svc'][0].best_estimator_
gscvs['svc'][0].best_params_


print(f'After 1 gridsearch, the SVC pipeline achieved a training accuracy of {best_estimators["svc"][0].score(X_train, y_train):.3f} and a test accuracy of {best_estimators["svc"][0].score(X_test, y_test):.3f}')


fig, ax = plt.subplots(1,2, sharey= True, figsize = (10,5))
ConfusionMatrixDisplay.from_predictions(best_estimators['svc'][0].predict(X_train), y_train, ax = ax[0], normalize = 'true')
ConfusionMatrixDisplay.from_predictions(best_estimators['svc'][0].predict(X_test), y_test, ax = ax[1], normalize  = 'true')
ax[0].tick_params(rotation = 45)
ax[1].tick_params(rotation = 45)
ax[0].set_title('Confusion Matrix for \n SVC Training Predictions after 1 iterations')
ax[1].set_title('Confusion Matrix for \n SVC Validation Predictions after 1 iterations')


param_grids['svc'].append(None)
param_grids['svc'][1] = {key: [val] for key, val in gscvs['svc'][0].best_params_.items()}
param_grids['svc'][1]['Vectorizing__max_df'] = [.5, .75]
param_grids['svc'][1]['Classifying__C'] = [1]
param_grids['svc'][1]['Classifying__kernel'] = ['poly']
param_grids['svc'][1]['Classifying__degree'] = [1,2,3]
param_grids['svc'][1]


gscvs['svc'].append(None)
gscvs['svc'][1] = GridSearchCV(pipelines['svc'], param_grids['svc'][1], cv = 2, verbose = 1)
gscvs['svc'][1].fit(X_train, y_train)


if len(cv_results['svc']) < 2: cv_results['svc'].append(None) 
cv_results['svc'][1] = pd.DataFrame(gscvs['svc'][1].cv_results_)
cv_results['svc'][1].sort_values('rank_test_score').head(10)



gscvs['svc'][1].best_params_


fig, ax = plt.subplots(1,2, sharey = True)

sns.lineplot(cv_results['svc'][1], x = 'rank_test_score', y = 'mean_test_score', ax = ax[0])
ax[0].set_title('Mean 2-Fold Cross Validation Score \n vs Cross Validation Rank', fontsize = 10)
ax[0].set_ylabel('Mean Cross Validation Accuracy (n = 2)')
ax[0].set_xlabel('2-Fold Cross Validation Rank')

sns.boxplot(cv_results['svc'][1], x = 'param_Classifying__degree', y = 'mean_test_score', ax = ax[1])
ax[1].set_title('Mean 2-Fold Cross Validation Score \n vs SVC Polynomial Kernel Degree', fontsize = 10)
ax[1].set_xlabel('SVC Polynomial Kernel Degree')
fig.get_label()


if len(best_estimators['svc']) < 2: best_estimators['svc'].append(None)
best_estimators['svc'][1] = gscvs['svc'][1].best_estimator_
best_estimators['svc'][1].score(X_test, y_test)


print(f'After 2 gridsearches, the SVC pipeline achieved a training accuracy of {best_estimators["svc"][1].score(X_train, y_train):.3f} and a test accuracy of {best_estimators["svc"][1].score(X_test, y_test):.3f}')


fig, ax = plt.subplots(1,2, sharey= True, figsize = (10,5))
ConfusionMatrixDisplay.from_predictions(best_estimators['svc'][1].predict(X_train), y_train, ax = ax[0], normalize = 'true')
ConfusionMatrixDisplay.from_predictions(best_estimators['svc'][1].predict(X_test), y_test, ax = ax[1], normalize  = 'true')
ax[0].tick_params(rotation = 45)
ax[1].tick_params(rotation = 45)


test['Category'] = best_estimators['svc'][1].predict(test['Docs_sm'])
test[['ArticleId','Category']].set_index('ArticleId').to_csv('svc_submission.csv')


lcs = {}
lcs['nmf'] = learning_curve(best_estimators['nmf'][3],X_train, y_train, train_sizes = np.linspace(.1,1,20), cv = two_fold, verbose = 1, n_jobs = -1)
lcs['svc'] = learning_curve(best_estimators['svc'][1],X_train, y_train, train_sizes = np.linspace(.1,1,20), cv = two_fold, verbose = 1, n_jobs = -1)


lc_nmf_sizes, lc_nmf_train, lc_nmf_val = lcs['nmf']
lc_svc_sizes, lc_svc_train, lc_svc_val = lcs['svc']

nmf_train_columns = ['nmf_Train_Score_' + str(i+1) for i in range(2)]
nmf_val_columns = ['nmf_Validation_Score_' + str(i + 1) for i in range(2)]
svc_train_columns = ['svc_Train_Score_' + str(i+1) for i in range(2)]
svc_val_columns = ['svc_Validation_Score_' + str(i + 1) for i in range(2)]

lc_nmf_train = pd.DataFrame(lc_nmf_train, index = lc_nmf_sizes, columns = nmf_train_columns)
lc_nmf_val = pd.DataFrame(lc_nmf_val, index = lc_nmf_sizes, columns = nmf_val_columns)
lc_svc_train = pd.DataFrame(lc_svc_train, index = lc_svc_sizes, columns = svc_train_columns)
lc_svc_val = pd.DataFrame(lc_svc_val, index = lc_svc_sizes, columns = svc_val_columns)

lc_nmf_train = lc_nmf_train.melt(var_name = 'nmf_Train', value_name = 'nmf_Score', ignore_index = False)
lc_nmf_val = lc_nmf_val.melt(var_name = 'nmf_Train', value_name = 'nmf_Score', ignore_index = False)
lc_svc_train = lc_svc_train.melt(var_name = 'svc_Train', value_name = 'svc_Score', ignore_index = False)
lc_svc_val = lc_svc_val.melt(var_name = 'svc_Train', value_name = 'svc_Score', ignore_index = False)

lc_nmf_train = lc_nmf_train.reset_index(names = 'Train_Size')
lc_nmf_val = lc_nmf_val.reset_index(names = 'Train_Size')
lc_svc_train = lc_svc_train.reset_index(names = 'Train_Size')
lc_svc_val = lc_svc_val.reset_index(names = 'Train_Size')

lc_nmf_train['nmf_Train'] = lc_nmf_train['nmf_Train'].map(lambda x: 'Training')
lc_nmf_val['nmf_Train'] = lc_nmf_val['nmf_Train'].map(lambda x: 'Validation')
lc_svc_train['svc_Train'] = lc_svc_train['svc_Train'].map(lambda x: 'Training')
lc_svc_val['svc_Train'] = lc_svc_val['svc_Train'].map(lambda x: 'Validation')

lc_nmf_df = pd.concat([lc_nmf_train, lc_nmf_val]).groupby(['nmf_Train', 'Train_Size']).mean('Score')
lc_svc_df = pd.concat([lc_svc_train, lc_svc_val]).groupby(['svc_Train', 'Train_Size']).mean('Score')
lc_df = pd.concat([lc_nmf_df, lc_svc_df], axis = 1)
lc_df = lc_df.reset_index().rename(columns = {'level_0': 'Training'})
lc_df = lc_df.melt(id_vars = ['Train_Size', 'Training'], 
                   value_vars = ['nmf_Score', 'svc_Score'], 
                   var_name = 'Model', 
                   value_name = 'Score')
lc_df['Model'] = lc_df['Model'].apply(lambda model: model[0:3])


lc_df.sample(8)


fig, ax = plt.subplots(1,1, sharey = True, figsize = (8,8))

sns.lineplot(lc_df, x = 'Train_Size', y = 'Score', hue = 'Model', style = 'Training', ax = ax)
ax.set_title('K-Fold Cross Validation Score (K = 2) vs Training Sample Size\n for SVC and NMF Models')
ax.set_xlabel('Training Sample Size')
ax.set_ylabel('Mean Cross Validation Accuracy (n = 2)')


