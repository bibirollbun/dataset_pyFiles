#Packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import Lasso
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

import xgboost as xgb

import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

import os
import re
from copy import deepcopy
import string
from collections import Counter


# nltk wordnet.zip do not unzip itself, so use this chunk of codes to fix the problem
os.system("python3 -m nltk.downloader wordnet")
os.system("unzip /usr/share/nltk_data/corpora/wordnet.zip -d /usr/share/nltk_data/corpora/")


# Stopwords counted from frequency analysis
more_stopwords = '''quarter, year, think, go, thank, see, continu, oper, million, question, busi, u, market, expect, look, like, next, would, 
                  realli, come, weve, that, custom, product, time, one, also, call, get, revenu, new, last, kind, point, third, margin, q, 
                  result, first, line, talk, want, today, mayb, basi, make, say, plea, end, work, could, compani, your, earn, there, around, 
                  number, across, way, financi, team, second, got, fourth, im, morn, mention, month, said, compar, actual, incom, per, may'''

'''
Function that use PowerTransformer() to minimize skewness, InterativeImputer() to impute missing values,
and Normalizer() to normalize the data, minimizing outliers effect
'''
def nan_impute_with_normalization(df_numerical_only, initial_strategy = 'median', imputation_order = 'roman'):

    pipe = make_pipeline(PowerTransformer(), 
                         IterativeImputer(initial_strategy = initial_strategy, # since too many outliers, mean will distort the parameter estimates
                                          imputation_order = imputation_order, # consideration for longitudinal variable
                                          skip_complete = True,
                                          verbose = 0),
                         Normalizer()).set_output(transform = 'pandas')

    return pipe.fit_transform(df_numerical_only)

'''
Function that cleans texts (according to fixed pattern existed because of web scrapping), stem and lemmatize words with 
nltk package, and transform texts into Bag of Words (BOW), e.g. a list with individual words as elements
'''
def clean_stem_lemmatize(text):
    
    translator = str.maketrans('', '', string.punctuation + string.digits)
    stopwords = nltk.corpus.stopwords.words('english')
    stemmer = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()


    # strip out transcript contexts
    match = re.search("(Quarter|quarter).*Read more current", text)

    if not match:
        return ['']

    s = text[match.start():match.end()]

    # remove digits and punctuation, lowercasing
    s = s.translate(translator).lower()

    # stemming, lemmatizing and tokenization
    tokens_ = []
    for word in nltk.word_tokenize(s):
        if word not in stopwords:
            word = lemmatizer.lemmatize(stemmer.stem(word))
            if word not in more_stopwords.split(', '):
                tokens_.append(word)

    token_count = Counter(tokens_)

    # keep only words with more than 1 occurrence 
    return [token for token in tokens_ if token_count[token] > 1]

'''
Function that vectorize bag of words, transform texts into numbers using HashingVectorizer()
'''
def NLP_transformation(list_of_texts, ngram_range = (1, 2), norm = 'l1', n_features = 500):
    vectorizer = HashingVectorizer(tokenizer = clean_stem_lemmatize, 
                                   preprocessor = None, 
                                   ngram_range = ngram_range, 
                                   norm = norm, 
                                   lowercase = False, # important: not sure what happened but has to set as False to run correctly
                                   n_features = n_features)
    
    return vectorizer.fit_transform(list_of_texts).todense()

'''
Function that concatenate transformed texts (that is, CSR matrix) with original dataset
'''
def concat_csr(csr, df):
    
    vec_df = pd.DataFrame(csr)
    # a copy of df, for later resetting index and column names
    df_reset = df.reset_index(drop = True)

    # concat df and matrix
    df_concat = pd.concat([df_reset, vec_df], axis = 1, ignore_index = True)

    # originate index
    df_concat.index = df.index

    #originate column names
    df_concat.columns = list(df.columns) + [f'col_{i}' for i in range(500)]
    
    return df_concat

'''
Function that ensembles all the functions above to transform raw data into modelable format
'''
def data_cleaning_preprocessing(df, initial_strategy = 'median', imputation_order = 'roman',ngram_range = (1, 2), norm = 'l1', n_features = 500):
    
    X = deepcopy(df)
    
    # see whether training set or testing set, simply for reuseability
    if 'FinancialSector' in X.columns.tolist():
        X.drop(['FinancialSector', 'FinancialRisk'], axis = 1, inplace = True)
        
    # drop all unnecessary columns, including those found in EDA analysis
    X.drop(['url', 'exchangeCountry', 'securityType', 'CIK', 'name', 'securityID', 'exchangeName', 'fiscalDint', 'exchangeID', 'businessDescription'], axis = 1, inplace = True)
    
    # rest of dataset needs 3 parts of work: categoricals encoding, nan imputation and NLP transformation
    
    # 1. categoricals encoding
    X.loc[X['incorporationCountry'] != 'USA', 'incorporationCountry'] = 'other'
    cate = X['incorporationCountry'].replace(['USA', 'other'], [True, False])
    
    # 2. nan imputation
    num = nan_impute_with_normalization(X.select_dtypes([np.number]), initial_strategy, imputation_order)
    
    # 3. NLP transformation
    csr = NLP_transformation(X['call_transcript'].tolist(), ngram_range, norm, n_features)
    
    # concatenate results above together
    X = concat_csr(csr, pd.concat([cate, num], axis = 1))
    
    
    return X

'''
Function that when using multimetric, display best scores and corresponding parameters in .cv_results_
'''
def display_best_score(cv_results_:dict):
    scoring = ['accuracy', 'precision', 'recall', 'auc']
    
    for score in scoring:

        for i, rank in enumerate(cv_results_['rank_test_' + score]):
            if rank == 1:
                print('best '+ score +' params combination:\n', cv_results_['params'][i])
                print('with mean '+ score, cv_results_['mean_test_' + score][i])
                print()

'''
df: raw data
X: transformed df into modelable data, for naively testing different models in Model Building section
y: label of training data
'''
df = pd.read_parquet("/kaggle/input/navigating-financial-instability/20231124_Financial_Risk_Project_train (1).parquet")
X = data_cleaning_preprocessing(df)
y = df['FinancialSector'].astype(bool)


pd.set_option('display.max_columns', None) # set max shaown columns to none
pd.set_option('display.max_rows', 100)


# First, I would like to know what it really means for some columns

df['fiscalDint'].value_counts() # date (mostly 20221231)

len(df) # 415 with high dimensions, not suitable to use NN

df['businessDescription'][0] # should be dealt with NLP

df['exchangeCountry'].value_counts() # all domestic

df['securityType'].value_counts() # only 1 different value

df['CIK'].nunique() # 414?

df['incorporationCountry'].value_counts() # mostly USA

df.describe() # there are missing values existed

df['exchangeID'].value_counts()



df['securityType'].value_counts() # other variables in the same situation: incorporationCountry


df[df['securityType'] == 'American Depository Receipt (ADR)']


df['incorporationCountry'].value_counts()


df.drop(['url', 'exchangeCountry', 'securityType', 'CIK', 'name', 'securityID', 'exchangeName', 'fiscalDint'], axis = 1, inplace = True)
df.head()


# lump incorporationCountry variable:
df.loc[df['incorporationCountry'] != 'USA', 'incorporationCountry'] = 'other'

df['incorporationCountry'].value_counts()


x = df['FinancialSector'].value_counts()
print(x,'\n','\n', 'Is FinancialSector among all: {0}'.format(x[1]/(x[0] + x[1])))


#check data types
df.info()


#numerical features
df.describe().style.set_sticky()


# split categorical and numerical features for convenience
nlp_cols = ['call_transcript', 'businessDescription']

num_cols = df.columns[(df.dtypes == 'float64') | (df.dtypes == 'int64')]

# not the current task so we drop it
num_cols = num_cols.drop(['FinancialRisk'])

bool_col = 'incorporationCountry'


sns.barplot(x = 'incorporationCountry', y = 'FinancialSector', data = df)


sns.barplot(x = 'exchangeID', y = 'FinancialSector', data = df)


facts = [
       'Assets', 'Capital Expenditure', 'Cash',
       'Depreciation', 'Dividend', 'Earnings', 'Equity', 'FCF', 'Income Tax',
       'Interest Expense', 'Long Liabilities', 'Long Term Debt', 'Market Cap',
       'Minority Interest', 'Operating Cash Flow', 'Operating Expense',
       'Operating Income', 'Operating Income Before Depreciation',
       'Operating Margin', 'Preferred Stock', 'Profit Margin', 'R&D',
       'R&D/Sales', 'Sales',
       'Short Term Debt', 'TEV', 'Working Capital',
       'close', 'dividendFactor', 'floatShares', 'outstandingShares',
       'shortInterestFloat']
ratios = ['VWAP', 'Accrual Ratio', 'B/P', 'CF/P', 'Debt/Equity', 'E/P', 'EBIT', 'EBIT/P',
       'EBIT/TEV', 'Earnings Growth (1Y)', 'Earnings Growth (2Y)',
       'Earnings Growth (3Y)', 'Earnings Growth (4Y)', 'Earnings Growth (5Y)',
       'Earnings Variability', 'FCF/P', 'ROA', 'ROE', 'S/P', 'SG&A', 'SG&A/Sales',  'Sales Growth (1Y)', 'Sales Growth (2Y)', 'Sales Growth (3Y)',
       'Sales Growth (4Y)', 'Sales Growth (5Y)', 'Sales Variability']
label = 'FinancialSector'


df.isna().sum()


# find columns with nan
nan_rich_cols = df.isna().sum()[df.isna().sum() > 0].index.tolist()

nan_portion = {}
for col in nan_rich_cols:
    nan_portion[col] = [df[df[col].isna() == True]['FinancialSector'].mean(),\
                        df[df[col].isna() == True]['FinancialSector'].count()]

nan_portion
nan_ratio = pd.DataFrame.from_dict(nan_portion).rename(index={0: 'ratio classified as 1', 1: '#nan'}).transpose()
nan_ratio


sns.scatterplot(x = nan_ratio['#nan'], y = nan_ratio['ratio classified as 1'])


# find rows with nan
nan_rich_rows = df.isna().sum(1)[df.isna().sum(1) > 0].index.tolist()

nan_label = pd.DataFrame(df.isna().sum(1)[df.isna().sum(1) > 0], columns = ["#nan"]).merge(df['FinancialSector'], how = 'inner', left_index = True, right_index = True)
nan_label 


sns.violinplot(data=nan_label, y='#nan', x='FinancialSector')


f = plt.figure(figsize = [20, 70])
for i in range(len(facts)):
    f.add_subplot(11, 3, i+1)
    sns.boxplot(y = facts[i], x = 'FinancialSector', data = df).set_title(facts[i])
plt.show()


f = plt.figure(figsize = [20, 50])
for i in range(len(ratios)):
    f.add_subplot(9, 3, i+1)
    sns.boxplot(y = ratios[i], x = 'FinancialSector', data = df).set_title(ratios[i])
plt.show()


# include only numerical columns
temp = df[facts + ratios]
k = 3

Q1 = temp.quantile(0.25)
Q3 = temp.quantile(0.75)
IQR = Q3 - Q1
low_lim = Q1 - k * IQR
high_lim = Q3 + k * IQR

# filter out outliers beyond the limit for each column, and calculate how many of
# them are classified as 1
outliers_portion = {}

for col, low, high in zip(temp.columns, low_lim, high_lim):
    outliers_portion[col] = [df[(temp[col] >= high)]['FinancialSector'].mean(),\
                             df[(temp[col] >= high)]['FinancialSector'].count(),\
                             df[(temp[col] <= low)]['FinancialSector'].mean(),\
                             df[(temp[col] <= low)]['FinancialSector'].count()]

outliers_ratio = pd.DataFrame.from_dict(outliers_portion).rename(index={0: 'higher outliers classified as 1',\
                                                                        1: '#higher outliers',\
                                                                        2: 'lower outliers classified as 1',\
                                                                        3: '#lower outliers'}).transpose()
outliers_ratio


corr = df[num_cols].corr()

# We have too many columns so we need to simplify the matrix
# trim dataframe to upper triangle, eliminate duplicate correlations
corr = corr.where(np.tril(np.ones_like(corr, dtype = bool), k = -1))

# identify high correlation exceeding 0.95 & -0.95, a rule of thumb
rank = corr.stack().sort_values(ascending = False)

rank[(rank >= 0.95) | (rank <= -0.95)]


heatmap_segment_size = 20
num_columns = len(corr.columns)

for i in range(0, num_columns, heatmap_segment_size):
    for j in range(0, num_columns, heatmap_segment_size):
        subset_corr = corr.iloc[i:i+heatmap_segment_size, j:j+heatmap_segment_size]
        
        # skip subset with only nan values
        if all(subset_corr.isna().all()): continue
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(subset_corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap - Columns {} to {}".format(i+1, i+heatmap_segment_size))
        plt.show()


highest_corr = rank[(rank >= 0.95) | (rank <= -0.95)].index.tolist()

f = plt.figure(figsize = [20, 20])
for i, pair in enumerate(highest_corr):
    f.add_subplot(4, 3, i+1)
    sns.scatterplot(y = pair[0], x = pair[1], data = df).set_title("{} - {}".format(pair[1], pair[0]))
plt.show()


high_corr = rank[((rank >= 0.7) & (rank < 0.95)) | ((rank > -0.95) & (rank <= -0.7))].index.tolist()
rank[((rank >= 0.7) & (rank < 0.95)) | ((rank > -0.95) & (rank <= -0.7))]


f = plt.figure(figsize = [20, 60])
for i, pair in enumerate(high_corr):
    f.add_subplot(11, 4, i+1)
    sns.scatterplot(y = pair[0], x = pair[1], data = df).set_title("{} - {}".format(pair[1], pair[0]))
plt.show()


f = plt.figure(figsize = [20, 70])
for i in range(len(facts)):
    f.add_subplot(11, 3, i+1)
    sns.kdeplot(x = facts[i], data = df).set_title(facts[i])
plt.show()


f = plt.figure(figsize = [20, 50])
for i in range(len(ratios)):
    f.add_subplot(9, 3, i+1)
    sns.kdeplot(x = ratios[i], data = df).set_title(ratios[i])
plt.show()


df.isna().sum()


# see all companies with nan in first year Earnings Growth

earnings_growth = ['Earnings Growth (1Y)', 'Earnings Growth (2Y)', 'Earnings Growth (3Y)', 'Earnings Growth (4Y)', 'Earnings Growth (5Y)']

with pd.option_context("display.max_rows", 1000):
    display(df[df['Earnings Growth (4Y)'].isna() == True][earnings_growth])


sales_growth = ['Sales Growth (1Y)', 'Sales Growth (2Y)', 'Sales Growth (3Y)', 'Sales Growth (4Y)', 'Sales Growth (5Y)']
df[df['Sales Growth (2Y)'].isna() == True][sales_growth]


# include only numerical features
def nan_impute_with_normalization(df_numerical_only, initial_strategy = 'median', imputation_order = 'roman'):

    pipe = make_pipeline(PowerTransformer(), 
                         IterativeImputer(initial_strategy = initial_strategy, # since too many outliers, mean will distort the parameter estimates
                                          imputation_order = imputation_order, # consideration for longitudinal variable
                                          skip_complete = True,
                                          verbose = 0),
                         Normalizer()).set_output(transform = 'pandas')

    return pipe.fit_transform(df_numerical_only)

df_num = nan_impute_with_normalization(df[facts + ratios])


# check distributions
f = plt.figure(figsize = [20, 70])
for i in range(len(facts)):
    f.add_subplot(11, 3, i+1)
    sns.kdeplot(x = facts[i], data = df_num).set_title(facts[i])
plt.show()


# f = plt.figure(figsize = [20, 50])
for i in range(len(ratios)):
    f.add_subplot(9, 3, i+1)
    sns.kdeplot(x = ratios[i], data = df_num).set_title(ratios[i])
plt.show()


# saves index that transcripts cannot be successfully stripped
err = []

# saves stripped transcript for every index
cleaned = {}

for index in df.index.tolist():
    s = df.loc[index, 'call_transcript']
    transcript = re.search("(Quarter|quarter).*Read more current", s)
    if transcript:
        cleaned[index] = s[transcript.start():transcript.end()]
    else:
        err.append(index)




# find potential useless transcripts where they probably have length < 6000
index, _ = zip(*filter(lambda x: len(x[1]) <= 6000, zip(df.index.tolist(), df['call_transcript'])))

print(len(index),'\n', sorted(index), '\n') # 56 transcripts error
print(len(err), '\n', sorted(err)) # 59 transcripts did not successfully stripped

# find difference between two lists
for x in err:
    if x not in index: print(x) # RXRX, BEAM, VERV


s = cleaned['FICO']

# use translator to remove punctuations and digits
translator = str.maketrans('', '', string.punctuation + string.digits)
s = s.translate(translator).lower()

# use default stop words list from nltk
stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

tokens = []
for word in nltk.word_tokenize(s):
    if word not in stopwords:
        tokens.append(lemmatizer.lemmatize(stemmer.stem(word)))

token_count = Counter(tokens)
# keep only words with more than 1 occurrence 
tokens = [token for token in tokens if token_count[token] > 1]
Counter(tokens)
    


more_stopwords = "quarter, year, think, go, thank, see, continu, oper, million, question, busi, u, market, expect, look, like, next, would, realli, come, weve, that, custom, product, time, one, also, call, get, revenu, new, last, kind, point, third, margin, q, result, first, line, talk, want, today, mayb, basi, make, say, plea, end, work, could, compani, your, earn, there, around, number, across, way, financi, team, second, got, fourth, im, morn, mention, month, said, compar, actual, incom, per, may"


def clean_transcripts(df_dirty):
    
    df = deepcopy(df_dirty)
    
    translator = str.maketrans('', '', string.punctuation + string.digits)
    stopwords = nltk.corpus.stopwords.words('english')
    stemmer = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()
    

    for index in df.index.tolist():
        s = df.loc[index, 'call_transcript']
        
        # strip out transcript contexts
        match = re.search("(Quarter|quarter).*Read more current", s)
        
        if not match:
            df.loc[index, 'call_transcript'] = ['']
            continue
        
        s = s[match.start():match.end()]
        
        # remove digits and punctuation, lowercasing
        s = s.translate(translator).lower()
        
        # stemming, lemmatizing and tokenization
        tokens = []
        for word in nltk.word_tokenize(s):
            if word not in stopwords:
                word = lemmatizer.lemmatize(stemmer.stem(word))
                if word not in more_stopwords.split(', '):
                    tokens.append(word)

        token_count = Counter(tokens)
        
        # keep only words with more than 1 occurrence 
        tokens = [token for token in tokens if token_count[token] > 1]
        
        # overwrite values of df call_transcript column to tokens
        df.at[index, 'call_transcript'] = tokens
        
    
    return df
        


df_TranscriptsCleaned = clean_transcripts(df)

count = Counter()
for index in df_TranscriptsCleaned.index.tolist():
    s = df_TranscriptsCleaned.loc[index, 'call_transcript']
    count += Counter(s)


# instead of absolute values of word frequencies, we show the relative frequency ratio among all words' occurence
total = count.total()
for key, value in count.items():
    count[key] = round(value/total, 4) * 100


count.most_common(20)


# before using hashing vectorizer, we have to modify a little bit of our text cleaning function
def clean_stem_lemmatize(text):
    
    translator = str.maketrans('', '', string.punctuation + string.digits)
    stopwords = nltk.corpus.stopwords.words('english')
    stemmer = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()


    # strip out transcript contexts
    match = re.search("(Quarter|quarter).*Read more current", text)

    if not match:
        return ['']

    s = text[match.start():match.end()]

    # remove digits and punctuation, lowercasing
    s = s.translate(translator).lower()

    # stemming, lemmatizing and tokenization
    tokens_ = []
    for word in nltk.word_tokenize(s):
        if word not in stopwords:
            word = lemmatizer.lemmatize(stemmer.stem(word))
            if word not in more_stopwords.split(', '):
                tokens_.append(word)

    token_count = Counter(tokens_)

    # keep only words with more than 1 occurrence 
    return [token for token in tokens_ if token_count[token] > 1]




def NLP_trans(list_of_texts, ngram_range = (1, 2), norm = 'l1', n_features = 500):
    vectorizer = HashingVectorizer(tokenizer = clean_stem_lemmatize, 
                                   preprocessor = None, 
                                   ngram_range = ngram_range, 
                                   norm = norm, 
                                   lowercase = False, # important: not sure what happened but has to set as False to run correctly
                                   n_features = n_features)
    
    return vectorizer.fit_transform(list_of_texts)



x = NLP_trans(df['call_transcript'].tolist())
x.shape


def concat_csr(csr, df):
    
    vec_df = pd.DataFrame(csr)
    # a copy of df, for later resetting index and column names
    df_index = df.index
    df_reset = df.reset_index(drop = True)

    # concat df and matrix
    df_concat = pd.concat([df_reset, vec_df], axis = 1, ignore_index = True)

    # originate index
    df_concat.index = df_index

    #originate column names
    df_concat.columns = list(df.columns) + [f'col_{i}' for i in range(500)]
    
    return df_concat




y = df['FinancialSector']
y = y.astype('bool')
X = df.drop(['FinancialRisk', 'businessDescription', 'FinancialSector', 'exchangeID'], axis = 1)

# since we only have 1 column for NLP, 1 column for encoding, rest of columns for normalization, I manually do it. Otherwise
# we can use sklearn.compose.ColumnTransformer() to do it in a single pipeline.
cat = X['incorporationCountry']
texts = X['call_transcript']
num = X[facts + ratios]


# encoding
cat.replace(['USA', 'other'], [True, False], inplace = True)


# numerical transformation
num = nan_impute_with_normalization(num)


# NLP transformation
csr_matrix = NLP_transformation(texts.tolist())


csr_matrix.shape


# concat them all together
X = concat_csr(csr_matrix, pd.concat([cat, num], axis = 1))
X




def L1_regularization(X, y, alpha = 0.001, max_iter = 1000, tol = 1e-4):
    regularizer = Lasso(alpha = alpha, max_iter = max_iter, tol = tol)
    regularizer.fit(X, y)
    feature_importance = pd.DataFrame()
    feature_importance['feature'] = X.columns.tolist()
    feature_importance['importance'] = np.abs(regularizer.coef_)
    return feature_importance[feature_importance['importance'] != 0].sort_values('importance', ascending = False).reset_index().drop(['index'], axis = 1)

feature_importance = L1_regularization(X, y, alpha = 0.001)
feature_importance


for corr in highest_corr:
    if corr[0] in feature_importance['feature']\
    and corr[1] in feature_importance['feature']:
        print(corr)


df = pd.read_parquet("/kaggle/input/navigating-financial-instability/20231124_Financial_Risk_Project_train (1).parquet")
X = data_cleaning_preprocessing(df)
y = df['FinancialSector'].astype(bool)


dummy_clf = DummyClassifier()
dummy_clf.fit(X, y)
DummyClassifier()
dummy_clf.predict(X)
dummy_clf.score(X, y)


# check multicollinearity with VIF for regularized features


alpha = 0.001
feat = L1_regularization(X, y, alpha)['feature'].tolist()

# keep necessary features
X_LR = X[feat].drop(['incorporationCountry', 'R&D/Sales', 'Profit Margin'], axis = 1) # drop incorporationCountry since VIF cannot process bool

# calculate VIF
def VIF(X):
    vif_df = pd.DataFrame() 
    vif_df["feature"] = X.columns 

    # calculating VIF for each feature 
    vif_df["VIF"] = [variance_inflation_factor(X.values, i) 
                              for i in range(len(X.columns))] 
    return vif_df
VIF(X_LR)


X_LR = X_LR.drop(['SG&A', 'Sales'], axis = 1)
VIF(X_LR)


# check multicollinearity with correlation
corr_LR = X_LR.corr().where(np.tril(np.ones_like(X_LR.corr(), dtype = bool), k = -1))
plt.figure(figsize=(12, 8))
sns.heatmap(corr_LR, annot=True, cmap='coolwarm', fmt=".1f")


X_LR = X_LR.drop(['Depreciation', 'Earnings Growth (3Y)', 'E/P'], axis = 1)
X_LR


# naive training and testing using LR


LR = LogisticRegression(penalty = None, 
                        C = 1.0,
                        class_weight = {1: 0.35, 0: 0.65},
                        solver = 'lbfgs',
                        max_iter = 500)

scoring = ['accuracy','precision','recall', 'roc_auc']

cv_LR = cross_validate(
    LR,
    X_LR,
    y,
    scoring=scoring,
    cv=5
)


cv_LR_accuracy = cv_LR['test_accuracy'].mean()
cv_LR_precision = cv_LR['test_precision'].mean()
cv_LR_recall = cv_LR['test_recall'].mean()
cv_LR_auc = cv_LR['test_roc_auc'].mean()


print('LR accuracy:', '{:,}'.format(cv_LR_accuracy))
print('LR precision:', '{:,}'.format(cv_LR_precision))
print('LR recall:', '{:,}'.format(cv_LR_recall))
print('LR auc:', '{:,}'.format(cv_LR_auc))



LR = LogisticRegression(penalty = 'elasticnet', 
                        C = 1,
                        class_weight = {1: 0.35, 0: 0.65},
                        solver = 'saga',
                        max_iter = 500,
                        l1_ratio = 0.99)

scoring = ['accuracy','precision','recall', 'roc_auc']

cv_LR = cross_validate(
    LR,
    X,
    y,
    scoring=scoring,
    cv=5,
    error_score = 'raise'
)


cv_LR_accuracy = cv_LR['test_accuracy'].mean()
cv_LR_precision = cv_LR['test_precision'].mean()
cv_LR_recall = cv_LR['test_recall'].mean()
cv_LR_auc = cv_LR['test_roc_auc'].mean()


print('LR accuracy:', '{:,}'.format(cv_LR_accuracy))
print('LR precision:', '{:,}'.format(cv_LR_precision))
print('LR recall:', '{:,}'.format(cv_LR_recall))
print('LR auc:', '{:,}'.format(cv_LR_auc))



# a way to customize decision threshold
# def custom_predict(X, threshold):
#     probs = LR.predict_proba(X) 
#     return (probs[:, 1] > threshold).astype(int)
    
    
# custom_predict(X=X_LR, threshold=0.4) 





RF = RandomForestClassifier()

scoring = ['accuracy','precision','recall', 'roc_auc']

cv_RF = cross_validate(RF,
                       X,
                       y,
                      scoring = scoring)

cv_RF_accuracy = cv_RF['test_accuracy'].mean()
cv_RF_precision = cv_RF['test_precision'].mean()
cv_RF_recall = cv_RF['test_recall'].mean()
cv_RF_auc = cv_RF['test_roc_auc'].mean()

print('RF accuracy:', '{:,}'.format(cv_RF_accuracy))
print('RF precision:', '{:,}'.format(cv_RF_precision))
print('RF recall:', '{:,}'.format(cv_RF_recall))
print('RF auc:', '{:,}'.format(cv_RF_auc))



XGB = xgb.XGBClassifier()

scoring = ['accuracy','precision','recall', 'roc_auc']

cv_XGB = cross_validate(XGB,
                       X,
                       y,
                      scoring = scoring)

cv_XGB_accuracy = cv_XGB['test_accuracy'].mean()
cv_XGB_precision = cv_XGB['test_precision'].mean()
cv_XGB_recall = cv_XGB['test_recall'].mean()
cv_XGB_auc = cv_XGB['test_roc_auc'].mean()

print('XGB accuracy:', '{:,}'.format(cv_XGB_accuracy))
print('XGB precision:', '{:,}'.format(cv_XGB_precision))
print('XGB recall:', '{:,}'.format(cv_XGB_recall))
print('XGB auc:', '{:,}'.format(cv_XGB_auc))


LR = LogisticRegression(penalty = 'l1', class_weight = {1: 0.35, 0: 0.65}, max_iter = 1000)



params = {'solver': ['liblinear', 'saga'],
          'tol': [1e-2, 1e-4],
          'C': [1e2, 1e3, 1e4, 1e5]}

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'auc': make_scorer(roc_auc_score, multi_class='ovr')
}

LR_cv = GridSearchCV(LR,
                     param_grid = params,
                     scoring = scoring,
                     verbose = 2,
                     refit = False)


LR_cv.fit(data_cleaning_preprocessing(df), y)


display_best_score(LR_cv.cv_results_)


params = {'solver': ['liblinear', 'saga'],
          'tol': [1e-4, 1e-5],
          'C': [1e4, 1e5, 1e6],
          'max_iter': [5000]}

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'auc': make_scorer(roc_auc_score, multi_class='ovr')
}

LR_cv = GridSearchCV(LR,
                     param_grid = params,
                     scoring = scoring,
                     verbose = 0,
                     refit = False)

LR_cv.fit(X, y)
display_best_score(LR_cv.cv_results_)


def data_cleaning(df, y = None):
    
    X = deepcopy(df)
    
    # see whether training set or testing set, simply for reuseability
    if 'FinancialSector' in X.columns.tolist():
        X.drop(['FinancialSector', 'FinancialRisk'], axis = 1, inplace = True)
        
    # drop all unnecessary columns, including those found in EDA analysis
    X.drop(['url', 'exchangeCountry', 'securityType', 'CIK', 'name', 'securityID', 'exchangeName', 'fiscalDint', 'exchangeID', 'businessDescription'], axis = 1, inplace = True)
    
    # categoricals encoding
    X.loc[X['incorporationCountry'] != 'USA', 'incorporationCountry'] = 'other'
    X['incorporationCountry'].replace(['USA', 'other'], [True, False], inplace = True)
    
    return X




cleaning_transformer = FunctionTransformer(data_cleaning)

nan_imputation_pipe = make_pipeline(PowerTransformer(), # minimize skewness and stablize variance
                                     IterativeImputer(sample_posterior = True, # linear regression, suitable for our data
                                                      initial_strategy = 'median', # since too many outliers, mean will distort the parameter estimates
                                                      imputation_order = 'roman', # consideration for longitudinal variable
                                                      skip_complete = True,
                                                      n_nearest_features = 10,
                                                      verbose = 0),
                                     Normalizer() # other scalers suffers from outliers
                                   )

word2vec = HashingVectorizer(tokenizer = clean_stem_lemmatize, 
                               token_pattern = None,
                               preprocessor = None, 
                               ngram_range = (1,1), 
                               norm = 'l1', 
                               lowercase = False, # important: not sure what happened but has to set as False to run correctly
                               n_features = 500)

featurization = ColumnTransformer([('NLP', word2vec, 0),
                                   ('nan_imputation_norm', nan_imputation_pipe, [1] + list(range(3, 61)))],
                                  sparse_threshold = 0)

RF_pipeline = Pipeline([('cleaning_transformer', cleaning_transformer),
                        ('featurization', featurization),
                        ('RF', RandomForestClassifier(class_weight = {1:0.35, 0:0.65},
                                                      criterion = 'entropy',
                                                      max_depth = None, 
                                                      n_estimators = 200
                                                     ))])




#Halving Grid Search CV on entire pipeline

params = {
#     'featurization__NLP__ngram_range': [(1, 1), (1, 2)],
#     'featurization__NLP__n_features': [50, 300, 500],
#     'featurization__nan_imputation_norm__iterativeimputer__n_nearest_features': [10, 30, None],
#     'RF__n_estimators': [200, 300],
#     'RF__criterion': ['gini', 'entropy'],
#     'RF__max_depth': [500, None],
#     'RF__min_samples_leaf': [0.01],
#     'RF__max_features':[100],
#     'RF__ccp_alpha': [0.0, 0.01]
}


RF_cv = HalvingGridSearchCV(RF_pipeline,
                     param_grid = params,
                     scoring = 'roc_auc',
                     verbose = 1,
                     n_jobs = -1,
                     cv = 5, # cv = 3 if tuning
                     refit = True)

RF_cv.fit(df, y)




RF_cv.best_params_
RF_cv.best_score_


best_RF = RF_cv.best_estimator_
best_RF


'''
Note: this code chunk runtime > 12h since it runs too many hyperparameter searches. Better tune under 3 of hyperparameters 
below for a responsive result or use RandomSearchCV.
'''


# #Traditional Grid Search CV on entire pipeline

# params = {
#     'featurization__NLP__ngram_range': [(1, 1), (1, 2)],
#     'featurization__NLP__n_features': [20, 300, 500],
#     'featurization__nan_imputation_norm__iterativeimputer__n_nearest_features': [10, 30, None],
#     'RF__criterion': ['gini', 'entropy'],
#     'RF__max_depth': [100, 200, None],
#     'RF__min_samples_leaf': [1, 0.1, 0.01],
#     'RF__max_features':[25, 'sqrt', 100],
#     'RF__ccp_alpha': [0.0, 0.01]
# }

# scoring = {
#     'accuracy': make_scorer(accuracy_score),
#     'precision': make_scorer(precision_score),
#     'recall': make_scorer(recall_score),
#     'auc': make_scorer(roc_auc_score, multi_class='ovr')
# }

# RF_cv = HalvingGridSearchCV(RF_pipeline,
#                      param_grid = params,
#                      scoring = scoring,
#                      verbose = 2,
#                      n_jobs = -1,
#                      cv = 3,
#                      refit = False)

# RF_cv.fit(df, y)
# display_best_score(RF_cv.cv_results_)



# since the best parameter has been applied into transformers (and it is costly to tune it again), we cv only XGBoost
XGB = xgb.XGBClassifier(learning_rate = 0.1, colsample_bytree = 0.1)


params = {
#     'n_estimators': [200, 300, None],
#     'max_depth': [100, 200],
#     'max_leaves': [200, 300],
#     'learning_rate': [0.1],
#     'colsample_bytree': [0.1]
}


XGB_cv = HalvingGridSearchCV(XGB,
                     param_grid = params,
                     scoring = 'roc_auc',
                     verbose = 1,
                     cv = 5, # cv = 3 if tuning
                     refit = True)

XGB_cv.fit(X, y)




# print(XGB_cv.best_params_)
# print(XGB_cv.best_score_)


best_XGB_pipe = Pipeline([('cleaning_transformer', cleaning_transformer),
                        ('featurization', featurization),
                        ('RF', xgb.XGBClassifier(learning_rate = 0.1, colsample_bytree = 0.1))])


X_test = pd.read_parquet("/kaggle/input/navigating-financial-instability/20231124_Financial_Risk_Project_test_public (1).parquet")


best_XGB_pipe.fit(df, y)
y_submission = best_XGB_pipe.predict(X_test)
y_submission


y_submission_df = pd.DataFrame(y_submission, index = X_test.index, columns = ['FinancialSector'])
y_submission_csv = y_submission_df.to_csv('submission.csv')



LR = LogisticRegression(penalty = 'l1', class_weight = {1: 0.35, 0: 0.65}, max_iter = 5000, C = 1000000.0, solver = 'liblinear', random_state = 43 )


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 50)
LR.fit(X_train, y_train)
predictions = LR.predict_proba(X_test)


from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

accuracy, precision, recall, auc = [], [], [], []

for threshold in thresholds:
    
    predicts = predictions[:, 1] > threshold

    accuracy.append(accuracy_score(y_test, predicts))
    precision.append(precision_score(y_test, predicts))
    recall.append(recall_score(y_test, predicts))
    auc.append(roc_auc_score(y_test, predicts))
    
    print('Threshold:', threshold)
    print('  ', 'accuracy:', accuracy_score(y_test, predicts))
    print('  ', 'precision:', precision_score(y_test, predicts))
    print('  ', 'recall:', recall_score(y_test, predicts))
    print('  ', 'auc:', roc_auc_score(y_test, predicts))
    print()




d = pd.DataFrame({'accuracy': accuracy, 'precision': precision, 'recall': recall, 'auc': auc}, index = thresholds)
sns.lineplot(d)

