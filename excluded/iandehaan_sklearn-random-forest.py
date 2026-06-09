import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')

df_train = pd.read_csv('/kaggle/input/home-depot/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('/kaggle/input/home-depot/test.csv', encoding="ISO-8859-1")
# df_attr = pd.read_csv('/kaggle/input/home-depot/attributes.csv')
df_pro_desc = pd.read_csv('/kaggle/input/home-depot/product_descriptions.csv')
df_merged = pd.merge(df_train, df_pro_desc, how='left', on='product_uid')

def str_stemmer(s):
	return " ".join([stemmer.stem(word) for word in s.lower().split()])

def str_common_word(str1, str2):
	return sum(int(str2.find(word)>=0) for word in str1.split())

def df_preprocessing(df, stemming = True):
    df = df.copy()
    if stemming:
        for column in ('search_term', 'product_title', 'product_description'):
            df[column] = df[column].map(lambda x:str_stemmer(x))
            
    df['len_of_query'] = df['search_term'].map(lambda x:len(x.split())).astype(np.int64)
    
    df['product_info'] = df['search_term']+"\t"+df['product_title']+"\t"+df['product_description']
    
    df['word_in_title'] = df['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
    df['word_in_description'] = df['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
    
    df = df.drop(['search_term','product_title','product_description','product_info'],axis=1)
    return df

# df_preprocessed = df_preprocessing(df_merged)
# print(df_preprocessed)
# df_no_stemming = df_preprocessing(df_merged, False)


from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time

split = train_test_split(df_preprocessed.drop(['id','relevance'],axis=1).values, df_preprocessed['relevance'].values, test_size=0.2, random_state=42)
split_no_stemming = train_test_split(df_no_stemming.drop(['id','relevance'],axis=1).values, df_no_stemming['relevance'].values, test_size=0.2, random_state=42)



def BaselineModel(X_train,X_test,y_train,y_test):
    start = time.time()
    rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
    clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'elapsed time: {time.time() - start}')
    print(f'rmse for non-stemmered data: {mse**0.5}')

print("with stemming:")
BaselineModel(*split)
print("without stemming:")
BaselineModel(*split_no_stemming)



import spacy
from tqdm import tqdm
!python -m spacy download en_core_web_md
tqdm.pandas()
def similarity(nlp, text_a, text_b):
    return nlp(text_a).similarity(nlp(text_b))

def alternative_preprocessing(df, stemming=True):
    df = df.copy()
    if stemming:
        for column in ('search_term', 'product_title', 'product_description'):
            df[column] = df[column].map(lambda x:str_stemmer(x))
    nlp=spacy.load("en_core_web_md")
    df['product_info'] = df['search_term']+"\t"+df['product_title']+"\t"+df['product_description']
    df['title_similarity'] = df['product_info'].progress_map(lambda x:similarity(nlp,x.split('\t')[0],x.split('\t')[1]))
    df['description_similarity'] = df['product_info'].progress_map(lambda x:similarity(nlp,x.split('\t')[0],x.split('\t')[2]))
    return df
    
alternarive_df = alternative_preprocessing(df_merged, stemming=False)


alternarive_df.to_csv('alternative.csv')


nlp=spacy.load("en_core_web_md")
title = nlp(df_merged['product_title'][4])
description = nlp(df_merged['product_description'][3])
search = nlp(df_merged['search_term'][4])
search.similarity(title)


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model

def LinearModel(X_train,X_test,y_train,y_test):
    start = time.time()
    reg = linear_model.Ridge(alpha=.1)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'elapsed time: {time.time() - start}')
    print(f'rmse for non-stemmered data: {mse**0.5}')

LinearModel(*split)


from sklearn.ensemble import AdaBoostRegressor

def AdaBoostModel(X_train,X_test,y_train,y_test):
    start = time.time()
    regr = AdaBoostRegressor(random_state=0, n_estimators=20)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'elapsed time: {time.time() - start}')
    print(f'rmse for non-stemmered data: {mse**0.5}')

AdaBoostModel(*split)


from sklearn.neighbors import KNeighborsRegressor

def KNeighborsModel(X_train,X_test,y_train,y_test):
    start = time.time()
    neigh = KNeighborsRegressor(n_neighbors=2)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'elapsed time: {time.time() - start}')
    print(f'rmse for non-stemmered data: {mse**0.5}')

KNeighborsModel(*split)




