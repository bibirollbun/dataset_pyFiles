!7z x /kaggle/input/mercari-price-suggestion-challenge/train.tsv.7z


!7z x /kaggle/input/mercari-price-suggestion-challenge/test_stg2.tsv.zip


import pandas as pd

data = pd.read_csv("train.tsv", sep='\t')
data


data.info()


data.brand_name.unique().shape


data.category_name.unique().shape


import matplotlib.pyplot as plt

plt.hist(data.item_description.apply(lambda x: len(x.split()) if isinstance(x, str) else 0), bins=100);


import re
max_text_length=100

def clean_str(text):
    if isinstance(text, str):
        text = ' '.join( [w for w in text.split()[:max_text_length]] )        
        text = text.lower()
        text = re.sub(u"é", u"e", text)
        text = re.sub(u"ē", u"e", text)
        text = re.sub(u"è", u"e", text)
        text = re.sub(u"ê", u"e", text)
        text = re.sub(u"à", u"a", text)
        text = re.sub(u"â", u"a", text)
        text = re.sub(u"ô", u"o", text)
        text = re.sub(u"ō", u"o", text)
        text = re.sub(u"ü", u"u", text)
        text = re.sub(u"ï", u"i", text)
        text = re.sub(u"ç", u"c", text)
        text = re.sub(u"\u2019", u"'", text)
        text = re.sub(u"\xed", u"i", text)
        text = re.sub(u"w\/", u" with ", text)
        
        text = re.sub(u"[^a-z0-9]", " ", text)
        text = u" ".join(re.split('(\d+)',text) )
        text = re.sub( u"\s+", u" ", text ).strip()
        text = ''.join(text)
    else:
        text = ""
    return text


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from sklearn.preprocessing import OneHotEncoder

def prepare_data(df, is_train=True):
    df.item_condition_id.fillna(2, inplace=True)
    df.shipping.fillna(0, inplace=True)
    df_train.brand_name.fillna('', inplace=True)
    if is_train:
        df = df.loc[df.price>0].reset_index(drop=True)
        df.price = np.log1p(df.price).astype(np.float32)
        df.drop('train_id', axis=1, inplace=True)
    else:
        df.drop('test_id', axis=1, inplace=True)

    df.category_name.fillna('//', inplace=True)
    df['category1'] = df.category_name.apply(lambda x : x.split('/')[0].strip())
    df['category2'] = df.category_name.apply(lambda x : x.split('/')[1].strip())
    df['category3'] = df.category_name.apply(lambda x : x.split('/')[2].strip())
    df['category_name'] = df.category_name.apply( lambda x : ' '.join( x.split('/') ).strip())

    df['brand_name'] = df['brand_name'].apply(clean_str)
    df['name'] = df['name'].apply(clean_str)
    df['item_description'] = df['item_description'].apply(clean_str)

    df['name_desc'] = df['name'] + ' ' + df['item_description']
    df.drop('name', axis=1, inplace=True)
    df.drop('category_name', axis=1, inplace=True)
    df.drop('item_description', axis=1, inplace=True)

    return df


df_train = pd.read_csv('train.tsv', sep='\t', encoding='utf-8')

df_train = prepare_data(df_train)

y = df_train['price']

nltk.download('stopwords')

nltk_stop_words = stopwords.words('english')

tfidf = TfidfVectorizer(max_df=0.99, min_df=0.01, stop_words=nltk_stop_words)

tfidf_matrix = tfidf.fit_transform(df_train['name_desc'])

ohe = OneHotEncoder(handle_unknown='ignore')

ohe_df_train = ohe.fit_transform(df_train.drop('name_desc', axis=1).drop('price', axis=1))


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.sparse import hstack
from catboost import CatBoostRegressor

X = hstack([ohe_df_train, tfidf_matrix])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = CatBoostRegressor(iterations=1000, learning_rate=0.5, verbose=False, max_depth=10)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


!ls


df_test = pd.read_csv('test_stg2.tsv', sep='\t', encoding='utf-8')

submission=pd.DataFrame()
submission['test_id'] = df_test['test_id']

df_test = prepare_data(df_test, False)

tfidf_matrix_test = tfidf.transform(df_test['name_desc'])

ohe_df_test = ohe.transform(df_test.drop('name_desc', axis=1))

X_test = hstack([ohe_df_test, tfidf_matrix_test])

y_pred_test = np.expm1(model.predict(X_test))

y_pred_test[y_pred_test<3]=3
y_pred_test[y_pred_test>1000]=1000
submission['price'] = y_pred_test
submission.to_csv('submission.csv', index=False)

