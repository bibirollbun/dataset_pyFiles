import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer


df_train = pd.read_csv('/kaggle/input/home-depot-product-search-relevance/train.csv.zip', encoding = "ISO-8859-1")
df_test = pd.read_csv('/kaggle/input/home-depot-product-search-relevance/test.csv.zip', encoding = "ISO-8859-1")


# Detailed product introduction is useful, because we need more corpus information to support our search
df_desc = pd.read_csv('/kaggle/input/home-depot-product-search-relevance/product_descriptions.csv.zip')


df_train.head()


df_desc.head()


# The training set and the test set are merged first to facilitate unified preprocessing
df_all = pd.concat((df_train, df_test), axis = 0, ignore_index = True)
df_all.head()


df_all.shape


# The product introduction information also needs to be merged
df_all = pd.merge(df_all, df_desc, how = 'left', on = 'product_uid')
df_all.head()


# Next, text preprocessing is carried out
stemmer = SnowballStemmer('english')

# Part-of-speech normalization process
def str_stemmer(s):
    return " ".join([stemmer.stem(word) for word in s.lower().split()])

# To calculate the validity of keywords, see how many times the words appear
def str_common_word(str1, str2):
    return sum(int(str2.find(word) >= 0) for word in str1.split())


# Unify the word forms of all text data
df_all['search_term'] = df_all['search_term'].map(lambda x : str_stemmer(x))

df_all['product_title'] = df_all['product_title'].map(lambda x : str_stemmer(x))

df_all['product_description'] = df_all['product_description'].map(lambda x : str_stemmer(x))


# Next, we can create some text features
df_all['len_of_query'] = df_all['search_term'].map(lambda x : len(x.split())).astype(np.int64)

df_all['commons_in_title'] = df_all.apply(lambda x : str_common_word(x['search_term'], x['product_title']), axis = 1)

df_all['commons_in_desc'] = df_all.apply(lambda x : str_common_word(x['search_term'], x['product_description']), axis = 1)


# The columns that cannot be processed by the machine learning model will be droped
df_all = df_all.drop(['search_term', 'product_title', 'product_description'], axis = 1)


# reshape the train/data set
df_train = df_all.loc[df_train.index]
df_test = df_all.loc[df_test.index]


# Record the test set id
test_ids = df_test['id']


y_train = df_train['relevance'].values
X_train = df_train.drop(['id', 'relevance'], axis = 1).values
X_test = df_test.drop(['id', 'relevance'], axis = 1).values


# Establish the Ridge model and debug the alpha value
from sklearn.model_selection import cross_val_score

params = [1, 3, 5, 6, 7, 8, 9, 10]
test_scores = []
for param in params:
    clf = RandomForestRegressor(n_estimators = 30, max_depth = param)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv = 5, scoring = 'neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))

plt.plot(params, test_scores)
plt.title("Param vs CV Error")


# Upload the result
rf = RandomForestRegressor(n_estimators = 30, max_depth = 7)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
output_path = "/kaggle/working/submission.csv"  
result_df = pd.DataFrame({"id": test_ids, "relevance": y_pred})
result_df.to_csv(output_path, index=False)




