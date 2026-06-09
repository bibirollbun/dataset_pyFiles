import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB


cats_df = pd.read_csv('/kaggle/input/21vek-query-classification/categories.csv')
train_df = pd.read_csv('/kaggle/input/21vek-query-classification/train.csv')
test_df = pd.read_csv('/kaggle/input/21vek-query-classification/test.csv')


cats_df


train_df = train_df.merge(cats_df, on='CategoryID', how='left')
train_df


test_df


fig, ax = plt.subplots(1,1, figsize=(10, 5))
train_df.groupby('CategoryName').Query.count().sort_values().plot.bar(x='CategoryName', y='Query')
plt.xticks(rotation=90)
plt.show()


model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])
model.fit(train_df['Query'], train_df['CategoryID'])


test_df['CategoryID'] = model.predict(test_df['Query'])
test_df[['ID', 'CategoryID']].to_csv('The_Naivest_Baseline.csv', index=None)


fig, ax = plt.subplots(1,1, figsize=(10, 5))
df = test_df.merge(cats_df, on='CategoryID', how='left')
df.groupby('CategoryName').Query.count().sort_values().plot.bar(x='CategoryName', y='Query')
plt.xticks(rotation=90)
plt.show()

