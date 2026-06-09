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


from sklearn.model_selection import train_test_split

df = pd.read_csv('/kaggle/input/short-project-rec-sys/train.csv')
train, val = train_test_split(df, test_size=0.2, random_state=7)


class TopPopRecommender():

    def fit(self, train):

        item_popularity = train[['movie_id','rating']].groupby(by='movie_id').count()

        self.train = train
        # We are not interested in sorting the popularity value,
        # but to order the items according to it
        self.popular_items = item_popularity.sort_values(by='rating',ascending=False).index
    
    
    def predict_top(self, user_id, at=5, remove_seen=True):

        if remove_seen:
            seen_items = self.train[self.train.user_id==user_id].movie_id.values
            unseen_items_mask = np.in1d(self.popular_items, seen_items, assume_unique=True, invert = True)
            unseen_items = self.popular_items[unseen_items_mask]
            recommended_items = unseen_items[0:at]

        else:
            recommended_items = self.popular_items[0:at]
    
        return recommended_items


import csv

topPopular = TopPopRecommender()
topPopular.fit(train)


test = pd.read_csv('/kaggle/input/short-project-rec-sys/kaggle_baseline.csv')

# open the file in the write mode
with open('submission.csv', 'w',encoding='UTF8') as f:
    # create the csv writer
    writer = csv.writer(f)
    # write a row to the csv file
    writer.writerow(['user_id', 'prediction'])
    for user_id in test.user_id.unique():
        relevant_items = topPopular.predict_top(user_id, at=25)
        list_relevants = ' '.join([str(elem) for elem in relevant_items])
        writer.writerow([str(user_id),list_relevants])




