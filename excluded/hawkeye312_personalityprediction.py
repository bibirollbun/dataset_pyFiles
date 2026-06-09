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


train_data = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train_data.head()


test_data.head()


train_data.isna().sum()


train_data.shape


test_data.isna().sum()


test_data.shape


import seaborn as sns
import matplotlib.pyplot as plt
sns.catplot(data=train_data.drop(columns=['id']), kind='box')
plt.xticks(rotation=45)


# Use median for one with outlier, mean for rest.
aggs = {
    'Time_spent_Alone': 'median',
    'Social_event_attendance': 'mean',
    'Going_outside': 'mean',
    'Friends_circle_size': 'mean',
    'Post_frequency': 'mean'
}


for column, agg in aggs.items():
    train_data[column] = train_data[column].fillna(round(train_data[column].agg(agg)))
    test_data[column] = test_data[column].fillna(round(test_data[column].agg(agg)))

train_data['Stage_fear'] = train_data['Stage_fear'].ffill()
test_data['Stage_fear'] = test_data['Stage_fear'].ffill()
train_data['Drained_after_socializing'] = train_data['Drained_after_socializing'].ffill()
test_data['Drained_after_socializing'] = test_data['Drained_after_socializing'].ffill()


train_data.head()


train_data.isna().sum()


test_data.isna().sum()


dummy_features = ['Stage_fear', 'Drained_after_socializing']
train_data = pd.get_dummies(train_data, columns=dummy_features, drop_first=True)
test_data = pd.get_dummies(test_data, columns=dummy_features, drop_first=True)


y = train_data['Personality']
X_train = train_data.drop(columns=['Personality', 'id'])
X_test = test_data.drop(columns=['id'])


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y)


predictions = knn.predict(X_test)


output = pd.DataFrame({'id': test_data.id, 'Personality': predictions})
output.to_csv('submission.csv', index=False)
print("Submission was successfully saved!")




