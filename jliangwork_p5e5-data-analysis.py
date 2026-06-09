# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting
import seaborn as sns # plotting for pandas dataframes

from sklearn.model_selection import KFold # split data into folds

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


train_w_source = train.copy()
test_w_source = test.copy()
train_w_source['Source'] = 'Train'
test_w_source['Source'] = 'Test'
test_w_source['Calories'] = -1.0
combined_data = pd.concat([train_w_source,test_w_source])


train.info()


test.info()


sample


train[['Sex']].describe()


375721/750000


test[['Sex']].describe()


125281/250000


sns.boxplot(data = train, x = 'Sex', y = 'Calories')


train.drop('id',axis=1).describe()


test.drop('id',axis=1).describe()


combined_data['Age']


sns.boxplot(data = combined_data, x = 'Source', y = 'Age')
plt.title("Age in Data")
plt.show()


sns.histplot(data=train, x= 'Age', binwidth = 1)
plt.title("Age in Train Data")


sns.histplot(data=test, x= 'Age', binwidth = 1)
plt.title("Age in Test Data")


sns.boxplot(data = train, x = 'Age', y = 'Calories')
plt.title("Age and Calories")
plt.show()


train_w_bins = train.copy()
train_w_bins['Age Bin'] = (train_w_bins['Age']/5).astype(int)
train_w_bins['Age Bin']


sns.boxplot(data = train_w_bins, x = 'Age Bin', y = 'Calories')
plt.title("Binned Age and Calories")
plt.show()


combined_data['Height'].head(30)


sns.boxplot(data = combined_data, x = 'Source', y = 'Height')
plt.show()


sns.scatterplot(data = train, x = 'Height', y = 'Calories')


train_w_bins['Height Bin'] = train_w_bins['Height'].apply(
    lambda x: 
    1 if x < 164 else (
        2 if x < 174 else (
            3 if x < 184 else 4))
)


sns.boxplot(data = train_w_bins, x = 'Height Bin', y = 'Calories')


combined_data['Weight'].head(30)


sns.boxplot(combined_data, x= 'Source', y = 'Weight')


sns.scatterplot(data = train, x = 'Weight', y = 'Calories')


train_w_bins['Weight Bin'] = train_w_bins['Weight'].apply(
    lambda x:
        1 if x < 63 else (
            2 if x < 74 else (
                3 if x < 87 else (
                    4
                )
            )
        )
)


sns.boxplot(data = train_w_bins, x = 'Weight Bin', y = 'Calories')


combined_data['Duration'].head(50)


sns.boxplot(data = combined_data, x = 'Source', y = 'Duration')
plt.show()


sns.histplot(data = train, x = 'Duration', discrete = True)


sns.histplot(data = test, x = 'Duration', discrete = True)


sns.boxplot(data = train, x = 'Duration', y = 'Calories')


combined_data['Heart_Rate'].head(30)


sns.boxplot(data = combined_data, x = 'Source', y = 'Heart_Rate')


sns.histplot(data = train, x = 'Heart_Rate', discrete = True)


sns.histplot(data = test, x = 'Heart_Rate', discrete = True)


sns.boxplot(data = train, x = 'Heart_Rate', y = 'Calories')


train_w_bins['Heart_Rate Bin'] = (train_w_bins['Heart_Rate']/3).astype(int)


sns.boxplot(data = train_w_bins, x = 'Heart_Rate Bin', y = 'Calories')


combined_data['Body_Temp'].head(30)


sns.boxplot(data = combined_data, x = 'Source', y = 'Body_Temp')


sns.histplot(data = train, x = 'Body_Temp')


sns.histplot(data = test, x = 'Body_Temp')


sns.boxplot(data = train, x = 'Body_Temp', y = 'Calories')


train_w_bins['Body_Temp Bin'] = ((train_w_bins['Body_Temp']/0.1)/3).astype(int)


sns.boxplot(data = train_w_bins, x = 'Body_Temp Bin', y = 'Calories')


train['Calories'].head(10)


sns.boxplot(data = train, y = 'Calories')


sns.histplot(data = train, x = 'Calories',binwidth = 10)

