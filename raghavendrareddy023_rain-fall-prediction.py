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


data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
data.head()


test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
test.head()


data.info()


test.info()


data.describe().T


data.isnull().sum()


test.isnull().sum()


from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(strategy = 'mean')
test = pd.DataFrame(imputer.fit_transform(test), columns = test.columns)
test


data.duplicated().sum()


check = []
for i in data.columns:
    check.append(data[i].unique())
check


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', ConvergenceWarning)


for i in data.columns:
    sns.distplot(data[i], kde=True)
    plt.title(i)
    plt.show()


for i in data.columns:
    skewness = data[i].skew()
    if data[i].nunique() > 15 :
        sns.distplot(data[i], bins=30, kde=True)
        plt.title(f'{i} skewness is  {skewness:.2f}')
        plt.axvline(data[i].mean(), color='green', linestyle='dashed', label='Mean')
        plt.axvline(data[i].median(), color='red', linestyle=':', label='Median')
        plt.show()


data


plt.figure(figsize=(12, 10))
sns.heatmap(data.corr()*100, annot=True, cmap='Blues')
plt.show()


for i in data.columns:
    sns.boxplot(data[i])
    plt.title(i)
    plt.show()


from scipy.stats import zscore

z_scores = data.drop('rainfall', axis=1).apply(zscore)
outliers = (z_scores.abs() > 3).sum()
outliers


data


# for i in data.columns:
#     Q1 = data[i].quantile(0.25)
#     Q3 = data[i].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - (1.5 * IQR)
#     upper_bound = Q3 + (1.5 * IQR)
#     data = data[(data[i] >= lower_bound) & (data[i] <= upper_bound)]


data['rainfall'].value_counts()


from sklearn.preprocessing import MinMaxScaler



X = data.drop(columns=['id', 'rainfall'], axis=1)
test_data = test.drop('id', axis=1)
Y = data['rainfall']


test_data.head()


ms = MinMaxScaler()
scaled_data = ms.fit_transform(X)
scaled_test = ms.transform(test_data)
# pd.DataFrame(scaled_test, columns = X.columns)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# x_train, x_test, y_train, y_test = train_test_split(scaled_data, Y, test_size=0.2, random_state=23, stratify = Y)


# x_train


# y_train


model = LogisticRegression()
model.fit(scaled_data, Y)


predict = model.predict(scaled_test)
predict


pred_prob = model.predict_proba(scaled_test)[:, 1]
pred_prob


ids = np.arange(2190, 2190 + len(pred_prob))
ids


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# print(accuracy_score(predict))
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, scaled_data, Y, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy:", cv_scores.mean())  # Average accuracy



print(accuracy_score(Y, model.predict(scaled_data)))


result = pd.DataFrame({"id": ids,"rainfall": pred_prob})
result


result.to_csv('sample_submission.csv', index=False)


sub = pd.read_csv('sample_submission.csv')
sub


pd.read_csv('/kaggle/working/sample_submission.csv')

