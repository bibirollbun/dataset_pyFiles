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


dataset=pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")


dataset.head()


dataset.info()


dataset.describe()


dataset.columns


print("No.of rows : ",dataset.shape[0])


print("No.of cols : ",dataset.shape[1])


dataset.isnull().sum()


dataset=dataset.dropna(how='any')


dataset["Misconception"].isnull().sum()


X=dataset.drop("Misconception",axis=1)


Y=dataset["Misconception"]


from sklearn.preprocessing import LabelEncoder

le_mc_answer = LabelEncoder()
dataset['MC_Answer'] = le_mc_answer.fit_transform(dataset['MC_Answer'])

le_category = LabelEncoder()
dataset['Category'] = le_category.fit_transform(dataset['Category'])

le_miscon = LabelEncoder()
dataset['Misconception'] = le_miscon.fit_transform(dataset['Misconception'])



from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_q = TfidfVectorizer(max_features=300)
q_vec = tfidf_q.fit_transform(dataset['QuestionText'].fillna("")).toarray()

tfidf_e = TfidfVectorizer(max_features=500)
e_vec = tfidf_e.fit_transform(dataset['StudentExplanation'].fillna("")).toarray()


import numpy as np

X = np.concatenate([q_vec, e_vec, 
                    dataset[['MC_Answer', 'Category']].values], axis=1)

Y = dataset['Misconception'].values


X.shape,Y.shape


from sklearn.model_selection import train_test_split


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)


X_train.shape,X_test.shape,Y_train.shape,Y_test.shape


from sklearn.svm import SVC


model=SVC()


dataset.head()


model.fit(X_train,Y_train)


prediction=model.predict(X_test)


from sklearn.metrics import accuracy_score,classification_report


print("Accuracy score : ",accuracy_score(Y_test,prediction))


print("Classification report : ",classification_report(Y_test,prediction))


































