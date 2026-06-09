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


df=pd.read_csv('/kaggle/input/human-vs-ai-text-classification-feb2024/train.csv')
df.head()


df.info()


df.duplicated().sum()


df.generated.value_counts().plot.bar()


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
x=df.text
y=df.generated
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=44)
text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', RandomForestClassifier(n_estimators=100,max_depth=50)),])
text_clf.fit(x_train, y_train)  


predictions = text_clf.predict(x_test)


print(f'Score For Training {text_clf.score(x_train,y_train)}')
print(f'Score For Testing {text_clf.score(x_test,y_test)}')
print(f'Accuracy {accuracy_score(y_test,predictions)}')



dic_ac={'s_train':text_clf.score(x_train,y_train).round(2),'s_test':text_clf.score(x_test,y_test).round(2),'accuracy':accuracy_score(y_test,predictions).round(2)}
pd.Series(dic_ac).plot()


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


import matplotlib.pyplot as plt
from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test, predictions)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])

cm_display.plot()
plt.show()


test=pd.read_csv(r'/kaggle/input/human-vs-ai-text-classification-feb2024/test.csv')
test_pred=text_clf.predict(test.text)
#submission = pd.DataFrame({'Id': test['Id'], 'generated': test_pred})
#submission.to_csv('submission.csv', index=False)


submission = pd.DataFrame({'Id': test['Id'], 'generated': test_pred})
submission.to_csv('submission.csv', index=False)


import re
from gensim.parsing.preprocessing import remove_stopwords
df_analysis=df.copy()
df_analysis['NOfWords']=df_analysis.text.apply(lambda x:len(x.split()))
df_analysis['NOfCharacters']=df_analysis.text.apply(lambda x:len(x))
df_analysis['CountsOfNumbers']=df_analysis.text.apply(lambda x:len(re.findall(r'\d',x)))
df_analysis['NOfSCharacters']=df_analysis.text.apply(lambda x:re.search(r'["!@#$%^&*(){}[]:?]',x))
df_analysis['NOfEmails']=df_analysis.text.apply(lambda x:len(re.findall(r'[\w\.-]+@[\w\.-]+', x)))
df_analysis['NOfWUnique']=df_analysis.text.apply(lambda x:len(set(x)))
df_analysis['AfterRemoveStopWords']=df_analysis.text.apply(lambda x:remove_stopwords(x))

df_analysis.head()




