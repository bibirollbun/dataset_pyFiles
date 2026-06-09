import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


sample_submission = pd.read_csv('/kaggle/input/czii-cryo-et-object-identification/sample_submission.csv')


sample_submission.head()


sample_submission.tail()


sample_new = sample_submission.drop(['id','experiment','x','y','z'],axis=1)


sample_new.head()


Selection_level = []
for i in range(len(sample_new)):
    if sample_new.iloc[i]['particle_type'] == 'beta-amylase':
        Selection_level.append ('0')
    elif sample_new.iloc[i]['particle_type'] == 'beta-galactosidase':
        Selection_level.append ('2')
    elif sample_new.iloc[i]['particle_type'] == 'ribosome' or sample_new.iloc[i]['particle_type'] == 'apo-ferritin' or sample_new.iloc[i]['particle_type'] == 'virus-like-particle':
        Selection_level.append ('1')
sample_new['Selection_level'] = Selection_level



sample_new.head()


sns.countplot(x='particle_type', data=sample_new)
plt.xticks(rotation=45)


selection= pd.get_dummies(sample_new, prefix=['particle_type'], columns=['particle_type'])
x = selection.drop(['Selection_level'], axis=1)
y = selection['Selection_level']
y = y.astype('int')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)


selection.head()


logreg = LogisticRegression()
logreg.fit(x_train, y_train)

score = logreg.score(x_train, y_train)
score2 = logreg.score(x_test, y_test)
print("Training set accuracy",'%.3f'%(score))
print("Test set accuracy",'%.3f'%(score2))


logreg2 = SVC()  
logreg2.fit(x_train, y_train) 

score = logreg2.score(x_train, y_train)
score2 = logreg2.score(x_test, y_test)
print("Training set accuracy",'%.3f'%(score))
print("Test set accuracy",'%.3f'%(score2))

