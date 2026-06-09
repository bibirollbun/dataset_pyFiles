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


# !git clone https://github.com/PriorLabs/tabpfn-extensions
!pip install tabpfn
# !pip install -e tabpfn-extensions <-- allegedly should have better performance but couldnt get it working on kaggle :(


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
dataset = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_dataset.csv")

dataset = (
    dataset
    .rename(columns={'Personality': 'match_p'})
    .drop_duplicates(['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
                      'Going_outside', 'Drained_after_socializing', 
                      'Friends_circle_size', 'Post_frequency'])
)

merge_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
              'Going_outside', 'Drained_after_socializing', 
              'Friends_circle_size', 'Post_frequency']

train = train.merge(dataset, how='left', on=merge_cols)
test = test.merge(dataset, how='left', on=merge_cols)

train = train.drop(columns='id')
test = test.drop(columns='id')

display(train.info(), train.head(), train.describe().T)


train['Stage_fear'] = train['Stage_fear'].str.lower().map({'yes': 1, 'no': 0})
train['Drained_after_socializing'] = train['Drained_after_socializing'].str.lower().map({'yes': 1, 'no': 0})

test['Stage_fear'] = test['Stage_fear'].str.lower().map({'yes': 1, 'no': 0})
test['Drained_after_socializing'] = test['Drained_after_socializing'].str.lower().map({'yes': 1, 'no': 0})

train['match_p'] = train['match_p'].map({'Extrovert': 0, 'Introvert': 1})
test['match_p'] = test['match_p'].map({'Extrovert': 0, 'Introvert': 1})

display(train.head(), test.head())


X = train.drop(columns=["Personality"])
y = train["Personality"]
display(X.info(), y.info())


from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = .2, random_state = 42, stratify = y)


from sklearn.metrics import accuracy_score
from tabpfn import TabPFNClassifier

# train
model = TabPFNClassifier(device = "cuda", ignore_pretraining_limits=True, random_state = 12)
model.fit(X_train, y_train)


predictions = model.predict(X_val)
print("Accuracy: ", accuracy_score(y_val, predictions))

# WITH STRATIFY = Y
# BEFORE ADDING ORIG: Accuracy:  0.9721997300944669
# AFTER ADDING ORIG (DATASERT): Accuracy:  0.9727395411605938
# AFTER ADDING ORIG (DATASET): Accuracy:  0.9746288798920378   <--- Best Result, adding both sets give same score but has redundancies
# AFTER ADDING BOTH (SET AND SERT): Accuracy:  0.9746288798920378

# WITHOUT STRATIFY
# AFTER ADDING ORIG (DATASET): Accuracy:  0.9697705802968961


test_probs = model.predict(test)


# create submission
submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
submission['Personality'] = test_probs
submission.to_csv('submission_final.csv', index=False)
submission.head()




