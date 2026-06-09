import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/train-and-test-data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns



train_df = pd.read_csv('/kaggle/input/train-and-test-data/train_clean.csv')
test_df = pd.read_csv('/kaggle/input/train-and-test-data/test_clean.csv')

train_df.shape
test_df.shape


X = train_df.drop(['id', 'sii'], axis=1)
y = train_df['sii']

X_test = test_df.drop(['id'], axis=1)

test_ids = test_df['id'].copy()


rf = RandomForestClassifier(
    n_estimators=300,      # 樹的數量
    random_state=42,
    class_weight= 'balanced'
)




rf.fit(X, y)
y_test_pred = rf.predict(X_test)

submission = pd.DataFrame({
    'id': test_ids,
    'sii': y_test_pred.astype(int)
})


submission


submission.to_csv(os.path.join('/kaggle/working', 'submission.csv'), index=False)

