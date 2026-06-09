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


import zipfile
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def load_json_from_zip(zip_path, json_filename):
    with zipfile.ZipFile(zip_path,'r') as z:
        with z.open(json_filename,'r') as f:
            return json.load(f)


train_data= load_json_from_zip('/kaggle/input/whats-cooking/train.json.zip', "train.json")
test_data= load_json_from_zip('/kaggle/input/whats-cooking/test.json.zip', "test.json")


train_df= pd.DataFrame(train_data)
test_df= pd.DataFrame(test_data)

train_df['ingredients_str'] = train_df['ingredients'].apply(lambda x: ' '.join(x))
test_df['ingredients_str'] = test_df['ingredients'].apply(lambda x: ' '.join(x))


label_encoder= LabelEncoder()
y= label_encoder.fit_transform(train_df['cuisine'])


train_df.head()


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Naive Bayes': MultinomialNB()
}

best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    score = accuracy_score(y_val, y_pred)
    print(f"{name}: {score:.4f}")
    
    if score > best_score:
        best_score = score
        best_model = model


print(best_model)


test_predictions = best_model.predict(X_test)
test_predictions_cuisine = label_encoder.inverse_transform(test_predictions)


submission = pd.DataFrame({
    'id': test_df['id'],
    'cuisine': test_predictions_cuisine
})

submission.to_csv('submission.csv', index=False)




