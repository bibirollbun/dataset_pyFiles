#dgut150106
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


import pandas as pd


import pandas as pd

train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')



train.head()


test.head()





X = train.drop(columns = ['id','Fertilizer Name'])
y = train['Fertilizer Name']


X = train.drop(columns=['id', 'Fertilizer Name'])
y = train['Fertilizer Name']

X_test = test.drop(columns=['id'])




from sklearn.preprocessing import LabelEncoder

encoder_soil = LabelEncoder()
encoder_crop = LabelEncoder()
encoder_fertilizer = LabelEncoder()




X['Soil Type'] = encoder_soil.fit_transform(X['Soil Type'])
X['Crop Type'] = encoder_crop.fit_transform(X['Crop Type'])

X.head()


y_encode = encoder_fertilizer.fit_transform(y)


y_encode[:5]

y.head()



X_test['Soil Type'] = encoder_soil.fit_transform(X_test['Soil Type'])
X_test['Crop Type'] = encoder_crop.fit_transform(X_test['Crop Type'])


X_test.head()


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y_encode, test_size=0.2, random_state=42)




from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)





y_probs = model.predict_proba(X_test)


y_pred_encoded = model.predict(X_test)


y_pred_labels = encoder_fertilizer.inverse_transform(y_pred_encoded)

print(y_probs[:5])
print(y_pred_labels[:5])


import numpy as np

top3 = np.argsort(y_probs, axis=1)[:, -3:][:, ::-1]


top3_labels_list = [encoder_fertilizer.inverse_transform(row) for row in top3]

submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': [' '.join(pred_list) for pred_list in top3_labels_list]
})

submission.to_csv("submission.csv", index=False)

print("Submission file created successfully!")
print(submission.head())






















