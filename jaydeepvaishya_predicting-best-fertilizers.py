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


df= pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
df1=pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv').sample(15)


df.head()


df.shape


df['Soil Type'].unique()


df['Crop Type'].unique()


df['Fertilizer Name'].unique()


df.isna().sum()


df.head()


from sklearn.preprocessing import OrdinalEncoder
import joblib

# Columns to encode
categorical_cols = ['Soil Type', 'Crop Type']

# Fit encoder
encoder = OrdinalEncoder()
df[categorical_cols] = encoder.fit_transform(df[categorical_cols])

# Save encoder for later use
joblib.dump(encoder, 'encoder.pkl')



df.head()


df['Soil Type'] = df['Soil Type'].astype(int)
df['Crop Type'] = df['Crop Type'].astype(int)


df.head()


df.drop('id',inplace=True,axis=1)


df.head()


np.mean(df['Temparature'])


np.mean(df['Humidity'])


import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df['Temparature'],kde=True)


X= df.drop('Fertilizer Name',axis=1)
y= df['Fertilizer Name']


from sklearn.model_selection import train_test_split as tts


# X_train,X_test,y_train,y_test= tts(X,y,test_size=0.2,random_state=42)


df['Fertilizer Name'].value_counts()



# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(n_estimators=150,max_depth=15)
# model.fit(X, y)



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y)  # y is your original Fertilizer Name column



from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

model.fit(X, y_encoded)






# joblib.dump(model, 'model.pkl')


encoder = joblib.load('encoder.pkl')
# model = joblib.load('model.pkl')

df1[categorical_cols] = encoder.transform(df1[categorical_cols])



df1.drop('id',inplace=True,axis=1)


preds = model.predict(df1)
decoded_preds = le.inverse_transform(preds)



df1=pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


submission = pd.DataFrame({
    'id': df1['id'],  # same ID column from test data
    'Fertilizer Name': decoded_preds
})


submission.to_csv('submission.csv', index=False)





