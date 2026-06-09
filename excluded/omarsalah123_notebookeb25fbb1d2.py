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


train_Data = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
test_Data = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")


train_Data.head()



pd.isnull(train_Data)


df= train_Data.dropna()


np.sum(pd.isnull(df))


df.describe(include="all")
#df["cyto_score"].unique()


target = df['efs']
target


train = df.drop(columns=['ID','efs','efs_time'])
train


test_Data


test_target = test_Data.drop(columns=['ID'])
test_target


train


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(train)
encoded_train = encoder.transform(train)
encoded_test = encoder.transform(test_target)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=8)
model.fit(encoded_train,target)


prediction = model.predict_proba(encoded_test)


prediction


test_Data


ids = test_Data['ID']
ids


output = pd.DataFrame({'ID':ids, 'prediction': prediction[:,1]})


print(output)


output.to_csv('submission.csv',index=False)




