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


sample_code=pd.read_csv("/kaggle/input/pascmlsig/sample_submission.csv")
sample_code.head()


train_data=pd.read_csv("/kaggle/input/pascmlsig/train.csv")
train_data.head()


test_data=pd.read_csv("/kaggle/input/pascmlsig/test.csv")
test_data.head()


train_data.isnull().sum()


x=train_data.drop(columns=["yield"])
y=train_data["yield"]
x.shape
y.shape


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


print(x_train.shape)
print(y_train.shape)


test_data_id=test_data["id"]


from sklearn.preprocessing import StandardScaler as ss
model0=ss()
x_train=model0.fit_transform(x_train)
x_test=model0.fit_transform(x_test)
test_data=model0.fit_transform(test_data)


from sklearn.linear_model import LinearRegression as lr
model=lr()
model.fit(x_train,y_train)


model.score(x_train,y_train).round(4)


model.score(x_test,y_test)


y_prd=model.predict(x_test)


import matplotlib.pyplot as plt
plt.scatter(train_data["seeds"],train_data["yield"])
plt.show()


arr=np.abs(model.coef_)
np.sort(arr)


model.intercept_


output=model.predict(test_data)
output


submission_df = pd.DataFrame({
    'id': test_data_id, # Use the 'id' column extracted from the original test_data_for_submission
    'yield': output
})


submission_df


submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully!")

