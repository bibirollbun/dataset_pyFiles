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
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')


df=pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
df


test=pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


#Training dataset
df=pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df


df.head()


df.info()


df.nunique()


df.value_counts().sum()


#To check whether it conatins null values or not
df.isnull().sum()


df.duplicated()


df.head(2)


df.groupby('Sex')['Duration'].sum()


Sex=['female','male']
Duration=[5824056.0,5741705.0]
plt.figure(figsize=(10,5))
plt.bar(Sex,Duration,color=['skyblue', 'salmon'],width=0.4)
plt.xlabel=('Sex')
plt.ylablel=('Caloroes Burned')
plt.title('Calories Burn')
plt.show()



x = df.drop(['Calories'], axis=1)  # Features
y = df['Calories']   


combined = pd.concat([x, test], axis=0)
combined_encoded = pd.get_dummies(combined, drop_first=True)
x_encoded = combined_encoded.iloc[:len(df), :]
test_encoded = combined_encoded.iloc[len(df):, :]


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_encoded, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)


val_predictions = model.predict(x_val)


val_predictions = np.maximum(val_predictions, 0)
y_val = np.maximum(y_val, 0)


from sklearn.metrics import mean_squared_log_error 
rmsle = np.sqrt(mean_squared_log_error(y_val, val_predictions))
print(f"RMSLE on validation set: {rmsle:.5f}")


model.fit(x_encoded, y)


predictions = model.predict(test_encoded)
predictions = np.maximum(predictions, 0) 


submission = pd.DataFrame({
    'id': range(750000, 750000 + len(test)),
    'Calories': predictions
})


submission.to_csv('submission.csv', index=False)
print("✅ Submission file saved as 'submission.csv'")

