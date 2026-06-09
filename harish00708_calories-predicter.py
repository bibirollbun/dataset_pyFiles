import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df.head()


df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
corr = df.corr()


plt.Figure(figsize=(8,8))
sns.heatmap(corr,cmap='Blues',annot=True, fmt='.2f')
plt.title('Correlation')
plt.show()


X = df.drop(['id','Height','Calories'],axis=1)
y = df['Calories']


pipe = Pipeline([
    ('Scaler', StandardScaler()),
    ('linear_regression', DecisionTreeRegressor())
])


X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=5)


pipe.fit(X_train,y_train)


y_predict = pipe.predict(X_test)

r2_Scor = r2_score(y_test,y_predict)
mean_error = mean_absolute_error(y_test,y_predict)
print(f'r2_score is {r2_Scor} and Mean Error Score is {mean_error}')


df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
df_test.shape


df_test['Sex'] = df_test['Sex'].apply(lambda x: 1 if x == 'male' else 0)
X = df_test.drop(['id','Height'],axis=1)


df_test_prediction = pipe.predict(X)


csv = {'id':df_test['id'],'Calories':df_test_prediction}
csv = pd.DataFrame(csv)
csv.head()


csv.to_csv('output.csv' , index = False)

