# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")

df.sample(5)


print(df.info())
print('\n', df.isna().sum())


df.nunique()


df.columns = df.columns.str.strip()
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)


df[df.select_dtypes(include='object').columns] = df.select_dtypes(include='object').apply(lambda x: x.fillna(x.mode()[0]))


df= df.fillna(df['Weight Capacity (kg)'].mean())


df.drop(columns=['id'],inplace=True)
df


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Size_encoded']=le.fit_transform(df['Size'])
df.drop(columns=['Size'],inplace=True)
df=pd.get_dummies(df, columns=['Brand','Material','Style','Color','Laptop Compartment','Waterproof'])


X=df.drop(columns=['Price'])
y=df['Price']
X


from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbr.fit(X_train, y_train)
y_pred = gbr.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse}")


tdf=pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
tdf[tdf.select_dtypes(include='object').columns] = tdf.select_dtypes(include='object').apply(lambda x: x.fillna(x.mode()[0]))
tdf= tdf.fillna(tdf['Weight Capacity (kg)'].mean())
PID=tdf['id']
tdf.drop(columns=['id'],inplace=True)
tdf['Size_encoded']=le.fit_transform(tdf['Size'])
tdf.drop(columns=['Size'],inplace=True)
tdf=pd.get_dummies(tdf, columns=['Brand','Material','Style','Color','Laptop Compartment','Waterproof'])
test_pred = gbr.predict(tdf)
test_pred


submission = pd.DataFrame({
    'id': PID,  
    'Price': test_pred   
})
submission.to_csv('submission.csv', index = False)
print('Submission file saved as submission.csv')




