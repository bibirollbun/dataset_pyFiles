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


#%load_ext cudf.pandas 



train_path = "/kaggle/input/playground-series-s5e9/train.csv"
test_path = "/kaggle/input/playground-series-s5e9/test.csv"
df_train=pd.read_csv(train_path)
df_test=pd.read_csv(test_path)
df_test


df_train.head()


df_train.isnull().sum()


df_train.fillna(df_train.mean(),inplace=True)


df_train.isnull().sum()


df_train.info()


import seaborn as sns
import matplotlib.pyplot as plt

# Create a small DataFrame
data = {
    "Type": ["Min", "Max"],
    "BeatsPerMinute": [df_train["BeatsPerMinute"].min(), df_train["BeatsPerMinute"].max()]
}

df_plot = pd.DataFrame(data)

# Plot
sns.barplot(x="Type", y="BeatsPerMinute", data=df_plot)
plt.title("Min and Max of BeatsPerMinute")
plt.show()



from sklearn.model_selection import train_test_split
x=df_train.drop(["id","BeatsPerMinute"],axis=1)
y=df_train["BeatsPerMinute"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
model.fit(x_train,y_train)
pred=model.predict(x_test)


from sklearn.metrics import r2_score
r2_score(y_test,pred)
print(r2_score(y_test,pred))


test_data=df_test.drop(["id"],axis=1)
df_test


predict_test=model.predict(test_data)


df=pd.DataFrame({"id":df_test["id"],"BeatsPerMinute":predict_test})


df.to_csv("submission.csv",index=False)


from xgboost import XGBRegressor

xgbr=XGBRegressor(n_estimators=100,learning_rate=0.01)

xgbr.fit(x_train,y_train)
pred2=xgbr.predict(x_test)

from sklearn.metrics import r2_score
r2_score(y_test,pred)
print(r2_score(y_test,pred2))



predict_test=xgbr.predict(test_data)


df2=pd.DataFrame({"id":df_test["id"],"BeatsPerMinute":predict_test})


df2.to_csv("submission2.csv",index=False)
df2




