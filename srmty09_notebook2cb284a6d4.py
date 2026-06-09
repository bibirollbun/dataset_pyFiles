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


df=pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df.info()


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
sns.regplot(
    data=df.sample(7500),
    x="Episode_Length_minutes",
    y="Listening_Time_minutes",
    scatter_kws={"s": 5, "alpha": 0.3}
)
plt.xlim(-5, 120)   
plt.ylim(-5, 120)   

plt.title("Listening_Time_minutes versus Episode_Length_minutes")
plt.xlabel("Episode_Length_minutes")
plt.ylabel("Listening_Time_minutes")
plt.show()



import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import math




mask1 = df["Episode_Length_minutes"].notna()
X1 = df.loc[mask1, "Listening_Time_minutes"].values.reshape(-1, 1)  
y1 = df.loc[mask1, "Episode_Length_minutes"].values 
rfr1 = RandomForestRegressor(n_estimators=100, random_state=42)
rfr1.fit(X1, y1)


mask2 = df["Episode_Length_minutes"].isna()
X2 = df.loc[mask2, "Listening_Time_minutes"].values.reshape(-1, 1)  

y_pred2 = rfr1.predict(X2)  
df.loc[mask2, "Episode_Length_minutes"] = y_pred2


test_data=pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


test_data["Episode_Length_minutes"].fillna(test_data["Episode_Length_minutes"].median(),inplace=True)


a = df[["Episode_Length_minutes","Genre","Number_of_Ads","Episode_Sentiment"]]
a = pd.concat([a, pd.get_dummies(a[["Genre","Episode_Sentiment"]]).astype(int)], axis=1)
a = a.drop(["Genre","Episode_Sentiment"], axis=1)



a["Number_of_Ads"].fillna(1,inplace=True)


X=a
y=df["Listening_Time_minutes"]
rfr = RandomForestRegressor(n_estimators=100, random_state=42)
rfr.fit(X,y)


b = test_data[["Episode_Length_minutes","Genre","Number_of_Ads","Episode_Sentiment"]]
b = pd.concat([b, pd.get_dummies(b[["Genre","Episode_Sentiment"]]).astype(int)], axis=1)
b = b.drop(["Genre","Episode_Sentiment"], axis=1)


y_prediction=rfr.predict(b)


sub=pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
sub["Listening_Time_minutes"]=y_prediction


sub.to_csv("submission_three.csv",index=False)




