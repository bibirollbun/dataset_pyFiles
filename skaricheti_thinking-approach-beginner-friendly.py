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


import pandas as pd #as working with structured data
import numpy as np #for numerical Computations
import matplotlib.pyplot as plt #for data visualisation
import warnings as W #read the markdown
W.filterwarnings('ignore') #to ignore warning of the -inf to inf
traind=pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv') #loaded train data
testd=pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv') #loaded test data
id=testd['id'] #I didn't use this in beginning of the code but while submitting it will be usefull to make a datafram for submission which resmbeles with sample submission
traind.head() #used to view the first five rows of the train data I also viewed test data as well but removed after seeing 


traind.info() # it give non-null-count and datatype of the variables
print("\n") 
testd.info()


cases=[traind,testd] #I just used loop here lol its unecessary, you can simply print actually like print(traind.describe())
for case in cases:
    print(case.describe())
    print('\n')
    


train_coloumn=['Episode_Length_minutes','Guest_Popularity_percentage','Number_of_Ads']
for col in train_coloumn:
    traind[col]=traind[col].fillna(traind[col].median()) #replaced with the median
traind.info()
test_coloumn=['Episode_Length_minutes','Guest_Popularity_percentage']
for col1 in test_coloumn:
    testd[col1]=testd[col1].fillna(testd[col1].median())
testd.info()


#remove unecessary coloumns
cols=['id','Podcast_Name']
for col in cols:
    traind=traind.drop(col, axis=1)
    testd=testd.drop(col, axis=1)


import seaborn as sns #I forgot to import this before seaborn is used to make different types of visuals

cols = ['Episode_Length_minutes', 'Number_of_Ads']

for col in cols:
    plt.figure() #used to customize the figure of the visual
    sns.boxplot(x=traind[col])
    plt.title(col.replace('_', ' '))
    
plt.show()


cols=['Episode_Length_minutes', 'Number_of_Ads']
for col in cols:
    plt.figure()
    sns.histplot(traind[col])
plt.show()


train_coloumn=['Episode_Length_minutes','Number_of_Ads']
for col in train_coloumn:
    low,high=traind[col].quantile([0.01,0.99]) #giving limits of the low and high with in the quantile
    traind[col]=traind[col].clip(lower=low,upper=high) #making clip if some thing is lower than low it makes the lower value to low and vice versa
traind.describe()


test_cols=['Episode_Length_minutes','Number_of_Ads']
for col in test_cols:
    low,high=testd[col].quantile([0.01,0.99]) #giving limits of the low and high with in the quantile
    testd[col]=testd[col].clip(lower=low, upper=high) #making clip if some thing is lower than low it makes the lower value to low and vice versa
testd.describe()


traind.head()


#feature Engineering
from sklearn.preprocessing import LabelEncoder
cattrain=['Episode_Title','Genre','Publication_Day','Publication_Time','Number_of_Ads','Episode_Sentiment']
le=LabelEncoder()
for col in cattrain:
        traind[col]=le.fit_transform(traind[col])
        testd[col]=le.fit_transform(testd[col])
traind.head()
testd.head()


#Making both the guest and host popularity as combine rating out of 10
traind['Rating_GH']=(((traind['Host_Popularity_percentage']/100+traind['Guest_Popularity_percentage']/100)/2)*10).round(0).astype(int)
testd['Rating_GH']=(((testd['Host_Popularity_percentage']/100+testd['Guest_Popularity_percentage']/100)/2)*10).round(0).astype(int)
traind.head()



#Splited the data with the help of train_test_split with the split of 80/20
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

x=traind.drop(columns=['Listening_Time_minutes'])
y=traind['Listening_Time_minutes']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


from lightgbm import LGBMRegressor
model=LGBMRegressor(random_state=42)
model.fit(x_train,y_train) #trained the data
ypred=model.predict(x_test) #tested the data


from sklearn.metrics import mean_squared_error
rmse=np.sqrt(mean_squared_error(y_test,ypred))
print("rmse:",rmse)


pred=model.predict(testd) #predicting with actual testdata
submission=pd.DataFrame({
    'id':id,
    'Listening_Time_minutes': pred
})
submission.to_csv('submission.csv',index=False)

