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


train = pd.read_csv('/kaggle/input/analyze-the-insights-over-mental-health-data/train.csv')
test = pd.read_csv('/kaggle/input/analyze-the-insights-over-mental-health-data/test.csv')


import sklearn
print(sklearn.__version__)


import category_encoders as ce


bachelor_degrees = ['BHM', 'LLB', 'B.Pharm', 'BBA', 'BCA', 'BE', 'B.Ed', 'B.Com', 'BA', 'B.Tech', 'B.Arch', 'BSc', 'B.Sc', 'BPharm', 'BBA', 'BPA', 'BB', 'B B.Com', 'B BA', 'B.Student']
master_degrees = ['MCA', 'ME', 'MA', 'MBA', 'M.Com', 'MHM', 'M.Tech', 'M.Ed', 'MSc', 'LLM', 'M.Pharm', 'MPA', 'M.Arch', 'MPharm', 'MTech', 'MEd', 'M.S', 'M_Tech', 'M. Business Analyst']
phd_degrees = ['PhD', 'MD', 'MBBS']

def replace_degree(degree):
    if degree in bachelor_degrees:
        return 'Bachelor'
    elif degree in master_degrees:
        return 'Master'
    elif degree in phd_degrees:
        return 'PhD'
    else:
        return np.nan 


train['Degree'] = train['Degree'].apply(replace_degree)


#train['Degree'].fillna(train['Degree'].mode()[0],inplace=True)


train['Degree'].value_counts()


x_train = train.copy()
x_test = test.copy()


x_train = x_train.ffill()
x_test = x_test.ffill()


x_train = x_train.bfill()
x_test = x_test.bfill()


for col in x_train.select_dtypes(include=['object']).columns:
    encoder = ce.TargetEncoder()
    x_train[col] = encoder.fit_transform(x_train[col],train['Depression'])
    x_test[col] = encoder.transform(x_test[col])
    if col in ['Working Professional or Student','Profession','Degree','Have you ever had suicidal thoughts ?']:
        df = pd.DataFrame({col : train[col].unique()})
        df['encoded'] = encoder.transform(df[col])
        df.to_csv(f"{col}.csv")


x_train.corr()['Depression']


x_train = x_train[['Age','Working Professional or Student','Profession','Work Pressure','Degree','Job Satisfaction','Have you ever had suicidal thoughts ?','Work/Study Hours','Financial Stress']] #,'Academic Pressure' ,'Study Satisfaction' , 'degree'
y_train = train['Depression']
x_test = x_test[['Age','Working Professional or Student','Profession','Work Pressure','Degree','Job Satisfaction','Have you ever had suicidal thoughts ?','Work/Study Hours','Financial Stress']] 


x_train['Financial Stress'].unique()


from sklearn.ensemble import RandomForestClassifier


model = RandomForestClassifier().fit(x_train,y_train) #class_weight = { 0 : 1 , 1 : 4}


model.score(x_train,y_train)


predicted = model.predict(x_test)


df = pd.DataFrame({
    "id" : test['id'],
    "Depression" : predicted})


df.to_csv('output.csv',index = False)


import joblib


joblib_file = "random_forest_model.pkl"
joblib.dump(model, joblib_file)

