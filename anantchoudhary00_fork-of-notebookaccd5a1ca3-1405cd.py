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


TRAIN_PATH = r"/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT.csv"
TEST_PATH  = r"/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT_UNK.csv"


import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt
import seaborn as sns



df=pd.read_csv(TRAIN_PATH)
df1=pd.read_csv(TEST_PATH)
df.head()


df1.info()


df.info()


df.describe()


df.isnull().mean()*100


numericalcols= df.select_dtypes(['int64','float64'])
categoricalcols= df.select_dtypes(['object'])



dfclean = df.dropna(subset=['CORRUCYSTIC_DENSITY'])
x= dfclean.drop('CORRUCYSTIC_DENSITY', axis=1)
y = dfclean['CORRUCYSTIC_DENSITY']



from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=2)


xtrain.head()


plt.figure(figsize=(20, 15))
sns.heatmap(numericalcols.corr(),cmap='YlGnBu')


numericalcols.corr()


sns.boxplot(df['CORRUCYSTIC_DENSITY'])


sns.kdeplot(df['CORRUCYSTIC_DENSITY'])


sns.kdeplot(df['CORRUCYSTIC_DENSITY'])


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor  
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV


numericfeatures = [col for col in xtrain.columns if xtrain[col].dtype in ['int64', 'float64']]
categoricfeatures = [col for col in xtrain.columns if xtrain[col].dtype == 'object']


numerical_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')), 
    ('scale', StandardScaler())                
])
categorical_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')), 
    ('onehotencoding', OneHotEncoder(handle_unknown='ignore'))   
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numericfeatures), 
        ('cat', categorical_transformer, categoricfeatures)
    ],
    remainder='passthrough' 
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor), 
    ('regressor', RandomForestRegressor(random_state=42)) 
])
                               


param_grid = {
   
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [3, 5, 9, None],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4],
    'regressor__max_features': ['sqrt', 'log2', 1.0],
    'regressor__bootstrap': [True, False],
    'regressor__max_samples': [0.5, 0.7, 0.9, 1.0],
    'regressor__oob_score': [True, False]
}



random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=20, 
    cv=5, 
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42)


random_search.fit(x,y)


X_unk=df1
y_unk_pred = random_search.predict(X_unk)

# Actually creating the submission CSV file
submission = pd.DataFrame({
    'LOCAL_IDENTIFIER': df1['LOCAL_IDENTIFIER'],
    'CORRUCYSTIC_DENSITY': y_unk_pred
})

submission['LOCAL_IDENTIFIER'] = submission['LOCAL_IDENTIFIER'].astype(int)
submission['CORRUCYSTIC_DENSITY'] = submission['CORRUCYSTIC_DENSITY'].astype(float)

submission.to_csv('submission.csv', index=False)
print(submission.head(10))









































