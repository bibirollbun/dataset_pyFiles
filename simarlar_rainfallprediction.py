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


#importing the libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Exporting data
training_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
testing_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
submission_data = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


training_data.head()


training_data.isnull().sum()


training_data.duplicated().sum()


testing_data.head()


testing_data.isnull().sum()


sns.kdeplot(data=testing_data,x='winddirection',fill=True)


testing_data.winddirection.mean()


testing_data.winddirection.describe()


testing_data = testing_data.fillna(testing_data.mean())


testing_data.isnull().sum()











testing_data.duplicated().sum()


#We can see that there is no null and duplicate values


training_data.rainfall.value_counts()


training_data.info()


training_data.describe()


testing_data.describe()


#Univariet Analysis


training_data.corr()


training_data.corr().rainfall


cols_name = ['day','pressure','maxtemp','temparature','mintemp','dewpoint','humidity','cloud','sunshine','winddirection','windspeed','rainfall']


plt.figure(figsize=(12,12))
for i ,cols in enumerate(cols_name,1):
    plt.subplot(3,4,i)
    sns.boxplot(y = training_data[cols])
    plt.title(f"Boxplot of {cols}")
plt.tight_layout()
plt.show()
    


#For detecting outliers
def detect_outliers(df,cols):
    Q1 = df[cols].quantile(0.25)
    Q3 = df[cols].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)
    outliers = df.loc[(df[cols] < lower_bound) | (df[cols] > upper_bound), cols]
    return outliers
    
    


plt.figure(figsize=(20,20))
for i ,cols in enumerate(cols_name,1):
    plt.subplot(3,4,i)
    sns.histplot(x=training_data[cols],kde=True,bins=30,orientation='vertical')
    plt.title(f"Histogram of {cols}")
plt.tight_layout()
plt.show()


plt.figure(figsize=(20,20))
for i ,cols in enumerate(cols_name,1):
    plt.subplot(3,4,i)
    sns.boxplot(data=training_data,x='rainfall',y=cols)
    plt.title(f"BoxPlot of {cols} by RainFall")
    plt.legend(title='rainfall', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()


corr_matrix = training_data.corr()


plt.figure(figsize=(20,20))
for i ,cols in enumerate(cols_name,1):
    plt.subplot(3,4,i)
    sns.histplot(data=training_data, x=cols, hue='rainfall', kde=True)
    plt.title(f"KdePlot of {cols} by RainFall")
    
plt.tight_layout()
plt.show()


rainfall_corr = corr_matrix[['rainfall']]


print(rainfall_corr)


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


training_data


X = training_data.drop(columns=['id','day','rainfall'],axis=1)
y = training_data['rainfall']


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


dt = DecisionTreeClassifier()


dt.fit(x_train,y_train)


dt_pred = dt.predict(x_test)


print(f"The accuracy is {accuracy_score(y_test,dt_pred)}")


rf = RandomForestClassifier()


rf.fit(x_train,y_train)


rf_predict = rf.predict(x_test)


print(f"The accuracy of rf is {accuracy_score(y_test,rf_predict)}")


gb = GradientBoostingClassifier()


gb.fit(x_train,y_train)


gb_predict = gb.predict(x_test)


print(f"The accuracy of gb is {accuracy_score(y_test,gb_predict)}")


xg = XGBClassifier()


xg.fit(x_train,y_train)


xg_predict = xg.predict(x_test)


print(f"The accuracy of xg is {accuracy_score(y_test,xg_predict)}")


from sklearn.model_selection import GridSearchCV


gb_param_grid = {
    'n_estimators': [50, 100, 150],  
    'learning_rate': [0.01, 0.1, 0.2],  
    'max_depth': [3, 5, 7],  
    'min_samples_split': [2, 5, 10],  
    'subsample': [0.8, 1.0] 
}


xgb_param_grid = {
    'n_estimators': [50, 100, 150],  
    'learning_rate': [0.01, 0.1, 0.2],  
    'max_depth': [3, 5, 7],  
    'subsample': [0.8, 1.0],  
    'colsample_bytree': [0.8, 1.0]  
}


gb_grid_search = GridSearchCV(param_grid=gb_param_grid,estimator=gb,cv=5,n_jobs=-1,verbose=1)
gb_grid_search.fit(x_train, y_train)


print(gb_grid_search.best_params_)


best_gb_model = gb_grid_search.best_estimator_


best_gd_predict = best_gb_model.predict(x_test)


print(f"The accuracy is {accuracy_score(y_test,best_gd_predict)}")


xgb_grid_search = GridSearchCV(param_grid=xgb_param_grid,estimator=xg,cv=5,n_jobs=-1,verbose=1)
xgb_grid_search.fit(x_train, y_train)


xgb_best_model = xgb_grid_search.best_estimator_


print(xgb_grid_search.best_params_)


xgb_predictors = xgb_best_model.predict(x_test)


print(f"The accuracy for xgb is {accuracy_score(y_test,xgb_predictors)}")


gb_model = GradientBoostingClassifier(learning_rate= 0.01, max_depth= 3, min_samples_split= 2, n_estimators= 150, subsample= 0.8)


gb_model.fit(X,y)


submission_data


testing_id = testing_data['id']


testing_data.drop(columns=['id','day'],axis=1,inplace=True)


testing_id.shape


from sklearn.metrics import roc_auc_score


y_probs = gb_model.predict_proba(testing_data)[:, 1]

submission = pd.DataFrame(data=testing_id,columns=['id'])


submission['rainfall'] = y_probs


submission


submission.to_csv('submission.csv',index=False)




