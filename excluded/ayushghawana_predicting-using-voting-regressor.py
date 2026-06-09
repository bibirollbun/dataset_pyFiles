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


train_df=pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sample_submission=pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


train_df.head()


train_df.isnull().sum()


dataframes = [train_df,test_df]


for df in dataframes:
    df['Episode_Length_minutes']=df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].median())
    df['Guest_Popularity_percentage']=df['Guest_Popularity_percentage'].fillna(df['Guest_Popularity_percentage'].median())
    df['Number_of_Ads']=df['Number_of_Ads'].fillna(df['Number_of_Ads'].median())


train_df.isnull().sum()


test_df.isnull().sum()


train_df['Podcast_Name'].value_counts()


import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(x=train_df['Publication_Day'], y=train_df['Listening_Time_minutes'], data=train_df)
plt.title("Listening Time by Day of Week")


plt.figure(figsize=(15,10))
sns.boxplot(x=train_df.Publication_Time,y=train_df.Listening_Time_minutes,data=train_df)



train_df.info()


cols_to_drop=['Episode_Title','Publication_Day',"Publication_Time"]
train_df=train_df.drop(columns=cols_to_drop)
test_df=test_df.drop(columns=cols_to_drop)


test_df.info()


train_df.head()


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x='Genre', y='Listening_Time_minutes', data=train_df)
plt.xticks(rotation=45)
plt.show()


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from category_encoders import TargetEncoder


train_df.info()


train_df['popularity avg'] = (train_df['Host_Popularity_percentage']+train_df['Guest_Popularity_percentage'])/2
train_df


test_df['popularity avg'] = (test_df['Host_Popularity_percentage']+test_df['Guest_Popularity_percentage'])/2


X=train_df.drop(columns=['Listening_Time_minutes'])
X


y=train_df['Listening_Time_minutes']
y


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


y_train


X_train


X_test['Episode_Sentiment'].value_counts()


te = TargetEncoder()
te.fit(X_train['Podcast_Name'],y_train)
X_train['Podcast_Name_enc'] = te.transform(X_train['Podcast_Name'])
X_test['Podcast_Name_enc'] = te.transform(X_test['Podcast_Name'])
test_df['Podcast_Name_enc'] = te.transform(test_df['Podcast_Name'])


le = LabelEncoder()
X_train['Episode_Sentiment']=le.fit_transform(X_train['Episode_Sentiment'])
X_test['Episode_Sentiment']=le.transform(X_test['Episode_Sentiment'])
test_df['Episode_Sentiment']=le.transform(test_df['Episode_Sentiment'])



X_train.isnull().sum()


encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_train_genre = encoder.fit_transform(X_train[['Genre']])
X_train_genre_df = pd.DataFrame(X_train_genre, columns=encoder.get_feature_names_out(['Genre']), index=X_train.index)
X_train = pd.concat([X_train, X_train_genre_df], axis=1)
X_train.isnull().sum()


X_test_genre = encoder.transform(X_test[['Genre']])
X_test_genre_df = pd.DataFrame(X_test_genre, columns=encoder.get_feature_names_out(['Genre']), index=X_test.index)
X_test = pd.concat([X_test, X_test_genre_df], axis=1)
X_test


test_genre_df = encoder.transform(test_df[['Genre']])
test_genre_df_1 = pd.DataFrame(test_genre_df,columns=encoder.get_feature_names_out(['Genre']),index=test_df.index)
test_df=pd.concat([test_df,test_genre_df_1],axis=1)
test_df


def feature_1(dataset):
    for genre_col in dataset.columns:
        if genre_col.startswith("Genre_"):
            dataset[f'{genre_col} x popularity avg'] = dataset[genre_col]*dataset['popularity avg']


feature_1(X_train)
feature_1(X_test)
feature_1(test_df)


X_train


cols_to_drop_1 = ['Podcast_Name','Genre','Genre_Business','Genre_Comedy','Genre_Education','Genre_Health','Genre_Lifestyle','Genre_Music','Genre_News','Genre_Sports','Genre_Technology','Genre_True Crime']
X_train=X_train.drop(columns=cols_to_drop_1)
X_test=X_test.drop(columns=cols_to_drop_1)
test_df=test_df.drop(columns=cols_to_drop_1)


X_train.info()


from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor 
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error


#models = {
    #'RF': RandomForestRegressor(),
    #'XG': xgb.XGBRegressor(),
    #'AD': AdaBoostRegressor()
#}


#for model_name,model in models.items():
    #print({model_name})
    #model.fit(X_train,y_train)
    #y_pred=model.predict(X_test)
    #mse=mean_squared_error(y_test,y_pred)
    #print(np.sqrt(mse))


test_df


#rf_params= {'max_depth':[5,8,None,10],
            #'max_features': [5,7,8,20],
            #'min_samples_split':[2,8,15,20],
            #'n_estimators': [100, 200,500]}
#xgb_params = {
    #'n_estimators': [100, 200, 500],
    #'max_depth': [3, 5, 7,10],
    #'learning_rate': [0.01, 0.1, 0.2],
    #'min_child_weight': [1,3, 5,4,10],
    #'subsample': [0.6, 0.8, 1.0],
    #'colsample_bytree': [0.6, 0.8, 1.0],
    #'reg_alpha': [0, 0.1,0.5,0.8,1],
    #'reg_lambda': [1, 1.5, 2,4],
    #'gamma':[0, 0.1, 0.5, 1, 2, 5]
#}
#dt_params = {
    #'criterion': ['squared_error', 'friedman_mse', 'absolute_error'], 
    #'splitter': ['best', 'random'],  
    #'max_depth': [2, 5, 10, 20, 30, 40, 50], 
    #'min_samples_split': [2, 5, 10, 15, 20],  
    #'min_samples_leaf': [1, 2, 4, 6, 8],  
    #'max_features': ['auto', 'sqrt', 'log2', None],  
    #'random_state': [42] 
#}
#ad_params = {
    #'n_estimators': [50, 100, 150, 200],  
    #'learning_rate': [0.01, 0.1, 0.5, 1.0, 1.5],  
    #'loss': ['linear', 'square', 'exponential'],  
    #'estimator': [DecisionTreeRegressor(max_depth=2), DecisionTreeRegressor(max_depth=3)]  
#}



#models = [
    #('XGB',xgb.XGBRegressor(),xgb_params),
    #('AD',AdaBoostRegressor(),ad_params)
#]


#from sklearn.model_selection import RandomizedSearchCV


#model_params={}
#best_model={}
#for model_name,model,parameters in models:
    #print(model_name)
    #random = RandomizedSearchCV(estimator=model,param_distributions=parameters,n_iter=80,n_jobs=-1,verbose=3,random_state=42,cv=3)
    #random.fit(X_train,y_train)
    #y_pred_new = random.best_estimator_.predict(X_test)
    #model_params[model_name]=random.best_params_
    #best_model[model_name]=random.best_estimator_
    #mse=mean_squared_error(y_test,y_pred_new)
    #print(np.sqrt(mse))


#for i in model_params:
    #print(model_params[i])
    #print('-------------------------')
#for j in best_model:
    #print(best_model[j])
    #print('-------------------------')


model1 = xgb.XGBRegressor(
    subsample= 1.0, 
    reg_lambda= 1, 
    reg_alpha=0.1, 
    n_estimators= 400, 
    min_child_weight= 3, 
    max_depth= 10, 
    learning_rate= 0.1, 
    gamma= 1, 
    colsample_bytree=0.8)
model2 = RandomForestRegressor()
model3 = xgb.XGBRegressor(
    subsample= 1.0, 
    reg_lambda= 1, 
    reg_alpha=0.1, 
    n_estimators= 300, 
    min_child_weight= 3, 
    max_depth= 10, 
    learning_rate= 0.1, 
    gamma= 1, 
    colsample_bytree=0.8)
model4 = xgb.XGBRegressor(
    n_estimators=250,
    learning_rate=0.09,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.85,
    reg_alpha=0.4,
    reg_lambda=6,
    min_child_weight=3,
    random_state=42)

model5 = xgb.XGBRegressor(n_estimators=300,
    learning_rate=0.04,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.5,
    reg_alpha=0.4,
    reg_lambda=6,
    min_child_weight=3,
    random_state=42)

model6 = xgb.XGBRegressor(n_estimators=220,
    learning_rate=0.02,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_alpha=0.4,
    reg_lambda=4,
    min_child_weight=3,
    random_state=42)

estimators=[
        ('xgb1', model1),
        ('RF', model2),
        ('xgb2', model3),
        ('xgb3', model4),
        ('xgb4',model5),
        ('xgb5',model6)
    ]


for model_name,model in estimators:
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    mse=mean_squared_error(y_test,y_pred)
    print(f'{model_name} rmse :{np.sqrt(mse)}')


from sklearn.ensemble import VotingRegressor
voting_regressor = VotingRegressor(estimators=estimators)
voting_regressor.fit(X_train, y_train)

y_pred_new = voting_regressor.predict(X_test)
rmse_1 = np.sqrt(mean_squared_error(y_test, y_pred_new))
print(rmse_1)


y_pred_test=voting_regressor.predict(test_df)


submission = pd.DataFrame({
    'id':test_df['id'],
    'Listening_Time_minutes':y_pred_test
})
submission.to_csv('submission.csv', index=False)




