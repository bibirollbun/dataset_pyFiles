import pandas as pd 
file =  pd.read_csv('train.csv')


from sklearn.preprocessing import LabelEncoder

encoding_columns =  ['Stage_fear','Drained_after_socializing']
encoders = {}
for i in encoding_columns:
    encoder =  LabelEncoder()
    file[i] =  encoder.fit_transform(file[i])
    encoders[i] =  encoder





y_encoder =  LabelEncoder()
file['Personality'] =  y_encoder.fit_transform(file['Personality'])


file.isnull().sum()


file.loc[file['Time_spent_Alone'].isnull() , 'Time_spent_Alone'   ] =  file['Time_spent_Alone'].mean()
file.loc[file['Social_event_attendance'].isnull() , 'Social_event_attendance'   ] =  file['Social_event_attendance'].mean()
file.loc[file['Going_outside'].isnull() , 'Going_outside'   ] =  file['Going_outside'].mean()
file.loc[file['Friends_circle_size'].isnull() , 'Friends_circle_size'   ] =  file['Friends_circle_size'].mean()
file.loc[file['Post_frequency'].isnull() , 'Post_frequency'   ] =  file['Post_frequency'].mean()



from sklearn.preprocessing import MinMaxScaler
scaler =  MinMaxScaler()

file[['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency']] =  scaler.fit_transform(file[['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency']] )


from sklearn.model_selection import train_test_split

x =  file.drop(columns={'id','Personality'})
y =  file['Personality']


xtrain ,xtest , ytrain , ytest  =  train_test_split(x,y,random_state=42,train_size=.80)


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
gridsearch = GridSearchCV()
log_model =  LogisticRegression()
param_grid = [
    {
        'solver': ['lbfgs'],
        'penalty': ['l2', None],
        'C': [0.01, 0.1, 1, 10],
        'max_iter': [200]
    },
    {
        'solver': ['liblinear'],
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10],
        'max_iter': [200]
    },
    {
        'solver': ['saga'],
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'C': [0.01, 0.1, 1, 10],
        'l1_ratio': [0, 0.5, 1],  # only used for elasticnet
        'max_iter': [200]
    }
]

grid = GridSearchCV(log_model, param_grid, cv=5, scoring='accuracy')

grid.fit(xtrain,ytrain)





