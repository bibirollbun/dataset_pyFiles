import pandas as pd 

file =  pd.read_csv('train.csv')



from sklearn.preprocessing import LabelEncoder , MinMaxScaler
from sklearn.model_selection import train_test_split

# dropping id column 
file =  file.drop(columns='id')
# encoding 
encodecolumns =  ['job','marital','education','housing','loan','contact','month','poutcome','default']

encoders = {}

for col in encodecolumns:
    encoder =  LabelEncoder()
    file[col] =  encoder.fit_transform(file[col])
    encoders[col] = encoder

# scalling 
scaler =  MinMaxScaler()
scaledcolumns =  ['age','balance','day','duration','campaign','pdays','previous']
file[scaledcolumns] =  scaler.fit_transform(file[scaledcolumns])


x=  file.drop(columns='y')
y =  file['y']


# Traing,test,split 
xtrain,xtest,ytrain,ytest =  train_test_split(x,y,random_state=42,train_size=.85)


# Modelling 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
logesticmodel = LogisticRegression(max_iter=500)


param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']   
}

grid_search = GridSearchCV(
    estimator=logesticmodel,
    param_grid=param_grid,
    cv=5,             # 5-fold cross-validation
    scoring='roc_auc',   # or 'f1', 'roc_auc', depending on your problem
    n_jobs=-1          # use all CPU cores
)

grid_search.fit(xtrain, ytrain)


# fitting best model 

print(grid_search.best_score_)

best_logesticmodel =  grid_search.best_estimator_



testdf =  pd.read_csv('test.csv')
# dropping id , saving to new df 
iddf =  testdf['id']
testdf =  testdf.drop(columns='id')


# encoding
for col in encodecolumns:
    testdf[col] = encoders[col].transform(testdf[col])

# scaling
testdf[scaledcolumns] = scaler.transform(testdf[scaledcolumns])



y_pred_proba = grid_search.best_estimator_.predict_proba(testdf)[:, 1]


testdf['y'] =  y_pred_proba


testdf['id'] =  iddf


testdf[['id','y']].to_csv('logestic_model_results.csv')


testdf[['id','y']]


testdf =  testdf.drop(columns=['id','y'])


logesticmodel.fit(xtrain,ytrain)


pd.DataFrame(logesticmodel.predict_proba(testdf)[:,1])


# saving for later practise 

import joblib 

joblib.dump(encoders,'encoders.pkl')
joblib.dump(scaler,'scaler.pkl')


from sklearn.ensemble import RandomForestClassifier

x=  file.drop(columns='y')
y =  file['y']

# Traing,test,split 
xtrain,xtest,ytrain,ytest =  train_test_split(x,y,random_state=42,train_size=.85)


from sklearn.model_selection import GridSearchCV
Random_model =  RandomForestClassifier(n_estimators=100,max_depth=15)






Random_model.fit(xtrain,ytrain)


Random_model.predict_proba(xtest)[0:,1]


testdf['y']  =   Random_model.predict_proba(testdf)[0:,1]


testdf['id'] = iddf


testdf[['id','y']].to_csv('random_results.csv')

