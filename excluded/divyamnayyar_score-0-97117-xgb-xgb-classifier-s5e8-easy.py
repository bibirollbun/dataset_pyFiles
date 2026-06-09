import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
# from sklearn.linear_model import LogisticRegression
# from xgboost import XGBRegressor
from xgboost import XGBClassifier
# from imblearn.over_sampling import SMOTE
from category_encoders import MEstimateEncoder
# from sklearn.calibration import CalibratedClassifierCV
import numpy as np
# from catboost import CatBoostClassifier
from xgboost import plot_importance
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import StratifiedKFold
# from sklearn.ensemble import VotingClassifier
pd.set_option('display.max_columns',None)



train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv',index_col=0) 


train.head()


for i in train.columns:
    print(f"Cardinality of {i} = {train[i].nunique()} , and its dtype = {train[i].dtype}")


#Unknows in cols
for i in train.columns:
    print(f"Unknows in {i} = {(train[i] =='unknown').sum()}")


def mapping(df):
    df['default'] = df['default'].map({'no':0,'yes':1})
    df['housing'] = df['housing'].map({'no':0,'yes':1})
    df['loan'] = df['loan'].map({'no':0,'yes':1})

    marital_map = {'single':0,'married':1,'divorced':2} 
    education_map = {'unknown':-1,'secondary':0,'primary':1,'tertiary':2}
    contact_map = {'unknown':-1,'cellular':0,'telephone':1}
    df['marital'] = df['marital'].map(marital_map)
    df['education'] = df['education'].map(education_map)
    df['contact'] = df['contact'].map(contact_map)
    month_map = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
             'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
    df['month'] = df['month'].map(month_map)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)

    poutcome_map = {'unknown':-1,'other':0,'failure':1,'success':2}
    # df['poutcome'] = df['poutcome'].map(poutcome_map)

    # df['people_contacted_over_phone_for_long_duration'] = df['people_contacted_over_phone_for_long_duration'].map({False:0,True:1})

def feature_engineer(df):
    # mask_3 = df['duration'] > 200
    # mask_4 = df['contact'] == 0
    # df['people_contacted_over_phone_for_long_duration'] = mask_3 & mask_4
    df['crat'] = df['duration']/df['campaign']
    df['age*time'] = df['duration']*df['age']
    df['cap'] =df['balance']*df['job']



validation = train.sample(frac = 0.1)
train.drop(validation.index,inplace= True)



# Target encoding setup
validation_y = validation.pop('y')
validation_x = validation.copy()
encoder = MEstimateEncoder(cols = ['job'],m = 0.5)
encoder.fit(validation_x,validation_y)



y = train.pop('y')
X = train.copy()


train_x,val_x,train_y,val_y = train_test_split(X,y,test_size = 0.2,random_state=42)


train_x = encoder.transform(train_x) # Target Encoding 


# # One Hot encoding  
# OH_cols = ['job']
# OH_encoder = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
# OH_encoder_train = pd.DataFrame(OH_encoder.fit_transform(train_x[OH_cols]))
# OH_encoder_train.index = train_x.index 
# train_x.drop(OH_cols,axis=1,inplace=True)
# train_x = pd.concat([OH_encoder_train,train_x],axis = 1)



mapping(train_x)
feature_engineer(train_x)


train_x.drop(['poutcome'],inplace = True,axis = 1)
train_x


train_x.columns = train_x.columns.astype(str)


val_x = encoder.transform(val_x)


# OH_encode_val = pd.DataFrame(OH_encoder.transform(val_x[OH_cols]))
# OH_encode_val.index = val_x.index 
# OH_encode_val.columns = OH_encode_val.columns.astype(str)
# val_x.drop(OH_cols,axis= 1,inplace=True)
# val_x = pd.concat([OH_encode_val,val_x],axis = 1)


mapping(val_x)
feature_engineer(val_x)


val_x.drop(['poutcome'],axis = 1,inplace=True)
val_x


train_y.value_counts()


# adding the below hyper parameter to handle the class imbalance
scale_pos_weight_v = 659512/90488 # Total Negatives / Total Positives
scale = 474882/65118 # Negatives in train_x / Positives in train_x
print(scale,scale_pos_weight_v)


model_xgb = XGBClassifier(subsample = 0.9,grow_policy="depthwise",max_bin = 4096,colsample_bytree = 0.8,scale_pos_weight = scale_pos_weight_v,learning_rate = 0.07,n_estimators = 3000,min_child_weight = 2 , gamma = 2,eval_metric = 'auc',reg_alpha = 0.5,reg_lambda = 3)
model_xgb.fit(train_x,train_y)



roc_auc_score(val_y,model_xgb.predict(val_x,output_margin=True))


plot_importance(model_xgb,max_num_features=20)


test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv',index_col= 0)
# test = test.set_index('id')
test.head()


test = encoder.transform(test)


mapping(test)
feature_engineer(test)


# OH_encode_test = pd.DataFrame(OH_encoder.transform(test[OH_cols]))
# OH_encode_test.index = test.index
# OH_encode_test.columns = OH_encode_test.columns.astype(str)
# test.drop(OH_cols,inplace=True,axis= 1)
# test = pd.concat([OH_encode_test,test],axis = 1)


test.drop('poutcome',axis = 1,inplace = True)
test.head()


preds = model_xgb.predict(test,output_margin=True)
submission = pd.DataFrame({'id':test.index,'y':preds})


submission.to_csv('submission.csv',index=False)

