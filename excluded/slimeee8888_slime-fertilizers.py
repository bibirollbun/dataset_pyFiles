import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

import warnings
warnings.simplefilter('ignore')



data=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv',index_col=0)
data_submission=pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv',index_col=0)

original = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
original.rename(columns={"Humidity ": "Humidity"},inplace=True)
print(f'Columns: \n {data.columns.values}')

data=pd.concat([data,original],axis=0)
data.reset_index(drop=True,inplace=True)


from scipy import stats as st

y_value_count=data['Fertilizer Name'].value_counts()
y_value_unique=np.unique(data['Fertilizer Name'])
print(y_value_unique)
total_df=pd.DataFrame()
for y_elem in y_value_unique:
    temp_data=data.loc[data['Fertilizer Name']==y_elem]
    temp_arr_for_mode=[]
    for column in temp_data.columns.values:
        # print(column)
        if column in ['Soil Type','Crop Type','Fertilizer Name']:
            continue
        

        temp_arr_for_mode.append(st.mode(temp_data[f'{column}']).mode)
        # print(t)
        # ()
    tt=pd.Series(temp_arr_for_mode,index=['Temparature','Humidity','Moisture','Nitrogen',
 'Potassium','Phosphorous'],name=y_elem)
    total_df=pd.concat([total_df,tt],axis=1)
total_df=total_df.T
print(total_df)



percent=0.00



# new_data=pd.DataFrame()
# for col in data.columns.values:
#     temp_col=pd.DataFrame()
#     if (col=='Soil Type') or (col=='Crop Type') or (col == 'Fertilizer Name'):
#         continue
#     for Y_elem in y_value_unique:
        
#         miwn=data.loc[data['Fertilizer Name'] == Y_elem,col].min()
#         lower_bound = data.loc[data['Fertilizer Name'] == Y_elem,col].quantile(0+percent)
#         # print(lower_bound,miwn)
#         upper_bound = data.loc[data['Fertilizer Name'] == Y_elem,col].quantile(1-percent)
        
#         rrwrw=data[col].loc[data['Fertilizer Name']==Y_elem].index.values
#         # print(rrwrw)
#         # data[col].hist()
#         temp_data= data.loc[data['Fertilizer Name'] == Y_elem,col]
        
#         temp_data=temp_data[(temp_data >= lower_bound) & (temp_data <= upper_bound)]
        
#         temp_col=pd.concat([temp_col,temp_data],axis=0 )
#     temp_col['ID'] = temp_col.index
#     if col=='Temparature':
        
#         new_data=temp_col
#     else:
        
#         new_data=pd.merge(new_data,temp_col,on='ID',how='inner')
 


# categorical_columns=data[['Soil Type','Crop Type']]
# categorical_columns['ID'] = categorical_columns.index
# new_data=pd.merge(new_data,categorical_columns,on='ID',how='inner')

# Y_set_for_training=data[['Fertilizer Name','Crop Type']]
# Y_set_for_training['ID'] = Y_set_for_training.index
# Y_set_for_training.drop(columns=['Crop Type'],inplace=True)

# new_data=pd.merge(new_data,Y_set_for_training,on='ID',how='inner')


# Y_set_for_training=new_data['Fertilizer Name']

# new_data.drop(columns=['ID','Fertilizer Name'],inplace=True)



# data=new_data
# print(data)

Y_set_for_training=data[['Fertilizer Name','Crop Type']]
Y_set_for_training.drop(columns=['Crop Type'],inplace=True)
data.drop(columns=['Fertilizer Name'],inplace=True)



Y_set_for_training



# data.drop(columns=['Fertilizer Name'],inplace=True)

Y_set_for_training=pd.Series(le.fit_transform(Y_set_for_training),name='Fertilizer Name',index=Y_set_for_training.index.values)
y_set_dict=dict()
[y_set_dict.update({ind: i}) for ind,i in enumerate(le.classes_)]

y_set_dict


data_indexes=data.index.values

concatenate_train_test=pd.concat([data,data_submission])



concatenate_train_test


min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(concatenate_train_test.drop(columns=['Soil Type','Crop Type'],axis=1 ))
data_normalized = pd.DataFrame(x_scaled,columns=(data.drop(columns=['Soil Type','Crop Type'],axis=1 )).columns.values)
data_normalized['ID']=concatenate_train_test.index.values


data_normalized




for column in data[['Soil Type','Crop Type']].columns.values:
    x_scaled=pd.DataFrame(le.fit_transform(concatenate_train_test[column]),columns=[column],index=concatenate_train_test.index.values)
    x_scaled.index=data_normalized.index
    data_normalized=pd.concat([data_normalized,x_scaled],axis=1)

    temp_dict=dict()
    [temp_dict.update({i: ind}) for i,ind in enumerate(le.classes_)]
    print('\n Label encodings for column: ',column,'\n',temp_dict)


data_normalized.drop(columns=['ID'],inplace=True)



data_normalized



print(data.head(5))




set_for_training=data_normalized.iloc[data_indexes,:]
corr=set_for_training.corr()
set_for_submission=data_normalized.iloc[np.max(data_indexes)+1:,:]
print(np.max(data_indexes)+1)


set_for_training.shape



x_train, x_test, y_train, y_test = train_test_split(
    set_for_training, Y_set_for_training, test_size=0.3, random_state=42)

x_valid, x_test, y_valid, y_test = train_test_split(
    x_test, y_test, test_size=0.3, random_state=42)






print(f'Data normalized and splitted into train and test datasets. \n '
      f'Train dataset size: {x_train.shape[0]}, test dataset size:{x_test.shape[0]}')



sets=[x_train,x_test,y_train,y_test]
[print(len(i)) for i in sets]




for column in x_train.columns.values:
    plt.figure()
    plt.hist(x_train[column])
    plt.suptitle(f'{column}')
    plt.show()



def map_at_k(y_true_ind,y_pred_proba,k=3):
    if len(y_true_ind)!=len(y_pred_proba):
        raise ValueError('Lenght of arrays is not matching')
    if k>6:
        raise ValueError('k bigger than 6')
    # k=3
    total_rel=0
    count=0
    rel_arr=[]
    for y_true_elem,y_pred_elem in zip(y_true_ind,y_pred_proba):
        rel=None
        y_true_val=y_set_dict[y_true_elem]
        y_pred_sorted = np.argsort(y_pred_elem)[::-1]
        top_k_predict=[y_set_dict[i] for i in y_pred_sorted[:k]]
        for ind in range(k):
            if rel is not None:
                break
            first_k=top_k_predict[:k]
            for ind,elem in enumerate(first_k):
                if elem==y_true_val:
                    if ind==0:
                        rel=1
                    if ind==1:
                        rel=0.5
                    if ind==2:
                        rel=1/3
            if rel is None: rel=0
        total_rel+=rel
        rel_arr.append(rel)
        count+=1

    rel_arr=pd.Series(rel_arr)
    print(rel_arr.value_counts())

    
    plt.figure()
    plt.title(f'{total_rel/count}')
    
    rel_arr.hist()
    plt.show()


    
    return total_rel/count
    


import optuna
from sklearn.model_selection import cross_val_score


def check_metric(model,x_fold_train,y_fold_train,x_fold_test,y_fold_test):
    
    y_pred = model.predict(x_fold_train)
    y_pred_proba_train=model.predict_proba(x_fold_train)
    
    
    MAP_train = map_at_k(y_fold_train,y_pred_proba_train,3)
    print("MAP train ",MAP_train)
    
    y_pred = model.predict(x_fold_test)
    y_pred_proba_test=model.predict_proba(x_fold_test)
    MAP_test = map_at_k(y_fold_test,y_pred_proba_test,3)
    print("MAP test ",MAP_test)


xgb_params={'learning_rate': 0.013075404355262733, 'max_depth': 13, 'gamma': 0.7923016271246413, 'subsample': 0.5364050102780459, 'colsample_bytree': 0.5486599932091898, 'reg_lambda': 1.4270183474090032, 'reg_alpha': 1.8181702361326582}
cat_params={'learning_rate': 0.010043935739595129, 'depth': 13, 'l2_leaf_reg': 8.25135906253988, 'bagging_temperature': 0.5847235478226895, 'random_strength': 0.09286288143167892}
lgb_params={'learning_rate': 0.010217053837832588, 'num_leaves': 21, 'max_depth': 13, 'min_child_samples': 48, 'subsample': 0.9087819252484214, 'colsample_bytree': 0.744354832453605, 'reg_alpha': 3.256541343411315, 'reg_lambda': 4.931214449300567}




xgb_params.update({'n_estimators':300})

cat_params.update({'n_estimators':300})

lgb_params.update({'n_estimators':300})



from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
kf = StratifiedKFold(n_splits=9)

model_predicts=pd.DataFrame()
def check_metric(model,x_fold_train,y_fold_train,x_fold_test,y_fold_test):
    
    y_pred = model.predict(x_fold_train)
    y_pred_proba_train=model.predict_proba(x_fold_train)
    
    
    MAP_train = map_at_k(y_fold_train,y_pred_proba_train,3)
    print("MAP train ",MAP_train)
    
    y_pred = model.predict(x_fold_test)
    y_pred_proba_test=model.predict_proba(x_fold_test)
    MAP_test = map_at_k(y_fold_test,y_pred_proba_test,3)
    print("MAP test ",MAP_test)


best_score=0
for i,(train_index, test_index) in enumerate(kf.split(x_train,y_train)):
    
    x_fold_train, x_fold_test = x_train.iloc[train_index],x_train.iloc[test_index]
    y_fold_train, y_fold_test = y_train.iloc[train_index], y_train.iloc[test_index]
    
    if (i==0):
        model_0=XGBClassifier(objective='multi:softprob',verbose=1,**xgb_params)
        model_0.fit(x_fold_train,y_fold_train)
        check_metric(model_0,x_fold_train,y_fold_train,x_fold_test,y_fold_test)
        model_predicts['model_0']=model_0.predict(set_for_training)
    if i==1:
        model_1=XGBClassifier(objective='multi:softprob',verbose=1,**xgb_params)
        model_1.fit(x_fold_train,y_fold_train)
        check_metric(model_1,x_fold_train,y_fold_train,x_fold_test,y_fold_test)
        model_predicts['model_1']=model_1.predict(set_for_training)
    if i==2:

        model_2=XGBClassifier(objective='multi:softprob',verbose=1,**xgb_params)
        model_2.fit(x_fold_train,y_fold_train)
        check_metric(model_2,x_fold_train,y_fold_train,x_fold_test,y_fold_test)
        model_predicts['model_2']=model_2.predict(set_for_training)
      
    if i==3:
   
        model_3=CatBoostClassifier(loss_function='MultiClass', verbose=0,**cat_params)
        model_3.fit(x_fold_train,y_fold_train)
        check_metric(model_3,x_fold_train,y_fold_train,x_fold_test,y_fold_test)
        model_predicts['model_3']=model_3.predict(set_for_training)
    if i==4:
        # study_cat = optuna.create_study(direction='maximize')
        # study_cat.optimize(objective_cat, n_trials=50)
        # print("Best params model 4: CatBoostClassifier:", study_cat.best_params)
        model_4=CatBoostClassifier(loss_function='MultiClass', verbose=0,**cat_params)
        model_4.fit(x_fold_train,y_fold_train)
        check_metric(model_4,x_fold_train,y_fold_train,x_fold_test,y_fold_test)
        model_predicts['model_4']=model_4.predict(set_for_training)
    if i==5:
  
        model_5=CatBoostClassifier(loss_function='MultiClass', verbose=0,**cat_params)
        model_5.fit(x_fold_train,y_fold_train)
        check_metric(model_5,x_fold_train,y_fold_train,x_fold_test,y_fold_test)
        model_predicts['model_5']=model_5.predict(set_for_training)

    if i==6:
  
        model_6=lgb.LGBMClassifier(verbose=-1,**lgb_params)
        model_6.fit(x_fold_train,y_fold_train)
        check_metric(model_6,x_fold_train,y_fold_train,x_fold_test,y_fold_test)
        model_predicts['model_6']=model_6.predict(set_for_training)
    if i==7:
  
        model_7=lgb.LGBMClassifier(verbose=-1,**lgb_params)
        model_7.fit(x_fold_train,y_fold_train)
        check_metric(model_7,x_fold_train,y_fold_train,x_fold_test,y_fold_test)
        model_predicts['model_7']=model_7.predict(set_for_training)
    if i==8:
       
        model_8=lgb.LGBMClassifier(verbose=-1,**lgb_params)
        model_8.fit(x_fold_train,y_fold_train)
        check_metric(model_8,x_fold_train,y_fold_train,x_fold_test,y_fold_test)
        model_predicts['model_8']=model_8.predict(set_for_training)




corr_models=model_predicts.corr()
corr_models[corr_models>0.5]


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
model = StackingClassifier(
    estimators=[('m0',model_0),('m1',model_1),('m2',model_2),('m3',model_3)
                ,('m4',model_4),('m5',model_5),('m6',model_6),('m7',model_7),('m8',model_8)
               ],
    final_estimator=LogisticRegression(),
    cv="prefit", passthrough=False
)
model.fit(x_valid, y_valid)



y_pred=model.predict(x_test)
y_pred_proba=model.predict_proba(x_test)
from sklearn.metrics import f1_score

f1_scr_test=f1_score(y_test, y_pred, average='micro')


print(y_pred)
print(f1_scr_test)



preds=[]
for i in y_pred_proba:
    sorted_indices = np.argsort(i)[::-1]
    temp_subm=''
    for i in sorted_indices[:3]:
        temp_subm+=f'{y_set_dict[i]} '
    preds.append(temp_subm)



len(y_set_dict)


set_for_submission



import time
start_time=time.time()
map_3=map_at_k(y_test,y_pred_proba,3)
print(map_3)
end_time=time.time()-start_time
print(end_time)
print(len(y_test))


submission_answer=model.predict(set_for_submission)
submission_answer_proba=model.predict_proba(set_for_submission)
submission_preds=[]
for i in submission_answer_proba:
    sorted_indices = np.argsort(i)[::-1]
    temp_subm=''
    for i in sorted_indices[:3]:
        temp_subm+=f'{y_set_dict[i]} '
    submission_preds.append(temp_subm)
# print(submission_preds)
submission_answer=submission_preds




submission=pd.concat([pd.Series(range(750000,1000000,1),name='id'),pd.Series(submission_answer,name='Fertilizer Name')],names=['id','Fertilizer Name'],axis=1)
print(submission)
submission.to_csv('submission.csv',index=False)

