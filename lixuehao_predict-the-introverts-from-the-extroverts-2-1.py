# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.impute import KNNImputer

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt 
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



train_file="/kaggle/input/playground-series-s5e7/train.csv"
test_pred_file="/kaggle/input/playground-series-s5e7/test.csv"
sample_submission_file="/kaggle/input/playground-series-s5e7/sample_submission.csv"


df=pd.read_csv(train_file)
df


#将"Stage_fear"与"Drained_after_socializing"的字符类型填充(用No填补)
df["Stage_fear"].fillna("No",inplace=True)
df["Drained_after_socializing"].fillna("No",inplace=True)


#进行分类转化为编码
df_labelencoder_model=LabelEncoder()
df["Stage_fear"]=df_labelencoder_model.fit_transform(df["Stage_fear"])
df["Drained_after_socializing"]=df_labelencoder_model.fit_transform(df["Drained_after_socializing"])


#得到其他类型的特征
df_columns=list(set(df.columns)-set(["id","Stage_fear","Drained_after_socializing","Personality"]))
df_columns


#进行缺失值的填补（用众数去填充）
EX_modes ={
    "Time_spent_Alone": df[df["Personality"]=="Extrovert"]["Time_spent_Alone"].mode().iloc[0],
    "Social_event_attendance": df[df["Personality"]=="Extrovert"]["Social_event_attendance"].mode().iloc[0],
    "Going_outside": df[df["Personality"]=="Extrovert"]["Going_outside"].mode().iloc[0],
    "Friends_circle_size": df[df["Personality"]=="Extrovert"]["Friends_circle_size"].mode().iloc[0],
    "Post_frequency": df[df["Personality"]=="Extrovert"]["Post_frequency"].mode().iloc[0]
}

IN_modes = {
    "Time_spent_Alone": df[df["Personality"]=="Introvert"]["Time_spent_Alone"].mode().iloc[0],
    "Social_event_attendance": df[df["Personality"]=="Introvert"]["Social_event_attendance"].mode().iloc[0],
    "Going_outside":df[df["Personality"]=="Introvert"]["Going_outside"].mode().iloc[0],
    "Friends_circle_size": df[df["Personality"]=="Introvert"]["Friends_circle_size"].mode().iloc[0],
    "Post_frequency": df[df["Personality"]=="Introvert"]["Post_frequency"].mode().iloc[0]  
}

for column in df_columns :
    df[column]=df.apply(
    lambda x : EX_modes[column]
    if(pd.isna(x[column]) and x["Personality"]=="Extrovert")
    else (IN_modes[column]
            if pd.isna(x[column]) and x["Personality"]=="Introvert"
            else x[column]),
    axis=1)


df.isnull().sum()


#将float类型转化为int类型
for column in df_columns :
    df[column]=df[column].astype(int)
for col in df.columns :
    print("-----"+col+"------")
    print(df[col].dtype)


train_df,test_df=train_test_split(df,test_size=0.2,random_state=42)
train_df_X=train_df[["id","Time_spent_Alone","Stage_fear","Social_event_attendance","Going_outside","Drained_after_socializing","Friends_circle_size","Post_frequency"]]
train_df_Y=train_df["Personality"]
test_df_X=test_df[["id","Time_spent_Alone","Stage_fear","Social_event_attendance","Going_outside","Drained_after_socializing","Friends_circle_size","Post_frequency"]]
test_df_Y=test_df["Personality"]


train_df_X


#catboost模型的建立与训练
cat_model=CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, verbose=10)
train_pool=Pool(train_df_X,train_df_Y,cat_features=[1,2,3,4,5,6,7])
cat_model.fit(train_pool)


#保存catboost模型
cat_model.save_model('/kaggle/working/catboost_model_fillna_No.cbm')


#对测试集进行预测
test_pool=Pool(test_df_X,cat_features=[1,2,3,4,5,6,7])
test_pred=cat_model.predict(test_pool)
unique_test_pred,counts_unique_test_pred=np.unique(test_pred,return_counts=True)
for unique,count in zip(unique_test_pred,counts_unique_test_pred):
    print(f'value:{unique}----count:{count}') 


#将预测值与test的标签进行对比得到准确率
test_df_pred=pd.Series(test_pred)
accuracy=accuracy_score(test_df_Y,test_df_pred,)
accuracy


pred_df=pd.read_csv(test_pred_file)
pred_df


#将"Stage_fear"与"Drained_after_socializing"的字符类型填充(用No填补)
pred_df["Stage_fear"].fillna("No",inplace=True)
pred_df["Drained_after_socializing"].fillna("No",inplace=True)


#进行分类转化为编码
pred_df_labelencoder_model=LabelEncoder()
pred_df["Stage_fear"]=pred_df_labelencoder_model.fit_transform(pred_df["Stage_fear"])
pred_df["Drained_after_socializing"]=pred_df_labelencoder_model.fit_transform(pred_df["Drained_after_socializing"])
pred_df_labelencoder_model.classes_


#处理其他类型的缺失值
pred_df_columns=list(set(pred_df.columns)-set(["Stage_fear","Drained_after_socializing","id"]))
for col in pred_df_columns :
    pred_df[col].fillna(0,inplace=True)


#将float类型变为int类型
for i in pred_df_columns :
    pred_df[i]=pred_df[col].astype(int)
for i in pred_df_columns :
    print("----"+i+"----")
    print(pred_df[i].dtype)
pred_df


#进行预测
pred_pool=Pool(pred_df,cat_features=[1,2,3,4,5,6,7])
pred=cat_model.predict(pred_pool)
pred


#将预测结果放入到sample_submission中
sample_submission=pd.read_csv(sample_submission_file)
sample_submission["Personality"]=pred
sample_submission.to_csv('/kaggle/working/sample_submission.csv',index=False)
sample_submission










