import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
df.head()


df.info()


df.shape


nums=df.select_dtypes(include=np.number)
for i in nums:
    sns.boxplot(x=i,data=df)
    plt.title(i)
    plt.show()


pd.Series(df['diagnosed_diabetes']).value_counts().plot(kind='bar')
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Class Distribution (0 vs 1)")
plt.show()


x=df.drop(['id','diagnosed_diabetes'],axis=1)
y=df['diagnosed_diabetes'].astype(int)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,test_size=0.2,random_state=2)


x_train.select_dtypes(include='number').columns


x_train.select_dtypes(exclude='number').columns


num=['age', 'alcohol_consumption_per_week',
       'physical_activity_minutes_per_week', 'diet_score',
       'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
       'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
       'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
       'triglycerides', 'family_history_diabetes', 'hypertension_history',
       'cardiovascular_history']
cat=['gender', 'ethnicity', 'education_level', 'income_level',
       'smoking_status', 'employment_status']


from sklearn.preprocessing import RobustScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
numeric=RobustScaler()
category=OneHotEncoder(handle_unknown='ignore')


from sklearn.compose import ColumnTransformer
preprocess=ColumnTransformer([('num',numeric,num),('cat',category,cat)])


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report,confusion_matrix
from lightgbm import LGBMClassifier


model_1=Pipeline([('pre',preprocess),('lr',LogisticRegression(class_weight='balanced',penalty='l2',random_state=2,C=0.01,max_iter=400))])
model_2=Pipeline([('pre',preprocess),('rf',RandomForestClassifier(n_estimators=300,max_depth=6,random_state=2,class_weight='balanced'))])
model_3=Pipeline([('pre',preprocess),('xbg',XGBClassifier(learning_rate=0.01,n_estimators=100,max_depth=6,random_state=2))])
model_4=Pipeline([('pre',preprocess),('lgb',LGBMClassifier())])


model_1.fit(x_train,y_train)


y_pred_1=model_1.predict(x_test)


model_2.fit(x_train,y_train)


y_pred_2=model_2.predict(x_test)


model_3.fit(x_train,y_train)


y_pred_3=model_3.predict(x_test)


model_4.fit(x_train,y_train)


y_pred_4=model_4.predict(x_test)


print("Logistic Regression Performance")
print(classification_report(y_test,y_pred_1))
print("Random Forest Performance")
print(classification_report(y_test,y_pred_2))
print("XGBoost Performance")
print(classification_report(y_test,y_pred_3))
print("LightGBM Performance")
print(classification_report(y_test,y_pred_4))


from sklearn.neural_network import MLPClassifier
model_5=Pipeline([('pre',preprocess),('mlp',MLPClassifier(activation='relu',learning_rate='adaptive',max_iter=1000,random_state=2,hidden_layer_sizes=(120,50),solver='adam'))])


model_5.fit(x_train,y_train)


y_pred_5=model_5.predict(x_test)


print(classification_report(y_pred_5,y_test))


df_2=pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


df_2.head()


y_pred_6=model_1.predict(df_2)


y_pred_6[:]


import os
path='/kaggle/working/'
df_3=pd.DataFrame(df_2['id'])
df_3=


df_3.head()

