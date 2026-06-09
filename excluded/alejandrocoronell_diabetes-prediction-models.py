import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import missingno as msno
from imblearn.over_sampling import  RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,roc_auc_score
import lightgbm as lgb
from sklearn.linear_model import SGDClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')


pd.options.display.max_columns = None


df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')


df_test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
df_sa = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


df


df.describe()


df.info()


df.drop(['id'],axis=1,inplace=True)


ids_test = df_test['id']
df_test = df_test.drop(columns=['id'])


def create_medical_features(df):
    df = df.copy()
    
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    
    df['bp_ratio'] = df['systolic_bp'] / (df['diastolic_bp'] + 1e-6)
    
    df['non_hdl_cholesterol'] = df['cholesterol_total'] - df['hdl_cholesterol']
    
    df['cholesterol_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1e-6)
    
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1e-6)
    
    screen_minutes = df['screen_time_hours_per_day'] * 60
    
    df['activity_screen_ratio'] = (df['physical_activity_minutes_per_week'] + 1) / (screen_minutes + 1)
    
    df['alcohol_bmi_interaction'] = df['alcohol_consumption_per_week'] * df['bmi']
    
    df['age_bmi_interaction'] = df['age'] * df['bmi']
    
    df['age_bp_interaction'] = df['age'] * df['systolic_bp']
    
    df['log_triglycerides'] = np.log1p(df['triglycerides'])
    
    return df

print("Generating new variables...")
df = create_medical_features(df)
df_test = create_medical_features(df_test)


df


df['gender'].value_counts()


le = LabelEncoder()


df['gender']= le.fit_transform(df['gender'])


df_test['gender']=le.fit_transform(df_test['gender'])


df['gender'].value_counts()


df_test['gender'].value_counts()


df['ethnicity'].value_counts()


df = pd.get_dummies(df, columns=['ethnicity'], dtype=int)


df_test = pd.get_dummies(df_test,columns = ['ethnicity'],dtype=int)


df_test


df


df['education_level'].value_counts()


df['income_level'].value_counts()


df['smoking_status'].value_counts()


income_m = {
    'Low': 0,
    'Lower-Middle': 1,
    'Middle': 2,
    'Upper-Middle': 3,
    'High': 4
}

education_m = {
    'No formal': 0,
    'Highschool': 1,
    'Graduate': 2,
    'Postgraduate': 3
}

smoking_m = {
    'Never': 0,
    'Former': 1,
    'Current': 2
}



df['income_level'] = df['income_level'].replace(income_m)
df['education_level'] = df['education_level'].replace(education_m)
df['smoking_status']= df['smoking_status'].replace(smoking_m)
df_test['income_level'] = df_test['income_level'].replace(income_m)
df_test['education_level'] = df_test['education_level'].replace(education_m)
df_test['smoking_status']= df_test['smoking_status'].replace(smoking_m)


df['income_level'].value_counts()



df_test['income_level'].value_counts()



df['education_level'].value_counts()



df['smoking_status'].value_counts()


df['employment_status'].value_counts()


df['employment_status']=le.fit_transform(df['employment_status'])


df_test['employment_status']=le.fit_transform(df_test['employment_status'])


df['employment_status'].value_counts()


df.hist(figsize=(25,25))


sns.set(style='whitegrid',font_scale= 1.5)
plt.figure(figsize=(15,15))
sns.heatmap(df.corr(),vmax=0.8,vmin=-0.8,annot=False,cmap='GnBu')


plt.figure(figsize=(40,40))
c = df.columns[:-1]
for i in enumerate(c):
    plt.subplot(7,6,i[0]+1)
    sns.boxplot(x=i[1],data=df)


for i in df.columns[:-1]:
    q1 = df[i].quantile(0.05)
    q3 = df[i].quantile(0.95)
    df[i][df[i]<=q1] = q1
    df[i][df[i]>=q3] = q3


for i in df_test.columns[:-1]:
    q1 = df_test[i].quantile(0.05)
    q3 = df_test[i].quantile(0.95)
    df_test[i][df_test[i]<=q1] = q1
    df_test[i][df_test[i]>=q3] = q3


plt.figure(figsize=(40,40))
c = df.columns[:-1]
for i in enumerate(c):
    plt.subplot(7,6,i[0]+1)
    sns.boxplot(x=i[1],data=df)


msno.bar(df)


msno.bar(df_test)


X = df.drop('diagnosed_diabetes',axis=1)
y = df['diagnosed_diabetes']


X


df_test=df_test.reindex(columns=X.columns,fill_value=0)


y.value_counts()


ros = RandomOverSampler(random_state=1000)
X_ros,y_ros = ros.fit_resample(X,y)


y_ros.value_counts()


rus = RandomUnderSampler(random_state=1000)
X_rus, y_rus = rus.fit_resample(X, y)


scaler = StandardScaler()
X_scaler_ros = scaler.fit_transform(X_ros)


X_test_scaled = scaler.transform(df_test)


X_scaler_rus = scaler.fit_transform(X_rus)


X_test_scaled_rus = scaler.transform(df_test)


X_train_ros, X_test_ros, y_train_ros, y_test_ros = train_test_split(X_scaler_ros, y_ros, test_size=0.2, random_state=1000)


X_train_rus,X_test_rus,y_train_rus,y_test_rus = train_test_split(X_scaler_rus,y_rus,test_size=0.2,random_state=1000)


modelo_xgb_ros = XGBClassifier(
    objective ='binary:logistic',
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',   
    eval_metric='auc',
    random_state=42,
    predictor='gpu_predictor'
)
modelo_xgb_ros.fit(X_train_ros,y_train_ros)


y_pred_xgb_ros_test = modelo_xgb_ros.predict(X_test_ros)
y_pred_xgb_ros_train = modelo_xgb_ros.predict(X_train_ros)


print("Accuracy XGB RUS Test:", accuracy_score(y_test_ros, y_pred_xgb_ros_test))
print("Accuracy XGB RUS Train:", accuracy_score(y_train_ros, y_pred_xgb_ros_train))


print("Classification Report Test:\n",classification_report(y_test_ros, y_pred_xgb_ros_test))
print("Classification Report Train:\n",classification_report(y_train_ros, y_pred_xgb_ros_train))



print("Confucion XGB ROS Test:\n", confusion_matrix(y_test_ros, y_pred_xgb_ros_test))
print("Confucion XGB ROS Train:\n", confusion_matrix(y_train_ros, y_pred_xgb_ros_train))


modelo_xgb_rus = XGBClassifier(
    objective ='binary:logistic',
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',   
    eval_metric='auc',
    random_state=42,
    predictor='gpu_predictor'
)
modelo_xgb_rus.fit(X_train_rus,y_train_rus)


y_pred_xgb_rus_test = modelo_xgb_rus.predict(X_test_rus)
y_pred_xgb_rus_train = modelo_xgb_rus.predict(X_train_rus)


print("Accuracy XGB RUS Test:", accuracy_score(y_test_rus, y_pred_xgb_rus_test))
print("Accuracy XGB RUS Train:", accuracy_score(y_train_rus, y_pred_xgb_rus_train))


print("Classification Report Test:\n",classification_report(y_test_rus, y_pred_xgb_rus_test))
print("Classification Report Train:\n",classification_report(y_train_rus, y_pred_xgb_rus_train))


print("Confucion XGB RUS Test:\n", confusion_matrix(y_test_rus, y_pred_xgb_rus_test))
print("Confucion XGB RUS Train:\n", confusion_matrix(y_train_rus, y_pred_xgb_rus_train))


model_lgbm_ros = lgb.LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=40,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',   # clave para recall
    random_state=42,
    n_jobs=-1
)
model_lgbm_ros.fit(X_train_ros,y_train_ros)


y_pred_lgbm_ros_test = model_lgbm_ros.predict(X_test_ros)
y_pred_lgbm_ros_train = model_lgbm_ros.predict(X_train_ros)


print("Accuracy LGBM ROS Test:", accuracy_score(y_test_ros, y_pred_lgbm_ros_test))
print("Accuracy LGBM ROS Train:", accuracy_score(y_train_ros, y_pred_lgbm_ros_train))


print("Classification Report Test:\n",classification_report(y_test_ros, y_pred_lgbm_ros_test))
print("Classification Report Train:\n",classification_report(y_train_ros, y_pred_lgbm_ros_train))


print("Confucion LGBM ROS Test:\n", confusion_matrix(y_test_ros, y_pred_lgbm_ros_test))
print("Confucion LGBM ROS Train:\n", confusion_matrix(y_train_ros, y_pred_lgbm_ros_train))


model_lgbm_rus = lgb.LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=40,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',   # clave para recall
    random_state=42,
    n_jobs=-1
)
model_lgbm_rus.fit(X_train_rus,y_train_rus)



y_pred_lgbm_rus_test = model_lgbm_rus.predict(X_test_rus)
y_pred_lgbm_rus_train = model_lgbm_rus.predict(X_train_rus)


print("Accuracy LGBM RUS Test:", accuracy_score(y_test_rus, y_pred_lgbm_rus_test))
print("Accuracy LGBM RUS Train:", accuracy_score(y_train_rus, y_pred_lgbm_rus_train))


print("Classification Report Test:\n",classification_report(y_test_rus, y_pred_lgbm_rus_test))
print("Classification Report Train:\n",classification_report(y_train_rus, y_pred_lgbm_rus_train))


print("Confucion LGBM RUS Test:\n", confusion_matrix(y_test_rus, y_pred_lgbm_rus_test))
print("Confucion LGBM RUS Train:\n", confusion_matrix(y_train_rus, y_pred_lgbm_rus_train))


model_sgd_ros = SGDClassifier(
    loss='log_loss',        
    alpha=1e-4,
    max_iter=2000,
    tol=1e-3,
    random_state=42
)
model_sgd_ros.fit(X_train_ros,y_train_ros)


y_pred_sgd_ros_test = model_sgd_ros.predict(X_test_ros)
y_pred_sgd_ros_train = model_sgd_ros.predict(X_train_ros)


print("Accuracy LGBM ROS Test:", accuracy_score(y_test_ros, y_pred_sgd_ros_test))
print("Accuracy LGBM ROS Train:", accuracy_score(y_train_ros, y_pred_sgd_ros_train))


print("Classification Report Test:\n",classification_report(y_test_ros, y_pred_sgd_ros_test))
print("Classification Report Train:\n",classification_report(y_train_ros, y_pred_sgd_ros_train))


print("Confucion LGBM ROS Test:\n", confusion_matrix(y_test_ros, y_pred_sgd_ros_test))
print("Confucion LGBM ROS Train:\n", confusion_matrix(y_train_ros, y_pred_sgd_ros_train))


model_sgd_rus = SGDClassifier(
    loss='log_loss',        
    alpha=1e-4,
    max_iter=2000,
    tol=1e-3,
    random_state=42
)
model_sgd_rus.fit(X_train_rus,y_train_rus)


y_pred_sgd_rus_test = model_sgd_rus.predict(X_test_rus)
y_pred_sgd_rus_train = model_sgd_rus.predict(X_train_rus)


print("Accuracy LGBM RUS Test:", accuracy_score(y_test_rus, y_pred_sgd_rus_test))
print("Accuracy LGBM RUS Train:", accuracy_score(y_train_rus, y_pred_sgd_rus_train))


print("Classification Report Test:\n",classification_report(y_test_rus, y_pred_sgd_rus_test))
print("Classification Report Train:\n",classification_report(y_train_rus, y_pred_sgd_rus_train))


print("Confucion LGBM RUS Test:\n", confusion_matrix(y_test_rus, y_pred_sgd_rus_test))
print("Confucion LGBM RUS Train:\n", confusion_matrix(y_train_rus, y_pred_sgd_rus_train))


ann_ros = Sequential()
ann_ros.add(Dense(units=50,activation='relu'))
#dropout
ann_ros.add(Dense(units=12,activation='relu'))
ann_ros.add(Dense(units=8,activation='relu'))
ann_ros.add(Dense(units=4,activation='sigmoid'))
ann_ros.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
ann_ros.fit(X_train_ros,y_train_ros, validation_data=(X_test_ros,y_test_ros), batch_size=32,epochs=50)


history = ann_ros.history.history
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()


history = ann_ros.history.history
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()


ann_rus = Sequential()
ann_rus.add(Dense(units=50,activation='relu'))
#dropout
ann_rus.add(Dense(units=12,activation='relu'))
ann_rus.add(Dense(units=8,activation='relu'))
ann_rus.add(Dense(units=4,activation='sigmoid'))
ann_rus.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
ann_rus.fit(X_train_rus,y_train_rus, validation_data=(X_test_rus,y_test_rus), batch_size=32,epochs=50)


history = ann_rus.history.history
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()


history = ann_rus.history.history
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()


print("Predicting...")
submission_xgb_ros = modelo_xgb_ros.predict_proba(X_test_scaled)[:,1]


submission_xgb_ros = pd.DataFrame({'id': df_sa['id'], 'diagnosed_diabetes': submission_xgb_ros})
submission_xgb_ros.to_csv('submission_xgb_ros.csv', index=False)
submission_xgb_ros[:5]


print("Predicting...")
submission_xgb_rus = modelo_xgb_rus.predict_proba(X_test_scaled_rus)[:,1]


submission_xgb_rus = pd.DataFrame({'id': df_sa['id'], 'diagnosed_diabetes': submission_xgb_rus})
submission_xgb_rus.to_csv('submission_xgb_rus.csv', index=False)
submission_xgb_rus[:5]


print("Predicting...")
submission_lgbm_ros = model_lgbm_ros.predict_proba(X_test_scaled)[:,1]


submission_lgbm_ros = pd.DataFrame({'id': df_sa['id'], 'diagnosed_diabetes': submission_lgbm_ros})
submission_lgbm_ros.to_csv('submission_lgbm_ros.csv', index=False)
submission_lgbm_ros[:5]


print("Predicting...")
submission_lgbm_rus = model_lgbm_rus.predict_proba(X_test_scaled_rus)[:,1]


submission_lgbm_rus = pd.DataFrame({'id': df_sa['id'], 'diagnosed_diabetes': submission_lgbm_rus})
submission_lgbm_rus.to_csv('submission_lgbm_rus.csv', index=False)
submission_lgbm_rus[:5]


print("Predicting...")
submission_sgd_ros = model_sgd_ros.predict_proba(X_test_scaled)[:,1]


submission_sgd_ros = pd.DataFrame({'id': df_sa['id'], 'diagnosed_diabetes': submission_sgd_ros})
submission_sgd_ros.to_csv('submission_sgd_ros.csv', index=False)
submission_sgd_ros[:5]


print("Predicting...")
submission_sgd_rus = model_sgd_rus.predict_proba(X_test_scaled_rus)[:,1]


submission_sgd_rus = pd.DataFrame({'id': df_sa['id'], 'diagnosed_diabetes': submission_sgd_rus})
submission_sgd_rus.to_csv('submission_sgd_rus.csv', index=False)
submission_sgd_rus[:5]


print("Predicting...")
submission_ann_ros = ann_ros.predict(X_test_scaled)[:,1]


submission_ann_ros = pd.DataFrame({'id': df_sa['id'], 'diagnosed_diabetes': submission_ann_ros})
submission_ann_ros.to_csv('submission_ann_ros.csv', index=False)
submission_ann_ros[:5]


print("Predicting...")
submission_ann_rus = ann_rus.predict(X_test_scaled_rus)[:,1]


submission_ann_rus = pd.DataFrame({'id': df_sa['id'], 'diagnosed_diabetes': submission_ann_rus})
submission_ann_rus.to_csv('submission_ann_rus.csv', index=False)
submission_ann_rus[:5]

