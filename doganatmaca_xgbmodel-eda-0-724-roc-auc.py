# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
data_test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
pd.set_option('display.max_columns',None)


df.info()


df.diagnosed_diabetes.value_counts()


df.head(3)


df.shape


df = df.drop(['id','ethnicity','education_level','income_level','employment_status'],axis=1)


df['physical_activity_minutes_per_week_category']=pd.cut(df['physical_activity_minutes_per_week'],bins=20,labels=False)
df['age_category']=pd.cut(df['age'],bins=20,labels=False)
df['ldl_cholesterol_category']=pd.cut(df['ldl_cholesterol'],bins=10,labels=False)
df['systolic_bp_category']=pd.cut(df['systolic_bp'],bins=10,labels=False)
df['bmi_category']=pd.cut(df['bmi'],bins=10,labels=False)


df.drop(['smoking_status','gender'],axis=1).corr()['diagnosed_diabetes'].sort_values(ascending=False)


corr_matrix = df.drop(['smoking_status','gender'], axis=1).corr()

plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, 
            annot=True,        
            fmt=".2f",         
            cmap='coolwarm',   
            square=True,       
            cbar_kws={'shrink':0.8})  
plt.title("Correlation Matrix")
plt.show()



sns.countplot(x=df['diagnosed_diabetes'])


x = df[['family_history_diabetes','age','age_category','physical_activity_minutes_per_week_category','gender','smoking_status','physical_activity_minutes_per_week',
       'systolic_bp','bmi','ldl_cholesterol','systolic_bp_category','bmi_category','ldl_cholesterol_category','triglycerides','cholesterol_total',
       'waist_to_hip_ratio']]
y = df['diagnosed_diabetes']


x_scale = pd.get_dummies(x,drop_first=True)


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


pos = (y == 1).sum()
neg = (y == 0).sum()
scale = neg / pos

# Veri böl
X_train, X_test, y_train, y_test = train_test_split(x_scale, y, test_size=0.2, random_state=42)


# Model oluştur
model = xgb.XGBClassifier(
    n_estimators=2000,
    max_depth=3,
    learning_rate=0.09,
    objective='binary:logistic',
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=scale,
    use_label_encoder=False,
    eval_metric='auc'
)

# Modeli eğit
model.fit(X_train, y_train)

# Tahmin ve AUC
y_proba = model.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, y_proba)
print("ROC-AUC:", auc)



from sklearn.metrics import roc_curve, auc

# ROC eğrisi
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Tahminleri 0/1 olarak al
y_pred = (y_proba >= 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()



feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
feat_importances.nlargest(20).plot(kind='barh', figsize=(10,8))
plt.title("Top 20 Feature Importances")
plt.show()



df_test = data_test.drop(['id','ethnicity','education_level','income_level','employment_status'],axis=1)


df_test['physical_activity_minutes_per_week_category']=pd.cut(df_test['physical_activity_minutes_per_week'],bins=20,labels=False)
df_test['age_category']=pd.cut(df_test['age'],bins=20,labels=False)
df_test['ldl_cholesterol_category']=pd.cut(df_test['ldl_cholesterol'],bins=10,labels=False)
df_test['systolic_bp_category']=pd.cut(df_test['systolic_bp'],bins=10,labels=False)
df_test['bmi_category']=pd.cut(df_test['bmi'],bins=10,labels=False)


x2 = df_test[['family_history_diabetes','age','age_category','physical_activity_minutes_per_week_category','gender','smoking_status','physical_activity_minutes_per_week',
       'systolic_bp','bmi','ldl_cholesterol','systolic_bp_category','bmi_category','ldl_cholesterol_category','triglycerides','cholesterol_total',
       'waist_to_hip_ratio']]


x2_scaled = pd.get_dummies(x2,drop_first=True)


predd = model.predict_proba(x2_scaled)[:,1]


result=pd.DataFrame()
result['id']=data_test['id']
result['diagnosed_diabetes']=predd
result=result.set_index('id')


result.head()

