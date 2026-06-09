import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings

warnings.filterwarnings('ignore')


df_train = pd.read_csv('/kaggle/input/mlolympiadbd2025/train.csv')


df_train.head()


df_train.isnull().sum()


df_train.duplicated().sum()


df_train.shape


df_train.info()


df_train.describe()


df_train.select_dtypes(include=['int64' , 'float64']).columns


numerical_cols = ['Age', 'SystolicBP', 'DiastolicBP', 'Blood glucose', 'BodyTemp',
       'HeartRate', 'RiskLevel']
df_train[numerical_cols].hist(bins=20 )
plt.tight_layout()


sns.boxplot(y=df_train['Age'] , x=df_train['RiskLevel'])


sns.heatmap(df_train.corr(numeric_only=True) , cmap='coolwarm')


df_train.head()


df_encode = df_train.drop(columns=['Id' , 'Usage'] , axis=1)


df_encode['HighBP_Flag'] = (df_encode['SystolicBP'] >= 140) | (df_encode['DiastolicBP'] >= 90)
df_encode['HighGlucose_Flag'] = df_encode['Blood glucose'] >= 7.0
df_encode[['HighBP_Flag' , 'HighGlucose_Flag']]   = df_encode[['HighBP_Flag' , 'HighGlucose_Flag']].astype(int)


X = df_encode.drop(columns='RiskLevel' , axis=1)
y = df_encode['RiskLevel']


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import xgboost as xgb 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score , f1_score , confusion_matrix
from sklearn.model_selection import GridSearchCV


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


df_encode


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model_LR = LogisticRegression()
model_LR.fit(X_train , y_train)


y_pred_LR = model_LR.predict(X_test)


print(accuracy_score(y_test , y_pred_LR))


confusion_matrix(y_test , y_pred_LR)


f1_macro = f1_score(y_test, y_pred_LR, average='macro')

# For weighted-average F1 (accounts for class imbalance)
f1_weighted = f1_score(y_test, y_pred_LR, average='weighted')

# For per-class F1
f1_per_class = f1_score(y_test, y_pred_LR, average=None)

print("Macro F1:", f1_macro)
print("Weighted F1:", f1_weighted)
print("Per-class F1:", f1_per_class)


model_xgb = xgb.XGBClassifier()
model_xgb.fit(X_train , y_train)


y_pred_xgb = model_xgb.predict(X_test)
accuracy_score(y_test , y_pred_xgb)


f1_score(y_test , y_pred_xgb , average=None)


model_svm = SVC()
model_svm.fit(X_train , y_train)


y_pred_svm = model_svm.predict(X_test)
accuracy_score(y_test , y_pred_svm)


model_knn = KNeighborsClassifier()
model_knn.fit(X_train , y_train)


y_pred_knn = model_knn.predict(X_test)
accuracy_score(y_test , y_pred_knn)


df_test = pd.read_csv('/kaggle/input/mlolympiadbd2025/test.csv')


df_test_id = df_test['Id']


df_test['HighBP_Flag'] = (df_test['SystolicBP'] >= 140) | (df_test['DiastolicBP'] >= 90)
df_test['HighGlucose_Flag'] = df_test['Blood glucose'] >= 7.0
df_test[['HighBP_Flag' , 'HighGlucose_Flag']]   = df_test[['HighBP_Flag' , 'HighGlucose_Flag']].astype(int)


df_test = scaler.transform(df_test)


predicts = model_xgb.predict(df_test)


predictions = [ 
    {0: 'Low Risk', 1: 'Mid Risk', 2: 'High Risk'}[val] 
    for val in predicts
]



 # submission = pd.DataFrame({
  #  'Id' : df_test_id,
   #  'RiskLevel' : predictions
#}) 
#submission.to_csv('submission.csv' , index=False) 




