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


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier,LGBMRegressor
from xgboost import XGBClassifier,XGBRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,r2_score
train=pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
##submission=pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


train=train.drop(columns=['id'])
le=LabelEncoder()
train['Sex']=le.fit_transform(train['Sex'])
test['Sex']=le.fit_transform(test['Sex'])


y = train['Calories'] 
X=train.drop(['Calories'],axis=1)


X['Height_m'] = X['Height'] / 100
X['BMI'] = X['Weight'] / (X['Height_m'] ** 2)
X.drop('Height_m', axis=1, inplace=True)



import matplotlib.pyplot as plt
plt.hist(X['BMI'], bins=50)
plt.title('BMI dağılımı')
plt.show()



X = X.copy()
y_log = np.log1p(y)
numeric_cols = X.select_dtypes(include=[np.number]).columns
positive_cols = [col for col in numeric_cols if (X[col] > 0).all()]
X[positive_cols] = X[positive_cols].apply(np.log1p)
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)


# Eğer train'de 'id' sütunu varsa kaldır
if 'id' in train.columns:
    train = train.drop(columns=['id'])

# Height birimini metreye çevir ve BMI hesapla
train['Height_m'] = train['Height'] / 100
train['BMI'] = train['Weight'] / (train['Height_m'] ** 2)
train = train.drop(columns=['Height_m'])



test['Height_m'] = test['Height'] / 100
test['BMI'] = test['Weight'] / (test['Height_m'] ** 2)
test.drop(columns=['Height_m'], inplace=True)



from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.model_selection import KFold


##params = {
  ##  'iterations': 1050,  
    ##'learning_rate': 0.02,  
    ##'depth': 8,
    ##'loss_function': 'RMSE',
    #'eval_metric': 'RMSE',
    #'random_seed': 42,
    #'od_type': 'Iter',  
    #'od_wait': 50,
    #'border_count': 254,  
    #'grow_policy': 'Lossguide',
    #'used_ram_limit': '8gb',  
    #'task_type': 'CPU'
#}

#X_train, X_val, y_train_log, y_val_log = train_test_split(X, y_log, test_size=0.2, random_state=42)
#kf = KFold(n_splits=5, shuffle=True, random_state=42)

#rmse_scores = []
#best_iterations = []

#for train_idx, val_idx in kf.split(X_train):
 #   X_tr, X_kval = X_train.iloc[train_idx], X_train.iloc[val_idx]
  #  y_tr, y_kval = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]
   # model = CatBoostRegressor(**params)
    #model.fit(
     #   X_tr, y_tr,
      #  eval_set=(X_kval, y_kval),
       # early_stopping_rounds=62,
        #use_best_model=True,
        #verbose=100
    #)
    #preds = model.predict(X_kval)
    #rmse = np.sqrt(np.mean((y_kval - preds) ** 2))
    #rmse_scores.append(rmse)
    #best_iterations.append(model.get_best_iteration())

#best_iter = int(np.mean(best_iterations))
#print(f"En iyi iterasyon sayısı (K-Fold ile): {best_iter}")
#print(f"Katlama başına RMSE: {rmse_scores}")
#print(f"Ortalama RMSE: {np.mean(rmse_scores)}")



#X_full_train = np.concatenate((X_train, X_val))
#y_full_train = np.concatenate((y_train_log, y_val_log))

#final_model = CatBoostRegressor(
 #   iterations=best_iter,
  #  learning_rate=0.05,
   # depth=7,
    #loss_function='RMSE',
    #eval_metric='RMSE',
    #random_seed=42,
    #verbose=100
#)
#final_model.fit(
 #   X_full_train, y_full_train,
  #  eval_set=(X_val, y_val_log),
   # use_best_model=True,
   # verbose=100
#)



from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import numpy as np

params = {
    'iterations': 1500,
    'learning_rate': 0.05,
    'depth': 8,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'od_type': 'Iter',
    'od_wait': 50,
    'task_type': 'CPU',
    'verbose': 100
}

X_tr, X_val, y_tr, y_val = train_test_split(X, y_log, test_size=0.2, random_state=42)

model = CatBoostRegressor(**params)
model.fit(
    X_tr, y_tr,
    eval_set=(X_val, y_val),
    early_stopping_rounds=50,
    use_best_model=True,
    verbose=100
)

preds_val = model.predict(X_val)
rmse_val = np.sqrt(np.mean((y_val - preds_val) ** 2))
print(f"Validation RMSE: {rmse_val}")



preds_log = model.predict(X_tr)
preds = np.expm1(preds_log)
y_true = np.expm1(y_tr)
rmsle = np.sqrt(mean_squared_log_error(y_true, preds))
print(f"RMSLE (orijinal ölçekte): {rmsle:.5f}")



X_train, X_val, y_train_log, y_val_log = train_test_split(X, y_log, test_size=0.2, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
X_full_train = np.concatenate((X_train, X_val))
y_full_train = np.concatenate((y_train_log, y_val_log))




test_cleaned = test.drop(columns=["id"])


numeric_cols = X.select_dtypes(include=[np.number]).columns
positive_cols = [col for col in numeric_cols if (X[col] > 0).all()]
X[positive_cols] = X[positive_cols].apply(np.log1p)
test_cleaned = test.drop(columns=["id"])
test_cleaned[positive_cols] = test_cleaned[positive_cols].apply(np.log1p)



prediction=model.predict(test_cleaned)
prediction=np.expm1(prediction)


submission=pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
submission = pd.DataFrame({
    "id": test["id"],          
    "Calories": prediction     
})
submission.to_csv("submission.csv", index=False)





