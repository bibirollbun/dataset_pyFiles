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



from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from xgboost import XGBClassifier







train=pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
train.head()


test=pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
test.head()
#X_test=test


train.isna().sum()
train.info()


train.describe()


train.columns





for df in [train, test]:
    df['N_to_P'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-5)
    df['N_to_K'] = df['Nitrogen'] / (df['Potassium'] + 1e-5)
    df['P_to_K'] = df['Phosphorous'] / (df['Potassium'] + 1e-5)
    df['Total_NPK'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
    df['Climate_Index'] = (df['Temparature'] + df['Humidity']) / 2
    df['Water_Stress'] = df['Humidity'] - df['Moisture']
    df['moistemp']=df['Moisture']- df['Temparature']





categorical_features = ['Soil Type', 'Crop Type'] 
# One-hot encode and update the original DataFrame
train_encoded = pd.get_dummies(train, columns=categorical_features)
train_encoded
test_encoded = pd.get_dummies(test, columns=categorical_features)
test_encoded





train_encoded



train_encoded.columns



X_test=train_encoded.drop(['Fertilizer Name'],axis=1)

y=train_encoded['Fertilizer Name']


x_test=test_encoded


y=le.fit_transform(y)


scaler = StandardScaler()
X_scale = scaler.fit_transform(X_test)



x_scale = scaler.transform(x_test)


x_scale


x_train,x_test,y_train,y_test=train_test_split(X_scale,y,train_size=0.7)


import xgboost as xgb
import lightgbm as lgb



# ========== Train XGBoost ==========
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss',gamma=0,
    learning_rate=0.01,
    max_depth=5,
    min_child_weight=3,
    subsample=1.0,
    n_estimators=500,        
    tree_method='hist',      
    n_jobs=-1,               
    verbosity=1 ,
    enable_categorical=True# Optional: show training progress
)
xgb_model.fit(X_scale, y)

# ========== Train LightGBM ==========
lgb_model = lgb.LGBMClassifier(learning_rate=0.05,        
    num_leaves=31,              
    max_depth=-1,              
    min_child_samples=20,       
    subsample=0.8,              
    colsample_bytree=0.8,       
    reg_alpha=0.0,              
    reg_lambda=0.0,             
    n_estimators=100,          
    random_state=42 ) 

lgb_model.fit(X_scale, y)

# ========== Predict Probabilities ==========
xgb_probs = xgb_model.predict_proba(x_scale)
lgb_probs = lgb_model.predict_proba(x_scale)

# ========== Average the Probabilities ==========
avg_probs = (xgb_probs + lgb_probs) / 2

# ========== Get Top-3 Predictions ==========
top3_indices = np.argsort(avg_probs, axis=1)[:, -3:][:, ::-1]  # shape: (n_samples, 3)
print(top3_indices)

# ========== Decode to Original Labels ==========
decoded_top3 = le.inverse_transform(top3_indices.ravel()).reshape(top3_indices.shape)

# ========== View Results ==========
print("Top-3 Predictions per Sample:\n", decoded_top3)





print(top3_indices)



output = pd.DataFrame({'id': test.id, 'Fertilizer Name': [' '.join(row) for row in decoded_top3]})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


submission=pd.read_csv('submission.csv')


submission

