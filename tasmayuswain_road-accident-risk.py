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


import numpy as np
import pandas as pd 
import xgboost as xgb 
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns 


from sklearn.model_selection import train_test_split, cross_val_score, KFold


from sklearn.preprocessing import LabelEncoder


df_train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test= pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


df_train.head()


print(df_train.isnull().sum())


target= df_train.columns.tolist()[-1]


print(df_train.shape)


# we don't need id as an extra feature 
X= df_train.drop(['id', 'accident_risk'], axis=1).copy()
Y= df_train['accident_risk'].copy()


X_test= df_test.drop(['id'],axis=1).copy()
test_ids = df_test['id'].copy()


print(X.columns.tolist())


plt.figure(figsize=(10,8))
sns.histplot(data=df_train, x="accident_risk")
plt.title('Distribution of Accident risk')
plt.show()


categorical_columns= X.select_dtypes(include=['object']).columns.tolist()
print(categorical_columns)


# now boolean columns to encode
bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
print(f"Boolean columns found: {bool_cols}")

# Then convert them
for col in bool_cols:
    if col in X.columns:
        X[col] = X[col].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0, 1: 1, 0: 0})
        X_test[col] = X_test[col].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0, 1: 1, 0: 0})
        print(f"Converted boolean column '{col}': TRUE=1, FALSE=0")



label_encoders = {}

for col in categorical_columns:
    if col not in bool_cols:  
        le = LabelEncoder()
        # Fit on combined train+test to ensure same encoding
        combined = pd.concat([X[col], X_test[col]], axis=0)
        le.fit(combined)
        
        X[col] = le.transform(X[col])
        X_test[col] = le.transform(X_test[col])
        label_encoders[col] = le
        
        print(f"\nEncoded '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")



X = X.astype(float)
X_test = X_test.astype(float)


print(X.head())


X_train, X_val, Y_train, Y_val= train_test_split(X, Y, test_size=0.2, random_state= 174)


print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")


model= XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=174,
    objective='reg:squarederror'
)


model.fit(X_train,Y_train)


y_val_pred= model.predict(X_val)


from sklearn.metrics import mean_squared_error, r2_score


mse= mean_squared_error(Y_val,y_val_pred)
rmse= np.sqrt(mse)
val_r2 = r2_score(Y_val, y_val_pred)

print(f"Mean squared error: {mse}")
print(f"Root mean squared error: {rmse}")
print(val_r2)


kf= KFold(n_splits=5, shuffle= True , random_state=32)

xgb_model = XGBRegressor(
    learning_rate=0.1,
    max_depth=5,
    n_estimators=200,
    random_state=174
)


# Now we will evaluate the model with negative rmse 
cv_scores_neg_mse = cross_val_score(
    xgb_model, X, Y,
    cv=kf,
    scoring='neg_mean_squared_error',
    
)




cv_score_rmse= np.sqrt(-cv_scores_neg_mse)
print(f"RMSE of CV fold sets are:{cv_score_rmse}")
print(f"RMSE of CV set is:{np.mean(cv_score_rmse)}")


test_predictions=model.predict(X_test)


test_predictions


submission = pd.DataFrame({
    'id': test_ids,
    'accident_risk': test_predictions
})

# saving the csv file 
submission.to_csv('submission.csv', index=False)

