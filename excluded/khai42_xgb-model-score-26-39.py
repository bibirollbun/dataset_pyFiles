import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb


train_df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


print('Train data: ',train_df.shape)
display(train_df.head(3))
print()
print('Test data: ',test_df.shape)
display(test_df.head(3))


X_train = train_df.drop(columns=['id', 'BeatsPerMinute'])
y_train = train_df['BeatsPerMinute'] 
X_test = test_df.drop(columns=['id'])

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(
    n_estimators=1000,      
    learning_rate=0.05,     
    max_depth=6,            
    colsample_bytree=0.8,   
    subsample=0.8,          
    random_state=42
)

model.fit(X_train_split, y_train_split, early_stopping_rounds=50, eval_metric="rmse", 
          eval_set=[(X_val_split, y_val_split)], verbose=True)

y_val_pred = model.predict(X_val_split)
rmse = np.sqrt(mean_squared_error(y_val_split, y_val_pred))
print(f"Val RMSE: {rmse}")

y_test_pred = model.predict(X_test)


submission = pd.DataFrame({'id': test_df['id'],'BeatsPerMinute': y_test_pred})
submission.to_csv('submission.csv', index=False)
print(submission.head(3))




