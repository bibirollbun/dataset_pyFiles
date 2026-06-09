import pandas as pd
import numpy as np
import matplotlib.pyplot as pl


train_df_1= pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
train_df_1.drop('id',axis=1,inplace = True)
train_df_2=pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
train_df_2.drop('id',axis=1,inplace=True)
test_df= pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
test_id = test_df['id']
test_df.drop('id',axis=1,inplace = True)



train_df = pd.concat([train_df_1, train_df_2], axis=0, ignore_index=True)


train_df.shape


train_df.info()


train_df.isnull().sum()


for col in train_df.columns:
    if col == "Price":  # Skip target column
        continue
    if train_df[col].dtype == 'object':  # Categorical columns
        train_df[col].fillna("Unknown", inplace=True)
        test_df[col].fillna("Unknown", inplace=True)
    else:  # Numerical columns
        train_df[col].fillna(train_df[col].median(), inplace=True)
        if col in test_df.columns:  # Only fill if the column exists in test_data
            test_df[col].fillna(test_df[col].median(), inplace=True)


train_df.isnull().sum()


#encoding
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

for col in categorical_cols:
    train_df[col] = encoder.fit_transform(train_df[col])
    test_df[col] = encoder.fit_transform(test_df[col])



train_df.head()


test_df.head()


train_df.info()


test_df.info()


X = train_df.drop('Price', axis=1)
y = train_df['Price']


from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error 
model = XGBRegressor( objective='reg:squarederror',
    random_state=42,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    enable_categorical=True)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("Validation Mean Absolute Error (MAE):", mae)


test_predictions = model.predict(test_df)

# Create a submission file
submission = pd.DataFrame({'id':test_id, 'Price': test_predictions})
submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'.")

