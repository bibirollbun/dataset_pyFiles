import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import re
import missingno as msno


test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')



test_df.head()


test_df.tail()


test_df.shape


train_df.head()


train_df.tail()


train_df.shape


train_df=train_df.drop('id',axis=1)


train_df.shape


test_df=test_df.drop('id',axis=1)


test_df.shape


train_df['Soil Type'].unique()


train_df['Crop Type'].unique()


test_df['Soil Type'].unique()


test_df['Crop Type'].unique()


X_train = train_df.drop(columns=["Fertilizer Name"]) #contains fertilizer info only
Y_train = train_df["Fertilizer Name"]                #contains fertilizer names only
X_test = test_df


combine = pd.concat([X_train,X_test] ,axis=0)


combined_encoded=pd.get_dummies(combine, columns=['Soil Type','Crop Type'],dtype=int, drop_first=True)



combined_encoded.shape


combined_encoded.head()


combined_encoded.tail()


combined_encoded.to_csv('combined_encoded.csv', index=False)

from IPython.display import FileLink
FileLink('combined_encoded.csv')



X_train_encoded = combined_encoded.iloc[:750000, :]
X_test_encoded = combined_encoded.iloc[750000:, :]


X_train_encoded.tail()


X_test_encoded.tail()


Y_train.to_csv('fertilizers.csv', index=False)

from IPython.display import FileLink
FileLink('fertilizers.csv')


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
Y_train_encoded = le.fit_transform(Y_train)


print(Y_train_encoded)


len(Y_train_encoded)


from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    eval_metric='mlogloss'
)

model.fit(X_train_encoded, Y_train_encoded)




y_preds = model.predict(X_test_encoded)


print(y_preds)


decoded_preds = le.inverse_transform(y_preds)


print(len(X_train_encoded), len(Y_train))  # Should match the size of your full training data
print(set(Y_train))  # How many unique fertilizer names



predictions = pd.DataFrame({'Fertilizer Name': decoded_preds})
predictions.insert(0, 'id', range(750000, 750000 + len(predictions)))
predictions.to_csv('submission.csv', index=False)
from IPython.display import FileLink
FileLink('submission.csv')


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_tr, X_val, Y_tr, Y_val = train_test_split(X_train_encoded, Y_train_encoded, test_size=0.2, random_state=42)

model.fit(X_tr, Y_tr)
y_preds = model.predict(X_val)

acc = accuracy_score(Y_val, y_preds)
print("Validation Accuracy:", acc)





