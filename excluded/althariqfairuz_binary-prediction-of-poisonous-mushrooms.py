import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, matthews_corrcoef


train_df = pd.read_csv('/kaggle/input/playground-series-s4e8/train.csv', index_col = 'id')
test_df = pd.read_csv('/kaggle/input/playground-series-s4e8/test.csv', index_col = 'id')
sample_df = pd.read_csv('/kaggle/input/playground-series-s4e8/sample_submission.csv')


train_df.info()


train_df.head()


train_df.isnull().sum()


test_df.info()


test_df.isnull().sum()


missing_percentage = (train_df.isnull().sum() / len(train_df)) * 100
print(missing_percentage)


copy_train = train_df.copy()


categorical_cols = train_df.select_dtypes(include=['object']).columns
copy_train[categorical_cols]  = copy_train[categorical_cols].fillna('unknown')


copy_train.isnull().sum()


numerical_cols = train_df.select_dtypes(exclude=['object']).columns
for col in numerical_cols:
     mode_value = copy_train[col].mode()[0]
     copy_train[col] = copy_train[col].fillna(mode_value)


# copy_train = copy_train.drop(['spore-print-color', 'veil-color', 'stem-root', 'stem-surface', 'veil-type', 'gill-spacing', 'cap-surface', 'gill-attachment'], axis = 1)


copy_train.isnull().sum()/len(copy_train) * 100


copy_train.head()


X_train, X_val, y_train, y_val = train_test_split(
    copy_train.drop(['class'], axis=1), copy_train['class'], test_size=0.2, random_state=42)


categorical_cols = X_train.select_dtypes(include=['object']).columns

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

X_train_cat_encoded = encoder.fit_transform(X_train[categorical_cols].astype(str))

X_val_cat_encoded = encoder.transform(X_val[categorical_cols].astype(str))

X_train_encoded = X_train.copy()

X_val_encoded = X_val.copy()

X_train_encoded[categorical_cols] = X_train_cat_encoded

X_val_encoded[categorical_cols] = X_val_cat_encoded



le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)


X_train_encoded.head()


y_train_encoded


model = XGBClassifier(
    n_estimators=100,    
    learning_rate=0.1,   
    max_depth=5,         
    random_state=42
)

model.fit(X_train_encoded,y_train_encoded)


y_val_pred = model.predict(X_val_encoded)


matthews_corrcoef(y_val_encoded, y_val_pred)


X_test = test_df.copy()


X_test[categorical_cols] = X_test[categorical_cols].fillna('unknown')
for col in numerical_cols:
     mode_value = X_test[col].mode()[0]
     X_test[col] = X_test[col].fillna(mode_value)


X_test.isnull().sum()


X_test_cat_encoded = encoder.transform(X_test[categorical_cols])
X_test_encoded = X_test.copy()
X_test_encoded[categorical_cols] = X_test_cat_encoded 


X_test_encoded.head()


result = model.predict(X_test_encoded)
result


decoded_result = le.inverse_transform(result)
decoded_result


sample_df.info()


sample_df['class'] = decoded_result


sample_df.to_csv('submission.csv',index=False)

