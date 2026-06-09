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


from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost  import XGBClassifier
from sklearn.ensemble  import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



df=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')


df.head()


df.isnull().sum()


df.info()


numerical= df.select_dtypes(include=["int64"]).columns
categorical = df.select_dtypes(include=["object"]).columns

print(numerical)
print(categorical)


categorical=['Soil Type','Crop Type']


y_cat=['Fertilizer Name']


df[categorical]


oh= OneHotEncoder(handle_unknown='ignore', sparse=False)
le=LabelEncoder()
sc=StandardScaler( )



def enc(data,tip):
    encoder= oh.fit_transform(data[tip])
    columns_name=oh.get_feature_names_out(data[tip].columns)
    df_encoder=pd.DataFrame(encoder, columns=columns_name)
    return df_encoder


def lb(data,tip):
    encoded_labels = le.fit_transform(data[tip].values.ravel())
    return encoded_labels


def sce(data,tip):
    df_sc=sc.fit_transform(data[tip])
    df_scaled = pd.DataFrame(df_sc, columns=tip)

    return df_scaled


def uni (data1,data2):
    df_dat=pd.concat([data1, data2], axis=1)
    return df_dat


df_enc=enc(df,categorical)
df_lb=lb(df,y_cat)
df_sce=sce(df,numerical)
df_dat= uni(df_sce,df_enc)


df_dat


X=df_dat.drop(columns=['id'])
y=df_lb


X_train,X_val,y_train,y_val=train_test_split(X,y,train_size=0.8,test_size=0.1, random_state=42)


xgb=XGBClassifier(learning_rate=0.1,n_estimators=1000, random_state=7).fit(X_train,y_train)
pret = xgb.predict(X_val)


accuracy = accuracy_score(y_val, pret)
print(accuracy)


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
cm = confusion_matrix(y_val, pret)  # ambas deben estar codificadas

# Mostrar la matriz de confusi贸n con etiquetas decodificadas
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Matriz de Confusi贸n')
plt.tight_layout()
plt.show()

print("Clasification Report:")
print(classification_report(y_val, pret))



df_test=pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


cat=df_test.select_dtypes(include=['object']).columns
num=df_test.select_dtypes(include=['int64']).columns

print(cat)
print(num)


num=['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium','Phosphorous']


df_hot=enc(df_test,cat)
df_num=sce(df_test,num)
df_uni=uni(df_test[num],df_hot)


df_uni


parse_id=df_test['id']

pred_encoded  = xgb.predict(df_uni)
pred_decoded = le.inverse_transform(pred_encoded.astype(int))


parse_id = df_test['id']
df_resultado = pd.DataFrame({
    'id': parse_id,
    'Fertilizer Name': pred_decoded
})
print(df_resultado)
df_resultado.to_csv('cienciano.csv', index=False)
print("Your submission was successfully saved!")

