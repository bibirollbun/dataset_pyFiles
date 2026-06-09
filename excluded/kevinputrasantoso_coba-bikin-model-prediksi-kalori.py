############################
# Disini, aku bakal make beberapa pendekatan:
# 1. Loading Data
# 2. Preprocessing (normalisasi, one-hot encoding, dsb)
# 3. Penetapan model
# 4. Splitting data jadi train split, val split, dan test split
# 5. Training
# 6. Prediksi pada test set
# 7. Evaluasi
###########################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")


df.head(5)


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])


df.drop(columns=['id'], inplace=True)


def minmax_scaling(data):
    data_min = data.min()
    data_max = data.max()
    data = (data - data_min) / (data_max - data_min)
    return data

def standardize(data):
    data_mean = data.mean()
    data_std = data.std()
    data = (data - data_mean) / data_std
    return data


df_copy = df.copy()


df_copy['Age'] = standardize(df_copy['Age'])
df_copy['Height'] = standardize(df_copy['Height'])
df_copy['Weight'] = standardize(df_copy['Weight'])
df_copy['Duration'] = standardize(df_copy['Duration'])
df_copy['Heart_Rate'] = standardize(df_copy['Heart_Rate'])
df_copy['Body_Temp'] = standardize(df_copy['Body_Temp'])


import seaborn as sns

sns.heatmap(df_copy.corr(), annot=True, cmap='coolwarm', fmt=".2f")


df_copy.drop(columns=['Sex', 'Height'], inplace=True)


sns.heatmap(df_copy.corr(), annot=True, cmap='coolwarm', fmt=".2f")


model_linreg = LinearRegression()


X = df_copy[[col for col in list(df_copy.columns)[:-1]]]
y = df_copy['Calories']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)


model_linreg.fit(X_train, y_train)


y_pred = model_linreg.predict(X_val)


from sklearn.metrics import mean_squared_error, r2_score

r2 = r2_score(y_val, y_pred)


plt.plot(y_val.reset_index()['Calories'][0:100])
plt.plot(y_pred[0:100])




