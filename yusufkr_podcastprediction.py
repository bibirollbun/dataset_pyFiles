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


import seaborn as sns
import matplotlib.pyplot as plt


trainDf = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')


trainDf


trainDf.shape


trainDf.info()


print(trainDf.columns)


print(trainDf.describe())


print(trainDf.isnull().sum())


# Count non-null values in each column
non_null_counts = trainDf.notna().sum()

# Create the bar plot
plt.figure(figsize=(12, 6))
non_null_counts.plot(kind='bar', color='lightcoral')

# Add title and labels
plt.title('Count of Non-Null Values in Each Column')
plt.xlabel('Columns')
plt.ylabel('Non-Null Value Count')

# Display the plot
plt.xticks(rotation=90)  # Rotate x-axis labels
plt.tight_layout()  # Adjust layout to prevent label cutoff
plt.show()


trainDf['Episode_Length_minutes'] = trainDf['Episode_Length_minutes'].fillna(trainDf['Episode_Length_minutes'].median())


trainDf['Episode_Length_minutes'].isnull().sum()


trainDf['Guest_Popularity_percentage'] = trainDf.groupby('Genre')['Guest_Popularity_percentage'].transform(lambda x: x.fillna(x.mean()))


trainDf['Guest_Popularity_percentage'].isnull().sum()


trainDf['Number_of_Ads'] = trainDf['Number_of_Ads'].fillna(trainDf['Number_of_Ads'].mean())


trainDf['Number_of_Ads'].isnull().sum()


trainDf.isnull().sum()


# Detect outliers for each column
for column in trainDf.columns:
    if trainDf[column].dtype in ['int64', 'float64']:  # Check if numeric column
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=trainDf, x=column)
        plt.title(f'Outliers in {column} Column')
        plt.show()


q1 = trainDf['Episode_Length_minutes'].quantile(0.25)
q3 = trainDf['Episode_Length_minutes'].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = trainDf[trainDf['Episode_Length_minutes'] > upper_bound]

# outlier olmayanların ortalamasını al
mean_value = trainDf[trainDf['Episode_Length_minutes'] <= upper_bound]['Episode_Length_minutes'].mean()

# outlier'ı ortalama ile değiştir
trainDf.loc[trainDf['Episode_Length_minutes'] > upper_bound, 'Episode_Length_minutes'] = mean_value


trainDf["Number_of_Ads"] = trainDf["Number_of_Ads"].apply(lambda x: x if x in [0,1,2,3] else 3)


trainDf.isnull().sum()


trainDf.info()


from sklearn.preprocessing import LabelEncoder


list(trainDf['Genre'].unique())


le = LabelEncoder()


trainDf["Genre"] = le.fit_transform(trainDf["Genre"])


trainDf['Genre'].head(10)


trainDf["Episode_Title"] = le.fit_transform(trainDf["Episode_Title"])


day_order = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

trainDf["Publication_Day"] = trainDf["Publication_Day"].map(day_order)


trainDf['Publication_Time'].unique()


trainDf["Publication_Time"] = le.fit_transform(trainDf["Publication_Time"])


trainDf['Publication_Time'].unique()


trainDf["Podcast_Name"] = le.fit_transform(trainDf["Podcast_Name"])


trainDf["Podcast_Name"].unique().sum()


trainDf["Episode_Sentiment"] = le.fit_transform(trainDf["Episode_Sentiment"])


print(trainDf["Number_of_Ads"].value_counts())


trainDf


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

trainDf['Episode_Length_minutes'] = scaler.fit_transform(trainDf[['Episode_Length_minutes']])


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
trainDf['Host_Popularity_percentage'] = scaler.fit_transform(trainDf[['Host_Popularity_percentage']])


trainDf['Guest_Popularity_percentage'] = scaler.fit_transform(trainDf[['Guest_Popularity_percentage']])


trainDf


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


X = trainDf.drop(columns=['id','Listening_Time_minutes'])
y = trainDf['Listening_Time_minutes']


X


y


print(type(X))  # DataFrame mi array mi gör
print(type(y))  # Aynı tipte mi diye bak


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


'''
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

# Verinin %10'unu örnek olarak ayır
X_sample, _, y_sample, _ = train_test_split(
    X, y,
    train_size=0.1,
    random_state=42,
    # stratify=y  # Sınıflandırma problemlerinde kullanılır, regresyonda genellikle gerek yok
)

# Model nesnesi
rf = RandomForestRegressor(random_state=42)

# Hiperparametre arama için grid tanımı
param_grid = {
    'n_estimators': [200, 300, 400, 500],        # Ağaç sayısı
    'max_depth': [10, 20],              # Maksimum derinlik
    'min_samples_split': [2, 5],        # Bir node'u bölmek için gereken minimum örnek sayısı
}

# GridSearchCV nesnesi
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,                                   # 3 katlı çapraz doğrulama
    scoring='neg_mean_squared_error',       # Negatif MSE ile değerlendirme
    n_jobs=-1,                              # Tüm işlemcileri kullan
    verbose=2                               # İşlem adımlarını detaylı göster
)

# GridSearchCV'yi örnek veri üzerinde çalıştır
grid_search.fit(X_sample, y_sample)

# Sonuçları yazdır
print("\nEn İyi Parametreler:", grid_search.best_params_)
print("En İyi Negatif MSE Skoru:", grid_search.best_score_)
print("En İyi MSE (Pozitif):", abs(grid_search.best_score_))
'''


rfr = RandomForestRegressor(
    n_estimators=600,
    max_depth=None,
    min_samples_split=2,
    max_features=None,
    random_state=42,
    n_jobs=-1
)


rfr.fit(X_train, y_train)


y_pred = rfr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)


mse


import math
rmse = math.sqrt(mse)


rmse


import statsmodels.api as sm


X_train_ols = sm.add_constant(X_train)


X_train_ols


sm_model = sm.OLS(y_train,X_train_ols)


sonuc = sm_model.fit()


print(sonuc.summary())


from sklearn.metrics import r2_score
# R^2 hesaplama
r2 = r2_score(y_test, y_pred)
print("R^2 Skoru:", r2)


testDf = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


testDf.isnull().sum()


testDf['Episode_Length_minutes'] = testDf['Episode_Length_minutes'].fillna(testDf['Episode_Length_minutes'].median())
testDf['Guest_Popularity_percentage'] = testDf.groupby('Genre')['Guest_Popularity_percentage'].transform(lambda x: x.fillna(x.mean()))


testDf.isnull().sum()


import matplotlib.pyplot as plt
import seaborn as sns
# Detect outliers for each column
for column in testDf.columns:
    if testDf[column].dtype in ['int64', 'float64']:  # Check if numeric column
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=testDf, x=column)
        plt.title(f'Outliers in {column} Column')
        plt.show()


# Uygula
testDf["Genre"] = le.fit_transform(testDf["Genre"])
testDf["Episode_Title"] = le.fit_transform(testDf["Episode_Title"])


gun_sirasi = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

testDf["Publication_Day"] = testDf["Publication_Day"].map(gun_sirasi)


testDf["Publication_Time"] = le.fit_transform(testDf["Publication_Time"])


testDf["Podcast_Name"] = le.fit_transform(testDf["Podcast_Name"])
testDf["Episode_Sentiment"] = le.fit_transform(testDf["Episode_Sentiment"])


# 0-3 dışındaki değerleri 3 olarak kabul et (örnek)
testDf["Number_of_Ads"] = testDf["Number_of_Ads"].apply(lambda x: x if x in [0,1,2,3] else 3)


q1 = testDf['Episode_Length_minutes'].quantile(0.25)
q3 = testDf['Episode_Length_minutes'].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = testDf[testDf['Episode_Length_minutes'] > upper_bound]

# outlier olmayanların ortalamasını al
mean_value = testDf[testDf['Episode_Length_minutes'] <= upper_bound]['Episode_Length_minutes'].mean()

# outlier'ı ortalama ile değiştir
testDf.loc[testDf['Episode_Length_minutes'] > upper_bound, 'Episode_Length_minutes'] = mean_value


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

testDf['Episode_Length_minutes'] = scaler.fit_transform(testDf[['Episode_Length_minutes']])


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
testDf['Host_Popularity_percentage'] = scaler.fit_transform(testDf[['Host_Popularity_percentage']])
testDf['Guest_Popularity_percentage'] = scaler.fit_transform(testDf[['Guest_Popularity_percentage']])


testDf


x_test = testDf.drop(columns=['id'])


y_pred = rfr.predict(x_test)


y_pred[0:10]


submission_df = pd.DataFrame({
    'id': testDf['id'],
    'Listening_Time_minutes': y_pred
}) 


submission_df.to_csv('submission.csv', index=False)
submission_df.head()

