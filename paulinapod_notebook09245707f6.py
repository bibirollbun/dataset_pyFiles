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


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Krok 1: Wczytywanie danych
train_data = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv') 
test_data = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')  




train_data.info()


test_data.info()


train_data.describe()


train_data.head()


print(train_data.isna().sum())  # Liczba NaN w każdej kolumnie
print(np.isinf(train_data).sum())  # Liczba inf w każdej kolumnie


train_data.fillna(train_data.mean(), inplace=True)


print(train_data.isna().sum())  # Liczba NaN w każdej kolumnie


print(list(train_data.columns))


train_data.head()


test_data.head()


# Krok 2: Przygotowanie zbioru danych
features = ["id", "Episode_Length_minutes","Host_Popularity_percentage","Guest_Popularity_percentage", 
            "Number_of_Ads"]
# Przygotowanie zbioru treningowego
X_train = pd.get_dummies(train_data[features])
y_train = train_data["Listening_Time_minutes"]


from sklearn.model_selection import train_test_split
# Podział na zbiory treningowy i walidacyjny (80% trening, 20% walidacja)
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)




# Wydzielenie 5 próbek z zestawu walidacyjnego
sample_val_data = X_val.sample(n=5, random_state=42)
print("Próbki z zestawu walidacyjnego:")
print(sample_val_data)


from sklearn.ensemble import RandomForestRegressor



rf_model = RandomForestRegressor(n_estimators=20)


rf_model.fit(X_train_split, y_train_split)


# Ocena modelu
y_val_pred = rf_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f'RMSE na zbiorze walidacyjnym: {rmse}')


X_test = pd.get_dummies(test_data[features])
X_test = X_test.reindex(columns=X_train_split.columns, fill_value=0)


predictions = rf_model.predict(X_test)


print(X_test.isna().sum())


test_data.head()


# Krok: Usuwanie zbednych wartosci
test_data.drop(['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment'], axis=1,inplace=True)


print(train_data.isna().sum())  # Liczba NaN w każdej kolumnie
print(np.isinf(train_data).sum())  # Liczba inf w każdej kolumnie


X_test = pd.get_dummies(test_data[features])
X_test = X_test.reindex(columns=X_train_split.columns, fill_value=0)


# Przygotowanie zbioru testowego
X_test = pd.get_dummies(test_data[features])

# Ujednolicenie kolumn, aby upewnić się, że obie zbiory mają tę samą strukturę
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


predictions = rf_model.predict(X_test)


print(X_test.isna().sum())

