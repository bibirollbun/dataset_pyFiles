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


train_path = "/kaggle/input/playground-series-s5e3/train.csv"
test_path = "/kaggle/input/playground-series-s5e3/test.csv"

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

train_data.head()


test_data.head()


#X = train_data.drop(columns=['rainfall', 'id', 'day']) 
X = train_data[['humidity', 'sunshine', 'dewpoint', 'pressure', 'cloud', 'temparature']]
y = train_data['rainfall'] 


from sklearn.model_selection import train_test_split

# Split the data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Видаляємо ті ж самі колонки з test_data
test_features = test_data.drop(columns=['id', 'day'])
test_features = test_features[X_train.columns]

# Check the size
print(X_train.shape, X_test.shape)



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

for n in [50, 100, 200, 300]:
    
    model = RandomForestClassifier(n_estimators=n, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"n_estimators: {n}, Accuracy: {accuracy:.4f}")

    

    importances = model.feature_importances_
    feature_names = X_train.columns



# Сортуємо та виводимо найважливіші фічі
sorted_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
for feature, importance in sorted_features:
    print(f"{feature}: {importance:.4f}")


#from sklearn.metrics import accuracy_score

#y_pred = model.predict(X_test)
#accuracy = accuracy_score(y_test, y_pred)

#print(f"n_estimators: {n}, Accuracy: {accuracy:.4f}")


print(test_features.dtypes)
print(X_train.dtypes)


print(test_features.isnull().sum())


test_features = test_features.fillna(test_features.mean())


# 1. Робимо передбачення для тестового набору Kaggle
test_predictions = model.predict(test_features)

# 2. Формуємо submission DataFrame
submission = pd.DataFrame({
    "id": test_data["id"],  # ID з тестового набору
    "rainfall": test_predictions  # Передбачені значення
})

# 3. Зберігаємо у CSV
submission.to_csv("submission.csv", index=False)
print("Submission file saved successfully!")


!ls /kaggle/working/


!head submission.csv

