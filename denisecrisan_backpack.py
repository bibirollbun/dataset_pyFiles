import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))





trainData = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
testData = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")

trainData.describe()




trainData.columns


features = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof',
            'Style', 'Color', 'Compartments', 'Weight Capacity (kg)']

trainData = trainData.dropna(axis=0)

# Datenvisualisierung: relevante Plots
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(trainData["Price"], bins=30, kde=True)
plt.title("Verteilung der Rucksack-Preise")
plt.xlabel("Preis")
plt.ylabel("Anzahl")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=trainData, x='Brand', y='Price')
plt.xticks(rotation=45)
plt.title("Preisverteilung pro Marke")
plt.show()


y = trainData.Price
X = trainData[features]
X_test = testData[features]

X_encoded = pd.get_dummies(X)
X_test_encoded=pd.get_dummies(X_test)





X.head()


# error 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

X_test_encoded = X_test_encoded.reindex(columns=X_encoded.columns, fill_value=0)
X_test_encoded = X_test_encoded.fillna(0)
train_X, val_X, train_y, val_y = train_test_split(X_encoded, y, random_state = 1)
backpack_model = DecisionTreeRegressor()
backpack_model.fit(train_X, train_y)

val_predictions = backpack_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

print(backpack_model.predict(X_encoded.head()))
print(y.head())


predictions = backpack_model.predict(X_test_encoded)
output = pd.DataFrame({'id': testData.id, 'Price': predictions})
output.to_csv('baseline_submission.csv', index=False)
print(output)



