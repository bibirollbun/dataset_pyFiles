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


data = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')


data.size


data.head()


data.isnull().sum()


data = data.dropna()


data.isnull().sum()


data.size


percent_data_loss = ((166716 - 91701)/166716 ) * 100
percent_data_loss


#fill missing values with mean and mode
data = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')


data = data.drop('id', axis = 1)


data.nunique()


#fill missing values with mode
for i in data.columns:
    mode = data[i].mode()[0]
    data[i].fillna(mode, inplace = True)


#label encoding
from sklearn.preprocessing import LabelEncoder


cols_to_encode = ['Stage_fear', 'Drained_after_socializing', 'Personality']
df = data
label_encoders = {}
encoding_maps = {}

for col in cols_to_encode:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    encoding_maps[col] = dict(zip(le.classes_, le.transform(le.classes_)))

# Display encoding map
for col, mapping in encoding_maps.items():
    print(f"Encoding for column '{col}':")
    for original, encoded in mapping.items():
        print(f"  '{original}' → {encoded}")
    print()



data.head()


import seaborn as sns
corr = data.corr()
sns.heatmap(corr, annot=True)


from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Split features and target
X = df.drop(columns=['Personality'])
y = df['Personality']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)



#random forest algorithm
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print(classification_report(y_test, y_pred_rf))


# Train LightGBM
model = LGBMClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoders['Personality'].classes_))


from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy}")


