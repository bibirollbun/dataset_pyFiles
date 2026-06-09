# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline

#Two lines Required to Plot Plotly
import plotly.io as pio
pio.renderers.default = 'iframe'

import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


train.tail()


train.info()


test.info()


train.describe().loc[['mean','min','max']].T


#By Pasindu Dilmin https://www.kaggle.com/code/pazindushane/fertilizers-recommendation

labels = train["Fertilizer Name"].unique()
counts = list(train["Fertilizer Name"].value_counts())

plt.figure(figsize = (9,5))
plt.barh(labels, counts)
  
for index, value in enumerate(counts):
    plt.text(value, index,
             str(value))
plt.title('Fertilizer Name')    
plt.show()


#By Rob Mulla https://www.kaggle.com/code/robikscube/sign-language-recognition-eda-twitch-stream

fig, ax = plt.subplots(figsize=(4, 4))
train["Crop Type"].value_counts().head().sort_values(ascending=True).plot(
    kind="barh",color='g', ax=ax, title="Crop type"
)
ax.set_xlabel("Number of Training Examples")
plt.show()


labels = 'Sandy', 'Black', 'Clayey', 'Red', 'Loamy'
sizes = [156710, 150956, 148382, 148102, 145850]  #must have same number labels, sizes and explode
explode = (0, 0.2, 0, 0, 0)  # only "explode" the 2nd slice 

fig1, ax1 = plt.subplots(figsize=(8,8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title('Soil Type')
plt.show()


#Lucas_DatAArtist https://www.kaggle.com/code/lucasdataartist/eda-prediction-of-obesity-risk

# correlation matrix
plt.figure(figsize = (8, 4), facecolor = "white")

# plotting
sns.heatmap(
    data = train.corr(numeric_only = True),
    cmap = "summer",
    vmin = -1, vmax = 1,
    linecolor = "white", linewidth = 0.5,
    annot = True,
    fmt = ".2f"
)

plt.title('Correlation Heatmap')
plt.show()


#relation of Crop type and Temperature with output variable
plt.figure(figsize=(15,6))
sns.boxplot(x=train['Crop Type'],y=train['Temparature'],hue=train['Fertilizer Name']);


#relation of crop type with Humidity
plt.figure(figsize=(15,8))
sns.boxplot(x=train['Crop Type'],y=train['Humidity'], hue=train['Fertilizer Name']);


#relation of soil type and Temperature with output variable
plt.figure(figsize=(15,6))
sns.boxplot(x=train['Soil Type'],y=train['Temparature'],hue=train['Fertilizer Name']);


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report


y = train['Fertilizer Name'].copy()
X = train.drop('Fertilizer Name', axis=1).copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)


##Gabriel Atkin https://www.kaggle.com/code/gcdatkin/fertilizer-type-prediction

nominal_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(sparse=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('nominal', nominal_transformer, ['Soil Type', 'Crop Type'])
], remainder='passthrough')

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])


model.fit(X_train, y_train)


print("Test Accuracy: {:.2f}%".format(model.score(X_test, y_test) * 100))


#Gabriel Atkin https://www.kaggle.com/code/gcdatkin/fertilizer-type-prediction

y_pred = model.predict(X_test)

clr = classification_report(y_test, y_pred)
print("Classification Report:\n----------------------\n", clr)

