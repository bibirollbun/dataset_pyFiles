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


#Importd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
warnings.filterwarnings('ignore')


#get the data
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


train.info()


train


# train.map?


train['Fertilizer Name'] = train['Fertilizer Name'].map({
                                '28-28': '28-28-0',
                                'DAP': '18-46-0',
                                '20-20':'20-20-0',
                                'Urea':'46-0-0',
                                '17-17-17':'17-17-17',
                                '14-35-14':'14-35-14',
                                '10-26-26':'10-26-26'}, na_action = 'ignore')
train


# # Actual Set(With classified N-P-K for fertilizers)
# Fertilizer_nutrients =  [train['Fertilizer Name'].iloc[i].split(sep = '-') for i in range(train.shape[0])]
# train['Fertilizer_nutrients'] = Fertilizer_nutrients
# train[['Fertilizer_nutrients']]


# train['FertilizerN'] = [Fertilizer_nutrients[i][0] for i in range(750000)]
# train['FertilizerP'] = [Fertilizer_nutrients[i][1] for i in range(750000)]
# train['FertilizerK'] = [Fertilizer_nutrients[i][2] for i in range(750000)]
# train = train.drop(['Fertilizer_nutrients'], axis = 1)
# train


test


print(train['Crop Type'].unique().flatten().tolist())
print(test['Crop Type'].unique().flatten().tolist())


Fertilizers = train['Fertilizer Name'].unique().flatten().tolist()
Crop_types = train['Crop Type'].unique().flatten().tolist()
Soil_Type = train['Soil Type'].unique().flatten().tolist()


# np.shape(train[['Soil Type']])


def printList(lst, name):
    for item in lst:
        print(f"{lst.index(item)+1}: {item}")

#impute values to numeric
leSoilType = LabelEncoder()
leSoilType.fit(Soil_Type)

train['Soil Type'] = leSoilType.transform(train['Soil Type'])
test['Soil Type'] = leSoilType.transform(test['Soil Type'])
print(f"""Diffrent Soil Types are :""")
printList(Soil_Type, "Soil Type")

leCropType = LabelEncoder()
leCropType.fit(Crop_types)

train['Crop Type'] = leCropType.transform(train['Crop Type'])
test['Crop Type'] = leCropType.transform(test['Crop Type'])
print(f"""\nDiffrent Crop Types are :""")
printList(Crop_types, "Crop Type")

leFertilizer = LabelEncoder()
leFertilizer.fit(Fertilizers)

train['Fertilizer Name'] = leFertilizer.transform(train['Fertilizer Name'])
print(f"""\nDiffrent Fertilizer Names are :""")
printList(Fertilizers, "Fertilizer Names")



train


Y = train[['Fertilizer Name']]
#Join both datasets
X = train.drop(['Fertilizer Name'], axis = 1)
fullData = pd.concat([X, test])


fullData





#Correlation heatmap 
fig, ax = plt.subplots(figsize=(15, 10))
heatmap = sns.heatmap(train.corr(), cmap = 'crest', annot = True, ax = ax)



train["SALI"] = (train['Temparature'] * train['Humidity'] * train['Moisture']) / 100000
train['Acid'] = train['SALI'] > 0.8
test["SALI"] = (test['Temparature'] * test['Humidity'] * test['Moisture']) / 100000
test['Acid'] = test['SALI'] > 0.8

train.corr()


#See visualisations of all features with histplots
cols = ['Temparature','Humidity','Moisture','Soil Type','Crop Type','Nitrogen','Potassium','Phosphorous','SALI','Acid']
colors = sns.color_palette('husl', len(cols))

fig, axes = plt.subplots(3,4, figsize = (12,8))
axes = axes.flatten()

for i in range(len(cols)):
    sns.histplot(train[cols[i]], kde = True, ax = axes[i], color = colors[i])
    axes[i].set_ylabel("")

plt.tight_layout()
plt.show()


# see relationship between the features and optimal fertilizer selected
cols = ['Temparature','Humidity','Moisture','Soil Type','Crop Type','Nitrogen','Potassium','Phosphorous','SALI','Acid']

fig, axes = plt.subplots(3,4, figsize=(12,8))
axes = axes.flatten()

for i in range(len(axes)):
    if i < len(cols):
        sns.scatterplot(x = cols[i], y = 'Fertilizer Name', data = train, ax = axes[i])
        axes[i].set_ylabel("")
    else:
        fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


# As a trial I'll use basic RandomForest to classify once
X = train.drop(['Fertilizer Name'], axis = 1)
X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size = 0.2,
                                                   random_state = 42)

rf = RandomForestClassifier()
rf.fit(X_train, Y_train)

preds = rf.predict(X_test)
print(f"Accuracy : {accuracy_score(Y_test, preds)}")


#Crop Needs & Soil Needs



train.sample(10)


# 0:-28-28-----: 28%N - 28%P - 0%K
# 1:-17-17-17--: 17%N - 17%P - 17%K
# 2:-10-26-26--: 10%N - 26%P - 26%K
# 3:-DAP-------: 18%N - 46%P - 0%K (Diammonium Phosphate)
# 4:-20-20-----: 20%N - 20%P - 0%K
# 5:-14-35-14--: 14%N - 35%P - 14%K
# 6:-Urea------: 46%N - 0%P - 0%K
fertilizers = {0:'28-28',1:'17-17-17',2:'10-26-26',3:'DAP',4:'20-20',5:'14-35-14',6:'Urea'}


train['Fertilizer Name'].value_counts()


npk_classes = {0:'High N&P',1:'Balanced',2:'High P&K', 3:'High P',4:'High N&P',5:'High P',6:'High N'}
train['NPK Group'] = [npk_classes[i] for i in train['Fertilizer Name']]
train


Y = train[['Fertilizer Name']]
#Join both datasets
X = train.drop(['Fertilizer Name'], axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                   Y,
                                                   test_size = 0.2,
                                                   random_state = 42)


#KNN input derived from NPK of soil
Soil = test[['Nitrogen','Potassium','Phosphorous']]
NPK_modelknn = KNeighborsClassifier()

NPK_modelknn.fit(X_train[['Nitrogen','Potassium','Phosphorous']], X_train['NPK Group'])

pred = NPK_modelknn.predict(Soil)


test['NPK Group'] = [pred[i] for i in range(test.shape[0])]


test


Y = train[['Fertilizer Name']]
#Join both datasets
X = train.drop(['Fertilizer Name'], axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                   Y,
                                                   test_size = 0.2,
                                                   random_state = 42)


#trial Catboost
model = CatBoostClassifier(verbose = 2)

cat_features = ['NPK Group']
model.fit(X_train, Y_train, cat_features)


preds = model.predict(X_test)
print(f"Accuracy : {accuracy_score(Y_test, preds)}")





final_model = CatBoostClassifier(verbose = 2)
cat_feat = ['NPK Group']

final_model.fit(X, Y, cat_features = cat_feat)


final_preds = final_model.predict(test)
finalPredList = final_preds.ravel().tolist()
final = [fertilizers[i] for i in finalPredList]


output = pd.DataFrame({'id':test['id'],
                      'Fertilizer Name': final})
output.to_csv('submission.csv', index = False)




