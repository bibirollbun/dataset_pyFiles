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


#Importing Required Packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt


#Display all rows & columns
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns', 200)


train_data=pd.read_csv('/kaggle/input/leaf-classification/train.csv.zip')
test_data=pd.read_csv('/kaggle/input/leaf-classification/test.csv.zip')
train_data.head(10)


#shape
print(f'data contains {train_data.shape[0]} rows and {train_data.shape[1]} columns \n')
#missing data
print(f'missing data per column is \n {train_data.isna().sum()}')
#duplicate
print('dup')
duplicated_data=train_data.duplicated()
print(type(duplicated_data))
a = duplicated_data.array
b = np.where(a == True)
print(type(b[0]))
print(len(b[0]))
# print(duplicated_data)
# print(f'Number of duplicated rows = {len(duplicated_data[duplicated_data[1]==True])}')
print(f'Number of duplicated rows = {len(b[0])}')


#classes distribution
print(train_data.groupby('species').size())
#it's a balanced problem


print(train_data.dtypes)


#summary statistics
train_data.describe()


def identify_outliers(data):
    # Calculate quartiles
    quartiles = np.percentile(data, [25, 75])
    lower_quartile = quartiles[0]
    upper_quartile = quartiles[1]
    
    # Calculate interquartile range
    iqr = upper_quartile - lower_quartile
    
    # Define the upper and lower bounds
    lower_bound = lower_quartile - 1.5 * iqr
    upper_bound = upper_quartile + 1.5 * iqr
    
    # Identify outliers
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    
    return outliers





train_dataoutlier=train_data.drop('species',axis=1)


for col in train_dataoutlier.columns:
    print(col)
    # Identify outliers
    outliers = identify_outliers(train_data[col])
    if outliers:
        print("Outliers:", outliers,len(outliers))
        
    else:
        print("No outliers found.")


#split into X and y to split to train and test

X=train_data.loc[0:,train_data.columns!='species']
X=X.drop("id",axis=1)
y=LabelEncoder().fit_transform(train_data.loc[0:,train_data.columns=='species'])
y=y.reshape((-1,1))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)


# bin the values into three intervals
discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans')
X_binned_train = discretizer.fit_transform(X_train)
X_binned_test=discretizer.transform(X_test)


clf = RandomForestClassifier(max_depth=None, random_state=42,n_estimators=2000,criterion='gini')
clf.fit(X_binned_train, y_train)


y_train_pred=clf.predict(X_binned_train)
print("Train Accuracy = ",accuracy_score(y_train_pred,y_train))
y_test_pred=clf.predict(X_binned_test)
print("Test Accuracy = ",accuracy_score(y_test_pred,y_test))


classes = ['Acer_Capillipes', 'Acer_Circinatum', 'Acer_Mono', 'Acer_Opalus', 'Acer_Palmatum', 'Acer_Pictum', 'Acer_Platanoids', 'Acer_Rubrum', 'Acer_Rufinerve', 'Acer_Saccharinum', 'Alnus_Cordata', 'Alnus_Maximowiczii', 'Alnus_Rubra', 'Alnus_Sieboldiana', 'Alnus_Viridis', 'Arundinaria_Simonii', 'Betula_Austrosinensis', 'Betula_Pendula', 'Callicarpa_Bodinieri', 'Castanea_Sativa', 'Celtis_Koraiensis', 'Cercis_Siliquastrum', 'Cornus_Chinensis', 'Cornus_Controversa', 'Cornus_Macrophylla', 'Cotinus_Coggygria', 'Crataegus_Monogyna', 'Cytisus_Battandieri', 'Eucalyptus_Glaucescens', 'Eucalyptus_Neglecta', 'Eucalyptus_Urnigera', 'Fagus_Sylvatica', 'Ginkgo_Biloba', 'Ilex_Aquifolium', 'Ilex_Cornuta', 'Liquidambar_Styraciflua', 'Liriodendron_Tulipifera', 'Lithocarpus_Cleistocarpus', 'Lithocarpus_Edulis', 'Magnolia_Heptapeta', 'Magnolia_Salicifolia', 'Morus_Nigra', 'Olea_Europaea', 'Phildelphus', 'Populus_Adenopoda', 'Populus_Grandidentata', 'Populus_Nigra', 'Prunus_Avium', 'Prunus_X_Shmittii', 'Pterocarya_Stenoptera', 'Quercus_Afares', 'Quercus_Agrifolia', 'Quercus_Alnifolia', 'Quercus_Brantii', 'Quercus_Canariensis', 'Quercus_Castaneifolia', 'Quercus_Cerris', 'Quercus_Chrysolepis', 'Quercus_Coccifera', 'Quercus_Coccinea', 'Quercus_Crassifolia', 'Quercus_Crassipes', 'Quercus_Dolicholepis', 'Quercus_Ellipsoidalis', 'Quercus_Greggii', 'Quercus_Hartwissiana', 'Quercus_Ilex', 'Quercus_Imbricaria', 'Quercus_Infectoria_sub', 'Quercus_Kewensis', 'Quercus_Nigra', 'Quercus_Palustris', 'Quercus_Phellos', 'Quercus_Phillyraeoides', 'Quercus_Pontica', 'Quercus_Pubescens', 'Quercus_Pyrenaica', 'Quercus_Rhysophylla', 'Quercus_Rubra', 'Quercus_Semecarpifolia', 'Quercus_Shumardii', 'Quercus_Suber', 'Quercus_Texana', 'Quercus_Trojana', 'Quercus_Variabilis', 'Quercus_Vulcanica', 'Quercus_x_Hispanica', 'Quercus_x_Turneri', 'Rhododendron_x_Russellianum', 'Salix_Fragilis', 'Salix_Intergra', 'Sorbus_Aria', 'Tilia_Oliveri', 'Tilia_Platyphyllos', 'Tilia_Tomentosa', 'Ulmus_Bergmanniana', 'Viburnum_Tinus', 'Viburnum_x_Rhytidophylloides', 'Zelkova_Serrata']


#training using all data 
X_binned=discretizer.transform(X)
clf_final = RandomForestClassifier(max_depth=None, random_state=42,n_estimators=2000,criterion='gini')
clf_final.fit(X_binned, y)


index=test_data['id']
test=test_data.drop('id',axis=1)
test_dist=discretizer.transform(test)
output=clf_final.predict_proba(test_dist)

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)



submission = pd.DataFrame(output, columns=classes)
submission.insert(0, 'id', index)
print(submission)
submission.to_csv('submission.csv',index=False)

