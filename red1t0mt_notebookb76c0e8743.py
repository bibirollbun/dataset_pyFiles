# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_raw = pd.read_csv('../input/forest-cover-type-prediction/train.csv', index_col= 'Id')
test_all = pd.read_csv('../input/forest-cover-type-prediction/test.csv', index_col='Id')

target = train_raw['Cover_Type']


train_raw.describe()


train_raw.drop(['Soil_Type7', 'Soil_Type15'], axis = 1, inplace = True)
test_all.drop(['Soil_Type7', 'Soil_Type15'],axis =1, inplace = True)


train_raw.info()


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.figure(figsize=(10,8))

y_target = np.array(target)

chart = sns.countplot(x=y_target, palette='Set3')

chart.set_xticks(range(7))  # 0, 1, 2, 3, 4, 5, 6

labels = [
    "Spruce/Fir (1)",
    "Lodgepole Pine (2)",
    "Ponderosa Pine (3)",
    "Cottonwood/Willow (4)",
    "Aspen (5)",
    "Douglas-fir (6)",
    "Krummholtz (7)"
]

chart.set_xticklabels(labels, rotation=45)

plt.xlabel('Forest Cover Types')
plt.ylabel('Count')
plt.title('Distribution of Forest Cover Types')
plt.tight_layout()
plt.show()


corrMatrix = train_raw.corr()

corrAbs = corrMatrix.abs().unstack()
corrSorted = corrAbs.sort_values(ascending = False).drop_duplicates()

corrSorted.head(50)


train_all = train_raw.copy()
train_all.drop(['Cover_Type'], axis=1, inplace=True)

X_train, X_valid, Y_train, Y_valid = train_test_split(train_all, target, 
                                                                train_size=0.9, test_size=0.1, random_state=5, stratify = target)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state = 20, n_jobs = -1, n_estimators = 100, bootstrap = True, max_depth = 50,
                            max_features = 0.5)
rfc.fit(X_train, Y_train)

pred_valid_rf = rfc.predict(X_valid)

from sklearn import metrics
print(metrics.accuracy_score(Y_valid, pred_valid_rf))


pred_test_rf = rfc.predict(test_all)


importance = rfc.feature_importances_

from matplotlib import pyplot
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()



train_all.columns


from scipy import stats
from scipy.cluster import hierarchy as hc

corr = np.round(stats.spearmanr(train_all).correlation, 2)
plt.figure(figsize=(20,20))

hc.dendrogram(hc.linkage(hc.distance.squareform(1-corr), 
                         method='average'), 
              labels=train_all.columns, orientation='left', 
              leaf_font_size=14)
plt.show()


import math
cols = list(train_all.columns)

for data in [train_raw, test_all]:
    data['Hillshade'] = data['Hillshade_9am'] + data['Hillshade_3pm'] + data['Hillshade_Noon']
    data['binned_elev'] = [math.floor(v/50.0) for v in data['Elevation']]
    data['Elevation_Fire_Points'] = data['Elevation']+data['Horizontal_Distance_To_Fire_Points']
    data['Road_Fire'] = data['Horizontal_Distance_To_Roadways'] + data['Horizontal_Distance_To_Fire_Points']
    data['Road-Fire'] = data['Horizontal_Distance_To_Roadways'] - data['Horizontal_Distance_To_Fire_Points']
    data['Ele_Road_Fire_Hydro'] = data['Elevation'] + data['Horizontal_Distance_To_Roadways']  + data['Horizontal_Distance_To_Fire_Points'] + data['Horizontal_Distance_To_Hydrology']
    data['Ele-Road'] = data['Elevation'] + data['Horizontal_Distance_To_Roadways']
    data['Ele_Road'] = data['Elevation'] - data['Horizontal_Distance_To_Roadways']
    data['Ele-Fire'] = data['Elevation'] + data['Horizontal_Distance_To_Fire_Points']
    data['Ele_Fire'] = data['Elevation'] - data['Horizontal_Distance_To_Fire_Points']
    data['Ele_Hillshade'] = data['Elevation'] - data['Hillshade']
    data['Ele-Hillshade'] = data['Elevation'] + data['Hillshade']
    #None elevation combos:
    data['Soil_W1'] = data['Soil_Type29'] + data['Wilderness_Area1']
    data['Soil_W4'] = data['Wilderness_Area4'] + data['Soil_Type3']
    data['Hydrology_Total'] = abs(data["Horizontal_Distance_To_Hydrology"])+abs(data['Vertical_Distance_To_Hydrology'])
    #Summary metrics
    data["mean"] = data[cols].mean(axis=1)
    data["min"] = data[cols].min(axis=1)
    data["max"] = data[cols].max(axis=1)
    data["std"] = data[cols].std(axis=1)


train_all = train_raw.copy()
train_all.drop(['Cover_Type'], axis=1, inplace=True)

X_train, X_valid, Y_train, Y_valid = train_test_split(train_all, target, 
                                                                train_size=0.9, test_size=0.1, random_state=5, stratify = target)

from sklearn.ensemble import RandomForestClassifier
rfc2 = RandomForestClassifier(random_state = 20, n_jobs = -1, n_estimators = 100, bootstrap = True, max_depth = 50,
                            max_features = 0.5)
rfc2.fit(X_train, Y_train)

pred_valid_rf2 = rfc2.predict(X_valid)

from sklearn import metrics
print(metrics.accuracy_score(Y_valid, pred_valid_rf2))


from catboost import CatBoostClassifier
cbc = CatBoostClassifier(random_state = 20, iterations = 3000, learning_rate = 0.03,od_wait = 1000,
                         depth = 7, l2_leaf_reg = 3, eval_metric = 'Accuracy', verbose = 1000)
cbc.fit(X_train, Y_train)

pred_valid_cbc = cbc.predict(X_valid)
print(metrics.accuracy_score(Y_valid, pred_valid_cbc))


from sklearn.ensemble import ExtraTreesClassifier
etc = ExtraTreesClassifier(random_state = 20, n_jobs = -1, max_features = 'auto')

etc.fit(X_train, Y_train)
pred_valid_etc = etc.predict(X_valid)
pred_test_etc = etc.predict(test_all)

print(metrics.accuracy_score(Y_valid, pred_valid_etc))


from xgboost import XGBClassifier
from sklearn import metrics

Y_train_adj = Y_train - 1
Y_valid_adj = Y_valid - 1

xgb = XGBClassifier(
    random_state=20, 
    learning_rate=0.4, 
    objective='multi:softprob', 
    num_class=7, 
    eval_metric='merror',
    verbose=False
)

xgb.fit(X_train, Y_train_adj)

pred_valid_xgb = xgb.predict(X_valid)

print(metrics.accuracy_score(Y_valid_adj, pred_valid_xgb))

pred_valid_original = pred_valid_xgb + 1


import lightgbm as lgb

lb = lgb.LGBMClassifier(learning_rate=0.09,max_depth=-5,random_state=42, application = 'multiclass')

lb.fit(X_train, Y_train)

pred_valid_lb = lb.predict(X_valid)
print(metrics.accuracy_score(Y_valid, pred_valid_lb))


output = pd.DataFrame({'Id': test_all.index,
                       'Cover_Type': pred_test_etc})
output.head()
output.to_csv('submission.csv', index=False)




