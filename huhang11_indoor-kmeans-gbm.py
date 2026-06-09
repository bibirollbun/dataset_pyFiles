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


# Prepare paths:
import glob
from pathlib import Path
inpath = '/kaggle/input/indoor-location-navigation/'
metapath = inpath + 'metadata/'
trainpath = inpath + 'train/'
testpath = inpath + 'test/'

# Extract testing files, buildings and sites:
os.system(f'grep SiteID {testpath}/* > test_buildings.txt')
test_buildings = pd.read_csv('test_buildings.txt',sep='\t',header=None,names=['file','building','site'])
test_buildings['file'] = test_buildings['file'].apply(lambda x: x[:-2])
test_buildings['building'] = test_buildings['building'].apply(lambda x: x[7:])

# How many buildings in the testing set?
buildings = np.unique(test_buildings['building'])
print('There are',len(buildings),'buildings in the testing set.')

test_buildings.head()


# Compile C++ pre-processing code:
er=os.system("g++ /kaggle/input/indoor-cpp/1_preprocess.cpp -std=c++11 -o preprocess")
if(er): print("Error")

# Reformat the testing set:
os.system('mkdir test')
for i,(path_filename,building) in enumerate(zip(test_buildings['file'],test_buildings['building'])):
    er=os.system(f'./preprocess {path_filename} test {building} {0}') #since we do not know the floor, we use 0.
    if(er): print("Error:",path_filename)


# Wifi testing data:
os.system('mkdir test_wifi')
os.system("g++ /kaggle/input/indoor-cpp/2_preprocess_wifi.cpp -std=c++11 -o preprocess_wifi")
for building in buildings:
    os.system(f'./preprocess_wifi {building}')


from sklearn.cluster import KMeans

import lightgbm as lgb
lgb_params = {'objective': 'multiclass',
              'boosting_type': 'gbdt',
              'n_estimators': 50000,
              'learning_rate': 0.1,
              'num_leaves': 90,
              'colsample_bytree': 0.4,
              'subsample': 0.6,
              'subsample_freq': 2,
              'bagging_seed': 42,
              'reg_alpha': 10,
              'reg_lambda': 2,
              'random_state': 42,
              'n_jobs': -1,
#               'device':'gpu'
}


from sklearn.model_selection import StratifiedKFold

result = pd.DataFrame(columns=['floor','proba'])

for building in buildings:
    
    # Training set:
    xyw = pd.DataFrame()
    for floor in np.arange(-3,10):
        file = f'/kaggle/input/indoor-xy-floor/{building}_{floor}.csv'
        if Path(file).is_file():
            xyi = pd.read_csv(file,index_col=0)
            bcols = [c for c in xyi.columns if len(c.split('_'))==3] #beacon cols
            wcols = [c for c in xyi.columns if c not in ['x','y','count','magn']+bcols] #wifi cols
            xyi = xyi.loc[~np.isnan(xyi['count']),['x','y','count','magn']+wcols]
            xyi.insert(0,'floor',floor)
            if(len(xyw)):
                xyw = xyw.merge(xyi,how='outer')
            else: xyw = xyi
    xyw.replace(np.nan,-99.0,inplace=True)

    # XY clustering:
    kmeans = KMeans(n_clusters=4,random_state=0).fit(xyw[['x','y']])
    xyw.insert(0,'cluster',kmeans.labels_)

    # Testing set:
    tfw = pd.read_csv(f'test_wifi/{building}.txt')
    tfw = tfw.pivot_table(index=['path_id','t1_wifi'],columns='bssid_wifi',values='rssid_wifi')
    tfw = tfw.reindex(columns=xyw.columns[6:],fill_value=np.nan)
    tfw.fillna(-99.0,inplace=True)

    # Arrays:
    dfmat = np.array(xyw.iloc[:,6:])
    mtest = np.array(tfw)
    labs = np.array([str(f)+'_'+str(c) for (f,c) in zip(xyw['floor'],xyw['cluster'])])
    features = list(np.unique(labs))
    yvalid = pd.DataFrame(np.zeros([len(labs),len(features)]),index=xyw.index,columns=features)
    ytest = pd.DataFrame(np.zeros([len(tfw),len(features)]),index=tfw.index,columns=features)

    # K-fold CV of coordinates:
    seeds, folds = 1, 10
    skf = StratifiedKFold(n_splits=folds,random_state=42,shuffle=True)
    for fold, (idt,idv) in enumerate(skf.split(dfmat,labs)):
        print('\r',f'{fold}',end='\t')
        mtrain, mvalid = dfmat[idt], dfmat[idv]
        ltrain, lvalid = labs[idt], labs[idv]
        modelf = lgb.LGBMClassifier(**lgb_params)
        modelf.fit(mtrain,ltrain,eval_set=[(mvalid,lvalid)],
            eval_metric='softmax',early_stopping_rounds=10,verbose=False)
        yvalid.loc[xyw.index[idv],modelf.classes_] = modelf.predict_proba(mvalid)
        ytest[modelf.classes_] += modelf.predict_proba(mtest) / folds

    # Performance:
    yvalid['truth'] = xyw['floor']
    yvalid = yvalid.melt(id_vars='truth')
    yvalid['pred'] = [int(x.split('_')[0]) for x in yvalid.variable]
    frmse = np.mean(np.sqrt((yvalid['pred']-yvalid['truth'])**2))
    print(building,f'floor rmse = {frmse}')

    # Prediction:
    ytest = ytest.groupby('path_id').mean().melt(ignore_index=False,value_name='proba')
    ytest['floor'] = [x.split('_')[0] for x in ytest.variable]
    ytest = ytest.groupby(['path_id','floor'])['proba'].sum().reset_index()
    ytest = ytest.loc[ytest.groupby('path_id')['proba'].transform(max) == ytest['proba']]
    ytest.index = [building+'_'+x for x in ytest.path_id]
    result = pd.concat([result,ytest[['floor','proba']]])
    result.to_csv('result_floor.csv')
    
result.head()


# Example:
import matplotlib.pyplot as plt
plt.figure(figsize=(15,15))
plt.scatter(xyw.x,xyw.y,c=xyw.cluster)
plt.show()




