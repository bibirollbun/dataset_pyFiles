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


df = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv")


df.head()


df.columns


df['ord_2'].value_counts()


df.ord_2.fillna("NONE").value_counts()


df.ord_4 = df.ord_4.fillna("NONE")


df['ord_4'].values


df.loc[df['ord_4'].value_counts()[df['ord_4']].values < 2000,'ord_4']='RARE'





mapping = {
    "Freezing": 0,
    "Warm" : 1 ,
    "Cold" : 2 ,
    "Boiling Hot": 3 ,
    "Hot": 4 ,
    "Lava Hot": 5
}


df.loc[:,'ord_2'] = df.ord_2.map(mapping)


df


import numpy as np

example = np.array(
    [
        [0,0,1],
        [1,0,0],
        [1,0,1]
    ]
)

print(example.nbytes)


df[df.ord_2 == 3].shape


df.groupby(['ord_2'])['id'].count()


df.ord_2.value_counts()





df_train = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv")


df_test = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/test.csv")


df_test.loc[:,'target'] = -1


 data = pd.concat([df_train, df_test]).reset_index(drop=True)


data


features = [x for x in df_train.columns if x not in ['id','target']]


features


from sklearn import preprocessing


for feature in features :
    lbl_enc = preprocessing.LabelEncoder()
    temp_col = data[feature].fillna("NONE").astype(str).values
    data.loc[:, feature] = lbl_enc.fit_transform(temp_col)


df_train = data[data.target != -1].reset_index(drop=True)
df_test = data[data.target == -1].reset_index(drop=True)


from sklearn import linear_model
from sklearn import metrics 
from sklearn import preprocessing 
import pandas as pd
from sklearn import model_selection


df = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv")


df['kfold'] = -1
df = df.sample(frac=1).reset_index(drop=True)
y = df.target.values
kf = model_selection.StratifiedKFold(n_splits=5)
for f,(t_,v_) in enumerate(kf.split(X=df,y=y)):
    df.loc[v_,'kfold']=f


df


x = ['ord_1', 'bin_1']
df.loc[:,x]


def run(fold):
    feature = [f for f in df.columns if f not in ("id","target","kfold") ]
    for col in feature :
        df.loc[:,col] = df[col].astype(str).fillna("NONE")
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_test = df[df.kfold == fold].reset_index(drop=True)
    ohe = preprocessing.OneHotEncoder()
    full_data = pd.concat([df_train[feature], df_test[feature]], axis=0)
    ohe.fit(full_data[feature])
    x_train = ohe.transform(df_train[feature])
    x_test  = ohe.transform(df_test[feature])
    model = linear_model.LogisticRegression()
    model.fit(x_train, df_train.target.values)
    test_preds = model.predict_proba(x_test)[:,1]

    auc = metrics.roc_auc_score(df_test.target.values, test_preds)
    print(auc)


if __name__ == "__main__":
    for i in range(5):
        run(i)




