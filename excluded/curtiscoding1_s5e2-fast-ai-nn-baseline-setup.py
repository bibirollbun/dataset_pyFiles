from fastai.tabular.all import *
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import random

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")

train.drop("id", axis = 1, inplace = True)
test.drop("id", axis = 1, inplace = True)


conts = train.select_dtypes(include = np.number).columns.tolist()
conts.remove('Compartments')
cats = [col for col in train.columns if col not in conts]
conts.remove("Price")


idx_all = np.array(train.index) 
val_size = len(idx_all) // 3  
idx_val = np.random.choice(idx_all, size=val_size, replace=False)  
idx_train = np.setdiff1d(idx_all, idx_val, assume_unique=True) 


# Filling missing categorical values
for col in cats:
    colmode = train[col].mode()[0]  
    train[col] = train[col].fillna(colmode)
    test[col] = test[col].fillna(colmode)

# Filling missing continuous values
for col in conts:
    colmean = train[col].mean()
    train[col] = train[col].fillna(colmean).astype('float32')
    test[col] = test[col].fillna(colmean).astype('float32')


dls = TabularDataLoaders.from_df(train, 
                                cat_names=cats, 
                                cont_names=conts,
                                y_names="Price",
                                valid_idx=idx_train.tolist(),
                                procs=[Categorify, FillMissing],  # Let FastAI handle encoding
                                bs=256,
                                val_bs=256
                               )


learn = tabular_learner(dls, metrics = rmse)


learn.fit_one_cycle(6)


preds = learn.get_preds(dl=dls.test_dl(test))


preds_array = preds[0].numpy()


preds_array


sub = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")


sub["Price"] = preds_array


sub.to_csv("submission.csv", index = False)


sub.head()


doc(TabularDataLoaders.from_df)


doc(tabular_learner)




