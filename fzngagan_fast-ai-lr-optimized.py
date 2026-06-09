# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


# !pip install -Uqq fastai
!pip install -Uqq pyarrow

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session"



from pathlib import Path
base_path = Path("/kaggle/input/drw-crypto-market-prediction/")


import pyarrow.parquet as pq


train_table  = pq.read_pandas(base_path/'train.parquet')


train_df = train_table.to_pandas()


train_df.head()


from fastai.tabular.all import *


clist = list(train_df.columns.values)


train_cols = clist.copy()


train_cols.remove("label")


splits = EndSplitter(valid_pct=0.1)(range_of(train_df))


train_to = TabularPandas(train_df,
                   # procs=[Normalize],
                   cont_names = train_cols,
                   y_names='label',
                   splits=splits)


train_to.xs.head()


dls = train_to.dataloaders(bs=64)


from fastai.metrics import AccumMetric
import scipy.stats as scs
def pearsonr(x,y): return scs.pearsonr(x,y)[0]
def MyPearsonCorrCoef(dim_argmax=None, **kwargs):
    "Pearson correlation coefficient for regression problem"
    return AccumMetric(pearsonr, invert_arg=False, dim_argmax=dim_argmax, **kwargs)



pearson = MyPearsonCorrCoef()


learn = tabular_learner(dls, metrics=pearson)


learn.model


def replace_relu_to_lrelu(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.LeakyReLU())
        else:
            # recurse
            replace_relu_to_lrelu(child)

replace_relu_to_lrelu(learn.model)


learn.lr_find(suggest_funcs=(slide, valley))


learn.fit_one_cycle(2, 0.001)


learn.model


learn.export('/kaggle/working/drw_newdata_fastai_lr_optimized_model_lrelu.pkl')


test_table  = pq.read_pandas(base_path/"test.parquet")
test_df = test_table.to_pandas()


    to = TabularPandas(test_df,
           # procs=[Normalize],
           cont_names = train_cols,
           y_names='label')




    dl = learn.dls.test_dl(to.xs, bs=64)


    preds, _ = learn.get_preds(dl=dl)


sub_df = pd.DataFrame()


    sub_df["ID"] = to.xs.index
    sub_df['prediction'] = preds
    sub_df.to_csv("/kaggle/working/submission_newdata_fastai_tabular_lr_optimized_lrelu.csv", header=True, index=False)




