#| default_exp load_data


#| hide
from nbdev.showdoc import *


!pip install git+https://github.com/ozpau/kaggle_backpack_sales.git


!pip install oztools


!pip install -Uqq dtreeviz


!pip install -Uqq fastbook


#| export
from fastcore.all import *
from oztools.kgl import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from dtreeviz.trees import *
from fastai.tabular.all import *
from fastbook import *
import warnings


#| export
path = setup_comp(
    'playground-series-s5e2',
    'git+https://github.com/ozpau/kaggle_backpack_sales.git'
)
Path.BASE_PATH = path
path.ls()


train = pd.read_csv(path/"train.csv", index_col=0)
test = pd.read_csv(path/"test.csv", index_col=0)
train_extra = pd.read_csv(path/"training_extra.csv", index_col=0)
sample_sub = pd.read_csv(path/"sample_submission.csv", index_col=0)

train.shape, test.shape


train


sample_sub


test


train_conds = np.arange(len(train)) < len(train)*0.8

train_idx = np.where( train_conds)[0]
valid_idx = np.where(~train_conds)[0]
splits = (list(train_idx), list(valid_idx))

procs = [Categorify, FillMissing]
dep_vars = "Price"
cont,cat = cont_cat_split(train, 40, dep_var=dep_vars)


cont


cat


with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    to = TabularPandas(train, procs, cat, cont, y_names=dep_vars, splits=splits)


def preprocess_test_data(to):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dls = to.dataloaders()
        dl = dls.test_dl(test)
        return dl.dataset.items


processed_test = preprocess_test_data(to)


xs,y = to.train.xs,to.train.y
valid_xs,valid_y = to.valid.xs,to.valid.y


m = DecisionTreeRegressor(max_leaf_nodes=4)
m.fit(xs, y);

draw_tree(m, xs, size=10, leaves_parallel=True, precision=2)


def predict(to):
    df = preprocess_test_data(to).reindex(columns = m.feature_names_in_)
    df = pd.DataFrame({"Price": m.predict(df)}, index=df.index)
    df.index.name = "id"
    return df


df = predict(to)
df.to_csv("submission.csv")


!head submission.csv


#| hide
import nbdev; nbdev.nbdev_export()

