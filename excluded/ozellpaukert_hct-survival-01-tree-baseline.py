try: import kaggle_hct_survival
except:
    !pip install -Uqq git+https://github.com/ozpau/kaggle_hct_survival.git


#| default_exp tree_baseline


#| hide
from nbdev.showdoc import *


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
    'equity-post-HCT-survival-predictions',
    'git+https://github.com/ozpau/kaggle_hct_survival.git'
)
Path.BASE_PATH = path
path.ls()


train = pd.read_csv(path/"train.csv", index_col=0)
test = pd.read_csv(path/"test.csv", index_col=0)
desc = pd.read_csv(path/"data_dictionary.csv", index_col=0).T
sample_sub = pd.read_csv(path/"sample_submission.csv", index_col=0)

train.shape, test.shape


ys = train.columns.difference(test.columns)
ys


desc[ys].T


dep_vars = ["efs", "efs_time"]


sample_sub.columns


time_vars = ['year_hct', 'donor_age', 'age_at_hct']
train[time_vars].describe()


test[time_vars]


pvalid = 0.2
#train.loc[len(train)


sample_sub.head().T


vars_submit = ["efs"]
train[vars_submit].sample(5).T


train[vars_submit].value_counts()


train.query('efs == 0')['efs_time'].plot.hist(label="Dead")
train.query('efs == 1')['efs_time'].plot.hist(label="Alive")
plt.legend()


train_conds = np.arange(len(train)) < len(train)*0.8

train_idx = np.where( train_conds)[0]
valid_idx = np.where(~train_conds)[0]
splits = (list(train_idx), list(valid_idx))

procs = [Categorify, FillMissing]
cont,cat = cont_cat_split(train, 40, dep_var=dep_vars)


with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    # debug
    to = TabularPandas(train, procs, cat, cont, y_names=dep_vars, splits=splits)
    #to = TabularPandas(train.drop(, procs, cat, cont, y_names=dep_vars, splits=splits)


def preprocess_test_data(to):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dls = to.dataloaders()
        dl = dls.test_dl(test)
        return dl.dataset.items


processed_test = preprocess_test_data(to)


'efs' in processed_test or 'efs_time' in processed_test


'efs_time' in processed_test.columns


list(zip(to.items.drop(columns=['efs','efs_time']).columns, processed_test.columns))


processed_test.columns


pd.concat([train,test]).shape, train.shape, test.shape


xs,y = to.train.xs,to.train.y
valid_xs,valid_y = to.valid.xs,to.valid.y


m = DecisionTreeRegressor(max_leaf_nodes=4)
m.fit(xs, y);

draw_tree(m, xs, size=10, leaves_parallel=True, precision=2)


desc['conditioning_intensity']


to.classes['conditioning_intensity']


to.classes['graft_type']


len(preprocess_test_data(to).columns), len(m.feature_names_in_)


df = preprocess_test_data(to)
df.reindex(columns = m.feature_names_in_)


list(zip(preprocess_test_data(to).reindex(columns = m.feature_names_in_).columns, m.feature_names_in_))


def predict(to):
    df = preprocess_test_data(to).reindex(columns = m.feature_names_in_)
    df = pd.DataFrame({"prediction": m.predict(df)}, index=df.index)
    df.index.name = "ID"
    return df


Path("var").mkdir(exist_ok=True)


df = predict(to)
df.to_csv("var/submission.csv")


!head {path}/sample_submission.csv


from fastkaggle import *





#| hide
comp_short = 'hct-survival'
if not iskaggle:
    push_notebook(
        'ozellpaukert', f'{comp_short}-tree-baseline',
        title='HCT survival: 01 Tree baseline',
        file='01_tree_baseline.ipynb',
        competition='equity-post-HCT-survival-predictions',
        private=False, gpu=False)


#| hide
if not iskaggle: import nbdev; nbdev.nbdev_export()




