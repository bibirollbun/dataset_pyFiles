import warnings
warnings.filterwarnings('ignore')


from fastai.imports import *


path = Path('../input/bluebook-for-bulldozers')
Path.BASE_PATH = path
# !ls {path}
path.ls(file_type='text')


df = pd.read_csv(path/'TrainAndValid.csv', low_memory=False)


df.head()


df.columns


df['ProductSize'].unique()


sizes = 'Large','Large / Medium','Medium','Small','Mini','Compact'


df['ProductSize'] = df['ProductSize'].astype('category')
df['ProductSize'] = df['ProductSize'].cat.set_categories(sizes, ordered=True)
df['ProductSize']


dep_var = 'SalePrice'

df[dep_var] = np.log(df[dep_var])
df[dep_var]


from fastai.tabular.all import *


df = add_datepart(df, 'saledate')


df_test = pd.read_csv(path/'Test.csv', low_memory=False)
df_test = add_datepart(df_test, 'saledate')


' '.join(o for o in df.columns if o.startswith('sale'))


procs = [Categorify, FillMissing]


cond = (df.saleYear<2011) | (df.saleMonth<10)
train_idx = np.where( cond)[0]
valid_idx = np.where(~cond)[0]

splits = (list(train_idx),list(valid_idx))


cont, cat = cont_cat_split(df, 1, dep_var=dep_var)


to = TabularPandas(df, procs, cat, cont, y_names=dep_var, splits=splits)


len(to.train), len(to.valid)


to.show(3)


to1 = TabularPandas(df, procs, ['state', 'ProductGroup', 'Drive_System', 'Enclosure'], [], y_names=dep_var, splits=splits)
to1.show(3)


to.items.head(3)


to1.items[['state', 'ProductGroup', 'Drive_System', 'Enclosure']].head(3)


to.classes['ProductSize'], to.classes['Drive_System']


save_path = Path('/kaggle/working')
save_pickle(save_path/'to.pkl', to)


to = load_pickle(save_path/'to.pkl')


xs, y = to.train.xs, to.train.y
valid_xs, valid_y = to.valid.xs, to.valid.y


from sklearn.tree import DecisionTreeRegressor


m = DecisionTreeRegressor(max_leaf_nodes=4)
m.fit(xs, y);


!pip install fastbook --quiet

from fastbook import draw_tree


draw_tree(m, xs, size=10, leaves_parallel=True, precision=2)


!pip install dtreeviz==1.4.1 --quiet


from dtreeviz.trees import dtreeviz


samp_idx = np.random.permutation(len(y))[:500]

dtreeviz(m, xs.iloc[samp_idx], y.iloc[samp_idx], xs.columns, dep_var,
        fontname='DejaVu Sans', scale=1.6, label_fontsize=10,
        orientation='LR')


xs.loc[xs['YearMade']<1900, 'YearMade'] = 1950
valid_xs.loc[valid_xs['YearMade']<1900, 'YearMade'] = 1950


m = DecisionTreeRegressor(max_leaf_nodes=4).fit(xs, y)

dtreeviz(m, xs.iloc[samp_idx], y.iloc[samp_idx], xs.columns, dep_var,
        fontname='DejaVu Sans', scale=1.6, label_fontsize=10,
        orientation='LR')


m = DecisionTreeRegressor()
m.fit(xs, y)


def r_mse(pred,y):
    return round(math.sqrt(((pred-y)**2).mean()), 6)

def m_rmse(m, xs, y):
    return r_mse(m.predict(xs), y)


m_rmse(m, xs, y)


m_rmse(m, valid_xs, valid_y)


m.get_n_leaves(), len(xs)


m = DecisionTreeRegressor(min_samples_leaf=25)
m.fit(to.train.xs, to.train.y)
m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y)


m.get_n_leaves()


from sklearn.ensemble import RandomForestRegressor


def rf(xs, y, n_estimators=40, max_samples=200_000,
       max_features=0.5, min_samples_leaf=5, **kwargs):
    return RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators,
                                 max_samples=max_samples, max_features=max_features,
                                 min_samples_leaf=min_samples_leaf, oob_score=True).fit(xs, y)


m = rf(xs, y);


m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y)


preds = np.stack([t.predict(valid_xs) for t in m.estimators_])


r_mse(preds.mean(0), valid_y)


plt.plot([r_mse(preds[:i+1].mean(0), valid_y) for i in range(40)]);


r_mse(m.oob_prediction_, y)


preds = np.stack([t.predict(valid_xs) for t in m.estimators_])


preds.shape


preds_std = preds.std(0)

preds_std[:5]


def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)


fi = rf_feat_importance(m, xs)
fi[:10]


def plot_fi(fi):
    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)

plot_fi(fi[:30]);


to_keep = fi[fi.imp>0.005].cols
len(to_keep)


xs_imp = xs[to_keep]
valid_xs_imp = valid_xs[to_keep]


m = rf(xs_imp, y)


m_rmse(m, xs_imp, y), m_rmse(m, valid_xs_imp, valid_y)


len(xs.columns), len(xs_imp.columns)


plot_fi(rf_feat_importance(m, xs_imp));


from scipy.cluster import hierarchy as hc
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

def cluster_columns(df, figsize=(10,6), font_size=12):
    corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
    corr_condensed = hc.distance.squareform(1 - corr)
    z = hc.linkage(corr_condensed, method='average')
    fig = plt.figure(figsize=figsize)
    hc.dendrogram(z, labels=df.columns, orientation='left', leaf_font_size=font_size)
    plt.show()


cluster_columns(xs_imp)


def get_oob(df):
    m = RandomForestRegressor(n_estimators=40, min_samples_leaf=15,
                              max_samples=50000, max_features=0.5, n_jobs=-1, oob_score=True)
    m.fit(df, y)
    return m.oob_score_


get_oob(xs_imp)


{c:get_oob(xs_imp.drop(c, axis=1)) for c in (
    'saleYear', 'saleElapsed', 'ProductGroupDesc','ProductGroup',
    'fiModelDesc', 'fiBaseModel',
    'Hydraulics_Flow','Grouser_Tracks', 'Coupler_System')}


to_drop = ['saleYear', 'ProductGroupDesc', 'fiBaseModel', 'Grouser_Tracks']
get_oob(xs_imp.drop(to_drop, axis=1))


xs_final = xs_imp.drop(to_drop, axis=1)
valid_xs_final = valid_xs_imp.drop(to_drop, axis=1)


save_pickle(save_path/'xs_final.pkl', xs_final)
save_pickle(save_path/'valid_xs_final.pkl', valid_xs_final)


xs_final = load_pickle(save_path/'xs_final.pkl')
valid_xs_final = load_pickle(save_path/'valid_xs_final.pkl')


m = rf(xs_final, y)
m_rmse(m, xs_final, y), m_rmse(m, valid_xs_final, valid_y)


p = valid_xs_final['ProductSize'].value_counts(sort=False).plot.barh()
c = to.classes['ProductSize']
plt.yticks(range(len(c)), c)


ax = valid_xs_final['YearMade'].hist()


from sklearn.inspection import PartialDependenceDisplay

fig, ax = plt.subplots(figsize=(12, 4))
PartialDependenceDisplay.from_estimator(m, valid_xs_final, features=['YearMade', 'ProductSize'],
                                        grid_resolution=20, ax=ax)


!pip install treeinterpreter --quiet
!pip install waterfallcharts --quiet


import warnings
warnings.simplefilter('ignore', FutureWarning)

from treeinterpreter import treeinterpreter
from waterfall_chart import plot as waterfall


row = valid_xs_final.iloc[:5]


prediction, bias, contributions = treeinterpreter.predict(m, row.values)


prediction[0], bias[0], contributions[0].sum()


waterfall(valid_xs_final.columns, contributions[0], threshold=0.08, rotation_value=45, formatting='{:,.3f}')


np.random.seed(42)


x_lin = torch.linspace(0,20, steps=40)
y_lin = x_lin + torch.randn_like(x_lin)
plt.scatter(x_lin, y_lin)


xs_lin = x_lin.unsqueeze(1)
x_lin.shape, xs_lin.shape


x_lin[:,None].shape


m_lin = RandomForestRegressor().fit(xs_lin[:30], y_lin[:30])


plt.scatter(x_lin, y_lin, 20)
plt.scatter(x_lin, m_lin.predict(xs_lin), color='red', alpha=0.5)


df_dom = pd.concat([xs_final, valid_xs_final])
is_valid = np.array([0]*len(xs_final) + [1]*len(valid_xs_final))

m = rf(df_dom, is_valid)
rf_feat_importance(m, df_dom)[:6]


m = rf(xs_final, y)
print('orig', m_rmse(m, valid_xs_final, valid_y))

for c in ('SalesID','saleElapsed','MachineID'):
    m = rf(xs_final.drop(c,axis=1), y)
    print(c, m_rmse(m, valid_xs_final.drop(c,axis=1), valid_y))


time_vars = ['SalesID','MachineID']
xs_final_time = xs_final.drop(time_vars, axis=1)
valid_xs_time = valid_xs_final.drop(time_vars, axis=1)

m = rf(xs_final_time, y)
m_rmse(m, valid_xs_time, valid_y)


xs['saleYear'].hist()


filt = xs['saleYear']>2004
xs_filt = xs_final_time[filt]
y_filt = y[filt]


m = rf(xs_filt, y_filt)
m_rmse(m, xs_filt, y_filt), m_rmse(m, valid_xs_time, valid_y)


df_nn = pd.read_csv(path/'TrainAndValid.csv', low_memory=False)
df_nn['ProductSize'] = df_nn['ProductSize'].astype('category')
df_nn['ProductSize'] = df_nn['ProductSize'].cat.set_categories(sizes, ordered=True)
df_nn[dep_var] = np.log(df_nn[dep_var])
df_nn = add_datepart(df_nn, 'saledate')


df_nn_final = df_nn[list(xs_final_time.columns) + [dep_var]]


cont_nn, cat_nn = cont_cat_split(df_nn_final, max_card=9000, dep_var=dep_var)


cont_nn


df_nn_final[cat_nn].nunique()


xs_filt2 = xs_filt.drop('fiModelDescriptor', axis=1)
valid_xs_time2 = valid_xs_time.drop('fiModelDescriptor', axis=1)
m2 = rf(xs_filt2, y_filt)
m_rmse(m2, xs_filt2, y_filt), m_rmse(m2, valid_xs_time2, valid_y)


cat_nn.remove('fiModelDescriptor')


procs_nn = [Categorify, FillMissing, Normalize]
to_nn = TabularPandas(df_nn_final, procs_nn, cat_nn, cont_nn,
                      splits=splits, y_names=dep_var)


dls = to_nn.dataloaders(1024)


y = to_nn.train.y
y.min(), y.max()


learn = tabular_learner(dls, y_range=(8,12), layers=[500,250],
                        n_out=1, loss_func=F.mse_loss)


learn.lr_find()


learn.fit_one_cycle(5, 1e-2)


preds,targs = learn.get_preds()
r_mse(preds,targs)


learn.save('nn')


rf_preds = m.predict(valid_xs_time)
ens_preds = (to_np(preds.squeeze()) + rf_preds) /2


r_mse(ens_preds,valid_y)

