import os
from pathlib import Path

iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')
if iskaggle: path = Path('../input/playground-series-s5e2')
else:
    path = Path('playground-series-s5e2')
    if not path.exists():
        import zipfile,kaggle
        kaggle.api.competition_download_cli(str(path))
        zipfile.ZipFile(f'{path}.zip').extractall(path)


from fastai.tabular.all import *


df = pd.read_csv(path/'train.csv')
df.head()


df.describe(include=np.number)


df["Compartments"].nunique()


df.describe(include='object')


dep_var = 'Price'


df.isna().sum()


df.dtypes


procs=[Categorify, FillMissing]


cont, cat = cont_cat_split(df, 1, dep_var)


cont


cat


from sklearn.model_selection import train_test_split

train, valid = train_test_split(df,test_size=0.20,random_state=442)
splits = (list(train.index), list(valid.index))


to = TabularPandas(df,procs,cat,cont,dep_var,splits=splits)
xs,y=to.train.xs,to.train.y
valid_xs,valid_y=to.valid.xs,to.valid.y


xs = xs.reset_index(drop=True).drop('id',axis=1)
y = y.reset_index(drop=True)
valid_xs = valid_xs.reset_index(drop=True).drop('id',axis=1)
valid_y = valid_y.reset_index(drop=True)


xs.head()


valid_xs.head()


from sklearn.ensemble import RandomForestRegressor

def rf(xs,y,n_estimators=50, max_samples=120_000, max_features=0.5,min_samples_leaf=5,**kwargs):
    return RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators,max_samples=max_samples, max_features=max_features, 
                                 min_samples_leaf=min_samples_leaf, oob_score=True).fit(xs,y)


m = rf(xs,y);


from sklearn.metrics import mean_squared_error

def rmse(m,xs,y):
    return np.sqrt(mean_squared_error(y, m.predict(xs)))


rmse(m,xs,y),rmse(m,valid_xs,valid_y)


np.sqrt(mean_squared_error(y,m.oob_prediction_))


def get_oob(xs,y,n_estimators=50):
    m = RandomForestRegressor(n_jobs=-1,n_estimators=n_estimators,max_samples=120_000, max_features=0.5, min_samples_leaf=5, oob_score=True)
    m.fit(xs,y)
    return m.oob_score_


get_oob(xs,y)


get_oob(xs,y,n_estimators=100)


get_oob(xs,y,n_estimators=500)


import warnings
warnings.simplefilter('ignore', UserWarning)


preds=np.stack([t.predict(valid_xs) for t in m.estimators_])


plt.plot([np.sqrt(mean_squared_error(valid_y,preds[:i+1].mean(0))) for i in range(preds.shape[0])]);


np.sqrt(mean_squared_error(valid_y,preds.mean(0)))


def fi_df(df):
    return pd.DataFrame({'cols':df.columns, 
                         'imp':m.feature_importances_}).sort_values(by='imp',ascending=False)


fi=fi_df(xs)


fi


fi.plot.barh('cols','imp')


to_keep = fi[fi['imp']>0.005].cols
to_keep


len(to_keep),len(xs.columns)


xs_imp=xs[to_keep]
valid_xs_imp=valid_xs[to_keep]


xs_imp.head()


m = rf(xs_imp,y);


rmse(m,xs_imp,y), rmse(m,valid_xs_imp,valid_y)


get_oob(xs_imp,y)


from scipy.cluster import hierarchy as hc

def cluster_columns(df, figsize=(10,6), font_size=12):
    corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
    corr_condensed = hc.distance.squareform(1-corr)
    z = hc.linkage(corr_condensed, method='average')
    fig = plt.figure(figsize=figsize)
    hc.dendrogram(z, labels=df.columns, orientation='left', leaf_font_size=font_size)
    plt.show()


cluster_columns(xs_imp)


from sklearn.inspection import PartialDependenceDisplay
fig,ax=plt.subplots(figsize=(12,4))
PartialDependenceDisplay.from_estimator(m,valid_xs_imp,['Weight Capacity (kg)','Compartments'],grid_resolution=20, ax=ax);


preds.std(0)[:5]


preds.std(0).max()


preds.std(0).min()

