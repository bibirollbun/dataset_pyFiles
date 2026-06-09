import os
from pathlib import Path

iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')
if iskaggle: path = Path('../input/playground-series-s5e1')
else:
    path = Path('playground-series-s5e1')
    if not path.exists():
        import zipfile,kaggle
        kaggle.api.competition_download_cli(str(path))
        zipfile.ZipFile(f'{path}.zip').extractall(path)


from fastai.tabular.all import *
# For controlling the float format
pd.set_option('display.float_format', '{:,.2f}'.format)

df = pd.read_csv(path/'train.csv')
df.head()


df_test = pd.read_csv(path/'test.csv')


df_test.head()


dep_var='num_sold'


df.describe(include=np.number)


df.describe(include=['object'])


df["date"] = df["date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))


df_test["date"] = df_test["date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))


df.describe(include='datetime')


df["date"].dt.year.unique()


df["date"].dt.month.unique()


df_test["date"].dt.year.unique()


delta = datetime(2025,1,27)-datetime(2025, 1, 20)
delta.days


today=datetime.now()
print(f"You can verify that today is day number {today.weekday()} of the week and week number {today.strftime('%U')} of the year.")


df=add_datepart(df,"date")
df.head()


df.isna().sum()


median=df["num_sold"].median()
df["num_sold"]=df["num_sold"].fillna(median)
df.isna().sum()


procs=[Categorify]


cont,cat=cont_cat_split(df,1,dep_var)


print(f"Continuous variables: {cont}")
print(f"Categorical variables: {cat}")


cond = df["Year"]<2014
train_idx = np.where(cond)[0]
valid_idx = np.where(~cond)[0]
splits = (list(train_idx),list(valid_idx))


len(splits[0])


to = TabularPandas(df, procs, cat, cont, dep_var, splits=splits)


train_xs, train_y = to.train.xs, to.train.y
valid_xs, valid_y = to.valid.xs, to.valid.y


train_xs.dtypes


valid_xs.dtypes


from sklearn.tree import DecisionTreeRegressor, export_graphviz

m = DecisionTreeRegressor(max_leaf_nodes=4)
m.fit(train_xs,train_y);


import graphviz

def draw_tree(t, df, size=10, ratio=0.6, precision=2, **kwargs):
    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True, leaves_parallel=True, rounded=True,
                      special_characters=True, rotate=False, precision=precision, **kwargs)
    return graphviz.Source(re.sub('Tree {', f'Tree {{ size={size}; ratio={ratio}', s))


draw_tree(m,train_xs,size=7)


value = train_y.mean()
value


squared_error=((train_y - value)**2).mean()
squared_error


train_xs["country"].unique()


train_xs["product"].unique()


train_xs["store"].unique()


# Tree without limits
m = DecisionTreeRegressor()
m.fit(train_xs,train_y);


(m.get_n_leaves(),len(train_xs))


from sklearn.metrics import mean_absolute_percentage_error

def mape(m,xs,y):
    return mean_absolute_percentage_error(y,m.predict(xs))


mape(m,train_xs,train_y)


mape(m,valid_xs,valid_y)


m=DecisionTreeRegressor(min_samples_leaf=10)
m.fit(train_xs,train_y)
m.get_n_leaves(), len(train_xs)


mape(m,train_xs,train_y), mape(m,valid_xs,valid_y)


df_test = add_datepart(df_test,"date")
to_test = TabularPandas(df_test, procs, cat, cont)


test_xs = to_test.xs


predictions = m.predict(test_xs)


submission = pd.DataFrame({
    "id": df_test["id"],
    "num_sold": predictions
})
submission.head()


submission.to_csv("submission_decision_tree.csv",index=False)

