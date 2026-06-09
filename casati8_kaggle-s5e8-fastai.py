from fastai.tabular.all import * 
from fastai.test_utils import show_install

from sklearn.metrics import roc_auc_score

import seaborn as sns


show_install()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


def set_seed_value(seed=718):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed_value()


path = Path('/kaggle/input/playground-series-s5e8')
Path.BASE_PATH = path
path.ls()


train_df = pd.read_csv(os.path.join(path, 'train.csv')).set_index('id')
test_df = pd.read_csv(os.path.join(path, 'test.csv')).set_index('id')
sample_submission = pd.read_csv(os.path.join(path, 'sample_submission.csv'))

dep_var = "y"


path = Path('/kaggle/input/bank-marketing-dataset-full')
Path.BASE_PATH = path
path.ls()


original_df = pd.read_csv(os.path.join(path, 'bank-full.csv'), sep=';')


train_df.describe().T


train_df.info()


train_df.isna().sum()


test_df.isna().sum()


corr = train_df.select_dtypes(exclude = [object]).corr()

fig, axes = plt.subplots(figsize=(10 , 10))
sns.heatmap(corr,  linewidths=.5, annot=True, cmap='rainbow')

plt.show()


train_df[dep_var].hist()


def do_preprocess(df):
    cols_to_convert = list(df.columns[(df.dtypes.values == np.dtype('object'))])
    for col in cols_to_convert:
        df[[col]] = df[[col]].apply(lambda col:pd.Categorical(col).codes).astype(int)

    for c in ["balance","duration", "campaign", "previous"]:
        for i in range(1,6):
            df[f'{c}_{i}']=df[c]//(10**(i))%10
            
    df["balance"] = np.log1p(df["balance"].clip(lower=0)).astype(float)
    df['contacted_before'] = (df["pdays"] != -1).astype(int)

    return df


def add_periodicals(df):

    my_dict = { "duration": 400}

    for key, value in my_dict.items():
        df[f"{key}_sin"] = np.sin(2 * np.pi * df[key] / value)
        df[f"{key}_cos"] = np.cos(2 * np.pi * df[key] / value)
    
    return df


def do_add_original(df, orig_df):

    copy_df = df.copy()
    dep_var_mean = train_df[dep_var].mean()
    dep_var_std = train_df[dep_var].std()
    
    for col in copy_df.columns:
        if col != dep_var:
            new_col = f"{col}_new"
            copy_df[new_col] = copy_df[col].map( orig_df.groupby(col).y.mean() ).fillna(dep_var_mean)      

            new_col = f"{col}_std"
            copy_df[new_col] = copy_df[col].map( orig_df.groupby(col).y.std() ).fillna(dep_var_std) 
            
    return copy_df


train_df = do_preprocess(train_df)
test_df = do_preprocess(test_df)
original_df = do_preprocess(original_df)


train_df = do_add_original(train_df, original_df)
test_df = do_add_original(test_df, original_df)


train_df = add_periodicals(train_df)
test_df = add_periodicals(test_df)


for col in list(train_df.columns[(train_df.dtypes.values == np.dtype('int64'))]):
    print (col, train_df[col].nunique())


cont_vars, cat_vars = cont_cat_split(train_df, dep_var=dep_var, max_card=2048)
len(cont_vars), len(cat_vars), cont_vars, cat_vars


def getData(df, batchSize=512):
       
    to_train = TabularPandas(df, 
                           [Normalize, FillMissing,  Categorify],
                            cat_names=cat_vars,
                            cont_names=cont_vars, 
                            splits=RandomSplitter(valid_pct=0.20)(df),  
                            device = device,
                            y_block = RegressionBlock(),
                            y_names=dep_var) 

    return to_train.dataloaders(bs=batchSize)


dls = getData(train_df, batchSize=1024)
len(dls.train), len(dls.valid)


dls.show_batch()


def my_roc_auc(inp, targ):
    "Simple wrapper around scikit's roc_auc_score function for regression problems"
    inp,targ = flatten_check(inp,targ)
    return roc_auc_score(targ.cpu().numpy(), inp.cpu().numpy())


my_config = tabular_config(ps=0.2, embed_p=0.2, use_bn=True,  y_range=(0, 1.0))

my_loss = CrossEntropyLossFlat()

learn = tabular_learner(dls,
                        config = my_config,
                        metrics = [my_roc_auc],
                        #  layers = [512,256],                  
                        )

learn.summary()


lr_min,lr_steep, lr_valley = learn.lr_find(suggest_funcs=(minimum, steep, valley))


print(f"Minimum: {lr_min:.2e}, steepest point: {lr_steep:.2e} valley point: {lr_valley:.2e}")


learn.fit_one_cycle(30, 2e-2, wd=0.01, cbs=[SaveModelCallback(fname='kaggle_s5e8')])


learn.show_results()


learn.load('kaggle_s5e8')


dlt = learn.dls.test_dl(test_df) 
nn_preds,_ ,preds = learn.get_preds(dl=dlt , with_decoded=True) 

preds, preds.min(), preds.max()


sample_submission[dep_var] =  preds
sample_submission.to_csv("submission.csv", index=False)
sample_submission.head(10)


ls -la 

