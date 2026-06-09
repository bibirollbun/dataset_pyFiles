from fastai.tabular.all import * 
from fastai.test_utils import show_install
import seaborn as sns
import matplotlib.pyplot as plt


from IPython.display import display, clear_output
import seaborn as sns


from fastai.test_utils import show_install
show_install()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


def set_seed_value(seed=718):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed_value()


path = Path('/kaggle/input/playground-series-s5e10')
Path.BASE_PATH = path
path.ls()


train_df = pd.read_csv(os.path.join(path, 'train.csv')).set_index('id')
test_df = pd.read_csv(os.path.join(path, 'test.csv')).set_index('id')
sample_submission = pd.read_csv(os.path.join(path, 'sample_submission.csv'))

dep_var = "accident_risk"


original_df = pd.read_csv("/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv")


train_df.head(10)


train_df.describe()


train_df.info()


train_df.isna().sum()


for col in train_df.columns:
   print(col , train_df[col].nunique())


corr = train_df.select_dtypes(exclude = [object]).corr()

fig, axes = plt.subplots(figsize=(10 , 10))
sns.heatmap(corr,  linewidths=.5, annot=True, cmap='rainbow')

plt.show()


train_df[dep_var].hist(bins=25)


def do_correct_curvature(df, shift_factor=-0.35):
    used_indexes = df[df.loc[:, dep_var] == 0.0].index
    df.loc[used_indexes, "curvature"] += shift_factor

    used_indexes = df[df.loc[:, dep_var] == 0.1].index
    df.loc[used_indexes, "curvature"] += shift_factor

    used_indexes = df[df.loc[:, dep_var] == 0.2].index
    df.loc[used_indexes, "curvature"] += shift_factor

    return df


def risc_factor(X): 
    return \
    0.3 * X["curvature"] + \
    0.2 * (X["lighting"] == "night").astype(int) + \
    0.1 * (X["weather"] != "clear").astype(int) + \
    0.2 * (X["speed_limit"] >= 60).astype(int) + \
    0.1 * (X["num_reported_accidents"] > 2).astype(int)

def clip(f):
    def clip_f(X):
        sigma = 0.05
        mu = f(X)
        a, b = -mu/sigma, (1-mu)/sigma
        Phi_a, Phi_b = scipy.stats.norm.cdf(a), scipy.stats.norm.cdf(b)
        phi_a, phi_b = scipy.stats.norm.pdf(a), scipy.stats.norm.pdf(b)
        return mu*(Phi_b-Phi_a)+sigma*(phi_a-phi_b)+1-Phi_b
    return clip_f



def do_preprocess(df):

    z= clip(risc_factor)(df)
    df["risk_factor"] =  z.values * 100
    df["risk_factor"] = df["risk_factor"].astype("int8")
    return df


def do_add_original(df, orig_df):

    copy_df = df.copy()
    dep_var_mean = orig_df[dep_var].mean()
    dep_var_std = orig_df[dep_var].std()
    for col in copy_df.columns:
        if col != dep_var:
            new_col = f"te_{col}_mean"
            copy_df[new_col] = copy_df[col].map( orig_df.groupby(col).accident_risk.std() ).fillna(dep_var_mean) 
            
    return copy_df


train_df = train_df[train_df.accident_risk <1.0]


train_df = do_preprocess(train_df)
original_df = do_preprocess(original_df)
test_df = do_preprocess(test_df)


train_df = do_add_original(train_df, original_df)
test_df = do_add_original(test_df, original_df)


train_df.duplicated().value_counts()
train_df.drop_duplicates(inplace=True)


for col in train_df.columns:
   print(col , train_df[col].nunique())


train_df.head(10)


plt.scatter(train_df["risk_factor"], train_df[dep_var], cmap='brg',  marker='o', s=1)
plt.ylabel(dep_var)
plt.xlabel("risk_factor")
plt.show()


cont_vars, cat_vars = cont_cat_split(train_df, dep_var=dep_var, max_card=256)
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


dls = getData(train_df, batchSize=512)
len(dls.train), len(dls.valid)


dls.show_batch()


def my_rmse(y_pred, target):
    return torch.sqrt(F.mse_loss(y_pred, target)).mean()


my_config = tabular_config(ps=0.2, embed_p=0.2, use_bn=True,  y_range=(0, 1.0))

learn = tabular_learner(dls,
                        config = my_config,
                        loss_func= my_rmse,
                        layers = [128,256,256,128],
                        )
learn.summary()


lr_min,lr_steep, lr_valley = learn.lr_find(suggest_funcs=(minimum, steep, valley))


print(f"Minimum: {lr_min:.2e}, steepest point: {lr_steep:.2e} valley point: {lr_valley:.2e}")


learn.fit_one_cycle(150, 5e-3, wd=0.01, cbs=[SaveModelCallback(fname='kaggle_s5e10', with_opt=False)])


learn.load('kaggle_s5e10')


learn.show_results()


dlt = learn.dls.test_dl(test_df) 
nn_preds,_ ,preds = learn.get_preds(dl=dlt, with_decoded=True) 

preds, preds.min(), preds.max()


sample_submission[dep_var] =  preds
sample_submission.to_csv("submission.csv", index=False)
sample_submission.head(10)


sample_submission[dep_var].hist(bins=30)


!ls -la

