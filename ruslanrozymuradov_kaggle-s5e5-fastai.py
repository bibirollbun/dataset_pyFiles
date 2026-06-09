from fastai.tabular.all import * 
from fastai.test_utils import show_install
import seaborn as sns

from IPython.display import display, clear_output
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import PowerTransformer
import random
import numpy as np
import torch


import pandas as pd
from collections import OrderedDict
from fastprogress.fastprogress import progress_bar
from IPython.display import clear_output
import matplotlib.pyplot as plt
import copy

show_install()


from fastai.test_utils import show_install
show_install()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




def set_seed_value(seed=718):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  

set_seed_value()



path = Path('/kaggle/input/playground-series-s5e5')
Path.BASE_PATH = path
path.ls()


train_df = pd.read_csv(os.path.join(path, 'train.csv')).set_index('id')
test_df = pd.read_csv(os.path.join(path, 'test.csv')).set_index('id')
sample_submission = pd.read_csv(os.path.join(path, 'sample_submission.csv'))

dep_var = 'Calories'


train_df.head()


train_df.info()


train_df[dep_var].hist(bins=50)


corr = train_df.select_dtypes(exclude = [object]).corr()

fig, axes = plt.subplots(figsize=(9, 8))

sns.heatmap(corr,  linewidths=.5, annot=True, cmap='rainbow')

plt.show()


from sklearn.preprocessing import PowerTransformer
import numpy as np

class TargetProcessor:
    def __init__(self, dep_var, mode="log1p"):
        self._dep_var = dep_var
        self._mode = mode
        
        if self._mode not in ["log1p", "power"]:
            raise ValueError("mode must be either 'log1p' or 'power'")
        
        self._pt_offset = 3.0
        self._transformer = None  # PowerTransformer objesi fit edildikten sonra atanacak
    
    def fit(self, df):
        if self._mode == "power":
            self._transformer = PowerTransformer(method='yeo-johnson')
            self._transformer.fit(df[self._dep_var].values.reshape(-1, 1))
    
    def transform(self, df):
        if self._mode == "log1p":
            transformed = np.log1p(df[self._dep_var].values)
        else:  # power
            if self._transformer is None:
                raise RuntimeError("PowerTransformer not fitted. Call fit() before transform().")
            transformed = self._transformer.transform(df[self._dep_var].values.reshape(-1, 1)) + self._pt_offset
            transformed = transformed.flatten()
        return transformed
    
    def re_transform(self, preds):
        if self._mode == "log1p":
            reTransformed = np.expm1(preds)
        else:
            if self._transformer is None:
                raise RuntimeError("PowerTransformer not fitted. Call fit() before re_transform().")
            reTransformed = self._transformer.inverse_transform((preds - self._pt_offset).reshape(-1, 1))
            reTransformed = reTransformed.flatten()
        return reTransformed



myTransformer = TargetProcessor(dep_var)


def do_feature_engineering(df):

    heart_rate_mean = df['Heart_Rate'].mean()
    df['BMI'] = np.round(df['Weight'] / ((df['Height'] / 100.0) ** 2),2)
    df["HR_Div"] = df['Heart_Rate'] - heart_rate_mean

    df['Intensity'] = df['Heart_Rate'] / df['Duration']
    df['Temp_Duration'] = df['Body_Temp'] * df['Duration']
    df["Body_Temp"] =  df["Body_Temp"] -37.0
    
    columns = df.columns.tolist()
    if dep_var in columns:
       df[dep_var] = myTransformer.transform(df)

    df.drop(["Height"], axis =1,inplace = True)
    return df


train_df = do_feature_engineering(train_df)
test_df = do_feature_engineering(test_df)


train_df[dep_var].hist(bins=50)


cont_vars, cat_vars = cont_cat_split(train_df, dep_var=dep_var)
len(cont_vars), len(cat_vars),cont_vars,cat_vars


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


dls.show()


class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))



my_config = tabular_config(ps=0.2, embed_p=0.2, use_bn=True, y_range=(0, 6.0))

learn = tabular_learner(dls,
                        config = my_config,
                        metrics=[rmse, msle] ,
                        loss_func= RMSLELoss(),
                        )

learn.summary()


lr_min,lr_steep, lr_valley = learn.lr_find(suggest_funcs=(minimum, steep, valley))


print(f"Minimum: {lr_min:.2e}, steepest point: {lr_steep:.2e} valley point: {lr_valley:.2e}")


learn.fit_one_cycle(50, 1e-3, wd=0.1, cbs=SaveModelCallback(fname='kaggle_season5_exp5', with_opt=True))


learn.show_results()


learn.load('kaggle_season5_exp5')


dlt = learn.dls.test_dl(test_df) 
nn_preds,_ ,preds = learn.get_preds(dl=dlt , with_decoded=True) 

preds, preds.min(), preds.max()


sample_submission[dep_var] =  myTransformer.re_transform(preds)

sample_submission.to_csv("submission.csv", index=False)
sample_submission.head(10)


!ls -la


import numpy as np
import pandas as pd
from collections import OrderedDict
from fastprogress.fastprogress import progress_bar
from IPython.display import clear_output
import matplotlib.pyplot as plt
import copy

class PermutationImportance:
    "Calculate and plot permutation importance safely and efficiently"

    def __init__(self, learn, df=None, bs=512, metric_idx=1):
        """
        learn: fastai Learner object
        df: test dataframe (optional)
        bs: batch size
        metric_idx: index of the metric to use in learn.validate()
        """
        self.learn = learn
        self.df = df
        self.bs = bs if bs is not None else learn.dls.bs
        self.metric_idx = metric_idx
        
        if self.df is not None:
            self.dl = learn.dls.test_dl(self.df, bs=self.bs)
            self.dataset = self.df.copy()
        else:
            self.dl = learn.dls[1]
            # Dataset extraction (pandas DataFrame) güvenli şekilde yapmak lazım, burada sadece referans veriliyor:
            self.dataset = self.dl.items.copy() if hasattr(self.dl, 'items') else None
        
        self.x_names = [x for x in learn.dls.x_names if '_na' not in x]
        self.na_features = [x for x in learn.dls.x_names if '_na' in x]
        self.importance = None
        
    def _permute_column(self, df, col):
        "Return a new dataframe with the given column permuted"
        df_copy = df.copy()
        # Eğer ilgili '_na' feature varsa beraber permute et
        related_cols = [col]
        na_col = f'{col}_na'
        if na_col in df_copy.columns:
            related_cols.append(na_col)
        for c in related_cols:
            df_copy[c] = np.random.permutation(df_copy[c].values)
        return df_copy

    def measure_col(self, col):
        "Shuffle one column and calculate metric"
        permuted_df = self._permute_column(self.dataset, col)
        # Yeni dataloader oluştur
        perm_dl = self.learn.dls.test_dl(permuted_df, bs=self.bs)
        # Model performansını ölç
        metric = self.learn.validate(dl=perm_dl)[self.metric_idx]
        return metric

    def calc_feat_importance(self):
        "Calculate permutation importance for all features"
        print('Calculating base error on original data...')
        base_error = self.learn.validate(dl=self.dl)[self.metric_idx]
        
        self.importance = {}
        pbar = progress_bar(self.x_names)
        
        for col in pbar:
            score = self.measure_col(col)
            # Importance: göreceli performans düşüşü
            self.importance[col] = np.abs(base_error - score) / base_error
        
        self.importance = OrderedDict(sorted(self.importance.items(), key=lambda kv: kv[1], reverse=True))
        return self.importance

    def ord_dic_to_df(self, importance_dict):
        return pd.DataFrame(list(importance_dict.items()), columns=['feature', 'importance'])

    def plot_importance(self, df=None, limit=20, asc=False, figsize=(10, 6), **kwargs):
        if df is None:
            if self.importance is None:
                raise ValueError("Calculate importance first!")
            df = self.ord_dic_to_df(self.importance)

        df = df.copy()
        df['feature'] = df['feature'].str.slice(0, 40)
        df = df.sort_values(by='importance', ascending=asc).head(limit)
        df = df.sort_values(by='importance', ascending=not asc)

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(df['feature'], df['importance'], **kwargs)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title('Permutation Feature Importance')

        for bar in bars:
            width = bar.get_width()
            ax.annotate(f'{width:.4f}', xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0), textcoords='offset points', ha='left', va='center')
        plt.show()



res = PermutationImportance(learn, train_df)
res.get_results()

