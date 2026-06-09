from fastai.tabular.all import * 
from fastai.test_utils import show_install
import seaborn as sns

from IPython.display import display, clear_output
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import PowerTransformer

show_install()


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


class TargetProcessor:

    def __init__(self, dep_var, mode="log1p"):
        self._dep_var = dep_var
        self._mode = mode
        self._pt_offset = 3.0
        self._transformer = PowerTransformer(method=mode)

    def transform(self, df):
        if self._mode == "log1p":
            print ("log fransform")
            transformed = np.log1p(df[dep_var])
        else:
            print ("power fransform")
            self._transformer.fit(df[self._dep_var].values.reshape(-1, 1))
            transformed = self._transformer.transform(df[self._dep_var].values.reshape(-1, 1)) + self._pt_offset

        return transformed

    def re_transform(self, preds):
        if self._mode == "log1p":
           reTransformed = np.expm1(preds)
        else:
            reTransformed = self._transformer.inverse_transform(preds - self._pt_offset )
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
                           splits=RandomSplitter(valid_pct=0.30)(df),  
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



my_config = tabular_config(ps=0.3, embed_p=0.2, use_bn=True, y_range=(0, 6.0))

learn = tabular_learner(dls,
                        config = my_config,
                        metrics=[rmse, msle] ,
                        loss_func= RMSLELoss(),
                        )

learn.summary()


lr_min,lr_steep, lr_valley = learn.lr_find(suggest_funcs=(minimum, steep, valley))


print(f"Minimum: {lr_min:.2e}, steepest point: {lr_steep:.2e} valley point: {lr_valley:.2e}")


learn.fit_one_cycle(50, 1e-3, wd=0.1, cbs=[SaveModelCallback(fname='kaggle_season5_exp5', with_opt=True)])


learn.show_results()


learn.load('kaggle_season5_exp5')


dlt = learn.dls.test_dl(test_df) 
nn_preds,_ ,preds = learn.get_preds(dl=dlt , with_decoded=True) 

preds, preds.min(), preds.max()


sample_submission[dep_var] =  myTransformer.re_transform(preds)

sample_submission.to_csv("submission.csv", index=False)
sample_submission.head(10)


!ls -la


class PermutationImportance():
      "Calculate and plot the permutation importance"
      def __init__(self, learn:Learner, df=None, bs=512):
        "Initialize with a test dataframe, a learner, and a metric"
        self.learn = learn
        self.df = df
        bs = bs if bs is not None else learn.dls.bs
        if self.df is not None:
          self.dl = learn.dls.test_dl(self.df, bs=bs)
        else:
          self.dl = learn.dls[1]
        self.x_names = learn.dls.x_names.filter(lambda x: '_na' not in x)
        self.na = learn.dls.x_names.filter(lambda x: '_na' in x)
        self.y = dls.y_names
        self.results = self.calc_feat_importance()
        self.plot_importance(self.ord_dic_to_df(self.results))
            
      def get_results(self): return self.results
        
      def measure_col(self, name:str):
          "Measures change after column shuffle"
          col = [name]
          if f'{name}_na' in self.na: col.append(name)
          orig = self.dl.items[col].values
          perm = np.random.permutation(len(orig))
          self.dl.items[col] = self.dl.items[col].values[perm]
          metric = learn.validate(dl=self.dl)[1]
          self.dl.items[col] = orig
          clear_output()
          return metric

      def calc_feat_importance(self):
          "Calculates permutation importance by shuffling a column on a percentage scale"
          print('Getting base error')
          base_error = self.learn.validate(dl=self.dl)[1]
          self.importance = {}
          pbar = progress_bar(self.x_names)
          print('Calculating Permutation Importance')
          for col in pbar:
            self.importance[col] = self.measure_col(col)
          for key, value in self.importance.items():# for col in ['Latitude', 'Longitude']:
            self.importance[key] = np.abs(base_error-value)/base_error #this can be adjusted
          return OrderedDict(sorted(self.importance.items(), key=lambda kv: kv[1], reverse=True))

      def ord_dic_to_df(self, dict:OrderedDict):
          return pd.DataFrame([[k, v] for k, v in dict.items()], columns=['feature', 'importance'])

      def plot_importance(self, df:pd.DataFrame, limit=20, asc=False, **kwargs):
          "Plot importance with an optional limit to how many variables shown"
          df_copy = df.copy()
          df_copy['feature'] = df_copy['feature'].str.slice(0,40)
          df_copy = df_copy.sort_values(by='importance', ascending=asc)[:limit].sort_values(by='importance', ascending=not(asc))
          ax = df_copy.plot.barh(x='feature', y='importance',  **kwargs)
          for p in ax.patches:
            ax.annotate(f'{p.get_width():.4f}', ((p.get_width() * 1.005), p.get_y()  * 1.005))


res = PermutationImportance(learn, train_df)
res.get_results()

