from fastai.tabular.all import * 
from fastai.test_utils import show_install
import seaborn as sns

from IPython.display import display, clear_output

show_install()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


def set_seed_value(seed=718):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed_value()


path = Path('/kaggle/input/playground-series-s5e4')
Path.BASE_PATH = path
path.ls()


train_df = pd.read_csv(os.path.join(path, 'train.csv')).set_index('id')
test_df = pd.read_csv(os.path.join(path, 'test.csv')).set_index('id')
sample_submission = pd.read_csv(os.path.join(path, 'sample_submission.csv'))

dep_var = 'Listening_Time_minutes'
from IPython.display import display, clear_output


train_df.describe()


train_df.info()


train_df.head()


train_df[dep_var].hist(bins=50)


train_df.isna().sum()


test_df.isna().sum()


train_df.describe()


def do_add_descimal_digit(df):
                      
    def get_decimal_digits(floatValue):
        if pd.isna(floatValue):
            return 0
        stringValue = str(floatValue)
        if '.' in stringValue:
            return len(stringValue.split('.')[1]) 
        return 0
        
    df['decimal_digits'] = df['Episode_Length_minutes'].apply(get_decimal_digits)
    df['decimal_digits'] = df['decimal_digits'].astype(int)
    return df



def do_encode(df):
    cols_to_transform = ["Podcast_Name", "Genre", "Episode_Title", "Podcast_Name", "Publication_Day", "Publication_Time", "Episode_Sentiment" ]
    for c in cols_to_transform:
        df[[c]] = df[[c]].apply(lambda col:pd.Categorical(col).codes)

    return df


train_df = do_encode(train_df)
test_df = do_encode(test_df)


def do_impute(df):
    numeric_features = ['Episode_Length_minutes', 'Number_of_Ads','Guest_Popularity_percentage']

    for col in numeric_features:
      df[col] = df.groupby(['Podcast_Name', 'Episode_Title'])[col].transform(lambda x: x.fillna(x.mean()))
    return df


train_df = do_impute(train_df)
test_df = do_impute(test_df)


train_df = do_add_descimal_digit(train_df)
test_df = do_add_descimal_digit(test_df)


train_df = train_df[train_df['decimal_digits'] < 3]
train_df = train_df[train_df['Episode_Length_minutes'] <= 140]


def do_transform(df):
   
    for col in ['Host_Popularity_percentage', 'Guest_Popularity_percentage']:
        df[col] = np.clip(df[col], a_min=0, a_max=100)

    df["Number_of_Ads"] = np.clip(df["Number_of_Ads"], a_min=0, a_max=5)
           
    cols = ["Number_of_Ads","Episode_Length_minutes" , "Host_Popularity_percentage", "Guest_Popularity_percentage"]
    for col in cols:
        df[col] = df[col].astype(int)


    combis = [
        ['Episode_Length_minutes', 'Host_Popularity_percentage'],
        #['Episode_Length_minutes', 'Guest_Popularity_percentage'],
        ['Episode_Length_minutes', 'Number_of_Ads'],
       # ['Host_Popularity_percentage', 'Guest_Popularity_percentage'],
       # ['Host_Popularity_percentage', 'Number_of_Ads']
        # ['Host_Popularity_percentage', 'Episode_Sentiment'] 
    ]

    for comb in combis:
        name = '--'.join(comb)  

        a = 0
        for idx, col in enumerate(comb):   
            a += 10** (idx *2) * df[comb[idx]]
        
        df[name] = a
        df[name] = df[name].astype(int)  

    cols_to_drop = ["Genre", "Number_of_Ads"]
    df.drop(cols_to_drop,axis =1,inplace = True)

    return df


train_df = do_transform(train_df)
test_df = do_transform(test_df)


corr = train_df.corr()

ig, axes = plt.subplots(figsize=(20, 20))
sns.heatmap(corr, linewidths=.5, annot=True, cmap='rainbow')

plt.show()


train_df.info()


cont_vars, cat_vars = cont_cat_split(train_df, dep_var=dep_var, max_card=128)
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


dls = getData(train_df) 
len(dls.train), len(dls.valid)


dls.show()


def my_rmse(y_pred, target):
    return  torch.sqrt(F.mse_loss(y_pred, target)).mean()


my_config = tabular_config(ps=0.2, embed_p=0.2, use_bn=True, y_range=(0,120))

learn = tabular_learner(dls, 
                        config = my_config,
                         metrics=[rmse, mae] ,
                        loss_func= my_rmse,
                        )

learn.summary()


lr_min,lr_steep, lr_valley = learn.lr_find(suggest_funcs=(minimum, steep, valley))


print(f"Minimum: {lr_min:.2e}, steepest point: {lr_steep:.2e} valley point: {lr_valley:.2e}")


learn.fit_one_cycle(50, 3e-3, wd=0.1, cbs=SaveModelCallback(fname="kaggle_s5e4", with_opt=True))


learn.show_results()


learn.load("kaggle_s5e4")


dlt = learn.dls.test_dl(test_df) 
nn_preds,_ ,preds = learn.get_preds(dl=dlt , with_decoded=True) 

preds, preds.min(), preds.max()


sample_submission[dep_var] = preds

sample_submission.to_csv("submission.csv", index=False)
sample_submission.head(10)


sample_submission[dep_var].hist(bins=50)


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


#res = PermutationImportance(learn, train_df)
#res.get_results()

