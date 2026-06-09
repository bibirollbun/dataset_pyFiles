


%%time 

!pip install -q -r /kaggle/input/playgrounds5e01-public-imports-v1/req_kaggle.txt

exec(open(f"/kaggle/input/playgrounds5e01-public-imports-v1/myimports.py", "r").read())
exec(open(f"/kaggle/input/playgrounds5e01-public-imports-v1/mypp.py", "r").read())
exec(open(f"/kaggle/input/playgrounds5e01-public-imports-v1/myutils.py", "r").read())
exec(open(f"/kaggle/input/playgrounds5e01-public-imports-v1/mytraining.py", "r").read())


%%time 

class CFG:
    """
    Configuration class for parameters and CV strategy for tuning and training
    Some parameters may be unused here as this is a general configuration class
    """;

    # Data preparation:-
    version_nb  = 1
    model_id    = "V1_7"
    model_label = "ML"

    test_req           = False
    test_sample_frac   = 0.01

    gpu_switch         = "OFF"
    state              = 42
    target             = f"num_sold"
    grouper            = f""
    tgt_mapper         = {}

    ip_path            = f"/kaggle/input/playground-series-s5e1"
    op_path            = f"/kaggle/working"
    orig_path          = f""

    dtl_preproc_req    = False
    ftre_plots_req     = False
    ftre_imp_req       = True

    nb_orig            = 0
    orig_all_folds     = False

    # Model Training:-
    pstprcs_oof        = True
    pstprcs_train      = True
    pstprcs_test       = True
    
    ML                 = False
    test_preds_req     = False

    pseudo_lbl_req     = "N"
    pseudolbl_up       = 0.975
    pseudolbl_low      = 0.00

    n_splits           = 3 if test_req == True else 5
    n_repeats          = 1
    nbrnd_erly_stp     = 50
    mdlcv_mthd         = 'GKF'

    # Ensemble:-
    ensemble_req       = True
    optuna_req         = False
    metric_obj         = 'minimize'
    ntrials            = 10 if test_req == True else 300

    # Global variables for plotting:-
    grid_specs = {'visible'  : True,
                  'which'    : 'both',
                  'linestyle': '--',
                  'color'    : 'lightgrey',
                  'linewidth': 0.75
                 }

    title_specs = {'fontsize'   : 9,
                   'fontweight' : 'bold',
                   'color'      : '#992600',
                  }

PrintColor(f"\n---> Configuration done!\n")

cv_selector = \
{
 "RKF"   : RKF(n_splits = CFG.n_splits, n_repeats= CFG.n_repeats, random_state= CFG.state),
 "RSKF"  : RSKF(n_splits = CFG.n_splits, n_repeats= CFG.n_repeats, random_state= CFG.state),
 "SKF"   : SKF(n_splits = CFG.n_splits, shuffle = True, random_state= CFG.state),
 "KF"    : KFold(n_splits = CFG.n_splits, shuffle = True, random_state= CFG.state),
 "GKF"   : GKF(n_splits = CFG.n_splits)
}

collect()



%%time 

pp = Preprocessor(2010, 2019)
pp.DoPreprocessing()


%%time

# Collecting the GDP figures
print()
gdp = GDPRequestor(country = pp.train.country.unique(), )
gdp_snp = gdp.ScrapGDP()
gdp_snp = \
gdp_snp.reset_index().\
melt(id_vars = ['index']).\
rename({'index': 'country', 'variable': 'year', 'value': 'GDP'}, axis=1)


%%time 

display(
    pp.train[["country", "store", "product", CFG.target]].\
    groupby(["country", "store", "product"])[CFG.target].\
    apply(lambda x : x.isna().sum()).\
    reset_index().\
    style.\
    set_caption(
        f"Train data - null targets by country, store and product"
    ).
    applymap(
        lambda x : "color : blue; font-size : 16px ; font-weight: bold; border: dashed maroon 1.5px" 
        if x > 0 else "color : black; background-color : white",
        subset = [CFG.target]
    )
)


%%time 

xform = \
ColumnTransformer([("E", OrdinalEncoder(dtype = np.int16), ['country', 'store', 'product'])
                  ],
                  verbose_feature_names_out = False,
                  remainder = "passthrough",
                 ).set_output(transform = "pandas")

train = xform.fit_transform(pp.train)
train['Ftre_Comb_Lbl'] = \
train['country'].astype(str) + train['store'].astype(str) + train['product'].astype(str)

# Developing a link between the encoded labels and the original column values:-
PrintColor(
    f"\nFeature combinations between encoded values and original feature labels\n"
)
display(
    train[['country', 'store', 'product', 'Ftre_Comb_Lbl']].drop_duplicates().T
)


df = \
pd.concat(
    [train.groupby(["Ftre_Comb_Lbl"])[CFG.target].apply(lambda x : x.isna().sum()),
     train.groupby(["Ftre_Comb_Lbl"])[CFG.target].count()
    ], 
    axis = 1
)

df.columns = ["Null_Count", "Count"]

print("\n\n\n")
display(
    df.loc[df.Count == 0].
    style.
    set_caption(f"Combinations with all null targets")
)


%%time 

# Converting the data into a pivot for further analysis:-
Sales_Prf = train.pivot(index= 'date', columns= 'Ftre_Comb_Lbl', values = CFG.target);

Sales_Prf.insert(0, 'Year', Sales_Prf.index.year)
Sales_Prf['Year'] = Sales_Prf['Year'].astype(np.uint16)

Sales_Prf.insert(1, 'Month', Sales_Prf.index.month)
Sales_Prf['Month'] = Sales_Prf['Month'].astype(np.int8)

Sales_Prf.insert(2, 'Day', Sales_Prf.index.day)
Sales_Prf['Day'] = Sales_Prf['Day'].astype(np.int8)

Sales_Prf.insert(3, 'WeekNb', Sales_Prf.index.isocalendar().week)
Sales_Prf['WeekNb'] = Sales_Prf['WeekNb'].astype(np.int8)

Sales_Prf.insert(4, 'DayNb',Sales_Prf.index.weekday)
Sales_Prf['DayNb'] = Sales_Prf['DayNb'].astype(np.int8)

Sales_Prf.insert(5, 'IsWeekend', np.where(Sales_Prf.DayNb >= 5, 1, 0))
Sales_Prf['IsWeekend'] = Sales_Prf['IsWeekend'].astype(np.int8)

# Merging the sales profile table with public holidays:-
Sales_Prf = pp.Holidays.merge(Sales_Prf, how = 'right', left_index= True, right_index= True)
Sales_Prf[pp.Holidays.columns] = Sales_Prf[pp.Holidays.columns].fillna(0.0).astype(np.int8)


if CFG.ftre_plots_req :
    # Displaying the pivot information:-
    PrintColor(
        f"\nComplete pivot table columns for sales across all combinations\n"
    )
    
    with np.printoptions(linewidth = 150, threshold = 1000):
        print(np.array(Sales_Prf.columns))

    PrintColor(f"\nSales Profile information\n")
    display(
        Sales_Prf.info(verbose = True)
    )

Sales_Prf.drop(["000", "300"], axis=1, errors = "ignore", inplace = True)

PrintColor(f"\nComplete combinations\n");
Cols = Sales_Prf.columns[13:]
display(Cols)


%%time

# Creating xticks for end of quarters with labels:-
Date_Labels             = pd.DataFrame(data= Sales_Prf.index[Sales_Prf.index.is_quarter_end])
Date_Labels['date_lbl'] = Date_Labels['date'].dt.year*100 + Date_Labels['date'].dt.month


%%time

def MakeGrpLinePlot(
    dtpart:str, 
    figsize= (20,96),
    Sales_Prf = Sales_Prf,
    Cols = Cols
):
        
    _ = Sales_Prf[['Year', dtpart] + list(Cols)]
    _.insert(0, 'Id', _['Year']* 100.0 + _[dtpart])
    _['Id'] = _['Id'].astype(np.int32).astype(str)
    _ = _.drop(['Year', dtpart], axis=1).groupby('Id').agg([np.sum])
    _.columns = [j+'_'+i for i, j in _.columns]

    combs = [str(i) + str(j) for i,j in list(product(range(0,6,1), [0,1,2]))]

    with sns.axes_style("white"):
        fig, ax = plt.subplots(18,1, figsize = figsize, sharex= True)
        
        for i, comb in enumerate(combs):    
            sns.lineplot(data = _[_.columns[_.columns.str[4:6] == comb]],
                         palette= ['black','#014F1C', '#890E03','#034AD0', "cyan"],
                         ax= ax[i],
                         **{"linewidth": 2.0}
                        )
            
            ax[i].grid(**CFG.grid_specs)
            ax[i].set_ylabel('Units sold')
            ax[i].set_title(
                f"\nTotal units sold for combination = {comb}\n",
                **CFG.title_specs
            )
            ax[i].set_xlabel('')
        
        plt.suptitle(
            f"Total Sales by {dtpart}", 
            **CFG.title_specs,
            y = 1.0
        )
        
        plt.xticks(rotation= 90)
        plt.tight_layout()
        plt.show()

    del combs
    collect()


%%time

if CFG.ftre_plots_req :
    MakeGrpLinePlot(dtpart='Month',  figsize= (30,108))


%%time 

def PltDtPrtSales(
    df:pd.DataFrame, dtpart:str
):
    "This function plots the line plots for the given series to elicit seasonality and cyclicality"

    with sns.axes_style("white"):
    
        fig, ax = plt.subplots(2,3, figsize= (30,11), sharex= True, 
                               gridspec_kw = {'hspace': 0.25, "wspace": 0.2}
                              )
        
        for i in range(0,6,1):
            try:
                a = ax[(i) // 3, (i) % 3]
                df[df.columns[df.columns.str.startswith(str(i))]].plot.line(ax = a, marker = 'o')
                
                a.set_title(
                    f"\nTotal sales per product and store across country {i}\n", 
                    **CFG.title_specs
                )
                
                a.grid(**CFG.grid_specs)
                a.legend(loc = 'upper left', fontsize= 6)
                a.set(xlabel = '', ylabel = '')
                a.legend(bbox_to_anchor = (1,1))
            except:
                pass
        
        plt.suptitle(f"Total daily sales for year = {yy} across {dtpart.upper()}",
                     color = 'blue', 
                     fontsize = 12, 
                     fontweight = 'bold', 
                     y = 0.99
                    )
        plt.tight_layout()
        plt.show()
    collect()


%%time

if CFG.ftre_plots_req :
    for yy in tqdm(range(2010,2017,1)):
        PltDtPrtSales(
            df= \
            Sales_Prf.loc[Sales_Prf.Year == yy, ['DayNb'] + list(Cols)].groupby('DayNb').sum()/1000,
            dtpart = "DayNb"
        )
    
collect()
print()


%%time 

if CFG.ftre_plots_req :
    
    for yy in tqdm(range(2010, 2017,1)):
        PltDtPrtSales(
            df= Sales_Prf.loc[Sales_Prf.Year == yy, ['WeekNb'] + list(Cols)].groupby('WeekNb').sum()/1000,
            dtpart = "WeekNb"    
        )
        
print()
collect()


%%time 

if CFG.ftre_plots_req :
    
    for yy in tqdm(range(2010, 2017,1)):
        PltDtPrtSales(
            df= Sales_Prf.loc[Sales_Prf.Year == yy, ['Month'] + list(Cols)].groupby('Month').sum()/1000,
            dtpart = "Month"    
        )
        
print()
collect()


%%time 

def DisplayAdjTbl(*args):
    """
    This function displays pandas tables in an adjacent manner, sourced from the below link-
    https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side
    """
    
    html_str=''
    
    for df in args:
        html_str+= df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)

def CalcStoreComb(store1, store2):
    "This function calculates the monthly sales ratio across provided store ids"
    
    _ = \
    pd.DataFrame(
        np.sum(Sales_Prf[['Month', 'Year'] + list(Cols[Cols.str[1] == store1])].\
               groupby(['Year','Month']).sum(), axis=1)/ np.sum(Sales_Prf[['Month', 'Year'] + list(Cols[Cols.str[1] == store2])].\
                                                                groupby(['Year','Month']).sum(), axis=1
                                                               )
    ).\
    reset_index()
    
    _ = \
    _.pivot(index= 'Month', columns= 'Year').\
    style.highlight_max(props= "color:red;fontweight:bold;background:lightgrey").\
    format(precision= 4).set_caption(f"Sales Ratio {store1}{store2}").\
    set_table_attributes("style='display:inline'")

    return _
    
collect()


%%time

if CFG.ftre_plots_req :

    # Calculating store contribution ratios:-
    if CFG.ftre_plots_req :
        DisplayAdjTbl(*[CalcStoreComb(store1 = "0", store2 = "1"),
                        CalcStoreComb(store1 = "0", store2 = "2"),
                        CalcStoreComb(store1 = "1", store2 = "2"),
                       ]
                     )

print()
collect()


%%time

if CFG.ftre_plots_req :

    # Calculating sales per store across country and products 
    sales_store = []
    for store in np.sort(train.store.unique()) :
        sales = Sales_Prf[list(Cols[Cols.str[1] == str(store)])].sum(axis=1)
        sales_store.append(sales)
    
    
    with sns.axes_style("white"):
        fig, ax = plt.subplots(1,1, figsize = (25, 6))
        
        (pd.concat(sales_store, axis=1).
         apply(lambda x : x/ x.sum(), axis=1).
         plot(
             ax = ax, 
             linewidth = 1.50, 
             color = ["black", "red", "blue"]
         )
        )
        ax.set_title(f"Sales per store per day across products", **CFG.title_specs)
        ax.set_xticks(
            list(Date_Labels["date"].values), labels = list(Date_Labels["date_lbl"].values), 
            rotation = 90, 
            fontsize = 9
        ) 
        ax.set_yticks(np.arange(0, 0.50, 0.05), labels = np.round(np.arange(0, 0.50, 0.05), 2))
        ax.legend(bbox_to_anchor = (1.0, -0.25))
        
        plt.tight_layout()
        plt.show()


%%time 

if CFG.ftre_plots_req :

    # Calculating and displaying total daily sales rate:-
    df = \
    pp.train.groupby(['date','product'])[[CFG.target]].sum().reset_index().\
    join(pp.train.groupby('date')[[CFG.target]].sum(), 
         on = 'date',
         rsuffix = '_daily'
        );
    
    df['SalesRate'] = df[CFG.target]/df[f'{CFG.target}_daily']
    
    with sns.axes_style("white"):
        fig, ax = plt.subplots(1,1, figsize = (16, 6))
        for product in df['product'].unique():
            X = df[df['product'] == product]
            plt.plot(X['date'], X['SalesRate'], label = product)
    
        plt.title(
            f"\nProduct level daily sales rate across model period\n", **CFG.title_specs
        )
        plt.legend(bbox_to_anchor = (1.0, -0.10))
        plt.show()

collect()
print()


%%time 

if CFG.ftre_plots_req :

    sales_ctry = []
    
    for country in np.sort(train.country.unique()) :
        sel_cols = Cols[Cols.str[0] == str(country)]
        df = Sales_Prf[sel_cols].sum(axis=1)
        sales_ctry.append(df)
    
    with sns.axes_style("white"):
        fig, axes = \
        plt.subplots(1,2, 
                     figsize = (25, 9),
                     width_ratios = [0.8, 0.2],
                    )
    
        ax = axes[0]
        (pd.concat(sales_ctry, axis=1).
         apply(lambda x : x/ x.sum(), axis=1).
         plot(
             ax = ax, 
             linewidth = 1.50, 
             color = sns.color_palette("icefire", n_colors =  train.country.nunique() )
         )
        )
        ax.set_title(
            f"Sales per country per day across products and stores", 
            **CFG.title_specs
        )
        
        ax.set_xticks(
            list(Date_Labels["date"].values), labels = list(Date_Labels["date_lbl"].values), 
            rotation = 90, 
            fontsize = 9
        ) 
        ax.set_yticks(np.arange(0, 0.50, 0.02), labels = np.round(np.arange(0, 0.50, 0.02), 2))
        ax.legend(bbox_to_anchor = (1.0, -0.25))
    
        ax = axes[1]
        pp.train[["country", CFG.target]].groupby("country")[CFG.target].sum().plot.bar(color = "tab:blue", ax = ax)
        ax.set_title(
            f"Total sales by country", **CFG.title_specs
        )
        
        plt.tight_layout()
        plt.show()


%%time 

if CFG.ftre_plots_req :

    with sns.axes_style("white"):
        fig, axes = \
        plt.subplots(
            3,1, figsize=(15, 15), gridspec_kw = {"hspace": 0.2}, sharex = True
        )
        
        for i, dtpart in enumerate(['d', "W", "MS"]):
            
            _ = pp.train.groupby(
                [pd.Grouper(key="date", freq= dtpart)]
            )[CFG.target].sum().reset_index()
        
            ax = axes[i]
            sns.lineplot(
                data = _, x = "date", y= CFG.target, ax = ax, color = "tab:blue"
            )
            
            ax.set_xticks(
                Date_Labels.date.values, 
                labels = Date_Labels.date_lbl.values, 
                rotation = 45,
                fontsize = 7
            )
            ax.set(xlabel = '', ylabel = '')
            ax.set_title(f"Grouped sales by {dtpart.upper()}", **CFG.title_specs)
        
        plt.suptitle(
            f"Total sales by date-part across all stores during training period", 
            **CFG.title_specs, 
            y = 0.925
        )
        plt.tight_layout()
        plt.show()

collect()
print()


%%time 

class Xformer(BaseEstimator, TransformerMixin):
    "This class creates secondary features from the provided data"
    
    def __init__(self, GDP_Snp : pd.DataFrame, use_gdp: bool = True):
        self.GDP_Snp = GDP_Snp
        self.use_gdp = use_gdp

    def fit(self, X, y= None, **params):       
        return self

    def transform(self, df: pd.DataFrame, y = None, **params):
        "This method develops new features from the date column"

        X = df.copy()
        
        X["month"]       = X["date"].dt.month
        X["qtr"]         = X["date"].dt.quarter
        X["day"]         = X["date"].dt.day
        X["day_of_week"] = X["date"].dt.dayofweek
        X["day_of_year"] = X["date"].dt.dayofyear
        X["week_nb"]     = X["date"].dt.isocalendar().week
        X["year"]        = X["date"].dt.year

        X['group']       = (X['year'] - 2010 ) *48 + X['month'] * 4 + X['day'] // 7
        
        X["month_sin"]   = np.sin(X['month'] * (2 * np.pi / 12))
        X["month_cos"]   = np.cos(X['month'] * (2 * np.pi / 12))
        X["day_sin"]     = np.sin(X['day'] * (2 * np.pi / 365))
        X["day_cos"]     = np.cos(X['day'] * (2 * np.pi / 365))
        X["week_sin"]    = np.sin(X["week_nb"] * (2 * np.pi/ 53))
        X["week_cos"]    = np.cos(X["week_nb"] * (2 * np.pi/ 53))       
        
        for day in range(24, 32):
            X[f'dec{day}'] = (X.date.dt.day.eq(day) & X.date.dt.month.eq(12))

        X.columns = X.columns.str.replace(r"-","M", regex = True)
        X.columns = X.columns.str.replace(r"\s+","", regex = True)

        try:
            bool_cols = X.select_dtypes(include = "bool").columns
            X[bool_cols] = X[bool_cols].astype(np.uint8)
        except:
            pass

        if self.use_gdp :
            X = X.merge(self.GDP_Snp, how = "left", on = ["year", "country"])
        else :
            pass

        X[["month", "qtr", "day", "day_of_week", "day_of_year", "week_nb", "year"]] = \
        X[["month", "qtr", "day", "day_of_week", "day_of_year", "week_nb", "year"]].astype("string")
            
        return X



%%time 

class HolidayMapper(BaseEstimator, TransformerMixin):
    def __init__(self, years : list):
        self.years_list = years

    def fit(self, X, y= None, **params):
        self.holidays = {}
        
        for country in set(X.country):
            self.holidays[country] = CountryHoliday(country, years = self.years_list)

        self.holidays = pd.DataFrame.from_dict(self.holidays).reset_index()
        self.holidays = \
        self.holidays.melt(id_vars = "index").\
        rename(
            columns = {"index" : "date",
                       "variable": "country",
                       "value" : "holiday",
                      }
        )
        self.holidays["date"] = pd.to_datetime(self.holidays["date"])
        return self
    
    def transform(self, X, y = None):
        df = X.copy()
        df = df.merge(self.holidays, how = "left", on = ["date", "country"])
        df["holiday"] = df["holiday"].fillna("NotHoliday")
       
        try:
            bool_cols = df.select_dtypes(include = "bool").columns
            df[bool_cols] = df[bool_cols].astype(np.uint8)
        except:
            pass

        sel_cols = df.select_dtypes(include = ["object", "category"]).columns
        df[sel_cols] = df[sel_cols].astype("string") 
        df.columns = df.columns.str.lower()
        return df


%%time 

xform = make_pipeline(Xformer(gdp_snp, False), HolidayMapper(list(range(2010, 2020, 1))))

Xtrain = pp.train.dropna()
Xtrain.index = range(len(Xtrain))
ytrain = np.log1p(Xtrain[CFG.target])
Xtrain = Xtrain.drop(CFG.target, axis = 1, errors = "ignore")

Xtrain = xform.fit_transform(Xtrain, ytrain,)
Xtest  = xform.transform(pp.test)

Xtrain["Source"], Xtest["Source"] = ("Competition", "Competition")

cat_cols = list(Xtest.drop("Source", axis=1).select_dtypes("string").columns)
PrintColor(f"---> Shapes = {Xtrain.shape} {Xtest.shape}")


%%time 

folds = np.zeros(len(Xtrain))
cv = cv_selector[CFG.mdlcv_mthd]

if "G" in CFG.mdlcv_mthd :
    for fold_nb, (_, dev_idx) in enumerate(cv.split(Xtrain, ytrain,groups = Xtrain["year"])):
        folds[dev_idx] = fold_nb
else:
    for fold_nb, (_, dev_idx) in enumerate(cv.split(Xtrain, ytrain)):
        folds[dev_idx] = fold_nb    

ygrp = pd.Series(folds, name = "fold_nb", dtype = np.uint8)


%%time 

Mdl_Master = \
{     
 f'LGBM1R' : LGBMR(**{"objective"           : "regression_l2",
                      'device'              : "gpu" if CFG.gpu_switch == "ON" else "cpu",
                      'metric'              : "rmse",
                      'learning_rate'       : 0.05,
                      'n_estimators'        : 10_000,
                      'max_depth'           : 10,
                      'num_leaves'          : 32,
                      'colsample_bytree'    : 0.80,
                      'min_child_samples'   : 32,
                      'lambda_l1'           : 0.001,
                      'lambda_l2'           : 0.001,
                      'verbosity'           : -1,
                      'random_state'        : CFG.state,
                     }
                  ),

 f'LGBM2R' : LGBMR(**{"objective"           : "regression_l2",
                      'device'              : "gpu" if CFG.gpu_switch == "ON" else "cpu",
                      'metric'              : "rmse",
                      'learning_rate'       : 0.07,
                      'n_estimators'        : 10_000,
                      'max_depth'           : 8,
                      'num_leaves'          : 50,
                      'colsample_bytree'    : 0.65,
                      'min_child_samples'   : 32,
                      'lambda_l1'           : 0.01,
                      'lambda_l2'           : 0.01,
                      'verbosity'           : -1,
                      'random_state'        : CFG.state,
                     }
                  ),

 f'LGBM3R' : LGBMR(**{"objective"           : "regression_l2",
                      'device'              : "gpu" if CFG.gpu_switch == "ON" else "cpu",
                      'metric'              : "rmse",
                      'data_sample_strategy': 'goss',
                      'learning_rate'       : 0.05,
                      'n_estimators'        : 10_000,
                      'max_depth'           : 13,
                      'num_leaves'          : 85,
                      'colsample_bytree'    : 0.85,
                      'min_child_samples'   : 32,
                      'lambda_l1'           : 0.01,
                      'lambda_l2'           : 0.01,
                      'verbosity'           : -1,
                      'random_state'        : CFG.state,
                     }
                  ),

}

# Initializing model outputs
OOF_Preds    = {}
Mdl_Preds    = {}
FittedModels = {}
FtreImp      = {}
SelMdlCols   = {}


%%time

# Model training:-
drop_cols = ["Source", "id", "Id", "Label", CFG.target, "fold_nb",  "date"]

for method, mymodel in tqdm(Mdl_Master.items()):

    PrintColor(f"\n{'=' * 20} {method.upper()} MODEL TRAINING {'=' * 20}\n")

    md = \
    ModelTrainer(
        problem_type   = "regression",
        es             = CFG.nbrnd_erly_stp,
        target         = CFG.target,
        orig_req       = False,
        orig_all_folds = False,
        metric_lbl     = "mape",
        drop_cols      = drop_cols,
        pp_preds       = False,
        )

    sel_mdl_cols = list(Xtest.columns) 
    PrintColor(
        f"Selected columns = {len(sel_mdl_cols) :,.0f}", 
        color = Fore.RED
    )
    SelMdlCols[method] = (sel_mdl_cols, cat_cols)

    Xtrain_ = Xtrain.copy()
    Xtest_  = Xtest.copy()

    if "CB" in method :
        pass
    else:
        Xtrain_[cat_cols] = Xtrain_[cat_cols].astype("category")
        Xtest_[cat_cols]  = Xtest_[cat_cols].astype("category")

    fitted_models, oof_preds, test_preds, ftreimp, mdl_best_iter =  \
    md.MakeOfflineModel(
        Xtrain_,
        ytrain,
        ygrp,
        Xtest_,
        clone(mymodel),
        method,
        test_preds_req   = True,
        ftreimp_plot_req = CFG.ftre_imp_req,
        ntop = 50,
    )

    OOF_Preds[method]    = oof_preds
    Mdl_Preds[method]    = test_preds
    FittedModels[method] = fitted_models
    FtreImp[method]      = ftreimp

    del fitted_models, oof_preds, test_preds, ftreimp, sel_mdl_cols, Xtrain_, Xtest_
    print()
    collect();

_ = utils.CleanMemory();


%%time

for method, oof_preds in OOF_Preds.items() :
    score = \
    utils.ScoreMetric(
        np.expm1(ytrain),
        np.round(np.clip(1.01 * np.expm1(oof_preds), 5, 5939),0)
    )
    PrintColor(
        f"---> OOF score = {score :,.8f} | {method}",
        color = Fore.CYAN,
    )

oof_preds = \
np.round(
    np.clip(
        1.01 * np.expm1(pd.DataFrame(OOF_Preds).mean(axis=1)), 
        5, 5939
    ),
    0
)
score = utils.ScoreMetric(np.expm1(ytrain), oof_preds)
PrintColor(
    f"\n---> Ensemble OOF score = {score :,.8f}"
)

test_preds = \
np.round(
    np.clip(
        1.01 * np.expm1(pd.DataFrame(Mdl_Preds).mean(axis=1)), 
        5, 5939
    ),
    0
)

try:
    test_preds = test_preds.values.flatten()
except:
    pass


%%time 

try:
    df = \
    pd.concat(
        [pp.train[["date", "country", "store", "product", CFG.target]].dropna(subset = [CFG.target], axis=0),
         pp.test[["date", "country", "store", "product"]].assign(**{CFG.target : test_preds})
        ], axis=0, ignore_index = True
    )
    
    all_combs = list(product(df.country.unique(), df["product"].unique(), df.store.unique()))
    
    with sns.axes_style("white"):
        fig, axes = plt.subplots(30, 3, 
                                 figsize = (36, 210), 
                                 gridspec_kw = {"hspace" : 0.25 , "wspace" : 0.25}
                                )
    
        for i, (c, p, s) in tqdm(enumerate(all_combs)) :
            df_1 = df.loc[(df.country == c) & (df["product"] == p) & (df.store == s)]
    
            ax = axes[i//3, i % 3]
            sns.lineplot(
                data = df_1, 
                x = df_1["date"], 
                y = df_1[CFG.target], 
                color = "#56ccf8", 
                ax = ax
            )
            ax.axvline(x = pd.to_datetime("2016-12-31"), linewidth = 1.5, color = "maroon")
            
            ax.set_title(
                f"{c} - {s} - {p}", fontsize = 9, fontweight = "bold"
            )
            ax.set(xlabel = "", ylabel = "")
            del df_1
    
        plt.show()

except:
    pass


%%time 

# Plotting by country total sales
df_1 = df.groupby(["date", "country"])[CFG.target].sum().reset_index()
df_2 = df.groupby(["date",])[CFG.target].sum().reset_index()
df_1 = df_1.merge(df_2, how = "left", on = ["date"])
df_1["prop_sales"] = df_1[f"{CFG.target}_x"] / df_1[f"{CFG.target}_y"]

with sns.axes_style("white") :
    fig, ax = plt.subplots(1,1, figsize = (20, 9))
    colors  = sns.color_palette("icefire", n_colors = 6)

    for j, country in enumerate(df_1.country.unique()) :
        df_2 = df_1.loc[df_1.country == country, ["date", "prop_sales"]]
        
        sns.lineplot(
            data = df_2, 
            x = "date",
            y = "prop_sales", 
            color = colors[j], 
            ax = ax,
        )
  
    plt.suptitle(
        f"Forecast sales by country - total", 
        **CFG.title_specs
    )
    plt.show()


%%time

pp.sub_fl[CFG.target] = test_preds
pp.sub_fl.to_csv("submission.csv", index = True)

!ls
print()
!head submission.csv

