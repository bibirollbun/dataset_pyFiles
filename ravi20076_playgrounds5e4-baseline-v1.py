


!uv pip install -q --system -r /kaggle/input/playgrounds5e4-public-imports-v1/req_kaggle.txt

exec( open(f"/kaggle/input/playgrounds5e4-public-imports-v1/myimports.py", "r").read() )
exec( open(f"/kaggle/input/playgrounds5e4-public-imports-v1/myutils.py", "r").read() )
exec( open(f"/kaggle/input/playgrounds5e4-public-imports-v1/training.py", "r").read() )
exec( open(f"/kaggle/input/playgrounds5e4-public-imports-v1/mypp.py", "r").read() )

print()


%%time

class CFG:
    """
    Configuration class for parameters and CV strategy for tuning and training
    Some parameters may be unused here as this is a general configuration class
    """;

    # Data preparation:-
    version_nb         = 1
    model_id           = "V1_3"
    model_label        = "ML"
    test_req           = False
    test_iter          = 50
    gpu_switch         = "ON" if torch.cuda.is_available() else "OFF"
    state              = 42
    target             = f"Listening_Time_minutes"
    grouper            = f""
    tgt_mapper         = {}
    ip_path            = f"/kaggle/input/playground-series-s5e4"
    op_path            = f"/kaggle/working"
    orig_path          = f"/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv"
    data_path          = f""
    dtl_preproc_req    = True
    ftre_plots_req     = True
    ftre_imp_req       = True
    nb_orig            = 1
    orig_all_folds     = True

    # Model Training:-
    pstprcs_oof        = False
    pstprcs_train      = False
    pstprcs_test       = False
    ML                 = True
    test_preds_req     = True
    n_splits           = 5
    n_repeats          = 1
    nbrnd_erly_stp     = 0
    mdlcv_mthd         = 'KF'
    metric_obj         = 'minimize'

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

cv_selector = \
{
 "RKF"   : RKF(n_splits   = CFG.n_splits, n_repeats= CFG.n_repeats, random_state= CFG.state),
 "RSKF"  : RSKF(n_splits  = CFG.n_splits, n_repeats= CFG.n_repeats, random_state= CFG.state),
 "SKF"   : SKF(n_splits   = CFG.n_splits, shuffle = True, random_state= CFG.state),
 "KF"    : KFold(n_splits = CFG.n_splits, shuffle = True, random_state= CFG.state),
 "GKF"   : GKF(n_splits   = CFG.n_splits)
}

collect()


%%time 

pp = Preprocessor()
pp.DoPreprocessing();


%%time 

if CFG.dtl_preproc_req :
    advcv = AdversarialCVMaker()
    advcv.make_cv(pp.train[pp.test.columns], pp.test)


%%time 

if CFG.dtl_preproc_req :

    train = pp.train.copy()
    test  = pp.test.copy()
    
    train["Episode_Title"] = train["Episode_Title"].str.split(" ", expand = True)[1].astype(np.int16)
    test["Episode_Title"]  = test["Episode_Title"].str.split(" ", expand = True)[1].astype(np.int16)


%%time 

if CFG.dtl_preproc_req :

    with sns.axes_style("white") :
        fig, axes = \
        plt.subplots( 
            16, 3, 
            figsize = (30, 80), 
            gridspec_kw = {"hspace": 0.30, "wspace" : 0.35}
        )
    
        df = train.groupby(["Podcast_Name", "Episode_Title"], as_index = False)[CFG.target].mean()
        
        for i, lbl in enumerate( np.unique(df["Podcast_Name"] ) ) :
            ax   = axes[i //  3, i % 3 ]
            df_1 = df.loc[df.Podcast_Name == lbl]
            sns.lineplot(
                data = df_1, 
                x = "Episode_Title", 
                y = CFG.target, 
                c = "tab:blue", 
                ax = ax,
            )
            ax.set_title(f"{lbl}", **CFG.title_specs)
            ax.set(xlabel = "", ylabel = "")
    
        plt.tight_layout()
        plt.suptitle(
            f"Average viewing time per episode per podcast", 
            y        = 0.89,
            color    = "black",
            fontsize = 14,
            fontweight = "bold",
        )
        plt.show()
        
        


%%time 

if CFG.dtl_preproc_req :

    with sns.axes_style("white") :
        fig, axes = \
        plt.subplots( 
            5, 2, 
            figsize = (30, 30), 
            gridspec_kw = {"hspace": 0.30, "wspace" : 0.35}
        )
    
        df = train.groupby(["Genre", "Episode_Title"], as_index = False)[CFG.target].mean()
        
        for i, lbl in enumerate( np.unique(df["Genre"] ) ) :
            ax   = axes[i //  2, i % 2 ]
            df_1 = df.loc[df["Genre"] == lbl]
            sns.lineplot(
                data = df_1, 
                x = "Episode_Title", 
                y = CFG.target, 
                c = "tab:blue", 
                ax = ax,
            )
            ax.set_title(f"{lbl}", **CFG.title_specs)
            ax.set(xlabel = "", ylabel = "")
    
        plt.tight_layout()
        plt.suptitle(
            f"Average viewing time per episode per genre", 
            y        = 0.91,
            color    = "black",
            fontsize = 14,
            fontweight = "bold",
        )
        plt.show()
        
        


%%time 

if CFG.dtl_preproc_req :
    with sns.axes_style("white", ) :
    
        cat_cols = \
        ['Podcast_Name',  'Genre',
         'Publication_Day', 'Publication_Time',
         'Number_of_Ads', 'Episode_Sentiment',
        ]
        
        fig, axes = \
        plt.subplots( 
            len(cat_cols ), 1, 
            figsize = (12, len(cat_cols) * 6), 
            gridspec_kw = {"hspace": 0.90, "wspace" : 0.35}
        )
        
        for i, col in enumerate( cat_cols ):
            ax = axes[i]
            df = train[[col, CFG.target]]
    
            if col == "Number_of_Ads" :
                df[col] = df[col].fillna(-1).round(0).astype(np.int16)
            else:
                df[col] = df[col].fillna("missing")
                
            df.groupby(col)[CFG.target].mean().sort_values(ascending = False).plot.bar(ax = ax)
            ax.set_title(f"Average episode viewing by {col}", **CFG.title_specs)
            ax.set(xlabel = "", ylabel = "")
    
        plt.tight_layout()
        plt.suptitle(
            f"Average viewing time per episode per category feature", 
            y        = 0.91,
            color    = "black",
            fontsize = 14,
            fontweight = "bold",
        )
        plt.show()


%%time 

if CFG.dtl_preproc_req :

    num_cols = \
    ['Episode_Length_minutes','Host_Popularity_percentage','Guest_Popularity_percentage', CFG.target]
    
    display(
        train.groupby("Genre")[num_cols].
        agg(["min", "mean", "max"]).
        style.
        format(formatter = '{:,.2f}').
        set_caption("Numeric column distribution by genre")
    )
    
    print("\n\n\n")
    display(
        train.groupby("Podcast_Name")[num_cols].
        agg(["min", "mean", "max"]).
        style.
        format(formatter = '{:,.2f}').
        set_caption("Numeric column distribution by podcast name")
    )



%%time 

def make_ftre( X: pd.DataFrame) :
    "This function makes secondary features from the provided data"

    df = X.copy()

    df["Pub_DateTime"]  = df['Publication_Day'].astype("string") + "-" + df['Publication_Time'].astype("string")
    df["Number_of_Ads"] = df["Number_of_Ads"].fillna(0).clip(0,3).astype(np.uint8)
    df["GuestPop_Int"]  = df["Guest_Popularity_percentage"].fillna(-1).astype(np.int16)
    df["GuestPop_Dec"]  = (df["Guest_Popularity_percentage"] - df["GuestPop_Int"]).fillna(-1)
    df["Total_Pop"]     = df["Guest_Popularity_percentage"] + df["Host_Popularity_percentage"]
    df["Diff_Pop"]      = df["Guest_Popularity_percentage"] - df["Host_Popularity_percentage"]
    df["TotalPop_vs_Ads"] = np.log1p(df["Total_Pop"] ) - np.log1p(df["Number_of_Ads"])

    return df
    


%%time 

Xtrain = make_ftre( pp.train.drop(CFG.target, axis=1) )
Xtest  = make_ftre( pp.test )
ytrain = pp.train[CFG.target]

cat_cols = Xtrain.drop("Source", axis=1).nunique()
cat_cols = \
list(
    set( 
        Xtrain.drop("Source", axis=1).
        select_dtypes(["string", "object", "category"]).
        columns 
    ).union(
        set( cat_cols.loc[cat_cols <= 500].index )
    )
)

Xtrain[cat_cols] = Xtrain[cat_cols].astype("string").fillna("missing")
Xtest[cat_cols]  = Xtest[cat_cols].astype("string").fillna("missing")

PrintColor(f"\n---> Category columns\n")
with np.printoptions(linewidth = 100):
    pprint(np.array(cat_cols))

# Initializing the cv scheme:-
cv = cv_selector[CFG.mdlcv_mthd]

if CFG.nb_orig > 0:
    all_df = []
    
    for mysource in ["Competition", "Original"]:
        df = pd.concat([Xtrain.loc[Xtrain.Source == mysource], ytrain], axis=1, join = "inner")
        df.index = range(len(df))
        for fold_nb, (_, dev_idx) in enumerate(cv.split(df, df[CFG.target])):
            df.loc[dev_idx, "fold_nb"] = fold_nb
            
        all_df.append(df)      
    ygrp = pd.concat(all_df, axis=0, ignore_index = True)["fold_nb"].astype(np.uint8)
                      
else:
    df = Xtrain.loc[Xtrain.Source == "Competition"]
    df.index = range(len(df))
    
    for fold_nb, (_, dev_idx) in enumerate(cv.split(df, ytrain.iloc[df.index])):
        df.loc[dev_idx, "fold_nb"] = fold_nb 
    ygrp = df["fold_nb"].astype(np.uint8)

PrintColor(
    f"\n---> Shapes = {Xtrain.shape} {Xtest.shape} {ytrain.shape} {ygrp.shape}"
)


%%time 

Mdl_Master = \
{   
 f'CB1R'    : CBR(**{"loss_function"         : "RMSE",
                     "eval_metric"           : "RMSE",
                     'task_type'             : "GPU" if CFG.gpu_switch == "ON" else "CPU",
                     'learning_rate'         : 0.0225,
                     'iterations'            : 4_500 if CFG.test_req == False else 50,
                     'max_depth'             : 6,
                     'min_data_in_leaf'      : 39 ,
                     'colsample_bylevel'     : 0.55 if CFG.gpu_switch == "OFF" else None,
                     'l2_leaf_reg'           : 2.50,
                     'random_strength'       : 0.025,
                     'leaf_estimation_method': "Newton",
                     'od_wait'               : 25, 
                     'verbose'               : 0,
                     'random_state'          : CFG.state,
                    }
                 ),
    
 f'XGB1R'  : XGBR(**{  "objective"            : "reg:squarederror",
                       "eval_metric"          : "rmse",
                       'device'               : "cuda" if CFG.gpu_switch == "ON" else "cpu",
                       'learning_rate'        : 0.02,
                       'n_estimators'         : 4_500 if CFG.test_req == False else 50,
                       'max_depth'            : 5,
                       'colsample_bytree'     : 0.60,
                       'colsample_bynode'     : 0.65,
                       'subsample'            : 0.65,
                       'reg_lambda'           : 0.001,
                       'reg_alpha'            : 0.001,
                       'verbosity'            : 0,
                       'random_state'         : CFG.state,
                       'early_stopping_rounds': None if CFG.nbrnd_erly_stp == 0 else CFG.nbrnd_erly_stp,
                      } 
                   ),
}

# Initializing model outputs
OOF_Preds    = {}
Mdl_Preds    = {}
FtreImp      = {}


%%time

drop_cols = ["Source", "id", "Id", "Label", CFG.target, "fold_nb"]

for method, mymodel in tqdm(Mdl_Master.items()):

    PrintColor(
        f"\n{'=' * 20} {method.upper()} MODEL TRAINING {'=' * 20}\n"
    )

    md = \
    ModelTrainer(
        problem_type   = "regression",
        es             = CFG.nbrnd_erly_stp,
        target         = CFG.target,
        orig_req       = True if CFG.nb_orig > 0 else False,
        orig_all_folds = CFG.orig_all_folds,
        metric_lbl     = "rmse",
        drop_cols      = drop_cols,
        pp_preds       = CFG.pstprcs_oof,
    )

    sel_mdl_cols = list(Xtest.columns) 
    PrintColor(
        f"Selected columns = {len(sel_mdl_cols) :,.0f}", 
        color = Fore.RED
    )

    ct = \
    ColumnTransformer(
        [("TE", TargetEncoder(random_state = CFG.state) , cat_cols)],
        remainder = "passthrough",
        verbose_feature_names_out = False,
    )
    mypipe = Pipeline([("PP", ct ), ("M", mymodel)])
    
    fitted_models, oof_preds, test_preds, ftreimp, mdl_best_iter =  \
    md.MakeOfflineModel(
        Xtrain,
        ytrain,
        ygrp,
        Xtest,
        mypipe,
        method,
        test_preds_req   = True,
        ftreimp_plot_req = CFG.ftre_plots_req,
        ntop = 50,
    )

    OOF_Preds[method]    = oof_preds
    Mdl_Preds[method]    = test_preds
    FtreImp[method]      = ftreimp

    del fitted_models, oof_preds, test_preds, ftreimp, sel_mdl_cols
    print()
    collect()


_ = utils.CleanMemory()


%%time

len_train = Xtrain.loc[Xtrain.Source == "Competition"].shape[0]
method    = "L21R"
model     = Ridge(max_iter = 10000, random_state = CFG.state)


md = \
ModelTrainer(
    problem_type   = "regression",
    es             = CFG.nbrnd_erly_stp,
    target         = CFG.target,
    orig_req       = False,
    orig_all_folds = CFG.orig_all_folds,
    metric_lbl     = "rmse",
    drop_cols      = drop_cols,
    pp_preds       = CFG.pstprcs_oof,
)

_, oof_ens_preds, test_preds, _, _ =  \
md.MakeOfflineModel(
    pd.DataFrame(OOF_Preds).iloc[0 : len_train].assign(Source = "Competition"),
    ytrain.iloc[0 : len_train],
    ygrp.iloc[0 : len_train],
    pd.DataFrame(Mdl_Preds).assign(Source = "Competition"),
    model,
    method,
    test_preds_req   = True,
    ftreimp_plot_req = False,
    ntop = 50,
)

score = utils.ScoreMetric(ytrain.iloc[0 : len_train],oof_ens_preds)
PrintColor(f"\n\n---> Overall score = {score:,.8f}")


%%time

try:
    pd.DataFrame(OOF_Preds).iloc[0: len_train].assign(**{"Ensemble": oof_ens_preds}).\
    to_parquet(
        os.path.join(CFG.op_path, f"OOF_Preds_{CFG.model_label}{CFG.model_id}.parquet")
    )

    pd.DataFrame(Mdl_Preds).assign(**{"Ensemble": test_preds}).\
    to_parquet(
        os.path.join(CFG.op_path, f"Mdl_Preds_{CFG.model_label}{CFG.model_id}.parquet")
    )

    pp.sub_fl[CFG.target] = test_preds
    
except:
    pd.DataFrame(OOF_Preds).\
    to_parquet(
        os.path.join(CFG.op_path, f"OOF_Preds_{CFG.model_label}{CFG.model_id}.parquet")
    )  
    
    pd.DataFrame(Mdl_Preds).\
    to_parquet(
        os.path.join(CFG.op_path, f"Mdl_Preds_{CFG.model_label}{CFG.model_id}.parquet")
    )

    pp.sub_fl[CFG.target] = Mdl_Preds[method].flatten()

# Public submission blend
try:
    sub = \
    pd.read_csv(f"/kaggle/input/playgrounds5e4publicsubsv1/submission.csv")[CFG.target].values.flatten()
    
    pp.sub_fl[CFG.target] = (pp.sub_fl[CFG.target].values * 0.4) + (sub * 0.6)
except:
    PrintColor(
        f"---> Could not blend public submission due to code error - check!",
        color = Fore.RED,
    )
    
pp.sub_fl.to_csv(
    os.path.join(CFG.op_path, f"submission.csv"), index = None
)

print()
!ls
print()
!head submission.csv

_ = utils.CleanMemory()
print()




