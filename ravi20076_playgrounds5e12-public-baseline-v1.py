


!uv pip install \
    -q \
    --system \
    -r /kaggle/input/playgrounds5e12-public-imports-v1/req_kaggle.txt

season  = 5
episode = 12

exec( open(f"/kaggle/input/playgrounds{season}e{episode}-public-imports-v1/myimports.py", "r").read() )
exec( open(f"/kaggle/input/playgrounds{season}e{episode}-public-imports-v1/myutils.py", "r").read() )
exec( open(f"/kaggle/input/playgrounds{season}e{episode}-public-imports-v1/training.py", "r").read() )
exec( open(f"/kaggle/input/playgrounds{season}e{episode}-public-imports-v1/mypp.py", "r").read() )

print()


%%time

class CFG:
    """
    Configuration class for parameters and CV strategy for tuning and training
    Some parameters may be unused here as this is a general configuration class
    """

    # Data preparation:-
    version_nb         = 1
    model_id           = "V1_2"
    model_label        = "ML"
    test_req           = False
    test_iter          = 20
    gpu_switch         = "ON" if torch.cuda.is_available() else "OFF"
    state              = 42
    target             = f"diagnosed_diabetes"
    grouper            = f""
    
    tgt_mapper         = {}
    
    ip_path            = f"/kaggle/input/playground-series-s5e12"
    op_path            = f"/kaggle/working"
    orig_path          = f"/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv"
    data_path          = f""
    dtl_preproc_req    = True
    ftre_plots_req     = False
    ftre_imp_req       = False
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
    mdlcv_mthd         = 'SKF'
    metric_obj         = 'maximize'

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
 "RKF"   : RepeatedKFold(n_splits = CFG.n_splits, n_repeats= CFG.n_repeats, random_state= CFG.state),
 "RSKF"  : RepeatedStratifiedKFold(n_splits  = CFG.n_splits, n_repeats= CFG.n_repeats, random_state= CFG.state),
 "SKF"   : StratifiedKFold(n_splits = CFG.n_splits, shuffle = True, random_state= CFG.state),
 "KF"    : KFold(n_splits = CFG.n_splits, shuffle = True, random_state= CFG.state),
 "GKF"   : GroupKFold(n_splits = CFG.n_splits)
}

if CFG.test_req :
    PrintColor(
        f"\n---> THIS IS A TEST RUN\n",
        color = Fore.RED,
    )

collect()


%%time 

pp = Preprocessor()
pp.DoPreprocessing();


def make_ftre(
    train, 
    test, 
    target : str, 
    orig_as_cols: bool,
    nb_orig : int,
    orig_all_folds :  bool, 
):
    "Combines the train-test data and builds state-less features for the model"

    strt_cols = list(test.columns)
    num_cols  = test.select_dtypes(include = np.number).columns.tolist()
    cat_cols  = test.select_dtypes(exclude = np.number).columns.tolist()
    te_cols   = []

    if orig_as_cols:
        orig  = train.loc[train.Source == "Original"]
        train = train.loc[train.Source != "Original"]
        
        print(f"\n---> Using original data as columns")
        for col in strt_cols :
            df_ = orig.groupby(col).agg({CFG.target : ["mean", "count",]})
            df_.columns = [
                f"O{c.replace('%', 'pctl')}_{col}" 
                for c in ["mean","count"]
            ]
            
            train = train.merge(
                df_,
                left_on = col,
                right_index = True,
                how = "left"
            )
        
            test = test.merge(
                df_,
                left_on = col,
                right_index = True,
                how = "left"
            )
            del df_

        CFG.nb_orig = 0

    print(f"---> Shapes = {train.shape} {test.shape}")

    ytrain = train[target].astype(np.uint8)
    train  = train.drop(target, axis=1, errors = "ignore")

    print(f"\n---> Appending train-test data")
    df = pd.concat([train, test[train.columns]])
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean()).values
    
    for col in cat_cols :
        df[f"{col}"] = df[f"{col}"].astype("string").fillna("missing")
    print(f"---> Shape = {df.shape}")

    print(f"\n---> Making string-twins from numeric columns")
    for col in num_cols :
    
        df[f"C{col}"] = df[col].astype("string").fillna("missing")
        enc           = LabelEncoder()
        df[f"C{col}"] = enc.fit_transform(df[f"C{col}"].values)
        df[f"C{col}"] = df[f"C{col}"].astype("string")
        cat_cols.append(f"C{col}")

    print(f"---> Shape = {df.shape}")
    
    print(f"\n---> Making combination features")
    combs  = combinations(num_cols, 2)

    for col1, col2 in combs :
        df[f"{col1}_p_{col2}"] = df[col1] + df[col2]
    
    print(f"---> Shape = {df.shape}")
      
    cat_cols = df.select_dtypes(exclude = np.number).columns.tolist()

    print(f"\n---> Making leaky count encoder")
    for col in cat_cols:
        df_         = df[col].value_counts().to_frame()
        df_.columns = [f"CE_{col}"]
        df          = df.merge(df_, how = "left", left_on = col, right_index = True)

    print(f"---> Shape = {df.shape}")  

    cat_cols = (
        df.drop("Source", axis=1, errors = "ignore").
        select_dtypes(exclude = np.number).
        columns.
        tolist()
    )

    Xtrain = df.iloc[0 : len(train)]
    Xtest  = df.iloc[-1*len(test):]
    print(f"---> Shape = {df.shape} {Xtrain.shape} {ytrain.shape} {Xtest.shape}")

    try:
        del Xtrain[0], Xtest[0]
    except:
        pass

    if nb_orig > 0 and orig_all_folds == True :
        extra = [
            Xtrain.loc[Xtrain.Source == "Original"],
            ytrain.iloc[0 : len(Xtrain.loc[Xtrain.Source == "Original"])]
        ]
        ytrain = ytrain.iloc[0 : len(Xtrain.loc[Xtrain.Source == "Competition"])]
        Xtrain = Xtrain.loc[Xtrain.Source == "Competition"]
        print(f"---> Shape = {Xtrain.shape} {ytrain.shape} {extra[0].shape} {extra[1].shape}")

    else:
        extra = None
    
    return (Xtrain, ytrain, Xtest, extra, cat_cols)
    


%%time
    
(Xtrain, ytrain, Xtest, extra, cat_cols) = make_ftre(
    pp.train.copy(), 
    pp.test.copy(),
    CFG.target,
    orig_as_cols = False,
    nb_orig = CFG.nb_orig,
    orig_all_folds = CFG.orig_all_folds
)

print(f"\n---> Shape = {Xtrain.shape} {Xtest.shape} {ytrain.shape}\n")
_ = utils.CleanMemory()


%%time 

# Initializing the cv scheme:-
cv = cv_selector[CFG.mdlcv_mthd]
ygrp = np.zeros(len(Xtrain))

for fold_nb, (_, dev_idx) in enumerate(cv.split(Xtrain, ytrain)):
    ygrp[dev_idx] = fold_nb

ygrp = pd.Series(ygrp, name = "fold_nb", dtype = np.uint8)


PrintColor(
    f"\n---> Shapes = {Xtrain.shape} {Xtest.shape} {ytrain.shape} {ygrp.shape}"
)


%%time 

ct = ColumnTransformer(
    [("TE", TargetEncoder(random_state = CFG.state), cat_cols)],
    verbose_feature_names_out = False,
    remainder = "passthrough",
)

Mdl_Master = \
{     
 f'XGB1C'  : [
                 Pipeline(
                     steps = [
                                 ("PP", ct),
                                 ("M", XGBC(**{"objective"            : "binary:logistic",
                                               "eval_metric"          : "auc",
                                               'device'               : "cuda:0" if CFG.gpu_switch == "ON" else "cpu",
                                               'learning_rate'        : 0.008,
                                               'n_estimators'         : 2500 if CFG.test_req == False else CFG.test_iter,
                                               'max_depth'            : 5,
                                               'subsample'            : 0.40,
                                               'colsample_bytree'     : 0.30,
                                               'reg_lambda'           : 3.50,
                                               'reg_alpha'            : 0.10,
                                               'verbosity'            : 0,
                                               'random_state'         : CFG.state,
                                               'enable_categorical'   : True,
                                              } 
                                           )
                                 )
                             ]
              ),
              {"M__verbose" : 0},
              {"dev" : {}, "test" : {}},
             ]
}

# Initializing model outputs
OOF_Preds    = {}
Mdl_Preds    = {}



%%time 

drop_cols = ["Source", "id", "Id", "Label", CFG.target, "fold_nb"]

for method, (mymodel, fit_params, predict_params) in tqdm(Mdl_Master.items()) :
    md = ModelTrainer(
        drop_cols    = drop_cols, 
        problem_type = "binary",
        len_train    = Xtrain.loc[Xtrain.Source == "Competition"].shape[0],
    )

    _, oof_preds, mdl_preds = \
    md.fit_predict(
        Xtrain.copy(), 
        ytrain,
        Xtest.copy(), 
        ygrp,
        extra    = extra,
        method   = method,
        mymodel  = mymodel,
        cat_cols = cat_cols,
        predict_params = predict_params,
        **fit_params,
    )   

    OOF_Preds[method] = oof_preds.flatten()
    Mdl_Preds[method] = mdl_preds.flatten()
    collect();

    score = utils.ScoreMetric(
        ytrain.values[0 : len(oof_preds)], 
        utils.pp_preds(oof_preds),
    )
    PrintColor(
        f"---> Overall score = {score :,.8f} | Competition metric",
        color = Fore.CYAN
    )

_ = utils.CleanMemory()
print()


pp.sub_fl[CFG.target] = mdl_preds
pp.sub_fl.to_csv(f"submission.csv", index = False)

with sns.axes_style("white") :
    fig, axes = plt.subplots(
        2,1, 
        figsize = (10, 9), 
        sharex = True,
        gridspec_kw = {"hspace" : 0.35, "wspace" : 0.35}
    )

    ax = axes[0]
    oof_preds = pd.DataFrame(OOF_Preds)
    oof_preds.plot.hist(bins = 100, ax = ax)
    ax.set_title("Prediction histogram - OOF", **CFG.title_specs)
    ax.set(xlabel = "", ylabel = "")
    ax.set_xticks(
        np.arange(0, 1.01, 0.05), 
        labels = np.round( np.arange(0, 1.01, 0.05) , 2), 
        rotation = 90,
        fontsize = 7.5,
    )

    ax = axes[1]
    oof_preds = pd.DataFrame(Mdl_Preds)
    oof_preds.plot.hist(bins = 100, ax = ax)
    ax.set_title("Prediction histogram - Test", **CFG.title_specs)
    ax.set(xlabel = "", ylabel = "")
    ax.set_xticks(
        np.arange(0, 1.01, 0.05), 
        labels = np.round( np.arange(0, 1.01, 0.05) , 2), 
        rotation = 90,
        fontsize = 7.5,
    )

    plt.tight_layout()
    plt.show()

print()
!ls
print()
!head submission.csv

