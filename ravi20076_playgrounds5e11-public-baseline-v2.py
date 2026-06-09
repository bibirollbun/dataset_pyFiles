


!uv pip install \
    -q \
    --system \
    -r /kaggle/input/playgrounds5e11-public-imports-v1/req_kaggle.txt

season  = 5
episode = 11

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
    model_id           = "V1_1"
    model_label        = "ML"
    test_req           = False
    test_iter          = 20
    gpu_switch         = "ON" if torch.cuda.is_available() else "OFF"
    state              = 42
    target             = f"loan_paid_back"
    grouper            = f""
    
    tgt_mapper         = {}
    
    ip_path            = f"/kaggle/input/playground-series-s5e11"
    op_path            = f"/kaggle/working"
    orig_path          = f"/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv"
    data_path          = f""
    dtl_preproc_req    = True
    ftre_plots_req     = False
    ftre_imp_req       = False
    nb_orig            = 1
    orig_all_folds     = False

    # Model Training:-
    pstprcs_oof        = False
    pstprcs_train      = False
    pstprcs_test       = False
    ML                 = True
    test_preds_req     = True
    n_splits           = 5
    n_repeats          = 1
    nbrnd_erly_stp     = 200
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


%%time 

def make_origcol_ftre(Xtrain, Xtest):
    "Makes target encoded features with original data as columns across train and test sets"
    
    Xtrain = Xtrain.loc[Xtrain.Source == "Competition"]
    orig   = Xtrain.loc[Xtrain.Source == "Original"]
    
    sel_cols = pp.strt_ftre[0: -1]
    for col in sel_cols :
        df_ = orig.groupby(col).agg({CFG.target : ["mean", "count",]})
        df_.columns = [
            f"O{c.replace('%', 'pctl')}_{col}" 
            for c in ["mean","count"]
        ]
        
        Xtrain = Xtrain.merge(
            df_,
            left_on = col,
            right_index = True,
            how = "left"
        )
    
        Xtest = Xtest.merge(
            df_,
            left_on = col,
            right_index = True,
            how = "left"
        )
    
        del df_
    
    print(f"\n---> Shape = {Xtrain.shape} {Xtest.shape}")
    std_      = Xtrain.select_dtypes(include = np.number).std()
    drop_cols = std_.loc[std_ <= 1e-5].index
    
    Xtrain = Xtrain.drop(drop_cols, axis=1)
    Xtest  = Xtest.drop(drop_cols, axis=1)
    print(f"---> Shape = {Xtrain.shape} {Xtest.shape}")

    return Xtrain, Xtest


%%time
    
ytrain        = pp.train[CFG.target]
Xtrain, Xtest = pp.train.copy(), pp.test.copy()
Xtrain, Xtest = make_origcol_ftre(Xtrain, Xtest)

cat_cols  = (
    Xtest.
    drop("Source", axis=1).
    select_dtypes(exclude = np.number).
    columns.
    tolist()
)

print(f"\n---> Shape = {Xtrain.shape} {Xtest.shape}\n")

CFG.nb_orig = 0
_ = utils.CleanMemory()


%%time 

# Initializing the cv scheme:-
CFG.nb_orig = 0
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
 f'XGB1C'  : [
              XGBC(**{ "objective"            : "binary:logistic",
                       "eval_metric"          : "auc",
                       'device'               : "cuda:0" if CFG.gpu_switch == "ON" else "cpu",
                       'learning_rate'        : 0.01,
                       'n_estimators'         : 10000 if CFG.test_req == False else CFG.test_iter,
                       'max_depth'            : 8,
                       'subsample'            : 0.90,
                       'colsample_bytree'     : 0.75,
                       'reg_lambda'           : 0.75,
                       'reg_alpha'            : 0.001,
                       'verbosity'            : 0,
                       'random_state'         : CFG.state,
                       'enable_categorical'   : True,
                       'early_stopping_rounds' : CFG.nbrnd_erly_stp,
                      } 
                   ),
              {"verbose" : 0}
             ],

f"CB1C" : [
             CBC(
                     loss_function       = "Logloss",
                     max_depth           = 6,
                     iterations          = 10000 if CFG.test_req == False else CFG.test_iter,
                     task_type           = "GPU" if CFG.gpu_switch == "ON" else "CPU", 
                     learning_rate       = 0.02,
                     l2_leaf_reg         = 1.25, 
                     random_state        = CFG.state,
                     verbose             = 0,
                     colsample_bylevel   = None if CFG.gpu_switch == "ON" else 0.7,
                     cat_features        = cat_cols,
                     early_stopping_rounds = CFG.nbrnd_erly_stp,
             ),
             {"verbose" : 0,} 
           ],

 f'LGBM1C'  : [
            LGBMC(**{  "objective"            : "binary",
                       "eval_metric"          : "auc",
                       'device'               : "gpu" if CFG.gpu_switch == "ON" else "cpu",
                       'learning_rate'        : 0.01,
                       'n_estimators'         : 10000 if CFG.test_req == False else CFG.test_iter,
                       'max_depth'            : 7,
                       'subsample'            : 0.90,
                       'colsample_bytree'     : 0.60,
                       'reg_lambda'           : 1.25,
                       'reg_alpha'            : 0.001,
                       'verbosity'            : -1,
                       'random_state'         : CFG.state,
                      } 
                   ),
              {
                  "callbacks" : [
                      log_evaluation(0), 
                      early_stopping(CFG.nbrnd_erly_stp, verbose = False)
                  ]
              }
             ],

 f'LGBM2C'  : [
            LGBMC(**{  "objective"            : "binary",
                       "data_sample_strategy" : "goss",
                       "eval_metric"          : "auc",
                       'device'               : "gpu" if CFG.gpu_switch == "ON" else "cpu",
                       'learning_rate'        : 0.01,
                       'n_estimators'         : 10000 if CFG.test_req == False else CFG.test_iter,
                       'max_depth'            : 6,
                       'subsample'            : 0.825,
                       'colsample_bytree'     : 0.55,
                       'reg_lambda'           : 0.85,
                       'reg_alpha'            : 0.001,
                       'verbosity'            : -1,
                       'random_state'         : CFG.state,
                      } 
                   ),
              {
                  "callbacks" : [
                      log_evaluation(0), 
                      early_stopping(CFG.nbrnd_erly_stp, verbose = False)
                  ]
              }
             ],
}

# Initializing model outputs
OOF_Preds    = {}
Mdl_Preds    = {}



%%time 

drop_cols = ["Source", "id", "Id", "Label", CFG.target, "fold_nb"]

for method, (mymodel, fit_params) in tqdm(Mdl_Master.items()) :
    md = ModelTrainer(
        drop_cols    = drop_cols, 
        problem_type = "binary",
        len_train    = Xtrain.loc[Xtrain.Source == "Competition"].shape[0]
    )

    _, oof_preds, mdl_preds = \
    md.fit_predict(
        Xtrain.copy(), 
        ytrain,
        Xtest.copy(), 
        ygrp,
        extra    = None,
        method   = method,
        mymodel  = mymodel,
        cat_cols = cat_cols,
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


model     = LRC(C = 0.05, max_iter = 100_000)
method    = "L2Blend1C"
oof_preds = pd.DataFrame(OOF_Preds)
mdl_preds = pd.DataFrame(Mdl_Preds)

md = ModelTrainer(
    drop_cols    = drop_cols, 
    problem_type = "binary",
    len_train    = len(oof_preds)
)

_, oof_ens_preds, test_preds = \
md.fit_predict(
    oof_preds.copy(), 
    ytrain.iloc[0 : len(oof_preds)],
    mdl_preds.copy(), 
    ygrp.iloc[0 : len(oof_preds)],
    extra    = None,
    method   = method,
    mymodel  = mymodel,
    cat_cols = None,
    **fit_params,
) 

print()
with sns.axes_style("white") :
    fig, ax = plt.subplots(1,1, figsize = (6, 4))
    sns.histplot(data = oof_ens_preds, bins = 25, color = "tab:blue")
    ax.tick_params(axis = 'x', labelsize = 7.5, rotation = 90)
    ax.set_title("Prediction histogram\n", **CFG.title_specs)
    plt.show()


pp.sub_fl[CFG.target] = test_preds
pp.sub_fl.to_csv(f"submission.csv", index = False)

!ls
print()
!head submission.csv

