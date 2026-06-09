


!uv pip install -q --system -r /kaggle/input/playgrounds5e9-public-imports-v1/req_kaggle.txt

season  = 5
episode = 9

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
    model_id           = "V1_8"
    model_label        = "ML"
    test_req           = False
    test_iter          = 20
    gpu_switch         = "ON" if torch.cuda.is_available() else "OFF"
    state              = 42
    target             = f"BeatsPerMinute"
    grouper            = f""
    
    tgt_mapper         = {}
    
    ip_path            = f"/kaggle/input/playground-series-s5e9"
    op_path            = f"/kaggle/working"
    orig_path          = f"/kaggle/input/bpm-prediction-challenge/Train.csv"
    data_path          = f""
    dtl_preproc_req    = True
    ftre_plots_req     = False
    ftre_imp_req       = False
    nb_orig            = 0
    orig_all_folds     = False

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

def make_ftre(df) :
    "This function adds secondary features for the model"
    
    df["is_high_energy"] = (df["Energy"] > 0.7).astype(np.uint8)
    df["is_acoustic"]    = (df["AcousticQuality"] > 0.5).astype(np.uint8)
    df["is_live"]        = (df["LivePerformanceLikelihood"] > 0.5).astype(np.uint8)

    return df
    

Xtrain = make_ftre(pp.train)
Xtest  = make_ftre(pp.test)
ytrain = pp.train[CFG.target]

print(f"\n---> Shape = {Xtrain.shape} {Xtest.shape}\n")
_ = utils.CleanMemory()


%%time 

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
 f'XGB1R'  : [
              XGBR(**{ "objective"            : "reg:squarederror",
                       "eval_metric"          : "rmse",
                       'device'               : "cuda:0" if CFG.gpu_switch == "ON" else "cpu",
                       'learning_rate'        : 0.01,
                       'n_estimators'         : 800 if CFG.test_req == False else CFG.test_iter,
                       'max_depth'            : 3,
                       'min_child_weight'     : 8,
                       'subsample'            : 0.65,
                       'reg_alpha'            : 0.01,
                       'reg_lambda'           : 3.50, 
                       'gamma'                : 1.76,
                       'verbosity'            : 0,
                       'random_state'         : CFG.state,
                       'enable_categorical'   : True,
                      } 
                   ),
              {"verbose" : 0}
             ],

 f"LGBM1R" : [
                 LGBMR(
                         objective           = "l2",
                         max_depth           = 9,
                         num_leaves          = 64, 
                         n_estimators        = 1_000 if CFG.test_req == False else CFG.test_iter,
                         device              = "gpu" if CFG.gpu_switch == "ON" else "cpu", 
                         learning_rate       = 0.01,
                         feature_fraction    = 0.95, 
                         subsample           = 0.90,
                         reg_alpha           = 0.01,
                         reg_lambda          = 4.00, 
                         random_state        = CFG.state,
                         verbosity           = -1,
                 ),
                 {"callbacks"   : [log_evaluation(0)],
                  "eval_metric" : "rmse",
                 } 
             ],

 f"LGBM2R" : [
                 LGBMR(
                         objective            = "l2",
                         max_depth            = 6,
                         num_leaves           = 64, 
                         data_sample_strategy = "goss",
                         n_estimators         = 1_000 if CFG.test_req == False else CFG.test_iter,
                         device               = "gpu" if CFG.gpu_switch == "ON" else "cpu", 
                         learning_rate        = 0.01,
                         reg_alpha            = 0.05,
                         reg_lambda           = 4.50, 
                         random_state         = CFG.state,
                         verbosity            = -1,
                 ),
                 {"callbacks"   : [log_evaluation(0)],
                  "eval_metric" : "rmse",
                 } 
             ],

 f"LGBM3R" : [
                 LGBMR(
                         objective           = "l2",
                         max_depth           = 5,
                         num_leaves          = 32, 
                         n_estimators        = 1_500 if CFG.test_req == False else CFG.test_iter,
                         device              = "gpu" if CFG.gpu_switch == "ON" else "cpu", 
                         learning_rate       = 0.015,
                         reg_alpha           = 0.20,
                         reg_lambda          = 2.00, 
                         random_state        = CFG.state,
                         verbosity           = -1,
                 ),
                 {"callbacks"   : [log_evaluation(0)],
                  "eval_metric" : "rmse",
                 } 
             ],

 f"LGBM4R" : [
                 LGBMR(
                         objective           = "l2",
                         max_depth           = 3,
                         num_leaves          = 256, 
                         n_estimators        = 800 if CFG.test_req == False else CFG.test_iter,
                         device              = "gpu" if CFG.gpu_switch == "ON" else "cpu", 
                         learning_rate       = 0.018,
                         reg_alpha           = 0.01,
                         reg_lambda          = 3.00, 
                         random_state        = CFG.state,
                         verbosity           = -1,
                 ),
                 {"callbacks"   : [log_evaluation(0)],
                  "eval_metric" : "rmse",
                 } 
             ],

 f"LGBM5R" : [
                 LGBMR(
                         objective           = "l2",
                         max_depth           = 2,
                         num_leaves          = 512, 
                         n_estimators        = 256 if CFG.test_req == False else CFG.test_iter,
                         device              = "gpu" if CFG.gpu_switch == "ON" else "cpu", 
                         learning_rate       = 0.01,
                         reg_alpha           = 0.01,
                         reg_lambda          = 1.50, 
                         random_state        = CFG.state,
                         verbosity           = -1,
                 ),
                 {"callbacks"   : [log_evaluation(0)],
                  "eval_metric" : "rmse",
                 } 
             ],
}


# Initializing model outputs
OOF_Preds    = {}
Mdl_Preds    = {}



%%time 

drop_cols = ["Source", "id", "Id", "Label", CFG.target, "fold_nb"]
cat_cols  = []

for method, (mymodel, fit_params) in tqdm(Mdl_Master.items()) :
    md = ModelTrainer(
        drop_cols    = drop_cols, 
        problem_type = "regression",
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
        cat_cols = None,
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


%%time 

oof_preds     = pd.DataFrame(OOF_Preds)
mdl_preds     = pd.DataFrame(Mdl_Preds)
ens_oof_preds = oof_preds.mean(axis=1).values.flatten()
ens_mdl_preds = mdl_preds.mean(axis=1).values.flatten()

score = utils.ScoreMetric(
    ytrain.values[0 : len(ens_oof_preds)], 
    ens_oof_preds,
)
PrintColor(
    f"\n---> Ensemble OOF score = {score:,.8f} | Before calibration", 
    color = Fore.RED
)

clb = sk.isotonic.IsotonicRegression(out_of_bounds = "clip")
clb.fit(ens_oof_preds, ytrain.values[0 : len(ens_oof_preds)])
ens_oof_preds = clb.transform(ens_oof_preds)
ens_mdl_preds = clb.transform(ens_mdl_preds)

score = utils.ScoreMetric(
    ytrain.values[0 : len(ens_oof_preds)], 
    ens_oof_preds,
)
PrintColor(
    f"---> Ensemble OOF score = {score:,.8f} | After calibration\n\n", 
    color = Fore.BLUE
)


%%time 

pp.sub_fl[CFG.target] = ens_mdl_preds
pp.sub_fl.to_csv("submission.csv", index = None)

print()
!ls submission.csv
print()
!head submission.csv

