


%%time 

!uv pip install -q --system -r /kaggle/input/diamond-public-imports-v1/req_kaggle.txt

exec( open(f"/kaggle/input/diamond-public-imports-v1/myimports.py", "r").read() )
exec( open(f"/kaggle/input/diamond-public-imports-v1/myutils.py", "r").read() )
exec( open(f"/kaggle/input/diamond-public-imports-v1/training.py", "r").read() )
exec( open(f"/kaggle/input/diamond-public-imports-v1/mypp.py", "r").read() )

print()


%%time

utils = Utils(327, 18797)

class CFG:
    """
    Configuration class for parameters and CV strategy for tuning and training
    Some parameters may be unused here as this is a general configuration class
    """

    # Data preparation:-
    version_nb         = 1
    model_id           = "V1_3"
    model_label        = "ML"
    test_req           = False
    test_iter          = 20
    gpu_switch         = "ON" if torch.cuda.is_available() else "OFF"
    state              = 42
    target             = f"price"
    grouper            = f""
    
    tgt_mapper         = {}
    
    ip_path            = f"/kaggle/input/predicting-the-price-of-diamond"
    op_path            = f"/kaggle/working"
    orig_path          = f"/kaggle/input/diamonds/diamonds.csv"
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

def make_ftre(df : pd.DataFrame, cat_cols, num_cols):
    "Makes secondary features for the dataset"
    
    X            = df.copy()
    X[cat_cols]  = X[cat_cols].astype("string").fillna("missing")

    for c1, c2 in combinations( num_cols, 2) :
        lbl = f"{c1}-{c2}"
        X[lbl]           = X[c1] *  X[c2]   
        X[f"l2_{lbl}"]   = X[c1]**2 +  X[c2]**2 
        X[f"hsum_{lbl}"] = X[c1] +  X[c2]
        X[f"hlsum_{lbl}"]= X[c1] +  np.log1p(X[c2])
        X[f"hdif_{lbl}"] = X[c1] -  X[c2]
      
    print(f"---> Shape = {X.shape} 2-grams")
    
    for c1, c2, c3 in combinations( num_cols, 3) :
        lbl              = f"{c1}-{c2}-{c3}"
        X[lbl]           = X[c1] * X[c2] * X[c3]
        X[f"l2_{lbl}"]   = X[c1]**2 + X[c2]**2 + X[c3]**2
        X[f"hsum_{lbl}"] = X[c1] +  X[c2] + X[c3]
        X[f"hlsum_{lbl}"]= X[c1] +  np.log1p(X[c2]) + np.log1p(X[c3]) 
        X[f"hdif_{lbl}"] = X[c1] -  X[c2] - X[c3]
        
    print(f"---> Shape = {X.shape} 3-grams")
    return X
    



%%time 

Xtrain   = pp.train.copy()
ytrain   = pp.train[CFG.target]
Xtest    = pp.test.copy()

num_cols = list( 
    set(Xtest.columns).
    difference(set(pp.cat_cols)).
    difference({"Source"}) 
)

del Xtrain[CFG.target]
print(f"\n---> Shape = {Xtrain.shape} {Xtest.shape}")

print(f"---> Creating base features")
Xtrain = make_ftre(Xtrain, pp.cat_cols, num_cols)
Xtest  = make_ftre(Xtest, pp.cat_cols, num_cols)
print(f"---> Shape = {Xtrain.shape} {Xtest.shape}\n")

print(f"---> Using count encoder")
df      = pd.concat([pp.train[pp.strt_ftre], pp.test], axis=0)
ce_cols = []

for col in pp.strt_ftre[0 : -1] :
    mapper = df[col].value_counts().to_dict()
    df[f"CE_{col}"] = df[col].map(mapper).astype(np.int32)
    ce_cols.append(f"CE_{col}")
    del mapper

Xtrain[ce_cols] = df.iloc[0 :-1* len(pp.test) , :][ce_cols].values
Xtest[ce_cols]  = df.iloc[-1* len(pp.test) : , :][ce_cols].values  
print(f"---> Shape = {Xtrain.shape} {Xtest.shape}\n")

print(f"---> Using original as columns")
for col in pp.strt_ftre[0 : -1] : 
    df_ = pp.original[[col, CFG.target]].groupby(col, as_index = True).agg({CFG.target : ["mean", "count"]})
    df_.columns = [f"Omean_{col}", f"Ocount_{col}"]

    Xtrain = Xtrain.merge(df_, how = "left", left_on = col, right_index = True)
    Xtest  = Xtest.merge(df_, how = "left", left_on = col, right_index = True)
    del df_
print(f"---> Shape = {Xtrain.shape} {Xtest.shape}\n")

_ = utils.CleanMemory()
print()


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
                 XGBR(**{  "objective"            : "reg:squarederror",
                           "eval_metric"          : "rmse",
                           'device'               : "cuda:0" if CFG.gpu_switch == "ON" else "cpu",
                           'learning_rate'        : 0.01,
                           'n_estimators'         : 800 if CFG.test_req == False else CFG.test_iter,
                           'max_depth'            : 6,
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
                         max_depth           = 5,
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
                 {"callbacks"   : [log_evaluation(0)], "eval_metric" : "rmse",},
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
                 {"callbacks"   : [log_evaluation(0)], "eval_metric" : "rmse",},
             ],

 f"CB1R"  : [CBR(
                     iterations        = 750,
                     loss_function     = "RMSE",
                     max_depth         = 5,
                     learning_rate     = 0.02,
                     l2_leaf_reg       = 1.25,
                     colsample_bylevel = None if CFG.gpu_switch == "ON" else 0.55,
                     task_type         = "GPU" if CFG.gpu_switch == "ON" else "CPU",
                     random_state      = CFG.state,
                     verbose           = 0,
                     cat_features      = pp.cat_cols,
                ), 
             {"verbose" : 0}
            ]
}

# Initializing model outputs
OOF_Preds    = {}
Mdl_Preds    = {}



%%time 

drop_cols = ["Source", "id", "Id", "Label", CFG.target, "fold_nb"]

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
        cat_cols = pp.cat_cols,
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

