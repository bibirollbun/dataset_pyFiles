


!uv pip install -q --system -r /kaggle/input/playgrounds5e7-public-imports-v1/req_kaggle.txt

season  = 5
episode = 7

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
    model_id           = "V1_6"
    model_label        = "ML"
    test_req           = False
    test_iter          = 200
    gpu_switch         = "ON" if torch.cuda.is_available() else "OFF"
    state              = 42
    target             = f"Personality"
    grouper            = f""
    
    tgt_mapper         = {"Extrovert" : 0, "Introvert" : 1}
    
    ip_path            = f"/kaggle/input/playground-series-s5e7"
    op_path            = f"/kaggle/working"
    orig_path          = f"/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_dataset.csv"
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

collect()


%%time 

pp = Preprocessor()
pp.DoPreprocessing();


%%time 

if CFG.dtl_preproc_req == True :
    advcv = AdversarialCVMaker()
    advcv.make_cv(pp.train[pp.test.columns], pp.test)


%%time 

Xtrain    = pp.train.drop(CFG.target, axis=1)
Xtest     = pp.test.copy()
ytrain    = pp.train[CFG.target]
PrintColor(f"---> Shapes = {Xtrain.shape} {Xtest.shape}")

df       = pd.concat([Xtrain, Xtest], axis = 0, ignore_index = True)
num_cols = Xtest.select_dtypes(np.number).columns.tolist()

pipe = \
ColumnTransformer(
    [
        ("Cat", 
         make_pipeline(
            *[SimpleImputer(fill_value = "missing", strategy = "constant"), 
              OrdinalEncoder(dtype = np.int16),
              FunctionTransformer( lambda x : x + 1 )
             ],
          ),
          pp.cat_cols
        ),
        ("Num", 
         make_pipeline(
             *[SimpleImputer(fill_value = -1, strategy = "constant"), 
               FunctionTransformer( lambda x : (x+1).astype(np.int16) )
              ] ,
          ),
          num_cols 
        )
    ], 
    remainder = "passthrough", 
    verbose_feature_names_out = False
).set_output(transform = "pandas")

Xtrain = pipe.fit_transform(Xtrain)
Xtest  = pipe.transform(Xtest)
print(f"---> Shapes = {Xtrain.shape} {Xtest.shape}")

df = pp.original.copy()
y  = df[CFG.target].map(CFG.tgt_mapper).astype(np.uint8)
df = pipe.transform(df.drop(CFG.target, axis=1))
df[CFG.target] = y
del y

for col in pp.strt_ftre[0: -1] :
    df_ = df.groupby( col ).agg({CFG.target : ["mean", "count"]})
    df_.columns = [f"O{col}_mean", f"O{col}_count"]
    
    Xtrain = Xtrain.merge(df_, how = "left", left_on = col, right_index = True)
    Xtest  = Xtest.merge(df_, how = "left", left_on = col, right_index = True)
    del df_

del df
print(f"---> Shapes = {Xtrain.shape} {Xtest.shape} | Original as columns")

for col1, col2 in combinations(list(pp.strt_ftre[0:-1]), 2) :
    label = f"{col1}-{col2}"
    Xtrain[label] = Xtrain[col1].astype("string") + "-" + Xtrain[col2].astype("string")
    Xtest[label]  = Xtest[col1].astype("string")  + "-" + Xtest[col2].astype("string")

print(f"---> Shapes = {Xtrain.shape} {Xtest.shape} | 2-grams")

for col1, col2, col3 in combinations(list(pp.strt_ftre[0:-1]), 3) :
    label = f"{col1}-{col2}-{col3}"
    Xtrain[label] = Xtrain[col1].astype("string") + "-" + Xtrain[col2].astype("string") + Xtrain[col3].astype("string") 
    Xtest[label]  = Xtest[col1].astype("string")  + "-" + Xtest[col2].astype("string")  + Xtest[col3].astype("string") 

print(f"---> Shapes = {Xtrain.shape} {Xtest.shape} | 3-grams")

enc_cols = list(Xtest.filter(regex = "-", axis=1).columns)

PrintColor(f"\n---> Target Encoding columns\n")
with np.printoptions(linewidth = 150, threshold = 5000) :
    pprint(np.array(enc_cols))

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
 f'XGB1C'  : [Pipeline(
                 steps = [("TE", ColumnTransformer(
                                     [("Enc", TargetEncoder(random_state = CFG.state), enc_cols)],
                                     remainder = "passthrough", verbose_feature_names_out = False
                                 )
                          ), 
                          ("M" , XGBC(**{  "objective"            : "binary:logistic",
                                           "eval_metric"          : "logloss",
                                           'device'               : "cuda:0" if CFG.gpu_switch == "ON" else "cpu",
                                           'learning_rate'        : 0.02,
                                           'n_estimators'         : 1500,
                                           'max_depth'            : 5,
                                           'colsample_bytree'     : 0.90,
                                           'verbosity'            : 0,
                                           'random_state'         : CFG.state,
                                           'early_stopping_rounds': None,
                                           'enable_categorical'   : True,
                                          } 
                                       ),
                          )
                         ]
     
              ), 
              {"M__verbose" : 0}
             ],

 f"CB1C"   : [Pipeline(
                 steps = [("TE", ColumnTransformer(
                                     [("Enc", TargetEncoder(random_state = CFG.state), enc_cols)],
                                     remainder = "passthrough", verbose_feature_names_out = False
                                 )
                          ),
                          ("M" , CBC(**{   "loss_function"        : "Logloss",
                                           'task_type'            : "GPU" if CFG.gpu_switch == "ON" else "CPU",
                                           'learning_rate'        : 0.02,
                                           'iterations'           : 1500,
                                           'max_depth'            : 5,
                                           'colsample_bylevel'    : 0.90 if CFG.gpu_switch == "OFF" else None,
                                           'verbose'              : 0,
                                           'random_state'         : CFG.state,
                                           'early_stopping_rounds': None,
                                        } 
                                    ),
                          )
                         ]
              ), 
              {"M__verbose" : 0}
             ], 

 f'LGBM1C' : [Pipeline(
                 steps = [("TE", ColumnTransformer(
                                     [("Enc", TargetEncoder(random_state = CFG.state), enc_cols)],
                                     remainder = "passthrough", verbose_feature_names_out = False
                                 )
                          ),
                          ("M" , LGBMC(**{ "objective"            : "binary",
                                           "metric"               : "logloss",
                                           'device'               : "gpu" if CFG.gpu_switch == "ON" else "cpu",
                                           'learning_rate'        : 0.02,
                                           'n_estimators'         : 1500,
                                           'max_depth'            : 5,
                                           'colsample_bytree'     : 0.45,
                                           'verbosity'            : -1,
                                           'random_state'         : CFG.state,
                                           'reg_lambda'           : 1.50,
                                          } 
                                       ),
                          )
                         ]
              ), 
              {"M__callbacks" : [log_evaluation(0)]}
             ],

 f'LGBM2C' : [Pipeline(
                 steps = [("TE", ColumnTransformer(
                                     [("Enc", TargetEncoder(random_state = CFG.state), enc_cols)],
                                     remainder = "passthrough", verbose_feature_names_out = False
                                 )
                          ),
                          ("M" , LGBMC(**{ "objective"            : "binary",
                                           "metric"               : "logloss",
                                           "data_sample_strategy" : 'goss',
                                           'device'               : "gpu" if CFG.gpu_switch == "ON" else "cpu",
                                           'learning_rate'        : 0.0175,
                                           'n_estimators'         : 700,
                                           'max_depth'            : 4,
                                           'colsample_bytree'     : 0.35,
                                           'verbosity'            : -1,
                                           'random_state'         : CFG.state,
                                           'reg_lambda'           : 2.50,
                                          } 
                                       ),
                          )
                         ]
              ), 
              {"M__callbacks" : [log_evaluation(0)]}
             ],

 f'RF1C'  : [Pipeline(
                 steps = [("TE", ColumnTransformer(
                                     [("Enc", TargetEncoder(random_state = CFG.state), enc_cols)],
                                     remainder = "passthrough", verbose_feature_names_out = False
                                 )
                          ),
                          ("M" , RFC(**{"n_estimators"     : 100,
                                        "max_depth"        : 6,
                                        "min_samples_leaf" : 16, 
                                        'verbose'          : 0,
                                       } 
                                    ),
                          )
                         ]
              ), 
              {},
             ],

 f'HGB1C' : [Pipeline(
                 steps = [("TE", ColumnTransformer(
                                     [("Enc", TargetEncoder(random_state = CFG.state), enc_cols)],
                                     remainder = "passthrough", verbose_feature_names_out = False
                                 )
                          ),
                          ("M" , HGBC(**{'learning_rate'    : 0.03,
                                         'min_samples_leaf' : 12,
                                         'max_iter'         : 500,
                                         'max_depth'        : 5,
                                         'l2_regularization': 0.75,
                                        } 
                                    ),
                          )
                         ]
              ), 
              {},
             ],
}

# Initializing model outputs
OOF_Preds    = {}
Mdl_Preds    = {}


%%time 

drop_cols = ["Source", "id", "Id", "Label", CFG.target, "fold_nb"]
ytrain_   = ytrain.map(CFG.tgt_mapper).astype(np.uint8)

for method, (mymodel, fit_params) in tqdm(Mdl_Master.items()) :
    md = ModelTrainer(
        drop_cols    = drop_cols, 
        problem_type = "binary",
        len_train    = Xtrain.loc[Xtrain.Source == "Competition"].shape[0]
    )

    _, oof_preds, mdl_preds = \
    md.fit_predict(
        Xtrain.copy(), 
        ytrain_,
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

    score = utils.ScoreMetric(ytrain_.values[0 : len(oof_preds)], oof_preds)
    PrintColor(
        f"---> Overall score = {score :,.8f} | Competition metric",
        color = Fore.CYAN
    )

_ = utils.CleanMemory()
print()


%%time 

drop_cols = ["Source", "id", "Id", "Label", CFG.target, "fold_nb"]
ytrain_   = ytrain.map(CFG.tgt_mapper).astype(np.uint8)
oof_preds = pd.DataFrame(OOF_Preds).assign(Source = "Competition")
mdl_preds = pd.DataFrame(Mdl_Preds).assign(Source = "Competition")

method     = "LR1C"
mymodel    = LRC(C = 0.01, max_iter = 10000)
fit_params = {}

md = ModelTrainer(
    drop_cols    = drop_cols, 
    problem_type = "binary",
    len_train    = oof_preds.shape[0]
)

_, oof_ens_preds, test_preds = \
md.fit_predict(
    oof_preds, 
    ytrain_,
    mdl_preds, 
    ygrp,
    extra    = None,
    method   = method,
    mymodel  = mymodel,
    cat_cols = None,
    **fit_params,
)   

score = utils.ScoreMetric(ytrain_.values[0 : len(oof_ens_preds)], oof_ens_preds)
PrintColor(
    f"---> Overall score = {score :,.8f} | Competition metric",
    color = Fore.CYAN
)

_ = utils.CleanMemory()
print()


%%time 

thresholds    = 0

for fold_nb, (train_idx, dev_idx) in enumerate( cv.split(oof_preds, ytrain_) , start = 1) :
    Xtr  = oof_ens_preds[train_idx].flatten()
    ytr  = ytrain_.iloc[train_idx].values
    Xdev = oof_ens_preds[dev_idx].flatten()
    ydev = ytrain_.iloc[dev_idx].values

    best_score, best_cutoff  = 0, 0
    for cutoff in np.arange(0.05, 0.9501, 0.001) :
        score = utils.ScoreMetric(ytr, Xtr, cutoff = cutoff)
        if score >= best_score :
            best_score  = score
            best_cutoff = cutoff
        else:
            pass

    print(f"{fold_nb}. Cutoff = {best_cutoff:,.4f} {best_score:,.8f}")
    thresholds += best_cutoff / CFG.n_splits

score = utils.ScoreMetric(ytrain_.values, oof_ens_preds.flatten(), cutoff = thresholds)
PrintColor(f"\n---> Best score = {score:,.8f} | cutoff = {thresholds:,.4f} \n")


%%time 

pp.sub_fl[CFG.target] = np.where(test_preds >= thresholds, 1, 0)
pp.sub_fl[CFG.target] = pp.sub_fl[CFG.target].map({v:k for k, v in CFG.tgt_mapper.items()})

pp.sub_fl.to_csv("submission.csv", index = None)
display(
    pp.sub_fl[CFG.target].value_counts(normalize = True)
)

print()
!ls submission.csv
print()
!head submission.csv

