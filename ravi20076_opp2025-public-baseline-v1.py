


%%time 

!pip install -q polars==1.29.0      --no-index --find-links=/kaggle/input/opp2025-public-imports-v1/packages
!pip install -q scikit-learn==1.7.0 --no-index --find-links=/kaggle/input/opp2025-public-imports-v1/packages
!pip install -q xgboost==3.0.2      --no-index --find-links=/kaggle/input/opp2025-public-imports-v1/packages
!pip install -q lightgbm==4.6.0     --no-index --find-links=/kaggle/input/opp2025-public-imports-v1/packages
!pip install -q rdkit               --no-index --find-links=/kaggle/input/opp2025-public-imports-v1/packages

exec( open(f"/kaggle/input/opp2025-public-imports-v1/myimports.py", "r").read() )
exec( open(f"/kaggle/input/opp2025-public-imports-v1/myutils.py", "r").read() )
exec( open(f"/kaggle/input/opp2025-public-imports-v1/training.py", "r").read() )
exec( open(f"/kaggle/input/opp2025-public-imports-v1/mypp.py", "r").read() )

os.environ["TOKENIZERS_PARALLELISM"] = "false"
print()


%%time

utils = Utils()

class CFG:
    """
    Configuration class for parameters and CV strategy for tuning and training
    Some parameters may be unused here as this is a general configuration class
    """;

    # Data preparation:-
    version_nb         = 1
    model_id           = "V1_4"
    model_label        = "ML"
    test_req           = False
    test_iter          = 200
    gpu_switch         = "ON" if torch.cuda.is_available() else "OFF"
    state              = 42
    
    target             = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    grouper            = f""
    tgt_mapper         = {}
    ip_path            = f"/kaggle/input/neurips-open-polymer-prediction-2025"
    op_path            = f"/kaggle/working"
    orig_path          = f""
    data_path          = f""
    dtl_preproc_req    = True
    ftre_plots_req     = True
    ftre_imp_req       = True
    nb_orig            = 0
    orig_all_folds     = False

    # Model Training:-
    pstprcs_oof        = True
    pstprcs_train      = False
    pstprcs_test       = False
    ML                 = True
    test_preds_req     = True
    n_splits           = 5
    n_repeats          = 1
    nbrnd_erly_stp     = 100
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
 "RKF"   : RepeatedKFold(n_splits   = CFG.n_splits, n_repeats= CFG.n_repeats, random_state= CFG.state),
 "RSKF"  : RepeatedStratifiedKFold(n_splits  = CFG.n_splits, n_repeats= CFG.n_repeats, random_state= CFG.state),
 "SKF"   : StratifiedKFold(n_splits   = CFG.n_splits, shuffle = True, random_state= CFG.state),
 "KF"    : KFold(n_splits = CFG.n_splits, shuffle = True, random_state= CFG.state),
 "GKF"   : GroupKFold(n_splits   = CFG.n_splits),
 "SGKF"  : StratifiedGroupKFold(n_splits = CFG.n_splits, shuffle = True, random_state= CFG.state),
}

collect()


%%writefile myfe.py

class FeatureMaker :
    "Secondary feature developer for the input data"

    def __init__(self):
        pass

    def _compute_all_descriptors(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [None] * len(desc_names)
        return [desc[1](mol) for desc in Descriptors.descList]

    def fit(self, X: pd.DataFrame, y = None, **params):
        return self

    def transform(self, X: pd.DataFrame, y = None, **params):
        "Creates secondary features for the dataset provided. Also cleans Ipc column with log transform"

        desc_names  = [desc[0] for desc in Descriptors.descList]
        descriptors = [self._compute_all_descriptors(smi) for smi in X['SMILES'].to_list()]
        descriptors = pd.DataFrame(descriptors, columns = desc_names)

        df        = pd.concat([X, descriptors], axis=1)
        df["Ipc"] = np.log1p(df["Ipc"])
        
        self.feature_names_out_ = list(df.columns)
        return df.replace([np.inf, -1*np.inf], np.nan)

    def fit_transform(self, X: pd.DataFrame, y = None, **params) :
        "Fits and transforms using the methods here"

        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, X = None) :
        return self.feature_names_out_
        


%%time 

exec( open(f"myfe.py", "r").read() )

train    = pd.read_csv(f"{CFG.ip_path}/train.csv")
test     = pd.read_csv(f"{CFG.ip_path}/test.csv")
sel_cols = train.drop(CFG.target, axis=1).columns

fe      = FeatureMaker()
Xtrain  = fe.fit_transform( train[sel_cols] )
Xtest   = fe.transform( test[sel_cols] )

print(f"\n---> Shapes = {Xtrain.shape} {Xtest.shape}")

std_      = Xtrain.select_dtypes(np.number).std()
nulls_    = Xtrain.isna().mean()
drop_cols = std_.loc[std_ <= 1e-6].index.tolist() + nulls_.loc[nulls_ >= 0.9975].index.tolist()
Xtrain    = Xtrain.drop(drop_cols, axis=1)
Xtest     = Xtest.drop(drop_cols, axis=1)

with np.printoptions(linewidth = 100):
    PrintColor(f"\n---> Dropped columns\n")
    pprint(np.array(drop_cols))
    print()
del std_, drop_cols

print(f"---> Shapes = {Xtrain.shape} {Xtest.shape}\n")

Ytrain = train[CFG.target]
display(
    Ytrain.isna().
    mean(axis=0).
    to_frame().
    rename({0 : "Nulls-Mean"}, axis=1).
    style.
    set_caption(f"Target null analysis").
    format(precision = 4)
)

print(f"\n---> Shapes = {Xtrain.shape} {Xtest.shape}")
_ = utils.CleanMemory()


%%time 

num_cols = Xtest.select_dtypes(np.number).columns.tolist()

for col in tqdm(num_cols) :
    mean_       = pd.concat([ Xtrain[col], Xtest[col] ], axis=0).dropna().mean()
    Xtrain[col] = Xtrain[col].fillna(mean_)
    Xtest[col]  = Xtest[col].fillna(mean_)
    del mean_

PrintColor(f"\n---> Nulls after imputation- Train")
_ = Xtrain.isna().sum()
print(_.loc[_ > 0])
PrintColor(f"\n---> Nulls after imputation- Test")
_ = Xtest.isna().sum()
print(_.loc[_ > 0])

print()


%%time 

OOF_Preds    = []
Mdl_Preds    = []
Artefacts    = {}
drop_cols    = ["Source", "id", "Id", "Label", "fold_nb", "SMILES"] + CFG.target



%%time 

Mdl_Master = \
{   
 f'XGB1R'  :  [XGBR(**{ 'device'                : "cuda:0" if CFG.gpu_switch == "ON" else "cpu",
                        'objective'             : "reg:absoluteerror",
                        "n_estimators"          : 5000 if CFG.test_req == False else CFG.test_iter, 
                        "learning_rate"         : 0.02,
                        'max_depth'             : 6,
                        "colsample_bytree"      : 0.30,
                        "colsample_bylevel"     : 0.35,
                        "random_state"          : CFG.state,
                        "verbosity"             : 0,
                        "reg:lambda"            : 1.25,
                        "early_stopping_rounds" : CFG.nbrnd_erly_stp if CFG.nbrnd_erly_stp > 0 else None,
                        "enable_categorical"    : True,
                      }
                   ),
               {"verbose" : 0},
              ],

 f'HGB1R' :  [HGBR(**{  'max_iter'              : 800 if CFG.test_req == False else CFG.test_iter,
                        'loss'                  : "absolute_error",
                        "learning_rate"         : 0.02,
                        "max_depth"             : 5,
                        "min_samples_leaf"      : 8,
                        "random_state"          : CFG.state,
                        "verbose"               : 0,
                        "l2_regularization"     : 0.85,
                      }
                   ),
               {},
              ],

 f'CB1R'    : [CBR(**{'task_type'             : "GPU" if CFG.gpu_switch == "ON" else "CPU",
                      'loss_function'         : "MAE",
                      'eval_metric'           : "MAE",
                      'learning_rate'         : 0.015,
                      'iterations'            : 15000 if CFG.test_req == False else CFG.test_iter,
                      'max_depth'             : 5,
                      'min_data_in_leaf'      : 12,
                      'colsample_bylevel'     : 0.30 if CFG.gpu_switch == "OFF" else None,
                      'l2_leaf_reg'           : 1.25,
                      'verbose'               : 0,
                      'random_state'          : CFG.state,
                      'early_stopping_rounds' : CFG.nbrnd_erly_stp if CFG.nbrnd_erly_stp > 0 else None,
                    }
                 ),
               {"verbose" : 0 },
              ],

 f'LGBM1R'  : [LGBMR(**{'device'             : "gpu" if CFG.gpu_switch == "ON" else "cpu",
                        'objective'          : "regression_l1",
                        "n_estimators"       : 5000 if CFG.test_req == False else CFG.test_iter, 
                        "learning_rate"      : 0.015,
                        "colsample_bytree"   : 0.25,               
                        "n_jobs"             : -1,
                        "num_leaves"         : 100,
                        "random_state"       : CFG.state,
                        "subsample"          : 0.25,
                        "verbosity"          : -1,
                      }
                   ),
               {"callbacks" : [log_evaluation(0), early_stopping(CFG.nbrnd_erly_stp, verbose = False)]},
              ],

 f'LGBM2R'  : [LGBMR(**{'device'               : "gpu" if CFG.gpu_switch == "ON" else "cpu",
                        'objective'            : "regression_l1",
                        'data_sample_strategy' : 'goss',
                        "n_estimators"         : 12000 if CFG.test_req == False else CFG.test_iter, 
                        "max_depth"            : 5,
                        "learning_rate"        : 0.015,
                        "colsample_bytree"     : 0.35,               
                        "n_jobs"               : -1,
                        "num_leaves"           : 32,
                        "random_state"         : CFG.state,
                        "subsample"            : 0.325,
                        "verbosity"            : -1,
                      }
                   ),
               {"callbacks" : [log_evaluation(0)]},
              ],
}


%%time 

target = "FFV"

PrintColor(f"\n ==== {target} ====\n", color = Fore.RED)

Xtrain_            = Xtrain.copy()
Xtrain_[target]    = Ytrain[target].values
Xtrain_            = Xtrain_.dropna(subset = [target])
idx                = Xtrain_.index.values
Xtrain_.index      = range(len(Xtrain_))
ytrain_            = Xtrain_[target]

ygrp = np.zeros(len(Xtrain_))
cv   = cv_selector[CFG.mdlcv_mthd]

for fold_nb, (train_idx, dev_idx) in tqdm( enumerate(cv.split(Xtrain_, ytrain_)) ) :
    ygrp[dev_idx] = fold_nb
ygrp_ = pd.Series(ygrp, name = "fold_nb", dtype = np.uint8, index = range(len(ygrp)))
del ygrp

print(f"---> Shapes = {Xtrain_.shape} {ytrain_.shape} {ygrp_.shape}")

oof   = []
preds = []

for method, (mymodel, fit_params) in tqdm(Mdl_Master.items()):

    md = \
    ModelTrainer(
        problem_type     = "regression",
        drop_cols        = drop_cols, 
        len_train        = Xtrain_.shape[0],
        test_preds_req   = CFG.test_preds_req, 
        target           = target,
        pp_preds         = CFG.pstprcs_oof,
    )

    (fitted_models, oof_preds, test_preds) =  \
    md.fit_predict(
        Xtrain_,
        ytrain_,
        Xtest,
        ygrp_,
        extra    = None,
        method   = method,
        mymodel  = mymodel,
        cat_cols = None,
        **fit_params, 
    )

    oof_preds.index = idx
    oof.append( oof_preds.rename({0 : f"{method}"}, axis=1) )
    preds.append( test_preds.rename({0 : f"{method}"}, axis=1) )
    Artefacts[f"{target}_{method}"] = fitted_models

    del fitted_models, oof_preds, test_preds
    collect()

OOF_Preds.append(
    pd.concat(oof, axis=1).
    mean(axis=1).
    to_frame().
    rename({ 0 : f"{target}"}, axis=1)
)
Mdl_Preds.append(
    pd.concat(preds, axis=1).
    mean(axis=1).
    to_frame().
    rename({ 0 : f"{target}"}, axis=1)
)

score = md.score(
    ytrain_, pd.concat(oof, axis=1).mean(axis=1).values.flatten() 
)
PrintColor(f"---> Ensemble CV score = {score:,.8f}\n")

del preds, oof
_ = utils.CleanMemory()


%%time

Mdl_Master = \
{    
 f'LGBM1R'  : [LGBMR(**{'device'             : "gpu" if CFG.gpu_switch == "ON" else "cpu",
                        'objective'          : "regression_l1",
                        "n_estimators"       : 400 if CFG.test_req == False else CFG.test_iter, 
                        "max_depth"          : 4,
                        "learning_rate"      : 0.015,
                        "colsample_bytree"   : 0.25,               
                        "n_jobs"             : -1,
                        "num_leaves"         : 50,
                        "random_state"       : CFG.state,
                        "reg_alpha"          : 0.001,
                        "reg_lambda"         : 0.001,
                        "subsample"          : 0.20,
                        "verbosity"          : -1,
                      }
                   ),
               {"callbacks" : [log_evaluation(0), early_stopping(CFG.nbrnd_erly_stp, verbose = False)]},
              ],

 f'CB1R'    : [CBR(**{'task_type'             : "GPU" if CFG.gpu_switch == "ON" else "CPU",
                      'loss_function'         : "MAE",
                      'eval_metric'           : "MAE",
                      'learning_rate'         : 0.02,
                      'iterations'            : 400 if CFG.test_req == False else CFG.test_iter,
                      'max_depth'             : 4,
                      'min_data_in_leaf'      : 4,
                      'colsample_bylevel'     : 0.30 if CFG.gpu_switch == "OFF" else None,
                      'l2_leaf_reg'           : 1.25,
                      'verbose'               : 0,
                      'random_state'          : CFG.state,
                      'early_stopping_rounds' : CFG.nbrnd_erly_stp if CFG.nbrnd_erly_stp > 0 else None,
                    }
                 ),
               {"verbose" : 0 },
              ], 

 f'XGB1R'  :  [XGBR(**{ 'device'                : "cuda:0" if CFG.gpu_switch == "ON" else "cpu",
                        'objective'             : "reg:absoluteerror",
                        'max_depth'             : 3,
                        "n_estimators"          : 300 if CFG.test_req == False else CFG.test_iter, 
                        "learning_rate"         : 0.0125,
                        "colsample_bytree"      : 0.30,
                        "colsample_bylevel"     : 0.30,
                        "random_state"          : CFG.state,
                        "verbosity"             : 0,
                        "reg:lambda"            : 1.25,
                        "early_stopping_rounds" : None,
                        "enable_categorical"    : True,
                      }
                   ),
               {"verbose" : 0},
              ],

 f'HGB1R' :  [HGBR(**{  'max_iter'              : 125 if CFG.test_req == False else CFG.test_iter,
                        'loss'                  : "absolute_error",
                        "learning_rate"         : 0.015,
                        "max_depth"             : 4,
                        "min_samples_leaf"      : 3,
                        "random_state"          : CFG.state,
                        "verbose"               : 0,
                        "l2_regularization"     : 1.25,
                      }
                   ),
               {},
              ],

 f'RF1R' :  [RFR(**{    'n_estimators'          : 75 if CFG.test_req == False else CFG.test_iter,
                        'criterion'             : "absolute_error",
                        "max_depth"             : 4,
                        "min_samples_split"     : 3,
                        "random_state"          : CFG.state,
                        "verbose"               : 0,
                    }
                ),
               {},
            ],
}


%%time 

for target in tqdm(CFG.target) :
    if target == "FFV" :
        pass
        
    else:
        PrintColor(f"\n ==== {target} ====\n", color = Fore.RED)
        
        Xtrain_            = Xtrain.copy()
        Xtrain_[target]    = Ytrain[target].values
        Xtrain_            = Xtrain_.dropna(subset = [target])
        idx                = Xtrain_.index.values
        Xtrain_.index      = range(len(Xtrain_))
        ytrain_            = Xtrain_[target]
        
        ygrp = np.zeros(len(Xtrain_))
        cv   = cv_selector[CFG.mdlcv_mthd]
        
        for fold_nb, (train_idx, dev_idx) in tqdm( enumerate(cv.split(Xtrain_, ytrain_)) ) :
            ygrp[dev_idx] = fold_nb
        ygrp_ = pd.Series(ygrp, name = "fold_nb", dtype = np.uint8, index = range(len(ygrp)))
        del ygrp
        
        print(f"---> Shapes = {Xtrain_.shape} {ytrain_.shape} {ygrp_.shape}")
        
        oof   = []
        preds = []
        
        for method, (mymodel, fit_params) in tqdm(Mdl_Master.items()):
        
            md = \
            ModelTrainer(
                problem_type     = "regression",
                drop_cols        = drop_cols, 
                len_train        = Xtrain_.shape[0],
                test_preds_req   = CFG.test_preds_req, 
                target           = target,
                pp_preds         = CFG.pstprcs_oof,
            )
        
            (fitted_models, oof_preds, test_preds) =  \
            md.fit_predict(
                Xtrain_,
                ytrain_,
                Xtest,
                ygrp_,
                extra    = None,
                method   = method,
                mymodel  = mymodel,
                cat_cols = None,
                **fit_params, 
            )
        
            oof_preds.index = idx
            oof.append( oof_preds.rename({0 : f"{method}"}, axis=1) )
            preds.append( test_preds.rename({0 : f"{method}"}, axis=1) )
            Artefacts[f"{target}_{method}"] = fitted_models
        
            del fitted_models, oof_preds, test_preds
            collect()
        
        OOF_Preds.append(
            pd.concat(oof, axis=1).
            mean(axis=1).
            to_frame().
            rename({ 0 : f"{target}"}, axis=1)
        )
        Mdl_Preds.append(
            pd.concat(preds, axis=1).
            mean(axis=1).
            to_frame().
            rename({ 0 : f"{target}"}, axis=1)
        )
        
        score = md.score(
            ytrain_, pd.concat(oof, axis=1).mean(axis=1).values.flatten() 
        )
        PrintColor(f"---> Ensemble CV score = {score:,.8f}", color = Fore.BLACK)
        del preds, oof
        
_ = utils.CleanMemory()    


%%time 

sub_fl = pd.read_csv(f"{CFG.ip_path}/sample_submission.csv")

oof_preds = pd.concat(OOF_Preds, axis=1).sort_index(ascending = True)
mdl_preds = pd.concat(Mdl_Preds, axis=1).sort_index(ascending = True)

score = \
utils.ScoreMetric(
    Ytrain.reset_index().rename(columns = {"index": "id"}),
    oof_preds[CFG.target].reset_index().rename(columns = {"index": "id"})
)

PrintColor(f"\n---> Final CV score after all models = {score:.8f}\n\n")

mdl_preds["id"] = sub_fl["id"].values
mdl_preds[["id"] + CFG.target].to_csv(f"submission.csv", index = None)
!head submission.csv

print()

