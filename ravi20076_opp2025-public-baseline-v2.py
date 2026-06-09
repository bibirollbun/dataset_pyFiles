


%%time 

!pip install -q polars==1.31.0      --no-index --find-links=/kaggle/input/opp2025-public-imports-v1/packages
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
    version_nb         = 2
    model_id           = "V2_3"
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
    data_path          = f"/kaggle/input/opp2025-public-imports-v1/XYtrain.csv"
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

    def __init__(
        self, 
        max_autocorr : int   = 10, 
        verbosity    : int   = 0
    ):
        self.max_autocorr = max_autocorr
        self.verbosity    = verbosity

    def _get_molecular_descriptors(self)-> tuple:
        """
        Get molecular descriptors - either hardcoded list or auto-discovered
        """
    
        descriptor_list_all = []
        test_mol = Chem.MolFromSmiles('CCO')
    
        for name in dir(Descriptors):
            if not name.startswith('_'):
                try:
                    func = getattr(Descriptors, name)
                    if callable(func):
                        result = func(test_mol)
                        if isinstance(result, (int, float)) and not np.isnan(result):
                            descriptor_list_all.append((name, func))
                except:
                    pass
    
        print(f"---> Total discovered descriptors before filtering: {len(descriptor_list_all)}")
    
        autocorr_descriptors = [
            (name, func)
            for name, func in descriptor_list_all
            if name.startswith('AUTOCORR2D_')
        ]
        autocorr_descriptors.sort(key=lambda x: int(x[0].split('_')[-1]))
        limited_autocorr = autocorr_descriptors[: self.max_autocorr]
    
        other_descriptors = [
            (name, func)
            for name, func in descriptor_list_all
            if not name.startswith('AUTOCORR2D_')
        ]
    
        descriptor_list = limited_autocorr + other_descriptors
        print(f"---> Auto-discovered {len(descriptor_list)} descriptors (limited to {self.max_autocorr} AUTOCORR2D):")
        names = [name for name, _ in descriptor_list]

        if self.verbosity > 0:
            print("  " + ", ".join(names))
    
        feature_names = [name for name, _ in descriptor_list]
        return (descriptor_list, feature_names)

    def _smiles_to_features(self, smiles_list, descriptor_functions)-> np.ndarray:
       """
       Convert SMILES strings to raw feature matrix
       """
       
       features = []
       total    = len(smiles_list)
           
       for i, smiles in enumerate(smiles_list):        
           mol_features = []
           try:
               mol = Chem.MolFromSmiles(smiles)
               if mol is None:
                   mol_features = [np.nan] * len(descriptor_functions)
               else:
                   for name, func in descriptor_functions:
                       try:
                           value = func(mol)
                           if np.isinf(value) or abs(value) > 1e10:
                               value = np.nan
                           mol_features.append(value)
                       except:
                           mol_features.append(np.nan)
           except:
               mol_features = [np.nan] * len(descriptor_functions)
               
           features.append(mol_features)
       
       return np.array(features, dtype=float)

    def fit(self, X: pd.DataFrame, y = None, **params):
        "Fits the transformer to the training data"
        return self
        
    def transform(self, X: pd.DataFrame, y = None, **params) -> pd.DataFrame:
        "Creates secondary features for the dataset provided. Also cleans Ipc column with log transform"

        descriptor_functions, feature_names = self._get_molecular_descriptors()
        self.feature_names_out_ = feature_names
        
        return pd.DataFrame(
            self._smiles_to_features(X["SMILES"].values, descriptor_functions),
            columns = feature_names
        )

    def fit_transform(self, X: pd.DataFrame, y = None, **params) -> pd.DataFrame :
        "Fits and transforms using the methods here"

        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, X = None) :
        return self.feature_names_out_ 


%%time

exec( open(f"myfe.py", "r").read() )

train    = pd.read_csv( CFG.data_path )
test     = pd.read_csv(f"{CFG.ip_path}/test.csv")
sel_cols = train.drop(CFG.target, axis=1).columns

fe = FeatureMaker(max_autocorr = 40)
PrintColor(f"---> Train feature engineering")
Xtrain  = fe.fit_transform( train[sel_cols] )
PrintColor(f"\n---> Test feature engineering")
Xtest   = fe.transform( test[sel_cols] )

print(f"\n---> Shapes = {Xtrain.shape} {Xtest.shape}")

std_      = Xtrain.select_dtypes(np.number).std()
nulls_    = Xtrain.isna().mean()
drop_cols = std_.loc[std_ <= 1e-5].index.tolist() + nulls_.loc[nulls_ >= 0.9500].index.tolist()
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
    imp_       = pd.concat([ Xtrain[col], Xtest[col] ], axis=0).dropna().median()
    Xtrain[col] = Xtrain[col].fillna(imp_)
    Xtest[col]  = Xtest[col].fillna(imp_)
    del imp_

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

def make_offline_models(
    targets: list, Xtrain, Ytrain, Mdl_Master, drop_cols,
):
    "Makes a CV loop for the provided models and the provided targets and stores the predictions in global variables"

    global OOF_Preds, Mdl_Preds, Artefacts

    for target in tqdm( targets ) :
        PrintColor(
            f"\n {'=' * 10} {target} {'=' * 10} \n", 
            color = Fore.RED
        )
        
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
            median(axis=1).
            to_frame().
            rename({ 0 : f"{target}"}, axis=1)
        )
        Mdl_Preds.append(
            pd.concat(preds, axis=1).
            median(axis=1).
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
 f'XGB1R'  :  [XGBR(**{ 'device'                : "cuda:0" if CFG.gpu_switch == "ON" else "cpu",
                        'objective'             : "reg:squarederror",
                        "n_estimators"          : 7000 if CFG.test_req == False else CFG.test_iter, 
                        "learning_rate"         : 0.02,
                        'max_depth'             : 6,
                        "colsample_bytree"      : 0.30,
                        "colsample_bylevel"     : 0.35,
                        "random_state"          : CFG.state,
                        "verbosity"             : 0,
                        "reg:lambda"            : 1.25,
                        "early_stopping_rounds" : None,
                        "enable_categorical"    : True,
                      }
                   ),
               {"verbose" : 0},
              ],

 f'CB1R'    : [CBR(**{'task_type'             : "GPU" if CFG.gpu_switch == "ON" else "CPU",
                      'loss_function'         : "RMSE",
                      'eval_metric'           : "MAE",
                      'learning_rate'         : 0.015,
                      'iterations'            : 15_000 if CFG.test_req == False else CFG.test_iter,
                      'max_depth'             : 5,
                      'min_data_in_leaf'      : 12,
                      'colsample_bylevel'     : 0.30 if CFG.gpu_switch == "OFF" else None,
                      'l2_leaf_reg'           : 1.25,
                      'verbose'               : 0,
                      'random_state'          : CFG.state,
                      'early_stopping_rounds' : None,
                    }
                 ),
               {"verbose" : 0 },
              ],

 f'LGBM1R'  : [LGBMR(**{'device'             : "gpu" if CFG.gpu_switch == "ON" else "cpu",
                        'objective'          : "regression",
                        "n_estimators"       : 15_000 if CFG.test_req == False else CFG.test_iter, 
                        "learning_rate"      : 0.015,
                        "colsample_bytree"   : 0.25,               
                        "n_jobs"             : -1,
                        "num_leaves"         : 100,
                        "random_state"       : CFG.state,
                        "subsample"          : 0.25,
                        "verbosity"          : -1,
                      }
                   ),
               {"callbacks" : [log_evaluation(0)]},
              ],

 f'LGBM2R'  : [LGBMR(**{'device'               : "gpu" if CFG.gpu_switch == "ON" else "cpu",
                        'objective'            : "regression",
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

make_offline_models( ["FFV", "Tg"], Xtrain, Ytrain, Mdl_Master, drop_cols)


%%time

Mdl_Master = \
{    
 f'LGBM1R'  : [LGBMR(**{'device'             : "gpu" if CFG.gpu_switch == "ON" else "cpu",
                        'objective'          : "regression",
                        "n_estimators"       : 700 if CFG.test_req == False else CFG.test_iter, 
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
               {"callbacks" : [log_evaluation(0)]},
              ],

 f'CB1R'    : [CBR(**{'task_type'             : "GPU" if CFG.gpu_switch == "ON" else "CPU",
                      'loss_function'         : "RMSE",
                      'eval_metric'           : "MAE",
                      'learning_rate'         : 0.02,
                      'iterations'            : 700 if CFG.test_req == False else CFG.test_iter,
                      'max_depth'             : 4,
                      'min_data_in_leaf'      : 4,
                      'colsample_bylevel'     : 0.30 if CFG.gpu_switch == "OFF" else None,
                      'l2_leaf_reg'           : 1.25,
                      'verbose'               : 0,
                      'random_state'          : CFG.state,
                      'early_stopping_rounds' : None,
                    }
                 ),
               {"verbose" : 0 },
              ], 

 f'XGB1R'  :  [XGBR(**{ 'device'                : "cuda:0" if CFG.gpu_switch == "ON" else "cpu",
                        'objective'             : "reg:squarederror",
                        'max_depth'             : 4,
                        "n_estimators"          : 700 if CFG.test_req == False else CFG.test_iter, 
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
              ]
}

make_offline_models( ["Density", "Tc", "Rg"], Xtrain, Ytrain, Mdl_Master, drop_cols)


%%time 

sub_fl = pd.read_csv(f"{CFG.ip_path}/sample_submission.csv")

oof_preds = pd.concat(OOF_Preds, axis=1).sort_index(ascending = True)
mdl_preds = pd.concat(Mdl_Preds, axis=1).sort_index(ascending = True)

score = \
utils.ScoreMetric(
    Ytrain.reset_index().rename(columns = {"index": "id"}),
    oof_preds[CFG.target].reset_index().rename(columns = {"index": "id"})
)

oof_preds.index.name = "id"
PrintColor(f"\n---> Final CV score after all models = {score:.8f}\n\n")

oof_preds.to_csv(f"OOF_Preds_{CFG.model_label}{CFG.model_id}.csv")

mdl_preds["id"] = sub_fl["id"].values
mdl_preds[["id"] + CFG.target].to_csv(f"submission.csv", index = None)
!head submission.csv

print()

