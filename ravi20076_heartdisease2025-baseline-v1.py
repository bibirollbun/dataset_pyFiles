


!uv pip install -q --system -r /kaggle/input/heartdisease2025-public-imports/req_kaggle.txt

exec( open(f"/kaggle/input/heartdisease2025-public-imports/myimports.py", "r").read() )
exec( open(f"/kaggle/input/heartdisease2025-public-imports/myutils.py", "r").read() )
exec( open(f"/kaggle/input/heartdisease2025-public-imports/training.py", "r").read() )
exec( open(f"/kaggle/input/heartdisease2025-public-imports/mypp.py", "r").read() )

print()


%%time

class CFG:
    """
    Configuration class for parameters and CV strategy for tuning and training
    Some parameters may be unused here as this is a general configuration class
    """;

    # Data preparation:-
    version_nb         = 1
    model_id           = "V1_1"
    model_label        = "ML"
    test_req           = False
    test_iter          = 50
    gpu_switch         = "ON" if torch.cuda.is_available() else "OFF"
    state              = 42
    target             = f"HeartDisease"
    grouper            = f""
    tgt_mapper         = {}
    ip_path            = f"/kaggle/input/heart-disease-prediction-dataquest"
    op_path            = f"/kaggle/working"
    orig_path          = f""
    data_path          = f""
    dtl_preproc_req    = True
    ftre_plots_req     = True
    ftre_imp_req       = True
    nb_orig            = 0
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
 "RKF"   : RKF(n_splits   = CFG.n_splits, n_repeats= CFG.n_repeats, random_state= CFG.state),
 "RSKF"  : RSKF(n_splits  = CFG.n_splits, n_repeats= CFG.n_repeats, random_state= CFG.state),
 "SKF"   : SKF(n_splits   = CFG.n_splits, shuffle = True, random_state= CFG.state),
 "KF"    : KFold(n_splits = CFG.n_splits, shuffle = True, random_state= CFG.state),
 "GKF"   : GKF(n_splits   = CFG.n_splits)
}

collect()


%%time 

train    = pd.read_csv(f"{CFG.ip_path}/heart_train.csv")
test     = pd.read_csv(f"{CFG.ip_path}/heart_test.csv")
sub_fl   = pd.read_csv(f"{CFG.ip_path}/sample_submission.csv", index_col = "id")
cat_cols = list( train.select_dtypes(["string", "object", "category"]).columns )

PrintColor(f"\n---> Shape = {train.shape} {test.shape} {sub_fl.shape} \n")





PrintColor(f"\n---> Train nunique\n")
display(train.nunique())

PrintColor(f"\n---> Test nunique\n")
display(test.nunique())

PrintColor(f"\n---> Target distribution\n")
display( train[CFG.target].value_counts(normalize = True) )


%%time 

Xtrain = train.copy()
Xtest  = test.copy()

Xtrain["Source"], Xtest["Source"] = ("Competition", "Competition")
Xtrain[cat_cols] = Xtrain[cat_cols].astype("string").fillna("missing")
Xtest[cat_cols]  = Xtest[cat_cols].astype("string").fillna("missing")

ytrain = Xtrain[CFG.target].astype(np.uint8)
Xtrain = Xtrain.drop(CFG.target, axis=1)

PrintColor(f"\n---> Shape = {Xtrain.shape} {ytrain.shape} {Xtest.shape} {sub_fl.shape} \n")


%%time 

cv    = cv_selector[CFG.mdlcv_mthd]
folds = np.zeros( len(Xtrain) )

for fold_nb, (_, dev_idx) in enumerate(cv.split(Xtrain, ytrain)):
    folds[dev_idx] = fold_nb 
ygrp = pd.Series(folds, name = "fold_nb", dtype = np.uint8)


Mdl_Master = \
{   
 f'CB1C'    : CBC(**{"loss_function"         : "Logloss",
                     "eval_metric"           : "AUC",
                     'task_type'             : "GPU" if CFG.gpu_switch == "ON" else "CPU",
                     'learning_rate'         : 0.02,
                     'iterations'            : 200,
                     'max_depth'             : 3,
                     'min_data_in_leaf'      : 3 ,
                     'colsample_bylevel'     : 0.60,
                     'l2_leaf_reg'           : 0.10,
                     'random_strength'       : 0.025,
                     'leaf_estimation_method': "Newton",
                     'od_wait'               : 5, 
                     'verbose'               : 0,
                     'random_state'          : CFG.state,
                    }
                 ),
    
 f'XGB1C'  : XGBC(**{  "objective"            : "binary:logistic",
                       "eval_metric"          : "auc",
                       'device'               : "cpu",
                       'learning_rate'        : 0.02,
                       'n_estimators'         : 180,
                       'max_depth'            : 3,
                       'colsample_bytree'     : 0.60,
                       'colsample_bynode'     : 0.65,
                       'subsample'            : 0.65,
                       'reg_lambda'           : 0.001,
                       'reg_alpha'            : 0.001,
                       'verbosity'            : 0,
                       'random_state'         : CFG.state,
                       'early_stopping_rounds': None,
                      } 
                   ),

 f'LGBM1C' : LGBMC(**{ "objective"            : "binary",
                       "eval_metric"          : "auc",
                       'device'               : "cpu",
                       'learning_rate'        : 0.02,
                       'n_estimators'         : 180,
                       'max_depth'            : 4,
                       'subsample'            : 0.60,
                       'colsample_bytree'     : 0.55,
                       'reg_lambda'           : 0.001,
                       'reg_alpha'            : 0.001,
                       'verbosity'            : -1,
                       'random_state'         : CFG.state,
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
        problem_type   = "binary",
        es             = CFG.nbrnd_erly_stp,
        target         = CFG.target,
        orig_req       = False,
        orig_all_folds = CFG.orig_all_folds,
        metric_lbl     = "auc",
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
method    = "L21C"
model     = LRC(max_iter = 10000, random_state = CFG.state, C = 0.05)

md = \
ModelTrainer(
    problem_type   = "binary",
    es             = CFG.nbrnd_erly_stp,
    target         = CFG.target,
    orig_req       = False,
    orig_all_folds = CFG.orig_all_folds,
    metric_lbl     = "auc",
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

score = utils.ScoreMetric(ytrain.iloc[0 : len_train], oof_ens_preds)
PrintColor(f"\n\n---> Overall score = {score:,.8f}")


sub_fl["target"] = np.where(test_preds >= 0.56, 1, 0)
sub_fl.to_csv("submission.csv", index = True)
print()
!ls
print()
!head submission.csv

_ = utils.CleanMemory()
print()


