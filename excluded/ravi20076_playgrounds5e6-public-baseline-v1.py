


!uv pip install -q --system -r /kaggle/input/playgrounds5e6-public-imports-v1/req_kaggle.txt

season  = 5
episode = 6

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
    test_iter          = 200
    gpu_switch         = "ON" if torch.cuda.is_available() else "OFF"
    state              = 42
    target             = f"FertilizerName"
    grouper            = f""
    
    tgt_mapper         = {'14-35-14': 0,
                          '10-26-26': 1,
                          '17-17-17': 2,
                          '28-28': 3,
                          '20-20': 4,
                          'DAP': 5,
                          'Urea': 6
                         }
    
    ip_path            = f"/kaggle/input/playground-series-s5e6"
    op_path            = f"/kaggle/working"
    orig_path          = f"/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv"
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
    nbrnd_erly_stp     = 100
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

if CFG.dtl_preproc_req == True and torch.cuda.is_available() == False :
    advcv = AdversarialCVMaker()
    advcv.make_cv(pp.train[pp.test.columns], pp.test)


%%time 

Xtrain    = reduce_mem_usage( pp.train.drop(CFG.target, axis=1), "Train")
Xtest     = reduce_mem_usage( pp.test, "Test")
strt_ftre = pp.strt_ftre[0:-1]
ytrain    = pp.train[CFG.target]

PrintColor(f"---> Shapes = {Xtrain.shape} {Xtest.shape}")

cat_cols = deepcopy(pp.cat_cols)

with np.printoptions(linewidth = 100, threshold = 1000):
    PrintColor(f"\n\n---> Category columns")
    pprint(np.array(cat_cols))

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
 f'XGB1C'  : [XGBC(**{ "objective"            : "multi:softprob",
                       "eval_metric"          : "mlogloss",
                       'device'               : "cuda" if CFG.gpu_switch == "ON" else "cpu",
                       'learning_rate'        : 0.03,
                       'n_estimators'         : 10_000 if CFG.test_req == False else 75,
                       'max_depth'            : 7,
                       'subsample'            : 0.80,
                       'colsample_bytree'     : 0.60,
                       'colsample_bynode'     : 0.65,
                       'colsample_bylevel'    : 0.825,
                       'verbosity'            : 0,
                       'random_state'         : CFG.state,
                       'early_stopping_rounds': None if CFG.nbrnd_erly_stp == 0 else CFG.nbrnd_erly_stp,
                       'enable_categorical'   : True,
                      } 
                   ),
              {"verbose" : 0}
             ]
}

# Initializing model outputs
Mdl_Preds    = {}


%%time 

drop_cols = ["Source", "id", "Id", "Label", CFG.target, "fold_nb"]
ytrain_   = ytrain.map(CFG.tgt_mapper).astype(np.uint8)

for method, (mymodel, fit_params) in tqdm(Mdl_Master.items()) :
    md = ModelTrainer(drop_cols = drop_cols)

    _, oof_preds, mdl_preds = \
    md.fit_predict(
        Xtrain.copy(), 
        ytrain_,
        Xtest.copy(), 
        ygrp,
        method   = method,
        mymodel  = mymodel,
        cat_cols = cat_cols,
        **fit_params,
    )   
    
    Mdl_Preds[method] = mdl_preds
    collect();

_ = utils.CleanMemory()
print()


%%time 

mdl_preds_ = pd.DataFrame(np.argsort(mdl_preds, axis=1)[:, -3:][:, ::-1])
for col in mdl_preds_.columns:
    mdl_preds_[col] = mdl_preds_[col].map({v:k for k, v in CFG.tgt_mapper.items()})

pp.sub_fl["Fertilizer Name"] = mdl_preds_.apply(lambda x: " ".join(x), axis=1).values.flatten()
pp.sub_fl.to_csv("submission.csv", index = None)
del mdl_preds_

!ls submission.csv
!head submission.csv

