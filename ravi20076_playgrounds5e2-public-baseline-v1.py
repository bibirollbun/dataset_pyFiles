


%%time 

!pip install -q -r /kaggle/input/playgrounds5e2-public-imports-v1/req_kaggle.txt

exec(open('/kaggle/input/playgrounds5e2-public-imports-v1/myimports.py','r').read())
exec(open('/kaggle/input/playgrounds5e2-public-imports-v1/myutils.py','r').read())
exec(open('/kaggle/input/playgrounds5e2-public-imports-v1/mytrainer.py','r').read())
exec(open('/kaggle/input/playgrounds5e2-public-imports-v1/myensembler.py','r').read())
exec(open('/kaggle/input/playgrounds5e2-public-imports-v1/mypp.py','r').read())

%matplotlib inline
print()


%%time 

class CFG:
    """
    Configuration class for parameters and CV strategy for tuning and training
    Some parameters may be unused here as this is a general configuration class
    """

    # Data preparation:-
    version_nb  = 1
    model_id    = "V1_4"
    model_label = "ML"

    test_req           = False
    fulltrain_req      = True
    test_sample_frac   = 0.01

    gpu_switch         = "OFF"
    state              = 42
    target             = f"Price"
    grouper            = f""
    tgt_mapper         = {}

    ip_path            = f"/kaggle/input/playground-series-s5e2"
    op_path            = f"/kaggle/working"
    orig_path          = f"/kaggle/input/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv"

    dtl_preproc_req    = True
    ftre_plots_req     = False
    ftre_imp_req       = True

    nb_orig            = 1
    orig_all_folds     = False

    # Model Training:-
    pstprcs_oof        = True
    pstprcs_train      = True
    pstprcs_test       = True
    
    ML                 = True
    test_preds_req     = True

    pseudo_lbl_req     = "N"
    pseudolbl_up       = 0.975
    pseudolbl_low      = 0.00

    n_splits           = 5
    n_repeats          = 1
    nbrnd_erly_stp     = 100
    mdlcv_mthd         = 'KF'

    # Ensemble:-
    ensemble_req       = False

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

PrintColor(f"\n---> Configuration done!\n")

cv_selector = \
{
 "RKF"   : RKF(n_splits = CFG.n_splits, n_repeats= CFG.n_repeats, random_state= CFG.state),
 "RSKF"  : RSKF(n_splits = CFG.n_splits, n_repeats= CFG.n_repeats, random_state= CFG.state),
 "SKF"   : SKF(n_splits = CFG.n_splits, shuffle = True, random_state= CFG.state),
 "KF"    : KFold(n_splits = CFG.n_splits, shuffle = True, random_state= CFG.state),
 "GKF"   : GKF(n_splits = CFG.n_splits)
}

collect()


%%time 

pp = Preprocessor();
pp.DoPreprocessing();


%%time 

plotter = \
FeaturePlotter(
    target         = CFG.target,
    ftre_plots_req = CFG.ftre_plots_req,
    title_specs    = CFG.title_specs,
    grid_specs     = CFG.grid_specs,
)

plotter.MakeCatFtrePlots(
    pp.cat_cols, pp.train.copy(), pp.test.copy(), pp.original.copy()
)


%%time 

plotter.MakeContColPlots(
    pp.cont_cols, pp.train, pp.test, pp.original
)


%%time 

Xtrain = pp.train.copy()

if CFG.test_req:
    Xtrain = Xtrain.groupby(["Source"], as_index = False).sample(frac = CFG.test_sample_frac)
    
    Xtrain.index = range(len(Xtrain))
    PrintColor(
        f"---> Syntax check mode - shape = {Xtrain.shape}", color = Fore.RED
    )
else:
    pass
    
ytrain = Xtrain[CFG.target]     
Xtrain = Xtrain.drop(CFG.target, axis=1)  
Xtest  = pp.test.copy()

Xtrain['Compartments'] = Xtrain["Compartments"].fillna(-1).astype(np.int16)
Xtest['Compartments']  = Xtest["Compartments"].fillna(-1).astype(np.int16)

cat_cols = pp.cat_cols + ["Compartments"]

Xtrain[cat_cols] = Xtrain[cat_cols].astype("string").fillna("missing")
Xtest[cat_cols]  = Xtest[cat_cols].astype("string").fillna("missing")

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

_ = utils.CleanMemory()


%%time 

Mdl_Master = \
{   
 f'CBMAE1R' : CBR(**{"loss_function"         : "MAE",
                     "eval_metric"           : "MAE",
                     'task_type'             : "GPU" if CFG.gpu_switch == "ON" else "CPU",
                     'learning_rate'         : 0.05,
                     'iterations'            : 5_000 if CFG.test_req == False else 450,
                     'max_depth'             : 8,
                     'colsample_bylevel'     : 0.70 if CFG.gpu_switch == "OFF" else None,
                     'l2_leaf_reg'           : 0.25,
                     'random_strength'       : 0.20,
                     'verbose'               : 0,
                     'random_state'          : CFG.state,
                     'cat_features'          : cat_cols,
                    }
                 ),
}

# Initializing model outputs
OOF_Preds    = {}
Mdl_Preds    = {}
FtreImp      = {}


%%time

# Model training:-
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
        metric_lbl     = "mae",
        drop_cols      = drop_cols,
        pp_preds       = CFG.pstprcs_oof,
        )

    sel_mdl_cols = list(Xtest.columns) 
    PrintColor(
        f"Selected columns = {len(sel_mdl_cols) :,.0f}", 
        color = Fore.RED
    )

    Xtrain_ = Xtrain.copy()
    Xtest_  = Xtest.copy()

    if "CB" not in method :
        Xtrain_[cat_cols] = Xtrain_[cat_cols].astype("category")
        Xtest_[cat_cols]  = Xtest_[cat_cols].astype("category")
    else:
        pass

    fitted_models, oof_preds, test_preds, ftreimp, mdl_best_iter =  \
    md.MakeOfflineModel(
        Xtrain_,
        ytrain,
        ygrp,
        Xtest_,
        clone(mymodel),
        method,
        test_preds_req   = True,
        ftreimp_plot_req = CFG.ftre_imp_req,
        ntop = 50,
    )

    OOF_Preds[method]    = oof_preds
    Mdl_Preds[method]    = test_preds
    FtreImp[method]      = ftreimp

    del fitted_models, oof_preds, test_preds, ftreimp, sel_mdl_cols, Xtrain_, Xtest_
    print()
    collect()


# Augmenting the train-test data with the OOF predictions from MAE model
Xtrain = Xtrain.assign(**OOF_Preds)
Xtest  = Xtest.assign(**Mdl_Preds)

_ = utils.CleanMemory()


%%time 

Mdl_Master = \
{   
 f'CBMSE1R' : CBR(**{"loss_function"         : "RMSE",
                     "eval_metric"           : "RMSE",
                     'task_type'             : "GPU" if CFG.gpu_switch == "ON" else "CPU",
                     'learning_rate'         : 0.04,
                     'iterations'            : 5_000 if CFG.test_req == False else 300,
                     'max_depth'             : 8,
                     'colsample_bylevel'     : 0.70 if CFG.gpu_switch == "OFF" else None,
                     'l2_leaf_reg'           : 0.25,
                     'random_strength'       : 0.20,
                     'verbose'               : 0,
                     'random_state'          : CFG.state,
                     'cat_features'          : cat_cols,
                    }
                 ),
}

# Initializing model outputs
OOF_Preds    = {}
Mdl_Preds    = {}
FtreImp      = {}


%%time

# Model training:-
drop_cols = ["Source", "id", "Id", "Label", CFG.target, "fold_nb"]

for method, mymodel in tqdm(Mdl_Master.items()):

    PrintColor(f"\n{'=' * 20} {method.upper()} MODEL TRAINING {'=' * 20}\n")

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

    Xtrain_ = Xtrain.copy()
    Xtest_  = Xtest.copy()

    if "CB" not in method :
        Xtrain_[cat_cols] = Xtrain_[cat_cols].astype("category")
        Xtest_[cat_cols]  = Xtest_[cat_cols].astype("category")
    else:
        pass

    fitted_models, oof_preds, test_preds, ftreimp, mdl_best_iter =  \
    md.MakeOfflineModel(
        Xtrain_,
        ytrain,
        ygrp,
        Xtest_,
        clone(mymodel),
        method,
        test_preds_req   = True,
        ftreimp_plot_req = CFG.ftre_imp_req,
        ntop = 50,
    )

    OOF_Preds[method]    = oof_preds
    Mdl_Preds[method]    = test_preds
    FtreImp[method]      = ftreimp

    del fitted_models, oof_preds, test_preds, ftreimp, sel_mdl_cols, Xtrain_, Xtest_
    print()
    collect()

_ = utils.CleanMemory()


%%time 

oof_preds = \
pd.DataFrame(OOF_Preds).assign(**{CFG.target : ytrain.values.flatten()})

with sns.axes_style("white") : 
    fig, axes = \
    plt.subplots(
        len(Mdl_Master.keys()) ,1, 
        figsize = (20, 6 * len(Mdl_Master.keys()),),
        gridspec_kw = {"hspace" : 0.35},
        sharex = True,
    )
    
    for i, method in tqdm(enumerate(Mdl_Master.keys() ) ):
        if len(Mdl_Master.keys()) == 1:
            ax = axes
        else :
            ax = axes[i] 
        
        sns.scatterplot(
            data = oof_preds,
            y = method,
            x = CFG.target,
            color = "tab:blue",
            ax = ax
        )
    
        r2 = r2_score(oof_preds[CFG.target], oof_preds[method])
    
        ax.set_title(f"{method} R2 score= {r2: ,.8f} ", **CFG.title_specs)
        ax.set_xticks(range(15, 151, 5), labels = range(15, 151, 5), fontsize = 9)

    plt.tight_layout()
    plt.show()



%%time

try:
    oof_preds.assign(**{"Ensemble": oof_ens_preds}).\
    to_parquet(
        os.path.join(CFG.op_path, f"OOF_Preds_{CFG.model_label}{CFG.model_id}.parquet")
    )

    mdl_preds.assign(**{"Ensemble": test_preds}).\
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

pp.sub_fl.to_csv(
    os.path.join(CFG.op_path, f"submission.csv"), index = None
)


print()
!ls
print()
!head submission.csv

_ = utils.CleanMemory()
print()

