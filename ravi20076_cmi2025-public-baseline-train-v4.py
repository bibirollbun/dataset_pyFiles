


%%time 

!pip install -q -r /kaggle/input/cmi2025-public-imports-v3/req_kaggle.txt 

exec( open(f"/kaggle/input/cmi2025-public-imports-v3/myimports.py", "r").read() )
exec( open(f"/kaggle/input/cmi2025-public-imports-v3/myutils.py", "r").read() )
exec( open(f"/kaggle/input/cmi2025-public-imports-v3/training.py", "r").read() )

print()


%%time

class CFG:
    """
    Configuration class for parameters and CV strategy for tuning and training
    Some parameters may be unused here as this is a general configuration class
    """;

    # Data preparation:-
    version_nb         = 4
    model_id           = "V4_1"
    model_label        = "ML"
    test_req           = False
    test_iter          = 150
    gpu_switch         = "ON" if torch.cuda.is_available() else "OFF"
    state              = 42
    
    target             = f'gesture'
    grouper            = f"subject"
    tgt_mapper         = {
                            "Above ear - pull hair" : 0,
                            "Cheek - pinch skin" : 1,
                            "Eyebrow - pull hair" : 2,
                            "Eyelash - pull hair" : 3, 
                            "Forehead - pull hairline" : 4,
                            "Forehead - scratch" : 5,
                            "Neck - pinch skin" : 6, 
                            "Neck - scratch" : 7,
                            
                            "Drink from bottle/cup" : 8,
                            "Feel around in tray and pull out an object" : 9,
                            "Glasses on/off" : 10,
                            "Pinch knee/leg skin" : 11, 
                            "Pull air toward your face" : 12,
                            "Scratch knee/leg skin" : 13,
                            "Text on phone" : 14,
                            "Wave hello" : 15,
                            "Write name in air" : 16,
                            "Write name on leg" : 17,
                        }
    
    ip_path            = f"/kaggle/input/cmi-detect-behavior-with-sensor-data"
    op_path            = f"/kaggle/working"
    orig_path          = f""
    data_path          = f""
    dtl_preproc_req    = True
    ftre_plots_req     = True
    ftre_imp_req       = True
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
    nbrnd_erly_stp     = 100
    mdlcv_mthd         = 'SGKF'
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
 "RKF"   : RepeatedKFold(n_splits   = CFG.n_splits, n_repeats= CFG.n_repeats, random_state= CFG.state),
 "RSKF"  : RepeatedStratifiedKFold(n_splits  = CFG.n_splits, n_repeats= CFG.n_repeats, random_state= CFG.state),
 "SKF"   : StratifiedKFold(n_splits   = CFG.n_splits, shuffle = True, random_state= CFG.state),
 "KF"    : KFold(n_splits = CFG.n_splits, shuffle = True, random_state= CFG.state),
 "GKF"   : GroupKFold(n_splits   = CFG.n_splits),
 "SGKF"  : StratifiedGroupKFold(n_splits = CFG.n_splits, shuffle = True, random_state= CFG.state),
}

collect()


%%writefile myfe.py

def make_ftre(sequence : pl.DataFrame, demographics : pl.DataFrame) :
    """
    Makes aggregate columns for the model
    Source - https://www.kaggle.com/code/farisalahmdi/lgbm-inference
    """
    
    agg_exprs = []
    stat_cols = ['acc_x','acc_y','acc_z','rot_w','rot_x','rot_y','rot_z']
    
    for c in stat_cols:
        agg_exprs += [
            pl.col(c).mean().alias(f"{c}_mean"),
            pl.col(c).std().alias(f"{c}_std"),
            pl.col(c).var().alias(f"{c}_var"),
            pl.col(c).quantile(0.25).alias(f"{c}_q25"),
            pl.col(c).median().alias(f"{c}_q50"),
            pl.col(c).quantile(0.75).alias(f"{c}_q75"),
            pl.col(c).max().alias(f"{c}_max"),
            pl.col(c).min().alias(f"{c}_min"),
            pl.col(c).first().alias(f"{c}_first"),
            pl.col(c).last().alias(f"{c}_last"),
            pl.col(c).quantile(0.25, "nearest").alias(f"{c}_t25"),
            pl.col(c).quantile(0.75, "nearest").alias(f"{c}_t75"),
            (pl.col(c).last() - pl.col(c).first()).alias(f"{c}_delta"),
            pl.corr("sequence_counter", c).alias(f"{c}_corr_time"),
            pl.col(c).diff().mean().alias(f"{c}_diff_mean"),
            pl.col(c).diff().std().alias(f"{c}_diff_std"),
            pl.col(c).skew().alias(f"{c}_skew"),
            pl.col(c).kurtosis().alias(f"{c}_kurt"),
            pl.col(c).diff().abs().gt(0).sum().alias(f"{c}_n_changes")
        ]
        
        agg_exprs += [
                pl.when(pl.col("sequence_counter") < 0.1 * pl.max("sequence_counter"))
                  .then(pl.col(c)).otherwise(None).mean().alias(f"{c}_seg1_mean"),
                pl.when(pl.col("sequence_counter") > 0.9 * pl.max("sequence_counter"))
                  .then(pl.col(c)).otherwise(None).mean().alias(f"{c}_seg3_mean"),
        ]
    
    return \
    (
        sequence.
        group_by(pl.col(["sequence_id", "subject"]), maintain_order=True).
        agg(agg_exprs).
        select(pl.all().shrink_dtype()).
        join(
            demographics.select(pl.all().shrink_dtype()),
            how = "left",
            on = ["subject"],
        ).
        drop(["subject"], strict = False).
        to_pandas()
    )


%%time 

exec( open(f"myfe.py", "r").read() )

train    = pl.read_csv(f"{CFG.ip_path}/train.csv")
traind   = pl.read_csv(f"{CFG.ip_path}/train_demographics.csv")
test     = pl.read_csv(f"{CFG.ip_path}/test.csv")
testd    = pl.read_csv(f"{CFG.ip_path}/test_demographics.csv")

sel_cols = [ c for c in test.collect_schema().names() if "thm" not in c and "tof" not in c ]

Xtrain  = make_ftre(train.select(pl.col(sel_cols)), traind)
Xtest   = make_ftre(test.select(pl.col(sel_cols)),  testd)

Ytrain             = train.select(pl.col(["sequence_id", CFG.grouper, CFG.target])).unique().to_pandas()
Ytrain[CFG.target] = Ytrain[CFG.target].map(CFG.tgt_mapper).astype(np.int8)
Ytrain             = Xtrain[["sequence_id"]].merge(Ytrain, how = "inner", on = ["sequence_id"])
ytrain             = Ytrain[CFG.target]

ygrp = np.zeros(len(Xtrain))
cv   = cv_selector[CFG.mdlcv_mthd]

for fold_nb, (train_idx, dev_idx) in tqdm( 
    enumerate(cv.split(Xtrain, Ytrain[CFG.target], Ytrain[CFG.grouper] )) 
) :
    ygrp[dev_idx] = fold_nb

cv = PredefinedSplit(ygrp)
print(
    f"\n\n---> Shape = {Xtrain.shape} {ytrain.shape} {ygrp.shape} {Xtest.shape}"
)

_ = utils.CleanMemory()
print()


%%time 

Mdl_Master = \
{ 
 f'LGBM1C'  : [LGBMC(**{'device'             : "gpu" if CFG.gpu_switch == "ON" else "cpu",
                        'objective'          : "multiclass",
                        "n_estimators"       : 10000 if CFG.test_req == False else CFG.test_iter, 
                        "max_depth"          : 8,
                        "learning_rate"      : 0.025,
                        "colsample_bytree"   : 0.55,               
                        "n_jobs"             : -1,
                        "num_leaves"         : 75,
                        "random_state"       : CFG.state,
                        "reg_alpha"          : 0.001,
                        "reg_lambda"         : 0.001,
                        "subsample"          : 0.40,
                        "verbosity"          : -1,
                      }
                   ),
               {"callbacks" : [log_evaluation(0), early_stopping(CFG.nbrnd_erly_stp, verbose = False)]},
              ],

 f'LGBM2C'  : [LGBMC(**{"data_sample_strategy"   : "goss",
                        "n_estimators"           : 10000 if CFG.test_req == False else CFG.test_iter, 
                        "max_depth"              : 7, 
                        'device'                 : "gpu" if CFG.gpu_switch == "ON" else "cpu",
                        "learning_rate"          : 0.025,
                        "colsample_bytree"       : 0.55,
                        "min_child_samples"      : 32,
                        "n_jobs"                 : -1,
                        "num_leaves"             : 60,
                        "random_state"           : CFG.state,
                        "reg_alpha"              : 0.001,
                        "reg_lambda"             : 0.001,
                        "subsample"              : 0.21,
                        "verbosity"              : -1,
                     }
                   ),
               {"callbacks" : [log_evaluation(0), early_stopping(CFG.nbrnd_erly_stp, verbose = False)]},
              ],

 f'CB1C'    : [CBC(**{'task_type'             : "GPU" if CFG.gpu_switch == "ON" else "CPU",
                      'loss_function'         : "MultiClass",
                      'learning_rate'         : 0.04,
                      'iterations'            : 10000 if CFG.test_req == False else CFG.test_iter,
                      'max_depth'             : 8,
                      'min_data_in_leaf'      : 12,
                      'colsample_bylevel'     : 0.30 if CFG.gpu_switch == "OFF" else None,
                      'l2_leaf_reg'           : 0.35,
                      'leaf_estimation_method': "Newton",
                      'verbose'               : 0,
                      'random_state'          : CFG.state,
                      'early_stopping_rounds' : CFG.nbrnd_erly_stp if CFG.nbrnd_erly_stp > 0 else None,
                    }
                 ),
               {"verbose" : 0 },
              ],

 f'XGB1C'  : [  XGBC(**{'device'                : "cuda:0" if CFG.gpu_switch == "ON" else "cpu",
                        'objective'             : "multi:softprob",
                        "n_estimators"          : 10000 if CFG.test_req == False else CFG.test_iter, 
                        "max_depth"             : 8,
                        "learning_rate"         : 0.025,
                        "colsample_bytree"      : 0.55,               
                        "n_jobs"                : -1,
                        "random_state"          : CFG.state,
                        "reg_alpha"             : 0.001,
                        "reg_lambda"            : 0.001,
                        "verbosity"             : 0,
                        "enable_categorical"    : True,
                        'early_stopping_rounds' : CFG.nbrnd_erly_stp if CFG.nbrnd_erly_stp > 0 else None,
                      }
                   ),
               {"verbose" : 0},
              ],
}

# Initializing model outputs
OOF_Preds    = []
FittedModels = {}


%%time 

drop_cols = \
["Source", "id", "Id", "Label", CFG.target, "fold_nb", "sequence_id", CFG.grouper]

for method, (mymodel, fit_params) in tqdm(Mdl_Master.items()):

    md = \
    ModelTrainer(
        problem_type     = "multiclass",
        drop_cols        = drop_cols, 
        len_train        = Xtrain.shape[0],
        test_preds_req   = CFG.test_preds_req, 
        bin_cutoff       = 0.50,
    )

    (fitted_models, oof_preds, test_preds) =  \
    md.fit_predict(
        Xtrain,
        ytrain,
        Xtest,
        ygrp,
        method,
        mymodel,
        cat_cols = None,
        **fit_params, 
    )

    OOF_Preds.append(oof_preds)
    FittedModels[method] = fitted_models

    del fitted_models, oof_preds, test_preds
    print()
    collect()

_ = utils.CleanMemory()


%%time 

score, bscore, mscore = \
utils.ScoreMetric(
    ytrain.values.flatten(),
    pd.concat(
        OOF_Preds, axis=0, ignore_index = False
    ).
    groupby(level = 0).
    mean().
    idxmax(axis=1).
    to_numpy().
    flatten()
)

PrintColor(
    f"\n---> Ensemble Score = {score:,.8f} | Binary = {bscore:,.8f} | Multiclass = {mscore:,.8f}\n"
)


joblib.dump(
    FittedModels, 
    f"{CFG.op_path}/FittedModels{CFG.model_label}{CFG.model_id}.joblib"
)

