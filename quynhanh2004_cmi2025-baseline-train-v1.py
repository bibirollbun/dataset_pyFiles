


!uv pip install -q --system -r /kaggle/input/cmi2025-public-imports-v1/req_kaggle.txt

exec( open(f"/kaggle/input/cmi2025-public-imports-v1/myimports.py", "r").read() )
exec( open(f"/kaggle/input/cmi2025-public-imports-v1/myutils.py", "r").read() )
exec( open(f"/kaggle/input/cmi2025-public-imports-v1/training.py", "r").read() )

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
    test_iter          = 200
    gpu_switch         = "ON" if torch.cuda.is_available() else "OFF"
    state              = 42
    target             = f'gesture'
    grouper            = f""
    tgt_mapper         = {'Neck - pinch skin'        : 0, 
                          'Cheek - pinch skin'       : 1, 
                          'Above ear - pull hair'    : 2,
                          'Eyelash - pull hair'      : 3, 
                          'Eyebrow - pull hair'      : 4,
                          'Forehead - pull hairline' : 5, 
                          'Forehead - scratch'       : 6, 
                          'Neck - scratch'           : 7,
                         }
    ip_path            = f"/kaggle/input/cmi2025-data-v1"
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
 "RKF"   : RepeatedKFold(n_splits   = CFG.n_splits, n_repeats= CFG.n_repeats, random_state= CFG.state),
 "RSKF"  : RepeatedStratifiedKFold(n_splits  = CFG.n_splits, n_repeats= CFG.n_repeats, random_state= CFG.state),
 "SKF"   : StratifiedKFold(n_splits   = CFG.n_splits, shuffle = True, random_state= CFG.state),
 "KF"    : KFold(n_splits = CFG.n_splits, shuffle = True, random_state= CFG.state),
 "GKF"   : GroupKFold(n_splits   = CFG.n_splits)
}

collect()


%%time

test = \
pl.scan_parquet( os.path.join(CFG.ip_path, "test.parquet" )).collect(engine = "streaming").to_pandas()

train = \
(
    pl.scan_parquet( os.path.join(CFG.ip_path, "train.parquet" )).
    filter(pl.col("sequence_type").eq("Target")).
    collect(engine = "streaming").
    to_pandas()
)

Xtrain  = train[test.columns]
Xtest   = test.copy()
PrintColor(f"---> Shapes = {Xtrain.shape} {Xtest.shape}")


%%writefile fe.py

class FeatureMaker:
    def __init__(self, fill_miss_val : bool = True):
        self.fill_miss_val = fill_miss_val
        
    def make_ftre(self, df):
        """
        Create comprehensive features from sensor data
        Source - https://www.kaggle.com/code/mohanapavanbezawada/detect-behavior-with-sensor-data
        """
        
        sensor_cols     = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
        all_sensor_cols = [col for col in df.columns if any(s in col for s in ['acc_', 'rot_', 'thm_', 'tof_'])]
        features_df     = df[sensor_cols].copy()
    
        if self.fill_miss_val:
            features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(-1)
        
        seq_features = []
        for seq_id in tqdm( df['sequence_id'].unique() ):
            seq_data = df[df['sequence_id'] == seq_id][sensor_cols]
           
            seq_feat = {}
            
            seq_feat.update({f'{col}_mean': seq_data[col].mean() for col in sensor_cols})
            seq_feat.update({f'{col}_std': seq_data[col].std() for col in sensor_cols})
            seq_feat.update({f'{col}_min': seq_data[col].min() for col in sensor_cols})
            seq_feat.update({f'{col}_max': seq_data[col].max() for col in sensor_cols})
            seq_feat.update({f'{col}_median': seq_data[col].median() for col in sensor_cols})
            seq_feat.update({f'{col}_q25': seq_data[col].quantile(0.25) for col in sensor_cols})
            seq_feat.update({f'{col}_q75': seq_data[col].quantile(0.75) for col in sensor_cols})
            seq_feat.update({f'{col}_skew': seq_data[col].skew() for col in sensor_cols})
            seq_feat.update({f'{col}_kurt': seq_data[col].kurtosis() for col in sensor_cols})
            
            seq_data_np = seq_data[['acc_x', 'acc_y', 'acc_z']].values
            magnitude = np.sqrt(np.sum(seq_data_np**2, axis=1))
            seq_feat['acc_magnitude_mean'] = np.mean(magnitude)
            seq_feat['acc_magnitude_std'] = np.std(magnitude)
            seq_feat['acc_magnitude_max'] = np.max(magnitude)
            
            rot_data = seq_data[['rot_x', 'rot_y', 'rot_z']].values
            rot_magnitude = np.sqrt(np.sum(rot_data**2, axis=1))
            seq_feat['rot_magnitude_mean'] = np.mean(rot_magnitude)
            seq_feat['rot_magnitude_std'] = np.std(rot_magnitude)
            
            for col in sensor_cols:
                values = seq_data[col].values
                if len(values) > 1:
                    diff = np.diff(values)
                    seq_feat[f'{col}_diff_mean'] = np.mean(diff)
                    seq_feat[f'{col}_diff_std'] = np.std(diff)
                    seq_feat[f'{col}_diff_max'] = np.max(np.abs(diff))
            
            for col in sensor_cols:
                values = seq_data[col].values - seq_data[col].mean()
                values = values[~np.isnan(values)] 
                if len(values) > 1:
                    zero_crossings = np.sum(np.diff(np.sign(values)) != 0)
                    seq_feat[f'{col}_zero_crossings'] = zero_crossings / len(values)
                else:
                    seq_feat[f'{col}_zero_crossings'] = 0
            
            for col in sensor_cols:
                seq_feat[f'{col}_energy'] = np.sum(seq_data[col].values**2)
                seq_feat[f'{col}_rms'] = np.sqrt(np.mean(seq_data[col].values**2))
            
            seq_feat['sequence_length'] = len(seq_data)
            seq_feat['sequence_id']     = seq_id
            seq_features.append(seq_feat)
        
        return pd.DataFrame(seq_features)



%%time 

exec( open( "fe.py", "r").read())
fe     = FeatureMaker(True)
Xtrain = fe.make_ftre(Xtrain)
Xtest  = fe.make_ftre(Xtest)

Xtrain["Source"], Xtest["Source"] = ("Competition", "Competition")

print(f"---> Shapes  = {Xtrain.shape} {Xtest.shape}")
print()
_ = utils.CleanMemory()


%%time 

ytrain = train[['sequence_id', CFG.target]].drop_duplicates()
ytrain[CFG.target] = ytrain[CFG.target].map(CFG.tgt_mapper).astype(np.uint8)
Xtrain = Xtrain.merge(ytrain, on = "sequence_id", how = "left")

ytrain = Xtrain[CFG.target]
del Xtrain[CFG.target]

cv   = cv_selector[CFG.mdlcv_mthd]
ygrp = np.zeros(len(Xtrain))

for fold_nb, (train_idx, dev_idx) in enumerate( cv.split(Xtrain, ytrain) ):
    ygrp[dev_idx] = fold_nb
ygrp = pd.Series( ygrp , name = "fold_nb", dtype = np.uint8 )

PrintColor(f"---> Shapes  = {Xtrain.shape} {ytrain.shape} {ygrp.shape} {Xtest.shape}")
_ = utils.CleanMemory()


%%time 

Mdl_Master = \
{ 
 f"XGB1C"   : XGBC(**{  "device"                 : "cuda" if CFG.gpu_switch == "ON" else "cpu", 
                        "n_estimators"           : 250,
                        "learning_rate"          : 0.02,
                        "colsample_bylevel"      : 0.35,
                        "gamma"                  : 0.75,
                        "max_depth"              : 4,
                        "n_jobs"                 : -1,
                        "random_state"           : CFG.state,
                        "reg_alpha"              : 0.01,
                        "reg_lambda"             : 0.10,
                        "subsample"              : 0.25,
                        "verbosity"              : 0,
                    }
                  ),
    
 f'LGBM1C'  : LGBMC(**{ 'device'             : "gpu" if CFG.gpu_switch == "ON" else "cpu",
                        "n_estimators"       : 300, 
                        "max_depth"          : 4,
                        "learning_rate"      : 0.02,
                        "colsample_bytree"   : 0.35,               
                        "n_jobs"             : -1,
                        "num_leaves"         : 45,
                        "random_state"       : CFG.state,
                        "reg_alpha"          : 0.01,
                        "reg_lambda"         : 0.001,
                        "subsample"          : 0.35,
                        "verbosity"          : -1
                      }
                   ),

 f'LGBM2C'  : LGBMC(**{ "data_sample_strategy"   : "goss",
                        "n_estimators"           : 300,
                        "max_depth"              : 4, 
                        'device'                 : "gpu" if CFG.gpu_switch == "ON" else "cpu",
                        "learning_rate"          : 0.015,
                        "colsample_bytree"       : 0.35,
                        "min_child_samples"      : 25,
                        "n_jobs"                 : -1,
                        "num_leaves"             : 30,
                        "random_state"           : CFG.state,
                        "reg_alpha"              : 0.001,
                        "reg_lambda"             : 0.001,
                        "subsample"              : 0.21,
                        "verbosity"              : -1,
                     }
                   ),

 f'CB1C'    : CBC(**{'task_type'             : "GPU" if CFG.gpu_switch == "ON" else "CPU",
                     'learning_rate'         : 0.02,
                     'iterations'            : 250,
                     'max_depth'             : 4,
                     'min_data_in_leaf'      : 8 ,
                     'colsample_bylevel'     : 0.30 if CFG.gpu_switch == "OFF" else None,
                     'l2_leaf_reg'           : 1.05,
                     'leaf_estimation_method': "Newton",
                     'verbose'               : 0,
                     'random_state'          : CFG.state,
                    }
                 ),

 f"RF1C"     : RFC(n_estimators = 300,
                   max_depth    = 5,
                   random_state = CFG.state,
                   n_jobs       = -1,
                  ),

 f"HGB1C"    : HGBC( learning_rate = 0.015,
                     max_iter      = 350,
                     max_depth     = 4,
                     verbose       = 0,
                     random_state  = CFG.state,
                   ),
}

# Initializing model outputs
OOF_Preds    = {}
Mdl_Preds    = {}
FtreImp      = {}
FittedModels = {}


%%time

drop_cols = ["Source", "id", "Id", "Label", CFG.target, "fold_nb", "sequence_id"]

for method, mymodel in tqdm(Mdl_Master.items()):

    PrintColor(
        f"\n{'=' * 20} {method.upper()} MODEL TRAINING {'=' * 20}\n"
    )

    md = \
    ModelTrainer(
        problem_type   = "multiclass",
        es             = CFG.nbrnd_erly_stp,
        target         = CFG.target,
        orig_req       = False,
        orig_all_folds = CFG.orig_all_folds,
        metric_lbl     = "auc_multi",
        drop_cols      = drop_cols,
        pp_preds       = CFG.pstprcs_oof,
    )

    fitted_models, oof_preds, test_preds, ftreimp, _ =  \
    md.MakeOfflineModel(
        Xtrain,
        ytrain,
        ygrp,
        Xtest,
        mymodel,
        method,
        test_preds_req   = True,
        ftreimp_plot_req = CFG.ftre_plots_req,
        ntop = 50,
    )

    OOF_Preds[method]    = oof_preds
    Mdl_Preds[method]    = test_preds
    FtreImp[method]      = ftreimp
    FittedModels[method] = fitted_models

    del fitted_models, oof_preds, test_preds, ftreimp
    print()
    collect()


_ = utils.CleanMemory()


%%time 

oof_preds = pd.DataFrame(index = range(len(Xtrain)))
for method, preds in OOF_Preds.items():
    oof_preds = pd.concat([oof_preds, pd.DataFrame(preds).add_prefix(method)], axis=1)

mdl_preds = pd.DataFrame(index = range(len(Xtest)))
for method, preds in Mdl_Preds.items():
    mdl_preds = pd.concat([mdl_preds, pd.DataFrame(preds).add_prefix(method)], axis=1)

method = "L21C"
model  = LRC(C = 0.05, random_state = CFG.state, max_iter = 10000)

md = \
ModelTrainer(
    problem_type   = "multiclass",
    es             = CFG.nbrnd_erly_stp,
    target         = CFG.target,
    orig_req       = False,
    orig_all_folds = CFG.orig_all_folds,
    metric_lbl     = "auc_multi",
    drop_cols      = drop_cols,
    pp_preds       = CFG.pstprcs_oof,
)

ens_models, oof_ens_preds, test_ens_preds, _, _ =  \
md.MakeOfflineModel(
    oof_preds.assign(Source = "Competition"),
    ytrain,
    ygrp,
    mdl_preds.assign(Source = "Competition"),
    model,
    method,
    test_preds_req   = True,
    ftreimp_plot_req = False,
    ntop = 50,
)


solution = pd.DataFrame(ytrain.map({v:k for k, v in CFG.tgt_mapper.items()}))
solution["sequence_id"] = Xtrain["sequence_id"].values

submission = pd.Series(np.argmax(oof_ens_preds, axis = 1)).map({v:k for k, v in CFG.tgt_mapper.items()})
submission = submission.to_frame()
submission["sequence_id"] = Xtrain["sequence_id"].values
submission = submission.rename({0 : CFG.target}, axis=1)

score = utils.ScoreMetric(solution, submission)
PrintColor(f"---> Ensemble OOF score = {score: ,.8f} | Competition Metric")


%%time 

oof_preds.assign(sequence_id = Xtrain["sequence_id"].values).to_csv("OOF_Preds.csv")
mdl_preds.assign(sequence_id = Xtest["sequence_id"].values).to_csv("Mdl_Preds.csv")

joblib.dump(FittedModels,  "FittedModels.joblib")
joblib.dump(ens_models,    "EnsembleModels.joblib")

!ls

