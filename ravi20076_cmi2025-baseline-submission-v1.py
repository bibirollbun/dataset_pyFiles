


!pip install -q polars==1.29.0      --no-index --find-links=/kaggle/input/cmi2025-public-imports-v1/packages
!pip install -q xgboost==3.0.1      --no-index --find-links=/kaggle/input/cmi2025-public-imports-v1/packages
!pip install -q scikit-learn==1.6.1 --no-index --find-links=/kaggle/input/cmi2025-public-imports-v1/packages
!pip install -q pytorch_tabnet      --no-index --find-links=/kaggle/input/cmi2025-public-imports-v1/packages
!pip install -q tabpfn              --no-index --find-links=/kaggle/input/cmi2025-public-imports-v1/packages

exec( open(f"/kaggle/input/cmi2025-public-imports-v1/myimports.py", "r").read() )
exec( open(f"/kaggle/input/cmi2025-public-imports-v1/myutils.py", "r").read() )
exec( open(f"/kaggle/input/cmi2025-public-imports-v1/training.py", "r").read() )

exec( open(f"/kaggle/input/cmi2025-baseline-train-v1/fe.py", "r").read() )
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

l1_models = joblib.load(f"/kaggle/input/cmi2025-baseline-train-v1/FittedModels.joblib")
l2_models = joblib.load(f"/kaggle/input/cmi2025-baseline-train-v1/EnsembleModels.joblib")
fe        = FeatureMaker(True)



%%time 

class VotingModelMaker:
    """
    This class collates the test set predictions per model method and averages the predictions through the folds
    """
    
    def __init__(self, estimators: list, method: str):
        "Defines the estimators and method for the predictions"

        self.estimators = estimators
        self.method     = method

    def fit(self, X : pd.DataFrame, y = None, **fit_params):
        return self

    def predict(self, X: pd.DataFrame, y = None, **params) -> pd.DataFrame:
        "Returns probability predictions from the fitted models"

        df_list = []
        for model in self.estimators :
            df = \
            pd.DataFrame( 
                model.predict_proba(
                    X.drop("sequence_id", axis=1, errors = "ignore")
                ) 
            )
            df.index = range( len(df) )
            
        df_list.append(df)
        df = pd.concat(df_list, axis=0).groupby(level = 0).mean().add_prefix(self.method)
        return df

    def fit_predict(self, X: pd.DataFrame, y = None, **params ) -> pd.DataFrame:
        self.fit(X, y)
        return self.predict(X)
     


%%time 

import kaggle_evaluation.cmi_inference_server
verbosity = 0

def predict(
    sequence     : pl.DataFrame, 
    demographics : pl.DataFrame,
) -> str:
    """
    Prediction function for Kaggle evaluation
    """

    test  = sequence.to_pandas()
    Xtest = fe.make_ftre(test)

    if verbosity > 0: 
        print(f"---> Shape = {Xtest.shape}")


    mdl_preds = []  
    for method, estimators in l1_models.items() :
        vmm   = VotingModelMaker(estimators, method)
        preds = vmm.predict(Xtest)
        mdl_preds.append(preds)
        del preds

    mdl_preds = pd.concat(mdl_preds, axis = 1)
    if verbosity > 0: 
        print(f"---> Shape = {mdl_preds.shape}")

    mdl_ens_preds = 0
    vmm = VotingModelMaker(l2_models, "L21C")
    mdl_ens_preds = vmm.predict(mdl_preds).to_numpy()
    mdl_ens_preds = pd.DataFrame(np.argmax(mdl_ens_preds, axis=1), columns = [CFG.target])  
    mdl_ens_preds[CFG.target] = mdl_ens_preds[CFG.target].map({v: k for k,v in CFG.tgt_mapper.items() })

    if verbosity > 0: 
        print(f"---> Prediction = {mdl_ens_preds[CFG.target].values[0]}")

    return mdl_ens_preds[CFG.target].values[0]


%%time 

inference_server = \
kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )

