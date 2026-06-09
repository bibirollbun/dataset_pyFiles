


%%time 

!pip install -q polars==1.29.0      --no-index --find-links=/kaggle/input/cmi2025-public-imports-v3/packages
!pip install -q xgboost==3.0.1      --no-index --find-links=/kaggle/input/cmi2025-public-imports-v3/packages
!pip install -q lightgbm==4.6.0     --no-index --find-links=/kaggle/input/cmi2025-public-imports-v3/packages
!pip install -q scikit-learn==1.6.1 --no-index --find-links=/kaggle/input/cmi2025-public-imports-v3/packages
!pip install -q tabpfn              --no-index --find-links=/kaggle/input/cmi2025-public-imports-v3/packages
!pip install -q pytorch_tabnet      --no-index --find-links=/kaggle/input/cmi2025-public-imports-v3/packages

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
    data_path          = f"/kaggle/input/cmi2025-public-baseline-train-v4"
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

collect()


%%time 

test     = pl.read_csv(f"{CFG.ip_path}/test.csv")
sel_cols = [ c for c in test.collect_schema().names() if "thm" not in c and "tof" not in c ]

exec( open( f"{CFG.data_path}/myfe.py", "r").read())

fitted_models = joblib.load(
    f"{CFG.data_path}/FittedModels{CFG.model_label}{CFG.model_id}.joblib"
)

vmms = []
for method, estimators in fitted_models.items() :
    vmm = VotingModelMaker(estimators, method, problem_type = "multiclass", as_numpy = False)
    vmms.append(vmm)


%%time 

import kaggle_evaluation.cmi_inference_server

def predict(
    sequence     : pl.DataFrame, 
    demographics : pl.DataFrame,
) -> str:
    """
    Prediction function for Kaggle evaluation
    """

    Xtest     = make_ftre(sequence, demographics)
    Mdl_Preds = []
    for model in vmms:
        Mdl_Preds.append(model.predict_proba(Xtest))

    pred = \
    (
        pd.concat(Mdl_Preds, axis = 0, ignore_index = False).
        groupby(level = 0).
        mean().
        idxmax(axis=1).
        map({v: k for k, v in CFG.tgt_mapper.items()}).
        values[0]
    )
    
    print(f"Final prediction = {pred}")
    return pred


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

print()

