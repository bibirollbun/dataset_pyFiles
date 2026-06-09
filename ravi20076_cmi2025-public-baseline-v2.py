


%%time 

!pip install -q polars==1.29.0      --no-index --find-links=/kaggle/input/cmi2025-public-imports-v2/packages
!pip install -q xgboost==3.0.1      --no-index --find-links=/kaggle/input/cmi2025-public-imports-v2/packages
!pip install -q scikit-learn==1.6.1 --no-index --find-links=/kaggle/input/cmi2025-public-imports-v2/packages
!pip install -q pytorch_tabnet      --no-index --find-links=/kaggle/input/cmi2025-public-imports-v2/packages
!pip install -q tabpfn              --no-index --find-links=/kaggle/input/cmi2025-public-imports-v2/packages

exec( open(f"/kaggle/input/cmi2025-public-imports-v2/myimports.py", "r").read() )
exec( open(f"/kaggle/input/cmi2025-public-imports-v2/myutils.py", "r").read() )
exec( open(f"/kaggle/input/cmi2025-public-imports-v2/training.py", "r").read() )

print()


%%time 

target       = "gesture"
version_lbl  = "V1_3"
model_lbl    = "ML"
state        = 42
n_splits     = 5
verbosity    = 1

mapper = {
    "Text on phone"            : 0,
    "Forehead - pull hairline" : 1,    
    "Neck - scratch"           : 2,        
    "Neck - pinch skin"        : 3,
    "Eyelash - pull hair"      : 4,        
    "Forehead - scratch"       : 5,          
    "Eyebrow - pull hair"      : 6,
    "Above ear - pull hair"    : 7,   
    "Cheek - pinch skin"       : 8,
}


%%time 

train  = pd.read_csv(
    f"/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv", 
    index_col = ["row_id"]
)

test   = pd.read_csv(
    f"/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv", 
    index_col = ["row_id"]
)

traind = pd.read_csv(
    f"/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv"
)
testd  = pd.read_csv(
    f"/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv"
)

train  = reduce_mem_usage(train,  "train")
test   = reduce_mem_usage(test,   "test")
traind = reduce_mem_usage(traind, "traind")
testd  = reduce_mem_usage(testd,  "testd")

PrintColor(
    f"\n---> Shapes = {train.shape} {test.shape} {traind.shape} {testd.shape}"
)

_ = utils.CleanMemory()
print()


def make_ftre(
    df: pd.DataFrame, demo_df : pd.DataFrame, sel_cols: list
) :
    "Creates aggregate features and joins with the demographics table"
    
    df_out = \
    (
        df.
        groupby(["sequence_id"])[sel_cols + ["subject", "sequence_id"]].
        tail(1).
        set_index(["subject", "sequence_id"]).
        astype(np.float32).
        add_prefix("last_").
        reset_index().
        merge(
            demo_df, on = ["subject"], how = "left"
        ).
        drop("subject", axis=1)
    )

    return df_out


%%time 

sel_cols = list(test.columns[3:])
Xtrain   = make_ftre(train, traind, sel_cols)
Xtest    = make_ftre(test,  testd,  sel_cols)

Ytrain = \
train[["sequence_id", "sequence_type", target]].drop_duplicates().set_index("sequence_id")

Ytrain[target] = \
np.where(
    Ytrain.sequence_type == "Non-Target", 
    'Text on phone', 
    Ytrain[target].values
)

PrintColor(f"\n---> Multiclass targets\n")
display( Ytrain[target].value_counts() )

PrintColor(f"\n---> Binary targets\n")
display( Ytrain["sequence_type"].value_counts())

Xtrain = Xtrain.merge(Ytrain, how = "left", left_on = ["sequence_id"], right_index = True)
Ytrain = Xtrain[["sequence_id", target, "sequence_type"]]
Xtrain = Xtrain.drop([target, "sequence_type"], axis=1)

PrintColor(
    f"\n---> Shapes = {Xtrain.shape} {Xtest.shape}"
)

ygrp = np.zeros(len(Ytrain))
cv   = StratifiedKFold(n_splits = n_splits, random_state = state, shuffle = True)

for fold_nb, (_, dev_idx) in enumerate(cv.split(Xtrain, Ytrain[target])):
    ygrp[dev_idx] = fold_nb

ygrp = pd.Series(ygrp, name = "fold_nb", dtype = np.uint8)

PrintColor(
    f"\n---> Shapes = {Xtrain.shape} {Ytrain.shape} {Xtest.shape} {ygrp.shape}"
)


%%time 

model = CBC(
    iterations    = 400,
    loss_function = "Logloss",
    eval_metric   = "Logloss",
    learning_rate = 0.025,
    max_depth     = 5,
    l2_leaf_reg   = 0.25,
    random_seed   = state,
    verbose       = 0,    
)

method = "BC_CB1C"
ytrain = Ytrain["sequence_type"]

ytrain = \
pd.Series( 
    np.where( ytrain == "Non-Target", 0, 1), 
    name = "sequence_type",
    dtype = np.uint8
)

drop_cols = ["subject", "sequence_id", "Label", "Source", "id", "Id"]

md = ModelTrainer(
    problem_type   = "binary", 
    test_preds_req = False, 
    drop_cols      = drop_cols,
    verbosity      = verbosity,
)

binary_models, bc_oof_preds, _ = \
md.fit_predict(
    Xtrain, ytrain, None, ygrp, model, method, 
    **{"verbose" : 0}
)

joblib.dump( binary_models , "BCModels.joblib")
bc_oof_preds.to_csv("BC_OOF_Preds.csv")

try:
    del Xtrain[method]
except:
    pass

_ = utils.CleanMemory()
print()


%%time 

model = CBC(
    iterations    = 400,
    loss_function = "MultiClass",
    eval_metric   = "MultiClass",
    learning_rate = 0.025,
    max_depth     = 5,
    l2_leaf_reg   = 0.25,
    random_seed   = state,
    verbose       = 0,    
)

method    = "MC_CB1C"
ytrain    = Ytrain[target].map(mapper).astype(np.uint8)
drop_cols = ["subject", "sequence_id", "Label", "Source", "id", "Id"]
Xtrain    = pd.concat([Xtrain, bc_oof_preds], axis=1)

md = ModelTrainer(
    problem_type   = "multiclass", 
    test_preds_req = False, 
    drop_cols      = drop_cols,
    verbosity      = verbosity,
)

mc_models, mc_oof_preds, _ = \
md.fit_predict(
    Xtrain, ytrain, None, ygrp, model, method, 
    **{"verbose" : 0}
)

joblib.dump( mc_models , "MCModels.joblib")
mc_oof_preds.to_csv("MC_OOF_Preds.csv")

_ = utils.CleanMemory()
print()


import kaggle_evaluation.cmi_inference_server

def predict(
    sequence     : pl.DataFrame, 
    demographics : pl.DataFrame,
) -> str:
    """
    Prediction function for Kaggle evaluation
    """

    print()
    test  = reduce_mem_usage( sequence.to_pandas(), "test-sequence" )
    testd = reduce_mem_usage( demographics.to_pandas(), "test-demographics" )
    Xtest = make_ftre(test, testd, sel_cols)

    if verbosity > 0: 
        print(f"---> Shape = {Xtest.shape}")
  
    vmm   = VotingModelMaker(binary_models, "BC_CB1C")
    preds = vmm.predict(Xtest).to_numpy()[:,1].flatten()
    Xtest["BC_CB1C"] = preds

    mdl_preds = []
    vmm   = VotingModelMaker(mc_models, "MC_CB1C")
    preds = vmm.predict(Xtest)
    mdl_preds.append(preds)
    
    mdl_preds = pd.concat(mdl_preds, axis = 0, ignore_index = False)
    mdl_preds = mdl_preds.groupby(level = 0).mean().to_numpy()
    mdl_preds = \
    ( 
        pd.Series(np.argmax( mdl_preds , axis = 1 ), name = target).
        map({v: k for k,v in mapper.items() })
    )
    
    if verbosity > 0: 
        print(f"---> Prediction = {mdl_preds.values[0]}")

    if verbosity == 0: 
        clear_output()
    return mdl_preds.values[0]



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

