


%%time 

!pip install -q polars==1.31.0      --no-index --find-links=/kaggle/input/map2025-public-imports-v1/packages
!pip install -q scikit-learn==1.7.0 --no-index --find-links=/kaggle/input/map2025-public-imports-v1/packages
!pip install -q xgboost==3.0.2      --no-index --find-links=/kaggle/input/map2025-public-imports-v1/packages
!pip install -q lightgbm==4.6.0     --no-index --find-links=/kaggle/input/map2025-kaggle-imports-v1/packages
!pip install -q pytorch_tabnet      --no-index --find-links=/kaggle/input/map2025-kaggle-imports-v1/packages
!pip install -q tabpfn              --no-index --find-links=/kaggle/input/map2025-kaggle-imports-v1/packages

exec( open(f"/kaggle/input/map2025-public-imports-v1/myimports.py",   "r"   ).read() )
exec( open(f"/kaggle/input/map2025-public-imports-v1/myutils.py",     "r"   ).read() )
exec( open(f"/kaggle/input/map2025-public-imports-v1/training.py",    "r"   ).read() )

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
    test_iter          = 50
    gpu_switch         = "ON" if torch.cuda.is_available() else "OFF"
    state              = 42
    
    target             = 'target_cat'
    grouper            = f""
    tgt_mapper         = {}
    ip_path            = f"/kaggle/input/map-charting-student-math-misunderstandings"
    op_path            = f"/kaggle/working"
    orig_path          = f""
    data_path          = f""
    dtl_preproc_req    = True
    ftre_plots_req     = True
    ftre_imp_req       = True

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
 "GKF"   : GroupKFold(n_splits   = CFG.n_splits),
 "SGKF"  : StratifiedGroupKFold(n_splits = CFG.n_splits, shuffle = True, random_state= CFG.state),
}

collect()


class TextCleaner:
    "Cleans the text data provided as a preprocessing step for the model"
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        text = re.sub(r'(\d+)\s*/\s*(\d+)', r'FRAC_\1_\2', text)
        text = re.sub(r'\\frac\{([^\}]+)\}\{([^\}]+)\}', r'FRAC_\1_\2', text)
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\s_]', '', text)
        return text.strip().lower()

    def extract_math_features(self, text):
        features = {}
        features['frac_count']     = len(re.findall(r'FRAC_\d+_\d+|\\frac', text))
        features['number_count']   = len(re.findall(r'\b\d+\b', text))
        features['operator_count'] = len(re.findall(r'[\+\-\*\/\=]', text))
        return features

    def _lemmatize(self, text):
        return " ".join([self.lemmatizer.lemmatize(word) for word in text.split()])

    def create_features(self, df, is_train=True):
        df['mc_answer_len']                 = df['MC_Answer'].astype(str).str.len()
        df['explanation_len']               = df['StudentExplanation'].astype(str).str.len()
        df['question_len']                  = df['QuestionText'].astype(str).str.len()
        df['explanation_to_question_ratio'] = df['explanation_len'] / (df['question_len'] + 1)
    
        for col in ['QuestionText', 'MC_Answer']:
            math_features = df[col].apply(self.extract_math_features).apply(pd.Series)
            prefix        = 'mc_' if col == 'MC_Answer' else ''
            math_features.columns = [f'{prefix}{c}' for c in math_features.columns]
            df = pd.concat([df, math_features], axis=1)

        df['cleaned_text'] = (
            "Question: "     + df['QuestionText'].astype("string") +
            " Answer: "      + df['MC_Answer'].astype("string")    +
            " Explanation: " + df['StudentExplanation'].astype("string")
        )

        df["cleaned_text"] = df["cleaned_text"].apply(self.clean_text).apply(self._lemmatize)
        return df


%%time 

train = pd.read_csv(f"{CFG.ip_path}/train.csv")
test  = pd.read_csv(f"{CFG.ip_path}/test.csv")

train['Misconception'] = train['Misconception'].astype("string").fillna('NA')
train['target_cat']    = train["Category"].astype("string") + ":" + train["Misconception"]

print(f"\n---> Shapes = {train.shape} {test.shape} | Data loading")

cleaner = TextCleaner()
train   = cleaner.create_features(train)
test    = cleaner.create_features(test)
print(f"---> Shapes = {train.shape} {test.shape} | Data cleaning")

display(
    train["Misconception"].
    value_counts().
    to_frame().
    transpose().
    style.
    set_caption("Misconceptions in train - counts")
)

print("\n\n\n")
display(
    train["Category"].
    value_counts().
    to_frame().
    transpose().
    style.
    set_caption("Categories in train - counts")
)

num_cols = [
    'mc_answer_len',
    'explanation_len', 
    'question_len', 
    'explanation_to_question_ratio',
    'frac_count', 
    'number_count', 
    'operator_count',
    'mc_frac_count',
    'mc_number_count', 
    'mc_operator_count',
]

tftdf = \
TfidfVectorizer(
    stop_words     = 'english',
    ngram_range    = (1, 3), 
    analyzer       = 'word', 
    max_df         = 0.95, 
    min_df         = 2,
    max_features   = 5000 if CFG.test_req == False else 10,
)

df = pd.concat([train['cleaned_text'], test['cleaned_text']])
tftdf.fit(df)
del df

Xtrain = reduce_mem_usage( 
    pd.DataFrame( tftdf.transform(train['cleaned_text']).toarray() ).add_prefix("C"), "Train" 
)
Xtrain = pd.concat([Xtrain, train[num_cols]], axis = 1)

Xtest  = reduce_mem_usage( 
    pd.DataFrame( tftdf.transform(test['cleaned_text']).toarray()  ).add_prefix("C"), "Test" 
)
Xtest  = pd.concat([Xtest, test[num_cols]], axis = 1)



%%time

mapper   = train[CFG.target].value_counts()
mapper   = mapper.reset_index()
to_remap = mapper.loc[mapper["count"] <= CFG.n_splits, CFG.target].values
mapper.loc[mapper["count"] <= CFG.n_splits, CFG.target] = "True_Misconception:Inversion"

mapper         = mapper.drop_duplicates(subset = [CFG.target])
CFG.tgt_mapper = {v:k for k, v in mapper.to_dict()[CFG.target].items()}

ytrain = train[CFG.target]
ytrain.loc[ytrain.isin(to_remap.tolist())] = "True_Misconception:Inversion"
ytrain = ytrain.map(CFG.tgt_mapper)


PrintColor(
    f"\n---> Shapes = {Xtrain.shape} {ytrain.shape} {Xtest.shape}\n"
)

print()
_ = utils.CleanMemory()


%%time 

cv    = cv_selector[CFG.mdlcv_mthd]
folds = np.zeros(len(train))

for fold_nb, (train_idx, dev_idx) in enumerate( cv.split(train, ytrain) ) :
    folds[dev_idx] = fold_nb

ygrp = pd.Series(folds, dtype = np.uint8, name = "fold_nb")
cv   =  PredefinedSplit(ygrp)

display(
    ygrp.value_counts().
    to_frame().
    sort_index(ascending = True).
    transpose().
    style.set_caption("CV folds")
)

print()


%%time 

Mdl_Master = \
{     
 f'XGB1C'  : [XGBC(**{ "objective"            : "multi:softprob",
                       'device'               : "cuda" if CFG.gpu_switch == "ON" else "cpu",
                       'learning_rate'        : 0.03,
                       'n_estimators'         : 700 if CFG.test_req == False else CFG.test_iter,
                       'max_depth'            : 8,
                       'subsample'            : 0.30,
                       'colsample_bytree'     : 0.30,
                       'colsample_bynode'     : 0.30,
                       'colsample_bylevel'    : 0.25,
                       'reg_alpha'            : 0.10,
                       'reg_lambda'           : 1.50, 
                       'verbosity'            : 0,
                       'random_state'         : CFG.state,
                       'early_stopping_rounds': None if CFG.nbrnd_erly_stp == 0 else CFG.nbrnd_erly_stp,
                       'enable_categorical'   : True,
                      }  
                   ),
              {"verbose" : 0}
             ],

 f'LGBM1C' : [LGBMC(**{"objective"            : "multiclass",
                       "eval_metric"          : "logloss",
                       'device'               : "gpu" if CFG.gpu_switch == "ON" else "cpu",
                       'learning_rate'        : 0.03,
                       'n_estimators'         : 1000 if CFG.test_req == False else CFG.test_iter,
                       'max_depth'            : 6,
                       'subsample'            : 0.30,
                       'verbosity'            : -1,
                       'random_state'         : CFG.state,
                       'reg_alpha'            : 0.01,
                       'reg_lambda'           : 1.80,
                       'class_weight'         : 'balanced',
                      } 
                   ),
              {"callbacks" : [log_evaluation(0)], 'eval_metric' : 'logloss'},
             ], 
}

# Initializing model outputs
Mdl_Preds    = []
OOF_Preds    = []
drop_cols    = ["Source", "id", "Id", "Label", "fold_nb"] + [CFG.target]


%%time 

for method, (mymodel, fit_params) in tqdm(Mdl_Master.items()) :
    md = ModelTrainer(
        drop_cols      = drop_cols, 
        len_train      = Xtrain.shape[0],
        k              =  3,
        test_preds_req = True
    )

    _, oof_preds, mdl_preds = \
    md.fit_predict(
        Xtrain, 
        ytrain,
        Xtest, 
        ygrp,
        method   = method,
        mymodel  = mymodel,
        cat_cols = None,
        **fit_params,
    )   
    
    Mdl_Preds.append( pd.DataFrame(mdl_preds).add_prefix(CFG.target) )
    OOF_Preds.append( pd.DataFrame(oof_preds).add_prefix(CFG.target) )
    collect();

print()

oof_preds = pd.concat(OOF_Preds).groupby(level = 0).mean().to_numpy()
mdl_preds = pd.concat(Mdl_Preds).groupby(level = 0).mean().to_numpy()

score = utils.ScoreMetric(
    [[label] for label in ytrain.values],
    np.argsort(oof_preds, axis=1)[:, -3:][:, ::-1].tolist(), 
)

PrintColor(f"---> Overall OOF score = {score:,.8f}\n\n")
_ = utils.CleanMemory()


%%time 

test_preds = pd.DataFrame(np.argsort(-mdl_preds, 1)[:,:3])
for col in test_preds.columns:
    test_preds[col] = test_preds[col].map({v:k for k,v in CFG.tgt_mapper.items()})
    
test_preds["Pred"] = test_preds.apply(lambda x: " ".join(x), axis=1)

sub_fl = pd.read_csv(f"{CFG.ip_path}/sample_submission.csv")
sub_fl['Category:Misconception'] = test_preds["Pred"].values
sub_fl.to_csv("submission.csv", index=False)

print()
display(
    sub_fl.head(10)
)

!ls
print()
!head submission.csv

