


%%time 

ip_path     = f"/kaggle/input/playground-series-s5e4"
orig_path   = f"/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv"
op_path     = f"/kaggle/working"
version_lbl = "MLV1_1"
target      = f"Listening_Time_minutes"
state       = 42

test_req = False
if test_req :
    n_iter   = 200
    n_splits = 5
else:
    n_iter   = 4000
    n_splits = 5
    


!uv pip install -q --system -r /kaggle/input/playgrounds5e4-public-imports-v1/req_kaggle.txt

exec( open(f"/kaggle/input/playgrounds5e4-public-imports-v1/myimports.py", "r").read() )
from itertools import combinations

print()


%%time 

train      = pd.read_csv(f"{ip_path}/train.csv", index_col = "id")
test       = pd.read_csv(f"{ip_path}/test.csv", index_col = "id")
sub_fl     = pd.read_csv(f"{ip_path}/sample_submission.csv", index_col = "id")
original   = pd.read_csv(f"{orig_path}")
strt_ftre  = list( test.columns )
original   = original.drop_duplicates().dropna(subset = [target])

if test_req :
    train    = train.iloc[0:10000]
    test     = test.iloc[0:10000]
    sub_fl   = sub_fl.iloc[0:10000]
    original = original.iloc[0:10000]
    PrintColor(f"\n---> Shape = {train.shape} {test.shape} {original.shape} | Syntax check\n")



%%time 

df = \
pd.concat(
    [train[strt_ftre] , original[strt_ftre], test], axis=0, 
    ignore_index = True
)

PrintColor(
    f"\n---> Shape = {train.shape} {test.shape} {original.shape} {df.shape}"
)

df["Weekday"] = \
df["Publication_Day"].\
map(
    {"Sunday"    : 0, 
     "Monday"    : 1, 
     "Tuesday"   : 2, 
     "Wednesday" : 3, 
     "Thursday"  : 4,
     "Friday"    : 5,
     "Saturday"  : 6,
    }
)

df["SinWeekday"] = np.sin(2 * np.pi * df["Weekday"]/ 7)
df["CosWeekday"] = np.cos(2 * np.pi * df["Weekday"]/ 7)

df["Time"] = \
df["Publication_Time"].\
map(
    {"Morning"   : 0, 
     "Afternoon" : 1, 
     "Evening"   : 2, 
     "Night"     : 3, 
    }
)
    
df["SinTime"] = np.sin(2 * np.pi * df["Time"]/ 4)
df["CosTime"] = np.cos(2 * np.pi * df["Time"]/ 4)

df["Episode_Title"] = \
df["Episode_Title"].str.split(" ", expand = True)[1].astype(np.uint16)
df["Number_of_Ads"] = df["Number_of_Ads"].fillna(0).clip(0,3).astype(np.uint8)

df["Episode_Length_minutes"] = df['Episode_Length_minutes'].fillna(60)
df['SinEpLen']               = np.sin(2*np.pi * df['Episode_Length_minutes']/60 )
df['CosEpLen']               = np.cos(2*np.pi * df['Episode_Length_minutes']/60 )

del df["Publication_Time"] , df["Publication_Day"]

df["ELen_Int"] = np.floor( df["Episode_Length_minutes"] )
df["ELen_Dec"] = df["Episode_Length_minutes"] - df["ELen_Int"]

cat_cols = \
["Podcast_Name", "Episode_Title", "Genre", "Number_of_Ads", 
 "Episode_Sentiment", "ELen_Int"
]

df[cat_cols] = df[cat_cols].astype("string").fillna("missing")

for col1, col2 in combinations(cat_cols, 2) :
    df[f"{col1}-{col2}"] = df[col1] + "-" + df[col2]

if test_req == False:
    for col1, col2, col3 in combinations(cat_cols, 3) :
        df[f"{col1}-{col2}-{col3}"] = df[col1] + "-" + df[col2] + "-" + df[col3]

PrintColor(
    f"---> Shape = {train.shape} {test.shape} {original.shape} {df.shape}\n"
)


%%time 

orig    = df.iloc[len(train) : -len(test)]
yorig   = original[target]
Xtrain  = df.iloc[0 : len(train)]
Xtest   = df.iloc[-len(test) :]

PrintColor(f"\n---> Original dataset features\n")
for col in orig.select_dtypes(["string", "category"]).columns :
    df_1         = orig[[col]]
    df_1[target] = yorig.values
    
    df_1 = df_1.groupby(col)[target].describe()[["mean"]]
    df_1 = df_1.add_prefix(f"{col}_")

    Xtrain[f"TE-{col}"] = Xtrain[col].values
    Xtest[f"TE-{col}"]  = Xtest[col].values
    
    Xtrain = Xtrain.merge(df_1, how = "left", on = [col])
    Xtest  = Xtest.merge(df_1, how = "left", on = [col])
    del df_1
    print(f"---> Shape = {Xtrain.shape} {Xtest.shape} | {col}")

    


PrintColor(f"\n---> Resultant columns\n\n")
with np.printoptions(linewidth = 150):
    pprint(np.array( Xtrain.columns ))

Xtrain.to_parquet(f"{op_path}/Xtrain.parquet")
Xtest.to_parquet(f"{op_path}/Xtest.parquet")
train[[target]].to_parquet(f"{op_path}/ytrain.parquet")


%%time 

method   = "XGB1R"
mymodel  = \
XGBR(**{"objective"            : "reg:squarederror",
        "eval_metric"          : "rmse",
        'device'               : "cuda" if torch.cuda.is_available() else "cpu",
        'learning_rate'        : 0.02,
        'n_estimators'         : n_iter,
        'max_depth'            : 6,
        'colsample_bytree'     : 0.30,
        'colsample_bynode'     : 0.35,
        'subsample'            : 0.40,
        'reg_alpha'            : 0.25,
        'verbosity'            : 0,
        'random_state'         : 42,
      } 
   )

cv         = KFold(n_splits = n_splits, random_state = state, shuffle = True)
te_cols    = list( Xtest.select_dtypes(["category", "string"]).columns )
test_preds = 0
scores     = []
ftreimp    = 0

for fold_nb, (train_idx, dev_idx) in enumerate( cv.split(Xtrain, train[target]) ) :
    PrintColor(
        f"\n ============ {method} - FOLD {fold_nb + 1} / {n_splits} ============ \n",
        color = Fore.BLACK,
    )

    Xtr  = Xtrain.iloc[train_idx]
    Xdev = Xtrain.iloc[dev_idx]
    ytr  = train.loc[train_idx, target]
    ydev = train.loc[dev_idx, target]

    if test_req :
        Xtr = Xtr.iloc[0: 1000]
        ytr = ytr.iloc[0: 1000]
        print(f"---> Shapes = {Xtr.shape} {ytr.shape} {Xdev.shape} {ydev.shape} | Syntax check")
    else:
        print(f"---> Shapes = {Xtr.shape} {ytr.shape} {Xdev.shape} {ydev.shape}")

    model = \
    Pipeline(
        [( "PP",
            ColumnTransformer(
                [("TE", TargetEncoder(random_state = state), te_cols)], 
                remainder = "passthrough", 
                verbose_feature_names_out = False
            )
         ),
         ("M", clone(mymodel))
        ]
    )

    model.fit( Xtr, ytr )
    dev_preds = model.predict(Xdev)
    score     = root_mean_squared_error(ydev, dev_preds)
    PrintColor(f"---> Score = {score :,.8f}")
    
    test_preds += ( model.predict(Xtest) / n_splits )
    scores.append(score)

    try:
        ftreimp += model["M"].feature_importances_
    except:
        pass

    del Xtr, Xdev, ytr, ydev
    collect();


PrintColor(
    f"\n\n---> OOF score = {np.mean(scores) :,.8f} +- {np.std(scores) :,.8f} | {method}\n\n"
)

try:
    with sns.axes_style("white") :
        fig, ax = plt.subplots(1,1, figsize = (25, 8))
        
        (
            pd.Series(ftreimp, index = Xtest.columns, name = "FtreImp").
            sort_values(ascending = False).
            head(50).
            plot.bar(ax = ax)
        )

        ax.set_title(
            f"{method} feature importances", 
            fontweight = "bold", 
            color = "maroon", 
            fontsize = 15
        )
        plt.show()
        
except: 
    pass



%%time 

try:
    sub = \
    pd.read_csv(
        f"/kaggle/input/12-38095-predict-podcast-listening-time/submission.csv"
    )[target].values.flatten()[0 : len(test_preds)]
    sub_fl[target] = test_preds * 0.20 + sub * 0.80
except:
    sub_fl[target] = test_preds
    print(f"---> Check the public blend - did not blend here\n")
    
sub_fl.to_csv("submission.csv")
print()
!ls
print()
!head submission.csv

