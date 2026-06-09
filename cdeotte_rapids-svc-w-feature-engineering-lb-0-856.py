%load_ext cudf.pandas
import pandas as pd, numpy as np

train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
train['year_group'] = train['id']//365 
print("Train shape:", train.shape )
train.head()


test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
print("Test shape:", test.shape )
test.head()


import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


RMV = ['rainfall','id','bucket','year_group']
FEATURES = [c for c in list( train.columns ) if not c in RMV]
print(f"We have {len(FEATURES)} basic features:")
print( FEATURES )


for c in FEATURES:

    # PLOT TRAIN DISTRIBUTION COMPARED WITH TEST DISTRIBUTION
    plt.figure(figsize=(12,3))
    plt.subplot(1,2,1)
    sns.distplot(train[c],label='train')
    sns.distplot(test[c],label='test')
    plt.legend()
    plt.title(f"{c}")    

    # PLOT TARGET RELATIONSHIP WITH BINNED NUMERIC FEATURES
    plt.subplot(1,2,2)
    train['bucket'], bin_edges = pd.cut(train[c], bins=10, retbins=True, labels=False)
    bucket_means = train.groupby('bucket')['rainfall'].mean()
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.plot(bin_midpoints, bucket_means, marker='o', linestyle='-')
    plt.xlabel(f'{c} (Binned)')
    plt.ylabel('Mean Rainfall')
    plt.title(f'Mean Rainfall per {c} (train)')
    plt.xticks(bin_midpoints, rotation=45)
    plt.grid()
    
    plt.show()


from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import roc_auc_score
#from sklearn.svm import SVC, LinearSVC
from cuml.svm import SVC, LinearSVC


INTERACT = []
for i,c1 in enumerate(FEATURES):
    for j,c2 in enumerate(FEATURES[i+1:]):
        n = f"{c1}_{c2}"
        train[n] = train[c1] * train[c2]
        test[n] = test[c1] * test[c2]
        INTERACT.append(n)
print(f"There are {len(INTERACT)} interaction features:")
print( INTERACT )


ADD  = []
best_auc = 0
best_oof = None
best_pred = None

# FORWARD FEATURE SELECTION 
for k,col in enumerate(['baseline']+INTERACT):

    FOLDS = train.year_group.nunique()
    kf = GroupKFold(n_splits=FOLDS) 
    
    oof_svc = np.zeros(len(train))
    pred_svc = np.zeros(len(test))

    if col!='baseline': ADD.append(col)

    # GROUP K FOLD USING YEAR AS GROUP
    for i, (train_index, test_index) in enumerate(kf.split(train,groups=train.year_group)):
        #print("#"*25)
        #print(f"### Fold {i+1}")
        #print("#"*25)

        # TRAIN AND VALID DATA
        x_train = train.loc[train_index,FEATURES+ADD ].copy()
        y_train = train.loc[train_index,"rainfall"]
        x_valid = train.loc[test_index,FEATURES+ADD ].copy()
        y_valid = train.loc[test_index,"rainfall"]
        x_test = test[FEATURES+ADD ].copy()

        # SVC WANTS STANDARIZED FEATURES
        for c in FEATURES+ADD:
            m = x_train[c].mean()
            s = x_train[c].std()
            x_train[c] = (x_train[c]-m)/s
            x_valid[c] = (x_valid[c]-m)/s
            x_test[c] = (x_test[c]-m)/s
            x_test[c] = x_test[c].fillna(0)

        # TRAIN SVC MODEL
        #model = SVC(C=0.1, probability=True, kernel='poly', degree=1)
        model = LinearSVC(C=0.1, probability=True)
        model.fit(x_train.values, y_train.values)
    
        # INFER OOF
        oof_svc[test_index] = model.predict_proba(x_valid.values)[:,1]
        # INFER TEST
        pred_svc += model.predict_proba(x_test.values)[:,1]
    
    # COMPUTE AVERAGE TEST PREDS
    pred_svc /= FOLDS

    # COMPUTE CV VALIDATION AUC SCORE
    true = train.rainfall.values
    m = roc_auc_score(true, oof_svc)
    
    if m>best_auc:
        print(f"NEW BEST with {col} at {m}")
        best_auc = m
        best_oof = oof_svc.copy()
        best_pred = pred_svc.copy()
    else:
        print(f"Worse with {col} at {m}")
        ADD.remove(col)


print(f"We achieved CV SVC AUC = {best_auc:.4f} adding {len(ADD)} interactions features:")
print( ADD )


sub = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
sub.rainfall = best_pred
print("Submission shape:", sub.shape )
sub.to_csv(f"submission.csv",index=False)
sub.head()

