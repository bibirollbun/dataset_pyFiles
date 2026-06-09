%load_ext cudf.pandas
import pandas as pd
import numpy as np


import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
sample_df = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


train_df.head()


train_df.info()


train_df.isnull().sum() / len(train_df) * 100


test_df.info()


test_df.isnull().sum() / len(test_df) * 100


NOT_FEATURES = ['rainfall','id','bucket','year_group']
FEATURES = [feat for feat in list(train_df.columns) if feat not in NOT_FEATURES]


for feat in FEATURES:

    # Comparison train data with test data
    plt.figure(figsize=(12,3))
    plt.subplot(1,2,1)
    sns.distplot(train_df[feat], label='train_df')
    sns.distplot(test_df[feat],label='test_df')
    plt.legend()
    plt.title(f"{feat}")

    # PLOT TARGET RELATIONSHIP WITH BINNED NUMERIC FEATURES
    plt.subplot(1,2,2)
    train_df['bucket'], bin_edges = pd.cut(train_df[feat], bins=10, retbins=True, labels=False)
    bucket_means= train_df.groupby('bucket')['rainfall'].mean()
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.plot(bin_midpoints, bucket_means, marker='o', linestyle='-')
    plt.xlabel(f'{feat} (Binned)')
    plt.ylabel('Mean Rainfall')
    plt.title(f'Mean Rainfall per {feat} (train)')
    plt.xticks(bin_midpoints, rotation=45)
    plt.grid()
    
    plt.show()


from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import roc_auc_score
from cuml.svm import SVC, LinearSVC # we use cuml instead of scikit's SVM since it can run on GPU


INTERACT = []
for i,c1 in enumerate(FEATURES):
    for j,c2 in enumerate(FEATURES[i+1:]):
        n = f"{c1}_{c2}"
        train_df[n] = train_df[c1] * train_df[c2]
        test_df[n] = test_df[c1] * test_df[c2]
        INTERACT.append(n)
print(f"There are {len(INTERACT)} interaction features:")
print( INTERACT )


train_df['year_group'] = train_df['id']//365 


ADD = []
best_auc = 0
best_oof = None
best_pred = None

# Feature Selection
for k, col in enumerate(['baseline']+INTERACT):
    FOLDS = train_df.year_group.nunique()
    kf = GroupKFold(n_splits=FOLDS) 
    
    oof_svc = np.zeros(len(train_df))
    pred_svc = np.zeros(len(test_df))
    if col!='baseline': ADD.append(col)

    # Group K-Fold using year as the group
    for i, (train_index, test_index) in enumerate(kf.split(train_df,groups=train_df.year_group)):

        # TRAIN AND VALID DATA
        x_train = train_df.loc[train_index,FEATURES+ADD ].copy()
        y_train = train_df.loc[train_index,"rainfall"]
        x_valid = train_df.loc[test_index,FEATURES+ADD ].copy()
        y_valid = train_df.loc[test_index,"rainfall"]
        x_test = test_df[FEATURES+ADD].copy()

        # SVC WANTS STANDARIZED FEATURES
        for c in FEATURES+ADD:
            m = x_train[c].mean()
            s = x_train[c].std()
            x_train[c] = (x_train[c]-m)/s
            x_valid[c] = (x_valid[c]-m)/s
            x_test[c] = (x_test[c]-m)/s
            x_test[c] = x_test[c].fillna(0)

        # TRAIN SVC MODEL
        model = LinearSVC(C=0.1, probability=True)
        model.fit(x_train.values, y_train.values)
    
        # INFER OOF
        oof_svc[test_index] = model.predict_proba(x_valid.values)[:,1]
        # INFER TEST
        pred_svc += model.predict_proba(x_test.values)[:,1]

    # Compute avg test preds
    pred_svc /= FOLDS

    # Compute CV Validation AUC Score
    true = train_df.rainfall.values
    m = roc_auc_score(true, oof_svc)

    if m >best_auc:
        print(f"NEW BEST with {col} at {m}")
        best_auc = m
        best_oof = oof_svc.copy()
        best_pred = pred_svc.copy()
    else :
        print(f"Worse with {col} at {m}")
        ADD.remove(col)


print(f"We achieved CV SVC AUC = {best_auc:.4f} adding {len(ADD)} interactions features:")
print( ADD )


sample_df.rainfall = best_pred
print("Submission shape:", sample_df.shape )
sample_df.to_csv(f"submission.csv",index=False)
sample_df.head()

