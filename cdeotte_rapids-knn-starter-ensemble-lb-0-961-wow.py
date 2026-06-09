import pandas as pd, numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
print("Train shape", train.shape )
train.head()


test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
print("Test shape:", test.shape )
test.head()


RMV = ['rainfall','id']
FEATURES = [c for c in train.columns if not c in RMV]
print("Our features are:")
print( FEATURES )


from sklearn.model_selection import KFold
from cuml.neighbors import KNeighborsClassifier


# WEIGHTS TO ADJUST IMPORTANCE OF FEATURES DURING KNN
WGT = {'day': 24, 'pressure': 1, 'maxtemp': 1, 'temparature': 1, 'mintemp': 1, 'dewpoint': 1, 'humidity': 1, 
       'cloud': 1, 'sunshine': 1, 'winddirection': 1, 'windspeed': 1}


%%time
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=777)
    
oof_knn = np.zeros(len(train))
pred_knn = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"rainfall"]    
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"rainfall"]
    x_test = test[FEATURES].copy()

    for c in FEATURES:
        m = x_train[c].mean()
        s = x_train[c].std()
        x_train[c] = WGT[c] * (x_train[c]-m)/s
        x_valid[c] = WGT[c] * (x_valid[c]-m)/s
        x_test[c] = WGT[c] * (x_test[c]-m)/s
        x_test[c] = x_test[c].fillna(0)
        x_train[c] = x_train[c].fillna(0)

    model = KNeighborsClassifier(n_neighbors=201, p=1)
    model.fit(x_train.values, y_train.values)

    # INFER OOF
    oof_knn[test_index] = model.predict_proba(x_valid.values)[:,1]
    # INFER TEST
    pred_knn += model.predict_proba(x_test.values)[:,1]

# COMPUTE AVERAGE TEST PREDS
pred_knn /= FOLDS


from sklearn.metrics import roc_auc_score
true = train.rainfall.values
m = roc_auc_score(true, oof_knn)
print(f"KNN CV Score AUC = {m:.3f}")


print("Best Public Notebook achieves LB = 0.954!")
best_public = pd.read_csv("/kaggle/input/lb-915-public-notebook/submission95427.csv")
display( best_public.head() )
best_public = best_public.rainfall.values


from scipy.stats import rankdata

print("Ensemble achieves LB = 0.961! Hooray!")
sub = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
sub.rainfall = -0.25 * rankdata( pred_knn ) + 1.25 * rankdata( best_public )
sub.rainfall = rankdata( sub.rainfall ) / len(sub)
print( sub.shape )
sub.to_csv(f"submission_ensemble.csv",index=False)
sub.head()

