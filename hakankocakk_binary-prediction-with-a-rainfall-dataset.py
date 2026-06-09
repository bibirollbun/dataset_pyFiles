import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
orig = pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


train['group'] = train['id']//365
orig['group'] = 6

train = train.drop(columns=['id'])


orig.columns = orig.columns.str.strip()


orig["rainfall"] = orig["rainfall"].map({'yes': 1, 'no': 0})


train2 = pd.concat([train, orig], ignore_index=True)


RMV = ['rainfall','id', 'group']
FEATURES = [c for c in train2.columns if not c in RMV]
print("Our features are:")
print( FEATURES )


from sklearn.model_selection import KFold, GroupKFold
from xgboost import XGBRegressor, XGBClassifier


%%time
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_xgb = np.zeros(len(train2))
pred_xgb = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train2)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train2.loc[train_index,FEATURES].copy()
    y_train = train2.loc[train_index,"rainfall"]    
    x_valid = train2.loc[test_index,FEATURES].copy()
    y_valid = train2.loc[test_index,"rainfall"]
    x_test = test[FEATURES].copy()

    model = XGBClassifier(
        device="cpu",
        max_depth=6,  
        colsample_bytree=0.9, 
        subsample=0.9, 
        n_estimators=10_000,  
        learning_rate=0.1, 
        eval_metric="auc",
        early_stopping_rounds=100,
        alpha=1,
    )
    model.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],  
        verbose=100 
    )

    # INFER OOF
    oof_xgb[test_index] = model.predict_proba(x_valid)[:,1]
    # INFER TEST
    pred_xgb += model.predict_proba(x_test)[:,1]

# COMPUTE AVERAGE TEST PREDS
pred_xgb /= FOLDS


from sklearn.metrics import roc_auc_score
true = train2.rainfall.values
m = roc_auc_score(true, oof_xgb)
print(f"XGBoost CV Score AUC = {m:.3f}")


feature_importance = model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": FEATURES,  
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 5))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance")
plt.gca().invert_yaxis()  
plt.show()


submission.rainfall = pred_xgb
submission.to_csv(f"submission_XGBoost.csv",index=False)


pip install tabpfn


from tabpfn import TabPFNClassifier


%%time
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_tabpfn = np.zeros(len(train2))
pred_tabpfn = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train2)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train2.loc[train_index,FEATURES].copy()
    y_train = train2.loc[train_index,"rainfall"]    
    x_valid = train2.loc[test_index,FEATURES].copy()
    y_valid = train2.loc[test_index,"rainfall"]
    x_test = test[FEATURES].copy()

    model = TabPFNClassifier(device='cuda')
    
    model.fit(x_train, y_train)

    # INFER OOF
    oof_tabpfn[test_index] = model.predict_proba(x_valid)[:,1]
    # INFER TEST
    pred_tabpfn += model.predict_proba(x_test)[:,1]

# COMPUTE AVERAGE TEST PREDS
pred_tabpfn /= FOLDS


from sklearn.metrics import roc_auc_score
true = train2.rainfall.values
m = roc_auc_score(true, oof_tabpfn)
print(f"TabPFN CV Score AUC = {m:.3f}")


submission.rainfall = pred_tabpfn
submission.to_csv(f"submission_tabpfn.csv",index=False)


%%time
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_xgb_tabpfn = np.zeros(len(train2))
pred_xgb_tabpfn = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train2)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train2.loc[train_index,FEATURES].copy()
    y_train = train2.loc[train_index,"rainfall"]    
    x_valid = train2.loc[test_index,FEATURES].copy()
    y_valid = train2.loc[test_index,"rainfall"]
    x_test = test[FEATURES].copy()

    xgb_model = XGBClassifier(
        device="cpu",
        max_depth=6,  
        colsample_bytree=0.9, 
        subsample=0.9, 
        n_estimators=10_000,  
        learning_rate=0.1, 
        eval_metric="auc",
        early_stopping_rounds=100,
        alpha=1,
    )
    xgb_model.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],  
        verbose=100 
    )

    tabpfn_model = TabPFNClassifier(device='cuda')
    tabpfn_model.fit(x_train, y_train)

    # INFER OOF
    oof_xgb_tabpfn[test_index] = (xgb_model.predict_proba(x_valid)[:,1] + tabpfn_model.predict_proba(x_valid)[:,1])/2
    # INFER TEST
    pred_xgb_tabpfn += (xgb_model.predict_proba(x_test)[:,1] + tabpfn_model.predict_proba(x_test)[:,1])/2

# COMPUTE AVERAGE TEST PREDS
pred_xgb_tabpfn /= FOLDS


from sklearn.metrics import roc_auc_score
true = train2.rainfall.values
m = roc_auc_score(true, oof_xgb_tabpfn)
print(f"XGBoost and TabPFN CV Score AUC = {m:.3f}")


submission.rainfall = pred_tabpfn
submission.to_csv(f"submission_blend_xgboost_tabpfn.csv",index=False)


from cuml.svm import SVC, LinearSVC


m = train.rainfall.mean()
COLS = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
       'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
for c in COLS:
    n = f"{c}2"
    train[n] = train[c].map( orig.groupby(c).rainfall.mean() )
    train[n] = train[n].fillna(m)
    test[n] = test[c].map( orig.groupby(c).rainfall.mean() )
    test[n] = test[n].fillna(m)


test["winddirection"] = test["winddirection"].fillna(test.winddirection.mean())


RMV = ['rainfall','id', 'group']
FEATURES = [c for c in train.columns if not c in RMV]
print("Our features are:")
print( FEATURES )


%%time
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_svc = np.zeros(len(train))
pred_svc = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy().values.astype(np.float32)
    y_train = train.loc[train_index,"rainfall"].values.astype(np.float32)    
    x_valid = train.loc[test_index,FEATURES].copy().values.astype(np.float32)
    y_valid = train.loc[test_index,"rainfall"].values.astype(np.float32)
    x_test = test[FEATURES].copy().values.astype(np.float32)

    model = LinearSVC(C=0.1, probability=True)
    
    model.fit(
        x_train, y_train
    )

    # INFER OOF
    oof_svc[test_index] = model.predict_proba(x_valid)[:,1]
    # INFER TEST
    pred_svc += model.predict_proba(x_test)[:,1]

# COMPUTE AVERAGE TEST PREDS
pred_svc /= FOLDS


from sklearn.metrics import roc_auc_score
true = train.rainfall.values
m = roc_auc_score(true, oof_svc)
print(f"SVC CV Score AUC = {m:.3f}")


submission.rainfall = pred_svc
submission.to_csv(f"submission_rapids_svc.csv",index=False)


submission.rainfall = (pred_xgb + pred_tabpfn + pred_svc) / 3
submission.to_csv(f"submission_blend_xgboost_tabpfn_svc.csv",index=False)


from cuml.svm import SVC, LinearSVC
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import roc_auc_score


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
orig = pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


train['group'] = train['id']//365
orig['group'] = 6

train = train.drop(columns=['id'])

orig.columns = orig.columns.str.strip()

orig["rainfall"] = orig["rainfall"].map({'yes': 1, 'no': 0})


train = pd.concat([train, orig], ignore_index=True)


RMV = ['rainfall','id', 'group']
FEATURES = [c for c in train.columns if not c in RMV]
print("Our features are:")
print( FEATURES )


INTERACT = []

for i, c1 in enumerate(FEATURES):
    for j, c2 in enumerate(FEATURES[i+1:]):
        n = f"{c1}_{c2}"
        train[n] = train[c1]*train[c2]
        test[n] = test[c1]*test[c2]
        INTERACT.append(n)
print(f"There are {len(INTERACT)} interaction features: ")
print(INTERACT)


ADD = []
best_auc = 0
best_oof = None
best_pred = None

for k, col in enumerate(["baseline"] + INTERACT):
    
    FOLDS = train.group.nunique()
    kf = GroupKFold(n_splits=FOLDS)
    
    oof_svc = np.zeros(len(train))
    pred_svc = np.zeros(len(test))

    if col != "baseline": ADD.append(col)

    for i, (train_index, test_index) in enumerate(kf.split(train, groups = train.group)):
        #print("#"*25)
        #print(f"### Fold {i+1}")
        #print("#"*25)
        
        x_train = train.loc[train_index, FEATURES+ADD].copy()
        y_train = train.loc[train_index, "rainfall"]
        x_valid = train.loc[test_index, FEATURES+ADD].copy()
        y_valid = train.loc[test_index, "rainfall"]
        x_test = test[FEATURES+ADD].copy()

        for c in FEATURES+ADD:
            m = x_train[c].mean()
            s = x_train[c].std()

            x_train[c] = (x_train[c]-m)/s
            x_valid[c] = (x_valid[c]-m)/s
            x_test[c] = (x_test[c]-m)/s
            x_train[c] = x_train[c].fillna(x_train[c].mean())
            x_valid[c] = x_valid[c].fillna(x_valid[c].mean())
            x_test[c] = x_test[c].fillna(x_test[c].mean())


        model = SVC(C=0.1, probability=True, kernel="poly", degree=1)
        model.fit(x_train.values, y_train.values)

        oof_svc[test_index] = model.predict_proba(x_valid.values)[:,1]

        pred_svc += model.predict_proba(x_test.values)[:,1]

    pred_svc /= FOLDS

    true = train.rainfall.values
    m = roc_auc_score(true, oof_svc)

    if m>best_auc:
        print(f"NEW BEST with {col} at {m}")
        best_auc = m
        best_oof = oof_svc.copy
        best_pred = pred_svc.copy()
    else:
        print(f"Worse with {col} at {m}")
        ADD.remove(col)


print(f"We achieved CV SVC AUC = {best_auc:.4f} adding {len(ADD)} interactions features:")
print(ADD)


submission.rainfall = best_pred
submission.to_csv(f"submission_rapids_svc_feature_engineering.csv",index=False)


submission.rainfall = (pred_xgb + pred_tabpfn + best_pred) / 3
submission.to_csv(f"submission_blend_xgboost_tabpfn_svc(fe).csv",index=False)

