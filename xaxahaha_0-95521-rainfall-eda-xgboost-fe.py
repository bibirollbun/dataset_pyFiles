#Importing required libs
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBRegressor, XGBClassifier
from scipy.stats import rankdata
import warnings 
warnings.filterwarnings('ignore')


#Loading our data
path = '/kaggle/input/playground-series-s5e3'
train = pd.read_csv(f"{path}/train.csv")
test = pd.read_csv(f"{path}/test.csv")


train.head()


train.describe()


test.describe()


train.info()


test.info()


train.isnull().sum()


test.isnull().sum()


target = train.rainfall
features_names = train.columns.drop(['id', 'rainfall']).tolist()
features_names


for c in features_names:
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


# i deleted all of the FE


y = target
X = train.drop(columns=["rainfall",
                           "id"])
X_test = test.drop(columns=["id"])


FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
oof_xgb = np.zeros(len(train))
pred_xgb = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("-"*25)
    print(f"Fold {i+1}")
    print("-"*25)
    
    X_train = train.loc[train_index,features_names].copy()
    y_train = train.loc[train_index,"rainfall"]    
    X_valid = train.loc[test_index,features_names].copy()
    y_valid = train.loc[test_index,"rainfall"]
    X_test = test[features_names].copy()

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
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],  
        verbose=100 
    )

    # INFER OOF
    oof_xgb[test_index] = model.predict_proba(X_valid)[:,1]
    # INFER TEST
    pred_xgb += model.predict_proba(X_test)[:,1]

# COMPUTE AVERAGE TEST PREDS
pred_xgb /= FOLDS


true = train.rainfall.values
m = roc_auc_score(true, oof_xgb)
print(f"XGBoost CV Score AUC = {m:.3f}")


feature_importance = model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": features_names,  
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 5))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance")
plt.gca().invert_yaxis()  
plt.show()


best_public = pd.read_csv("/kaggle/input/lb-915-public-notebook/submission95427.csv")
display( best_public.head() )
best_public = best_public.rainfall.values


sub = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
sub.rainfall = -0.4 * rankdata( pred_xgb ) + 2.6 * rankdata( best_public )
sub.rainfall = rankdata( sub.rainfall ) / len(sub)
print( sub.shape )
sub.to_csv(f"submission_ensemble2.csv",index=False)
sub.head()

