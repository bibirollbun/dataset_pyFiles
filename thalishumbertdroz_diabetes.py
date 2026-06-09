pip install feature_engine


import os
import json
from datetime import datetime
import time
from functools import reduce

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import phik
from phik.report import plot_correlation_matrix

from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

import sklearn 
sklearn.set_config(enable_metadata_routing=True)
from sklearn.preprocessing import KBinsDiscretizer,PowerTransformer,StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score,log_loss
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

from feature_engine.outliers import OutlierTrimmer

from catboost import CatBoostClassifier, Pool, cv

from xgboost import XGBClassifier


TRAIN_DATA_FILEPATH = "/kaggle/input/playground-series-s5e12/train.csv"
TEST_DATA_FILEPATH = "/kaggle/input/playground-series-s5e12/test.csv"
SUBMISSIONS_PATH = "/kaggle/working/submissions/"
BINNING_PATH = "/kaggle/working/binnings/"
os.makedirs(SUBMISSIONS_PATH,exist_ok=True)
os.makedirs(BINNING_PATH,exist_ok=True)


def load_data(train=True,ret_id=False,map_income_level=True):
    if train:
        pdf = pd.read_csv(TRAIN_DATA_FILEPATH)
    else:
        pdf = pd.read_csv(TEST_DATA_FILEPATH)
    if not ret_id:
        pdf.drop("id",axis=1,inplace=True)
    if map_income_level:
        income_level_map = {
            "Low":0,
            "Lower-Middle":1,
            "Middle":2,
            "Upper-Middle":3,
            "High":4
        }
    pdf["income_level"] = pdf["income_level"].map(lambda x: income_level_map[x])
    return pdf

def plot_dists(DF,grid_size=(5,5),bins=25,figsize=(10,20)):
    fig,ax = plt.subplots(grid_size[1],grid_size[0],figsize=figsize)
    ix,iy = 0, 0
    for i_column,column in enumerate(TRAIN_DATA.columns):
        ax[iy,ix].hist(TRAIN_DATA[column],bins=bins)
        ax[iy,ix].tick_params(axis="x",rotation=90)
        ax[iy,ix].set_title(f"{column}")
        ax[iy,ix].set_xlabel("Value")
        ax[iy,ix].set_ylabel("Occurencies")
        if ix == grid_size[0]-1:
            iy = (iy + 1) % grid_size[1]
        ix = (ix + 1) % grid_size[0]
    plt.tight_layout()

def find_unnoised_binning(TRAIN_DATA,column="age",min_bins=10,max_bins=40):
    diffs=[]
    data = TRAIN_DATA[column]
    kde = gaussian_kde(data)
    for b in range(min_bins,max_bins+1):
        cut = pd.cut(data,bins=b)
        vc = cut.value_counts().sort_index()
        summ = vc.values.sum()
        kde_estim = []
        real = []
        for interval,count in vc.items():
            kde_estim.append(kde.integrate_box_1d(interval.left,interval.right))
            real.append(count/summ)
        kde_estim = np.array(kde_estim)
        real = np.array(real)
        diffs.append(np.abs(kde_estim-real).sum())
    diffs = np.array(diffs)/sum(diffs)
    peaks = find_peaks(1-diffs)[0]
    return (peaks+min_bins).tolist(), diffs[peaks]


TRAIN_DATA = load_data()
print(TRAIN_DATA.shape)
TRAIN_DATA.head()


TRAIN_DATA.isna().sum()


plot_dists(TRAIN_DATA,bins=45,grid_size=(3,9))


data_types = {
    # Interval
    "age": 'interval',
    "physical_activity_minutes_per_week":'interval',
    "diet_score":'interval',
    "sleep_hours_per_day":'interval',
    "screen_time_hours_per_day":'interval',
    "bmi":'interval',
    "waist_to_hip_ratio":'interval',
    "systolic_bp":'interval',
    "diastolic_bp":'interval',
    "heart_rate":'interval',
    "cholesterol_total":'interval',
    "hdl_cholesterol":'interval',
    "ldl_cholesterol":'interval',
    "triglycerides":'interval',

    # Ordinal
    "alcohol_consumption_per_week":'ordinal',
    "income_level":'ordinal',

    # Nominal
    "gender":'nominal',
    "ethnicity":'nominal',
    "education_level":'nominal',
    "smoking_status":'nominal',
    "employment_status":'nominal',
    "family_history_diabetes":'nominal',
    "hypertension_history":'nominal',
    "cardiovascular_history":'nominal',
    "family_history_diabetes":'nominal',

    # Target
    "diagnosed_diabetes":'nominal'
}



for col,data_type in data_types.items():
    if data_type == "interval":
        filepath = BINNING_PATH+col+".json"
        if not os.path.exists(filepath):
            vals = find_unnoised_binning(TRAIN_DATA,column=col)
            data = list(zip(vals[0],vals[1]))
            with open(filepath, 'w') as f:
                json.dump(data, f)


filepath = BINNING_PATH+"correlations.csv"
if not os.path.exists(filepath):
    column = []
    binning = []
    target_correlation = []
    interval_cols = [c for c,dtype in data_types.items() if dtype=="interval"]
    for c in interval_cols:
        filepath = BINNING_PATH+c+".json"
        with open(filepath, 'r') as f:
            binnings = json.load(f)
        for n_bins,loss in binnings:
            num_vars = [c]
            corr = phik.phik_from_array(
                TRAIN_DATA[num_vars[0]],
                TRAIN_DATA["diagnosed_diabetes"],
                num_vars = num_vars,
                bins={c:n_bins})
            target_correlation.append(corr)
            column.append(c)
            binning.append(n_bins)
    d = {'column': column, 'binning': binning, 'target_corr':target_correlation}
    df = pd.DataFrame(data=d)
    df.to_csv(filepath,index=False)
else:
    df = pd.read_csv(filepath)


def get_best_correlated_binning(TRAIN_DATA,df_corr):
    interval_binning = {}
    df_corr.sort_values(by="target_corr",inplace=True,ascending=False)
    df_corr.reset_index(drop=True,inplace=True)
    for row in df_corr.iterrows():
        col = row[1]["column"]
        if col not in interval_binning:
            interval_binning[col] = row[1]["binning"]
    return interval_binning
interval_binning = get_best_correlated_binning(TRAIN_DATA,df)
interval_cols = list(interval_binning)


phik_overview = TRAIN_DATA[interval_cols+["diagnosed_diabetes"]].phik_matrix(interval_cols=interval_cols,bins=interval_binning)
plot_correlation_matrix(phik_overview.values, x_labels=phik_overview.columns, y_labels=phik_overview.index, 
                        vmin=0, vmax=1, color_map='Blues', title=r'correlation $\phi_K$', fontsize_factor=0.8,
                        figsize=(10,7))


def get_default_features_corr(TRAIN_DATA,data_types,interval_cols,interval_binning,min_phik_target=0.1):
    col_target_corrs = []
    for c,dtype in data_types.items():
        if c == "diagnosed_diabetes":
            continue
        num_vars = []
        if dtype == "interval":
            num_vars.append(c)
        corr = phik.phik_from_array(
            TRAIN_DATA[c],
            TRAIN_DATA["diagnosed_diabetes"],
            num_vars = num_vars,
            bins=interval_binning)
        if corr >= min_phik_target:
            col_target_corrs.append((c,corr))
    pre_selected_cols = [c for c,_ in col_target_corrs]
    gphik = phik.global_phik_array(TRAIN_DATA[pre_selected_cols],interval_cols=interval_cols,bins=interval_binning)
    gphik = dict(list(zip(gphik[1].tolist(),gphik[0].ravel().tolist())))
    col_target_corrs = [(c,target_corr-gphik[c]) for c,target_corr in col_target_corrs]
    col_target_corrs.sort(key=lambda x: x[1],reverse=True) 
    return col_target_corrs    

def get_selected_features(col_target_corrs):
    col_target_corrs.sort(key=lambda x: x[1],reverse=True) 
    return [c for c,_ in col_target_corrs]



col_target_corrs = get_default_features_corr(
    TRAIN_DATA,
    data_types,
    interval_cols,
    interval_binning,
    min_phik_target=0.06
)
selected_features = get_selected_features(col_target_corrs)
dict_tc = dict(col_target_corrs[:len(selected_features)])
maxx = np.max(list(dict_tc.values()))
minn = np.min(list(dict_tc.values()))
maxx_minn = maxx-minn
for k,v in dict_tc.items():
    dict_tc[k] = ((v-minn)/maxx_minn-1/2)/2+1
dict_tc


phik_overview = TRAIN_DATA[selected_features+["diagnosed_diabetes"]].phik_matrix(interval_cols=interval_cols,bins=interval_binning)
plot_correlation_matrix(phik_overview.values, x_labels=phik_overview.columns, y_labels=phik_overview.index, 
                        vmin=0, vmax=1, color_map='Blues', title=r'correlation $\phi_K$', fontsize_factor=0.8,
                        figsize=(10,7))


cat_features = [c for c,dtype in data_types.items() if (dtype=="nominal") and (c in selected_features)]

params_catboost = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 200,
    "learning_rate": 0.1,
    "depth": 7,
    "l2_leaf_reg": 7,
    "random_seed": 42,
    "verbose": False,
    "feature_weights":dict_tc,
    #"grow_policy":"Depthwise",
    "random_strength":1.1
}

n_splits = 3
sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
report = {
    "train_loss":[],
    "val_loss":[],
    "val_auc":[],
}
for fold, (train_idx, val_idx) in enumerate(sss.split(TRAIN_DATA[selected_features], TRAIN_DATA["diagnosed_diabetes"]), 1):
    X_train, X_val = TRAIN_DATA[selected_features].iloc[train_idx], TRAIN_DATA[selected_features].iloc[val_idx]
    y_train, y_val = TRAIN_DATA["diagnosed_diabetes"].iloc[train_idx], TRAIN_DATA["diagnosed_diabetes"].iloc[val_idx]

    for c in cat_features:
        X_train[c] = X_train[c].astype("category")
        X_val[c] = X_val[c].astype("category")
    
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool   = Pool(X_val, y_val, cat_features=cat_features)
    
    model = CatBoostClassifier(**params_catboost)
    model.fit(train_pool, eval_set=val_pool, verbose=False)

    results = model.get_evals_result()
    report["val_auc"].append(np.array(results["validation"]["AUC"]))
    report["val_loss"].append(np.array(results["validation"]["Logloss"]))
    report["train_loss"].append(np.array(results["learn"]["Logloss"]))
    
for i,k in enumerate(report):
    report[k] = reduce(lambda x,y: x+y,report[k])/len(report[k])


fig,ax = plt.subplots(2,1)
ax[0].plot(report["train_loss"],label="train")
ax[0].plot(report["val_loss"],label="val",c="orange")
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("Logloss")
ax[0].legend()
ax[1].plot(report["val_auc"],c="orange")
ax[1].set_xlabel("Iteration")
ax[1].set_ylabel("AUC")
plt.tight_layout()


cat_features = [c for c,dtype in data_types.items() if (dtype=="nominal") and (c in selected_features)]

params_xgboost ={
    "n_estimators":200, 
    "tree_method":"hist",
    "learning_rate":0.1,
    "max_depth":4,
    "lambda":1.2,
    "eval_metric":["auc","logloss"],
    "enable_categorical":True,
    "max_cat_to_onehot":3,
    "objective":'binary:logistic',
    "seed":42
}

n_splits = 3
sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
report = {
    "train_loss":[],
    "val_loss":[],
    "val_auc":[],
}
for fold, (train_idx, val_idx) in enumerate(sss.split(TRAIN_DATA[selected_features], TRAIN_DATA["diagnosed_diabetes"]), 1):
    X_train, X_val = TRAIN_DATA[selected_features].iloc[train_idx], TRAIN_DATA[selected_features].iloc[val_idx]
    y_train, y_val = TRAIN_DATA["diagnosed_diabetes"].iloc[train_idx], TRAIN_DATA["diagnosed_diabetes"].iloc[val_idx]

    for c in cat_features:
        X_train[c] = X_train[c].astype("category")
        X_val[c] = X_val[c].astype("category")
    
    model = XGBClassifier(**params_xgboost)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val),(X_train, y_train)],verbose=False)
    
    results = model.evals_result()
    report["val_auc"].append(np.array(results["validation_0"]["auc"]))
    report["val_loss"].append(np.array(results["validation_0"]["logloss"]))
    report["train_loss"].append(np.array(results["validation_1"]["logloss"]))

for i,k in enumerate(report):
    report[k] = reduce(lambda x,y: x+y,report[k])/len(report[k])


fig,ax = plt.subplots(2,1)
ax[0].plot(report["train_loss"],label="train")
ax[0].plot(report["val_loss"],label="val",c="orange")
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("Logloss")
ax[0].legend()
ax[1].plot(report["val_auc"],c="orange")
ax[1].set_xlabel("Iteration")
ax[1].set_ylabel("AUC")
plt.tight_layout()


cat_features = [c for c,dtype in data_types.items() if (dtype=="nominal") and (c in selected_features)]

params_xgboost ={
    "n_estimators":200, 
    "tree_method":"hist",
    "learning_rate":0.1,
    "max_depth":4,
    "lambda":1.2,
    "eval_metric":["auc","logloss"],
    "enable_categorical":True,
    "max_cat_to_onehot":3,
    "objective":'binary:logistic',
    "seed":42
}
params_catboost = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 200,
    "learning_rate": 0.1,
    "depth": 7,
    "l2_leaf_reg": 7,
    "random_seed": 42,
    "verbose": False,
    "feature_weights":dict_tc,
    #"grow_policy":"Depthwise",
    "random_strength":1.1
}

n_splits = 1
sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
report = {
    "train_loss":[],
    "val_loss":[],
    "val_auc":[],
}
for fold, (train_idx, val_idx) in enumerate(sss.split(TRAIN_DATA[selected_features], TRAIN_DATA["diagnosed_diabetes"]), 1):
    X_train, X_val = TRAIN_DATA[selected_features].iloc[train_idx], TRAIN_DATA[selected_features].iloc[val_idx]
    y_train, y_val = TRAIN_DATA["diagnosed_diabetes"].iloc[train_idx], TRAIN_DATA["diagnosed_diabetes"].iloc[val_idx]

    for c in cat_features:
        X_train[c] = X_train[c].astype("category")
        X_val[c] = X_val[c].astype("category")
    
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool   = Pool(X_val, y_val, cat_features=cat_features)
    
    model_cat = CatBoostClassifier(**params_catboost)
    model_cat.fit(train_pool, eval_set=val_pool, verbose=False)
    
    model_xg = XGBClassifier(**params_xgboost)
    model_xg.fit(X_train, y_train, eval_set=[(X_val, y_val),(X_train, y_train)],verbose=False)
    
    pred_cat = model_cat.predict_proba(X_val)[:,1]
    pred_xg = model_xg.predict_proba(X_val)[:,1]


plt.scatter(pred_cat,pred_xg,c=y_val,alpha=0.5,s=0.5)



#### WRAPPERS
class CatBoostWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, cat_features=[], **params):
        self.cat_features = cat_features
        self.params = params
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        # sklearn
        self.classes_ = np.unique(y)
        # fit
        pool = Pool(X, y, cat_features=self.cat_features)
        self.model = CatBoostClassifier(cat_features=self.cat_features, **self.params)
        self.model.fit(pool, verbose=False)
        return self

    def predict(self, X):
        pool = Pool(X, cat_features=self.cat_features)
        return self.model.predict(pool)

    def predict_proba(self, X):
        pool = Pool(X, cat_features=self.cat_features)
        return self.model.predict_proba(pool)

#### MODEL
class Model(BaseEstimator, ClassifierMixin):
    def __init__(self,dict_features_weights,data_types,target_name,xgboost_params,catboost_params,stacker_params,ot_params):
        # Attributes
        self.dict_features_weights = dict_features_weights
        self.data_types = data_types
        self.interval_cols = [c for c,dtype in data_types.items() if (dtype=="interval") and (c != target_name)]
        self.nominal_cols = [c for c,dtype in data_types.items() if (dtype=="nominal") and (c != target_name)]
        self.target_name=target_name
        self.xgboost_params = xgboost_params
        self.catboost_params = catboost_params
        self.stacker_params = stacker_params
        self.ot_params = ot_params
        self.selected_features = [c for c,dtype in dict_features_weights.items()]
        self.selected_nominal_cols = [c for c in self.nominal_cols if c in self.selected_features]
        self.selected_interval_cols = [c for c in self.interval_cols if c in self.selected_features]
        self.model=None
        self.ot=None
        self.classes_=None
        
    def fit(self,X,y):
        # sklearn
        self.classes_ = np.unique(y)
        
        # Transform nominal columns
        for c in self.selected_nominal_cols:
            X.loc[:,c] = X[c].astype("category")

        # Instanciate
        self.ot = OutlierTrimmer(variables=self.selected_interval_cols, **self.ot_params)
        catboost = CatBoostWrapper(
            cat_features = self.selected_nominal_cols,
            **self.catboost_params
        )
        xgboost = XGBClassifier(
            **self.xgboost_params
        )
        clf_stacker = Pipeline([
            ("scaler",StandardScaler()),
            ("estimator",LogisticRegression(
                **self.stacker_params
            ))
        ])
        self.model = StackingClassifier(
            [
                ("catboost",catboost),
                ("xgboost",xgboost)
            ],
            final_estimator=clf_stacker,
            stack_method="predict_proba"
        )
        
        # Outliers
        self.ot.fit(X[self.selected_features])
        X,y = self.ot.transform_x_y(
            X[self.selected_features],
            y
        )
        # Optimize
        self.model.fit(
            X[self.selected_features],
            y
        ) 
        return self

    def predict(self, X):
        return self.model.predict(X[self.selected_features])

    def predict_proba(self, X):
        return self.model.predict_proba(X[self.selected_features])

TRAIN_DATA = load_data()
catboost_params={
    "loss_function":"Logloss",
    "iterations":10,
    "learning_rate":0.3,
    "depth":7,
    "l2_leaf_reg":7,
    "random_seed":42,
    "verbose":False,
    "feature_weights":dict_tc,
    "random_strength":1.1
}
xgboost_params = {
    "n_estimators":10,
    "tree_method":"hist",
    "learning_rate":0.3,
    "colsample_bylevel":0.8,
    "enable_categorical":True,
    "seed":42,
    "max_cat_to_onehot":3,
    "objective":'binary:logistic',
}
stacker_params = {
    "solver":"newton-cholesky",
    "max_iter":150,
    "C":0.8,
    "cv":StratifiedShuffleSplit(n_splits=5,test_size=0.2)
}
ot_params = {
    "capping_method":'mad', 
    "tail":'both', 
    "fold":6,
}

target_name = "diagnosed_diabetes"
model = Model(
    dict_tc,
    data_types,
    target_name,
    xgboost_params,
    catboost_params,
    stacker_params,
    ot_params
)
model.fit(TRAIN_DATA[selected_features],TRAIN_DATA[target_name])


TEST_DATA = load_data(train=False)
model.predict(TEST_DATA)








n_splits = 5
sss = StratifiedShuffleSplit(n_splits=n_splits, shuffle=True, random_state=42)

aucs = []

# Boucle CV
for fold, (train_idx, val_idx) in enumerate(sss.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Entraînement
    model.fit(X_train, y_train)
    
    # Prédiction
    y_pred = model.predict_proba(X_val)[:, 1]
    
    # Calcul AUC
    auc = roc_auc_score(y_val, y_pred)
    print(f"Fold {fold} AUC: {auc:.4f}")
    aucs.append(auc)








TEST_DATA = load_data(train=False,ret_id=True)
preds = model.predict_proba(TEST_DATA[selected_features])


data = np.ndarray((len(TEST_DATA),2))
data[:,0] = TEST_DATA["id"]
data[:,1] = preds[:,1]
submission = pd.DataFrame(data,columns=["id","diagnosed_diabetes"])
submission["id"] = submission["id"].astype(int)

current_timestamp = int(time.time())
date_time = datetime.fromtimestamp(current_timestamp)
formatted = date_time.strftime("%m_%d_%Y-%H_%M_%S")

submission.to_csv(SUBMISSIONS_PATH+"submission-"+formatted+".csv",index=False)

