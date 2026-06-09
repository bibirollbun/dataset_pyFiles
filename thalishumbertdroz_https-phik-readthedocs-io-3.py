# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

os.system("ls -l /kaggle/working/submissions")


TRAIN_DATA_FILEPATH = "/kaggle/input/playground-series-s5e11/train.csv"
TEST_DATA_FILEPATH = "/kaggle/input/playground-series-s5e11/test.csv"
SUBMISSION_FILEPATH = "/kaggle/working/submissions/submission.csv"
os.makedirs("/kaggle/working/submissions",exist_ok=True)


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd 
from sklearn.preprocessing import PowerTransformer,StandardScaler,KBinsDiscretizer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import davies_bouldin_score,accuracy_score,roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer,make_column_transformer
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import phik
from phik import resources
from phik.binning import bin_data
from phik.report import plot_correlation_matrix
from phik import report
import itertools


# Tabular data and shape
def load_data(train=True):
    if train:
        filepath = TRAIN_DATA_FILEPATH
    else:
        filepath = TEST_DATA_FILEPATH
    DATA = pd.read_csv(filepath)
    DATA.drop("id",axis=1,inplace=True)
    return DATA
TRAIN_DATA = load_data()
TEST_DATA = load_data(False)
print("Train shape:\n",TRAIN_DATA.shape)
print("Test shape:\n",TEST_DATA.shape)
TRAIN_DATA.head()


# Data types
TRAIN_DATA.dtypes


TRAIN_DATA.isna().sum()


TRAIN_DATA.isnull().sum()


(TRAIN_DATA == "").sum()


((TRAIN_DATA == float("inf")) | (TRAIN_DATA == -float("inf"))).sum()


TRAIN_DATA.duplicated().any()


def plot_dists(DF,grid_size=(2,6),bins=25,figsize=(10,20)):
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

plot_dists(TRAIN_DATA)


data_types = {
    "annual_income": 'interval',
    "debt_to_income_ratio":'interval',
    "credit_score":'interval',
    "loan_amount":'interval',
    "interest_rate":'interval',
    "grade_subgrade":'ordinal',
    "gender":'nominal',
    "marital_status":'nominal',
    "education_level":'nominal',
    "employment_status":'nominal',
    "loan_purpose":'nominal'
}
interval_binning = {
    "annual_income": 25,
    "debt_to_income_ratio":25,
    "credit_score":15,
    "loan_amount":15,
    "interest_rate":15,
}
# retrieve the interval columns
interval_cols = [col for col, v in data_types.items() if v=='interval' and col in TRAIN_DATA.columns]
interval_cols


# https://nbviewer.org/github/kaveio/phik/blob/master/phik/notebooks/phik_tutorial_basic.ipynb
plt.rc('text', usetex=False)

# automatical binning of the interval data
data_binned, binning_dict = bin_data(TRAIN_DATA, cols=interval_cols, retbins=True, bins=interval_binning)

# plot the correlations
n=0
for i in range(len(TRAIN_DATA.columns)):
    n=n+i
    
ncols=3
nrows=int(np.ceil(n/ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(15,4*nrows))
ndecimals = 0

for i, comb in enumerate(itertools.combinations(data_binned.columns.values, 2)):
    
    c = int(i%ncols)
    r = int((i-c)/ncols )

    # get data
    c0, c1 = comb
    datahist = data_binned.groupby([c0,c1])[c0].count().to_frame().unstack().fillna(0)
    datahist.columns = datahist.columns.droplevel()
    
    # plot data
    img = axes[r][c].pcolormesh(datahist.values, edgecolor='w', linewidth=1)
    
    # axis ticks and tick labels
    if c0 in binning_dict.keys():
        ylabels = ['{1:.{0}f}_{2:.{0}f}'.format(ndecimals, binning_dict[c0][i][0], binning_dict[c0][i][1])
                   for i in range(len(binning_dict[c0]))]
    else:
        ylabels = datahist.index

    if c1 in binning_dict.keys():        
        xlabels = ['{1:.{0}f}_{2:.{0}f}'.format(ndecimals, binning_dict[c1][i][0], binning_dict[c1][i][1])
                    for i in range(len(binning_dict[c1]))]
    else:
        xlabels = datahist.columns
    
    # axis labels
    axes[r][c].set_yticks(np.arange(len(ylabels)) + 0.5)
    axes[r][c].set_xticks(np.arange(len(xlabels)) + 0.5)
    axes[r][c].set_xticklabels(xlabels, rotation='vertical')
    axes[r][c].set_yticklabels(ylabels, rotation='horizontal')    
    axes[r][c].set_xlabel(datahist.columns.name)
    axes[r][c].set_ylabel(datahist.index.name)    
    axes[r][c].set_title('data')
    
plt.tight_layout()


phik_overview = TRAIN_DATA.phik_matrix(interval_cols=interval_cols,bins=interval_binning)
phik_overview


plot_correlation_matrix(phik_overview.values, x_labels=phik_overview.columns, y_labels=phik_overview.index, 
                        vmin=0, vmax=1, color_map='Blues', title=r'correlation $\phi_K$', fontsize_factor=0.8,
                        figsize=(10,7))


significance_overview = TRAIN_DATA.significance_matrix(interval_cols=interval_cols)
significance_overview


plot_correlation_matrix(significance_overview.fillna(0).values, x_labels=significance_overview.columns, 
                        y_labels=significance_overview.index, vmin=-5, vmax=5, title='significance', 
                        usetex=False, fontsize_factor=0.8, figsize=(10,7))


data_types = {
    "annual_income": 'interval',
    "debt_to_income_ratio":'interval',
    "credit_score":'interval',
    "loan_amount":'interval',
    "interest_rate":'interval',
    "grade_subgrade":'ordinal',
    "gender":'nominal',
    "marital_status":'nominal',
    "education_level":'nominal',
    "employment_status":'nominal',
    "loan_purpose":'nominal'
}
interval_binning = {
    "annual_income": 25,
    "debt_to_income_ratio":25,
    "credit_score":15,
    "loan_amount":15,
    "interest_rate":15,
}
# retrieve the interval columns
interval_cols = [col for col, v in data_types.items() if v=='interval' and col in TRAIN_DATA.columns]
interval_cols
n_features_to_remove = 6
TRAIN_DATA = load_data()
phik_overview = TRAIN_DATA.phik_matrix(interval_cols=interval_cols,bins=interval_binning)
selected_features = phik_overview["loan_paid_back"].sort_values()[n_features_to_remove:].index.values
TRAIN_DATA = TRAIN_DATA[selected_features]
selected_features


phik_overview = TRAIN_DATA.phik_matrix(interval_cols=interval_cols,bins=interval_binning)
plot_correlation_matrix(phik_overview.values, x_labels=phik_overview.columns, y_labels=phik_overview.index, 
                        vmin=0, vmax=1, color_map='Blues', title=r'correlation $\phi_K$', fontsize_factor=0.8,
                        figsize=(10,7))


class AnomalyCreator():
    def __init__(self,train_data,interval_cols:list[str],interval_binning:dict,min_phik:float = 0.5):
        # params
        self.phik_matrix = None
        self.train_data = train_data
        self.interval_cols = interval_cols
        self.interval_binning = interval_binning
        self.column_pairs = []
        self.column_pairs_outlier_signifs = []
        self.column_pairs_transform_sums = []
        self.min_phik = min_phik
        # init
        self.set_phik_matrix()
        self.set_column_pairs()
        
    def fit(self):
        for col1,col2 in self.column_pairs:
            outlier_signifs, binning_dict = self.train_data[[col1,col2]].outlier_significance_matrix(interval_cols=self.interval_cols,bins=self.interval_binning,retbins=True)
            data = {
                "outlier_signifs":outlier_signifs,
                "binning_dict":binning_dict,
            }
            self.column_pairs_outlier_signifs.append(data)
        res = self.transform(self.train_data,fitted=False)
        for anomaly in res:
            self.column_pairs_transform_sums.append(np.sum(np.abs(anomaly)).item())

    def _get_cat_bins(self,data,unique_cats,col):
        unique_cats.sort()
        unique_cats = unique_cats.tolist()
        col_bins = data[col].map(lambda x: unique_cats.index(x))
        return col_bins
        
    def _get_interval_bins(self,data,i_pair,col):
        binning_dict_col = self.column_pairs_outlier_signifs[i_pair]["binning_dict"][col]
        col_intervals = pd.IntervalIndex.from_tuples(binning_dict_col)
        col_bins = pd.cut(data[col], bins=col_intervals).cat.codes  
        return col_bins
    
    def transform(self,data,fitted=True):
        results = []
        for i_pair,(col1,col2) in enumerate(self.column_pairs):
            result = np.full(len(data), 0.0)
            if col1 in self.interval_cols:
                col1_bins = self._get_interval_bins(data,i_pair,col1)
                if col2 in self.interval_cols:
                    col2_bins = self._get_interval_bins(data,i_pair,col2)
                    mask = (col1_bins >= 0) & (col2_bins >= 0)
                    result[mask] = self.column_pairs_outlier_signifs[i_pair]["outlier_signifs"].values[col1_bins[mask], col2_bins[mask]]
                else:
                    col2_unique_cats = self.train_data[col2].unique()
                    col2_bins = self._get_cat_bins(data,col2_unique_cats,col2)
                    mask = (col1_bins >= 0) & data[col2].isin(col2_unique_cats)
                    result[mask] = self.column_pairs_outlier_signifs[i_pair]["outlier_signifs"].values[col1_bins[mask], col2_bins[mask]]
            else:
                col1_unique_cats = self.train_data[col1].unique()
                col1_bins = self._get_cat_bins(data,col1_unique_cats,col1)
                if col2 in self.interval_cols:
                    col2_bins = self._get_interval_bins(data,i_pair,col2)
                    mask = data[col1].isin(col1_unique_cats) & (col2_bins >= 0)
                else:
                    col2_unique_cats = self.train_data[col2].unique()
                    col2_bins = self._get_cat_bins(data,col2_unique_cats,col2)
                    mask = data[col1].isin(col1_unique_cats) & data[col2].isin(col2_unique_cats)
            result[mask] = self.column_pairs_outlier_signifs[i_pair]["outlier_signifs"].values[col1_bins[mask], col2_bins[mask]]
            if fitted:
                result = result / self.column_pairs_transform_sums[i_pair]
            results.append(result)
        return results
            
        
    def set_column_pairs(self,target_name="loan_paid_back"):
        self.column_pairs = []
        for i,c in enumerate(self.phik_matrix.columns):
            for d in self.phik_matrix.columns[i+1:]:
                col1 = self.phik_matrix.columns[i]
                col2 = d
                if (col1 != target_name) and (col2 != target_name):
                    if self.phik_matrix[col1][col2] >= self.min_phik:
                        self.column_pairs.append((col1,col2))
        
    def set_phik_matrix(self):
        self.phik_matrix = self.train_data.phik_matrix(interval_cols=self.interval_cols,bins=self.interval_binning)


ac = AnomalyCreator(TRAIN_DATA,interval_cols,interval_binning,min_phik=0.1)
ac.fit()


res = ac.transform(TRAIN_DATA)
#res_test = ac.transform(TEST_DATA)


new_features = []
for i,(col1,col2) in enumerate(ac.column_pairs):
    new_features.append(f"{col1}_{col2}_anomaly")
    TRAIN_DATA[f"{col1}_{col2}_anomaly"] = res[i]
    #TEST_DATA[f"{col1}_{col2}_anomaly"] = res_test[i]
#TEST_DATA.fillna(0,inplace=True)


new_features_interval_cols = interval_cols+new_features
new_features_interval_binning = interval_binning.copy()
for f in new_features:
    new_features_interval_binning[f] = 15

phik_overview = TRAIN_DATA.phik_matrix(interval_cols=new_features_interval_cols,bins=new_features_interval_binning)
plot_correlation_matrix(phik_overview.values, x_labels=phik_overview.columns, y_labels=phik_overview.index, 
                        vmin=0, vmax=1, color_map='Blues', title=r'correlation $\phi_K$', fontsize_factor=0.8,
                        figsize=(10,7))


TRAIN_DATA.columns


pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=2)),
])
X_pca = pipeline.fit_transform(TRAIN_DATA[[c for c in interval_cols if c in selected_features]])
X_pca_new = pipeline.fit_transform(TRAIN_DATA[[c for c in interval_cols if c in selected_features]+new_features])


fig,ax = plt.subplots(1,2,figsize=(12,5))
ax[0].scatter(X_pca[:,0],X_pca[:,1],c=TRAIN_DATA["loan_paid_back"],s=2,alpha=0.5)
ax[1].scatter(X_pca_new[:,0],X_pca_new[:,1],c=TRAIN_DATA["loan_paid_back"],s=2,alpha=0.5)
ax[0].set_xlabel("PC0")
ax[0].set_ylabel("PC1")
ax[0].set_title("selected features")
ax[1].set_xlabel("PC0")
ax[1].set_ylabel("PC1")
ax[1].set_title("with anomaly features")
plt.tight_layout()


std_scaler = StandardScaler()
X = std_scaler.fit_transform(TRAIN_DATA[[c for c in interval_cols if c in selected_features]])
X_new = std_scaler.fit_transform(TRAIN_DATA[[c for c in interval_cols if c in selected_features]+new_features])
score = davies_bouldin_score(X,TRAIN_DATA["loan_paid_back"])
score_new = davies_bouldin_score(X_new,TRAIN_DATA["loan_paid_back"])
print(
    f"selected features: {score}",
    "\n"
    f"with anomaly features: {score_new}"
)


anomaly_columns = [c for c in TRAIN_DATA.columns if "_anomaly" in c]
TRAIN_DATA["anomaly_sum"] = 0
for c in anomaly_columns:
    TRAIN_DATA["anomaly_sum"] += TRAIN_DATA[c]
TRAIN_DATA["anomaly_sum"].hist(bins=20)


binner = KBinsDiscretizer(
    n_bins=30,
    encode='ordinal',
    strategy='quantile',
    subsample=None
)
binner.fit(TRAIN_DATA[["anomaly_sum"]])
X_train_binned = binner.transform(TRAIN_DATA[["anomaly_sum"]])



TRAIN_DATA["anomaly_sum"]=X_train_binned


phik_overview = TRAIN_DATA[list(selected_features)+["anomaly_sum"]].phik_matrix(interval_cols=interval_cols,bins=interval_binning)
plot_correlation_matrix(phik_overview.values, x_labels=phik_overview.columns, y_labels=phik_overview.index, 
                        vmin=0, vmax=1, color_map='Blues', title=r'correlation $\phi_K$', fontsize_factor=0.8,
                        figsize=(10,7))


pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=2)),
])
X_pca = pipeline.fit_transform(TRAIN_DATA[[c for c in interval_cols if c in selected_features]])
X_pca_new = pipeline.fit_transform(TRAIN_DATA[[c for c in interval_cols if c in selected_features]+["anomaly_sum"]])


fig,ax = plt.subplots(1,2,figsize=(12,5))
ax[0].scatter(X_pca[:,0],X_pca[:,1],c=TRAIN_DATA["loan_paid_back"],s=2,alpha=0.5)
ax[1].scatter(X_pca_new[:,0],X_pca_new[:,1],c=TRAIN_DATA["loan_paid_back"],s=2,alpha=0.5)
ax[0].set_xlabel("PC0")
ax[0].set_ylabel("PC1")
ax[0].set_title("selected features")
ax[1].set_xlabel("PC0")
ax[1].set_ylabel("PC1")
ax[1].set_title("with anomaly sum")
plt.tight_layout()


std_scaler = StandardScaler()
X = std_scaler.fit_transform(TRAIN_DATA[[c for c in interval_cols if c in selected_features]])
X_new = std_scaler.fit_transform(TRAIN_DATA[[c for c in interval_cols if c in selected_features]+new_features])
score = davies_bouldin_score(X,TRAIN_DATA["loan_paid_back"])
score_new = davies_bouldin_score(X_new,TRAIN_DATA["loan_paid_back"])
print(
    f"selected features: {score}",
    "\n"
    f"with anomaly sum: {score_new}"
)


TRAIN_DATA = load_data()
data_types = {
    "annual_income": 'interval',
    "debt_to_income_ratio":'interval',
    "credit_score":'interval',
    "loan_amount":'interval',
    "interest_rate":'interval',
    "grade_subgrade":'ordinal',
    "gender":'nominal',
    "marital_status":'nominal',
    "education_level":'nominal',
    "employment_status":'nominal',
    "loan_purpose":'nominal'
}
interval_binning = {
    "annual_income": 25,
    "debt_to_income_ratio":25,
    "credit_score":15,
    "loan_amount":15,
    "interest_rate":15,
}
# retrieve the interval columns
interval_cols = [col for col, v in data_types.items() if v=='interval' and col in TRAIN_DATA.columns]
interval_cols
n_features_to_remove = 6
phik_overview = TRAIN_DATA.phik_matrix(interval_cols=interval_cols,bins=interval_binning)
selected_features = phik_overview["loan_paid_back"].sort_values()[n_features_to_remove:].index.values
TRAIN_DATA = TRAIN_DATA[selected_features]
selected_features


TRAIN_DATA = load_data()[selected_features]
grade_subgrade_unique = TRAIN_DATA["grade_subgrade"].unique().tolist()
grade_subgrade_unique.sort()
TRAIN_DATA["grade_subgrade"] = TRAIN_DATA["grade_subgrade"].map(lambda x: grade_subgrade_unique.index(x))


# adding a column to stratify
#TRAIN_DATA["train_label"] = TRAIN_DATA["loan_paid_back"].astype(str)+TRAIN_DATA["grade_subgrade"].astype("str")

TRAIN_DATA["train_label"] = \
TRAIN_DATA["loan_paid_back"].astype(str)+ \
pd.qcut(TRAIN_DATA["interest_rate"],5).astype("str")+ \
pd.qcut(TRAIN_DATA["credit_score"],5).astype(str)

#TRAIN_DATA["train_label"] = \
#TRAIN_DATA["loan_paid_back"].astype(str) + \
#TRAIN_DATA["employment_status"].astype(str)



TRAIN_DATA["train_label"].value_counts()


data_types = {
    "annual_income": 'interval',
    "debt_to_income_ratio":'interval',
    "credit_score":'interval',
    "loan_amount":'interval',
    "interest_rate":'interval',
    "anomaly":'interval',
    "pca0":'interval',
    "pca1":'interval',
    "grade_subgrade":'ordinal',
    "gender":'nominal',
    "marital_status":'nominal',
    "education_level":'nominal',
    "employment_status":'nominal',
    "loan_purpose":'nominal'
}
interval_binning = {
    "annual_income": 25,
    "debt_to_income_ratio":25,
    "credit_score":15,
    "loan_amount":15,
    "interest_rate":15,
    "anomaly":15,
}
cat_cols = [
    "employment_status",
    #"loan_purpose"
]
interval_cols = [col for col, v in data_types.items() if v=='interval']
interval_cols

pipeline_pca = Pipeline([
    ("scaler",PowerTransformer(method='yeo-johnson', standardize=True)),
    ("PCA",PCA(n_components=0.9))
])

pipeline = Pipeline([
    ("model", CatBoostClassifier(
        iterations=600,
        learning_rate=0.045,
        depth=10,
        eval_metric="AUC",
        verbose=200,
        loss_function="Logloss",
        cat_features=cat_cols,
        use_best_model=True,
        l2_leaf_reg=8,
        random_strength=1.8
    ))
])

n_splits = 1
min_phik = 0.1
sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=44)
for i, (train_index, test_index) in enumerate(sss.split(TRAIN_DATA, TRAIN_DATA["train_label"])):
    train_split = TRAIN_DATA.loc[train_index,:]
    test_split = TRAIN_DATA.loc[test_index,:]
    train_split.drop("train_label",axis=1,inplace=True)
    test_split.drop("train_label",axis=1,inplace=True)
    # calculate anomaly column
    ac = AnomalyCreator(train_split,interval_cols,interval_binning,min_phik=min_phik)
    ac.fit()
    res_train = ac.transform(train_split)
    res_test = ac.transform(test_split)
    anomaly_columns = []
    for i,(col1,col2) in enumerate(ac.column_pairs):
        anomaly_columns.append(f"{col1}_{col2}_anomaly")
        train_split[f"{col1}_{col2}_anomaly"] = res_train[i]
        test_split[f"{col1}_{col2}_anomaly"] = res_test[i]
    preprocess = make_column_transformer(
        (pipeline_pca, anomaly_columns),
        #(anomaly_pip,["anomaly_mean"]),
        remainder="passthrough",verbose_feature_names_out=False
    )
    #train_split.drop(anomaly_columns,axis=1,inplace=True)
    #test_split.drop(anomaly_columns,axis=1,inplace=True)
    # fit the model
    y_train = train_split["loan_paid_back"]
    train_split.drop("loan_paid_back",axis=1,inplace=True)
    y_test = test_split["loan_paid_back"]
    test_split.drop("loan_paid_back",axis=1,inplace=True)
    train_split = pd.DataFrame(preprocess.fit_transform(train_split))
    train_split.columns = preprocess.get_feature_names_out()
    test_split = pd.DataFrame(preprocess.transform(test_split))
    test_split.columns = preprocess.get_feature_names_out()
    """phik_overview = train_split.phik_matrix(interval_cols=interval_cols,bins=interval_binning)
    plot_correlation_matrix(phik_overview.values, x_labels=phik_overview.columns, y_labels=phik_overview.index, 
                        vmin=0, vmax=1, color_map='Blues', title=r'correlation $\phi_K$', fontsize_factor=0.8,
                        figsize=(10,7))
    break"""
    pipeline.fit(train_split,y_train,model__eval_set=(test_split, y_test))


results = pipeline[0].get_evals_result()
results["validation"].keys()

fig,ax = plt.subplots(2,1)
ax[0].plot(results["learn"]["Logloss"],alpha=0.5,label="train")
ax[0].plot(results["validation"]["Logloss"],alpha=0.5,label="val")
ax[1].plot(results["validation"]["AUC"])
#ax[0].set_ylim(0.2,0.9)
ax[0].legend()
plt.tight_layout()
np.max(results["validation"]["AUC"])


# load data
TRAIN_DATA = load_data()
TEST_DATA = load_data(False)

# Set the columns types
data_types = {
    "annual_income": 'interval',
    "debt_to_income_ratio":'interval',
    "credit_score":'interval',
    "loan_amount":'interval',
    "interest_rate":'interval',
    "anomaly":'interval',
    "pca0":'interval',
    "pca1":'interval',
    "grade_subgrade":'ordinal',
    "gender":'nominal',
    "marital_status":'nominal',
    "education_level":'nominal',
    "employment_status":'nominal',
    "loan_purpose":'nominal'
}
interval_binning = {
    "annual_income": 25,
    "debt_to_income_ratio":25,
    "credit_score":15,
    "loan_amount":15,
    "interest_rate":15,
    "anomaly":15,
}
cat_cols = [
    "employment_status",
    #"loan_purpose"
]
interval_cols = [col for col, v in data_types.items() if v=='interval']
interval_cols


# Select the features
n_features_to_remove = 6
phik_overview = TRAIN_DATA.phik_matrix(interval_cols=interval_cols,bins=interval_binning)
selected_features = phik_overview["loan_paid_back"].sort_values()[n_features_to_remove:].index.values
TRAIN_DATA = TRAIN_DATA[selected_features]
selected_features = selected_features.tolist()
selected_features.remove("loan_paid_back")
TEST_DATA = TEST_DATA[selected_features]


# Convert grade_subgrade to integers
grade_subgrade_unique = TRAIN_DATA["grade_subgrade"].unique().tolist()
grade_subgrade_unique.sort()
TRAIN_DATA["grade_subgrade"] = TRAIN_DATA["grade_subgrade"].map(lambda x: grade_subgrade_unique.index(x))
TEST_DATA["grade_subgrade"] = TEST_DATA["grade_subgrade"].map(lambda x: grade_subgrade_unique.index(x))


# model
pipeline_pca = Pipeline([
    ("scaler",PowerTransformer(method='yeo-johnson', standardize=True)),
    ("PCA",PCA(n_components=4))
])

pipeline = Pipeline([
    ("model", CatBoostClassifier(
        iterations=1500,
        learning_rate=0.035,
        depth=9,
        eval_metric="AUC",
        verbose=200,
        loss_function="Logloss",
        cat_features=cat_cols,
        l2_leaf_reg=6,
        random_strength=1.3
    ))
])

# calculate anomaly columns based on phik threshold
min_phik = 0.1
ac = AnomalyCreator(TRAIN_DATA,interval_cols,interval_binning,min_phik=min_phik)
ac.fit()
res_train = ac.transform(TRAIN_DATA)
res_test = ac.transform(TEST_DATA)
anomaly_columns = []
for i,(col1,col2) in enumerate(ac.column_pairs):
    anomaly_columns.append(f"{col1}_{col2}_anomaly")
    TRAIN_DATA[f"{col1}_{col2}_anomaly"] = res_train[i]
    TEST_DATA[f"{col1}_{col2}_anomaly"] = res_test[i]

# apply PCA on the anomaly columns
preprocess = make_column_transformer(
    (pipeline_pca, anomaly_columns),
    remainder="passthrough",verbose_feature_names_out=False
)
# fit the model
y_train = TRAIN_DATA["loan_paid_back"]
TRAIN_DATA.drop("loan_paid_back",axis=1,inplace=True)
TRAIN_DATA = pd.DataFrame(preprocess.fit_transform(TRAIN_DATA))
TRAIN_DATA.columns = preprocess.get_feature_names_out()
TEST_DATA = pd.DataFrame(preprocess.transform(TEST_DATA))
TEST_DATA.columns = preprocess.get_feature_names_out()
"""phik_overview = train_split.phik_matrix(interval_cols=interval_cols,bins=interval_binning)
plot_correlation_matrix(phik_overview.values, x_labels=phik_overview.columns, y_labels=phik_overview.index, 
                    vmin=0, vmax=1, color_map='Blues', title=r'correlation $\phi_K$', fontsize_factor=0.8,
                    figsize=(10,7))
break"""
pipeline.fit(TRAIN_DATA,y_train)


td = pd.read_csv(TEST_DATA_FILEPATH)
data = np.ndarray((len(td),2))
data[:,0] = td["id"]
data[:,1] = pipeline.predict_proba(TEST_DATA)[:,1]
submission = pd.DataFrame(data,columns=["id","loan_paid_back"])
submission["id"] = submission["id"].astype(int)
submission.to_csv(SUBMISSION_FILEPATH,index=False)

