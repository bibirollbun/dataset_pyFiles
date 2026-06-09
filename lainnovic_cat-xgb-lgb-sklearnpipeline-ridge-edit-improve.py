import matplotlib.pyplot as plt
from itertools import permutations
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings, os, gc, sys, math, json, random, itertools
from pathlib import Path
from sklearn.metrics import roc_auc_score,confusion_matrix
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import numpy as np, pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold,GridSearchCV,cross_val_predict
from sklearn.metrics import average_precision_score, make_scorer,log_loss,precision_recall_curve

import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

warnings.filterwarnings('ignore')

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

ONLINE = bool(os.environ.get('KAGGLE_KERNEL_RUN_TYPE'))

sys.path.insert(0, '../')
%load_ext autoreload
%autoreload 2
%config Completer.use_jedi = False

if not ONLINE:
    from viz.plot import *

DATA_ROOT = Path("/kaggle/input/playground-series-s5e8") if ONLINE else Path("..")
WORK_ROOT = Path("/kaggle/working") if ONLINE else Path("..")

def load_files():
    train= pd.read_csv(DATA_ROOT / "train.csv")
    test = pd.read_csv(DATA_ROOT / "test.csv")
    if ONLINE:
        original =pd.read_csv("/kaggle/input/bank-marketing-dataset-full/bank-full.csv", sep=';')
        save_path       = WORK_ROOT
        submission_path = WORK_ROOT / "submission.csv"
    else:
        original = pd.read_csv(DATA_ROOT / "original.csv", sep=";") ##GOT A WEIRD SEP HERE
        save_path       = WORK_ROOT / "OOF_PRED"
        submission_path = WORK_ROOT / "SUBMISSION" / "submission.csv"

    return train, test, original, save_path, submission_path

train_raw, test_raw, original_raw, save_path, submission_path = load_files()

train,test,original = train_raw.copy(),test_raw.copy(),original_raw.copy()

for df in [train,test,original]:
    display(df.head(3))



# original['y'] = original['y'].map({'yes': 1, 'no': 0}).astype(int)
# train = pd.concat([train.drop(columns=['id']),original], axis=0).reset_index(drop=True)


focus_cols = ['age','job','marital','education','default','balance','housing','loan','contact','day','month',
 'duration','campaign','pdays','previous','poutcome']

cat_cols = [c for c in focus_cols if train[c].dtype == "object"]
num_cols = [c for c in focus_cols if train[c].dtype != "object"]

y = train["y"]


def scan_df(df, name="df"):
    """
    Print NaN counts and [min / max] counts
    for every numeric column in `df`.
    """
    num_cols = df.select_dtypes(include=np.number).columns

    print(f"\n=== {name} ({len(df):,} rows) ===")

    nan_ct = df[num_cols].isna().sum()
    print("\nNaN counts:")
    print(nan_ct[nan_ct > 0].to_string() if nan_ct.any() else "  (none)")
    
    extreme_ct = {}
    for col in num_cols:
        col_min, col_max = df[col].min(), df[col].max()
        extreme_ct[col] = ((df[col] == col_min) | (df[col] == col_max)).sum()
    extreme_ser = pd.Series(extreme_ct, name="min_or_max_ct")
    extreme_ser = extreme_ser[extreme_ser > 0]
    
    print("\nExtreme value counts (equal to that column’s min or max):")
    print(extreme_ser.to_string() if not extreme_ser.empty else "  (none)")

for d, n in [(train, "train"), (test, "test"), (original, "original")]:
    scan_df(d, n)



train.info()


def bar_percent_grid(df, cat_list, target='y', cols=2):
    """
    Draw stacked % bar charts for each categorical feature and lay them out
    in a grid with `cols` plots per row.  Call with, e.g.,
    bar_percent_grid(train, cat_cols, cols=2).
    """
    import math, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

    rows = math.ceil(len(cat_list) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3), squeeze=False)

    for i, cat in enumerate(cat_list):
        r, c = divmod(i, cols)

        tab = (pd.crosstab(df[cat], df[target], normalize='index') * 100
               ).reset_index().melt(id_vars=cat,
                                    var_name=target,
                                    value_name='percent')

        sns.barplot(data=tab, x=cat, y='percent', hue=target,
                    dodge=False, palette='Set2', ax=axes[r][c])

        axes[r][c].set_ylabel('% within ' + cat)
        axes[r][c].tick_params(axis='x', rotation=30, labelsize=8)
        axes[r][c].legend_.remove()

    # hide any leftover blank axes
    for j in range(i + 1, rows * cols):
        fig.delaxes(axes[divmod(j, cols)])

    # single shared legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.tight_layout()
    plt.show()
    
#bar_percent_grid(train, cat_cols, cols=2)   # two charts per row



#This is a weird thing but as only little rows involve, we'll skip it for now. Maybe some ppl ignored the calls.

inconsistent = train[(train['pdays'] == -1) & (train['previous'] > 0)]
print(f"Inconsistent rows: {len(inconsistent)}")
display(inconsistent[['pdays', 'previous']].head(3))


from sklearn.preprocessing import QuantileTransformer
import pandas as pd
import numpy as np

class BankFeatureEngineer(BaseEstimator, TransformerMixin):
    
    def __init__(self, include_target: False, retirement_age: int = 65):
        self.include_target = include_target
        self.retirement_age = retirement_age
        self.q1_ = None
        self.q3_ = None
        self.iqr_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """blue color iqr IQR"""
        df = X.copy()
        blue_collar_balance = df[df['job'] == 'blue-collar']['balance']
        self.q1_ = blue_collar_balance.quantile(0.25)
        self.q3_ = blue_collar_balance.quantile(0.75)
        self.iqr_ = self.q3_ - self.q1_
        return self  

    def transform(self, X: pd.DataFrame):
        df = X.copy()

        # Existing features
        df['is_month_end']   = (df['day'] >= 26).astype(int)
        df['is_month_start'] = (df['day'] <= 5).astype(int)

        month_to_quarter = {
            'jan': 1, 'feb': 1, 'mar': 1, 'apr': 2, 'may': 2, 'jun': 2,
            'jul': 3, 'aug': 3, 'sep': 3, 'oct': 4, 'nov': 4, 'dec': 4
        }
        df['quarter'] = df['month'].map(month_to_quarter)
        df['is_q4']   = (df['quarter'] == 4).astype(int)

        df['recent_contact_density'] = np.where(
            df['pdays'] > 0,
            df['campaign'] / (df['pdays'] + 1),
            0
        )

        housing_map = {'yes': 1, 'no': 0}
        loan_map    = {'yes': 1, 'no': 0}
        df['debt_pressure_index'] = (
            df['housing'].map(housing_map) + df['loan'].map(loan_map)
        ) / (df['balance'].abs() + 1)
        
        df['abnormal_cash_flow'] = np.where(
            (df['job'] == 'blue-collar') &
            (df['balance'] > (self.q3_ + 3 * self.iqr_)),
            1, 0
        )
        
        df['high_value_behavior'] = np.where(
            (df['previous'] > 0) &
            (df['poutcome'] == 'success') &
            (df['duration'] > 300),
            1, 0
        )
        
        if self.include_target and 'y' in df.columns:
            df['anti_persuasion'] = np.where(
                (df['campaign'] > 5) &
                (df['previous'] == 0) &
                (df['y'] == 'no'),
                1, 0
            )
        else:
            df['anti_persuasion'] = np.nan

        mismatch = (
            (df['job'] == 'admin.')     & (df['education'] == 'primary') |
            (df['job'] == 'management') & (df['education'] == 'primary') |
            (df['job'] == 'technician') & (df['education'] == 'unknown')
        )
        df['job_education_mismatch'] = mismatch.astype(int)

        df['marital_debt_stress'] = np.where(
            (df['marital'] == 'single') &
            (df['housing'] == 'yes') &
            (df['balance'] < 0),
            1, 0
        )
 
        df['duration_bin'] = pd.cut(
            df['duration'],
            bins=[0, 60, 300, 600, float('inf')],
            labels=['short', 'medium', 'long', 'very_long'],
            right=False
        )
        df['age_group'] = pd.cut(
            df['age'],
            bins=[0, 30, 45, 60, 100],
            labels=['young', 'mid', 'senior', 'elder']
        )
        
        # NEW FEATURES
        
        # Age-based features
        df['is_senior'] = (df['age'] >= 60).astype(int)
        df['years_until_retirement'] = np.maximum(0, self.retirement_age - df['age'])
        
        # Log transformations
        df['duration_log'] = np.log1p(df['duration'])
        df['campaign_log'] = np.log1p(df['campaign'])
        df['pdays_log'] = np.log1p(df['pdays'] + 2)
        df['previous_log'] = np.log1p(df['previous'] + 1)
        
        # Duration interaction features
        df['duration_balance'] = df['duration'] * df['balance']
        df['duration_age'] = df['duration'] * df['age']
        df['duration_age_balance'] = df['duration'] * df['age'] * df['balance']
        df['duration_day'] = df['duration'] * df['day']
        df['duration_age_day'] = df['duration'] * df['age'] * df['day']
        
        # Poutcome and pdays interaction
        # Create a combined categorical-numeric feature
        df['poutcome_pdays'] = df['poutcome'].astype(str) + '_' + df['pdays'].astype(str)
        
        return df


for df_ in [train, test]:
    df_['has_prev_contact'] = np.where(df_['previous'] > 0, 'Yes', 'No')
    df_['balance_positive'] = np.where(df_['balance'] > 0, 'Yes', 'No')
    df_['was_contacted'] = np.where(df_['pdays'] != -1, 'Yes', 'No')

def extra_cols_generator(df, num_cols, output_distribution="uniform"):
    qt = QuantileTransformer(output_distribution=output_distribution, random_state=42)
    df_qt = pd.DataFrame(index=df.index)

    for col in [c for c in num_cols if c != "day"]:
        transformed = qt.fit_transform(df[[col]])[:, 0]  
        df_qt[f"{col}_qt"] = transformed

    df_qt["day_str"] = df["day"].astype(str)
    return df_qt

class ExtraColsGenerator(BaseEstimator, TransformerMixin):
    """
    Quantile-transform selected numeric columns and add a string version of 'day'.
    this contest only!
    """
    def __init__(self,num_cols,output_distribution="uniform",random_state=42):
        self.num_cols            = num_cols
        self.output_distribution = output_distribution
        self.random_state        = random_state
        self.qt_dict_ = {}

    def fit(self, X, y=None):
        self.qt_dict_ = {}
        for col in [c for c in self.num_cols if c != "day"]:
            qt = QuantileTransformer(output_distribution=self.output_distribution,
                                     random_state=self.random_state)
            qt.fit(X[[col]])
            self.qt_dict_[col] = qt
        return self  

    def transform(self, X):
        """Return a DataFrame with new *_qt columns and a 'day_str' column."""
        X_out = pd.DataFrame(index=X.index)

        for col, qt in self.qt_dict_.items():
            X_out[f"{col}_qt"] = qt.transform(X[[col]])[:, 0]
            
        if "day" in X.columns:
            X_out["day_str"] = X["day"].astype(str)

        return pd.concat([X.reset_index(drop=True),     
                          X_out.reset_index(drop=True)], axis=1)


fill_nan_value = "__MISSING__"

class CatCleaner(BaseEstimator, TransformerMixin):
    """
    for object and categorical cols：
      1. NaN -> str "__MISSING__"
      2. transfer into str
    """
    def __init__(self, fill_value="__MISSING__"):
        self.fill_value = fill_value
        
    def fit(self, X: pd.DataFrame, y=None):   
        self.obj_cols_ = X.select_dtypes(include=["object","string","category"]).columns.tolist()
        return self
    
    def transform(self, X):
        X = X.copy()
        for c in self.obj_cols_:
            X[c] = X[c].astype("object").fillna(self.fill_value).astype(str)
        return X

class CatCrossInteractor(BaseEstimator, TransformerMixin):
    """
    for str cols, do interaction. Can handle nans.
    If cols input contains numerical dtype, it'll be transformed into str
       r_vals=(2,3) 
    """
    def __init__(self, cols=None, r_vals=(2,3), sep="_", max_features=None,fill_value="__MISSING__"):
        self.cols = cols
        self.r_vals = r_vals
        self.sep = sep
        self.max_features = max_features  
        self.fill_value = fill_value
    
    def fit(self, X: pd.DataFrame, y=None):
        self.cols_ = self.cols if self.cols is not None else X.select_dtypes(include=["object","string","category"]).columns.tolist()
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        X[self.cols_] = X[self.cols_].astype("object").fillna(self.fill_value).astype(str)
        
        new_cols = []
        for r in self.r_vals:
            for combo in itertools.combinations(self.cols_, r):
                new_name = self.sep.join(combo)
                X[new_name] = X[list(combo)].agg(self.sep.join, axis=1)
                new_cols.append(new_name)
                if self.max_features and len(new_cols) >= self.max_features:
                    break
        return X


class Binner(BaseEstimator, TransformerMixin):
    """
    Normal binning for numerical columns,cannot handle nans
    """
    def __init__(self, cols=None, n_bins=10, strategy="quantile",
                 drop_original=False, min_unique_frac=0.01):
        self.cols = cols
        self.n_bins = n_bins
        self.strategy = strategy
        self.drop_original = drop_original
        self.min_unique_frac = min_unique_frac  

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.cols_ = self.cols or X.select_dtypes("number").columns.tolist()
        self.bin_edges_ = {}
        eps = 1e-9

        for c in self.cols_:
            col = X[c].dropna()
            # continue if number of unique values is too little
            if col.nunique() / max(len(col), 1) < self.min_unique_frac: continue

            if self.strategy == "quantile":
                try:
                    _, edges = pd.qcut(
                        col,
                        q=self.n_bins,
                        retbins=True,
                        duplicates="drop"
                    )
                except ValueError:
                    edges = np.array([col.min() - eps, col.max() + eps])
            else:  # uniform
                edges = np.linspace(col.min(), col.max(), self.n_bins + 1)

            if len(edges) < 2:
                continue
            edges = np.maximum.accumulate(edges)
            for i in range(1, len(edges)):
                if edges[i] <= edges[i - 1]:
                    edges[i] = edges[i - 1] + eps

            edges[0]  -= eps
            edges[-1] += eps
            self.bin_edges_[c] = edges

        return self

    def transform(self, X, y=None):
        X = X.copy()
        for c, edges in self.bin_edges_.items():
            X[f"{c}_bin"] = pd.cut(
                X[c], bins=edges, labels=False, include_lowest=True
            ).astype("str")  
        return X.drop(columns=self.bin_edges_.keys()) if self.drop_original else X

    
class ArithMix(BaseEstimator, TransformerMixin):
    """
    Perform + - * / for cols
    """
    def __init__(self, cols=None, ops=("/",)):
        self.cols = cols
        self.ops  = ops
        
    def fit(self, X, y=None):
        if self.cols is None:
            self.cols_ = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        else:
            self.cols_ = list(self.cols)
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for i, c1 in enumerate(self.cols_):
            for c2 in self.cols_[i+1:]:
                if "*" in self.ops:
                    X[f"{c1}*{c2}"] = X[c1] * X[c2]
                if "+" in self.ops:
                    X[f"{c1}+{c2}"] = X[c1] + X[c2]
                if "-" in self.ops:
                    X[f"{c1}-{c2}"] = X[c1] - X[c2]
                if "/" in self.ops:
                    eps = 1e-8
                    X[f"{c1}/{c2}"] = X[c1] / (X[c2]+ eps)
                    X[f"{c2}/{c1}"] = X[c2] / (X[c1]+ eps)
        return X

class ObjectToCategoryConverter(BaseEstimator, TransformerMixin):

    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X, y=None):
        if self.cols is None:
            self.cols_ = X.select_dtypes(include='object').columns.tolist()
        elif self.cols == 'all':
            self.cols_ = X.columns.tolist()
        elif isinstance(self.cols, list):
            self.cols_ = self.cols
        else:
            raise ValueError("cols must be None, 'all', or a list of column names.")
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols_:
            if col in X.columns:
                X[col] = X[col].astype('category')
        return X


class ReduceCardinality(BaseEstimator, TransformerMixin):
    '''Reduce the overwhelming amount of unique value within categorical cols'''
    def __init__(self, cols=None, top_n=200):
        self.cols = cols
        self.top_n = top_n

    def fit(self, X, y=None):
        self.cols_ = self.cols if self.cols is not None else X.select_dtypes(include=["category","string","object"]).columns.tolist()
        self.top_vals_ = {}  
        for col in self.cols_:
            self.top_vals_[col] = X[col].value_counts().nlargest(self.top_n).index
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols_:
            X[col] = np.where(X[col].isin(self.top_vals_[col]), X[col], '__OTHER__')
        return X



from xgboost import XGBClassifier
lgb_model = {
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': 12,    
    'n_estimators': 10000,
    'learning_rate': 0.06,
    'reg_alpha': 6,
    'reg_lambda': 64,#It's my mistake, but performs good and I'll keep it
    'colsample_bytree': 0.5,
    'subsample': 0.8,
    'verbosity': -1,
    'random_state': 42,
    'n_jobs': -1,
   # 'device_type': 'gpu',  
    'max_bin': 255,           
}
xgb_model = {
   # "tree_method":"hist",
    "enable_categorical":True,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 8,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "early_stopping_rounds": 100,
    "reg_alpha": 3,
    "reg_lambda": 3,
    "n_estimators": 10000,
}
cat_model = {'random_state': 42,
            'early_stopping_rounds': 100,
            'eval_metric': "Logloss",
            'n_estimators' : 5000,
            'learning_rate': 0.06524873965257823,
            'l2_leaf_reg': 0.8867612905712001,
            'bagging_temperature': 0.1317347791955057,
            'random_strength': 0.9922857768340815,
            'depth': 7,
            'min_data_in_leaf': 8,
            'task_type': "GPU",
            }


MODELS = {
    "XGB": {
        "cls": XGBClassifier,
        "params": xgb_model,
        "fit_kwargs_cb":
            lambda X_tr, y_tr, X_val, y_val: dict(
                eval_set=[(X_val, y_val)],    
                verbose=False
            )
    },
    "LGBM": {
        "cls": lgb.LGBMClassifier,
        "params": lgb_model,
        "fit_kwargs_cb": 
            lambda X_tr, y_tr, X_val, y_val: dict(
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
    },
    "CAT": {
        "cls": CatBoostClassifier,
        "params": cat_model,
        "fit_kwargs_cb":
            lambda X_tr, y_tr, X_val, y_val,cat_feats: dict(
                eval_set=(X_val, y_val),
                use_best_model=True,
                verbose=False,
                cat_features=cat_feats
            )
    }
}


X = train[focus_cols].copy()
y = train["y"]
X_test = test[focus_cols].copy()

full_pipe = Pipeline([
   # ("binner",  Binner()),
   # ("cross",   CatCrossInteractor()),
    ("bank_fe",BankFeatureEngineer(include_target=False)),
    ("extra", ExtraColsGenerator(num_cols=num_cols,output_distribution="normal")),
    ("reduce_card", ReduceCardinality(top_n=200)),
    ("to_cat",  ObjectToCategoryConverter()),
    ("to_cat_2",  ObjectToCategoryConverter(cols = focus_cols)),
    ("clf",     LGBMClassifier(**lgb_model))   
])
pipe = full_pipe[:-1]  

TEST_LGB = False  #single model test
if TEST_LGB:
    K = 10
    cv = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

    oof      = np.zeros(len(X))          
    test_pred = np.zeros(len(X_test))   
    overall_auc = 0                 

    for i, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        X_tr = pipe.fit_transform(X_tr, y_tr)   
        X_val = pipe.transform(X_val)
        X_test_pipe = pipe.transform(X_test)
        if i==1: print ("X_tr.shape : ", X_tr.shape)
        
        clf = lgb.LGBMClassifier(**lgb_model)
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(50,verbose=False)]              
        )
        
        oof[val_idx] = clf.predict_proba(X_val)[:,1]
        test_pred   += clf.predict_proba(X_test_pipe)[:,1] / K

        auc = clf.best_score_['valid_0']['auc']
        overall_auc += auc/K

        print(f"ROC AUC: {auc:.4f}")
        gc.collect()
        
    print("overall auc", overall_auc)


def train_one_fold(model_key, X_tr, y_tr, X_val, y_val):
    cfg = MODELS[model_key]
    params = cfg["params"]
    model = cfg["cls"](**params)

    #  category dtype for catboost
    if model_key == "CAT":
        cat_feats = [col for col in X_tr.columns if X_tr[col].dtype.name == "category"]
        fit_kwargs = cfg["fit_kwargs_cb"](X_tr, y_tr, X_val, y_val, cat_feats)
        
    else: fit_kwargs = cfg["fit_kwargs_cb"](X_tr, y_tr, X_val, y_val)
        
    model.fit(X_tr, y_tr, **fit_kwargs)
    val_pred = model.predict_proba(X_val)
    return model, val_pred

K=10

cv = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

oof_dict   = {m: np.zeros(len(X))        for m in MODELS}   
test_dict  = {m: np.zeros(len(X_test))   for m in MODELS}   
auc_dict   = {m: [] for m in MODELS}                         

for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    X_tr_pipe   = pipe.fit_transform(X_tr, y_tr)
    X_val_pipe  = pipe.transform(X_val)
    X_test_pipe = pipe.transform(X_test)

    for model_key in MODELS:
        model, val_pred = train_one_fold(
            model_key,
            X_tr_pipe, y_tr,
            X_val_pipe, y_val,
        )
        oof_dict[model_key][val_idx] += val_pred[:, 1]
        test_dict[model_key] += model.predict_proba(X_test_pipe)[:, 1] / K

        auc = roc_auc_score(y_val, val_pred[:, 1])
        auc_dict[model_key].append(auc)

        print(f"[Fold {fold}] {model_key} AUC = {auc:.4f}")

print("\n=== CV ===")
for m in MODELS:
    print(f"{m}: OOF AUC = {roc_auc_score(y, oof_dict[m]):.5f} "
          f"(folds → {np.round(auc_dict[m],4)})")



oof_meta   = np.column_stack([oof_dict[m]  for m in MODELS])
test_meta  = np.column_stack([test_dict[m] for m in MODELS])
print(oof_meta.shape)

from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score

alphas = np.logspace(-5, 2, 50) 
ridge = RidgeCV(alphas=alphas, cv=5)  
ridge.fit(oof_meta, y)

blend_oof = ridge.predict(oof_meta)
blend_auc = roc_auc_score(y, blend_oof)
print(f"Ridge blend OOF AUC = {blend_auc:.5f}")



blend_test_pred = ridge.predict(test_meta)
blend_test_pred


submission = pd.DataFrame({
    "id": test["id"],
    "target": blend_test_pred
})


submission.to_csv(submission_path, index=False)
print("✅ Saved")

display(submission.head(5))

