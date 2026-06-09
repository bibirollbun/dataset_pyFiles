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


df = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')


df.head(5)


df.shape


df[df['isFraud'] == 1].shape


df[df['isFraud'] == 0].shape


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


X = df.drop('isFraud', axis=1)
y = df['isFraud']


X = X.select_dtypes(include=['float64', 'int64'])


X = X.fillna(0)


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Giáº£ sá»­ báº¡n Ä‘Ã£ cÃ³ dá»¯ liá»‡u X vÃ  y
# X: Dá»¯ liá»‡u Ä‘áº·c trÆ°ng, y: NhÃ£n

# Chia dá»¯ liá»‡u thÃ nh táº­p huáº¥n luyá»‡n vÃ  táº­p kiá»ƒm tra (80% huáº¥n luyá»‡n, 20% kiá»ƒm tra)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khá»Ÿi táº¡o vÃ  huáº¥n luyá»‡n mÃ´ hÃ¬nh Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Dá»± Ä‘oÃ¡n xÃ¡c suáº¥t cho cÃ¡c lá»›p
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Láº¥y xÃ¡c suáº¥t cho lá»›p dÆ°Æ¡ng (1)

# TÃ­nh AUC (Area Under the Curve)
auc = roc_auc_score(y_test, y_pred_proba)

# TÃ­nh GIN
gini = 2 * auc - 1
print(f"AUC: {auc:.4f}")
print(f"GINI: {gini:.4f}")

# Váº½ ROC Curve Ä‘á»ƒ tham kháº£o
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()



import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# CÃ i Ä‘áº·t tham sá»‘ cho mÃ´ hÃ¬nh XGBoost
params = {
    'objective': 'binary:logistic',  # PhÃ¢n loáº¡i nhá»‹ phÃ¢n
    'eval_metric': 'logloss',        # Ä�Ã¡nh giÃ¡ báº±ng logloss
    'max_depth': 3,                  # Ä�á»™ sÃ¢u tá»‘i Ä‘a cá»§a cÃ¢y
    'eta': 0.1,                      # Tá»‘c Ä‘á»™ há»�c (learning rate)
    #'silent': 1                      # Táº¯t thÃ´ng bÃ¡o khi huáº¥n luyá»‡n
}

# Huáº¥n luyá»‡n mÃ´ hÃ¬nh vá»›i XGBoost
num_round = 100  # Sá»‘ vÃ²ng láº·p (iterations)
bst = xgb.train(params, dtrain, num_round)

# Dá»± Ä‘oÃ¡n vá»›i dá»¯ liá»‡u kiá»ƒm tra
y_pred_prob = bst.predict(dtest)  # Dá»± Ä‘oÃ¡n xÃ¡c suáº¥t
y_pred = (y_pred_prob > 0.5).astype(int)  # Chuyá»ƒn xÃ¡c suáº¥t thÃ nh nhÃ£n (0 hoáº·c 1)


# TÃ­nh AUC (Area Under the Curve)
auc = roc_auc_score(y_test, y_pred_prob)

# TÃ­nh GIN
gini = 2 * auc - 1
print(f"AUC: {auc:.4f}")
print(f"GINI: {gini:.4f}")




from xgboost import plot_importance
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')  # Style Ä‘áº¹p
fig, ax = plt.subplots(figsize=(10, 16))  # TÃ¹y chá»‰nh kÃ­ch thÆ°á»›c
plot_importance(bst,  # Model Ä‘Ã£ train xong
                max_num_features=30,  # Top 30 features quan trá»�ng nháº¥t
                importance_type='gain',  # Dá»±a trÃªn gain
                ax=ax)
plt.title("Top 30 Feature Importances (by Gain)")
plt.tight_layout()
plt.show()


import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# Táº¡o dataset cho LightGBM
dtrain = lgb.Dataset(X_train, label=y_train)
dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)

# CÃ i Ä‘áº·t tham sá»‘ cho LightGBM
params = {
    'objective': 'binary',           # PhÃ¢n loáº¡i nhá»‹ phÃ¢n
    'metric': 'binary_logloss',     # Sá»­ dá»¥ng logloss Ä‘á»ƒ Ä‘Ã¡nh giÃ¡
    'max_depth': 3,                 # Ä�á»™ sÃ¢u tá»‘i Ä‘a cá»§a cÃ¢y
    'learning_rate': 0.1,           # Tá»‘c Ä‘á»™ há»�c
    'verbosity': -1                 # Táº¯t log (tuá»³ chá»�n)
}

# Sá»‘ vÃ²ng láº·p
num_round = 100

# Huáº¥n luyá»‡n mÃ´ hÃ¬nh
bst = lgb.train(params, dtrain, num_boost_round=num_round)

# Dá»± Ä‘oÃ¡n xÃ¡c suáº¥t trÃªn táº­p kiá»ƒm tra
y_pred_prob = bst.predict(X_test)

# Chuyá»ƒn xÃ¡c suáº¥t thÃ nh nhÃ£n 0 hoáº·c 1
y_pred = (y_pred_prob > 0.5).astype(int)

# TÃ­nh AUC
auc = roc_auc_score(y_test, y_pred_prob)

# TÃ­nh GINI
gini = 2 * auc - 1

print(f"AUC: {auc:.4f}")
print(f"GINI: {gini:.4f}")



df.head(2)


def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))


display(missing_data(df))


missing_df = missing_data(df)


missing_df


# Láº¥y danh sÃ¡ch tÃªn cá»™t cÃ³ pháº§n trÄƒm missing > 80%
cols_missing_over_80 = missing_df.loc['Percent'][missing_df.loc['Percent'] > 80].index.tolist()

print("CÃ¡c cá»™t cÃ³ > 80% missing values:")
print(cols_missing_over_80)


missing_data(df[cols_missing_over_80])


missing_list = cols_missing_over_80


df['TransactionID'].nunique()


df.dtypes.value_counts()


df_int = df.select_dtypes(include=['int64'])


df_int


df_float = df.select_dtypes(include=['float64'])


df_float


df_cat = df.select_dtypes(include=['object'])


df_cat


for col in df_cat.columns:    
    print(f'{col}: {df_cat[col].unique().tolist()}')


def calc_iv(df, target, bins=10, min_category_freq=0, return_woe_detail=False):
    """
    TÃ­nh IV cho táº¥t cáº£ cÃ¡c biáº¿n trong DataFrame (trá»« biáº¿n má»¥c tiÃªu).
    
    Params:
        df     : pandas.DataFrame
        target : tÃªn cá»™t má»¥c tiÃªu (0/1)
        bins   : sá»‘ bin cho biáº¿n sá»‘ (default = 10)
        min_category_freq : ngÆ°á»¡ng tá»‘i thiá»ƒu Ä‘á»ƒ giá»¯ láº¡i nhÃ£n (náº¿u lÃ  biáº¿n phÃ¢n loáº¡i)
        return_woe_detail : náº¿u True, tráº£ thÃªm báº£ng WOE chi tiáº¿t cho tá»«ng biáº¿n

    Returns:
        iv_df  : DataFrame chá»©a tÃªn biáº¿n vÃ  giÃ¡ trá»‹ IV, Ä‘Ã£ sáº¯p xáº¿p giáº£m dáº§n
        woe_detail_dict (optional): dict chá»©a báº£ng chi tiáº¿t WOE/IV theo biáº¿n
    """
    iv_dict = {}
    woe_detail_dict = {}

    for col in df.columns:
        if col == target:
            continue

        temp_df = df[[col, target]].copy()

        # Binning
        if pd.api.types.is_numeric_dtype(temp_df[col]):
            try:
                temp_df['bin'] = pd.qcut(temp_df[col], q=bins, duplicates='drop')
            except ValueError:
                temp_df['bin'] = pd.cut(temp_df[col], bins=bins)
            temp_df['bin'] = temp_df['bin'].astype(str)
            temp_df.loc[temp_df[col].isna(), 'bin'] = 'missing'
        else:
            temp_df['bin'] = temp_df[col].astype(str)
            temp_df.loc[temp_df[col].isna(), 'bin'] = 'missing'

            if min_category_freq > 0:
                freq = temp_df['bin'].value_counts()
                rare_labels = freq[freq < min_category_freq].index
                temp_df['bin'] = temp_df['bin'].apply(lambda x: 'others' if x in rare_labels else x)

        # Group vÃ  tÃ­nh toÃ¡n
        grouped = temp_df.groupby('bin')[target].agg(['count', 'sum'])
        grouped.columns = ['total', 'bad']
        grouped['good'] = grouped['total'] - grouped['bad']
        grouped['dist_good'] = grouped['good'] / grouped['good'].sum()
        grouped['dist_bad'] = grouped['bad'] / grouped['bad'].sum()
        grouped['woe'] = np.log((grouped['dist_good'] + 1e-6) / (grouped['dist_bad'] + 1e-6))
        grouped['iv'] = (grouped['dist_good'] - grouped['dist_bad']) * grouped['woe']

        iv_value = grouped['iv'].sum()
        iv_dict[col] = iv_value

        if return_woe_detail:
            grouped = grouped.reset_index()
            grouped['variable'] = col
            woe_detail_dict[col] = grouped[['variable', 'bin', 'total', 'bad', 'good', 'woe', 'iv']]

    iv_df = pd.DataFrame.from_dict(iv_dict, orient='index', columns=['IV'])
    iv_df.index.name = 'Variable'
    iv_df = iv_df.sort_values(by='IV', ascending=False)

    if return_woe_detail:
        return iv_df, woe_detail_dict
    else:
        return iv_df


iv_df, woe_detail_dict = calc_iv(df, target = 'isFraud', bins=20, min_category_freq=10, return_woe_detail=True)


iv_df = iv_df.reset_index()


iv_df[iv_df['IV'] < 0.02].shape


iv_df[iv_df['Variable'] == 'V247']


df['V247'].unique()


woe_detail_dict['V247']


woe_detail_dict['TransactionAmt']


iv_df[iv_df['Variable'] == 'TransactionAmt']


woe_detail_dict['P_emaildomain']


iv_list = iv_df[iv_df['IV'] < 0.02]['Variable'].tolist()


for col in missing_list:
    print(iv_df[iv_df['Variable'] == col])


woe_detail_dict['D7']


df = df.drop(iv_list, axis = 1)


df.shape


df


df['TransactionDT'].dtypes


from datetime import datetime, timedelta

start_time = datetime(2023, 12, 31)

df['transaction_date'] = df['TransactionDT'].apply(lambda x: (start_time + timedelta(seconds=x)).date())
df['transaction_month'] = df['TransactionDT'].apply(lambda x: (start_time + timedelta(seconds=x)).month)


df[['transaction_date','transaction_month', 'TransactionDT', ]]


fraud_stats = df.groupby('transaction_month').agg(
    transaction_count=('isFraud', 'count'),
    fraud_count=('isFraud', 'sum')
).reset_index()

fraud_stats['fraud_rate'] = fraud_stats['fraud_count'] / fraud_stats['transaction_count']


fraud_stats


df.loc[df['transaction_month'] < 6, 'split'] = 'dev'
df.loc[df['transaction_month'] == 6, 'split'] = 'oot'


exd_cols = ['TransactionID', 'TransactionDT', 'P_emaildomain', 'R_emaildomain', 'isFraud', 'split', 'stratify']


random_seed = 42


df['stratify'] = df['transaction_date'].astype(str) + df['isFraud'].astype(str)

X_dev = df[df['split']=='dev'].drop(exd_cols, axis=1)
y_dev = df[df['split']=='dev']['isFraud']
z_dev1 = df[df['split']=='dev']['stratify']

X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev, stratify = z_dev1, test_size=0.2, random_state=random_seed)


X_train


X_train['transaction_month'].unique()


X_test['transaction_month'].unique()


X_train = X_train.drop(['transaction_date', 'transaction_month'], axis=1)


X_test = X_test.drop(['transaction_date', 'transaction_month'], axis=1)


X_train


X_oot = df[df['split']=='oot'].drop(exd_cols, axis=1)
y_oot = df[df['split']=='oot']['isFraud']


X_oot = X_oot.drop(['transaction_date', 'transaction_month'], axis=1)


X_oot


import pandas as pd
import numpy as np
from xgboost import DMatrix, train, Booster
from sklearn.metrics import roc_auc_score

# --- 1. Chuáº©n bá»‹ dá»¯ liá»‡u ---
# Chuyá»ƒn object â†’ category
for col in X_train.select_dtypes(include='object').columns:
    X_train[col] = X_train[col].astype('category')

# Táº¡o DMatrix
dtrain = DMatrix(X_train, label=y_train, enable_categorical=True)

# --- 2. Khai bÃ¡o tham sá»‘ ---
params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 4,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",  # hoáº·c "gpu_hist" náº¿u dÃ¹ng GPU
    "random_state": 42,
    "enable_categorical": True
}

# --- 3. Train mÃ´ hÃ¬nh ---
booster = train(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,
    evals=[(dtrain, "train")],
    verbose_eval=50
)

# --- 4. Dá»± Ä‘oÃ¡n & TÃ­nh Gini ---
y_pred = booster.predict(dtrain)
gini = 2 * roc_auc_score(y_train, y_pred) - 1
print(f"âœ… Gini on training set: {gini:.4f}")


for col in X_oot.select_dtypes(include='object').columns:
    X_oot[col] = X_oot[col].astype('category')

doot = DMatrix(X_oot, label=y_oot, enable_categorical=True)


y_pred_oot = booster.predict(doot)
gini = 2 * roc_auc_score(y_oot, y_pred_oot) - 1
print(f"âœ… Gini on oot set: {gini:.4f}")


for col in X_test.select_dtypes(include='object').columns:
    X_test[col] = X_test[col].astype('category')

dtest = DMatrix(X_test, label=y_test, enable_categorical=True)


y_pred_test = booster.predict(dtest)
gini = 2 * roc_auc_score(y_test, y_pred_test) - 1
print(f"âœ… Gini on test set: {gini:.4f}")


from xgboost import plot_importance
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')  # Style Ä‘áº¹p
fig, ax = plt.subplots(figsize=(10, 16))  # TÃ¹y chá»‰nh kÃ­ch thÆ°á»›c
plot_importance(booster,  # Model Ä‘Ã£ train xong
                max_num_features=30,  # Top 30 features quan trá»�ng nháº¥t
                importance_type='gain',  # Dá»±a trÃªn gain
                ax=ax)
plt.title("Top 30 Feature Importances (by Gain)")
plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import DMatrix, train, Booster
import joblib
import os

# --- Táº¡o thÆ° má»¥c Ä‘á»ƒ lÆ°u model ---
os.makedirs("xgb_models", exist_ok=True)

# --- 1. Chuyá»ƒn cÃ¡c cá»™t object thÃ nh category ---
for df in [X_train, X_test, X_oot]:
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category')

# --- 2. Khá»Ÿi táº¡o KFold ---
K = 5
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 4,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "random_state": 42,
    "enable_categorical": True
}

# --- 3. Train tá»«ng fold vÃ  lÆ°u model ---
gini_scores = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\nğŸ“¦ Training Fold {fold+1}/{K}")
    
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_val, y_val = X_train.iloc[valid_idx], y_train.iloc[valid_idx]

    dtrain = DMatrix(X_tr, label=y_tr, enable_categorical=True)
    dval = DMatrix(X_val, label=y_val, enable_categorical=True)

    booster = train(
        params=params,
        dtrain=dtrain,
        num_boost_round=500,
        evals=[(dval, "validation")],
        verbose_eval=50
    )

    # Predict & Gini
    val_pred = booster.predict(dval)
    gini = 2 * roc_auc_score(y_val, val_pred) - 1
    gini_scores.append(gini)
    print(f"âœ… Fold {fold+1} Gini: {gini:.4f}")

    # Save model
    booster.save_model(f"xgb_models/fold_{fold}.json")

print(f"\nğŸ�¯ Average Gini across folds: {np.mean(gini_scores):.4f}")


def ensemble_predict(X_input, model_dir="xgb_models"):
    # Chuyá»ƒn object -> category
    for col in X_input.select_dtypes(include='object').columns:
        X_input[col] = X_input[col].astype('category')

    preds = np.zeros(X_input.shape[0])
    for fold in range(K):
        booster = Booster()
        booster.load_model(f"{model_dir}/fold_{fold}.json")
        dmatrix = DMatrix(X_input, enable_categorical=True)
        preds += booster.predict(dmatrix)
    
    return preds / K



# Predict
y_test_pred = ensemble_predict(X_test)
y_oot_pred = ensemble_predict(X_oot)

# Gini náº¿u cÃ³ ground truth
gini_test = 2 * roc_auc_score(y_test, y_test_pred) - 1
gini_oot = 2 * roc_auc_score(y_oot, y_oot_pred) - 1

print(f"\nğŸ“Š Gini Test Set: {gini_test:.4f}")
print(f"ğŸ“Š Gini OOT  Set: {gini_oot:.4f}")



from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


def xgb_cv_bo(max_depth, gamma, eta, subsample, colsample_bytree, min_child_weight):
    model = XGBClassifier(
        n_estimators=100,
        max_depth=int(max_depth),
        gamma=gamma,
        eta=eta,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        tree_method='hist',
        enable_categorical=True
    )

    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
    return scores.mean()



%%time

xgb_bo = BayesianOptimization(
    f=xgb_cv_bo,
    pbounds={
        'max_depth': (3, 10),
        'gamma': (0, 5),
        'eta': (0.01, 0.3),
        'subsample': (0.5, 1),
        'colsample_bytree': (0.5, 1),
        'min_child_weight': (1, 10)
    },
    random_state=42,
    verbose=2
)

xgb_bo.maximize(init_points=3, n_iter=9)



best_params = xgb_bo.max['params']
best_params['max_depth'] = int(best_params['max_depth'])  # cáº§n Ã©p vá»� int

print("\nğŸ�¯ Best Hyperparameters:")
print(best_params)



best_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "random_state": 42,
    "enable_categorical": True,
    "use_label_encoder": False,
    **best_params
}
best_params['max_depth'] = int(best_params['max_depth'])



os.makedirs("xgb_models_opt", exist_ok=True)

K = 5
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

gini_scores = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\nğŸ“¦ Training Fold {fold+1}/{K}")
    
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_val, y_val = X_train.iloc[valid_idx], y_train.iloc[valid_idx]

    dtrain = DMatrix(X_tr, label=y_tr, enable_categorical=True)
    dval = DMatrix(X_val, label=y_val, enable_categorical=True)

    booster = train(
        params=best_params,
        dtrain=dtrain,
        num_boost_round=100,
        evals=[(dval, "validation")],
        verbose_eval=50
    )

    # Predict & Gini
    val_pred = booster.predict(dval)
    gini = 2 * roc_auc_score(y_val, val_pred) - 1
    gini_scores.append(gini)
    print(f"âœ… Fold {fold+1} Gini: {gini:.4f}")

    # Save model
    booster.save_model(f"xgb_models_opt/fold_{fold}.json")

print(f"\nğŸ�¯ Average Gini across folds: {np.mean(gini_scores):.4f}")


def ensemble_predict_opt(X_input, model_dir="xgb_models_opt"):
    # Chuyá»ƒn object -> category
    for col in X_input.select_dtypes(include='object').columns:
        X_input[col] = X_input[col].astype('category')

    preds = np.zeros(X_input.shape[0])
    for fold in range(K):
        booster = Booster()
        booster.load_model(f"{model_dir}/fold_{fold}.json")
        dmatrix = DMatrix(X_input, enable_categorical=True)
        preds += booster.predict(dmatrix)
    
    return preds / K



# Predict
y_test_pred = ensemble_predict_opt(X_test)
y_oot_pred = ensemble_predict_opt(X_oot)

# Gini náº¿u cÃ³ ground truth
gini_test = 2 * roc_auc_score(y_test, y_test_pred) - 1
gini_oot = 2 * roc_auc_score(y_oot, y_oot_pred) - 1

print(f"\nğŸ“Š Gini Test Set: {gini_test:.4f}")
print(f"ğŸ“Š Gini OOT  Set: {gini_oot:.4f}")



import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold


def objective(trial):
    # Gá»£i Ã½ giÃ¡ trá»‹ hyperparam tá»« Optuna
    max_depth = trial.suggest_int("max_depth", 3, 10)
    gamma = trial.suggest_float("gamma", 0, 5)
    eta = trial.suggest_float("eta", 0.01, 0.3)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    min_child_weight = trial.suggest_float("min_child_weight", 1, 10)

    # Táº¡o mÃ´ hÃ¬nh
    model = XGBClassifier(
        n_estimators=100,
        max_depth=max_depth,
        gamma=gamma,
        eta=eta,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        tree_method='hist',
        enable_categorical=True,
        random_state=42
    )

    # Ä�Ã¡nh giÃ¡ báº±ng Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()

    return score  # Optuna sáº½ tá»± maximize



%%time

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=12)  # Thá»­ 25 cáº¥u hÃ¬nh



print("âœ… Best score (AUC):", study.best_value)
print("ğŸ�¯ Best params:")
print(study.best_params)


best_params = study.best_params

best_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "random_state": 42,
    "enable_categorical": True,
    "use_label_encoder": False,
    **best_params
}


from xgboost import DMatrix, train

os.makedirs("xgb_models_optuna", exist_ok=True)

K = 5
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

gini_scores = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\nğŸ“¦ Training Fold {fold+1}/{K}")
    
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_val, y_val = X_train.iloc[valid_idx], y_train.iloc[valid_idx]

    dtrain = DMatrix(X_tr, label=y_tr, enable_categorical=True)
    dval = DMatrix(X_val, label=y_val, enable_categorical=True)

    booster = train(
        params=best_params,
        dtrain=dtrain,
        num_boost_round=100,
        evals=[(dval, "validation")],
        verbose_eval=50
    )

    # Predict & Gini
    val_pred = booster.predict(dval)
    gini = 2 * roc_auc_score(y_val, val_pred) - 1
    gini_scores.append(gini)
    print(f"âœ… Fold {fold+1} Gini: {gini:.4f}")

    # Save model
    booster.save_model(f"xgb_models_optuna/fold_{fold}.json")

print(f"\nğŸ�¯ Average Gini across folds: {np.mean(gini_scores):.4f}")


def ensemble_predict_optuna(X_input, model_dir="xgb_models_optuna"):
    # Chuyá»ƒn object -> category
    for col in X_input.select_dtypes(include='object').columns:
        X_input[col] = X_input[col].astype('category')

    preds = np.zeros(X_input.shape[0])
    for fold in range(K):
        booster = Booster()
        booster.load_model(f"{model_dir}/fold_{fold}.json")
        dmatrix = DMatrix(X_input, enable_categorical=True)
        preds += booster.predict(dmatrix)
    
    return preds / K



from xgboost import Booster

# Predict
y_test_pred = ensemble_predict_optuna(X_test)
y_oot_pred = ensemble_predict_optuna(X_oot)

# Gini náº¿u cÃ³ ground truth
gini_test = 2 * roc_auc_score(y_test, y_test_pred) - 1
gini_oot = 2 * roc_auc_score(y_oot, y_oot_pred) - 1

print(f"\nğŸ“Š Gini Test Set: {gini_test:.4f}")
print(f"ğŸ“Š Gini OOT  Set: {gini_oot:.4f}")


for col in X_train.select_dtypes(include='object').columns:
    X_train[col] = X_train[col].astype('category')


%%time

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score
from scipy.stats import randint, uniform

# Gini scorer
def gini_score(y_true, y_pred):
    from sklearn.metrics import roc_auc_score
    return 2 * roc_auc_score(y_true, y_pred) - 1

gini_scorer = make_scorer(gini_score, needs_proba=True)

# Param distributions tÆ°Æ¡ng Ä‘Æ°Æ¡ng Optuna
param_dist = {
    'max_depth': randint(3, 11),                         # int tá»« 3 Ä‘áº¿n 10
    'gamma': uniform(0, 5),                              # float tá»« 0 Ä‘áº¿n 5
    'eta': uniform(0.01, 0.29),                          # float tá»« 0.01 Ä‘áº¿n 0.3
    'subsample': uniform(0.5, 0.5),                      # float tá»« 0.5 Ä‘áº¿n 1.0
    'colsample_bytree': uniform(0.5, 0.5),               # float tá»« 0.5 Ä‘áº¿n 1.0
    'min_child_weight': uniform(1, 9),                   # float tá»« 1 Ä‘áº¿n 10
}

# Model
xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    tree_method='hist',
    enable_categorical=True,
    random_state=42
)

# KFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Randomized Search
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=12,                 # sá»‘ lÆ°á»£ng tá»• há»£p random
    scoring=gini_scorer,
    cv=cv,
    verbose=2,
    n_jobs=-1,
    random_state=11
)

# Fit
random_search.fit(X_train, y_train)

# Káº¿t quáº£
print("ğŸ�¯ Best Gini:", random_search.best_score_)
print("âœ… Best params:", random_search.best_params_)


best_params = random_search.best_params_

best_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "random_state": 42,
    "enable_categorical": True,
    "use_label_encoder": False,
    **best_params
}


from xgboost import DMatrix, train

os.makedirs("xgb_models_rand", exist_ok=True)

K = 5
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

gini_scores = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\nğŸ“¦ Training Fold {fold+1}/{K}")
    
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_val, y_val = X_train.iloc[valid_idx], y_train.iloc[valid_idx]

    dtrain = DMatrix(X_tr, label=y_tr, enable_categorical=True)
    dval = DMatrix(X_val, label=y_val, enable_categorical=True)

    booster = train(
        params=best_params,
        dtrain=dtrain,
        num_boost_round=100,
        evals=[(dval, "validation")],
        verbose_eval=50
    )

    # Predict & Gini
    val_pred = booster.predict(dval)
    gini = 2 * roc_auc_score(y_val, val_pred) - 1
    gini_scores.append(gini)
    print(f"âœ… Fold {fold+1} Gini: {gini:.4f}")

    # Save model
    booster.save_model(f"xgb_models_rand/fold_{fold}.json")

print(f"\nğŸ�¯ Average Gini across folds: {np.mean(gini_scores):.4f}")


K = 5

def ensemble_predict_rand(X_input, model_dir="xgb_models_rand"):
    # Chuyá»ƒn object -> category
    for col in X_input.select_dtypes(include='object').columns:
        X_input[col] = X_input[col].astype('category')

    preds = np.zeros(X_input.shape[0])
    for fold in range(K):
        booster = Booster()
        booster.load_model(f"{model_dir}/fold_{fold}.json")
        dmatrix = DMatrix(X_input, enable_categorical=True)
        preds += booster.predict(dmatrix)
    
    return preds / K



from xgboost import Booster

# Predict
y_test_pred = ensemble_predict_rand(X_test)
y_oot_pred = ensemble_predict_rand(X_oot)

# Gini náº¿u cÃ³ ground truth
gini_test = 2 * roc_auc_score(y_test, y_test_pred) - 1
gini_oot = 2 * roc_auc_score(y_oot, y_oot_pred) - 1

print(f"\nğŸ“Š Gini Test Set: {gini_test:.4f}")
print(f"ğŸ“Š Gini OOT  Set: {gini_oot:.4f}")

