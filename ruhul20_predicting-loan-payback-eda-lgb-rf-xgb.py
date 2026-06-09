!pip install -U scikit-learn imbalanced-learn


import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import  gc, time
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
import lightgbm as lgb
from sklearn.metrics import accuracy_score

SEED = 42

# utility
def seed_everything(seed=SEED):
    np.random.seed(seed)
seed_everything()

import warnings
warnings.filterwarnings('ignore')


# ====== 1. Load ======
# update path to match your Kaggle input directory
TRAIN_PATH = "/kaggle/input/playground-series-s5e11/train.csv"
TEST_PATH  = "/kaggle/input/playground-series-s5e11/test.csv"
SUB_PATH   = "/kaggle/input/playground-series-s5e11/sample_submission.csv"

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
print("train", train.shape, "test", test.shape)


train.isnull().sum().sort_values(ascending=False)


train.info()


test.info()


data = train.drop(columns=['id'], axis=1)
columns = data.columns
columns


for x in columns:
    print(f'{data[x].value_counts()}\n')


# Correlation matrix
corr_matrix = data.corr(numeric_only=True)

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="Oranges")
plt.title("Correlation Matrix for Dataset", fontsize=15)
plt.savefig('Correlation Matrix df.png')
plt.show()


plt.figure(figsize=(10, 6))
datalabel = sns.countplot(x='loan_paid_back', data=data, palette='viridis')

for i in datalabel.containers:
    datalabel.bar_label(i)


plt.title('Distribution of loan_paid_back')
plt.xlabel('loan_paid_back')
plt.ylabel('Count')
plt.savefig('Distribution of loan_paid_back.png')
plt.show()


# Continuous variables to visualize
continuous_vars = ['loan_purpose', 'employment_status', 'education_level','marital_status']

plt.figure(figsize=(12, 10))
for i, var in enumerate(continuous_vars):
    plt.subplot(2, 2, i+1)
    sns.histplot(data[var], kde=True, color="hotpink")
    plt.title(f'{var} Distribution')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('metformin.png')
plt.show()


# Continuous variables to visualize
continuous_vars = ['loan_purpose', 'employment_status', 'education_level','marital_status']

plt.figure(figsize=(12, 10))
for i, var in enumerate(continuous_vars):
    plt.subplot(2, 2, i+1)
    
    datalabel = sns.countplot(x=data[var], data=data, palette='Greens' )
    for i in datalabel.containers:
        datalabel.bar_label(i)
        
    plt.title(f'{var} Distribution')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('troglitazone.png')
plt.show()


categorical_vars =['loan_purpose', 'employment_status', 'education_level','gender']

plt.figure(figsize=(12, 8))
for i, var in enumerate(categorical_vars):
    plt.subplot(2, 2, i+1)
    
    datalabel = sns.countplot(x=var, hue='loan_paid_back', data=data)
    for i in datalabel.containers:
        datalabel.bar_label(i)
        
    plt.title(f'{var} by loan_paid_back')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('loan_paid_back.png')
plt.show()


from sklearn.metrics import  confusion_matrix, classification_report, make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score, roc_curve

from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
import joblib

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler , LabelEncoder
from sklearn.pipeline import Pipeline


df = data.copy()
#df['readmitted'] = df['readmitted'].map({'No':0, '>30':})

label_encoder = LabelEncoder()

cat_cols = df.select_dtypes(include=['object','category']).columns  # pick categorical columns

for col in cat_cols:
    df[col] = label_encoder.fit_transform(df[col].astype(str))

df.head()


X = df.drop(['loan_paid_back'],axis=1)
y = df['loan_paid_back']
 
X.describe()


# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, stratify=y, random_state=42
# )

# X_train.shape


numeric_features = list(X.select_dtypes(include=['int64', 'float64']).columns)


# standard_scaler = StandardScaler()

# X[numeric_features] = standard_scaler.fit_transform(X[numeric_features])

# X_test[numeric_features] = standard_scaler.transform(X_test[numeric_features])

# X_train.describe()


from catboost import CatBoostClassifier


# Model parameter dicts (you can tune further)
rf_params = dict(random_state=42, n_jobs=-1, n_estimators=800, verbose=0)

lgb_params = dict(
    n_estimators=1320, learning_rate=0.05, num_leaves=93, max_depth=5,
    colsample_bytree=0.975, subsample=0.743, reg_alpha=2.95, reg_lambda=0.0022,
    random_state=42, n_jobs=-1, objective='binary', metric='auc', verbosity=-1
)

xgb_params = dict(
    objective="binary:logistic", eval_metric="auc", tree_method="hist",
    max_depth=6, learning_rate=0.06694384217835, n_estimators=732,
    min_child_weight=8.3685, subsample=0.8639, colsample_bytree=0.92626,
    gamma=1.98801, reg_alpha=0.01047, reg_lambda=0.01006, max_bin=504,
    random_state=42, n_jobs=-1, verbosity=0
)

cat_params = dict(
    iterations=1500,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=10,
    eval_metric='AUC',
    task_type='CPU'  # change to 'GPU' if you have GPU and catboost built with GPU support
)

# Optional: if you have categorical feature column indices (list of ints)
# cat_features_idx = [0, 3, 5]  # example; set to None or [] if none
cat_features_idx = None

# CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
roc_curves, fold_scores = [], []

# Ensemble weights (tune as you like)
w_xgb = 0.5
w_lgb = 0.15
w_cat = 0.15
w_rf  = 0.2

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    print(f"--- Fold {fold}/{skf.n_splits} ---")
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    # --- instantiate fresh models each fold ---
    rf_model = RandomForestClassifier(**rf_params)
    lgb_model = LGBMClassifier(**lgb_params)
    xgb_model = XGBClassifier(**xgb_params)
    cat_model = CatBoostClassifier(**cat_params)

    # 1) RandomForest (no eval_set)
    rf_model.fit(X_tr, y_tr)
    rf_pred = rf_model.predict_proba(X_val)[:, 1]

    # 2) LightGBM with early stopping
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        
    )
    lgb_pred = lgb_model.predict_proba(X_val)[:, 1]

    # 3) XGBoost with early stopping
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    xgb_pred = xgb_model.predict_proba(X_val)[:, 1]

    # 4) CatBoost with early stopping (use cat_features if provided)
    if cat_features_idx:
        cat_model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            cat_features=cat_features_idx,
            early_stopping_rounds=100,
            use_best_model=True,
            verbose=10
        )
    else:
        cat_model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=100,
            use_best_model=True,
            verbose=10
        )
    cat_pred = cat_model.predict_proba(X_val)[:, 1]

    # Blend predictions
    val_pred = w_xgb * xgb_pred + w_lgb * lgb_pred + w_cat * cat_pred + w_rf * rf_pred

    auc = roc_auc_score(y_val, val_pred)
    fold_scores.append(auc)
    print(f"Fold {fold} AUC: {auc:.4f}")

    fpr, tpr, _ = roc_curve(y_val, val_pred)
    roc_curves.append((fpr, tpr, auc))

print("Fold AUCs:", [round(s, 4) for s in fold_scores])
simple_avg_score = np.mean(fold_scores)
print(f"\nSimple Average CV Score: {simple_avg_score:.5f} (+/- {np.std(fold_scores):.5f})")



# rf_model = RandomForestClassifier(
#                 random_state=42, 
#                 n_jobs=-1, 
#                 n_estimators=800,
#                 verbose=0
#         )

# rf_model.fit(X_train, y_train)


# y_pred = rf_model.predict(X_test)

# print(f'\nRandomForestClassifier Best score : {accuracy_score(y_test, y_pred)}\n')
    
# sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="seismic")
# plt.title(f'Confusion Matrix for RandomForestClassifier', fontsize=14)
# plt.savefig(f'Confusion Matrix RandomForestClassifier.png')
# plt.show()


# rf_final = joblib.load("best_rf.pkl")


# lgb_final = joblib.load("best_lgb.pkl")


# final_models = [rf_final, lgb_final]

# models_name =["RandomForestClassifier","LGBMClassifier"]


# lgb_model = LGBMClassifier(
#                 random_state=42, 
#                 n_jobs=-1, 
#                 #device="gpu", 
#                 verbosity=-1,
#                 # boost_from_average=False, 
#                 # force_row_wise=True,
#                 # scale_pos_weight=4,        # imbalance dataset positive class weight (ratio of neg/pos)
#                 # subsample=0.8,             # row sampling → overfitting reduce
#                 # colsample_bytree=0.8,      # feature sampling
#                 boosting_type="gbdt",      # default GBDT
#                 learning_rate=0.01, 
#                 n_estimators=800
#             )

# lgb_model.fit(X_train, y_train)


# y_pred = lgb_model.predict(X_test)

# print(f'\LGBMClassifier Best score : {accuracy_score(y_test, y_pred)}\n')
    
# sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="seismic")
# plt.title(f'Confusion Matrix for LGBMClassifier', fontsize=14)
# plt.savefig(f'Confusion Matrix LGBMClassifier.png')
# plt.show()


# from xgboost import XGBClassifier

# xgb_params = {
#     'n_estimators': 1000,
#     'learning_rate': 0.01,
#     'max_depth': 12,
#     'min_child_weight': 3,
    
#     'gamma': 0.2,
#     'reg_alpha': 0.1,
#     'reg_lambda': 0.3,
#     'tree_method': 'hist',  # 'gpu_hist' if GPU available
#     'random_state': 42
# }

# xgb_model = XGBClassifier(**xgb_params)
# xgb_model.fit(X_train, y_train)


# y_pred = xgb_model.predict(X_test)

# print(f'\nXGBClassifier Best score : {accuracy_score(y_test, y_pred)}\n')
    
# sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="seismic")
# plt.title(f'Confusion Matrix for XGBClassifier', fontsize=14)
# plt.savefig(f'Confusion Matrix XGBClassifier.png')
# plt.show()


# y_pred = xgb_model.predict(X_test)
# print(f'mean_square_error : {mean_squared_error(y_test, y_pred)}')


def draw_outlier_hist(data, column): 
    
    global lower, upper    
    q25, q75 = np.quantile(data[column], 0.25), np.quantile(data[column], 0.75)          
    iqr = q75 - q25    
    cut_off = iqr * 1.5          
    lower, upper = q25 - cut_off, q75 + cut_off     
    
    print('IQR : ',iqr)     
    print('lower bound : ', lower)     
    print('upper bound : ', upper)    
    
    data1 = data[data[column] > upper]     
    data2 = data[data[column] < lower]    
    
    print('# of Outlier : ', data1.shape[0] + data2.shape[0])
    
    plt.figure(figsize=(10,5))
    sns.distplot(data[column], kde=False)
    plt.axvspan(xmin=lower, xmax=data[column].min(), alpha=0.2, color='red')
    plt.axvspan(xmin=upper, xmax=data[column].max(), alpha=0.2, color='red')
    plt.show()
    
    return


draw_outlier_hist(X,'annual_income')
draw_outlier_hist(X,'loan_amount')


# Outlier Removal for all features
features = ['annual_income','debt_to_income_ratio','loan_amount','interest_rate']
df = data.copy()


Q1 = df[features].quantile(0.25)
Q3 = df[features].quantile(0.75)
IQR = Q3 - Q1

outlier = df[((df[features] < (Q1 - 1.5 * IQR)) | (df[features] > (Q3 + 1.5 * IQR))).any(axis=1)]

df1 = df[~((df[features] < (Q1 - 1.5 * IQR)) | (df[features] > (Q3 + 1.5 * IQR))).any(axis=1)]


print(f'Outlier found : {outlier.shape}')
print(f'Dataset after Outlier removed : {df1.shape}')


label_encoder = LabelEncoder()

cat_cols = df1.select_dtypes(include=['object','category']).columns  # pick categorical columns

for col in cat_cols:
    df1[col] = label_encoder.fit_transform(df1[col].astype(str))


X = df1.drop(['loan_paid_back'],axis=1)
y = df1['loan_paid_back']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)


scaler_std = StandardScaler()

X_train_scaled = scaler_std.fit_transform(X_train)
X_test_scaled = scaler_std.transform(X_test)



# rf_model = RandomForestClassifier(
#                 random_state=42, 
#                 n_jobs=-1, 
#                 n_estimators=800,
#                 verbose=0
#         )




# rf_model.fit(X_train_scaled, y_train)

# y_pred = rf_model.predict(X_test_scaled)

# print(f'\nRandomForestClassifier Best score : {accuracy_score(y_test, y_pred)}\n')
    
# sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="seismic")
# plt.title(f'Confusion Matrix for RandomForestClassifier', fontsize=14)
# plt.savefig(f'Confusion Matrix RandomForestClassifier.png')
# plt.show()


# lgb_model = LGBMClassifier(
#                 random_state=42, 
#                 n_jobs=-1, 
#                 #device="gpu", 
#                 verbosity=-1,
#                 boost_from_average=False, 
#                 force_row_wise=True,
#                 scale_pos_weight=4,        # imbalance dataset positive class weight (ratio of neg/pos)
#                 subsample=0.9,             # row sampling → overfitting reduce
#                 colsample_bytree=0.7,      # feature sampling
#                 boosting_type="gbdt",      # default GBDT
#                 learning_rate=0.01, 
#                 n_estimators=1000
#             )




# lgb_model.fit(X_train_scaled, y_train)

# y_pred = lgb_model.predict(X_test_scaled)

# print(f'\LGBMClassifier Best score : {accuracy_score(y_test, y_pred)}\n')
    
# sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="seismic")
# plt.title(f'Confusion Matrix for LGBMClassifier', fontsize=14)
# plt.savefig(f'Confusion Matrix LGBMClassifier.png')
# plt.show()


# from xgboost import XGBClassifier

# xgb_params = {
#     'n_estimators': 1000,
#     'learning_rate': 0.01,
#     'max_depth': 15,
#     'min_child_weight': 3,
   
#     'gamma': 0.2,
#     'reg_alpha': 0.1,
#     'reg_lambda': 0.3,
#     'tree_method': 'hist',  # 'gpu_hist' if GPU available
#     'random_state': 42
# }

# #xgb_model = XGBClassifier(**xgb_params)



# xgb_model.fit(X_train_scaled, y_train)

# y_pred = xgb_model.predict(X_test_scaled)

# print(f'\nXGBClassifier Best score : {accuracy_score(y_test, y_pred)}\n')
    
# sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="seismic")
# plt.title(f'Confusion Matrix for XGBClassifier', fontsize=14)
# plt.savefig(f'Confusion Matrix XGBClassifier.png')
# plt.show()


test_df = pd.read_csv(TEST_PATH)
test_df


df = test_df.drop(['id'], axis=1)


label_encoder = LabelEncoder()

cat_cols = df.select_dtypes(include=['object','category']).columns  # pick categorical columns

for col in cat_cols:
    df[col] = label_encoder.fit_transform(df[col].astype(str))

#test_sc = standard_scaler.transform(df[numeric_features])


lgb_pred = lgb_model.predict_proba(df)[:, 1]
xgb_pred = xgb_model.predict_proba(df)[:, 1]
rf_pred = rf_model.predict_proba(df)[:,1]
cat_pred = cat_model.predict_proba(df)[:,1]

ensemble_pred = w_xgb * xgb_pred + w_lgb * lgb_pred + w_cat * cat_pred + w_rf * rf_pred



sub = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')
sub


col = sub['id']

test_data = pd.DataFrame({
    'id' : col,
    'loan_paid_back' : ensemble_pred
}) 

test_data


# #df_final = test_data.head(10)
test_data.to_csv('submission.csv',index=False)
s = pd.read_csv('/kaggle/working/submission.csv')
s

