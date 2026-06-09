import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns


print(os.listdir("../input/"))


app_train = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
print('Training data shape: ', app_train.shape)
app_train.head()


# Testing data features
app_test = pd.read_csv('../input/home-credit-default-risk/application_test.csv')
print('Testing data shape: ', app_test.shape)
app_test.head()


app_train['TARGET'].value_counts()


app_train['TARGET'].astype(int).plot.hist();


# 缺失值探查
def missing_values_table(df):
        mis_val = df.isnull().sum()
        
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        return mis_val_table_ren_columns


missing_values = missing_values_table(app_train)
missing_values.head(20)


# 数据类型
app_train.dtypes.value_counts()


# unique
app_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


# 标签变量映射
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in app_train:
    if app_train[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(app_train[col].unique())) <= 2:
            # Train on the training data
            le.fit(app_train[col])
            # Transform both training and testing data
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)




app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)


# 数据集对齐
train_labels = app_train['TARGET']

app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)

app_train['TARGET'] = train_labels

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)


# 异常值探查

(app_train['DAYS_BIRTH'] / -365).describe()



app_train['DAYS_EMPLOYED'].describe()


# 异常值分析
app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
plt.xlabel('Days Employment');
anom = app_train[app_train['DAYS_EMPLOYED'] == 365243]
non_anom = app_train[app_train['DAYS_EMPLOYED'] != 365243]
print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean()))
print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean()))
print('There are %d anomalous days of employment' % len(anom))


import numpy as np

# 1. 異常値（365243）を除外したデータの中央値を計算
# app_train["DAYS_EMPLOYED"] != 365243 の条件を満たす値のみを抽出
median_days_employed = app_train.loc[app_train["DAYS_EMPLOYED"] != 365243, "DAYS_EMPLOYED"].median()

# 2. 異常値（365243）を計算した中央値で置き換える
# inplace=True で元のデータフレームを直接変更
app_train['DAYS_EMPLOYED'].replace({365243: median_days_employed}, inplace=True)

# 3. テストデータも同様に処理（重要：訓練データの中央値を使用）
# テストデータも、訓練データで計算した中央値で埋める
app_test['DAYS_EMPLOYED'].replace({365243: median_days_employed}, inplace=True)

# 結果の確認
print(f"DAYS_EMPLOYEDの異常値は、中央値 {median_days_employed:.0f} で置き換えられました。")
print(app_train['DAYS_EMPLOYED'].describe())


#分布は？
plt.figure(figsize = (10, 6)) # グラフのサイズを設定
# bins=100 で棒の数を増やし、より詳細な分布を表示
plt.hist(app_train['DAYS_EMPLOYED'], bins = 100, edgecolor = 'k') 

plt.title('Days Employment Distribution (After Imputation)') # グラフのタイトル
plt.xlabel('Days Employment (in days)') # X軸ラベル
plt.ylabel('Frequency') # Y軸ラベル
plt.savefig('days_employed_hist.png') # 画像ファイルとして保存
# plt.show() はKaggleノートブックでは不要/非推奨です


# 1. 訓練データの中央値の計算 (異常値 365243 を除外して計算)
# ※ この中央値は、既に前処理として計算・保持されていると仮定しています。
median_days_employed = app_train.loc[app_train["DAYS_EMPLOYED"] != 365243, "DAYS_EMPLOYED"].median()

# 2. テストデータに異常値フラグ列を作成（元の異常値を記録）
app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243

# 3. テストデータの異常値を、訓練データから計算した中央値で置き換える
# 訓練データの中央値 (median_days_employed) を使用することが重要です。
app_test["DAYS_EMPLOYED"].replace({365243: median_days_employed}, inplace = True)

print('テストデータのDAYS_EMPLOYEDの異常値は、訓練データの中央値で置き換えられました。')
print('テストデータ内の異常値件数:', app_test["DAYS_EMPLOYED_ANOM"].sum())
print(app_test['DAYS_EMPLOYED'].describe())


correlations = app_train.corr()['TARGET'].sort_values()

print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))


# DAYS_BIRTH 0.078239
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
app_train['DAYS_BIRTH'].corr(app_train['TARGET'])


plt.style.use('fivethirtyeight')

plt.hist(app_train['DAYS_BIRTH'] / 365, edgecolor = 'k', bins = 25)
plt.title('Age of Client'); plt.xlabel('Age (years)'); plt.ylabel('Count');


plt.figure(figsize = (10, 8))

sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'target == 0')

sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target == 1')

plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');


# 对年龄字段作进一步处理

age_data = app_train[['TARGET', 'DAYS_BIRTH']]
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365

age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
age_groups  = age_data.groupby('YEARS_BINNED').mean()

plt.figure(figsize = (8, 8))

plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])

plt.xticks(rotation = 75); plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group');


# EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3 负相关变量
ext_data = app_train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
ext_data_corrs = ext_data.corr()
plt.figure(figsize = (8, 6))

sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');



# 查看分布（同Age）

plt.figure(figsize = (10, 12))

for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
    
    plt.subplot(3, 1, i + 1)
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, source], label = 'target == 0')
    # plot loans that were not repaid
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, source], label = 'target == 1')
    
    plt.title('Distribution of %s by Target Value' % source)
    plt.xlabel('%s' % source); plt.ylabel('Density');
    
plt.tight_layout(h_pad = 2.5)


plot_data = ext_data.drop(columns = ['DAYS_BIRTH']).copy()

plot_data['YEARS_BIRTH'] = age_data['YEARS_BIRTH']

plot_data = plot_data.dropna().loc[:100000, :]

def corr_func(x, y, **kwargs):
    r = np.corrcoef(x, y)[0][1]
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.2, .8), xycoords=ax.transAxes,
                size = 20)

grid = sns.PairGrid(data = plot_data, height = 3, diag_sharey=False,
                    hue = 'TARGET', 
                    vars = [x for x in list(plot_data.columns) if x != 'TARGET'])

grid.map_upper(plt.scatter, alpha = 0.2)

grid.map_diag(sns.kdeplot)

grid.map_lower(sns.kdeplot, cmap = plt.cm.OrRd_r);

plt.suptitle('Ext Source and Age Features Pairs Plot', size = 32, y = 1.05);


# EXT_SOURCE & DAYS_BIRTH字段建立多项式特征

poly_features = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
poly_features_test = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')

poly_target = poly_features['TARGET']

poly_features = poly_features.drop(columns = ['TARGET'])

poly_features = imputer.fit_transform(poly_features)
poly_features_test = imputer.transform(poly_features_test)

from sklearn.preprocessing import PolynomialFeatures

poly_transformer = PolynomialFeatures(degree = 3)


poly_transformer.fit(poly_features)

poly_features = poly_transformer.transform(poly_features)
poly_features_test = poly_transformer.transform(poly_features_test)
print('Polynomial Features shape: ', poly_features.shape)


poly_transformer.get_feature_names_out(input_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])[:30]


# 自建指标自相关性计算
poly_features = pd.DataFrame(poly_features, 
                             columns = poly_transformer.get_feature_names_out(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                               'EXT_SOURCE_3', 'DAYS_BIRTH']))
poly_features['TARGET'] = poly_target

poly_corrs = poly_features.corr()['TARGET'].sort_values()

print(poly_corrs.head(10))
print(poly_corrs.tail(5))


# 插入新列
poly_features_test = pd.DataFrame(poly_features_test, 
                                  columns = poly_transformer.get_feature_names_out(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                                    'EXT_SOURCE_3', 'DAYS_BIRTH']))

poly_features['SK_ID_CURR'] = app_train['SK_ID_CURR']
app_train_poly = app_train.merge(poly_features, on = 'SK_ID_CURR', how = 'left')

poly_features_test['SK_ID_CURR'] = app_test['SK_ID_CURR']
app_test_poly = app_test.merge(poly_features_test, on = 'SK_ID_CURR', how = 'left')

app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join = 'inner', axis = 1)

print('Training data with polynomial features shape: ', app_train_poly.shape)
print('Testing data with polynomial features shape:  ', app_test_poly.shape)


# 含义变量 ：CREDIT_INCOME_PERCENT、ANNUITY_INCOME_PERCENT、CREDIT_TERM、DAYS_EMPLOYED_PERCENT
app_train_domain = app_train.copy()
app_test_domain = app_test.copy()

app_train_domain['CREDIT_INCOME_PERCENT'] = app_train_domain['AMT_CREDIT'] / app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['ANNUITY_INCOME_PERCENT'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['CREDIT_TERM'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT']
app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain['DAYS_EMPLOYED'] / app_train_domain['DAYS_BIRTH']

app_test_domain['CREDIT_INCOME_PERCENT'] = app_test_domain['AMT_CREDIT'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['ANNUITY_INCOME_PERCENT'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['CREDIT_TERM'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_CREDIT']
app_test_domain['DAYS_EMPLOYED_PERCENT'] = app_test_domain['DAYS_EMPLOYED'] / app_test_domain['DAYS_BIRTH']


plt.figure(figsize = (12, 20))

for i, feature in enumerate(['CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM', 'DAYS_EMPLOYED_PERCENT']):
    
    plt.subplot(4, 1, i + 1)
    sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET'] == 0, feature], label = 'target == 0')
    sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET'] == 1, feature], label = 'target == 1')
    
    plt.title('Distribution of %s by Target Value' % feature)
    plt.xlabel('%s' % feature); plt.ylabel('Density');
    
plt.tight_layout(h_pad = 2.5)


# 统一数据处理
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import pandas as pd

if 'TARGET' in app_train.columns:
    y_train = app_train['TARGET']
    X_train = app_train.drop(columns=['TARGET'])
else:
    y_train = None
    X_train = app_train.copy()


X_test = app_test.copy()

X_train, X_test = X_train.align(X_test, join='inner', axis=1)


imputer = SimpleImputer(strategy='median')
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train),
                           columns=X_train.columns, index=X_train.index)
X_test_imp  = pd.DataFrame(imputer.transform(X_test),
                           columns=X_test.columns, index=X_test.index)

scaler = MinMaxScaler(feature_range=(0, 1))
train = pd.DataFrame(scaler.fit_transform(X_train_imp),
                              columns=X_train_imp.columns, index=X_train_imp.index)
test  = pd.DataFrame(scaler.transform(X_test_imp),
                              columns=X_test_imp.columns, index=X_test_imp.index)

print('Training data shape:', train.shape)
print('Testing data shape :', test.shape)


# 通用预处理与数据集构造
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

assert 'TARGET' in app_train.columns, "app_train 中未找到 TARGET"
y_train = app_train['TARGET'].copy()

def build_X(app_train_like: pd.DataFrame, app_test_like: pd.DataFrame):

    if 'TARGET' in app_train_like.columns:
        X_tr = app_train_like.drop(columns=['TARGET']).copy()
    else:
        X_tr = app_train_like.copy()
    X_te = app_test_like.copy()

    X_tr, X_te = X_tr.align(X_te, join='inner', axis=1)

    imputer = SimpleImputer(strategy='median')
    X_tr_imp = pd.DataFrame(imputer.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
    X_te_imp = pd.DataFrame(imputer.transform(X_te), columns=X_te.columns, index=X_te.index)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_tr_scaled = pd.DataFrame(scaler.fit_transform(X_tr_imp), columns=X_tr_imp.columns, index=X_tr_imp.index)
    X_te_scaled = pd.DataFrame(scaler.transform(X_te_imp), columns=X_te_imp.columns, index=X_te_imp.index)

    return X_tr_scaled, X_te_scaled

train_orig,  test_orig  = build_X(app_train,        app_test)
train_poly,  test_poly  = build_X(app_train_poly,   app_test_poly)
train_domain,test_domain= build_X(app_train_domain, app_test_domain)

print("train_orig / test_orig:",   train_orig.shape,  test_orig.shape)
print("train_poly / test_poly:",   train_poly.shape,  test_poly.shape)
print("train_domain / test_domain:", train_domain.shape, test_domain.shape)

tr_idx, val_idx = train_test_split(
    np.arange(len(y_train)),
    test_size=0.2,
    stratify=y_train,
    random_state=42
)

def split_by_index(X: pd.DataFrame, y: pd.Series, tr_idx, val_idx):
    return X.iloc[tr_idx], X.iloc[val_idx], y.iloc[tr_idx], y.iloc[val_idx]

Xtr_o, Xval_o, ytr_o, yval_o = split_by_index(train_orig, y_train, tr_idx, val_idx)

Xtr_p, Xval_p, ytr_p, yval_p = split_by_index(train_poly, y_train, tr_idx, val_idx)

Xtr_d, Xval_d, ytr_d, yval_d = split_by_index(train_domain, y_train, tr_idx, val_idx)

print("原始数据划分:",   Xtr_o.shape, Xval_o.shape)
print("多项式数据划分:", Xtr_p.shape, Xval_p.shape)
print("含义变量划分:",   Xtr_d.shape, Xval_d.shape)



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

X_tr, X_val, y_tr, y_val = train_test_split(
    train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

log_reg = LogisticRegression(C=0.0001, max_iter=1000)
log_reg.fit(X_tr, y_tr)

val_pred = log_reg.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_pred)
print(f"Validation AUC: {val_auc:.4f}")



from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

def run_dt(Xtr, ytr, Xval, yval,
           max_depth=6, min_samples_leaf=10, class_weight=None):
    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=42
    )
    dt.fit(Xtr, ytr)
    val_pred = dt.predict_proba(Xval)[:, 1]
    return roc_auc_score(yval, val_pred), dt

dt_scores = {}
dt_models = {}

for name, Xtr, Xval, ytr, yval in [
    ("orig", Xtr_o, Xval_o, ytr_o, yval_o),
    ("poly", Xtr_p, Xval_p, ytr_p, yval_p),
    ("domain", Xtr_d, Xval_d, ytr_d, yval_d),
]:
    score, model = run_dt(Xtr, ytr, Xval, yval, max_depth=6, min_samples_leaf=10)
    dt_scores[name] = score
    dt_models[name] = model

print("DecisionTree AUCs:", dt_scores)



from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
import time

def run_knn(Xtr, ytr, Xval, yval,
            n_neighbors=5, weights='distance', metric='minkowski', p=2):

    knn = KNeighborsClassifier(n_neighbors=n_neighbors,
                               weights=weights,
                               metric=metric,
                               p=p,
                               n_jobs=-1)
    t0 = time.time()
    knn.fit(Xtr, ytr)
    t1 = time.time()
    val_pred = knn.predict_proba(Xval)[:, 1]
    auc = roc_auc_score(yval, val_pred)
    return auc, knn, t1 - t0

knn_scores = {}
knn_models = {}
knn_times = {}

for name, Xtr, Xval, ytr, yval in [
    ("orig", Xtr_o, Xval_o, ytr_o, yval_o),
    ("poly", Xtr_p, Xval_p, ytr_p, yval_p),
    ("domain", Xtr_d, Xval_d, ytr_d, yval_d),
]:
    score, model, elapsed = run_knn(Xtr, ytr, Xval, yval, n_neighbors=5, weights='distance')
    knn_scores[name] = score
    knn_models[name] = model
    knn_times[name] = elapsed

print("KNN AUCs:", knn_scores)


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
import time

def run_nb(Xtr, ytr, Xval, yval, var_smoothing=1e-9):

    nb = GaussianNB(var_smoothing=var_smoothing)
    t0 = time.time()
    nb.fit(Xtr, ytr)
    t1 = time.time()
    val_pred = nb.predict_proba(Xval)[:, 1]
    auc = roc_auc_score(yval, val_pred)
    return auc, nb, t1 - t0

nb_scores = {}
nb_models = {}
nb_times = {}

for name, Xtr, Xval, ytr, yval in [
    ("orig", Xtr_o, Xval_o, ytr_o, yval_o),
    ("poly", Xtr_p, Xval_p, ytr_p, yval_p),
    ("domain", Xtr_d, Xval_d, ytr_d, yval_d),
]:
    score, model, elapsed = run_nb(Xtr, ytr, Xval, yval, var_smoothing=1e-9)
    nb_scores[name] = score
    nb_models[name] = model
    nb_times[name] = elapsed

print("NaiveBayes AUCs:", nb_scores)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def run_rf(Xtr, ytr, Xval, yval):
    rf = RandomForestClassifier(
        n_estimators=400, 
        max_depth=None, 
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(Xtr, ytr)
    val_pred = rf.predict_proba(Xval)[:, 1]
    return roc_auc_score(yval, val_pred)

rf_scores = {
    "orig":   run_rf(Xtr_o, ytr_o, Xval_o, yval_o),
    "poly":   run_rf(Xtr_p, ytr_p, Xval_p, yval_p),
    "domain": run_rf(Xtr_d, ytr_d, Xval_d, yval_d),
}
print("RandomForest AUCs:", rf_scores)



from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

def run_et(Xtr, ytr, Xval, yval):
    et = ExtraTreesClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=False,
        n_jobs=-1,
        random_state=42
    )
    et.fit(Xtr, ytr)
    val_pred = et.predict_proba(Xval)[:, 1]
    return roc_auc_score(yval, val_pred)

et_scores = {
    "orig":   run_et(Xtr_o, ytr_o, Xval_o, yval_o),
    "poly":   run_et(Xtr_p, ytr_p, Xval_p, yval_p),
    "domain": run_et(Xtr_d, ytr_d, Xval_d, yval_d),
}
print("ExtraTrees AUCs:", et_scores)



# === Cell 5: 模型#4 - HistGradientBoosting ===
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

def run_hgb(Xtr, ytr, Xval, yval):
    hgb = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.06,
        max_depth=None,   # 让算法自动选择
        l2_regularization=0.0,
        early_stopping=True,
        random_state=42
    )
    hgb.fit(Xtr, ytr)
    val_pred = hgb.predict_proba(Xval)[:, 1]
    return roc_auc_score(yval, val_pred)

hgb_scores = {
    "orig":   run_hgb(Xtr_o, ytr_o, Xval_o, yval_o),
    "poly":   run_hgb(Xtr_p, ytr_p, Xval_p, yval_p),
    "domain": run_hgb(Xtr_d, ytr_d, Xval_d, yval_d),
}
print("HistGB AUCs:", hgb_scores)



from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import time

def run_xgb(Xtr, ytr, Xval, yval,
            n_estimators=1000, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=30):

    xgb = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        use_label_encoder=False,
        eval_metric='auc',
        n_jobs=-1,
        random_state=42
    )
    t0 = time.time()
    xgb.fit(Xtr, ytr, eval_set=[(Xval, yval)], early_stopping_rounds=early_stopping_rounds, verbose=False)
    t1 = time.time()
    val_pred = xgb.predict_proba(Xval)[:, 1]
    auc = roc_auc_score(yval, val_pred)
    return auc, xgb, t1 - t0

xgb_scores = {}
xgb_models = {}
xgb_times = {}

for name, Xtr, Xval, ytr, yval in [
    ("orig", Xtr_o, Xval_o, ytr_o, yval_o),
    ("poly", Xtr_p, Xval_p, ytr_p, yval_p),
    ("domain", Xtr_d, Xval_d, ytr_d, yval_d),
]:
    score, model, elapsed = run_xgb(Xtr, ytr, Xval, yval,
                                    n_estimators=1000, learning_rate=0.05, max_depth=6,
                                    subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=30)
    xgb_scores[name] = score
    xgb_models[name] = model
    xgb_times[name] = elapsed

print("XGBoost AUCs:", xgb_scores)


from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def run_stack(Xtr, ytr, Xval, yval):
    base_estimators = [
        ("lr", LogisticRegression(C=0.2, max_iter=1000, n_jobs=None, solver="lbfgs")),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
        ("gb", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)),
    ]
    meta = LogisticRegression(C=0.5, max_iter=2000, solver="lbfgs")
    stk = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1
    )
    stk.fit(Xtr, ytr)
    val_pred = stk.predict_proba(Xval)[:, 1]
    return roc_auc_score(yval, val_pred)

stack_scores = {
    "orig":   run_stack(Xtr_o, ytr_o, Xval_o, yval_o),
    "poly":   run_stack(Xtr_p, ytr_p, Xval_p, yval_p),
    "domain": run_stack(Xtr_d, ytr_d, Xval_d, yval_d),
}
print("Stacking AUCs:", stack_scores)


