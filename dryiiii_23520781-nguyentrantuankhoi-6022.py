! pip install catboost


!pip install imbalanced-learn optuna


from google.colab import drive

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import json

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, StratifiedKFold
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score
from imblearn.under_sampling import RandomUnderSampler

import shap

import optuna


drive.mount('/content/drive')


df_delay_4_6 = pd.read_csv('/content/drive/MyDrive/DS108/Datasets/Copy of delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv', encoding='utf-8')
df_not_delay_4_6 = pd.read_csv('/content/drive/MyDrive/DS108/Datasets/Copy of not_delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv', encoding='utf-8')

df_delay_7_9 = pd.read_csv('/content/drive/MyDrive/DS108/Datasets/Copy of delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv', encoding='utf-8')
df_not_delay_7_9 = pd.read_csv('/content/drive/MyDrive/DS108/Datasets/Copy of not_delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv', encoding='utf-8')


df_4_6 = pd.concat([df_delay_4_6, df_not_delay_4_6], axis=0, ignore_index=True)
df_4_6.drop_duplicates(inplace=True)

df_7_9 = pd.concat([df_delay_7_9, df_not_delay_7_9], axis=0, ignore_index=True)
df_7_9.drop_duplicates(inplace=True)

common_cols = df_4_6.columns.intersection(df_7_9.columns)
df_full = pd.concat([df_4_6[common_cols], df_7_9[common_cols]], axis=0, ignore_index=True)
df_full.drop_duplicates(inplace=True)


df_10 = pd.read_csv('/content/drive/MyDrive/DS108/Datasets/PILOT_10.csv', encoding='utf-8')


print('Shape of df_4_6:', df_4_6.shape)
print('Shape of df_7_9:', df_7_9.shape)
print('Shape of df_full:', df_full.shape)
print('Shape of df_10:', df_10.shape)


df_10.info()


df_full['Order date'] = pd.to_datetime(df_full['Order date'], format='mixed')


def missing_value_report(df, sort=True, ascending=False):
    total_missing = df.isnull().sum()
    percent_missing = 100 * total_missing / len(df)
    missing_df = pd.DataFrame({
        'Missing Count': total_missing,
        'Missing Rate (%)': percent_missing
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0]

    if sort:
        missing_df = missing_df.sort_values(by='Missing Rate (%)', ascending=ascending)

    return missing_df


missing_value_report(df_10)


# Clean 'OTHER AREA SHIP DIV' column
df_full['OTHER AREA SHIP DIV'] = df_full['OTHER AREA SHIP DIV'].fillna(0)
df_full['OTHER AREA SHIP DIV'] = df_full['OTHER AREA SHIP DIV'].replace([' '], 0)
df_full['OTHER AREA SHIP DIV'] = df_full['OTHER AREA SHIP DIV'].astype('int64')


# extract hour, minute, second as integers
padded = df_full['SO_TIME'].astype(str).str.zfill(6)
df_full['hour order'] = padded.str.slice(0, 2).astype(int)
df_full['minute order'] = padded.str.slice(2, 4).astype(int)
df_full['second order'] = padded.str.slice(4, 6).astype(int)

# convert to datetime, then extract month and day
date_cols = ['VSD']
for col in date_cols:
  df_full[col] = pd.to_datetime(df_full[col], format='mixed')
  df_full[col + ' month'] = df_full[col].dt.month
  df_full[col + ' day'] = df_full[col].dt.day


# Tạo cột day_range bằng cách trừ hai cột datetime
df_full['day_range'] = (df_full['VSD'] - df_full['Order date']).dt.days


df_full['SO_DAY_OF_WEEK_sin'] = np.sin(2 * np.pi * df_full['SO_DAY_OF_WEEK'] / 7)
df_full['SO_DAY_OF_WEEK_cos'] = np.cos(2 * np.pi * df_full['SO_DAY_OF_WEEK'] / 7)
# Tương tự cho hour order (chia cho 24) và SO_DAY_OF_MONTH (chia cho số ngày max của tháng đó, thường là 31)
df_full['SO_DAY_OF_MONTH_sin'] = np.sin(2 * np.pi * df_full['SO_DAY_OF_MONTH'] / 31)
df_full['SO_DAY_OF_MONTH_cos'] = np.cos(2 * np.pi * df_full['SO_DAY_OF_MONTH'] / 31)

df_full['hour_order_sin'] = np.sin(2 * np.pi * df_full['hour order'] / 24)
df_full['hour_order_cos'] = np.cos(2 * np.pi * df_full['hour order'] / 24)


df_full['Is_Weekend'] = df_full['Order date'].dt.dayofweek.isin([5, 6]).astype(int)


# df_not_9['Allocation_Ratio'] = df_not_9['ALLOCATION QTY'] / df_not_9['SO QTY'] # Tỷ lệ phân bổ
df_full['Total_Weight'] = df_full['PACK QTY'] * df_full['WEIGHT PER PIECE'] # Tổng trọng lượng gói hàng
df_full['Weight'] = df_full['SO QTY'] * df_full['WEIGHT PER PIECE'] # Tổng trọng lượng hàng


df_full['Consider count hodiday Saturday'] = df_full['Consider count hodiday Saturday'].replace(' ', 0).astype(int)


print('Columns of full data:', df_full.columns)


missing_value_report(df_full)


df_full.info()


cols_drop = ['QTUF_RCV_NO', 'SOUF_RCV_NO', 'REASON_CD', 'SHIP DECISION NO', 'second order',
             'GLOBAL_NO', 'VSD month', 'minute order', 'SO_TIME', 'SPECIAL_DIV']

rows_drop = ['Ship Mode', 'SUPPLIER_DIV']

df_full = df_full.drop(columns=cols_drop)
df_full = df_full.dropna(subset=rows_drop)


df_9 = df_full[df_full['Order date'].dt.month == 9]
df_not_9 = df_full[df_full['Order date'].dt.month != 9]


df_9 = df_9.drop(columns=['Order date', 'VSD'])
df_not_9 = df_not_9.drop(columns=['Order date', 'VSD'])


correlation_matrix = df_not_9.corr(numeric_only=True)


correlation_with_label = correlation_matrix['label'].sort_values(ascending=False)
print("\nMối tương quan với biến 'label':")
print(correlation_with_label)


plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Ma trận tương quan giữa các thuộc tính')
plt.show()

# Hoặc chỉ trực quan mối tương quan với 'label'
plt.figure(figsize=(8, 6))
sns.barplot(x=correlation_with_label.index, y=correlation_with_label.values, palette='viridis')
plt.title('Mối tương quan của các thuộc tính với Label')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# def clip_outliers(train_df, apply_dfs, columns=None, lower_q=0.01, upper_q=0.99):
#     if columns is None:
#         columns = train_df.select_dtypes(include=['number']).columns.tolist()

#     # Compute clipping thresholds from train data
#     lower_bounds = train_df[columns].quantile(lower_q)
#     upper_bounds = train_df[columns].quantile(upper_q)

#     # Clip train
#     for col in columns:
#         train_df[col] = train_df[col].clip(lower_bounds[col], upper_bounds[col])

#     # Clip other data
#     for df in apply_dfs:
#         for col in columns:
#             df[col] = df[col].clip(lower_bounds[col], upper_bounds[col])


import numpy as np
import pandas as pd

def iv_woe(data, target, bins=10, show_woe=False, force_categorical=[]):

    newDF, woeDF = pd.DataFrame(), pd.DataFrame()
    cols = data.columns

    for ivars in cols[~cols.isin([target])]:
        temp_data = data[[ivars, target]].dropna()

        # convert to categorical data
        if ivars in force_categorical:
            d0 = pd.DataFrame({'x': temp_data[ivars].astype(str), 'y': temp_data[target]})

        # if numeric plit bin qcut
        elif (temp_data[ivars].dtype.kind in 'bifc') and (len(np.unique(temp_data[ivars])) > 10):
            try:
                binned_x = pd.qcut(temp_data[ivars], bins, duplicates='drop')
                d0 = pd.DataFrame({'x': binned_x, 'y': temp_data[target]})
            except:
                print(f"Không thể bin cột {ivars}, có thể do giá trị trùng nhau quá nhiều.")
                continue

        #
        else:
            d0 = pd.DataFrame({'x': temp_data[ivars].astype(str), 'y': temp_data[target]})

        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Events']/d['% of Non-Events'])
        d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        print("Information value of " + ivars + " is " + str(round(d['IV'].sum(),6)))
        temp =pd.DataFrame({"Variable" : [ivars], "IV" : [d['IV'].sum()]}, columns = ["Variable", "IV"])
        newDF=pd.concat([newDF,temp], axis=0)
        woeDF=pd.concat([woeDF,d], axis=0)

        if show_woe == True:
            print(d)

    return newDF.sort_values(by="IV", ascending=False).reset_index(drop=True), woeDF


categorical_cols = ['CLASSIFY_CD', 'CUST_CD','BRAND_CD', 'INNER_CD', 'SUPPLIER_CD','Stock class',
                    'OTHER AREA SHIP DIV', 'PACKING RANK', 'PRODUCT_CD', 'PRODUCT ATTRIBUTION',
                    'SPECIAL DIV','LOGICAL PLANT', 'DIRECT SHIP FLG','DELI_DIV', 'Ship Mode',
                    'SHIP DECISION NO', 'SUPPLIER_DIV', 'SPECIAL_DIV', 'Is_Weekend', ]

iv, woe = iv_woe(
    data=df_not_9,
    target='label',
    bins=15,
    force_categorical=categorical_cols
)

print(iv)


use_cols = ['PRODUCT_CD', 'INNER_CD', 'CUST_CD', 'CLASSIFY_CD', 'DELI_DIV', 'SUPPLIER_CD', 'DIRECT SHIP FLG',
            'BRAND_CD', 'day_range', 'Weight', 'WEIGHT PER PIECE', 'SUPPLIER INV AMOUNT', 'PURCHASE AMOUNT',
            'VSD day', 'Ship Mode']


X_train = df_not_9.drop(columns=['label']) # Assuming 'label' is the target column
y_train = df_not_9['label']
X_test = df_9.drop(columns=['label'])
y_test = df_9['label']


df_not_9.info()


categorical_features = ['SUBSIDIARY_CD', 'CLASSIFY_CD', 'CUST_CD', 'BRAND_CD', 'INNER_CD', 'SUPPLIER_CD',
                        'Sales order line number', 'Stock class', 'OTHER AREA SHIP DIV', 'PRODUCT_CD',
                        'PRODUCT ATTRIBUTION', 'SPECIAL DIV', 'LOGICAL PLANT', 'DIRECT SHIP FLG',
                        'DELI_DIV', 'Ship Mode', 'SUPPLIER_DIV', 'Is_Weekend', 'PACKING RANK']


for col in categorical_features:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)


from catboost import Pool, CatBoostClassifier


train_pool = Pool(
    data=X_train,
    label=y_train,
    cat_features=categorical_features
)

catboost_model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.15,
    depth=6,
    random_seed=42,
    auto_class_weights='Balanced',
    loss_function='Logloss',
    verbose=0 # Tắt log để gọn
)

catboost_model.fit(train_pool)


explainer = shap.TreeExplainer(catboost_model)
shap_values = explainer.shap_values(X_test)


# Vẽ biểu đồ tóm tắt SHAP
shap.summary_plot(shap_values, X_test, max_display=X_test.shape[1])


# Chuyển đổi giá trị SHAP thành DataFrame
shap_values_df = pd.DataFrame(shap_values, columns=X_test.columns)

# Xem các thống kê mô tả của các giá trị SHAP
shap_values_df.describe().T


y_pred_cat = catboost_model.predict(X_test)

# F1-score
f1_CB = f1_score(y_test, y_pred_cat)
precision_CB = precision_score(y_test, y_pred_cat)
recall_CB = recall_score(y_test, y_pred_cat)

print("F1-score (test):", round(f1_CB, 4))
print("Precision (test):", round(precision_CB, 4))
print("Recall (test):", round(recall_CB, 4))

# report
print("\nClassification Report:\n", classification_report(y_test, y_pred_cat, digits=4))


# Import các thư viện cần thiết
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter


print(f"Trước khi undersample và oversample: {Counter(y_train)}")

# Calculate target counts based on desired ratios (majority:minority)
original_minority_count = Counter(y_train)[1]
desired_majority_under = 17 * original_minority_count
desired_minority_under = original_minority_count

under_sampler = RandomUnderSampler(sampling_strategy={0: desired_majority_under, 1: desired_minority_under}, random_state=42)
X_under, y_under = under_sampler.fit_resample(X_train, y_train)
print(f"Sau khi under-sample (30:1 majority:minority): {Counter(y_under)}")


from catboost import Pool, CatBoostClassifier


train_pool = Pool(
    data=X_under,
    label=y_under,
    cat_features=categorical_features
)

catboost_model = CatBoostClassifier(
    iterations=191,
    learning_rate=0.23167839441737345,
    depth=5,
    l2_leaf_reg= 4.090354306055502,
    border_count= 51,
    random_seed=42,
    loss_function='Logloss',
    eval_metric= "F1",
    verbose=0, # Tắt log để gọn
    task_type= "GPU"
)

catboost_model.fit(train_pool)


y_pred = catboost_model.predict(X_test)
f1 = f1_score(y_test, y_pred)
f1


explainer = shap.TreeExplainer(catboost_model)
shap_values = explainer.shap_values(X_test)


# day_range, SUPPLIER_CD, CUST_CD, VSD day, BRAND_CD, CLASSIFY_CD, DELI_DIV, PURCHASE AMOUNT, PRODUCT_CD, INNER_CD, SUPPLIER INV AMOUNT


# 0                        PRODUCT_CD  35.591908
# 1                          INNER_CD  10.325964
# 2                           CUST_CD   3.207646
# 3                       CLASSIFY_CD   1.651734
# 4                          DELI_DIV   1.585817
# 5                       SUPPLIER_CD   1.580959
# 6                   DIRECT SHIP FLG   0.709505
# 7                          BRAND_CD   0.333186
# 8                         day_range   0.324085
# 9                            Weight   0.302927
# 10                 WEIGHT PER PIECE   0.243452
# 11              SUPPLIER INV AMOUNT   0.221067
# 12                  PURCHASE AMOUNT   0.220906
# 13                          VSD day   0.173217
# 14                        Ship Mode   0.155638
# 15                     SUPPLIER_DIV   0.069567
# 16              OTHER AREA SHIP DIV   0.069318
# 17                   ALLOCATION QTY   0.050487
# 18                           SO QTY   0.050487
# 19                  SO_DAY_OF_MONTH   0.039946
# 20          Sales order line number   0.035888
# 21              SO_DAY_OF_MONTH_cos   0.035799
# 22  Consider count hodiday Saturday   0.033336
# 23              SO_DAY_OF_MONTH_sin   0.021692
# 24                    LOGICAL PLANT   0.021422
# 25                   hour_order_sin   0.017975
# 26                   hour_order_cos   0.017131
# 27                       hour order   0.014757
# 28                      SPECIAL DIV   0.013600
# 29               SO_DAY_OF_WEEK_sin   0.008007
# 30                   SO_DAY_OF_WEEK   0.008007
# 31               SO_DAY_OF_WEEK_cos   0.008007
# 32                     PACKING RANK   0.003690
# 33              PRODUCT ATTRIBUTION   0.002568
# 34                      Stock class   0.002568
# 35                       Is_Weekend   0.001024
# 36                    SUBSIDIARY_CD   0.000000
# 37                         PACK QTY   0.000000
# 38                     Total_Weight   0.000000


# Vẽ biểu đồ tóm tắt SHAP
shap.summary_plot(shap_values, X_test, max_display=X_test.shape[1])


# Lọc mẫu class 1
X_c1 = X_test[y_test == 1]
shap_c1 = shap_values[y_test == 1]
shap.summary_plot(shap_c1, X_c1, max_display=X_test.shape[1])


# 1. Lọc mẫu có ground truth = 1
mask_class1 = (y_test == 1)
X_class1 = X_test[mask_class1]
shap_class1 = shap_values[mask_class1]  # giữ đúng thứ tự dòng

# 2. Đưa vào DataFrame để dễ thao tác
shap_df = pd.DataFrame(shap_class1, columns=X_test.columns)

# 3. Tính phần trăm số mẫu có SHAP > 0 cho mỗi feature
pct_positive_shap = (shap_df > 0).sum(axis=0) / shap_df.shape[0] * 100

# 4. Đưa vào DataFrame kết quả
result_df = pd.DataFrame({
    "feature": pct_positive_shap.index,
    "pct_SHAP_positive_in_class1": pct_positive_shap.values
}).sort_values(by="pct_SHAP_positive_in_class1", ascending=False)

result_df


# Lọc mẫu class 0
X_c0 = X_test[y_test == 0]
shap_c0 = shap_values[y_test == 0]
shap.summary_plot(shap_c0, X_c0, max_display=X_test.shape[1])


# 1. Lọc các mẫu có label thực tế là 0
mask_class0 = (y_test == 0)
X_class0 = X_test[mask_class0]
shap_class0 = shap_values[mask_class0]

# 2. Đưa vào DataFrame
shap_df = pd.DataFrame(shap_class0, columns=X_test.columns)

# 3. Tính % số mẫu có SHAP > 0 trong class 0 (tức là feature đang đẩy nhầm về class 1)
pct_shap_positive = (shap_df > 0).sum(axis=0) / shap_df.shape[0] * 100

# 4. Đưa vào bảng kết quả
result_df_2 = pd.DataFrame({
    "feature": pct_shap_positive.index,
    "pct_SHAP_positive_in_class0": pct_shap_positive.values
}).sort_values(by="pct_SHAP_positive_in_class0", ascending=False)

result_df_2


missing_value_report(df_10)





from catboost import Pool, CatBoostClassifier

def objective_CatBoost_Under(trial):
    # Định nghĩa các tham số của CatBoost
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        # "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        # "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "eval_metric": "F1",
        "loss_function": "Logloss",
        "random_seed": 42,
        "verbose": 0,
        "task_type": "GPU"
    }

    # Tạo Pool cho CatBoost
    train_pool = Pool(data=X_under, label=y_under, cat_features=categorical_features)
    test_pool = Pool(data=X_test, label=y_test, cat_features=categorical_features)

    # Khởi tạo và huấn luyện mô hình
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=test_pool, use_best_model=True)

    # Dự đoán và tính F1 score
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    return f1



study_cb_under = optuna.create_study(direction='maximize')
study_cb_under.optimize(objective_CatBoost_Under, n_trials=10)


print("Best F1-score:", study_cb_under.best_value)
print("Best parameters:")
for k, v in study_cb_under.best_params.items():
    print(f"  {k}: {v}")


train_pool = Pool(data=X_under, label=y_under, cat_features=categorical_features)

best_params_cb_under = study_cb_under.best_params
best_model_cb_under = CatBoostClassifier(**best_params_cb_under)
best_model_cb_under.fit(train_pool)


train_pool = Pool(data=X_under, label=y_under, cat_features=categorical_features)

best_model_cb_under = CatBoostClassifier(
    iterations=601,
    depth=9,
    learning_rate=0.2048148456885514,
    l2_leaf_reg=8.266595779431615,
    border_count=240,
    verbose=100 # Add verbose=0 to silence training output
)

best_model_cb_under.fit(train_pool)


explainer = shap.TreeExplainer(best_model_cb_under)
shap_values = explainer.shap_values(X_test)


shap.summary_plot(shap_values, X_test, max_display=X_test.shape[1])


X_train = df_not_9.drop(columns=['label'])
y_train = df_not_9['label']
X_test = df_9.drop(columns=['label'])
y_test = df_9['label']


# X_train = X_train[['day_range', 'SUPPLIER_CD', 'CUST_CD', 'VSD day', 'BRAND_CD', 'CLASSIFY_CD',
#                   'DELI_DIV', 'PURCHASE AMOUNT', 'PRODUCT_CD', 'INNER_CD', 'SUPPLIER INV AMOUNT']]
# X_test = X_test[['day_range', 'SUPPLIER_CD', 'CUST_CD', 'VSD day', 'BRAND_CD', 'CLASSIFY_CD',
#                   'DELI_DIV', 'PURCHASE AMOUNT', 'PRODUCT_CD', 'INNER_CD', 'SUPPLIER INV AMOUNT']]


# categorical_features = ['SUPPLIER_CD', 'CUST_CD', 'BRAND_CD', 'CLASSIFY_CD', 'DELI_DIV', 'PRODUCT_CD', 'INNER_CD']


categorical_features = ['SUBSIDIARY_CD', 'CLASSIFY_CD', 'CUST_CD', 'BRAND_CD', 'INNER_CD', 'SUPPLIER_CD',
                        'Sales order line number', 'Stock class', 'OTHER AREA SHIP DIV', 'PRODUCT_CD',
                        'SPECIAL DIV', 'LOGICAL PLANT', 'DIRECT SHIP FLG',
                        'DELI_DIV', 'Ship Mode', 'SUPPLIER_DIV', 'Is_Weekend', 'PACKING RANK']

X_train['SUPPLIER_DIV'] = X_train['SUPPLIER_DIV'].astype(str)
X_test['SUPPLIER_DIV'] = X_test['SUPPLIER_DIV'].astype(str)


print(f"Trước khi undersample và oversample: {Counter(y_train)}")

# Calculate target counts based on desired ratios (majority:minority)
original_minority_count = Counter(y_train)[1]
desired_majority_under = 17 * original_minority_count
desired_minority_under = original_minority_count

under_sampler = RandomUnderSampler(sampling_strategy={0: desired_majority_under, 1: desired_minority_under}, random_state=42)
X_under, y_under = under_sampler.fit_resample(X_train, y_train)
print(f"Sau khi under-sample (30:1 majority:minority): {Counter(y_under)}")


from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import optuna
import numpy as np

def objective_CatBoost_Under(trial):

    X_tr, X_dev, y_tr, y_dev = train_test_split(
        X_under, y_under,
        test_size=0.20,          # bạn đổi tuỳ ý
        stratify=y_under,
        random_state=trial.number   # khác nhau mỗi trial
    )

    train_pool = Pool(X_tr,  y_tr, cat_features=categorical_features)
    dev_pool   = Pool(X_dev, y_dev, cat_features=categorical_features)

    # Định nghĩa các tham số của CatBoost
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        # "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        # "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "eval_metric": "F1",
        "loss_function": "Logloss",
        "random_seed": 42,
        "verbose": 0,
        "task_type": "GPU"
    }


    # Khởi tạo và huấn luyện mô hình
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=dev_pool, early_stopping_rounds=80)

    y_pred_dev = model.predict(X_dev)
    f1_dev = f1_score(y_dev, y_pred_dev)

    return f1_dev


study_cb_under = optuna.create_study(direction='maximize')
study_cb_under.optimize(objective_CatBoost_Under, n_trials=5)


print("Best F1-score:", study_cb_under.best_value)
print("Best parameters:")
for k, v in study_cb_under.best_params.items():
    print(f"  {k}: {v}")


train_pool = Pool(data=X_under, label=y_under, cat_features=categorical_features)

best_params_cb_under = study_cb_under.best_params
best_model_cb_under = CatBoostClassifier(**best_params_cb_under, task_type= "GPU", verbose=0)
best_model_cb_under.fit(train_pool)


y_pred = best_model_cb_under.predict(X_test)
f1 = f1_score(y_test, y_pred)
f1





X_final = df_full.drop(columns=['label'])
y_final = df_full['label']
X_test_final = df_10


X_final = X_final[['day_range', 'SUPPLIER_CD', 'CUST_CD', 'VSD day', 'BRAND_CD', 'CLASSIFY_CD',
                  'DELI_DIV', 'PURCHASE AMOUNT', 'PRODUCT_CD', 'INNER_CD', 'SUPPLIER INV AMOUNT']]
X_test_final = X_test_final[['day_range', 'SUPPLIER_CD', 'CUST_CD', 'VSD day', 'BRAND_CD', 'CLASSIFY_CD',
                              'DELI_DIV', 'PURCHASE AMOUNT', 'PRODUCT_CD', 'INNER_CD', 'SUPPLIER INV AMOUNT']]




