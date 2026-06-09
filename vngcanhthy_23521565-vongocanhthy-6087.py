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


import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
pd.set_option('display.max_columns', 200)


delay_4_6 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv', na_values=[' ', '', '   '], low_memory=False)
not_delay_4_6 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/not_delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv', na_values=[' ', '', '   '], low_memory=False)
delay_7_9 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv', na_values=[' ', '', '   '], low_memory=False)
not_delay_7_9 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/not_delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv', na_values=[' ', '', '   '], low_memory=False)


data_4_6 = pd.concat([delay_4_6, not_delay_4_6], axis = 0)


data_7_9 = pd.concat([delay_7_9, not_delay_7_9], axis = 0)


data_4_6.drop(columns=['HEAVY_FLG', 'EXPENSIVE_FLG', 'ACTUAL_SHIP_DAYS',
                       'SPECIFY_PRODUCTION_DAYS', 'SPECIFY_SHIP_DAYS',
                       'HAZARD_FLG', 'IO_UNFIT_FLG', 'WEIGHT_UNIT',
                       'SUPPLIER_CATEGORY_CD', 'PRODUCT_ASSORT'], inplace=True)


data_4_6.head(1)


data_7_9.head(1)


def ship_days_expect(data):
    data['Order date'] = pd.to_datetime(data['Order date'])
    data['VSD'] = pd.to_datetime(data['VSD'])
    
    data['SHIP_DAYS_EXPECT'] = (data['VSD'] - data['Order date']).dt.days
    return data


data_4_6 = ship_days_expect(data_4_6)
data_7_9 = ship_days_expect(data_7_9)


data = data_7_9.copy()


data.drop(columns=['Order date', 'VSD', 'REASON_CD', 'SOUF_RCV_NO', 'QTUF_RCV_NO'], inplace=True)


data = data.rename(columns={'Sales order line number': 'SALES_ORDER_LINE_NUMBER',
                            'Stock class': 'STOCK_CLASS',
                            'Consider count hodiday Saturday': 'CONSIDER_COUNT_HOLIDAY_SATURDAY',
                            'SO QTY': 'SO_QTY',
                            'OTHER AREA SHIP DIV': 'OTHER_AREA_SHIP_DIV',
                            'ALLOCATION QTY': 'ALLOCATION_QTY',
                            'SUPPLIER INV AMOUNT': 'SUPPLIER_INV_AMOUNT',
                            'PACKING RANK': 'PACKING_RANK',
                            'PRODUCT ATTRIBUTION': 'PRODUCT_ATTRIBUTION',
                            'SPECIAL DIV': 'SPECIALDIV',
                            'LOGICAL PLANT': 'LOGICAL_PLANT',
                            'PURCHASE AMOUNT': 'PURCHASE_AMOUNT',
                            'DIRECT SHIP FLG': 'DIRECT_SHIP_FLG',
                            'Ship Mode': 'SHIP_MODE',
                            'SHIP DECISION NO': 'SHIP_DECISION_NO',
                            'PACK QTY': 'PACK_QTY',
                            'WEIGHT PER PIECE': 'WEIGHT_PER_PIECE'})


missing_cols = data.isna().sum()[data.isna().sum() > 0]
missing_cols


print("**Missing ratio**")
for col in missing_cols.index:
  missing_ratio = data[col].isna().mean() * 100
  dtype = data[col].dtype
  print(f"{col:<33}- {dtype} - \t: {missing_ratio:.5f}%")


data = data.dropna(subset=['SHIP_MODE', 'SUPPLIER_DIV', 'CONSIDER_COUNT_HOLIDAY_SATURDAY'])


data.OTHER_AREA_SHIP_DIV.fillna(0, inplace=True) 


(data['SPECIAL_DIV'] == data['SPECIALDIV']).all()


(data.STOCK_CLASS == data.PRODUCT_ATTRIBUTION).all()


data.drop(columns=['SPECIALDIV', 'STOCK_CLASS', 'SUBSIDIARY_CD'], inplace=True)


data.duplicated().sum()


data.drop(columns=['GLOBAL_NO', 'INNER_CD', 'PRODUCT_CD', 'SHIP_DECISION_NO', 'SALES_ORDER_LINE_NUMBER'], inplace=True)


data.info()



def z_score_outlier_detection(data, columns, threshold=3):
  outliers_dict = {}
  for column in columns:
    Z_scores = (data[column] - np.mean(data[column])) / np.std(data[column])
    outliers = np.where(np.abs(Z_scores) > threshold)[0]
    outliers_dict[column] = outliers
    total_outliers = len(outliers)
    print(f"Total outliers {column:<40}: {total_outliers}")


numeric_cols = data.select_dtypes(include=['number']).columns
z_score_outlier_detection(data, numeric_cols)


data.loc[data['SO_QTY'] > 400, 'SO_QTY'] = int(data['SO_QTY'].mean())
data.loc[data['ALLOCATION_QTY'] > 400, 'ALLOCATION_QTY'] = int(data['ALLOCATION_QTY'].mean())
data.loc[data['SUPPLIER_INV_AMOUNT'] > 94000, 'SUPPLIER_INV_AMOUNT'] = int(data['SUPPLIER_INV_AMOUNT'].mean())
data.loc[data['PURCHASE_AMOUNT'] > 94000, 'PURCHASE_AMOUNT'] = int(data['PURCHASE_AMOUNT'].mean())
data.loc[data['PACK_QTY'] > 300, 'PACK_QTY'] = int(data['PACK_QTY'].mean())
data.loc[data['WEIGHT_PER_PIECE'] >18000, 'WEIGHT_PER_PIECE'] = int(data['WEIGHT_PER_PIECE'].mean())


cat_features = ['BRAND_CD', 'CLASSIFY_CD', 'SUPPLIER_CD']

for col in cat_features:
  data[col] = data[col].astype('category')
  counts = data[col].value_counts()
  threshold = len(data) * 0.02
  common_data = counts[counts > threshold].index
  data[col] = data[col].apply(lambda x: x if x in common_data else 'Other').astype('category')


data['CUST_CD'] = data['CUST_CD'].astype('category')
counts = data['CUST_CD'].value_counts()
threshold = 10000
common_data = counts[counts > threshold].index
data['CUST_CD'] = data['CUST_CD'].apply(lambda x: x if x in common_data else 'Other').astype('category')


categorical_cols = ['CLASSIFY_CD', 'CUST_CD', 'BRAND_CD', 'SUPPLIER_CD', 'DIRECT_SHIP_FLG',
                   'OTHER_AREA_SHIP_DIV', 'PRODUCT_ATTRIBUTION', 'LOGICAL_PLANT',
                   'SHIP_MODE', 'DELI_DIV', 'PACKING_RANK', 'SO_DAY_OF_MONTH',
                   'SO_DAY_OF_WEEK', 'SUPPLIER_DIV', 'SPECIAL_DIV']


data.SUPPLIER_DIV.value_counts()


one_hot_cols = ['CLASSIFY_CD', 'CUST_CD', 'SUPPLIER_CD', 'LOGICAL_PLANT',
                'SHIP_MODE', 'PACKING_RANK', 'DELI_DIV', 'SUPPLIER_DIV']

data = pd.get_dummies(data, columns=one_hot_cols, drop_first=False)


data['BRAND_CD'] = (data['BRAND_CD'] == 'MSM1').astype(int)


data.info()


X_train = data.drop(columns=['label'])
Y_train = data['label']


test = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/PILOT_10.csv', na_values=[' ', '', '   '], low_memory=False)


id = test['ID']


test.info()


test = ship_days_expect(test)


test.drop(columns=['ID', 'Order date', 'SUBSIDIARY_CD', 'Stock class', 'VSD', 'GLOBAL_NO', 
                   'INNER_CD', 'PRODUCT_CD', 'SHIP DECISION NO', 'Sales order line number', 
                   'SPECIAL DIV','REASON_CD', 'SOUF_RCV_NO', 'QTUF_RCV_NO'], inplace=True)


test = test.rename(columns={'Consider count hodiday Saturday': 'CONSIDER_COUNT_HOLIDAY_SATURDAY',
                            # 'Sales order line number': 'SALES_ORDER_LINE_NUMBER',
                            'SO QTY': 'SO_QTY',
                            'OTHER AREA SHIP DIV': 'OTHER_AREA_SHIP_DIV',
                            'ALLOCATION QTY': 'ALLOCATION_QTY',
                            'SUPPLIER INV AMOUNT': 'SUPPLIER_INV_AMOUNT',
                            'PACKING RANK': 'PACKING_RANK',
                            'PRODUCT ATTRIBUTION': 'PRODUCT_ATTRIBUTION',
                            'LOGICAL PLANT': 'LOGICAL_PLANT',
                            'PURCHASE AMOUNT': 'PURCHASE_AMOUNT',
                            'DIRECT SHIP FLG': 'DIRECT_SHIP_FLG',
                            'Ship Mode': 'SHIP_MODE',
                            # 'SHIP DECISION NO': 'SHIP_DECISION_NO',
                            'PACK QTY': 'PACK_QTY',
                            'WEIGHT PER PIECE': 'WEIGHT_PER_PIECE'})


test.OTHER_AREA_SHIP_DIV.fillna(0, inplace=True)


test = pd.get_dummies(test, columns=one_hot_cols, drop_first=False)
test = test.reindex(columns=X_train.columns, fill_value=0)


test['BRAND_CD'] = (test['BRAND_CD'] == 'MSM1').astype(int)


test.info()


if list(X_train.columns) == list(test.columns):
    print("True")
else:
    print("False")
    print("X_test.columns:", list(X_train.columns))
    print("test.columns   :", list(test.columns))



# pip install xgboost


import xgboost as xgb

modelXGB = xgb.XGBClassifier(
    enable_categorical=True,
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss'
)

modelXGB.fit(X_train, Y_train)


Y_pred_XGB = modelXGB.predict(test)


Y_pred_XGB = pd.DataFrame(Y_pred_XGB, columns=['label'])
Y_pred_XGB['ID'] = id
Y_pred_XGB = Y_pred_XGB[['ID', 'label']]


Y_pred_XGB['label'].value_counts()


# !pip install lightgbm


import lightgbm as lgb

modelLGB = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    # class_weight={0: 2, 1: 5},
    num_leaves=31,
    random_state=42
)
modelLGB.fit(X_train, Y_train)


Y_pred_LGB=modelLGB.predict(test)


Y_pred_LGB = pd.DataFrame(Y_pred_LGB, columns=['label'])
Y_pred_LGB['ID'] = id
Y_pred_LGB = Y_pred_LGB[['ID', 'label']]


Y_pred_LGB['label'].value_counts()


from sklearn.ensemble import RandomForestClassifier

modelRF = RandomForestClassifier(
    n_estimators=100, 
    max_depth=None, 
    random_state=42
)

modelRF.fit(X_train, Y_train)

Y_pred_RF = modelRF.predict(test)


Y_pred_RF = pd.DataFrame(Y_pred_RF, columns=['label'])
Y_pred_RF['ID'] = id
Y_pred_RF = Y_pred_RF[['ID', 'label']]


Y_pred_RF['label'].value_counts()


Y_pred_RF.to_csv('submission.csv', index=False)


from sklearn.metrics import confusion_matrix

# Confusion matrix
cm = confusion_matrix(Y_pred_RF['label'], Y_pred_LGB['label'])
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False)
plt.title('Confusion Matrix - LightGBM')
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.show()


from sklearn.metrics import confusion_matrix

# Confusion matrix
cm = confusion_matrix(Y_pred_RF['label'], Y_pred_XGB['label'])
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False)
plt.title('Confusion Matrix - LightGBM')
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.show()

