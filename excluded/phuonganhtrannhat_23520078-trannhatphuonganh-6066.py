import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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


df_delay_4_6 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv', na_values=[' ', '', '   '], low_memory=False)
df_not_delay_4_6 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/not_delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv', na_values=[' ', '', '   '], low_memory=False)
df_delay_7_9 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv', na_values=[' ', '', '   '], low_memory=False)
df_not_delay_7_9 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/not_delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv', na_values=[' ', '', '   '], low_memory=False)


# Gộp dữ liệu
A = pd.concat([df_delay_4_6, df_not_delay_4_6], ignore_index=True)
B = pd.concat([df_delay_7_9, df_not_delay_7_9], ignore_index=True)


if A["SPECIAL_DIV"].equals(A["SPECIAL DIV"]): 
    print('Hai cột giống nhau, chỉ giữ lại một cột')
    A.drop(columns=["SPECIAL DIV"], inplace=True) 


# Chuyển cột 'Order date' sang kiểu datetime
A['Order date'] = pd.to_datetime(A['Order date'])

# Chuyển cột 'VSD' sang kiểu datetime
A['VSD'] = pd.to_datetime(A['VSD'])


A['Order_Year'] = A['Order date'].dt.year
A['Order_Month'] = A['Order date'].dt.month
A['Time_to_Ship'] = (A['VSD'] - A['Order date']).dt.days

# Loại bỏ cột gốc sau khi trích xuất
A.drop(columns=['Order date', 'VSD'], inplace=True) 


# Kiểm tra missing values
total_missing = A.isnull().sum()
total_cells = A.size

# Tính phần trăm giá trị thiếu cho mỗi cột
percentage_missing = (total_missing / len(A)) * 100

for col, missing, percent in zip(A.columns, total_missing, percentage_missing):
    print(f"- {col}: {missing} giá trị thiếu ({percent:.2f}%)")


# Drop các cột có quá nhiều missing values
columns_to_drop = [ "SOUF_RCV_NO", "QTUF_RCV_NO"]

A = A.drop(columns=columns_to_drop, errors='ignore')


# Kiểm tra thuộc tính chỉ có khoảng trắng
str_cols = A.select_dtypes(include='object').columns

# Tạo dictionary để lưu kết quả
blankspace_counts = {}

for col in str_cols:
    series = A[col].dropna()

    count = series.apply(lambda x: isinstance(x, str) and (
        x.strip() == ''              # chỉ chứa khoảng trắng

    )).sum()

    blankspace_counts[col] = count

for col, count in blankspace_counts.items():
    print(f"{col}: {count} ô có khoảng trắng")


# Drop các cột chỉ chứa khoảng trống
A = A.drop(columns=["PRODUCT_ASSORT", "REASON_CD"], errors='ignore')


# Thay thế các khoảng trắng ở thuộc tính OTHER AREA SHIP DIV thành 0
A["OTHER AREA SHIP DIV"] = A["OTHER AREA SHIP DIV"].astype(str).str.strip().replace("", "0").fillna("0")


# Kiểm tra các cột chỉ có một giá trị
unique_value_counts = A.apply(lambda col: len(col.unique()))
single_value_cols = unique_value_counts[unique_value_counts == 1].index

for col in single_value_cols:
    print(f"- {col}: {A[col].unique()[0]}")


# Drop các cột chỉ có một giá trị
columns_to_drop = ["SUBSIDIARY_CD", "HAZARD_FLG", "Order_Year"]

A = A.drop(columns=columns_to_drop, errors='ignore')


# Drop các cột chỉ có ở tập A mà không có ở tập B
columns_to_drop = ["HEAVY_FLG", "EXPENSIVE_FLG", "ACTUAL_SHIP_DAYS", "SPECIFY_PRODUCTION_DAYS", "SPECIFY_SHIP_DAYS", "IO_UNFIT_FLG", "WEIGHT_UNIT", 
                   "SUPPLIER_CATEGORY_CD"]

A = A.drop(columns=columns_to_drop, errors='ignore')


# Lọc ra các cột có kiểu dữ liệu object
object_columns = A.select_dtypes(include=['object'])

for column in object_columns.columns:
    print(f"Cột {column}: {object_columns[column].unique()}")


# Chuyển đổi cột có thể thành số 
numeric_columns = ['Consider count hodiday Saturday', 'OTHER AREA SHIP DIV']
for column in numeric_columns:
    A[column] = pd.to_numeric(A[column], errors='coerce') 


# Chuyển đổi cột có thể thành category 
category_columns = ['PACKING RANK', 'DELI_DIV', 'Ship Mode']
for column in category_columns:
    A[column] = A[column].astype('category')


category_features = A.select_dtypes(include=['object', 'category']).columns

# Chỉ chọn những cột có số lượng giá trị duy nhất nhỏ hơn 100
selected_categories = [col for col in category_features if A[col].nunique() < 100]
A = pd.get_dummies(A, columns=selected_categories)

# Nếu số lượng giá trị duy nhất lớn, dùng Label Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in category_features:
    if A[col].nunique() >= 100:
        A[col] = le.fit_transform(A[col])


# Chỉ lấy các cột kiểu số
numeric_cols = A.select_dtypes(include=['number'])

# Vẽ heatmap
plt.figure(figsize=(25, 10))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Heatmap của ma trận tương quan giữa các biến số")
plt.show()


# Drop các cột đa cộng tuyến
A = A.drop(columns=['SUPPLIER INV AMOUNT', 'Stock class', 'ALLOCATION QTY'])


A.columns = A.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True)


if B["SPECIAL_DIV"].equals(B["SPECIAL DIV"]): 
    print('Hai cột giống nhau, chỉ giữ lại một cột')
    B.drop(columns=["SPECIAL DIV"], inplace=True) 


# Chuyển cột 'Order date' sang kiểu datetime
B['Order date'] = pd.to_datetime(B['Order date'])

# Chuyển cột 'VSD' sang kiểu datetime
B['VSD'] = pd.to_datetime(B['VSD'])


B['Order_Year'] = B['Order date'].dt.year
B['Order_Month'] = B['Order date'].dt.month
B['Time_to_Ship'] = (B['VSD'] - B['Order date']).dt.days

# Loại bỏ cột gốc sau khi trích xuất
B.drop(columns=['Order date', 'VSD'], inplace=True) 


# Kiểm tra missing values
total_missing = B.isnull().sum()
total_cells = B.size

# Tính phần trăm giá trị thiếu cho mỗi cột
percentage_missing = (total_missing / len(B)) * 100

for col, missing, percent in zip(B.columns, total_missing, percentage_missing):
    print(f"- {col}: {missing} giá trị thiếu ({percent:.2f}%)")


# Drop các cột có quá nhiều missing values
columns_to_drop = ["SOUF_RCV_NO", "QTUF_RCV_NO", "REASON_CD"]

B = B.drop(columns=columns_to_drop, errors='ignore')


# Thay thế các khoảng trắng thành 0
B["OTHER AREA SHIP DIV"] = B["OTHER AREA SHIP DIV"].astype(str).str.strip().replace("", "0").fillna("0")


# Kiểm tra thuộc tính chỉ có khoảng trắng
str_cols = B.select_dtypes(include='object').columns

# Tạo dictionary để lưu kết quả
blankspace_counts = {}

for col in str_cols:
    series = B[col].dropna()

    count = series.apply(lambda x: isinstance(x, str) and (
        x.strip() == ''              # chỉ chứa khoảng trắng

    )).sum()

    blankspace_counts[col] = count

for col, count in blankspace_counts.items():
    print(f"{col}: {count} ô có khoảng trắng")


# Kiểm tra các cột chỉ có một giá trị
unique_value_counts = B.apply(lambda col: len(col.unique()))
single_value_cols = unique_value_counts[unique_value_counts == 1].index

for col in single_value_cols:
    print(f"- {col}: {B[col].unique()[0]}")


# Drop các cột chỉ có một giá trị
columns_to_drop = ["SUBSIDIARY_CD", "Order_Year"]

B = B.drop(columns=columns_to_drop, errors='ignore')


# Lọc ra các cột có kiểu dữ liệu object
object_columns = B.select_dtypes(include=['object'])

for column in object_columns.columns:
    print(f"Cột {column}: {object_columns[column].unique()}")


# Chuyển đổi cột có thể thành số 
numeric_columns = ['OTHER AREA SHIP DIV']
for column in numeric_columns:
    B[column] = pd.to_numeric(B[column], errors='coerce')


# Chuyển đổi cột có thể thành category 
category_columns = ['PACKING RANK', 'DELI_DIV', 'Ship Mode']
for column in category_columns:
    B[column] = B[column].astype('category')


category_features = B.select_dtypes(include=['object', 'category']).columns

# Chỉ chọn những cột có số lượng giá trị duy nhất nhỏ hơn 100
selected_categories = [col for col in category_features if B[col].nunique() < 100]
B = pd.get_dummies(B, columns=selected_categories)

# Nếu số lượng giá trị duy nhất lớn, dùng Label Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in category_features:
    if B[col].nunique() >= 100:
        B[col] = le.fit_transform(B[col])


# Chỉ lấy các cột kiểu số
numeric_cols = B.select_dtypes(include=['number'])

# Vẽ heatmap
plt.figure(figsize=(25, 10))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Heatmap của ma trận tương quan giữa các biến số")
plt.show()


# Drop các cột đa cộng tuyến
B = B.drop(columns=['SUPPLIER INV AMOUNT', 'Stock class', 'ALLOCATION QTY'])


B.columns = B.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True)


data = pd.concat([A, B])


test = pd.read_csv('PILOT_10.csv')


if test["SPECIAL_DIV"].equals(test["SPECIAL DIV"]): 
    print('Hai cột giống nhau, chỉ giữ lại một cột')
    test.drop(columns=["SPECIAL DIV"], inplace=True) 


# Chuyển cột 'Order date' sang kiểu datetime
test['Order date'] = pd.to_datetime(test['Order date'])

# Chuyển cột 'VSD' sang kiểu datetime
test['VSD'] = pd.to_datetime(test['VSD'])


test['Order_Year'] = test['Order date'].dt.year
test['Order_Month'] = test['Order date'].dt.month
test['Time_to_Ship'] = (test['VSD'] - test['Order date']).dt.days

# Loại bỏ cột gốc sau khi trích xuất
test.drop(columns=['Order date', 'VSD'], inplace=True) 


# Drop các cột có quá nhiều missing values
columns_to_drop = [ "SOUF_RCV_NO", "QTUF_RCV_NO"]

test = test.drop(columns=columns_to_drop, errors='ignore')


# Drop các cột chỉ chứa khoảng trống
test = test.drop(columns=["PRODUCT_ASSORT", "REASON_CD"], errors='ignore')


# Thay thế các khoảng trắng ở thuộc tính OTHER AREA SHIP DIV thành 0
test["OTHER AREA SHIP DIV"] = test["OTHER AREA SHIP DIV"].astype(str).str.strip().replace("", "0").fillna("0")


# Drop các cột chỉ có một giá trị
columns_to_drop = ["SUBSIDIARY_CD", "HAZARD_FLG", "Order_Year"]

test = test.drop(columns=columns_to_drop, errors='ignore')


columns_to_drop = ['Stock class', 'ALLOCATION QTY', 'SUPPLIER INV AMOUNT'] 

test = test.drop(columns=columns_to_drop, errors='ignore')


# Chuyển đổi cột có thể thành số 
numeric_columns = ['Consider count hodiday Saturday', 'OTHER AREA SHIP DIV']
for column in numeric_columns:
    test[column] = pd.to_numeric(test[column], errors='coerce') 


# Chuyển đổi cột có thể thành category 
category_columns = ['PACKING RANK', 'DELI_DIV', 'Ship Mode']
for column in category_columns:
    test[column] = test[column].astype('category')


category_features = test.select_dtypes(include=['object', 'category']).columns

# Chỉ chọn những cột có số lượng giá trị duy nhất nhỏ hơn 100
selected_categories = [col for col in category_features if test[col].nunique() < 100]
test = pd.get_dummies(test, columns=selected_categories)

# Nếu số lượng giá trị duy nhất lớn, dùng Label Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in category_features:
    if test[col].nunique() >= 100:
        test[col] = le.fit_transform(test[col])


test.columns = test.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True)


columns_to_add = ['ship_mode_w', 'ship_mode_o', 'ship_mode_f', 'deli_div_0l'] 

test[columns_to_add] = False


test = test[sorted(test.columns)]
data = data[sorted(data.columns)]


test = test.rename(columns={"deli_div_0": "deli_div_00"})


X_train = data.drop(columns = ['label'])
y_train = data['label']

X_test = test.drop(columns = ['id'])


import numpy as np 
import pandas as pd 
from xgboost import XGBClassifier
from catboost import CatBoostClassifier 
from lightgbm import LGBMClassifier

xgb = XGBClassifier() 
cat = CatBoostClassifier() 
lgb = LGBMClassifier()

xgb.fit(X_train, y_train) 
cat.fit(X_train, y_train) 
lgb.fit(X_train, y_train)

xgb_pred = xgb.predict_proba(X_test)[:, 1] 
cat_pred = cat.predict_proba(X_test)[:, 1] 
lgb_pred = lgb.predict_proba(X_test)[:, 1]

avg_pred = (xgb_pred + cat_pred + lgb_pred) / 3
y_pred = (avg_pred > 0.5).astype(int)



pred = pd.DataFrame(y_pred)
pred.columns = ["label"]
pred.index = range(1, len(pred) + 1)
pred.index.name = "ID"


pred.to_csv('23520078.csv')

