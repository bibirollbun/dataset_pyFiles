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


delay_4_6 = pd.read_csv("/kaggle/input/ds-108-p-21-assigment-06/delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv")
delay_7_9 = pd.read_csv("/kaggle/input/ds-108-p-21-assigment-06/delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv")
not_delay_4_6 = pd.read_csv("/kaggle/input/ds-108-p-21-assigment-06/not_delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv", low_memory=False)
not_delay_7_9 = pd.read_csv("/kaggle/input/ds-108-p-21-assigment-06/not_delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv", low_memory=False)
PILOT_10 = pd.read_csv("/kaggle/input/ds-108-p-21-assigment-06/PILOT_10.csv")


from sklearn.model_selection import train_test_split

df = pd.concat([delay_4_6, not_delay_4_6, delay_7_9, not_delay_7_9], ignore_index=True)
test_df = pd.concat([PILOT_10], ignore_index=True)

# Đảm bảo cột 'label' của df sạch NaN 
df['label'] = df['label'].astype(str).str.strip()
df['label'] = df['label'].replace('', pd.NA)
df.dropna(subset=['label'], inplace=True)


# Chia train và val
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Tách X và y từ train_df
X_train = train_df.drop(columns=['label']).copy()
y_train = train_df['label'].copy()

# Tách X và y_true từ val_df (tập validation)
X_val = val_df.drop(columns=['label']).copy()
y_val_true = val_df['label'].copy()

# Gán test_df vào biến X_test 
X_test = test_df.copy()


# Danh sách các feature bị loại bỏ từ báo cáo (do tương quan thấp, ảnh hưởng thấp, hoặc target leak)
features_to_drop = [
    'SUBSIDIARY_CD', 'GLOBAL_NO', 'BRAND_CD', 'Sales order line number',
    'Stock class', 'ALLOCATION QTY', 'SPECIAL DIV', 'LOGICAL PLANT',
    'PURCHASE AMOUNT', 'DIRECT SHIP FLG', 'SHIP DECISION NO', 'PACK QTY',
    'SUPPLIER_DIV', 'SPECIAL_DIV', 'SO_DAY_OF_MONTH', 'SO_DAY_OF_WEEK',
    'SO_TIME', 'QTUF_RCV_NO', 'SOUF_RCV_NO','REASON_CD' 
]

# Loại bỏ các feature đã xác định khỏi X_train, X_val và X_test
X_train_processed = X_train.drop(columns=[col for col in features_to_drop if col in X_train.columns], errors='ignore')
X_val_processed = X_val.drop(columns=[col for col in features_to_drop if col in X_val.columns], errors='ignore')
X_test_processed = X_test.drop(columns=[col for col in features_to_drop if col in X_test.columns], errors='ignore') # Đã đổi final_test_df thành X_test

# Đảm bảo X_train, X_val và X_test có cùng tập hợp cột và thứ tự sau khi loại bỏ.
final_model_cols = list(set(X_train_processed.columns) & set(X_val_processed.columns) & set(X_test_processed.columns))

X_train_processed = X_train_processed[final_model_cols].copy()
X_val_processed = X_val_processed[final_model_cols].copy()
X_test_processed = X_test_processed[final_model_cols].copy() # Đã đổi final_test_df thành X_test

# Sắp xếp lại cột để đảm bảo thứ tự giống nhau cho LightGBM
X_train_processed = X_train_processed.reindex(columns=sorted(X_train_processed.columns))
X_val_processed = X_val_processed.reindex(columns=sorted(X_val_processed.columns))
X_test_processed = X_test_processed.reindex(columns=sorted(X_test_processed.columns)) # Đã đổi final_test_df thành X_test


# Xử lý giá trị thiếu (NaN)
numerical_cols = X_train_processed.select_dtypes(include=['number']).columns
for col in numerical_cols:
    X_train_processed[col] = pd.to_numeric(X_train_processed[col], errors='coerce')
    X_val_processed[col] = pd.to_numeric(X_val_processed[col], errors='coerce')
    X_test_processed[col] = pd.to_numeric(X_test_processed[col], errors='coerce') # Áp dụng cho X_test

    median_val = X_train_processed[col].median() # Tính từ X_train_processed
    X_train_processed[col].fillna(median_val, inplace=True)
    X_val_processed[col].fillna(median_val, inplace=True)
    X_test_processed[col].fillna(median_val, inplace=True) # Áp dụng cho X_test

# Lấy danh sách các cột object sau khi điền NaN (để chuyển thành category)
categorical_features_for_lgbm = X_train_processed.select_dtypes(include='object').columns.tolist()

for col in categorical_features_for_lgbm:
    mode_val = X_train_processed[col].mode()[0] # Tính từ X_train_processed
    X_train_processed[col].fillna(mode_val, inplace=True)
    X_val_processed[col].fillna(mode_val, inplace=True)
    X_test_processed[col].fillna(mode_val, inplace=True) # Áp dụng cho X_test

# Chuyển đổi sang Category
for col in categorical_features_for_lgbm:
    # Đảm bảo các cấp độ (categories) chỉ được xác định từ X_train_processed
    all_categories = X_train_processed[col].astype(str).unique()
    
    # Áp dụng các cấp độ này cho tất cả các tập
    X_train_processed[col] = pd.Categorical(X_train_processed[col].astype(str), categories=all_categories)
    X_val_processed[col] = pd.Categorical(X_val_processed[col].astype(str), categories=all_categories)
    X_test_processed[col] = pd.Categorical(X_test_processed[col].astype(str), categories=all_categories)


print(X_val_processed.info())


print(X_test_processed.info())


from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

model = LGBMClassifier(
    objective='binary',
    metric='logloss',
    random_state=42,
    n_jobs=-1,
    categorical_feature=categorical_features_for_lgbm
)

model.fit(X_train_processed, y_train)


# Predict & Evaluate trên tập validation (X_val_processed)
y_val_pred = model.predict(X_val_processed)
f1_val = f1_score(y_val_true, y_val_pred, average='macro')
print(f"Macro F1-score on validation set: {f1_val:.4f}")


# Dự đoán trên tập test cuối cùng (X_test_processed)
y_test_pred = model.predict(X_test_processed)


submission_df = pd.DataFrame({
    'ID': test_df['ID'],
    'label': y_test_pred
})

# Lưu thành file CSV
submission_df.to_csv('submission.csv', index=False)


from collections import Counter

print(Counter(y_test_pred))

