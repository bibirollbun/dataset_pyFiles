import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import gc


!unzip /kaggle/input/santander-product-recommendation/train_ver2.csv.zip
!unzip /kaggle/input/santander-product-recommendation/test_ver2.csv.zip


def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce memory usage of a DataFrame by downcasting numeric columns
    and converting object columns to categorical when appropriate.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Initial memory usage: {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type).startswith('int'):
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                df[col] = df[col].astype(np.float32)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Final memory usage: {end_mem:.2f} MB")
    print(f"Memory reduced by {(start_mem - end_mem) / start_mem * 100:.1f}%")
    return df


train_data = pd.read_csv('/kaggle/working/train_ver2.csv', nrows=500000)
test_data = pd.read_csv('/kaggle/working/test_ver2.csv') 


drop_cols = ['fecha_alta', 'ult_fec_cli_1t', 'tipodom', 'cod_prov', 'conyuemp', 'fecha_dato']
train_data.drop(columns=drop_cols, errors='ignore', inplace=True)
test_data.drop(columns=drop_cols, errors='ignore', inplace=True)


train_data.head()


train_data.info()


test_data.head()


test_data.info()


numeric_cols = train_data.select_dtypes(include=['number']).columns
train_data[numeric_cols] = train_data[numeric_cols].fillna(-1)


cat_cols = train_data.select_dtypes(include=['object', 'category']).columns
for col in cat_cols:
    combined = pd.concat([train_data[col], test_data[col]], axis=0).astype('category')
    train_data[col] = combined[:len(train_data)].cat.codes
    test_data[col] = combined[len(train_data):].cat.codes


train_data = reduce_memory_usage(train_data)


y = train_data.iloc[:, -1]
X = train_data.drop(train_data.columns[-1], axis=1)


target_cols = [col for col in train_data.columns if col not in test_data.columns]

print(f"Target number of products: {len(target_cols)}")


y = train_data.iloc[:, -1]
X = train_data.drop(columns=target_cols, axis=1)
test_ids = test_data['ncodpers']
X_test = test_data[X.columns]


gc.collect()


print(f"Education (X) columns: {X.columns.tolist()}")


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestClassifier


model = RandomForestClassifier(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score


y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"Model Accuracy: {acc:.4f}")


expected_columns = X.columns


for col in expected_columns:
    if col in X_test.columns and X_test[col].dtype == 'object':
        print(f"Under repair: {col}")
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(-1)


print("Missing values are filled with -1...")
X_test = X_test.fillna(-1)


if X_test.isnull().values.any():
    print("WARNING: There are still empty values!")
else:
    print("The data is clean, no empty values.")


final_preds = model.predict(X_test)


target_product_name = "ind_recibo_ult1" 
 
product_preds = []
for p in final_preds:
    if p == 1:
        product_preds.append(target_product_name)
    else: 
        product_preds.append("ind_cco_fin_ult1") 
 
submission = pd.DataFrame({
    'ncodpers': test_ids,
    'added_products': product_preds
})
 
submission.to_csv('submission.csv', index=False)
print("The corrected submission.csv has been created.")


submission

