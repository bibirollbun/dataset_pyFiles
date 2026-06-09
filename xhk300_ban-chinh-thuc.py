import zipfile
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ctypes
import gc
from tqdm import tqdm
import pickle
from scipy import stats
import collections
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Định nghĩa đường dẫn
input_dir = "/kaggle/input/santander-product-recommendation/"
output_dir = "/kaggle/working/"

# Danh sách file cần giải nén
zip_files = [
    "train_ver2.csv.zip",
    "test_ver2.csv.zip",
    "sample_submission.csv.zip"
]

# Giải nén từng file
for zip_file in zip_files:
    zip_path = os.path.join(input_dir, zip_file)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)


df = pd.read_csv(os.path.join(output_dir, "train_ver2.csv"), low_memory=False)


df_test = pd.read_csv(os.path.join(output_dir, "test_ver2.csv"), low_memory=False)


def reduce_memory_usage(df, verbose=True):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


# Trước khi giảm
before_df = df.memory_usage().sum() / 1024 ** 2  # Đổi sang MB
before_df_test = df_test.memory_usage().sum() / 1024 ** 2

# Giảm bộ nhớ
df = reduce_memory_usage(df)
df_test = reduce_memory_usage(df_test)

# Sau khi giảm
after_df = df.memory_usage().sum() / 1024 ** 2  
after_df_test = df_test.memory_usage().sum() / 1024 ** 2


print("So sanh truoc va sau khi giam")
print("Tap train: ", before_df, " - ",after_df)
print("Tap test: ", before_df_test, " - ",after_df_test)


df["fecha_dato"] = pd.to_datetime(df["fecha_dato"],format="%Y-%m-%d")
df["fecha_alta"] = pd.to_datetime(df["fecha_alta"],format="%Y-%m-%d")
df["age"]   = pd.to_numeric(df["age"], errors="coerce")


df_test["fecha_dato"] = pd.to_datetime(df_test["fecha_dato"],format="%Y-%m-%d")
df_test["fecha_alta"] = pd.to_datetime(df_test["fecha_alta"],format="%Y-%m-%d")
df_test["age"]   = pd.to_numeric(df_test["age"], errors="coerce")


df.isnull().any()


df["age"].fillna(df["age"].mean(),inplace=True)
df["age"] = df["age"].astype(int)


df_test["age"].fillna(df_test["age"].mean(),inplace=True)
df_test["age"] = df_test["age"].astype(int)


df.loc[df["ind_nuevo"].isnull(),"ind_nuevo"] = 1


df_test.loc[df_test["ind_nuevo"].isnull(),"ind_nuevo"] = 1


dates=df.loc[:,"fecha_alta"].sort_values().reset_index()
median_date = int(np.median(dates.index.values))
df.loc[df.fecha_alta.isnull(),"fecha_alta"] = dates.loc[median_date,"fecha_alta"]
df["fecha_alta"].describe()


dates=df_test.loc[:,"fecha_alta"].sort_values().reset_index()
median_date = int(np.median(dates.index.values))
df_test.loc[df_test.fecha_alta.isnull(),"fecha_alta"] = dates.loc[median_date,"fecha_alta"]


df["antiguedad"] = pd.to_numeric(df["antiguedad"], errors="coerce")
mean_value = df["antiguedad"].mean()
df["antiguedad"].fillna(mean_value, inplace=True)
df.loc[df["antiguedad"] < 0, "antiguedad"] = 0


df_test.antiguedad = pd.to_numeric(df_test.antiguedad,errors="coerce")
df_test["antiguedad"].fillna(mean_value, inplace=True)
df_test.loc[df_test.antiguedad <0, "antiguedad"]      = 0


df.loc[df.indrel.isnull(),"indrel"] = 1


df_test.loc[df_test.indrel.isnull(),"indrel"] = 1


df.loc[df.ind_actividad_cliente.isnull(),"ind_actividad_cliente"] = df["ind_actividad_cliente"].median()


df_test.loc[df_test.ind_actividad_cliente.isnull(),"ind_actividad_cliente"] = df_test["ind_actividad_cliente"].median()


df_test.loc[df.nomprov=="CORU\xc3\x91A, A","nomprov"] = "CORUNA, A"
df_test.loc[df.nomprov.isnull(),"nomprov"] = "UNKNOWN"


df_test.loc[df_test.nomprov=="CORU\xc3\x91A, A","nomprov"] = "CORUNA, A"
df_test.loc[df_test.nomprov.isnull(),"nomprov"] = "UNKNOWN"


# Gộp cột 'nomprov' và 'renta' của df và df_test
combined_df = pd.concat([df[['nomprov', 'renta']], df_test[['nomprov', 'renta']]])

# Tạo bảng median_map từ dữ liệu gộp
median_map = combined_df.groupby('nomprov')['renta'].median().to_dict()

# Điền giá trị null từ median theo 'nomprov'
df['renta'] = df.apply(lambda row: median_map.get(row['nomprov'], row['renta']) if pd.isna(row['renta']) else row['renta'], axis=1)

# Điền tiếp các giá trị null còn lại bằng median toàn cột
df['renta'].fillna(combined_df['renta'].median(), inplace=True)


df_test['renta'] = df_test.apply(lambda row: median_map.get(row['nomprov'], row['renta']) if pd.isna(row['renta']) else row['renta'], axis=1)
df_test['renta'].fillna(combined_df['renta'].median(), inplace=True)


df.loc[df.ind_nomina_ult1.isnull(), "ind_nomina_ult1"] = 0
df.loc[df.ind_nom_pens_ult1.isnull(), "ind_nom_pens_ult1"] = 0


df_test.loc[df_test.ind_nomina_ult1.isnull(), "ind_nomina_ult1"] = 0
df_test.loc[df_test.ind_nom_pens_ult1.isnull(), "ind_nom_pens_ult1"] = 0


string_data = df.select_dtypes(include=["object"])
missing_columns = [col for col in string_data if string_data[col].isnull().any()]
for col in missing_columns:
    print("Unique values for {0}:\n{1}\n".format(col,string_data[col].unique()))
del string_data


# Xử lý cột 'indfall'
df.loc[df.indfall.isnull(), 'indfall'] = 'N'
df_test.loc[df_test.indfall.isnull(), 'indfall'] = 'N'

# Xử lý cột 'tiprel_1mes'
df.loc[df.tiprel_1mes.isnull(), 'tiprel_1mes'] = 'A'
df_test.loc[df_test.tiprel_1mes.isnull(), 'tiprel_1mes'] = 'A'
df.tiprel_1mes = df.tiprel_1mes.astype("category")
df_test.tiprel_1mes = df_test.tiprel_1mes.astype("category")

map_dict = { 1.0  : "1",
            "1.0" : "1",
            "1"   : "1",
            "3.0" : "3",
            "P"   : "P",
            3.0   : "3",
            2.0   : "2",
            "3"   : "3",
            "2.0" : "2",
            "4.0" : "4",
            "4"   : "4",
            "2"   : "2"}

df.indrel_1mes.fillna("P",inplace=True)
df.indrel_1mes = df.indrel_1mes.apply(lambda x: map_dict.get(x,x))
df.indrel_1mes = df.indrel_1mes.astype("category")

df_test.indrel_1mes.fillna("P",inplace=True)
df_test.indrel_1mes = df_test.indrel_1mes.apply(lambda x: map_dict.get(x,x))
df_test.indrel_1mes = df_test.indrel_1mes.astype("category")


unknown_cols = [col for col in missing_columns if col not in ["indfall","tiprel_1mes","indrel_1mes"]]
for col in unknown_cols:
    df.loc[df[col].isnull(),col] = "UNKNOWN"
    df_test.loc[df_test[col].isnull(),col] = "UNKNOWN"


df.isnull().any()


df_test.isnull().any()


df.loc[df.age < 18,"age"]  = df.loc[(df.age >= 18) & (df.age <= 30),"age"].mean(skipna=True)
df.loc[df.age > 100,"age"] = df.loc[(df.age >= 30) & (df.age <= 100),"age"].mean(skipna=True)


min_value = 0.
max_value = 256.
df["antiguedad"] = df["antiguedad"].clip(lower=min_value, upper=max_value)


min_value = 0.
max_value = 1500000.
df["renta"] = df["renta"].clip(lower=min_value, upper=max_value)


def scaleAge(df):
    min_age = 18.
    max_age = 100.
    # Chuẩn hóa cột 'age' và làm tròn đến 4 chữ số thập phân
    df['age'] = round((df['age'] - min_age) / (max_age - min_age), 4)
    return df

# Gọi hàm
df = scaleAge(df)


def scaleAnti(df):
    min = 0.
    max = 100.
    # Chuẩn hóa cột 'age' và làm tròn đến 4 chữ số thập phân
    df['antiguedad'] = round((df['antiguedad'] - min) / (max - min), 4)
    return df

# Gọi hàm
df = scaleAnti(df)


def scaleRent(df):
    min = 0.
    max = 1500000.
    # Chuẩn hóa cột 'age' và làm tròn đến 4 chữ số thập phân
    df['renta'] = round((df['renta'] - min) / (max - min), 4)
    return df

# Gọi hàm
df = scaleRent(df)


from sklearn.preprocessing import LabelEncoder

cols = ['ind_empleado','pais_residencia', 'sexo','ind_nuevo','indrel_1mes','tiprel_1mes', 'indresi', 
        'indext', 'conyuemp', 'canal_entrada','indfall', 'nomprov','segmento','indrel','ind_actividad_cliente']

df[cols] = df[cols].astype(str).apply(LabelEncoder().fit_transform)


for col in df.columns:
    print(f"Column: {col}")
    print(df[col].unique())
    print("-" * 50)


df.drop(columns=['tipodom'],inplace = True)
df.drop(columns=['cod_prov'],inplace = True)


df.drop(columns=['ind_ahor_fin_ult1', 'ind_aval_fin_ult1'],inplace = True)


target_cols = df.columns.values[22:]
target_cols


df_days_column = (df['fecha_dato'] - df['fecha_alta']).dt.days
df.insert(loc=6, column='days', value=df_days_column)


customers42015 = df[df['fecha_dato'] == '2015-04-28']['ncodpers'].to_numpy()
customers52015 = df[df['fecha_dato'] == '2015-05-28']['ncodpers'].to_numpy()
inf_prod_52015 = df[df['fecha_dato'] == '2015-05-28'].set_index('ncodpers')[target_cols]

# Thông tin mua sản phẩm của những khách hàng vào tháng 6-2015 đã từng mua sản phẩm vào tháng 5-2015
inf_prod_52015_old = df[(df['fecha_dato'] == '2015-06-28') & (df['ncodpers'].isin(customers52015))].set_index(['ncodpers'])[target_cols]
new_prod = inf_prod_52015 - inf_prod_52015_old
q = (new_prod[target_cols] == -1).sum(1)
june_custs = q[q > 0].index

june_new_customers = df[(df['fecha_dato'] == '2015-06-28') & (~df['ncodpers'].isin(customers52015))]
df_sept = df[(df['fecha_dato'] == '2015-06-28') & (df['ncodpers'].isin(june_custs))]


print(df_sept.shape)


# Lọc các dòng có fecha_dato = '2015-05-28' và ncodpers có trong customers52015
df_52015_own = df[(df['fecha_dato'] == '2015-05-28') & (df['ncodpers'].isin(customers52015))]

# Chọn các cột cần thiết
df_52015_own = df_52015_own[['ncodpers', 'ind_actividad_cliente', 'tiprel_1mes'] + list(target_cols)]

# Đổi tên các cột thành feature lag
df_52015_own.rename(columns={'ind_actividad_cliente': 'ind_actividad_cliente_last', 
                             'tiprel_1mes': 'tiprel_1mes_last'}, inplace=True)




# Tạo feature số sản phẩm mà khách hàng đã sở hữu vào tháng 5-2015
df_52015_own['n_product_last'] = df_52015_own[target_cols].sum(axis=1)
df_52015_own.drop(columns=target_cols, inplace=True)


df_sept = df_sept.merge(
    df_52015_own[['ncodpers', 'ind_actividad_cliente_last', 'tiprel_1mes_last', 'n_product_last']],
    on='ncodpers',
    how='left'
)


import warnings
warnings.filterwarnings('ignore')

train_total = pd.DataFrame()
t=0
for i in tqdm(target_cols):
    train = df_sept[df_sept['ncodpers'].isin(new_prod[new_prod[i] == -1].index)]
    train2 = june_new_customers[june_new_customers[i] == 1]

    train2 = train2.assign(
        ind_actividad_cliente_last=0,
        tiprel_1mes_last=0,
        n_product_last=0
    )
    
    train.drop(columns=target_cols,inplace = True)
    train2.drop(columns=target_cols,inplace = True)
    train['target'] = t
    train2['target'] = t

    train = pd.concat([train, train2], ignore_index=True, sort=False)
    train_total = pd.concat([train_total, train], ignore_index=True, sort=False)
    t+=1

del train
del train2
warnings.filterwarnings('default')


print(train_total.shape)


#Creating the final train dataset
X_train = df_total[~df_total['target'].isnull()]
y_train = X_train['target'].astype(int)
X_train.drop(columns=['target'],inplace = True)

train_cust_ids = X_train['ncodpers']
X_train.drop(columns=['fecha_dato','ncodpers'],inplace = True)

X_train = X_train.values.tolist()

del df
del train_total
del df_test
del df_group

k=0
for i in tqdm(train_cust_ids):
    l = train_lags.get(i,[[0]*22])

    try:
        lag_1 = list(l[-1])
    except:
        lag_1 = [0]*22
    try:
        lag_2 = list(l[-2])
    except:
        lag_2 = [0]*22
    try:
        lag_3 = list(l[-3])
    except:
        lag_3 = [0]*22
    try:
        lag_4 = list(l[-4])
    except:
        lag_4 = [0]*22
    try:
        lag_5 = list(l[-5])
    except:
        lag_5 = [0]*22
    
    X_train[k].extend(lag_1+lag_5+lag_4+lag_3+lag_2)
    k+=1  

X_train = np.array(X_train)



from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from xgboost.sklearn import XGBClassifier

params = {'learning_rate':[0.01,0.03,0.1,0.2],
          'max_depth':[3, 5, 8],
          'n_estimators':[10,50],
          'colsample_bytree':[0.5,0.6,0.7,0.8,0.9,1],
          'subsample':[0.5,0.6,0.7,0.8,0.9,1],
          'min_child_weight': `[1,3,5,7,10,14]}

clf = RandomizedSearchCV(XGBClassifier(objective = 'multi:softprob',eval_metric = 'mlogloss'), params, cv=3,scoring='roc_auc_ovo',return_train_score=True, n_jobs = -1,verbose = 10,error_score="raise",n_iter = 15)
clf.fit(X_train, y_train)


clf.best_params_


from xgboost.sklearn import XGBClassifier
xgb = XGBClassifier(objective = 'multi:softprob',eval_metric = 'mlogloss',max_depth=5,n_estimators=50,learning_rate=0.1,colsample_bytree=0.8,subsample=0.9,min_child_weight = 1)
xgb.fit(X_train, y_train)


y_pred = xgb.predict_proba(X_test)
test_users = {}
for i in tqdm(df[(df['fecha_dato'] == '2016-05-28') & (df['ncodpers'].isin(df_test['ncodpers'].to_numpy()))].iloc):
    test_users[i['ncodpers']] = np.where(i['ind_cco_fin_ult1':'ind_recibo_ult1'] == 0)[0]


test_user_ratings = {}
k=0
for i in tqdm(test_cust_ids):
    test_user_ratings[i] = y_pred[k][test_users[i]]
    k+=1

final_preds = []
test_ids = []
for i in tqdm(test_user_ratings.keys()):
    top_seven = test_users[i][np.argsort(test_user_ratings[i])[::-1][:7]]
    final_preds.append(" ".join(target_cols[top_seven]))
    test_ids.append(i)


xgb_submission = pd.DataFrame({'ncodpers':test_ids, 'added_products':final_preds})
xgb_submission = xgb_submission.sort_values('ncodpers')
xgb_submission.to_csv('drive/MyDrive/Santander product recommendation/submit67.csv', index=False)

