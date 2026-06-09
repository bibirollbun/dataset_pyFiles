!pip install numpy
!pip install pandas
!pip install matplotlib
!pip install randomforest
!pip install seaborn
!pip install mplcyberpunk calplot
!pip install scikit-learn --upgrade
!pip install deep-translator
!pip install holidays


import numpy as np
import pandas as pd
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import seaborn as sns
from calplot import calplot as clp
import mplcyberpunk
plt.style.use("cyberpunk")

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from deep_translator import GoogleTranslator

import gc
import requests
import holidays

pd.set_option('display.float_format', lambda x: '%.4f' % x)


# Loading Dataset


df_sales = pd.read_csv(r"/kaggle/input/ml-zoomcamp-2024-competition/sales.csv", index_col=0, parse_dates=["date"])
df_online = pd.read_csv(r"/kaggle/input/ml-zoomcamp-2024-competition/online.csv", index_col=0, parse_dates=["date"])
df_markdowns = pd.read_csv(r"/kaggle/input/ml-zoomcamp-2024-competition/markdowns.csv", index_col=0, parse_dates=["date"])
df_price_history = pd.read_csv(r"/kaggle/input/ml-zoomcamp-2024-competition/price_history.csv", index_col=0, parse_dates=["date"])
df_discounts_history = pd.read_csv(r"/kaggle/input/ml-zoomcamp-2024-competition/discounts_history.csv", index_col=0, parse_dates=["date"])
df_actual_matrix = pd.read_csv(r"/kaggle/input/ml-zoomcamp-2024-competition/actual_matrix.csv", index_col=0, parse_dates=["date"])
df_catalog = pd.read_csv(r"/kaggle/input/ml-zoomcamp-2024-competition/catalog.csv", index_col=0)
df_stores = pd.read_csv(r"/kaggle/input/ml-zoomcamp-2024-competition/stores.csv", index_col=0)
df_test = pd.read_csv(r"/kaggle/input/ml-zoomcamp-2024-competition/test.csv", sep=";", index_col="row_id", parse_dates=["date"], dayfirst = True)
df_sample_submission = pd.read_csv("/kaggle/input/ml-zoomcamp-2024-competition/sample_submission.csv", index_col=0)


df_sales.head()


df_sales.describe(include="all")


df_sales.isna().sum()


df_sales.duplicated()


df_sales.drop_duplicates()


df_sales[(df_sales['quantity'] <= 0) |
         (df_sales['price_base'] <= 0) |
         (df_sales['sum_total'] <= 0)]


mask = (df_sales['quantity'] <= 0) | (df_sales['price_base'] <= 0) | (df_sales['sum_total'] <= 0)
df_sales.drop(df_sales[mask].index, axis=0, inplace=True)


df_online.describe(include="all")


mask_online = (df_online['price_base'] <= 0) | (df_online['sum_total'] <= 0)
df_online.drop(df_online[mask_online].index, axis=0, inplace=True)
df_online[(df_online['price_base'] <= 0) |
         (df_online['sum_total'] <= 0)]


df_online.isna().sum()


df_actual_matrix.isna().sum()


df_actual_matrix['is_available'] = True
df_sales_merged_actual_matrix = pd.merge(df_sales, df_actual_matrix, on=['date', 'item_id', "store_id"], how='left')
df_sales_merged_actual_matrix.isna().sum()


df_sales_merged_actual_matrix.fillna(False, inplace=True)
df_sales_merged_actual_matrix.isna().sum()


df_online = df_online.rename(columns={"price_base":"price_base_online", "sum_total":"sum_total_online"})
df_online["online"] = True

df = pd.merge(df_sales_merged_actual_matrix, df_online, on=['date', 'item_id', "store_id"], how='outer', suffixes=('_x', '_y'))
df["quantity"] = df[['quantity_x', 'quantity_y']].sum(axis=1)

df.isna().sum()


df['online'].fillna(False, inplace = True)
df['is_available'].fillna(True, inplace = True)
df.isna().sum()


df.fillna(0, inplace = True)
df.isna().sum()


df["sum_total_both"] = df[['sum_total', 'sum_total_online']].sum(axis=1)
df["price_base_both"] = df["sum_total_both"] / df["quantity"]
df[(df['online'] == True)].head()


df = df[['date', 'item_id', 'store_id', 'is_available', 'online', 'quantity', 'sum_total_both', 'price_base_both']]
df.describe(include = "all")


df_markdowns.describe(include="all")


mask = (df_markdowns.normal_price <= df_markdowns.price)
df_markdowns[mask]


df_markdowns.drop(df_markdowns[mask].index, axis=0, inplace=True)


df_markdowns.isna().sum()


df = df.merge(df_stores, how='left', left_on=["store_id"], right_on=["store_id"])
df.drop(["format", 'area'], axis=1, inplace = True)
df.head()


df_discounts_history.describe(include="all")


mask = (df_discounts_history.sale_price_before_promo <= df_discounts_history.sale_price_time_promo) | (df_discounts_history.sale_price_before_promo <= 0) | (df_discounts_history.sale_price_time_promo < 0) | (df_discounts_history.date > df_test.date.max())
df_discounts_history[mask]


df_discounts_history.drop(df_discounts_history[mask].index, axis=0, inplace=True)


df_discounts_history.isna().sum()


mask = ((df_discounts_history.date >= df_test.date.min()) & (df_discounts_history.date <= df_test.date.max()))
df_discounts_history[mask]


df_discounts_history["discount_percentage"] = (df_discounts_history["sale_price_before_promo"] - df_discounts_history["sale_price_time_promo"]) / df_discounts_history["sale_price_before_promo"]
df_discounts_history[mask]


mask = (df_price_history.price <= 0)
df_price_history.drop(df_price_history[mask].index, axis=0, inplace=True)





df_price_history = pd.read_csv(r"/kaggle/input/ml-zoomcamp-2024-competition/price_history.csv", index_col=0, parse_dates=["date"])


df_price_history.duplicated(subset=['item_id', 'store_id', 'date']).sum()


df_price_history = df_price_history.drop_duplicates(subset=['item_id', 'store_id', 'date'], keep='last')


df['price'] = pd.NA



# Lọc giá trị tối đa date <= row['date'] và gộp lại với df_price_history
price_history_filtered = df_price_history[df_price_history['date'] <= df['date'].iloc[0]]
latest_prices = price_history_filtered.groupby(['item_id', 'store_id'])['date'].max().reset_index()
latest_prices = pd.merge(latest_prices, df_price_history, on=['item_id', 'store_id', 'date'], how='left')

# Lọc giá trị tối thiểu date > row['date'] và gộp lại với df_]price_history
price_history_filtered_next = df_price_history[df_price_history['date'] > df['date'].iloc[0]]
earliest_prices = price_history_filtered_next.groupby(['item_id', 'store_id'])['date'].min().reset_index()
earliest_prices = pd.merge(earliest_prices, df_price_history, on=['item_id', 'store_id', 'date'], how='left')

# Merge kết quả để có cột giá trị price cho mỗi item_id tại store_id
df = pd.merge(df, latest_prices[['item_id', 'store_id', 'price']], on=['item_id', 'store_id'], how='left', suffixes=('', '_latest'))
df = pd.merge(df, earliest_prices[['item_id', 'store_id', 'price']], on=['item_id', 'store_id'], how='left', suffixes=('', '_earliest'))
# # Sử dụng điều kiện để chọn giá trị cuối cùng (price_latest nếu không có giá trị, chọn price_earliest)
# df['price'] = df['price_latest'].fillna(df['price_earliest'])

# # Xóa các cột tạm thời
# df.drop(['price_latest', 'price_earliest'], axis=1, inplace=True)



print(df.columns)


# Sử dụng điều kiện để chọn giá trị cuối cùng (price_latest nếu không có giá trị, chọn price_earliest)
df['price'] = df['price_latest'].fillna(df['price_earliest'])

# Xóa các cột tạm thời
df.drop(['price_latest', 'price_earliest'], axis=1, inplace=True)


df.isna().sum()


df['price'] = df['price'].fillna(df['price_base_both'])


df.head()


df.loc[df['price'] < df['price_base_both'], 'price'] = df['price_base_both']
df.head()


columns = df.columns.to_list()
df = df.merge(
    df_discounts_history,
    left_on=['item_id', 'store_id', 'date'],
    right_on=['item_id', 'store_id', 'date'],
    how='left',
    suffixes=('', '_price')
)

df = df[columns + ['discount_percentage', 'number_disc_day']]
df.head()


df = df.fillna(0)


columns = df_test.columns.to_list()
df_test = df_test.merge(
    df_discounts_history,
    left_on=['item_id', 'store_id', 'date'],
    right_on=['item_id', 'store_id', 'date'],
    how='left',
    suffixes=('', '_price')
)

df_test = df_test.merge(df_stores, how='left', left_on=["store_id"], right_on=["store_id"])



df_test = df_test[columns+['discount_percentage', 'number_disc_day', 'division', 'city']]
df_test.head()


df_test.isna().sum()


df_test = df_test.fillna(0)


#!pip install deep-translator


#df_catalog = pd.read_csv("/kaggle/input/ml-zoomcamp-2024-competition/catalog.csv", index_col=0)

#dept_name = df_catalog.dept_name.unique()
#for name in dept_name:
    #df_catalog.loc[df_catalog['dept_name'] == name, 'dept_name'] = GoogleTranslator(source='ru', target='en').translate(name).lower().replace(' ', '_')

#class_name = df_catalog.class_name.unique()
#for name in class_name:
    #df_catalog.loc[df_catalog['class_name'] == name, 'class_name'] = GoogleTranslator(source='ru', target='en').translate(name).lower().replace(' ', '_')

#subclass_name = df_catalog.subclass_name.unique()
#for name in subclass_name:
    #df_catalog.loc[df_catalog['subclass_name'] == name, 'subclass_name'] = GoogleTranslator(source='ru', target='en').translate(name).lower().replace(' ', '_')

#df_catalog.item_type = df_catalog.item_type.fillna("other")

#item_type = df_catalog.item_type.unique()
#for name in item_type:
    #df_catalog.loc[df_catalog['item_type'] == name, 'item_type'] = GoogleTranslator(source='ru', target='en').translate(name).lower().replace(' ', '_')

#df_catalog.to_csv("/kaggle/working/translated_catalog.csv", index=True)


df_catalog.head()


df = df.merge(df_catalog, how='left', left_on=["item_id"], right_on=["item_id"])
df.head()


df_test = df_test.merge(df_catalog, how='left', left_on=["item_id"], right_on=["item_id"])
df_test.head()


def get_colums_with_nan(df):
    return df.columns[df.isna().sum() > 0]
    
cols = get_colums_with_nan(df)

df[cols].isna().sum()*100/len(df)


df_test[cols].isna().sum()*100/len(df_test)


df = df.drop(["fatness"], axis=1)
df_test = df_test.drop(["fatness"], axis=1)


def fill_catalog(dataframe, item_name="other"):
    dataframe.dept_name = dataframe.dept_name.fillna("other")
    dataframe.class_name = dataframe.class_name.fillna("other")
    dataframe.subclass_name = dataframe.subclass_name.fillna("other")
    dataframe.item_type = dataframe.item_type.fillna("other")
    return dataframe

df = fill_catalog(df, item_name="other")
df_test = fill_catalog(df_test, item_name="other")


cols = get_colums_with_nan(df)


def weight_fill_nan(dataframe):
    dataframe.weight_volume = dataframe.groupby(by=["item_id", "dept_name", "class_name", "subclass_name", "item_type"]).weight_volume.transform(lambda x: x.fillna(x.mean()))
    dataframe.weight_netto = dataframe.groupby(by=["item_id", "dept_name", "class_name", "subclass_name", "item_type"]).weight_netto.transform(lambda x: x.fillna(x.mean()))
    
    dataframe.weight_volume = dataframe.groupby(by=["dept_name", "class_name", "subclass_name", "item_type"]).weight_volume.transform(lambda x: x.fillna(x.mean()))
    dataframe.weight_netto = dataframe.groupby(by=["dept_name", "class_name", "subclass_name", "item_type"]).weight_netto.transform(lambda x: x.fillna(x.mean()))
    
    dataframe.weight_volume = dataframe.groupby(by=["dept_name", "class_name", "subclass_name"]).weight_volume.transform(lambda x: x.fillna(x.mean()))
    dataframe.weight_netto = dataframe.groupby(by=["dept_name", "class_name", "subclass_name"]).weight_netto.transform(lambda x: x.fillna(x.mean()))
    
    dataframe.weight_volume = dataframe.groupby(by=["dept_name", "class_name"]).weight_volume.transform(lambda x: x.fillna(x.mean()))
    dataframe.weight_netto = dataframe.groupby(by=["dept_name", "class_name"]).weight_netto.transform(lambda x: x.fillna(x.mean()))
    
    dataframe.weight_volume = dataframe.groupby(by=["dept_name"]).weight_volume.transform(lambda x: x.fillna(x.mean()))
    dataframe.weight_netto = dataframe.groupby(by=["dept_name"]).weight_netto.transform(lambda x: x.fillna(x.mean()))
    return dataframe

train_index = len(df)
test_index = len(df_test)

all_data = pd.concat([df, df_test], axis=0)

all_data = weight_fill_nan(all_data)
df = all_data.iloc[:train_index]
df_test = all_data.iloc[train_index:test_index+train_index]


df = df.fillna(-1)
df[cols].isna().sum()*100/len(df)


df_test = df_test.fillna(-1)
df_test[cols].isna().sum()*100/len(df_test)


df_price_history.sort_values(by=['store_id', 'item_id', 'date'], inplace=True)
from collections import defaultdict

price_dict = defaultdict(list)
for row in df_price_history.itertuples(index=False):
    key = (row.store_id, row.item_id)
    price_dict[key].append((row.date, row.price))


from bisect import bisect_right

def get_latest_price(key, date):
    entries = price_dict.get(key, [])
    dates = [d for d, _ in entries]
    idx = bisect_right(dates, date) - 1  # tìm vị trí gần nhất trước ngày
    if idx >= 0:
        return entries[idx][1]
    return None  # hoặc giữ nguyên -1 nếu không có giá nào trước đó

# Áp dụng từng dòng
df_test['price'] = [
    get_latest_price((row.store_id, row.item_id), row.date)
    for row in df_test.itertuples(index=False)
]


df_test.isna().sum()


df_test = df_test.fillna(-1)


#...


def date_features(dataframe):
    dataframe["dayofmonth"] = dataframe.date.dt.day
    dataframe["month"] = dataframe.date.dt.month
    dataframe["dayofyear"] = dataframe.date.dt.dayofyear
    dataframe["year"] = dataframe.date.dt.year
    dataframe['dayofweek'] = dataframe['date'].dt.dayofweek 
    dataframe['week'] = dataframe['date'].dt.isocalendar().week
    return dataframe
    
df = date_features(df)
df_test = date_features(df_test)


# Quantity per year
plt.figure(figsize=(3, 3))
aux = df.groupby(["year"]).quantity.mean().reset_index()
aux.year = aux.year.apply(str)
sns.barplot(y=aux.quantity, x=aux.year);


def transform2cyclic(dataframe):
    dataframe['dayofmonth_sin'] = np.sin(2 * np.pi * (dataframe['dayofmonth']-1)/31)
    dataframe['dayofmonth_cos'] = np.cos(2 * np.pi * (dataframe['dayofmonth']-1)/31)

    dataframe['dayofyear_sin'] = np.sin(2 * np.pi * (dataframe['dayofyear']-1)/365)
    dataframe['dayofyear_cos'] = np.cos(2 * np.pi * (dataframe['dayofyear']-1)/365)
    
    dataframe['dayofweek_sin'] = np.sin(2 * np.pi * dataframe['dayofweek']/6)
    dataframe['dayofweek_cos'] = np.cos(2 * np.pi * dataframe['dayofweek']/6)
    
    dataframe['week_sin'] = np.sin(2 * np.pi * (dataframe['week']-1)/52)
    dataframe['week_cos'] = np.cos(2 * np.pi * (dataframe['week']-1)/52)
    
    dataframe['month_sin'] = np.sin(2 * np.pi * (dataframe['month']-1)/12)
    dataframe['month_cos'] = np.cos(2 * np.pi * (dataframe['month']-1)/12)
    return dataframe

df = transform2cyclic(df)
df_test = transform2cyclic(df_test)


cols = ["dayofmonth", "dayofweek", "dayofyear", "week", "month"]
n = len(cols)
rows = (n + 1) // 2  # 2 plots per row

fig, axes = plt.subplots(rows, 2, figsize=(15, 5 * rows))
axes = axes.flatten()  # Flatten to 1D array for easier indexing

for i, col in enumerate(cols):
    aux = df.groupby([col, col+"_sin", col+"_cos"]).quantity.mean().reset_index()
    aux[col] = aux[col].apply(str)
    sns.lineplot(data=aux[[col+"_sin", col+"_cos"]], ax=axes[i])
    axes[i].set_title(f"{col} - sin & cos")

# Hide any unused subplots (if n is odd)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


def get_seasons(dataframe):
    dataframe["season"] = 0
    dataframe.loc[(dataframe.month >= 3) & (dataframe.month <= 5), "season"] = 1
    dataframe.loc[(dataframe.month >= 6) & (dataframe.month <= 8), "season"] = 2
    dataframe.loc[(dataframe.month == 9) & (dataframe.month <= 11), "season"] = 3
    dataframe.loc[((dataframe.month >= 1) & (dataframe.month <= 2)) | (dataframe.month == 12), "season"] = 4
    return dataframe

df = get_seasons(df)
df_test = get_seasons(df_test)


df.columns


def get_holidays(dataframe):
    RU_holidays = holidays.CountryHoliday('RU', years=[2022, 2023, 2024])
    dataframe["holidays"] = False
    dataframe.loc[df.date.isin(RU_holidays.keys()), "holidays"] = True
    return dataframe

df = get_holidays(df)
df_test = get_holidays(df_test)


def get_sundays(dataframe):
    dataframe["is_sunday"] = dataframe['dayofweek'].eq(6)
    return dataframe

df = get_sundays(df)
df_test = get_sundays(df_test)


def get_weekends(dataframe):
    dataframe["is_weekend"] = dataframe['dayofweek'].isin([4, 5, 6])
    return dataframe

df = get_weekends(df)
df_test = get_weekends(df_test)


df.columns


df_test.columns


cols = ['date', 
        'dayofmonth', 
        'dayofyear',
        'dayofweek', 
        'week', 
        'online',
        'is_available',
        'month', 
        'price_base_both', 
        'sum_total_both',
        'price'
       ]
df.drop(columns=cols, inplace=True)
df_test.drop(columns=cols+["quantity"], inplace=True)


df_test.columns


X = df.drop(["quantity"], axis=1)
y = df["quantity"]


numerical_cols = X.select_dtypes([np.int32, np.int64, np.float32, np.float64]).columns.to_list()
categorical_cols = X.select_dtypes('object').columns.to_list()
numerical_cols, categorical_cols


column_transformer = make_column_transformer(
    # Numerical columns
    (
        StandardScaler(),
        numerical_cols
    ),
    # Categorical columns
    (
        OneHotEncoder(handle_unknown='ignore', drop='first'),
        categorical_cols
    ),
    remainder='passthrough',
    verbose_feature_names_out=False
)

X_transformed = column_transformer.fit_transform(X)
X_test_transformed = column_transformer.transform(df_test)


seed_value = 42
X_train, X_val, y_train, y_val = train_test_split(X_transformed, y, train_size=0.8, random_state=seed_value)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)


import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ====================
# 1. Huấn luyện mô hình Random Forest
# ====================
model = RandomForestRegressor(
    n_estimators=2000,         
    max_depth=12,              
    min_samples_leaf=1,        
    max_features='sqrt',       
    random_state=seed_value,
    n_jobs = -1,
    verbose=1
)

model.fit(X_train, y_train)

# Dự đoán trên tập validation
preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, preds))
print(f"✅ Validation RMSE: {rmse:.4f}")

# Lưu model
joblib.dump(model, "best_model_randomforest.pkl")

# ====================
# 2. Load model tốt nhất và dự đoán
# ====================
best_model = joblib.load("best_model_randomforest.pkl")

# Dự đoán trên tập test
quantity_pred = best_model.predict(X_test_transformed)

# ====================
# 3. Tạo submission
# ====================
df_test["quantity"] = quantity_pred
df_submission = df_test[["quantity"]]

# Xem trước kết quả
print(df_submission.head())



#quantity_pred = model.predict(X_test_transformed)
#df_test["quantity"] = quantity_pred
#df_submission = df_test[["quantity"]]
#df_submission.head()


missing_items = df_test[~df_test.item_id.isin(df.item_id)]
missing_items.head()


df_submission.loc[missing_items.index, "quantity"] = 0


df_submission.to_csv("/kaggle/working/submission.csv", index_label='row_id')

