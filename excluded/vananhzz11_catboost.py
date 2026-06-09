!pip install numpy # cai thu vien tinh toan 
!pip istall pandas #du lieu dang bang - doc cac file excel 
!pip install matplotlib #thu vien ve bieu do
!pip install catboost
!pip install seaborn #ve bieu do cao cap dua tren matplotlib
!pip install mplcyberpunk calplot #theme giao dien cho mau sac bat mat
!pip install scikit-learn --upgrade #thư viện ML cổ điển và rất mạnh.
#Hỗ trợ các mô hình như linear regression, decision tree, SVM, clustering, preprocessing dữ liệu, pipeline.
!pip install deep-translator #dich van ban tu dong
!pip install holidays #tạo danh sách ngày lễ của từng quốc gia


import numpy as np
import pandas as pd
import os #dung de tuong tac voi he dieu hanh
import warnings # thu vien chuan dung de xu ly canh cao khi chay code
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import seaborn as sns
from calplot import calplot as clp
import mplcyberpunk
plt.style.use("cyberpunk")

from catboost import CatBoostRegressor

from sklearn.metrics import root_mean_squared_error as RMSE
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from deep_translator import GoogleTranslator

import gc
import requests
import holidays

pd.set_option('display.float_format', lambda x: '%.4f' % x)


# Đọc dữ liệu với parse_dates và index_col
 # parse_dates - tự động chuyển cột "date" từ chuỗi (string) sang định dạng datetime
df_sales = pd.read_csv(r"/kaggle/input/ml-zoomcamp-2024-competition/sales.csv", index_col=0, parse_dates=["date"])
df_online = pd.read_csv(r"/kaggle/input/ml-zoomcamp-2024-competition/online.csv", index_col=0, parse_dates=["date"])
df_markdowns = pd.read_csv(r"/kaggle/input/ml-zoomcamp-2024-competition/markdowns.csv", index_col=0, parse_dates=["date"])
df_price_history = pd.read_csv(r"/kaggle/input/ml-zoomcamp-2024-competition/price_history.csv", index_col=0, parse_dates=["date"])
df_discounts_history = pd.read_csv(r"/kaggle/input/ml-zoomcamp-2024-competition/discounts_history.csv", index_col=0, parse_dates=["date"])
df_actual_matrix = pd.read_csv(r"/kaggle/input/ml-zoomcamp-2024-competition/actual_matrix.csv", index_col=0, parse_dates=["date"])
df_catalog = pd.read_csv(r"/kaggle/input/trans-cat/translated_catalog.csv", index_col=0)
df_stores = pd.read_csv(r"/kaggle/input/ml-zoomcamp-2024-competition/stores.csv", index_col=0) # Sửa: index_col thay vì index_cols
df_test = pd.read_csv(r"/kaggle/input/ml-zoomcamp-2024-competition/test.csv", sep=";", index_col="row_id", parse_dates=["date"], dayfirst = True)
# sep=";" : chi dinh dau phan cach giua cac cot #Với dayfirst=True, nó hiểu định dạng kiểu châu Âu: (DD/MM/YYYY)
df_sample_submission = pd.read_csv("/kaggle/input/ml-zoomcamp-2024-competition/sample_submission.csv", index_col=0)


df_sales.head()


#tóm tắt thống kê toàn bộ các cột
df_sales.describe(include="all")


#Hien thi cac gia tri am - khong hop ly
df_sales[(df_sales['quantity'] <= 0) |
         (df_sales['price_base'] <= 0) |
         (df_sales['sum_total'] <= 0)]


#Xoa cac gia tri am - khong hop ly
mask = (df_sales['quantity'] <= 0) | (df_sales['price_base'] <= 0) | (df_sales['sum_total'] <= 0)
df_sales.drop(df_sales[mask].index, axis=0, inplace=True) #axis=0 chỉ xóa theo hàng (dòng).


# Kiem tra missing value- gia tri thieu
df_sales.isna().sum()


#kiem tra cac dòng trùng lặp

df_sales.duplicated()



# => So dong trung lap la 0 
df_sales.duplicated().sum()


df_online.describe(include="all")


mask_online = (df_online['price_base'] <= 0) | (df_online['sum_total'] <= 0)

# Xoa cac gia tri bat thuong
df_online.drop(df_online[mask_online].index, axis=0, inplace=True)

# check cac gia tri khong hop ly sau khi đã xóa 
df_online[(df_online['price_base'] <= 0) |
         (df_online['sum_total'] <= 0)]


# kiem tra missing value
df_online.isna().sum()


# so luong trung lap
df_online.duplicated().sum()


df_actual_matrix.describe(include="all")


# Kiem tra missing value
df_actual_matrix.isna().sum()


df_actual_matrix.duplicated().sum()


# Them một cột mới tên "is_available"
#→ Mục đích: đánh dấu các dòng này là "có hàng", hoặc "sẵn sàng bán".
df_actual_matrix['is_available'] = True #"có hàng


df_sales_merged_actual_matrix = pd.merge(df_sales, df_actual_matrix, on=['date', 'item_id', "store_id"], how='left')
df_sales_merged_actual_matrix.isna().sum()
#how='left': giữ nguyên tất cả dòng của df_sales, và chỉ nối thông tin từ df_actual_matrix nếu khớp.
#Nếu không khớp → cột is_available sẽ bị NaN (giá trị thiếu).


# Handling missing value 
df_sales_merged_actual_matrix.fillna(False, inplace=True) # False : hàng không có sẵn
df_sales_merged_actual_matrix.isna().sum()


#doi ten tranh bi trung voi df_sales_merged_actual_matrix
df_online = df_online.rename(columns={"price_base":"price_base_online", "sum_total":"sum_total_online"})
df_online["online"] = True

df = pd.merge(df_sales_merged_actual_matrix, df_online, on=['date', 'item_id', "store_id"], how='outer', suffixes=('_x', '_y'))
df["quantity"] = df[['quantity_x', 'quantity_y']].sum(axis=1)
#how='outer': giữ tất cả dữ liệu từ cả hai bảng (kể cả nếu không khớp).
#Khi cột trùng tên (ví dụ quantity), Pandas sẽ đặt hậu tố _x, _y:
#quantity_x: từ bảng df_sales_merged_actual_matrix (offline).
#quantity_y: từ bảng df_online (online).

df.isna().sum()


# Khi online == NaN → hiểu là sản phẩm không có bán online.
#Handling missing value 

df['online'].fillna(False, inplace = True) # không có thông tin → coi như không online
df['is_available'].fillna(True, inplace = True) #Mặc định coi là có hàng
df.isna().sum()


# COi như giá bằng 0
#Cột số lượng bán (quantity_x, quantity_y) → 0 nghĩa là không bán
df.fillna(0, inplace = True)
df.isna().sum()


# gop sum_total lai
df["sum_total_both"] = df[['sum_total', 'sum_total_online']].sum(axis=1) ##tong hang ngang df['summ_total'+'sum_total_online']
df["price_base_both"] = df["sum_total_both"] / df["quantity"]
df[(df['online'] == True)].head()


#loc cac bang 
df = df[['date', 'item_id', 'store_id', 'is_available', 'online', 'quantity', 'sum_total_both', 'price_base_both']]
df.describe(include = "all")


df_markdowns.describe(include="all")


# Lọc các dữ liệu không hợp lý
mask = (df_markdowns.normal_price <= df_markdowns.price)
df_markdowns[mask]


# Xóa các giá trị không hợp lệ 
df_markdowns.drop(df_markdowns[mask].index, axis=0, inplace=True)


#Kiểm tra missing value
df_markdowns.isna().sum()


# Kiểm tra trùng lặp
df_markdowns.duplicated().sum()


df_markdowns[df_markdowns.duplicated(keep=False)].sort_values(by=['item_id', 'date'])




df_markdowns_clean = df_markdowns.drop_duplicates(keep='first').reset_index(drop=True)
df_markdowns_clean



df_markdowns_clean[df_markdowns_clean.duplicated(keep=False)].sort_values(by=['item_id', 'date'])


df = df.merge(df_stores, how='left', left_on=["store_id"], right_on=["store_id"])
df.drop(["format", 'area'], axis=1, inplace = True)
df.head()


df_discounts_history.describe(include="all")


mask = (df_discounts_history.sale_price_before_promo <= df_discounts_history.sale_price_time_promo) | (df_discounts_history.sale_price_before_promo <= 0) | (df_discounts_history.sale_price_time_promo < 0) | (df_discounts_history.date > df_test.date.max())
df_discounts_history[mask]


df_discounts_history.drop(df_discounts_history[mask].index, axis=0, inplace=True)


mask = ((df_discounts_history.date >= df_test.date.min()) & (df_discounts_history.date <= df_test.date.max()))
df_discounts_history[mask]


print(df_test.date.max())
print(df_test.date.min())


# lọc dữ liệu khuyến mãi nằm trong khoảng thời gian của tập test
mask = ((df_discounts_history.date >= df_test.date.min()) & (df_discounts_history.date <= df_test.date.max()))
df_discounts_history[mask]


# Ty le giam gia
df_discounts_history["discount_percentage"] = (df_discounts_history["sale_price_before_promo"] - df_discounts_history["sale_price_time_promo"])  / df_discounts_history["sale_price_before_promo"]
df_discounts_history[mask]


# Kiểm tra missing value 
df_discounts_history.isna().sum()


df_discounts_history.duplicated().sum()


df_price_history.describe(include="all")


mask = (df_price_history.price <= 0)

# Xoa truong hop bat hop ly
df_price_history.drop(df_price_history[mask].index, axis=0, inplace=True)


df_price_history = pd.read_csv(r"/kaggle/input/ml-zoomcamp-2024-competition/price_history.csv", index_col=0, parse_dates=["date"])


df_price_history.duplicated(subset=['item_id', 'store_id', 'date']).sum()


#loại bỏ các dòng trùng lặp, chỉ giữ lại dòng cuối cùng 
df_price_history = df_price_history.drop_duplicates(subset=['item_id', 'store_id', 'date'], keep='last')


# gán toàn bộ cột price trong DataFrame df thành giá trị thiếu (NA)
df['price'] = pd.NA


# Tìm giá gần nhất trong quá khứ
df_price_history['date'].head()


df['date'].iloc[0]



#  Lọc giá trị tối đa date <= row['date'] và gộp lại với df_price_history
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


# Kiem tra missing value
df.isna().sum()



df['price'] = df['price'].fillna(df['price_base_both'])


df.head()


# lọc những dòng mà giá thực tế 
df.loc[df['price'] < df['price_base_both'], 'price'] = df['price_base_both']
df.head()


#  lưu lại danh sách các cột hiện có của df de giu dung thu tu
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


df.isna().sum()


def get_colums_with_nan(df):
    return df.columns[df.isna().sum() > 0]
    
cols = get_colums_with_nan(df)

df[cols].isna().sum()*100/len(df)


df_test[cols].isna().sum()*100/len(df_test)


# Xoa fatness
df = df.drop(["fatness"], axis=1)
df_test = df_test.drop(["fatness"], axis=1)


# dữ liệu phân loại => điền giá trị mặc định "không rõ"
def fill_catalog(dataframe, item_name="other"):
    dataframe.dept_name = dataframe.dept_name.fillna("other")
    dataframe.class_name = dataframe.class_name.fillna("other")
    dataframe.subclass_name = dataframe.subclass_name.fillna("other")
    dataframe.item_type = dataframe.item_type.fillna("other")
    return dataframe

df = fill_catalog(df, item_name="other")
df_test = fill_catalog(df_test, item_name="other")


cols = get_colums_with_nan(df)
cols


#numerical data
#Điền NaN bằng trung bình theo nhóm. Ưu tiên nhóm chi tiết trước.=> trung bình theo nhóm, từ chi tiết đến tổng quát
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

#tạo ra một dictionary mà giá trị mặc định là một danh sách rỗng [] nếu key chưa tồn tại.
price_dict = defaultdict(list)
for row in df_price_history.itertuples(index=False):
    key = (row.store_id, row.item_id)
    price_dict[key].append((row.date, row.price))


from bisect import bisect_right #tìm nhanh ngày gần nhất trước hoặc bằng

def get_latest_price(key, date):
    entries = price_dict.get(key, []) # Lấy danh sách (date, price) của sản phẩm tại cửa hàng
    dates = [d for d, _ in entries] # Trích danh sách các ngày
    idx = bisect_right(dates, date) - 1  # Tìm index của ngày gần nhất trước hoặc bằng ngày cần

    if idx >= 0:
        return entries[idx][1] # Trả về giá tương ứng
    return None  # hoặc giữ nguyên -1 nếu không có giá nào trước đó

# Áp dụng từng dòng
df_test['price'] = [
    get_latest_price((row.store_id, row.item_id), row.date)
    for row in df_test.itertuples(index=False)
]


df.isna().sum()


# Kiem tra missing value
df_test.isna().sum()


df_test = df_test.fillna(-1) #khong xac dinh


#tổng số lượng (quantity) bán ra theo từng ngày 
aux = df.groupby('date', as_index=False).quantity.sum() #as_index=False: kết quả trả về sẽ không dùng date làm chỉ số (index)
plt.figure(figsize=(15,3))
sns.histplot(data=aux, x='quantity', kde=True, stat='density')
quant_50 = aux.quantity.quantile(0.5)
quant_75 = aux.quantity.quantile(0.75)
quant_95 = aux.quantity.quantile(0.95)

quants = [[quant_50,1,2], [quant_75, 0.8, 0.6], [quant_95, 0.6, 0.3]] #alpha (độ trong suốt của đường)

#ymax (độ cao tương đối của đường, từ 0 đến 1)

#Vẽ các đường phân vị dọc
for i in quants :
    plt.axvline(i[0], alpha=i[1], ymax=i[2], linestyle=":") 

# Them chu thich 
plt.text(quant_50-.13, .000051, "50th", size=12, alpha = 0.85) #plt.text(x, y, text, ...): thêm nhãn văn bản tại vị trí (x, y) trên biểu đồ.

plt.text(quant_75-.13, .000032, "75th", size = 10.5, alpha = .85)
plt.text(quant_95-.25, .000016, "95th Percentile", size = 10, alpha =.8);

plt.title("Quantity distribution")


aux = df.groupby('date', as_index=False).quantity.sum()
plt.figure(figsize=(15,5))

sns.lineplot(aux, x='date', y='quantity')
plt.xlabel('Date')
plt.ylabel('Quantity')


aux = df.groupby(['date']).quantity.sum()
clp(aux, cmap="BuPu")


colors =[
    '#08F7FE',
    '#FE53BB',
    '#F5D300',
    '#00ff41',
]

aux = df.groupby(['date', 'store_id'], as_index=False).quantity.sum()

plt.figure(figsize=(12,5))
for i in range(4):
    aux2=aux[aux.store_id == i+1]
    plt.plot(aux2.date, aux2.quantity, label=f"store{i+1}", color=colors[i])

plt.legend() #hiển thị chú thích 
plt.xlim(aux.date.min(), aux.date.max())
plt.xlabel('Date')
plt.ylabel('Quantity')


aux = df.groupby(['store_id'], as_index=False).quantity.mean()

aux.store_id = aux.store_id.apply(str) #chuyen sang kiểu chuỗi (string)

plt.figure(figsize=(12,5))
sns.barplot(data=aux, y='store_id', x='quantity', orient='h') # huong ngang

plt.ylabel('Store ID')
plt.xlabel('Quantity')


aux = df.groupby(['date', 'store_id'], as_index=False).quantity.sum()

plt.figure(figsize=(12,5))

for i in range(4):
    sns.kdeplot(aux , x='quantity', hue='store_id', palette=colors)
# hue='store_id' giúp biểu đồ phân biệt dữ liệu theo từng cửa hàng bằng màu sắc


aux= df.groupby(by=['date', 'store_id'], as_index=False).quantity.sum()

plt.figure(figsize=(12,5))

for i in range(4):
    aux2=aux[aux.store_id == i+1]
    k = aux2.quantity.median()
    plt.plot(aux2.date, aux2.quantity/k , label=f"store{i+1}", color=colors[i])

plt.legend()
plt.xlim(aux.date.min(), aux.date.max())
plt.xlabel('Date')
plt.ylabel('Quantity')
    


aux = df.groupby(by=['date', 'store_id'], as_index=False).quantity.sum()

plt.figure(figsize=(13,5))
for i in range(4):
    aux2 = aux[aux.store_id == i+1]
    k = aux2.quantity.median()
    sns.kdeplot(aux2.quantity/k, color=colors[i])


plt.figure(figsize=(12,5))
quantity_fractions = {}
quantity_per_column = df.groupby(['date', "store_id"])["quantity"].sum().reset_index().pivot(index="date", columns="store_id", values='quantity').reset_index(drop=True)
quantity_fractions["store_id"] = quantity_per_column.divide(quantity_per_column.sum(axis=1), axis=0)
for i in range(4):
    sns.lineplot(x=df.date.unique(), y=quantity_fractions["store_id"][i+1]);


aux = df.groupby(["item_type"], as_index=False).quantity.sum().sort_values(by="quantity", ascending=False)[:20]

plt.figure(figsize=(25, 15))
sns.barplot(aux, y="item_type", x="quantity", color=colors[1]);

plt.xlabel("Quantity", fontsize=20)
plt.ylabel("Item Type", fontsize=20)
plt.title("Top 20 Item Types by Quantity", fontsize=24)

# Tăng cỡ chữ cho nhãn trục
plt.tick_params(axis='both', labelsize=18)


def date_features(dataframe):
    dataframe["dayofmonth"] = dataframe.date.dt.day
    dataframe["month"] = dataframe.date.dt.month
    dataframe["dayofyear"] = dataframe.date.dt.dayofyear
    dataframe["year"] = dataframe.date.dt.year
    dataframe['dayofweek'] = dataframe['date'].dt.dayofweek 
    dataframe['week'] = dataframe['date'].dt.isocalendar().week
    return dataframe

# áp dụng cho cả tập huấn luyện và tập kiểm tra
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
    dataframe["holidays"] = False #Mặc định không phải ngày lễ
    dataframe.loc[df.date.isin(RU_holidays.keys()), "holidays"] = True #Gán True nếu date rơi vào ngày nghỉ
    return dataframe

df = get_holidays(df)
df_test = get_holidays(df_test)


def get_sundays(dataframe):
    dataframe["is_sunday"] = dataframe['dayofweek'].eq(6) # 6 tương ứng chủ nhật - 0 tương ứng t2
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
        'week', #Tránh trùng lặp thông tin
        'online',
        'is_available', #không mang giá trị dự đoán tốt
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


#Tạo một pipeline xử lý đặc trưng
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
    remainder='passthrough', #Các cột không được liệt kê (không phải số cũng không phải chuỗi) sẽ được giữ nguyên
    verbose_feature_names_out=False
)

X_transformed = column_transformer.fit_transform(X) # huấn luyện bộ biến đổi trên X và áp dụng biến đổi
X_test_transformed = column_transformer.transform(df_test) #áp dụng đúng biến đổi đã học từ tập huấn luyện lên df_test


seed_value = 42
X_train, X_val, y_train, y_val = train_test_split(X_transformed, y, train_size=0.8, random_state=seed_value)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)


import optuna
import numpy as np
import joblib
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

# 1. Hàm objective cho Optuna
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 1000, 10000),
        'learning_rate': trial.suggest_float('learning_rate', 0.3, 0.6),
        'depth': trial.suggest_int('depth', 10, 14),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 5.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 0.1),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.5, 1.0),
        'langevin': True,
        'random_seed': 42,
        'loss_function': 'RMSE',
        'verbose': False
    }

    model = CatBoostRegressor(**params)

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=20
    )

    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))

    # Lưu model tốt nhất nếu RMSE cải thiện
    if rmse < objective.best_rmse:
        joblib.dump(model, "best_model_catboost.pkl")
        objective.best_rmse = rmse

    return rmse

# Gán giá trị RMSE ban đầu lớn nhất
objective.best_rmse = float("inf")

# 2. Tối ưu với Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30, timeout=3600)

print(" Best params:", study.best_params)
print(" Best RMSE:", study.best_value)

# 3. Load model tốt nhất và dự đoán
best_model = joblib.load("best_model_catboost.pkl")

# Dự đoán trên tập test
quantity_pred = best_model.predict(X_test_transformed)

# 4. Tạo submission
df_test["quantity"] = quantity_pred
df_submission = df_test[["quantity"]]

# Xem trước kết quả
print(df_submission.head())



quantity_pred = best_model.predict(X_test_transformed)
df_test["quantity"] = quantity_pred
df_submission = df_test[["quantity"]]
df_submission.head()


missing_items = df_test[~df_test.item_id.isin(df.item_id)]
missing_items.head()


df_submission.loc[missing_items.index, "quantity"] = 0


df_submission.to_csv("/kaggle/working/submission.csv", index_label='row_id')

