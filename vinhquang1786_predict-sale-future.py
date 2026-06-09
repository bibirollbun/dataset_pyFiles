!pip install pandas numpy seaborn matplotlib scipy scikit-learn
!pip install nltk
!pip install xgboost
!pip install lightgbm
!pip install catboost



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Mount Google Drive if using Google Colab
# from google.colab import drive
# drive.mount('/content/drive')

path = "/kaggle/working/"
path2 = "/kaggle/input/competitive-data-science-predict-future-sales/"

# # Đọc dữ liệu, adjust the file paths if necessary
# sales = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv') # Updated file path
# items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv') # Updated file path
# categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv') # Updated file path
# shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv') # Updated file path

# # Gộp sales với items (theo item_id)
# df = sales.merge(items, on='item_id', how='left')

# # Gộp với categories (theo item_category_id)
# df = df.merge(categories, on='item_category_id', how='left')

# # Gộp với shops (theo shop_id)
# df = df.merge(shops, on='shop_id', how='left')
# df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")



# #Dọc dữ liệu từ test.csv
# test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
# test = test.merge(items, on="item_id", how="left")
# test = test.merge(categories, on="item_category_id", how="left")


# last_tx_shop_0 = df[df['shop_id'] == 0].sort_values('date', ascending=False).head(1)

# # Giao dịch đầu tiên của shop 57
# first_tx_shop_57 = df[df['shop_id'] == 57].sort_values('date', ascending=True).head(1)

# # In kết quả
# print("Giao dịch cuối cùng của shop 0:")
# print(last_tx_shop_0)

# print("\nGiao dịch đầu tiên của shop 57:")
# print(first_tx_shop_57)


# print("Số Dòng Dữ Liệu: ",df.shape[0])
# print("Số Cột Dữ Liệu: ",df.shape[1])



# df.head(5)



# df.info()


# pd.set_option('display.float_format', '{:.4f}'.format)  # Hiển thị 4 chữ số thập phân

# df.describe()



# plt.figure(figsize=(12, 6))
# sns.histplot(df['item_price'], bins=30, kde=True)
# plt.title('Phân phối giá sản phẩm')
# plt.show()



# # # # Box plot để phát hiện outliers
# plt.figure(figsize=(12, 6))
# sns.boxplot(
#     y=df['item_price'],
#     width=0.5,      # độ rộng của box
#     fliersize=10,     # kích thước outlier

# )
# plt.title('Boxplot phát hiện outliers trong giá sản phẩm')
# plt.show()



# plt.figure(figsize=(12, 6))
# sns.histplot(df['item_cnt_day'], bins=20, kde=True)
# plt.title('Phân phối số lượng sản phẩm')
# plt.show()



# # # # Box plot để phát hiện outliers
# plt.figure(figsize=(12, 6))
# sns.boxplot(
#     y=df['item_cnt_day'],
#     width=0.5,      # độ rộng của box
#     fliersize=10,     # kích thước outlier

# )
# plt.title('Boxplot phát hiện outliers trong số sản phẩm')
# plt.show()



# df.sort_values('item_cnt_day', ascending=False).head(5)



# # sns.scatterplot(x='item_price', y='item_cnt_day', data=df)

# # Biểu đồ phân tán
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='item_price', y='item_cnt_day', data=df)
# plt.title('Mối quan hệ giữa giá sản phẩm và số lượng bán')
# plt.show()

# # Tính hệ số tương quan
# correlation = df['item_price'].corr(df['item_cnt_day'])
# print(f"Hệ số tương quan: {correlation}")



# monthly_sales = sales.groupby('date_block_num')['item_cnt_day'].sum()

# plt.figure(figsize=(12,6))
# monthly_sales.plot(kind='line')
# plt.xticks(ticks=range(34))
# plt.xlabel('Tháng (date_block_num)')
# plt.ylabel('Tổng số sản phẩm bán')
# plt.title('Tổng số sản phẩm bán theo từng tháng')
# plt.grid(True)
# plt.show()


# # Box plot cho item_price theo item_category_id
# plt.figure(figsize=(20, 6))
# sns.boxplot(x='item_category_id', y='item_price', data=df)
# plt.title('Phân phối giá sản phẩm theo danh mục')
# plt.xticks(rotation=45)
# plt.show()





# plt.figure(figsize=(20, 6))

# sales_by_shop = df.groupby('shop_id')['item_cnt_day'].sum().reset_index()

# plt.figure(figsize=(12, 6))
# sns.barplot(x='shop_id', y='item_cnt_day', data=sales_by_shop)
# plt.title('Tổng Số Lượng Bán Theo Cửa Hàng')
# plt.xlabel('Cửa Hàng')
# plt.ylabel('Tổng Số Lượng Bán')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()



# sales_by_category = df.groupby('item_category_id')['item_cnt_day'].sum().reset_index()
# sales_by_category = sales_by_category.sort_values('item_cnt_day', ascending=False)  # Sắp xếp giảm dần


# plt.figure(figsize=(20, 6))
# sns.barplot(x='item_category_id', y='item_cnt_day', data=sales_by_category)
# plt.title('Tổng Số Lượng Bán Theo Danh Mục Sản Phẩm')
# plt.xlabel('Danh Mục Sản Phẩm')
# plt.ylabel('Tổng Số Lượng Bán')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()




# correlation_matrix = df[['item_price', 'item_cnt_day', 'date_block_num']].corr()
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
# plt.show()



# test.head(5)



# print(f"There are {len(test['shop_id'].unique())} unique shop_id's and {len(test['item_id'].unique())} unique item_id's in the test set.")
# print(f"There are {len(test)} test items in total.")


# unique_items_date_block = [df.loc[df.date_block_num==x,:].item_id.unique().size for x in range(34)]
# unique_items_date_block.append(test.item_id.unique().size)
# ax = sns.lineplot(x = range(len(unique_items_date_block)), y = unique_items_date_block, marker='.', markersize=12)
# _ = ax.set(xlabel = "Date block", ylabel = "Unique items sold", title = "Unique items per month (test set=date block 34)")


# intersection = [len(set(df.loc[df.date_block_num==x,:].item_id).intersection(set(test.item_id))) for x in range(33)]
# intersection.append(len(set(test.item_id)))
# ax = sns.lineplot(x = range(len(intersection)), y = intersection)
# _ = ax.set(xlabel = "Date block", ylabel = "Intersection with date block 34")
# _ = ax.set_title("Sold items also found in test set")


# intersection = [len(set(df.loc[df.date_block_num==i+1,:].item_id) -
#     set(df.loc[df.date_block_num==i,:].item_id)) for i in range(33)]
# intersection.append(len(set(df.loc[df.date_block_num==33].item_id) - set(test.item_id)))
# ax = sns.lineplot(x = range(1, len(intersection)+1), y = intersection)
# _ = ax.set(xlabel = "Date block", ylabel = "New items")
# _ = ax.set_title("Items not seen in the preceding month")


def reduce_mem_usage(df, silent=True, allow_categorical=True, float_dtype="float32"):
    """
    Iterates through all the columns of a dataframe and downcasts the data type
     to reduce memory usage. Can also factorize categorical columns to integer dtype.
    """
    def _downcast_numeric(series, allow_categorical=allow_categorical):
        """
        Downcast a numeric series into either the smallest possible int dtype or a specified float dtype.
        """
        if isinstance(series.dtype, pd.SparseDtype):
            return series
        elif pd.api.types.is_numeric_dtype(series.dtype) is False:
            if pd.api.types.is_datetime64_any_dtype(series.dtype):
                return series
            else:
                if allow_categorical:
                    return series
                else:
                    codes, uniques = series.factorize()
                    series = pd.Series(data=codes, index=series.index)
                    series = _downcast_numeric(series)
                    return series
        else:
            series = pd.to_numeric(series, downcast="integer")
        if pd.api.types.is_float_dtype(series.dtype):
            series = series.astype(float_dtype)
        return series

    if silent is False:
        start_mem = np.sum(df.memory_usage()) / 1024 ** 2
        print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
    if df.ndim == 1:
        df = _downcast_numeric(df)
    else:
        for col in df.columns:
            df.loc[:, col] = _downcast_numeric(df.loc[:,col])
    if silent is False:
        end_mem = np.sum(df.memory_usage()) / 1024 ** 2
        print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
        print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def shrink_mem_new_cols(matrix, oldcols=None, allow_categorical=False):
    # Calls reduce_mem_usage on columns which have not yet been optimized
    if oldcols is not None:
        newcols = matrix.columns.difference(oldcols)
    else:
        newcols = matrix.columns
    matrix.loc[:,newcols] = reduce_mem_usage(matrix.loc[:,newcols], allow_categorical=allow_categorical)
    oldcols = matrix.columns  # This is used to track which columns have already been downcast
    return matrix, oldcols


def list_if_not(s, dtype=str):
    # Puts a variable in a list if it is not already a list
    if type(s) not in (dtype, list):
        raise TypeError
    if (s != "") & (type(s) is not list):
        s = [s]
    return s


# #Xử lí null
# df.isnull().sum()


# duplicates=df[df.duplicated(keep=False)]
# print("Số hàng trùng nhau:",len(duplicates))
# duplicates.head(6)


# #Xóa duplicate
# df.drop_duplicates(inplace=True)


# pd.set_option('display.float_format', '{:.4f}'.format)  # Hiển thị 4 chữ số thập phân

# df.describe()



# df.drop(df[(df['item_price'] < 0) | (df['item_cnt_day'] < 0)].index, inplace=True)





# #Chuyển lại định dạng ngày tháng phù hợp
# df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y') # Changed the format to '%d.%m.%Y'


# #giữ lại những shop id có trong tập test
# df= df.loc[df.shop_id.isin(test["shop_id"].unique()), :]



# # Tạo một CategoricalDtype với thứ tự mong muốn
# desired_order = [0, 57, 1, 58, 11, 10, 40, 39]
# shop_id_order = pd.CategoricalDtype(desired_order, ordered=True)

# # Chuyển đổi cột 'shop_id' thành CategoricalDtype
# shopscp=shops.copy()
# shopscp['shop_id'] = shopscp['shop_id'].astype(shop_id_order)

# # Lọc dữ liệu và sắp xếp theo 'shop_id'
# filtered_df = shopscp[shopscp['shop_id'].isin(desired_order)]
# sorted_df = filtered_df.sort_values(by=['shop_id'], key=lambda x: x.cat.codes)

# # Hiển thị kết quả
# print(sorted_df)


# df["shop_id"] = df["shop_id"].replace({0: 57, 1: 58, 11: 10, 40: 39})



# import scipy.stats as stats

# # Histogram

# plt.figure(figsize=(6, 4))
# sns.histplot(df['item_price'], bins=30, kde=True)
# plt.title('Phân phối giá sản phẩm')
# plt.show()

# plt.figure(figsize=(6, 4))
# sns.histplot(df['item_cnt_day'], bins=30, kde=True)
# plt.title('Phân phối giá sản phẩm')
# plt.show()


# # Q-Q Plot
# for col in ['item_price', 'item_cnt_day']:
#     plt.figure(figsize=(6, 4))
#     stats.probplot(df[col], dist="norm", plot=plt)
#     plt.title(f"Q-Q Plot - {col}")
#     plt.show()











# #Xác định outlier theo biến item_cnt_day
# Q1 = df['item_cnt_day'].quantile(0.25)
# Q3 = df['item_cnt_day'].quantile(0.75)
# IQR = Q3 - Q1

# outliers0 = df[(df['item_cnt_day'] < Q1 - 1.5*IQR) | (df['item_cnt_day'] > Q3 + 1.5*IQR)]
# print("Lower Bound:",Q1 - 1.5*IQR )
# print("Upper Bound:",Q3 + 1.5*IQR )
# print("Số lượng outlier:", outliers0.shape[0])
# outliers0.head()



# Q1 = df['item_price'].quantile(0.25)
# Q3 = df['item_price'].quantile(0.75)
# IQR = Q3 - Q1

# outliers1 = df[(df['item_price'] < Q1 - 1.5*IQR) | (df['item_price'] > Q3 + 1.5*IQR)]
# print("Lower Bound:",Q1 - 1.5*IQR )
# print("Upper Bound:",Q3 + 10*IQR )
# print("Số lượng outlier:", outliers1.shape[0])
# outliers1.head()



# outliers_intersection = pd.merge(outliers0,outliers1, how='inner', on=df.columns.tolist())
# outliers_intersection.head()
# print("Độ dài của outliers0:",len(outliers0))
# print("Độ dài của outliers1:",len(outliers1))
# print("Độ dài của outliers sau khi giao lại:",len(outliers_intersection))
# print("Số lượng hàng của data: ",len(df))


# df.drop(df[(df['item_price'] > 50000) | (df['item_cnt_day'] > 1000)].index, inplace=True)




import itertools
# def create_testlike_train(sales_train, test=None):
#     indexlist = []
#     for i in sales_train.date_block_num.unique():
#         x = itertools.product(
#             [i],
#             sales_train.loc[sales_train.date_block_num == i].shop_id.unique(),
#             sales_train.loc[sales_train.date_block_num == i].item_id.unique(),
#         )
#         indexlist.append(np.array(list(x)))
#     df = pd.DataFrame(
#         data=np.concatenate(indexlist, axis=0),
#         columns=["date_block_num", "shop_id", "item_id"],
#     )

#     # Add revenue column to sales_train
#     sales_train["item_revenue_day"] = sales_train["item_price"] * sales_train["item_cnt_day"]
#     # Aggregate item_id / shop_id item_cnts and revenue at the month level
#     sales_train_grouped = sales_train.groupby(["date_block_num", "shop_id", "item_id"]).agg(
#         item_cnt_month=pd.NamedAgg(column="item_cnt_day", aggfunc="sum"),
#         item_revenue_month=pd.NamedAgg(column="item_revenue_day", aggfunc="sum"),
#     )

#     # Merge the grouped data with the index
#     df = df.merge(
#         sales_train_grouped, how="left", on=["date_block_num", "shop_id", "item_id"],
#     )

#     if test is not None:
#         test["date_block_num"] = 34
#         test["date_block_num"] = test["date_block_num"].astype(np.int8)
#         test["shop_id"] = test.shop_id.astype(np.int8)
#         test["item_id"] = test.item_id.astype(np.int16)
#         test = test.drop("ID",axis=1)

#         df = pd.concat([df, test[["date_block_num", "shop_id", "item_id"]]])

#     # Fill empty item_cnt entries with 0
#     df.item_cnt_month = df.item_cnt_month.fillna(0)
#     df.item_revenue_month = df.item_revenue_month.fillna(0)

#     return df



# matrix = create_testlike_train(df, test)
# del(test)





# matrix


# oldcols = matrix.columns


 # items.query("item_id>3564").head(5)


!pip install fuzzywuzzy


import re

from fuzzywuzzy import fuzz


# def add_item_name_groups(matrix, train, items, sim_thresh, feature_name="item_name_group"):
#     def partialmatchgroups(items, sim_thresh=sim_thresh):
#         def strip_brackets(string):
#             string = re.sub(r".∗?.*?", "", string)
#             string = re.sub(r".∗?.*?", "", string)
#             return string

#         items = items.copy()
#         items["nc"] = items.item_name.apply(strip_brackets)
#         items["ncnext"] = np.concatenate((items["nc"].to_numpy()[1:], np.array([""])))

#         def partialcompare(s):
#             return fuzz.partial_ratio(s["nc"], s["ncnext"])

#         items["partialmatch"] = items.apply(partialcompare, axis=1)
#         # Assign groups
#         grp = 0
#         for i in range(items.shape[0]):
#             items.loc[i, "partialmatchgroup"] = grp
#             if items.loc[i, "partialmatch"] < sim_thresh:
#                 grp += 1
#         items = items.drop(columns=["nc", "ncnext", "partialmatch"])
#         return items

#     items = partialmatchgroups(items)
#     items = items.rename(columns={"partialmatchgroup": feature_name})
#     items = items.drop(columns="partialmatchgroup", errors="ignore")

#     items[feature_name] = items[feature_name].apply(str)
#     items[feature_name] = items[feature_name].factorize()[0]
#     matrix = matrix.merge(items[["item_id", feature_name]], on="item_id", how="left")
#     train = train.merge(items[["item_id", feature_name]], on="item_id", how="left")
#     return matrix, train


# matrix, train = add_item_name_groups(matrix, df, items, 65)


# matrix



import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# def add_first_word_features(matrix, items=items, feature_name="artist_name_or_first_word"):
#     # This extracts artist names for music categories and adds them as a feature.
#     def extract_artist(st):
#         st = st.strip()
#         if st.startswith("V/A"):
#             artist = "V/A"
#         elif st.startswith("СБ"):
#             artist = "СБ"
#         else:
#             # Retrieves artist names using the double space or all uppercase pattern
#             mus_artist_dubspace = re.compile(r".{2,}?(?=\s{2,})")
#             match_dubspace = mus_artist_dubspace.match(st)
#             mus_artist_capsonly = re.compile(r"^([^a-zа-я]+\s)+")
#             match_capsonly = mus_artist_capsonly.match(st)
#             candidates = [match_dubspace, match_capsonly]
#             candidates = [m[0] for m in candidates if m is not None]
#             # Sometimes one of the patterns catches some extra words so choose the shortest one
#             if len(candidates):
#                 artist = min(candidates, key=len)
#             else:
#                 # If neither of the previous patterns found something, use the dot-space pattern
#                 mus_artist_dotspace = re.compile(r".{2,}?(?=\.\s)")
#                 match = mus_artist_dotspace.match(st)
#                 if match:
#                     artist = match[0]
#                 else:
#                     artist = ""
#         artist = artist.upper()
#         artist = re.sub(r"[^A-ZА-Я ]||\bTHE\b", "", artist)
#         artist = re.sub(r"\s{2,}", " ", artist)
#         artist = artist.strip()
#         return artist

#     items = items.copy()
#     all_stopwords = stopwords.words("russian")
#     all_stopwords = all_stopwords + stopwords.words("english")

#     def first_word(string):
#         # This cleans the string of special characters, excess spaces and stopwords then extracts the first word
#         string = re.sub(r"[^\w\s]", "", string)
#         string = re.sub(r"\s{2,}", " ", string)
#         tokens = string.lower().split()
#         tokens = [t for t in tokens if t not in all_stopwords]
#         token = tokens[0] if len(tokens) > 0 else ""
#         return token

#     music_categories = [55, 56, 57, 58, 59, 60]
#     items.loc[items.item_category_id.isin(music_categories), feature_name] = items.loc[
#         items.item_category_id.isin(music_categories), "item_name"
#     ].apply(extract_artist)
#     items.loc[items[feature_name] == "", feature_name] = "other music"
#     items.loc[~items.item_category_id.isin(music_categories), feature_name] = items.loc[
#         ~items.item_category_id.isin(music_categories), "item_name"
#     ].apply(first_word)
#     items.loc[items[feature_name] == "", feature_name] = "other non-music"
#     items[feature_name] = items[feature_name].factorize()[0]
#     matrix = matrix.merge(items[["item_id", feature_name]], on="item_id", how="left",)
#     return matrix


# matrix = add_first_word_features(
#     matrix, items=items, feature_name="artist_name_or_first_word"
# )


# matrix


import re
# def clean_item_name(string):
#     # Removes bracketed terms, special characters and extra whitespace
#     string = re.sub(r".∗?.*?", "", string)
#     string = re.sub(r".∗?.*?", "", string)
#     string = re.sub(r"[^A-ZА-Яa-zа-я0-9 ]", "", string)
#     string = re.sub(r"\s{2,}", " ", string)
#     string = string.lower()
#     return string

# items["item_name_cleaned_length"] = items["item_name"].apply(clean_item_name).apply(len)
# items["item_name_length"] = items["item_name"].apply(len)
# matrix = matrix.merge(items[['item_id', 'item_name_length', 'item_name_cleaned_length']], how='left', on='item_id')
# items = items.drop(columns=['item_name_length', 'item_name_cleaned_length'])



# print("Created name features")



# oldcols


# def add_time_features(m, train, correct_item_cnt_day=False):
#     from pandas.tseries.offsets import Day, MonthBegin, MonthEnd

#     def item_shop_age_months(m):
#         m["item_age"] = m.groupby("item_id")["date_block_num"].transform(
#             lambda x: x - x.min()
#         )
#         # Sales tend to plateau after 12 months
#         m["new_item"] = m["item_age"] == 0
#         m["new_item"] = m["new_item"].astype("int8")
#         m["shop_age"] = (
#             m.groupby("shop_id")["date_block_num"]
#             .transform(lambda x: x - x.min())
#             .astype("int8")
#         )
#         return m

#     # Add dummy values for the test month so that features are created correctly
#     dummies = m.loc[m.date_block_num == 34, ["date_block_num", "shop_id", "item_id"]]
#     dummies = dummies.assign(
#         date=pd.to_datetime("2015-11-30"), item_price=1, item_cnt_day=0, item_revenue_day=0,
#     )
#     train = pd.concat([train, dummies])
#     del dummies

#     month_last_day = train.groupby("date_block_num").date.max().rename("month_last_day")
#     month_last_day[~month_last_day.dt.is_month_end] = (
#         month_last_day[~month_last_day.dt.is_month_end] + MonthEnd()
#     )
#     month_first_day = train.groupby("date_block_num").date.min().rename("month_first_day")
#     month_first_day[~month_first_day.dt.is_month_start] = (
#         month_first_day[~month_first_day.dt.is_month_start] - MonthBegin()
#     )
#     month_length = (month_last_day - month_first_day + Day()).rename("month_length")
#     first_shop_date = train.groupby("shop_id").date.min().rename("first_shop_date")
#     first_item_date = train.groupby("item_id").date.min().rename("first_item_date")
#     first_shop_item_date = (
#         train.groupby(["shop_id", "item_id"]).date.min().rename("first_shop_item_date")
#     )
#     first_item_name_group_date = (
#         train.groupby("item_name_group").date.min().rename("first_name_group_date")
#     )

#     m = m.merge(month_first_day, left_on="date_block_num", right_index=True, how="left")
#     m = m.merge(month_last_day, left_on="date_block_num", right_index=True, how="left")
#     m = m.merge(month_length, left_on="date_block_num", right_index=True, how="left")
#     m = m.merge(first_shop_date, left_on="shop_id", right_index=True, how="left")
#     m = m.merge(first_item_date, left_on="item_id", right_index=True, how="left")
#     m = m.merge(
#         first_shop_item_date, left_on=["shop_id", "item_id"], right_index=True, how="left"
#     )
#     m = m.merge(
#         first_item_name_group_date, left_on="item_name_group", right_index=True, how="left"
#     )

#     # Calculate how long the item was sold for in each month and use this to calculate average sales per day
#     m["shop_open_days"] = m["month_last_day"] - m["first_shop_date"] + Day()
#     m["item_first_sale_days"] = m["month_last_day"] - m["first_item_date"] + Day()
#     m["item_in_shop_days"] = (
#         m[["shop_open_days", "item_first_sale_days", "month_length"]].min(axis=1).dt.days
#     )
#     m = m.drop(columns="item_first_sale_days")
#     m["item_cnt_day_avg"] = m["item_cnt_month"] / m["item_in_shop_days"]
#     m["month_length"] = m["month_length"].dt.days

#     # Calculate the time differences from the beginning of the month so they can be used as features without lagging
#     m["shop_open_days"] = m["month_first_day"] - m["first_shop_date"]
#     m["first_item_sale_days"] = m["month_first_day"] - m["first_item_date"]
#     m["first_shop_item_sale_days"] = m["month_first_day"] - m["first_shop_item_date"]
#     m["first_name_group_sale_days"] = m["month_first_day"] - m["first_name_group_date"]
#     m["shop_open_days"] = m["shop_open_days"].dt.days.fillna(0).clip(lower=0)
#     m["first_item_sale_days"] = (
#         m["first_item_sale_days"].dt.days.fillna(0).clip(lower=0).replace(0, 9999)
#     )
#     m["first_shop_item_sale_days"] = (
#         m["first_shop_item_sale_days"].dt.days.fillna(0).clip(lower=0).replace(0, 9999)
#     )
#     m["first_name_group_sale_days"] = (
#         m["first_name_group_sale_days"].dt.days.fillna(0).clip(lower=0).replace(0, 9999)
#     )

#     # Add days since last sale
#     def last_sale_days(matrix):
#         last_shop_item_dates = []
#         for dbn in range(1, 35):
#             lsid_temp = (
#                 train.query(f"date_block_num<{dbn}")
#                 .groupby(["shop_id", "item_id"])
#                 .date.max()
#                 .rename("last_shop_item_sale_date")
#                 .reset_index()
#             )
#             lsid_temp["date_block_num"] = dbn
#             last_shop_item_dates.append(lsid_temp)

#         last_shop_item_dates = pd.concat(last_shop_item_dates)
#         matrix = matrix.merge(
#             last_shop_item_dates, on=["date_block_num", "shop_id", "item_id"], how="left"
#         )

#         def days_since_last_feat(m, feat_name, date_feat_name, missingval):
#             m[feat_name] = (m["month_first_day"] - m[date_feat_name]).dt.days
#             m.loc[m[feat_name] > 2000, feat_name] = missingval
#             m.loc[m[feat_name].isna(), feat_name] = missingval
#             return m

#         matrix = days_since_last_feat(
#             matrix, "last_shop_item_sale_days", "last_shop_item_sale_date", 9999
#         )

#         matrix = matrix.drop(columns=["last_shop_item_sale_date"])
#         return matrix

#     m = last_sale_days(m)
#     # Month id feature
#     m["month"] = m["month_first_day"].dt.month

#     m = m.drop(
#         columns=[
#             "first_day",
#             "month_first_day",
#             "month_last_day",
#             "first_shop_date",
#             "first_item_date",
#             "first_name_group_date",
#             "item_in_shop_days",
#             "first_shop_item_date",
#             "month_length",
#         ],
#         errors="ignore",
#     )

#     m = item_shop_age_months(m)

#     if correct_item_cnt_day == True:
#         m["item_cnt_month_original"] = m["item_cnt_month"]
#         m["item_cnt_month"] = m["item_cnt_day_avg"] * m["month_length"]

#     return m


# matrix = add_time_features(matrix, train, False)
# print("Time features created")


# def add_price_features(matrix, train):
#     # Get mean prices per month from train dataframe
#     price_features = train.groupby(["date_block_num", "item_id"]).item_price.mean()
#     price_features = pd.DataFrame(price_features)
#     price_features = price_features.reset_index()
#     # Calculate normalized differenced from mean category price per month
#     price_features = price_features.merge(
#         items[["item_id", "item_category_id"]], how="left", on="item_id"
#     )
#     price_features["norm_diff_cat_price"] = price_features.groupby(
#         ["date_block_num", "item_category_id"]
#     )["item_price"].transform(lambda x: (x - x.mean()) / x.mean())
#     # Retain only the necessary features
#     price_features = price_features[
#         [
#             "date_block_num",
#             "item_id",
#             "item_price",
#             "norm_diff_cat_price",
#         ]
#     ]

#     features = [
#         "item_price",
#         "norm_diff_cat_price",
#     ]
#     newnames = ["last_" + f for f in features]
#     aggs = {f: "last" for f in features}
#     renames = {f: "last_" + f for f in features}
#     features = []
#     for dbn in range(1, 35):
#         f_temp = (
#             price_features.query(f"date_block_num<{dbn}")
#             .groupby("item_id")
#             .agg(aggs)
#             .rename(columns=renames)
#         )
#         f_temp["date_block_num"] = dbn
#         features.append(f_temp)
#     features = pd.concat(features).reset_index()
#     matrix = matrix.merge(features, on=["date_block_num", "item_id"], how="left")
#     return matrix



# matrix = add_price_features(matrix, train)
# del(train)


# matrix = matrix.merge(items[['item_id', 'item_category_id']], on='item_id', how='left')

# platform_map = {
#     0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 8, 10: 1, 11: 2,
#     12: 3, 13: 4, 14: 5, 15: 6, 16: 7, 17: 8, 18: 1, 19: 2, 20: 3, 21: 4, 22: 5,
#     23: 6, 24: 7, 25: 8, 26: 9, 27: 10, 28: 0, 29: 0, 30: 0, 31: 0, 32: 8, 33: 11,
#     34: 11, 35: 3, 36: 0, 37: 12, 38: 12, 39: 12, 40: 13, 41: 13, 42: 14, 43: 15,
#     44: 15, 45: 15, 46: 14, 47: 14, 48: 14, 49: 14, 50: 14, 51: 14, 52: 14, 53: 14,
#     54: 8, 55: 16, 56: 16, 57: 17, 58: 18, 59: 13, 60: 16, 61: 8, 62: 8, 63: 8, 64: 8,
#     65: 8, 66: 8, 67: 8, 68: 8, 69: 8, 70: 8, 71: 8, 72: 8, 73: 0, 74: 10, 75: 0,
#     76: 0, 77: 0, 78: 0, 79: 8, 80: 8, 81: 8, 82: 8, 83: 8,
# }
# matrix['platform_id'] = matrix['item_category_id'].map(platform_map)

# supercat_map = {
#     0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 2, 9: 2, 10: 1, 11: 1, 12: 1,
#     13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 3, 19: 3, 20: 3, 21: 3, 22: 3, 23: 3,
#     24: 3, 25: 0, 26: 2, 27: 3, 28: 3, 29: 3, 30: 3, 31: 3, 32: 2, 33: 2, 34: 2,
#     35: 2, 36: 2, 37: 4, 38: 4, 39: 4, 40: 4, 41: 4, 42: 5, 43: 5, 44: 5, 45: 5,
#     46: 5, 47: 5, 48: 5, 49: 5, 50: 5, 51: 5, 52: 5, 53: 5, 54: 5, 55: 6, 56: 6,
#     57: 6, 58: 6, 59: 6, 60: 6, 61: 0, 62: 0, 63: 0, 64: 0, 65: 0, 66: 0, 67: 0,
#     68: 0, 69: 0, 70: 0, 71: 0, 72: 0, 73: 7, 74: 7, 75: 7, 76: 7, 77: 7, 78: 7,
#     79: 2, 80: 2, 81: 0, 82: 0, 83: 0
# }
# matrix['supercategory_id'] = matrix['item_category_id'].map(supercat_map)


# matrix


# def add_city_codes(matrix, shops):
#     shops.loc[
#         shops.shop_name == 'Сергиев Посад ТЦ "7Я"', "shop_name"
#     ] = 'СергиевПосад ТЦ "7Я"'
#     shops["city"] = shops["shop_name"].str.split(" ").map(lambda x: x[0])
#     shops.loc[shops.city == "!Якутск", "city"] = "Якутск"
#     shops["city_code"] = shops["city"].factorize()[0]
#     shop_labels = shops[["shop_id", "city_code"]]
#     matrix = matrix.merge(shop_labels, on='shop_id', how='left')
#     return matrix

# matrix = add_city_codes(matrix, shops)
# del(shops)



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import gc

# def cluster_feature(matrix, target_feature, clust_feature, level_feature,
#                     n_components=4, n_clusters=5, aggfunc="mean", exclude=None):
#     # Chọn khoảng thời gian để phân tích: từ tháng 21 đến 32
#     start_month = 20
#     end_month = 32
#     pt = matrix.query(f"date_block_num>{start_month} & date_block_num<={end_month}")

#     # Phân cụm từ tháng 20 đến 32 là để:
#       # - Bỏ qua dữ liệu lỗi thời và nhiễu đầu kỳ.
#       # - Tập trung vào xu hướng gần thời điểm dự đoán.
#       # - Cải thiện chất lượng phân cụm và hiệu quả mô hình.

#     # Loại bỏ các đối tượng (shops hoặc item categories) nếu cần
#     if exclude is not None:
#         pt = matrix[~matrix[clust_feature].isin(exclude)]

#     # Tạo pivot table: mỗi dòng là một shop/item category, mỗi cột là một tháng,
#     # giá trị là doanh số trung bình hoặc tổng (tuỳ theo aggfunc)
#     pt = pt.pivot_table(values=target_feature,
#                         columns=clust_feature,
#                         index=level_feature,
#                         fill_value=0,
#                         aggfunc=aggfunc)

#     pt = pt.transpose()  # Đưa các đối tượng thành dòng để thực hiện PCA

#     # PCA để giảm chiều dữ liệu
#     pca = PCA(n_components=10)
#     components = pca.fit_transform(pt)
#     components = pd.DataFrame(components)

#     # Vẽ biểu đồ tỉ lệ phương sai được giải thích bởi từng thành phần chính
#     sns.set_theme()
#     features = list(range(pca.n_components_))
#     fig = plt.figure(figsize=(10,4))
#     ax = fig.add_subplot(121)
#     sns.barplot(x=features, y=pca.explained_variance_ratio_, ax=ax)
#     plt.title("Variance by PCA components")
#     plt.xlabel("component")
#     plt.ylabel("explained variance")
#     plt.xticks(features)

#     # Đánh giá chất lượng phân cụm với silhouette score
#     scorelist = []
#     nrange = range(2, 10)  # Thử từ 2 đến 9 cụm
#     for n in nrange:
#         clusterer = AgglomerativeClustering(n_clusters=n)
#         labels = clusterer.fit_predict(components)
#         silscore = silhouette_score(pt, labels)
#         scorelist.append(silscore)

#     # Vẽ biểu đồ silhouette score theo số cụm
#     ax = fig.add_subplot(122)
#     sns.lineplot(x=nrange, y=scorelist, ax=ax)
#     plt.title("Clustering quality by number of clusters")
#     plt.xlabel("n clusters")
#     plt.ylabel("silhouette score")

#     # Phân cụm thực tế với số cụm được chọn
#     pca = PCA(n_components=n_components)
#     components = pca.fit_transform(pt)
#     components = pd.DataFrame(components)
#     clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage="average")
#     labels = clusterer.fit_predict(components)

#     # Vẽ biểu đồ phân cụm theo 2 thành phần PCA đầu tiên
#     x = components[0]
#     y = components[1]
#     fig = plt.figure(figsize=(10, 4))
#     ax = fig.add_subplot(111)
#     sns.scatterplot(x=x, y=y, hue=labels,
#                     palette=sns.color_palette("hls", n_clusters), ax=ax)
#     plt.title("Items by cluster")
#     plt.xlabel("component 1 score")
#     plt.ylabel("component 2 score")

#     # Gắn nhãn (annotate) các đối tượng để dễ tra cứu
#     for i, txt in enumerate(pt.index.to_list()):
#         ax.annotate(str(txt), (x[i], y[i]))

#     # Trả về dictionary chứa id của shop/item category → số cụm tương ứng
#     groups = {}
#     for i, s in enumerate(pt.index):
#         groups[s] = labels[i]

#     return groups



# # Dựa trên doanh số trung bình mỗi tháng, phân cụm các loại sản phẩm (item_category_id)
# category_group_dict = cluster_feature(
#     matrix,
#     'item_cnt_month',         # Biến mục tiêu: số lượng bán mỗi tháng
#     'item_category_id',       # Feature dùng để phân cụm: loại sản phẩm
#     'date_block_num',         # Mỗi loại sản phẩm được biểu diễn bằng chuỗi doanh số qua các tháng
#     n_components=2,           # Số lượng thành phần PCA để trực quan hóa (vì ít nhóm)
#     n_clusters=4,             # Số cụm muốn chia (đã chọn sau khi đánh giá silhouette score)
#     aggfunc="mean",           # Sử dụng trung bình số lượng bán mỗi tháng để biểu diễn đặc trưng
#     exclude=[]                # Không loại trừ nhóm nào
# )

# # Gán nhãn cụm vừa phân cụm vào matrix
# matrix['category_cluster'] = matrix['item_category_id'].map(category_group_dict)



# # Phân cụm cửa hàng dựa trên tổng doanh số theo từng loại sản phẩm
# shop_group_dict = cluster_feature(
#     matrix,
#     'item_cnt_month',         # Số lượng bán theo tháng
#     'shop_id',                # Đối tượng cần phân cụm là cửa hàng
#     'item_category_id',       # Dùng phân phối bán hàng theo từng loại để làm đặc trưng
#     n_components=4,
#     n_clusters=4,
#     aggfunc="mean",           # Doanh số trung bình theo loại sản phẩm
#     exclude=[36]              # Loại bỏ shop 36 vì có dữ liệu quá ít
# )

# # Shop 36 có dữ liệu chỉ 1 tháng → gán nó chung cụm với shop 37 (gần nhất về hành vi)
# shop_group_dict[36] = shop_group_dict[37]

# # Gán nhãn cụm vào matrix
# matrix['shop_cluster'] = matrix['shop_id'].map(shop_group_dict)



# # Dọn bộ nhớ RAM sau khi xử lý cụm
# gc.collect()



# matrix


# # Hàm tạo đặc trưng đếm số item_id duy nhất trong mỗi nhóm (có thể giới hạn theo điều kiện)
# def uniques(matrix, groupers, name, limitation=None):
#     if limitation is not None:
#         # Nếu có điều kiện lọc (ví dụ: chỉ đếm các sản phẩm mới)
#         s = (
#             matrix.query(limitation)
#             .groupby(groupers)
#             .item_id.nunique()
#             .rename(name)
#             .reset_index()
#         )
#     else:
#         # Đếm item_id duy nhất trong mỗi nhóm (không có điều kiện lọc)
#         s = matrix.groupby(groupers).item_id.nunique().rename(name).reset_index()

#     # Merge kết quả vào ma trận gốc
#     matrix = matrix.merge(s, on=groupers, how="left")
#     matrix[name] = matrix[name].fillna(0)
#     return matrix



# # Tổng số mặt hàng duy nhất được bán trong mỗi tháng
# matrix = uniques(matrix, ["date_block_num"], "unique_items_month")

# # Số mặt hàng duy nhất thuộc mỗi nhóm tên trong từng tháng
# matrix = uniques(matrix, ["date_block_num", "item_name_group"], "name_group_unique_month")

# # Số mặt hàng duy nhất thuộc mỗi nhóm tên và loại sản phẩm trong từng tháng
# matrix = uniques(
#     matrix,
#     ["date_block_num", "item_category_id", "item_name_group"],
#     "name_group_cat_unique_month",
# )

# # Số mặt hàng mới duy nhất thuộc mỗi nhóm tên trong từng tháng
# matrix = uniques(
#     matrix,
#     ["date_block_num", "item_name_group"],
#     "name_group_new_unique_month",
#     limitation="new_item==True",  # Chỉ tính các mặt hàng mới
# )

# # Số mặt hàng mới duy nhất thuộc mỗi nhóm tên và loại sản phẩm trong từng tháng
# matrix = uniques(
#     matrix,
#     ["date_block_num", "item_category_id", "item_name_group"],
#     "name_group_new_cat_unique_month",
#     limitation="new_item==True",
# )

# # Số mặt hàng duy nhất có từ đầu tên giống nhau trong mỗi tháng
# matrix = uniques(
#     matrix, ["date_block_num", "artist_name_or_first_word"], "first_word_unique_month"
# )

# # Số mặt hàng duy nhất có từ đầu tên giống nhau trong mỗi loại sản phẩm mỗi tháng
# matrix = uniques(
#     matrix,
#     ["date_block_num", "item_category_id", "artist_name_or_first_word"],
#     "first_word_cat_unique_month",
# )

# # Số mặt hàng mới duy nhất có từ đầu tên giống nhau trong mỗi tháng
# matrix = uniques(
#     matrix,
#     ["date_block_num", "artist_name_or_first_word"],
#     "first_word_new_unique_month",
#     limitation="new_item==True",
# )

# # Số mặt hàng mới duy nhất có từ đầu tên giống nhau trong mỗi loại sản phẩm mỗi tháng
# matrix = uniques(
#     matrix,
#     ["date_block_num", "item_category_id", "artist_name_or_first_word"],
#     "first_word_new_cat_unique_month",
#     limitation="new_item==True",
# )

# # Số mặt hàng duy nhất thuộc mỗi loại sản phẩm trong mỗi tháng
# matrix = uniques(matrix, ["date_block_num", "item_category_id"], "unique_items_cat")

# # Số mặt hàng mới duy nhất thuộc mỗi loại sản phẩm trong mỗi tháng
# matrix = uniques(
#     matrix,
#     ["date_block_num", "item_category_id"],
#     "new_items_cat",
#     limitation="new_item==True",
# )

# # Số mặt hàng mới duy nhất trong toàn bộ tháng (không phân loại)
# matrix = uniques(matrix, ["date_block_num"], "new_items_month", limitation="new_item==True")

# # Tỷ lệ số mặt hàng duy nhất theo từng loại sản phẩm so với toàn bộ mặt hàng trong tháng
# matrix["cat_items_proportion"] = matrix["unique_items_cat"] / matrix["unique_items_month"]

# # Tỷ lệ số mặt hàng mới trong mỗi nhóm tên so với tổng số mặt hàng trong nhóm đó
# matrix["name_group_new_proportion_month"] = (
#     matrix["name_group_new_unique_month"] / matrix["name_group_unique_month"]
# )

# # Xoá bớt các cột trung gian không cần thiết sau khi đã dùng để tạo đặc trưng
# matrix = matrix.drop(columns=["unique_items_month", "name_group_unique_month"])


# matrix


# def add_pct_change(
#     matrix,  # DataFrame chứa dữ liệu ban đầu
#     group_feats,  # Danh sách các tính năng nhóm (ví dụ: sản phẩm, danh mục sản phẩm)
#     target="item_cnt_month",  # Tính năng mục tiêu cần tính thay đổi phần trăm (số lượng sản phẩm bán được trong tháng)
#     aggfunc="mean",  # Hàm tổng hợp để tính giá trị trung bình
#     periods=1,  # Số kỳ (tháng) để tính sự thay đổi
#     lag=1,  # Số tháng để trễ tính toán (sử dụng để tính thêm các tính năng lag)
#     clip_value=None,  # Nếu có, giới hạn giá trị thay đổi tỷ lệ phần trăm trong khoảng này
# ):
#     # Kiểm tra và đảm bảo periods và group_feats là dạng danh sách
#     periods = list_if_not(periods, int)
#     group_feats = list_if_not(group_feats)

#     # Tạo pivot_table để tính giá trị trung bình của target (item_cnt_month) cho các nhóm
#     group_feats_full = ["date_block_num"] + group_feats
#     dat = matrix.pivot_table(
#         index=group_feats + ["date_block_num"],  # Nhóm theo các tính năng và tháng
#         values=target,  # Giá trị cần tính toán là số lượng sản phẩm bán được
#         aggfunc=aggfunc,  # Hàm tổng hợp (tính trung bình)
#         fill_value=0,  # Thay thế giá trị thiếu bằng 0
#         dropna=False,  # Không loại bỏ các giá trị thiếu
#     ).astype("float32")  # Đảm bảo sử dụng kiểu dữ liệu float32 để tiết kiệm bộ nhớ

#     # Đảm bảo giá trị của target là NaN cho các tháng trước lần xuất hiện đầu tiên của mỗi nhóm
#     for g in group_feats:
#         firsts = matrix.groupby(g).date_block_num.min().rename("firsts")  # Tìm tháng đầu tiên của từng nhóm
#         dat = dat.merge(firsts, left_on=g, right_index=True, how="left")  # Kết hợp với pivot_table
#         dat.loc[dat.index.get_level_values("date_block_num") < dat["firsts"], target] = float(
#             "nan"
#         )  # Gán NaN cho các giá trị trước tháng đầu tiên
#         del dat["firsts"]  # Xóa cột "firsts" sau khi sử dụng

#     # Tính toán sự thay đổi phần trăm (percentage change) cho các kỳ
#     for period in periods:
#         # Tạo tên tính năng mới cho sự thay đổi theo tỷ lệ phần trăm
#         feat_name = "_".join(
#             group_feats + [target] + [aggfunc] + ["delta"] + [str(period)] + [f"lag_{lag}"]
#         )
#         print(f"Adding feature {feat_name}")  # In ra tên tính năng mới
#         # Tính toán sự thay đổi phần trăm
#         dat = (
#             dat.groupby(group_feats)[target]
#             .transform(lambda x: x.pct_change(periods=period, fill_method="pad"))
#             .rename(feat_name)  # Đổi tên cột thành tính năng mới
#         )
#         # Giới hạn giá trị thay đổi phần trăm nếu clip_value được chỉ định
#         if clip_value is not None:
#             dat = dat.clip(lower=-clip_value, upper=clip_value)

#     # Reset lại chỉ số và tăng tháng cho tính năng lag
#     dat = dat.reset_index()
#     dat["date_block_num"] += lag  # Tăng số tháng để tính toán lag

#     # Thêm tính năng mới vào matrix và giảm bộ nhớ cho cột tính năng
#     matrix = matrix.merge(dat, on=["date_block_num"] + group_feats, how="left")

#     return matrix  # Trả về matrix đã được cập nhật với tính năng mới





# # Áp dụng hàm add_pct_change cho các nhóm khác nhau
# matrix = add_pct_change(matrix, ["item_id"], "item_cnt_month", clip_value=3)  # Dự đoán thay đổi phần trăm theo sản phẩm
# matrix = add_pct_change(matrix, ["item_category_id"], "item_cnt_month", clip_value=3)  # Dự đoán thay đổi phần trăm theo danh mục sản phẩm
# matrix = add_pct_change(matrix, ["item_name_group"], "item_cnt_month", clip_value=3)  # Dự đoán thay đổi phần trăm theo nhóm tên sản phẩm

# # Dự đoán thay đổi phần trăm với lag 12 tháng để nắm bắt xu hướng theo mùa vụ
# matrix = add_pct_change(matrix, ["item_category_id"], "item_cnt_month", lag=12, clip_value=3,)
# gc.collect()  # Dọn dẹp bộ nhớ



# def add_rolling_stats(
#     matrix,  # DataFrame chứa dữ liệu ban đầu
#     features,  # Danh sách các tính năng nhóm để tính toán thống kê (ví dụ: item_id, shop_id)
#     window=12,  # Kích thước cửa sổ (số tháng tính toán)
#     kind="rolling",  # Loại cửa sổ (rolling, expanding, ewm)
#     argfeat="item_cnt_month",  # Tính năng cần tính toán (số lượng sản phẩm bán được mỗi tháng)
#     aggfunc="mean",  # Hàm tổng hợp mặc định (tính trung bình)
#     rolling_aggfunc="mean",  # Hàm tổng hợp cho cửa sổ rolling (tính trung bình)
#     dtype="float16",  # Kiểu dữ liệu của tính năng (để tiết kiệm bộ nhớ)
#     reshape_source=True,  # Nếu True, tạo lại bảng pivot để xử lý dữ liệu
#     lag_offset=0,  # Số tháng trễ cho tính năng (để tạo tính năng lag)
# ):
#     def rolling_stat(
#         matrix,
#         source,  # Dữ liệu gốc đã được chuẩn bị
#         feats,  # Các tính năng nhóm để tính toán thống kê
#         feat_name,  # Tên tính năng mới sẽ được tạo
#         window=12,  # Kích thước cửa sổ
#         argfeat="item_cnt_month",  # Tính năng cần tính toán
#         aggfunc="mean",  # Hàm tổng hợp
#         dtype=dtype,  # Kiểu dữ liệu
#         lag_offset=0,  # Số tháng trễ
#     ):
#         store = []  # Danh sách lưu các tính năng tính toán
#         # Tính toán cho mỗi tháng trong khoảng thời gian đã chỉ định
#         for i in range(2 + lag_offset, 35 + lag_offset):
#             if len(feats) > 0:
#                 # Nếu có nhóm tính năng, tính toán trung bình theo nhóm
#                 mes = (
#                     source[source.date_block_num.isin(range(max([i - window, 0]), i))]
#                     .groupby(feats)[argfeat]
#                     .agg(aggfunc)
#                     .astype(dtype)
#                     .rename(feat_name)
#                     .reset_index()
#                 )
#             else:
#                 # Nếu không có nhóm tính năng, chỉ tính toán tổng hợp cho toàn bộ dữ liệu
#                 mes = {}
#                 mes[feat_name] = (
#                     source.loc[
#                         source.date_block_num.isin(range(max([i - window, 0]), i)), argfeat
#                     ]
#                     .agg(aggfunc)
#                     .astype(dtype)
#                 )
#                 mes = pd.DataFrame(data=mes, index=[i])
#             mes["date_block_num"] = i - lag_offset  # Cập nhật tháng cho tính năng
#             store.append(mes)  # Lưu tính năng vào danh sách
#         store = pd.concat(store)  # Kết hợp tất cả các tính năng lại
#         matrix = matrix.merge(store, on=feats + ["date_block_num"], how="left")  # Kết hợp vào dữ liệu gốc
#         return matrix

#     # Xử lý dữ liệu nếu reshape_source == True hoặc loại cửa sổ là "ewm"
#     if (reshape_source == True) or (kind == "ewm"):
#         source = matrix.pivot_table(
#             index=features + ["date_block_num"],  # Tạo bảng pivot theo các nhóm tính năng và tháng
#             values=argfeat,  # Tính toán giá trị tính toán
#             aggfunc=aggfunc,  # Hàm tổng hợp (tính trung bình)
#             fill_value=0,  # Thay thế giá trị thiếu bằng 0
#             dropna=False,  # Không loại bỏ các giá trị thiếu
#         ).astype(dtype)  # Đảm bảo sử dụng kiểu dữ liệu tiết kiệm bộ nhớ
#         for g in features:
#             firsts = matrix.groupby(g).date_block_num.min().rename("firsts")  # Lấy tháng đầu tiên của nhóm
#             source = source.merge(firsts, left_on=g, right_index=True, how="left")
#             # Đặt giá trị NaN cho các tháng trước khi xuất hiện sản phẩm lần đầu
#             source.loc[
#                 source.index.get_level_values("date_block_num") < source["firsts"], argfeat
#             ] = float("nan")
#             del source["firsts"]  # Xóa cột "firsts" sau khi sử dụng
#         source = source.reset_index()  # Đặt lại chỉ số cho bảng
#     else:
#         source = matrix  # Nếu không cần tạo lại bảng pivot, sử dụng dữ liệu gốc

#     # Tạo tính năng cho cửa sổ rolling
#     if kind == "rolling":
#         feat_name = (
#             f"{'_'.join(features)}_{argfeat}_{aggfunc}_rolling_{rolling_aggfunc}_win_{window}"
#         )
#         print(f'Creating feature "{feat_name}"')  # In ra tên tính năng mới
#         return rolling_stat(
#             matrix,
#             source,
#             features,
#             feat_name,
#             window=window,
#             argfeat=argfeat,
#             aggfunc=rolling_aggfunc,
#             dtype=dtype,
#             lag_offset=lag_offset,
#         )
#     # Tạo tính năng cho cửa sổ expanding
#     elif kind == "expanding":
#         feat_name = f"{'_'.join(features)}_{argfeat}_{aggfunc}_expanding_{rolling_aggfunc}"
#         print(f'Creating feature "{feat_name}"')
#         return rolling_stat(
#             matrix,
#             source,
#             features,
#             feat_name,
#             window=100,  # Cửa sổ mở rộng sẽ tính toàn bộ dữ liệu từ đầu
#             argfeat=argfeat,
#             aggfunc=aggfunc,
#             dtype=dtype,
#             lag_offset=lag_offset,
#         )
#     # Tạo tính năng cho cửa sổ ewm
#     elif kind == "ewm":
#         feat_name = f"{'_'.join(features)}_{argfeat}_{aggfunc}_ewm_hl_{window}"
#         print(f'Creating feature "{feat_name}"')
#         source[feat_name] = (
#             source.groupby(features)[argfeat]
#             .ewm(halflife=window, min_periods=1)
#             .agg(rolling_aggfunc)
#             .to_numpy(dtype=dtype)
#         )
#         del source[argfeat]  # Xóa cột tính năng gốc sau khi tính toán
#         source["date_block_num"] += 1 - lag_offset  # Cập nhật tháng
#         return matrix.merge(source, on=["date_block_num"] + features, how="left")  # Kết hợp với dữ liệu gốc




# # Trung bình động theo shop, nghệ sĩ (hoặc từ đầu tiên của tên), danh mục, độ tuổi sản phẩm
# matrix = add_rolling_stats(
#     matrix, ["shop_id", "artist_name_or_first_word", "item_category_id", "item_age"],
#     window=12, reshape_source=False
# )

# # Trung bình mở rộng theo shop, nghệ sĩ, danh mục, sản phẩm mới
# matrix = add_rolling_stats(
#     matrix, ["shop_id", "artist_name_or_first_word", "item_category_id", "new_item"],
#     kind="expanding", reshape_source=False
# )

# # Trung bình mở rộng theo shop, nghệ sĩ, sản phẩm mới
# matrix = add_rolling_stats(
#     matrix, ["shop_id", "artist_name_or_first_word", "new_item"],
#     kind="expanding", reshape_source=False
# )

# # Trung bình động theo shop và cụm danh mục
# matrix = add_rolling_stats(matrix, ["shop_id", "category_cluster"], window=12)

# # Trung bình mở rộng theo shop, danh mục sản phẩm, độ tuổi
# matrix = add_rolling_stats(
#     matrix, ["shop_id", "item_category_id", "item_age"],
#     kind="expanding", reshape_source=False
# )

# # Trung bình động theo shop, danh mục sản phẩm, độ tuổi
# matrix = add_rolling_stats(
#     matrix, ["shop_id", "item_category_id", "item_age"],
#     window=12, reshape_source=False
# )

# # Trung bình mũ (ewm) theo shop và danh mục sản phẩm (cửa sổ = 1, nhấn mạnh dữ liệu gần nhất)
# matrix = add_rolling_stats(
#     matrix, ["shop_id", "item_category_id"],
#     kind="ewm", window=1
# )

# # Trung bình mở rộng theo shop, danh mục sản phẩm, sản phẩm mới
# matrix = add_rolling_stats(
#     matrix, ["shop_id", "item_category_id", "new_item"],
#     kind="expanding", reshape_source=False
# )

# # Trung bình động theo shop, danh mục sản phẩm, sản phẩm mới
# matrix = add_rolling_stats(
#     matrix, ["shop_id", "item_category_id", "new_item"],
#     window=12, reshape_source=False
# )

# # Trung bình động theo từng shop
# matrix = add_rolling_stats(matrix, ["shop_id"], window=12)

# # Trung bình mũ theo shop và sản phẩm
# matrix = add_rolling_stats(matrix, ["shop_id", "item_id"], kind="ewm", window=1)

# # Trung bình động theo shop và sản phẩm
# matrix = add_rolling_stats(matrix, ["shop_id", "item_id"], window=12)

# # Trung bình động theo shop, nhóm tên sản phẩm, danh mục, sản phẩm mới
# matrix = add_rolling_stats(
#     matrix, ["shop_id", "item_name_group", "item_category_id", "new_item"],
#     window=12, reshape_source=False
# )

# # Trung bình mở rộng theo shop, nhóm tên sản phẩm và sản phẩm mới
# matrix = add_rolling_stats(
#     matrix, ["shop_id", "item_name_group", "new_item"],
#     kind="expanding", reshape_source=False
# )

# # Trung bình động theo shop, nhóm siêu danh mục và sản phẩm mới
# matrix = add_rolling_stats(
#     matrix, ["shop_id", "supercategory_id", "new_item"],
#     window=12, reshape_source=False
# )

# # Trung bình mũ theo cụm cửa hàng và sản phẩm
# matrix = add_rolling_stats(matrix, ["shop_cluster", "item_id"], kind="ewm", window=1)

# # Trung bình mở rộng theo cụm shop, danh mục sản phẩm, độ tuổi
# matrix = add_rolling_stats(
#     matrix, ["shop_cluster", "item_category_id", "item_age"],
#     kind="expanding", reshape_source=False
# )

# # Trung bình động theo cụm shop, nhóm tên sản phẩm, sản phẩm mới
# matrix = add_rolling_stats(
#     matrix, ["shop_cluster", "item_name_group", "new_item"],
#     window=12, reshape_source=False
# )

# # Trung bình mở rộng theo cụm danh mục và độ tuổi sản phẩm
# matrix = add_rolling_stats(
#     matrix, ["category_cluster", "item_age"],
#     kind="expanding", reshape_source=False
# )

# # Trung bình mở rộng theo cụm danh mục và sản phẩm mới
# matrix = add_rolling_stats(
#     matrix, ["category_cluster", "new_item"],
#     kind="expanding", reshape_source=False
# )

# # Trung bình động theo từng sản phẩm
# matrix = add_rolling_stats(matrix, ["item_id"], window=12)

# # Trung bình động theo nghệ sĩ hoặc từ đầu tiên của tên sản phẩm
# matrix = add_rolling_stats(matrix, ["artist_name_or_first_word"], window=12)

# # Trung bình mũ theo nghệ sĩ hoặc từ đầu tiên của tên sản phẩm
# matrix = add_rolling_stats(matrix, ["artist_name_or_first_word"], kind="ewm", window=1)

# # Trung bình động theo nghệ sĩ và độ tuổi sản phẩm
# matrix = add_rolling_stats(
#     matrix, ["artist_name_or_first_word", "item_age"],
#     window=12, reshape_source=False
# )

# # Trung bình động theo nghệ sĩ, danh mục, độ tuổi sản phẩm
# matrix = add_rolling_stats(
#     matrix, ["artist_name_or_first_word", "item_category_id", "item_age"],
#     window=12, reshape_source=False
# )

# # Trung bình mở rộng theo nghệ sĩ và sản phẩm mới
# matrix = add_rolling_stats(
#     matrix, ["artist_name_or_first_word", "new_item"],
#     kind="expanding", reshape_source=False
# )

# # Trung bình mở rộng theo danh mục và độ tuổi sản phẩm
# matrix = add_rolling_stats(
#     matrix, ["item_category_id", "item_age"],
#     kind="expanding", reshape_source=False
# )

# # Trung bình động theo từng danh mục sản phẩm
# matrix = add_rolling_stats(matrix, ["item_category_id"], window=12)

# # Trung bình mũ theo từng danh mục sản phẩm
# matrix = add_rolling_stats(matrix, ["item_category_id"], kind="ewm", window=1)

# # Trung bình mở rộng theo danh mục và sản phẩm mới
# matrix = add_rolling_stats(
#     matrix, ["item_category_id", "new_item"],
#     kind="expanding", reshape_source=False
# )

# # Trung bình động theo nhóm tên sản phẩm và độ tuổi
# matrix = add_rolling_stats(
#     matrix, ["item_name_group", "item_age"],
#     window=12, reshape_source=False
# )

# # Trung bình mũ theo nhóm tên sản phẩm
# matrix = add_rolling_stats(matrix, ["item_name_group"], kind="ewm", window=1)

# # Trung bình động theo nhóm tên sản phẩm
# matrix = add_rolling_stats(matrix, ["item_name_group"], window=12)

# # Trung bình động theo nền tảng (platform)
# matrix = add_rolling_stats(matrix, ["platform_id"], window=12)

# # Trung bình mũ theo nền tảng
# matrix = add_rolling_stats(matrix, ["platform_id"], kind="ewm", window=1)

# # Dọn bộ nhớ tạm
# gc.collect()




# # Tính tổng doanh số 12 tháng gần nhất theo từng shop và item (rolling window)
# matrix = add_rolling_stats(
#     matrix,
#     ["shop_id", "item_id"],
#     aggfunc="sum",                # Tổng doanh số mỗi tháng
#     rolling_aggfunc="sum",        # Tổng 12 tháng gần nhất
#     kind="rolling",               # Dùng rolling window
#     window=12,
#     reshape_source=False,
# )

# # Tính tổng doanh số cộng dồn theo item (expanding window)
# matrix = add_rolling_stats(
#     matrix,
#     ["item_id"],
#     aggfunc="sum",
#     rolling_aggfunc="sum",
#     kind="expanding",            # Từ tháng đầu tiên đến hiện tại
#     reshape_source=False,
# )

# # === TÍNH DOANH SỐ TRUNG BÌNH MỖI NGÀY (TRÊN CƠ SỞ TỔNG DOANH SỐ VÀ SỐ NGÀY BÁN) ===

# # Gán số ngày trong 1 năm (365 ngày)
# matrix["1year"] = 365

# # Doanh số trung bình mỗi ngày (expanding): Tổng doanh số theo item chia cho số ngày bán hàng
# matrix["item_id_day_mean_expanding"] = matrix[
#     "item_id_item_cnt_month_sum_expanding_sum"
# ] / matrix[["first_item_sale_days"]].min(axis=1)  # Lấy số ngày bán đầu tiên để tránh lỗi

# # Doanh số trung bình mỗi ngày (rolling): Tổng doanh số rolling 12 tháng gần nhất theo shop & item chia cho số ngày tối thiểu
# matrix["shop_id_item_id_day_mean_win_12"] = matrix[
#     "shop_id_item_id_item_cnt_month_sum_rolling_sum_win_12"
# ] / matrix[["first_item_sale_days", "shop_open_days", "1year"]].min(axis=1)

# # Gán NaN cho các mặt hàng mới vì chưa có dữ liệu để tính trung bình
# matrix.loc[matrix.new_item == True, "item_id_day_mean_expanding"] = float("nan")

# # Xóa cột trung gian sau khi đã dùng xong
# matrix = matrix.drop(columns=["1year", "item_id_item_cnt_month_sum_expanding_sum"])

# # === TÍNH DOANH THU TRUNG BÌNH THEO TÊN NHÓM SẢN PHẨM VÀ SHOP TRONG 12 THÁNG ===

# # Rolling mean 12 tháng doanh thu theo shop và nhóm tên sản phẩm
# matrix = add_rolling_stats(
#     matrix,
#     ["shop_id", "item_name_group"],
#     window=12,
#     argfeat="item_revenue_month",  # Đặc trưng đầu vào là doanh thu
#     dtype="float32",
# )

# # === TÍNH SỐ LƯỢNG SẢN PHẨM MỚI THEO DANH MỤC VÀ TÊN NHÓM SẢN PHẨM ===

# # Tính rolling mean 12 tháng cho số lượng sản phẩm mới theo category
# matrix = add_rolling_stats(
#     matrix,
#     ["item_category_id"],
#     argfeat="new_items_cat",       # Tổng sản phẩm mới trong danh mục mỗi tháng
#     window=12,
#     reshape_source=True,
#     lag_offset=1,                  # Trễ 1 tháng để tránh leakage
# )

# # Tính rolling mean 12 tháng cho số lượng nhóm tên mới (item_name_group)
# matrix = add_rolling_stats(
#     matrix,
#     ["item_name_group"],
#     argfeat="name_group_new_unique_month",  # Số nhóm tên mới xuất hiện mỗi tháng
#     window=12,
#     reshape_source=True,
#     lag_offset=1,
# )

# # === TÍNH TỶ LỆ SẢN PHẨM MỚI TRÊN DANH MỤC TRUNG BÌNH ===

# # Tỷ lệ sản phẩm mới trong tháng hiện tại so với trung bình của 12 tháng gần nhất theo từng danh mục
# matrix["new_items_cat_1_12_ratio"] = (
#     matrix["new_items_cat"]
#     / matrix["item_category_id_new_items_cat_mean_rolling_mean_win_12"]
# )

# gc.collect()



# matrix.to_pickle(path+"matrixcheckpoint.pkl")


# matrix = pd.read_pickle("/kaggle/input/checkpoint1/matrixcheckpoint.pkl")


# # HÀM TẠO CÁC ĐẶC TRƯNG LAG (ĐỘ TRỄ) CHO MỖI shop-item
# def simple_lag_feature(matrix, lag_feature, lags):
#     for lag in lags:
#         newname = lag_feature + f"_lag_{lag}"  # Đặt tên cột mới theo dạng: <tên_cũ>_lag_<độ_trễ>
#         print(f"Adding feature {newname}")

#         # Tạo dataframe chứa cột mục tiêu cần lag và thông tin khóa ghép
#         targetseries = matrix.loc[:, ["date_block_num", "item_id", "shop_id"] + [lag_feature]]

#         # Dịch thời gian đi một số tháng tương ứng với độ trễ
#         targetseries["date_block_num"] += lag

#         # Đổi tên cột đặc trưng mục tiêu thành tên mới (ví dụ: item_cnt_month_lag_1)
#         targetseries = targetseries.rename(columns={lag_feature: newname})

#         # Nối đặc trưng lag này vào `matrix` dựa trên các khóa: tháng, item, shop
#         matrix = matrix.merge(
#             targetseries, on=["date_block_num", "item_id", "shop_id"], how="left"
#         )

#         # Xử lý các giá trị thiếu (NaN) do không có dữ liệu các tháng trước:
#         # → nếu item_age và shop_age đủ lớn (>= độ trễ) nhưng vẫn thiếu → gán 0
#         matrix.loc[
#             (matrix.item_age >= lag) & (matrix.shop_age >= lag) & (matrix[newname].isna()),
#             newname,
#         ] = 0

#     return matrix

# # TẠO ĐẶC TRƯNG LAG CHO item_cnt_month (số lượng bán mỗi tháng) VỚI CÁC ĐỘ TRỄ: 1, 2, 3 THÁNG
# matrix = simple_lag_feature(matrix, 'item_cnt_month', lags=[1, 2, 3])

# # TẠO ĐẶC TRƯNG LAG CHO item_cnt_day_avg (trung bình bán mỗi ngày) VỚI CÁC ĐỘ TRỄ: 1, 2, 3 THÁNG
# matrix = simple_lag_feature(matrix, 'item_cnt_day_avg', lags=[1, 2, 3])

# # TẠO ĐẶC TRƯNG LAG CHO item_revenue_month (doanh thu sản phẩm mỗi tháng) VỚI ĐỘ TRỄ: 1 THÁNG
# matrix = simple_lag_feature(matrix, 'item_revenue_month', lags=[1])

# # Giải phóng bộ nhớ sau khi tạo xong các đặc trưng
# gc.collect()

# # In ra thông báo khi hoàn tất
# print("Lag features created")



# # Hàm tạo và áp dụng đặc trưng mean encoding hoặc sum encoding theo thời gian (lag)
# def create_apply_ME(matrix, grouping_fields, lags=[1], target="item_cnt_day_avg", aggfunc="mean"):
#     # Đảm bảo grouping_fields luôn là danh sách
#     grouping_fields = list_if_not(grouping_fields)

#     for lag in lags:
#         # Tạo tên cột mới thể hiện đặc trưng được tạo, ví dụ: item_id_item_cnt_day_avg_mean_lag_1
#         newname = "_".join(grouping_fields + [target] + [aggfunc] + [f"lag_{lag}"])
#         print(f"Adding feature {newname}")

#         # Tính trung bình hoặc tổng của target theo từng nhóm trong mỗi tháng
#         me_series = (
#             matrix.groupby(["date_block_num"] + grouping_fields)[target]
#             .agg(aggfunc)
#             .rename(newname)
#             .reset_index()
#         )

#         # Dịch date_block_num về tương lai bằng độ trễ (lag) để sử dụng giá trị quá khứ cho tháng hiện tại
#         me_series["date_block_num"] += lag

#         # Merge đặc trưng mới vào ma trận dữ liệu chính
#         matrix = matrix.merge(me_series, on=["date_block_num"] + grouping_fields, how="left")

#         # Giải phóng bộ nhớ
#         del me_series

#         # Điền các giá trị thiếu còn lại bằng 0 tạm thời
#         matrix[newname] = matrix[newname].fillna(0)

#         # Gán NaN cho các dòng không đủ dữ liệu quá khứ (nhóm xuất hiện sau lag tháng)
#         for g in grouping_fields:
#             # Tìm tháng đầu tiên xuất hiện của mỗi nhóm
#             firsts = matrix.groupby(g).date_block_num.min().rename("firsts")
#             matrix = matrix.merge(firsts, left_on=g, right_index=True, how="left")
#             # Gán NaN cho các dòng mà tháng hiện tại < tháng đầu tiên xuất hiện + lag
#             matrix.loc[matrix["date_block_num"] < (matrix["firsts"] + (lag)), newname] = float("nan")
#             del matrix["firsts"]

#         # Tối ưu bộ nhớ cho cột mới
#         matrix[newname] = reduce_mem_usage(matrix[newname])

#     return matrix

# # Áp dụng mean encoding/sum encoding cho nhiều tổ hợp biến phân loại khác nhau

# # Mã hóa trung bình doanh số theo nhóm tên sản phẩm (item_name_group) - giúp học được xu hướng chung của từng nhóm sản phẩm
# matrix = create_apply_ME(matrix, ["item_name_group"], target="item_cnt_month")

# # Mã hóa tổng doanh số theo nhóm tên sản phẩm - cho mô hình biết tổng doanh số nhóm có thể quan trọng hơn trung bình
# matrix = create_apply_ME(matrix, ["item_name_group"], target="item_cnt_month", aggfunc="sum")

# # Mã hóa trung bình doanh số theo ID sản phẩm (item_id) - đặc trưng cơ bản về mức độ phổ biến của từng sản phẩm
# matrix = create_apply_ME(matrix, ["item_id"], target="item_cnt_month")

# # Mã hóa trung bình item_cnt_day_avg theo item_id - thể hiện thói quen mua hàng trong ngày của từng sản phẩm
# matrix = create_apply_ME(matrix, ["item_id"])

# # Mã hóa trung bình item_cnt_day_avg theo nền tảng (platform_id) - mỗi nền tảng có xu hướng bán hàng khác nhau
# matrix = create_apply_ME(matrix, ["platform_id"])

# # Mã hóa trung bình item_cnt_day_avg theo nhóm tên sản phẩm - tương tự dòng trên nhưng ở cấp nhóm sản phẩm
# matrix = create_apply_ME(matrix, ["item_name_group"])

# # Mã hóa trung bình doanh số (item_cnt_month) theo nền tảng - hiểu xu hướng tiêu thụ hàng tháng của từng nền tảng
# matrix = create_apply_ME(matrix, ["platform_id"], target="item_cnt_month")

# # Mã hóa trung bình item_cnt_day_avg theo siêu danh mục (supercategory_id) - giúp hiểu các nhóm sản phẩm lớn hơn
# matrix = create_apply_ME(matrix, ["supercategory_id"])

# # Mã hóa doanh số trung bình theo tổ hợp loại sản phẩm và flag sản phẩm mới - rất hữu ích với sản phẩm mới
# matrix = create_apply_ME(matrix, ["item_category_id", "new_item"], target="item_cnt_month")

# # Mã hóa doanh số trung bình theo tổ hợp shop và loại sản phẩm - mỗi shop có thể bán mạnh ở một số ngành hàng
# matrix = create_apply_ME(matrix, ["shop_id", "item_category_id"], target="item_cnt_month")

# # Mã hóa doanh số trung bình theo cụm shop và sản phẩm - cụm shop phản ánh khu vực hoặc thị trường
# matrix = create_apply_ME(matrix, ["shop_cluster", "item_id"], target="item_cnt_month")

# # Mã hóa trung bình item_cnt_day_avg theo cụm shop và sản phẩm - cho thấy tần suất bán hàng theo ngày
# matrix = create_apply_ME(matrix, ["shop_cluster", "item_id"])

# # Mã hóa trung bình item_cnt_day_avg theo mã thành phố và sản phẩm - giúp nhận biết nhu cầu theo khu vực địa lý
# matrix = create_apply_ME(matrix, ["city_code", "item_id"])

# # Mã hóa trung bình item_cnt_day_avg theo mã thành phố và nhóm tên sản phẩm - kết hợp yếu tố địa lý và loại sản phẩm
# matrix = create_apply_ME(matrix, ["city_code", "item_name_group"])

# # Tạo đặc trưng tỷ lệ giữa doanh số trung bình gần đây (lag=1) và doanh số trung bình dài hạn (trung bình 12 tháng)
# matrix["item_id_item_cnt_1_12_ratio"] = (
#     matrix["item_id_item_cnt_month_mean_lag_1"]
#     / matrix["item_id_item_cnt_month_mean_rolling_mean_win_12"]
# )

# # Tạo đặc trưng tỷ lệ giữa doanh số trung bình ngày (tháng trước) với doanh số trung bình năm
# matrix["shop_id_item_id_item_cnt_1_12_ratio"] = (
#     matrix["item_cnt_day_avg_lag_1"] / matrix["shop_id_item_id_day_mean_win_12"]
# )

# gc.collect()  # Dọn dẹp bộ nhớ
# print("Mean encoding features created")



# surplus_columns = [
#     "item_revenue_month",
#     "item_cnt_day_avg",
#     "item_name_group",
#     "artist_name_or_first_word",
#     "item_age",
#     "shop_open_days",
#     "shop_age",
#     "platform_id",
#     "supercategory_id",
#     "city_code",
#     "category_cluster",
#     "shop_cluster",
#     "new_items_cat",
#     "shop_id_item_id_day_mean_win_12",
#     "item_id_item_cnt_month_mean_rolling_mean_win_12",
# ]
# matrix = matrix.drop(columns=surplus_columns)


import re
import warnings

warnings.filterwarnings("ignore", module="sklearn")  # Tắt các cảnh báo từ sklearn

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, f_regression

# def name_token_feats(matrix, items, k=50, item_n_threshold=5, target_month_start=33):
#     # Hàm làm sạch chuỗi tên sản phẩm
#     def name_correction(st):
#         st = re.sub(r"[^\w\s]", "", st)  # Loại bỏ ký tự đặc biệt
#         st = re.sub(r"\s{2,}", " ", st)  # Loại bỏ khoảng trắng dư
#         st = st.lower().strip()         # Đưa về chữ thường và loại bỏ khoảng trắng đầu/cuối
#         return st

#     items["item_name_clean"] = items["item_name"].apply(name_correction)

#     # Tạo ma trận Bag-of-Words từ tên sản phẩm
#     def create_item_id_bow_matrix(items):
#         all_stopwords = stopwords.words("russian") + stopwords.words("english")  # Dừng từ Nga + Anh
#         vectorizer = CountVectorizer(stop_words=all_stopwords)  # Vector hóa văn bản
#         X = vectorizer.fit_transform(items["item_name_clean"])  # Ma trận từ (items × từ)
#         X = pd.DataFrame.sparse.from_spmatrix(X)
#         print(f"{len(vectorizer.vocabulary_)} words found in all items")

#         # Đổi tên các cột dạng word_tên
#         featuremap = {
#             col: "word_" + token
#             for col, token in zip(
#                 range(len(vectorizer.vocabulary_)), vectorizer.get_feature_names_out()
#             )
#         }
#         X = X.rename(columns=featuremap)
#         return X

#     # Tạo ma trận từ
#     items_bow = create_item_id_bow_matrix(items)
#     items_bow = items_bow.clip(0, 1)  # Chuyển ma trận sang nhị phân (có từ hay không)

#     # Chỉ giữ lại những từ xuất hiện ở nhiều hơn item_n_threshold sản phẩm
#     common_word_mask = items_bow.sum(axis=0) > item_n_threshold

#     # Giữ lại những từ xuất hiện trong sản phẩm mới từ tháng target_month_start trở đi
#     target_items = matrix.query(
#         f"date_block_num>={target_month_start} & new_item==True"
#     ).item_id.unique()
#     target_item_mask = items_bow.loc[target_items, :].sum(axis=0) > 1

#     # Chỉ giữ các từ thỏa cả 2 điều kiện
#     items_bow = items_bow.loc[:, common_word_mask & target_item_mask]
#     print(f"{items_bow.shape[1]} words of interest")

#     # Tạo tập huấn luyện để chọn lọc các từ có tính dự đoán cao
#     mxbow = matrix[["date_block_num", "item_id", "item_cnt_month"]].query("date_block_num<34")
#     mxbow = mxbow.merge(items_bow, left_on="item_id", right_index=True, how="left")
#     X = mxbow.drop(columns=["date_block_num", "item_id", "item_cnt_month"])
#     y = mxbow["item_cnt_month"].clip(0, 20)  # Giới hạn số lượng bán để tránh outliers

#     # Chọn ra k từ có tương quan cao nhất với target (item_cnt_month)
#     selektor = SelectKBest(f_regression, k=k)
#     selektor.fit(X, y)
#     tokencols = X.columns[selektor.get_support()]  # Lấy danh sách cột từ được chọn
#     print(f"{k} word features selected")

#     return items_bow[tokencols]  # Trả về ma trận các từ đã chọn

# # Đọc dữ liệu item từ file
# items = pd.read_csv(path2 + "items.csv")

# # Trích xuất đặc trưng từ tên sản phẩm (50 từ có tính dự đoán cao nhất)
# word_frame = name_token_feats(matrix, items, k=50, item_n_threshold=5)

# # Nối đặc trưng word vào ma trận chính thông qua item_id
# matrix = matrix.merge(word_frame, left_on='item_id', right_index=True, how='left')

# # LightGBM không hỗ trợ đặc trưng dạng sparse tốt → chuyển sang dense và ép kiểu int8 để tiết kiệm RAM
# sparsecols = [c for c in matrix.columns if pd.api.types.is_sparse(matrix[c].dtype)]
# matrix[sparsecols] = matrix[sparsecols].sparse.to_dense().astype('int8')

# gc.collect()  # Dọn dẹp bộ nhớ
# matrix.to_pickle(path+"checkpoint_final_0.84.pkl") # Lưu trạng thái ra file để tái sử dụng
# print("All features generated, dataframe saved")  # Thông báo hoàn tất



import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu đã xử lý (dạng .pkl nhanh và tiết kiệm bộ nhớ hơn .csv)
df = pd.read_pickle("/kaggle/input/feature-engineering/checkpoint_final_0.84.pkl")

# Giới hạn giá trị doanh số bán ra từ 0 đến 20
df['item_cnt_month'] = df['item_cnt_month'].clip(0, 20)
df = df.rename(columns={"item_cnt_month": "item_cnt"})
df = df[df != np.inf]


# import warnings
# warnings.filterwarnings("ignore", module="lightgbm")

# import lightgbm as lgbm
# from lightgbm import early_stopping, log_evaluation

# # Định nghĩa hàm huấn luyện mô hình LightGBM với API của sklearn
# def fit_booster(
#     X_train,
#     y_train,
#     X_test=None,
#     y_test=None,
#     test_flag=False,
#     params=None,
#     categoricals=[],
#     dropcols=[],
#     use_early_stopping=True,
# ):
#     # Nếu không truyền tham số thì dùng mặc định
#     if params is None:
#         params = {"learning_rate": 0.1, "subsample_for_bin": 300000, "n_estimators": 50}

#     X_train = X_train.drop(columns=dropcols)
    
#     # Tập dùng để đánh giá trong huấn luyện
#     if test_flag:
#         eval_set = [(X_test, y_test)]
#     else:
#         eval_set = [(X_train, y_train)]

#     booster = lgbm.LGBMRegressor(**params)

#     # Chỉ giữ lại các cột phân loại có tồn tại trong X_train
#     categoricals = [c for c in categoricals if c in X_train.columns]

#     callbacks = []
#     # if use_early_stopping:
#     #     callbacks.append(early_stopping(stopping_rounds=10))
#     callbacks.append(log_evaluation(period=100))  # log mỗi 100 vòng


#     # Huấn luyện mô hình
#     booster.fit(
#         X_train,
#         y_train,
#         eval_set=eval_set,
#         eval_metric=["rmse"],  # Sử dụng RMSE làm thước đo lỗi
#         categorical_feature=categoricals,
#         callbacks=callbacks,
#     )

#     return booster


# Loại bỏ một số cột không cần thiết
dropcols = [
    "item_id",
    "shop_id",
    "item_category_id",
    
]


# import pandas as pd
# !pip install plotly
# import plotly.express as px

# def visualize_tuning_results_random(param_list, loss_list, model_name="XGBoost", negate_score=False):
#     """
#     Trực quan hóa kết quả Random Search với biểu đồ Parallel Coordinates.
    
#     Tham số:
#     - param_list: danh sách dict các tham số đã thử
#     - loss_list: danh sách loss (RMSE) tương ứng với từng cấu hình tham số
#     - model_name: tên mô hình để hiển thị tiêu đề
#     - negate_score: nếu True thì sẽ đổi dấu loss (dùng khi loss là negative score)
#     """

#     # Tạo DataFrame từ danh sách tham số và loss
#     vis_df = pd.DataFrame(param_list)
#     vis_df["Loss"] = [-x if negate_score else x for x in loss_list]

#     # Ánh xạ chuỗi thành số (nếu có)
#     str_mappings = {}
#     for col in vis_df.columns:
#         if vis_df[col].dtype == "object" or isinstance(vis_df[col].iloc[0], str):
#             unique_vals = vis_df[col].unique()
#             mapping = {val: idx for idx, val in enumerate(unique_vals)}
#             str_mappings[col] = mapping
#             vis_df[col] = vis_df[col].map(mapping)

#     # Vẽ biểu đồ
#     fig = px.parallel_coordinates(
#         vis_df,
#         color="Loss",
#         labels={col: col for col in vis_df.columns},
#         title=f"Biểu đồ trực quan hóa Random Search - {model_name}",
#         color_continuous_scale=px.colors.sequential.Viridis_r,
#     )
#     fig.show()

#     # In ánh xạ nếu có dùng chuỗi
#     if str_mappings:
#         print(f"\nÁnh xạ giá trị chuỗi -> số cho {model_name}:")
#         for col, mapping in str_mappings.items():
#             print(f" - {col}: {mapping}")



# from sklearn.model_selection import ParameterSampler
# from sklearn.metrics import mean_squared_error
# import xgboost as xgb
# import numpy as np

# # --- Chia dữ liệu theo tháng ---
# X_train = df[df["date_block_num"] < 33].drop(["item_cnt", "date_block_num"], axis=1)
# y_train = df.loc[df["date_block_num"] < 33, "item_cnt"]
# X_train = X_train.drop(columns=dropcols)

# X_val = df[df["date_block_num"] == 33].drop(["item_cnt", "date_block_num"], axis=1)
# y_val = df.loc[df["date_block_num"] == 33, "item_cnt"]

# X_val = X_val.drop(columns=dropcols)

# # --- Không gian siêu tham số ---
# param_dist = {
#     "learning_rate": [0.01, 0.03, 0.05],
#     "max_depth": [6, 8, 10],
#     "max_leaves": [512, 768, 1024],
#     "min_child_weight": [1, 5, 10],
#     "subsample": [0.6, 0.8, 1.0],
#     "colsample_bytree": [0.6, 0.8, 1.0],
#     "n_estimators": [200, 300, 500],
# }

# param_list = list(ParameterSampler(param_dist, n_iter=20, random_state=42))

# # --- Tìm kiếm mô hình tốt nhất ---
# best_rmse = float("inf")
# best_model = None
# best_params = None

# all_losses = []

# for i, params in enumerate(param_list):
#     xgb_model = xgb.XGBRegressor(
#         **params,
#         eval_metric="rmse",
#     )

#     xgb_model.fit(
#         X_train, y_train,
#         eval_set=[(X_val, y_val)],
#         verbose=100
#     )

#     val_pred = xgb_model.predict(X_val).clip(0, 20)
#     rmse = np.sqrt(mean_squared_error(y_val, val_pred))
#     all_losses.append(rmse)

#     print(f"Trial {i+1}")

#     if rmse < best_rmse:
#         best_rmse = rmse
#         best_model = xgb_model
#         best_params = params

# print("\nBest RMSE:", best_rmse)
# print("Best Params:", best_params)

# with open("tuning_xgb.csv", "w") as f:
#     for loss in all_losses:
#         f.write(f"{loss}\n")



# param_list = list(ParameterSampler(param_dist, n_iter=20, random_state=42))

# visualize_tuning_results_random(param_list, all_losses, model_name="XGBoost")


# from sklearn.model_selection import ParameterSampler
# from sklearn.metrics import mean_squared_error
# import numpy as np

# # Chia tập train/val theo tháng
# X_train = df[df["date_block_num"] < 33].drop(["item_cnt", "date_block_num"], axis=1)
# y_train = df.loc[df["date_block_num"] < 33, "item_cnt"]

# X_val = df[df["date_block_num"] == 33].drop(["item_cnt", "date_block_num"], axis=1)
# y_val = df.loc[df["date_block_num"] == 33, "item_cnt"]
# X_val = X_val.drop(columns=dropcols)

# # Các cột categorical
# categoricals = ["month"]

# # Define parameter distributions
# param_dist = {
#     "num_leaves": [256, 512, 768, 966],
#     "cat_smooth": [10, 20, 45, 70],
#     "min_child_samples": [10, 20, 30, 40],
#     "min_child_weight": [0.01, 0.02, 0.05, 0.1],
#     "max_bin": [128, 160, 214, 256],
#     "learning_rate": [0.01, 0.03, 0.05],
#     "subsample_for_bin": [200000, 300000, 400000],
#     "min_data_in_bin": [5, 7, 9],
#     "subsample": [0.6, 0.8, 1.0],
#     "subsample_freq": [1, 3, 5],
#     "colsample_bytree": [0.6, 0.8, 1.0],
#     "n_estimators": [200, 300, 500],
# }

# # Random lấy 30 tổ hợp
# param_list = list(ParameterSampler(param_dist, n_iter=20, random_state=42))

# best_rmse = float("inf")
# best_model = None
# best_params = None

# all_losses = []

# for i, params in enumerate(param_list):

#     model = fit_booster(
#         X_train, y_train,
#         X_test=X_val, y_test=y_val,
#         test_flag=True,
#         params=params,
#         categoricals=categoricals,
#         dropcols=dropcols
#     )

#     val_pred = model.predict(X_val).clip(0, 20)
#     rmse = np.sqrt(mean_squared_error(y_val, val_pred))
#     all_losses.append(rmse)
#     print(f"Trial {i+1}")

#     if rmse < best_rmse:
#         best_rmse = rmse
#         best_model = model
#         best_params = params

# print("\nBest RMSE:", best_rmse)
# print("Best Params:", best_params)

# with open("tuning_lgbm.csv", "w") as f:
#     for loss in all_losses:
#         f.write(f"{loss}\n")



params = {
    "num_leaves": 768,
    "cat_smooth": 10,
    "min_child_samples": 40,
    "min_child_weight": 0.05,
    "max_bin": 160,
    "learning_rate": 0.03,
    "subsample_for_bin": 300000,
    "min_data_in_bin": 5,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 3,
    "n_estimators": 300,
}


# import xgboost as xgb
# from sklearn.metrics import mean_squared_error

# X_train = df[df["date_block_num"] < 33].drop(["item_cnt", "date_block_num"], axis=1)
# y_train = df.loc[df["date_block_num"] < 33, "item_cnt"]
# X_train = X_train.drop(columns=dropcols)

# X_val = df[df["date_block_num"] == 33].drop(["item_cnt", "date_block_num"], axis=1)
# y_val = df.loc[df["date_block_num"] == 33, "item_cnt"]

# X_val = X_val.drop(columns=dropcols)

# # Khởi tạo mô hình XGBoost
# xgb_model = xgb.XGBRegressor(
#     subsample=1.0,
#     learning_rate=0.03,
#     max_leaves=768,
#     n_estimators=300,
#     min_child_weight=10,
#     eval_metric='rmse',
#     colsample_bytree=0.8,
#     max_depth=10
# )

# xgb_model.fit(
#     X_train,
#     y_train,
#     eval_set=[(X_val, y_val)],
#     verbose=False  # Tắt log ra màn hình
# )

# # Dự đoán và tính RMSE
# val_pred = xgb_model.predict(X_val).clip(0, 20)
# rmse = np.sqrt(mean_squared_error(y_val, val_pred))
# print(rmse)

# # Lấy độ quan trọng của các feature theo gain
# importances = xgb_model.get_booster().get_score(importance_type='gain')

# # Tạo DataFrame với các feature và gain tương ứng
# feat_imp = pd.DataFrame(list(importances.items()), columns=['Feature', 'Gain'])

# # Sắp xếp các feature theo gain giảm dần
# feat_imp = feat_imp.sort_values(by='Gain', ascending=False)

# # In ra bảng kết quả
# print(feat_imp)

# # Vẽ biểu đồ top 30 feature quan trọng nhất dựa trên gain
# plt.figure(figsize=(10, 12))
# sns.barplot(data=feat_imp.head(30), x="Gain", y="Feature", palette="viridis")
# plt.title("Top 30 Feature Importances (XGBoost) - Gain", fontsize=16)
# plt.xlabel("Gain", fontsize=14)
# plt.ylabel("Feature", fontsize=14)
# plt.tight_layout()
# plt.show()



# !pip install memory_profiler

# import time
# import psutil
# import threading
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# # Biến toàn cục để theo dõi RAM
# ram_usages = []
# stop_monitoring = False

# # Hàm theo dõi RAM trong luồng riêng
# def monitor_memory():
#     process = psutil.Process()
#     while not stop_monitoring:
#         current_ram = process.memory_info().rss  # RAM sử dụng hiện tại (bytes)
#         ram_usages.append(current_ram)
#         time.sleep(0.1)  # Theo dõi mỗi 0.1 giây



# # Tạo mô hình
# model = xgb.XGBRegressor(
#     subsample=1.0,
#     learning_rate=0.03,
#     max_leaves=768,
#     n_estimators=300,
#     min_child_weight=10,
#     eval_metric='rmse',
#     colsample_bytree=0.8,
#     max_depth=10
# )

# # Bắt đầu theo dõi RAM trong luồng riêng
# monitor_thread = threading.Thread(target=monitor_memory)
# monitor_thread.start()

# # Đo thời gian và huấn luyện mô hình
# start_time = time.time()
# model.fit(X_train, y_train)
# end_time = time.time()

# # Dừng theo dõi RAM
# stop_monitoring = True
# monitor_thread.join()

# # Tính toán min, max, average RAM
# if ram_usages:
#     min_ram = min(ram_usages)
#     max_ram = max(ram_usages)
#     average_ram = sum(ram_usages) / len(ram_usages)
# else:
#     min_ram = max_ram = average_ram = 0

# # Dự đoán trên tập validation
# y_pred = model.predict(X_val)

# # Tính các độ đo
# mse = mean_squared_error(y_val, y_pred)
# mae = mean_absolute_error(y_val, y_pred)
# mape = mean_absolute_percentage_error(y_val, y_pred)

# # Thời gian huấn luyện
# training_time = end_time - start_time

# # Tên model
# model_name = "XGBoost"

# # Chuẩn bị dữ liệu cho CSV
# results = {
#     "Model": [model_name],
#     "MSE": [mse],
#     "MAE": [mae],
#     "MAPE": [mape],
#     "Training Time (s)": [training_time],
#     "Min RAM (MB)": [min_ram / 1024**2],
#     "Max RAM (MB)": [max_ram / 1024**2],
#     "Average RAM (MB)": [average_ram / 1024**2]
# }

# # Tạo DataFrame và ghi vào file CSV
# df = pd.DataFrame(results)
# df.to_csv("model_results.csv", index=False)

# print("Kết quả đã được ghi vào file model_results.csv")


# # Sắp xếp theo gain tăng dần
# feat_imp_sorted = feat_imp.sort_values(by='Gain', ascending=False)
# low_gain_features = feat_imp_sorted.head(20)['Feature'].tolist()
# print("10 feature có gain thấp nhất:", low_gain_features)



# # Duyệt theo thứ tự dòng thực tế (0, 1, 2, ...) và kiểm tra 'length' trong tên feature
# for idx, row in feat_imp_sorted.reset_index(drop=True).iterrows():
#     if 'length' in row['Feature']:
#         print(f"Thứ tự: {idx}, Tên: {row['Feature']}, Độ quan trọng: {row['Gain']}")



# for i in range(1, 11):
#     low_gain_features = feat_imp_sorted.head(i)['Feature'].tolist()
    
#     X_train_filtered = X_train.drop(columns=low_gain_features, errors='ignore') 
#     X_val_filtered = X_val.drop(columns=low_gain_features, errors='ignore')
    
#     # Huấn luyện lại XGBoost
#     model = xgb.XGBRegressor(
#         subsample=1.0,
#         learning_rate=0.03,
#         max_leaves=768,
#         n_estimators=300,
#         min_child_weight=10,
#         eval_metric='rmse',
#         colsample_bytree=0.8,
#         max_depth=10
#     )
    
#     model.fit(
#         X_train_filtered, y_train,
#         eval_set=[(X_val_filtered, y_val)],
#         verbose=False
#     )
    
#     # Dự đoán và tính RMSE
#     val_pred = model.predict(X_val_filtered).clip(0, 20)
#     rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    
#     print(f"RMSE: {rmse:.5f}")


params = {
    "num_leaves": 768,
    "cat_smooth": 10,
    "min_child_samples": 40,
    "min_child_weight": 0.05,
    "max_bin": 160,
    "learning_rate": 0.03,
    "subsample_for_bin": 300000,
    "min_data_in_bin": 5,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 3,
    "n_estimators": 300,
}


# !pip install memory_profiler

# import lightgbm as lgbm
# from lightgbm import early_stopping, log_evaluation
# import time
# import psutil
# import threading
# import pandas as pd
# from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# # Biến toàn cục để theo dõi RAM
# ram_usages = []
# stop_monitoring = False

# # Hàm theo dõi RAM trong luồng riêng
# def monitor_memory():
#     process = psutil.Process()
#     while not stop_monitoring:
#         current_ram = process.memory_info().rss  # RAM sử dụng hiện tại (bytes)
#         ram_usages.append(current_ram)
#         time.sleep(0.1)  # Theo dõi mỗi 0.1 giây



# # Tạo mô hình
# model = lgbm.LGBMRegressor(**params)

# # Bắt đầu theo dõi RAM trong luồng riêng
# monitor_thread = threading.Thread(target=monitor_memory)
# monitor_thread.start()

# # Đo thời gian và huấn luyện mô hình
# start_time = time.time()
# model.fit(X_train, y_train)
# end_time = time.time()

# # Dừng theo dõi RAM
# stop_monitoring = True
# monitor_thread.join()

# # Tính toán min, max, average RAM
# if ram_usages:
#     min_ram = min(ram_usages)
#     max_ram = max(ram_usages)
#     average_ram = sum(ram_usages) / len(ram_usages)
# else:
#     min_ram = max_ram = average_ram = 0

# # Dự đoán trên tập validation
# y_pred = model.predict(X_val)

# # Tính các độ đo
# mse = mean_squared_error(y_val, y_pred)
# mae = mean_absolute_error(y_val, y_pred)
# mape = mean_absolute_percentage_error(y_val, y_pred)

# # Thời gian huấn luyện
# training_time = end_time - start_time

# # Tên model
# model_name = "LightGBM"

# # Chuẩn bị dữ liệu cho CSV
# results = {
#     "Model": [model_name],
#     "MSE": [mse],
#     "MAE": [mae],
#     "MAPE": [mape],
#     "Training Time (s)": [training_time],
#     "Min RAM (MB)": [min_ram / 1024**2],
#     "Max RAM (MB)": [max_ram / 1024**2],
#     "Average RAM (MB)": [average_ram / 1024**2]
# }

# # Tạo DataFrame và ghi vào file CSV
# df = pd.DataFrame(results)
# df.to_csv("model_results.csv", index=False)

# print("Kết quả đã được ghi vào file model_results.csv")


# X_train = df[df["date_block_num"] < 34].drop(["item_cnt", "date_block_num"], axis=1)
# y_train = df.loc[df["date_block_num"] < 34, "item_cnt"]
# X_train = X_train.drop(columns=dropcols)

# X_val = df[df["date_block_num"] == 34].drop(["item_cnt", "date_block_num"], axis=1)
# y_val = df.loc[df["date_block_num"] == 34, "item_cnt"]
# X_val = X_val.drop(columns=dropcols)

# xgb_model = xgb.XGBRegressor(
#     subsample=1.0,
#     learning_rate=0.03,
#     max_leaves=768,
#     n_estimators=300,
#     min_child_weight=10,
#     eval_metric='rmse',
#     colsample_bytree=0.8,
#     max_depth=10
# )

# xgb_model.fit(
#     X_train,
#     y_train,
#     eval_set=[(X_train, y_train)],
#     verbose=False  # Tắt log ra màn hình
# )

# val_pred = xgb_model.predict(X_val).clip(0, 20)




# import lightgbm as lgbm

# X_train = df[df["date_block_num"] < 34].drop(["item_cnt", "date_block_num"], axis=1)
# y_train = df.loc[df["date_block_num"] < 34, "item_cnt"]
# X_train = X_train.drop(columns=dropcols)

# X_val = df[df["date_block_num"] == 34].drop(["item_cnt", "date_block_num"], axis=1)
# y_val = df.loc[df["date_block_num"] == 34, "item_cnt"]
# X_val = X_val.drop(columns=dropcols)

# lgbm_model = model = lgbm.LGBMRegressor(**params)

# lgbm_model.fit(X_train, y_train)

# val_pred = lgbm_model.predict(X_val).clip(0, 20)




# from catboost import CatBoostRegressor

# X_train = df[df["date_block_num"] < 34].drop(["item_cnt", "date_block_num"], axis=1)
# y_train = df.loc[df["date_block_num"] < 34, "item_cnt"]
# X_train = X_train.drop(columns=dropcols)

# X_val = df[df["date_block_num"] == 34].drop(["item_cnt", "date_block_num"], axis=1)
# y_val = df.loc[df["date_block_num"] == 34, "item_cnt"]
# X_val = X_val.drop(columns=dropcols)

# CatBoost = CatBoostRegressor(
#         iterations = 500,
#         depth = 10,
#         subsample = 0.8,
#         random_seed = 42,
#         min_data_in_leaf = 5,
#         learning_rate = 0.05,
#         l2_leaf_reg = 1,
#         verbose=100
# )

# CatBoost.fit(X_train, y_train)

# val_pred = CatBoost.predict(X_val).clip(0, 20)




# from sklearn.ensemble import RandomForestRegressor

# X_train = df[df["date_block_num"] < 34].drop(["item_cnt", "date_block_num"], axis=1)
# y_train = df.loc[df["date_block_num"] < 34, "item_cnt"]
# X_train = X_train.drop(columns=dropcols)

# X_val = df[df["date_block_num"] == 34].drop(["item_cnt", "date_block_num"], axis=1)
# y_val = df.loc[df["date_block_num"] == 34, "item_cnt"]
# X_val = X_val.drop(columns=dropcols)

# Forest = RandomForestRegressor(n_estimators=100, min_samples_split=6, min_samples_leaf=2, max_features='sqrt', max_depth=7)

# Forest.fit(X_train, y_train)

# val_pred = Forest.predict(X_val).clip(0, 20)




# Các thư viện cần thiết
import xgboost as xgb
import lightgbm as lgbm
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import time
import psutil
import gc
import os

# Khởi tạo các list lưu dự đoán và đặc trưng phục vụ cho stacking
preds_arr_lgb=[]              # Dự đoán từ mô hình LightGBM native API
vals_arr_lgb=[]               # Dự đoán từ mô hình LightGBM native API
vals_arr_lgb_84=[]            # Dự đoán từ LightGBM (fit_booster - sklearn API)
preds_arr_xgb=[]              # Dự đoán từ mô hình XGBoost
vals_arr_xgb=[]               # Dự đoán từ mô hình XGBoost
vals_arr_cat=[]
vals_arr_forest=[]
shop_id=[]
item_id=[]
cat_id=[]
month_arr=[]
# # Danh sách lưu trữ RAM sử dụng trong quá trình huấn luyện
# ram_usages_training = []

# # Hàm lấy RAM sử dụng hiện tại (MB)
# def get_memory_usage():
#     process = psutil.Process(os.getpid())
#     mem_info = process.memory_info()
#     return mem_info.rss / 1024 ** 2  # Chuyển sang MB





# """
# Tạo tập huấn luyện và tập kiểm tra

# """
# # Đo tổng thời gian
# # total_start = time.perf_counter() 

# # Lặp qua từng tháng từ 25 đến 34 để huấn luyện mô hình
# for i in range(21, 35):
    
#     # Chuẩn bị dữ liệu huấn luyện: lấy tất cả dữ liệu trước tháng i
#     X_train = df[df["date_block_num"] < i].drop(['item_cnt', "date_block_num"], axis=1)
#     y_train = df.loc[df["date_block_num"] < i, 'item_cnt']
#     X_train = X_train.drop(columns=dropcols)

#     # Dữ liệu kiểm tra (validation): là tháng hiện tại (i)
#     X_val = df[df["date_block_num"] == i].drop(['item_cnt', "date_block_num"], axis=1)
#     y_val = df.loc[df["date_block_num"] == i, 'item_cnt']
#     X_val = X_val.drop(columns=dropcols)

#     """
#     Huấn luyện và đánh giá mô hình theo từng tháng (date_block_num từ 25 đến 34), mô phỏng quá trình dự đoán thực tế.
#     Đây là hình thức time series cross-validation, cực kỳ phù hợp với bài toán chuỗi thời gian như dự đoán doanh số.

#     Huấn luyện các tháng từ 25 đến 34 vì:

#     - Đang thực hiện việc huấn luyện theo từng tháng, trong đó:

#       + X_train: gồm tất cả dữ liệu trước tháng i.

#       + X_val: là dữ liệu của tháng i.

#     - Nếu bắt đầu từ tháng 0 → không còn "tháng -1" để làm X_train.

#     - Nếu bắt đầu từ tháng 1 → X_train sẽ rất ít (chỉ có tháng 0).

#     - Từ tháng 0 đến 24 không đủ thông tin huấn luyện ổn định cho mô hình.

#     """

#     """
#     -------------------Huấn luyện mô hình XGBoost-------------------------
#     """
#     # Khởi tạo mô hình XGBoost
#     xgb_model = xgb.XGBRegressor(
#         subsample=1.0,
#         learning_rate=0.03,
#         max_leaves=768,
#         n_estimators=300,
#         min_child_weight=10,
#         eval_metric='rmse',
#         colsample_bytree=0.8,
#         max_depth=10
#     )

#     # # Đo thời gian và RAM cho huấn luyện
#     # xgb_start = time.perf_counter()
#     # mem_before_training = get_memory_usage()
#     # ram_usages_training.append(mem_before_training)

#     xgb_model.fit(X_train,y_train,)

#     # # Đo RAM sau huấn luyện
#     # mem_after_training = get_memory_usage()
#     # ram_usages_training.append(mem_after_training)
#     # xgb_end = time.perf_counter()
    
#     # Dự đoán và giới hạn kết quả trong khoảng [0, 20] như yêu cầu bài toán
#     val_pred = xgb_model.predict(X_val).clip(0, 20)
#     vals_arr_xgb.append(val_pred)  # Lưu dự đoán

#     del xgb_model
#     gc.collect()

# # # Tổng thời gian
# # total_end = time.perf_counter()

# # # Tính max, min, average RAM cho quá trình huấn luyện
# # ram_max_training = np.max(ram_usages_training)
# # ram_min_training = np.min(ram_usages_training)
# # ram_avg_training = np.mean(ram_usages_training)

# # # Chuẩn bị dữ liệu cho CSV
# # results = {
# #     "Training Time (s)": [total_end - total_start],
# #     "Min RAM (MB)": [ram_min_training],
# #     "Max RAM (MB)": [ram_max_training],
# #     "Average RAM (MB)": [ram_avg_training]
# # }

# # # Tạo DataFrame và ghi vào file CSV
# # df = pd.DataFrame(results)
# # df.to_csv("xgb_stack.csv", index=False)

# # # """
# # # Lưu kết quả dự đoán ra file .pkl

# # # """
# vals_arr_xgb_series=pd.Series(vals_arr_xgb)
# vals_arr_xgb_series.to_pickle(path+'model-con-test-xgb.pkl')




# # # Danh sách lưu trữ RAM sử dụng trong quá trình huấn luyện
# # ram_usages_training = []

# """
# Tạo tập huấn luyện và tập kiểm tra

# """
# # # Đo tổng thời gian
# # total_start = time.perf_counter() 

# # Lặp qua từng tháng từ 25 đến 34 để huấn luyện mô hình
# for i in range(21, 35):
    
#     # Chuẩn bị dữ liệu huấn luyện: lấy tất cả dữ liệu trước tháng i
#     X_train = df[df["date_block_num"] < i].drop(['item_cnt', "date_block_num"], axis=1)
#     y_train = df.loc[df["date_block_num"] < i, 'item_cnt']
#     X_train = X_train.drop(columns=dropcols)

#     # Dữ liệu kiểm tra (validation): là tháng hiện tại (i)
#     X_val = df[df["date_block_num"] == i].drop(['item_cnt', "date_block_num"], axis=1)
#     y_val = df.loc[df["date_block_num"] == i, 'item_cnt']
#     X_val = X_val.drop(columns=dropcols)

#     """
#     -------------------Huấn luyện mô hình LightGBM với sklearn API-------------------------
#     """

#     # Huấn luyện mô hình LightGBM sử dụng sklearn API (fit_booster)
#     lgbooster = lgbm.LGBMRegressor(**params)

#     # # Đo thời gian và RAM cho huấn luyện
#     # mem_before_training = get_memory_usage()
#     # ram_usages_training.append(mem_before_training)

#     lgbooster.fit(X_train,y_train)

#     # # Đo RAM sau huấn luyện
#     # mem_after_training = get_memory_usage()
#     # ram_usages_training.append(mem_after_training)
    
#     # Dự đoán và giới hạn giá trị
#     val_pred = lgbooster.predict(X_val).clip(0, 20)
#     vals_arr_lgb_84.append(val_pred)  # Lưu dự đoán từ mô hình này

#     del lgbooster
#     gc.collect()
#     """
#     Dùng LightGBM dưới dạng LGBMRegressor thông qua hàm fit_booster .

#     Sử dụng categoricals giúp LightGBM xử lý tốt hơn đặc trưng phân loại → tăng độ chính xác.

#     Dự đoán được lưu lại phục vụ cho stacking.

#     """

# # # Tổng thời gian
# # total_end = time.perf_counter()

# # # Tính max, min, average RAM cho quá trình huấn luyện
# # ram_max_training = np.max(ram_usages_training)
# # ram_min_training = np.min(ram_usages_training)
# # ram_avg_training = np.mean(ram_usages_training)

# # # Chuẩn bị dữ liệu cho CSV
# # results = {
# #     "Training Time (s)": [total_end - total_start],
# #     "Min RAM (MB)": [ram_min_training],
# #     "Max RAM (MB)": [ram_max_training],
# #     "Average RAM (MB)": [ram_avg_training]
# # }

# # # Tạo DataFrame và ghi vào file CSV
# # df = pd.DataFrame(results)
# # df.to_csv("lgbm_stack.csv", index=False)

# """
# Lưu kết quả dự đoán ra file .pkl

# """

# vals_arr_lgb_series_84s=pd.Series(vals_arr_lgb_84)
# vals_arr_lgb_series_84s.to_pickle(path+'model-con-test-lgbm.pkl')

# # print("Các kết quả đã được ghi vào các file")


# # # Danh sách lưu trữ RAM sử dụng trong quá trình huấn luyện
# # ram_usages_training = []

# """
# Tạo tập huấn luyện và tập kiểm tra

# """
# # # Đo tổng thời gian
# # total_start = time.perf_counter() 

# # Lặp qua từng tháng từ 25 đến 34 để huấn luyện mô hình
# for i in range(21, 35):
    
#     # Chuẩn bị dữ liệu huấn luyện: lấy tất cả dữ liệu trước tháng i
#     X_train = df[df["date_block_num"] < i].drop(['item_cnt', "date_block_num"], axis=1)
#     y_train = df.loc[df["date_block_num"] < i, 'item_cnt']
#     X_train = X_train.drop(columns=dropcols)

#     # Dữ liệu kiểm tra (validation): là tháng hiện tại (i)
#     X_val = df[df["date_block_num"] == i].drop(['item_cnt', "date_block_num"], axis=1)
#     y_val = df.loc[df["date_block_num"] == i, 'item_cnt']
#     X_val = X_val.drop(columns=dropcols)

#     """
#     -------------------Huấn luyện mô hình CatBoost-------------------------
#     """

#     # Huấn luyện mô hình 
#     CatBoost = CatBoostRegressor(
#         iterations = 500,
#         depth = 10,
#         subsample = 0.8,
#         random_seed = 42,
#         min_data_in_leaf = 5,
#         learning_rate = 0.05,
#         l2_leaf_reg = 1,
#         verbose=100
#     )

#     # # Đo thời gian và RAM cho huấn luyện
#     # mem_before_training = get_memory_usage()
#     # ram_usages_training.append(mem_before_training)

#     CatBoost.fit(X_train,y_train)

#     # # Đo RAM sau huấn luyện
#     # mem_after_training = get_memory_usage()
#     # ram_usages_training.append(mem_after_training)
    
#     # Dự đoán và giới hạn giá trị
#     val_pred = CatBoost.predict(X_val).clip(0, 20)
#     vals_arr_cat.append(val_pred)  # Lưu dự đoán từ mô hình này

#     del CatBoost
#     gc.collect()
#     """
#     Dùng LightGBM dưới dạng LGBMRegressor thông qua hàm fit_booster .

#     Sử dụng categoricals giúp LightGBM xử lý tốt hơn đặc trưng phân loại → tăng độ chính xác.

#     Dự đoán được lưu lại phục vụ cho stacking.

#     """

# # # Tổng thời gian
# # total_end = time.perf_counter()

# # # Tính max, min, average RAM cho quá trình huấn luyện
# # ram_max_training = np.max(ram_usages_training)
# # ram_min_training = np.min(ram_usages_training)
# # ram_avg_training = np.mean(ram_usages_training)

# # # Chuẩn bị dữ liệu cho CSV
# # results = {
# #     "Training Time (s)": [total_end - total_start],
# #     "Min RAM (MB)": [ram_min_training],
# #     "Max RAM (MB)": [ram_max_training],
# #     "Average RAM (MB)": [ram_avg_training]
# # }

# # # Tạo DataFrame và ghi vào file CSV
# # df = pd.DataFrame(results)
# # df.to_csv("cat_stack.csv", index=False)

# """
# Lưu kết quả dự đoán ra file .pkl

# """

# vals_arr_cat_series=pd.Series(vals_arr_cat)
# vals_arr_cat_series.to_pickle(path+'model-con-test-catboost.pkl')

# # print("Các kết quả đã được ghi vào các file")


# # Danh sách lưu trữ RAM sử dụng trong quá trình huấn luyện
# ram_usages_training = []

# """
# Tạo tập huấn luyện và tập kiểm tra

# """
# # Đo tổng thời gian
# total_start = time.perf_counter() 

# # Lặp qua từng tháng từ 25 đến 34 để huấn luyện mô hình
# for i in range(21, 34):
    
#     # Chuẩn bị dữ liệu huấn luyện: lấy tất cả dữ liệu trước tháng i
#     X_train = df[df["date_block_num"] < i].drop(['item_cnt', "date_block_num"], axis=1)
#     y_train = df.loc[df["date_block_num"] < i, 'item_cnt']
#     X_train = X_train.drop(columns=dropcols)

#     # Dữ liệu kiểm tra (validation): là tháng hiện tại (i)
#     X_val = df[df["date_block_num"] == i].drop(['item_cnt', "date_block_num"], axis=1)
#     y_val = df.loc[df["date_block_num"] == i, 'item_cnt']
#     X_val = X_val.drop(columns=dropcols)

#     """
#     -------------------Huấn luyện mô hình Random Forest-------------------------
#     """

#     # Huấn luyện mô hình 
#     Forest = RandomForestRegressor(n_estimators=100, min_samples_split=6, min_samples_leaf=2, max_features='sqrt', max_depth=7)

#     # Đo thời gian và RAM cho huấn luyện
#     mem_before_training = get_memory_usage()
#     ram_usages_training.append(mem_before_training)

#     Forest.fit(X_train,y_train)

#     # Đo RAM sau huấn luyện
#     mem_after_training = get_memory_usage()
#     ram_usages_training.append(mem_after_training)
    
#     # Dự đoán và giới hạn giá trị
#     val_pred = Forest.predict(X_val).clip(0, 20)
#     vals_arr_forest.append(val_pred)  # Lưu dự đoán từ mô hình này

#     print(f"Trial {i}")

#     del Forest
#     gc.collect()
#     """
#     Dự đoán được lưu lại phục vụ cho stacking.

#     """

# # Tổng thời gian
# total_end = time.perf_counter()

# # Tính max, min, average RAM cho quá trình huấn luyện
# ram_max_training = np.max(ram_usages_training)
# ram_min_training = np.min(ram_usages_training)
# ram_avg_training = np.mean(ram_usages_training)

# # Chuẩn bị dữ liệu cho CSV
# results = {
#     "Training Time (s)": [total_end - total_start],
#     "Min RAM (MB)": [ram_min_training],
#     "Max RAM (MB)": [ram_max_training],
#     "Average RAM (MB)": [ram_avg_training]
# }

# # Tạo DataFrame và ghi vào file CSV
# df = pd.DataFrame(results)
# df.to_csv("forest_stack.csv", index=False)

# """
# Lưu kết quả dự đoán ra file .pkl

# """

# vals_arr_forest_series=pd.Series(vals_arr_forest)
# vals_arr_forest_series.to_pickle(path+'model-con-randomforest.pkl')

# print("Các kết quả đã được ghi vào các file")


# Lặp qua từng tháng từ 25 đến 34 để huấn luyện mô hình
for i in range(21, 35):
    
    # Chuẩn bị dữ liệu huấn luyện: lấy tất cả dữ liệu trước tháng i
    X_train = df[df["date_block_num"] < i].drop(['item_cnt', "date_block_num"], axis=1)
    y_train = df.loc[df["date_block_num"] < i, 'item_cnt']
    X_train = X_train.drop(columns=dropcols)

    # Dữ liệu kiểm tra (validation): là tháng hiện tại (i)
    X_val = df[df["date_block_num"] == i].drop(['item_cnt', "date_block_num"], axis=1)
    y_val = df.loc[df["date_block_num"] == i, 'item_cnt']
    X_val = X_val.drop(columns=dropcols)

    preds_arr_lgb.append([y_val])



# Đọc lại 3 file pickle chứa kết quả dự đoán của các mô hình con
vals_arr_xgb = pd.read_pickle("/kaggle/input/model-stack/model-con-test-xgb.pkl")
vals_arr_lgb = pd.read_pickle("/kaggle/input/model-stack/model-con-test-lgbm.pkl")
vals_arr_lgb_84s = pd.read_pickle("/kaggle/input/model-stack/model-con-test-catboost.pkl")


# Giải nén từng mảng dự đoán thành danh sách Series, rồi gộp thành một Series lớn
# Mỗi phần tử ban đầu là một mảng các giá trị dự đoán của từng fold trong cross-validation

s = []
for i in vals_arr_xgb[:-1]:
    a = pd.Series(i)
    s.append(a)
series_xgb = pd.concat(s)

s = []
for i in vals_arr_lgb[:-1]:
    a = pd.Series(i)
    s.append(a)
series_lgb = pd.concat(s)

s = []
for i in vals_arr_lgb_84s[:-1]:
    a = pd.Series(i)
    s.append(a)
series_lgbs = pd.concat(s)


# Lấy lại giá trị nhãn thực tế (y_val) từ kết quả dự đoán lưu trước đó
# Mỗi phần tử là (y_val, y_pred), ta chỉ lấy y_val

s = []
for i in preds_arr_lgb[:-1]:  # preds_arr_lgb là danh sách các (y_val, y_pred)
    s.append(pd.Series(i[0].values))  # Lấy y_val
actual_all = pd.concat(s)


# # Chuyển danh sách các đặc trưng shop_id, item_id, category_id, month thành các Series

# s = []
# for i in shop_id[:-1]:
#     s.append(pd.Series(i))
# series_shop = pd.concat(s)

# s = []
# for i in item_id[:-1]:
#     s.append(pd.Series(i))
# series_item = pd.concat(s)

# s = []
# for i in cat_id[:-1]:
#     s.append(pd.Series(i))
# series_cat = pd.concat(s)

# s = []
# for i in month_arr[:-1]:
#     s.append(pd.Series(i))
# series_month = pd.concat(s)



# Gộp tất cả các Series lại thành một DataFrame duy nhất
# Đây là dữ liệu đầu vào cho mô hình stacking

datframe = pd.concat([
    series_xgb,     # dự đoán từ mô hình xgb
    series_lgb,     # dự đoán từ mô hình LightGBM với native API
    series_lgbs,    # dự đoán từ mô hình LightGBM với sklearn API
    actual_all,     # giá trị thực tế
], axis=1)


# Đặt tên cột cho rõ ràng
datframe = datframe.rename(columns={
    0: "xgb",
    1: "lgb",
    2: "cat",
    3: "actual"
})



# Tách đặc trưng (X) và nhãn (Y) để huấn luyện mô hình stacking cuối cùng
# Loại bỏ cột 'actual' và 'month' khỏi X

Y = datframe["actual"]
X = datframe.drop(columns=["actual"])


# from sklearn.linear_model import LinearRegression
# reg = LinearRegression().fit(X, Y)
# from sklearn.linear_model import Ridge

# # Khởi tạo và huấn luyện Ridge Regression với alpha = 1.0 (có thể điều chỉnh)
# ridge = Ridge(alpha=1.0)
# ridge.fit(X, Y)

from sklearn.svm import SVR

# Khởi tạo và huấn luyện SVR với kernel RBF (mặc định)
svr = SVR(C=1.0, epsilon=0.1, kernel='rbf')
svr.fit(X, Y)




# Lấy kết quả dự đoán của mô hình XGBoost tương ứng trong test.csv
series_xgb = pd.Series(vals_arr_xgb[-1:].reset_index(drop=True)[0])

# Lấy kết quả dự đoán của mô hình LightGBM với native API tương ứng trong test.csv
series_lgb = pd.Series(vals_arr_lgb[-1:].reset_index(drop=True)[0])

# Lấy kết quả dự đoán của mô hình LightGBM với sklearn API tương ứng trong test.csv
series_lgbs = pd.Series(vals_arr_lgb_84s[-1:].reset_index(drop=True)[0])

# # Lấy thông tin tháng tương ứng trong test.csv
# series_month_ser = pd.Series(month_arr[-1:][0])
# 
# # Lấy thông tin shop_id tương ứng trong test.csv
# series_shop_ser = pd.Series(shop_id[-1])

# # Lấy thông tin item_id tương ứng trong test.csv
# series_item_ser = pd.Series(item_id[-1])

# # Lấy thông tin category_id tương ứng trong test.csv
# series_cat_ser = pd.Series(cat_id[-1])



datframe_test = pd.concat([
    series_xgb.reset_index(drop=True),       # dự đoán từ xgb
    series_lgb.reset_index(drop=True),       # dự đoán từ lgb với native API
    series_lgbs.reset_index(drop=True),      # dự đoán từ lgb với sklearn API
], axis=1)

# Đặt tên cột cho DataFrame test
datframe_test = datframe_test.rename(columns={
    0: "xgb",
    1: "lgb",
    2: "cat",
})

X_test = datframe_test.drop(columns=[])


val_pred=svr.predict(X_test)


submission = pd.read_csv(path2+'sample_submission.csv')
submission['item_cnt_month'] = pd.Series(val_pred).clip(0,20)

items = pd.read_csv(path2+"items.csv")
shops = pd.read_csv(path2+"shops.csv")
train = pd.read_csv(path2+"sales_train.csv")
tr=pd.merge(train,items,how="left",on="item_id")
test = pd.read_csv(path2+"test.csv")
test=pd.merge(test,items,how="left",on="item_id")
submission=pd.merge(submission,test,on="ID")



# error_analysis_items_list=error_analysis_file[(error_analysis_file["release_day"].isin([29,30,31]))&\
#                                               (error_analysis_file["unique_in_33"]==1)]["item_id"].unique()

#categories in shop 55 
cat_55_shop=[39,25,31,76]
submission.loc[((submission["item_category_id"].isin(cat_55_shop))&\
           (submission["shop_id"]!=55)),"item_cnt_month"]=0



###############one direction with id 5320 and name ONE DIRECTION Made In The A.M.	 was definitely a hit
submission.loc[((submission["item_id"]==5320)&(submission["shop_id"]!=55)),"item_cnt_month"]=\
submission.loc[((submission["item_id"]==5320)&(submission["shop_id"]!=55)),"item_cnt_month"]*3


submission.loc[((submission["item_id"]==18743)&(submission["shop_id"]!=55)),"item_cnt_month"]=\
submission.loc[((submission["item_id"]==18743)&(submission["shop_id"]!=55)),"item_cnt_month"]*2




#category id 34 always remain above 20 so
submission.loc[((submission["item_category_id"]==34)&\
(submission["shop_id"]==55)),"item_cnt_month"]=20

submission.loc[((submission["item_category_id"]==34)&\
(submission["shop_id"]!=55)),"item_cnt_month"]=0

submission.loc[(submission["item_id"]==7223),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==7223),"item_cnt_month"]*1.3
# #

##5268 need for speed might be a popular game
submission.loc[(submission["item_id"]==5268),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==5268),"item_cnt_month"]*3



#2323 is call of duty black hawks down is very popular and for PS4 it was a super hit as seen from google ratings and wikipedia 
submission.loc[(submission["item_id"]==2323),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==2323),"item_cnt_month"]*3

# 2327 s call of duty black hawks down is very popular and for PS4 it was a super hit as seen from google ratings and wikipedia
submission.loc[(submission["item_id"]==2327),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==2327),"item_cnt_month"]*3


#3408 Fallout 4 eleased in 2015 is very popular
submission.loc[(submission["item_id"]==3408),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==3408),"item_cnt_month"]*3

# 3405 Fallout 4 eleased in 2015 is very popular
submission.loc[(submission["item_id"]==3405),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==3405),"item_cnt_month"]*3

# 3407 Fallout 4 eleased in 2015 is very popular
submission.loc[(submission["item_id"]==3407),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==3407),"item_cnt_month"]*3

#6729 and 6732,6731  are starwars battlefront variants and they all are extremely popular
submission.loc[(submission["item_id"]==6729),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==6729),"item_cnt_month"]*1.5

submission.loc[(submission["item_id"]==6731),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==6731),"item_cnt_month"]*1.5

submission.loc[(submission["item_id"]==6732),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==6732),"item_cnt_month"]*1.5

#7782 wasteland is somewhat popular but not so 
submission.loc[(submission["item_id"]==7782),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==7782),"item_cnt_month"]*1.5
#PC game football
submission.loc[(submission["item_id"]==3538),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==3538),"item_cnt_month"]*1.5
#rise of tomb rider xbox1
submission.loc[(submission["item_id"]==6153),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==6153),"item_cnt_month"]*1.5
submission.loc[(submission["item_id"]==6152),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==6152),"item_cnt_month"]*1.5

#rise of tomb rider xbox1
submission.loc[(submission["item_id"]==2326),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==2326),"item_cnt_month"]*1.5

#rise of tomb rider xbox1
submission.loc[(submission["item_id"]==2328),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==2328),"item_cnt_month"]*2
#rise of tomb rider xbox1
submission.loc[(submission["item_id"]==1577),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==1577),"item_cnt_month"]*1.5
# #rise of tomb rider xbox1
submission.loc[(submission["item_id"]==5269),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==5269),"item_cnt_month"]*1.5
# #rise of tomb rider xbox1
submission.loc[(submission["item_id"]==4060),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==4060),"item_cnt_month"]*2








# .comic high rating
submission.loc[(submission["item_id"]==13310),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==13310),"item_cnt_month"]*3

# submission.loc[(submission["item_id"]==13338),"item_cnt_month"]=\
# submission.loc[(submission["item_id"]==13338),"item_cnt_month"]*1.5

# submission.loc[(submission["item_id"]==13309),"item_cnt_month"]=\
# submission.loc[(submission["item_id"]==13309),"item_cnt_month"]*1.5

#3571  angels and ghost was quite popular
submission.loc[(submission["item_id"]==3571),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==3571),"item_cnt_month"]*2

#3984 JARRE JEAN MICHEL  Electronika 1  The Time Machine released on october16 15 will also be sold in november
submission.loc[(submission["item_id"]==3984),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==3984),"item_cnt_month"]*2

#Selena gomez revival
submission.loc[(submission["item_id"]==3604),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==3604),"item_cnt_month"]*1.5

#botelli 
submission.loc[(submission["item_id"]==1732),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==1732),"item_cnt_month"]*1.5

#botelli 
submission.loc[(submission["item_id"]==1246),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==1246),"item_cnt_month"]*1.5


# 6335 STEWART ROD  Another Country
submission.loc[(submission["item_id"]==6335),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==6335),"item_cnt_month"]*1.5

# 10203 quite popular when starts
submission.loc[(submission["item_id"]==10203),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==10203),"item_cnt_month"]*1.5

# 6152 tom raider is a famous game
submission.loc[(submission["item_id"]==6152),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==6152),"item_cnt_month"]*1.5

# 6153 tom raider is a famous game
submission.loc[(submission["item_id"]==6153),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==6153),"item_cnt_month"]*3

# 6742 statecraft is a famous game
submission.loc[(submission["item_id"]==6742),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==6742),"item_cnt_month"]*3



#13745 BLACK WITCHER IS PLAYED AND IS FAMOUS
submission.loc[(submission["item_id"]==13745),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==13745),"item_cnt_month"]*3
               

# #5269 \nfs will be a a hot selll
submission.loc[(submission["item_id"]==5269),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==5269),"item_cnt_month"]*3

# # #ЛЕВША famous movie
# submission.loc[(submission["item_id"]==13804),"item_cnt_month"]=\
# submission.loc[(submission["item_id"]==13804),"item_cnt_month"]*2

# #mission impossible  famous movie
submission.loc[(submission["item_id"]==14647),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==14647),"item_cnt_month"]*2






special_shops=[25,31,42]

# #mission impossible  famous movie
submission.loc[(submission["item_id"]==14648),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==14648),"item_cnt_month"]*3
submission.loc[(submission["item_id"]==14648)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]=20

special_shops=[25,31,42]

submission.loc[(submission["item_id"]==2427)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]=8


submission.loc[(submission["item_id"]==5268)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==5268)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]*2

submission.loc[(submission["item_id"]==5269)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==5269)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]*2

submission.loc[(submission["item_id"]==13745)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==13745)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]*2

submission.loc[(submission["item_id"]==3408)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==3408)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]*2

submission.loc[(submission["item_id"]==3405)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==3405)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]*2

submission.loc[(submission["item_id"]==3407)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==3407)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]*2

submission.loc[(submission["item_id"]==2327)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==2327)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]*2

submission.loc[(submission["item_id"]==6742)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==6742)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]*2



special_shops_new=[25,31,42]
submission.loc[(submission["item_id"]==10447)&(submission["shop_id"].isin(special_shops_new)),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==10447)&(submission["shop_id"].isin(special_shops_new)),"item_cnt_month"]*1.5

submission.loc[(submission["item_id"]==19655)&(submission["shop_id"].isin(special_shops_new)),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==19655)&(submission["shop_id"].isin(special_shops_new)),"item_cnt_month"]*1.5

submission.loc[(submission["item_id"]==19657)&(submission["shop_id"].isin(special_shops_new)),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==19657)&(submission["shop_id"].isin(special_shops_new)),"item_cnt_month"]*1.5

submission.loc[(submission["item_id"]==10449)&(submission["shop_id"].isin(special_shops_new)),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==10449)&(submission["shop_id"].isin(special_shops_new)),"item_cnt_month"]*1.5

submission.loc[(submission["item_id"]==10449)&(submission["shop_id"].isin(special_shops_new)),"item_cnt_month"]=\
submission.loc[(submission["item_id"]==10449)&(submission["shop_id"].isin(special_shops_new)),"item_cnt_month"]*1.5


submission.loc[(submission["item_id"].isin([5268]))&(submission["shop_id"]==39),"item_cnt_month"]=20
submission.loc[(submission["item_id"].isin([204075]))&(submission["shop_id"]==39),"item_cnt_month"]=20
submission.loc[(submission["item_id"].isin([204077]))&(submission["shop_id"]==39),"item_cnt_month"]=18
submission.loc[(submission["item_id"].isin([6742]))&(submission["shop_id"]==39),"item_cnt_month"]=20
submission.loc[(submission["item_id"].isin([2327]))&(submission["shop_id"]==39),"item_cnt_month"]=20
submission.loc[(submission["item_id"].isin([2323]))&(submission["shop_id"]==39),"item_cnt_month"]=20

submission.loc[(submission["item_id"].isin([13247]))&(submission["shop_id"]==39),"item_cnt_month"]=20
submission.loc[(submission["item_id"].isin([13247]))&(submission["shop_id"]==25),"item_cnt_month"]=20

submission.loc[(submission["item_id"].isin([14959])),"item_cnt_month"]=\
submission.loc[(submission["item_id"].isin([14959])),"item_cnt_month"]*2
submission.loc[(submission["item_id"].isin([13303])),"item_cnt_month"]=\
submission.loc[(submission["item_id"].isin([13303])),"item_cnt_month"]*2



submission.loc[(submission["item_id"].isin([21811]))&(submission["shop_id"]==39),"item_cnt_month"]=20
submission.loc[(submission["item_id"].isin([17270]))&(submission["shop_id"]==39),"item_cnt_month"]=20
submission.loc[(submission["item_id"].isin([6732]))&(submission["shop_id"]==39),"item_cnt_month"]=20

submission.loc[(submission["item_id"].isin([2326]))&(submission["shop_id"]==39),"item_cnt_month"]=4
submission.loc[(submission["item_id"].isin([13246]))&(submission["shop_id"]==39),"item_cnt_month"]=5
submission.loc[(submission["item_id"].isin([1583]))&(submission["shop_id"]==39),"item_cnt_month"]=5


#good rating 
submission.loc[(submission["item_id"].isin([10449]))&(submission["shop_id"]==39),"item_cnt_month"]=16
submission.loc[(submission["item_id"].isin([16629]))&(submission["shop_id"]==39),"item_cnt_month"]=16
submission.loc[(submission["item_id"].isin([13805]))&(submission["shop_id"]==39),"item_cnt_month"]=16




submission['item_cnt_month']=submission['item_cnt_month'].clip(0,20)
submission[['ID', 'item_cnt_month']].to_csv('submission.csv', index=False)


