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


df1 = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv')
df2 = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv')
df3 = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv')
df4 = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv')
df5 = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv')
df6= pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv')
print(df1)
print(df2)
print(df3)
print(df4)
print(df5)
print(df6)


def display_dataset_info(dataset, name):  
    print("-----------------------------------------------------------------")  
    print(f"{name} DataFrame Shape: Rows = {dataset.shape[0]}, Columns = {dataset.shape[1]}")  
    
    # Numerical and categorical columns information  
    num_cols = dataset.select_dtypes(include='number')  
    cat_cols = dataset.select_dtypes(exclude='number')  
    print(f"{name} DataFrame has {len(num_cols.columns)} numeric columns and {len(cat_cols.columns)} categorical columns.")  
    
    # Missing values information  
    total_missing = dataset.isnull().sum().sum()  
    if total_missing > 0:  
        missing_perc = (total_missing / (dataset.shape[0] * dataset.shape[1])) * 100  
        print(f"There are a total of {total_missing} missing values in the {name} DataFrame ({missing_perc:.2f}% of all values).")  
        print("Missing values per column:")  
        print(dataset.isnull().sum().sort_values(ascending=False).head(10))  
    else:  
        print(f"There are no missing values in the {name} DataFrame.")  
    
    # Duplicate rows information  
    total_duplicates = dataset.duplicated().sum()  
    if total_duplicates > 0:  
        print(f"There are {total_duplicates} duplicate rows in the {name} DataFrame.")  
    else:  
        print(f"There are no duplicate rows in the {name} DataFrame.")  
    
    # Check for column data types  
    print("\nColumn data types:")  
    print(dataset.dtypes.value_counts())  
    
    print("-----------------------------------------------------------------")  
    

datasets = [df1, df2, df3, df4]
names = [ "sales_train", "sales_test", "inventory", "calendar"]
for i in range(len(datasets)):
    display_dataset_info(datasets[i], names[i])


df1.describe().T


print(df2.describe().T)
print(df3.describe().T)
print(df4.describe().T)
print(df5.describe().T)
print(df6.describe().T)


print(df1.isna())
print('..........................')
print(df2.isna())
print('..........................')
print(df3.isna())
print('..........................')
print(df4.isna())
print('..........................')
print(df5.isna())


import matplotlib.pyplot as plt
import seaborn as sns



plt.figure(figsize=(8, 1.5))
sns.boxplot(x=df3['total_orders'], color='salmon')
plt.title(" Boxplot - Phân phối đơn hàng & phát hiện giá trị ngoại lai")
plt.tight_layout()
plt.show()



# Chuyển về datetime
df3['date'] = pd.to_datetime(df3['date'], errors='coerce')

# Tạo cột tháng-năm
df3['year_month'] = df3['date'].dt.to_period('M').astype(str)

# Tổng hợp doanh thu
monthly_sales = df3.groupby('year_month')['sales'].sum().reset_index()

# Vẽ biểu đồ
plt.figure(figsize=(14, 5))
sns.lineplot(data=monthly_sales, x='year_month', y='sales', marker='o')
plt.title("Tổng doanh thu theo thời gian (theo tháng)")
plt.xlabel("Tháng")
plt.ylabel("Tổng doanh thu")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# Chuyển cột ngày về datetime
df1['date'] = pd.to_datetime(df1['date'], errors='coerce')
df3['date'] = pd.to_datetime(df3['date'], errors='coerce')

# Tổng doanh thu theo ngày
daily_sales = df3.groupby('date')['sales'].sum().reset_index()

# Merge thông tin shops_closed
daily_sales = daily_sales.merge(df1[['date', 'shops_closed']], on='date', how='left')

# Vẽ biểu đồ
plt.figure(figsize=(15, 5))
sns.lineplot(data=daily_sales, x='date', y='sales', label='Doanh thu hằng ngày')

# Highlight những ngày đóng cửa
closed_days = daily_sales[daily_sales['shops_closed'] == 1]
plt.scatter(closed_days['date'], closed_days['sales'], color='red', label='Shops Closed', zorder=5)

plt.title(" Doanh thu theo thời gian (highlight các ngày shops_closed)")
plt.xlabel("Ngày")
plt.ylabel("Tổng doanh thu")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Đảm bảo cột ngày đúng định dạng
df3['date'] = pd.to_datetime(df3['date'])

# Tạo cột 'year_month' để phân tích theo tháng
df3['year_month'] = df3['date'].dt.to_period('M').astype(str)  

# Hiển thị tổng total_orders từng tháng
monthly_orders = df3.groupby('year_month')['total_orders'].sum().sort_index()

# Hiển thị top 5 tháng đầu tiên để kiểm tra
print(monthly_orders.head())

# Vẽ lại biểu đồ
plt.figure(figsize=(14, 5))
monthly_orders.plot(marker='o')
plt.title(' Tổng số đơn hàng theo tháng (toàn hệ thống)')
plt.xlabel('Tháng')
plt.ylabel('Tổng đơn hàng (tính theo dữ liệu gốc)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# Chuyển cột ngày về datetime
df1['date'] = pd.to_datetime(df1['date'], errors='coerce')
df3['date'] = pd.to_datetime(df3['date'], errors='coerce')

# Tổng số đơn hàng theo ngày
daily_orders = df3.groupby('date')['total_orders'].sum().reset_index()

# Merge thông tin shops_closed
daily_orders = daily_orders.merge(df1[['date', 'shops_closed']], on='date', how='left')

# Vẽ biểu đồ
plt.figure(figsize=(15, 5))
sns.lineplot(data=daily_orders, x='date', y='total_orders', label='Số đơn hàng hằng ngày')

# Highlight những ngày đóng cửa
closed_days = daily_orders[daily_orders['shops_closed'] == 1]
plt.scatter(closed_days['date'], closed_days['total_orders'], color='red', label='Shops Closed', zorder=5)

plt.title("Số đơn hàng theo thời gian (highlight các ngày shops_closed)")
plt.xlabel("Ngày")
plt.ylabel("Tổng số đơn hàng")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




cols_to_plot = ['total_orders', 'sales', 'availability', 'sell_price_main']

for col in cols_to_plot:
    plt.figure(figsize=(8, 4))
    sns.histplot(df3[col], bins=50, kde=True, color='steelblue')
    plt.title(f" Phân phối biến: {col}")
    plt.xlabel(col)
    plt.ylabel("Tần suất")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# Đếm tổng số hàng hóa (tính cả trùng lặp) theo kho
total_items_per_warehouse = df2.groupby('warehouse')['name'].count().reset_index()
total_items_per_warehouse.columns = ['warehouse', 'total_products']

# Hiển thị bảng
print(total_items_per_warehouse)

# Vẽ biểu đồ
plt.figure(figsize=(10,5))
sns.barplot(data=total_items_per_warehouse, x='warehouse', y='total_products', palette='crest')
plt.title(" Tổng số lượng tất cả các mặt hàng theo từng kho")
plt.xlabel("Kho")
plt.ylabel("Tổng số sản phẩm (bao gồm trùng lặp)")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# Đảm bảo date đúng format
df3['date'] = pd.to_datetime(df3['date'])
df1['date'] = pd.to_datetime(df1['date'])

# Merge sales_train với calendar
merged = df3.merge(df1, on=['date', 'warehouse'], how='left')

# Merge tiếp với inventory để lấy thông tin ngành hàng
merged = merged.merge(df2[['unique_id', 'name', 'L1_category_name_en', 'L2_category_name_en']], on='unique_id', how='left')


top_cats = merged.groupby('L1_category_name_en')['sales'].sum().sort_values(ascending=False)
top_cats.plot(kind='bar', figsize=(10,4), title='Top ngành hàng theo doanh thu')
plt.ylabel("Tổng doanh thu")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()



warehouse_sales = merged.groupby('warehouse')['sales'].sum().sort_values(ascending=False)
warehouse_sales.plot(kind='bar', title='Tổng doanh thu theo kho', figsize=(10,4))
plt.ylabel("Doanh thu")
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()


merged['month'] = merged['date'].dt.to_period('M').astype(str)
monthly_sales = merged.groupby(['month', 'warehouse'])['sales'].sum().reset_index()
pivot_sales = monthly_sales.pivot(index='month', columns='warehouse', values='sales').fillna(0)
pivot_sales.plot(marker='o', figsize=(14,5), title='Doanh số theo tháng và kho')
plt.xticks(rotation=45)
plt.ylabel("Doanh thu")
plt.grid(True)
plt.tight_layout()
plt.show()


# Tổng hợp đơn hàng theo tháng và kho
monthly_orders_warehouse = df3.groupby(['year_month', 'warehouse'])['total_orders'].sum().reset_index()

# Pivot để vẽ biểu đồ line nhiều kho
pivot_orders = monthly_orders_warehouse.pivot(index='year_month', columns='warehouse', values='total_orders').fillna(0)

# Vẽ biểu đồ line nhiều kho
plt.figure(figsize=(16, 6))
pivot_orders.plot(marker='o')
plt.title(' Số lượng đơn hàng theo tháng cho từng kho')
plt.xlabel('Tháng')
plt.ylabel('Tổng đơn hàng')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend(title='Warehouse')
plt.tight_layout()
plt.show()



df3['date'] = pd.to_datetime(df3['date'])
df3['month'] = df3['date'].dt.month
df3['year'] = df3['date'].dt.year

# Tổng hợp doanh thu và đơn hàng theo tháng
monthly_stats = df3.groupby('month')[ 'total_orders'].mean()

# Vẽ biểu đồ
monthly_stats.plot(kind='bar', figsize=(10, 4))
plt.title(" Đơn hàng trung bình theo tháng")
plt.ylabel("Giá trị trung bình")
plt.xlabel("Tháng")
plt.grid(axis='y')
plt.tight_layout()
plt.show()



# Đảm bảo 'date' là datetime
df1['date'] = pd.to_datetime(df1['date'])
df3['date'] = pd.to_datetime(df3['date'])

# Merge sales với calendar để lấy thông tin ngày lễ
merged = df3.merge(df1[['date', 'holiday']], on='date', how='left')

# Phân tích doanh thu trung bình 
holiday_stats = merged.groupby('holiday')['sales'].mean()

# Vẽ biểu đồ
holiday_stats.plot(kind='bar', title=" Doanh thu trung bình: Ngày thường vs Ngày lễ", color=['#66c2a5', '#fc8d62'])
plt.ylabel("Doanh thu trung bình")
plt.xticks([0, 1], ['Không lễ', 'Ngày lễ'], rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()



# Phân loại availability thành nhóm (ví dụ: thiếu, đầy đủ)
df3['availability_status'] = pd.cut(df3['availability'], bins=[-0.1, 0.5, 0.99, 1.0], labels=['Thiếu', 'Gần đủ', 'Đầy đủ'])

# Trung bình sales theo availability
avail_stats = df3.groupby('availability_status')['sales'].mean()

# Vẽ biểu đồ
avail_stats.plot(kind='bar', color='orange', title=" Doanh số theo mức độ sẵn có")
plt.ylabel("Doanh số trung bình")
plt.grid(axis='y')
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,4))
sns.scatterplot(data=df3, x='sell_price_main', y='sales', alpha=0.2)
plt.title(" Giá bán vs Doanh thu")
plt.xlabel("Giá bán chính")
plt.ylabel("Doanh thu")
plt.grid(True)
plt.tight_layout()
plt.show()


# Merge df3 với df2 để lấy tên sản phẩm
df3 = df3.merge(df2[['unique_id', 'name']], on='unique_id', how='left')
# Tính giá trung bình theo sản phẩm
avg_price_by_product = df3.groupby('name')['sell_price_main'].mean().sort_values(ascending=False).head(50)

# Vẽ biểu đồ
plt.figure(figsize=(12,5))
avg_price_by_product.plot(kind='bar')
plt.title("Giá bán trung bình theo sản phẩm (Top 50)")
plt.ylabel("Giá trung bình")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.show()



# Gộp tên ngành hàng từ inventory
df3 = df3.merge(df2[['unique_id', 'L1_category_name_en']], on='unique_id', how='left')

# Vẽ boxplot
plt.figure(figsize=(14,6))
sns.boxplot(data=df3, x='L1_category_name_en', y='sell_price_main')
plt.xticks(rotation=45, ha='right')
plt.title(" Phân phối giá bán theo ngành hàng (L1)")
plt.ylabel("Giá bán")
plt.xlabel("Ngành hàng")
plt.tight_layout()
plt.show()




# Merge holiday thông tin
df1['date'] = pd.to_datetime(df1['date'], errors='coerce')
df3['date'] = pd.to_datetime(df3['date'], errors='coerce')

df_holiday = df3.merge(df1[['date', 'holiday']], on='date', how='left')

# Thay nhãn 0/1 thành tên dễ hiểu
df_holiday['holiday_label'] = df_holiday['holiday'].map({0: 'Ngày thường', 1: 'Ngày lễ'})

# Vẽ histogram phân phối SALES
plt.figure(figsize=(14, 5))


# Vẽ histogram phân phối TOTAL ORDERS
plt.subplot(1, 2, 1)
sns.histplot(data=df_holiday, x='total_orders', hue='holiday_label', bins=50, kde=True, stat="density", common_norm=False)
plt.title(" Phân phối số đơn hàng: Ngày thường vs Ngày lễ")
plt.xlabel("Số đơn hàng")

plt.tight_layout()
plt.show()



# Chắc chắn có cột 'unique_id' trong df_holiday
if 'unique_id' not in df_holiday.columns:
    # Nếu thiếu thì merge lại từ df3
    df_holiday = df_holiday.merge(df3[['date', 'unique_id']], on='date', how='left')
if 'L1_category_name_en' in df_holiday.columns:
    df_holiday.drop(columns=['L1_category_name_en'], inplace=True)

# Merge thêm L1_category_name_en từ inventory
df_holiday = df_holiday.merge(df2[['unique_id', 'L1_category_name_en']], on='unique_id', how='left')

# Tính tổng doanh số theo ngành hàng và ngày lễ
sales_by_category = df_holiday.groupby(['L1_category_name_en', 'holiday_label'])['sales'].sum().reset_index()

# Vẽ biểu đồ
plt.figure(figsize=(14, 6))
sns.barplot(data=sales_by_category, x='L1_category_name_en', y='sales', hue='holiday_label', palette='Set3')
plt.title(" Tổng doanh số theo ngành hàng và ngày lễ")
plt.ylabel("Tổng doanh số")
plt.xlabel("Ngành hàng")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.show()



# Tính tổng doanh số theo kho và ngày lễ
sales_by_warehouse = df_holiday.groupby(['warehouse', 'holiday_label'])['sales'].sum().reset_index()

# Vẽ biểu đồ
plt.figure(figsize=(12, 5))
sns.barplot(data=sales_by_warehouse, x='warehouse', y='sales', hue='holiday_label', palette='Set2')
plt.title(" Tổng doanh số theo kho hàng và ngày lễ")
plt.ylabel("Tổng doanh số")
plt.xlabel("Kho hàng")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()



# Chuyển ngày về datetime nếu chưa
df1['date'] = pd.to_datetime(df1['date'], errors='coerce')
df3['date'] = pd.to_datetime(df3['date'], errors='coerce')

# Merge calendar vào sales
df_merged = df3.merge(df1[['date', 'holiday']], on='date', how='left')

# Tạo nhãn dễ đọc
df_merged['holiday_label'] = df_merged['holiday'].map({0: 'Ngày thường', 1: 'Ngày lễ'})

# Tính doanh thu trung bình theo warehouse & loại ngày
avg_sales = df_merged.groupby(['warehouse', 'holiday_label'])['sales'].mean().reset_index()

# Vẽ biểu đồ
plt.figure(figsize=(12, 6))
sns.barplot(data=avg_sales, x='warehouse', y='sales', hue='holiday_label', palette='Set2')
plt.title("Doanh thu trung bình theo kho và loại ngày")
plt.xlabel("Kho")
plt.ylabel("Doanh thu trung bình")
plt.xticks(rotation=45)
plt.legend(title="Loại ngày")
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# Chuyển date sang datetime nếu chưa làm
df1['date'] = pd.to_datetime(df1['date'], errors='coerce')

# Thêm cột tháng
df1['month_n'] = df1['date'].dt.month

# Tổng hợp dữ liệu theo kho và tháng
grouped_df = df1.groupby(['warehouse', 'month_n'])[
    ['holiday', 'shops_closed', 'winter_school_holidays', 'school_holidays']
].sum().reset_index()

# Danh sách các kho
warehouses = grouped_df['warehouse'].unique()

# Thiết lập khung vẽ phù hợp
import matplotlib.pyplot as plt

rows = (len(warehouses) + 2) // 3
fig, axes = plt.subplots(nrows=rows, ncols=3, figsize=(18, rows * 4))
axes = axes.flatten()

# Màu cho từng biến
colors = ['royalblue', 'darkorange', 'mediumseagreen', 'crimson']

# Vẽ từng warehouse
for i, warehouse in enumerate(warehouses):
    data = grouped_df[grouped_df['warehouse'] == warehouse].set_index('month_n')
    data = data[['holiday', 'shops_closed', 'winter_school_holidays', 'school_holidays']]

    data.plot(kind='bar', stacked=True, ax=axes[i], color=colors)
    axes[i].set_title(f'{warehouse}', fontsize=16)
    axes[i].set_xlabel('Tháng')
    axes[i].set_ylabel('Số ngày')
    axes[i].legend(loc='upper right', fontsize=10)
    axes[i].tick_params(axis='x', rotation=0)
    axes[i].set_xticks(range(0, 12))

# Xoá subplot thừa
for j in range(len(warehouses), len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Phân bố ngày nghỉ lễ và nghỉ học theo tháng và từng kho", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



# Nếu đã tồn tại cột này từ lần merge trước, thì xóa đi
if 'L1_category_name_en' in df3.columns:
    df3.drop(columns=['L1_category_name_en'], inplace=True)

# Merge lại sau khi đã xử lý
df3 = df3.merge(df2[['unique_id', 'L1_category_name_en']], on='unique_id', how='left')


# Tính giá bán trung bình theo (kho, ngành hàng)
pivot_category_price = df3.groupby(['warehouse', 'L1_category_name_en'])['sell_price_main'].mean().unstack()

# Vẽ heatmap
plt.figure(figsize=(14, 6))
sns.heatmap(pivot_category_price, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title(" Giá bán trung bình theo ngành hàng và kho")
plt.xlabel("Ngành hàng")
plt.ylabel("Kho")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



# Đảm bảo định dạng ngày giống nhau
df1['date'] = pd.to_datetime(df1['date'], errors='coerce')
df3['date'] = pd.to_datetime(df3['date'], errors='coerce')

# Merge các cột calendar vào sales
df_merged = df3.merge(
    df1[['date', 'holiday', 'shops_closed', 'winter_school_holidays', 'school_holidays']],
    on='date', how='left'
)

# Chọn các cột số để tính tương quan
corr_cols = ['holiday', 'shops_closed', 'winter_school_holidays', 'school_holidays',
             'sales', 'total_orders', 'availability',
             'type_0_discount', 'type_1_discount', 'type_2_discount', 'type_3_discount', 'type_4_discount', 
             'type_5_discount', 'type_6_discount']
             
corr_df = df_merged[corr_cols].dropna()

# Tính hệ số tương quan
correlation_matrix = corr_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title(" Ma trận tương quan giữa ngày lễ và các yếu tố bán hàng")
plt.tight_layout()
plt.show()


