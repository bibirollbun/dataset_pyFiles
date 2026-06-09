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


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

file_path_train = "/kaggle/input/rossmann-store-sales/train.csv"  # Cập nhật đường dẫn file
df_train = pd.read_csv(file_path_train)

# Hiển thị 5 dòng đầu tiên để kiểm tra
df_train.head()


file_path_store = "/kaggle/input/rossmann-store-sales/store.csv"  # Cập nhật đường dẫn file
df_store = pd.read_csv(file_path_store)

# Hiển thị 5 dòng đầu tiên để kiểm tra
df_store.head()


df_merged = pd.merge(df_train, df_store, on='Store', how='left')  # 'inner', 'left', 'right', 'outer'

df_merged.head()



df_merged.shape


df_merged.columns


df_merged.isnull().sum()


df_store.isnull().sum()


df_train.isnull().sum()


df_merged.describe()


df_merged.info()


df_train.describe()


df_train.info()


df_store.describe()


df_store.info()


df_merged.duplicated().sum()



import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))
sns.histplot(df_merged['Sales'], bins=50, kde=True)
plt.xlabel('Sales')
plt.ylabel('Count')
plt.title('Distribution of Sales')
plt.show()



# Danh sách các biến phân loại cần phân tích đơn biến (không chú thích cho Promo, Promo2)
categorical_vars = ['StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment', 'Open']
# Chuẩn hóa giá trị trong cột StateHoliday: chuyển tất cả giá trị 0 (số) thành chuỗi '0'
df_merged['StateHoliday'] = df_merged['StateHoliday'].astype(str).replace('0.0', '0')
# Chú thích giải thích cho từng biến (loại bỏ Promo và Promo2)
annotations = {
    'StateHoliday': ['a: Public holiday', 'b: Easter', 'c: Christmas', '0: None'],
    'SchoolHoliday': ['0: Không có', '1: Có (trường học nghỉ)'],
    'StoreType': ['a: Mô hình A', 'b: Mô hình B', 'c: Mô hình C', 'd: Mô hình D'],
    'Assortment': ['a: Basic', 'b: Extra', 'c: Extended'],
    'Open': ['0: Cửa hàng đóng cửa', '1: Cửa hàng mở cửa']
}

# Tạo lưới biểu đồ: 2 hàng x 3 cột (có 5 biến nên tạo 2x3 để đẹp)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()  # Chuyển về danh sách để dễ truy cập

# Vẽ biểu đồ cho từng biến phân loại
for i, var in enumerate(categorical_vars):
    ax = axes[i]
    
    # Vẽ biểu đồ và lấy danh sách màu
    bars = sns.countplot(x=var, data=df_merged, ax=ax, palette="viridis")
    colors = [bar.get_facecolor() for bar in bars.patches]  # Lấy màu từng cột
    
    ax.set_xlabel(var)
    ax.set_ylabel("Số lượng")
    
    # Chỉ thêm chú thích nếu biến đó có trong annotations
    if var in annotations:
        annotation_text = "\n".join(annotations[var])
        
        # Thêm chú thích vào góc trên bên phải của biểu đồ
        ax.text(0.98, 0.95, annotation_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))

# Xóa ô trống nếu có (vì ta đang dùng 2x3 nhưng chỉ có 5 biểu đồ)
for j in range(len(categorical_vars), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



# Bivariate Analysis

import matplotlib.pyplot as plt
import seaborn as sns

# Tạo một lưới biểu đồ với 2 hàng và 3 cột
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Doanh thu trung bình theo StateHoliday (chỉnh lại 4 class)
sns.boxplot(x='StateHoliday', y='Sales', data=df_merged, order=['0', 'a', 'b', 'c'], ax=axes[0, 0], palette="viridis")
axes[0, 0].set_title("Doanh thu trung bình theo StateHoliday")
axes[0, 0].set_xlabel("StateHoliday (0: không có, a: public, b: Easter, c: Christmas)")
axes[0, 0].set_ylabel("Doanh thu trung bình")

# 2. Doanh thu trung bình theo SchoolHoliday
sns.boxplot(x='SchoolHoliday', y='Sales', data=df_merged, ax=axes[0, 1], palette="viridis")
axes[0, 1].set_title("Doanh thu trung bình theo SchoolHoliday")
axes[0, 1].set_xlabel("SchoolHoliday (0: không có, 1: có)")
axes[0, 1].set_ylabel("Doanh thu trung bình")

# 3. Doanh thu trung bình theo Assortment
sns.boxplot(x='Assortment', y='Sales', data=df_merged, ax=axes[0, 2], palette="viridis")
axes[0, 2].set_title("Doanh thu trung bình theo Assortment")
axes[0, 2].set_xlabel("Assortment (a: cơ bản, b: extra, c: extended)")
axes[0, 2].set_ylabel("Doanh thu trung bình")

# 4. Doanh thu trung bình theo Open
sns.boxplot(x='Open', y='Sales', data=df_merged, ax=axes[1, 0], palette="viridis")
axes[1, 0].set_title("Doanh thu trung bình theo Open")
axes[1, 0].set_xlabel("Open (0: đóng cửa, 1: mở cửa)")
axes[1, 0].set_ylabel("Doanh thu trung bình")

# 5. Doanh thu trung bình theo DayOfWeek
sns.boxplot(x='DayOfWeek', y='Sales', data=df_merged, ax=axes[1, 1], palette="viridis")
axes[1, 1].set_title("Doanh thu trung bình theo DayOfWeek")
axes[1, 1].set_xlabel("DayOfWeek (1: Monday, ..., 7: Sunday)")
axes[1, 1].set_ylabel("Doanh thu trung bình")

# Ẩn ô trống còn lại
axes[1, 2].axis("off")

plt.tight_layout()
plt.show()


### StateHoliday: Doanh thu thấp vào ngày lễ (a = public holiday, b = Easter holiday, c = Christmas) do hầu hết cửa hàng đóng cửa. SchoolHoliday: Ảnh hưởng nhỏ, doanh thu nhỉnh hơn 1 chút khi nghỉ học nhưng không quá đáng kể. Assortment: Cửa hàng loại b có doanh thu cao nhất, cho thấy danh mục sản phẩm mở rộng thu hút khách hơn. Open: Cửa hàng mở (1) có doanh thu cao, đóng (0) gần như bằng 0.


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_merged["Customers"], y=df_merged["Sales"], alpha=0.5)

plt.xlabel("Số lượng khách hàng")
plt.ylabel("Doanh thu (Sales)")
plt.title("Mối quan hệ giữa số khách hàng và doanh thu")
plt.show()



plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_merged['CompetitionDistance'], y=df_merged['Sales'], alpha=0.5)
plt.xlabel("Khoảng cách đến đối thủ (CompetitionDistance)")
plt.ylabel("Doanh thu (Sales)")
plt.title("Mối quan hệ giữa khoảng cách đối thủ và doanh thu")
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Boxplot cho Promo vs Sales
sns.boxplot(x="Promo", y="Sales", data=df_merged, ax=axes[0], palette="viridis")
axes[0].set_title("Ảnh hưởng của Promo đến Sales")

# Boxplot cho Promo2 vs Sales
sns.boxplot(x="Promo2", y="Sales", data=df_merged, ax=axes[1], palette="magma")
axes[1].set_title("Ảnh hưởng của Promo2 đến Sales")

plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# Tạo figure với 2 hàng, 2 cột
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# 1. Customers vs Promo (Boxplot)
sns.boxplot(x='Promo', y='Customers', data=df_merged, ax=axes[0, 0], palette="viridis")
axes[0, 0].set_title("Số lượng khách hàng theo Promo")
axes[0, 0].set_xlabel("Promo (0: không, 1: có)")
axes[0, 0].set_ylabel("Số lượng khách hàng")

# 2. Customers vs StoreType (Boxplot)
sns.boxplot(x='StoreType', y='Customers', data=df_merged, ax=axes[0, 1], palette="viridis")
axes[0, 1].set_title("Số lượng khách hàng theo loại cửa hàng")
axes[0, 1].set_xlabel("StoreType (a, b, c, d)")
axes[0, 1].set_ylabel("Số lượng khách hàng")

# 3. CompetitionDistance vs Customers (Scatterplot)
sns.scatterplot(x='CompetitionDistance', y='Customers', data=df_merged, ax=axes[1, 0], alpha=0.5)
axes[1, 0].set_title("Mối quan hệ giữa khoảng cách đối thủ và số khách hàng")
axes[1, 0].set_xlabel("Khoảng cách đến đối thủ (mét)")
axes[1, 0].set_ylabel("Số lượng khách hàng")

# 4. Promo vs StateHoliday (Clustered Bar Chart - Chỉ có 4 nhóm 0, a, b, c)
df_merged['StateHoliday'] = df_merged['StateHoliday'].astype(str)  # Chuẩn hóa dữ liệu tránh lỗi 2 cột '0'
sns.countplot(x='StateHoliday', hue='Promo', data=df_merged, ax=axes[1, 1], palette="viridis")
axes[1, 1].set_title("Tần suất khuyến mãi theo ngày lễ")
axes[1, 1].set_xlabel("StateHoliday (0: không, a: public, b: Easter, c: Christmas)")
axes[1, 1].set_ylabel("Số lượng cửa hàng")

plt.tight_layout()  # Căn chỉnh layout để không bị chồng chéo
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Giả sử df là DataFrame chứa dữ liệu của bạn

# Chỉ giữ các biến số học để tính toán ma trận tương quan
numeric_cols = ['Sales', 'Customers', 'Open', 'CompetitionDistance', 'Promo', 'Promo2']

# Tính toán ma trận tương quan
correlation_matrix = df_merged[numeric_cols].corr()

# Vẽ heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numeric Variables')
plt.show()


