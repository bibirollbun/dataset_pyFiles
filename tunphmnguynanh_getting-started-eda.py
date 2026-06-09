%%capture
# Install relevant libraries
!pip install geopandas folium 


# Import libraries
import pandas as pd
import numpy as np
import random
import os
from tqdm.notebook import tqdm

import geopandas as gpd
from shapely.geometry import Point
import folium

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
pd.options.display.float_format = '{:.5f}'.format
pd.options.display.max_rows = None

%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

# You can ignore the Shapely GEOS warning :-)


# Set seed for reproducability
SEED = 2023
random.seed(SEED)
np.random.seed(SEED)


DATA_PATH = '/kaggle/input/playground-series-s3e20'
# Load files
train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
samplesubmission = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))

# Preview train dataset
train.head()


# Preview test dataset
test.head()


# Preview sample submission file
samplesubmission.head()


# Check size and shape of datasets
train.shape, test.shape, samplesubmission.shape


# Train to test sets ratio
(test.shape[0]) / (train.shape[0] + test.shape[0])


# Train statistical summary
train.describe(include = 'all')


# Target variable distribution
sns.set_style('darkgrid')
plt.figure(figsize = (13, 7))
sns.histplot(train.emission, kde = True, bins = 15)
plt.title('Target variable distribution', y = 1.02, fontsize = 15)
display(plt.show(), train.emission.skew())


# Plotting boxplot for the CO2 emissions
sns.set_style('darkgrid')
plt.figure(figsize = (13, 7))
sns.boxplot(train.emission)
plt.title('Boxplot showing CO2 emission outliers', y = 1.02, fontsize = 15)  
plt.show()


# Combine train and test for easy visualisation
train_coords = train.drop_duplicates(subset = ['latitude', 'longitude'])
test_coords = test.drop_duplicates(subset = ['latitude', 'longitude'])
train_coords['set_type'], test_coords['set_type'] = 'train', 'test'

all_data = pd.concat([train_coords, test_coords], ignore_index = True)
# Create point geometries

geometry = gpd.points_from_xy(all_data.longitude, all_data.latitude)
geo_df = gpd.GeoDataFrame(
    all_data[["latitude", "longitude", "set_type"]], geometry=geometry
)

# Preview the geopandas df
geo_df.head()


# Create a canvas to plot your map on
all_data_map = folium.Map(prefer_canvas=True)

# Create a geometry list from the GeoDataFrame
geo_df_list = [[point.xy[1][0], point.xy[0][0]] for point in geo_df.geometry]

# Iterate through list and add a marker for each volcano, color-coded by its type.
i = 0
for coordinates in geo_df_list:
    # assign a color marker for the type set
    if geo_df.set_type[i] == "train":
        type_color = "green"
    elif geo_df.set_type[i] == "test":
        type_color = "orange"

    # Place the markers 
    all_data_map.add_child(
        folium.CircleMarker(
            location=coordinates,
            radius = 1,
            weight = 4,
            zoom =10,
            popup= 
            "Set: " + str(geo_df.set_type[i]) + "<br>"
            "Coordinates: " + str([round(x, 2) for x in geo_df_list[i]]),
            color =  type_color),
        )
    i = i + 1
all_data_map.fit_bounds(all_data_map.get_bounds())
all_data_map


# Check for missing values
train.isnull().sum().any(), test.isnull().sum().any() 


# Plot missing values in train set
ax = train.isna().sum().sort_values(ascending = False)[:15][::-1].plot(kind = 'barh', figsize = (9, 10))
plt.title('Percentage of Missing Values Per Column in Train Set', fontdict={'size':15})
for p in ax.patches:
    percentage ='{:,.0f}%'.format((p.get_width()/train.shape[0])*100)
    width, height =p.get_width(),p.get_height()
    x=p.get_x()+width+0.02
    y=p.get_y()+height/2
    ax.annotate(percentage,(x,y))


# Check for duplicates
train.duplicated().any(), test.duplicated().any()


# Year countplot
plt.figure(figsize = (14, 7))
sns.countplot(x = 'year', data = train)
plt.title('Year count plot - Train')
plt.show()


# Year countplot
plt.figure(figsize = (4, 7))
sns.countplot(x = 'year', data = test)
plt.title('Year count plot - Test')
plt.show()


# Week countplot
plt.figure(figsize = (14, 7))
sns.countplot(x = 'week_no', data = train)
plt.title('Week count plot - Train')
plt.show()


train.drop_duplicates(subset = ['year', 'week_no']).groupby(['year'])[['week_no']].count()


# Top 20 correlated features to the target
top20_corrs = abs(train.corr()['emission']).sort_values(ascending = False).head(20)
top20_corrs


# Quantify correlations between features
corr = train[list(top20_corrs.index)].corr()
plt.figure(figsize = (13, 8))
sns.heatmap(corr, cmap='RdYlGn', annot = True, center = 0)
plt.title('Correlogram', fontsize = 15, color = 'darkgreen')
plt.show()


# Sample a unique location and visualize its emissions across the years
train.latitude, train.longitude = round(train.latitude, 2), round(train.longitude, 2)
sample_loc = train[(train.latitude == -0.510) & (train.longitude == 29.290)]

# Plot a line plot
sns.set_style('darkgrid')
fig, axes = plt.subplots(nrows = 3, ncols = 1, figsize = (13, 10))
fig.suptitle('Co2 emissions for location lat -23.75 lon 28.75', y=1.02, fontsize = 15)

for ax, data, year, color, in zip(axes.flatten(), sample_loc, sample_loc.year.unique(), ['#882255','#332288', '#999933' , 'orangered']):
  df = sample_loc[sample_loc.year == year]
  sns.lineplot(x=df.week_no,y= df.emission, ax = ax, label = year, color = color)
plt.legend()
plt.tight_layout()


# Loại bỏ các cột có tỷ lệ thiếu >90%
missing_ratio = train.isna().mean()
columns_to_drop = missing_ratio[missing_ratio > 0.9].index
train = train.drop(columns=columns_to_drop)
test = test.drop(columns=columns_to_drop)


# Điền các giá trị thiếu còn lại bằng trung vị
for col in train.columns:
    if train[col].isna().any():
        train[col].fillna(train[col].median(), inplace=True)
        if col in test.columns:
            test[col].fillna(train[col].median(), inplace=True)


# 2. Xử lý ngoại lai trong biến mục tiêu
# Áp dụng biến đổi log cho biến mục tiêu (emission)
train['emission'] = np.log1p(train['emission'])  # log1p để xử lý giá trị bằng 0

# 3. Kỹ thuật đặc trưng
# Mã hóa chu kỳ cho week_no
train['sin_week'] = np.sin(2 * np.pi * train['week_no'] / 52)
train['cos_week'] = np.cos(2 * np.pi * train['week_no'] / 52)
test['sin_week'] = np.sin(2 * np.pi * test['week_no'] / 52)
test['cos_week'] = np.cos(2 * np.pi * test['week_no'] / 52)


# 4. Lựa chọn đặc trưng
# Lấy top 20 đặc trưng có tương quan cao nhất với emission từ EDA
top20_corrs = abs(train.corr()['emission']).sort_values(ascending=False).head(20)
selected_features = top20_corrs.index.tolist()
# Loại bỏ 'emission' khỏi danh sách đặc trưng (nếu có)
if 'emission' in selected_features:
    selected_features.remove('emission')
# Đảm bảo các đặc trưng kỹ thuật được bao gồm nếu cần
additional_engineered_features = ['sin_week', 'cos_week', 'month', 'quarter']
for feat in additional_engineered_features:
    if feat not in selected_features:
        selected_features.append(feat)
# Giới hạn ở 20 đặc trưng nếu danh sách vượt quá
selected_features
# In danh sách đặc trưng được chọn để kiểm tra
print("Các đặc trưng được chọn:", selected_features)



# Loại bỏ cột không cần thiết
train = train.drop(columns=['ID_LAT_LON_YEAR_WEEK'])
test_ids = test['ID_LAT_LON_YEAR_WEEK']  # Lưu để nộp bài
test = test.drop(columns=['ID_LAT_LON_YEAR_WEEK'])


from sklearn.preprocessing import StandardScaler
# 5. Chuẩn bị đặc trưng và mục tiêu
# Chỉ giữ các cột có trong selected_features và tồn tại trong tập train/test
selected_features = [col for col in selected_features if col in train.columns and col in test.columns]
X = train[selected_features]
y = train['emission']
X_test = test[selected_features]

# Chuẩn hóa các đặc trưng số
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


# --- Huấn luyện mô hình ---

# Chia dữ liệu
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2023)

# Huấn luyện RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=2023, n_jobs=-1)
model.fit(X_train, y_train)

# Đánh giá trên tập xác thực
y_pred_val = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f'RMSE trên tập xác thực: {rmse:.5f}')


# --- Dự đoán trên tập kiểm tra ---

# Dự đoán trên tập kiểm tra
y_pred_test = model.predict(X_test)

# Đảo ngược biến đổi log
y_pred_test = np.expm1(y_pred_test)  # expm1 để đảo ngược log1p

# Tạo file nộp bài
submission = pd.DataFrame({
    'ID_LAT_LON_YEAR_WEEK': test_ids,
    'emission': y_pred_test
})
submission.to_csv('submission.csv', index=False)
print("File nộp bài đã được tạo thành công!")

