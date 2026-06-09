import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from colorama import Fore, Style
from sklearn.linear_model import LinearRegression # Giữ lại phần này cho context mô hình gốc

# Cài đặt tùy chỉnh hiển thị và đồ thị
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [15, 6]
plt.rcParams['font.size'] = 12

# --- TẢI DỮ LIỆU ---
# LƯU Ý: Đảm bảo đường dẫn file là chính xác trong môi trường Kaggle của bạn.
try:
    ci = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/city_indexes.csv') 
    csi = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/city_search_index.csv') 
    sp = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/sector_POI.csv') 
    
    train_lt = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/land_transactions.csv')
    train_ltns = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/land_transactions_nearby_sectors.csv')
    train_pht = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions.csv')
    train_phtns = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions_nearby_sectors.csv')
    train_nht = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv')
    train_nhtns = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions_nearby_sectors.csv')
    test = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/test.csv')
except FileNotFoundError as e:
    print(Fore.RED + f"LỖI: Không tìm thấy file tại đường dẫn. Vui lòng kiểm tra lại cấu trúc file trên Kaggle/máy tính của bạn: {e}" + Style.RESET_ALL)
    raise

month_codes = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

if not test.empty:
    test_id = test.id.str.split('_', expand=True)
    test['month'] = test_id[0]
    test['sector'] = test_id[1]
    del test_id

# --- TIỀN XỬ LÝ DỮ LIỆU & TẠO BIẾN THỜI GIAN/SECTOR_ID ---
dataframes_to_process_time = [train_lt, train_ltns, train_pht, train_phtns, train_nht, train_nhtns, csi, test]
dataframes_to_process_sector = [train_lt, train_ltns, train_pht, train_phtns, train_nht, train_nhtns, sp, test]

for df in [d for d in dataframes_to_process_time if not d.empty]:
    if 'month' in df.columns:
        df['year'] = df.month.str.slice(0, 4).astype(int)
        df['month_str'] = df.month.str.slice(5, None)
        df['month_num'] = df['month_str'].map(month_codes)
        df['time'] = (df['year'] - 2019) * 12 + df['month_num'] - 1 
        
        df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month_num'].astype(str) + '-01')
        df['quarter'] = df['date'].dt.quarter
        df['month_name'] = df['date'].dt.strftime('%b')
        
        df.drop(columns=['month_str', 'month_num'], inplace=True, errors='ignore')

for df in [d for d in dataframes_to_process_sector if not d.empty]:
    if 'sector' in df.columns:
        df['sector_id'] = df.sector.str.slice(7, None).astype(int)
        df.drop(columns=['sector'], inplace=True, errors='ignore')

# --- HỢP NHẤT DỮ LIỆU CHÍNH (FEATURE ENGINEERING) ---
print(Fore.BLUE + Style.BRIGHT + "\n--- DATA PREPARATION VÀ HỢP NHẤT DỮ LIỆU ---" + Style.RESET_ALL)
df_train = train_nht.copy()

# Xử lý Hợp nhất city_indexes (ci) - Sửa lỗi KeyError
ci_year_col = next((col for col in ['city_indicator_data_year', 'month'] if col in ci.columns), None)
if ci_year_col:
    ci.rename(columns={ci_year_col: 'year'}, inplace=True)
    ci['year'] = ci['year'].astype(int)
    
df_train = pd.merge(df_train, ci, on='year', how='left')

# Hợp nhất các chỉ số POI (sp)
df_train = pd.merge(df_train, sp, on='sector_id', how='left')

# Tạo dataframe tổng nhu cầu cho Time Series Analysis
total_demand = df_train.groupby('date')['amount_new_house_transactions'].sum().to_frame()

# --- PREP CHO LINEAR REGRESSION GỐC ---
amount_new_house_transactions = train_nht.set_index(['time', 'sector_id']).amount_new_house_transactions.unstack()
amount_new_house_transactions = amount_new_house_transactions.fillna(0)
if 95 not in amount_new_house_transactions.columns:
    amount_new_house_transactions[95] = 0
amount_new_house_transactions = amount_new_house_transactions[np.arange(1, 97)]

# Hiển thị đầu dữ liệu Wide Format
print(Fore.GREEN + "Dữ liệu giao dịch nhà mới (Wide Format) cho mô hình:" + Style.RESET_ALL)
print(amount_new_house_transactions.astype(int).head())


# --- ADVANCED EDA: PHÂN TÍCH PHÂN PHỐI VÀ BIẾN ĐỘNG ---
print(Fore.BLUE + Style.BRIGHT + "\n--- ADVANCED EDA: PHÂN TÍCH PHÂN PHỐI & BIẾN ĐỘNG ---" + Style.RESET_ALL)

# 1. Phân tích Phân phối Biến Mục tiêu
plt.figure(figsize=(15, 5))

# Biểu đồ 1: Phân phối gốc
plt.subplot(1, 2, 1)
sns.histplot(df_train['amount_new_house_transactions'], bins=50, kde=True, color='skyblue')
plt.title('Distribution of Target Variable (Amount - Raw)')
plt.xlabel('Amount (10k Yuan)')

# Biểu đồ 2: Phân phối sau khi log-transform
# Thêm 1 để tránh log(0)
df_train['log_amount'] = np.log1p(df_train['amount_new_house_transactions'])
plt.subplot(1, 2, 2)
sns.histplot(df_train['log_amount'], bins=30, kde=True, color='salmon')
plt.title('Distribution of Target Variable (Amount - Log Transformed)')
plt.xlabel('Log(1 + Amount)')
plt.show()

# 2. Phân tích Sector: Top 10 Sector theo Tổng Nhu cầu
sector_summary = df_train.groupby('sector_id')['amount_new_house_transactions'].sum().sort_values(ascending=False)

plt.figure(figsize=(15, 6))
# Top 10 Sector
plt.subplot(1, 2, 1)
sns.barplot(x=sector_summary.head(10).index, y=sector_summary.head(10).values, palette='viridis')
plt.title('Top 10 Sectors by Total Transaction Amount', fontsize=14)
plt.xlabel('Sector ID')
plt.ylabel('Total Amount (10k Yuan)')

# Bottom 10 Sector (có thể là 0)
plt.subplot(1, 2, 2)
sns.barplot(x=sector_summary.tail(10).index, y=sector_summary.tail(10).values, palette='plasma')
plt.title('Bottom 10 Sectors by Total Transaction Amount', fontsize=14)
plt.xlabel('Sector ID')
plt.ylabel('Total Amount (10k Yuan)')
plt.show()

# 3. Phân tích Giá (Price vs Area) - Thêm trực quan hóa nâng cao
# Tạo biến Price log và Area log để trực quan hóa mối quan hệ phi tuyến tính
df_train['log_price'] = np.log1p(df_train['price_new_house_transactions'])
df_train['log_area'] = np.log1p(df_train['area_new_house_transactions'])

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='log_area',
    y='log_price',
    data=df_train.sample(frac=0.3, random_state=42), # Sample để tránh quá tải
    alpha=0.6,
    color='purple'
)
plt.title('Scatter Plot: Log(Area) vs Log(Price) for New House Transactions', fontsize=14)
plt.xlabel('Log(1 + Total Area Transaction)')
plt.ylabel('Log(1 + Avg Price per sqm)')
plt.show()


# --- ADVANCED EDA: PHÂN TÍCH CHUỖI THỜI GIAN & TƯƠNG QUAN ---
print(Fore.BLUE + Style.BRIGHT + "\n--- ADVANCED EDA: PHÂN TÍCH CHUỖI THỜI GIAN VÀ TƯƠNG QUAN ---" + Style.RESET_ALL)

# 1. Phân tích Chuỗi Thời gian (Time Series Decomposition)
print(Fore.CYAN + "Phân tích Chuỗi Thời gian: Tổng Nhu cầu Thị trường" + Style.RESET_ALL)
ts = total_demand['amount_new_house_transactions']

try:
    decomposition = seasonal_decompose(ts, model='additive', period=12)

    fig = decomposition.plot()
    fig.set_size_inches(15, 8)
    plt.suptitle('Time Series Decomposition of Total New House Transactions', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()

    # Phân tích Tính Thời vụ theo Tháng
    print(Fore.CYAN + "Phân tích Tính Thời Vụ Theo Tháng (Seasonal Effect)" + Style.RESET_ALL)
    seasonal_component = decomposition.seasonal.to_frame(name='seasonal_effect')
    seasonal_component['month_name'] = seasonal_component.index.strftime('%b')

    month_order = [calendar.month_abbr[i] for i in range(1, 13)]

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x='month_name',
        y='seasonal_effect',
        data=seasonal_component,
        order=month_order,
        palette='Spectral'
    )
    plt.title('Seasonal Effect on Transaction Amount by Month')
    plt.xlabel('Month')
    plt.ylabel('Seasonal Effect (Transaction Amount)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.show()

except Exception as e:
    print(Fore.RED + f"Lỗi khi thực hiện Time Series Decomposition: {e}" + Style.RESET_ALL)


# 2. Ma Trận Tương Quan (Correlation Heatmap)
print(Fore.CYAN + "Ma Trận Tương Quan giữa các Tính năng Chính" + Style.RESET_ALL)
correlation_features = [
    'amount_new_house_transactions',
    'area_new_house_transactions',
    'price_new_house_transactions',
    'land_transaction_amount', 
    'amount_pre_owned_house_transactions', 
    'gdp_100m', 
    'per_capita_disposable_income_absolute_yuan', # Từ city_indexes
    'resident_population_dense', # Từ sector_POI
    'number_of_shops_dense', # Từ sector_POI
    'year',
    'quarter'
]

valid_features = [col for col in correlation_features if col in df_train.columns]
df_corr = df_train[valid_features].copy()
df_corr = df_corr.fillna(df_corr.median()) 

corr_matrix = df_corr.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    annot=True,          
    cmap='coolwarm',     
    fmt=".2f",           
    linewidths=.5,       
    linecolor='black',
    cbar_kws={'label': 'Correlation Coefficient'}
)
plt.title('Correlation Heatmap of Key Features (Advanced EDA)', fontsize=16)
plt.show()

# Trực quan hóa Tương quan với Target
print(Fore.CYAN + "Tương Quan của các Biến với Biến Mục Tiêu (Target)" + Style.RESET_ALL)
target_corr = corr_matrix['amount_new_house_transactions'].sort_values(ascending=False).drop('amount_new_house_transactions', errors='ignore')

plt.figure(figsize=(10, 5))
sns.barplot(x=target_corr.values, y=target_corr.index, palette='RdBu_r')
plt.title('Correlation with New House Transaction Amount (Target)', fontsize=14)
plt.xlabel('Correlation Coefficient')
plt.show()


# --- ADVANCED EDA: PHÂN TÍCH NÂNG CAO VỀ KINH TẾ VÀ POI ---
print(Fore.BLUE + Style.BRIGHT + "\n--- ADVANCED EDA: PHÂN TÍCH CHỈ SỐ KINH TẾ & POI ---" + Style.RESET_ALL)

# 1. Mối quan hệ giữa GDP và Tổng Nhu cầu
gdp_demand = df_train.groupby('year').agg({
    'gdp_100m': 'first', # GDP là chỉ số năm
    'amount_new_house_transactions': 'sum' # Tổng giao dịch theo năm
}).reset_index()

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.lineplot(x='year', y='amount_new_house_transactions', data=gdp_demand, marker='o', color='darkblue')
plt.title('Total Real Estate Demand Over Time', fontsize=14)
plt.ylabel('Total Amount (10k Yuan)')

plt.subplot(1, 2, 2)
# Phân tích tương quan giữa GDP và Nhu cầu BĐS
sns.regplot(x='gdp_100m', y='amount_new_house_transactions', data=gdp_demand, scatter_kws={'alpha':0.8}, line_kws={'color':'red'})
plt.title('Relationship between City GDP and Total Annual Demand', fontsize=14)
plt.xlabel('GDP (100 million yuan)')
plt.ylabel('Total Amount (10k Yuan)')
plt.show()

# 2. Tác động của Mật độ Dân số (POI Density) lên Giao dịch
# Chọn 2 tính năng mật độ quan trọng nhất
plt.figure(figsize=(15, 6))

# Tác động của Mật độ Dân số Cư trú
plt.subplot(1, 2, 1)
sns.scatterplot(
    x='resident_population_dense', 
    y='amount_new_house_transactions', 
    data=df_train.sample(frac=0.5, random_state=42), # Sample 50% dữ liệu
    alpha=0.3, 
    color='green'
)
plt.title('Demand vs Resident Population Density (by Month/Sector)', fontsize=14)
plt.xlabel('Resident Population Density')
plt.ylabel('Amount (10k Yuan)')

# Tác động của Mật độ Cơ sở Thương mại (Number of Shops)
plt.subplot(1, 2, 2)
sns.scatterplot(
    x='number_of_shops_dense', 
    y='amount_new_house_transactions', 
    data=df_train.sample(frac=0.5, random_state=42),
    alpha=0.3, 
    color='orange'
)
plt.title('Demand vs Shop Density (by Month/Sector)', fontsize=14)
plt.xlabel('Number of Shops Density')
plt.ylabel('Amount (10k Yuan)')
plt.show()


# --- PHẦN MÔ HÌNH LINEAR REGRESSION (MODIFIED FOR VISUALIZATION) ---

# Dữ liệu Wide Format đã chuẩn bị ở trên: amount_new_house_transactions
# Tạo biến thời gian X (time) và biến mục tiêu Y (amount_new_house_transactions)
X = amount_new_house_transactions.index.to_frame(name='time')
Y = amount_new_house_transactions.copy()

# Huấn luyện mô hình Linear Regression cho từng sector
models = {}
for sector in Y.columns:
    model = LinearRegression()
    # Y[sector].reset_index(name='actual') sẽ tạo ra cột 'time' (từ index) và 'actual' (từ data)
    historical_data_for_fit = Y[sector].reset_index(name='actual')
    model.fit(historical_data_for_fit[['time']], historical_data_for_fit['actual'])
    models[sector] = model

# --- DỰ ĐOÁN CHO TEST SET ---
# Chuẩn bị dữ liệu test (time index)
test_month = test.month.str.split(' ', expand=True)
test_year = test_month[0].astype(int)
test_month_str = test_month[1]
test_month_num = test_month_str.map(month_codes)

test_X = pd.DataFrame()
test_X['time'] = (test_year - 2019) * 12 + test_month_num - 1

# Tạo prediction
test_sector_id = test.sector_id.values
test['predicted_amount'] = 0.0

for i, row in test_X.iterrows():
    sector = test_sector_id[i]
    if sector in models:
        model = models[sector]
        test.loc[i, 'predicted_amount'] = model.predict(row.to_frame().T[['time']])[0]

# Làm sạch (Nếu prediction < 0 thì đặt bằng 0)
test.loc[test.predicted_amount < 0, 'predicted_amount'] = 0

# TẠO FILE SUBMISSION GỐC
submission = test[['id', 'predicted_amount']].rename(columns={'predicted_amount': 'new_house_transaction_amount'})

# --- TRỰC QUAN HÓA XU HƯỚNG DỰ ĐOÁN (LINEAR REGRESSION) ---
print(Fore.BLUE + Style.BRIGHT + "\n--- TRỰC QUAN HÓA XU HƯỚNG DỰ ĐOÁN (LINEAR REGRESSION) ---" + Style.RESET_ALL)

# 1. Chọn Top 3 sectors theo tổng khối lượng giao dịch
total_volume = Y.sum().sort_values(ascending=False)
top_sectors = total_volume.head(3).index.tolist()

# 2. Tạo DataFrame cho đường hồi quy trên dữ liệu lịch sử và tương lai
X_all = pd.concat([X, test_X], ignore_index=True).sort_values(by='time').drop_duplicates().reset_index(drop=True)

# 3. Vẽ biểu đồ
fig, axes = plt.subplots(len(top_sectors), 1, figsize=(15, 5 * len(top_sectors)))
if len(top_sectors) == 1:
    axes = [axes]

for i, sector in enumerate(top_sectors):
    ax = axes[i]
    
    # SỬA LỖI KEYERROR TẠI ĐÂY:
    # Chỉ cần reset_index. DataFrame kết quả sẽ có 2 cột: 'time' và 'actual'
    historical_data = Y[sector].reset_index(name='actual')
    
    # Lấy điểm cuối của dữ liệu lịch sử
    last_historical_point = historical_data.sort_values(by='time').iloc[-1]
    
    # Dự đoán trên toàn bộ dải thời gian (Historical + Future)
    X_all['predicted'] = models[sector].predict(X_all[['time']])
    
    # Plot Dữ liệu Lịch sử (Actual)
    ax.scatter(historical_data['time'], historical_data['actual'], color='blue', alpha=0.6, label='Historical Amount (Actual)')
    
    # Plot Đường Hồi quy (Linear Trend) - Toàn bộ dải thời gian
    ax.plot(X_all['time'], X_all['predicted'], color='red', linestyle='--', alpha=0.7, label='Linear Regression Trend')

    # Plot Dữ liệu Dự đoán (Future)
    future_plot_data = X_all[X_all['time'] >= last_historical_point['time']].copy()
    
    # Nối điểm cuối của lịch sử với điểm đầu của dự đoán
    first_future_time = future_plot_data['time'].min()
    future_plot_data.loc[future_plot_data['time'] == first_future_time, 'predicted'] = models[sector].predict(pd.DataFrame({'time': [first_future_time]}))[0]
    
    ax.plot(future_plot_data['time'], future_plot_data['predicted'], color='orange', linewidth=3, label='Predicted Future Trend')

    # Đánh dấu ranh giới Train/Test
    ax.axvline(x=last_historical_point['time'], color='gray', linestyle=':', label='Train/Test Boundary')
    
    ax.set_title(f'Sector {sector}: Historical Data and Linear Regression Trend', fontsize=16)
    ax.set_xlabel('Time Index (Month since 2019 Jan)')
    ax.set_ylabel('Transaction Amount (10k Yuan)')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

