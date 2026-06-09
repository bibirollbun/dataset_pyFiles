import numpy as np 
import pandas as pd 
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
import seaborn as sns





application_train = pd.read_csv(r"/kaggle/input/home-credit-default-risk/application_train.csv")
application_train.head(10)


application_train_small = application_train[['TARGET', 'CODE_GENDER', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'OCCUPATION_TYPE','REGION_RATING_CLIENT', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'EXT_SOURCE_1','CNT_FAM_MEMBERS']]
application_train_small.head(10)


data = application_train_small
print(data.isna().sum())
data.head(10)
data = data.copy()
data['DAYS_BIRTH'] = data['DAYS_BIRTH'] // -365
data['NAME_EDUCATION_TYPE'].value_counts()


#BINNING


def binning_all_variables(df, threshold=0.05, custom_bins=None, feature=None):
    df = df.copy()
    
    columns_to_process = [feature] if feature else df.columns
    
    for column in columns_to_process:
        if column == 'TARGET':  # Bỏ qua cột target
            continue
        
        missing_mask = df[column].isna()
        
        if df[column].dtype == 'O':  # Xử lý biến dạng chữ
            df['GRP_' + column] = df[column].astype(object)
            df.loc[missing_mask, 'GRP_' + column] = 'Missing'
        else:  # Xử lý biến số
            if custom_bins and column in custom_bins:
                bins = [-np.inf] + custom_bins[column] + [np.inf]
            else:
                value_counts = df[column].value_counts(normalize=True)
                unique_values = value_counts[value_counts > threshold].index.tolist()
                
                sorted_values = np.sort(df[column].dropna().unique())
                cumulative = 0
                bins = [-np.inf]
                
                for val in sorted_values:
                    prop = (df[column] == val).mean()
                    if prop > threshold:
                        bins.append(val)
                    else:
                        cumulative += prop
                        if cumulative >= threshold:
                            bins.append(val)
                            cumulative = 0
                
                bins.append(np.inf)
                bins = sorted(set(bins))
            
            df['GRP_' + column] = pd.cut(df[column], bins=bins, include_lowest=True, duplicates='drop')
        
        if missing_mask.any():
            df['GRP_' + column] = df['GRP_' + column].astype(object)
            df.loc[missing_mask, 'GRP_' + column] = 'Missing'
        
        # Sắp xếp lại danh mục theo giá trị số học hoặc theo danh sách chữ
        def extract_lower_bound(x):
            if isinstance(x, pd.Interval):
                return x.left  # Lấy giá trị lower bound của khoảng
            return float('inf')
        
        if df[column].dtype == 'O':
            categories = list(df[column].dropna().unique()) + ['Missing']
        else:
            categories = sorted(
                [cat for cat in df['GRP_' + column].dropna().unique() if isinstance(cat, pd.Interval)],
                key=extract_lower_bound
            )
            if 'Missing' in df['GRP_' + column].values:
                categories.append('Missing')  # Thêm Missing vào danh mục cuối cùng nếu có NaN
        
        df['GRP_' + column] = pd.Categorical(df['GRP_' + column], categories=categories, ordered=True)
    
    return df

# Thiết lập điểm cắt tùy chỉnh, ví dụ
custom_bins = {
    'AMT_INCOME_TOTAL': [67500.0, 81000.0, 90000.0, 112500.0, 117000.0, 135000.0, 148500.0, 157500.0, 180000.0, 202500.0, 216000.0, 225000.0, 270000.0, 315000.0],
    'EXT_SOURCE_1' : [ 0.3, 0.5, 0.7, 0.9],
    'DAYS_BIRTH' : [25, 30, 35, 40, 45, 50],
    'DAYS_EMPLOYED' : [-10950, -9125, -7300, -5475, -3650, -1825, -750, 0, 365242],
    'REGION_RATING_CLIENT' : [1, 2]
}
custom_bins_1 = {
    'AMT_INCOME_TOTAL': [117000.0, 216000.0],
    'EXT_SOURCE_1' : [ 0.3, 0.7],
    'DAYS_BIRTH' : [30, 50],
    'DAYS_EMPLOYED' : [-10950, -5475, 0, 365242],
    'REGION_RATING_CLIENT' : [1, 2]
}


#Tính WOE, IV:
def caculate_WOE_IV(df,feature,target):
    df = df.groupby(feature,observed=False)[target].agg(['count','sum']).reset_index()
    df.columns = ['bin','num_of_obs','num_of_event']
    df['num_of_non_event']= df['num_of_obs'] - df['num_of_event']
    df['num_of_event'] = df['num_of_event'].astype(float)
    df['num_of_non_event'] = df['num_of_non_event'].astype(float)
    df['num_of_non_event_c'] = df['num_of_non_event'].copy()
    df['num_of_event_c'] = df['num_of_event'].copy()
    total_non_event = df['num_of_non_event'].sum()
    total_event = df['num_of_event'].sum()
    # Nếu event hoặc non_event = 0 thì cộng 0.5 vào cả hai
    mask = (df['num_of_event'] == 0) | (df['num_of_non_event'] == 0)
    df.loc[mask, 'num_of_event_c'] += 0.5
    df.loc[mask, 'num_of_non_event_c'] += 0.5
    df['prct_non_event']= df['num_of_non_event_c']/total_non_event
    df['prct_event']= df['num_of_event_c']/total_event
    df['WOE'] = np.log(df['prct_non_event']/df['prct_event'])
    df['IV'] = (df['prct_non_event']- df['prct_event']) * df['WOE']
    IV = df['IV'].sum()
    df.index = range(1,len(df)+1)
    df['bin'] = df['bin'].astype(str)

    df = df.drop(columns=['num_of_non_event_c','num_of_event_c'])
    df['event_rate'] = df['num_of_event']/df['num_of_obs']
    df['prct_obs'] = df['num_of_obs']/(df['num_of_obs'].sum())
    return df , IV


#CROSS BIẾN GENDER+DAYS_BIRTH


from itertools import combinations

def cross_selected_binned_variables(df, custom_bins, selected_features):
    df = binning_all_variables(df, custom_bins=custom_bins)  # Binned dữ liệu trước
    
    selected_binned = [f"GRP_{col}" for col in selected_features if f"GRP_{col}" in df.columns]
    
    if len(selected_binned) != 2:
        raise ValueError("Bạn phải chọn đúng 2 biến đã được binning để tạo biến giao nhau.")
    
    col1, col2 = selected_binned
    cross_col_name = f"{col1.replace('GRP_', '')}_{col2.replace('GRP_', '')}_c"
    df[cross_col_name] = df[col1].astype(str) + "_" + df[col2].astype(str)
    
    # Xóa toàn bộ các cột đã binned
    binned_columns = [col for col in df.columns if col.startswith('GRP_')]
    df = df.drop(columns=binned_columns, errors="ignore")
    
    return df

# Áp dụng binning theo custom_bins_1 và tạo biến giao nhau cho 2 cột cụ thể
data = cross_selected_binned_variables(data, custom_bins_1, selected_features=['CODE_GENDER', 'DAYS_BIRTH'])
data



data['CODE_GENDER_DAYS_BIRTH_c'].value_counts().sort_index()


WOE_CODE_GENDER_DAYS_BIRTH_c, IV = caculate_WOE_IV(data,'CODE_GENDER_DAYS_BIRTH_c','TARGET')
print('Tổng IV là:', IV)
WOE_CODE_GENDER_DAYS_BIRTH_c




WOE_CODE_GENDER_DAYS_BIRTH_c[['code_gender', 'days_birth']] = WOE_CODE_GENDER_DAYS_BIRTH_c['bin'].str.extract(r'(\w+)_((?:-?[\w.]+, ?[\w.]+]|\(-?[\w.]+, ?[\w.]+\]))')
WOE_CODE_GENDER_DAYS_BIRTH_c


pivot_df = WOE_CODE_GENDER_DAYS_BIRTH_c.pivot(index="code_gender", columns="days_birth", values=["event_rate", "num_of_obs", "prct_obs", "WOE"])

# Sắp xếp lại thứ tự cột theo đúng logic
pivot_df = pivot_df.sort_index(axis=1, level=1)


# Hiển thị kết quả
pivot_df


data["CODE_GENDER_DAYS_BIRTH_c"] = data["CODE_GENDER_DAYS_BIRTH_c"].map(lambda x: 
    "Group1" if x in ["XNA_(-inf, 30.0]", "M_(-inf, 30.0]"] else
    "Group2" if x in ["F_(-inf, 30.0]", "XNA_(30.0, 50.0]", "M_(30.0, 50.0]"] else
    "Group3" if x in ["F_(30.0, 50.0]", "XNA_(50.0, inf]", "M_(50.0, inf]"] else
    "Group4"
)



data["CODE_GENDER_DAYS_BIRTH_c"].value_counts().sort_index()


data.head(10)


#CROSS BIẾN AMT_INCOOME_TOTAL VÀ REGION_RATING_CLIENT
# Áp dụng binning theo custom_bins_1 và tạo biến giao nhau cho 2 cột cụ thể
data = cross_selected_binned_variables(data, custom_bins_1, selected_features=['AMT_INCOME_TOTAL', 'REGION_RATING_CLIENT'])



data['AMT_INCOME_TOTAL_REGION_RATING_CLIENT_c'].value_counts().sort_index()


WOE_AMT_INCOME_TOTAL_REGION_RATING_CLIENT_c, IV = caculate_WOE_IV(data,'AMT_INCOME_TOTAL_REGION_RATING_CLIENT_c','TARGET')
print('Tổng IV là:', IV)
WOE_AMT_INCOME_TOTAL_REGION_RATING_CLIENT_c


WOE_AMT_INCOME_TOTAL_REGION_RATING_CLIENT_c[['income', 'region']] = WOE_AMT_INCOME_TOTAL_REGION_RATING_CLIENT_c['bin'].str.extract(r'((?:-?[\w.]+, ?[\w.]+]|\(-?[\w.]+, ?[\w.]+\]))_((?:-?[\w.]+, ?[\w.]+]|\(-?[\w.]+, ?[\w.]+\]))')
WOE_AMT_INCOME_TOTAL_REGION_RATING_CLIENT_c





pivot_df1 = WOE_AMT_INCOME_TOTAL_REGION_RATING_CLIENT_c.pivot(index="income", columns="region", values=["event_rate", "num_of_obs", "prct_obs", "WOE"])

# Sắp xếp lại thứ tự cột theo đúng logic
pivot_df1 = pivot_df1.sort_index(axis=1, level=1)


# Hiển thị kết quả
pivot_df1


data["AMT_INCOME_TOTAL_REGION_RATING_CLIENT_c"] = data["AMT_INCOME_TOTAL_REGION_RATING_CLIENT_c"].map(lambda x: 
    "Group1" if x in ["(216000.0, inf]_(-inf, 1.0]"] else
    "Group2" if x in ["(-inf, 117000.0]_(-inf, 1.0]", "(117000.0, 216000.0]_(-inf, 1.0]"] else
    "Group3" if x in ["(-inf, 117000.0]_(1.0, 2.0]", "(117000.0, 216000.0]_(1.0, 2.0]", "(216000.0, inf]_(1.0, 2.0]"] else
    "Group4" if x in ["(216000.0, inf]_(2.0, inf]"] else
    "Group5"                                                                                                
)



data["AMT_INCOME_TOTAL_REGION_RATING_CLIENT_c"].value_counts().sort_index()


data.head()


#CROSS BIẾN INCOME VÀ DAYS_BIRTH
data = cross_selected_binned_variables(data, custom_bins_1, selected_features=['AMT_INCOME_TOTAL', 'DAYS_BIRTH'])
data.head()


data['AMT_INCOME_TOTAL_DAYS_BIRTH_c'].value_counts().sort_index()


WOE_AMT_INCOME_TOTAL_DAYS_BIRTH_c, IV = caculate_WOE_IV(data,'AMT_INCOME_TOTAL_DAYS_BIRTH_c','TARGET')
print('Tổng IV là:', IV)
WOE_AMT_INCOME_TOTAL_DAYS_BIRTH_c


WOE_AMT_INCOME_TOTAL_DAYS_BIRTH_c[['income', 'age']] = WOE_AMT_INCOME_TOTAL_DAYS_BIRTH_c['bin'].str.extract(r'((?:-?[\w.]+, ?[\w.]+]|\(-?[\w.]+, ?[\w.]+\]))_((?:-?[\w.]+, ?[\w.]+]|\(-?[\w.]+, ?[\w.]+\]))')
WOE_AMT_INCOME_TOTAL_DAYS_BIRTH_c


pivot_df2 = WOE_AMT_INCOME_TOTAL_DAYS_BIRTH_c.pivot(index="income", columns="age", values=["event_rate", "num_of_obs", "prct_obs", "WOE"])

# Sắp xếp lại thứ tự cột theo đúng logic
pivot_df2 = pivot_df2.sort_index(axis=1, level=1)


# Hiển thị kết quả
pivot_df2


data["AMT_INCOME_TOTAL_DAYS_BIRTH_c"] = data["AMT_INCOME_TOTAL_DAYS_BIRTH_c"].map(lambda x: 
    "Group1" if x in ["(216000.0, inf]_(-inf, 30.0]"] else
    "Group2" if x in ["(-inf, 117000.0]_(-inf, 30.0]", "(117000.0, 216000.0]_(-inf, 30.0]"] else
    "Group3" if x in ["(216000.0, inf]_(30.0, 50.0]"] else
    "Group4" if x in ["(-inf, 117000.0]_(30.0, 50.0]", "(117000.0, 216000.0]_(30.0, 50.0]"] else
    "Group5"                                                                                                
)



data["AMT_INCOME_TOTAL_DAYS_BIRTH_c"].value_counts()


data.head()


def woe_feature_table(data, custom_bins, feature):
    df_binned = binning_all_variables(data, threshold=0.05, custom_bins=custom_bins, feature=feature)
    WOE_TABLE, IV = caculate_WOE_IV(df_binned, f'GRP_{feature}', 'TARGET')

    df = WOE_TABLE[['bin', 'num_of_obs', 'prct_obs', 'num_of_non_event', 'num_of_event', 'event_rate', 'WOE', 'IV']]
    df.columns = ['bin','count','count(%)','non_event','event','event_rate','WOE','IV']

    total_row = pd.DataFrame({
        'bin': [''],
        'count': [df['count'].sum()],
        'count(%)': [df['count(%)'].sum()],
        'non_event': [df['non_event'].sum()],
        'event': [df['event'].sum()],
        'event_rate': [df['event_rate'].sum()],
        'WOE': [''],
        'IV': [df['IV'].sum()]
    })

    df = pd.concat([df, total_row], ignore_index=True)
    df.index = df.index.astype(str)
    df.index.values[-1] = "TOTAL"
   
    return df
df2 = woe_feature_table(data, custom_bins = custom_bins, feature = 'AMT_INCOME_TOTAL')
df2


df2.dtypes


df2['WOE'] = pd.to_numeric(df2['WOE'], errors='coerce') #chuyển đổi WOE về dạng số
df2.dtypes


#Vẽ biểu đồ:
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches # Import để tạo mẫu màu




def woe_plot(df, feature):
    df = df[df.index != "TOTAL"]   # Bỏ dòng TOTAL nếu có

    # Tạo colormap từ đỏ -> vàng -> xanh
    cmap = plt.colormaps["RdYlGn"]
    norm = mcolors.Normalize(vmin=df['WOE'].min(), vmax=df['WOE'].max())  # Chuẩn hóa giá trị WOE

    fig, ax1 = plt.subplots(figsize=(9, 7))
    bins = np.arange(len(df))

    # Tô màu cột theo giá trị WOE
    colors = [cmap(norm(woe)) for woe in df['WOE']]

    # Vẽ cột Event + Non-Event
    ax1.bar(bins, df['event'], label='Event', color=colors, alpha=1)
    ax1.bar(bins, df['non_event'], label='Non-event', bottom=df['event'], color=colors, alpha=1)

    # Cấu hình trục chính
    ax1.set_xlabel('Bin')
    ax1.set_ylabel('Count')
    ax1.set_xticks(bins)
    ax1.set_xticklabels(df['bin'], rotation=45, ha='right')

    # Trục phụ cho WOE
    ax2 = ax1.twinx()
    ax2.plot(bins, df['WOE'], color='black', marker='o', markersize=6, linestyle='-', linewidth=2, label='WOE')
    ax2.set_ylabel('WOE')

    # Tiêu đề và chú thích 
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    plt.title(f'WOE Plot for {feature}', fontsize=12, fontweight='bold')

    # Định nghĩa màu sắc tương ứng với colormap RdYlGn
    high_risk_patch = mpatches.Patch(facecolor='red', label='Rủi ro cao')
    medium_risk_patch = mpatches.Patch(facecolor='yellow', label='Rủi ro trung bình')
    low_risk_patch = mpatches.Patch(facecolor='green', label='Rủi ro thấp')

    # Thêm chú thích vào biểu đồ
    ax1.legend(handles=[high_risk_patch, medium_risk_patch, low_risk_patch],  
               bbox_to_anchor=(0.9, -0.5),  # Đưa chú thích xuống dưới
               frameon=True, ncol=3)  # Sắp xếp theo hàng ngang

    plt.tight_layout()
    plt.show()

a = woe_plot(df2, 'AMT_INCOME_TOTAL')
a


def transform_to_woe(df, features, target, custom_bins=None, feature=None):
    df_transformed = df.copy()
    df_transformed = binning_all_variables(df, threshold=0.05, custom_bins=custom_bins, feature=feature)  # Binning
    
    features_to_process = [feature] if feature else features
    
    for feature in features_to_process:
        if 'GRP_' + feature not in df_transformed.columns:
            print(f'Warning: GRP_{feature} không tồn tại trong dữ liệu sau binning!')
            continue
        
        woe_iv, iv = caculate_WOE_IV(df_transformed, 'GRP_' + feature, target)
        
        # Chuyển đổi kiểu dữ liệu về chuỗi để tránh lỗi NaN
        woe_iv['bin'] = woe_iv['bin'].astype(str)
        df_transformed['GRP_' + feature] = df_transformed['GRP_' + feature].astype(str)
        
        woe_mapping = woe_iv.set_index('bin')['WOE'].rename('WOE_' + feature)
        
        df_transformed['WOE_' + feature] = df_transformed['GRP_' + feature].map(woe_mapping)
        
        # Đảm bảo tất cả các giá trị có cùng kiểu dữ liệu
        df_transformed['WOE_' + feature] = df_transformed['WOE_' + feature].astype(float)
    
    return df_transformed



import time
start_time = time.time()  # Bắt đầu đo thời gian
# Chuyển đổi dữ liệu gốc sang WOE
woe_data1 = transform_to_woe(data, ['CODE_GENDER', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'OCCUPATION_TYPE','REGION_RATING_CLIENT', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'EXT_SOURCE_1','CNT_FAM_MEMBERS','CODE_GENDER_DAYS_BIRTH_c','AMT_INCOME_TOTAL_REGION_RATING_CLIENT_c','AMT_INCOME_TOTAL_DAYS_BIRTH_c'], 'TARGET', custom_bins=custom_bins, feature= None)

end_time = time.time()  # Kết thúc đo thời gian
print(f"Thời gian chạy: {end_time - start_time:.4f} giây")  # Hiển thị thời gian chạy


woe_data1.head(10)


print(woe_data1.columns.tolist())


import statsmodels.api as sm


woe_columns = [col for col in woe_data1.columns if col.startswith('WOE_')]
sel_col = ['TARGET'] + woe_columns 
data_woe = woe_data1[woe_columns]
data_woe



corr_matrix = data_woe.corr()
corr_matrix


print(data_woe.columns.tolist())


data_woe = data_woe.copy()
data_woe['TARGET'] = woe_data1['TARGET']
data_woe


X = data_woe[['WOE_CODE_GENDER', 'WOE_DAYS_EMPLOYED', 'WOE_OCCUPATION_TYPE', 'WOE_REGION_RATING_CLIENT', 'WOE_NAME_EDUCATION_TYPE', 'WOE_NAME_FAMILY_STATUS', 'WOE_EXT_SOURCE_1','WOE_CNT_FAM_MEMBERS']]
X = sm.add_constant(X)  # Thêm intercept
y = 1-data_woe['TARGET']

# Khởi tạo mô hình hồi quy logistic với solver Newton-Raphson
model = sm.Logit(y, X)
result = model.fit(method='newton')  # Sử dụng Newton solver
print(result.summary())


from sklearn.metrics import roc_auc_score

# Dự báo xác suất từ mô hình
y_pred = result.predict(X)  # X đã có const

# Tính AUC
auc = roc_auc_score(y, y_pred)

# Tính Gini
gini = 2 * auc - 1

print(f"AUC: {auc:.4f}")
print(f"Gini: {gini:.4f}")


n = len(data)
n


def calculate_scorecard(coefficients, woe_table, base_score=600, odds=50, pdo=20, a=-2.4319, n=10):
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(50)
    
    scorecard = []
    for feature, coef in coefficients.items():
        if feature.startswith('WOE_'):
            feature_name = feature.replace('WOE_', '')
            woe_col = f'WOE_{feature_name}'
            grp_col = f'GRP_{feature_name}'
            
            if woe_col in woe_table.columns and grp_col in woe_table.columns:
                grouped_woe = woe_table[[grp_col, woe_col]].drop_duplicates()
                for _, row in grouped_woe.iterrows():
                    score = ((row[woe_col] * coef + (a / n)) * factor) + (offset / n)
                    scorecard.append({
                        'Feature': feature_name,
                        'Bin': row[grp_col],
                        'WOE': row[woe_col],
                        'Score': round(score, 2)
                    })
            else:
                raise KeyError(f"Missing columns in woe_table: Expected {woe_col} and {grp_col}, found {woe_table.columns}")
    
    return pd.DataFrame(scorecard)
coefficients = result.params.to_dict()



# Tạo bảng điểm
scorecard = calculate_scorecard(coefficients, woe_data1)
scorecard


scorecard.head(20)

