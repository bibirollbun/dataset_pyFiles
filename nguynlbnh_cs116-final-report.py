import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler

import os
import re
from sklearn.base import clone
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import polars as pl
import polars.selectors as cs
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, PercentFormatter
import seaborn as sns

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
import torch
import torch.nn as nn
import torch.optim as optim

from colorama import Fore, Style
from IPython.display import clear_output
import warnings
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None


def plot_related_cols(df, columns_list):
    # Thiết lập số cột và hàng
    n_cols = 4
    n_rows = -(-len(columns_list) // n_cols)  # Tính số hàng cần thiết (chia làm tròn lên)
    
    # Tạo figure lớn
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))  # Điều chỉnh kích thước phù hợp
    axes = axes.flatten()  # Chuyển ma trận axes thành 1D để dễ xử lý
    
    # Vẽ biểu đồ cho từng cột
    for i, col in enumerate(columns_list):
        value_counts = df[col].value_counts(sort=False).sort_index()  # Đếm tần suất
        
        value_counts.plot(kind='bar', ax=axes[i], title=col)  # Vẽ biểu đồ
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Số lượng')
    
    # Ẩn các ô không sử dụng nếu số cột không chia hết
    for i in range(len(columns_list), len(axes)):
        axes[i].axis('off')  # Tắt ô không dùng
    
    # Tinh chỉnh layout
    plt.tight_layout()
    plt.show()

def plot_kde(df, column_name):
    """
    Vẽ biểu đồ KDE cho một cột trong DataFrame.

    Parameters:
    - df: DataFrame chứa dữ liệu.
    - column_name: Tên cột mà bạn muốn vẽ KDE.

    Returns:
    - Không trả về giá trị, chỉ vẽ biểu đồ.
    """
    if column_name not in df.columns:
        print(f"Cột '{column_name}' không tồn tại trong DataFrame.")
        return

    # Vẽ KDE cho cột với tham số 'fill=True' thay cho 'shade=True'
    sns.kdeplot(df[column_name], fill=True)
    
    # Thêm tiêu đề và nhãn cho trục
    plt.title(f"Kernel Density Estimate (KDE) for {column_name}")
    plt.xlabel(column_name)
    plt.ylabel('Density')
    
    # Hiển thị biểu đồ
    plt.show()


def plot_kde_and_box(df, column_name):
    """
    Vẽ cả KDE plot và Box plot cho một cột trong DataFrame,
    đồng thời trả về các giá trị outliers trong Box plot.

    Parameters:
    - df: DataFrame chứa dữ liệu.
    - column_name: Tên cột mà bạn muốn vẽ biểu đồ.

    Returns:
    - outliers: Danh sách các giá trị outliers trong cột.
    """
    # Kiểm tra nếu cột tồn tại trong DataFrame
    if column_name not in df.columns:
        print(f"Cột '{column_name}' không tồn tại trong DataFrame.")
        return
    dataframe = df[column_name]
    
    # Tính toán Q1, Q3, IQR để xác định các outliers
    Q1 = dataframe.quantile(0.25)
    Q3 = dataframe.quantile(0.75)
    IQR = Q3 - Q1

    # Xác định các outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Lọc các giá trị outliers
    outliers = dataframe[dataframe < lower_bound].tolist() + dataframe[dataframe > upper_bound].tolist()
    

    # Vẽ KDE và Boxplot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Vẽ KDE trên axes[0]
    sns.kdeplot(dataframe, fill=True, ax=axes[0])
    axes[0].set_title(f"KDE Plot for {column_name}")
    axes[0].set_xlabel(column_name)
    axes[0].set_ylabel('Density')

    # Vẽ Boxplot trên axes[1]
    sns.boxplot(x=dataframe, ax=axes[1])
    axes[1].set_title(f"Boxplot for {column_name}")
    axes[1].set_xlabel(column_name)

    # Hiển thị biểu đồ
    plt.tight_layout()
    plt.show()

    # Trả về các giá trị outliers
    return outliers

def replace_outliers(df, column_name, outliers, replacement_value=None):
    """
    Thay thế các giá trị outliers trong một cột của DataFrame bằng giá trị thay thế.
    
    Parameters:
    - df: DataFrame chứa dữ liệu.
    - column_name: Tên cột cần xử lý outliers.
    - outliers: Danh sách các giá trị outliers.
    - replacement_value: Giá trị thay thế. Mặc định là None (thay bằng trung vị của cột).
    
    Returns:
    - df: DataFrame sau khi đã thay thế outliers.
    """
    # Nếu không có giá trị thay thế, sử dụng trung vị của cột
    if replacement_value is None:
        replacement_value = np.nan #df[column_name].median()
    
    # Thay thế các giá trị outliers
    df[column_name] = df[column_name].apply(lambda x: replacement_value if x in outliers else x)
    
    return df

def remove_outlier(df, columns):
    for column in columns:
        # # Step 0: Create a boxplot
        # plt.figure(figsize=(8, 6))
        # sns.boxplot(x=df[column])
        # plt.title('Boxplot for column_name (Outliers replaced with NaN)')
        # plt.show()

        # Step 1: Identify the outliers
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define the outlier thresholds
        lower_bound = Q1 - 5 * IQR
        upper_bound = Q3 + 5 * IQR
        
        # Step 2: Replace outliers with NaN
        df[column] = df[column].apply(lambda x: np.nan if x < lower_bound or x > upper_bound else x)
        
    return df

# Tính phần trăm missing data
def missing_percentage(df, column_name):
    total_values = len(df[column_name])
    missing_values = df[column_name].isna().sum()
    percentage_missing = (missing_values / total_values) * 100
    return percentage_missing

# read file parquet
def process_file(filename, dirname):
    df = pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet'))
    df.drop('step', axis=1, inplace=True)
    return df.describe().values.reshape(-1), filename.split('=')[1]

# load time series from parquet files
def load_time_series(dirname) -> pd.DataFrame:
    ids = os.listdir(dirname)
    
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda fname: process_file(fname, dirname), ids), total=len(ids)))
    
    stats, indexes = zip(*results)
    
    df = pd.DataFrame(stats, columns=[f"stat_{i}" for i in range(len(stats[0]))])
    df['id'] = indexes
    return df


train = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
test = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')
data_dict = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/data_dictionary.csv')
sample = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/sample_submission.csv')
train_ts = load_time_series("/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet")
test_ts = load_time_series("/kaggle/input/child-mind-institute-problematic-internet-use/series_test.parquet")


data_dict.head()


groups = data_dict.groupby('Instrument')['Field'].apply(list).to_dict()

for instrument, features in groups.items():
    print(f"{instrument}: {features}\n")


PCIAT_cols = data_dict[data_dict['Instrument'] == 'Parent-Child Internet Addiction Test']['Field']
for col in PCIAT_cols:
    print(f" Missing percent of {col}: {missing_percentage(df=train, column_name=col)}")
print(f" Missing percent of sii: {missing_percentage(df=train, column_name='sii')}")


plt.figure(figsize=(8, 6))
plt.scatter(train['PCIAT-PCIAT_Total'], train['sii'], alpha=0.7)
plt.title('')
plt.ylabel('sii')
plt.xlabel('PCIAT-PCIAT_Total')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


BIA_cols = data_dict[data_dict['Instrument'] == 'Bio-electric Impedance Analysis']['Field']
for col in BIA_cols:
    print(f" Missing percent of {col}: {missing_percentage(df=train, column_name=col)}")


train[BIA_cols].describe()


for col in BIA_cols:
    if 'Season' not in col:
        outlier = plot_kde_and_box(train, col)
        # print(outlier)
        print(len(outlier))
        replace_value = train[col].median()

        train = replace_outliers(train, col, outlier, replace_value)


for col in BIA_cols:
    if 'Season' not in col:
        plot_kde(train, col)


for i in range(1,17):
    print(len(train[train[BIA_cols].isnull().sum(axis=1) >= i][BIA_cols]), end=', ')


CGAS_cols = data_dict[data_dict['Instrument'] == "Children's Global Assessment Scale"]['Field']
for col in CGAS_cols:
    print(f" Missing percent of {col}: {missing_percentage(df=train, column_name=col)}")


plot_related_cols(train, CGAS_cols)


train.loc[train['id'] == '83525bbe', 'CGAS-CGAS_Score'] = np.nan
train[train['id'] == '83525bbe']


plot_kde(train, 'CGAS-CGAS_Score')


len(train[train['CGAS-CGAS_Score'].isna() ^ train['sii'].isna()])


plt.figure(figsize=(8, 6))
plt.scatter(train['CGAS-CGAS_Score'], train['sii'], alpha=0.7)
plt.title('')
plt.ylabel('sii')
plt.xlabel('CGAS-CGAS_Score')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


BD_cols = data_dict[data_dict['Instrument'] == "Demographics"]['Field']
for col in BD_cols:
    print(f" Missing percent of {col}: {missing_percentage(df=train, column_name=col)}")


plot_related_cols(train, BD_cols)


FGC_cols = data_dict[data_dict['Instrument'] == "FitnessGram Child"]['Field']
for col in FGC_cols:
    print(f" Missing percent of {col}: {missing_percentage(df=train, column_name=col)}")


train[FGC_cols].describe()


plot_related_cols(train, FGC_cols)


# Phân bố thể lực
for col in FGC_cols:
    if 'Season' not in col and 'Zone' not in col:
        plot_kde(train, col)


# Miss nhiều quả nên cho nghỉ, maybe cái trên cũng sẽ khong cần xài tới.
FE_cols = data_dict[data_dict['Instrument'] == "FitnessGram Vitals and Treadmill"]['Field']
for col in FE_cols:
    print(f" Missing percent of {col}: {missing_percentage(df=train, column_name=col)}")


IU_cols = data_dict[data_dict['Instrument'] == "Internet Use"]['Field']
for col in IU_cols:
    print(f" Missing percent of {col}: {missing_percentage(df=train, column_name=col)}")


plot_related_cols(train, IU_cols)


confusion_matrix = pd.crosstab(train['sii'], train['PreInt_EduHx-computerinternet_hoursday'], rownames=['sii'], colnames=['PreInt_EduHx-computerinternet_hoursday'])
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.show()


PAQ_A_cols = data_dict[data_dict['Instrument'] == "Physical Activity Questionnaire (Adolescents)"]['Field']
for col in PAQ_A_cols:
    print(f" Missing percent of {col}: {missing_percentage(df=train, column_name=col)}")


plot_related_cols(train, PAQ_A_cols)


plot_kde(train, 'PAQ_A-PAQ_A_Total')


PAQ_C_cols = data_dict[data_dict['Instrument'] == "Physical Activity Questionnaire (Children)"]['Field']
for col in PAQ_C_cols:
    print(f" Missing percent of {col}: {missing_percentage(df=train, column_name=col)}")


plot_related_cols(train, PAQ_C_cols)


plot_kde(train, 'PAQ_C-PAQ_C_Total')


print(len(train[train['PAQ_C-PAQ_C_Total'].notna() & train['PAQ_A-PAQ_A_Total'].notna()]))
print(len(train[train['PAQ_C-PAQ_C_Total'].isna() & train['PAQ_A-PAQ_A_Total'].isna() & train['sii'].notna()]))


train[train['PAQ_C-PAQ_C_Total'].notna() & train['PAQ_A-PAQ_A_Total'].notna()]


# Định nghĩa hàm kết hợp 2 cột
def combine_columns(col1, col2):
    return col1.combine(col2, lambda x, y: x if pd.notna(x) and pd.notna(y) 
                        else x if pd.notna(x) 
                        else y if pd.notna(y)
                        else np.nan)

train['PAQ_Combine'] = combine_columns(train['PAQ_C-PAQ_C_Total'], train['PAQ_A-PAQ_A_Total'])


print(f" Missing percent of 'PAQ_Combine': {missing_percentage(df=train, column_name='PAQ_Combine')}")


plot_kde(train, 'PAQ_Combine')


PM_cols = data_dict[data_dict['Instrument'] == "Physical Measures"]['Field']
for col in PM_cols:
    print(f" Missing percent of {col}: {missing_percentage(df=train, column_name=col)}")


train[PM_cols].describe()


for col in PM_cols:
    if 'Season' not in col:
        outlier = plot_kde_and_box(train, col)
        # print(outlier)
        print(len(outlier))
        replace_value = train[col].median()

        train = replace_outliers(train, col, outlier, replace_value)


plot_related_cols(train, PM_cols)


for col in PM_cols:
    if 'Season' not in col:
        plot_kde(train, col)


SDS_cols = data_dict[data_dict['Instrument'] == "Sleep Disturbance Scale"]['Field']
for col in SDS_cols:
    print(f" Missing percent of {col}: {missing_percentage(df=train, column_name=col)}")


train[SDS_cols].describe()


plot_related_cols(train, SDS_cols)


train = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
test = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')
data_dict = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/data_dictionary.csv')
sample = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/sample_submission.csv')
train_ts = load_time_series("/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet")
test_ts = load_time_series("/kaggle/input/child-mind-institute-problematic-internet-use/series_test.parquet")


# Auto Encoder to handle parquet data
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim*3),
            nn.ReLU(),
            nn.Linear(encoding_dim*3, encoding_dim*2),
            nn.ReLU(),
            nn.Linear(encoding_dim*2, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim*2),
            nn.ReLU(),
            nn.Linear(input_dim*2, input_dim*3),
            nn.ReLU(),
            nn.Linear(input_dim*3, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def perform_autoencoder(df, autoencoder, encoding_dim=50, epochs=50, batch_size=32):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    data_tensor = torch.FloatTensor(df_scaled)
    
    input_dim = data_tensor.shape[1]

    train = True
    if autoencoder == None:
        train = False        
        autoencoder = AutoEncoder(input_dim, encoding_dim)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters())
    
    if train == True:
        for epoch in range(epochs):
            for i in range(0, len(data_tensor), batch_size):
                batch = data_tensor[i : i + batch_size]
                optimizer.zero_grad()
                reconstructed = autoencoder(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()
                
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}]')
                     
    with torch.no_grad():
        encoded_data = autoencoder.encoder(data_tensor).numpy()
        
    df_encoded = pd.DataFrame(encoded_data, columns=[f'Enc_{i + 1}' for i in range(encoded_data.shape[1])])
    
    return df_encoded, autoencoder


train = train.dropna(subset='sii')

df_train = train_ts.drop('id', axis=1)
df_test = test_ts.drop('id', axis=1)

train_ts_encoded, autoencoder= perform_autoencoder(df_train, None, encoding_dim=60, epochs=100, batch_size=32)
test_ts_encoded, autoencoder = perform_autoencoder(df_test, autoencoder, encoding_dim=60, epochs=100, batch_size=32)

time_series_cols = train_ts_encoded.columns.tolist()
train_ts_encoded["id"]=train_ts["id"]
test_ts_encoded['id']=test_ts["id"]

train = pd.merge(train, train_ts_encoded, how="left", on='id')
test = pd.merge(test, test_ts_encoded, how="left", on='id')

imputer = KNNImputer(n_neighbors=5)
numeric_cols = train.select_dtypes(include=['float64','float32', 'int64']).columns
numeric_cols = [col for col in numeric_cols if not (("PCIAT" in col) or ("sii" in col))]

imputed_data = imputer.fit_transform(train[numeric_cols])
train_imputed = pd.DataFrame(imputed_data, columns=numeric_cols)

test_imputed_data = imputer.transform(test[numeric_cols])
test_imputed = pd.DataFrame(test_imputed_data, columns=numeric_cols)

for col in train.columns:
    if col not in numeric_cols:
        train_imputed[col] = train[col]

for col in test.columns:
    if col not in numeric_cols:
        test_imputed[col] = test[col]
        
train = train_imputed
test = test_imputed


def feature_engineering(df):
    season_cols = [col for col in df.columns if 'Season' in col]
    df = df.drop(season_cols, axis=1) 
    df['BMI_Age'] = df['Physical-BMI'] * df['Basic_Demos-Age']
    df['Internet_Hours_Age'] = df['PreInt_EduHx-computerinternet_hoursday'] * df['Basic_Demos-Age']
    df['BMI_Internet_Hours'] = df['Physical-BMI'] * df['PreInt_EduHx-computerinternet_hoursday']
    df['BFP_BMI'] = df['BIA-BIA_Fat'] / df['BIA-BIA_BMI']
    df['FFMI_BFP'] = df['BIA-BIA_FFMI'] / df['BIA-BIA_Fat']
    df['FMI_BFP'] = df['BIA-BIA_FMI'] / df['BIA-BIA_Fat']
    df['LST_TBW'] = df['BIA-BIA_LST'] / df['BIA-BIA_TBW']
    df['BFP_BMR'] = df['BIA-BIA_Fat'] * df['BIA-BIA_BMR']
    df['BFP_DEE'] = df['BIA-BIA_Fat'] * df['BIA-BIA_DEE']
    df['BMR_Weight'] = df['BIA-BIA_BMR'] / df['Physical-Weight']
    df['DEE_Weight'] = df['BIA-BIA_DEE'] / df['Physical-Weight']
    df['SMM_Height'] = df['BIA-BIA_SMM'] / df['Physical-Height']
    df['Muscle_to_Fat'] = df['BIA-BIA_SMM'] / df['BIA-BIA_FMI']
    df['Hydration_Status'] = df['BIA-BIA_TBW'] / df['Physical-Weight']
    df['ICW_TBW'] = df['BIA-BIA_ICW'] / df['BIA-BIA_TBW']
    df['BMI_PHR'] = df['Physical-BMI'] * df['Physical-HeartRate']
    
    return df


train = feature_engineering(train)
train = train.dropna(thresh=10, axis=0)
test = feature_engineering(test)

train = train.drop('id', axis=1)
test  = test.drop('id', axis=1)   


featuresCols = ['Basic_Demos-Age', 'Basic_Demos-Sex',
                'CGAS-CGAS_Score', 'Physical-BMI',
                'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumference',
                'Physical-Diastolic_BP', 'Physical-HeartRate', 'Physical-Systolic_BP',
                'Fitness_Endurance-Max_Stage',
                'Fitness_Endurance-Time_Mins', 'Fitness_Endurance-Time_Sec',
                'FGC-FGC_CU', 'FGC-FGC_CU_Zone', 'FGC-FGC_GSND',
                'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD', 'FGC-FGC_GSD_Zone', 'FGC-FGC_PU',
                'FGC-FGC_PU_Zone', 'FGC-FGC_SRL', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR',
                'FGC-FGC_SRR_Zone', 'FGC-FGC_TL', 'FGC-FGC_TL_Zone',
                'BIA-BIA_Activity_Level_num', 'BIA-BIA_BMC', 'BIA-BIA_BMI',
                'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_ECW', 'BIA-BIA_FFM',
                'BIA-BIA_FFMI', 'BIA-BIA_FMI', 'BIA-BIA_Fat', 'BIA-BIA_Frame_num',
                'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_SMM',
                'BIA-BIA_TBW', 'PAQ_A-PAQ_A_Total',
                'PAQ_C-PAQ_C_Total', 'SDS-SDS_Total_Raw',
                'SDS-SDS_Total_T',
                'PreInt_EduHx-computerinternet_hoursday', 'BMI_Age','Internet_Hours_Age','BMI_Internet_Hours',
                'BFP_BMI', 'FFMI_BFP', 'FMI_BFP', 'LST_TBW', 'BFP_BMR', 'BFP_DEE', 'BMR_Weight', 'DEE_Weight',
                'SMM_Height', 'Muscle_to_Fat', 'Hydration_Status', 'ICW_TBW','BMI_PHR']

featuresCols += time_series_cols
train = train[featuresCols + ['sii']]
test = test[featuresCols]


print(train.columns)
print(test.columns)
print(len(featuresCols))


train = train.replace([np.inf], np.nan)
train = train.replace([-np.inf], np.nan)

# if np.any(np.isinf(test)):
test = test.replace([np.inf], np.nan)
test = test.replace([-np.inf], np.nan)

nan_indices = train[train.isnull().any(axis=1)].index
train = train.fillna(0)
test = test.fillna(0)


!pip -q install /kaggle/input/pytorchtabnet/pytorch_tabnet-4.1.0-py3-none-any.whl


from pytorch_tabnet.tab_model import TabNetRegressor
import torch


# New: TabNet

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from pytorch_tabnet.callbacks import Callback
import os
import torch
from pytorch_tabnet.callbacks import Callback

class TabNetWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.model = TabNetRegressor(**kwargs)
        self.kwargs = kwargs
        self.imputer = SimpleImputer(strategy='median')
        self.best_model_path = 'best_tabnet_model.pt'
        
    def fit(self, X, y):
        # Handle missing values
        X_imputed = self.imputer.fit_transform(X)
        
        if hasattr(y, 'values'):
            y = y.values
            
        # Create internal validation set
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_imputed, 
            y, 
            test_size=0.2,
            random_state=42
        )
        
        # Train TabNet model
        history = self.model.fit(
            X_train=X_train,
            y_train=y_train.reshape(-1, 1),
            eval_set=[(X_valid, y_valid.reshape(-1, 1))],
            eval_name=['valid'],
            eval_metric=['mse'],
            max_epochs=200,
            patience=20,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False,
            callbacks=[
                TabNetPretrainedModelCheckpoint(
                    filepath=self.best_model_path,
                    monitor='valid_mse',
                    mode='min',
                    save_best_only=True,
                    verbose=True
                )
            ]
        )
        
        # Load the best model
        if os.path.exists(self.best_model_path):
            self.model.load_model(self.best_model_path)
            os.remove(self.best_model_path)  # Remove temporary file
        
        return self
    
    def predict(self, X):
        X_imputed = self.imputer.transform(X)
        return self.model.predict(X_imputed).flatten()
    
    def __deepcopy__(self, memo):
        # Add deepcopy support for scikit-learn
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

# TabNet hyperparameters
TabNet_Params = {
    'n_d': 64,              # Width of the decision prediction layer
    'n_a': 64,              # Width of the attention embedding for each step
    'n_steps': 5,           # Number of steps in the architecture
    'gamma': 1.5,           # Coefficient for feature selection regularization
    'n_independent': 2,     # Number of independent GLU layer in each GLU block
    'n_shared': 2,          # Number of shared GLU layer in each GLU block
    'lambda_sparse': 1e-4,  # Sparsity regularization
    'optimizer_fn': torch.optim.Adam,
    'optimizer_params': dict(lr=2e-2, weight_decay=1e-5),
    'mask_type': 'entmax',
    'scheduler_params': dict(mode="min", patience=10, min_lr=1e-5, factor=0.5),
    'scheduler_fn': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'verbose': 1,
    'device_name': 'cuda' if torch.cuda.is_available() else 'cpu'
}

class TabNetPretrainedModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_loss', mode='min', 
                 save_best_only=True, verbose=1):
        super().__init__()  # Initialize parent class
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best = float('inf') if mode == 'min' else -float('inf')
        
    def on_train_begin(self, logs=None):
        self.model = self.trainer  # Use trainer itself as model
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return
        
        # Check if current metric is better than best
        if (self.mode == 'min' and current < self.best) or \
           (self.mode == 'max' and current > self.best):
            if self.verbose:
                print(f'\nEpoch {epoch}: {self.monitor} improved from {self.best:.4f} to {current:.4f}')
            self.best = current
            if self.save_best_only:
                self.model.save_model(self.filepath)  # Save the entire model


SEED = 42
n_splits = 5


# Model parameters for LightGBM
Params = {
    'learning_rate': 0.046,
    'max_depth': 12,
    'num_leaves': 478,
    'min_data_in_leaf': 13,
    'feature_fraction': 0.893,
    'bagging_fraction': 0.784,
    'bagging_freq': 4,
    'lambda_l1': 10,  # Increased from 6.59
    'lambda_l2': 0.01,  # Increased from 2.68e-06
    'device': 'cpu'

}


# XGBoost parameters
XGB_Params = {
    'learning_rate': 0.05,
    'max_depth': 6,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1,  # Increased from 0.1
    'reg_lambda': 5,  # Increased from 1
    'random_state': SEED,
    'tree_method': 'gpu_hist',

}


CatBoost_Params = {
    'learning_rate': 0.05,
    'depth': 6,
    'iterations': 200,
    'random_seed': SEED,
    'verbose': 0,
    'l2_leaf_reg': 10,  # Increase this value
    'task_type': 'GPU'

}


def create_mapping(column, dataset):
    unique_values = dataset[column].unique()
    return {value: idx for idx, value in enumerate(unique_values)}

def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def threshold_Rounder(oof_non_rounded, thresholds):
    return np.where(oof_non_rounded < thresholds[0], 0,
                    np.where(oof_non_rounded < thresholds[1], 1,
                             np.where(oof_non_rounded < thresholds[2], 2, 3)))

def evaluate_predictions(thresholds, y_true, oof_non_rounded):
    rounded_p = threshold_Rounder(oof_non_rounded, thresholds)
    return -quadratic_weighted_kappa(y_true, rounded_p)


def TrainML(model_class, test_data):
    X = train.drop(['sii'], axis=1)
    y = train['sii']

    SKF = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    train_S = []
    test_S = []
    
    oof_non_rounded = np.zeros(len(y), dtype=float) 
    oof_rounded = np.zeros(len(y), dtype=int) 
    test_preds = np.zeros((len(test_data), n_splits))

    for fold, (train_idx, test_idx) in enumerate(tqdm(SKF.split(X, y), desc="Training Folds", total=n_splits)):
        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

        model = clone(model_class)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        oof_non_rounded[test_idx] = y_val_pred
        y_val_pred_rounded = y_val_pred.round(0).astype(int)
        oof_rounded[test_idx] = y_val_pred_rounded

        train_kappa = quadratic_weighted_kappa(y_train, y_train_pred.round(0).astype(int))
        val_kappa = quadratic_weighted_kappa(y_val, y_val_pred_rounded)

        train_S.append(train_kappa)
        test_S.append(val_kappa)
        
        test_preds[:, fold] = model.predict(test_data)
        
        print(f"Fold {fold+1} - Train QWK: {train_kappa:.4f}, Validation QWK: {val_kappa:.4f}")
        clear_output(wait=True)

    print(f"Mean Train QWK --> {np.mean(train_S):.4f}")
    print(f"Mean Validation QWK ---> {np.mean(test_S):.4f}")

    KappaOPtimizer = minimize(evaluate_predictions,
                              x0=[0.5, 1.5, 2.5], args=(y, oof_non_rounded), 
                              method='Nelder-Mead')
    assert KappaOPtimizer.success, "Optimization did not converge."
    
    oof_tuned = threshold_Rounder(oof_non_rounded, KappaOPtimizer.x)
    tKappa = quadratic_weighted_kappa(y, oof_tuned)

    print(f"----> || Optimized QWK SCORE :: {Fore.CYAN}{Style.BRIGHT} {tKappa:.3f}{Style.RESET_ALL}")

    tpm = test_preds.mean(axis=1)
    tpTuned = threshold_Rounder(tpm, KappaOPtimizer.x)
    
    submission = pd.DataFrame({
        'id': sample['id'],
        'sii': tpTuned
    })

    return submission


labels = train_imputed[['PCIAT-PCIAT_01', 'PCIAT-PCIAT_02',
       'PCIAT-PCIAT_03', 'PCIAT-PCIAT_04', 'PCIAT-PCIAT_05', 'PCIAT-PCIAT_06',
       'PCIAT-PCIAT_07', 'PCIAT-PCIAT_08', 'PCIAT-PCIAT_09', 'PCIAT-PCIAT_10',
       'PCIAT-PCIAT_11', 'PCIAT-PCIAT_12', 'PCIAT-PCIAT_13', 'PCIAT-PCIAT_14',
       'PCIAT-PCIAT_15', 'PCIAT-PCIAT_16', 'PCIAT-PCIAT_17', 'PCIAT-PCIAT_18',
       'PCIAT-PCIAT_19', 'PCIAT-PCIAT_20']]

labels = labels.fillna(0)


def baseline0(model, final_model, X_train , y_train, labels, X_test):
    n_model = len(labels.columns)
    X_valid, y_valid  = X_train, y_train
    train_labels = labels
    valid_labels = labels

    # X_train, X_valid, y_train, y_valid, train_labels, valid_labels = train_test_split(X_train, y_train, labels,
    #                                                                                 test_size=0.2, random_state=SEED+1)
    models = [Pipeline([('model', model)]) for i in range(n_model)]
    
    train_pred = []
    valid_pred = []
    test_pred = []
    
    for i in range(n_model):
        print(f"########### Model {i} ##############")
        
        models[i].fit(X_train, train_labels[labels.columns[i]].to_numpy().ravel())
        train_pred.append(models[i].predict(X_train))
        valid_pred.append(models[i].predict(X_valid))
        test_pred.append(models[i].predict(X_test))

    train_pred_combined = np.concatenate((np.array(np.transpose(train_pred)), np.array(train_labels)), axis=0)
    y_train_pred_combined = np.concatenate((np.array(y_train), np.array(y_train)), axis=0)
    
    valid_pred = np.array(valid_pred)
    test_pred = np.array(test_pred)

    print("Lens before and after combined")
    print(len(train_pred), len(train_pred_combined))
    print(len(y_train), len(y_train_pred_combined))

    final_model.fit(train_pred_combined, y_train_pred_combined.ravel())
    final_train_pred = final_model.predict(np.transpose(train_pred))
    KappaOPtimizer = minimize(evaluate_predictions,
                          x0=[0.5, 1.5, 2.5], args=(y_train, final_train_pred), 
                          method='Nelder-Mead')

    
    final_valid_pred = final_model.predict(np.transpose(valid_pred))
    
    KappaOPtimizer = minimize(evaluate_predictions,
                          x0=[0.5, 1.5, 2.5], args=(y_valid, final_valid_pred), 
                          method='Nelder-Mead')
    new_valid_pred = threshold_Rounder(final_valid_pred, KappaOPtimizer.x)
    final_valid_score = quadratic_weighted_kappa(y_valid, new_valid_pred)
    print("Validation score: ", final_valid_score)

    final_test_pred = final_model.predict(np.transpose(test_pred))
    final_test_pred = threshold_Rounder(final_test_pred, KappaOPtimizer.x)
    
    return final_test_pred


model = VotingRegressor(estimators=[
    ('lgb', Pipeline(steps=[('regressor', LGBMRegressor(**Params, random_state=SEED, verbose=-1, n_estimators=300))])),
    # ('xgb', Pipeline(steps=[('regressor', XGBRegressor(**XGB_Params))])),
    ('cat', Pipeline(steps=[('regressor', CatBoostRegressor(**CatBoost_Params))])),
    ('rf', Pipeline(steps=[('regressor', RandomForestRegressor(random_state=SEED))])),
    ('gb', Pipeline(steps=[('regressor', GradientBoostingRegressor(random_state=SEED))]))
])


X_train = train.copy().drop(['sii'], axis=1)
y_train = train['sii']


labels.isna().sum()


Submission0 = baseline0(model, model, X_train , y_train, labels, test)
Submission0 = pd.DataFrame({
    'id': sample['id'],
    'sii': Submission0
})

Submission0


voting_model = VotingRegressor(estimators=[
    ('lightgbm', LGBMRegressor(**Params, random_state=SEED, verbose=-1, n_estimators=300)),
    ('xgboost', XGBRegressor(**XGB_Params)),
    ('catboost', CatBoostRegressor(**CatBoost_Params)),
    ('tabnet', TabNetWrapper(**TabNet_Params))
],weights=[4.0,4.0,5.0,4.0])


Submission1 = TrainML(voting_model, test)

Submission1


# Combine models using Voting Regressor
voting_model = VotingRegressor(estimators=[
    ('lightgbm', LGBMRegressor(**Params, random_state=SEED, verbose=-1, n_estimators=300)),
    ('xgboost', XGBRegressor(**XGB_Params)),
    ('catboost', CatBoostRegressor(**CatBoost_Params))
])

# Train the ensemble model
Submission2 = TrainML(voting_model, test)

# Save submission
Submission2


ensemble = VotingRegressor(estimators=[
    ('lgb', Pipeline(steps=[('imputer', imputer), ('regressor', LGBMRegressor(random_state=SEED))])),
    ('xgb', Pipeline(steps=[('imputer', imputer), ('regressor', XGBRegressor(random_state=SEED))])),
    ('cat', Pipeline(steps=[('imputer', imputer), ('regressor', CatBoostRegressor(random_state=SEED, silent=True))])),
    ('rf', Pipeline(steps=[('imputer', imputer), ('regressor', RandomForestRegressor(random_state=SEED))])),
    ('gb', Pipeline(steps=[('imputer', imputer), ('regressor', GradientBoostingRegressor(random_state=SEED))]))
])

Submission3 = TrainML(ensemble, test)
Submission3


sub0 = Submission0
sub1 = Submission1
sub2 = Submission2
sub3 = Submission3

sub0 = sub0.sort_values(by='id').reset_index(drop=True)
sub1 = sub1.sort_values(by='id').reset_index(drop=True)
sub2 = sub2.sort_values(by='id').reset_index(drop=True)
sub3 = sub3.sort_values(by='id').reset_index(drop=True)

combined = pd.DataFrame({
    'id': sub1['id'],
    'sii_0': sub0['sii'],
    'sii_1': sub1['sii'],
    'sii_2': sub2['sii'],
    'sii_3': sub3['sii']
})

def majority_vote(row):
    return row.mode()[0]

combined['final_sii'] = combined[['sii_0', 'sii_1', 'sii_2', 'sii_3']].apply(majority_vote, axis=1)

final_submission = combined[['id', 'final_sii']].rename(columns={'final_sii': 'sii'})

final_submission.to_csv('submission.csv', index=False)

print("Majority voting completed and saved to 'Final_Submission.csv'")


final_submission
















