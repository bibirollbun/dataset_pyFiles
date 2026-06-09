import numpy as np
import pandas as pd
import os
from sklearn.base import clone
from sklearn.metrics import cohen_kappa_score, make_scorer, confusion_matrix
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import warnings
from sklearn.linear_model import ElasticNetCV, LassoCV, Lasso, LinearRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import shap
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')


print(1+1)


SEED = 643
n_splits = 10
optimize_params = False
n_trials = 25 
voting = True
base_thresholds = [30, 50, 80]


TRAIN_PATH = '/kaggle/input/child-mind-institute-problematic-internet-use/train.csv'
TEST_PATH = '/kaggle/input/child-mind-institute-problematic-internet-use/test.csv'
TRAIN_TS_PATH = '/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet'
TEST_TS_PATH = '/kaggle/input/child-mind-institute-problematic-internet-use/series_test.parquet'


# 加载数据
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
train.head(3)


train.shape


def describe_x(df):
    X = df['X']
    return [
        X.std(),
    ]

def describe_y(df):
    Y = df['Y']
    return [
        Y.std(),
    ]

def describe_z(df):
    Z = df['Z']
    return [
        Z.std(),  
    ]

def describe_enmo(df):
    enmo = df['enmo']
    return [
        enmo.mean(),  
    ]

def describe_anglez(df):
    anglez = df['anglez']
    return [
        anglez.std(),
    ]
    
# Light level thresholds (in lux)
light_bins = [
    (0, 5, 'Twilight'),
    (5, 10, 'Minimal Street Lighting'),
    (10, 50, 'Sunset'),
    (50, 80, 'Family Living Room'),
    (80, 100, 'Hallway'),
    (100, 320, 'Very Dark Overcast Day'),
    (320, 500, 'Office Lighting'),
    (500, 1000, 'Sunrise/Sunset'),
    (1000, 10000, 'Overcast Day'),
    (10000, 25000, 'Full Daylight'),
    (25000, 130000, 'Direct Sunlight')
]


def categorize_light(light_value):
    for low, high, label in light_bins:
        if low <= light_value < high:
            return label
    return 'Unknown'

def describe_light(df):
    df['light_category'] = df['light'].apply(categorize_light)
    light_categories = df['light_category'].value_counts(normalize=True).to_dict()
    
    features = [light_categories.get(label, 0) for _, _, label in light_bins]
    return features

def longest_inactivity_streaks(df, window_size=100, threshold=10, top_n=5):
    rolling_cumsum = df['enmo'].rolling(window=window_size).sum()
    inactive = rolling_cumsum <= threshold
    
    # Calculate streaks
    streak_lengths = []
    current_streak = 0
    for is_inactive in inactive:
        if is_inactive:
            current_streak += 1
        else:
            if current_streak > 0:
                streak_lengths.append(current_streak)
            current_streak = 0
    
    # If the last streak is still active, add it
    if current_streak > 0:
        streak_lengths.append(current_streak)
    
    # Sort streaks in descending order and pick top N
    streak_lengths = sorted(streak_lengths, reverse=True)[:top_n]
    
    # Pad with zeros if there are fewer than N streaks
    streak_lengths += [0] * (top_n - len(streak_lengths))
    return streak_lengths


def longest_activity_streaks(df, window_size=100, threshold=1, top_n=5):
    # Calculate cumsum of enmo in the defined window
    rolling_cumsum = df['enmo'].rolling(window=window_size).sum()
    
    # Identify active windows (cumsum > threshold)
    active = rolling_cumsum > threshold
    
    # Calculate streaks
    streak_lengths = []
    current_streak = 0
    for is_active in active:
        if is_active:
            current_streak += 1
        else:
            if current_streak > 0:
                streak_lengths.append(current_streak)
            current_streak = 0
    
    # If the last streak is still active, add it
    if current_streak > 0:
        streak_lengths.append(current_streak)
    
    # Sort streaks in descending order and pick top N
    streak_lengths = sorted(streak_lengths, reverse=True)[:top_n]
    
    # Pad with zeros if there are fewer than N streaks
    streak_lengths += [0] * (top_n - len(streak_lengths))
    return streak_lengths


def process_file(filename, dirname):
    df = pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet'))
    df.drop(['step'], axis=1, inplace=True)
   
    features = []
    features.extend(describe_x(df))
    features.extend(describe_y(df))
    features.extend(describe_z(df))
    features.extend(describe_enmo(df))
    features.extend(describe_anglez(df))
    features.extend(describe_light(df))  
    
    enmo_active_ratio = (df['enmo'] > 0).mean()
    features.append(enmo_active_ratio)
    features.extend(longest_inactivity_streaks(df, threshold=1))
    features.extend(longest_activity_streaks(df, threshold=5))
   
    return np.array(features), filename.split('=')[1]



def load_time_series(dirname) -> pd.DataFrame:
    ids = os.listdir(dirname)
    
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda fname: process_file(fname, dirname), ids), total=len(ids)))
    
    stats, indexes = zip(*results)
    
    df = pd.DataFrame(stats, columns=[f"stat_{i}" for i in range(len(stats[0]))])
    df['id'] = indexes
    return df


train_ts = load_time_series("/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet")
test_ts = load_time_series("/kaggle/input/child-mind-institute-problematic-internet-use/series_test.parquet")


df_train = train_ts.drop('id', axis=1)


scaler = StandardScaler()
df_train = pd.DataFrame(scaler.fit_transform(df_train), columns=df_train.columns)


df_train.head(3)


df_test = test_ts.drop('id', axis=1)


df_test = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns)


for c in df_train.columns:
    m = np.mean(df_train[c])
    df_train[c].fillna(m, inplace=True)
    df_test[c].fillna(m, inplace=True)


print(df_train.shape)



df_train['id'] = train_ts['id']
df_test['id'] = test_ts['id']

train = pd.merge(train, df_train, how="left", on='id')
test = pd.merge(test, df_test, how="left", on='id')
train.shape


train.columns


def clean_features(df):
    # Remove highly implausible values
    # Clip Grip
    df[['FGC-FGC_GSND', 'FGC-FGC_GSD']] = df[['FGC-FGC_GSND', 'FGC-FGC_GSD']].clip(lower=9, upper=60)
    # Remove implausible body-fat(Body Fat Percentage)
    df["BIA-BIA_Fat"] = np.where(df["BIA-BIA_Fat"] <= 0, np.nan, df["BIA-BIA_Fat"])
    # Basal Metabolic Rate(基础代谢率)
    df["BIA-BIA_BMR"] = np.where(df["BIA-BIA_BMR"] > 4000, np.nan, df["BIA-BIA_BMR"])
    # Daily Energy Expenditure(每日能耗)
    df["BIA-BIA_DEE"] = np.where(df["BIA-BIA_DEE"] > 8000, np.nan, df["BIA-BIA_DEE"])
    # Bone Mineral Content（骨矿物含量）
    df["BIA-BIA_BMC"] = np.where(df["BIA-BIA_BMC"] <= 0, np.nan, df["BIA-BIA_BMC"])
    df["BIA-BIA_BMC"] = np.where(df["BIA-BIA_BMC"] > 30, np.nan, df["BIA-BIA_BMC"])
    # Fat Free Mass Index（肌肉质量指数，去脂体重与身高平方的比值）
    df["BIA-BIA_FFM"] = np.where(df["BIA-BIA_FFM"] > 300, np.nan, df["BIA-BIA_FFM"])
    # Fat Mass Index(脂肪质量指数)
    df["BIA-BIA_FMI"] = np.where(df["BIA-BIA_FMI"] <= 0, np.nan, df["BIA-BIA_FMI"])
    # Extra Cellular Water（细胞外液）
    df["BIA-BIA_ECW"] = np.where(df["BIA-BIA_ECW"] > 100, np.nan, df["BIA-BIA_ECW"])
    # Intra Cellular Water（细胞内液）
    df["BIA-BIA_ICW"] = np.where(df["BIA-BIA_ICW"] > 100, np.nan, df["BIA-BIA_ICW"])
    # Lean Dry Mass（去脂肪体重）s
    df["BIA-BIA_LDM"] = np.where(df["BIA-BIA_LDM"] > 100, np.nan, df["BIA-BIA_LDM"])
    # Lean Soft Tissue（瘦软组织是指人体中除了脂肪组织以外的软组织部分）
    df["BIA-BIA_LST"] = np.where(df["BIA-BIA_LST"] > 400, np.nan, df["BIA-BIA_LST"])
    # Skeletal Muscle Mass（骨骼肌）
    df["BIA-BIA_SMM"] = np.where(df["BIA-BIA_SMM"] > 300, np.nan, df["BIA-BIA_SMM"])
    # Total Body Water（水份）
    df["BIA-BIA_TBW"] = np.where(df["BIA-BIA_TBW"] > 300, np.nan, df["BIA-BIA_TBW"])

    df["BIA-BIA_BMI"] = np.where(df["BIA-BIA_BMI"] <10, np.nan, df["BIA-BIA_BMI"])

    df["BIA-BIA_FFMI"] = np.where(df["BIA-BIA_FFMI"] > 100, np.nan, df["BIA-BIA_FFMI"])
    
    return df

train = clean_features(train)
test = clean_features(test)


train.head(3)


train.shape


def feature_engineering(df):
    season_cols = [col for col in df.columns if 'Season' in col]
    df = df.drop(season_cols, axis=1) 
    
    # From here on own features
    def assign_group(age):
        thresholds = [5, 6, 7, 8, 10, 12, 14, 17, 22]
        for i, j in enumerate(thresholds):
            if age <= j:
                return i
        return np.nan
    
    # Age groups
    df["group"] = df['Basic_Demos-Age'].apply(assign_group)
    
    # BMI (body mass index)体质指数,衡量健康水平
    BMI_map = {0: 16.3,1: 15.9,2: 16.1,3: 16.8,4: 17.3,5: 19.2,6: 20.2,7: 22.3, 8: 23.6}
    df['BMI_mean_norm'] = df[['Physical-BMI', 'BIA-BIA_BMI']].mean(axis=1) / df["group"].map(BMI_map)
    
    # FGC zone aggregate （aerobic capacity, muscular strength, muscular endurance, flexibility, and body composition.）（可改）
    zones = ['FGC-FGC_CU_Zone', 'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD_Zone',
            'FGC-FGC_PU_Zone', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR_Zone',
            'FGC-FGC_TL_Zone']
    
    df['FGC_Zones_mean'] = df[zones].mean(axis=1)
    df['FGC_Zones_min'] = df[zones].min(axis=1)
    df['FGC_Zones_max'] = df[zones].max(axis=1)
    
    # Grip（抓力）
    GSD_max_map = {0: 9, 1: 9, 2: 9, 3: 9, 4: 16.2, 5: 19.9, 6: 26.1, 7: 31.3, 8: 35.4}
    GSD_min_map = {0: 9, 1: 9, 2: 9, 3: 9, 4: 14.4, 5: 17.8, 6: 23.4, 7: 27.8, 8: 31.1}
    
    df['GS_max'] = df[['FGC-FGC_GSND', 'FGC-FGC_GSD']].max(axis=1) / df["group"].map(GSD_max_map)
    df['GS_min'] = df[['FGC-FGC_GSND', 'FGC-FGC_GSD']].min(axis=1) / df["group"].map(GSD_min_map)
    
    # Curl-ups（仰卧起坐）, push-ups（俯卧撑）, trunk-lifts（起身式）... normalized based on age-group
    cu_map = {0: 1.0, 1: 3.0, 2: 5.0, 3: 7.0, 4: 10.0, 5: 14.0, 6: 20.0, 7: 20.0, 8: 20.0}
    pu_map = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 7.0, 6: 8.0, 7: 10.0, 8: 14.0}
    tl_map = {0: 8.0, 1: 8.0, 2: 8.0, 3: 9.0, 4: 9.0, 5: 10.0, 6: 10.0, 7: 10.0, 8: 10.0}
    
    df["CU_norm"] = df['FGC-FGC_CU'] / df['group'].map(cu_map)
    df["PU_norm"] = df['FGC-FGC_PU'] / df['group'].map(pu_map)
    df["TL_norm"] = df['FGC-FGC_TL'] / df['group'].map(tl_map)
    
    # Reach （坐位体前屈）
    df["SR_min"] = df[['FGC-FGC_SRL', 'FGC-FGC_SRR']].min(axis=1)
    df["SR_max"] = df[['FGC-FGC_SRL', 'FGC-FGC_SRR']].max(axis=1)

    # BIA Features
    # Energy Expenditure    
    bmr_map = {0: 934.0, 1: 941.0, 2: 999.0, 3: 1048.0, 4: 1283.0, 5: 1255.0, 6: 1481.0, 7: 1519.0, 8: 1650.0}
    dee_map = {0: 1471.0, 1: 1508.0, 2: 1640.0, 3: 1735.0, 4: 2132.0, 5: 2121.0, 6: 2528.0, 7: 2566.0, 8: 2793.0}
    df["BMR_norm"] = df["BIA-BIA_BMR"] / df["group"].map(bmr_map)  # 基础代谢率
    df["DEE_norm"] = df["BIA-BIA_DEE"] / df["group"].map(dee_map) # 每日能耗
    df["DEE_BMR"] = df["BIA-BIA_DEE"] - df["BIA-BIA_BMR"]

    # FMM （去脂体重）
    ffm_map = {0: 42.0, 1: 43.0, 2: 49.0, 3: 54.0, 4: 60.0, 5: 76.0, 6: 94.0, 7: 104.0, 8: 111.0}
    df["FFM_norm"] = df["BIA-BIA_FFM"] / df["group"].map(ffm_map)

    # ECW ICW
    df["ECW_to_ICW"] = df["BIA-BIA_ECW"] / df["BIA-BIA_ICW"]
    
    # 
    df['Hydration_Status'] = df['BIA-BIA_TBW'] / df['Physical-Weight']

    # 
    df['Muscle_to_Fat'] = df['BIA-BIA_SMM'] / df['BIA-BIA_FMI']

    #
    df['ECW_to_SMM'] = df["BIA-BIA_ECW"] / df['BIA-BIA_SMM']
    
    # Fitness_Endurance
    df["Fitness_Endurance-Time"] = df["Fitness_Endurance-Time_Mins"]*60+df["Fitness_Endurance-Time_Sec"]
    
    drop_feats = ['FGC-FGC_GSND', 'FGC-FGC_GSD', 'FGC-FGC_CU_Zone', 'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD_Zone',
                'FGC-FGC_PU_Zone', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR_Zone', 'FGC-FGC_TL_Zone',
                'Physical-BMI', 'BIA-BIA_BMI', 'FGC-FGC_CU', 'FGC-FGC_PU', 'FGC-FGC_TL', 'FGC-FGC_SRL', 'FGC-FGC_SRR',
                'BIA-BIA_BMR', 'BIA-BIA_DEE', "BIA-BIA_FFM","Fitness_Endurance-Time_Mins","Fitness_Endurance-Time_Sec"]
    df = df.drop(drop_feats, axis=1) 
    return df


# Perform feature_engineering on train and test
train = feature_engineering(train)
test = feature_engineering(test)





train.shape


train.columns


import pandas as pd

def bin_data(train, test, columns, n_bins=10):
    # 检查输入是否为DataFrame
    if not isinstance(train, pd.DataFrame) or not isinstance(test, pd.DataFrame):
        raise ValueError("输入的train和test必须是pandas DataFrame类型。")
    
    # Combine train and test for consistent bin edges
    combined = pd.concat([train, test], axis=0)
    
    bin_edges = {}
    for col in columns:
        try:
            # Compute quantile bin edges
            edges = pd.qcut(combined[col], n_bins, retbins=True, labels=range(n_bins), duplicates="drop")[1]
            # 检查分桶边界是否单调递增
            if not pd.Series(edges).is_monotonic_increasing:
                raise ValueError(f"列 {col} 的分桶边界不是单调递增的。")
            bin_edges[col] = edges
        except ValueError as e:
            print(f"处理列 {col} 时出错: {e}，跳过该列。")
    
    # Apply the same bin edges to both train and test
    for col, edges in bin_edges.items():
        train[col] = pd.cut(
            train[col], bins=edges, labels=range(len(edges) - 1), include_lowest=True
        ).astype(float)
        test[col] = pd.cut(
            test[col], bins=edges, labels=range(len(edges) - 1), include_lowest=True
        ).astype(float)
    
    return train, test

# 假设train和test是已经定义好的DataFrame
# 示例代码中没有给出train和test的定义，你需要确保这两个变量已经正确定义
# train = ...
# test = ...

columns_to_bin = [
    "PAQ_A-PAQ_A_Total", "PAQ_C-PAQ_C_Total","BMR_norm", "DEE_norm", "GS_min", "GS_max", "BIA-BIA_FFMI", 
    "BIA-BIA_BMC", "Physical-HeartRate", "BIA-BIA_ICW", "BIA-BIA_ECW",
    "BIA-BIA_LDM", "BIA-BIA_LST","BIA-BIA_SMM", "BIA-BIA_TBW", "DEE_BMR", "ECW_to_ICW",'Hydration_Status',
    'Muscle_to_Fat','ECW_to_SMM'
]
# Bin specified columns in train and test
try:
    train, test = bin_data(train, test, columns_to_bin, n_bins=10)
except ValueError as e:
    print(f"出现错误: {e}")



train.head(3)


train.columns


# Features to exclude, because they're not in test
exclude = ['PCIAT-Season', 'PCIAT-PCIAT_01', 'PCIAT-PCIAT_02', 'PCIAT-PCIAT_03',
        'PCIAT-PCIAT_04', 'PCIAT-PCIAT_05', 'PCIAT-PCIAT_06', 'PCIAT-PCIAT_07',
        'PCIAT-PCIAT_08', 'PCIAT-PCIAT_09', 'PCIAT-PCIAT_10', 'PCIAT-PCIAT_11',
        'PCIAT-PCIAT_12', 'PCIAT-PCIAT_13', 'PCIAT-PCIAT_14', 'PCIAT-PCIAT_15',
        'PCIAT-PCIAT_16', 'PCIAT-PCIAT_17', 'PCIAT-PCIAT_18', 'PCIAT-PCIAT_19',
        'PCIAT-PCIAT_20', 'PCIAT-PCIAT_Total', 'sii', 'id']

y_model = "PCIAT-PCIAT_Total" # Score, target for the model
y_comp = "sii" # Index, target of the competition
features = [f for f in train.columns if f not in exclude]    
train = train[train["sii"].notna()] # Keep rows where target is available
train.shape


class Impute_With_Model:
    def __init__(self, na_frac=0.5, min_samples=0):
        self.model_dict = {} # Dictionary storing models for imputation
        self.mean_dict = {} # Dictionary storing mean of feature
        self.features = None
        self.na_frac = na_frac # Maximum fraction of missing values allowed for features to be used
        self.min_samples = min_samples # Minimum number of samples required for model fitting
        
    def find_features(self, data, feature, tmp_features):
        # Finds valid features where the fraction of missing values is <= na_frac
        missing_rows = data[feature].isna()
        na_fraction = data[missing_rows][tmp_features].isna().mean(axis=0) # 对于缺失行看其他特征是否完整
        valid_features = np.array(tmp_features)[na_fraction <= self.na_frac]
        return valid_features

    def fit_models(self, model, data, features):
        self.features = features
        n_data = data.shape[0]
        for feature in features:
            self.mean_dict[feature] = np.mean(data[feature])
        # Iterate over all features
        for feature in tqdm(features):
            # Impute if there are missing values in the data
            if data[feature].isna().sum() > 0:
                model_clone = clone(model)
                X = data[data[feature].notna()].copy() # Select data where target values are available as trainings data
                tmp_features = [f for f in features if f != feature]
                tmp_features = self.find_features(data, feature, tmp_features)
                if len(tmp_features) >= 1 and X.shape[0] > self.min_samples:
                    # Fit model if enough features and sufficient samples
                    for f in tmp_features:
                        X[f] = X[f].fillna(self.mean_dict[f])
                    model_clone.fit(X[tmp_features], X[feature])
                    # Add model and features to dictionary
                    self.model_dict[feature] = (model_clone, tmp_features.copy())
                else:
                    # Revert to mean imputation if too few 
                    self.model_dict[feature] = ("mean", np.mean(data[feature]))
            
    def impute(self, data):
        imputed_data = data.copy()
        # Iterate over models
        for feature, model in self.model_dict.items():
            missing_rows = imputed_data[feature].isna() # Identify rows to be imputed
            if missing_rows.any():
                if model[0] == "mean":
                    # Mean imputation if "mean"
                    imputed_data[feature].fillna(model[1], inplace=True)
                else:
                    # Prepare data for imputation and predict
                    tmp_features = [f for f in self.features if f != feature]
                    X_missing = data.loc[missing_rows, tmp_features].copy()
                    for f in tmp_features:
                        X_missing[f] = X_missing[f].fillna(self.mean_dict[f])
                    imputed_data.loc[missing_rows, feature] = model[0].predict(X_missing[model[1]])
        return imputed_data


model = LassoCV(cv=5, random_state=SEED)
imputer = Impute_With_Model(na_frac=0.4) 
imputer.fit_models(model, train, features)
train = imputer.impute(train)
test = imputer.impute(test)


def round_with_thresholds(raw_preds, thresholds):
    return np.where(raw_preds < thresholds[0], int(0),
                    np.where(raw_preds < thresholds[1], int(1),
                            np.where(raw_preds < thresholds[2], int(2), int(3))))

def optimize_thresholds(y_true, raw_preds, start_vals=[0.5, 1.5, 2.5]):
    def fun(thresholds, y_true, raw_preds):
        rounded_preds = round_with_thresholds(raw_preds, thresholds)
        return -cohen_kappa_score(y_true, rounded_preds, weights='quadratic')

    res = minimize(fun, x0=start_vals, args=(y_true, raw_preds), method='Powell')
    assert res.success
    return res.x # 返回优化后的阈值


def calculate_weights(series):
    # Create bins for the target variable and assign weights based on frequency
    bins = pd.cut(series, bins=10, labels=False)
    weights = bins.value_counts().reset_index()
    weights.columns = ['target_bins', 'count']
    weights['count'] = 1 / weights['count']
    weight_map = weights.set_index('target_bins')['count'].to_dict()
    weights = bins.map(weight_map)
    return weights / weights.mean() 


def cross_validate(model_, data, features, score_col, index_col, cv, sample_weights=False, verbose=False):
    """
    Perform cross-validation with a given model and compute the out-of-fold 
    predictions and Cohen's Kappa score for each fold.

    Returns:
    float: Mean Kappa score across all folds.
    array: Out-of-fold score predictions for the entire dataset.
    """
    kappa_scores = [] 
    oof_score_predictions = np.zeros(len(data))  

    score_to_index_thresholds = base_thresholds  
    thresholds = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(data, data[index_col])):
        X_train, X_val = data[features].iloc[train_idx], data[features].iloc[val_idx]  #划分训练集与测试集
        y_train_score = data[score_col].iloc[train_idx]  # 训练集分数
        y_train_index = data[index_col].iloc[train_idx]  # 训练集分类
        y_val_score = data[score_col].iloc[val_idx]  # 测试集分数
        y_val_index = data[index_col].iloc[val_idx]  # 测试集分类 
        
        # Train model with sample weights if provided
        if sample_weights:
            weights = calculate_weights(y_train_score)
            model_.fit(X_train, y_train_score, sample_weight=weights)
        else:
            model_.fit(X_train, y_train_score)

        y_pred_train_score = model_.predict(X_train)
        y_pred_val_score = model_.predict(X_val)
        
        oof_score_predictions[val_idx] = y_pred_val_score 

        # Find optimal threshold in sample 
        t_1 = optimize_thresholds(y_train_index, y_pred_train_score, start_vals=base_thresholds) # 返回优化后的阈值
        thresholds.append(t_1)

        y_pred_val_index = round_with_thresholds(y_pred_val_score, t_1)

        kappa_score = cohen_kappa_score(y_val_index, y_pred_val_index, weights='quadratic')
        kappa_scores.append(kappa_score)
        
        if verbose:
            print(f"Fold {fold_idx}: Optimized Kappa Score = {kappa_score}")
    
    if verbose:
        print(f"## Mean CV Kappa Score: {np.mean(kappa_scores)} ##")
        print(f"## Std CV: {np.std(kappa_scores)}")
    
    return np.mean(kappa_scores), oof_score_predictions, thresholds

def n_cross_validate(model_, data, features, score_col, index_col, cv, seeds, sample_weights=False, verbose=False):
    # Performs repeated cross-validation by reseeding the cv object
    scores = []
    for seed in seeds:
        cv.random_state=seed
        score, oof, _ = cross_validate(model_, data, features, score_col, index_col, cv, sample_weights=True, verbose=False)
        scores.append(score)
    return np.mean(score), oof # score


# for optuna
def objective(trial, model_type, X, features, score_col, index_col, cv, sample_weights=False):
    # Parameter space to explore if model is xgboost
    if model_type == 'xgboost':
        params = {
            'objective': trial.suggest_categorical('objective', ['reg:tweedie', 'reg:pseudohubererror']),
            'random_state': SEED,
            'num_parallel_tree': trial.suggest_int('num_parallel_tree', 2, 30),
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 2, 4),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.02, 0.05),
            'subsample': trial.suggest_float('subsample', 0.5, 0.8),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-5, 1e-1),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-5, 1e-1),
        }
        if params['objective'] == 'reg:tweedie':
            params['tweedie_variance_power'] = trial.suggest_float('tweedie_variance_power', 1, 2)
        model = XGBRegressor(**params, use_label_encoder=False)
    
    # Parameter space to explore if model is lightgbm
    elif model_type == 'lightgbm':
        params = {
            'objective': trial.suggest_categorical('objective', ['poisson', 'tweedie', 'regression']),
            'random_state': SEED,
            'verbosity': -1,
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 2, 4),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.05),
            'subsample': trial.suggest_float('subsample', 0.5, 0.8),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100)
        }
        if params['objective'] == 'tweedie':
            params['tweedie_variance_power'] = trial.suggest_float('tweedie_variance_power', 1, 2)
        model = LGBMRegressor(**params)
    
    # Parameter space to explore if model is catboost
    elif model_type == 'catboost':
        params = {
            'loss_function': trial.suggest_categorical('objective', ['Tweedie:variance_power=1.5', 
                                                                    'Poisson', 'RMSE']),
            'random_state': SEED,
            'iterations': trial.suggest_int('iterations', 100, 300),
            'depth': trial.suggest_int('depth', 2, 4),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.05),
            'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 1e-1),
            'subsample': trial.suggest_float('subsample', 0.5, 0.7),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 60),
        }
        model = CatBoostRegressor(**params, verbose=0)
    
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
        
    seeds = [random.randint(1, 10000) for _ in range(20)] # Seeds for repeated KFold

    score, _ = n_cross_validate(model, X, features, score_col, index_col, cv, seeds, sample_weights=True, verbose=True)

    return score


def run_optimization(X, features, score_col, index_col, model_type, n_trials=30, cv=None, sample_weights=False):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, model_type, X, features, score_col, index_col, cv, sample_weights), 
                n_trials=n_trials)
    
    print(f"Best params for {model_type}: {study.best_params}")
    print(f"Best score: {study.best_value}")
    return study.best_params


features


lgb_features = features
xgb_features = features
cat_features = features
print(len(features))


# 用于在交叉验证过程中生成折叠（folds），同时保证每个折叠中的各类别的比例与整个数据集中的相同
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


train.head(3)


# lgb_params = run_optimization(train, lgb_features, 'PCIAT-PCIAT_Total', 'sii', 'lightgbm', n_trials=n_trials, cv=kf, sample_weights=True)


lgb_params = {
    'objective': 'tweedie', 
    'n_estimators': 104, 
    'max_depth': 4, 
    'learning_rate':0.03803102883584067, 
    'subsample':0.5018987434259187, 
    'colsample_bytree': 0.7796775245856102, 
    'min_data_in_leaf': 26,
    'tweedie_variance_power': 1.1151854443935258
}


# xgb_params = run_optimization(train, xgb_features, 'PCIAT-PCIAT_Total', 'sii', 'xgboost', n_trials=n_trials, cv=kf, sample_weights=True)


xgb_params = {
    'objective': 'reg:tweedie', 
    'num_parallel_tree': 10, 
    'n_estimators': 103, 
    'max_depth': 4, 
    'learning_rate': 0.041516820857180226, 
    'subsample':  0.5584143457265243, 
    'colsample_bytree': 0.5452085572379025, 
    'reg_alpha': 7.342903529712683e-05,
    'reg_lambda': 0.0005153495971550512, 
    'tweedie_variance_power':  1.2404020613161162
}


# cat_params = run_optimization(train, cat_features, 'PCIAT-PCIAT_Total', 'sii', 'catboost', n_trials=n_trials, cv=kf, sample_weights=True)


cat_params = {
    'objective': 'RMSE', 
    'iterations': 300, 
    'depth': 3, 
    'learning_rate': 0.026355664467056426, 
    'l2_leaf_reg':0.03478233218557398, 
    'subsample': 0.6211868950201196, 
    'bagging_temperature': 0.6743370890806787, 
    'random_strength': 0.07766429401725408, 
    'min_data_in_leaf': 33
}


lgb_model = LGBMRegressor(**lgb_params, random_state=SEED, verbosity=-1)
weights = calculate_weights(train['PCIAT-PCIAT_Total'])
# Cross-validate LGBM model
score_lgb, oof_lgb, lgb_thresholds = cross_validate(
    lgb_model, train, lgb_features, 'PCIAT-PCIAT_Total', 'sii', kf, verbose=True, sample_weights=True
)
# Fit final model and predict test samples
lgb_model.fit(train[lgb_features], train['PCIAT-PCIAT_Total'], sample_weight=weights)
test_lgb = lgb_model.predict(test[lgb_features])
print(score_lgb)


# 创建 SHAP 解释器
explainer = shap.Explainer(lgb_model)

# 计算 SHAP 值
shap_values = explainer(train[lgb_features])

# 可视化 SHAP 总结图
shap.summary_plot(shap_values.values, train[lgb_features], feature_names=xgb_features)    


lgb_thresholds_ens = np.mean(np.array(lgb_thresholds), axis=0)
lgb_thresholds_ens




# 绘制散点图
scatter1 = plt.scatter(train['PCIAT-PCIAT_Total'], oof_lgb, c=train["sii"], cmap="autumn", alpha=0.5)

# 设置坐标轴标签
plt.xlabel("True Score")
plt.ylabel("OOF Predictions - LGBM")

# 设置坐标轴范围
plt.ylim(0, np.max(train['PCIAT-PCIAT_Total']))
plt.xlim(0, np.max(train['PCIAT-PCIAT_Total']))

# 设置坐标轴比例
plt.gca().set_aspect('equal', adjustable='box')

# 绘制阈值线
thresholds = [30, 50, 80]
for threshold in thresholds:
    plt.axvline(threshold, color="blue", linestyle="--", lw=1)
for threshold in lgb_thresholds_ens:
    plt.axhline(threshold, color="blue", linestyle="--", lw=1)

# 显示图形
plt.show()


xgb_model = XGBRegressor(**xgb_params, random_state=SEED, verbosity=0)
# Cross-validate XGBoost model
score_xgb, oof_xgb, xgb_thresholds = cross_validate(
    xgb_model, train, xgb_features, 'PCIAT-PCIAT_Total', 'sii', kf, verbose=True, sample_weights=True
)
# Fit final model and predict test samples
xgb_model.fit(train[xgb_features], train['PCIAT-PCIAT_Total'], sample_weight=weights)
test_xgb = xgb_model.predict(test[xgb_features])
print(score_xgb)


# 创建 SHAP 解释器
explainer = shap.Explainer(xgb_model)

# 计算 SHAP 值
shap_values = explainer(train[xgb_features])

# 可视化 SHAP 总结图
shap.summary_plot(shap_values.values, train[xgb_features], feature_names=xgb_features)    


xgb_thresholds_ens = np.mean(np.array(xgb_thresholds), axis=0)
xgb_thresholds_ens



# 绘制散点图
scatter1 = plt.scatter(train['PCIAT-PCIAT_Total'], oof_xgb, c=train["sii"], cmap="autumn", alpha=0.5)

# 设置坐标轴标签
plt.xlabel("True Score")
plt.ylabel("OOF Predictions - XGB")

# 设置坐标轴范围
plt.ylim(0, np.max(train['PCIAT-PCIAT_Total']))
plt.xlim(0, np.max(train['PCIAT-PCIAT_Total']))

# 设置坐标轴比例
plt.gca().set_aspect('equal', adjustable='box')

# 绘制阈值线
thresholds = [30, 50, 80]
for threshold in thresholds:
    plt.axvline(threshold, color="blue", linestyle="--", lw=1)
for threshold in xgb_thresholds_ens:
    plt.axhline(threshold, color="blue", linestyle="--", lw=1)

# 显示图形
plt.show()


cat_model = CatBoostRegressor(**cat_params, random_state=SEED, verbose=0)
# Cross-validate CatBoost model
score_cat, oof_cat, cat_thresholds = cross_validate(
    cat_model, train, cat_features, 'PCIAT-PCIAT_Total', 'sii', kf, verbose=True, sample_weights=True
)
# Fit final model and predict test samples
cat_model.fit(train[cat_features], train['PCIAT-PCIAT_Total'], sample_weight=weights)
test_cat = cat_model.predict(test[cat_features])

print(score_cat)


# 创建 SHAP 解释器
explainer = shap.Explainer(cat_model)

# 计算 SHAP 值
shap_values = explainer(train[cat_features])

# 打印第一个样本的 SHAP 值
print("第一个样本的 SHAP 值：")
print(shap_values.values[0])

# 可视化 SHAP 总结图
shap.summary_plot(shap_values.values, train[cat_features], feature_names=cat_features)    


cat_thresholds_ens = np.mean(np.array(cat_thresholds), axis=0)
cat_thresholds_ens


# 绘制散点图
scatter1 = plt.scatter(train['PCIAT-PCIAT_Total'], oof_cat, c=train["sii"], cmap="autumn", alpha=0.5)

# 设置坐标轴标签
plt.xlabel("True Score")
plt.ylabel("OOF Predictions - XCat")

# 设置坐标轴范围
plt.ylim(0, np.max(train['PCIAT-PCIAT_Total']))
plt.xlim(0, np.max(train['PCIAT-PCIAT_Total']))

# 设置坐标轴比例
plt.gca().set_aspect('equal', adjustable='box')

# 绘制阈值线
thresholds = [30, 50, 80]
for threshold in thresholds:
    plt.axvline(threshold, color="blue", linestyle="--", lw=1)
for threshold in cat_thresholds_ens:
    plt.axhline(threshold, color="blue", linestyle="--", lw=1)

# 显示图形
plt.show()


sns.set_theme(style="white")
fig, axes = plt.subplots(1,3, figsize=(14, 6))

scatter1 = axes[0].scatter(train['PCIAT-PCIAT_Total'], oof_xgb, c=train["sii"], cmap="autumn", alpha=0.5)
axes[0].set_xlabel("True Score")
axes[0].set_ylabel("OOF Predictions - XGB")
axes[0].set_ylim(0,np.max(train['PCIAT-PCIAT_Total']))
axes[0].set_xlim(0,np.max(train['PCIAT-PCIAT_Total']))
axes[0].set_aspect('equal', adjustable='box')
thresholds = [30, 50, 80]
for threshold in thresholds:
    axes[0].axvline(threshold, color="blue", linestyle="--", lw=1)
for threshold in xgb_thresholds_ens:
    axes[0].axhline(threshold, color="blue", linestyle="--", lw=1)
    
scatter2 = axes[1].scatter(train['PCIAT-PCIAT_Total'], oof_lgb, c=train["sii"], cmap="autumn", alpha=0.5)
axes[1].set_xlabel("True Score")
axes[1].set_ylabel("OOF Predictions - LGBM")
axes[1].set_ylim(0,np.max(train['PCIAT-PCIAT_Total']))
axes[1].set_xlim(0,np.max(train['PCIAT-PCIAT_Total']))
axes[1].set_aspect('equal', adjustable='box')

for threshold in thresholds:
    axes[1].axvline(threshold, color="blue", linestyle="--", lw=1)
for threshold in lgb_thresholds_ens:
    axes[1].axhline(threshold, color="blue", linestyle="--", lw=1)
    
scatter3 = axes[2].scatter(train['PCIAT-PCIAT_Total'], oof_cat, c=train["sii"], cmap="autumn", alpha=0.5)
axes[2].set_xlabel("True Score")
axes[2].set_ylabel("OOF Predictions - Cat")
axes[2].set_ylim(0,np.max(train['PCIAT-PCIAT_Total']))
axes[2].set_xlim(0,np.max(train['PCIAT-PCIAT_Total']))
axes[2].set_aspect('equal', adjustable='box')

for threshold in thresholds:
    axes[2].axvline(threshold, color="blue", linestyle="--", lw=1)
for threshold in cat_thresholds_ens:
    axes[2].axhline(threshold, color="blue", linestyle="--", lw=1)

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1,2, figsize=(20,6))

model_preds = pd.DataFrame({
    'lgb': oof_lgb,
    'xgb': oof_xgb,
    'cat': oof_cat
})

corr_df = model_preds.corr()
sns.heatmap(corr_df, annot=True, cmap="autumn", cbar=False, linewidths=0.5, linecolor='black', ax=axes[0])
axes[0].set_title("Correlation Between Models")

lgb_thresholds_ens = np.mean(np.array(lgb_thresholds), axis=0)
xgb_thresholds_ens = np.mean(np.array(xgb_thresholds), axis=0)
cat_thresholds_ens = np.mean(np.array(cat_thresholds), axis=0)
thresholds_df = pd.DataFrame({
    "LGB Thresholds": lgb_thresholds_ens,
    "XGB Thresholds": xgb_thresholds_ens,
    "Cat Thresholds": cat_thresholds_ens
})

sns.heatmap(thresholds_df, annot=True, cmap="autumn", cbar=False, linewidths=0.5, linecolor='black', ax=axes[1])
axes[1].set_title("Optimal Thresholds Derived from CV")
axes[1].set_xticklabels(thresholds_df.columns, rotation=45)  
plt.show()


oof_lgb_index = round_with_thresholds(oof_lgb, lgb_thresholds_ens)
print(f"LGBM optimized Kappa: {cohen_kappa_score(train['sii'], oof_lgb_index , weights='quadratic')}")


oof_xgb_index  = round_with_thresholds(oof_xgb, xgb_thresholds_ens)
print(f"XGB optimized Kappa: {cohen_kappa_score(train['sii'], oof_xgb_index , weights='quadratic')}")

oof_cat_index  = round_with_thresholds(oof_cat, cat_thresholds_ens)
print(f"CAT optimized Kappa: {cohen_kappa_score(train['sii'], oof_cat_index , weights='quadratic')}")


# Showing how sensitive QWK is with changes in threshold
scores = []
ts = []
m = 30
for i in tqdm(np.linspace(-2,10, 100)):
    thresholds = [m+i, 50, 80]
    pred = round_with_thresholds(oof_xgb, thresholds)
    score = cohen_kappa_score(train["sii"], pred, weights='quadratic')
    ts.append(m+i)
    scores.append(score)
    
plt.plot(ts, scores,color='orange')
plt.title("Demonstration of QWK sensitivity to changes in threshold 0")
plt.show()


lgb_importances = pd.DataFrame({
    'Feature': lgb_features,
    'Importance': lgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

xgb_importances = pd.DataFrame({
    'Feature': xgb_features,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

cat_importances = pd.DataFrame({
    'Feature': cat_features,
    'Importance': cat_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Set the number of features to display
n_top_features = 40

fig, axes = plt.subplots(1, 3, figsize=(18, 8))
sns.set_theme(style="whitegrid")

sns.barplot(ax=axes[0], data=lgb_importances.head(n_top_features),
            x='Importance', y='Feature', palette="autumn")
axes[0].set_title('LightGBM Top Feature Importances')

sns.barplot(ax=axes[1], data=xgb_importances.head(n_top_features),
            x='Importance', y='Feature', palette="autumn")
axes[1].set_title('XGBoost Top Feature Importances')

sns.barplot(ax=axes[2], data=cat_importances.head(n_top_features),
            x='Importance', y='Feature', palette="autumn")
axes[2].set_title('CatBoost Top Feature Importances')

plt.tight_layout()
plt.show()


(set(lgb_importances[:10]["Feature"]) & set(xgb_importances[:10]["Feature"]) & set(cat_importances[:10]["Feature"]))


weights = [0.3,0.5,0.2]
oof_preds = np.array([oof_lgb_index, oof_xgb_index, oof_cat_index])
weighted_oof = np.average(oof_preds, axis=0, weights=weights)
final_oof = np.round(weighted_oof).astype(int)


final_oof


pd.DataFrame(final_oof).to_csv("final_oof.csv")


# Calculate Kappa score for voted OOF predictions
kappa_score = cohen_kappa_score(train["sii"], final_oof, weights='quadratic')
print(f"Ensemble Kappa score: {kappa_score}")


# Plot confusion matrix
conf_matrix = confusion_matrix(train["sii"], final_oof)
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="autumn", cbar=False, linewidths=0.5, linecolor='black')
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.show()


conf_matrix = confusion_matrix(train["sii"], oof_lgb_index)
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="autumn", cbar=False, linewidths=0.5, linecolor='black')
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.show()


conf_matrix = confusion_matrix(train["sii"], oof_xgb_index)
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="autumn", cbar=False, linewidths=0.5, linecolor='black')
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.show()


conf_matrix = confusion_matrix(train["sii"], oof_cat_index)
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="autumn", cbar=False, linewidths=0.5, linecolor='black')
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.show()


# 设置绘图主题
sns.set_theme(style="whitegrid")
# 创建包含 1 行 3 列子图的图形
fig, axes = plt.subplots(1, 3, figsize=(14, 6))

# 绘制 XGBoost 模型的混淆矩阵
conf_matrix = confusion_matrix(train["sii"], oof_xgb_index)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="autumn", cbar=False, linewidths=0.5, linecolor='black', ax=axes[0])
axes[0].set_title('Confusion Matrix (XGBoost)', fontsize=16)
axes[0].set_xlabel('Predicted', fontsize=12)
axes[0].set_ylabel('True', fontsize=12)

# 绘制 LightGBM 模型的混淆矩阵
conf_matrix = confusion_matrix(train["sii"], oof_lgb_index)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="autumn", cbar=False, linewidths=0.5, linecolor='black', ax=axes[1])
axes[1].set_title('Confusion Matrix (LightGBM)', fontsize=16)
axes[1].set_xlabel('Predicted', fontsize=12)
axes[1].set_ylabel('True', fontsize=12)

# 绘制 CatBoost 模型的混淆矩阵
conf_matrix = confusion_matrix(train["sii"], oof_cat_index)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="autumn", cbar=False, linewidths=0.5, linecolor='black', ax=axes[2])
axes[2].set_title('Confusion Matrix (CatBoost)', fontsize=16)
axes[2].set_xlabel('Predicted', fontsize=12)
axes[2].set_ylabel('True', fontsize=12)

# 自动调整子图布局
plt.tight_layout()
# 显示图形
plt.show()


test_lgb_index = round_with_thresholds(test_lgb, lgb_thresholds_ens)
test_xgb_index = round_with_thresholds(test_xgb, xgb_thresholds_ens)
test_cat_index = round_with_thresholds(test_cat, cat_thresholds_ens)
if voting:
    test_preds = np.array([test_lgb_index, test_xgb_index, test_cat_index])
    voted_test = stats.mode(test_preds, axis=0).mode.flatten().astype(int)
    final_test = np.round(voted_test).astype(int)


submission = pd.read_csv("/kaggle/input/child-mind-institute-problematic-internet-use/sample_submission.csv")


submission['sii'] = final_test
submission.to_csv("submission.csv", index=False)

