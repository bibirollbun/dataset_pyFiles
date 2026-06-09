import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
import warnings

warnings.filterwarnings('ignore')




def process_file(filename, dirname):
    """Process a single parquet file and extract time-based features"""
    df = pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet'))
    
    # Drop 'step' column if it exists
    if 'step' in df.columns:
        df.drop('step', axis=1, inplace=True)
    
    # Convert time_of_day to hours
    df["hours"] = df["time_of_day"] // (3_600 * 1_000_000_000)
    
    # Define time periods
    night = ((df["hours"] >= 22) | (df["hours"] <= 5))
    day = ((df["hours"] <= 20) & (df["hours"] >= 7))
    
    # Initialize features dictionary
    features = {}
    
    # Basic activity features
    features['non_wear_mean'] = df["non-wear_flag"].mean()
    features['active_enmo_sum'] = df["enmo"][df["enmo"] >= 0.05].sum()
    
    # Process each column for different time periods
    for col in ['enmo', 'anglez', 'light', 'battery_voltage']:
        # Full day statistics
        features[f"{col}_mean"] = df[col].mean()
        features[f"{col}_std"] = df[col].std()
        features[f"{col}_max"] = df[col].max()
        features[f"{col}_min"] = df[col].min()
        features[f"{col}_diff_mean"] = df[col].diff().mean()
        features[f"{col}_diff_std"] = df[col].diff().std()
        
        # Night time statistics
        night_data = df.loc[night, col]
        features[f"{col}_night_mean"] = night_data.mean()
        features[f"{col}_night_std"] = night_data.std()
        features[f"{col}_night_max"] = night_data.max()
        features[f"{col}_night_min"] = night_data.min()
        
        # Day time statistics
        day_data = df.loc[day, col]
        features[f"{col}_day_mean"] = day_data.mean()
        features[f"{col}_day_std"] = day_data.std()
        features[f"{col}_day_max"] = day_data.max()
        features[f"{col}_day_min"] = day_data.min()
    
    return features, filename.split('=')[1]

def load_data_parquet(dirname) -> pd.DataFrame:
    """Load and process time series data from directory in parallel"""
    ids = os.listdir(dirname)
    
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda fname: process_file(fname, dirname), ids), total=len(ids)))
    
    features_list, indexes = zip(*results)
    
    # Create DataFrame with extracted features and IDs
    df = pd.DataFrame(features_list)
    df['id'] = indexes
    
    return df


from sklearn.preprocessing import OneHotEncoder
train = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
test = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')
sample = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/sample_submission.csv')

# chia thành 3 nhóm features chính (Bộ dữ liệu khách quan csv)
demographicFeatures = ['Basic_Demos-Enroll_Season', 'Basic_Demos-Age', 'Basic_Demos-Sex']

seasonFeatures = ['Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season', 
          'Fitness_Endurance-Season', 'FGC-Season', 'BIA-Season', 
          'PAQ_A-Season', 'PAQ_C-Season', 'SDS-Season', 'PreInt_EduHx-Season', 'PCIAT-Season']

train_ts = load_data_parquet('/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet')
test_ts = load_data_parquet('/kaggle/input/child-mind-institute-problematic-internet-use/series_test.parquet')

time_series_cols = train_ts.columns.tolist()
time_series_cols.remove("id")

train = pd.merge(train, train_ts, how="left", on='id')
test = pd.merge(test, test_ts, how="left", on='id')



print(train.shape)
print(test.shape)



print(set(train.columns) - set(test.columns))


columns_not_in_test = sorted(list(set(train.columns) - set(test.columns)))

columns_to_exclude = ['PCIAT-PCIAT_Total', 'PCIAT-Season', 'sii']
question_columns = [
    col for col in columns_not_in_test if col not in columns_to_exclude
]

question_columns


na_total_rows = train[train['sii'].isna()]
na_total_rows


train = train.dropna(subset=['PCIAT-PCIAT_Total'])
train


wh_cols = [
    'Physical-BMI', 'Physical-Height',
    'Physical-Weight', 'Physical-Waist_Circumference'
]

train[wh_cols] = train[wh_cols].replace(0, np.nan)
test[wh_cols] = test[wh_cols].replace(0, np.nan)
train[wh_cols].describe()


object_columns = train.select_dtypes(include='object').columns
print("Các cột kiểu object:")
print(object_columns)



# Danh sách các cột thuộc nhóm BIA
bia_cols = [
    'BIA-BIA_BMC', 'BIA-BIA_BMI', 'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_ECW', 'BIA-BIA_FFM',
    'BIA-BIA_FFMI', 'BIA-BIA_FMI', 'BIA-BIA_Fat', 'BIA-BIA_ICW', 'BIA-BIA_LDM',
    'BIA-BIA_LST', 'BIA-BIA_SMM', 'BIA-BIA_TBW'
]
# Các thuộc tính nhóm này không thể là âm => Replace bằng NaN

print("\n======== Trước khi Replace ========")
print("\nTập train:")
print((train[bia_cols] < 0).sum())
print("\nTập test:")
print((test[bia_cols] < 0).sum())

train[bia_cols] = train[bia_cols].applymap(lambda x: np.nan if x < 0 else x)
# Tập test ko có nên ko cần replace

# Kiểm tra lại sau khi Replace
print("\n======== Sau khi Replace ========")
print("\nTập train:")
print((train[bia_cols] < 0).sum())


def cleaning_features(df):
    # Xóa các dữ liệu bị sai lệch lớn (vô lý)
    
    # % mỡ cơ thể
    df['BIA-BIA_Fat'] = np.where(df['BIA-BIA_Fat'] < 5, np.nan, df['BIA-BIA_Fat'])
    df['BIA-BIA_Fat'] = np.where(df['BIA-BIA_Fat'] > 60, np.nan, df['BIA-BIA_Fat'])
    # Bone Mineral Content (Khối lượng khoáng trong xương) (kg)
    df['BIA-BIA_BMC'] = np.where(df['BIA-BIA_BMC'] < 0.5, np.nan, df['BIA-BIA_BMC'])
    df['BIA-BIA_BMC'] = np.where(df['BIA-BIA_BMC'] > 5, np.nan, df['BIA-BIA_BMC'])
    # Body Mass Index (chỉ số khối cơ thể)
    df['BIA-BIA_BMI'] = np.where(df['BIA-BIA_BMI'] < 10, np.nan, df['BIA-BIA_BMI'])
    df['BIA-BIA_BMI'] = np.where(df['BIA-BIA_BMI'] > 40, np.nan, df['BIA-BIA_BMI'])
    # Basal Metabolic Rate (kcal/ngày)
    df['BIA-BIA_BMR'] = np.where(df['BIA-BIA_BMR'] < 600, np.nan, df['BIA-BIA_BMR'])
    df['BIA-BIA_BMR'] = np.where(df['BIA-BIA_BMR'] > 3000, np.nan, df['BIA-BIA_BMR'])
    # Daily Energy Expenditure (kcal/ngày)
    df['BIA-BIA_DEE'] = np.where(df['BIA-BIA_DEE'] < 800, np.nan, df['BIA-BIA_DEE'])
    df['BIA-BIA_DEE'] = np.where(df['BIA-BIA_DEE'] > 5000, np.nan, df['BIA-BIA_DEE'])
    # Extracellular Water – Nước ngoài tế bào (L)
    df['BIA-BIA_ECW'] = np.where(df['BIA-BIA_ECW'] < 3, np.nan, df['BIA-BIA_ECW'])
    df['BIA-BIA_ECW'] = np.where(df['BIA-BIA_ECW'] > 15, np.nan, df['BIA-BIA_ECW'])
    # Fat-Free Mass – Khối lượng không mỡ (kg)
    df['BIA-BIA_FFM'] = np.where(df['BIA-BIA_FFM'] < 10, np.nan, df['BIA-BIA_FFM'])
    df['BIA-BIA_FFM'] = np.where(df['BIA-BIA_FFM'] > 60, np.nan, df['BIA-BIA_FFM'])
    # Fat-Free Mass Index
    df['BIA-BIA_FFMI'] = np.where(df['BIA-BIA_FFMI'] < 8, np.nan, df['BIA-BIA_FFMI'])
    df['BIA-BIA_FFMI'] = np.where(df['BIA-BIA_FFMI'] > 25, np.nan, df['BIA-BIA_FFMI'])
    # Fat Mass Index
    df['BIA-BIA_FMI'] = np.where(df['BIA-BIA_FMI'] < 1, np.nan, df['BIA-BIA_FMI'])
    df['BIA-BIA_FMI'] = np.where(df['BIA-BIA_FMI'] > 15, np.nan, df['BIA-BIA_FMI'])
    # Intracellular Water – Nước trong tế bào (L)
    df['BIA-BIA_ICW'] = np.where(df['BIA-BIA_ICW'] < 5, np.nan, df['BIA-BIA_ICW'])
    df['BIA-BIA_ICW'] = np.where(df['BIA-BIA_ICW'] > 20, np.nan, df['BIA-BIA_ICW'])
    # Lean Dry Mass – Khối lượng nạc không nước (kg)
    df['BIA-BIA_LDM'] = np.where(df['BIA-BIA_LDM'] < 5, np.nan, df['BIA-BIA_LDM'])
    df['BIA-BIA_LDM'] = np.where(df['BIA-BIA_LDM'] > 40, np.nan, df['BIA-BIA_LDM'])
    # Lean Soft Tissue – Mô mềm không mỡ (kg)
    df['BIA-BIA_LST'] = np.where(df['BIA-BIA_LST'] < 10, np.nan, df['BIA-BIA_LST'])
    df['BIA-BIA_LST'] = np.where(df['BIA-BIA_LST'] > 55, np.nan, df['BIA-BIA_LST'])
    # Skeletal Muscle Mass – Khối cơ xương (kg)
    df['BIA-BIA_SMM'] = np.where(df['BIA-BIA_SMM'] < 5, np.nan, df['BIA-BIA_SMM'])
    df['BIA-BIA_SMM'] = np.where(df['BIA-BIA_SMM'] > 40, np.nan, df['BIA-BIA_SMM'])
    # Total Body Water – Tổng lượng nước trong cơ thể (L)
    df['BIA-BIA_TBW'] = np.where(df['BIA-BIA_TBW'] < 10, np.nan, df['BIA-BIA_TBW'])
    df['BIA-BIA_TBW'] = np.where(df['BIA-BIA_TBW'] > 40, np.nan, df['BIA-BIA_TBW'])

    return df

train = cleaning_features(train)
test = cleaning_features(test)


train = train.drop(columns=question_columns, errors='ignore')
train = train.drop(columns=seasonFeatures, errors='ignore')
test = test.drop(columns=seasonFeatures, errors='ignore')
train = train.drop(columns='PCIAT-PCIAT_Total', errors='ignore')



filtered_features = train.drop(columns=['id', 'sii'])  

# Chuẩn bị dữ liệu X và y
X = filtered_features
y = train['sii']               

# Định nghĩa pipeline xử lý dữ liệu số
num_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=3)),
    ('scaler', StandardScaler())
])

# Định nghĩa ColumnTransformer để áp dụng pipeline cho các cột số
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, filtered_features.columns.tolist())
])

# Fit và transform X
preprocessor.fit(X)
X_transformed = pd.DataFrame(preprocessor.transform(X), columns=filtered_features.columns)

# Kiểm tra các dòng đầu tiên của dữ liệu đã transform
print("Transformed X DataFrame:")
print(X_transformed.head())


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_transformed, y, test_size=0.2, random_state=42)


seed = 2023



LGB_Param = {
    'learning_rate': 0.046,
    'max_depth': 12,
    'num_leaves': 478,
    'min_data_in_leaf': 13,
    'feature_fraction': 0.893,
    'bagging_fraction': 0.784,
    'bagging_freq': 4,
    'lambda_l1': 10,
    'lambda_l2': 0.01
}


XGBoost_Param = {
    'learning_rate': 0.05,
    'max_depth': 6,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1,  
    'reg_lambda': 3,  
    'random_state': seed,
}

XGBoost_Param.update({
    'learning_rate': 0.01,
    'max_depth': 5,
    'n_estimators': 500,
    'reg_alpha': 5,
    'reg_lambda': 10,
})

CatBoost_Param = {
    'learning_rate': 0.05,
        'depth': 6,
        'iterations': 200,
        'random_seed': seed,
        'verbose': 0,
        'l2_leaf_reg': 10,
}


GradientBoost_Param = {
    'n_estimators': 204,
    'learning_rate': 0.03055022094677994,
    'max_depth': 7,
    'min_samples_split': 20,
    'min_samples_leaf': 18,
    'subsample': 0.8363079223697193,
    'max_features': 'log2'
}


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor, 
    AdaBoostRegressor,
    ExtraTreesRegressor,
    BaggingRegressor,
    StackingRegressor,
    VotingRegressor
)
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from scipy.optimize import minimize

# Cấu hình
seed = 2023
np.random.seed(seed)
warnings.filterwarnings("ignore")

# Khởi tạo các mô hình Regressor
LGB_Model = LGBMRegressor(**LGB_Param,random_state=seed, verbose=-1)
CatBoost_Model = CatBoostRegressor(**CatBoost_Param)
XGB_Model = XGBRegressor(**XGBoost_Param)

# Hàm tính Quadratic Weighted Kappa
def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred.round().astype(int), weights='quadratic')

# Hàm tối ưu Threshold Rounding
def threshold_Rounder(oof_non_rounded, thresholds):
    return np.where(oof_non_rounded < thresholds[0], 0,
                    np.where(oof_non_rounded < thresholds[1], 1,
                             np.where(oof_non_rounded < thresholds[2], 2, 3)))

def evaluate_predictions(thresholds, y_true, oof_non_rounded):
    rounded_p = threshold_Rounder(oof_non_rounded, thresholds)
    return -quadratic_weighted_kappa(y_true, rounded_p)

# Tạo mô hình StackingRegressor
stacking_model = StackingRegressor(
    estimators=[
        ('lgb', LGB_Model),
        ('cat', CatBoost_Model),
        ('xgb', XGB_Model)
    ]
)


# Preprocess the test data
X_test = test.drop(columns=['id'])
X_test = pd.DataFrame(preprocessor.transform(X_test), columns=filtered_features.columns)

# Train mô hình StackingRegressor
best_model = StackingRegressor(
    estimators=[
        ('lgb', LGB_Model),
        ('cat', CatBoost_Model),
        ('xgb', XGB_Model)
    ],
    #final_estimator=GradientBoostingRegressor(**GradientBoost_Param, random_state=seed)  # Meta-model
)

#best_model = GradientBoostingRegressor(**GradientBoost_Param, random_state=seed)

best_model.fit(X_train, y_train)

# Dự đoán trên tập test
y_test_pred = best_model.predict(X_test)

# Tối ưu hóa threshold để làm tròn
KappaOptimizer = minimize(
    evaluate_predictions,
    x0=[0.5, 1.5, 2.5], args=(y_train, best_model.predict(X_train)),
    method='Nelder-Mead'
)
optimized_thresholds = KappaOptimizer.x

# Làm tròn kết quả về nhãn classification
y_test_pred_rounded = threshold_Rounder(y_test_pred, optimized_thresholds)

# Tạo file submission
submission = pd.DataFrame({
    'id': test['id'],
    'sii': y_test_pred_rounded
})
submission.to_csv('submission.csv', index=False)

# Load và kiểm tra kết quả
submiss = pd.read_csv('submission.csv')
print(submiss)
print("✅ Submission file created successfully.")  # In ra thông báo khi hoàn thành




