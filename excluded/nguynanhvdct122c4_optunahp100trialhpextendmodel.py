import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize


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

featuresCols = ['Basic_Demos-Enroll_Season', 'Basic_Demos-Age', 'Basic_Demos-Sex',
                'CGAS-Season', 'CGAS-CGAS_Score', 'Physical-Season', 'Physical-BMI',
                'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumference',
                'Physical-Diastolic_BP', 'Physical-HeartRate', 'Physical-Systolic_BP',
                'Fitness_Endurance-Season', 'Fitness_Endurance-Max_Stage',
                'Fitness_Endurance-Time_Mins', 'Fitness_Endurance-Time_Sec',
                'FGC-Season', 'FGC-FGC_CU', 'FGC-FGC_CU_Zone', 'FGC-FGC_GSND',
                'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD', 'FGC-FGC_GSD_Zone', 'FGC-FGC_PU',
                'FGC-FGC_PU_Zone', 'FGC-FGC_SRL', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR',
                'FGC-FGC_SRR_Zone', 'FGC-FGC_TL', 'FGC-FGC_TL_Zone', 'BIA-Season',
                'BIA-BIA_Activity_Level_num', 'BIA-BIA_BMC', 'BIA-BIA_BMI',
                'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_ECW', 'BIA-BIA_FFM',
                'BIA-BIA_FFMI', 'BIA-BIA_FMI', 'BIA-BIA_Fat', 'BIA-BIA_Frame_num',
                'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_SMM',
                'BIA-BIA_TBW', 'PAQ_A-Season', 'PAQ_A-PAQ_A_Total', 'PAQ_C-Season',
                'PAQ_C-PAQ_C_Total', 'SDS-Season', 'SDS-SDS_Total_Raw',
                'SDS-SDS_Total_T', 'PreInt_EduHx-Season',
                'PreInt_EduHx-computerinternet_hoursday','sii']

# chia thành 3 nhóm features chính (Bộ dữ liệu khách quan csv)
demographicFeatures = ['Basic_Demos-Enroll_Season', 'Basic_Demos-Age', 'sii']

phycsicsFeatures = ['CGAS-Season', 'CGAS-CGAS_Score', 'Physical-Season', 'Physical-BMI',
                'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumference',
                'Physical-Diastolic_BP', 'Physical-HeartRate', 'Physical-Systolic_BP',
                'Fitness_Endurance-Season', 'Fitness_Endurance-Max_Stage',
                'Fitness_Endurance-Time_Mins', 'Fitness_Endurance-Time_Sec',
                'FGC-Season', 'FGC-FGC_CU', 'FGC-FGC_CU_Zone', 'FGC-FGC_GSND',
                'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD', 'FGC-FGC_GSD_Zone', 'FGC-FGC_PU',
                'FGC-FGC_PU_Zone', 'FGC-FGC_SRL', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR',
                'FGC-FGC_SRR_Zone', 'FGC-FGC_TL', 'FGC-FGC_TL_Zone', 'BIA-Season',
                'BIA-BIA_Activity_Level_num', 'BIA-BIA_BMC', 'BIA-BIA_BMI',
                'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_ECW', 'BIA-BIA_FFM',
                'BIA-BIA_FFMI', 'BIA-BIA_FMI', 'BIA-BIA_Fat', 'BIA-BIA_Frame_num',
                'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_SMM',
                'BIA-BIA_TBW','sii' ]

behaviorFeatures = ['PAQ_A-Season', 'PAQ_A-PAQ_A_Total', 'PAQ_C-Season',
                    'SDS-SDS_Total_T', 'PreInt_EduHx-Season',
                'PreInt_EduHx-computerinternet_hoursday','sii']

cat_c = ['Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season', 
          'Fitness_Endurance-Season', 'FGC-Season', 'BIA-Season', 
          'PAQ_A-Season', 'PAQ_C-Season', 'SDS-Season', 'PreInt_EduHx-Season']

train_ts = load_data_parquet('/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet')
test_ts = load_data_parquet('/kaggle/input/child-mind-institute-problematic-internet-use/series_test.parquet')

time_series_cols = train_ts.columns.tolist()
time_series_cols.remove("id")

train = pd.merge(train, train_ts, how="left", on='id')
test = pd.merge(test, test_ts, how="left", on='id')


featuresCols += time_series_cols

train_df = train[featuresCols]


# Loại bỏ các feature liên quan đến "season"
filtered_features = [feature for feature in featuresCols if feature not in cat_c and feature != 'sii']

# Loại bỏ các hàng có giá trị NaN trong y
train_df = train_df[featuresCols].dropna(subset=['sii'])

# Chuẩn bị dữ liệu X và y
X = train_df[filtered_features]
y = train_df['sii']

# Định nghĩa pipeline xử lý dữ liệu số
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Định nghĩa ColumnTransformer để áp dụng pipeline cho các cột số
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, filtered_features)
])

# Fit và transform X
preprocessor.fit(X)
X_transformed = pd.DataFrame(preprocessor.transform(X), columns=filtered_features)

# Kiểm tra các dòng đầu tiên của dữ liệu đã transform
print("Transformed X DataFrame:")
print(X_transformed.head())


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_transformed, y, test_size=0.2, random_state=42)


def threshold_Rounder(oof_non_rounded, thresholds):
    return np.where(oof_non_rounded < thresholds[0], 0,
                    np.where(oof_non_rounded < thresholds[1], 1,
                             np.where(oof_non_rounded < thresholds[2], 2, 3)))


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
LGB_Model = LGBMRegressor(random_state=seed, verbose=-1, n_estimators=300)
CatBoost_Model = CatBoostRegressor(random_state=seed, verbose=0)
XGB_Model = XGBRegressor(random_state=seed)

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

# Danh sách mô hình
models = [stacking_model]
model_names = ['StackingRegressor']

# Hàm đánh giá mô hình
def generate_baseline_results(models, model_names, X, y, cv=5, plot_results=False):
    """
    Đánh giá nhiều mô hình bằng cross-validation với kỹ thuật threshold rounding.
    
    Parameters:
    -----------
    models: list - Danh sách các mô hình đã khởi tạo.
    model_names: list - Danh sách tên các mô hình.
    X: DataFrame - Ma trận đặc trưng.
    y: Series - Biến mục tiêu.
    cv: int - Số fold cho cross-validation.
    plot_results: bool - Có vẽ đồ thị kết quả hay không.

    Returns:
    --------
    DataFrame - Kết quả hiệu suất của các mô hình.
    """
    print(f"Đánh giá {len(models)} mô hình với {cv}-fold cross-validation...")

    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    entries = []

    for model, model_name in zip(models, model_names):
        print(f"Đang đánh giá: {model_name}")
        try:
            oof_predictions = np.zeros(len(y), dtype=float)  # Lưu dự đoán liên tục
            
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model.fit(X_train, y_train)  # Train mô hình
                y_val_pred = model.predict(X_val)  # Dự đoán hồi quy
                
                oof_predictions[val_idx] = y_val_pred  # Lưu lại dự đoán

            # Tối ưu hóa thresholds
            KappaOptimizer = minimize(
                evaluate_predictions,
                x0=[0.5, 1.5, 2.5], args=(y, oof_predictions), method='Nelder-Mead'
            )
            optimized_thresholds = KappaOptimizer.x

            # Áp dụng thresholds đã tối ưu
            rounded_preds = threshold_Rounder(oof_predictions, optimized_thresholds)

            # Tính toán QWK
            qwk_score = quadratic_weighted_kappa(y, rounded_preds)

            entries.append((model_name, qwk_score))
            print(f"  • {model_name} - QWK Score: {qwk_score:.4f}")
        
        except Exception as e:
            print(f"  • Lỗi với {model_name}: {e}")

    # Tạo DataFrame kết quả
    results_df = pd.DataFrame(entries, columns=['model_name', 'qwk_score'])
    results_df.sort_values(by='qwk_score', ascending=False, inplace=True)

    # Vẽ biểu đồ kết quả
    if plot_results and not results_df.empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(x='model_name', y='qwk_score', data=results_df, color='lightblue', edgecolor='black')
        plt.title("Hiệu suất mô hình (QWK Score)", fontsize=14)
        plt.xlabel("Mô hình", fontsize=12)
        plt.ylabel("Quadratic Weighted Kappa", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    print("\n===== KẾT QUẢ CUỐI CÙNG =====")
    print(results_df)
    return results_df

# Sử dụng hàm để chạy toàn bộ đánh giá
print("Bắt đầu đánh giá mô hình...\n")
cv_results = generate_baseline_results(models, model_names, X_train, y_train, cv=5, plot_results=True)



# Tạo cell mới để đánh giá mô hình đơn lẻ
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize

# Định nghĩa hàm để đánh giá mô hình đơn lẻ
def evaluate_single_models(X_train, y_train, X_val, y_val, verbose=True):
    """
    Đánh giá hiệu suất của các mô hình đơn lẻ trước khi kết hợp.
    
    Tham số:
    ---------
    X_train: DataFrame - Dữ liệu huấn luyện
    y_train: Series - Nhãn huấn luyện
    X_val: DataFrame - Dữ liệu kiểm tra
    y_val: Series - Nhãn kiểm tra
    verbose: bool - In kết quả chi tiết hay không
    
    Trả về:
    --------
    dict - Từ điển chứa mô hình tốt nhất và điểm QWK của nó
    """
    # Khởi tạo các mô hình cơ bản
    models = {
        'LightGBM': LGBMRegressor(random_state=seed, verbose=-1, n_estimators=300),
        'XGBoost': XGBRegressor(random_state=seed),
        'CatBoost': CatBoostRegressor(random_state=seed, verbose=0),
        'RandomForest': RandomForestRegressor(random_state=seed),
        'GradientBoosting': GradientBoostingRegressor(random_state=seed)
    }
    
    results = {}
    
    if verbose:
        print("===== ĐÁNH GIÁ MÔ HÌNH ĐƠN LẺ =====")
    
    # Huấn luyện và đánh giá từng mô hình
    for name, model in models.items():
        if verbose:
            print(f"Đang huấn luyện: {name}...")
        
        # Huấn luyện mô hình
        model.fit(X_train, y_train)
        
        # Dự đoán trên tập validation
        val_preds = model.predict(X_val)
        
        # Tối ưu ngưỡng cho QWK
        optimizer = minimize(
            evaluate_predictions,
            x0=[0.5, 1.5, 2.5], 
            args=(y_val, val_preds),
            method='Nelder-Mead'
        )
        optimized_thresholds = optimizer.x
        
        # Áp dụng ngưỡng đã tối ưu
        rounded_preds = threshold_Rounder(val_preds, optimized_thresholds)
        
        # Tính điểm QWK
        qwk = quadratic_weighted_kappa(y_val, rounded_preds)
        results[name] = {
            'model': model,
            'score': qwk,
            'thresholds': optimized_thresholds
        }
        
        if verbose:
            print(f"  • {name} - QWK Score: {qwk:.4f}")
    
    # Sắp xếp kết quả theo điểm số
    sorted_results = sorted(results.items(), key=lambda x: x[1]['score'], reverse=True)
    best_model_name, best_model_info = sorted_results[0]
    
    if verbose:
        print("\n===== KẾT QUẢ CUỐI CÙNG =====")
        print(f"Mô hình tốt nhất: {best_model_name} với QWK Score: {best_model_info['score']:.4f}")
    
    return {
        'best_model_name': best_model_name,
        'best_model': best_model_info['model'],
        'best_score': best_model_info['score'],
        'best_thresholds': best_model_info['thresholds'],
        'all_results': results
    }

# Chạy đánh giá
single_model_results = evaluate_single_models(X_train, y_train, X_val, y_val)

# Lấy top 3 mô hình tốt nhất để kết hợp
top_models = sorted(single_model_results['all_results'].items(), 
                    key=lambda x: x[1]['score'], 
                    reverse=True)[:3]

print("\n===== TOP 3 MÔ HÌNH TỐT NHẤT =====")
for i, (name, info) in enumerate(top_models, 1):
    print(f"{i}. {name}: {info['score']:.4f}")


# Cell mới với không gian tìm kiếm mở rộng cho từng loại mô hình
def objective_extended(trial, X_data, y_data, model_type):
    """
    Hàm mục tiêu Optuna với không gian tìm kiếm hyperparameter mở rộng cho một loại mô hình cụ thể.
    
    Tham số:
    ---------
    trial: Đối tượng trial
    X_data: Dữ liệu đặc trưng
    y_data: Dữ liệu nhãn
    model_type: Loại mô hình cần tối ưu
    
    Trả về:
    --------
    float: Điểm QWK trung bình
    """
    if model_type == 'LightGBM':
        params = {
            'objective': trial.suggest_categorical('objective', 
                        ['regression', 'poisson', 'tweedie', 'mape', 'huber', 'quantile']),
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
            'max_depth': trial.suggest_int('max_depth', 2, 15),
            'num_leaves': trial.suggest_int('num_leaves', 20, 200),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1e-1),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1e-1),
            'random_state': seed,
            'verbose': -1
        }
        if params['objective'] == 'tweedie':
            params['tweedie_variance_power'] = trial.suggest_float('tweedie_variance_power', 1.1, 1.9)
        elif params['objective'] == 'quantile':
            params['alpha'] = trial.suggest_float('alpha', 0.05, 0.95)
        elif params['objective'] == 'huber':
            params['alpha'] = trial.suggest_float('huber_alpha', 0.5, 0.95)
        model = LGBMRegressor(**params)

    elif model_type == 'XGBoost':
        params = {
            'objective': trial.suggest_categorical('objective', 
                        ['reg:squarederror', 'reg:pseudohubererror', 'reg:tweedie']),
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
            'max_depth': trial.suggest_int('max_depth', 2, 15),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1e-1),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1e-1),
            'gamma': trial.suggest_float('gamma', 0, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state': seed
        }
        if params['objective'] == 'reg:tweedie':
            params['tweedie_variance_power'] = trial.suggest_float('tweedie_variance_power', 1.1, 1.9)
        model = XGBRegressor(**params)

    elif model_type == 'CatBoost':
        # Định nghĩa loss_function trước
        loss_function = trial.suggest_categorical('loss_function', 
                        ['RMSE', 'MAE', 'Poisson', 'Huber', 'Tweedie', 'Quantile'])
        
        # Xử lý các tham số đặc biệt cho các loss function
        if loss_function == 'Huber':
            delta = trial.suggest_float('huber_delta', 0.1, 10.0)
            loss_function = f'Huber:delta={delta}'
        elif loss_function == 'Tweedie':
            variance_power = trial.suggest_float('tweedie_variance_power', 1.1, 1.9)
            loss_function = f'Tweedie:variance_power={variance_power}'
        elif loss_function == 'Quantile':
            q = trial.suggest_float('quantile_q', 0.1, 0.9)
            loss_function = f'Quantile:alpha={q}'
        
        # Tạo tham số cho CatBoost sau khi đã xử lý loss_function
        params = {
            'loss_function': loss_function,
            'iterations': trial.suggest_int('iterations', 50, 1000),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
            'depth': trial.suggest_int('depth', 2, 10),
            'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-8, 1e-1),
            'random_strength': trial.suggest_float('random_strength', 1e-8, 10),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10),
            'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
            'od_wait': trial.suggest_int('od_wait', 10, 50),
            'random_state': seed,
            'verbose': 0
        }
        
        model = CatBoostRegressor(**params)

    elif model_type == 'RandomForest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 2, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': seed
        }
        model = RandomForestRegressor(**params)

    elif model_type == 'GradientBoosting':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': seed
        }
        model = GradientBoostingRegressor(**params)
    
    # Đánh giá mô hình
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []

    for train_idx, val_idx in kfold.split(X_data, y_data):
        X_train_fold, X_val_fold = X_data.iloc[train_idx], X_data.iloc[val_idx]
        y_train_fold, y_val_fold = y_data.iloc[train_idx], y_data.iloc[val_idx]

        model.fit(X_train_fold, y_train_fold)
        val_preds = model.predict(X_val_fold)

        optimizer = minimize(
            evaluate_predictions,
            x0=[0.5, 1.5, 2.5], 
            args=(y_val_fold, val_preds),
            method='Nelder-Mead'
        )
        optimized_thresholds = optimizer.x
        rounded_preds = threshold_Rounder(val_preds, optimized_thresholds)
        qwk = quadratic_weighted_kappa(y_val_fold, rounded_preds)
        scores.append(qwk)

    return np.mean(scores)

def run_extended_optimization(X_data, y_data, n_trials=50):
    """
    Chạy qúa trình tối ưu hóa cho mỗi loại mô hình riêng biệt.
    
    Tham số:
    ---------
    X_data: Dữ liệu đặc trưng
    y_data: Dữ liệu nhãn
    n_trials: Số lượng trials cho mỗi mô hình
    
    Trả về:
    --------
    dict: Thông tin về các mô hình tốt nhất cho mỗi loại
    """
    print(f"Bắt đầu quá trình tối ưu hóa hyperparameter với {n_trials} trials cho mỗi mô hình...")
    
    # Danh sách các loại mô hình cần tối ưu
    model_types = ['LightGBM', 'XGBoost', 'CatBoost', 'RandomForest', 'GradientBoosting']
    
    # Dictionary lưu kết quả tối ưu cho từng loại mô hình
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Tối ưu hóa mô hình {model_type}")
        print(f"{'='*50}")
        
        # Tạo nghiên cứu mới cho mỗi loại mô hình
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=seed)
        )
        
        # Chạy tối ưu hóa
        study.optimize(
            lambda trial: objective_extended(trial, X_data, y_data, model_type),
            n_trials=n_trials
        )
        
        # In kết quả
        print(f"\n===== KẾT QUẢ TỐI ƯU CHO {model_type} =====")
        print(f"Điểm QWK tốt nhất: {study.best_value:.4f}")
        
        # Tạo mô hình tốt nhất với hyperparameters tối ưu
        best_params = study.best_params
        
        if model_type == 'LightGBM':
            best_model = LGBMRegressor(**best_params)
        elif model_type == 'XGBoost':
            best_model = XGBRegressor(**best_params)
        elif model_type == 'CatBoost':
            best_model = CatBoostRegressor(**best_params)
        elif model_type == 'RandomForest':
            best_model = RandomForestRegressor(**best_params)
        elif model_type == 'GradientBoosting':
            best_model = GradientBoostingRegressor(**best_params)
        
        # Lưu thông tin vào dictionary kết quả
        results[model_type] = {
            'best_model': best_model,
            'best_score': study.best_value,
            'best_params': best_params
        }
        
        # In tham số chi tiết
        print("\nTham số tối ưu:")
        for param_name, param_value in best_params.items():
            print(f"  • {param_name}: {param_value}")
    
    # Tìm mô hình tốt nhất trong tất cả các loại
    best_model_type = max(results.items(), key=lambda x: x[1]['best_score'])[0]
    best_overall = results[best_model_type]
    
    print("\n\n================================================================")
    print(f"MÔ HÌNH TỐT NHẤT TỔNG THỂ: {best_model_type} với QWK score: {best_overall['best_score']:.4f}")
    print("================================================================")
    
    # Tạo bảng so sánh các mô hình
    comparison_data = []
    for model_type, info in results.items():
        comparison_data.append({
            'Model': model_type,
            'QWK Score': info['best_score']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('QWK Score', ascending=False).reset_index(drop=True)
    
    print("\nSo sánh hiệu suất các mô hình:")
    print(comparison_df)
    
    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Model', y='QWK Score', data=comparison_df, palette='viridis')
    plt.title('Hiệu suất các mô hình sau khi tối ưu hóa', fontsize=14)
    plt.xlabel('Loại mô hình', fontsize=12)
    plt.ylabel('Điểm QWK', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Thêm thông tin mô hình tốt nhất tổng thể vào kết quả
    results['best_overall'] = {
        'model_type': best_model_type,
        'best_model': best_overall['best_model'],
        'best_score': best_overall['best_score'],
        'best_params': best_overall['best_params']
    }
    
    return results

# Chạy tối ưu hóa mở rộng cho từng loại mô hình
extended_results = run_extended_optimization(X_train, y_train, n_trials=150)  # Điều chỉnh n_trials nếu cần


# Cell mới để kết hợp mô hình tốt nhất
def combine_best_models(single_results, extended_results, X_train, y_train, X_val, y_val):
    """
    Kết hợp các mô hình tốt nhất để tạo ensemble.
    
    Tham số:
    ---------
    single_results: Kết quả từ đánh giá mô hình đơn lẻ
    extended_results: Kết quả từ tối ưu mở rộng
    X_train, y_train: Dữ liệu huấn luyện
    X_val, y_val: Dữ liệu kiểm tra
    
    Trả về:
    --------
    dict: Thông tin về mô hình ensemble tốt nhất
    """
    print("===== KẾT HỢP CÁC MÔ HÌNH TỐT NHẤT =====")
    
    # Tạo mô hình từ đánh giá đơn lẻ
    best_single_model = single_results['best_model']
    best_single_name = single_results['best_model_name']
    
    # Tạo mô hình từ tối ưu hóa mở rộng
    # Truy cập đúng mô hình từ 'best_overall' trong extended_results
    best_extended_type = extended_results['best_overall']['model_type']
    best_extended_model = extended_results[best_extended_type]['best_model']
    
    # Thêm các mô hình top vào danh sách
    models_to_combine = []
    model_weights = []
    
    # Thêm mô hình tốt nhất từ đánh giá đơn lẻ
    models_to_combine.append((f'best_single_{best_single_name}', best_single_model))
    model_weights.append(1.0)
    
    # Thêm mô hình tốt nhất từ tối ưu mở rộng
    models_to_combine.append((f'best_extended_{best_extended_type}', best_extended_model))
    model_weights.append(1.0)
    
    # Thêm các mô hình top từ đánh giá đơn lẻ
    for i, (name, info) in enumerate(top_models[:2], 1):  # Chỉ lấy 2 mô hình top
        if name != best_single_name:  # Tránh trùng lặp
            models_to_combine.append((f'top_{i}_{name}', info['model']))
            model_weights.append(0.8)  # Trọng số thấp hơn mô hình tốt nhất
    
    # Tạo VotingRegressor
    voting_model = VotingRegressor(
        estimators=models_to_combine,
        weights=model_weights
    )
    
    # Đánh giá ensemble
    print("Đang đánh giá ensemble model...")
    voting_model.fit(X_train, y_train)
    val_preds = voting_model.predict(X_val)
    
    # Tối ưu ngưỡng 
    optimizer = minimize(
        evaluate_predictions,
        x0=[0.5, 1.5, 2.5], 
        args=(y_val, val_preds),
        method='Nelder-Mead'
    )
    optimized_thresholds = optimizer.x
    
    # Áp dụng ngưỡng tối ưu
    rounded_preds = threshold_Rounder(val_preds, optimized_thresholds)
    
    # Tính QWK
    ensemble_qwk = quadratic_weighted_kappa(y_val, rounded_preds)
    print(f"Ensemble Model - QWK Score: {ensemble_qwk:.4f}")
    
    # So sánh với các mô hình đơn lẻ
    print("\n===== SO SÁNH HIỆU SUẤT =====")
    print(f"Mô hình đơn lẻ tốt nhất ({best_single_name}): {single_results['best_score']:.4f}")
    # Sử dụng điểm số từ best_overall
    print(f"Mô hình tối ưu mở rộng ({best_extended_type}): {extended_results['best_overall']['best_score']:.4f}")
    print(f"Ensemble Model: {ensemble_qwk:.4f}")
    
    # Tạo StackingRegressor (kết hợp khác)
    meta_regressor = RandomForestRegressor(n_estimators=100, random_state=seed)
    stacking_model = StackingRegressor(
        estimators=models_to_combine,
        final_estimator=meta_regressor
    )
    
    # Đánh giá stacking
    print("\nĐang đánh giá stacking model...")
    stacking_model.fit(X_train, y_train)
    stack_val_preds = stacking_model.predict(X_val)
    
    # Tối ưu ngưỡng cho stacking
    stack_optimizer = minimize(
        evaluate_predictions,
        x0=[0.5, 1.5, 2.5], 
        args=(y_val, stack_val_preds),
        method='Nelder-Mead'
    )
    stack_thresholds = stack_optimizer.x
    
    # Áp dụng ngưỡng tối ưu cho stacking
    stack_rounded_preds = threshold_Rounder(stack_val_preds, stack_thresholds)
    
    # Tính QWK cho stacking
    stack_qwk = quadratic_weighted_kappa(y_val, stack_rounded_preds)
    print(f"Stacking Model - QWK Score: {stack_qwk:.4f}")
    
    # Sửa lại cách lấy điểm số của mô hình tối ưu mở rộng
    best_extended_score = extended_results['best_overall']['best_score']
    
    # Xác định mô hình tốt nhất cuối cùng
    best_ensemble = None
    best_ensemble_thresholds = None
    best_ensemble_type = ""
    best_ensemble_score = 0
    
    if ensemble_qwk > stack_qwk and ensemble_qwk > single_results['best_score'] and ensemble_qwk > best_extended_score:
        best_ensemble = voting_model
        best_ensemble_thresholds = optimized_thresholds
        best_ensemble_type = "VotingRegressor"
        best_ensemble_score = ensemble_qwk
    elif stack_qwk > ensemble_qwk and stack_qwk > single_results['best_score'] and stack_qwk > best_extended_score:
        best_ensemble = stacking_model
        best_ensemble_thresholds = stack_thresholds
        best_ensemble_type = "StackingRegressor"
        best_ensemble_score = stack_qwk
    elif single_results['best_score'] > best_extended_score:
        best_ensemble = best_single_model
        best_ensemble_thresholds = single_results['best_thresholds']
        best_ensemble_type = f"Single Model ({best_single_name})"
        best_ensemble_score = single_results['best_score']
    else:
        best_ensemble = best_extended_model
        # Tính ngưỡng tối ưu cho mô hình này
        best_extended_model.fit(X_train, y_train)
        ext_preds = best_extended_model.predict(X_val)
        ext_optimizer = minimize(
            evaluate_predictions,
            x0=[0.5, 1.5, 2.5], 
            args=(y_val, ext_preds),
            method='Nelder-Mead'
        )
        best_ensemble_thresholds = ext_optimizer.x
        best_ensemble_type = f"Extended Optimized Model ({best_extended_type})"
        best_ensemble_score = best_extended_score
    
    print("\n===== MÔ HÌNH TỐT NHẤT CUỐI CÙNG =====")
    print(f"Loại: {best_ensemble_type}")
    print(f"Điểm QWK: {best_ensemble_score:.4f}")
    
    return {
        'model': best_ensemble,
        'thresholds': best_ensemble_thresholds,
        'model_type': best_ensemble_type,
        'score': best_ensemble_score
    }

# Kết hợp các mô hình tốt nhất
final_model_info = combine_best_models(single_model_results, extended_results, 
                                     X_train, y_train, X_val, y_val)


# Extended Model Submission
def make_extended_model_predictions(X_train, y_train, X_val, y_val, X_test_transformed, test, extended_results):
    """
    Tạo dự đoán và file submission cho Extended model (với hyperparameter tuning)
    """
    print("\n===== ĐANG TẠO DỰ ĐOÁN CHO EXTENDED MODEL =====")
    
    # Lấy mô hình tốt nhất từ quá trình tối ưu hóa mở rộng
    best_model_type = extended_results['best_overall']['model_type']
    best_model = extended_results[best_model_type]['best_model']
    
    # Huấn luyện lại trên toàn bộ dữ liệu
    print(f"Đang huấn luyện extended model ({best_model_type})...")
    best_model.fit(X_train, y_train)
    
    # Đánh giá trên tập validation
    val_preds = best_model.predict(X_val)
    
    # Tìm ngưỡng tối ưu
    optimizer = minimize(
        evaluate_predictions,
        x0=[0.5, 1.5, 2.5], 
        args=(y_val, val_preds),
        method='Nelder-Mead'
    )
    extended_thresholds = optimizer.x
    
    # Tính QWK score trên validation
    rounded_val_preds = threshold_Rounder(val_preds, extended_thresholds)
    extended_qwk = quadratic_weighted_kappa(y_val, rounded_val_preds)
    
    # Dự đoán trên tập test
    test_preds = best_model.predict(X_test_transformed)
    
    # Áp dụng ngưỡng đã tối ưu
    rounded_test_preds = threshold_Rounder(test_preds, extended_thresholds)
    
    # Tạo DataFrame kết quả
    submission = pd.DataFrame({
        'id': test['id'],
        'sii': rounded_test_preds.astype(int)
    })
    
    # Phân bố dự đoán
    pred_distribution = pd.Series(rounded_test_preds).value_counts().sort_index()
    print(f"\n===== PHÂN BỐ DỰ ĐOÁN CHO EXTENDED MODEL ({best_model_type}) =====")
    print(pred_distribution)
    
    # Lưu file kết quả - Dùng tên chuẩn cho Kaggle
    submission_file = 'submission.csv'
    submission.to_csv(submission_file, index=False)
    print(f"\nĐã lưu kết quả vào file: {submission_file}")
    
    print(f"\nĐiểm QWK trên tập validation: {extended_qwk:.4f}")
    print(f"Ngưỡng tối ưu: {extended_thresholds}")
    
    return submission

# Tạo dự đoán với extended model
test_df = test[filtered_features]
X_test_transformed = pd.DataFrame(preprocessor.transform(test_df), columns=filtered_features)
extended_submission = make_extended_model_predictions(X_train, y_train, X_val, y_val, X_test_transformed, test, extended_results)

