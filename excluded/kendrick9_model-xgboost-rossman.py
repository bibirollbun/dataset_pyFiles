import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
import warnings

warnings.filterwarnings('ignore')

print("Bước 1: Đang tải dữ liệu...")
try:
    train_df_orig = pd.read_csv("../input/train.csv", low_memory=False, parse_dates=['Date'])
    store_df_orig = pd.read_csv("../input/store.csv", low_memory=False)
    test_df_orig = pd.read_csv("../input/test.csv", low_memory=False, parse_dates=['Date'])
    print("Tải dữ liệu thành công.")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy tệp dữ liệu. Vui lòng đảm bảo các tệp train.csv, store.csv, và test.csv nằm trong cùng thư mục.")
    # exit()

# Tạo bản sao để tránh thay đổi dữ liệu gốc
train_df = train_df_orig.copy()
store_df = store_df_orig.copy()
test_df = test_df_orig.copy()

# Kết hợp dữ liệu train/test với store
print("Đang kết hợp dữ liệu...")
train_df = pd.merge(train_df, store_df, on='Store', how='left')
test_df = pd.merge(test_df, store_df, on='Store', how='left')
print("Kết hợp dữ liệu thành công.")


print("\nBước 2: Đang tiền xử lý dữ liệu...")

def preprocess_data(df, is_train=True):
    # Xử lý giá trị thiếu
    if not is_train and 'Open' in df.columns:
        df['Open'].fillna(1, inplace=True)

    df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
    for col in ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear']:
        df[col].fillna(0, inplace=True)
    df['PromoInterval'].fillna('', inplace=True)

    # Kỹ thuật Đặc trưng
    # Cột Date đã được parse khi đọc file, không cần parse lại ở đây nếu đã dùng parse_dates
    # df['Date'] = pd.to_datetime(df['Date'], errors='coerce') 
    
    df['Year'] = df.Date.dt.year
    df['Month'] = df.Date.dt.month
    df['Day'] = df.Date.dt.day
    df['DayOfWeek'] = df.Date.dt.dayofweek
    df['WeekOfYear'] = df.Date.dt.isocalendar().week.astype(int)
    df['DayOfYear'] = df.Date.dt.dayofyear

    df['CompetitionOpen'] = 12 * (df.Year - df.CompetitionOpenSinceYear) + \
                            (df.Month - df.CompetitionOpenSinceMonth)
    df['CompetitionOpen'] = df['CompetitionOpen'].apply(lambda x: x if x > 0 else 0)

    df['PromoOpen'] = 12 * (df.Year - df.Promo2SinceYear) + \
                      (df.WeekOfYear - df.Promo2SinceWeek) / 4.0
    df['PromoOpen'] = df['PromoOpen'].apply(lambda x: x if x > 0 else 0)

    month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    df['MonthStr'] = df.Month.map(month_map)
    df['IsPromoMonth'] = 0
    for interval in df.PromoInterval.unique():
        if interval != '':
            for month_abbr in interval.split(','):
                df.loc[(df.MonthStr == month_abbr) & (df.PromoInterval == interval), 'IsPromoMonth'] = 1
    
    df['StateHoliday'] = df['StateHoliday'].map({'0': 0, 'a': 1, 'b': 2, 'c': 3, 0:0})
    df['StateHoliday'].fillna(0, inplace=True)

    for col in ['StoreType', 'Assortment']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    if 'Open' in df.columns:
         df['Open'] = df['Open'].astype(int)
    
    # Giữ lại cột Date gốc cho việc chia theo thời gian
    if not is_train: # Chỉ drop Date gốc trong test_df sau khi đã dùng
        df.drop(['MonthStr', 'PromoInterval', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear'], axis=1, inplace=True, errors='ignore')
    else: # Giữ Date cho train_df để split
         df.drop(['MonthStr', 'PromoInterval', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear'], axis=1, inplace=True, errors='ignore')

    return df

train_df = preprocess_data(train_df.copy(), is_train=True)
test_df = preprocess_data(test_df.copy(), is_train=False)


# Xử lý Doanh số (Sales Handling - cho tập huấn luyện)
train_df = train_df[train_df['Open'] == 1]
train_df = train_df[train_df['Sales'] > 0]
train_df['Sales_log'] = np.log1p(train_df['Sales'])

print("Tiền xử lý dữ liệu hoàn tất.")



features = [
    'Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday',
    'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
    'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear',
    'Year', 'Month', 'Day', 'WeekOfYear', 'DayOfYear',
    'CompetitionOpen', 'PromoOpen', 'IsPromoMonth'
]
if 'Customers' in features: # Đảm bảo Customers không còn trong features
    features.remove('Customers')
if 'Open' in features: # Đảm bảo Open không còn trong features (vì đã lọc)
    features.remove('Open')
print("Xử lý dữ liệu hoàn tất.")


print("\nBước 2.2: Đang phân chia tập dữ liệu theo thời gian...")

# features_train là danh sách các cột đặc trưng sẽ được sử dụng
features_train = [f for f in features if f in train_df.columns]


# Sắp xếp dữ liệu theo Store và Date để đảm bảo tính nhất quán khi chia
# Cột 'Date' gốc cần được giữ lại trong train_df cho đến bước này
train_df_sorted = train_df.sort_values(by=['Store', 'Date'])
last_date = train_df_sorted['Date'].max()
split_date = last_date - pd.Timedelta(weeks=6)

X_train_df_time = train_df_sorted[train_df_sorted['Date'] < split_date]
X_val_df_time = train_df_sorted[train_df_sorted['Date'] >= split_date]

X_train = X_train_df_time[features_train]
y_train_log = X_train_df_time['Sales_log']
y_train_orig = X_train_df_time['Sales'] # Sales gốc cho tập train (dùng cho MASE)

X_val = X_val_df_time[features_train]
y_val_log = X_val_df_time['Sales_log']
y_val_orig = X_val_df_time['Sales'] # Sales gốc cho tập val (dùng cho MASE và các metric khác)

# Bây giờ có thể drop cột 'Date' khỏi X_train và X_val nếu các đặc trưng thời gian đã được tạo
# và các mô hình không trực tiếp sử dụng cột 'Date' kiểu datetime
# X_train = X_train.drop('Date', axis=1, errors='ignore')
# X_val = X_val.drop('Date', axis=1, errors='ignore')

# Toàn bộ dữ liệu huấn luyện (X_all, y_all_log, y_all_orig) cho việc huấn luyện mô hình cuối cùng
X_all = train_df_sorted[features_train] # train_df_sorted đã loại bỏ Open=0, Sales=0
y_all_log = train_df_sorted['Sales_log']
y_all_orig = train_df_sorted['Sales']

# Drop cột Date khỏi X_all nếu không cần nữa
# X_all = X_all.drop('Date', axis=1, errors='ignore')
# Tương tự cho test_df
test_df_final = test_df.drop('Date', axis=1, errors='ignore')


print(f"Kích thước tập huấn luyện: {X_train.shape}, tập kiểm định: {X_val.shape}")



def rmspe_func(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    valid_indices = y_true != 0
    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices]
    if len(y_true_valid) == 0: return 0.0
    return np.sqrt(np.mean(((y_true_valid - y_pred_valid) / y_true_valid) ** 2))

def mape_func(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    valid_indices = y_true != 0
    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices]
    if len(y_true_valid) == 0: return 0.0
    return np.mean(np.abs((y_true_valid - y_pred_valid) / y_true_valid)) * 100

def mase_func(y_true_val_orig, y_pred_val_orig, y_true_train_orig):
    mae_model = mean_absolute_error(y_true_val_orig, y_pred_val_orig)
    if len(y_true_train_orig) <= 7: return np.inf
    y_naive_pred_train = y_true_train_orig.iloc[:-7].values
    mae_naive = mean_absolute_error(y_true_train_orig.iloc[7:].values, y_naive_pred_train)
    if mae_naive == 0: return np.inf
    return mae_model / mae_naive

def evaluate_model(model_name, y_val_log_true, y_val_log_pred, y_train_orig_for_mase, y_val_orig_true):
    y_val_orig_pred = np.expm1(y_val_log_pred)
    print(f"\n--- Đánh giá cho mô hình: {model_name} ---")
    print(f"RMSPE: {rmspe_func(y_val_log_true, y_val_log_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_val_orig_true, y_val_orig_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_val_orig_true, y_val_orig_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_val_orig_true, y_val_orig_pred)):.4f}")
    print(f"MAPE: {mape_func(y_val_log_true, y_val_log_pred):.4f}%")
    print(f"R-Squared: {r2_score(y_val_orig_true, y_val_orig_pred):.4f}")
    mase = mase_func(y_val_orig_true, y_val_orig_pred, y_train_orig_for_mase)
    print(f"MASE: {mase:.4f}")

# Hàm RMSPE cho XGBoost (chỉ trả về 2 giá trị)
def rmspe_xgb_feval(y_pred_log, y_true_dmatrix):
    y_true_log_label = y_true_dmatrix.get_label()
    y_true_orig = np.expm1(y_true_log_label)
    y_pred_orig = np.expm1(y_pred_log)
    valid_indices = y_true_orig != 0
    y_true_valid = y_true_orig[valid_indices]
    y_pred_valid = y_pred_orig[valid_indices]
    if len(y_true_valid) == 0:
        return 'rmspe', 0.0
    return 'rmspe', np.sqrt(np.mean(((y_true_valid - y_pred_valid) / y_true_valid) ** 2))

# Hàm RMSPE cho LightGBM (trả về 3 giá trị)
def rmspe_lgb_feval(y_pred_log, y_true_dmatrix):
    y_true_log_label = y_true_dmatrix.get_label()
    y_true_orig = np.expm1(y_true_log_label)
    y_pred_orig = np.expm1(y_pred_log)
    valid_indices = y_true_orig != 0
    y_true_valid = y_true_orig[valid_indices]
    y_pred_valid = y_pred_orig[valid_indices]
    if len(y_true_valid) == 0:
        return 'rmspe', 0.0, False # LightGBM cần is_higher_better
    return 'rmspe', np.sqrt(np.mean(((y_true_valid - y_pred_valid) / y_true_valid) ** 2)), False

print("Xây hàm đánh giá hoàn tất.")


print("\n--- Mô hình 1: XGBoost ---")
dtrain_xgb = xgb.DMatrix(X_train, label=y_train_log)
dval_xgb = xgb.DMatrix(X_val, label=y_val_log)
watchlist_xgb = [(dtrain_xgb, 'train'), (dval_xgb, 'eval')]

def objective_xgb(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse', 
        'eta': trial.suggest_float('eta', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'seed': 42
    }
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "eval-rmse")
    model = xgb.train(params, dtrain_xgb, num_boost_round=300,
                      evals=watchlist_xgb, early_stopping_rounds=20, 
                      callbacks=[pruning_callback], 
                      feval=rmspe_xgb_feval, # Sử dụng hàm feval đúng cho XGBoost
                      maximize=False,
                      verbose_eval=False)
    # Optuna sẽ tối ưu hóa dựa trên giá trị trả về của hàm này.
    # Nếu feval được dùng, XGBoost sẽ lưu best_score dựa trên feval.
    # Nếu eval_metric được dùng, XGBoost sẽ lưu best_score dựa trên eval_metric.
    # Để Optuna tối ưu RMSPE, hàm objective phải trả về RMSPE.
    # Chúng ta đã dùng feval=rmspe_xgb_feval, nên model.best_score sẽ là RMSPE.
    return model.best_score 


# study_xgb = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
# study_xgb.optimize(objective_xgb, n_trials=1) 
# best_params_xgb = study_xgb.best_params if 'study_xgb' in locals() and hasattr(study_xgb, 'best_params') else {}
# print("XGBoost - Siêu tham số tốt nhất:", best_params_xgb)
# print("XGBoost - RMSPE tốt nhất từ Optuna:", study_xgb.best_value if 'study_xgb' in locals() and hasattr(study_xgb, 'best_value') else "Chưa chạy Optuna")

best_params_xgb = {
    'objective': 'reg:squarederror', 'eta': 0.03, 'max_depth': 10,
    'subsample': 0.9, 'colsample_bytree': 0.7, 'seed': 10,
}


model_xgb_final = xgb.train(
    best_params_xgb, dtrain_xgb, num_boost_round=6000,
    evals=watchlist_xgb, early_stopping_rounds=100,
    feval=rmspe_xgb_feval, maximize=False, verbose_eval=100
)
y_pred_xgb_log = model_xgb_final.predict(dval_xgb)
evaluate_model("XGBoost", y_val_log, y_pred_xgb_log, y_train_orig, y_val_orig)

# --- Thử nghiệm "Whole Correction" cho XGBoost ---
print("\nĐang thực hiện 'Whole Correction' cho XGBoost...")
weights_to_try = np.arange(0.990, 1.011, 0.001) # Dải weights giống notebook top 1% (0.990 đến 1.010)
rmspe_scores_wc = []

for w in weights_to_try:
    corrected_preds_log = y_pred_xgb_log * w
    rmspe_w = rmspe_func(y_val_log, corrected_preds_log)
    rmspe_scores_wc.append(rmspe_w)
    # print(f'RMSPE cho weight {w:.3f}: {rmspe_w:.6f}')

best_rmspe_wc = min(rmspe_scores_wc)
w_optimal_whole_xgb = weights_to_try[np.argmin(rmspe_scores_wc)]
print(f'XGBoost - Best RMSPE sau "Whole Correction": {best_rmspe_wc:.6f} với weight tối ưu: {w_optimal_whole_xgb:.3f}')

y_pred_xgb_log_val_corrected = y_pred_xgb_log_val * w_optimal_whole_xgb
evaluate_model("XGBoost (sau hiệu chỉnh toàn cục)", y_val_log, y_pred_xgb_log_val_corrected, y_train_orig, y_val_orig)



# --- Thử nghiệm "Whole Correction" cho XGBoost ---
print("\nĐang thực hiện 'Whole Correction' cho XGBoost...")
weights_to_try = np.arange(0.990, 1.011, 0.001) # Dải weights giống notebook top 1% (0.990 đến 1.010)
rmspe_scores_wc = []

for w in weights_to_try:
    corrected_preds_log = y_pred_xgb_log * w
    rmspe_w = rmspe_func(y_val_log, corrected_preds_log)
    rmspe_scores_wc.append(rmspe_w)
    # print(f'RMSPE cho weight {w:.3f}: {rmspe_w:.6f}')

best_rmspe_wc = min(rmspe_scores_wc)
w_optimal_whole_xgb = weights_to_try[np.argmin(rmspe_scores_wc)]
print(f'XGBoost - Best RMSPE sau "Whole Correction": {best_rmspe_wc:.6f} với weight tối ưu: {w_optimal_whole_xgb:.3f}')

y_pred_xgb_log_val_corrected = y_pred_xgb_log * w_optimal_whole_xgb
evaluate_model("XGBoost (sau hiệu chỉnh toàn cục)", y_val_log, y_pred_xgb_log_val_corrected, y_train_orig, y_val_orig)



print("\n--- Mô hình 2: LightGBM ---")
dtrain_lgb = lgb.Dataset(X_train, label=y_train_log)
dval_lgb = lgb.Dataset(X_val, label=y_val_log, reference=dtrain_lgb)

def objective_lgb(trial):
    params = {
        'objective': 'regression_l1', 
        'metric': 'rmse', 
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 5, 10),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.9), # Tương tự colsample_bytree
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.9), # Tương tự subsample
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 0.5, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 0.5, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    # n_estimators được xử lý bởi num_boost_round trong lgb.train
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "rmspe", valid_name="valid_1") # Theo dõi valid_1-rmspe
    model = lgb.train(
        params, dtrain_lgb, num_boost_round=300, # Giảm số vòng cho Optuna
        valid_sets=[dtrain_lgb, dval_lgb],
        callbacks=[pruning_callback, lgb.early_stopping(30, verbose=-1)],
        feval=rmspe_lgb_feval # Sử dụng hàm feval đúng cho LightGBM
    )
    # LightGBM sẽ lưu best_score dựa trên feval nếu được cung cấp
    return model.best_score['valid_1']['rmspe']


# study_lgb = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
# study_lgb.optimize(objective_lgb, n_trials=1)
# best_params_lgb = study_lgb.best_params if 'study_lgb' in locals() and hasattr(study_lgb, 'best_params') else {}
# print("LightGBM - Siêu tham số tốt nhất:", best_params_lgb)
# print("LightGBM - RMSPE tốt nhất từ Optuna:", study_lgb.best_value if 'study_lgb' in locals() and hasattr(study_lgb, 'best_value') else "Chưa chạy Optuna")

best_params_lgb = {
    'objective': 'regression_l1', 'metric': 'rmse', 'boosting_type': 'gbdt',
    'learning_rate': 0.03, 'num_leaves': 80, 'max_depth': 8,
    'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 5,
    'random_state': 42, 'n_jobs': -1,
    'verbose': -1, 'reg_alpha': 0.1, 'reg_lambda': 0.1
}

model_lgb_final = lgb.train(
    best_params_lgb,
    dtrain_lgb,
    num_boost_round=1000,
    valid_sets=[dtrain_lgb, dval_lgb],
    valid_names=['train', 'eval'],  # tên này sẽ hiện trong log
    feval=rmspe_lgb_feval,  # dùng hàm đã định nghĩa của bạn
    callbacks=[
        lgb.early_stopping(50),
        lgb.log_evaluation(period=100)  # in log mỗi 100 vòng, đổi thành 1 nếu cần tất cả
    ]
)
y_pred_lgb_log = model_lgb_final.predict(X_val, num_iteration=model_lgb_final.best_iteration)
evaluate_model("LightGBM", y_val_log, y_pred_lgb_log, y_train_orig, y_val_orig)


print("\n--- Mô hình 3: Random Forest ---")

def objective_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 8, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 5, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.7]),
        'random_state': 42, 'n_jobs': -1
    }
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train_log)
    preds = model.predict(X_val)
    rmspe_val = rmspe_func(y_val_log, preds)
    return rmspe_val

# study_rf = optuna.create_study(direction='minimize')
# study_rf.optimize(objective_rf, n_trials=1)
# best_params_rf = study_rf.best_params if 'study_rf' in locals() and hasattr(study_rf, 'best_params') else {}
# print("Random Forest - Siêu tham số tốt nhất:", best_params_rf)
# print("Random Forest - RMSPE tốt nhất từ Optuna:", study_rf.best_value if 'study_rf' in locals() and hasattr(study_rf, 'best_value') else "Chưa chạy Optuna")

best_params_rf = {
    'n_estimators': 150, 'max_depth': 12, 'min_samples_split': 10,
    'min_samples_leaf': 5, 'random_state': 42, 'n_jobs': -1, 'max_features': 'sqrt'
}

model_rf_final = RandomForestRegressor(**best_params_rf)
model_rf_final.fit(X_train, y_train_log)
y_pred_rf_log = model_rf_final.predict(X_val)
evaluate_model("Random Forest", y_val_log, y_pred_rf_log, y_train_orig, y_val_orig)


print("\n--- Mô hình 4: Linear Regression ---")
scaler = StandardScaler()
X_train_lr = X_train.copy()
X_val_lr = X_val.copy()

numerical_cols_for_scaling = ['CompetitionDistance', 'CompetitionOpen', 'PromoOpen', 
                              'Year', 'Month', 'Day', 'WeekOfYear', 'DayOfYear'] # Thêm các cột thời gian
# Cần đảm bảo các cột này là số và không phải là cột đã one-hot encoded nếu bạn dùng one-hot

# Loại bỏ các cột không phải số hoặc đã được mã hóa mà không muốn scale
cols_to_scale_lr = [col for col in numerical_cols_for_scaling if col in X_train_lr.columns and X_train_lr[col].dtype in [np.int64, np.float64]]


if cols_to_scale_lr: # Chỉ scale nếu có cột để scale
    X_train_lr[cols_to_scale_lr] = scaler.fit_transform(X_train[cols_to_scale_lr])
    X_val_lr[cols_to_scale_lr] = scaler.transform(X_val[cols_to_scale_lr])
else:
    print("Không có cột số nào được chọn để scale cho Linear Regression.")


def objective_lr(trial):
    params = {
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
    }
    model = LinearRegression(**params)
    model.fit(X_train_lr, y_train_log)
    preds = model.predict(X_val_lr)
    rmspe_val = rmspe_func(y_val_log, preds)
    return rmspe_val

# study_lr = optuna.create_study(direction='minimize')
# study_lr.optimize(objective_lr, n_trials=1)
# best_params_lr = study_lr.best_params if 'study_lr' in locals() and hasattr(study_lr, 'best_params') else {}
# print("Linear Regression - Siêu tham số tốt nhất:", best_params_lr)
# print("Linear Regression - RMSPE tốt nhất từ Optuna:", study_lr.best_value if 'study_lr' in locals() and hasattr(study_lr, 'best_value') else "Chưa chạy Optuna")

best_params_lr = {'fit_intercept': True}

model_lr_final = LinearRegression(**best_params_lr)
model_lr_final.fit(X_train_lr, y_train_log)
y_pred_lr_log = model_lr_final.predict(X_val_lr)
evaluate_model("Linear Regression", y_val_log, y_pred_lr_log, y_train_orig, y_val_orig)


print("\n--- Tìm tham số tối ưu: XGBoost ---")
dtrain_xgb = xgb.DMatrix(X_train, label=y_train_log)
dval_xgb = xgb.DMatrix(X_val, label=y_val_log)
watchlist_xgb = [(dtrain_xgb, 'train'), (dval_xgb, 'eval')]
from optuna.integration.xgboost import XGBoostPruningCallback


def objective_xgb(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse', 
        'eta': trial.suggest_float('eta', 0.01, 0.07, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'seed': 42
    }
    pruning_callback = XGBoostPruningCallback(trial, "eval-rmse")
    model = xgb.train(params, dtrain_xgb, num_boost_round=2500,
                      evals=watchlist_xgb, early_stopping_rounds=50, 
                      callbacks=[pruning_callback], 
                      feval=rmspe_xgb_feval, # Sử dụng hàm feval đúng cho XGBoost
                      maximize=False,
                      verbose_eval=False)
    # Optuna sẽ tối ưu hóa dựa trên giá trị trả về của hàm này.
    # Nếu feval được dùng, XGBoost sẽ lưu best_score dựa trên feval.
    # Nếu eval_metric được dùng, XGBoost sẽ lưu best_score dựa trên eval_metric.
    # Để Optuna tối ưu RMSPE, hàm objective phải trả về RMSPE.
    # Chúng ta đã dùng feval=rmspe_xgb_feval, nên model.best_score sẽ là RMSPE.
    return model.best_score 

study_xgb = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
study_xgb.optimize(objective_xgb, n_trials=50) 
best_params_xgb = study_xgb.best_params if 'study_xgb' in locals() and hasattr(study_xgb, 'best_params') else {}
print("XGBoost - Siêu tham số tốt nhất:", best_params_xgb)
print("XGBoost - RMSPE tốt nhất từ Optuna:", study_xgb.best_value if 'study_xgb' in locals() and hasattr(study_xgb, 'best_value') else "Chưa chạy Optuna")

# best_params_xgb = {
#     'objective': 'reg:squarederror', 'eta': 0.08742551955888661, 'max_depth': 10,
#     'subsample': 0.7, 'colsample_bytree': 0.7, 'seed': 42,
#     'lambda': 0.5, 'alpha':0.5 
# }


model_xgb_final = xgb.train(
    best_params_xgb, dtrain_xgb, num_boost_round=10000,
    evals=watchlist_xgb, early_stopping_rounds=50,
    feval=rmspe_xgb_feval, maximize=False, verbose_eval=100
)
y_pred_xgb_log = model_xgb_final.predict(dval_xgb)
evaluate_model("XGBoost", y_val_log, y_pred_xgb_log, y_train_orig, y_val_orig)


pip install optuna-integration[xgboost]


print("\n--- Tìm tham số tối ưu: XGBoost ---")
dtrain_xgb = xgb.DMatrix(X_train, label=y_train_log)
dval_xgb = xgb.DMatrix(X_val, label=y_val_log)
watchlist_xgb = [(dtrain_xgb, 'train'), (dval_xgb, 'eval')]
from optuna.integration.xgboost import XGBoostPruningCallback

best_params_xgb = {
    
    'objective': 'reg:squarederror', 'booster' : 'gbtree', 'eta': 0.059377153102338466, 'max_depth': 9,
    'subsample': 0.9003558616040022, 'colsample_bytree': 0.8334712704090349, 'seed': 42,
    'lambda': 0.004755215517421936, 'alpha': 0.0011774972149392502 
}

# XGBoost - Siêu tham số tốt nhất: {'eta': 0.059377153102338466, 'max_depth': 9, 'subsample': 0.9003558616040022, 'colsample_bytree': 0.8334712704090349, 'lambda': 0.004755215517421936, 'alpha': 0.0011774972149392502}
# XGBoost - RMSPE tốt nhất từ Optuna: 0.12831
# [I 2025-05-31 04:55:44,674] Trial 8 finished with value: 0.126011 and parameters: {'eta': 0.05581224358507298, 'max_depth': 10, 'subsample': 0.685806062958686, 'colsample_bytree': 0.8984734799096694, 'lambda': 0.3122857705095174, 'alpha': 0.009093340234409449}. Best is trial 8 with value: 0.126011.

model_xgb_final = xgb.train(
    best_params_xgb, dtrain_xgb, num_boost_round=10000,
    evals=watchlist_xgb, early_stopping_rounds=50,
    feval=rmspe_xgb_feval, maximize=False, verbose_eval=100
)
y_pred_xgb_log = model_xgb_final.predict(dval_xgb)
evaluate_model("XGBoost", y_val_log, y_pred_xgb_log, y_train_orig, y_val_orig)


print("\nBước 5: Đang tạo dự đoán cho tệp test với XGBoost (đã hiệu chỉnh toàn cục)...")
X_submission_test = test_df_final[features_train].copy()
dtest_final_xgb = xgb.DMatrix(X_submission_test)

print("Huấn luyện lại XGBoost trên toàn bộ dữ liệu huấn luyện có sẵn (X_all, y_all_log)...")
dtrain_full_xgb = xgb.DMatrix(X_all, label=y_all_log)
final_num_boost_round_xgb = model_xgb_final.best_iteration if model_xgb_final and hasattr(model_xgb_final, 'best_iteration') and model_xgb_final.best_iteration > 0 else 6000 

model_xgb_for_submission = xgb.train(
    best_params_xgb, dtrain_full_xgb, 
    num_boost_round=final_num_boost_round_xgb + 50, 
    evals=[(dtrain_full_xgb, 'train')],
    feval=rmspe_xgb_feval, maximize=False, verbose_eval=100 
)

predictions_log_submission = model_xgb_for_submission.predict(dtest_final_xgb)
corrected_predictions_log_submission = predictions_log_submission * w_optimal_whole_xgb 
predictions_original_submission = np.expm1(corrected_predictions_log_submission)

open_status_test_orig = test_df_orig['Open'].fillna(1).astype(int)
if len(predictions_original_submission) == len(test_df_orig):
    predictions_original_submission[test_df_orig[open_status_test_orig == 0].index] = 0
else:
    print("Cảnh báo: Độ dài của dự đoán và dữ liệu test gốc không khớp.")

submission_final_df = pd.DataFrame({
    "Id": test_df_orig["Id"],
    "Sales": predictions_original_submission
})
submission_final_df['Sales'] = submission_final_df['Sales'].clip(lower=0)
submission_final_df.to_csv("submission_xgb_wc.csv", index=False)
print("Tệp submission_xgb_wc_fixed.csv đã được tạo.")

print("\nHoàn thành!")


#------------------------------------------------------------------------------
# Bước 5: Dự đoán và Nộp kết quả (Sử dụng XGBoost đã hiệu chỉnh toàn cục)
#------------------------------------------------------------------------------
print("\nBước 5: Đang tạo dự đoán cho tệp test với XGBoost (đã hiệu chỉnh toàn cục)...")
# X_submission_test đã được tạo ở trên từ test_df_final_submission[features_to_keep]
dtest_final_xgb = xgb.DMatrix(X_submission_test)

print("Huấn luyện lại XGBoost trên toàn bộ dữ liệu huấn luyện có sẵn (X_all, y_all_log)...")
dtrain_full_xgb = xgb.DMatrix(X_all, label=y_all_log)
final_num_boost_round_xgb = model_xgb_final.best_iteration if model_xgb_final and hasattr(model_xgb_final, 'best_iteration') and model_xgb_final.best_iteration > 0 else 6000 

model_xgb_for_submission = xgb.train(
    best_params_xgb, dtrain_full_xgb, 
    num_boost_round=final_num_boost_round_xgb, # Sử dụng best_iteration trực tiếp
    evals=[(dtrain_full_xgb, 'train')],
    feval=rmspe_xg, maximize=False, verbose_eval=100 # Giống notebook top 1%
)

predictions_log_submission = model_xgb_for_submission.predict(dtest_final_xgb)
corrected_predictions_log_submission = predictions_log_submission * w_optimal_whole_xgb 
predictions_original_submission = np.expm1(corrected_predictions_log_submission)

open_status_test_orig = test_df_orig['Open'].fillna(1).astype(int)
if len(predictions_original_submission) == len(test_df_orig):
    # Đảm bảo test_df_orig có cùng index với predictions_original_submission
    # Hoặc reset index cho cả hai nếu cần
    # predictions_original_submission[test_df_orig[open_status_test_orig == 0].index] = 0
    # Cách an toàn hơn là tạo một Series boolean từ open_status_test_orig
    # và sử dụng nó để gán, đảm bảo index khớp với predictions_original_submission
    # nếu predictions_original_submission là một numpy array
    
    # test_df_orig['Id'] được dùng để tạo submission, nên index của nó là quan trọng
    # Giả sử X_submission_test và test_df_orig có cùng thứ tự hàng sau khi xử lý
    closed_store_indices_test = test_df_orig[open_status_test_orig == 0].index
    
    # Nếu predictions_original_submission là numpy array, cần đảm bảo nó khớp với index của test_df_orig
    # Hoặc, nếu X_submission_test có index giống test_df_orig, thì:
    pred_series = pd.Series(predictions_original_submission, index=X_submission_test.index)
    pred_series.loc[closed_store_indices_test] = 0
    predictions_original_submission = pred_series.values

else:
    print("Cảnh báo: Độ dài của dự đoán và dữ liệu test gốc không khớp.")

submission_final_df = pd.DataFrame({
    "Id": test_df_orig["Id"], # Lấy Id từ test_df_orig gốc
    "Sales": predictions_original_submission
})
submission_final_df['Sales'] = submission_final_df['Sales'].clip(lower=0)
submission_final_df.to_csv("submission_xgb_wc_final_preprocess.csv", index=False)
print("Tệp submission_xgb_wc_top1_preprocess.csv đã được tạo.")

print("\nHoàn thành!")



import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
import warnings

warnings.filterwarnings('ignore')

#------------------------------------------------------------------------------
# Bước 1: Tải và Kết hợp Dữ liệu
#------------------------------------------------------------------------------
print("Bước 1: Đang tải dữ liệu...")
try:
    train_df_orig = pd.read_csv("../input/train.csv", low_memory=False, parse_dates=['Date'])
    store_df_orig = pd.read_csv("../input/store.csv", low_memory=False)
    test_df_orig = pd.read_csv("../input/test.csv", low_memory=False, parse_dates=['Date'])
    print("Tải dữ liệu thành công.")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy tệp dữ liệu. Vui lòng đảm bảo các tệp train.csv, store.csv, và test.csv nằm trong cùng thư mục.")
    exit()

train_df = train_df_orig.copy()
store_df_global_filled = store_df_orig.copy() # Đổi tên để rõ ràng hơn
test_df = test_df_orig.copy()

# Xử lý NaN trong store_df THEO CÁCH CỦA NOTEBOOK TOP 1% (fillna(0) TRƯỚC KHI MERGE)
store_df_global_filled.fillna(0, inplace=True)


print("Đang kết hợp dữ liệu...")
train_df = pd.merge(train_df, store_df_global_filled, on='Store', how='left')
test_df = pd.merge(test_df, store_df_global_filled, on='Store', how='left') # test_df này sẽ được xử lý và có thể dùng cho X_val nếu không muốn tạo lại
print("Kết hợp dữ liệu thành công.")

#------------------------------------------------------------------------------
# Bước 2: Tiền xử lý dữ liệu và Kỹ thuật Đặc trưng
#------------------------------------------------------------------------------
print("\nBước 2: Đang tiền xử lý dữ liệu và tạo đặc trưng...")

# Các đặc trưng sẽ giữ lại cho mô hình (sau khi Date đã được dùng để tạo đặc trưng thời gian)
features_to_keep = [
    'Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday',
    'StoreType', 'Assortment', 'CompetitionDistance',
    'Promo2', 'Year', 'Month', 'Day', 'WeekOfYear', 'DayOfYear',
    'CompetitionOpen', 'PromoOpen', 'IsPromoMonth'
]
# Các đặc trưng cần giữ lại cho test_df trong hàm build_features trước khi drop
# (Open sẽ được dùng để set Sales=0 sau dự đoán)
features_to_keep_for_test_in_build = features_to_keep + ['Open']


def build_features(df, is_train=True):
    if not is_train and 'Open' in df.columns:
        df['Open'].fillna(1, inplace=True)

    df['Year'] = df.Date.dt.year
    df['Month'] = df.Date.dt.month
    df['Day'] = df.Date.dt.day
    df['DayOfWeek'] = df.Date.dt.dayofweek
    df['WeekOfYear'] = df.Date.dt.isocalendar().week.astype(int)
    df['DayOfYear'] = df.Date.dt.dayofyear

    df['CompetitionOpen'] = 12 * (df.Year - df.CompetitionOpenSinceYear) + \
                            (df.Month - df.CompetitionOpenSinceMonth)
    df['CompetitionOpen'] = df['CompetitionOpen'].apply(lambda x: x if x > 0 else 0)

    df['PromoOpen'] = 12 * (df.Year - df.Promo2SinceYear) + \
                      (df.WeekOfYear - df.Promo2SinceWeek) / 4.0
    df['PromoOpen'] = df['PromoOpen'].apply(lambda x: x if x > 0 else 0)
    
    month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    df['monthStr'] = df.Month.map(month_map)
    df.loc[df.PromoInterval == 0, 'PromoInterval'] = ''
    df['IsPromoMonth'] = 0
    for interval in df.PromoInterval.unique():
        if interval != '':
            for month_abbr in interval.split(','):
                df.loc[(df.monthStr == month_abbr) & (df.PromoInterval == interval), 'IsPromoMonth'] = 1
    
    mappings = {'0':0, 0:0, 'a':1, 'b':2, 'c':3, 'd':4} 
    if 'StateHoliday' in df.columns:
        df['StateHoliday'] = df['StateHoliday'].astype(str)
        df['StateHoliday'].fillna('0', inplace=True)
        df.StateHoliday.replace(mappings, inplace=True)
        df['StateHoliday'] = df['StateHoliday'].astype(int)

    if 'StoreType' in df.columns:
        df.StoreType.replace(mappings, inplace=True)
        df['StoreType'] = df['StoreType'].astype(int)

    if 'Assortment' in df.columns:
        df.Assortment.replace(mappings, inplace=True)
        df['Assortment'] = df['Assortment'].astype(int)
    
    if 'Open' in df.columns and df['Open'].isnull().any():
        df['Open'].fillna(1, inplace=True)
    if 'Open' in df.columns:
         df['Open'] = df['Open'].astype(int)
    
    cols_to_drop = ['Customers', 'PromoInterval', 'monthStr', 
                    'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 
                    'Promo2SinceWeek', 'Promo2SinceYear']
    if is_train:
        pass 
    else: # test_df
        cols_to_drop.append('Id') # Id sẽ được dùng lại từ test_df_orig
        # Giữ lại cột 'Open' trong df trả về từ build_features cho test set
        # để dùng cho việc set Sales = 0 sau này.
        # Nó sẽ không nằm trong features_to_keep cho mô hình.

    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True, errors='ignore')
    return df

train_df = build_features(train_df.copy(), is_train=True)
# test_df được xử lý ở đây chỉ để có thể dùng cho X_val nếu cần, hoặc bỏ qua nếu X_val được tạo lại từ đầu
# test_df = build_features(test_df.copy(), is_train=False) 

train_df = train_df[train_df['Open'] == 1]
train_df = train_df[train_df['Sales'] > 0]
train_df['Sales_log'] = np.log1p(train_df['Sales'])

print("Tiền xử lý dữ liệu và tạo đặc trưng hoàn tất.")

#------------------------------------------------------------------------------
# Bước 2.1: Lựa chọn đặc trưng cho mô hình
#------------------------------------------------------------------------------
# features_to_keep đã được định nghĩa ở trên
features_train = [f for f in features_to_keep if f in train_df.columns]


#------------------------------------------------------------------------------
# Bước 2.2: Phân chia Tập dữ liệu (Bằng phân chia theo thời gian cụ thể)
#------------------------------------------------------------------------------
print("\nBước 2.2: Đang phân chia tập dữ liệu theo thời gian...")
train_df_sorted_for_split = train_df.sort_values(['Date'], ascending=False)
num_ho_samples = 6 * 7 * 1115 
num_ho_samples = min(num_ho_samples, int(0.1 * len(train_df_sorted_for_split)))

X_val_df_time = train_df_sorted_for_split[:num_ho_samples]
X_train_df_time = train_df_sorted_for_split[num_ho_samples:]

X_train_df_time = X_train_df_time.sort_values(by=['Store', 'Date'])
X_val_df_time = X_val_df_time.sort_values(by=['Store', 'Date'])

X_train = X_train_df_time[features_train]
y_train_log = X_train_df_time['Sales_log']
y_train_orig = X_train_df_time['Sales']

X_val = X_val_df_time[features_train]
y_val_log = X_val_df_time['Sales_log']
y_val_orig = X_val_df_time['Sales']

X_all = train_df.sort_values(by=['Store', 'Date'])[features_train]
y_all_log = train_df.sort_values(by=['Store', 'Date'])['Sales_log']
y_all_orig = train_df.sort_values(by=['Store', 'Date'])['Sales']

print(f"Kích thước tập huấn luyện: {X_train.shape}, tập kiểm định: {X_val.shape}")

#------------------------------------------------------------------------------
# Bước 4.1: Hàm đánh giá
#------------------------------------------------------------------------------
def rmspe_func(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.maximum(0, y_pred)
    valid_indices = (y_true != 0)
    if not np.any(valid_indices):
        return float('inf') if np.any(y_pred != 0) else 0.0
    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices]
    if len(y_true_valid) == 0: return float('inf') 
    return np.sqrt(np.mean(((y_true_valid - y_pred_valid) / y_true_valid) ** 2))

def mape_func(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.maximum(0, y_pred)
    valid_indices = y_true != 0
    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices]
    if len(y_true_valid) == 0: return float('inf')
    return np.mean(np.abs((y_true_valid - y_pred_valid) / y_true_valid)) * 100

def mase_func(y_true_val_orig, y_pred_val_orig, y_true_train_orig):
    mae_model = mean_absolute_error(y_true_val_orig, y_pred_val_orig)
    if len(y_true_train_orig) <= 7: return np.inf
    y_naive_pred_train = y_true_train_orig.iloc[:-7].values
    mae_naive = mean_absolute_error(y_true_train_orig.iloc[7:].values, y_naive_pred_train)
    if mae_naive == 0: return np.inf
    return mae_model / mae_naive

def evaluate_model(model_name, y_val_log_true, y_val_log_pred, y_train_orig_for_mase, y_val_orig_true):
    y_val_orig_pred = np.expm1(y_val_log_pred)
    y_val_orig_pred = np.maximum(0, y_val_orig_pred)
    print(f"\n--- Đánh giá cho mô hình: {model_name} ---")
    print(f"RMSPE: {rmspe_func(y_val_log_true, y_val_log_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_val_orig_true, y_val_orig_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_val_orig_true, y_val_orig_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_val_orig_true, y_val_orig_pred)):.4f}")
    print(f"MAPE: {mape_func(y_val_log_true, y_val_log_pred):.4f}%")
    print(f"R-Squared: {r2_score(y_val_orig_true, y_val_orig_pred):.4f}")
    mase = mase_func(y_val_orig_true, y_val_orig_pred, y_train_orig_for_mase)
    print(f"MASE: {mase:.4f}")

def rmspe_xg(yhat, y): # Đổi tên cho XGBoost feval
    y = np.expm1(y.get_label()) 
    yhat = np.expm1(yhat)      
    yhat = np.maximum(0, yhat) 
    return "rmspe", np.sqrt(np.mean(((y - yhat)/y) ** 2))

def rmspe_lgb_feval(y_pred_log, y_true_dmatrix):
    y_true_log_label = y_true_dmatrix.get_label()
    y_true_orig = np.expm1(y_true_log_label)
    y_pred_orig = np.expm1(y_pred_log)
    y_pred_orig = np.maximum(0, y_pred_orig)
    valid_indices = y_true_orig != 0
    y_true_valid = y_true_orig[valid_indices]
    y_pred_valid = y_pred_orig[valid_indices]
    if len(y_true_valid) == 0: return 'rmspe', 0.0, False
    return 'rmspe', np.sqrt(np.mean(((y_true_valid - y_pred_valid) / y_true_valid) ** 2)), False

#------------------------------------------------------------------------------
# Mô hình 1: XGBoost
#------------------------------------------------------------------------------
print("\n--- Mô hình 1: XGBoost ---")
dtrain_xgb = xgb.DMatrix(X_train, label=y_train_log)
dval_xgb = xgb.DMatrix(X_val, label=y_val_log)
watchlist_xgb = [(dtrain_xgb, 'train'), (dval_xgb, 'eval')]

def objective_xgb(trial):
    params = {
        'objective': 'reg:linear', 
        'booster' : 'gbtree',      
        'eta': trial.suggest_float('eta', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 8, 12),     
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),  
        'silent': 1, 
        'seed': trial.suggest_int('seed', 1, 100) 
    }
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "eval-rmspe")
    model = xgb.train(params, dtrain_xgb, num_boost_round=300, 
                      evals=watchlist_xgb, early_stopping_rounds=30, 
                      callbacks=[pruning_callback], 
                      feval=rmspe_xg, maximize=False, verbose_eval=False)
    return model.best_score 

# study_xgb = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
# study_xgb.optimize(objective_xgb, n_trials=1) 
# best_params_xgb_optuna = study_xgb.best_params if 'study_xgb' in locals() and hasattr(study_xgb, 'best_params') else {}
# print("XGBoost - Siêu tham số tốt nhất từ Optuna:", best_params_xgb_optuna)
# print("XGBoost - RMSPE tốt nhất từ Optuna:", study_xgb.best_value if 'study_xgb' in locals() and hasattr(study_xgb, 'best_value') else "Chưa chạy Optuna")

best_params_xgb = {
    "objective": "reg:linear", "booster" : "gbtree", "eta": 0.03,
    "max_depth": 10, "subsample": 0.9, "colsample_bytree": 0.7,
    "silent": 100, "seed": 10
}
# if best_params_xgb_optuna: 
#     best_params_xgb.update(best_params_xgb_optuna) 
#     if 'objective' not in best_params_xgb: best_params_xgb['objective'] = 'reg:linear'
#     if 'seed' not in best_params_xgb: best_params_xgb['seed'] = 10 

model_xgb_final = None
y_pred_xgb_log_val = None
try:
    model_xgb_final = xgb.train(
        best_params_xgb, dtrain_xgb, num_boost_round=6000, 
        evals=watchlist_xgb, early_stopping_rounds=100,   
        feval=rmspe_xg, maximize=False, verbose_eval=True 
    )
    y_pred_xgb_log_val = model_xgb_final.predict(dval_xgb) 
    evaluate_model("XGBoost (trước hiệu chỉnh)", y_val_log, y_pred_xgb_log_val, y_train_orig, y_val_orig)
except Exception as e:
    print(f"Lỗi trong quá trình huấn luyện hoặc dự đoán XGBoost: {e}")

print("\nĐang thực hiện 'Whole Correction' cho XGBoost...")
w_optimal_whole_xgb = 1.0 
if y_pred_xgb_log_val is not None and (isinstance(y_pred_xgb_log_val, np.ndarray) and y_pred_xgb_log_val.size > 0):
    weights_to_try = np.arange(0.990, 1.011, 0.001) 
    rmspe_scores_wc = []
    for w in weights_to_try: 
        corrected_preds_log = y_pred_xgb_log_val * w
        rmspe_w = rmspe_func(y_val_log, corrected_preds_log)
        rmspe_scores_wc.append(rmspe_w)
    if rmspe_scores_wc: 
        best_rmspe_wc = min(rmspe_scores_wc)
        w_optimal_whole_xgb = weights_to_try[np.argmin(rmspe_scores_wc)]
        print(f'XGBoost - Best RMSPE sau "Whole Correction": {best_rmspe_wc:.6f} với weight tối ưu: {w_optimal_whole_xgb:.3f}')
        y_pred_xgb_log_val_corrected = y_pred_xgb_log_val * w_optimal_whole_xgb
        evaluate_model("XGBoost (sau hiệu chỉnh toàn cục)", y_val_log, y_pred_xgb_log_val_corrected, y_train_orig, y_val_orig)
    else: print("Không thể thực hiện 'Whole Correction' do không có điểm RMSPE.")
else: print("Không thể thực hiện 'Whole Correction' do y_pred_xgb_log_val không được định nghĩa hoặc rỗng.")

#------------------------------------------------------------------------------
# Mô hình 2: LightGBM 
#------------------------------------------------------------------------------
# print("\n--- Mô hình 2: LightGBM ---")
# dtrain_lgb = lgb.Dataset(X_train, label=y_train_log)
# dval_lgb = lgb.Dataset(X_val, label=y_val_log, reference=dtrain_lgb)
# best_params_lgb = {
#     'objective': 'regression_l1', 'metric': 'rmse', 'boosting_type': 'gbdt',
#     'learning_rate': 0.03, 'num_leaves': 80, 'max_depth': 8,
#     'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 5,
#     'random_state': 42, 'n_jobs': -1, 'verbose': -1, 
#     'reg_alpha': 0.1, 'reg_lambda': 0.1
# }
# model_lgb_final = lgb.train(
#     best_params_lgb, dtrain_lgb, num_boost_round=1000,
#     valid_sets=[dtrain_lgb, dval_lgb],
#     callbacks=[lgb.early_stopping(50, verbose=100)],
#     feval=rmspe_lgb_feval
# )
# y_pred_lgb_log_val = model_lgb_final.predict(X_val, num_iteration=model_lgb_final.best_iteration)
# evaluate_model("LightGBM", y_val_log, y_pred_lgb_log_val, y_train_orig, y_val_orig)

# #------------------------------------------------------------------------------
# # Mô hình 3: Random Forest 
# #------------------------------------------------------------------------------
# print("\n--- Mô hình 3: Random Forest ---")
# best_params_rf = {
#     'n_estimators': 150, 'max_depth': 12, 'min_samples_split': 10,
#     'min_samples_leaf': 5, 'random_state': 42, 'n_jobs': -1, 'max_features': 'sqrt'
# }
# model_rf_final = RandomForestRegressor(**best_params_rf)
# model_rf_final.fit(X_train, y_train_log)
# y_pred_rf_log_val = model_rf_final.predict(X_val)
# evaluate_model("Random Forest", y_val_log, y_pred_rf_log_val, y_train_orig, y_val_orig)

# #------------------------------------------------------------------------------
# # Mô hình 4: Linear Regression 
# #------------------------------------------------------------------------------
# print("\n--- Mô hình 4: Linear Regression ---")
# scaler = StandardScaler()
# X_train_lr = X_train.copy()
# X_val_lr = X_val.copy()
# numerical_cols_for_scaling = ['CompetitionDistance', 'CompetitionOpen', 'PromoOpen', 
#                               'Year', 'Month', 'Day', 'WeekOfYear', 'DayOfYear']
# cols_to_scale_lr = [col for col in numerical_cols_for_scaling if col in X_train_lr.columns and X_train_lr[col].dtype in [np.int64, np.float64]]
# if cols_to_scale_lr:
#     X_train_lr[cols_to_scale_lr] = scaler.fit_transform(X_train[cols_to_scale_lr])
#     X_val_lr[cols_to_scale_lr] = scaler.transform(X_val[cols_to_scale_lr])
# else:
#     print("Không có cột số nào được chọn để scale cho Linear Regression.")
# best_params_lr = {'fit_intercept': True}
# model_lr_final = LinearRegression(**best_params_lr)
# model_lr_final.fit(X_train_lr, y_train_log)
# y_pred_lr_log_val = model_lr_final.predict(X_val_lr)
# evaluate_model("Linear Regression", y_val_log, y_pred_lr_log_val, y_train_orig, y_val_orig)

#------------------------------------------------------------------------------
# Bước 5: Dự đoán và Nộp kết quả (Sử dụng XGBoost đã hiệu chỉnh toàn cục)
#------------------------------------------------------------------------------
print("\nBước 5: Đang tạo dự đoán cho tệp test với XGBoost (đã hiệu chỉnh toàn cục)...")

# Chuẩn bị X_submission_test từ test_df_orig (dữ liệu test gốc)
# 1. Merge test_df_orig với store_df_global_filled
submission_test_merged = pd.merge(test_df_orig.copy(), store_df_global_filled, on='Store', how='left')
# 2. Áp dụng build_features
submission_test_featured = build_features(submission_test_merged.copy(), is_train=False)
# 3. Chọn các cột đặc trưng cuối cùng
X_submission_test = submission_test_featured[features_to_keep].copy()


dtest_final_xgb = xgb.DMatrix(X_submission_test)

print("Huấn luyện lại XGBoost trên toàn bộ dữ liệu huấn luyện có sẵn (X_all, y_all_log)...")
dtrain_full_xgb = xgb.DMatrix(X_all, label=y_all_log)
final_num_boost_round_xgb = model_xgb_final.best_iteration if model_xgb_final and hasattr(model_xgb_final, 'best_iteration') and model_xgb_final.best_iteration > 0 else 6000 

model_xgb_for_submission = xgb.train(
    best_params_xgb, dtrain_full_xgb, 
    num_boost_round=final_num_boost_round_xgb, 
    evals=[(dtrain_full_xgb, 'train')],
    feval=rmspe_xg, maximize=False, verbose_eval=100 
)

predictions_log_submission = model_xgb_for_submission.predict(dtest_final_xgb)
corrected_predictions_log_submission = predictions_log_submission * w_optimal_whole_xgb 
predictions_original_submission = np.expm1(corrected_predictions_log_submission)

open_status_test_orig = test_df_orig['Open'].fillna(1).astype(int)
if len(predictions_original_submission) == len(test_df_orig):
    # Tạo một Series từ predictions_original_submission với index của X_submission_test
    # (X_submission_test được tạo từ test_df_orig nên index sẽ tương ứng nếu không có thay đổi lớn)
    # Hoặc tốt hơn là dùng index của test_df_orig trực tiếp nếu X_submission_test được tạo từ nó và giữ nguyên thứ tự
    
    # Cách tiếp cận an toàn hơn: tạo Series dự đoán và căn chỉnh theo test_df_orig
    # Điều này giả định rằng X_submission_test được tạo ra từ test_df_orig và giữ nguyên thứ tự các hàng.
    # Nếu không, cần một cách join hoặc map Id để đảm bảo đúng.
    
    # Hiện tại, X_submission_test được tạo từ test_df_final_submission, 
    # mà test_df_final_submission được tạo từ test_df_orig.copy() rồi merge và build_features.
    # Thứ tự hàng có thể được giữ nếu không có sort nào sau merge.
    
    # Để đảm bảo, chúng ta có thể tạo một DataFrame tạm thời với Id và dự đoán,
    # sau đó merge với test_df_orig để lấy trạng thái 'Open'.
    
    temp_submission_df = pd.DataFrame({
        'Id': test_df_orig['Id'], # Lấy Id từ test_df_orig
        'PredictedSalesLog': predictions_log_submission # Giữ ở dạng log để nhân weight
    })
    
    # Áp dụng weight
    temp_submission_df['CorrectedPredictedSalesLog'] = temp_submission_df['PredictedSalesLog'] * w_optimal_whole_xgb
    temp_submission_df['PredictedSales'] = np.expm1(temp_submission_df['CorrectedPredictedSalesLog'])
    
    # Merge với thông tin 'Open' từ test_df_orig (đã fillna)
    # Cần đảm bảo test_df_orig có cột 'Open' đã được fillna(1) nếu nó thiếu
    # open_status_test_orig đã làm điều này
    
    # Gán Sales = 0 cho các cửa hàng đóng
    # Tạo một mapping từ Id sang Open status
    id_to_open_status = pd.Series(open_status_test_orig.values, index=test_df_orig['Id'])
    
    # Áp dụng điều kiện
    for idx, row in temp_submission_df.iterrows():
        store_id = row['Id'] # Lấy Id từ temp_submission_df (cũng là từ test_df_orig)
        if id_to_open_status.get(store_id, 1) == 0: # Mặc định là mở nếu Id không tìm thấy (ít khả năng)
            temp_submission_df.loc[idx, 'PredictedSales'] = 0
            
    predictions_original_submission = temp_submission_df['PredictedSales'].values

else:
    print("Cảnh báo: Độ dài của dự đoán và dữ liệu test gốc không khớp.")


submission_final_df = pd.DataFrame({
    "Id": test_df_orig["Id"], 
    "Sales": predictions_original_submission
})
submission_final_df['Sales'] = submission_final_df['Sales'].clip(lower=0)
submission_final_df.to_csv("submission_xgb_wc_top1_preprocess_v2.csv", index=False)
print("Tệp submission_xgb_wc_top1_preprocess_v2.csv đã được tạo.")

print("\nHoàn thành!")



import pandas as pd
import numpy as np
import xgboost as xgb
from time import time
import matplotlib.pyplot as plt # Thêm thư viện để vẽ biểu đồ (nếu cần)
import seaborn as sns # Thêm thư viện để vẽ biểu đồ (nếu cần)

#------------------------------------------------------------------------------
# 1. Tải và Xử lý NaN Ban đầu
#------------------------------------------------------------------------------
print("Bước 1: Đang tải dữ liệu và xử lý NaN ban đầu...")
try:
    train_df_orig = pd.read_csv("../input/train.csv", parse_dates=[2], low_memory=False)
    test_df_orig = pd.read_csv("../input/test.csv", parse_dates=[3], low_memory=False)
    store_df_orig = pd.read_csv("../input/store.csv", low_memory=False)
    print("Tải dữ liệu thành công.")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy tệp dữ liệu. Vui lòng đảm bảo các tệp train.csv, store.csv, và test.csv nằm trong cùng thư mục.")
    exit()

# Tạo bản sao để làm việc
train = train_df_orig.copy()
test = test_df_orig.copy()
store = store_df_orig.copy()

# Xử lý NaN trong test.csv (Open) và store.csv
test.fillna(1, inplace=True) # Giả định cửa hàng mở nếu không có thông tin trong test
store.fillna(0, inplace=True) # Điền 0 cho tất cả NaN trong store_df (theo notebook top 1%)

print("Xử lý NaN ban đầu hoàn tất.")

#------------------------------------------------------------------------------
# 2. Kết hợp Dữ liệu
#------------------------------------------------------------------------------
print("\nBước 2: Đang kết hợp dữ liệu...")
train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')
print("Kết hợp dữ liệu thành công.")

#------------------------------------------------------------------------------
# 3. Tạo Tập Hold-out và Lọc Dữ liệu
#------------------------------------------------------------------------------
print("\nBước 3: Đang tạo tập hold-out và lọc dữ liệu...")
# Sắp xếp theo Date giảm dần để lấy 6 tuần cuối làm hold-out test
train = train.sort_values(['Date'], ascending=False)

# Kích thước tập hold-out (6 tuần * 7 ngày * 1115 cửa hàng)
# Notebook gốc dùng hardcode số lượng, ở đây ta có thể tính tương đối
# hoặc dùng một tỷ lệ nhỏ của tập train nếu số lượng mẫu quá lớn/nhỏ
num_ho_samples = 6 * 7 * 1115
num_ho_samples = min(num_ho_samples, len(train)) # Đảm bảo không vượt quá kích thước train

ho_test_df = train[:num_ho_samples]
ho_train_df = train[num_ho_samples:]

# Chỉ sử dụng dữ liệu có Sales > 0 và Open != 0
ho_test_df = ho_test_df[ho_test_df["Open"] != 0]
ho_test_df = ho_test_df[ho_test_df["Sales"] > 0]
ho_train_df = ho_train_df[ho_train_df["Open"] != 0]
ho_train_df = ho_train_df[ho_train_df["Sales"] > 0]

print(f"Kích thước tập huấn luyện hold-out (ho_train): {ho_train_df.shape}")
print(f"Kích thước tập kiểm định hold-out (ho_test): {ho_test_df.shape}")

#------------------------------------------------------------------------------
# 4. Kỹ thuật Đặc trưng (Feature Engineering)
#------------------------------------------------------------------------------
print("\nBước 4: Đang tạo đặc trưng...")

def features_create(data):
    mappings = {'0':0, 0:0, 'a':1, 'b':2, 'c':3, 'd':4} # Thêm 0:0 để xử lý trường hợp là số
    # Chuyển sang string trước khi map để nhất quán, sau đó fillna nếu cần
    if 'StoreType' in data.columns:
        data.StoreType = data.StoreType.astype(str).replace(mappings).astype(int)
    if 'Assortment' in data.columns:
        data.Assortment = data.Assortment.astype(str).replace(mappings).astype(int)
    if 'StateHoliday' in data.columns:
        data.StateHoliday = data.StateHoliday.astype(str) # Đảm bảo là string
        data.StateHoliday.fillna('0', inplace=True)      # Fill NaN bằng '0'
        data.StateHoliday.replace(mappings, inplace=True)
        data.StateHoliday = data.StateHoliday.astype(int) # Chuyển sang int
    
    data['Year'] = data.Date.dt.year
    data['Month'] = data.Date.dt.month
    data['Day'] = data.Date.dt.day
    data['DayOfWeek'] = data.Date.dt.dayofweek
    data['WeekOfYear'] = data.Date.dt.isocalendar().week.astype(int)
    
    data['CompetitionOpen'] = 12 * (data.Year - data.CompetitionOpenSinceYear) + \
                              (data.Month - data.CompetitionOpenSinceMonth)
    data['PromoOpen'] = 12 * (data.Year - data.Promo2SinceYear) + \
                        (data.WeekOfYear - data.Promo2SinceWeek) / 4.0
    
    data['CompetitionOpen'] = data.CompetitionOpen.apply(lambda x: x if x > 0 else 0)        
    data['PromoOpen'] = data.PromoOpen.apply(lambda x: x if x > 0 else 0)
    
    month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', \
                 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    data['monthStr'] = data.Month.map(month2str)
    
    # Xử lý PromoInterval là 0 (do fillna) thành chuỗi rỗng
    if 'PromoInterval' in data.columns:
        data.loc[data.PromoInterval == 0, 'PromoInterval'] = ''
        
    data['IsPromoMonth'] = 0
    for interval in data.PromoInterval.unique():
        if interval != '':
            for month_str_val in interval.split(','): # Đổi tên biến để tránh trùng
                data.loc[(data.monthStr == month_str_val) & (data.PromoInterval == interval), 'IsPromoMonth'] = 1
    return data

ho_train_df = features_create(ho_train_df.copy())
ho_test_df = features_create(ho_test_df.copy())
test_processed_df = features_create(test.copy()) # Xử lý cho tập test cuối cùng

print("Tạo đặc trưng hoàn tất.")

# Loại bỏ các cột không cần thiết (theo notebook top 1%)
cols_to_drop_train_ho = ['Date', 'Customers', 'Open', 'PromoInterval', 'monthStr']
ho_train_df.drop(columns=[col for col in cols_to_drop_train_ho if col in ho_train_df.columns], inplace=True)
ho_test_df.drop(columns=[col for col in cols_to_drop_train_ho if col in ho_test_df.columns], inplace=True)

cols_to_drop_test = ['Id', 'Date', 'Open', 'PromoInterval', 'monthStr']
xtest_final = test_processed_df.drop(columns=[col for col in cols_to_drop_test if col in test_processed_df.columns], errors='ignore')


# Chuẩn bị dữ liệu cho XGBoost
# Đảm bảo các cột đặc trưng nhất quán
final_features = [col for col in ho_train_df.columns if col not in ['Sales']]

ho_xtrain = ho_train_df[final_features]
ho_ytrain_log = np.log1p(ho_train_df.Sales)

ho_xtest = ho_test_df[final_features]
ho_ytest_log = np.log1p(ho_test_df.Sales)
ho_ytest_orig = ho_test_df.Sales # Giữ lại Sales gốc của ho_test để tính RMSPE

# Đảm bảo xtest_final có cùng các cột và thứ tự như ho_xtrain
xtest_final = xtest_final[final_features]


print(f"Đặc trưng sử dụng cho mô hình: {final_features}")
print(f"Kích thước ho_xtrain: {ho_xtrain.shape}, ho_xtest: {ho_xtest.shape}, xtest_final: {xtest_final.shape}")

#------------------------------------------------------------------------------
# 5. Hàm Đánh giá RMSPE
#------------------------------------------------------------------------------
def rmspe(y, yhat): # y và yhat là ở thang đo gốc
    y = np.array(y)
    yhat = np.array(yhat)
    yhat = np.maximum(0, yhat) # Đảm bảo dự đoán không âm
    
    # Lọc các phần tử y == 0 để tránh lỗi chia cho 0
    # Theo quy tắc cuộc thi, các cửa hàng đóng (Sales=0) không được tính
    # Dữ liệu huấn luyện đã được lọc Sales > 0
    valid_indices = (y != 0)
    if not np.any(valid_indices):
        return float('inf') if np.any(yhat != 0) else 0.0

    y_valid = y[valid_indices]
    yhat_valid = yhat[valid_indices]
    
    if len(y_valid) == 0:
        return float('inf')

    return np.sqrt(np.mean(((yhat_valid/y_valid) - 1) ** 2))

def rmspe_xg(yhat_log, y_true_dmatrix): # yhat_log và y_true_log (từ dmatrix) đều ở thang đo log
    y_true_log = y_true_dmatrix.get_label()
    y_true_orig = np.expm1(y_true_log)
    yhat_orig = np.expm1(yhat_log)
    return "rmspe", rmspe(y_true_orig, yhat_orig) # Truyền giá trị gốc vào hàm rmspe

#------------------------------------------------------------------------------
# 6. Huấn luyện Mô hình XGBoost
#------------------------------------------------------------------------------
print("\nBước 6: Đang huấn luyện mô hình XGBoost...")
params = {
    "objective": "reg:linear", # Trong XGBoost cũ, tương đương 'reg:squarederror'
    "booster": "gbtree",
    "eta": 0.03,
    "max_depth": 10,
    "subsample": 0.9,
    "colsample_bytree": 0.7,
    "silent": 1, # 0 để hiển thị log, 1 để im lặng
    "seed": 10
}
num_boost_round = 6000

dtrain_xgb = xgb.DMatrix(ho_xtrain, ho_ytrain_log)
dvalid_xgb = xgb.DMatrix(ho_xtest, ho_ytest_log)
watchlist = [(dtrain_xgb, 'train'), (dvalid_xgb, 'eval')]

start_time = time()
gbm = xgb.train(params, dtrain_xgb, num_boost_round, evals=watchlist,
                early_stopping_rounds=100, feval=rmspe_xg, verbose_eval=True) # Để True để xem log
end_time = time()
print(f'Thời gian huấn luyện: {end_time - start_time:.2f} giây.')

print("\nĐang kiểm định trên tập hold-out...")
yhat_ho_log = gbm.predict(xgb.DMatrix(ho_xtest))
error_ho_before_correction = rmspe(ho_ytest_orig, np.expm1(yhat_ho_log))
print(f'RMSPE trên tập hold-out (trước hiệu chỉnh): {error_ho_before_correction:.6f}')

#------------------------------------------------------------------------------
# 7. Hiệu chỉnh Dự đoán (Post-processing)
#------------------------------------------------------------------------------
print("\nBước 7: Đang thực hiện hiệu chỉnh dự đoán...")

# 7.1. Whole Correction
print("Đang thực hiện 'Whole Correction'...")
weights_to_try_whole = np.arange(0.990, 1.011, 0.001)
rmspe_scores_wc_whole = []

for w in weights_to_try_whole:
    error_w = rmspe(ho_ytest_orig, np.expm1(yhat_ho_log * w))
    rmspe_scores_wc_whole.append(error_w)
    # print(f'RMSPE cho weight {w:.3f}: {error_w:.6f}')

if rmspe_scores_wc_whole:
    best_rmspe_wc_whole = min(rmspe_scores_wc_whole)
    w_optimal_whole = weights_to_try_whole[np.argmin(rmspe_scores_wc_whole)]
    print(f'Best RMSPE sau "Whole Correction": {best_rmspe_wc_whole:.6f} với weight tối ưu: {w_optimal_whole:.3f}')
else:
    w_optimal_whole = 1.0 # Mặc định không hiệu chỉnh nếu có lỗi
    print("Không thể thực hiện 'Whole Correction', sử dụng weight mặc định 1.0")


# 7.2. Correction by Store
print("\nĐang thực hiện 'Correction by Store'...")
# Tạo DataFrame tạm thời từ ho_test để dễ dàng nhóm theo Store
res_ho = pd.DataFrame(data = ho_ytest_orig) # Sales gốc
res_ho['Prediction_log'] = yhat_ho_log
res_ho['Store'] = ho_xtest['Store'].values # Thêm cột Store vào res_ho để groupby

W_ho_store_correction = [] # Lưu trữ các weight đã hiệu chỉnh cho tập hold-out
W_test_store_correction = [] # Lưu trữ các weight sẽ áp dụng cho tập test

# Sắp xếp lại res_ho theo index của ho_xtest để đảm bảo thứ tự
res_ho = res_ho.reindex(ho_xtest.index)


for store_id in range(1, 1116): # Lặp qua tất cả các Store ID có thể có
    s1_ho = res_ho[res_ho['Store'] == store_id]
    s2_test = xtest_final[xtest_final['Store'] == store_id] # Dữ liệu của store này trong tập test
    
    current_best_w_store = w_optimal_whole # Bắt đầu với weight toàn cục tốt nhất
    
    if not s1_ho.empty: # Chỉ tìm weight nếu store này có trong hold-out
        W1_store = np.arange(0.990, 1.011, 0.001)
        S_store = []
        for w_s in W1_store:
            error_s = rmspe(np.expm1(s1_ho.Sales), np.expm1(s1_ho.Prediction_log * w_s))
            S_store.append(error_s)
        
        if S_store:
            Score_store = pd.Series(S_store, index=W1_store)
            # Lấy weight đầu tiên nếu có nhiều weight cùng cho RMSPE min
            current_best_w_store = Score_store[Score_store.values == Score_store.values.min()].index.values[0]

    # Áp dụng weight này cho các mẫu của store hiện tại trong ho_test và test_set
    b_ho_store = np.full(len(s1_ho), current_best_w_store)
    b_test_store = np.full(len(s2_test), current_best_w_store)
    
    W_ho_store_correction.extend(b_ho_store.tolist())
    W_test_store_correction.extend(b_test_store.tolist())


# Cách an toàn hơn: Tạo một cột weight trong ho_xtest_df (hoặc một bản sao)
ho_xtest_corrected = ho_xtest.copy()
ho_xtest_corrected['StoreWeight'] = w_optimal_whole # Mặc định

for store_id in range(1, 1116):
    s1_ho = res_ho[res_ho['Store'] == store_id]
    if not s1_ho.empty:
        W1_store = np.arange(0.990, 1.011, 0.001)
        S_store = []
        for w_s in W1_store:
            error_s = rmspe(np.expm1(s1_ho.Sales), np.expm1(s1_ho.Prediction_log * w_s))
            S_store.append(error_s)
        if S_store:
            Score_store = pd.Series(S_store, index=W1_store)
            current_best_w_store = Score_store[Score_store.values == Score_store.values.min()].index.values[0]
            ho_xtest_corrected.loc[ho_xtest_corrected['Store'] == store_id, 'StoreWeight'] = current_best_w_store

yhat_ho_log_store_corrected = yhat_ho_log * ho_xtest_corrected['StoreWeight'].values
error_ho_after_store_correction = rmspe(ho_ytest_orig, np.expm1(yhat_ho_log_store_corrected))
print(f'RMSPE trên tập hold-out sau "Correction by Store": {error_ho_after_store_correction:.6f}')


# Chuẩn bị W_test_store_correction cho tập test cuối cùng
xtest_final_corrected = xtest_final.copy()
xtest_final_corrected['StoreWeight'] = w_optimal_whole # Mặc định

for store_id in range(1, 1116): # Lặp lại để lấy weight cho tập test
    s1_ho = res_ho[res_ho['Store'] == store_id] # Dùng lại s1_ho để tìm weight
    current_best_w_store_for_test = w_optimal_whole
    if not s1_ho.empty:
        W1_store = np.arange(0.990, 1.011, 0.001)
        S_store = []
        for w_s in W1_store:
            error_s = rmspe(np.expm1(s1_ho.Sales), np.expm1(s1_ho.Prediction_log * w_s))
            S_store.append(error_s)
        if S_store:
            Score_store = pd.Series(S_store, index=W1_store)
            current_best_w_store_for_test = Score_store[Score_store.values == Score_store.values.min()].index.values[0]
    xtest_final_corrected.loc[xtest_final_corrected['Store'] == store_id, 'StoreWeight'] = current_best_w_store_for_test


#------------------------------------------------------------------------------
# 8. Tạo Dự đoán và Tệp Submission
#------------------------------------------------------------------------------
print("\nBước 8: Đang tạo dự đoán trên tập test và tệp submission...")
dtest_final_xgb = xgb.DMatrix(xtest_final[final_features]) # Chỉ lấy các đặc trưng đã huấn luyện
test_predictions_log = gbm.predict(dtest_final_xgb)

# Submission 1: Không hiệu chỉnh
sales_no_correction = np.expm1(test_predictions_log)
sales_no_correction[test_processed_df[test_processed_df['Open'] == 0].index] = 0 # Xử lý cửa hàng đóng
result_1 = pd.DataFrame({"Id": test_df_orig['Id'], 'Sales': sales_no_correction})
result_1['Sales'] = result_1['Sales'].clip(lower=0)
result_1.to_csv("Rossmann_submission_final.csv", index=False)
print("Đã tạo Rossmann_submission_1_replication.csv (không hiệu chỉnh)")

# Submission 2: Whole Correction
sales_whole_correction = np.expm1(test_predictions_log * w_optimal_whole)
sales_whole_correction[test_processed_df[test_processed_df['Open'] == 0].index] = 0
result_2 = pd.DataFrame({"Id": test_df_orig['Id'], 'Sales': sales_whole_correction})
result_2['Sales'] = result_2['Sales'].clip(lower=0)
result_2.to_csv("Rossmann_submission_2_final_wc.csv", index=False)
print("Đã tạo Rossmann_submission_2_replication_wc.csv (hiệu chỉnh toàn cục)")

# Submission 3: Correction by Store
# Sử dụng xtest_final_corrected['StoreWeight'] đã tính ở trên
sales_store_correction = np.expm1(test_predictions_log * xtest_final_corrected['StoreWeight'].values)

pen_status_submission = test_processed_df['Open'].values # Lấy trạng thái Open theo thứ tự của test_processed_df
if len(sales_store_correction) == len(open_status_submission):
    sales_store_correction[open_status_submission == 0] = 0
else:
    print("Cảnh báo: Độ dài không khớp khi áp dụng trạng thái Open cho submission 3.")


result_3 = pd.DataFrame({"Id": test_df_orig['Id'], 'Sales': sales_store_correction})
result_3['Sales'] = result_3['Sales'].clip(lower=0)
result_3.to_csv("Rossmann_submission_3_final_sc.csv", index=False)
print("Đã tạo Rossmann_submission_3_replication_sc.csv (hiệu chỉnh theo cửa hàng)")


# (Phần Ensemble được bỏ qua trong lần triển khai này để đơn giản hóa,
#  nhưng là một bước quan trọng để cải thiện thêm điểm số)

print("\nHoàn thành!")


