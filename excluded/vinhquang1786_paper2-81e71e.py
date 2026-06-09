# from tqdm import tqdm
# from itertools import product

# import numpy as np
# import pandas as pd

# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import GroupKFold
# from sklearn.impute import SimpleImputer, KNNImputer
# import warnings

# from sklearn.linear_model import Ridge
# from sklearn.ensemble import StackingRegressor
# warnings.filterwarnings("ignore")

# from itertools import product

# hours = range(0, 6, 1)
# minutes = range(0, 60, 5)

# target_col = "bg+1-00"
# group_col = "p_num"
# date_col = "time"

# bg_cols = [f"bg-{i}-{j:02d}" for i, j in product(hours, minutes)]
# insu_cols = [f"insulin-{i}-{j:02d}" for i, j in product(hours, minutes)]
# carb_cols = [f"carbs-{i}-{j:02d}" for i, j in product(hours, minutes)]
# hr_cols = [f"hr-{i}-{j:02d}" for i, j in product(hours, minutes)]
# step_cols = [f"steps-{i}-{j:02d}" for i, j in product(hours, minutes)]
# cals_cols = [f"cals-{i}-{j:02d}" for i, j in product(hours, minutes)]

# feature_cols = bg_cols + insu_cols + carb_cols + hr_cols + step_cols + cals_cols





# df_train = pd.read_csv(
#     '/kaggle/input/brist1d/train.csv',
#     index_col='id',
#     parse_dates=['time'],
# )

# df_train.columns = df_train.columns.str.replace(':', '-')


# import random
# import os
# import numpy as np
# import torch


# def seed_everything(seed):
#     random.seed(seed)
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = True


# df_train.columns = df_train.columns.str.replace(":", "-")


# for colset in [bg_cols, insu_cols, carb_cols, hr_cols, step_cols, cals_cols]:
#     df_train[colset] = (
#         df_train[colset]
#         .interpolate(axis=1)
#         .fillna(method="bfill", axis=1)
#         .fillna(method="ffill", axis=1)
#     )

# feature_columns = {
#     'bg': bg_cols,
#     'hr': hr_cols,
#     'steps': step_cols
# }

# def impute_time_series_features(df, columns, method='median', window=5):
#     for col in columns:
#         df[col] = df[col].fillna(df[col].rolling(window=window, min_periods=1).median())

#         df[col] = df[col].fillna(method='ffill')
#         df[col] = df[col].fillna(method='bfill')

#         df[col] = df.groupby('p_num')[col].transform(lambda x: x.fillna(x.median()))
#     return df

# for feature, columns in feature_columns.items():
#     df_train = impute_time_series_features(df_train, columns, method='median')

# all_imputed_columns = bg_cols + hr_cols + step_cols
# print("Remaining null values in training set:", df_train[all_imputed_columns].isnull().sum().sum())



# feature_columns_insulin_cals = {
#     'insulin': insu_cols,
#     'cals': cals_cols
# }

# def impute_non_temporal_features(df, columns, method='median', window=5):
#     for col in columns:
#         df[col] = df.groupby('p_num')[col].transform(lambda x: x.fillna(x.median()))

#         df[col] = df[col].fillna(df[col].rolling(window=window, min_periods=1).median())
#     return df

# for feature, columns in feature_columns_insulin_cals.items():
#     df_train = impute_non_temporal_features(df_train, columns, method='median')

# all_imputed_columns_insulin_cals = insu_cols + cals_cols
# print("Remaining null values in training set:", df_train[all_imputed_columns_insulin_cals].isnull().sum().sum())



# feature_columns_insulin_cals = {
#     'insulin': insu_cols,
#     'cals': cals_cols
# }

# def impute_non_temporal_features(df, columns, method='median', window=5):
#     for col in columns:
#         df[col] = df.groupby('p_num')[col].transform(lambda x: x.fillna(x.median()))

#         df[col] = df[col].fillna(df[col].rolling(window=window, min_periods=1).median())
#     return df

# for feature, columns in feature_columns_insulin_cals.items():
#     df_train = impute_non_temporal_features(df_train, columns, method='median')

# all_imputed_columns_insulin_cals = insu_cols + cals_cols
# print("Remaining null values in training set:", df_train[all_imputed_columns_insulin_cals].isnull().sum().sum())



# activity_cols = [col for col in df_train.columns if col.startswith('activity')]
# dominant_activity_train = df_train[activity_cols].mode(axis=1)[0]
# df_train['dominant_activity'] = dominant_activity_train
# print(df_train['dominant_activity'].value_counts())


# activity_replace = {
#     'Walk': 'Walking',
#     'Run': 'Running',
#     'Swim': 'Swimming',
# }
# df_train['dominant_activity'] = df_train['dominant_activity'].replace(activity_replace)
# print(df_train['dominant_activity'])


# df_train['dominant_activity'].fillna('No Activity', inplace=True)

# from sklearn.preprocessing import LabelEncoder

# label_encoder = LabelEncoder()

# combined_activities = pd.concat([
#     df_train['dominant_activity']
# ])

# label_encoder.fit(combined_activities)

# df_train['dominant_activity_encoded'] = label_encoder.transform(df_train['dominant_activity'])

# df_train = df_train.drop(['dominant_activity'], axis = 1)

# feature_cols = feature_cols + ['dominant_activity_encoded']


# imputer = SimpleImputer()

# df_train[feature_cols] = imputer.fit_transform(df_train[feature_cols])

# df_train['time_hour'] = pd.to_datetime(df_train['time']).dt.hour

# feature_cols.extend(["time_hour"])

# df_train_final = df_train[feature_cols]

# y_target = df_train[[target_col]]


# np.isnan(df_train_final).sum()



# df_train_final


# y_target


!pip install statsmodels


import numpy as np
import logging
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn import linear_model
from statsmodels.stats.multitest import multipletests
from mpmath import mp
import time

from joblib import Parallel, delayed
from multiprocessing import shared_memory

import os
from sklearn.linear_model import Lasso
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(2025)

def generate_data(n, p, true_beta):

    X = np.random.normal(loc=0, scale=1, size=(n, p))
    true_beta = np.reshape(true_beta, (p, 1))
    noise = np.random.normal(loc=0, scale=1, size=(n, 1))
    y = X @ true_beta + noise
    return X, y

mp.dps = 20

def construct_A_XA_Ac_XAc_bhA(X, bh):
    p = X.shape[1]
    A = sorted(list(np.where(bh != 0)[0]))
    XA = X[:, A] if A else np.empty((X.shape[0], 0))

    return A, XA

def construct_test_statistic(j, XA, y, A):
    ej = np.zeros((len(A), 1))
    ej[A.index(j)] = 1
    inv = np.linalg.pinv(XA.T @ XA)
    etaj = XA @ inv @ ej
    etajTy = (etaj.T @ y)[0, 0]

    return etaj, etajTy



def compute_yz(y, etaj, zk, n, sigma_squared):

    sq_norm_val =(etaj.T @ etaj).item()
    sq_norm = sigma_squared * sq_norm_val
    
    
    if sq_norm < 1e-12: sq_norm = 1e-12

    # --- PHẦN TỐI ƯU HÓA ---
    # Thay vì tạo ma trận proj_matrix (n x n), chúng ta tính toán trực tiếp.
    # a = (I - (ηηᵀ)/||η||²)y = y - η(ηᵀy)/||η||²
    
    # 1. Tính tích trong (inner product) trước -> ra một số vô hướng
    # etajTy_scalar = (etaj.T @ y)[0, 0]
    etajTy_scalar = (etaj.T @ y).item()

    
    # 2. Tính vector "a" bằng các phép toán trên vector
    a = y - etaj * (etajTy_scalar / sq_norm)*sigma_squared
    # -----------------------
    
    b = (etaj / sq_norm) * sigma_squared
    yz = a + b * zk
   
    return yz, b




def compute_polyhedron(X, M, s, lambda_val):
    n, p = X.shape
    if not M: return np.empty((0, n)), np.empty(0)
    
    X_M = X[:, M]
    X_notM_indices = np.setdiff1d(np.arange(p), M)
    X_notM = X[:, X_notM_indices]
    
    try:
        XATXA_inv = np.linalg.pinv(X_M.T @ X_M)
    except np.linalg.LinAlgError:
        return np.empty((0, n)), np.empty(0)

    A0, b0 = np.empty((0, n)), np.empty(0)
    if X_notM.shape[1] > 0:
        # TỐI ƯU HÓA TỐT HƠN NỮA
        # Tính X_notM.T @ P_M mà không cần tạo P_M (n x n)
        # X_notM.T @ P_M = X_notM.T @ (X_M @ inv(X_M.T @ X_M) @ X_M.T)
        #                = (X_notM.T @ X_M) @ inv(X_M.T @ X_M) @ X_M.T
        term_k_x_k = X_notM.T @ X_M @ XATXA_inv
        X_notM_T_P_M = term_k_x_k @ X_M.T
        
        # A0_T = (1/lambda_val) * (X_notM.T - X_notM.T @ P_M)
        A0_T = (1/lambda_val) * (X_notM.T - X_notM_T_P_M)
        
        A0 = np.vstack([A0_T, -A0_T])
        
        XT_M_pinv = np.linalg.pinv(X_M.T)
        s_term = X_notM.T @ XT_M_pinv @ s
        b0 = np.hstack([np.ones(X_notM.shape[1]) - s_term, 
                        np.ones(X_notM.shape[1]) + s_term])
    
    A1 = -np.diag(s) @ XATXA_inv @ X_M.T
    b1 = -lambda_val * np.diag(s) @ XATXA_inv @ s
    
    return np.vstack([A0, A1]), np.hstack([b0, b1])





def compute_truncation_bounds(A, b, eta, y):
    eta = eta.ravel()
    y = y.ravel()
    eta_norm = np.dot(eta, eta)
    if eta_norm < 1e-10:
        raise ValueError("eta has near-zero norm")
    c = eta / eta_norm
    z = y - np.dot(eta, y) * c
    Ac = A @ c
    Az = A @ z
    V_minus = None
    V_plus = None
    V_zero = np.inf
    for j in range(A.shape[0]):
        if abs(Ac[j]) == 0:
            V_zero = min(V_zero, b[j] - Az[j])
        elif Ac[j] < 0:
            if V_minus is None or (b[j] - Az[j]) / Ac[j] > V_minus:
                V_minus = (b[j] - Az[j]) / Ac[j]
        elif Ac[j] > 0:
            if V_plus is None or (b[j] - Az[j]) / Ac[j] < V_plus:
                V_plus = (b[j] - Az[j]) / Ac[j]
    if V_minus is None:
        V_minus = -np.inf
    if V_plus is None:
        V_plus = np.inf
    if V_zero < 0:
        raise ValueError("Invalid truncation bounds: V_zero < 0")

 

    return V_minus, V_plus
def parametric_lasso(X, yz, lamda, b, n, p,eta_j):
    yz_flatten = yz.flatten()
    clf = linear_model.Lasso(alpha=lamda/n, fit_intercept=False, tol=1e-10)
    clf.fit(X, yz_flatten)
    bhz = clf.coef_

    A, XA = construct_A_XA_Ac_XAc_bhA(X, bhz)
    s=np.sign(bhz[A])
    A1,b1= compute_polyhedron(X,A,s, lamda)
    V_l,V_r=compute_truncation_bounds(A1, b1, eta_j, yz_flatten)
    print("Done5")
    print("V_l: ",V_l)
    print("V_r: ",V_r)

    return V_l,V_r, A, bhz

def create_shared_array(arr: np.ndarray):
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    shm_arr[:] = arr[:]
    return shm, arr.shape, arr.dtype.str

def attach_shared_array(shm_name, shape, dtype_str):
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
    return shm, arr

# ============================
# Worker xử lý 1 đoạn zk
# ============================


def setup_worker_logging(worker_id):
    """Cấu hình logger cho mỗi tiến trình con."""
    logger = logging.getLogger(f"worker_{worker_id}")
    logger.setLevel(logging.INFO)  # Chỉ ghi log từ mức INFO trở lên

    # Tạo handler để ghi ra console
    handler = logging.StreamHandler()
    
    # Tạo formatter để định dạng log message
    # Bao gồm: Thời gian - ID Tiến trình - Tên Logger - Mức độ - Nội dung
    formatter = logging.Formatter(
        '%(asctime)s - PID:%(process)d - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)

    # Thêm handler vào logger (nếu chưa có)
    if not logger.handlers:
        logger.addHandler(handler)
        
    return logger





def process_segment(z_start, z_end,
                    X_shm_name, X_shape, X_dtype,
                    y_shm_name, y_shape, y_dtype,
                    etaj_shm_name, etaj_shape, etaj_dtype,
                    alpha, sigma_sq, small_eps=0.0000000000000001, max_iter=2000,
                    lasso_tol=1e-4, lasso_max_iter=20000):
    

    worker_id = os.getpid()
    logger = setup_worker_logging(worker_id)
    logger.info(f"BẮT ĐẦU xử lý đoạn [{float(z_start):.4f}, {float(z_end):.4f}]")

    
    start_time = time.time()

    # Attach shared arrays
    shmX, X = attach_shared_array(X_shm_name, X_shape, X_dtype)
    shmy, y = attach_shared_array(y_shm_name, y_shape, y_dtype)
    shmetaj, etaj = attach_shared_array(etaj_shm_name, etaj_shape, etaj_dtype)

    n = X.shape[0]
    results = []
    clf = Lasso(alpha=alpha / n, fit_intercept=False, warm_start=True,max_iter=80000,tol=1e-7)

    zk = float(z_start)
    it = 0
   
    log_interval = 250

    while zk < z_end:
        it += 1
        if it == 1 or it % log_interval == 0:
            logger.info(f"Vòng lặp #{it}... zk hiện tại = {float(zk):.6f}")


        yz, b_vec = compute_yz(y, etaj, zk, n, sigma_sq)
        yz_flat = yz.flatten()

    
        
       
        clf.fit(X, yz_flat)
       
        bhz = clf.coef_.copy()

        A, _ = construct_A_XA_Ac_XAc_bhA(X, bhz)

        if len(A) == 0:
            V_l, V_r = zk, z_end
        else:
            s = np.sign(bhz[A])
            A1, b1 = compute_polyhedron(X, A, s, alpha)
        
            V_l, V_r = compute_truncation_bounds(A1, b1, etaj, yz_flat)
            
        print(f"Vòng lặp #{it}: V_r={V_r}, V_l={V_l}, V_r-V_l={V_r-V_l}") # In ra để debug
        results.append((zk, A, bhz))
        
        # KIỂM TRA TÍNH HỢP LỆ CỦA KHOẢNG TRƯỚC KHI CẬP NHẬT
        if np.isfinite(V_r) and np.isfinite(V_l) and V_r > V_l and V_r > zk + 1e-12:
           
            zk = min(V_r + small_eps, z_end)
        else:

            logger.warning(f"Vòng lặp #{it}: Đa diện không hợp lệ hoặc không tiến lên (V_r={V_r}, V_l={V_l}). Dùng bước nhảy nhỏ.")
            zk = zk + small_eps
            
   
    shmX.close()
    shmy.close()
    shmetaj.close()
    end_time = time.time()
    duration = end_time - start_time
    
   

    logger.info(f"KẾT THÚC xử lý đoạn [{z_start.item():.4f}, {z_end.item():.4f}]. "
            f"Tổng cộng: {it} vòng lặp. Thời gian: {duration:.2f} giây.")

    return results




# ============================
# Hàm song song hóa chính
# ============================

def run_parametric_lasso(X, y, alpha, etaj, threshold, sigma_sq,
                                 n_segments, overlap=1e-8, **worker_kwargs):
    shmX, X_shape, X_dtype = create_shared_array(X)
    shmy, y_shape, y_dtype = create_shared_array(y.reshape(-1))
    shmetaj, etaj_shape, etaj_dtype = create_shared_array(etaj.reshape(-1))

    edges = np.linspace(-threshold, threshold, n_segments + 1)
    segments = []
    overlap=0.0001
    for i in range(n_segments):
        z0 = edges[i] - (overlap if i > 0 else 0.0)
        z1 = edges[i+1] + (overlap if i < n_segments - 1 else 0.0)
        z0 = max(z0, -threshold)
        z1 = min(z1, threshold)
        segments.append((z0, z1))

    parallel = Parallel(n_jobs=n_segments)
    jobs = (delayed(process_segment)(
                z0, z1,
                shmX.name, X_shape, X_dtype,
                shmy.name, y_shape, y_dtype,
                shmetaj.name, etaj_shape, etaj_dtype,
                alpha, sigma_sq,
                **worker_kwargs
            ) for (z0, z1) in segments)
    segment_results = parallel(jobs)

    shmX.close(); shmX.unlink()
    shmy.close(); shmy.unlink()
    shmetaj.close(); shmetaj.unlink()

    all_entries = []
    for seg in segment_results:
        all_entries.extend(seg)
    all_entries.sort(key=lambda t: t[0])

    merged_zk = []
    merged_active_sets = []
    merged_bhz = []
    tol = 0.00001
    for zk, A, bhz in all_entries:
        if merged_zk and abs(zk - merged_zk[-1]) < tol:
            continue
        merged_zk.append(float(zk))
        merged_active_sets.append(A)
        merged_bhz.append(bhz)

    if len(merged_zk) == 0 or merged_zk[-1] < threshold - 1e-12:
        merged_zk.append(float(threshold))

    return merged_zk, merged_active_sets, merged_bhz






















def p_value(A_obs, list_active_set, list_zk, etaj, z_obs, cov_matrix):
    tn_sigma = np.sqrt((etaj.T @ cov_matrix @ etaj))[0, 0]
    if tn_sigma < 1e-9: return None
    
    z_intervals = []
    for i, active_set in enumerate(list_active_set):
        if set(A_obs) == set(active_set):
            z_intervals.append([list_zk[i], list_zk[i+1]])

    print("z_intervals: ",z_intervals)

    if not z_intervals: return None

    tn_mu = 0
    numerator = mp.mpf(0)
    denominator = mp.mpf(0)
    
    for al, ar in z_intervals:
        prob_interval = mp.ncdf((ar - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)
        denominator += prob_interval
        if z_obs >= ar:
            numerator += prob_interval
        elif z_obs > al:
            numerator += mp.ncdf((z_obs - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)
            
    print("denominator:",denominator)
    if denominator < 1e-100: return None
    
    pivot_val = float(numerator / denominator)
    print("Done7")
    return 2 * min(pivot_val, 1 - pivot_val)

def ensure_series(y, name="y"):
    import numpy as np, pandas as pd
    # DataFrame 1 cột -> chuyển thành Series
    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 1:
            return y.iloc[:, 0].copy()
        else:
            raise ValueError("y là DataFrame nhiều cột — cần chỉ rõ cột target.")
    # ndarray
    if isinstance(y, np.ndarray):
        if y.ndim == 1:
            return pd.Series(y, name=name)
        if y.ndim == 2:
            # trường hợp (n,1) hoặc (1,n)
            if y.shape[1] == 1 or y.shape[0] == 1:
                return pd.Series(y.ravel(), name=name)
            else:
                raise ValueError(f"ndarray không phải vector 1D hay cột 1: shape={y.shape}")
    # nếu là list, tuple, pd.Series, v.v.
    return pd.Series(y, name=name)

# dùng:

# =============================================================================
# HÀM CHẠY CHÍNH: QUY TRÌNH PHÂN TÍCH DỮ LIỆU THỰC
# =============================================================================
import numpy as np
import random
if __name__ == '__main__':
    # --- BƯỚC 0: CHUẨN BỊ DỮ LIỆU ĐẦU VÀO ---
    
    print("Sử dụng dữ liệu X_clean và y_clean...")

    n = 177000
    p = 433
    true_beta = [0.25 if i < 25 else 0 for i in range(p)]

    X_clean, y_clean = generate_data(n, p, true_beta)  
    if isinstance(X_clean, np.ndarray):
        col_names = [f"x{i}" for i in range(X_clean.shape[1])]  # hoặc dùng tên khác
        X_clean = pd.DataFrame(X_clean, columns=col_names)
    else:
        # nếu đã là DataFrame, copy để tránh thay đổi ngoài ý muốn
        X_clean = X_clean.copy()

    y_clean = ensure_series(y_clean, name="y")
    # X_clean =df_train_final
    # y_clean =y_target

    
    
    print(f"Dữ liệu đầu vào: X_clean.shape={X_clean.shape}, y_clean.shape={y_clean.shape}")

    # --- GIAI ĐOẠN 1: LỰA CHỌN MÔ HÌNH BẰNG LASSO (KHÔNG DÙNG CV) ---
    print("\n--- Giai đoạn 1: Lựa chọn Mô hình bằng Lasso ---")

    # 1.1. Chuẩn hóa X
    print("Đang chuẩn hóa X_clean...")
    scaler = StandardScaler()
    X_columns = X_clean.columns
    X_scaled = scaler.fit_transform(X_clean)
    n, p = X_scaled.shape
    BEST_ALPHA = 7500
    print(f"Sử dụng alpha cố định: {BEST_ALPHA:.4f}")
    lasso_model = Lasso(alpha=BEST_ALPHA/n, fit_intercept=False).fit(X_scaled, y_clean)
    A_obs, XA_obs = construct_A_XA_Ac_XAc_bhA(X_scaled, lasso_model.coef_)

    if not A_obs:
        print("Lasso không chọn đặc trưng nào. Kết thúc.")
        exit()

    print(f"\nLasso đã chọn {len(A_obs)} đặc trưng.")
    print("Các đặc trưng được chọn:", X_columns[A_obs].tolist())

    print("Ước lượng sigma^2 từ phần dư của OLS trên các biến được chọn (refitted OLS)...")
    ols_refit = sm.OLS(y_clean, X_scaled[:, A_obs]).fit()
    df_residuals = n - len(A_obs)
    if df_residuals <= 0:
        raise ValueError("Không thể ước lượng sigma^2 vì bậc tự do <= 0.")
    sigma_squared_hat = np.sum(ols_refit.resid**2) / df_residuals
    print(f"Ước lượng phương sai nhiễu (σ^2) đã sửa: {sigma_squared_hat:.4f}")



    # import matplotlib.pyplot as plt
    # import numpy as np
    # from scipy.stats import gaussian_kde
    
    # # Lấy residual
    # residuals = ols_refit.resid
    
    # # Tạo histogram
    # plt.figure(figsize=(8, 5))
    # count, bins, _ = plt.hist(residuals, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')
    
    # # Vẽ đường mật độ kernel
    # kde = gaussian_kde(residuals)
    # x_range = np.linspace(min(residuals), max(residuals), 1000)
    # plt.plot(x_range, kde(x_range), color='red', linewidth=2, label='Mật độ ước lượng')
    
    # # Vẽ đường phân phối chuẩn tham chiếu
    # mean_res = np.mean(residuals)
    # std_res = np.std(residuals)
    # normal_pdf = (1/(std_res*np.sqrt(2*np.pi))) * np.exp(-0.5*((x_range-mean_res)/std_res)**2)
    # plt.plot(x_range, normal_pdf, 'k--', linewidth=2, label='Chuẩn N(μ,σ²)')
    
    # plt.title('Phân phối residual (Histogram + KDE)')
    # plt.xlabel('Residual')
    # plt.ylabel('Mật độ')
    # plt.legend()
    # plt.grid(alpha=0.3)
    # plt.show()
    
        

    # --- GIAI ĐOẠN 2: SUY LUẬN CHỌN LỌC ---
    print(f"\n--- Giai đoạn 2: Bắt đầu Suy luận Chọn lọc cho {len(A_obs)} đặc trưng ---")
    p_values_dict = {}
    
    y_col_vector = y_clean.values.reshape(-1, 1)

    for i, j_selected in enumerate(A_obs):
        
        feature_name = X_columns[j_selected]
        start_time = time.time()
        print(f"  [{i+1}/{len(A_obs)}] Đang phân tích '{feature_name}'...")
        etaj, etajTy_orig = construct_test_statistic(j_selected, XA_obs, y_col_vector, A_obs)
        sq_norm_eta = (etaj.T @ etaj)
        if sq_norm_eta < 1e-12: sq_norm_eta = 1e-12
        
  
        etajTy_scalar = (etaj.T @ y_col_vector)
        a_vec = y_col_vector - etaj * (etajTy_scalar / sq_norm_eta) # Cách hiệu quả
        z_obs = etajTy_orig 
        print("z_obs: ",z_obs)
        sigma_z = np.sqrt(sq_norm_eta * sigma_squared_hat)
        
        threshold = 20*sigma_z
        if (z_obs>threshold):
            threshold=z_obs*1.5
        print("threshold: ",threshold)
  
        
        n_segments = os.cpu_count()  # số core logic của CPU
        
        list_zk, list_active_set, list_bhz = run_parametric_lasso(
            X_scaled, y_col_vector, BEST_ALPHA, etaj,
            threshold=threshold, sigma_sq=sigma_squared_hat,
            n_segments=n_segments,
            small_eps=0.001, max_iter=10000
        )
    
        cov_matrix = np.identity(n) * sigma_squared_hat
    
        p_val = p_value(A_obs, list_active_set, list_zk, etaj, z_obs, cov_matrix)
        
        
        p_values_dict[feature_name] = p_val
        elapsed_time = time.time() - start_time
        print(f"    -> p-value = {p_val if p_val is not None else 'N/A'} (Thời gian: {elapsed_time:.2f}s)")
        break
        
    # --- GIAI ĐOẠN 3: PHÂN TÍCH KẾT QUẢ ---

    print("\n--- Giai đoạn 3: Phân tích Kết quả Cuối cùng ---")
    results_df = pd.DataFrame.from_dict(p_values_dict, orient='index', columns=['raw_p_value']).dropna()
    results_df.index.name = 'feature'
    results_df.sort_values('raw_p_value', inplace=True)

    if not results_df.empty:
        reject, p_adjusted, _, _ = multipletests(results_df['raw_p_value'], alpha=0.05, method='fdr_bh')
        results_df['adjusted_p_value'] = p_adjusted
        results_df['is_significant'] = reject
        
        lasso_coef_df = pd.DataFrame({
            'feature': X_columns[A_obs],
            'lasso_coefficient': lasso_model.coef_[A_obs]
        })
        results_df = pd.merge(results_df, lasso_coef_df, left_index=True, right_on='feature').set_index('feature')
        results_df = results_df.sort_values('adjusted_p_value')

        print("\nBảng kết quả Suy luận Chọn lọc:")
        print(results_df)

