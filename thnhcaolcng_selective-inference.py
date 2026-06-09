from tqdm import tqdm
from itertools import product

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer, KNNImputer
import warnings


from sklearn.ensemble import StackingRegressor
warnings.filterwarnings("ignore")

from itertools import product
from scipy.stats import norm
from scipy.special import logsumexp





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



# y_clean.shape


# X_clean.shape


# X_clean =pd.read_csv("/kaggle/input/x-clean/X_clean.csv")
# y_clean=pd.read_csv("/kaggle/input/y-clean/y_clean.csv")




# X=pd.concat([X_clean,y_clean],axis=1)


# X.shape


# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns


# # Nếu có quá nhiều đặc trưng, hãy vẽ cho từng nhóm nhỏ
# for i in range(0, 434, 20):
#     plt.figure(figsize=(15, 6))
#     sns.boxplot(data=X.iloc[:, i:i+20])
#     plt.title(f"Box Plot cho đặc trưng {i} đến {i+19}")
#     plt.show()


# import pandas as pd
# from collections import Counter

# outlier_indices = []

# for column in X.columns[-1]:
#     Q1 = X[column].quantile(0.25)
#     Q3 = X[column].quantile(0.75)
#     IQR_val = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR_val
#     upper_bound = Q3 + 1.5 * IQR_val
#     column_outliers = X[(X[column] < lower_bound) | (X[column] > upper_bound)].index
#     outlier_indices.extend(column_outliers)

# outlier_counts = Counter(outlier_indices)

# feature_threshold = 1

# multiple_outliers = [i for i, count in outlier_counts.items() if count > feature_threshold]

# print(f"Số hàng ban đầu: {len(X)}")
# print(f"Số hàng bị coi là outlier nghiêm trọng: {len(multiple_outliers)}")

# X_cleaned = X.drop(multiple_outliers)
# if 'y' in locals():
#     y_cleaned = y.drop(multiple_outliers)

# print(f"Số hàng sau khi xóa: {len(X_cleaned)}")






# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns


# # Nếu có quá nhiều đặc trưng, hãy vẽ cho từng nhóm nhỏ
# for i in range(0, 434, 20):
#     plt.figure(figsize=(15, 6))
#     sns.boxplot(data=X.iloc[:, i:i+20])
#     plt.title(f"Box Plot cho đặc trưng {i} đến {i+19}")
#     plt.show()


!pip install skglm
!pip install statsmodels
# !pip install celer
!pip install -U scikit-learn-intelex skglm



# def _log_cdf_diff(V_r, V_l):
#     """
#     Tính log(P(V_l < Z < V_r)) một cách ổn định về mặt số học.
#     Z ~ N(0, 1).
#     Sử dụng công thức: log(a - b) = log(a) + log(1 - exp(log(b) - log(a)))
#     """
#     if V_l >= V_r:
#         return -np.inf  # log(0)

#     log_cdf_r = norm.logcdf(V_r)
#     log_cdf_l = norm.logcdf(V_l)

#     # np.log1p(x) tính log(1 + x) một cách chính xác hơn cho x nhỏ.
#     # Ta có log(1 - exp(y)) = log1p(-exp(y))
#     # y = log_cdf_l - log_cdf_r luôn âm vì V_l < V_r
#     return log_cdf_r + np.log1p(-np.exp(log_cdf_l - log_cdf_r))

# def _calculate_pivot_under_null(V_intervals, V_obs):
#     """
#     Tính giá trị của đại lượng chốt (CDF của phân phối cụt) dưới H0.
#     Giá trị trả về tuân theo phân phối Uniform(0, 1).
#     """
#     # 1. Tính log(denominator) - Tổng khối xác suất của vùng cắt cụt
#     log_probs_denom = [_log_cdf_diff(V_r, V_l) for V_l, V_r in V_intervals]
#     log_denominator = logsumexp(log_probs_denom)
#     print("log_denominator:",log_denominator)
    


#     # 2. Tính log(numerator) - Khối xác suất từ -inf đến z_obs trong vùng cắt cụt
#     log_probs_num = []
#     for V_l, V_r in V_intervals:
#         # Khoảng tích phân là [V_l, min(V_r, V_obs)]
#         integration_end = min(V_r, V_obs)
#         if V_l < integration_end:
#             log_probs_num.append(_log_cdf_diff(integration_end, V_l))
    
    
#     log_numerator = logsumexp(log_probs_num)
#     print("log_numerator:",log_numerator)

#     # 3. Tính log của pivot
#     log_pivot = log_numerator - log_denominator
    
#     return np.exp(log_pivot)

# def p_value(A_obs, list_active_set, list_zk, etaj, z_obs, cov_matrix):
#     """
#     Tính p-value cho suy luận chọn lọc bằng cách trước tiên tính đại lượng chốt (pivot).
#     """
#     print("    Bắt đầu tính p-value thông qua pivot trên log-space...")
#     start_time_pval = time.time()
    
#     # --- Bước A: Xây dựng vùng cắt cụt z_intervals ---
#     z_intervals = []
#     set_A_obs = set(A_obs)
#     for i in range(len(list_zk) - 1):
#         if set(list_active_set[i]) == set_A_obs:
#             z_intervals.append((list_zk[i], list_zk[i+1]))

#     if not z_intervals:
#         print("    CẢNH BÁO: Không tìm thấy khoảng khớp. Trả về p-value = 1.0")
#         return 1.0

#     # --- Bước B: Tính tham số phân phối ---
#     sq_norm_eta = (etaj.T @ etaj).item()
#     sigma_squared_hat = cov_matrix[0, 0]
#     tn_sigma = np.sqrt(sq_norm_eta * sigma_squared_hat)
    
#     if tn_sigma < 1e-12: tn_sigma = 1e-12
#     print(f"    tn_sigma tính được: {tn_sigma:.6f}")

#     # --- Bước C: Chuẩn hóa sang không gian V ---
#     V_intervals = [(z_l / tn_sigma, z_r / tn_sigma) for z_l, z_r in z_intervals]
#     V_obs = z_obs / tn_sigma

#     # --- Bước D: Tính giá trị pivot dưới H0 ---
#     pivot_value = _calculate_pivot_under_null(V_intervals, V_obs)
#     print(f"    Giá trị Pivot (CDF) tính được: {pivot_value:.6f}")
    
#     # --- Bước E: Chuyển pivot thành p-value hai phía ---
#     final_p_value = 2 * min(pivot_value, 1 - pivot_value)
    
#     print(f"    Thời gian tính p-value: {(time.time() - start_time_pval):.4f}s")

#     return final_p_value


!pip install skglm
from tqdm import tqdm
from itertools import product

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer, KNNImputer
import warnings


from sklearn.ensemble import StackingRegressor
warnings.filterwarnings("ignore")

from itertools import product
from scipy.stats import norm
from scipy.special import logsumexp



import numpy as np
import logging
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
# from celer import Lasso
from skglm import Lasso
from sklearn import linear_model
from statsmodels.stats.multitest import multipletests
from mpmath import mp
import time

from joblib import Parallel, delayed
from multiprocessing import shared_memory

import os

mp.dps =400

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
    etajTy = (etaj.T @ y).item()

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
        s_term = s_term.flatten()
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
                    alpha, sigma_sq):


    small_eps=0.000001
    

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
    clf = Lasso(alpha=alpha / n, fit_intercept=False, warm_start=True,max_iter=80000,tol=1e-9)

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
        # Bước "làm sạch"
        coeff_threshold = 1e-9 
        bhz[np.abs(bhz) < coeff_threshold] = 0

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
                                 n_segments, overlap=0, **worker_kwargs):
    shmX, X_shape, X_dtype = create_shared_array(X)
    shmy, y_shape, y_dtype = create_shared_array(y.reshape(-1))
    shmetaj, etaj_shape, etaj_dtype = create_shared_array(etaj.reshape(-1))

    edges = np.linspace(-threshold, threshold, n_segments + 1)
    segments = []
    overlap=0
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
                alpha, sigma_sq
            ) for (z0, z1) in segments)
    start=time.time()
    segment_results = parallel(jobs)
    print("Thời gian cho ra segment_results: ",time.time()-start)

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
        if merged_active_sets and (merged_active_sets[-1]==A):
            continue
        merged_zk.append(float(zk))
        merged_active_sets.append(A)
        merged_bhz.append(bhz)

    if len(merged_zk) == 0 or merged_zk[-1] < threshold - 1e-12:
        merged_zk.append(float(threshold))

    return merged_zk, merged_active_sets, merged_bhz






















# def p_value(A_obs, list_active_set, list_zk, etaj, z_obs, cov_matrix):
#     start=time.time()
#     tn_sigma = (np.sqrt((etaj.T @ cov_matrix @ etaj))).item()
#     print("tn_sigma:",tn_sigma)
#     if tn_sigma < 1e-9: return None
    
#     z_intervals = []
#     for i, active_set in enumerate(list_active_set):
#         if set(A_obs) == set(active_set):
#             z_intervals.append([list_zk[i], list_zk[i+1]])

#     print("z_intervals: ",z_intervals)

#     if not z_intervals: return None

#     tn_mu = 0
#     numerator = mp.mpf(0)
#     denominator = mp.mpf(0)
    
#     for al, ar in z_intervals:
#         prob_interval = mp.ncdf((ar - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)
#         denominator += prob_interval
#         if z_obs >= ar:
#             numerator += prob_interval
#         elif z_obs > al:
#             numerator += mp.ncdf((z_obs - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)
#     print("numerator: ",numerator)       
#     print("denominator:",denominator)
   
#     # if denominator<1e-100:
    
#     #     return 0
    
#     pivot_val = float(numerator / denominator)
#     print("Thơi gian tính p value: ",time.time()-start)
#     return 2 * min(pivot_val, 1 - pivot_val)

import mpmath as mp
import time

mp.mp.dps = 2000  # độ chính xác (có thể chỉnh)

# ====== Các hàm hỗ trợ ======

def log_survival(t):
    return mp.log(0.5 * mp.erfc(t / mp.sqrt(2)))

def log_interval_mass(la, lb):
    delta = lb - la
    if delta <= -50:
        return la
    return la + mp.log1p(- mp.e**(delta))

def log_sum_exp(log_vals):
    if not log_vals:
        return mp.ninf
    L = max(log_vals)
    if L == mp.ninf:
        return mp.ninf
    s = mp.mpf('0')
    for lv in log_vals:
        s += mp.e**(lv - L)
    return L + mp.log(s)

# ====== Hàm p_value ổn định ======

def p_value(A_obs, list_active_set, list_zk, etaj, z_obs, cov_matrix):
    start = time.time()
    
    # 1) Tính sigma
    tn_sigma = (np.sqrt((etaj.T @ cov_matrix @ etaj))).item()
    print("tn_sigma:", tn_sigma)
    if tn_sigma < 1e-9:
        return None

    # 2) Lấy các khoảng cắt cụt
    z_intervals = []
    for i, active_set in enumerate(list_active_set):
        if set(A_obs) == set(active_set):
            z_intervals.append((list_zk[i], list_zk[i+1]))

    print("z_intervals:", z_intervals)
    if not z_intervals:
        return None

    tn_mu = 0

    # 3) Chuẩn hóa z và các khoảng
    z_std = (z_obs - tn_mu) / tn_sigma
    list_intervals_std = [((al - tn_mu)/tn_sigma, (ar - tn_mu)/tn_sigma) for al, ar in z_intervals]

    # 4) Tính logN và logD
    log_masses = []
    log_parts = []
    lz = log_survival(z_std)

    for (a_std, b_std) in list_intervals_std:
        la = log_survival(a_std)
        lb = log_survival(b_std)

        # toàn bộ khối lượng khoảng
        log_mk = log_interval_mass(la, lb)
        log_masses.append(log_mk)

        # phần cho numerator
        if z_std <= a_std:
            pass  # không đóng góp
        elif z_std >= b_std:
            log_parts.append(log_mk)
        else:
            log_pk = log_interval_mass(la, lz)
            log_parts.append(log_pk)

    logD = log_sum_exp(log_masses)
    logN = log_sum_exp(log_parts)

    if logD == mp.ninf:
        return None  # denominator = 0

    pivot = mp.e**(logN - logD)
    p_value = 2 * mp.mpf(min(pivot, 1 - pivot))

    print("pivot:", pivot)
    print("p-value:", p_value)
    print("Thời gian tính p value:", time.time() - start)

    return float(p_value)




# =============================================================================
# HÀM CHẠY CHÍNH: QUY TRÌNH PHÂN TÍCH DỮ LIỆU THỰC
# =============================================================================
import numpy as np
import random
from sklearn.preprocessing import RobustScaler
if __name__ == '__main__':
    # --- BƯỚC 0: CHUẨN BỊ DỮ LIỆU ĐẦU VÀO ---
    
    print("Sử dụng dữ liệu X_clean và y_clean...")

            
    X_clean =pd.read_csv("/kaggle/input/x-clean/X_clean.csv")
    y_clean=pd.read_csv("/kaggle/input/y-clean/y_clean.csv")
 

    
    
    print(f"Dữ liệu đầu vào: X_clean.shape={X_clean.shape}, y_clean.shape={y_clean.shape}")

    # --- GIAI ĐOẠN 1: LỰA CHỌN MÔ HÌNH BẰNG LASSO (KHÔNG DÙNG CV) ---
    print("\n--- Giai đoạn 1: Lựa chọn Mô hình bằng Lasso ---")

    # 1.1. Chuẩn hóa X
    print("Đang chuẩn hóa X_clean...")
    scaler = StandardScaler()


    X_columns = X_clean.columns
    X_scaled = scaler.fit_transform(X_clean)
   
    y_scaler = StandardScaler(with_std=False) 
    y_clean = y_scaler.fit_transform(y_clean)
    n, p = X_scaled.shape
    BEST_ALPHA = 10000
    print(f"Sử dụng alpha cố định: {BEST_ALPHA:.4f}")
    lasso_model = Lasso(alpha=BEST_ALPHA/n, fit_intercept=False,max_iter=80000,tol=1e-9).fit(X_scaled, y_clean)
    A_obs, XA_obs = construct_A_XA_Ac_XAc_bhA(X_scaled, lasso_model.coef_)

    if not A_obs:
        print("Lasso không chọn đặc trưng nào. Kết thúc.")
        exit()

    print(f"\nLasso đã chọn {len(A_obs)} đặc trưng.")
    print("Các đặc trưng được chọn:", X_columns[A_obs].tolist())


    print("Ước lượng sigma^2 từ phần dư của Lasso với bậc tự do điều chỉnh (phương pháp Reid et al.)...")
    
    # 1. Lấy y_pred trực tiếp từ mô hình Lasso đã fit
    y_pred_lasso = lasso_model.predict(X_scaled)
    
    # 2. Tính phần dư (residuals)
    # Đảm bảo cả hai đều là mảng 1D để trừ
    if isinstance(y_clean, pd.Series) or isinstance(y_clean, pd.DataFrame):
        y_true_flat = y_clean.values.flatten()
    else:
        y_true_flat = y_clean.flatten()
        
    residuals_lasso = y_true_flat - y_pred_lasso.flatten()



    # 3. Tính tổng bình phương phần dư (RSS)
    rss_lasso = np.sum(residuals_lasso**2)
    
    # 4. Xác định bậc tự do hiệu dụng (effective degrees of freedom)
    # Theo lý thuyết, bậc tự do hiệu dụng của Lasso chính là số lượng biến khác 0
    df_residuals = n - len(A_obs) 
    if df_residuals <= 0:
        raise ValueError(f"Không thể ước lượng sigma^2 vì bậc tự do (n - |A_obs|) = ({n} - {len(A_obs)}) <= 0.")
    
    # 5. Tính ước lượng phương sai nhiễu
    sigma_squared_hat = rss_lasso / df_residuals
    print(f"Ước lượng phương sai nhiễu (σ^2) : {sigma_squared_hat:.4f}")

    


        

    # --- GIAI ĐOẠN 2: SUY LUẬN CHỌN LỌC ---
    print(f"\n--- Giai đoạn 2: Bắt đầu Suy luận Chọn lọc cho {len(A_obs)} đặc trưng ---")
    p_values_dict = {}
    
    y_col_vector = y_clean.reshape(-1, 1)

    for i, j_selected in enumerate(A_obs):
        if (i==0):
        
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
            if (abs(z_obs)>threshold):
                print("Vì z_obs>20*sigma_z nên threshold=z_obs*1.5 ")
                threshold=abs(z_obs)*1.5
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


# def generate_data(n, p, true_beta, noise_std=1):
#     """Hàm tạo dữ liệu mô phỏng."""
#     print(f"Đang tạo dữ liệu với n={n}, p={p}...")
#     X = np.random.normal(loc=0, scale=1, size=(n, p))
#     true_beta_reshaped = np.reshape(true_beta, (p, 1))
#     noise = np.random.normal(loc=0, scale=noise_std, size=(n, 1))
#     y = X @ true_beta_reshaped + noise
#     print("Tạo dữ liệu xong.")
#     return X, y

# if __name__ == '__main__':
#     # --- BƯỚC 0: CHUẨN BỊ DỮ LIỆU MÔ PHỎNG ---
    
#     # 0.1. Thiết lập các tham số cho mô phỏng
#     n = 170000
#     p = 434
#     noise_std = 1
#     num_true_features = 25 # Số lượng đặc trưng thực sự có ảnh hưởng
    
#     # 0.2. Tạo vector hệ số beta thực
#     true_beta = np.zeros(p)
#     # Chọn ngẫu nhiên các chỉ số cho các đặc trưng có ảnh hưởng
#     true_feature_indices = np.random.choice(p, num_true_features, replace=False)
#     # Gán giá trị cho các hệ số beta thực (ví dụ: 5)
#     true_beta[true_feature_indices] = 1
    
#     print(f"Các đặc trưng có ảnh hưởng thực sự (chỉ số): {sorted(list(true_feature_indices))}")
    
#     # 0.3. Tạo dữ liệu X và y
#     X_sim, y_sim = generate_data(n, p, true_beta, noise_std)
    
#     # 0.4 Tạo tên cột giả để tương thích với mã cũ
#     X_columns = [f'feature_{i}' for i in range(p)]

#     print(f"Dữ liệu mô phỏng: X.shape={X_sim.shape}, y.shape={y_sim.shape}")

#     # --- GIAI ĐOẠN 1: LỰA CHỌN MÔ HÌNH BẰNG LASSO ---
#     print("\n--- Giai đoạn 1: Lựa chọn Mô hình bằng Lasso ---")

#     # 1.1. Chuẩn hóa X và y
#     print("Đang chuẩn hóa X và y...")
#     scaler_X = StandardScaler()
#     X_scaled = X_sim
   
#     scaler_y = StandardScaler(with_std=False) # Chỉ trừ trung bình, không chia cho std dev
#     y_scaled = y_sim
    
#     n, p = X_scaled.shape
    

#     BEST_ALPHA = 6000
  
#     lasso_model = Lasso(alpha=BEST_ALPHA/n, fit_intercept=False, max_iter=80000, tol=1e-9).fit(X_scaled, y_scaled.ravel())
#     A_obs, XA_obs = construct_A_XA_Ac_XAc_bhA(X_scaled, lasso_model.coef_)

#     if not A_obs:
#         print("Lasso không chọn đặc trưng nào. Kết thúc.")
#         exit()

#     print(f"\nLasso đã chọn {len(A_obs)} đặc trưng.")
#     print("Các đặc trưng được chọn (chỉ số):", A_obs)

#     # 1.4. Ước lượng sigma^2
#     print("Ước lượng sigma^2 từ phần dư của Lasso...")
#     y_pred_lasso = lasso_model.predict(X_scaled)
#     residuals_lasso = y_scaled.flatten() - y_pred_lasso.flatten()
#     rss_lasso = np.sum(residuals_lasso**2)
#     df_residuals = n - len(A_obs) 
#     if df_residuals <= 0:
#         raise ValueError(f"Không thể ước lượng sigma^2 vì bậc tự do (n - |A_obs|) = ({n} - {len(A_obs)}) <= 0.")
#     sigma_squared_hat = rss_lasso / df_residuals
#     print(f"Ước lượng phương sai nhiễu (σ^2) : {sigma_squared_hat:.4f} (Giá trị thực tế là {noise_std**2})")

#     # --- GIAI ĐOẠN 2: SUY LUẬN CHỌN LỌC ---
#     print(f"\n--- Giai đoạn 2: Bắt đầu Suy luận Chọn lọc cho {len(A_obs)} đặc trưng ---")
#     p_values_dict = {}
    
#     y_col_vector = y_scaled.reshape(-1, 1)

#     for i, j_selected in enumerate(A_obs):
#         feature_name = X_columns[j_selected]
#         start_time = time.time()
#         print(f"  [{i+1}/{len(A_obs)}] Đang phân tích '{feature_name}' (chỉ số {j_selected})...")
        
#         etaj, etajTy_orig = construct_test_statistic(j_selected, XA_obs, y_col_vector, A_obs)
#         sq_norm_eta = (etaj.T @ etaj).item()
#         if sq_norm_eta < 1e-12: sq_norm_eta = 1e-12
        
#         z_obs = etajTy_orig
#         print(f"    z_obs = {z_obs:.4f}")
        
#         sigma_z = np.sqrt(sq_norm_eta * sigma_squared_hat)
#         threshold = 20 * sigma_z
#         if abs(z_obs) > threshold:
#             print(f"    z_obs ({abs(z_obs):.2f}) vượt ngưỡng 20*sigma_z ({threshold:.2f}). Đặt threshold = |z_obs|*1.5")
#             threshold = abs(z_obs) * 1.5
#         print(f"    Sử dụng threshold = {threshold:.4f}")
        
#         # Sử dụng số core logic của CPU làm số đoạn xử lý
#         n_segments = os.cpu_count() or 4 # Mặc định là 4 nếu không phát hiện được
#         print(f"    Bắt đầu `run_parametric_lasso` với {n_segments} tiến trình...")
        
        
#         list_zk, list_active_set, list_bhz = run_parametric_lasso(
#             X_scaled, y_col_vector, BEST_ALPHA, etaj,
#             threshold=threshold, sigma_sq=sigma_squared_hat,
#             n_segments=n_segments
#         )
    
#         print(f"    `run_parametric_lasso` hoàn tất. Bắt đầu tính p-value...")
#         cov_matrix = np.identity(n) * sigma_squared_hat
#         p_val = p_value(A_obs, list_active_set, list_zk, etaj, z_obs, cov_matrix)
        
#         p_values_dict[feature_name] = p_val
#         elapsed_time = time.time() - start_time
#         print(f"    -> p-value = {p_val if p_val is not None else 'N/A'} (Thời gian: {elapsed_time:.2f}s)")
        
#         # LƯU Ý: Lệnh break này sẽ dừng vòng lặp sau khi phân tích xong đặc trưng đầu tiên.
#         # Xóa nó đi nếu bạn muốn phân tích TẤT CẢ các đặc trưng Lasso đã chọn.
#         break
        
#     # --- GIAI ĐOẠN 3: PHÂN TÍCH KẾT QUẢ ---
#     print("\n--- Giai đoạn 3: Phân tích Kết quả Cuối cùng ---")
    
#     # Lấy các hệ số Lasso cho các đặc trưng được chọn
#     lasso_coef_df = pd.DataFrame({
#         'feature': [X_columns[i] for i in A_obs],
#         'lasso_coefficient': lasso_model.coef_[A_obs]
#     })
    
#     # Tạo DataFrame kết quả từ p-values
#     results_df = pd.DataFrame.from_dict(p_values_dict, orient='index', columns=['raw_p_value']).dropna()
#     results_df.index.name = 'feature'
    
#     if not results_df.empty:
#         # Hiệu chỉnh p-value cho đa kiểm định
#         reject, p_adjusted, _, _ = multipletests(results_df['raw_p_value'], alpha=0.05, method='fdr_bh')
#         results_df['adjusted_p_value'] = p_adjusted
#         results_df['is_significant'] = reject
        
#         # Kết hợp với DataFrame hệ số Lasso
#         results_df = pd.merge(results_df.reset_index(), lasso_coef_df, on='feature').set_index('feature')
        
#         # Thêm cột cho biết đây có phải là đặc trưng "thật" không
#         true_feature_names = {X_columns[i] for i in true_feature_indices}
#         results_df['is_true_feature'] = results_df.index.isin(true_feature_names)
        
#         # Sắp xếp lại
#         results_df = results_df.sort_values('adjusted_p_value')

#         print("\nBảng kết quả Suy luận Chọn lọc:")
#         print(results_df[['lasso_coefficient', 'raw_p_value', 'adjusted_p_value', 'is_significant', 'is_true_feature']])
#     else:
#         print("Không có p-value nào được tính toán.")

