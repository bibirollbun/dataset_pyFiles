from tqdm import tqdm
from itertools import product

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer, KNNImputer
import warnings

from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
warnings.filterwarnings("ignore")

from itertools import product

hours = range(0, 6, 1)
minutes = range(0, 60, 5)

target_col = "bg+1-00"
group_col = "p_num"
date_col = "time"

bg_cols = [f"bg-{i}-{j:02d}" for i, j in product(hours, minutes)]
insu_cols = [f"insulin-{i}-{j:02d}" for i, j in product(hours, minutes)]
carb_cols = [f"carbs-{i}-{j:02d}" for i, j in product(hours, minutes)]
hr_cols = [f"hr-{i}-{j:02d}" for i, j in product(hours, minutes)]
step_cols = [f"steps-{i}-{j:02d}" for i, j in product(hours, minutes)]
cals_cols = [f"cals-{i}-{j:02d}" for i, j in product(hours, minutes)]

feature_cols = bg_cols + insu_cols + carb_cols + hr_cols + step_cols + cals_cols





df_train = pd.read_csv(
    '/kaggle/input/brist1d/train.csv',
    index_col='id',
    parse_dates=['time'],
)

df_train.columns = df_train.columns.str.replace(':', '-')


import random
import os
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


df_train.columns = df_train.columns.str.replace(":", "-")


for colset in [bg_cols, insu_cols, carb_cols, hr_cols, step_cols, cals_cols]:
    df_train[colset] = (
        df_train[colset]
        .interpolate(axis=1)
        .fillna(method="bfill", axis=1)
        .fillna(method="ffill", axis=1)
    )

feature_columns = {
    'bg': bg_cols,
    'hr': hr_cols,
    'steps': step_cols
}

def impute_time_series_features(df, columns, method='median', window=5):
    for col in columns:
        df[col] = df[col].fillna(df[col].rolling(window=window, min_periods=1).median())

        df[col] = df[col].fillna(method='ffill')
        df[col] = df[col].fillna(method='bfill')

        df[col] = df.groupby('p_num')[col].transform(lambda x: x.fillna(x.median()))
    return df

for feature, columns in feature_columns.items():
    df_train = impute_time_series_features(df_train, columns, method='median')

all_imputed_columns = bg_cols + hr_cols + step_cols
print("Remaining null values in training set:", df_train[all_imputed_columns].isnull().sum().sum())



feature_columns_insulin_cals = {
    'insulin': insu_cols,
    'cals': cals_cols
}

def impute_non_temporal_features(df, columns, method='median', window=5):
    for col in columns:
        df[col] = df.groupby('p_num')[col].transform(lambda x: x.fillna(x.median()))

        df[col] = df[col].fillna(df[col].rolling(window=window, min_periods=1).median())
    return df

for feature, columns in feature_columns_insulin_cals.items():
    df_train = impute_non_temporal_features(df_train, columns, method='median')

all_imputed_columns_insulin_cals = insu_cols + cals_cols
print("Remaining null values in training set:", df_train[all_imputed_columns_insulin_cals].isnull().sum().sum())



feature_columns_insulin_cals = {
    'insulin': insu_cols,
    'cals': cals_cols
}

def impute_non_temporal_features(df, columns, method='median', window=5):
    for col in columns:
        df[col] = df.groupby('p_num')[col].transform(lambda x: x.fillna(x.median()))

        df[col] = df[col].fillna(df[col].rolling(window=window, min_periods=1).median())
    return df

for feature, columns in feature_columns_insulin_cals.items():
    df_train = impute_non_temporal_features(df_train, columns, method='median')

all_imputed_columns_insulin_cals = insu_cols + cals_cols
print("Remaining null values in training set:", df_train[all_imputed_columns_insulin_cals].isnull().sum().sum())



activity_cols = [col for col in df_train.columns if col.startswith('activity')]
dominant_activity_train = df_train[activity_cols].mode(axis=1)[0]
df_train['dominant_activity'] = dominant_activity_train
print(df_train['dominant_activity'].value_counts())


activity_replace = {
    'Walk': 'Walking',
    'Run': 'Running',
    'Swim': 'Swimming',
}
df_train['dominant_activity'] = df_train['dominant_activity'].replace(activity_replace)
print(df_train['dominant_activity'])


df_train['dominant_activity'].fillna('No Activity', inplace=True)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

combined_activities = pd.concat([
    df_train['dominant_activity']
])

label_encoder.fit(combined_activities)

df_train['dominant_activity_encoded'] = label_encoder.transform(df_train['dominant_activity'])

df_train = df_train.drop(['dominant_activity'], axis = 1)

feature_cols = feature_cols + ['dominant_activity_encoded']


imputer = SimpleImputer()

df_train[feature_cols] = imputer.fit_transform(df_train[feature_cols])

df_train['time_hour'] = pd.to_datetime(df_train['time']).dt.hour

feature_cols.extend(["time_hour"])

df_train_final = df_train[feature_cols]

y_target = df_train[[target_col]]


np.isnan(df_train_final).sum()



df_train_final


y_target


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

mp.dps = 50

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

    sq_norm = sigma_squared*(etaj.T @ etaj)
    if sq_norm < 1e-12: sq_norm = 1e-12

    # --- PHẦN TỐI ƯU HÓA ---
    # Thay vì tạo ma trận proj_matrix (n x n), chúng ta tính toán trực tiếp.
    # a = (I - (ηηᵀ)/||η||²)y = y - η(ηᵀy)/||η||²
    
    # 1. Tính tích trong (inner product) trước -> ra một số vô hướng
    # etajTy_scalar = (etaj.T @ y)[0, 0]
    etajTy_scalar = (etaj.T @ y)
    
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

# SỬA ĐỔI 2: Hàm run_... giờ nhận sigma_squared
# def run_parametric_lasso(X, y, alpha, etaj, threshold, sigma_squared):
#     n, p = X.shape
#     zk = -threshold
#     list_zk = [zk]
#     list_active_set = []
#     list_bhz = []
    
#     while zk < threshold:
#         yz, b = compute_yz(y, etaj, zk, n,sigma_squared)
#         V_l,V_r, Akz, bhkz = parametric_lasso(X, yz, alpha, b, n, p,etaj)
#         zk = V_r + 0.0001


#         if zk <= threshold:
#             list_zk.append(zk-0.0001)
#         else:
#             list_zk.append(threshold)
#         list_active_set.append(Akz)
#         list_bhz.append(bhkz)

#     print("DOne6")
#     return list_zk, list_bhz, list_active_set










# ============================
# Hàm tạo/attach shared memory
# ============================

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

# def process_segment(z_start, z_end,
#                     X_shm_name, X_shape, X_dtype,
#                     y_shm_name, y_shape, y_dtype,
#                     etaj_shm_name, etaj_shape, etaj_dtype,
#                     alpha, sigma_sq, small_eps=1e-6, max_iter=2000,
#                     lasso_tol=1e-10, lasso_max_iter=20000):
#     # Attach shared arrays
#     shmX, X = attach_shared_array(X_shm_name, X_shape, X_dtype)
#     shmy, y = attach_shared_array(y_shm_name, y_shape, y_dtype)
#     shmetaj, etaj = attach_shared_array(etaj_shm_name, etaj_shape, etaj_dtype)

#     n = X.shape[0]
#     results = []
#     clf = Lasso(alpha=alpha / n, fit_intercept=False, tol=lasso_tol, warm_start=True, max_iter=lasso_max_iter)

#     zk = float(z_start)
#     it = 0
#     while zk < z_end and it < max_iter:
#         it += 1
#         yz, b_vec = compute_yz(y, etaj, zk, n, sigma_sq)
#         yz_flat = yz.flatten()

#         clf.fit(X, yz_flat)
#         bhz = clf.coef_.copy()

#         A, XA= construct_A_XA_Ac_XAc_bhA(X, bhz)

#         if len(A) == 0:
#             V_l, V_r = zk, z_end
#         else:
#             s = np.sign(bhz[A])
#             A1, b1 = compute_polyhedron(X, A, s, alpha)
#             try:
#                 V_l, V_r = compute_truncation_bounds(A1, b1, etaj, yz_flat)
#             except Exception:
#                 V_l, V_r = zk, zk + small_eps

    #     results.append((zk, A, bhz))

    #     if not np.isfinite(V_r) or V_r <= zk + 1e-12:
    #         zk = zk + small_eps
    #     else:
    #         zk = min(V_r + small_eps, z_end)

    # shmX.close()
    # shmy.close()
    # shmetaj.close()

    # return results



def process_segment(z_start, z_end,
                    X_shm_name, X_shape, X_dtype,
                    y_shm_name, y_shape, y_dtype,
                    etaj_shm_name, etaj_shape, etaj_dtype,
                    alpha, sigma_sq, small_eps=0.001, max_iter=2000,
                    lasso_tol=1e-4, lasso_max_iter=20000):
    
    # --- SỬA ĐỔI 1: Thiết lập logger và log khi bắt đầu ---
    worker_id = os.getpid()
    logger = setup_worker_logging(worker_id)
    logger.info(f"BẮT ĐẦU xử lý đoạn [{z_start:.4f}, {z_end:.4f}]")
    
    start_time = time.time()

    # Attach shared arrays
    shmX, X = attach_shared_array(X_shm_name, X_shape, X_dtype)
    shmy, y = attach_shared_array(y_shm_name, y_shape, y_dtype)
    shmetaj, etaj = attach_shared_array(etaj_shm_name, etaj_shape, etaj_dtype)

    n = X.shape[0]
    results = []
    clf = Lasso(alpha=alpha / n, fit_intercept=False, tol=lasso_tol, warm_start=True, max_iter=lasso_max_iter)

    zk = float(z_start)
    it = 0
    # --- SỬA ĐỔI 2: Tạo biến để kiểm soát tần suất log, tránh làm ngập màn hình ---
    log_interval = 20  # Chỉ in log sau mỗi 250 vòng lặp

    while zk < z_end and it < max_iter:
        it += 1
        
        # --- SỬA ĐỔI 3: Log "nhịp tim" của worker ---
        # Điều này cho bạn biết worker có bị "treo" hay không.
        if it == 1 or it % log_interval == 0:
            logger.info(f"Vòng lặp #{it}... zk hiện tại = {zk:.6f}")

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
            try:
                V_l, V_r = compute_truncation_bounds(A1, b1, etaj, yz_flat)
            except Exception as e:
                logger.error(f"Lỗi tại vòng lặp {it}, zk={zk:.4f}: {e}. Dùng bước nhảy nhỏ.")
                V_l, V_r = zk, zk + small_eps

        results.append((zk, A, bhz))
        
        # --- SỬA ĐỔI 4 (Tùy chọn): Log kết quả của mỗi bước nhảy ---
        # Bỏ comment dòng dưới nếu bạn muốn xem chi tiết tuyệt đối, nhưng nó sẽ tạo ra RẤT NHIỀU log.
        # logger.debug(f"Loop {it}: zk={zk:.4f} -> new V_r={V_r:.4f}. Active set size: {len(A)}")
        # Lưu ý: Phải đổi logger.setLevel(logging.DEBUG) ở trên để thấy log này.

        if not np.isfinite(V_r) or V_r <= zk + 1e-12:
            zk = zk + small_eps
        else:
            zk = min(V_r + small_eps, z_end)

    shmX.close()
    shmy.close()
    shmetaj.close()
    
    # --- SỬA ĐỔI 5: Log khi kết thúc và báo cáo tổng kết ---
    end_time = time.time()
    duration = end_time - start_time
    
    if it >= max_iter:
        logger.warning(f"ĐÃ ĐẠT GIỚI HẠN {max_iter} vòng lặp. Worker có thể chưa xử lý xong.")

    logger.info(f"KẾT THÚC xử lý đoạn [{z_start:.4f}, {z_end:.4f}]. "
                f"Tổng cộng: {it} vòng lặp. Thời gian: {duration:.2f} giây.")

    return results




# ============================
# Hàm song song hóa chính
# ============================

def run_parametric_lasso(X, y, alpha, etaj, threshold, sigma_sq,
                                 n_segments=5, overlap=1e-8, **worker_kwargs):
    shmX, X_shape, X_dtype = create_shared_array(X)
    shmy, y_shape, y_dtype = create_shared_array(y.reshape(-1))
    shmetaj, etaj_shape, etaj_dtype = create_shared_array(etaj.reshape(-1))

    edges = np.linspace(-threshold, threshold, n_segments + 1)
    segments = []
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
    tol = 1e-8
    for zk, A, bhz in all_entries:
        if merged_zk and abs(zk - merged_zk[-1]) < tol:
            continue
        merged_zk.append(float(zk))
        merged_active_sets.append(A)
        merged_bhz.append(bhz)

    if len(merged_zk) == 0 or merged_zk[-1] < threshold - 1e-12:
        merged_zk.append(float(threshold))

    return merged_zk, merged_active_sets, merged_bhz























# SỬA ĐỔI 3: Hàm p_value nhận z_obs và đã sửa lỗi logic
def p_value(A_obs, list_active_set, list_zk, etaj, z_obs, cov_matrix):
    tn_sigma = np.sqrt((etaj.T @ cov_matrix @ etaj))[0, 0]
    if tn_sigma < 1e-9: return None
    
    z_intervals = []
    for i, active_set in enumerate(list_active_set):
        if set(A_obs) == set(active_set):
            z_intervals.append([list_zk[i], list_zk[i+1]])

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
            
    if denominator < 1e-100: return None
    
    pivot_val = float(numerator / denominator)
    print("Done7")
    return 2 * min(pivot_val, 1 - pivot_val)

# =============================================================================
# HÀM CHẠY CHÍNH: QUY TRÌNH PHÂN TÍCH DỮ LIỆU THỰC
# =============================================================================

if __name__ == '__main__':
    # --- BƯỚC 0: CHUẨN BỊ DỮ LIỆU ĐẦU VÀO ---
    
    print("Sử dụng dữ liệu X_clean và y_clean...")
   
    X_clean =df_train_final
    y_clean =y_target
    
    print(f"Dữ liệu đầu vào: X_clean.shape={X_clean.shape}, y_clean.shape={y_clean.shape}")

    # --- GIAI ĐOẠN 1: LỰA CHỌN MÔ HÌNH BẰNG LASSO (KHÔNG DÙNG CV) ---
    print("\n--- Giai đoạn 1: Lựa chọn Mô hình bằng Lasso ---")

    # 1.1. Chuẩn hóa X
    print("Đang chuẩn hóa X_clean...")
    scaler = StandardScaler()
    X_columns = X_clean.columns
    X_scaled = scaler.fit_transform(X_clean)
    n, p = X_scaled.shape
    BEST_ALPHA = 6000
    print(f"Sử dụng alpha cố định: {BEST_ALPHA:.4f}")
    lasso_model = Lasso(alpha=BEST_ALPHA/n).fit(X_scaled, y_clean)
    A_obs, XA_obs = construct_A_XA_Ac_XAc_bhA(X_scaled, lasso_model.coef_)

    if not A_obs:
        print("Lasso không chọn đặc trưng nào. Kết thúc.")
        exit()

    print(f"\nLasso đã chọn {len(A_obs)} đặc trưng.")
    print("Các đặc trưng được chọn:", X_columns[A_obs].tolist())

    # SỬA ĐỔI 4: Ước lượng phương sai nhiễu (σ²)

    
    y_pred = lasso_model.predict(X_scaled)
    df_residuals = n - len(A_obs)
    if df_residuals <= 0:
        raise ValueError("Không thể ước lượng sigma^2.")
    # sigma_squared_hat = np.sum((y_clean.values - y_pred)**2) / df_residuals
    sigma_squared_hat = 1
    print(f"Ước lượng phương sai nhiễu (σ^2): {sigma_squared_hat:.4f}")

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
        
        # Tối ưu hóa việc tính a_vec, tránh tạo ma trận n x n
        # a_vec = y_col_vector - (etaj @ etaj.T @ y_col_vector) / sq_norm_eta # Cách không hiệu quả
        etajTy_scalar = (etaj.T @ y_col_vector)
        a_vec = y_col_vector - etaj * (etajTy_scalar / sq_norm_eta) # Cách hiệu quả
        z_obs = etajTy_orig      
        sigma_z = np.sqrt(sq_norm_eta * sigma_squared_hat)
        # threshold = 20 * sigma_z
        threshold = 20 * 1
        
        # list_zk, _, list_active_set = run_parametric_lasso(
        #     X_scaled, y_col_vector, BEST_ALPHA, etaj, threshold, sigma_squared_hat
        # )

        
        n_segments = os.cpu_count()  # số core logic của CPU
        
        list_zk, list_active_set, list_bhz = run_parametric_lasso(
            X_scaled, y_col_vector, BEST_ALPHA, etaj,
            threshold=20.0, sigma_sq=1.0,
            n_segments=n_segments,  # CHỈ THÊM DÒNG NÀY
            small_eps=0.001, max_iter=2000
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

