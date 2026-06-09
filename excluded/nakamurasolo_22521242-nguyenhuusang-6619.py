# # ==============================================================================
# # BƯỚC 0: KHỞI TẠO - TẢI THƯ VIỆN VÀ DỮ LIỆU
# # ==============================================================================
# import pandas as pd
# import numpy as np
# import lightgbm as lgb
# import optuna
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import f1_score
# import warnings

# # Tắt các cảnh báo không cần thiết để output gọn gàng hơn
# warnings.filterwarnings('ignore')

# print("BƯỚC 0: Đang tải dữ liệu...")
# # Tải tất cả các file dữ liệu
# try:
#     delay_4_6 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv')
#     delay_7_9 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv')
#     not_delay_4_6 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/not_delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv')
#     not_delay_7_9 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/not_delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv')
#     test_df = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/PILOT_10.csv')
#     test_ids = test_df['ID']
# except FileNotFoundError as e:
#     print(f"LỖI: Không tìm thấy file csv. Vui lòng kiểm tra lại đường dẫn.")
#     print(e)
#     exit()

# # Gộp dữ liệu huấn luyện và tạo biến mục tiêu
# delay_4_6['is_late'] = 1
# delay_7_9['is_late'] = 1
# not_delay_4_6['is_late'] = 0
# not_delay_7_9['is_late'] = 0
# train_df = pd.concat([delay_4_6, delay_7_9, not_delay_4_6, not_delay_7_9], ignore_index=True)

# # Hàm đổi tên cột trùng lặp
# def rename_duplicate_columns(df):
#     cols, count = [], {}
#     for column in df.columns:
#         clean_column = column.replace(' ', '_')
#         if clean_column in count:
#             count[clean_column] += 1
#             cols.append(f"{clean_column}_{count[clean_column]}")
#         else:
#             count[clean_column] = 1
#             cols.append(clean_column)
#     df.columns = cols
#     return df

# train_df = rename_duplicate_columns(train_df)
# test_df = rename_duplicate_columns(test_df)
# print("BƯỚC 0: Tải dữ liệu và chuẩn bị ban đầu hoàn tất.\n")

# # ==============================================================================
# # BƯỚC 1: KỸ THUẬT ĐẶC TRƯNG (FEATURE ENGINEERING)
# # ==============================================================================
# print("BƯỚC 1: Bắt đầu quá trình Feature Engineering...")

# def feature_engineer(df):
#     """Hàm tạo ra các đặc trưng mới và giàu thông tin."""
    
#     # 1. Chuyển đổi và tạo đặc trưng từ ngày tháng
#     for col in ['Order_date', 'VSD']:
#         df[col] = pd.to_datetime(df[col], errors='coerce')
    
#     df['planned_lead_time'] = (df['VSD'] - df['Order_date']).dt.days
#     df['order_day_of_week'] = df['Order_date'].dt.dayofweek
#     df['order_month'] = df['Order_date'].dt.month
#     df['order_day_of_year'] = df['Order_date'].dt.dayofyear
#     # Tính tuần trong năm, có thể tạo ra NaN nếu ngày tháng là NaT
#     week_of_year = df['Order_date'].dt.isocalendar().week

#     # Điền giá trị thay thế (-1) cho các NaN rồi mới chuyển sang kiểu số nguyên
#     df['order_week_of_year'] = week_of_year.fillna(0).astype(int)

#     # 2. Tạo đặc trưng tương tác (Interaction Features)
#     # Kết hợp các mã để tạo ra các thực thể duy nhất
#     df['cust_supplier'] = df['CUST_CD'].astype(str) + '_' + df['SUPPLIER_CD'].astype(str)
#     df['cust_brand'] = df['CUST_CD'].astype(str) + '_' + df['BRAND_CD'].astype(str)
#     df['supplier_brand'] = df['SUPPLIER_CD'].astype(str) + '_' + df['BRAND_CD'].astype(str)

#     # 3. Tạo đặc trưng tổng hợp (Aggregation Features)
#     # Tính toán các giá trị thống kê cho các nhóm để nắm bắt hành vi
#     agg_features = {
#         'planned_lead_time': ['mean', 'std', 'max'],
#         'SO_QTY': ['mean', 'std', 'sum'],
#         'PURCHASE_AMOUNT': ['mean', 'sum']
#     }
    
#     # Nhóm theo nhà cung cấp
#     supplier_agg = df.groupby('SUPPLIER_CD').agg(agg_features)
#     supplier_agg.columns = ['supplier_' + '_'.join(col).strip() for col in supplier_agg.columns.values]
#     df = df.merge(supplier_agg, on='SUPPLIER_CD', how='left')
    
#     # Nhóm theo khách hàng
#     cust_agg = df.groupby('CUST_CD').agg(agg_features)
#     cust_agg.columns = ['cust_' + '_'.join(col).strip() for col in cust_agg.columns.values]
#     df = df.merge(cust_agg, on='CUST_CD', how='left')

#     return df

# # --- Áp dụng theo phương pháp "Gộp - Xử lý - Tách" ---
# # 1. Lưu lại biến mục tiêu và gộp train/test
# y = train_df['is_late']
# combined_df = pd.concat([train_df.drop(columns=['is_late']), test_df], ignore_index=True)

# # 2. Áp dụng hàm feature_engineer
# combined_featured = feature_engineer(combined_df)
# print("BƯỚC 1: Feature Engineering hoàn tất.\n")


# # ==============================================================================
# # BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU (PREPROCESSING)
# # ==============================================================================
# print("BƯỚC 2: Bắt đầu Preprocessing...")

# def preprocess(df):
#     """Hàm dọn dẹp, xử lý giá trị thiếu và mã hóa."""
    
#     # 1. Loại bỏ các cột không cần thiết
#     cols_to_drop = [
#         'Order_date', 'VSD', 'GLOBAL_NO', 'label', 'ID',
#         'SOUF_RCV_NO', 'QTUF_RCV_NO', 'PRODUCT_ATTRIBUTION', 'Stock_class'
#     ]
#     existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
#     df = df.drop(columns=existing_cols_to_drop)

#     # 2. Điền giá trị thiếu (Impute Missing Values)
#     # Dùng median cho cột số, 'Unknown' cho cột object
#     for col in df.select_dtypes(include=np.number).columns:
#         df[col] = df[col].fillna(df[col].median())
#     for col in df.select_dtypes(include=['object', 'category']).columns:
#         df[col] = df[col].fillna('Unknown')
        
#     # 3. Mã hóa các cột phân loại
#     # Chuyển thành kiểu 'category' để LightGBM xử lý hiệu quả
#     for col in df.select_dtypes(include=['object']).columns:
#         df[col] = df[col].astype('category')
        
#     return df

# # Áp dụng hàm preprocess
# combined_processed = preprocess(combined_featured)

# # Tách lại thành train và test
# train_processed = combined_processed.iloc[:len(train_df)]
# test_processed = combined_processed.iloc[len(train_df):]

# print("BƯỚC 2: Preprocessing hoàn tất.")
# print(f"Số đặc trưng cuối cùng: {len(train_processed.columns)}\n")



# # ==============================================================================
# # BƯỚC 3: TỐI ƯU HÓA SIÊU THAM SỐ VỚI OPTUNA
# # ==============================================================================
# print("BƯỚC 3: Bắt đầu tối ưu hóa siêu tham số (Hyperparameter Tuning)...")

# def objective(trial):
#     """Hàm mục tiêu cho Optuna để tối đa hóa Macro F1-Score."""
#     params = {
#         'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1,
#         'boosting_type': 'gbdt',
#         'scale_pos_weight': 40.0, # Tham số quan trọng nhất
#         'n_estimators': trial.suggest_int('n_estimators', 500, 2500),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
#         'num_leaves': trial.suggest_int('num_leaves', 20, 300),
#         'max_depth': trial.suggest_int('max_depth', 5, 15),
#         'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
#     }
    
#     kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     f1_scores = []
    
#     for train_idx, val_idx in kf.split(train_processed, y):
#         X_train, X_val = train_processed.iloc[train_idx], train_processed.iloc[val_idx]
#         y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
#         model = lgb.LGBMClassifier(**params)
#         model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
#                   eval_metric='logloss', callbacks=[lgb.early_stopping(100, verbose=False)])
        
#         preds = model.predict(X_val)
#         f1_scores.append(f1_score(y_val, preds, average='macro'))
        
#     return np.mean(f1_scores)

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=15) 

# print("BƯỚC 3: Tối ưu hóa hoàn tất.")
# print(f"Điểm Macro F1-Score tốt nhất (CV): {study.best_value:.5f}")
# print("Bộ tham số tốt nhất:", study.best_params)
# print("\n")



# study.optimize(objective, n_trials=5)


# # ==============================================================================
# # BƯỚC 4: TÌM NGƯỠNG TỐI ƯU VÀ HUẤN LUYỆN MÔ HÌNH CUỐI CÙNG
# # ==============================================================================
# print("BƯỚC 4: Tìm ngưỡng tối ưu và huấn luyện mô hình cuối cùng...")

# # Lấy bộ tham số tốt nhất và thêm các tham số cố định
# best_params = study.best_params
# best_params.update({'objective': 'binary', 'metric': 'binary_logloss', 
#                     'verbosity': -1, 'boosting_type': 'gbdt', 'scale_pos_weight': 40.0})

# # best_params['n_estimators'] = 1752
# # best_params['learning_rate'] = 0.05285619391955323
# # best_params['num_leaves'] = 250
# # best_params['max_depth'] = 11
# # best_params['min_child_samples'] = 11
# # best_params['subsample'] = 0.8950150544982376
# # best_params['colsample_bytree'] = 0.7951485778113276
# # best_params['reg_alpha'] = 0.000511973595955625
# # best_params['reg_lambda'] = 0.04430188400980013

# # Lấy dự đoán Out-of-Fold (OOF) để tìm ngưỡng
# kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# oof_preds_proba = np.zeros(len(train_df))

# for train_idx, val_idx in kf.split(train_processed, y):
#     X_train, X_val = train_processed.iloc[train_idx], train_processed.iloc[val_idx]
#     y_train = y.iloc[train_idx]
    
#     model = lgb.LGBMClassifier(**best_params)
#     model.fit(X_train, y_train)
#     oof_preds_proba[val_idx] = model.predict_proba(X_val)[:, 1]

# # Tìm ngưỡng tốt nhất
# thresholds = np.arange(0.1, 0.9, 0.01)
# f1_scores = [f1_score(y, (oof_preds_proba >= t).astype(int), average='macro') for t in thresholds]
# best_threshold = thresholds[np.argmax(f1_scores)]

# print(f"Ngưỡng xác suất tối ưu tìm được: {best_threshold:.2f}")
# print(f"Macro F1-Score tương ứng trên tập OOF: {max(f1_scores):.5f}")

# # Huấn luyện mô hình cuối cùng trên TOÀN BỘ dữ liệu train
# final_model = lgb.LGBMClassifier(**best_params)
# final_model.fit(train_processed, y)
# print("BƯỚC 4: Huấn luyện mô hình cuối cùng hoàn tất.\n")




# # ==============================================================================
# # BƯỚC 5: DỰ ĐOÁN VÀ TẠO FILE SUBMISSION
# # ==============================================================================
# print("BƯỚC 5: Dự đoán trên tập test và tạo file submission...")

# # Dự đoán xác suất trên tập test
# test_preds_proba = final_model.predict_proba(test_processed)[:, 1]

# # Áp dụng ngưỡng tối ưu để ra quyết định cuối cùng
# test_preds_final = (test_preds_proba >= best_threshold).astype(int)

# # Tạo DataFrame để nộp bài
# submission_df = pd.DataFrame({'ID': test_ids, 'label': test_preds_final})
# submission_df.to_csv('submission_final_new.csv', index=False)

# print("Đã tạo file 'submission_final_new.csv' thành công!")
# print("Xem trước 5 dòng đầu của file submission:")
# print(submission_df.head())
# print("\nPhân phối dự đoán trên tập test:")
# print(submission_df['label'].value_counts(normalize=True))


"""
Final Kaggle pipeline for: Predict product shipping delay (binary) - maximize Macro F1
Combines: feature engineering, LightGBM + CatBoost stacking ensemble, Optuna tuning, OOF threshold search

How to use:
- Place train CSVs and test CSV in the same folder as this script (defaults below).
- Run: python final_solution_pipeline.py
- Outputs: submission_ensemble.csv and oof_predictions.npy

Notes:
- This script is written to be robust to column name differences by normalizing spaces to underscores.
- Tune `N_TRIALS`, `N_FOLDS` for more compute; defaults are conservative for quick runs.
- Requires: pandas, numpy, scikit-learn, lightgbm, catboost, optuna
"""

import os
import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import optuna

# ----------------------------- User Config -----------------------------
TRAIN_FILES = [
    '/kaggle/input/ds-108-p-21-assigment-06/delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv',
    '/kaggle/input/ds-108-p-21-assigment-06/delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv',
    '/kaggle/input/ds-108-p-21-assigment-06/not_delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv',
    '/kaggle/input/ds-108-p-21-assigment-06/not_delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv'
]
TEST_FILE = '/kaggle/input/ds-108-p-21-assigment-06/PILOT_10.csv'  # or 'test.csv' depending on user
SUBMISSION_NAME = 'submission_ensemble.csv'
TARGET_COL = 'label'  # or 'is_late' — script will normalize
ID_COL = 'ID'
N_FOLDS = 5
N_TRIALS = 30  # increase for better hyperparams (100+ recommended)
RANDOM_STATE = 42
VERBOSE = True

# ----------------------------- Utilities -----------------------------

def load_and_concatenate(train_files):
    dfs = []
    for f in train_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Train file not found: {f}")
        df = pd.read_csv(f)
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def normalize_columns(df):
    df = df.copy()
    new_cols = []
    for c in df.columns:
        nc = str(c).strip().replace(' ', '_')
        new_cols.append(nc)
    df.columns = new_cols
    return df


# ----------------------------- Load data -----------------------------
print('Loading data...')
train_df = load_and_concatenate(TRAIN_FILES)
# train_df = normalize_columns(train_df)

def rename_duplicate_columns(df):
    cols, count = [], {}
    for c in df.columns:
        clean = str(c).strip().replace(' ', '_')
        if clean in count:
            count[clean] += 1
            cols.append(f"{clean}_{count[clean]}")
        else:
            count[clean] = 1
            cols.append(clean)
    df.columns = cols
    return df

def reduce_mem(df):
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        col_min, col_max = df[col].min(), df[col].max()
        if str(df[col].dtype).startswith('int'):
            if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        else:
            df[col] = pd.to_numeric(df[col], downcast='float')
    return df


train_df = rename_duplicate_columns(train_df)
train_df = reduce_mem(train_df)


if os.path.exists(TEST_FILE):
    test_df = pd.read_csv(TEST_FILE)
    # test_df = normalize_columns(test_df)
    test_df = rename_duplicate_columns(test_df)
    test_df = reduce_mem(test_df)

    if ID_COL in test_df.columns:
        test_ids = test_df[ID_COL].values
    else:
        test_ids = np.arange(1, len(test_df) + 1)
else:
    raise FileNotFoundError(f"Test file not found: {TEST_FILE}")

# Detect target
if 'label' in train_df.columns:
    train_df[TARGET_COL] = train_df['label']
elif 'is_late' in train_df.columns:
    train_df[TARGET_COL] = train_df['is_late']
elif TARGET_COL not in train_df.columns:
    raise ValueError('Target column not found in training data')

# Drop exact duplicate rows if any
train_df = train_df.drop_duplicates().reset_index(drop=True)

print('Train shape:', train_df.shape, 'Test shape:', test_df.shape)
import gc
gc.collect()




# ----------------------------- Feature Engineering -----------------------------
print('Feature engineering...')

def add_date_features(df):
    for col in ['Order_date', 'VSD']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    if 'Order_date' in df.columns and 'VSD' in df.columns:
        df['planned_lead_time'] = (df['VSD'] - df['Order_date']).dt.days
    if 'Order_date' in df.columns:
        df['order_day_of_week'] = df['Order_date'].dt.dayofweek
        df['order_month'] = df['Order_date'].dt.month
        df['order_day_of_month'] = df['Order_date'].dt.day
        df['order_week_of_year'] = df['Order_date'].dt.isocalendar().week.astype('Int64')
    return df


def create_interactions_and_aggregations(df, group_cols=['SUPPLIER_CD','CUST_CD']):
    # Example interactions
    if 'CUST_CD' in df.columns and 'SUPPLIER_CD' in df.columns:
        df['cust_supplier'] = df['CUST_CD'].astype(str) + '_' + df['SUPPLIER_CD'].astype(str)
    # Aggregations: numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in [TARGET_COL]]
    for g in group_cols:
        if g in df.columns:
            agg = df.groupby(g)[num_cols].agg(['mean','std','sum']).reset_index()
            # flatten columns
            agg.columns = [g] + [f'{g}_{col}_{stat}' for col, stat in agg.columns.tolist()[1:]]
            df = df.merge(agg, on=g, how='left')
    return df


# Apply to combined (train+test) to prevent mismatch
combined = pd.concat([train_df.drop(columns=[TARGET_COL]), test_df], ignore_index=True)
combined = add_date_features(combined)
combined = create_interactions_and_aggregations(combined)

# Back to splits
train = combined.iloc[:len(train_df)].copy()
test = combined.iloc[len(train_df):].copy()
train[TARGET_COL] = train_df[TARGET_COL].values

# ----------------------------- Preprocessing -----------------------------
print('Preprocessing...')

def preprocess(df, fit_encoders=None):
    df = df.copy()
    # Replace obvious missing values
    df = df.replace([' ', ''], np.nan)
    # Numeric impute
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].fillna(df[c].median())
    # Object columns -> fillna 'Unknown'
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    for c in obj_cols:
        df[c] = df[c].fillna('Unknown').astype(str)
    # Label encode high-cardinality categoricals for LGB
    encoders = {} if fit_encoders is None else fit_encoders
    cat_cols = [c for c in obj_cols if df[c].nunique() < 10000]
    for c in cat_cols:
        if fit_encoders is None:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c])
            encoders[c] = le
        else:
            le = encoders.get(c)
            if le is not None:
                # unseen labels -> set to -1
                df[c] = df[c].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    return df, encoders, cat_cols

train_proc, encoders, cat_cols = preprocess(train)
test_proc, _, _ = preprocess(test, fit_encoders=encoders)

# Prepare features list
drop_cols = ['ID'] if 'ID' in train_proc.columns else []
# Ensure target not in features
if TARGET_COL in train_proc.columns:
    drop_cols.append(TARGET_COL)

features = [c for c in train_proc.columns if c not in drop_cols]
print('Number of features:', len(features))

# ----------------------------- CV and Blending Utilities -----------------------------

def get_oof_preds(clf_factory, X, y, X_test, n_folds=5, random_state=42, cat_features_for_cb=None):
    """Return OOF train probs and test probs (mean across folds), and per-fold models if needed"""
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_models = []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        clf = clf_factory()
        if isinstance(clf, CatBoostClassifier) and cat_features_for_cb is not None:
            train_pool = Pool(X_tr, label=y_tr, cat_features=cat_features_for_cb)
            val_pool = Pool(X_val, label=y_val, cat_features=cat_features_for_cb)
            clf.fit(train_pool, eval_set=val_pool, verbose=False)
            oof[val_idx] = clf.predict_proba(X_val)[:,1]
            test_preds += clf.predict_proba(X_test)[:,1] / n_folds
        else:
            clf.fit(X_tr, y_tr)
            oof[val_idx] = clf.predict_proba(X_val)[:,1]
            test_preds += clf.predict_proba(X_test)[:,1] / n_folds
        fold_models.append(clf)
        if VERBOSE:
            print(f'  Fold {fold+1} done')
    return oof, test_preds, fold_models

import gc
gc.collect()



# from lightgbm import early_stopping




# # ----------------------------- LightGBM: Optuna Tuning -----------------------------
# print('Tuning LightGBM with Optuna...')
# X = train_proc[features].reset_index(drop=True)
# y = train_proc[TARGET_COL].reset_index(drop=True)
# X_test_final = test_proc[features].reset_index(drop=True)

# # Prepare categorical feature indices for CatBoost (object columns in original combined)
# cat_features_for_cb = [i for i, c in enumerate(features) if c in cat_cols]

# # ✅ Convert non-numeric columns to hashed integers
# bad_cols = X.select_dtypes(include=['object', 'datetime']).columns
# for c in bad_cols:
#     X[c] = X[c].astype(str).apply(lambda x: hash(x) % (10 ** 8))
#     X_test_final[c] = X_test_final[c].astype(str).apply(lambda x: hash(x) % (10 ** 8))


# def lgb_objective(trial):
#     param = {
#         'objective': 'binary',
#         'metric': 'binary_logloss',
#         'verbosity': -1,
#         'boosting_type': 'gbdt',
#         'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
#         'num_leaves': trial.suggest_int('num_leaves', 31, 512),
#         'max_depth': trial.suggest_int('max_depth', 5, 20),
#         'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
#         'subsample': trial.suggest_float('subsample', 0.4, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
#         'random_state': RANDOM_STATE,
#         'n_jobs': -1,
#     }
#     # scale_pos_weight from class ratio
#     pos = y.sum(); neg = len(y) - pos
#     param['scale_pos_weight'] = max(1.0, neg/pos)

#     kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
#     scores = []
#     for tr_idx, val_idx in kf.split(X, y):
#         X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
#         y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
#         clf = lgb.LGBMClassifier(**param)
#         # clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='logloss', early_stopping_rounds=100, verbose=False)
#         clf.fit(
#             X_tr, y_tr,
#             eval_set=[(X_val, y_val)],
#             eval_metric='logloss',
#             callbacks=[early_stopping(stopping_rounds=100, verbose=False)]
#         )

#         preds = clf.predict(X_val)
#         scores.append(f1_score(y_val, preds, average='macro'))
#     return np.mean(scores)

# study = optuna.create_study(direction='maximize')
# study.optimize(lgb_objective, n_trials=N_TRIALS)
# print('Best LGB params:', study.best_params)

# # Build final tuned lgb factory
# best_lgb_params = study.best_params
# best_lgb_params.update({'objective':'binary','metric':'binary_logloss','random_state':RANDOM_STATE,'n_jobs':-1})

# import gc
# gc.collect()


# def make_lgb():
#     return lgb.LGBMClassifier(**best_lgb_params)

# # ----------------------------- CatBoost (default tuned minimal) -----------------------------
# print('Preparing CatBoost (no heavy tuning to save time)...')

# def make_cat():
#     # Base parameters: allow CatBoost to handle categories
#     params = {
#         'iterations': 1000,
#         'learning_rate': 0.05,
#         'depth': 6,
#         'loss_function': 'Logloss',
#         'verbose': 0,
#         'random_seed': RANDOM_STATE,
#         'early_stopping_rounds': 100,
#         'auto_class_weights': 'Balanced'
#     }
#     return CatBoostClassifier(**params)

# # ----------------------------- Get OOF preds for base models -----------------------------
# print('Generating OOF predictions for base models...')

# oof_lgb, test_lgb, lgb_models = get_oof_preds(make_lgb, X, y, X_test_final, n_folds=N_FOLDS)
# # For CatBoost we need to give original categorical features as indices
# # We'll re-create X_cat where categorical columns are left as original strings
# X_cat_full = train.iloc[:, :][features].copy()
# for c in cat_cols:
#     if c in X_cat_full.columns:
#         X_cat_full[c] = train[c].astype(str)
# # same for test
# X_test_cat_full = test[features].copy()
# for c in cat_cols:
#     if c in X_test_cat_full.columns:
#         X_test_cat_full[c] = test[c].astype(str)

# # For CatBoost factory that uses Pool, wrap factory to include cat feature names

# def make_cat_for_oof():
#     return CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6, loss_function='Logloss', verbose=0, random_seed=RANDOM_STATE, early_stopping_rounds=100, auto_class_weights='Balanced')

# # Custom OOF generator for CatBoost using Pools

# def get_oof_cat(clf_factory, X, y, X_test, n_folds=5, random_state=42, cat_features=None):
#     kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
#     oof = np.zeros(len(X))
#     test_preds = np.zeros(len(X_test))
#     for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
#         X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
#         y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
#         clf = clf_factory()
#         train_pool = Pool(X_tr, label=y_tr, cat_features=cat_features)
#         val_pool = Pool(X_val, label=y_val, cat_features=cat_features)
#         clf.fit(train_pool, eval_set=val_pool, verbose=False)
#         oof[val_idx] = clf.predict_proba(X_val)[:,1]
#         test_preds += clf.predict_proba(X_test)[:,1] / n_folds
#         print(f'  CatBoost Fold {fold+1} done')
#     return oof, test_preds

# # Map cat feature names to indices
# cat_feature_names = [c for c in features if c in cat_cols]
# cat_feature_indices = [features.index(c) for c in cat_feature_names if c in features]

# try:
#     oof_cat, test_cat = get_oof_cat(make_cat_for_oof, X_cat_full[features], y, X_test_cat_full[features], n_folds=N_FOLDS, cat_features=cat_feature_names)
# except Exception as e:
#     print('CatBoost OOF failed:', e)
#     # Fall back to converting categories with LabelEncoder then train
#     oof_cat, test_cat, _ = get_oof_preds(make_cat, X, y, X_test_final, n_folds=N_FOLDS)

# # Store OOFs
# base_oof = np.vstack([oof_lgb, oof_cat]).T
# base_test = np.vstack([test_lgb, test_cat]).T

# # ----------------------------- Stacker (Logistic Regression) -----------------------------
# print('Training stacker...')
# stack_oof = np.zeros_like(oof_lgb)
# stack_test = np.zeros(X_test_final.shape[0])
# kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
# for fold, (tr_idx, val_idx) in enumerate(kf.split(base_oof, y)):
#     X_tr, X_val = base_oof[tr_idx], base_oof[val_idx]
#     y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
#     lr = LogisticRegression(max_iter=2000)
#     lr.fit(X_tr, y_tr)
#     stack_oof[val_idx] = lr.predict_proba(X_val)[:,1]
#     stack_test += lr.predict_proba(base_test)[:,1] / N_FOLDS
#     print(f'  Stacker fold {fold+1} done')

# # Evaluate OOF macro F1 via threshold search
# print('Searching best threshold on OOF...')
# thresholds = np.linspace(0.01, 0.99, 99)
# f1s = [f1_score(y, (stack_oof >= t).astype(int), average='macro') for t in thresholds]
# best_t = thresholds[np.argmax(f1s)]
# print('Best OOF Macro F1:', max(f1s), 'at threshold', best_t)

# # Final test preds
# final_test_preds = (stack_test >= best_t).astype(int)

# # ----------------------------- Save submission -----------------------------
# submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: final_test_preds})
# submission.to_csv(SUBMISSION_NAME, index=False)
# print('Saved', SUBMISSION_NAME)

# # Save OOF arrays for debugging
# np.save('oof_stack.npy', stack_oof)

# print('Done.')



from lightgbm import early_stopping
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import gc

# ----------------------------- LightGBM: Optuna Tuning -----------------------------
print('Tuning LightGBM with Optuna...')
X = train_proc[features].reset_index(drop=True)
y = train_proc[TARGET_COL].reset_index(drop=True)
X_test_final = test_proc[features].reset_index(drop=True)

# Prepare categorical feature indices for CatBoost
cat_features_for_cb = [i for i, c in enumerate(features) if c in cat_cols]

# ✅ Convert non-numeric columns to hashed integers
bad_cols = X.select_dtypes(include=['object', 'datetime']).columns
for c in bad_cols:
    X[c] = X[c].astype(str).apply(lambda x: hash(x) % (10 ** 8))
    X_test_final[c] = X_test_final[c].astype(str).apply(lambda x: hash(x) % (10 ** 8))

def lgb_objective(trial):
    # param = {
    #     'objective': 'binary',
    #     'metric': 'binary_logloss',
    #     'verbosity': -1,
    #     'boosting_type': 'gbdt',
    #     'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
    #     'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
    #     'num_leaves': trial.suggest_int('num_leaves', 31, 512),
    #     'max_depth': trial.suggest_int('max_depth', 5, 20),
    #     'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
    #     'subsample': trial.suggest_float('subsample', 0.4, 1.0),
    #     'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
    #     'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
    #     'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
    #     'random_state': RANDOM_STATE,
    #     'n_jobs': -1,
    #     'device': 'gpu',           # ✅ GPU: use GPU device
    #     'gpu_platform_id': 0,      # ✅ GPU: default platform
    #     'gpu_device_id': 0         # ✅ GPU: default GPU device
    # }
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
        'num_leaves': trial.suggest_int('num_leaves', 31, 256),  # ✅ smaller limit
        'max_depth': trial.suggest_int('max_depth', 4, 12),       # ✅ avoid huge trees
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'min_data_in_bin': 10,          # ✅ ensures stable splits on GPU
        'max_bin': 255,                 # ✅ default safe GPU bin size
        'force_col_wise': True,         # ✅ avoids some GPU histogram issues
    }

    # scale_pos_weight from class ratio
    pos = y.sum(); neg = len(y) - pos
    param['scale_pos_weight'] = max(1.0, neg/pos)

    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for tr_idx, val_idx in kf.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        clf = lgb.LGBMClassifier(**param)
        # clf.fit(
        #     X_tr, y_tr,
        #     eval_set=[(X_val, y_val)],
        #     eval_metric='logloss',
        #     callbacks=[early_stopping(stopping_rounds=100, verbose=False)]
        # )
        try:
            clf.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric='logloss',
                callbacks=[early_stopping(stopping_rounds=100, verbose=False)]
            )
        except lgb.basic.LightGBMError as e:
            print("⚠️ GPU failed, retrying on CPU:", e)
            param['device'] = 'cpu'
            clf = lgb.LGBMClassifier(**param)
            clf.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric='logloss',
                callbacks=[early_stopping(stopping_rounds=100, verbose=False)]
            )

        preds = clf.predict(X_val)
        scores.append(f1_score(y_val, preds, average='macro'))
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(lgb_objective, n_trials=N_TRIALS)
print('Best LGB params:', study.best_params)

# Build final tuned lgb factory
best_lgb_params = study.best_params
best_lgb_params.update({
    'objective': 'binary',
    'metric': 'binary_logloss',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'device': 'gpu',           # ✅ GPU
    'gpu_platform_id': 0,      # ✅ GPU
    'gpu_device_id': 0         # ✅ GPU
})

gc.collect()

def make_lgb():
    return lgb.LGBMClassifier(**best_lgb_params)

# ----------------------------- CatBoost (GPU) -----------------------------
print('Preparing CatBoost (GPU)...')

def make_cat():
    params = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 6,
        'loss_function': 'Logloss',
        'verbose': 0,
        'random_seed': RANDOM_STATE,
        'early_stopping_rounds': 100,
        'auto_class_weights': 'Balanced',
        'task_type': 'GPU',       # ✅ GPU
        'devices': '0'            # ✅ GPU
    }
    return CatBoostClassifier(**params)

# ----------------------------- Get OOF preds for base models -----------------------------
print('Generating OOF predictions for base models...')

oof_lgb, test_lgb, lgb_models = get_oof_preds(make_lgb, X, y, X_test_final, n_folds=N_FOLDS)

X_cat_full = train.iloc[:, :][features].copy()
for c in cat_cols:
    if c in X_cat_full.columns:
        X_cat_full[c] = train[c].astype(str)
X_test_cat_full = test[features].copy()
for c in cat_cols:
    if c in X_test_cat_full.columns:
        X_test_cat_full[c] = test[c].astype(str)

def make_cat_for_oof():
    return CatBoostClassifier(
        iterations=1000, learning_rate=0.05, depth=6,
        loss_function='Logloss', verbose=0, random_seed=RANDOM_STATE,
        early_stopping_rounds=100, auto_class_weights='Balanced',
        task_type='GPU', devices='0'   # ✅ GPU
    )

def get_oof_cat(clf_factory, X, y, X_test, n_folds=5, random_state=42, cat_features=None):
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        clf = clf_factory()
        train_pool = Pool(X_tr, label=y_tr, cat_features=cat_features)
        val_pool = Pool(X_val, label=y_val, cat_features=cat_features)
        clf.fit(train_pool, eval_set=val_pool, verbose=False)
        oof[val_idx] = clf.predict_proba(X_val)[:,1]
        test_preds += clf.predict_proba(X_test)[:,1] / n_folds
        print(f'  CatBoost Fold {fold+1} done')
    return oof, test_preds

cat_feature_names = [c for c in features if c in cat_cols]
cat_feature_indices = [features.index(c) for c in cat_feature_names if c in features]

try:
    oof_cat, test_cat = get_oof_cat(make_cat_for_oof, X_cat_full[features], y, X_test_cat_full[features], n_folds=N_FOLDS, cat_features=cat_feature_names)
except Exception as e:
    print('CatBoost OOF failed:', e)
    oof_cat, test_cat, _ = get_oof_preds(make_cat, X, y, X_test_final, n_folds=N_FOLDS)

# ----------------------------- Stacker (Logistic Regression) -----------------------------
print('Training stacker...')
stack_oof = np.zeros_like(oof_lgb)
stack_test = np.zeros(X_test_final.shape[0])
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
for fold, (tr_idx, val_idx) in enumerate(kf.split(np.vstack([oof_lgb, oof_cat]).T, y)):
    X_tr, X_val = np.vstack([oof_lgb, oof_cat]).T[tr_idx], np.vstack([oof_lgb, oof_cat]).T[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_tr, y_tr)
    stack_oof[val_idx] = lr.predict_proba(X_val)[:,1]
    stack_test += lr.predict_proba(np.vstack([test_lgb, test_cat]).T)[:,1] / N_FOLDS
    print(f'  Stacker fold {fold+1} done')

print('Searching best threshold on OOF...')
thresholds = np.linspace(0.01, 0.99, 99)
f1s = [f1_score(y, (stack_oof >= t).astype(int), average='macro') for t in thresholds]
best_t = thresholds[np.argmax(f1s)]
print('Best OOF Macro F1:', max(f1s), 'at threshold', best_t)

final_test_preds = (stack_test >= best_t).astype(int)

submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: final_test_preds})
submission.to_csv(SUBMISSION_NAME, index=False)
print('Saved', SUBMISSION_NAME)
np.save('oof_stack.npy', stack_oof)
print('Done.')


