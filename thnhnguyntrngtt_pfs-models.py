import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
%matplotlib inline 
import seaborn as sns
from itertools import product
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn import metrics
import lightgbm as lgb

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)


df = pd.read_csv('/kaggle/input/processed-df/processed_df.csv')


columns_to_exclude = ['ID',
                      'item_cnt_month',
                      'item_cnt_month_uncl',
                      'revenue',
                      'purch_cnt_month',
                      'months_since_sh_it_first_s',
                      'months_since_it_first_s',
                      'months_since_sh_first_s', 
                      'avg_price_global', 
#                      'avg_price_mnth_lag1', 
#                      'avg_price_mnth_lag2',
#                      'avg_price_mnth_sh_lag1', 
#                      'avg_price_mnth_sh_lag2'
#                      'avg_price_mnth_to_gl', 
#                      'avg_price_mnth_sh_to_gl', 
#                      'months_from_sh_it_last_s',
#                      'months_from_it_last_s',
                      'lagged_sh_it_mean',
                      'lagged_it_mean',
#                      'it_had_sales_before'
#                      'sh_it_had_sales_before'
                     ]
cat_features = ['month',
#                'year',
                'shop_id',
                'shop_city_encoded',
                'shop_type_encoded',
                'item_category_id',
                'item_category_type_encoded',
                'item_category_subtype_encoded',
                'days_in_m'
               ]


import time
import psutil # Thư viện để kiểm tra tài nguyên hệ thống
import os     # Thư viện để tương tác với hệ điều hành (lấy process ID)
import numpy as np
import pandas as pd
import lightgbm as lgb # Thêm import cho lightgbm

# Giả định các biến df, columns_to_exclude, và cat_features đã được định nghĩa ở các cell trước
# Ví dụ:
# if 'df' not in locals():
#     print("Biến 'df' chưa được định nghĩa. Vui lòng nạp và xử lý dữ liệu trước.")
#     # df = pd.read_csv('your_processed_data.csv') # Hoặc cách bạn có df
# if 'columns_to_exclude' not in locals():
#     columns_to_exclude = [] # Khởi tạo nếu chưa có
# if 'cat_features' not in locals():
#     cat_features = [] # Khởi tạo nếu chưa có


# modeling params
params = {'metric': 'rmse',
          'objective': 'mse', # Hoặc 'regression' cho bài toán hồi quy
          'num_leaves': 255,
          'learning_rate': 0.005,
          'feature_fraction': 0.75,
          'bagging_fraction': 0.75,
          'bagging_freq': 5,
          'force_col_wise' : True,
          'random_state': 10,
          'verbosity': -1 # Giảm bớt log không cần thiết, vì đã có log_evaluation
         }


# Lấy process ID hiện tại của tiến trình Python này
pid = os.getpid()
process = psutil.Process(pid)

# --- Ghi lại thông tin bộ nhớ TRƯỚC KHI tạo Dataset và huấn luyện ---
mem_info_before_dataload = process.memory_info()
rss_before_dataload_MB = mem_info_before_dataload.rss / (1024 * 1024) # Chuyển byte sang MB
print(f"Bộ nhớ RSS của tiến trình trước khi tải dữ liệu vào lgb.Dataset: {rss_before_dataload_MB:.2f} MB")

# Prepare training and validation datasets with categorical features
# Đảm bảo các cột cần thiết tồn tại
if 'date_block_num' not in df.columns or 'item_cnt_month' not in df.columns:
    print("Lỗi: df thiếu cột 'date_block_num' hoặc 'item_cnt_month'.")
    # exit() # Hoặc xử lý lỗi phù hợp

# Đảm bảo columns_to_exclude và cat_features là list và các cột tồn tại
if not isinstance(columns_to_exclude, list):
    columns_to_exclude = list(columns_to_exclude) if hasattr(columns_to_exclude, '__iter__') else []
if not isinstance(cat_features, list):
    cat_features = list(cat_features) if hasattr(cat_features, '__iter__') else []

actual_columns_to_exclude = [col for col in columns_to_exclude if col in df.columns]
actual_cat_features = [col for col in cat_features if col in df.drop(actual_columns_to_exclude, axis=1, errors='ignore').columns]


print("Đang chuẩn bị lgb.Dataset...")
train_data = lgb.Dataset(
    df[(df['date_block_num'] >= 19) & (df['date_block_num'] < 33)].drop(actual_columns_to_exclude, axis=1, errors='ignore'),
    label=df[(df['date_block_num'] >= 19) & (df['date_block_num'] < 33)]['item_cnt_month'],
    categorical_feature=actual_cat_features
)

valid_data = lgb.Dataset(
    df[df['date_block_num'] == 33].drop(actual_columns_to_exclude, axis=1, errors='ignore'),
    label=df[df['date_block_num'] == 33]['item_cnt_month'],
    categorical_feature=actual_cat_features,
    reference=train_data
)
print("Chuẩn bị lgb.Dataset hoàn tất.")

# --- Ghi lại thông tin bộ nhớ TRƯỚC KHI huấn luyện (sau khi tạo Dataset) ---
mem_info_before_train = process.memory_info()
rss_before_train_MB = mem_info_before_train.rss / (1024 * 1024)
print(f"Bộ nhớ RSS của tiến trình trước khi huấn luyện: {rss_before_train_MB:.2f} MB")

lgb_model = None # Khởi tạo lgb_model
print("Bắt đầu huấn luyện mô hình LightGBM...")
start_time = time.time() # Ghi lại thời điểm bắt đầu

# Train model
lgb_model = lgb.train(
    params=params,
    train_set=train_data,
    num_boost_round=1500,
    valid_sets=[train_data, valid_data],
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True), lgb.log_evaluation(100)]
)

end_time = time.time() # Ghi lại thời điểm kết thúc
training_time = end_time - start_time # Tính toán thời gian huấn luyện

# --- Ghi lại thông tin bộ nhớ SAU KHI huấn luyện ---
mem_info_after_train = process.memory_info()
rss_after_train_MB = mem_info_after_train.rss / (1024 * 1024)

print(f"Hoàn tất huấn luyện mô hình.")
print(f"Thời gian huấn luyện: {training_time:.2f} giây")
print(f"Bộ nhớ RSS của tiến trình sau khi huấn luyện: {rss_after_train_MB:.2f} MB")

# Tính toán lượng bộ nhớ sử dụng thêm cho quá trình huấn luyện (và mô hình được tạo ra)
memory_used_for_training_MB = rss_after_train_MB - rss_before_train_MB
print(f"Lượng bộ nhớ RSS sử dụng thêm trong quá trình huấn luyện (và lưu mô hình): {memory_used_for_training_MB:.2f} MB")

# Tính toán tổng lượng bộ nhớ sử dụng thêm từ lúc trước khi load data vào Dataset
total_memory_increase_MB = rss_after_train_MB - rss_before_dataload_MB
print(f"Tổng lượng bộ nhớ RSS tăng thêm (bao gồm tải data và huấn luyện): {total_memory_increase_MB:.2f} MB")
print("="*50)

# --- TẠO FILE SUBMISSION CHO LIGHTGBM ---
print("\nBắt đầu tạo file submission cho LightGBM...")
if lgb_model is not None:
    # Chuẩn bị dữ liệu test (date_block_num == 34)
    df_test_lgbm = df[df['date_block_num'] == 34]
    
    if not df_test_lgbm.empty:
        X_test_lgbm = df_test_lgbm.drop(actual_columns_to_exclude, axis=1, errors='ignore')
        test_ids_lgbm = df_test_lgbm['ID'].values

        # Dự đoán trên tập test
        predictions_lgbm = lgb_model.predict(X_test_lgbm)
        
        # Clip giá trị dự đoán
        predictions_lgbm_clipped = predictions_lgbm.clip(0, 20)

        if len(test_ids_lgbm) == len(predictions_lgbm_clipped):
            submission_df_lgbm = pd.DataFrame({
                'ID': test_ids_lgbm,
                'item_cnt_month': predictions_lgbm_clipped
            })
            print(f"File submission_lgbm.csv đã được tạo thành công với {len(submission_df_lgbm)} dòng.")
            print(submission_df_lgbm.head())
        else:
            print(f"Lỗi: Độ dài của test_ids_lgbm ({len(test_ids_lgbm)}) không khớp với độ dài của predictions_lgbm_clipped ({len(predictions_lgbm_clipped)}).")
            print("Không thể tạo file submission.")
    else:
        print("Cảnh báo: Không có dữ liệu cho date_block_num == 34 để tạo submission.")
else:
    print("Lỗi: Mô hình LightGBM chưa được huấn luyện (lgb_model is None). Không thể tạo file submission.")
print("="*50)


# # XGBoost
# submission_df_xgb.to_csv('/kaggle/working/submission.csv', index=False)

# # LSTM
# submission_df_lstm.to_csv('/kaggle/working/submission.csv', index=False)

# # Ridge
# submission_df_ridge.to_csv('/kaggle/working/submission.csv', index=False)

# Light GBM
submission_df_lgbm.to_csv('/kaggle/working/submission.csv', index=False)

