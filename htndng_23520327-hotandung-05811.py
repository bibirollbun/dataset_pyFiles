# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# --- Thêm đoạn mã để kiểm tra sample_Solution.csv ---
print("\n--- Kiểm tra tệp sample_Solution.csv ---")
try:
    sample_solution_path = "/kaggle/input/ds-108-p-21-assigment-06/sample_Solution.csv"
    df_sample_solution = pd.read_csv(sample_solution_path)
    print(f"Đã tải thành công: {sample_solution_path}")
    df_sample_solution.info()
    print(df_sample_solution.head())
except FileNotFoundError:
    print(f"⚠️ Không tìm thấy tệp sample_Solution.csv tại đường dẫn: {sample_solution_path}. Vui lòng đảm bảo tệp tồn tại.")
except Exception as e:
    print(f"⚠️ Lỗi khi đọc tệp sample_Solution.csv: {e}")
print("--- Kết thúc kiểm tra sample_Solution.csv ---\n")
# ----------------------------------------------------

# THÊM CÁC LỆNH CÀI ĐẶT/NÂNG CẤP THƯ VIỆN NÀY ĐỂ ĐẢM BẢO TƯƠNG THÍCH
# Fix lỗi ModuleNotFoundError: No module named 'sklearn.utils._metadata_requests'
# do xung đột phiên bản giữa scikit-learn và imbalanced-learn.
# Môi trường Kaggle hiện tại có scikit-learn==1.2.2
# imbalanced-learn 0.13.0 yêu cầu scikit-learn >= 1.3.2,
# vì vậy chúng ta sẽ cài đặt imbalanced-learn 0.12.2 để tương thích.
print("Đang gỡ cài đặt phiên bản imbalanced-learn hiện tại (nếu có)...")
!pip uninstall -y imbalanced-learn
print("Đang cài đặt imbalanced-learn phiên bản 0.12.2 (tương thích với scikit-learn 1.2.2)...")
!pip install imbalanced-learn==0.12.2


# Các thư viện khác cần thiết cho mô hình
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import f1_score, precision_recall_curve
import lightgbm as lgb
from imblearn.over_sampling import SMOTE # Bây giờ sẽ import từ imbalanced-learn 0.12.2

# --- 1. Hàm tải và kết hợp dữ liệu (Đã điều chỉnh cho môi trường Kaggle) ---
def load_and_combine_data_kaggle(base_path):
    """
    Tải và kết hợp dữ liệu từ đường dẫn Kaggle.
    """
    print(f"Đang tải dữ liệu từ thư mục Kaggle: {base_path}")

    # Đường dẫn đầy đủ đến các file dữ liệu trong môi trường Kaggle
    delay_46_path = os.path.join(base_path, "delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv")
    delay_79_path = os.path.join(base_path, "delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv")
    not_delay_46_path = os.path.join(base_path, "not_delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv")
    not_delay_79_path = os.path.join(base_path, "not_delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv")
    test_df_path = os.path.join(base_path, "PILOT_10.csv")

    # Kiểm tra xem các file có tồn tại không
    required_files = [delay_46_path, delay_79_path, not_delay_46_path, not_delay_79_path, test_df_path]
    for f_path in required_files:
        if not os.path.exists(f_path):
            raise FileNotFoundError(f"File không tìm thấy: {f_path}. Vui lòng kiểm tra lại đường dẫn và đảm bảo tất cả các file dữ liệu đã được tải về.")

    # Đọc các file dữ liệu
    delay_46 = pd.read_csv(delay_46_path)
    delay_79 = pd.read_csv(delay_79_path)
    not_delay_46 = pd.read_csv(not_delay_46_path)
    not_delay_79 = pd.read_csv(not_delay_79_path)
    test_df = pd.read_csv(test_df_path)

    # Tạo tập huấn luyện gộp
    train_df = pd.concat([
        delay_46,
        delay_79,
        not_delay_46,
        not_delay_79
    ], ignore_index=True)

    print(f"✅ Đã tải xong dữ liệu. Kích thước tập huấn luyện: {train_df.shape}, Kích thước tập kiểm tra: {test_df.shape}")
    return train_df, test_df

# --- 2. Hàm tiền xử lý dữ liệu ---
def preprocess_data(train_df, test_df):
    """
    Thực hiện tiền xử lý dữ liệu: xử lý thiếu, ép kiểu, tạo đặc trưng thời gian, mã hóa, chuẩn hóa.
    Đảm bảo tính nhất quán của các cột giữa train và test.
    """
    print("\n--- Bắt đầu tiền xử lý dữ liệu ---")

    # Xử lý cột 'Order date': Ép kiểu và tạo đặc trưng thời gian
    for df in [train_df, test_df]:
        df['Order date'] = pd.to_datetime(df['Order date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        df['Order date'] = df['Order date'].fillna(pd.Timestamp('2022-01-01'))
        df['order_year'] = df['Order date'].dt.year
        df['order_month'] = df['Order date'].dt.month
        df['order_day'] = df['Order date'].dt.day
        df['order_dayofweek'] = df['Order date'].dt.dayofweek
        df['order_is_weekend'] = df['order_dayofweek'].isin([5, 6]).astype(int)

        # Xử lý SO_TIME
        df['SO_HOUR'] = df['SO_TIME'] // 10000
        df['SO_MINUTE'] = (df['SO_TIME'] // 100) % 100
        df['SO_SECOND'] = df['SO_TIME'] % 100

        df.drop(columns=['Order date', 'SO_TIME'], inplace=True)

    print("✅ Đã xử lý cột thời gian ('Order date', 'SO_TIME').")

    # Xử lý giá trị thiếu (Missing Values)
    cols_to_drop_due_to_nulls = ['QTUF_RCV_NO', 'SOUF_RCV_NO']
    for col in cols_to_drop_due_to_nulls:
        if col in train_df.columns:
            train_df.drop(columns=[col], inplace=True)
        if col in test_df.columns:
            test_df.drop(columns=[col], inplace=True)
    print("✅ Đã loại bỏ 'QTUF_RCV_NO' và 'SOUF_RCV_NO' (nếu tồn tại) do nhiều giá trị thiếu.")

    # Điền giá trị thiếu cho các cột còn lại
    categorical_fillna_cols = ['REASON_CD', 'OTHER AREA SHIP DIV', 'Ship Mode']
    for col in categorical_fillna_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna('UNKNOWN')
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna('UNKNOWN')

    numerical_fillna_cols = ['SHIP DECISION NO', 'SUPPLIER_DIV']
    for col in numerical_fillna_cols:
        if col in train_df.columns:
            median_val = train_df[col].median()
            train_df[col] = train_df[col].fillna(median_val)
            if col in test_df.columns:
                test_df[col] = test_df[col].fillna(median_val)
    print("✅ Đã xử lý các giá trị thiếu còn lại.")

    # Xử lý cột trùng lặp 'SPECIAL DIV' và 'SPECIAL_DIV'
    if 'SPECIAL DIV' in train_df.columns and 'SPECIAL_DIV' in train_df.columns:
        if (train_df['SPECIAL DIV'].astype(str).equals(train_df['SPECIAL_DIV'].astype(str))):
            train_df.drop(columns=['SPECIAL DIV'], inplace=True)
            if 'SPECIAL DIV' in test_df.columns:
                test_df.drop(columns=['SPECIAL DIV'], inplace=True)
            print("✅ Đã loại bỏ cột 'SPECIAL DIV' do trùng lặp với 'SPECIAL_DIV'.")
        else:
            print("⚠️ Cột 'SPECIAL DIV' và 'SPECIAL_DIV' khác nhau, không loại bỏ.")

    # Mã hóa các cột phân loại (Categorical Encoding)
    categorical_cols_in_train = train_df.select_dtypes(include='object').columns.tolist()
    categorical_cols_to_encode_in_both = [col for col in categorical_cols_in_train if col in test_df.columns]

    print(f"Các cột phân loại được mã hóa (chung cho cả train và test): {categorical_cols_to_encode_in_both}")

    for col in categorical_cols_to_encode_in_both:
        le = LabelEncoder()
        all_values = pd.concat([train_df[col].astype(str), test_df[col].astype(str)]).unique()
        le.fit(all_values)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
    print("✅ Đã mã hóa các cột phân loại chung bằng Label Encoding.")

    # Xử lý Outliers và Chuẩn hóa (Scaling)
    skewed_cols = ['PURCHASE AMOUNT', 'SUPPLIER INV AMOUNT']
    for col in skewed_cols:
        if col in train_df.columns:
            train_df[col] = np.log1p(train_df[col].clip(lower=0))
            if col in test_df.columns:
                test_df[col] = np.log1p(test_df[col].clip(lower=0))
    print("✅ Đã áp dụng log1p cho 'PURCHASE AMOUNT', 'SUPPLIER INV AMOUNT' (nếu tồn tại).\n")

    if 'WEIGHT PER PIECE' in train_df.columns:
        cap_value = train_df['WEIGHT PER PIECE'].quantile(0.99)
        train_df['WEIGHT PER PIECE'] = np.where(train_df['WEIGHT PER PIECE'] > cap_value, cap_value, train_df['WEIGHT PER PIECE'])
        if 'WEIGHT PER PIECE' in test_df.columns:
            test_df['WEIGHT PER PIECE'] = np.where(test_df['WEIGHT PER PIECE'] > cap_value, cap_value, test_df['WEIGHT PER PIECE'])
    print("✅ Đã giới hạn outliers cho 'WEIGHT PER PIECE' (nếu tồn tại).")

    numerical_features_for_scaling = [
        'PURCHASE AMOUNT', 'SUPPLIER INV AMOUNT', 'WEIGHT PER PIECE', 'PACK QTY', 'SO QTY',
        'SO_DAY_OF_WEEK', 'SO_DAY_OF_MONTH', 'SO_HOUR', 'SO_MINUTE', 'SO_SECOND',
        'order_year', 'order_month', 'order_day', 'order_dayofweek', 'order_is_weekend',
        'SHIP DECISION NO', 'SUPPLIER_DIV'
    ]
    numerical_features_for_scaling = [col for col in numerical_features_for_scaling if col in train_df.columns]

    scaler = StandardScaler()
    train_df[numerical_features_for_scaling] = scaler.fit_transform(train_df[numerical_features_for_scaling])
    test_df[numerical_features_for_scaling] = scaler.transform(test_df[numerical_features_for_scaling])
    print("✅ Đã chuẩn hóa các đặc trưng số bằng StandardScaler.")

    # Đảm bảo tập đặc trưng là nhất quán giữa train và test
    train_cols_set = set(train_df.columns)
    test_cols_set = set(test_df.columns)

    cols_to_exclude_train = {'label'}
    cols_to_exclude_test = {'ID'}

    final_train_features = list(train_cols_set - cols_to_exclude_train)
    final_test_features = list(test_cols_set - cols_to_exclude_test)

    common_features = sorted(list(set(final_train_features) & set(final_test_features)))

    for col in final_train_features:
        if col not in common_features:
            if col in train_df.columns:
                train_df.drop(columns=[col], inplace=True)
                print(f"⚠️ Loại bỏ cột '{col}' khỏi train_df vì không có trong test_df.")
            
    for col in final_test_features:
        if col not in common_features:
            if col in test_df.columns:
                test_df.drop(columns=[col], inplace=True)
                print(f"⚠️ Loại bỏ cột '{col}' khỏi test_df vì không có trong train_df.")
            
    train_df = train_df[common_features + ['label']]
    test_df = test_df[common_features + ['ID']]

    print(f"Kích thước tập huấn luyện sau tiền xử lý và đồng bộ cột: {train_df.shape}")
    print(f"Kích thước tập kiểm tra sau tiền xử lý và đồng bộ cột: {test_df.shape}")
    return train_df, test_df

# --- 3. Hàm huấn luyện mô hình với tối ưu hóa siêu tham số ---
def train_and_predict(train_df, test_df):
    """
    Huấn luyện mô hình LightGBM với KFold Cross-Validation,
    sử dụng SMOTE để xử lý mất cân bằng dữ liệu, và tối ưu ngưỡng.
    """
    print("\n--- Bắt đầu huấn luyện mô hình ---\n")

    X = train_df.drop(columns=['label'])
    y = train_df['label']
    X_test = test_df.drop(columns=['ID'])

    # Tính toán scale_pos_weight cho dữ liệu mất cân bằng (vẫn hữu ích ngay cả với SMOTE)
    if 0 in y.value_counts() and 1 in y.value_counts():
        neg_count = y.value_counts()[0]
        pos_count = y.value_counts()[1]
        scale_pos_weight_value = neg_count / pos_count
        print(f"Tính toán scale_pos_weight ban đầu: {scale_pos_weight_value:.2f}")
    else:
        scale_pos_weight_value = 1.0
        print("Cảnh báo: Chỉ có một lớp trong tập huấn luyện. Đặt scale_pos_weight = 1.0")

    # KHỐI CODE RandomizedSearchCV (GIỮ LẠI VÀ CHẠY VỚI GPU)
    # LƯU Ý: Với n_jobs=1, quá trình tìm kiếm sẽ tuần tự nhưng mỗi lần huấn luyện sẽ nhanh hơn nhiều nhờ GPU.
    param_dist = {
        'n_estimators': [300, 500, 700, 1000],
        'learning_rate': [0.005, 0.01, 0.05, 0.1],
        'num_leaves': [20, 31, 40, 60, 80],
        'max_depth': [-1, 5, 7, 9, 12],
        'min_child_samples': [10, 20, 30, 40],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
        'reg_lambda': [0, 0.01, 0.1, 0.5, 1.0],
        'scale_pos_weight': sorted(list(set([
            scale_pos_weight_value * 0.5,
            scale_pos_weight_value,
            scale_pos_weight_value * 1.5,
            scale_pos_weight_value * 2.0,
            scale_pos_weight_value * 3.0,
            scale_pos_weight_value * 5.0
        ])))
    }

    # Khởi tạo LightGBM cho RandomizedSearchCV VỚI GPU
    lgbm = lgb.LGBMClassifier(random_state=42, n_jobs=-1, objective='binary', device='gpu') 
    print("Đang tìm kiếm siêu tham số tốt nhất với RandomizedSearchCV (sử dụng KFold n_splits=3)...")
    rand_search = RandomizedSearchCV(
        estimator=lgbm,
        param_distributions=param_dist,
        n_iter=5, # Số lần lặp tìm kiếm. Có thể tăng lên nếu muốn tìm kiếm kỹ hơn (sẽ mất thêm thời gian).
        scoring='f1',
        cv=KFold(n_splits=3, shuffle=True, random_state=42), # Giảm n_splits để tìm kiếm nhanh hơn
        verbose=1,
        random_state=42,
        n_jobs=1 # ĐẶT n_jobs=1 ĐỂ TRÁNH LỖI RAM VÀ TRANH CHẤP GPU
    )
    rand_search.fit(X, y)
    best_params = rand_search.best_params_
    print(f"✅ Siêu tham số tốt nhất tìm được: {best_params}\n")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds_proba = np.zeros(len(X))
    test_preds_proba = np.zeros(len(X_test))

    smote = SMOTE(random_state=42) # Đã loại bỏ n_jobs=-1 khỏi SMOTE
    
    print("Đang huấn luyện mô hình cuối cùng với KFold Cross-Validation (5 folds), SMOTE và siêu tham số tốt nhất...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n--- Đang huấn luyện Fold {fold + 1} ---")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        print(f"   Phân bố lớp ban đầu của Fold {fold + 1} (trước SMOTE):")
        print(y_train.value_counts())

        # Áp dụng SMOTE CHỈ trên tập huấn luyện của fold hiện tại
        print(f"   Áp dụng SMOTE cho tập huấn luyện của Fold {fold + 1}...")
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"   Kích thước tập huấn luyện sau SMOTE: {X_train_resampled.shape}")
        print(f"   Phân bố lớp sau SMOTE của Fold {fold + 1}:")
        print(y_train_resampled.value_counts())

        # KHỞI TẠO MODEL CUỐI CÙNG VỚI GPU
        model = lgb.LGBMClassifier(**best_params, random_state=42, n_jobs=-1, objective='binary', device='gpu') # THÊM 'device='gpu'' TẠI ĐÂY
        model.fit(
            X_train_resampled, y_train_resampled, # Huấn luyện trên dữ liệu đã được SMOTE
            eval_set=[(X_val, y_val)], # Đánh giá trên dữ liệu validation gốc
            eval_metric='f1',
            callbacks=[lgb.early_stopping(100, verbose=100)]
        )

        oof_preds_proba[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds_proba += model.predict_proba(X_test)[:, 1] / kf.n_splits
    
    # --- Tối ưu ngưỡng phân loại trên OOF predictions ---
    precisions, recalls, thresholds = precision_recall_curve(y, oof_preds_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
    
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]
    
    oof_preds_optimized = (oof_preds_proba >= best_threshold).astype(int)
    f1_overall_optimized = f1_score(y, oof_preds_optimized)

    print(f"\n✅ F1 Score trung bình trên tập huấn luyện (OOF) (ngưỡng 0.5): {f1_score(y, (oof_preds_proba >= 0.5).astype(int)):.4f}")
    print(f"✅ Ngưỡng tối ưu tìm được trên OOF: {best_threshold:.4f}")
    print(f"✅ F1 Score trung bình trên tập huấn luyện (OOF) (ngưỡng tối ưu): {f1_overall_optimized:.4f}")

    return test_preds_proba, best_threshold

# --- 4. Hàm tạo file Submission ---
def create_submission_file(test_df_original, test_preds_proba, best_threshold, filename="submission.csv"):
    """
    Tạo file submission cuối cùng.
    Sử dụng test_df_original để lấy cột 'ID'.
    """
    print("\n--- Tạo file Submission ---")
    # Áp dụng ngưỡng tối ưu đã tìm được
    final_test_pred = (test_preds_proba >= best_threshold).astype(int)

    submission = pd.DataFrame({
        'ID': test_df_original['ID'],
        'Predicted': final_test_pred
    })
    submission.to_csv(filename, index=False)
    print(f"✅ Đã tạo file submission: {filename}")
    print(submission.head())

# --- Chạy toàn bộ quy trình ---
if __name__ == "__main__":
    # ĐẶT ĐƯỜNG DẪN ĐẾN THƯ MỤC CHỨA DỮ LIỆU CỦA BẠN TRÊN KAGGLE
    # THAY THẾ 'ds-108-p-21-assigment-06' BẰNG TÊN CHÍNH XÁC CỦA DATASET CỦA BẠN TRÊN KAGGLE NẾU KHÁC
    kaggle_data_path = "/kaggle/input/ds-108-p-21-assigment-06"

    # 1. Tải dữ liệu
    train_df, test_df_original = load_and_combine_data_kaggle(kaggle_data_path)

    # 2. Tiền xử lý dữ liệu (truyền bản sao để không ảnh hưởng đến DataFrame gốc)
    train_df_processed, test_df_processed = preprocess_data(train_df.copy(), test_df_original.copy())

    # 3. Huấn luyện mô hình và dự đoán (nhận lại xác suất và ngưỡng tối ưu)
    final_test_predictions_proba, optimal_threshold = train_and_predict(train_df_processed, test_df_processed)

    # 4. Tạo file Submission (truyền cả xác suất và ngưỡng tối ưu)
    create_submission_file(test_df_original, final_test_predictions_proba, optimal_threshold)

