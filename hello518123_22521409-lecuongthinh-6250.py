import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder


# Tải tất cả các file dữ liệu
path = "/kaggle/input/ds-108-p-21-assigment-06/"
try:
    # --- Dữ liệu huấn luyện (Train) ---
    delay_4_6 = pd.read_csv(path + 'delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv')
    delay_7_9 = pd.read_csv(path + 'delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv')
    not_delay_4_6 = pd.read_csv(path + 'not_delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv')
    not_delay_7_9 = pd.read_csv(path + 'not_delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv')

    # --- Dữ liệu kiểm tra (Test) ---
    test_df = pd.read_csv(path + 'PILOT_10.csv')
    # Giữ lại ID để tạo file submission
    test_ids = test_df['ID']

except FileNotFoundError as e:
    print(f"Lỗi: Không tìm thấy file. Hãy chắc chắn các file csv nằm đúng thư mục.")
    print(e)
    # Thoát nếu không có file
    exit()

# --- Gộp dữ liệu huấn luyện ---
delay_4_6['is_late'] = 1
delay_7_9['is_late'] = 1
not_delay_4_6['is_late'] = 0
not_delay_7_9['is_late'] = 0

train_df = pd.concat([delay_4_6, delay_7_9, not_delay_4_6, not_delay_7_9], ignore_index=True)


def rename_duplicate_columns(df):
    cols = []
    count = {}
    for column in df.columns:
        # Làm sạch tên cột và thay dấu cách bằng gạch dưới
        clean_column = column.replace(' ', '_')
        if clean_column in count:
            count[clean_column] += 1
            # Đổi tên cột thứ hai, thứ ba... bị trùng
            cols.append(f"{clean_column}_{count[clean_column]}")
        else:
            count[clean_column] = 1
            cols.append(clean_column)
    df.columns = cols
    return df

print("Tên cột gốc (có thể có trùng lặp):")
# In ra một vài tên cột có khả năng gây lỗi
print([col for col in train_df.columns if 'SPECIAL' in col.upper()])

train_df = rename_duplicate_columns(train_df)
test_df = rename_duplicate_columns(test_df) # Áp dụng cho cả tập test để nhất quán

print("\nTên cột đã được đổi tên (duy nhất):")
print([col for col in train_df.columns if 'SPECIAL' in col.upper()])
# ==============================================================================


print(f"\nSố dòng dữ liệu train: {len(train_df)}")
print(f"Số dòng dữ liệu test: {len(test_df)}")
print("Phân phối của biến mục tiêu 'is_late' trong tập train:")
print(train_df['is_late'].value_counts(normalize=True))


def preprocess(df):
    """
    Hàm này thực hiện tất cả các bước tiền xử lý và feature engineering.
    (Hàm này giữ nguyên, không cần thay đổi)
    """
    # 1. Xử lý các cột ngày tháng
    for col in ['Order_date', 'VSD']:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # 2. Kỹ thuật đặc trưng thời gian
    df['planned_lead_time'] = (df['VSD'] - df['Order_date']).dt.days
    df['order_day_of_week'] = df['Order_date'].dt.dayofweek
    df['order_month'] = df['Order_date'].dt.month
    df['order_day'] = df['Order_date'].dt.day

    # 3. Xử lý các cột có vấn đề
    cols_to_drop = [
        'Order_date', 'VSD',
        'SOUF_RCV_NO', 'QTUF_RCV_NO',
        'GLOBAL_NO',
        'PRODUCT_ATTRIBUTION', 'Stock_class', 'label'
    ]
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=existing_cols_to_drop)

    # 4. Xử lý giá trị thiếu
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].fillna('Unknown')

    # 5. Mã hóa cột phân loại
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')

    return df

# 1. Chuẩn bị để gộp
y = train_df['is_late'].copy() # Lưu lại biến mục tiêu
train_df_to_combine = train_df.drop(columns=['is_late'])

# Đảm bảo index của test không bị trùng với train khi gộp
original_test_index = test_df.index
test_df.index = test_df.index + len(train_df)

# 2. Gộp train và test
print("Gộp train và test để xử lý đồng nhất...")
combined_df = pd.concat([train_df_to_combine, test_df], ignore_index=False)

# 3. Áp dụng preprocess trên dữ liệu đã gộp
combined_processed = preprocess(combined_df)

# 4. Tách lại thành train và test đã xử lý
train_processed = combined_processed.loc[:len(train_df)-1]
test_processed = combined_processed.loc[len(train_df):]

# Khôi phục lại index gốc của test_processed để khớp với test_ids
test_processed.index = original_test_index

print("\nTách dữ liệu thành công.")
print(f"Số đặc trưng được sử dụng: {len(train_processed.columns)}")
print(f"Kích thước train_processed: {train_processed.shape}")
print(f"Kích thước test_processed: {test_processed.shape}")

# Kiểm tra sự nhất quán của cột
if not all(train_processed.columns == test_processed.columns):
    raise ValueError("Lỗi: Tên cột của train và test không khớp sau khi xử lý!")


from collections import Counter

def objective(trial):

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 300, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'min_split_gain': trial.suggest_float("min_split_gain", 0.0, 1.0),
        'min_data_in_leaf': trial.suggest_int("min_data_in_leaf", 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'bagging_freq': trial.suggest_int("bagging_freq", 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'boost_from_average': False
    }

    # Stratified K-Fold
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []

    
    for train_index, val_index in kf.split(train_processed, y):
        X_train, X_val = train_processed.iloc[train_index].copy(), train_processed.iloc[val_index].copy()
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Tạo thêm đặc trưng mới trong quá trình huấn luyện
        for df_part in [X_train, X_val]:
            if 'planned_lead_time' in df_part.columns and 'PO_QTY' in df_part.columns:
                df_part['lead_time_per_product'] = df_part['planned_lead_time'] / (df_part['PO_QTY'] + 1)
            if 'order_day_of_week' in df_part.columns and 'order_month' in df_part.columns:
                df_part['weekday_vs_month'] = df_part['order_day_of_week'] * df_part['order_month']
            if 'SUPPLIER_CODE' in df_part.columns and 'MATERIAL_CODE' in df_part.columns:
                comb = df_part['SUPPLIER_CODE'].astype(str) + '_' + df_part['MATERIAL_CODE'].astype(str)
                df_part['supplier_material_comb'] = LabelEncoder().fit_transform(comb)

        class_counts = Counter(y_train)
        neg_pos_ratio = class_counts[0] / class_counts[1]
        params['scale_pos_weight'] = neg_pos_ratio
        params['is_unbalance'] = False

        model = lgb.LGBMClassifier(**params)

        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='logloss',
                  callbacks=[lgb.early_stopping(100, verbose=False)])

        preds = model.predict(X_val)
        f1 = f1_score(y_val, preds, average='macro')
        f1_scores.append(f1)

    return np.mean(f1_scores)


# Tạo study và chạy tối ưu
study = optuna.create_study(direction="maximize")
print("\nBắt đầu quá trình tối ưu hóa...")
study.optimize(objective, n_trials=22)

print("Quá trình tối ưu hóa hoàn tất.")
print("Số lần thử nghiệm: ", len(study.trials))
print("Bộ tham số tốt nhất tìm được: ", study.best_params)
print("Điểm Macro F1-Score tốt nhất (cross-validation): ", study.best_value)


# Lấy bộ tham số tốt nhất từ study
best_params = study.best_params

# Thêm các tham số cố định
best_params['objective'] = 'binary'
best_params['metric'] = 'binary_logloss'
best_params['verbosity'] = -1
best_params['boosting_type'] = 'gbdt'
best_params['scale_pos_weight'] = 40.0

# Huấn luyện mô hình cuối cùng trên TOÀN BỘ dữ liệu train
final_model = lgb.LGBMClassifier(**best_params)
final_model.fit(train_processed, y)

print("\nHuấn luyện mô hình cuối cùng hoàn tất.")

# Dự đoán trên tập test
test_preds = final_model.predict(test_processed)

# Tạo file submission
submission_df = pd.DataFrame({'ID': test_ids, 'label': test_preds})

# Lưu file submission
submission_df.to_csv('submission.csv', index=False)

print("\nĐã tạo file 'submission.csv' thành công!")
print(submission_df.head())
print("Phân phối dự đoán trên tập test:")
print(submission_df['label'].value_counts(normalize=True))

